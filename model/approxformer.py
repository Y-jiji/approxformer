import torch as t
from typing import *

class IterSequential(t.nn.Module):
    def __init__(self, *args):
        super().__init__()
        t.nn.Sequential
        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def iterate(self, x: t.Tensor, cache: list[Any]):
        _cache = [None] * len(cache)
        for i, module in enumerate(self.children()):
            x, _cache[i] = module.iterate(x, cache[i])
        return x, _cache

    def init_cache(self) -> list[Any]:
        _cache = []
        for module in self.children():
            _cache.append(module.init_cache())
        return _cache

class ApproxFormer(t.nn.Module):
    def __init__(self, N: int, L: int) -> None:
        super().__init__()
        self.embed = t.nn.Embedding(N, 128)
        self.inner = IterSequential(
            RoPE(128),
            DecoderLayer(M=32, D=128, A=64, C=128, L=L, H=4, W=128, G=4, F=1024),
            DecoderLayer(M=32, D=128, A=64, C=128, L=L, H=2, W=128, G=4, F=1024),
        )
        self.output = t.nn.Linear(128, N, bias=False)
        with t.no_grad():
            self.output.weight.set_(-self.embed.weight / 100)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.output(self.inner(self.embed(x))).softmax(dim=-1)

    def init_cache(self) -> list[Any]:
        return self.inner.init_cache()

    def iterate(self, x: t.Tensor, cache: list[Any]) -> tuple[t.Tensor, list[Any]]:
        x, cache = self.inner.iterate(self.embed(x), cache)
        return self.output(x).softmax(dim=-1), cache

class RoPE(t.nn.Module):
    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N
        self.gamma = t.nn.Parameter(t.tensor(0.0))
        self.dummy = t.nn.Parameter(t.tensor(0.0))
        assert N % 2 == 0

    @property
    def device(self):
        return self.dummy.device

    def forward(self, x, OFFSET=0):
        with t.no_grad():
            L = x.shape[-2]
            assert self.N % 2 == 0
            y = t.arange(0, L, device=self.device).unsqueeze(dim=-1) + OFFSET
            f = t.arange(0, self.N, device=self.device) + 2
            w = y * 2 ** -(4 + f % (self.N // 2)) * t.pi + (2 * f > self.N) * t.pi / 2
        return (1 - self.gamma.sigmoid()) * x + self.gamma.sigmoid() * (t.concat([x[..., self.N//2:], x[..., :self.N//2]], dim=-1) * w.sin() + x * w.cos())

    def iterate(self, x: t.Tensor, offset: int) -> tuple[t.Tensor, int]:
        return self.forward(x, offset), offset + 1

    def init_cache(self) -> int:
        return 0

class CNNLayer(t.nn.Module):
    def __init__(self, I, W) -> None:
        super().__init__()
        self.ker = t.nn.Parameter(t.randn(I, 1, W))
        self.W = W
        self.I = I

    def forward(self, x: t.Tensor):
        with t.no_grad():
            pad = x[..., 0:1, :].repeat_interleave(self.W - 1, dim=-2).detach().transpose(-2, -1)
        return t.nn.functional.conv1d(t.concat([pad, x.transpose(-2, -1)], dim=-1), self.ker.softmax(dim=-1), t.zeros(self.I, device=x.device), groups=self.I).transpose(-2, -1)

    def iterate(self, x: t.Tensor, window: None | t.Tensor) -> tuple[t.Tensor, None | t.Tensor]:
        window = (x if window is None else
                  t.concat([window[..., 1:, :], x], dim=-2) if window.shape[-2] == self.W else
                  t.concat([window, x], dim=-2))
        return self.forward(window)[..., -1:, :, :], window

class GapAttention(t.nn.Module):
    POW = 4
    def __init__(self, D, H, C, M, L) -> None:
        super().__init__()
        # dummy variable as device indicator
        self.dummy = t.nn.Parameter(t.tensor(0.0))
        # miscellaneous layers
        self.pe = RoPE(D)
        self.ln = t.nn.LayerNorm(D * H)
        # key, query and value
        self.k = t.nn.Sequential(
            t.nn.Linear((D, M * H)), t.nn.Unflatten(-1, (M, H)))
        self.q = t.nn.Sequential(
            t.nn.Linear((D, M * H)), t.nn.Unflatten(-1, (M, H)))
        self.v = t.nn.Sequential(
            t.nn.Linear((D, D * H)), t.nn.Unflatten(-1, (D, H)))
        # precomputed coefficients for kernel function
        def combi(n, x):
            return t.arange(n+1-x, n+1).prod() / t.arange(1, x+1).prod()
        def coeff(u, pow, n):
            u = u + pow * n + pow - 1
            r = t.zeros_like(u).float()
            for i in range(pow):
                v = (u.unsqueeze(-1) - (2 * n + 1) * i - t.arange(pow-1)).relu()
                c = combi(pow, i)
                r += c * (2 * ((i + 1) % 2) - 1) * v.prod(-1) / (t.arange(pow-1) + 1).prod(-1)
            return r
        assert self.C % 4 == 0
        self._coeffs = coeff(t.arange(0, self.C+1), self.POW, self.C//self.POW)
        # some numbers
        self.M = M
        self.C = C
        self.L = L
        self.H = H
    
    @t.no_grad()
    def inp_position_code(self, OFF:int, LEN:int):
        """
        compute output position code (IPC) from positions
        """
        # encoding code
        N = self.C // self.POW
        X = t.arange(0, LEN, device=self.device)
        Y = t.arange(0, self.C+1, device=self.device)
        OFF = self.L / N + OFF
        X = t.cos(Y * (((X + OFF) / self.L) * t.pi).unsqueeze(-1))
        X[:, 0] /= 2
        # put coefficients into their place
        return self._coeffs.to(self.device) * X / N ** (self.POW-1) / 2

    @t.no_grad()
    def out_position_code(self, OFF:int, LEN:int):
        """
        compute output position code (OPC) from positions
        """
        # decoding code
        Y = t.arange(0, self.C+1, device=self.device)
        X = t.arange(0, self.L, device=self.device)
        X = t.cos(-Y * ((X / self.L) * t.pi).unsqueeze(-1)).cumsum(0) / self.L
        # a very small elementwise perturbation to stop information leak
        if self.training:
            X[:, 0] += 1e-3 * t.randn_like(X[:, 0])
        # return output position code
        return X[OFF:LEN+OFF]

    @property
    def device(self):
        return self.dummy.device

    def forward(self, x):
        """
        x: [L, D]
        return: [L, D, H]
        """
        L = x.shape[-2]
        lnx = self.ln(self.pe(x))
        k, q, v = t.exp(self.k(lnx)), t.exp(self.q(lnx)), self.v(x)
        ipc, opc = self.inp_position_code(0, L), self.out_position_code(0, L)
        out = t.einsum("...imh, ...idh, ...ic, ...omh, ...oc -> ...odh", k, v, ipc, q, opc)
        return out

    def iterate(self, x, cache) -> tuple[t.Tensor, tuple[int, t.Tensor]]:
        """
        x: [1, D]
        cache: int, [M, C+1, D, H]
        return: [1, D, H]
        """
        assert x.shape[-2] == 1
        off, mem = cache
        lnx = self.ln(x)
        k, q, v = t.exp(-self.k(lnx).relu()), t.exp(-self.q(lnx).relu()), self.v(x)
        ipc, opc = self.inp_position_code(off, 1), self.out_position_code(off, 1)
        mem = mem + t.einsum("...imh, ...idh, ...ic -> ...mcdh", k, v, ipc)
        out = t.einsum("...mcdh, ...oc -> ...odh", q, opc)
        cache = off + 1, mem
        return out, cache

    def init_cache(self) -> tuple[int, t.Tensor]:
        return 0, t.zeros(self.M, self.C+1, self.D, self.H, device=self.device)

class CumAttention(t.nn.Module):
    def __init__(self, D, A, H) -> None:
        super().__init__()
        """
        O[i] = sum{k in 0..D} sum{j in 0..i} Q[i][k] * K[j][k] * V[j]
        """
        # dummy variable as device indicator
        self.dummy = t.nn.Parameter(t.tensor(0.0))
        # miscellaneous layers
        self.pe = RoPE(D)
        self.ln = t.nn.LayerNorm(D)
        # key, query and value
        self.k = t.nn.Sequential(
            t.nn.Linear(D, A * H), t.nn.Unflatten(-1, A, H))
        self.q = t.nn.Sequential(
            t.nn.Linear(D, A * H), t.nn.Unflatten(-1, A, H))
        self.v = t.nn.Sequential(
            t.nn.Linear(D, D * H), t.nn.Unflatten(-1, D, H))
        # some numbers
        self.A = A
        self.D = D
        self.H = H

    @property
    def device(self):
        return self.dummy.device

    def forward(self, x):
        # key, query, value
        lnx = self.ln(self.pe(x))
        k, q, v = t.exp(self.k(lnx)), t.exp(self.q(lnx)), self.v(self.pe(x))
        mem = t.einsum("...lah, ...ldh -> ...ladh", k, v).cumsum(-4)
        return t.einsum("...lah, ...ladh -> ...ldh", q, mem)

    def iterate(self, x, cache):
        # key, query, value
        off, mem = cache
        lnx = self.pe(self.ln(x), off)
        k, q, v = t.exp(self.k(lnx)), t.exp(self.q(lnx)), self.v(self.pe(x, off))
        mem = mem + t.einsum("...lah, ...ldh -> ...ladh", k, v)
        return t.einsum("...lah, ...ladh -> ...ldh", q, mem), (off+1, mem)

    def init_cache(self):
        return 0, t.zeros(self.A, self.D, self.H)

class DecoderLayer(t.nn.Module):
    def __init__(self, M, D, A, C, L, H, W, G, F, p=0.1) -> None:
        """
        M: size of memory
        H: number of heads
        D: size of token representation
        A: size of attention matcher
        C: size of position code
        L: maximal length
        W: 1d cnn kernel width
        G: groups in feed-forward network
        F: feed forward dimensions
        p: dropout rate
        """
        super().__init__()
        self.cnn = CNNLayer(D * H, W)
        self.pe = RoPE(D)
        self.ln0 = t.nn.LayerNorm((A * H))
        self.ln1 = t.nn.LayerNorm((A * H))
        self.ln2 = t.nn.LayerNorm((D, H))
        self.ln3 = t.nn.LayerNorm((D, H))
        # scaling down will alleviate some numerical issues
        self.q_mem = t.nn.Parameter(t.randn(M, A * H) / 1000)
        self.k_mem = t.nn.Parameter(t.randn(M, A * H) / 1000)
        self.k_inp = t.nn.Linear(D, A * H)
        self.q_out = t.nn.Linear(D, A * H)
        self.v_inp = t.nn.Linear(D, D * H)
        self.dummy = t.nn.Parameter(t.tensor([]))
        self.ffn = t.nn.Sequential(
            t.nn.Unflatten(-1, (G, D*H // G)),
            t.nn.Linear(D*H // G, F // G),
            t.nn.Flatten(-2, -1),
            t.nn.ReLU(),
            t.nn.Dropout(p),
            t.nn.Linear(F, D),
        )
        self.C = C
        self.D = D
        self.A = A
        self.H = H
        self.M = M
        self.L = L
        self.W = W
        # precomputed coefficients for kernel function
        def combi(n, x):
            return t.arange(n+1-x, n+1).prod() / t.arange(1, x+1).prod()
        def coeff(u, pow, n):
            u = u + pow * n + pow - 1
            r = t.zeros_like(u).float()
            for i in range(pow):
                v = (u.unsqueeze(-1) - (2 * n + 1) * i - t.arange(pow-1)).relu()
                c = combi(pow, i)
                r += c * (2 * ((i + 1) % 2) - 1) * v.prod(-1) / (t.arange(pow-1) + 1).prod(-1)
            return r
        assert self.C % 4 == 0
        self._coeffs = coeff(t.arange(0, self.C+1), 4, self.C//4)

    @property
    def device(self):
        return self.dummy.device

    @staticmethod
    def normalize(x):
        return x / (x**2).sum(dim=-1, keepdim=True)**(1/2)

    @staticmethod
    def d_pow(x, l, n, pow, off):
        def combi(n, x):
            return t.arange(n+1-x, n+1).prod() / t.arange(1, x+1).prod()
        def coeff(u):
            u = u + pow * n + pow - 1
            r = t.zeros_like(u).float()
            for i in range(pow):
                v = (u.unsqueeze(-1) - (2 * n + 1) * i - t.arange(pow-1)).relu()
                c = combi(pow, i)
                r += c * (2 * ((i + 1) % 2) - 1) * v.prod(-1) / (t.arange(pow-1) + 1).prod(-1)
            return r
        a = t.arange(0, pow*n+1)
        u = coeff(a) * t.cos(a * (((x + off) / l) * t.pi).unsqueeze(-1)) / n ** (pow-1) / 2
        u[:, 0] /= 2
        v = t.cos(-a * ((x / l) * t.pi).unsqueeze(-1)).cumsum(0) / l
        return u, v

    @t.no_grad()
    def inp_position_code(self, OFF:int, LEN:int):
        """
        compute output position code (IPC) from positions
        """
        pow = 4
        n = self.C // pow
        a = t.arange(0, self.C+1, device=self.device)
        l = self.L
        x = t.arange(0, LEN, device=self.device)
        off = self.L / n + OFF
        x = t.cos(a * (((x + off) / l) * t.pi).unsqueeze(-1))
        x[:, 0] /= 2
        return self._coeffs.to(self.device) * x / n ** (pow-1) / 2

    @t.no_grad()
    def out_position_code(self, OFF:int, LEN:int):
        """
        compute output position code (OPC) from positions
        """
        a = t.arange(0, self.C+1, device=self.device)
        l = self.L
        off = OFF
        x = t.arange(0, LEN, device=self.device)
        x = t.cos(-a * (((x + off) / l) * t.pi).unsqueeze(-1)).cumsum(0) / l
        # a very small randomization to stop information leak
        if self.training:
            x[:, 0] += 1e-3 * t.randn_like(x[:, 0])
        return x

    def dot(self, x, y):
        """
        compute multi-head attention
        """
        a = t.einsum("...ijh, ...kjh -> ...ikh", x.unflatten(-1, (self.A, self.H)), y.unflatten(-1, (self.A, self.H)))
        assert (a < 15).all().item()
        return a.clamp_max(15).exp()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        parallel mode forward
        x: [..., L, INPUT]
        """
        L = x.shape[-2]
        ipc = self.inp_position_code(0, L)
        opc = self.out_position_code(0, L)
        a0 = self.dot(self.q_mem, self.ln0(self.k_inp(self.pe(x))))
        v0 = t.einsum("...mlh, ...lc, ...lvh -> ...mcvh", a0, ipc, self.v_inp(x).unflatten(-1, (self.D, self.H)))
        a1 = self.dot(self.ln1(self.q_out(x)), self.k_mem)
        v1 = t.einsum("...lmh, ...lc, ...mcvh -> ...lvh", a1, opc, v0)
        v2 = self.cnn(self.v_inp(x)).unflatten(-1, (self.D, self.H))
        return x + self.ffn(self.ln3(self.ln2(v1) + x.unsqueeze(dim=-1)).flatten(-2, -1))

    def iterate(self, x: t.Tensor, cache: tuple[int, t.Tensor, t.Tensor]) -> tuple[t.Tensor, tuple[int, t.Tensor, t.Tensor]]:
        L = x.shape[-2]
        assert L == 1
        assert self.training == False
        offset, v0, window = cache
        ipc = self.inp_position_code(offset, L)
        opc = self.out_position_code(offset, L)
        a0 = self.dot(self.q_mem, self.ln0(self.k_inp(self.pe(x, offset))))
        v0 = v0 + t.einsum("...mlh, ...lc, ...lvh -> ...mcvh", a0, ipc, self.v_inp(x).unflatten(-1, (self.D, self.H)))
        a1 = self.dot(self.ln1(self.q_out(x)), self.k_mem)
        v1 = t.einsum("...lmh, ...lc, ...mcvh -> ...lvh", a1, opc, v0)
        window = (x if window is None else
                  t.concat([window[..., 1:, :], x], dim=-2) if window.shape[-2] == self.W else
                  t.concat([window, x], dim=-2))
        v2 = self.cnn(self.v_inp(window)).unflatten(-1, (self.D, self.H))[..., -1:, :, :]
        cache = (offset + 1, v0, window)
        return x + self.ffn(self.ln3(self.ln2(v1) + x.unsqueeze(dim=-1)).flatten(-2, -1)), cache

    def init_cache(self) -> tuple[int, t.Tensor, t.Tensor]:
        return 0, t.zeros(self.M, self.C+1, self.D, self.H, device=self.device), None

if __name__ == '__main__':
    with t.no_grad():
        model = ApproxFormer(128, 8000)
        model.train(False)
        x = t.randint(0, 127, (10, 1000))
        cache = model.init_cache()
        ys = []
        for i in range(x.shape[-1]):
            if i % 100 == 0: print(i)
            y, cache = model.iterate(x[..., i:i+1], cache)
            ys.append(y)
        y0 = t.concat(ys, dim=-2)
        y1 = model.forward(x)
        y2 = model.forward(t.randint(0, 127, (10, 1000)))
        print(((y0 - y1).abs() / y1).mean(dim=-2))
        print(((y2 - y1).abs() / y1).mean(dim=-2))