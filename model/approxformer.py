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

class Forward(t.nn.Module):
    def __init__(self, inner: t.nn.Module) -> None:
        super().__init__()
        self.inner = inner
    def forward(self, x):
        return self.inner(x)
    def init_cache(self) -> None:
        return None
    def iterate(self, x: t.Tensor, cache: None) -> tuple[t.Tensor, None]:
        return self.forward(x), None

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
    def __init__(self, D, A, H):
        self.att = IterSequential(
            CumAttention(D, A, H),
            Forward(t.nn.LayerNorm((D, H))),
        )
        self.ffn = t.nn.Sequential(
            t.nn.Flatten(-2, -1),
            t.nn.LayerNorm((D * H, )),
            t.nn.Linear(D * H, 2048),
            t.nn.ReLU(),
            t.nn.Dropout(0.1),
            t.nn.Linear(2048, D),
        )

    def forward(self, x):
        return self.ffn(self.att(x) + x.unsqueeze(-1)) + x

    def iterate(self, x, cache):
        y, cache = self.att.iterate(x, cache)
        return self.ffn(y + x.unsqueeze(-1)) + x, cache

    def init_cache(self):
        return self.att.init_cache()