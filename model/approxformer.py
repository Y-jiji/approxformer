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
            RotaryPE(128),
            DecoderLayer(M=32, D=128, A=64, C=128, L=L, H=4, W=128, G=4, F=1024),
            DecoderLayer(M=32, D=128, A=64, C=128, L=L, H=4, W=128, G=4, F=1024),
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

class RotaryPE(t.nn.Module):
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
        assert C % 4 == 0
        self.cnn = CNNLayer(D * H, W)
        self.pe = RotaryPE(D)
        self.ln0 = t.nn.LayerNorm((A * H))
        self.ln1 = t.nn.LayerNorm((A * H))
        self.ln2 = t.nn.LayerNorm((D, H))
        self.ln3 = t.nn.LayerNorm((D, H))
        self.q_mem = t.nn.Parameter(t.randn(M, A * H) / 10)
        self.k_mem = t.nn.Parameter(t.randn(M, A * H) / 10)
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

    @property
    def device(self):
        return self.dummy.device

    @staticmethod
    def normalize(x):
        return x / (x**2).sum(dim=-1, keepdim=True)**(1/2)

    @t.no_grad()
    def inp_position_code(self, OFF:int, LEN:int):
        """
        compute output position code (IPC) from positions
        """
        N = self.C // 4
        ALPHA = 0.0
        i = t.arange(0, LEN, device=self.device) + OFF
        i = i.unsqueeze(-1) / self.L * t.pi
        f = t.arange(0, 2*N, device=self.device)
        c = (2*N - f.abs()) * ALPHA / (2*N) + (1-ALPHA)
        c[0] *= 1/2
        u = t.concat([(-i * f).cos() * c, (-i * f).sin() * c], dim=-1)
        return u.reshape(LEN, self.C)

    @t.no_grad()
    def out_position_code(self, OFF:int, LEN:int):
        """
        compute output position code (OPC) from positions
        """
        N = self.C // 4
        OFF += -2 * self.L / N
        i = t.arange(0, LEN, device=self.device) + OFF
        i = i.unsqueeze(-1) / self.L * t.pi
        f = t.arange(0, 2*N, device=self.device)
        v = t.concat([(-i * f).cos(), (-i * f).sin()], dim=-1).cumsum(dim=0) / self.L
        v = v * (1 + t.randn_like(v) * 0.1)
        return v.reshape(LEN, self.C)

    def dot(self, x, y):
        """
        compute multi-head attention
        """
        a = t.einsum("...ijh, ...kjh -> ...ikh", x.unflatten(-1, (self.A, self.H)), y.unflatten(-1, (self.A, self.H)))
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
        return x + self.ffn(self.ln3(self.ln2(v1) + v2 + x.unsqueeze(dim=-1)).flatten(-2, -1))

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
        return x + self.ffn(self.ln3(self.ln2(v1) + v2 + x.unsqueeze(dim=-1)).flatten(-2, -1)), cache

    def init_cache(self) -> tuple[int, t.Tensor, t.Tensor]:
        return 0, t.zeros(self.M, self.C, self.D, self.H, device=self.device), None

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