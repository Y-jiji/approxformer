import torch as t
from typing import *

"""
The main model. 
"""

class ApproxFormer(t.nn.Module):
    def __init__(self, N: int, L: int) -> None:
        super().__init__()
        self.embed = t.nn.Embedding(N, 128)
        self.inner = IterSequential(
            RoPE(128),
            DecoderLayer(128, 4, 128, 128, 128, L),
            DecoderLayer(128, 4, 128, 128, 128, L),
        )
        self.output = t.nn.Linear(128, N, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.output(self.inner(self.embed(x))).softmax(dim=-1)

    def init_cache(self) -> list[Any]:
        return self.inner.init_cache()

    def iterate(self, x: t.Tensor, cache: list[Any]) -> tuple[t.Tensor, list[Any]]:
        x, cache = self.inner.iterate(self.embed(x), cache)
        return self.output(x).softmax(dim=-1), cache

"""
Some tool layers are listed below. 
"""

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

class DeMax(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x) -> None:
        return x - (1-1e-3) * (x.max(-1, True).values - 15).relu()

class RoPE(t.nn.Module):
    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N
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
            w = y * 2 ** -(f % (self.N // 2)) * t.pi + (2 * f > self.N) * t.pi / 2
        return t.concat([x[..., self.N//2:], x[..., :self.N//2]], dim=-1) * w.sin() + x * w.cos()

    def iterate(self, x: t.Tensor, offset: int) -> tuple[t.Tensor, int]:
        return self.forward(x, offset), offset + 1

    def init_cache(self) -> int:
        return 0

class ReceptCNN(t.nn.Module):
    """
    CNN + receptence gate
    """
    def __init__(self, D, W) -> None:
        super().__init__()
        self.W = W
        self.D = D
        self.w = t.nn.Parameter(t.tensor(D, 1, W))
        self.r = t.nn.Sequential(t.nn.Linear(D, D), t.nn.Sigmoid())

    def forward(self, x: t.Tensor) -> t.Tensor:
        r = self.r(x)
        x = t.concat([x[0:1].repeat_interleave(self.W - 1, -2), x], -2)
        w = w.softmax(dim=-1)
        g = t.conv1d(x.transpose(-2, -1), w, groups=self.D).transpose(-2, -1)
        return (1 - r) * g + r * x

    def init_cache(self) -> t.Tensor | None:
        return None

    def iterate(self, x, cache) -> tuple[t.Tensor, t.Tensor | None]:
        cache = (
            x                           if cache is None else
            t.concat([cache, x], -2)    if cache.shape[-2] < self.W else
            t.concat([cache[1:], x], -2)
        )
        return self.forward(cache), cache

class GapAttention(t.nn.Module):
    POW = 4
    def __init__(self, D, H, C, M, L) -> None:
        super().__init__()
        # dummy variable as device indicator
        self.dummy = t.nn.Parameter(t.tensor(0.0))
        # miscellaneous layers
        self.pe = RoPE(D)
        self.ln = t.nn.LayerNorm(D * H, elementwise_affine=False)
        # key, query and value
        self.k = t.nn.Sequential(
            t.nn.Linear(D, M * H), DeMax(), t.nn.Unflatten(-1, (M, H)))
        self.q = t.nn.Sequential(
            t.nn.Linear(D, M * H), DeMax(), t.nn.Unflatten(-1, (M, H)))
        self.v = t.nn.Sequential(
            t.nn.Linear(D, D * H), t.nn.Unflatten(-1, (D, H)))
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
        assert C % 4 == 0
        self._coeffs = coeff(t.arange(0, C+1), self.POW, C//self.POW)
        # some numbers
        self.M = M
        self.C = C
        self.L = L
        self.H = H
        # smaller initialization works better
        t.nn.init.normal_(self.k.named_parameters('weight').__next__()[1], 1e-3, 1e-2)
        t.nn.init.normal_(self.q.named_parameters('weight').__next__()[1], 1e-3, 1e-2)
    
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
        k, q, v = t.exp(self.k(x)), t.exp(self.q(x)), self.v(x)
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
        k, q, v = t.exp(self.k(x)), t.exp(self.q(x)), self.v(x)
        ipc, opc = self.inp_position_code(off, 1), self.out_position_code(off, 1)
        mem = mem + t.einsum("...imh, ...idh, ...ic -> ...mcdh", k, v, ipc)
        out = t.einsum("...mcdh, ...oc -> ...odh", q, opc)
        cache = off + 1, mem
        return out, cache

    def init_cache(self) -> tuple[int, t.Tensor]:
        return 0, t.zeros(self.M, self.C+1, self.D, self.H, device=self.device)

class DecoderLayer(t.nn.Module):
    def __init__(self, D, H, C, M, W, L):
        super().__init__()
        self.att = IterSequential(
            ReceptCNN(D, W),
            GapAttention(D, H, C, M, L)
        )
        self.ffn = t.nn.Sequential(
            t.nn.Flatten(-2, -1),
            t.nn.Linear(D * H, 2048),
            t.nn.ReLU(),
            t.nn.Dropout(0.1),
            t.nn.Linear(2048, D),
        )

    @staticmethod
    def norm(x):
        return x * (x ** 2).sum(-2, True) ** (-1/2)

    def forward(self, x):
        return self.ffn(self.norm(self.att(x)) + x.unsqueeze(-1)) + x

    def iterate(self, x, cache):
        y, cache = self.att.iterate(x, cache)
        return self.ffn(self.norm(y) + x.unsqueeze(-1)) + x, cache

    def init_cache(self):
        return self.att.init_cache()