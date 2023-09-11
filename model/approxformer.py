import torch as t

class ApproxFormer(t.nn.Module):
    def __init__(self, N: int, L: int) -> None:
        super().__init__()
        self.embed = t.nn.Embedding(N, 128)
        self.inner = t.nn.Sequential(
            PositionalEmbedding(128),
            DecoderLayer(64, 128, 32, 128, L, 4, 128),
            DecoderLayer(64, 128, 32, 128, L, 4, 128),
        )
        self.output = t.nn.Linear(128, N, bias=False)
        with t.no_grad():
            self.output.weight.set_(self.embed.weight / 20)

    @staticmethod
    def normalize(x):
        return x / (x**2).sum(dim=-1, keepdim=True)**(1/2)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.output(self.inner(self.embed(x))).softmax(dim=-1)

class PositionalEmbedding(t.nn.Module):
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
        L = x.shape[-2]
        y = t.arange(0, L, device=self.device).unsqueeze(dim=-1) + OFFSET
        y = y + t.rand_like(y.float())
        f = t.arange(0, self.N, device=self.device) + 2
        return x * (1-self.gamma.sigmoid()) + self.gamma.sigmoid() * (y * t.pi ** (-f // 2) + (f % 2) * t.pi / 2).sin()

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

class DecoderLayer(t.nn.Module):
    def __init__(self, M, D, A, C, L, H, W, p=0.1) -> None:
        """
        M: size of memory
        H: number of heads
        D: size of token representation
        A: size of attention matcher
        C: size of position code
        L: maximal length
        W: 1d cnn kernel width
        p: dropout rate
        """
        super().__init__()
        assert C % 4 == 0
        self.cnn = CNNLayer(D * H, W)
        self.pe = PositionalEmbedding(D)
        self.ln0 = t.nn.LayerNorm((A * H))
        self.ln1 = t.nn.LayerNorm((A * H))
        self.ln2 = t.nn.LayerNorm((D, H))
        self.ln3 = t.nn.LayerNorm((D, H))
        self.q_mem = t.nn.Parameter(t.randn(M, A * H))
        self.k_mem = t.nn.Parameter(t.randn(M, A * H))
        self.k_inp = t.nn.Linear(D, A * H)
        self.q_out = t.nn.Linear(D, A * H)
        self.v_inp = t.nn.Linear(D, D * H)
        self.dummy = t.nn.Parameter(t.tensor([]))
        self.ffn = t.nn.Sequential(
            t.nn.Linear(D * H, 2048),
            t.nn.ReLU(),
            t.nn.Dropout(p),
            t.nn.Linear(2048, D)
        )
        self.C = C
        self.D = D
        self.A = A
        self.H = H
        self.M = M
        self.L = L

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
        ALPHA = 0.75
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
        v = v * (1 + t.randn_like(v) * 0.05)
        return v.reshape(LEN, self.C)

    def dot(self, x, y):
        """
        compute multi-head attention
        """
        a = t.einsum("...ijh, ...kjh -> ...ikh", x.unflatten(-1, (self.A, self.H)), y.unflatten(-1, (self.A, self.H)))
        return (a - a.max(dim=-2, keepdim=True).values + 2).exp()

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
        v1 = self.ln2(
            t.einsum("...lmh, ...lc, ...mcvh -> ...lvh", a1, opc, v0)
            / t.einsum("...lmh, ...mvh -> ...lvh", a1, a0.sum(dim=-2, keepdim=True))
        )
        v2 = self.cnn(self.v_inp(x)).unflatten(-1, (self.D, self.H))
        return x + self.ffn(self.ln3(v1 + v2 + x.unsqueeze(dim=-1)).flatten(-2, -1))