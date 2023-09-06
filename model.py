import torch as t

class PositionalEmbedding(t.nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.dummy = t.nn.Parameter(t.tensor(0.0))
        assert n % 2 == 0

    @property
    def device(self):
        return self.dummy.device

    def forward(self, x, OFFSET=0):
        L = x.shape[-2]
        y = t.arange(0, L, device=self.device).unsqueeze(dim=-1) + OFFSET
        f = t.arange(0, self.n, device=self.device) + 2
        return x + (y / (f // 2) + (f % 2) * t.pi / 2).sin()

class Decoder(t.nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.embed = t.nn.Embedding(n, 256)
        self.positional_embed = PositionalEmbedding(256)
        self.layer = t.nn.Sequential(
            LongTermDecoderLayer(256, 8, 256, 256),
            LongTermDecoderLayer(256, 8, 256, 256),
            LongTermDecoderLayer(256, 8, 256, 256),
        )
        self.output = t.nn.Sequential(
            t.nn.Linear(256, n, bias=False),
            t.nn.Softmax(dim=-1)
        )

    @staticmethod
    def normalize(x):
        return x / x.norm(dim=-1, keepdim=True)

    def forward(self, x: t.Tensor):
        embed = self.positional_embed(self.embed(x))
        return self.output(self.layer(self.normalize(embed)))

    @t.no_grad()
    def iterative_forward(self, x: t.Tensor, cache=None, OFF=0):
        x = self.positional_embed(self.embed(x))
        L = x.shape[-2]
        N = len(list(self.layer.modules()))
        if cache == None:
            cache = [None] * N
        y = []
        for OFF in range(L):
            _y = x[..., OFF, :].unsqueeze(dim=-2)
            for i, module in enumerate(self.layer.children()):
                _y, cache[i] = module.iterative_forward(_y, cache[i], OFF, L)
            y.append(self.output(_y).argmax(dim=-1))
        return t.stack(y, dim=-1), cache

class ShortTermDecoderLayer(t.nn.Module):
    def __init__(self, attention_length, input_features, value_features) -> None:
        super().__init__()
        self.q_out = t.nn.Linear(input_features, value_features)
        self.k_inp = t.nn.Linear(input_features, value_features)
        self.v_inp = t.nn.Linear(input_features, value_features)
        self.attention_length = attention_length

    def forward(self, x: t.Tensor):
        pass

    def iterative_forward(self, x: t.Tensor):
        pass

class LongTermDecoderLayer(t.nn.Module):
    def __init__(self, memory_size, position_code_features, input_features, value_features) -> None:
        super().__init__()
        self.q_mem = t.nn.Parameter(t.randn(memory_size, value_features) / 10)
        self.k_mem = t.nn.Parameter(t.randn(memory_size, value_features) / 10)
        self.k_inp = t.nn.Linear(input_features, value_features)
        self.v_inp = t.nn.Linear(input_features, value_features)
        self.q_out = t.nn.Linear(input_features, value_features)
        self.dummy = t.nn.Parameter(t.tensor([]))
        self.ffn = t.nn.Sequential(
            t.nn.Linear(value_features, 2048),
            t.nn.ReLU(),
            t.nn.Linear(2048, 2048),
            t.nn.ReLU(),
            t.nn.Linear(2048, value_features)
        )
        self.C = position_code_features
        self.D = value_features
        self.M = memory_size

    @property
    def device(self):
        return self.dummy.device

    @staticmethod
    def normalize(x):
        return x / x.norm(dim=-1, keepdim=True)

    @t.no_grad()
    def inp_position_code(self, LEN, OFF, L=None):
        """
        compute output position code (IPC) from positions
        """
        if L is None: L = LEN
        N = self.M * (self.C // 4)
        ALPHA = 0.5
        i = t.arange(0, L, device=self.device) + OFF
        i = i.unsqueeze(-1) / LEN * t.pi
        f = t.arange(0, 2*N, device=self.device)
        c = (2*N - f.abs()) * ALPHA / (2*N) + (1-ALPHA)
        c[0] *= 1/2
        u = t.concat([(-i * f).cos() * c, (-i * f).sin() * c], dim=-1)
        return u.reshape(L, self.M, 4 * (self.C // 4))

    @t.no_grad()
    def out_position_code(self, LEN, OFF, L=None):
        """
        compute output position code (OPC) from positions
        """
        if L is None: L = LEN
        N = self.M * (self.C // 4)
        OFF += -2 * LEN / N
        i = t.arange(0, L, device=self.device) + OFF
        i = i.unsqueeze(-1) / LEN * t.pi
        f = t.arange(0, 2*N, device=self.device)
        v = t.concat([(-i * f).cos(), (-i * f).sin()], dim=-1).cumsum(dim=0) / LEN
        v = v + 1e-2 * t.randn_like(v)
        return v.reshape(L, self.M, 4 * (self.C // 4))

    @t.no_grad()
    def attention_matrix(self, x: t.Tensor):
        """
        parallel mode forward
        x: [..., L, INPUT]
        """
        L = x.shape[-2]
        def dot(x, y):
            return (t.einsum("...ij, ...kj -> ...ik", x, y)).exp()
        a0 = dot(self.q_mem, self.normalize(self.k_inp(x)))
        ipc = self.inp_position_code(L, 0)
        a1 = dot(self.normalize(self.q_out(x)), self.k_mem)
        opc = self.out_position_code(L, 0)
        approx = t.einsum("ij, jic, ki, kic -> kj", a0, ipc, a1, opc)
        really = t.einsum("ji, kj -> ki", a0, a1) * t.eye(L, device=self.device).cumsum(dim=0)
        return approx / approx.sum(dim=1, keepdim=True), really / really.sum(dim=1, keepdim=True)

    @t.no_grad()
    def plot_changes(self, l=100, sample_rate_0=0.1, sample_rate_1=0.1):
        ipc = self.inp_position_code(l)
        opc = self.out_position_code(l)
        sam = t.randperm(l)[:int(l*sample_rate_0)].sort().values
        x = t.linspace(0, l-1, int(l*sample_rate_1)).to(t.int)
        mat = t.einsum("ij, kj -> ik", opc[sam], ipc[x])
        return x, mat, sam

    @t.no_grad()
    def iterative_forward(self, x: t.Tensor, cache: tuple[t.Tensor, t.Tensor], OFFSET: int, L: int):
        """
        iterative mode forward
        x: [..., 1, INPUT]
        """
        if cache == None:
            v, d = 0.0, 0.0
        else:
            v, d = cache
        def dot(x, y):
            return t.einsum("...ij, ...kj -> ...ik", x, y).exp()
        a0 = dot(self.q_mem, self.normalize(self.k_inp(x)))
        ipc = self.inp_position_code(L, OFFSET, 1)
        v = v + t.einsum("...ij, ...jic, ...jv -> ...icv", a0, ipc, self.v_inp(x))
        d = d + t.einsum("...ij, ...jic -> ...ic", a0, ipc)
        a1 = dot(self.normalize(self.q_out(x)), self.k_mem)
        opc = self.out_position_code(L, OFFSET, 1)
        _v = (
            t.einsum("...ji, ...jic, ...icv -> ...jv", a1, opc, v) /
            t.einsum("...ji, ...jic, ...ic -> ...j", a1, opc, d).unsqueeze(-1)
        )
        return self.ffn(_v + x) + x, (v, d)

    def forward(self, x: t.Tensor):
        """
        parallel mode forward
        x: [..., L, INPUT]
        TODO: add amending attention from nearby tokens
        """
        L = x.shape[-2]
        def dot(x, y):
            return t.einsum("...ij, ...kj -> ...ik", x, y).exp()
        # -- channel sharing version -- #
        a0 = dot(self.q_mem, self.normalize(self.k_inp(x)))
        ipc = self.inp_position_code(L, 0)
        v = t.einsum("...ij, ...jic, ...jv -> ...icv", a0, ipc, self.v_inp(x))
        d = t.einsum("...ij, ...jic -> ...ic", a0, ipc)
        a1 = dot(self.normalize(self.q_out(x)), self.k_mem)
        opc = self.out_position_code(L, 0)
        v = (
            t.einsum("...ji, ...jic, ...icv -> ...jv", a1, opc, v) / 
            t.einsum("...ji, ...jic, ...ic -> ...j", a1, opc, d).unsqueeze(-1)
        )
        # -- naive version -- #
        # a0 = dot(self.q_mem, self.normalize(self.k_inp(x)))
        # a1 = dot(self.normalize(self.q_out(x)), self.k_mem)
        # a = t.einsum("...ij, ...jk -> ...ik", a1, a0) * t.eye(L, device=self.device).cumsum(0)
        # v = t.einsum("...ij, ...jk -> ...ik", a, self.v_inp(x))
        # d = a.sum(dim=-1).unsqueeze(-1)
        return 0.01 * self.ffn(v + x) + x