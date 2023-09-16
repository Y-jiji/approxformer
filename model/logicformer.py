import torch as t

class TimeGate(t.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.flip = FlipGate(dim)
        self.decay = t.nn.Parameter(1e-3 * t.ones(dim))
    @property
    def device(self):
        return self.decay.device
    def apply_decay(self, x):
        i = 0
        w = -t.exp(self.decay)
        while 2 ** i < min(x.shape[-2], 2 ** 8):
            y = t.concat([t.ones_like(x[..., :2**i, :]) * -t.inf, x[..., :-2**i, :]], dim=-2)
            x = t.maximum(x, y + w)
            i = i + 1
            w = w * 2
        return x
    def forward(self, x) -> t.Tensor:
        return t.minimum(x, self.flip(self.apply_decay(self.flip(x))))

class NormLinear(t.jit.ScriptModule):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.w = t.nn.Parameter(t.randn(input_features, output_features))
        self.n = t.nn.LayerNorm((output_features, ), elementwise_affine=False)
    def forward(self, x) -> t.Tensor:
        weight = self.w * (self.w ** 2).sum(-1, True) ** (-1/2)
        return x.matmul(weight)

class FlipGate(t.nn.Module):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(2.01 * t.rand(input_features) - 1.005)
    def forward(self, x) -> t.Tensor:
        return x * (self.weight.tanh() + t.randn_like(self.weight) * 1e-2)

class LogicLayer(t.nn.Module):
    def __init__(self, features) -> None:
        super().__init__()
        self.linear = NormLinear(features, features)
        self.time_gate = TimeGate(features)
    def forward(self, x: t.Tensor):
        return self.linear(self.time_gate(x))

class LogicFormer(t.nn.Module):
    def __init__(self, tokens: int, dim: int) -> None:
        super().__init__()
        self.embed = t.nn.Embedding(tokens, dim)
        self.inner = t.nn.Sequential(
            *[LogicLayer(dim)] * 6, 
            TimeGate(dim),
            NormLinear(dim, tokens),
            t.nn.Softmax(-1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.inner(self.embed(x))