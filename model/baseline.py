import torch as t

class Baseline(t.nn.Module):
    def __init__(self, N: int) -> None:
        super().__init__()        
        self.embed = t.nn.Embedding(N, 128)
        self.inner = t.nn.Sequential(
            PositionalEmbedding(128),
            BaselineDecoderLayer(D=128, H=4, F=2048),
            BaselineDecoderLayer(D=128, H=4, F=2048),
        )
        self.output = t.nn.Linear(128, N, bias=False)
        with t.no_grad():
            self.output.weight.set_(self.embed.weight / 20)
    
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


class BaselineDecoderLayer(t.nn.Module):
    def __init__(self, D, H, F) -> None:
        super().__init__()
        self.inner = t.nn.TransformerEncoderLayer(D, H, F, batch_first=True, norm_first=True)

    def forward(self, x):
        return self.inner(x, t.nn.Transformer.generate_square_subsequent_mask(x.shape[-2], device=x.device))
