class BaselineDecoderLayer(t.nn.Module):
    def __init__(self, D) -> None:
        super().__init__()
        self.inner = t.nn.TransformerEncoderLayer(D, 1, batch_first=True)

    def forward(self, x):
        return self.inner(x, t.nn.Transformer.generate_square_subsequent_mask(x.shape[-2], device=x.device))
