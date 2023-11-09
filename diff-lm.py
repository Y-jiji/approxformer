import torch as t
import torch.utils.data as data

class RoPE(t.nn.Module):
    def __init__(self, E: int, β: float=0.01) -> None:
        super().__init__()
        self.register_buffer("β", t.tensor(β).requires_grad_(False))
        self.register_buffer("α", t.arange(1, E+1).requires_grad_(False))
    def forward(self, x: t.Tensor) -> t.Tensor:
        α, β = self.get_buffer("α"), self.get_buffer("β")
        L = x.shape[-2]
        γ = β ** (α * t.arange(1, L+1).unsqueeze(-1).requires_grad_(False))
        return t.concat([x * t.cos(γ), x * t.sin(γ)], dim=-1)

class Exp(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.exp()

class Layer(t.nn.Module):
    def __init__(self, E: int, C: int, H: int, p=0.1) -> None:
        super().__init__()
        self.k = t.nn.Sequential(t.nn.Linear(E, C*H), Exp(), RoPE(C*H), t.nn.Unflatten(-1, (H*2, C)))
        self.q = t.nn.Sequential(t.nn.Linear(E, C*H), Exp(), RoPE(C*H), t.nn.Unflatten(-1, (H*2, C)))
        self.v = t.nn.Sequential(t.nn.Linear(E, E*H*2), t.nn.Unflatten(-1, (H*2, E)))
        self.ln = t.nn.LayerNorm((E,))
        self.ffn = t.nn.Sequential(
            t.nn.Linear(E, 2*E),
            t.nn.ReLU(),
            t.nn.Dropout(p),
            t.nn.Linear(2*E, E),
        )

    def forward(self, mem: t.Tensor, x: t.Tensor) -> t.Tensor:
        q, k, v = self.q(x), self.k(mem), self.v(mem)
        r = t.einsum("...ihj, ...ihk -> ...ihjk", k, v).cumsum(-4)
        r = t.einsum("...ihj, ...ihjk -> ...ik", q, r)
        d = t.einsum("...ihj, ...ihj -> ...i", q, k.cumsum(-3)).unsqueeze(-1)
        return self.ffn(self.ln(r / d)) + x

class Block(t.nn.Module):
    def __init__(self, E: int, C: int, H: int):
        super().__init__()
        self.layers = t.nn.ModuleList([Layer(E, C, H) for i in range(3)])
        self.ln = t.nn.LayerNorm((E, ), elementwise_affine=False)

    def forward(self, z: t.Tensor, y: t.Tensor):
        """
        z: y with noise shifted left
        y: the reference
        """
        for layer in self.layers:
            z = layer(y, z)
        return z

    def bootstrap(self, n: int, β: float, dataloader: data.DataLoader):
        optim = t.optim.Adam(self.parameters())
        ξ = 1 - β
        for x in dataloader:
            y = (ξ ** (n/2) * x + n * t.randn_like(x))
            z = (ξ ** (1/2) * y + 1 * t.randn_like(x))
            w = self.forward(z[..., 1:, :], y[..., :-1, :])
            loss = (self.ln(w) - self.ln(y[..., 1:, :])) ** 2
            loss.backward()
            optim.step()
            self.zero_grad()

class DiffLM(t.nn.Module):
    def __init__(self, L: int, E: int, C: int, H: int, V: int, β: float):
        """
        L: number of layers
        E: embedding dimensions
        V: vocabulary size
        """
        super().__init__()
        self.embedding = t.nn.Embedding(V, E)
        self.blocks = t.nn.ModuleList([Block(E, C, H) for _ in range(L)])
        self.register_buffer("β", t.tensor(β).requires_grad_(False))

    @t.no_grad()
    def forward(self, x: t.Tensor):
        L = len(self.blocks)
        β = self.get_buffer("β")
        ξ = 1-β
        x = self.embedding(x)
        ε = t.randn(L, *x.shape, device=x.device).cumsum(0)
        z = t.randn_like(x) * L
        for i, block in enumerate(self.blocks):
            j = L-i-1
            z = block.forward(z, (ξ**(j/2)*x + ε[j]))
        return z

    def train(self, dataset: data.Dataset):
        L = len(self.blocks)
        β = self.get_buffer("β").item()
        loader = data.DataLoader(dataset, 128)
        # update each block independently
        # aggregate gradient for each block
        # finally update embedding layer
        # repeat

if __name__ == '__main__':
    x = t.randint(0, 127, (100, 50, ))
    model = DiffLM(1, 128, 32, 4, 128, β=0.1)
    print(model.forward(x) - model.forward(x))