import torch as t
import torch.utils.data as data

class RoPE(t.nn.Module):
    def __init__(self, E: int, β: float=2) -> None:
        super().__init__()
        self.register_buffer("β", t.tensor(β).requires_grad_(False))
        self.register_buffer("α", 1/t.arange(1, E+1).requires_grad_(False))
    def forward(self, x: t.Tensor) -> t.Tensor:
        α, β = self.get_buffer("α"), self.get_buffer("β")
        L = x.shape[-2]
        γ = β ** (α * t.arange(1, L+1, device=x.device).unsqueeze(-1).requires_grad_(False))
        return t.concat([x * t.cos(γ), x * t.sin(γ)], dim=-1)

class Exp(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.clamp_max(5).exp()

class Layer(t.nn.Module):
    def __init__(self, E: int, C: int, H: int, p=0.1) -> None:
        super().__init__()
        self.k = t.nn.Sequential(t.nn.Linear(E, C*H), Exp(), RoPE(C*H), t.nn.Unflatten(-1, (H*2, C)))
        self.q = t.nn.Sequential(t.nn.Linear(E, C*H*2), Exp(), t.nn.Unflatten(-1, (H*2, C)))
        self.v = t.nn.Sequential(t.nn.Linear(E, E*H*2), t.nn.Unflatten(-1, (H*2, E)))
        self.ln = t.nn.LayerNorm((E,))
        self.ffn = t.nn.Sequential(
            t.nn.Linear(E, 1024),
            t.nn.ReLU(),
            t.nn.Dropout(p),
            t.nn.Linear(1024, E),
        )

    def forward(self, mem: t.Tensor, x: t.Tensor) -> t.Tensor:
        q, k, v = self.q(x), self.k(mem), self.v(mem)
        r = t.einsum("...ihj, ...ihk -> ...ihjk", k, v).cumsum(-4)
        r = t.einsum("...ihj, ...ihjk -> ...ik", q, r)
        d = t.einsum("...ihj, ...ihj -> ...i", q, k.cumsum(-3)).unsqueeze(-1)
        return self.ffn(self.ln(r/d)) + x

class Embedding(t.nn.Module):
    def __init__(self, V: int, E: int) -> None:
        import math
        super().__init__()
        # mixing amptitude w.r.t. z-index
        self.weight = t.nn.Parameter(t.randn(V, E) * 0.8)
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.weight.index_select(0, x.flatten()).reshape(*x.shape, -1)
    def π(self, x: t.Tensor) -> t.Tensor:
        return t.einsum("ij, ...j -> ...i", self.weight, x).softmax(-1)
    def rand_like(self, n: int, ξ: float, x: t.Tensor) -> t.Tensor:
        if n == 0: return 0
        V = self.weight.shape[0]
        r = t.randint(0, V-1, (*x.shape[:-1], n), device=x.device)
        coeff = ξ ** t.arange(n, device=x.device).unsqueeze(-1)
        weight = self.weight.index_select(0, r.flatten()).reshape(*r.shape, -1)
        return (weight * coeff).sum(-2) * (1-ξ)

class Block(t.nn.Module):
    def __init__(self, E: int, C: int, H: int):
        super().__init__()
        self.layers = t.nn.ModuleList([Layer(E, C, H) for i in range(2)])
        self.ln = t.nn.LayerNorm((E, ), elementwise_affine=False)

    def parallel_forward(self, z: t.Tensor, y: t.Tensor):
        """
        z: y with noise shifted left
        y: the reference
        """
        for layer in self.layers:
            z = layer(y, z)
        return z

    def bootstrap(self, n: int, β: float, label: t.Tensor, embed: Embedding, last=False) -> float:
        ξ = 1 - β
        x = embed(label)
        r = embed.rand_like(n, ξ, x)
        y = r + ξ**n * x
        if not last:
            # if it is not the last layer, some recovered information can be leaked. 
            z = embed.rand_like(1, ξ, x) + ξ * y
        else:
            # if it is the last layer, nothing can be seen therefore we just sample it. 
            z = embed.rand_like(1, ξ, x) + ξ * r
        # training forward
        w = self.parallel_forward(z[:, 1:, :], x[:, :-1, :])
        loss_item = None
        if n == 0:
            # fit real label on last layer
            π_w = embed.π(w).flatten(0, -2)
            label = label[..., 1:].flatten()
            range = t.arange(label.shape[0], device=label.device)
            loss = -π_w[range, label].clamp_min(1e-20).log().mean()
            loss_item = loss.item()
            loss.backward(retain_graph=True)
        else:
            # fit blur label on other layers
            π_y, π_w = embed.π(y[:, 1:, :]), embed.π(w)
            loss = (π_y * (π_y.clamp_min(1e-20).log() - π_w.clamp_min(1e-20).log())).sum(-1).mean()
            loss_item = loss.item()
            loss.backward(retain_graph=True)
        return loss_item

class DiffLM(t.nn.Module):
    def __init__(self, L: int, E: int, C: int, H: int, V: int, β: float):
        """
        L: number of layers
        E: embedding dimensions
        V: vocabulary size
        """
        super().__init__()
        self.embedding = Embedding(V, E)
        self.blocks = t.nn.ModuleList([Block(E, C, H) for _ in range(L)])
        self.register_buffer("β", t.tensor(β).requires_grad_(False))

    @t.no_grad()
    def forward(self, x: t.Tensor):
        """
        TODO: fix it, distribution changed, no randn anymore
        """
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

    def train(self, dataset: data.Dataset, epoch: int):
        L = len(self.blocks)
        β = self.get_buffer("β").item()
        loader = data.DataLoader(dataset, 128)
        optimizer = t.optim.Adam(self.parameters())
        for e in range(epoch):
            for x in loader:
                loss = 0.0
                for i, block in enumerate(self.blocks):
                    loss = max(loss, block.bootstrap(L-1-i, β, x, self.embedding, last=(i==0)))
                print(f'{e}: {loss}')
                optimizer.step()
                self.zero_grad()

if __name__ == '__main__':
    # if the model is good, it should overfit this sample easily
    xs = t.randint(0, 127, (128, 80, )).to('cuda:0')
    model = DiffLM(32, 64, 32, 4, 128, β=0.1).to('cuda:0')
    model.train(xs, 1000)