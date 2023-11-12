import torch as t
import torch.distributions as distr

class Embedding(t.nn.Module):
    def __init__(self, V: int, D: int) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.randn(V, D))
        self.output = t.nn.Parameter(t.randn(D, V))

    def vsize(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        INPUT
            x: [..., L, V] shaped probability tensor
        OUTPUT
            [..., L, D] shaped word vectors
        """
        return x.matmul(self.weight)

    def predict(self, x: t.Tensor) -> None:
        """
        INPUT
            x: [..., L, D] shaped embedding tensor
        OUTPUT
            [..., L, V] shaped probability tensors
        """
        return x.matmul(self.output)

class Exp(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: t.Tensor) -> t.Tensor:
        return (x - (x - 15).relu().max().detach_()).exp()

class Layer(t.nn.Module):
    def __init__(self, D: int, H: int, C: int, Z: int) -> None:
        """
        Standard MH Linear attention + FFN Layer
        INPUT:
            D: input & output dimension
            H: # attention heads
            C: # channels
            Z: hidden state dimension
        """
        super().__init__()
        self.k = t.nn.Sequential(t.nn.Linear(D, C*H), Exp(), t.nn.Unflatten(-1, (C, H)))
        self.q = t.nn.Sequential(t.nn.Linear(D, C*H), Exp(), t.nn.Unflatten(-1, (C, H)))
        self.v = t.nn.Sequential(t.nn.Linear(D, D*H), t.nn.Unflatten(-1, (D, H)))
        self.ffn = t.nn.Sequential(
            t.nn.LayerNorm((D, )),
            t.nn.Linear(D, Z), 
            t.nn.ReLU(),
            t.nn.Linear(Z, D),
        )

    def forward(self, prior: t.FloatTensor, z: t.FloatTensor) -> t.FloatTensor:
        """
        INPUT: 
            z: [B, L, D] shaped tensor
        OUTPUT:
            [B, L, D] shaped tensor
        """
        v = t.einsum("...ch, ...dh -> ...chd"   , self.k(prior), self.v(prior)).cumsum(-4)
        v = t.einsum("...ich, ...ichd -> ...id" , self.q(z), v)
        d = t.einsum("...ich, ...ich -> ...i"   , self.q(z), self.k(prior).cumsum(-3))
        return self.ffn(v / d.unsqueeze(-1)) + z

class Block(t.nn.Module):
    def __init__(self, V: int, N: int, D: int, H: int, C: int, Z: int) -> None:
        super().__init__()
        """
        Blocks can be trained seperately. 
        INPUT:
            V: vocabulary size
            N: # of layers
            D, H, C, Z: layer configuration, see `class Layer`
        """
        self.embeds = Embedding(V, D)
        self.layers = t.nn.ModuleList([Layer(D, H, C, Z) for _ in range(N)])
    
    def forward(self, prior: t.FloatTensor, z: t.FloatTensor) -> t.FloatTensor:
        prior = self.embeds.forward(prior)
        z = self.embeds.forward(z)
        for layer in self.layers:
            z = layer(prior, z)
        z = self.embeds.predict(z)
        return z.clamp(-10, 10) * 0.01 + z * 0.99

class DiffusionLM(t.nn.Module):
    def __init__(self, M: int, V: int, D: int, N: int, H: int, C: int, Z: int) -> None:
        super().__init__()
        """
        LLM trained with diffusion. 
        INPUT:
            M: # of blocks
            V: vocabulary size
            N, D, H, C, Z: block configuration, see `class Block`
        ATTRIBUTES:
            emb_direct: directly trained embedding
            emd_moment: embedding updated with momentum
            blocks: denoising models that can be trained seperately
        """
        self.blocks = t.nn.ModuleList([Block(V, N, D, H, C, Z) for _ in range(M)])
        self.β = 0.25

    def diffuse(self, x: t.Tensor, α: float, n: int) -> t.Tensor:
        """
        INPUT
            x: [..., L, V] shaped probability tensor
        OUTPUT
            [..., L, V] shaped, noisy probability tensor
        """
        if n == 0: return x
        with t.no_grad():
            sample = distr.Dirichlet(self.β * t.ones(n, *x.shape, dtype=t.float, device=x.device)).sample()
            sample = t.einsum("i, i... -> ...", (1-α)**t.arange(n, device=x.device), sample)
        assert ((((1-α)**n) * x + α * sample).sum(-1) - 1.0).abs().max().item() < 1e-5
        return ((1-α)**n) * x + α * sample

    def diffuse_many(self, x: t.Tensor, α: float, n: int) -> t.Tensor:
        with t.no_grad():
            if n > 1:
                sample = distr.Dirichlet(self.β * t.ones(n-1, *x.shape, dtype=t.float, device=x.device)).sample()
                sample = t.concat([t.zeros(1, *x.shape, dtype=t.float, device=x.device), sample], 0)
            else:
                sample = t.zeros(1, *x.shape, dtype=t.float, device=x.device)
            sample = t.einsum("i, i... -> i...", (1-α)**-t.arange(n, device=x.device), sample).cumsum(0)
            sample = t.einsum("i, i... -> i...", (1-α)** t.arange(n, device=x.device), sample)
        x = t.einsum("i, ... -> i...", (1-α)**t.arange(n, device=x.device), x)
        assert ((x + α * sample).sum(-1) - 1).abs().max().item() < 1e-5, (x + α * sample).sum(-1)
        return x + α * sample

    def loss_local(self, i: int, label: t.LongTensor, α: float):
        """
        train one block with one batch. 
        INPUT:
            i: the trained block
            α: diffusion rate (must be positive and between 0, 1)
            label: [B, L] a batch of sentence
        OUTPUT:
            loss: loss of this prediction
        """
        M = len(self.blocks)
        V = self.blocks[0].embeds.vsize()
        block = self.blocks[i]
        label = t.nn.functional.one_hot(label, V).float()
        if i == 0:
            noise = self.diffuse(t.ones_like(label) / V, α, M-1)
            label = noise + (1-α)**(M-1) * (label - t.ones_like(label) / V)
            noise = self.diffuse(noise, α, 1)[..., 1:, :]
        else:
            label = self.diffuse(label, α, M-1-i)
            noise = self.diffuse(label, α, 1)[..., 1:, :]
        value = block.forward(label[...,:-1, :], noise).log_softmax(-1)
        return -t.einsum("...j, ...j -> ...", label[..., 1:, :], value).mean()

    def forward(self, label: t.LongTensor, α: float = 0) -> t.FloatTensor:
        """
        next word prediction. 
        INPUT:
            α: diffusion rate (must be positive and between 0, 1)
            label: [B, L] label of words
        OUTPUT:
            probability distribution of the next label for each label
        """
        M = len(self.blocks)
        V = self.blocks[0].embeds.vsize()
        label = t.nn.functional.one_hot(label, V).float()
        value = self.diffuse(t.ones_like(label) / V, α, M)
        label = self.diffuse_many(label, α, M)
        for i, b in enumerate(self.blocks): 
            value = b(label[M-1-i], value).softmax(-1)
        return value

if __name__ == '__main__':
    import torch.utils.data as data
    from tqdm import tqdm
    label = t.randint(0, 127, (129, 128)).to('cuda:0')
    model = DiffusionLM(4, 128, 64, 4, 20, 1, 1024).to('cuda:0')
    optim = t.optim.Adam(model.parameters())
    pbar = tqdm(range(2000))
    with open('log', 'w') as f:
        for e in pbar:
            for batch in data.DataLoader(label, 32):
                loss_sum = 0.0
                for i in range(len(model.blocks)):
                    loss = model.loss_local(i, batch, 0.2)
                    loss.backward()
                    optim.step()
                    model.zero_grad()
                    loss_sum += loss.item()
                pbar.set_description_str(f'{loss_sum}', refresh=True)
            acc = 0.0
            for batch in data.DataLoader(label, 32):
                acc += (model.forward(batch[:, :-1], 0.2).flatten(0, -2).argmax(-1) == batch[:, 1:].flatten()).sum().item()
            print(acc / label.numel(), file=f)
            if e % 100 == 0:
                f.flush()