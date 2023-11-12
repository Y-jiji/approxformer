import torch as t
import torch.distributions as distr

class Embedding(t.nn.Module):
    def __init__(self, V: int, D: int) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.randn(V, D))
        self.output = t.nn.Parameter(t.randn(D, V))

    def vsize(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: t.Tensor, α: float = 0) -> None:
        """
        INPUT
            x: [..., L, V] shaped probability tensor
        OUTPUT
            [..., L, D] shaped word vectors
        """
        sample = distr.Dirichlet(t.ones_like(x)).sample()
        x = x * (1-α) + sample * α
        return x.matmul(self.weight)
    
    def diffuse(self, x: t.Tensor, n: int, α: float) -> None:
        """
        INPUT
            x: [..., L, V] shaped probability tensor
        OUTPUT
            [..., L, V] shaped, noisy probability tensor
        """
        if n == 0: return x
        sample = distr.Dirichlet(t.ones_like(x)).sample_n(n)
        sample = t.einsum("...i, ...i -> ...", sample, (1-α)**t.arange(n, x.device))
        return ((1-α)**n) * x + (α) * sample

    def predict(self, x: t.Tensor) -> None:
        """
        INPUT
            x: [..., L, D] shaped embedding tensor
        OUTPUT
            [..., L, V] shaped probability tensors
        """
        return x.matmul(self.output).softmax(-1)

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
        self.k = t.nn.Sequential(t.nn.Linear(D, C*H), t.nn.LeakyReLU(), t.nn.Unflatten(-1, (C, H)))
        self.q = t.nn.Sequential(t.nn.Linear(D, C*H), t.nn.LeakyReLU(), t.nn.Unflatten(-1, (C, H)))
        self.v = t.nn.Sequential(t.nn.Linear(D, D*H), t.nn.Unflatten(-1, (D, H)))
        self.ffn = t.nn.Sequential(
            t.nn.LayerNorm((D, )),
            t.nn.Linear(D, Z), 
            t.nn.ReLU(),
            t.nn.Linear(Z, D),
        )

    def forward(self, z: t.FloatTensor) -> t.FloatTensor:
        """
        INPUT: 
            z: [B, L, D] shaped tensor
        OUTPUT:
            [B, L, D] shaped tensor
        """
        v = t.einsum("...ch, ...dh -> ...chd"   , self.k(z), self.v(z)).cumsum(-4)
        v = t.einsum("...ich, ...ichd -> ...id" , self.q(z), v)
        d = t.einsum("...ich, ...ich -> ...i"   , self.q(z), self.k(z).cumsum(-3))
        return self.ffn(v / d.unsqueeze(-1)) + z

class Block(t.nn.Module):
    def __init__(self, N: int, D: int, H: int, C: int, Z: int) -> None:
        super().__init__()
        """
        Blocks can be trained seperately. 
        INPUT:
            N: # of layers
            D, H, C, Z: layer configuration, see `class Layer`
        """
        self.layers = t.nn.Sequential(*[Layer(D, H, C, Z) for _ in range(N)])
    
    def forward(self, z: t.FloatTensor) -> t.FloatTensor:
        return self.layers(z)

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
        self.embed = Embedding(V, D)
        self.blocks = t.nn.ModuleList([Block(N, D, H, C, Z) for _ in range(M)])

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
        label = t.nn.functional.one_hot(label, self.embed.vsize())
        label = self.embed.diffuse(label, α, M-1-i)
        noise = self.embed.forward(label, α)
        value = self.embed.predict(self.blocks[i].forward(noise))
        return t.matmul("...j, ...j -> ...", label, value.log()).mean()

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
        label = t.nn.functional.one_hot(label, self.embed.vsize())
        label = self.embed.diffuse(label, α, M)
        value = self.embed.forward(label, 0)
        for b in self.blocks: 
            value = self.embed.predict(b(value))
        return value

if __name__ == '__main__':
    label = t.randint(0, 127, (128, 1024), device='cuda:0')
    model = DiffusionLM(32, 128, 64, 3, 5, 17, 1024).to('cuda:0')
    print(model.forward(0.5, label))