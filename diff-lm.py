import torch as t

class Embedding(t.nn.Module):
    def __init__(self, V: int, D: int):
        super().__init__()
        """
        An alternative interpretation of embedding. 
        INPUT:
            V: vocabulary size
            D: # of hidden state dimensions
        ATTRIBUTES: 
            μ: [V, D]   mean of each hidden state
            σ: [V]      standard deviation of each hidden state
            λlog: [V]   each label's present by probability λ, stored as ln(λ)
        """
        self.μ = t.nn.Parameter(t.randn(V, D))
        self.σ = t.nn.Parameter(t.randn(V))
        self.λlog = t.nn.Parameter(t.randn(V))

    def forward(self, label: t.LongTensor) -> t.FloatTensor:
        """
        INPUT:
            label:  [B, L] word label
        OUPUT:
            [B, L, D] hidden state for each label
        """
        return self.μ[label.flatten()].reshape(*label.shape, -1)

    def vsize(self) -> int:
        """
        get vocabulary size
        """
        return self.μ.shape[0]

    def diffuse(self, z: t.FloatTensor, scale: float) -> t.FloatTensor:
        """
        INPUT:
            z:      [B, L, D]   hidden state
            scale:  IS SCALAR   the scale factor of noise
        OUTPUT:
            [B, L, D] diffused hidden state
        """
        # sample gaussian variables, add to x
        return scale * t.randn_like(z) + z

    def predict(self, z: t.FloatTensor) -> t.FloatTensor:
        """
        INPUT: 
            z:  [B, L, D] hidden state
        OUTPUT:
            [B, L, V] label probability, 
                where V is the vocabulary size. 
        """
        # get lambda for each word
        λ = self.λlog.softmax(0)
        # compute distance to each distribution mean
        d = -2 * t.einsum("ij, ...j -> ...i", self.μ, z)
        d = d + t.einsum("ij, ij -> i", self.μ, self.μ)
        d = d + t.einsum("...j, ...j -> ...", z, z).unsqueeze(-1)
        # compute probability of joint distribution with label
        p = λ * (self.σ**-1) * (2*t.pi)**(-1/2) * (-0.5*p*self.σ**-2).exp()
        # return conditional probability of each label given hidden state
        return p / p.sum(-1, keepdim=True)

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

    def loss_local(self, i: int, β: float, label: t.LongTensor):
        """
        train one block with one batch. 
        INPUT:
            i: the trained block
            β: diffusion rate (must be positive)
            label: [B, L] a batch of sentence
        OUTPUT:
            loss: loss of this prediction
        """
        M = len(self.blocks)
        x = self.embed(label[:, :-1])
        y = self.embed(label[:, 1: ])
        λ = (i + 1) / M
        z_next = self.embed.diffuse((1 - λ) * x + λ * y , (1 + β) ** (M - 1 - i) - 1)
        z_fuse = self.embed.diffuse(z_next + (x - y) / M, (1 + β) ** (M - 1 - i) * β)
        z_pred = self.blocks[i].forward(z_fuse)
        return ((z_next - z_pred) ** 2).sum()
    
    def loss_final(self, label: t.LongTensor):
        """
        train the embedding by entropy. 
        """
        π = self.embed.predict(self.embed.forward(label))
        return π.flatten(0, -2)[label.flatten()].log().sum()

    def forward(self, β: float, label: t.LongTensor) -> t.FloatTensor:
        """
        next word prediction. 
        INPUT:
            β: diffusion rate (must be positive)
            label: [B, L] label of words
        OUTPUT:
            probability distribution of the next label for each label
        """
        M = len(self.blocks)
        z = self.embed.diffuse(self.embed(label), (1 + β) ** M - 1)
        for b in self.blocks: z = b(z)
        return self.embed.predict(z)

if __name__ == '__main__':
    label = t.randint(0, 127, (128, 1024), device='cuda:0')
    model = DiffusionLM(32, 128, 64, 3, 5, 17, 1024).to('cuda:0')
    print(model.forward(0.5, label))