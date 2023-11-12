import torch as t
from torch.distributions import Categorical

class Layer(t.nn.Module):
    def __init__(self, D: int, H: int, C: int, Z: int, F: int) -> None:
        """
        Standard MH Linear attention + FFN Layer
        INPUT:
            D: input & output dimension
            H: # attention heads
            C: # channels
            Z: hidden state dimension
            F: channel folding constant
        """
        super().__init__()
        self.k = t.nn.Sequential(t.nn.Linear(D, C*H*F), t.nn.Unflatten(-1, (H, F, C)))
        self.q = t.nn.Sequential(t.nn.Linear(D, C*H*F), t.nn.Unflatten(-1, (H, F, C)))
        self.v = t.nn.Sequential(t.nn.Linear(D, D*H), t.nn.Unflatten(-1, (H, D)))
        self.ffn = t.nn.Sequential(
            t.nn.LayerNorm((D, )),
            t.nn.Linear(D, Z), 
            t.nn.ReLU(),
            t.nn.Linear(Z, D),
        )
    def forward(self, x: t.Tensor) -> t.Tensor:
        # elements on channels
        k = self.k(x).softmax(-1)
        c = Categorical(k).sample_n(32)
        v = self.v(x)
        # for each channel, sample 1 element and get the probability
        # for each channel, forward the values that have the same channel
        u = t.index_add()