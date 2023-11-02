import torch as t
import matplotlib.pyplot as plt

class Matrix(t.nn.Module):
    def __init__(self, m, k) -> None:
        super().__init__()
        self.v = t.nn.Embedding(m, k)
        self.u = t.nn.Parameter(t.ones(m, 1))
        self.b = t.nn.Parameter(t.ones(1, k))
        self.m = m

    def forward(self):
        x = t.arange(self.m)
        v = self.v(x)
        u = self.u * v.cumsum(dim=0) + self.b
        return ((t.einsum("ij, kj -> ik", u, v) - t.eye(self.m).cumsum(dim=-1))**2)

    @t.no_grad()
    def matrix(self):
        x = t.arange(self.m)
        v = self.v(x)
        u = self.u * v.cumsum(dim=0) + self.b
        return t.einsum("ij, kj -> ik", u, v)

    @t.no_grad()
    def get_v(self):
        x = t.arange(self.m)
        v = self.v(x)
        return v

    @t.no_grad()
    def get_u(self):
        x = t.arange(self.m)
        v = self.v(x)
        u = self.u * v.cumsum(dim=0) + self.b
        return u

if __name__ == '__main__':
    model = Matrix(100, 10)
    optim = t.optim.Adam(model.parameters())
    for i in range(50000):
        loss = model()
        print(loss.mean().item())
        loss.mean().backward()
        optim.step()
        model.zero_grad()
    plt.subplot(1, 3, 1)
    plt.imshow(model.get_u())
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(model.get_v())
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(model.u.detach())
    plt.colorbar()
    plt.show()