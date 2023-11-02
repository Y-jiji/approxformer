import torch
from torch.utils import data
from typing import TypeVar

TProximalModel = TypeVar("TProximalModel", bound="ProximalModel")

class ProximalModel:
    def __init__(self, model: torch.nn.Module, proxy: torch.nn.Module) -> None:
        from copy import deepcopy
        self.model_new = model
        self.model_old = deepcopy(model)
        self.proxy_new = proxy
        self.proxy_old = deepcopy(proxy)
        for p in self.model_old.parameters():
            p.requires_grad_(False)
        for p in self.proxy_new.parameters():
            p.requires_grad_(False)
        self.optimizer_m = torch.optim.Adam(self.model_new.parameters())
        self.optimizer_p = torch.optim.Adam(self.proxy_new.parameters())
    def update(self, x: torch.Tensor, y: torch.Tensor, layer_next: TProximalModel | None, eta: tuple[float, float]) -> None:
        # update critic using next layer critic (with its old forward function)
        if layer_next is not None:
            loss_proxy = self.proxy_new(self.model_new(x), y)
            with torch.no_grad():
                loss_label = layer_next.proxy_new(layer_next.model_old(self.model_new(x)), y)
            (loss_proxy - loss_label).abs().mean().backward()
            self.optimizer_p.step()
            self.model_new.zero_grad()
            self.proxy_new.zero_grad()
            with torch.no_grad():
                for (p_new, p_old) in zip(self.proxy_new.parameters(), self.proxy_old.parameters()):
                    p_old = p_new * eta[0] + p_old * (1 - eta[0])
        # update model_new by critic
        loss_proxy = self.proxy_old(self.model_new(x), y).mean()
        print(loss_proxy.item())
        loss_proxy.backward()
        self.optimizer_m.step()
        self.model_new.zero_grad()
        # update model_old by eta
        with torch.no_grad():
            for (p_new, p_old) in zip(self.model_new.parameters(), self.model_old.parameters()):
                p_old = p_new * eta[1] + p_old * (1 - eta[1])
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model_old(x)

class ProximalTrainer:
    def __init__(self, layers: list[ProximalModel]) -> None:
        self.layers = layers
    def train(self, dataset: data.Dataset):
        loader = data.DataLoader(dataset, 16)
        for x, y in loader:
            for layer, layer_next in zip(self.layers[:-1], self.layers[1:]):
                layer.update(x, y, layer_next, eta=(1e-2, 1e-1))
            x = layer.forward(x)
            self.layers[-1].update(x, y, None, eta=(1e-2, 1e-1))
    @torch.no_grad()
    def apply(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class Critic(torch.nn.Module):
    def __init__(self, input_features: int, n_classes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_features, n_classes)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _x = self.linear(x.flatten(0, -2)).log_softmax(-1)
        _y = y.flatten(0, -1)
        return -_x[torch.arange(_y.shape[0], device=_y.device), _y]

class Tail(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _x = x.flatten(0, -2)
        _y = y.flatten(0, -1)
        return -_x[torch.arange(_y.shape[0], device=_y.device), _y]

class Block(torch.nn.Module):
    def __init__(self, d_model: int, d_attention: int, n_heads: int, hidden: int) -> None:
        super().__init__()
        self.ln0 = torch.nn.LayerNorm(d_model)
        self.ln1 = torch.nn.LayerNorm((d_model, n_heads))
        self.q = torch.nn.Sequential(torch.nn.Linear(d_model, d_attention * n_heads), torch.nn.Unflatten(-1, (d_attention, n_heads)))
        self.k = torch.nn.Sequential(torch.nn.Linear(d_model, d_attention * n_heads), torch.nn.Unflatten(-1, (d_attention, n_heads)))
        self.v = torch.nn.Sequential(torch.nn.Linear(d_model, d_model * n_heads), torch.nn.Unflatten(-1, (d_model, n_heads)))
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model * n_heads, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, d_model),
            torch.nn.ReLU(),
        )
    def forward(self, x: torch.Tensor):
        _x = self.ln0(x)
        q, k = self.q(_x).clamp(-10, 10).exp(), self.k(_x).clamp(-10, 10).exp()
        # qkv can be optimized to a very small kernel
        qkv = torch.einsum("...ijh, ...ijkh -> ...ikh", q,
            torch.einsum("...ijh, ...ikh -> ...ijkh", k, self.v(x)).cumsum(-4))
        den = torch.einsum("...ikh, ...ikh -> ...ih", q, k.cumsum(-3)).unsqueeze(-2)
        return self.ffn((x.unsqueeze(-1) + self.ln1(qkv / den)).flatten(-2, -1)) + x

if __name__ == '__main__':
    trainer = ProximalTrainer(
        [ProximalModel(Block(129, 32, 4, 2048).to('cuda:0'), Critic(129, 2048).to('cuda:0')) for i in range(31)] +
        [ProximalModel(torch.nn.Sequential(Block(129, 32, 4, 2048), torch.nn.Linear(129, 2048), torch.nn.LogSoftmax(-1)).to('cuda:0'), Tail().to('cuda:0'))]
    )
    l = [(torch.rand(256, 129).to('cuda:0'), torch.randint(0, 2047, (256, )).to('cuda:0')) for i in range(10)]
    for i in range(1000):
        trainer.train(l)
        print(f'finish {i}')