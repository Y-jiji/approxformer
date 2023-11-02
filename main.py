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
    def update(self, x: torch.Tensor, y: torch.Tensor, layer_next: TProximalModel | None, eta: float) -> None:
        # update critic using next layer critic (with its old forward function)
        if layer_next is not None:
            loss_proxy = self.proxy_new(self.model_new(x), y)
            with torch.no_grad():
                loss_label = layer_next.proxy_new(layer_next.model_old(self.model_new(x)), y)
            loss_delta = loss_proxy - loss_label
            loss_delta.abs().mean().backward()
            self.optimizer_p.step()
            self.proxy_new.zero_grad()
            with torch.no_grad():
                for (p_new, p_old) in zip(self.proxy_new.parameters(), self.proxy_old.parameters()):
                    p_old = p_new * eta + p_old * (1 - eta)
        # update model_new by critic
        loss_proxy = self.proxy_old(self.model_new(x), y)
        loss_proxy.backward()
        self.optimizer_m.step()
        self.model_new.zero_grad()
        # update model_old by eta
        with torch.no_grad():
            for (p_new, p_old) in zip(self.model_new.parameters(), self.model_old.parameters()):
                p_old = p_new * eta + p_old * (1 - eta)
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model_old(x)

class ProximalTrainer:
    def __init__(self, layers: list[ProximalModel]) -> None:
        self.layers = layers
    def train(self, dataset: data.Dataset):
        loader = data.DataLoader(dataset)
        for x, y in loader:
            for layer, layer_next in zip(self.layers[:-1], self.layers[1:]):
                layer.update(x, y, layer_next, eta=1e-2)
            x = layer.forward(x)
        self.layers[-1].update(x, y, eta=1e-2)
    @torch.no_grad()
    def apply(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class Critic(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)