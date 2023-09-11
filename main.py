import torch as t
from model.approxformer import *

def load_tiny_shakespeare():
    import random
    with open('dataset/tiny-shakespeare.txt') as f:
        text = f.read()
    data = t.frombuffer(text.encode("ascii"), dtype=t.uint8, requires_grad=False).to(t.int).to('cuda:0')
    L = 8000
    I = 128
    model = ApproxFormer(256, L).to('cuda:0')
    optim = t.optim.Adam(model.parameters())
    running_loss = 5.0
    for i in range(100000):
        OFF = random.randint(0, I-1) * L
        y = model.forward(data[OFF:OFF+L])
        loss = -(y[t.arange(y.shape[-2], device=y.device), data[OFF+1:OFF+L+1]] + 1e-30).log().mean()
        running_loss = running_loss * 0.99 + loss.item() * 0.01
        if i % 100 == 0:
            acc = (y.argmax(-1) == data[OFF+1:OFF+L+1]).sum().item() / L
            print(f"{i:05}\n\t{acc}\n\t{running_loss}")
        (loss / 32).backward()
        if i % 32 == 0:
            optim.step()
            model.zero_grad()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    load_tiny_shakespeare()