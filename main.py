import torch as t
from model.approxformer import *
from model.baseline import *

def load_tiny_shakespeare():
    import random
    import datetime
    with open('dataset/tiny-shakespeare.txt') as f:
        text = f.read()
    L = 14000
    I = 64
    model = ApproxFormer(256, L).to('cuda:0')
    optim = t.optim.Adam(model.parameters())
    running_loss = 5.0
    data = t.frombuffer(text.encode("ascii"), dtype=t.uint8, requires_grad=False).to(t.int).to('cuda:0')
    for i in range(100000):
        OFF = random.randint(0, I-1) * L
        y = model.forward(data[OFF:OFF+L])
        loss = -(y[t.arange(y.shape[-2], device=y.device), data[OFF+1:OFF+L+1]] + 1e-30).log().mean()
        running_loss = running_loss * 0.99 + loss.item() * 0.01
        (loss / 32).backward()
        if i % 100 == 0:
            acc = (y.argmax(-1) == data[OFF+1:OFF+L+1]).sum().item() / L
            print(f"{i:05}\n\t{acc}\n\t{running_loss}\n\t{datetime.datetime.now()}")
        if i % 32 == 0:
            optim.step()
            model.zero_grad()
    with t.no_grad():
        model.train(False)
        cache = model.init_cache()
        x = data[L*I:L*I+L+1]
        loss = 0.0
        for i in range(L):
            if i % 100 == 0: print(f'iterate {i}')
            y, cache = model.iterate(x[i:i+1], cache)
            loss += -(y[t.arange(y.shape[-2], device=y.device), x[i+1:i+2]] + 1e-30).log()
        print(loss / L)

if __name__ == '__main__':
    load_tiny_shakespeare()