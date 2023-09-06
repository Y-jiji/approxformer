import torch as t
from model import *

def load_tiny_shakespeare():
    import random
    with open('dataset/tiny-shakespeare.txt') as f:
        text = f.read()
    print(text.encode("utf-8"))
    data = t.frombuffer(text.encode("ascii"), dtype=t.uint8, requires_grad=False).to(t.int).to('cuda:0')
    print(data.size())
    model = Decoder(256).to('cuda:0')
    optim = t.optim.Adam(model.parameters())
    L = 4000
    I = 64
    for i in range(10000):
        OFF = random.randint(0, I-1) * L
        y = model.forward(data[OFF:OFF+L])
        loss = -(y[t.arange(y.shape[-2]), data[OFF+1:OFF+L+1]] + 1e-30).log().mean()
        if i % 100 == 0:
            acc = (y.argmax(-1) == data[OFF+1:OFF+L+1]).sum().item() / L
            print(f"{i:05}\n\t{acc}\n\t{loss.item()}")
        loss.backward()
        optim.step()
        model.zero_grad()
    with t.no_grad():
        y, _cache = model.iterative_forward(data[I*L:I*L+L])
        print((y == data[I*L+1:I*L+L+1]).sum().item() / L)

def plot_transformer_mask():
    L = 2000
    B = 4
    PAD = 0
    model = t.nn.TransformerEncoderLayer(128, 1, 1024, batch_first=True).to('cuda:0')
    x = t.randn(B, L + PAD + 1, 128).to('cuda:0')
    y = x[:, 1:]
    x = x[:, :-1]
    optim = t.optim.Adam(model.parameters())
    mask = t.nn.Transformer().generate_square_subsequent_mask(L).to('cuda:0')
    for i in range(1000):
        loss = (model.forward(x, mask) - y).abs().mean()
        if i % 10 == 0:
            print(f"{i:04}: {loss.item()}")
        loss.backward()
        optim.step()
        model.zero_grad()
    model.train(False)
    x = t.randn(B, L + PAD, 128).to('cuda:0')
    y = x + 20 * t.randn(B, L + PAD, 128).to('cuda:0') * (t.arange(L + PAD) >= (L + PAD)/2).unsqueeze(-1).to('cuda:0')
    out_x = model.forward(x, mask)[..., PAD:]
    out_y = model.forward(y, mask)[..., PAD:]
    print(t.cuda.memory_allocated(0))
    for i in range(B):
        plt.subplot(B, 1, i+1)
        e = ((out_x[i] - out_y[i]).abs() / (out_x[i].abs() + out_y[i].abs())).abs().cpu().T.detach()
        plt.imshow(-(e + 1e-4).log10().to(t.int), cmap='gray', aspect='auto', interpolation='none')
        plt.colorbar()

def plot_pseudo_mask():
    L = 2000
    B = 4
    PAD = 0
    model = t.nn.Sequential(
        LongTermDecoderLayer(128, 4, 128, 128).to('cuda:0'), 
        # DecoderLayer(128, 4, 128, 128).to('cuda:0'), 
        # DecoderLayer(128, 4, 128, 128).to('cuda:0'), 
    )
    x = t.randn(B, L + PAD + 1, 128).to('cuda:0')
    y = x[:, 1:]
    x = x[:, :-1]
    optim = t.optim.Adam(model.parameters())
    for i in range(1000):
        loss = (model.forward(x) - y).abs().mean()
        if i % 10 == 0:
            print(f"{i:04}: {loss.item()}")
        loss.backward()
        optim.step()
        model.zero_grad()
    model.train(False)
    x = t.randn(B, L + PAD, 128).to('cuda:0')
    y = x + 20 * t.randn(B, L + PAD, 128).to('cuda:0') * (t.arange(L + PAD) >= (L + PAD)/2).unsqueeze(-1).to('cuda:0')
    out_x = model.forward(x)[..., PAD:]
    out_y = model.forward(y)[..., PAD:]
    print(t.cuda.memory_allocated(0))
    for i in range(B):
        plt.subplot(B, 1, i+1)
        e = ((out_x[i] - out_y[i]).abs() / (out_x[i].abs() + out_y[i].abs())).abs().cpu().T.detach()
        plt.imshow(-(e + 1e-4).log10(), cmap='gray', aspect='auto', interpolation='none')
        plt.colorbar()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    load_tiny_shakespeare()