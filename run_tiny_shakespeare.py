import torch as t
t.set_float32_matmul_precision('high')

from model.approxformer import *
from model.baseline import *
from model.logicformer import *

def run_tiny_shakespeare(load, save):
    import random
    import datetime
    with open('data/tiny-shakespeare.txt') as f:
        text = f.read()
    L = 8192
    I = 64
    model = t.compile(ApproxFormer(256, L).to('cuda:0'))
    optim = t.optim.Adam(model.parameters())
    running_loss = 5.0
    data = t.frombuffer(text.encode("ascii"), dtype=t.uint8, requires_grad=False).to(t.int)
    data = data.to('cuda:0')
    if load:
        try:
            model_state_dict, optim_state_dict = t.load(f'ckpt/{datetime.datetime.date()}')
            model.load_state_dict(model_state_dict)
            optim.load_state_dict(optim_state_dict)
        except:
            pass
    for i in range(1, 100001):
        model.train(True)
        OFF = random.randint(0, I-1) * L
        y = model.forward(data[OFF:OFF+L].to('cuda:0'))
        loss = -(y[t.arange(y.shape[-2], device=y.device), data[OFF+1:OFF+L+1]] + 1e-30).log().mean()
        running_loss = running_loss * 0.99 + loss.item() * 0.01 if running_loss is not None else loss.item()
        (loss / 32).backward()
        if i % 32 == 0:
            optim.step()
            model.zero_grad()
        if i % 100 == 0:
            acc = (y.argmax(-1) == data[OFF+1:OFF+L+1].to('cuda:0')).sum().item() / L
            print(f"{i:05} its\n"
                  f"\taccuracy {acc}\n"
                  f"\tloss     {running_loss}\n"
                  f"\ttime     {datetime.datetime.now()}")
        if i % 10000 == 0 and save:
            t.save((model.state_dict(), optim.state_dict()), f'ckpt/{datetime.datetime.now().date()}')
            with t.no_grad():
                model.train(False)
                cache = model.init_cache()
                x = data[L*I:L*I+L+1].to('cuda:0')
                loss = 0.0
                acc = 0.0
                for i in range(L):
                    if i % 100 == 0: print(f'iterate {i:005} / {L}')
                    y, cache = model.iterate(x[i:i+1], cache)
                    loss += -(y[t.arange(y.shape[-2], device=y.device), x[i+1:i+2]] + 1e-30).log()
                    acc += (y.argmax(-1) == x[i+1:i+2])
                print(loss / L)
                print(acc / L)

if __name__ == '__main__':
    run_tiny_shakespeare(False, False)