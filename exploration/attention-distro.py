import torch as t
import matplotlib.pyplot as plt

def dot_attention(q, k):
    k = k / (k**2).sum(dim=-1, keepdim=True)**(1/2)
    q = q / (q**2).sum(dim=-1, keepdim=True)**(1/2)
    m = t.eye(k.shape[-2]).cumsum(0)
    a = t.matmul(q, k.T)
    y = t.exp(a - a.max(dim=-1, keepdim=True).values) * m
    return y / y.sum(dim=-1, keepdim=True)

def linear_attention(q, k):
    m = t.eye(k.shape[-2]).cumsum(0)
    k = k / (k**2).sum(dim=-1, keepdim=True)**(1/2)
    q = q / (q**2).sum(dim=-1, keepdim=True)**(1/2)
    y = t.matmul(q.exp(), k.T.exp()) * m
    return y / y.sum(dim=-1, keepdim=True)

def center_attention(q0, k0):
    def inner(q, k):
        k = k / (k**2).sum(dim=-1, keepdim=True)**(1/2)
        q = q / (q**2).sum(dim=-1, keepdim=True)**(1/2)
        m = t.eye(k.shape[-2]).cumsum(0)
        a = t.matmul(q0, k.T)
        a = (a - a.max(dim=-1, keepdim=True).values).exp()
        b = t.matmul(q, k0.T)
        b = (b - b.max(dim=-1, keepdim=True).values).exp()
        y = t.matmul(b, a) * m
        return y / y.sum(dim=-1, keepdim=True)
    return inner

def draw_attention_matrix():
    L = 10
    D = 100
    k = 100 * t.randn(L, D)
    cmap = 'gray'
    att = dot_attention
    # att = linear_attention
    # att = center_attention(t.eye(D, D), t.eye(D, D))
    plt.subplot(1, 3, 1)
    plt.imshow(att(t.randn(L, D), k), cmap=cmap)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(att(t.randn(L, D), k), cmap=cmap)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(att(t.randn(100, L, D), k).mean(dim=0), cmap=cmap)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    draw_attention_matrix()