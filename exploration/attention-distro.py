import torch as t
import matplotlib.pyplot as plt

def dot_attention(q, k):
    k = k / (k**2).sum(dim=-1, keepdim=True)**(1/2)
    q = q / (q**2).sum(dim=-1, keepdim=True)**(1/2)
    m = t.eye(k.shape[-2]).cumsum(0)
    a = t.matmul(q, k.T * 100)
    y = t.exp(a - a.max(dim=-1, keepdim=True).values) * (m * (1+0.1*t.randn_like(m)))
    return y / y.sum(dim=-1, keepdim=True)

def linear_attention(q, k):
    L = k.shape[-2]
    D = k.shape[-1]
    m = t.eye(L).cumsum(0)
    m = t.randn_like(1.0*m) * 0.05
    k = k / (k**2).sum(dim=-1, keepdim=True)**(1/2)
    q = q / (q**2).sum(dim=-1, keepdim=True)**(1/2)
    r = t.randn(D, D) * 10
    y = (
        t.matmul(q.matmul(r).cos(), k.matmul(r).cos().T) * m * D**(-1) +
        t.matmul(q.matmul(r).sin(), k.matmul(r).sin().T) * m * D**(-1)
    )
    return y / y.sum(dim=-1, keepdim=True)

def center_attention(q0, k0):
    def inner(q, k):
        k = k / (k**2).sum(dim=-1, keepdim=True)**(1/2)
        q = q / (q**2).sum(dim=-1, keepdim=True)**(1/2)
        m = t.eye(k.shape[-2] + 100).cumsum(0)[:-100, 100:]
        a = t.matmul(q0, k.T)
        a = (a - a.max(dim=-1, keepdim=True).values).exp()
        b = t.matmul(q, k0.T)
        b = (b - b.max(dim=-1, keepdim=True).values).exp()
        y = t.matmul(b, a)
        return y / y.sum(dim=-1, keepdim=True) * (m + 0.05 * t.randn_like(m)).relu()
    return inner

def draw_attention_matrix():
    L = 400
    D = 100
    k = t.randn(L, D)
    cmap = 'gray'
    # att = dot_attention
    # att = linear_attention
    qm = t.randn(32, D)
    km = t.randn(32, D)
    att = center_attention(km * 10, qm * 10)
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