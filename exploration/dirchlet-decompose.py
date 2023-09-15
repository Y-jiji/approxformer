import torch as t
import matplotlib.pyplot as plt

@t.no_grad()
def sample_and_plot(ikey, okey, mat, l, s0, s1):
    """
    matrix = sum{j} ikey[i][j] * okey[k][j]
    sample a list of i
    sample a list of j
    for each i
        + plot how matrix[i][j] changes with j
        + print the most unacceptable value in matrix[i][...]
    """
    s0 = t.randperm(l)[:s0].sort().values
    s1 = t.linspace(0, l-1, s1).to(t.int)
    lines = mat(ikey[s0], okey[s1]).T
    for i, oview in enumerate(lines.T):
        color_grad = (i/lines.T.shape[0])
        plt.plot(s1, oview, color=(color_grad, 0, 1-color_grad), label=f"line: {s0[i]}")
        plt.plot(s1, (s1 <= s0[i]) * 1.0, color=(color_grad, 0, 1-color_grad), label=f"line: {s0[i]}")
    plt.plot(s1, t.ones_like(s1) * 0.05, color='green')
    plt.legend()

@t.no_grad()
def sample_and_draw_matrix(ikey, okey, mat, l, s):
    """
    matrix = sum{j} ikey[i][j] * okey[k][j]
    use plt.imshow to plot a submatrix of matrix
    where the submatrix is indexed from a sample list
    """
    s = t.linspace(0, l-1, s).to(t.int)
    m = mat(ikey[s], okey[s])
    plt.imshow(m, cmap='gray')
    plt.colorbar()

@t.no_grad()
def draw_full_matrix(ikey, okey, mat):
    """
    draw full matrix
    """
    plt.imshow(mat(ikey, okey))
    plt.colorbar()

@t.no_grad()
def blur_conv(x, alpha, size):
    """
    compute convolution and get a blurred vector
    """
    if size <= 1: return x
    ker = (t.arange(1, size+1) * alpha).unsqueeze(dim=0).unsqueeze(dim=0)
    ker = ker.softmax(dim=-1).to(x.dtype)
    return t.conv1d(x.T.unsqueeze(-2), ker, padding=size-1).squeeze(-2).T[(size-1)//2:-(size//2)]

@t.no_grad()
def dirichlet_kernel(LEN, KAP):
    """
    compute dirchlet kernel, output (v, u)
    where sum{k} v[i][k] * u[j][k] is approximately 0 when j >= i and is 1 approximately when j < i
    """
    OF1 = 10
    OF2 = 0
    LEN = LEN + 2*OF1 + OF2
    i = t.arange(0, LEN)
    i = ((i.unsqueeze(-1) + 0.5) * t.pi * 2) / LEN
    f = t.arange(0, KAP) // 2
    p = (t.arange(0, KAP) % 2) * t.pi / 2
    u = (i * f + p).cos()
    v = (u / LEN).cumsum(dim=0)
    u[:, 0] /= 2
    return u[OF1+OF2:LEN-OF1], v[OF1:LEN-OF1-OF2], lambda ikey, okey: 2 * t.einsum("ij, kj -> ik", ikey, okey)

@t.no_grad()
def dirichlet_pow_kernel(LEN, KAP):
    """
    compute dirchlet kernel, output (v, u)
    where sum{k} v[i][k] * u[j][k] is approximately 0 when j >= i and is 1 approximately when j < i
    """
    N = KAP // 4
    OFF = -2 * LEN / N
    i = t.arange(0, LEN) + OFF
    i = i.unsqueeze(-1) / LEN * t.pi
    f = t.arange(0, 2*N)
    v = t.concat([(-i * f).cos(), (-i * f).sin()], dim=-1).cumsum(dim=0) / LEN
    ALPHA = 0.5
    i = t.arange(0, LEN)
    i = i.unsqueeze(-1) / LEN * t.pi
    f = t.arange(0, 2*N)
    c = (2*N - f.abs()) * ALPHA / (2*N) + (1-ALPHA)
    c[0] *= 1/2
    u = t.concat([(-i * f).cos() * c, (-i * f).sin() * c], dim=-1)
    u = u * (1 + t.randn_like(u) * 0.1)
    return v, u, lambda ikey, okey: t.einsum("ij, kj -> ik", ikey, okey)

def d_pow(x, l, n, pow, off):
    def combi(n, x):
        return t.arange(n+1-x, n+1).prod() / t.arange(1, x+1).prod()
    def coeff(u):
        u = u + pow * n + pow - 1
        r = t.zeros_like(u).float()
        for i in range(pow):
            v = (u.unsqueeze(-1) - (2 * n + 1) * i - t.arange(pow-1)).relu()
            c = combi(pow, i)
            r += c * (2 * ((i + 1) % 2) - 1) * v.prod(-1) / (t.arange(pow-1) + 1).prod(-1)
        return r
    a = t.arange(0, pow*n+1)
    u = coeff(a) * t.cos(a * (((x + off) / l) * t.pi).unsqueeze(-1)) / n ** (pow-1) / 2
    u[:, 0] /= 2
    v = t.cos(-a * ((x / l) * t.pi).unsqueeze(-1)).cumsum(0) / l
    return v, u

@t.no_grad()
def random_spinning_kernel(LEN, KAP):
    OF1 = 0
    OF2 = 100
    LEN = LEN + 2*OF1 + OF2
    i = t.arange(0, LEN) * 0.99 + t.randperm(LEN) * 0.01
    i = (i.unsqueeze(-1) * t.pi * 2) / LEN
    f = t.arange(0, KAP) // 2
    p = (t.arange(0, KAP) % 2) * t.pi / 2
    u = (i * f + p).cos()
    u[:, 1:] *= 2
    v = ((i * f + p).cos() / LEN).cumsum(dim=0)
    return u[OF1+OF2:LEN-OF1], v[OF1:LEN-OF1-OF2], lambda ikey, okey: t.einsum("ij, kj -> ik", ikey, okey)

@t.no_grad()
def plot_step_fn_tri_approx(LEN, KAP):
    x = t.linspace(-t.pi, t.pi, LEN).unsqueeze(-1)
    f = t.exp(1j * x * (t.arange(2, KAP+2) // 2))
    i = t.arange(0, KAP)
    n = i // 2 + 1
    g = (i % 2 == 0) * ((n%2)*2.0 - 1) / (1j * n) + (i % 2 == 1) / (1j * n)
    y = t.einsum("ij, j -> i", f, g).real ** 2
    plt.plot(x, y)
    plt.plot(x, t.pi / 2 * (2*(x > 0) - 1))

@t.no_grad()
def plot_dirichlet_pow_fn_tri_approx(S, LEN, KAP):
    x = t.linspace(-S, S, LEN).unsqueeze(-1)
    f = t.arange(-2*KAP, 2*KAP+1)
    c = 2*KAP + 1 - f.abs()
    y = (t.exp(1j * x * f) * c).sum(dim=1)
    y = y.real.cumsum(dim=0) / LEN / (2 * KAP + 1) * t.pi / S
    plt.plot(x, y, label=f'dpow-{KAP}')

@t.no_grad()
def plot_dirichlet_fn_tri_approx(S, LEN, KAP):
    x = t.linspace(-S, S, LEN).unsqueeze(-1)
    f = t.arange(-2*KAP, 2*KAP+1)
    y = t.exp(1j * x * f).sum(dim=1)
    y = y.real.cumsum(dim=0) / LEN * t.pi / S
    plt.plot(x[LEN], y, label=f'd-{KAP}')

@t.no_grad()
def step_fn_kernel(LEN, KAP):
    """
    """
    x = t.linspace(-t.pi, t.pi, LEN).unsqueeze(-1) / 2
    f = t.exp(1j * -x * (t.arange(2, KAP+2) // 2))
    h = t.exp(1j * (x + 0.001) * (t.arange(2, KAP+2) // 2))
    i = t.arange(0, KAP)
    n = i // 2 + 1
    g = (i % 2 == 0) * ((n%2)*2.0 - 1) / (1j * n) + (i % 2 == 1) / (1j * n)
    return f, h * g, lambda ikey, okey: (t.einsum("ij, kj -> ik", ikey, okey).real + t.pi/2) / t.pi

@t.no_grad()
def random_direction_kernel(LEN, KAP):
    x = t.einsum("...ij, ...j -> ...i", t.randn(KAP, KAP), t.exp(10*t.rand(LEN, KAP)))
    return x.cumsum(0), x, lambda ikey, okey: t.einsum("ij, kj -> ik", ikey, okey)

if __name__ == '__main__':
    t.manual_seed(154)
    with t.no_grad():
        LEN = 2_000_000
        KAP = 128
        ikey, okey, mat = dirichlet_pow_kernel(LEN, KAP)
        sample_and_plot(ikey, okey, mat, LEN, 10, 10000)
        # sample_and_draw_matrix(ikey, okey, mat, LEN, 100)
        plt.show()
