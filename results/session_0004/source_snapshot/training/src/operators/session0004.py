"""Three testable learning mechanisms, with no established novelty claim."""

import torch
from torch import nn
import torch.nn.functional as F


def mlp(input_dim, output_dim, width=32):
    return nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, width),
                         nn.Tanh(), nn.Linear(width, output_dim))


class BranchModel(nn.Module):
    """E: jointly generated pair, plus conventional matched-output controls."""
    def __init__(self, mode="paired"):
        super().__init__(); self.mode = mode
        self.net = mlp(2 if mode != "conditioned" else 3, 2 if mode in ["paired", "two_head"] else 1)
        if mode == "paired":
            self.fold_raw = nn.Parameter(torch.tensor(-.4, dtype=torch.float64))

    @property
    def critical(self):
        return 3.2 + F.softplus(self.fold_raw)

    def forward(self, lam, x):
        b, n = len(lam), len(x)
        l = lam[:, None].expand(b, n)
        spatial = (x - .5)**2
        z = torch.stack([l / 4, spatial[None].expand(b, n)], dim=-1)
        envelope = x * (1-x)
        if self.mode == "conditioned":
            out = []
            for sign in [-1., 1.]:
                zs = torch.cat([z, torch.full_like(z[..., :1], sign)], dim=-1)
                out.append(self.net(zs)[..., 0] * envelope * 8)
            return torch.stack(out, dim=1)
        y = self.net(z)
        if self.mode == "paired":
            center = 8 * envelope * y[..., 0]
            gap = 8 * envelope * F.softplus(y[..., 1]) * torch.sqrt(torch.clamp(self.critical - l, min=1e-10))
            return torch.stack([center-gap, center+gap], dim=1)
        if self.mode == "mean":
            return (8 * envelope * y[..., 0])[:, None].expand(b, 2, n)
        return (8 * envelope[None, :, None] * y).transpose(1, 2)


class InterventionModel(nn.Module):
    """F: finite edit singles and a learned symmetric cross-response channel."""
    def __init__(self, mode="pair", rank=16):
        super().__init__(); self.mode = mode; self.rank = rank
        if mode == "direct":
            self.direct = mlp(5, 1, 48)
        else:
            self.base = mlp(1, 1, 24)
            self.single = mlp(3, 1, 32)
            if mode == "pair":
                self.edit = mlp(2, rank, 24)
                self.query = mlp(1, rank, 24)

    def forward(self, inputs, x):
        # inputs = a,b,center1,center2; the source width .1 and lambda1.5 are fixed.
        a, b, c1, c2 = inputs.unbind(-1)
        batch, n = len(inputs), len(x)
        spatial = x[None].expand(batch, n)
        if self.mode == "direct":
            z = torch.stack([a[:, None].expand_as(spatial), b[:, None].expand_as(spatial),
                             c1[:, None].expand_as(spatial), c2[:, None].expand_as(spatial), spatial], -1)
            swapped = z[..., [1, 0, 3, 2, 4]]
            return x*(1-x)*(self.direct(z)[..., 0]+self.direct(swapped)[..., 0])/2
        result = self.base(x[:, None])[:, 0][None].expand(batch, n)
        for amplitude, center in [(a, c1), (b, c2)]:
            z = torch.stack([amplitude[:, None].expand_as(spatial), center[:, None].expand_as(spatial), spatial], -1)
            result = result + amplitude[:, None] * self.single(z)[..., 0]
        if self.mode == "pair":
            ea = self.edit(torch.stack([a, c1], -1))
            eb = self.edit(torch.stack([b, c2], -1))
            result = result + (a*b)[:, None] * torch.einsum('br,nr->bn', ea*eb, self.query(x[:, None])) / self.rank**.5
        return x*(1-x)*result


class CoGenerationModel(nn.Module):
    """G: temperatures and balanced fluxes interact inside a learned latent block."""
    def __init__(self, n=15, mode="feedback", width=48, depth=3):
        super().__init__(); self.n=n; self.mode=mode; self.depth=depth
        self.encode = nn.Linear(2*n+4, width)
        self.evolve = nn.Linear(width, width)
        self.decode = nn.Linear(width, n+1 if mode != "direct" else 2*n+1)
        self.gate = nn.Linear(width, width)
        self.reaction = nn.Linear(n+1, width, bias=False)
        self.mismatch = nn.Linear(n+1, width, bias=False)
        self.register_buffer("center", torch.zeros(2*n+4))
        self.register_buffer("scale", torch.ones(2*n+4))
        self.register_buffer("x", torch.arange(1, n+1, dtype=torch.float64)/(n+1))

    def generate(self, h, raw):
        n = self.n; k = raw[:, :n+1].exp(); q = raw[:, n+1:2*n+1]
        boundary = raw[:, 2*n+1:2*n+3]; lam = raw[:, -1]
        state = self.decode(h)
        t = boundary[:, :1]*(1-self.x) + boundary[:, 1:2]*self.x + self.x*(1-self.x)*state[:, :n]
        if self.mode == "direct":
            return t, state[:, n:]
        effective_q = q + lam[:, None]*t.exp()
        total = F.pad(torch.cumsum(effective_q, dim=1)/(n+1), (1, 0))
        j = state[:, n:n+1] + total
        return t, j

    def forward(self, raw):
        h = torch.tanh(self.encode((raw-self.center)/self.scale))
        for _ in range(self.depth):
            t,j = self.generate(h, raw)
            if self.mode == "feedback":
                full = torch.cat([raw[:, 2*self.n+1:2*self.n+2],t,raw[:, 2*self.n+2:2*self.n+3]],1)
                constitutive = -raw[:, :self.n+1].exp()*torch.diff(full,dim=1)*(self.n+1)
                r = j-constitutive
                h = torch.tanh(self.evolve(h)+self.mismatch(r)+torch.tanh(self.gate(h))*torch.tanh(self.reaction(r)))
            else:
                h = torch.tanh(self.evolve(h))
        return self.generate(h, raw)
