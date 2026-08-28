"""chaotic_gears.py — "Gears" hypothesis: multiple specialized mixers.

The d=512 scaling ceiling (exp41b/42/45) could come from a single 24-block
mixer that can't use its width. "Gears": distribute work across SEVERAL
mixers, each at its own window scale, connected by interleave
permutations. Three variants:

- DualChaoticMixer (exp50): local (Wl=64) → interleave → intermediate (256)
- TripleChaoticMixer (exp51): local (64) → interleave → mid (128) → interleave → global (256)
- BidirectionalChaoticMixer (exp52): forward + backward mixers, concat → Linear(2d,d)

Note: ТЗ window3=1024 exceeds the base W=256 (kept from exp41b for
apples-to-apples). Adapted to 64/128/256 — the same three-scale idea.

The interleave permutation connects the scales WITHOUT adding learnable
params (still only gates learn) — except exp52's projection (2d→d).
"""
import torch
import torch.nn as nn


class ChaoticBlock(nn.Module):
    """Hierarchical chaotic mixing at one window scale."""
    def __init__(self, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: self._perm(Wl, t) for t in range(1, bl + 1)}
        self._sig_g = {t: self._perm(self.Nw, t) for t in range(1, bg + 1)}

    @staticmethod
    def _perm(n, t):
        # multiplicative permutation (primitive root of n+1 style), deterministic
        from chaos_lib import permute_indices
        return torch.as_tensor(permute_indices(n, t), dtype=torch.long)

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        if N < 2:
            return h  # single element — no pairing possible
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even, odd = h[:, 0::2, :], h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, h):
        B, W, d = h.shape
        hw = h.view(B, self.Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l)
                           for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        return loc.reshape(B, W, d) + glob.mean(dim=1, keepdim=True)


class Interleave(nn.Module):
    """Cross-scale permutation: stride-interleave token positions."""
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        B, W, d = x.shape
        s = self.stride
        xg = x.view(B, W // s, s, d)
        return xg.transpose(1, 2).reshape(B, W, d)


class DualChaoticMixer(nn.Module):
    def __init__(self, W, d, blocks=12, window1=64, window2=256, stride=4):
        super().__init__()
        bg = max(1, blocks // 2)
        self.block1 = ChaoticBlock(W, window1, d, blocks, bg)
        self.block2 = ChaoticBlock(W, window2, d, blocks, bg)
        self.interleave = Interleave(stride)

    def forward(self, x):
        x = self.block1(x)
        x = self.interleave(x)
        x = self.block2(x)
        return x


class TripleChaoticMixer(nn.Module):
    def __init__(self, W, d, blocks=8, window1=64, window2=128, window3=256,
                 stride1=4, stride2=8):
        super().__init__()
        bg = max(1, blocks // 2)
        self.block1 = ChaoticBlock(W, window1, d, blocks, bg)
        self.block2 = ChaoticBlock(W, window2, d, blocks, bg)
        self.block3 = ChaoticBlock(W, window3, d, blocks, bg)
        self.inter1 = Interleave(stride1)
        self.inter2 = Interleave(stride2)

    def forward(self, x):
        x = self.block1(x)
        x = self.inter1(x)
        x = self.block2(x)
        x = self.inter2(x)
        x = self.block3(x)
        return x


class BidirectionalChaoticMixer(nn.Module):
    def __init__(self, W, d, blocks=12, window=64):
        super().__init__()
        bg = max(1, blocks // 2)
        self.fwd = ChaoticBlock(W, window, d, blocks, bg)
        self.bwd = ChaoticBlock(W, window, d, blocks, bg)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, x):
        xf = self.fwd(x)
        xb = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.proj(torch.cat([xf, xb], dim=-1))


def make_model(W, d, V, kind, **kw):
    """Wrap a gear mixer into a full LM head (embed + mixer + readout)."""
    class GearBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
            if kind == "dual":
                self.mixer = DualChaoticMixer(W, d, **kw)
            elif kind == "triple":
                self.mixer = TripleChaoticMixer(W, d, **kw)
            elif kind == "bi":
                self.mixer = BidirectionalChaoticMixer(W, d, **kw)
            else:
                raise ValueError(kind)
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            return self.norm(self.embed(x) + self.pos + self.mixer(self.embed(x) + self.pos))

    class GearLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = GearBase()
            self.readout = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, V))

        def forward(self, x):
            h = self.base(x)
            gvec = h.mean(dim=1)
            return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))

    return GearLM()
