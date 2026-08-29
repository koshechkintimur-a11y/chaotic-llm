"""eye.py — the Eye selector + Selective Chaotic LM.

The Eye does NOT mix tokens (chaos keeps doing that cheaply). It computes
per-position route scores in O(W·d) — which positions deserve to feed the
readout / be emphasized — then the chaotic mixer output is pooled with
those weights instead of uniformly.

Variants (ТЗ §3):
  A: global eye   — one global query, dot with each token
  B: local eye    — 1D conv over positions → per-token score
  C: global+local — sum of A and B (main candidate)

Modes (ТЗ §6):
  soft  — softmax(T) weights, full support
  topk  — keep top-K, renormalize
  hard  — one-hot argmax

Temperature T and entropy-λ are passed by the caller.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Eye(nn.Module):
    def __init__(self, d, variant="C", mode="soft", T=1.0, k=4,
                 local_kernel=7):
        super().__init__()
        assert variant in ("A", "B", "C")
        assert mode in ("soft", "topk", "hard")
        self.variant, self.mode, self.T, self.k = variant, mode, T, k
        self.d = d
        # global query (A, C)
        self.global_lin = nn.Linear(d, d, bias=False)
        # local conv (B, C)
        if variant in ("B", "C"):
            self.local_conv = nn.Conv1d(d, d, local_kernel, padding=local_kernel // 2, bias=False)
        # score projection
        self.score_proj = nn.Linear(d, 1, bias=False)

    def _scores(self, h):
        B, W, d = h.shape
        s = torch.zeros(B, W, device=h.device, dtype=h.dtype)
        if self.variant in ("A", "C"):
            q = self.global_lin(h.mean(dim=1, keepdim=True))          # (B,1,d)
            s = s + (h * q).sum(-1)                                   # (B,W)
        if self.variant in ("B", "C"):
            local = self.local_conv(h.transpose(1, 2)).transpose(1, 2)  # (B,W,d)
            s = s + self.score_proj(local).squeeze(-1)                # (B,W)
        if self.variant == "A":
            s = self.score_proj(h * q).squeeze(-1)
        return s

    def forward(self, h):
        s = self._scores(h)
        w = torch.softmax(s / self.T, dim=-1)                          # (B,W)
        if self.mode == "topk":
            top = torch.topk(w, self.k, dim=-1).indices
            mask = torch.zeros_like(w).scatter(-1, top, 1.0)
            w = w * mask
            w = w / (w.sum(-1, keepdim=True) + 1e-9)
        elif self.mode == "hard":
            top = torch.argmax(w, dim=-1, keepdim=True)
            w = torch.zeros_like(w).scatter(-1, top, 1.0)
        return w  # (B, W) route weights

    def entropy(self, h):
        w = self.forward(h)
        return -(w * torch.log(w.clamp_min(1e-9))).sum(-1).mean()


class SelectiveChaoticLM(nn.Module):
    """Chaotic mixer + Eye-weighted pooling (no pairwise attention).

    forward: embed+pos -> chaotic block (unchanged) -> norm -> h
             eye weights from h -> weighted global vector
             readout(concat(h_last, weighted_gvec))
    """
    def __init__(self, base, V, d, eye_variant="C", eye_mode="soft",
                 eye_T=1.0, eye_k=4):
        super().__init__()
        self.base = base
        self.eye = Eye(d, eye_variant, eye_mode, eye_T, eye_k)
        self.readout = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, x, eye_T=None):
        h = self.base.mix(x)
        if eye_T is not None:
            self.eye.T = eye_T
        w = self.eye(h)                                  # (B, W)
        gvec = (h * w.unsqueeze(-1)).sum(dim=1)          # weighted pool
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1)), w
