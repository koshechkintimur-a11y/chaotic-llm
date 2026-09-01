"""crt_probe2.py — verify the FIX: read CRT residue buckets SEPARATELY (no
vector compression) vs compressed (H3's mistake Linear(1664->128)).

User principle: «не допускать сжатие в вектор». H3 failed because it squeezed
all 26 residue buckets into ONE 128-d vector. Probe1 proved full-bucket linear
decoder recovers position 100%. Question: does keeping buckets separate in a
smaller per-bucket dim (d_crt=16, 26*16=416) still recover ~100%, while the
H3-style compression (->128) collapses?

CPU, seconds.
"""
import numpy as np
import torch
import torch.nn as nn

W, D, V = 256, 128, 512
PRIMES = (3, 5, 7, 11)


class CRTBuckets(nn.Module):
    """Residue buckets per prime, each bucket = mean of its residue class.
    Returns a FLAT concatenation of all buckets (no further compression)."""
    def __init__(self, d=D, primes=PRIMES):
        super().__init__()
        self.primes = primes

    def forward(self, e):
        B, W, d = e.shape
        outs = []
        for p in self.primes:
            buckets = torch.zeros(B, p, d, device=e.device)
            counts = torch.zeros(B, p, 1, device=e.device)
            idx = torch.arange(W, device=e.device) % p
            buckets.scatter_add_(1, idx.unsqueeze(0).unsqueeze(-1).expand(B, W, d), e)
            counts.scatter_add_(1, idx.unsqueeze(0).unsqueeze(-1).expand(B, W, 1),
                                torch.ones(B, W, 1, device=e.device))
            outs.append((buckets / counts.clamp(min=1)).reshape(B, p * d))
        return torch.cat(outs, dim=-1)


def probe_recovery(bucket_dim, label, L_list=(16, 64, 128, 240), trials=300):
    """Decoder from buckets (per-bucket projected to bucket_dim, kept separate)
    to 'which token at position L'. bucket_dim=None -> use raw full d per bucket."""
    rng = np.random.default_rng(0)
    embed = nn.Embedding(V, D)
    crt = CRTBuckets()
    if bucket_dim is None:
        proj = None
        in_dim = crt.forward(torch.zeros(1, W, D)).shape[-1]
    else:
        proj = nn.Linear(D, bucket_dim)
        in_dim = (sum(PRIMES) * bucket_dim)
    dec = in_dim + 1  # +bias in lstsq
    print(f"-- bucket_dim={bucket_dim}: total bucket features={in_dim:,} --")
    for L in L_list:
        X, Y = [], []
        for _ in range(trials):
            ids = rng.integers(1, V, size=W)
            e = embed(torch.tensor(ids))
            b = crt(e.unsqueeze(0))                    # [1, sum(p)*D]
            if proj is not None:
                # per-bucket projection: reshape to (1, sum_p, D) -> (1, sum_p, bd) -> flat
                bs = b.view(1, sum(PRIMES), D)
                b = proj(bs).reshape(1, -1)
            X.append(b[0].detach().numpy())
            Y.append(ids[L])
        X = np.array(X); Y = np.array(Y)
        Xt = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        w = np.linalg.lstsq(Xt, Y, rcond=None)[0]
        acc = np.mean(np.round(Xt @ w).astype(int) == Y)
        print(f"    L={L}: recovery acc = {acc:.3f}")


def probe_compressed(compress_to, L_list=(16, 64, 128, 240), trials=300):
    """H3-style: squeeze ALL buckets -> one vector of size compress_to, then decode."""
    rng = np.random.default_rng(0)
    embed = nn.Embedding(V, D)
    crt = CRTBuckets()
    sq = nn.Linear(sum(PRIMES) * D, compress_to)
    print(f"-- H3-style compression: all buckets -> {compress_to}-d vector --")
    for L in L_list:
        X, Y = [], []
        for _ in range(trials):
            ids = rng.integers(1, V, size=W)
            e = embed(torch.tensor(ids))
            b = crt(e.unsqueeze(0))
            b = sq(b)                                   # squeeze into ONE vector
            X.append(b[0].detach().numpy())
            Y.append(ids[L])
        X = np.array(X); Y = np.array(Y)
        Xt = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        w = np.linalg.lstsq(Xt, Y, rcond=None)[0]
        acc = np.mean(np.round(Xt @ w).astype(int) == Y)
        print(f"    L={L}: recovery acc = {acc:.3f}")


if __name__ == "__main__":
    print("=== FIX: keep buckets SEPARATE (per-bucket small projection) ===")
    probe_recovery(16, "separate buckets d_crt=16")      # 26*16=416 features
    print()
    print("=== H3's mistake: compress all buckets into ONE vector ===")
    probe_compressed(128)                                 # 1664 -> 128 (what H3 did)
