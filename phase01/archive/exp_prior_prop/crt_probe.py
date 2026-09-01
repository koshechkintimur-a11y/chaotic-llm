"""crt_probe.py — toy test: can CRT-based (Chinese Remainder) multi-scale
pooling recover positional specificity that mean/projector averaged away?

Mother-finds-child idea (user, 2026-08-30):
  - tokens = children in a crowd
  - identify token by MULTIPLE independent anchors (hair + height + gait)
  - CRT: position i is uniquely recovered from residues (i mod p_k) over
    several coprime moduli — small p = local, large p = global
  - ergodic Arnold map gives O(log N) connectivity (already in mixer)

Probe (CPU, seconds):
  - embed W=256 random token ids -> E [W,d]
  - CRT observer: for each prime p_k, pool E into p_k residue buckets
    (positions i mod p_k) -> each bucket = mean of its residue class
  - task: recover the embedding of the token at position L (or detect which
    token id sat at position L) from the residue-bucket readout
  - compare: CRT-pool vs single softmax-proj (H1/H2 style) vs full E (oracle)

If CRT recovers position-specific identity MUCH better than averaged pool,
the idea is alive and worth wiring into the model. If not, stop here.
"""
import numpy as np
import torch
import torch.nn as nn

W = 256
D = 128
V = 512
PRIMES = (3, 5, 7, 11, 13, 17)   # coprime moduli (CRT for W=256: product 3*5*7*11*13*17 > 256)
# note: 3*5*7*11*13*17 = 255255 >> 256, so residues determine position uniquely


class CRTObserver(nn.Module):
    """Pool input embeddings by residue classes of each prime modulus."""
    def __init__(self, d=D, primes=PRIMES):
        super().__init__()
        self.primes = primes
        self.out_dim = sum(primes) * d   # all buckets concatenated

    def forward(self, e):
        # e: [B, W, d] -> buckets per prime
        B, W, d = e.shape
        outs = []
        for p in self.primes:
            buckets = torch.zeros(B, p, d, device=e.device)
            counts = torch.zeros(B, p, 1, device=e.device)
            idx = torch.arange(W, device=e.device) % p
            buckets.scatter_add_(1, idx.unsqueeze(0).unsqueeze(-1).expand(B, W, d), e)
            counts.scatter_add_(1, idx.unsqueeze(0).unsqueeze(-1).expand(B, W, 1),
                                torch.ones(B, W, 1, device=e.device))
            buckets = buckets / counts.clamp(min=1)
            outs.append(buckets.reshape(B, p * d))
        return torch.cat(outs, dim=-1)   # [B, sum(p)*d]


def recover_embedding_accuracy(observer, embed, ids, L, trials=200):
    """Place a random KEY token at position L, then ask: can we recover which
    token is at position L from CRT buckets (via a learned decoder)?"""
    rng = np.random.default_rng(0)
    dec = nn.Linear(observer.out_dim, V)
    # build dataset: random id windows, KEY at position L
    X = []  # buckets
    Y = []  # the token id at position L
    for _ in range(trials):
        ids_i = rng.integers(1, V, size=W)
        e = embed(torch.tensor(ids_i))                     # [W,d]
        b = observer(e.unsqueeze(0))                       # [1, out]
        X.append(b[0])
        Y.append(ids_i[L])
    X = torch.stack(X)
    Y = torch.tensor(Y)
    # fit decoder (closed form ridge) to map buckets -> token at L
    Xn = X.detach().numpy()
    Yn = Y.numpy()
    Xt = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
    w = np.linalg.lstsq(Xt, Yn, rcond=None)[0]             # (out+1,)
    pred = Xt @ w
    acc = np.mean(np.round(pred).astype(int) == Yn)
    return acc


def main():
    embed = nn.Embedding(V, D)
    ids = torch.arange(V)
    obs = CRTObserver()
    print(f"CRT observer out_dim={obs.out_dim:,} (vs W*D={W*D:,} full, D={D} mean)")
    print(f"product of primes={np.prod(PRIMES):,} > W={W} -> CRT uniquely recovers position")
    for L in (16, 64, 128, 240):
        acc = recover_embedding_accuracy(obs, embed, ids, L, trials=300)
        print(f"  L={L}: token-at-position-L recovery acc = {acc:.3f}")
    # compare: mean pool (H1/H2 style) as baseline
    print("\n-- comparison: single-vector pool (what H1/H2 effectively did) --")
    for L in (16, 128):
        # mean pool -> [d], can't index position at all; try nearest-center decode
        rng = np.random.default_rng(0)
        X, Y = [], []
        for _ in range(300):
            ids_i = rng.integers(1, V, size=W)
            e = embed(torch.tensor(ids_i))
            X.append(e.mean(0).detach().numpy())
            Y.append(ids_i[L])
        X = np.array(X); Y = np.array(Y)
        Xt = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        w = np.linalg.lstsq(Xt, Y, rcond=None)[0]
        pred = Xt @ w
        print(f"  L={L}: mean-pool recovery acc = {np.mean(np.round(pred).astype(int) == Y):.3f}")


if __name__ == "__main__":
    main()
