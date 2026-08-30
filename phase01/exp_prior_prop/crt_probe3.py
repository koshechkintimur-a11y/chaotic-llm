"""crt_probe3.py — test the three geometric selectors (colleague's formalization
of architect's idea) on the SAME recovery task as crt_probe/probe2:
"which token sat at position L?" — isolating selector structure from averaging.

Selectors (all no-learning, O(W log W) claim):
  A. ArnoldSelector — iterate Arnold map on query position, gather orbit indices
  B. CRTSelector   — per-modulus residue subsets (separate, no vector squeeze)
  C. TorusSelector — quadratic-residue rays from query position

Metric: linear decode from selector output -> token id at position L.
Compare vs oracle (raw E[:,L]), separate-buckets (crt_probe2: 100%),
and mean-pool (crt_probe2: ~0%).
"""
import numpy as np
import torch
import torch.nn as nn

W, D, V = 256, 128, 512
PRIMES = (3, 5, 7, 11)


# ---------- A: Arnold orbit selector ----------
def arnold_orbit(query_idx, W, steps=8):
    """Iterate Arnold cat map [1 1;1 2] on (x=q, y=0) mod W -> orbit indices."""
    orbit = []
    x, y = int(query_idx), 0
    for _ in range(steps):
        x, y = (x + y) % W, (x + 2 * y) % W
        orbit.append(x)
    return list(dict.fromkeys(orbit))          # dedupe, keep order


# ---------- B: CRT residue selector ----------
def crt_subsets(query_pos, W, primes):
    """Per-modulus subsets: positions i with i % m == query_pos % m."""
    layers = []
    for m in primes:
        r = query_pos % m
        layers.append([i for i in range(W) if i % m == r])
    return layers


# ---------- C: torus quadratic-residue rays ----------
def quadratic_residues(W):
    qr = set()
    for a in range(W):
        qr.add((a * a) % W)
    return sorted(qr)


def torus_rays(query_pos, W, qr, k=4):
    """Rays: positions query_pos + s*t mod W for s in first k quadratic residues."""
    rays = []
    for s in qr[:k]:
        ray = [(query_pos + s * t) % W for t in range(1, W // 2)]
        rays.append(ray)
    return rays


def recover_acc(build_features, label, L_list=(16, 64, 128, 240), trials=300):
    rng = np.random.default_rng(0)
    embed = nn.Embedding(V, D)
    print(f"-- {label} --")
    for L in L_list:
        X, Y = [], []
        for _ in range(trials):
            ids = rng.integers(1, V, size=W)
            e = embed(torch.tensor(ids)).detach().numpy()      # [W, d]
            feat = build_features(e, W - L)                     # selector at query pos
            X.append(feat)
            Y.append(ids[L])
        X = np.array(X); Y = np.array(Y)
        Xt = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        w = np.linalg.lstsq(Xt, Y, rcond=None)[0]
        acc = np.mean(np.round(Xt @ w).astype(int) == Y)
        print(f"    L={L}: recovery acc = {acc:.3f}")


def main():
    qr = quadratic_residues(W)

    # oracle: raw embedding of the query position (upper bound)
    def feat_oracle(e, q):
        return e[q]
    recover_acc(feat_oracle, "ORACLE: raw E[:,query]", trials=200)

    # A: Arnold orbit -> sum of orbit embeddings (one vector, chaotic sum)
    def feat_arnold(e, q):
        orb = arnold_orbit(q, W, steps=8)
        return e[orb].sum(0)
    recover_acc(feat_arnold, "A: Arnold orbit (sum)")

    # B: CRT subsets -> per-modulus sums, concatenated (separate, no squeeze)
    def feat_crt(e, q):
        outs = []
        for layer in crt_subsets(q, W, PRIMES):
            outs.append(e[layer].sum(0))
        return np.concatenate(outs)
    recover_acc(feat_crt, "B: CRT residue layers (separate sums)")

    # C: torus rays -> per-ray sums, concatenated
    def feat_torus(e, q):
        outs = []
        for ray in torus_rays(q, W, qr, k=4):
            outs.append(e[ray].sum(0))
        return np.concatenate(outs)
    recover_acc(feat_torus, "C: Torus quadratic-residue rays (separate sums)")

    # reference: mean pool (the failure mode)
    def feat_mean(e, q):
        return e.mean(0)
    recover_acc(feat_mean, "REF: mean pool (H1/H2 failure)")


if __name__ == "__main__":
    main()
