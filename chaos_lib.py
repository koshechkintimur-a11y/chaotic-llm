"""
chaos_lib.py — shared library for the Chaotic LLM research project.

MATHEMATICAL FACTS (pre-registered):

1. The Arnold cat map is a bijection on Z_N^2 (the N×N grid).  It is a
   permutation of the N^2 cells.

2. When tokens occupy a subset T ⊂ Z_N^2 of size n_tokens, the induced map
   on T is a permutation ONLY if T is invariant under the map (i.e.,
   A^t(T) = T for all t).  For a rectangular grid T = {0..k-1}×{0..k-1},
   this holds iff k = N (i.e., all cells are occupied).  For arbitrary
   subsets, the map is not a permutation on T.

   CONSEQUENCE: permute_sequence is only an exact bijection on tokens
   when n_tokens is a perfect square AND the grid size equals sqrt(n_tokens).
   For non-square counts, we pad to the nearest perfect square.

3. A permutation on a finite state space conserves the L2 norm, the
   Hamming distance, and the support of the state vector.  It does NOT
   mix values — value mixing requires a separate operation (coupling_mix).

4. The "chaotic mixing" of the cat map manifests as the decorrelation of
   positions under iteration, NOT as exponential divergence of state values.
   The map is a finite group action; all orbits are periodic.
"""

import numpy as np

A_MATRIX = np.array([[1, 1], [1, 2]], dtype=np.int64)
A_INV_MATRIX = np.array([[2, -1], [-1, 1]], dtype=np.int64)


# ---------------------------------------------------------------------------
# Arnold cat map on Z_N^2
# ---------------------------------------------------------------------------

def arnold_map(pts, N, t=1):
    """Apply t iterations of the cat map to integer positions.

    pts: (..., 2) array of positions in [0, N)^2
    Returns positions (..., 2).  Negative t applies the inverse map.
    """
    pts = np.asarray(pts, dtype=np.int64)
    if t < 0:
        return arnold_inv(pts, N, -t)
    for _ in range(t):
        x = pts[..., 0]
        y = pts[..., 1]
        nx = (x + y) % N
        ny = (x + 2 * y) % N
        pts = np.stack([nx, ny], axis=-1)
    return pts


def arnold_inv(pts, N, t=1):
    """Apply t iterations of the INVERSE cat map (M = A^{-t}).

    Negative t applies the forward map.
    """
    pts = np.asarray(pts, dtype=np.int64)
    if t < 0:
        return arnold_map(pts, N, -t)
    for _ in range(t):
        x = pts[..., 0]
        y = pts[..., 1]
        nx = (2 * x - y) % N
        ny = (-x + y) % N
        pts = np.stack([nx, ny], axis=-1)
    return pts


def period(N):
    """Minimal t > 0 such that A^t == I (mod N).  T(N) of the cat map."""
    if N == 1:
        return 1
    a, b, c, d = 1, 0, 0, 1
    t = 0
    while True:
        t += 1
        na = (a + c) % N
        nb = (b + d) % N
        nc = (a + 2 * c) % N
        nd = (b + 2 * d) % N
        a, b, c, d = na, nb, nc, nd
        if (a, b, c, d) == (1, 0, 0, 1):
            return t


# ---------------------------------------------------------------------------
# Token <-> grid mapping (PERFECT SQUARE only)
# ---------------------------------------------------------------------------

def nearest_square(n):
    """Round up to nearest perfect square."""
    r = int(np.ceil(np.sqrt(n)))
    return r * r


def square_grid_size(n_tokens):
    """N = sqrt(padded n_tokens).  n_tokens must be a perfect square."""
    N = int(np.sqrt(n_tokens))
    assert N * N == n_tokens, (
        f"n_tokens must be a perfect square, got {n_tokens} -> sqrt={np.sqrt(n_tokens)}")
    return N


def token_positions(n_tokens, N=None):
    """Row-major positions of tokens on the N x N grid (N = sqrt(n_tokens))."""
    if N is None:
        N = square_grid_size(n_tokens)
    idx = np.arange(n_tokens, dtype=np.int64)
    return np.stack([idx // N, idx % N], axis=-1)


def permute_indices(n_tokens, t):
    """Return σ_t such that after permutation, X_after[i] = X_before[σ_t[i]].

    n_tokens must be a perfect square.  σ_t is a permutation of 0..n_tokens-1.
    It is computed as: for grid position pos_i = (i//N, i%N), σ_t(i) = the
    token index at position A^{-t}·pos_i (the token whose value arrives at
    cell i after t steps of the cat map).
    """
    N = square_grid_size(n_tokens)
    pos = token_positions(n_tokens, N)          # (n_tokens, 2)
    src_pos = arnold_inv(pos, N, t)               # A^{-t}·pos_i
    flat = src_pos[..., 0] * N + src_pos[..., 1]  # (n_tokens,)  — each is a token index
    return flat.astype(np.int64)


def permute_sequence(X, t):
    """Apply cat-map permutation at step t to a sequence X (..., n_tokens, d).

    n_tokens must be a perfect square.  Result: X_after[..., i, :] = X_before[..., σ_t(i), :].
    """
    n_tokens = X.shape[-2]
    sigma = permute_indices(n_tokens, t)
    return X[..., sigma, :]


# ---------------------------------------------------------------------------
# Mixing (value combination) primitives
# ---------------------------------------------------------------------------

def coupling_mix(X, step, half='alternate'):
    """Invertible coupling-layer-style mixing.

    Split token axis into two halves (by index).  Half A (source) is
    untouched; half B (destination) is updated as:
      X_B <- X_B + 0.5 * X_A

    Half 'alternate' swaps source/destination by step parity.
    This is the ONLY place where values actually mix across tokens.
    A pure permutation cannot mix values — it only rearranges them.
    """
    X = X.copy()
    n = X.shape[-2]
    half_n = n // 2
    if half == 'alternate':
        if step % 2 == 0:
            src, dst = slice(0, half_n), slice(half_n, n)
        else:
            src, dst = slice(half_n, n), slice(0, half_n)
    else:
        src, dst = slice(0, half_n), slice(half_n, n)
    X[..., dst, :] = X[..., dst, :] + 0.5 * X[..., src, :]
    return X


def coupling_unmix(X, step, half='alternate'):
    """Inverse of coupling_mix (same split, subtract the same source)."""
    X = X.copy()
    n = X.shape[-2]
    half_n = n // 2
    if half == 'alternate':
        if step % 2 == 0:
            src, dst = slice(0, half_n), slice(half_n, n)
        else:
            src, dst = slice(half_n, n), slice(0, half_n)
    else:
        src, dst = slice(0, half_n), slice(half_n, n)
    X[..., dst, :] = X[..., dst, :] - 0.5 * X[..., src, :]
    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def support_coverage(X, eps=1e-9):
    """Fraction of token positions whose value differs from zero."""
    nonzero = np.any(np.abs(X) > eps, axis=-1)
    return nonzero.mean(axis=-1)


def state_entropy(X, bins=32, eps=1e-12):
    """Discretized entropy of the flattened state (bits per value)."""
    flat = X.reshape(-1)
    lo, hi = flat.min(), flat.max()
    if hi - lo < 1e-12:
        return 0.0
    hist, _ = np.histogram(flat, bins=bins, range=(lo, hi))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def dispersion(pts, N):
    """Mean pairwise distance of points on the torus, normalized by N*sqrt(2).

    Higher = better spread.  1.0 = uniform-ish.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    d = 0.0
    count = 0
    for i in range(len(pts)):
        dx = np.abs(pts[i, 0] - pts[:, 0])
        dy = np.abs(pts[i, 1] - pts[:, 1])
        dx = np.minimum(dx, N - dx)
        dy = np.minimum(dy, N - dy)
        d += float(np.sqrt(dx**2 + dy**2).sum())
        count += len(pts)
    return d / (count * N * np.sqrt(2))


def chi2_uniform(pts, N, cells=None):
    """Chi-square statistic of cell occupancy vs uniform over the N x N grid.

    Lower = closer to uniform.
    """
    flat = pts[..., 0].astype(np.int64) * N + pts[..., 1].astype(np.int64)
    if cells is None:
        counts = np.bincount(flat, minlength=N * N)
        expected = len(flat) / (N * N)
    else:
        sample = np.random.choice(np.arange(N * N), size=min(cells, N * N),
                                  replace=False)
        counts = np.bincount(flat, minlength=N * N)[sample]
        expected = len(flat) / len(sample)
    return float(((counts - expected) ** 2 / expected).sum())


def interaction_reachability(n_tokens, T, schedule='arnold', n_trials=200,
                             rng=None):
    """Fraction of random (query, target) pairs whose info can reach each
    other within T steps of pairwise coupling.

    Model: each step t, tokens are paired by the permutation schedule and a
    coupling copies one-way info.  Reachability = query and target share an
    ancestor within the past-cone of depth T (i.e., the undirected graph
    formed by the union of the T matchings connects them).

    The graph depends only on (schedule, T), so we build the union-find ONCE
    per call and then test many pairs cheaply.

    schedule: 'arnold' (cat map), 'random' (fresh random permutation per
    step), 'fixed' (one fixed random permutation repeated), 'shift' (cyclic
    shift by 1 — a poor schedule).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    parent = np.arange(n_tokens)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for t in range(1, T + 1):
        if schedule == 'arnold':
            perm = permute_indices(n_tokens, t)
        elif schedule == 'random':
            perm = rng.permutation(n_tokens)
        elif schedule == 'fixed':
            if t == 1:
                fixed_perm = rng.permutation(n_tokens)
            perm = fixed_perm
        elif schedule == 'shift':
            perm = (np.arange(n_tokens) + 1) % n_tokens
        for i in range(n_tokens):
            union(i, perm[i])
    pairs = rng.integers(0, n_tokens, size=(n_trials, 2))
    roots = np.array([find(q) for q in range(n_tokens)])
    return float((roots[pairs[:, 0]] == roots[pairs[:, 1]]).mean())


# ---------------------------------------------------------------------------
# FLOP accounting (honest: every operation is counted)
# ---------------------------------------------------------------------------

def flops_attention(n, d, heads=1):
    """FLOPs of one full self-attention layer over n tokens, dim d."""
    proj = 3 * n * d * d * 2
    scores = n * n * d * 2 + 5 * n * n
    attn_out = n * n * d * 2
    out_proj = n * d * d * 2
    return proj + scores + attn_out + out_proj


def flops_mlp(n, d, hidden=None):
    """FLOPs of a per-token MLP (2-layer, hidden=4d)."""
    if hidden is None:
        hidden = 4 * d
    return n * (2 * d * hidden + 2 * hidden * d)


def flops_chaotic_step(n, d):
    """FLOPs of ONE chaotic step: permutation (index arithmetic ~ n*8) +
    coupling mixing (n/2 × d adds)."""
    perm = n * 8
    mix = (n // 2) * d * 2
    return perm + mix


def flops_gsf_pooled(n, d, hidden=None):
    """FLOPs of a GSF that pools all tokens (mean: n*d adds) and runs an MLP
    of size d -> hidden -> control.  The pooling is O(n*d) and is NOT hidden."""
    if hidden is None:
        hidden = 2 * d
    pool = n * d
    mlp = 2 * d * hidden + 2 * hidden * 1
    return pool + mlp