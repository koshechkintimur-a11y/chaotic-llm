"""
toy_data.py — data generators for toy tasks A–D (v2: one-hot + CE).

Uniform interface:
  input:  (B, N, d_in)  — one-hot/feature encoding per token
  target: (B,) long     — class index
  qmask:  (B, N)        — 1.0 on query token(s)

Task A — associative recall:  K unique (key,value) pairs + query key → value.
Task B — long-range dependency: A ... fillers ... B.  Predict B's value from A.
Task C — hierarchical retrieval: 3-level tree, query path (root, child) → grandchild.
Task D — compositional reasoning: chain A→B→C→D.  Predict D from A.

Input channels are one-hot so key-matching is a dot product (learnable easily).
"""
import numpy as np
import torch


def taskA_batch(bs, N=16, K=8, V=16, rng=None):
    """N memory tokens: K unique keys (each repeated N/K times), values random.
    Query token carries the query key (one-hot).  Answer: value of that key.

    We make the task UNAMBIGUOUS: keys 0..K-1 each appear exactly N/K times
    with the SAME value per key, so the answer is well-defined.
    """
    rng = rng if rng is not None else np.random.default_rng()
    per = N // K
    keys = np.repeat(np.arange(K), per)          # (N,)  each key appears per times
    # value per key (K values), then broadcast to tokens
    keyvals = rng.integers(0, V, size=(bs, K))
    vals = keyvals[:, keys]                       # (bs, N)
    # query: pick one key per sample
    qkey = rng.integers(0, K, size=(bs,))
    d_in = K + V
    inp = np.zeros((bs, N, d_in), dtype=np.float32)
    inp[:, :, :K] = np.eye(K)[keys][None, :, :]   # key one-hot
    inp[:, :, K:] = np.eye(V)[vals.astype(int)]   # value one-hot
    # query token: replace with query-key one-hot + zero value
    # NOTE: query at the END (N-1) — fair for sequential models (GRU) and
    # avoids the cat map's fixed point at grid (0,0).
    qpos = np.full(bs, N - 1, dtype=int)
    inp[np.arange(bs), qpos, :K] = np.eye(K)[qkey]
    inp[np.arange(bs), qpos, K:] = 0.0
    qmask = np.zeros((bs, N), dtype=np.float32)
    qmask[np.arange(bs), qpos] = 1.0
    target = keyvals[np.arange(bs), qkey].astype(np.int64)  # value of query key
    return (torch.tensor(inp), torch.tensor(qmask), torch.tensor(target))


def taskB_batch(bs, L, N_max=512, K=4, V=16, rng=None):
    """A ... L fillers ... B.  Predict A's value given the query at B's
    position (the answer must TRAVEL across the L fillers — honest long-range).

    Encoding: token0 = A (class one-hot over V + marker), fillers = zero,
    last = B (marker only, empty).  qmask at the LAST position: the model
    must bring A's identity to the end.
    """
    rng = rng if rng is not None else np.random.default_rng()
    N = L + 2
    d_in = V + 1  # V classes + marker
    a = rng.integers(0, V, size=(bs,))
    inp = np.zeros((bs, N, d_in), dtype=np.float32)
    inp[:, 0, :V] = np.eye(V)[a]
    inp[:, 0, V] = 1.0        # A marker
    inp[:, -1, V] = 2.0       # B marker
    qmask = np.zeros((bs, N), dtype=np.float32)
    qmask[:, L + 1] = 1.0     # query at B — needs info from A
    return (torch.tensor(inp), torch.tensor(qmask), torch.tensor(a.astype(np.int64)))


def taskC_batch(bs, N=16, K=8, V=32, rng=None):
    """3-level tree in DFS order.  Query (root, child) → grandchild value.
    Types: root=0, child=1, grandchild=2, filler=-1.
    d_in = V (node values one-hot) + 3 (type one-hot).
    """
    rng = rng if rng is not None else np.random.default_rng()
    # tree layout: root(0), child0(1), gc(2,3), child1(4), gc(5,6), child2(7),
    # gc(8,9), child3(10), gc(11,12), filler(13,14,15)
    n_nodes = 13
    vals = rng.integers(0, V, size=(bs, n_nodes))
    c = rng.integers(0, 4, size=(bs,))
    g = rng.integers(0, 2, size=(bs,))
    gc_idx = 2 + 3 * c + g
    types = [0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, -1, -1, -1]
    d_in = V + 3
    inp = np.zeros((bs, N, d_in), dtype=np.float32)
    inp[:, :, :V] = np.eye(V)[vals] if n_nodes == N else \
        np.concatenate([np.eye(V)[vals], np.zeros((bs, N - n_nodes, V))], axis=1)
    inp[:, :n_nodes, V + 1:] = 0.0
    # type one-hot (3 dims): 0=root, 1=child, 2=gc
    for i, tp in enumerate(types):
        if tp >= 0:
            inp[:, i, V + tp] = 1.0
    qmask = np.zeros((bs, N), dtype=np.float32)
    qmask[:, 0] = 1.0
    qmask[np.arange(bs), 1 + 3 * c] = 1.0
    target = vals[np.arange(bs), gc_idx].astype(np.int64)
    return (torch.tensor(inp), torch.tensor(qmask), torch.tensor(target))


def taskD_batch(bs, N=9, V=16, rng=None):
    """Chain A→B→C→D.  Given the chain (A,B,C,D shown), predict D's value
    from A's value (compositional: the answer follows the chain).
    N = 9 (3^2) with 4 chain tokens + 5 zero fillers (ChaoticMixer needs
    a perfect-square token count).  d_in = V + 4.
    """
    rng = rng if rng is not None else np.random.default_rng()
    a = rng.integers(0, V, size=(bs,))
    d = rng.integers(0, V, size=(bs,))  # independent target for compositional test
    d_in = V + 4
    inp = np.zeros((bs, N, d_in), dtype=np.float32)
    chain_vals = np.stack([a, rng.integers(0, V, size=(bs,)),
                           rng.integers(0, V, size=(bs,)), d], axis=1)  # (bs,4)
    for i in range(4):
        inp[:, i, :V] = np.eye(V)[chain_vals[:, i]]
        inp[:, i, V + i] = 1.0
    qmask = np.zeros((bs, N), dtype=np.float32)
    qmask[:, 0] = 1.0
    return (torch.tensor(inp), torch.tensor(qmask), torch.tensor(d.astype(np.int64)))


def d_in_for(task):
    if task == "A":
        return 8 + 16
    if task == "B":
        return 16 + 1
    if task == "C":
        return 32 + 3
    if task == "D":
        return 16 + 4
    raise ValueError(task)
