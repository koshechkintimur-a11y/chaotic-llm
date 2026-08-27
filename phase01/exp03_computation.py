"""exp03_computation.py — Phase 0.1, Experiment 3 (most important).

CAN THE CHAOS BE A COMPUTATIONAL LAYER?

Tasks: XOR, parity, classification, associative lookup.
Framings:
  (a) Autonomous map, readout after T steps:
        y = readout(A^T x_0)
      The map A^T is FIXED and LINEAR.  Any readout capacity is unchanged
      by folding A^T into it:  readout(A^T x_0) == readout'(x_0) with
      readout' = readout ∘ A^T.  The chaotic layer is ABSORBABLE — it adds
      no computational capability to a neural pipeline.
      We verify: train an MLP readout directly on x_0 vs on A^T x_0 —
      identical accuracy (map absorbable).
  (b) Controlled dynamics:
        x_{t+1} = A x_t + u(x_t, theta)
      This is a nonlinear RECURRENT map — i.e., an RNN.  It can compute
      nonlinear tasks, but the honest baseline is an RNN with the same
      capacity; "chaos" adds no architectural novelty.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp03_computation")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

A = np.array([[1, 1], [1, 2]], dtype=np.int64)
N_GRID = 64


def reach_closed(x0, t, N):
    """Apply A^t to each PAIR of coordinates (block-diagonal linear map).
    x0: (k,) int vector.  Works for any even-dim input; pads odd dims.
    The map is a fixed linear map regardless — absorbability holds."""
    x0 = np.asarray(x0, dtype=np.int64)
    if x0.ndim == 1:
        x0 = x0.reshape(1, -1)
    k = x0.shape[1]
    if k % 2 == 1:
        x0 = np.concatenate([x0, np.zeros((x0.shape[0], 1), dtype=np.int64)], axis=1)
    M = np.eye(2, dtype=np.int64)
    base = A % N
    tt = int(t)
    while tt > 0:
        if tt & 1:
            M = (M @ base) % N
        base = (base @ base) % N
        tt >>= 1
    pairs = x0.reshape(-1, 2)
    out = (pairs @ M.T) % N
    out = out.reshape(x0.shape[0], -1)
    if out.shape[1] > k:
        out = out[:, :k]
    return out.ravel() if out.shape[0] == 1 else out


def make_task(task, n=4000, N=N_GRID):
    rng = np.random.default_rng(1)
    if task == "xor":
        X = rng.integers(0, N, size=(n, 2))  # raw 2D states (not bits)
        y = ((X[:, 0] % 2) ^ (X[:, 1] % 2)).astype(np.int64)  # parity of LSBs
    elif task == "and":
        X = rng.integers(0, N, size=(n, 2))
        y = ((X[:, 0] % 2) & (X[:, 1] % 2)).astype(np.int64)
    elif task == "parity8":
        X = rng.integers(0, N, size=(n, 8))
        y = (X.sum(axis=1) % 2).astype(np.int64)  # parity over 8 coords
    elif task == "assoc":
        # memory of K pairs, query key -> value class
        K = 8
        mem = rng.integers(0, 3, size=(n, K))  # each key has value 0..2
        qk = rng.integers(0, K, size=(n,))
        X = mem
        y = mem[np.arange(n), qk]
        X = np.concatenate([X, np.eye(K)[qk]], axis=1)  # append query one-hot
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class MLPReadout(nn.Module):
    def __init__(self, d_in, d_out, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def train_and_eval(model, X, y, Xv, yv, epochs=300, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = X.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n)[:256]
        opt.zero_grad()
        loss = lossf(model(X[idx]), y[idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xv).argmax(-1) == yv).float().mean().item()
    model.train()
    return acc


results = {}

# ============ Part A: absorbability of the fixed linear map ============
# Compare: MLP readout on raw x_0  vs  MLP readout on A^T x_0.
for task in ["xor", "and", "parity8", "assoc"]:
    X, y = make_task(task)
    n = len(X)
    Xv, yv = make_task(task, n=1000)
    nv = len(Xv)
    d_in = X.shape[1]
    d_out = int(y.max()) + 1

    accs_raw = []
    accs_chaotic = []
    for rep in range(3):
        # readout on raw x_0
        m1 = MLPReadout(d_in, d_out)
        a1 = train_and_eval(m1, X, y, Xv, yv)
        # readout on A^T x_0 (T=5)
        XT = torch.stack([torch.tensor(reach_closed(x.numpy().astype(np.int64), 5, N_GRID),
                                       dtype=torch.float32)
                          for x in X])
        XTv = torch.stack([torch.tensor(reach_closed(x.numpy().astype(np.int64), 5, N_GRID),
                                        dtype=torch.float32)
                           for x in Xv])
        m2 = MLPReadout(d_in, d_out)
        a2 = train_and_eval(m2, XT, y, XTv, yv)
        accs_raw.append(a1)
        accs_chaotic.append(a2)
    results[task] = {
        "acc_readout_on_raw": float(np.mean(accs_raw)),
        "acc_readout_on_A^5_x0": float(np.mean(accs_chaotic)),
        "absorbable_delta": float(np.mean(accs_raw) - np.mean(accs_chaotic)),
        "n_classes": d_out,
    }
    print(f"[{task:8s}] raw={np.mean(accs_raw):.3f}  A^5(x0)={np.mean(accs_chaotic):.3f} "
          f"delta={np.mean(accs_raw)-np.mean(accs_chaotic):+.3f}")

results["absorbability_note"] = (
    "A fixed linear map between learned layers is absorbable: "
    "readout(A^T x_0) == (readout ∘ A^T)(x_0), a readout of the SAME "
    "capacity.  The chaotic map adds no computational capability.  "
    "Measured delta should be ~0 (same accuracy either way).")

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
