"""exp01_addressable_iteration.py — Phase 0.1, Experiment 1.

ADDRESSABLE ITERATION: can x_t = A^t x_0 (mod N) be computed WITHOUT
simulating all intermediate states?

The cat map A = [[1,1],[1,2]] has a CLOSED FORM:
    A^t = [[F_{2t-1}, F_{2t}], [F_{2t}, F_{2t+1}]]   (Fibonacci numbers)
so x_t is computable in O(log t) (fast doubling) or O(1) (matrix form),
versus O(t) for sequential iteration.

This is the foundation of the "chaos as addressable computational space"
idea: if the trajectory is closed-form, any state can be accessed directly.

CRITICAL HONESTY QUESTION (pre-registered):
  - The closed form exists because the map is LINEAR.
  - A linear map is a single linear function of x_0 — "depth" via iteration
    is fake: F^t(x_0) = (A^t) x_0 is ONE linear transformation, no richer
    than a single layer.
  - The baseline "sequential O(t)" is a strawman: nobody iterates a linear
    map step by step; the honest baseline for "reach state at time t" is
    O(log t) (matrix power) — the same as the chaotic claim.
  We measure all of this explicitly.
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp01_addressable_iteration")
os.makedirs(OUT, exist_ok=True)

A = np.array([[1, 1], [1, 2]], dtype=np.int64)
N_GRID = 1000  # grid N x N (state space 10^6 cells)


def fib_pair_mod(n, N):
    """(F_n, F_{n+1}) mod N via fast doubling, O(log n), all values mod N."""
    if n == 0:
        return (0, 1 % N)
    a, b = fib_pair_mod(n >> 1, N)
    c = (a * ((2 * b - a) % N)) % N
    d = (a * a + b * b) % N
    if n & 1:
        return (d, (c + d) % N)
    return (c, d)


def A_power_closed(t, N):
    """A^t mod N via Fibonacci closed form (all mod N)."""
    f2tm1, f2t = fib_pair_mod(2 * t - 1, N)
    _, f2tp1 = fib_pair_mod(2 * t, N)
    M = np.array([[f2tm1, f2t], [f2t, f2tp1]], dtype=np.int64)
    return M


def A_power_fast(t, N):
    """A^t mod N via binary exponentiation, O(log t) 2x2 matmuls."""
    result = np.eye(2, dtype=np.int64)
    base = A % N
    t = int(t)
    while t > 0:
        if t & 1:
            result = (result @ base) % N
        base = (base @ base) % N
        t >>= 1
    return result


def A_power_sequential(t, N):
    """A^t mod N by t sequential 2x2 matmuls (strawman baseline)."""
    result = np.eye(2, dtype=np.int64)
    for _ in range(t):
        result = (result @ A) % N
    return result


def reach_closed(x0, t, N):
    return (A_power_closed(t, N) @ x0) % N


def reach_fast(x0, t, N):
    return (A_power_fast(t, N) @ x0) % N


def reach_sequential(x0, t, N):
    x = x0.copy()
    for _ in range(t):
        x = (A @ x) % N
    return x


results = {"N_grid": N_GRID, "closed_form": "A^t = [[F_{2t-1},F_{2t}],[F_{2t},F_{2t+1}]]"}

# ---- 1. Correctness: closed form == fast power == sequential (small t) ----
x0 = np.array([123, 456])
correctness = {}
for t in [1, 2, 3, 7, 10, 100, 1000]:
    a = reach_closed(x0, t, N_GRID)
    b = reach_fast(x0, t, N_GRID)
    c = reach_sequential(x0, t, N_GRID)
    correctness[str(t)] = {
        "closed_vs_fast": bool(np.array_equal(a, b)),
        "closed_vs_seq": bool(np.array_equal(a, c)),
        "state": a.tolist(),
    }
results["correctness"] = correctness
results["correctness_all_ok"] = all(
    v["closed_vs_fast"] and v["closed_vs_seq"] for v in correctness.values())

# ---- 2. Timing: reach x_t for huge t ----
times = [10, 100, 1000, 10_000, 10**6, 10**9, 10**12, 10**18]
timing = {}
for t in times:
    # closed form (fast doubling)
    t0 = time.perf_counter()
    s1 = reach_closed(x0, t, N_GRID)
    t1 = time.perf_counter() - t0
    # fast matrix power
    t0 = time.perf_counter()
    s2 = reach_fast(x0, t, N_GRID)
    t2 = time.perf_counter() - t0
    # sequential — only for small t (t<=1000), else extrapolate
    if t <= 1000:
        t0 = time.perf_counter()
        s3 = reach_sequential(x0, t, N_GRID)
        t3 = time.perf_counter() - t0
        ok3 = bool(np.array_equal(s1, s3))
    else:
        # measure one step and extrapolate honestly: per-step cost * t
        t0 = time.perf_counter()
        for _ in range(1000):
            reach_sequential(x0, 1, N_GRID)
        per_step = (time.perf_counter() - t0) / 1000
        t3 = per_step * t
        ok3 = "extrapolated"
    timing[str(t)] = {
        "t_closed_s": t1, "t_fastpow_s": t2, "t_sequential_s": t3,
        "sequential_agree": ok3,
        "speedup_vs_sequential": (t3 / t1) if t1 > 0 else None,
        "log2_t": np.log2(t),
    }
results["timing"] = timing

# ---- 3. The decisive analysis: what IS the honest baseline? ----
# For a LINEAR map, "reach state at time t" = evaluate A^t x_0.
# The matrix power itself is O(log t).  The "sequential O(t)" baseline is
# a strawman.  The closed form is O(log t) — EQUAL to the honest baseline,
# not better.  (For this 2x2 map it's even O(1) with Binet-like formulas.)
results["honest_baseline_note"] = (
    "For an autonomous LINEAR map the honest baseline for x_t is the matrix "
    "power O(log t), NOT sequential iteration O(t).  The chaotic closed form "
    "matches the honest baseline; the O(t) comparison is a strawman.")

# ---- 4. The poverty of linearity: what can the trajectory compute? ----
# x_t = A^t x_0 is linear in x_0 for every t.  A linear map cannot implement
# XOR / parity / any nonlinear function of the input bits.
# Verify empirically: parity of a 2-bit input (x0 in {0,1}^2 on the grid).
from itertools import product

def parity_linear_check():
    """Check whether any LINEAR readout of x_t can solve parity on 2 bits.
    x0 = (a, b) with a,b in {0,1}; parity = (a+b) mod 2.
    For each t: x_t = A^t x_0.  A linear readout w·x_t + c over GF-ish mod 2
    can't separate the 4 parity classes... we check mod-2 linearity directly:
    parity is linear over GF(2): parity(a,b)=a XOR b.  x_t mod 2 is a linear
    function of (a,b) over GF(2).  XOR IS linear over GF(2)!
    BUT over Z_N (the actual state space), XOR is NOT a linear function of
    the Z_N-linear map.  The correct test: the state x_t mod 2 = M_t (a,b)^T
    mod 2, a LINEAR map over GF(2).  A linear map over GF(2) CAN represent
    XOR.  So parity IS linearly representable from x_t mod 2!
    The nontrivial nonlinearity test: MULTIPLICATION (a*b mod 2) — AND.
    AND is not GF(2)-linear: no linear readout of x_t can output a AND b.
    """
    res = {}
    for t in [1, 2, 3, 5, 10]:
        # enumerate all 4 inputs
        X = np.array([[a, b] for a in [0, 1] for b in [0, 1]])
        Xt = np.array([reach_closed(x, t, N_GRID) for x in X])
        y_and = np.array([(a & b) for a, b in X])
        y_xor = np.array([(a ^ b) for a, b in X])
        # try all linear readouts of Xt mod 2 (2 coefficients + bias):
        # a linear map over GF(2) from 2 dims: 2^2 * 2^2 * 2^2 possibilities
        def linear_representable(Xt, y, N):
            X2 = Xt % 2  # GF(2) features (2 dims)
            for w1 in [0, 1]:
                for w2 in [0, 1]:
                    for b in [0, 1]:
                        pred = (X2[:, 0] * w1 + X2[:, 1] * w2 + b) % 2
                        if np.array_equal(pred, y):
                            return True
            return False
        res[str(t)] = {
            "xor_linear_representable": linear_representable(Xt, y_xor, N_GRID),
            "and_linear_representable": linear_representable(Xt, y_and, N_GRID),
        }
    return res

results["linear_computation_limits"] = linear_parity_check() if False else None
# (fixed below — see run)

# run the check properly
from itertools import product as _p
lin = {}
for t in [1, 2, 3, 5, 10]:
    X = np.array([[a, b] for a in [0, 1] for b in [0, 1]])
    Xt = np.array([reach_closed(x, t, N_GRID) for x in X])
    y_and = np.array([(a & b) for a, b in X])
    y_xor = np.array([(a ^ b) for a, b in X])
    def lin_rep(Xt, y):
        X2 = Xt % 2
        for w1 in [0, 1]:
            for w2 in [0, 1]:
                for b in [0, 1]:
                    pred = (X2[:, 0] * w1 + X2[:, 1] * w2 + b) % 2
                    if np.array_equal(pred, y):
                        return True
        return False
    lin[str(t)] = {"xor": lin_rep(Xt, y_xor), "and": lin_rep(Xt, y_and)}
results["linear_computation_limits"] = lin

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print("=== Correctness (closed form == fast power == sequential) ===")
print("all ok:", results["correctness_all_ok"])
print("\n=== Timing ===")
print(f"{'t':>10} {'closed(s)':>12} {'fastpow(s)':>12} {'sequential(s)':>14} {'speedup':>10}")
for t, v in timing.items():
    su = v["speedup_vs_sequential"]
    print(f"{t:>10} {v['t_closed_s']:>12.2e} {v['t_fastpow_s']:>12.2e} "
          f"{v['t_sequential_s']:>14.2e} {su:>10.1e}")
print("\n=== Linear computation limits (can a linear readout of x_t solve?) ===")
for t, v in lin.items():
    print(f"  t={t:>3}: XOR (parity) linear-representable: {v['xor']}, AND: {v['and']}")
print("\nclosed form x_1e18 =", reach_closed(x0, 10**18, N_GRID).tolist())
