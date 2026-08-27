"""exp07_poincare_recurrence.py — Phase 0.1, Experiment 7.

The user's idea: the cat map turns an image into noise, and after EXACTLY
T(N) iterations (Poincare recurrence) the noise reassembles into the
original.  "We need to understand the iteration count and control it."

What we verify and measure:
  1. Reassembly is EXACT at t = T(N) (and only at multiples of T(N)).
  2. The number of DISTINCT scrambled states on the way back = T(N) — this
     is the "iteration count" one would need to control.
  3. Controlling t gives access to exactly the ORBIT of the image:
     T(N) states out of N^2 — a fraction T(N)/N^2 ~ 0.75/N of the space.
  4. What "period control" buys: selecting t among {1..T(N)} = selecting
     among the orbit states = a piecewise-linear function with <= T(N)
     pieces (each piece A^t x is linear).  Bounded by linearity.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp07_poincare_recurrence")
os.makedirs(OUT, exist_ok=True)

A = np.array([[1, 1], [1, 2]], dtype=np.int64)
Ainv = np.array([[2, -1], [-1, 1]], dtype=np.int64)
rng = np.random.default_rng(11)


def period(N):
    a, b, c, d = 1, 0, 0, 1
    t = 0
    while True:
        t += 1
        a, b, c, d = (a + c) % N, (b + d) % N, (a + 2 * c) % N, (b + 2 * d) % N
        if (a, b, c, d) == (1, 0, 0, 1):
            return t


def apply_map(img, N, inv=False):
    """Apply the cat map (or inverse) to an N x N image: value at (x,y)
    moves to A·(x,y) mod N."""
    M = Ainv if inv else A
    out = np.empty_like(img)
    for x in range(N):
        for y in range(N):
            nx = (M[0, 0] * x + M[0, 1] * y) % N
            ny = (M[1, 0] * x + M[1, 1] * y) % N
            out[nx, ny] = img[x, y]
    return out


results = {}

# ============ 1+2: exact reassembly at T(N) ============
for Ng in [16, 32, 64]:
    T = period(Ng)
    img = rng.integers(0, 256, size=(Ng, Ng))
    cur = img.copy()
    first_return = None
    distinct_states = set()
    for t in range(1, 2 * T + 2):
        cur = apply_map(cur, Ng)
        distinct_states.add(cur.tobytes())
        if np.array_equal(cur, img):
            first_return = t
            break
    # verify: T(N) == first return
    results[str(Ng)] = {
        "T(N)": T,
        "first_exact_return_t": first_return,
        "reassembly_exact_at_T": (first_return == T),
        "distinct_scramblings_before_return": len(distinct_states),
        "orbit_fraction_of_space": T / (Ng * Ng),
    }
    print(f"N={Ng:3d}: T(N)={T:4d}  first return={first_return}  "
          f"exact={first_return == T}  distinct scramblings={len(distinct_states)}  "
          f"orbit fraction={T/(Ng*Ng):.4f}")

# ============ 3: period control = reachable set = orbit ============
# "Controlling the iteration count" = choosing t -> state A^t(img).
# The reachable set is the ORBIT: exactly T(N) distinct states.
Ng = 64
T = period(Ng)
img = rng.integers(0, 256, size=(Ng, Ng))
reachable = set()
cur = img.copy()
for t in range(T):
    cur = apply_map(cur, Ng)
    reachable.add(cur.tobytes())
results["period_control_reachable"] = {
    "N": Ng,
    "states_reachable_by_controlling_t": len(reachable),
    "total_states": Ng * Ng,
    "fraction": len(reachable) / (Ng * Ng),
}
print(f"\nPeriod control (choosing t) reaches {len(reachable)}/{Ng*Ng} "
      f"states = {len(reachable)/(Ng*Ng):.4f} of the space (just the orbit)")

# ============ 4: what does reassembly give computationally? ============
# Reassembly at T-t is the INVERSE map:  A^{T-t}(A^t x) = x.
# So "unscrambling" = applying a LINEAR map A^{T-t} — no computation,
# just linear algebra, addressable in O(log t) (exp01).
results["reassembly_note"] = (
    "Reassembly after T(N) steps is exact (Poincare recurrence).  "
    "But 'unscrambling' the state at step t is simply applying the "
    "inverse LINEAR map A^{T-t} — addressable in O(log t) (exp01), "
    "and it performs NO computation: it is linear algebra.  "
    "Controlling t selects among the T(N) orbit states — a "
    "piecewise-linear function with <= T(N) pieces.  Expressivity is "
    "bounded by linearity (exp01: no AND).  And information never "
    "leaves the orbit (exp02: 0.75/N of the space).")

# ============ 5: orbit confinement of a REGION (image patch) ============
# A patch of the image travels along its orbit: it can only ever interact
# with cells on that orbit.  Measure the orbit of one pixel.
Ng = 64
px = np.array([5, 7])
cells = set()
cur = px.copy()
for t in range(period(Ng)):
    cells.add((int(cur[0]), int(cur[1])))
    cur = (A @ cur) % Ng
results["pixel_orbit"] = {
    "pixel": px.tolist(),
    "orbit_size": len(cells),
    "space_size": Ng * Ng,
    "fraction": len(cells) / (Ng * Ng),
}
print(f"Pixel {px.tolist()} visits {len(cells)} cells = "
      f"{len(cells)/(Ng*Ng):.4f} of the space (its orbit)")

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=int)
print("\nSaved to", OUT)
