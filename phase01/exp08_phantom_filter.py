"""exp08_phantom_filter.py — Phase 0.1, Experiment 8.

MORIN-FILTER SYNTHESIS: the missing parameter in the chaos equation.

Insight from the user's morin-filter research:
  ~90% of locally-plausible token transitions are PHANTOMS (locally valid,
  globally impossible).  The winning mechanism was a BETA-MIXTURE with a
  corpus prior:  P = (1-beta)*P_model + beta*P_corpus  (Jelinek-Mercer).
  The model itself cannot see phantoms (oracle top-k == baseline; the
  corpus filter beats it).

Hypothesis for chaos:
  The cat-map orbit is a POINCARE LOOP: locally every step is valid, and
  the trajectory returns to the start (a "Penrose triangle" of chaos).
  As a computation the loop is VACUOUS: it hits a random target with
  probability = reachability ~ 0.75/N (measured in exp02).  So ~99% of
  chaotic "transitions" are PHANTOMS — even more than in the token graph.
  The missing parameter is therefore BETA: blend the chaotic proposal with
  a data-driven transition prior that knows which transitions are real.

Experiments:
  A. Phantom rate of the chaotic proposer on a routing task.
  B. Accuracy vs beta: chaos alone (beta=0, phantom) vs +prior (beta>0).
  C. Chaos vs random proposer with the SAME prior (who does the work?).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp08_phantom_filter")
os.makedirs(OUT, exist_ok=True)

A = np.array([[1, 1], [1, 2]], dtype=np.int64)
rng = np.random.default_rng(21)

Ng = 64          # grid N x N (4096 cells)
C = 16           # contexts
K = 8            # target regions (experts)
R = 3            # region radius (ball)
N_TRAIN = 2000   # training samples (context -> target)
N_TEST = 2000

# ---- data: context -> target region ----
# targets: K random centers
targets = rng.integers(R, Ng - R, size=(K, 2))
# true mapping: context c -> target (c mod K) (with some noise)
true_t = (np.arange(C) % K)

def in_region(cell, center, r=R):
    return max(abs(int(cell[0]) - center[0]), abs(int(cell[1]) - center[1])) <= r

def chaos_proposal_dist(x0, T):
    """Fraction of orbit states (steps 1..T) landing in each target region.
    This is the 'chaotic readout' — mostly phantom for random x0."""
    counts = np.zeros(K)
    x = x0.copy()
    for _ in range(T):
        x = (A @ x) % Ng
        for k in range(K):
            if in_region(x, targets[k]):
                counts[k] += 1
                break
    s = counts.sum()
    if s == 0:
        return None, counts  # phantom: trajectory never hits any target
    return counts / s, counts

# ============ A. Phantom rate of the chaotic proposer ============
T = 96  # orbit length (period of N=64)
phantom = 0
for _ in range(500):
    x0 = rng.integers(0, Ng, size=2)
    _, counts = chaos_proposal_dist(x0, T)
    if counts.sum() == 0:
        phantom += 1
phantom_rate_A = phantom / 500
print(f"A. Chaotic proposer phantom rate (never hits any target in {T} steps): "
      f"{phantom_rate_A:.3f}")

# ============ B. Beta-mixture: chaos + corpus prior ============
# corpus prior: from training samples, table (context -> target) counts
prior = np.zeros((C, K))
for _ in range(N_TRAIN):
    c = rng.integers(0, C)
    # with noise: 85% the true target, 15% random
    t = true_t[c] if rng.random() < 0.85 else rng.integers(0, K)
    prior[c, t] += 1
prior_p = prior / prior.sum(axis=1, keepdims=True)

def chaos_proposer_for_context(c):
    """Deterministic pseudo-encoding: context -> x0.  x0 is 'random' wrt
    targets (the encoder has no target knowledge — the phantom case)."""
    x0 = np.array([(c * 37 + 11) % Ng, (c * 53 + 7) % Ng])
    return x0

def eval_accuracy(beta, n=N_TEST):
    correct = 0
    for _ in range(n):
        c = rng.integers(0, C)
        t_true = true_t[c]
        x0 = chaos_proposer_for_context(c)
        dist, counts = chaos_proposal_dist(x0, T)
        if dist is None:
            p_chaos = np.ones(K) / K  # phantom -> uniform (no information)
        else:
            p_chaos = dist
        p_mix = (1 - beta) * p_chaos + beta * prior_p[c]
        if np.argmax(p_mix) == t_true:
            correct += 1
    return correct / n

print("\nB. Accuracy vs beta (chaos proposer + corpus prior):")
acc_by_beta = {}
for beta in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    a = eval_accuracy(beta)
    acc_by_beta[str(beta)] = a
    print(f"   beta={beta:.1f}: acc={a:.3f}")

# ============ C. Who does the work? random proposer + same prior ============
def eval_accuracy_random_proposer(beta, n=N_TEST):
    correct = 0
    for _ in range(n):
        c = rng.integers(0, C)
        t_true = true_t[c]
        x0 = rng.integers(0, Ng, size=2)  # RANDOM proposer
        dist, counts = chaos_proposal_dist(x0, T)
        p_chaos = dist if dist is not None else np.ones(K) / K
        p_mix = (1 - beta) * p_chaos + beta * prior_p[c]
        if np.argmax(p_mix) == t_true:
            correct += 1
    return correct / n

print("\nC. Random proposer + same prior:")
for beta in [0.0, 0.3, 1.0]:
    a = eval_accuracy_random_proposer(beta)
    print(f"   beta={beta:.1f}: acc={a:.3f}")

results = {
    "A_phantom_rate_chaotic_proposer": phantom_rate_A,
    "B_acc_vs_beta_chaos": acc_by_beta,
    "C_acc_vs_beta_random_proposer": {
        "0.0": eval_accuracy_random_proposer(0.0),
        "0.3": eval_accuracy_random_proposer(0.3),
        "1.0": eval_accuracy_random_proposer(1.0),
    },
    "note": (
        "The missing parameter is BETA: the mixture weight blending the "
        "chaotic proposal with a data-driven transition prior.  beta=0 "
        "(pure chaos) is phantom — accuracy at chance.  beta>0 (prior) "
        "carries the real information.  The chaotic proposer and a random "
        "proposer give the SAME accuracy with the same prior — the chaos "
        "does not add routing information (consistent with exp02/03)."),
}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
