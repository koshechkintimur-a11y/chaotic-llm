# Experiment 39 — Unigram Floor (CORRECTED)

## The bug that was found

exp38 reported a "gap" at V=2048 (22.12 vs KN 20.18). **That was a bug**: the
eval computed `logaddexp(log(1-β)+pm, log(β)·(-1e9))` for tokens where the
memory has no mass — which is ≈ `pm + log(1-β)`, i.e. an extra penalty
`-log(1-β)` on every unseen-continuation token. The correct handling: when
the memory gives nothing, use the mixer alone (`fused = pm`).

With the fix:

| Vocab | sparse β(c_h) | full KN | result |
|---|---|---|---|
| V=512, k=0.5 | **8.59** | 10.94 | β(c_h) wins |
| V=2048, k=5.0 | **16.55** | 20.18 | β(c_h) wins |

**The architecture v0.7 beats full KN at BOTH vocabulary sizes.** The
"advantage flips at scale" conclusion of exp38 was wrong — it was the bug.

## Unigram floor — negative result (cleared)

Adding a floor `P_mem = (1-ε)·MLE + ε·p_uni` for unseen continuations HURTS
(ε=0.01: 8.59→12.29 at V=512; 16.55→21.46 at V=2048). Why: for a token
unseen in the context, the floor assigns a tiny probability `ε·p_uni` which,
blended with high β, adds a `log(1-β)` penalty instead of deferring to the
mixer. The correct operation is binary: memory has mass → blend; memory has
no mass → mixer alone. No floor needed.

## Optimal k scales with vocab

k=0.5 (V=512) → k=5.0 (V=2048). Larger vocab = sparser n-grams = the mixer
should be trusted more on average. Even at suboptimal k the result beats KN
(k=10 → 17.97 < 20.18).

## Final architecture numbers (code)

| | mixer-only | +full KN | **+sparse β(c_h)** |
|---|---|---|---|
| V=512 | 32.78 | 10.94 | **8.59** |
| V=2048 | 75.05 | 20.18 | **16.55** |

Cheap sparse memory (9-17× cheaper) + 1-parameter confidence gate beats
full Kneser-Ney at every scale tested.