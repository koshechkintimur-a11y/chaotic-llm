# Experiment 28 — Learned β-Controller (THE central experiment)

## Question

Can a learned β-controller route each token to memory or compute **per-token**,
and CUT the average β without losing PPL / generation quality?

If yes → the β-Architecture becomes a genuinely new compute regime:
**chaotic compute as the default (O(W log W), cheap) + sparse exact memory
(on demand, controlled per token)** — selective computation, the project's
original thesis.

## Setup

- Frozen mixer V1 (vocab 512, code, 90K params) + KN order-3 table.
- Per-position windows [i, i+W) → batched mixer forward → features
  (h_last ⊕ gvec) — exact exp18 eval convention (mixer-alone PPL 33.2
  reproduces exp18's 32.8 ✓).
- Controller: MLP(feats) → β ∈ (0,1). Loss: -log((1-β)p_mix(y) + β·p_mem(y)) + λ·β.
- Oracle: per-token binary β* = 1 iff p_mem(y) > p_mix(y).
- Honest train/test split (corpus_train / corpus_test).

## Results

### Fixed-β vs oracle (per-token binary β*) — eval (12K positions)

| Method | Avg β | PPL | top-1 |
|---|---|---|---|
| Mixer alone (β=0) | 0 | 33.24 | 23.1% |
| Best fixed β=0.9 | 0.900 | 10.88 | 41.4% |
| **Oracle (binary per-token)** | **0.758** | **8.28** | **44.5%** |

Oracle Pareto-domination at every operating point:

| Oracle avg β | Oracle PPL | Fixed-β PPL at same β |
|---|---|---|
| 0.758 | **8.28** | 10.88 (β=0.9) |
| 0.489 | **9.44** | 12.59 (β=0.5) |
| 0.275 | **12.93** | 14.94 (β=0.3) |
| 0.138 | **18.10** | 20.54 (β=0.1) |

**Per-token routing is strictly Pareto-superior to any fixed β** — same memory
budget buys less PPL, or same PPL at lower memory.

### Learned controller — the negative

| Features | AUC (predict "memory wins") |
|---|---|
| mixer h_last ⊕ gvec | 0.601 |
| KN-ctx cheap (c_h, n1, top prob) | 0.596 |
| both | 0.615 |
| oracle margin | 1.000 |

The controller collapses to β≈1 for all λ (avg β 0.97-1.00): the likelihood
benefit of high β on ~76% of tokens swamps the λ penalty, and the routing
signal from context is too weak to separate the 24% where compute wins.

## Interpretation (honest)

1. **The opportunity is real and large**: per-token routing beats fixed β by
   24% PPL at lower average β (oracle). The β-Architecture's memory axis CAN
   be made selective.

2. **The routing decision depends on the SPECIFIC true token y**, not just
   the context: "does memory beat compute here" is only observable after y.
   Context-only features (mixer state, cheap KN stats) reach only
   AUC ≈ 0.6 — 10 points above random. A context-only controller cannot
   realize the oracle.

3. **Consequence**: the λ-regularized controller collapses to β≈1. To be
   selective, the controller needs the memory probe (or y), which changes the
   architecture claim.

## Defensible formulation

- **Positive (new)**: the β-Architecture's memory channel admits per-token
  routing with a strictly better PPL/memory Pareto frontier than any fixed β.
  Selective memory is possible in principle.
- **Negative (honest)**: routing cannot be reliably predicted from the compute
  state or cheap n-gram statistics alone (AUC 0.6). A pure context-controller
  does not beat fixed β.

## Next

1. Mixer-confidence features (top-1 prob, entropy of the readout) — free,
   may push AUC toward 0.7.
2. Two-stage: cheap hash probe (O(1) existence/count) → conditional full KN
   distribution + fusion. Tunable threshold instead of a learned controller.
3. Or accept fixed-β as the operating point (already PPL 10.88, generates
   code) and treat learned sparsity as future work.

## Files

- `exp28_beta_controller.py` — the experiment (fixed curve, oracle, λ sweep).
- `exp28_beta_controller/features.npz` — extracted features (80K train / 12K eval).
- `exp28_beta_controller/results.json` — all numbers.
- `diag_route.py`, `diag_route2.py` — feature-informativeness diagnostics.
