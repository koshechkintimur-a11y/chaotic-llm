# Experiment 42 — Vector Gates: H3 REJECTED (stopped at step 14K)

## Question

exp41b diagnosed scalar gates (one coefficient shared across all d dims)
as the scaling ceiling. Test: d-dimensional gates per block — does d=512
then scale past the 460K point?

## Result (stopped at step 14K of 16K — trend already decisive)

| Step | Scalar gates loss (exp41b) | Vector gates loss (exp42) |
|---|---|---|
| 8000 | 4.86 | 4.86 |
| 10000 | 4.59 | 4.89 |
| 12000 | 4.41 | 4.84 |
| 14000 | 4.15 | **4.54** |

Checkpoint eval @ step 8000 (mixer-only control, β=0):
- scalar: PPL 121.3
- vector: PPL 159.3  (worse)

## Verdict

**H3 REJECTED — vector gates do not help.** The trajectory is worse than
scalar gates at every step after 8K. The scaling ceiling at d=512 is NOT
caused by the scalar nature of the mixing gates.

## What this means

The bottleneck is deeper than the gate representation. Candidates, in
order of plausibility:

1. **The mixing structure itself (24 permutation+coupling blocks) doesn't
   transfer capacity to width.** Both scalar and vector gates plateau near
   the same loss; the model can't use its 8× width regardless of how gates
   are represented.
2. **Data ratio (1.2M params / 990K tokens)**: the 460K model was the
   sweet spot for this corpus; 1.2M needs more data, not more mixing.
3. **The local window (Wl=64) + global pool architecture** caps the mixing
   that any block count can achieve at high width.

## Honest position

- 90K → 460K: scales (32.8 → 16.7 mixer PPL). Real.
- 460K → 1.2M: does NOT scale, and neither optimizer (exp41b) nor gate
  representation (exp42) fixes it.
- The fused numbers (~9.0 PPL) stay flat because the sparse-memory channel
  masks the compute channel at V=512. The mixer's contribution is invisible
  in fused PPL — the CONTROL (β=0) is the only honest mixer metric.

## Options (next)

1. **Scale data instead**: expand corpus 3-5×, retrain 1.2M — tests the
   data-ratio hypothesis (2).
2. **Deepen the hierarchy** (Wl=64 → 64→512→global, colleague's idea):
   tests whether the 2-level structure caps capacity (3).
3. **Accept 460K as the compute sweet spot** and measure the architecture's
   value at V=2048 / NL where memory doesn't mask compute.

## Files

- `exp42_vector_gates.py` — vector-gate training script
- `ckpt_4000.pt`, `ckpt_8000.pt` — checkpoints (partial run)
- `compare_step8000.json` — scalar vs vector mixer-only/gated at step 8K
