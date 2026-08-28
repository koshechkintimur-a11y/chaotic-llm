# Experiment 41b — d=512 Rerun with Diagnostics: ARCHITECTURAL CEILING

## Setup

d=512, 16+8 blocks, 1.18M params. lr=5e-4 + warmup 1000 + cosine 16K steps,
grad clipping 1.0. Full diagnostics every 500 steps: grad norms (total,
embed, readout), reversibility error. GPU RTX 3060, 4033s (~67 min).

## Result

| Metric | 90K (d=64) | 460K (d=256) | 1.2M (d=512) |
|---|---|---|---|
| train loss (final) | 3.96 | 2.52 | **4.01** |
| mixer-only PPL | 32.78 | 16.73 | **97.4** |
| +sparse β(c_h) k=0.5 | 8.59 | 8.70 | 9.00 |

## Hypotheses verdict

| Hyp | Result |
|---|---|
| H1 LR/schedule | Cosine+warmup improved train loss 4.65→4.01, but model still can't even overfit (train loss > 460K's). Schedule helps, isn't the ceiling. |
| H2 gradient attenuation | **REJECTED** — grad norms healthy whole run (embed ~0.12, readout ~2.6), no attenuation through 24 permutations. |
| **H3 scalar gates don't scale with width** | **PRIMARY** — each gate is a scalar `nn.Parameter(torch.zeros(bl))` shared across all d dims. `even + g*odd` mixes all 512 dims identically. Width grew 8× (64→512), mixing capacity grew 0× (24 scalars). Model stuck in a poor minimum. |

## Evidence for H3

- d=64→256 (scalar gates): real scaling, mixer PPL 32.8→16.7.
- d=256→512 (still scalar gates): train loss JUMPS UP (2.52→4.01). The model
  cannot even fit the training data — the mixing can't exploit the added
  width.
- This is NOT data-limited (1.2M could overfit 990K tokens if it could use
  its capacity) and NOT optimizer-limited (both schedule variants plateau).

## Answer to the scaling question

**460K is the last point where the CURRENT architecture scales.** The
ceiling comes from scalar gates: mixing capacity is independent of width.
For 5M, the fix is **vector gates** (d-dimensional gate per block) so mixing
capacity grows with width. Concrete, testable next step.

## What did NOT help

- Reversibility check was broken (wrong analytic inverse of the symmetric
  coupling) — logged a constant ~70, discarded. Not an input to the verdict.
- Note: the +sparse fused PPL (9.00) masks the compute failure — memory
  bails out a broken mixer. The CONTROL (β=0, mixer-only 97.4) is the
  honest number.

## Next

exp42: vector gates (d-dim per block) at d=512, same protocol. If mixer-only
breaks below 16.7 → H3 confirmed and the architecture scales again.
