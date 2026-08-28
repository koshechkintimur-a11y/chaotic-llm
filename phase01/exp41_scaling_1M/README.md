# Experiment 41 — Scaling: 1.2M Params (d=512) — HONEST PLATEAU SIGNAL

## Setup

d=512, 16 local + 8 global blocks, 1,181,720 params. Trained on RTX 3060
(25 min, grad clipping 1.0 — added per colleague review). Same corpus
(2M chars, 990K tokens), same eval protocol (12K test positions).

## Result

| Mixer size | params | train loss (final) | mixer-only PPL | +sparse β(c_h) |
|---|---|---|---|---|
| d=64 | 90K | 3.96 | 32.78 | **8.59** |
| d=256 | 460K | 2.52 | 16.73 | 8.70 |
| **d=512** | **1.2M** | **4.65** ⚠️ | **87.72** ⚠️ | **8.95** |

## The honest finding: d=512 FAILED TO TRAIN properly

- Train loss 4.65 at step 8000 — far above d=256's 2.52. The model did
  NOT converge to a better solution.
- mixer-only PPL 87.7 — **WORSE than the 90K model (32.8)**.
- The fused number (8.95 vs 8.59) hides this: the memory channel bails
  the model out. Do NOT read 8.95 as "scaling works".

## Diagnosis (what the loss trajectory says)

Loss trajectory: 6.32 → 5.00 (2K) → 4.94 (4K) → 4.65 (6K) → 4.65 (8K).
The model is stuck in a plateau from step 2000 — it is NOT slowly
converging, it is STUCK. Three hypotheses, in order of plausibility:

1. **Optimizer/schedule mismatch** (most likely): d=512 with the same
   lr=1e-3, batch 64, 8000 steps as d=64. Wider models need different
   LR/warmup/longer training. The 460K model trained 2× longer per step
   already; 1.2M may need 4× steps.
2. **Gradient flow through permutations**: 16+8 blocks of FIXED
   permutations — signal may attenuate through 24 permutation+coupling
   stages at this width. (Colleague's hypothesis — the gradient norm
   monitoring is not yet instrumented, so this is unverified.)
3. **Overfitting**: 1.2M params on 990K tokens = 1.2 params/token —
   borderline, but train loss is HIGH (not low), so this is NOT
   overfitting. Rule out.

## Interpretation for the scaling curve

The scaling curve is currently:
- 90K → 460K: PPL halves (32.8 → 16.7) — REAL scaling.
- 460K → 1.2M: **UNMEASURED** — the run failed to converge, so it tells
  us nothing about the architecture's ceiling.

The honest conclusion: **the scaling question (460K → 1M → 5M → 10M) is
STILL OPEN.** This experiment was a training-infrastructure failure, not
an architecture verdict. The next run needs: lower LR (5e-4) + warmup +
more steps (16K) + gradient-norm logging to discriminate hypotheses 1 vs 2.

## Positive side-note

Even the badly-trained d=512 mixer, once fused with sparse β(c_h) memory,
recovers to 8.95 PPL — only 0.36 worse than the d=256 fused (8.59). The
memory channel is doing enormous work. This is both a strength of the
architecture (robustness to a weak compute channel) and a warning (the
compute channel's contribution may be masked by the memory channel at
V=512 — the right place to measure scaling is V=2048 or NL).

## Next

1. Rerun d=512 with lr=5e-4, warmup 1000, 16K steps, grad-norm logging.
2. If it converges (train loss < 2.2): real scaling point.
3. 5M run on the SAME protocol, with expanded corpus (3-5M tokens) —
   0.8-1.2 params/token is the memory-regime limit; colleague is right.