# Experiment 45 — 1.2M Transformer CONTROL: verdict on the mixer

## The control that decides the debate

Same corpus (2M chars / 990K tokens), same protocol (lr 5e-4, warmup 1000,
cosine 16K, batch 64, clip 1.0), same eval (12K held-out positions) — but
a standard causal Transformer (d=128, 4 heads, 6 blocks, 1.35M params)
instead of the chaotic mixer (1.18M).

## Result

| Model (1.2M scale) | train loss (final) | mixer-only PPL | +sparse β(c_h) |
|---|---|---|---|
| chaotic mixer | 4.01 | 97.4 | 9.00 |
| **Transformer** | **1.82** | **8.65** | **7.97** |

Transformer reached 3.11 at step 2K (mixer: 5.03) and 2.17 at step 8K
(mixer: 4.86). It beat the mixer's FINAL loss (4.01) by step 6K.

## Verdict

**Hypothesis C CONFIRMED — the chaotic mixer has an architectural scaling
ceiling around 460K (d=256).**

- NOT data (the Transformer proves the corpus supports loss ~1.8).
- NOT optimization (identical protocol; the mixer's schedule variants
  exp41/41b both plateau).
- NOT gate representation (exp42: vector gates worse).
- The problem is the mixing structure itself at width d=512: 24 fixed
  permutation+coupling blocks with only gates as learnable mixing
  parameters cannot transfer capacity to width the way attention does.

## Secondary finding (important)

**Sparse MLE + β(c_h) improves the Transformer too: 8.65 → 7.97.** The
cheap memory channel is architecture-agnostic — it boosts any base model.
This validates the memory part of Architecture v0.7 independently of the
mixer.

## Honest position on the project

1. **The memory channel works and is generic**: sparse MLE + β(c_h) beats
   KN (10.94 → 8.59 with mixer; 8.65 → 7.97 with transformer) at 9-17×
   cheaper cost. This holds at V=512, V=2048, code and NL.
2. **The chaotic compute channel scales only to 460K** — beyond that,
   attention wins decisively. The O(W·d) complexity advantage remains
   real (vs O(W²) attention), but the QUALITY ceiling at width is now
   measured, not guessed.

## Next (if continuing)

- Test the mixer ceiling at intermediate widths (d=320, d=384) to find
  where the curve turns — is it smooth or a cliff?
- Or accept the measured ceiling and pivot the architecture claim to the
  memory channel (which is generic) + mixer at ≤460K for cheap contexts.