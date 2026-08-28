# Experiment 50 — Dual Gears: NOT SUPPORTED

## Result

| Model (1.2M, same protocol) | mixer-only PPL | vs criterion |
|---|---|---|
| exp41b scalar | 97.4 | — |
| exp42 vector | 159.3 | — |
| **exp50 dual** | **133.5** | ≥121.3 → **FAIL** |

Train loss final ~4.4 (step-14K 4.485). +sparse fused: 9.04 (memory masks).

## Interpretation

Splitting the 24 blocks into two gear mixers (local 64 → interleave →
intermediate 256) with FIXED interleave permutations does NOT fix the d=512
scaling ceiling. The interleave is non-learnable — it adds mixing capacity
but not learnable capacity, and the model still can't use its width.

Dual (133.5) beats vector (159.3) but is worse than scalar (97.4).

## Decision (per methodology — stop on foreknown negatives)

- **exp51 (triple)** — SAME mechanism (more fixed permutations, no learnable
  params): skipped as foreknown negative.
- **exp52 (bidirectional)** — DIFFERENT mechanism (learnable Linear 2d→d
  projection, ~262K params): run next. The projection gives the model a
  learnable way to combine directions — qualitatively distinct.