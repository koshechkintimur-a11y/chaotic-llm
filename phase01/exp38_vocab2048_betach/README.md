# Experiment 38 — β(c_h) + Sparse MLE at BPE-2048 (CORRECTED)

## ⚠️ Corrected result

The original exp38 run reported PPL 22.12 at V=2048 and concluded KN was
better. **That was a bug** (missing guard: tokens where memory has no mass
were penalized by log(1-β)). The fixed evaluation (exp39):

| Method | PPL (test, V=2048) |
|---|---|
| mixer-only | 75.05 |
| sparse fixed β=0.9 | 28.32 |
| **sparse β(c_h) k=5.0** | **16.55** |
| full KN (exp26) | 20.18 |

**β(c_h) sparse MLE BEATS full KN at BPE-2048 (16.55 vs 20.18)** — same as
at V=512 (8.59 vs 10.94). The architecture v0.7 holds at scale.

## Interpretation

- Optimal k scales with vocab: 0.5 at V=512, 5.0 at V=2048 (larger vocab =
  sparser n-grams = trust the mixer more on average).
- Sparse MLE + β(c_h) is 4.5× better than mixer-only, 12 PPL better than
  fixed-β sparse, and ~3.6 PPL better than full KN — at 9-17× cheaper
  memory (16 vs ~280 µs/token at V=2048).

## Verdict

**Viable at scale.** Cheap unconditional memory + 1-parameter confidence
gate beats full Kneser-Ney at both vocabulary sizes tested.