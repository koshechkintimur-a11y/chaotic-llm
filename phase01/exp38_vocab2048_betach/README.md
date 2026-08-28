# Experiment 38 — β(c_h) + Sparse MLE at BPE-2048

## Question

exp35/36 showed β(c_h) sparse MLE BEATS full KN at V=512 (code 8.59 vs
10.94; NL 18.85 vs 23.21). Does the advantage hold at 4× vocabulary?

## Result (code, BPE-2048)

| Method | PPL | per-token |
|---|---|---|
| mixer-only | 75.05 | 0.0366 |
| sparse fixed β=0.9 | 28.32 | 0.0138 |
| **sparse β(c_h) k=5** | **22.12** | 0.0108 |
| full KN (exp26) | 20.18 | 0.0099 |

## Interpretation (honest)

**The advantage flips at larger vocab.** At V=512 β(c_h) beats KN; at
V=2048 KN is ~2 PPL better. Optimal k scales with vocab (0.5 at V=512 →
5.0 at V=2048) — larger vocab makes MLE counts noisier, needing more
mixer trust.

Why: larger vocabulary → sparser n-gram observations → KN's backoff
provides more value. The cheap memory's blind spots (unseen
continuations, no backoff) grow with vocab.

Still, β(c_h) sparse MLE at V=2048:
- **4× better than mixer-only** (22.1 vs 75.0)
- **6.2 PPL better than fixed-β sparse** (22.1 vs 28.3)
- **~1.8 PPL from full KN at 9-17× cheaper memory** (16 vs ~280 µs/token)

## Verdict

**Viable but with a quality-cost trade-off at scale.** At V=512 the cheap
memory dominates KN outright; at V=2048 it trades ~2 PPL for 9-17× cheaper
memory. A hybrid (sparse MLE for most contexts + KN backoff only where
sparse MLE is weak, i.e. rare contexts) or a unigram floor for unseen
continuations could close the residual gap — next candidates.