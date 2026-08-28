# EXP30 — Utility-Gated Memory: Report

## 1. Hypothesis

> Can a cheap signal predict, BEFORE the expensive memory lookup, whether that
> lookup will actually help the current token — so we can PHYSICALLY skip
> memory computation without losing quality?

Tested as **conditional execution**: mixer always runs (cheap, 0.56 µs/token);
KN memory (132 µs/token — 230× more expensive) runs only when the gate says
its utility is high. Ground-truth utility ΔL = L_mixer − L_mixer+KN is
computed offline and NEVER shown to the gate at decision time.

## 2. Experimental setup

- Mixer V1 (vocab 512, code, 90K params) + KN order-3 table, β=0.9 when fused.
- 60K train / 10K val / 12K test positions, windows [i, i+W) → predict seq[i+W].
- Baseline always-on PPL: **10.923** (test). Mixer-only PPL: 32.775.
- Files: `exp30_utility_gate.py` (ground truth), `exp30_part2_gates.py` (gates,
  curves, wall-clock, generalization, long-range), `results/exp30_*.csv/json`.

## 3. Ground-truth utility

ΔL = L_mixer − L_mixer+KN per token, β=0.9. Distribution (test, 12K tokens):

| ΔL range | share | meaning |
|---|---|---|
| < −0.5 | 15.5% | memory strongly hurt |
| −0.5..−0.1 | 6.7% | memory hurt |
| −0.1..−0.01 | 2.1% | memory slightly hurt |
| **≈0 (±0.01)** | **0.5%** | **memory useless** |
| 0.01..0.1 | 2.8% | memory slightly helped |
| 0.1..0.5 | 11.7% | memory helped |
| > 0.5 | 60.7% | memory strongly helped |

**The decisive fact: only 0.5% of tokens have ΔL ≈ 0.** Memory is *never*
"almost useless": it either strongly helps (60.7% with ΔL > 0.5) or strongly
hurts (15.5% with ΔL < −0.5). The distribution is bimodal with an empty
middle. There is no population of "memory-indifferent" tokens to skip safely.

## 4. Gate designs

| Gate | Cost | Description |
|---|---|---|
| G0 membership | ~0 | context ∈ table? (exp29 baseline) |
| G1 top_prob | ~0 | KN confidence (max cont / c_h) |
| G1b −mixer_top1 | ~0 | mixer uncertainty |
| G2 LR (7 params) | 0.9% of KN | logistic regression on 6 cheap feats |
| G3 MLP (4353 params) | 1.19 µs/token | MLP on mixer state ⊕ cheap feats |

All gates see only pre-lookup information (hit, c_h, n1, top_prob, mixer
top-1/entropy, mixer hidden state). None sees ΔL.

## 5. Skip-vs-quality curve (test)

Quality-budget table — max skip at a given PPL budget:

| Budget | G0 | G1b | G2 | G3 |
|---|---|---|---|---|
| ΔPPL ≤ 0.01 | — | — | — | — |
| ΔPPL ≤ 0.05 | — | — | — | — |
| ΔPPL ≤ 0.10 | 1.4% | — | — | 5.0% |
| ΔPPL ≤ 0.25 | 1.4% | 5.0% | 5.0% | 5.0% |

**No gate reaches even 5% skip at ΔPPL ≤ 0.05.** Learned gates (G2/G3) barely
beat the 0-cost membership rule. The cheap signals cannot identify safe tokens
because there are (almost) none.

## 6. Skip-vs-latency curve

| Skip | ΔPPL (G3) | measured speedup |
|---|---|---|
| 5% | +0.09 | **0.96× (slower)** |

## 7. Gate overhead (measured)

| Component | µs/token |
|---|---|
| Mixer (batched) | 0.56 |
| G3 gate (feats + MLP) | 1.19 |
| KN full dist (no memo) | 132.6 |
| gate/KN ratio | **0.9%** |

The gate itself is nearly free (0.9% of KN cost). The problem is not gate
cost — it's that at 5% skip the saved KN time (≈6.6 µs/token) is smaller than
the gate + feature overhead.

## 8. Real wall-clock (5 warm-up + 5 timed runs, same hardware)

| Mode | mean | median | p95 |
|---|---|---|---|
| Baseline (always-on) | 3496 ms | 3446 ms | 3727 ms |
| Gated (G3, 5% skip) | 3649 ms | 3709 ms | 3853 ms |
| **Speedup** | **0.96×** | | |

Gating makes inference *slower* at 5% skip. Conditional execution only pays
off when the skip rate is large enough to amortize the gate — which quality
never allows on this corpus.

## 9. Generalization

| Context frequency | skip rate (G3) |
|---|---|
| c_h = 0 (not in table) | 100% |
| c_h 1–10 | 0.7% |
| c_h 10–100 | 1.3% |
| c_h 100+ | 5.0% |

The gate skips *more* on high-frequency contexts (where the mixer alone is
already strong) — a sensible direction, but the effect is too weak to matter.
Train/val/test were strictly separated; thresholds came from the curve sweep,
not from test tuning.

## 10. Long-range effect (K = 1, 4, 8, 16)

| K | always-on PPL | gated PPL | ΔPPL |
|---|---|---|---|
| 1 | 10.940 | 11.086 | +0.146 |
| 4 | 10.936 | 10.936 | 0.000 |
| 8 | 10.927 | 10.927 | 0.000 |
| 16 | 10.832 | 10.832 | 0.000 |

**No error accumulation.** Skipping memory at token t has zero effect on
t+1…t+16 (teacher-forcing: the mixer state and KN context at later tokens do
not depend on the skip decision). Memory utility is strictly local — which
means if a safe skip existed, it would be safe for arbitrarily long
continuations. The bottleneck is identifying the safe tokens, not error
propagation.

## 11. Comparison with exp29

| Method | Skip | ΔPPL |
|---|---|---|
| exp29 rule c_h==0 | 1.9% | +0.05 |
| exp29 controller λ=0.5 τ=0.5 | 14.1% | +0.80 |
| exp29 controller λ=1.0 τ=0.5 | 50.8% | +4.90 |
| exp30 G3 (best learned gate) | 5.0% | +0.09 |

exp30 confirms exp29: **no gate safely exceeds ~5% skip**. The utility-aware
gates (G2/G3) are not a qualitative improvement over the membership probe —
both are capped by the same wall: tokens with ΔL≈0 do not exist (0.5%).

## 12. Failure cases

- **The core failure is in the data, not the gate**: ΔL distribution is
  bimodal with an empty middle. Conditional execution needs a mass of
  near-zero-utility tokens; this corpus has none.
- G1 (KN confidence) never reaches any budget — the KN top probability does
  not predict whether memory will help *the true token*.
- G2 (7-param LR) saturates at the same 5% as G0 — the 6 cheap features carry
  ~no routing signal beyond membership.

## 13. Limitations

- Corpus is code with a very strong n-gram signal (60.7% of tokens have
  ΔL > 0.5): memory is *more* useful than in typical natural language.
- V=512 dict-based KN in Python (132 µs) vs the theoretical C/hash version
  (~1–2 µs): the real wall-clock favors KN much more; the qualitative
  conclusion (no safe mass to skip) is scale-independent.
- Teacher-forcing long-range test: no accumulation *by construction*; a
  free-running generation test would be needed to fully close §12 of the spec.

## 14. Conclusion

**Verdict: NOT SUPPORTED** (for significant safe skipping on this corpus).

- Conditional execution is *technically* real and cheap: the gate costs 0.9%
  of the KN lookup, and physical skipping works.
- But there is **no population of memory-indifferent tokens**: only 0.5% of
  tokens have ΔL ≈ 0; memory either strongly helps (60.7%) or strongly hurts
  (15.5%). A gate cannot skip what does not exist.
- Max safe skip: 1.4% (ΔPPL ≤ 0.10, membership) … 5% (ΔPPL ≤ 0.25, learned).
- Measured real speedup at 5% skip: **0.96× — gating is slower**.
- Positive: no error accumulation over K=16 (memory utility is local), and
  the gate overhead is negligible — the architecture is *ready* for
  conditional memory the moment a corpus offers a safe skip population.

## Summary numbers

```
Maximum safe skip rate:  5.0% (ΔPPL ≤ 0.25, G3); 1.4% (ΔPPL ≤ 0.10, G0)
Maximum measured real speedup: 0.96× (no speedup — gated is slower at 5% skip)
Quality degradation:      +0.09 PPL at 5% skip; +0.80 at 14%; +4.9 at 51%
Best gate:                G3 MLP (4353 params, 1.19 µs/token = 0.9% of KN)
Gate overhead:            0.9% of KN cost (negligible)
Key blocker:              only 0.5% of tokens have ΔL ≈ 0 (bimodal utility)
```

## Answer to the main question

> Can we teach the system to know in advance that an expensive computation
> won't be needed, and physically not execute it?

**On this corpus: no.** The utility signal is bimodal — memory is almost
always either a big help or a big harm, and the cheap pre-lookup features
cannot tell which (AUC ≈ 0.6). Conditional execution is implemented, cheap
(0.9% overhead), and has no error accumulation — but there is no safe mass to
skip. The idea remains architecturally sound; it needs a domain where a large
fraction of tokens genuinely do not need the memory (e.g., larger vocabularies
with sparser n-gram coverage, or mixed-domain streams).
