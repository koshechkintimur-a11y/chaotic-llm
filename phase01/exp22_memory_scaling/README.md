# Experiment 22 — Memory-Scaling (Phase 5: β-Architecture test)

## Hypothesis

The β-Architecture (Compute-Memory Split): accuracy scales with the MEMORY
channel (corpus table) at FIXED (tiny) compute channel. The corpus is a
scaling axis independent of parameters.

## Setup

- Model: exp18 V1 (tiny chaotic mixer + local readout, 6000 steps, FIXED).
  Mixer params: 49,292. Total: 90,828.
  Mixer alone: PPL 32.83.
- Memory channel: β-table (order-1/2/3, capped variants), β=0.3 fixed.
- Corpus: code, BPE-512, 990K train tokens.
- Eval: 20,000 test positions.

## Results

| Table | Contexts | Memory | PPL+β | top-1+β | PPL_model (fixed) |
|---|---|---|---|---|---|
| order-1 | 507 | 0.23 MB | 28.34 | 23.2% | 32.83 |
| order-2 cap 3K | 3,000 | 0.45 MB | 20.14 | 32.4% | 32.83 |
| order-2 cap 10K | 10,000 | 0.86 MB | 15.39 | 38.4% | 32.83 |
| order-2 full | 28,176 | 1.27 MB | 13.79 | 40.7% | 32.83 |
| **order-3 full** | **130,856** | **3.54 MB** | **10.35** | **52.4%** | 32.83 |

Table alone: PPL = inf (unsmoothed — zero probability on unseen contexts).

## Interpretation

1. **Memory-scaling CONFIRMED**: PPL+β improves monotonically with table size
   (28.3 → 10.35, 2.7×) at fixed compute. top-1+β: 23.2% → 52.4%. The corpus
   is a scaling axis INDEPENDENT of parameters.
2. **Best result of the whole project**: PPL 10.35 / top-1 52.4% (mixer 90K
   params + 3.54 MB table) beats the transformer (11.9 / 42.4%) — at a fraction
   of the transformer's compute.
3. **Channel complementarity**: table alone = inf (no coverage), mixer alone =
   32.83 (no precision), together = 10.35. **Compute = coverage/backoff,
   memory = precision.** The channels are complementary, not redundant —
   exactly the two-channel β-Architecture design.

## Conclusion

**Подтверждено (Prediction #1 of PHASE5_HYPOTHESIS).** The memory-scaling axis
exists: a nearly-parameter-free compute channel + a growing corpus table
scales accuracy — a property the transformer does not have (its memory lives
in parameters only). This is the empirical foundation of the β-Architecture.

## Caveats

- BPE-512 on code: n-gram coverage is strong here. Natural language (sparser
  n-grams) may shift β (exp23).
- Table alone is inf — the mixer is REQUIRED for smoothing; channels are
  complementary, not independent.
- Real LLM scale would need smoothed/backoff n-gram models (Kneser-Ney) and
  a hash-based table for vocab 50K+.

## Next

exp23 — β-law on natural language (does β drop with sparser n-grams?).
exp24 — fused/compiled implementation for wall-clock.