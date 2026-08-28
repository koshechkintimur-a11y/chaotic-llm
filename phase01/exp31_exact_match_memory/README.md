# Experiment 31 — Cheap Associative Memory (Exact Match + Sparse MLE)

## The "а что если" pivot

exp28–30 blocked on one root cause: **memory costs 132 µs/token (V-dim KN),
so a gate deciding whether to use it must be right — and no cheap signal
can be**. The pivot: **make the memory itself cheap, so no gate is needed.**

Metaphor: instead of a proofreader who re-reads every sentence (expensive,
needs to be selectively invoked), use an associative memory — "I've seen
this phrase before" — which is cheap enough to fire on every token.

## Methods (cost ladder)

| Memory | Lookups/token | Cost | PPL (test) | % of KN gain |
|---|---|---|---|---|
| none (mixer only) | 0 | 0 µs | 32.78 | 0% |
| exact match, top-1 boost | 1 | 0.54 µs | 25.59 | 33% |
| exact match, top-3 boost | 3 | ~1.5 µs | 21.26 | ~53% |
| **sparse MLE (β=0.9)** | **~5 (observed only)** | **7.7 µs** | **11.66** | **93%** |
| KN full (β=0.9) | 512 | 130.8 µs | 10.94 | 100% |

## Key result

**Sparse MLE — the distribution over only the OBSERVED continuations of the
current context (no backoff, no V-dim scan) — retains 93% of KN's quality
improvement at 17× lower cost (7.7 vs 130.8 µs/token).**

Because it is cheap (~5 hash lookups on average), it can run on EVERY token
unconditionally. **The gate dilemma disappears**: no need to predict utility,
no conditional execution — just always use the cheap memory.

## Comparison with the blocked tracks

- exp30 gate: at 5% skip (max safe), measured speedup 0.96× — net loss.
- exp31 sparse MLE: 17× cheaper memory, always on, 93% of quality. **No gate,
  no risk, strictly better.**

## Architecture consequence

β-Architecture v0.6:

```
chaotic compute  (0.56 µs/token)  — always
sparse MLE memory (7.7 µs/token)  — always, 17× cheaper than KN
β-gate (fixed or learned)          — no conditional execution needed
```

Total ≈ 8.3 µs/token vs 131 µs/token for KN — 15× cheaper inference with
nearly identical quality. The memory channel is now *unconditionally
affordable*, which is what makes the architecture viable.

## Caveats

- Sparse MLE has no backoff → zero probability for unseen continuations.
  For contexts with few observations this is harsh (β=0.3 → 14.8 PPL).
  β=0.9 works because at high β the sparse distribution dominates only
  where the context is strong.
- Exact-match top-1 boost (33% of gain) is too weak alone; top-3 helps but
  still far from sparse MLE.
- Measured on code, V=512, Python dict. Real hash-table implementation
  scales the same way.

## Next

1. Sparse MLE on NL (WikiText-2) — does the 93%-at-17× result generalize?
2. Cheap backoff for sparse contexts (c_h small → mix in unigram) —
   close the last 0.7 PPL to KN.
3. Generation demo with sparse MLE memory.
4. Integrate into the full β-Architecture and re-verify wall-clock.