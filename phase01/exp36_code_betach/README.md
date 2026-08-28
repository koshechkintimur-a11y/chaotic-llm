# Experiment 36 — β(c_h) on Code + Architecture Wall-Clock

## Result (code corpus)

| Method | PPL (test) | Memory cost |
|---|---|---|
| mixer-only | 32.78 | 0 |
| sparse MLE, fixed β=0.9 | 11.66 | ~8 µs/token |
| **sparse MLE + β(c_h)** | **8.59** | ~8 µs/token |
| full KN (target) | 10.94 | 131 µs/token |

**β(c_h) = c_h/(c_h + 0.5), βmax=1.0: PPL 8.59 — beats full KN (10.94) AND
the project's previous best (exp22 raw-MLE order-3: 10.35).**

## Both domains confirmed

| Domain | +KN | +sparse fixed β | +sparse β(c_h) |
|---|---|---|---|
| Code | 10.94 | 11.66 | **8.59** |
| NL (WikiText-2) | 23.21 | 28.28 | **18.85** |

The single confidence parameter c_h/(c_h+k) — free, taken from the same
memory lookup — turns cheap sparse memory from "slightly worse than KN"
into "strictly better than KN" on both domains.

## Why β(c_h) beats KN

- **Frequent contexts** (high c_h): β→β_max=1.0 — memory dominates, and
  n-grams are reliable there (deterministic code patterns, common phrases).
- **Rare contexts**: β→0 — the standalone-trained mixer (90K params, real
  representations) takes over. KN instead backoffs to lower-order n-grams,
  which are also data-poor on BPE-512; the mixer generalizes better.
- KN spreads its confidence smoothly across orders; β(c_h) does a sharper
  context-evidence-dependent trust switch, for free.

## Wall-clock (this CPU bench)

mixer + sparse MLE + β(c_h): **289 µs/token measured** — dominated by Python
loop overhead (per-token dict lookups + tuple slicing), not by the
architecture. Memory channel itself: ~8-16 µs/token vs KN 131-141 µs
(9-17× cheaper). A batched hash-table / GPU implementation is the
engineering next step.

## Architecture v0.7 — final form

```
chaotic compute (standalone mixer, 90K params)  — always, O(W log W)
sparse MLE memory (observed continuations)      — always, 9-17× cheaper than KN
β(c_h) = c_h/(c_h + k)                          — confidence gate, 1 param, free
```

Code PPL **8.59** (KN 10.94) · NL PPL **18.85** (KN 23.21).
Cheap, ungated, unconditional memory that beats full Kneser-Ney.