# Experiment 35 — Confidence-Gated β: cheap memory BEATS full KN

## Question

exp32: sparse MLE on NL = 28.28 PPL vs KN 23.21 (9× cheaper, 87% of gain).
The gap comes from fixed β=0.9 trusting the memory equally everywhere —
but memory is reliable only on frequent contexts (high c_h).

**"А что если"**: gate β by the context's own evidence, for free:
β(c_h) = β_max · c_h / (c_h + k)  — one parameter, c_h comes from the
same memory lookup. Rare contexts → trust the mixer; frequent contexts →
trust the memory.

## Result (NL, WikiText-2)

| Method | PPL (test) | Memory cost |
|---|---|---|
| mixer-only | 62.69 | 0 |
| sparse MLE, fixed β=0.9 | 28.28 | 16.3 µs/token |
| **sparse MLE + β(c_h)** | **18.85** | 16.3 µs/token |
| **full KN, β=0.9 (target)** | **23.21** | 141 µs/token |

**Confidence-gated sparse MLE BEATS full KN by 4.4 PPL — at 9× cheaper
memory.** Δ vs fixed-β: −9.44 PPL from a single parameter.

## Why it works

- Fixed β=0.9 forces memory trust even on rare contexts, where the sparse
  MLE is noise (few observations) — it actively hurts.
- β(c_h) automatically reduces memory weight on rare contexts; the
  standalone-trained mixer (a real 90K-param model) is more reliable there
  than an n-gram backoff.
- On frequent contexts memory dominates (β→β_max) — where it is strong.

This is the same *principle* as KN's discounting, but applied to the
fusion weight β instead of the counts — and it needs no V-dim backoff.

## Architecture v0.7

```
chaotic compute (standalone-trained mixer)  — always
sparse MLE memory                           — always, 9-17× cheaper than KN
β(c_h) = c_h/(c_h + k)                      — confidence gate, 1 param, free
```

NL PPL 18.85 — **better than full KN (23.21)** at a fraction of the cost.

## Note

exp34 (joint training with fused loss) failed — mixer-only PPL 4888 vs
62.7: the mixer becomes lazy and stops learning unseen tokens. The mixer
must be trained standalone; the memory is a post-hoc cheap enhancement.
That failure is what made β(c_h) necessary — and it worked.