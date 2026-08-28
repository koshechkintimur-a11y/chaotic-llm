# Experiment 33 — Cheap Backoff for Sparse MLE

## Question

exp32 showed sparse MLE on NL retains 87% of KN's gain at 9× cheaper, but
the absolute gap is 5 PPL (28.28 vs 23.21). Can a cheap hierarchical backoff
(order-3 → order-2 → order-1) close this gap without blowing up the cost?

## Result

| Method | PPL (test) | Cost/token | vs KN |
|---|---|---|---|
| mixer-only | 62.69 | 0 µs | — |
| mixer+KN | 23.21 | 141 µs | 1× |
| **mixer+sparse MLE** | **28.28** | **16.3 µs** | **9× cheaper** |
| mixer+sparse+backoff | 26.02 | 120 µs | 1.2× cheaper |

## Interpretation

**Backoff is not worth it.** The hierarchical backoff (blending order-3
sparse with order-2 sparse via λ = α/(α+c_h)) closes only 2.26 of the
5 PPL gap (28.28 → 26.02) while raising cost 7× (16 → 120 µs/token).
The cost explodes because the implementation creates a full V-dim array
at each order level, eliminating the sparsity advantage.

The optimal point is **sparse MLE without backoff**: 87% of KN's gain
at 9× cheaper. The remaining gap to KN is the price of cheap memory.

## "А что если" — joint training

The gap exists because the mixer was trained to work with KN (full backoff),
not with sparse MLE. If the mixer is trained jointly with the sparse-MLE
memory, it can learn to compensate for the missing backoff — the mixer
covers the unseen continuations that sparse MLE misses.

This is the next step: retrain the NL mixer with sparse-MLE memory in the
loop, and measure whether the gap closes.