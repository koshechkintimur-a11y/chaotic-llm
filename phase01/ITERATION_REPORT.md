# Iteration Report: β-Architecture Phase 5 (exp25–exp30)

## Core direction

**β-Architecture (Compute-Memory Split)**: chaotic mixer (compute, O(W log W),
covering) + n-gram table (memory, exact, scaling with corpus) + β-gate.
The architecture works as a generative model, beats fixed-β attention,
and scales with memory. The open question was: can we make memory *selective*
(conditional execution) to save its ~230× higher cost?

## Positive results (main track)

### 1. Vectorized mixer — wall-clock closed (exp25)
Chaotic mixer is now **2.9× faster than attention at W=4096** (7.4 vs 21.1 ms).
40-60× speedup over the per-window loop. O(W log W) is real in wall-clock.

### 2. Hash table memory — faithful at scale (exp26)
Probing hash table (KenLM-style) is **exact: 0 disagreements with dict**.
BPE-2048 + hash-KN: PPL 20.18, top-1 40.7% at 8.9 MB. The memory channel
scales to larger vocabularies without loss.

### 3. Text generation — architecture generates code (exp27)
β=0.9 gives coherent code (imports, JSX, API handlers, type definitions).
β=0 (mixer alone) degenerates to repeated tokens. **First generative demo
of the β-Architecture as a text-producing model, not just a PPL scorer.**

### 4. Per-token routing Pareto-dominates fixed β (exp28, oracle)
Oracle per-token binary routing: **PPL 8.28 at avg β 0.758** vs fixed β=0.9
at PPL 10.88 — 24% better PPL at lower memory. The theoretical ceiling of
selective memory is real and large.

### 5. No error accumulation over K=16 (exp30)
Memory utility is strictly local: skipping memory at token t has **zero
effect on t+1 … t+16**. If a safe skip existed, it would be safe on long
horizons.

### 6. Gate overhead is negligible (exp30)
G3 gate (4353 params, MLP on mixer + cheap features): **1.19 µs = 0.9%
of KN cost** (132.6 µs). Conditional execution infrastructure is cheap.

## Negative results (successfully cleared from the track)

### 1. Learned β-controller (exp28)
*Hypothesis*: MLP on mixer features can learn per-token routing.
*Result*: collapses to β≈1 (AUC 0.6). Routing depends on y (true token),
unknown at test time. Context-only features cannot predict it.
*Verdict*: **BLOCKED — context-only routing is insufficient.**

### 2. Context-only gate (exp29–30)
*Hypothesis*: cheap features (hit, c_h, mixer confidence, KN stats) can
predict memory utility before the lookup.
*Result*: no gate exceeds 5% skip at ΔPPL ≤ 0.25. Learned gates (G2/G3)
not better than membership rule.
*Cause*: ΔL distribution is bimodal — only 0.5% of tokens have ΔL ≈ 0.
Memory is never "almost useless".
*Verdict*: **BLOCKED — no safe-skip population exists on this corpus.**

### 3. Utility-gated conditional execution (exp30)
*Hypothesis*: physical skip + utility predictor = real speedup.
*Result*: measured speedup 0.96× at 5% skip — **gating is slower than
always-on**. Skip rate too small to amortize the gate.
*Verdict*: **BLOCKED — conditional execution architecture is ready, but
the skip signal (ΔL≈0 tokens) does not exist.**

## Synthesis — "а что если"

The three blocked tracks share a root cause: **the memory is too expensive
(132 µs/token, V-dim) for a gate to decide whether to use it. The gate
itself costs 1.19 µs, and at 5% skip it saves 6.6 µs — net loss.**

**"А что если" — make the memory itself cheap, so no gate is needed.**

## exp31 — Cheap Associative Memory: THE PIVOT ✅

| Memory | Cost/token | PPL | % of KN gain |
|---|---|---|---|
| none (mixer only) | 0 µs | 32.78 | 0% |
| exact match top-1 | 0.54 µs | 25.59 | 33% |
| **sparse MLE (β=0.9)** | **7.7 µs** | **11.66** | **93%** |
| KN full | 130.8 µs | 10.94 | 100% |

**Sparse MLE — distribution over only the OBSERVED continuations of the
context (~5 hash lookups, no backoff, no V-dim scan) — retains 93% of KN's
quality at 17× lower cost.** Cheap enough to run on every token
unconditionally: **the gate dilemma disappears.**

## Architecture v0.6

```
chaotic compute   (0.56 µs/token)  — always, O(W log W)
sparse MLE memory (7.7 µs/token)   — always, 17× cheaper than KN
β-gate (β=0.9)                     — no conditional execution needed
```

Total ≈ 8.3 µs/token vs 131 µs/token (KN) — **15× cheaper inference with
nearly identical quality**. The memory channel is unconditionally
affordable: the architecture is viable without gates, without utility
predictors, without conditional execution.

## Iteration verdict

- **exp25–27 (positive, kept)**: vectorized mixer 2.9× faster than attention
  at W=4096; hash memory faithful at scale; text generation works.
- **exp28–30 (blocked, cleared)**: learned β-controller (AUC 0.6), context
  gate (max safe skip 5%), utility-gated execution (0.96× — net loss).
  Root cause: memory too expensive for a gate. **Successfully passed —
  the failure was in the gate premise, not the memory.**
- **exp31 (positive, new track)**: cheap sparse-MLE memory — 93% of quality
  at 17× cost, no gate needed.

**The solution was there all along**: don't gate an expensive memory —
make the memory cheap. Next: NL validation, cheap backoff, generation
demo, full architecture wall-clock.