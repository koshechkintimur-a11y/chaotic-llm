# Phase 5 — Architecture Hypothesis: "β-Architecture" (Compute-Memory Split)

## Origin

Phase 4 (Candidate Recall) показал: хаос НЕ маршрутизатор (Recall@16=11.8%).
Но цепочка из 21 эксперимента выявила факт, который становится основой
новой гипотезы: **качество даёт канал памяти (β-таблица), а не канал
вычислений (миксер почти бесплатен).**

## The Honest Classification

| Component | Known analogue | Status |
|---|---|---|
| Structured linear / reversible mixing | SSM (Mamba/S4), MLP-Mixer, linear attention | Known |
| β-interpolation with n-gram prior | neural+n-gram LM interpolation (classic) | Known |
| Local readout | standard | Known |
| **The COMBINATION (compute-light + memory-heavy + learnable β-gate, attention-free)** | — | **Not published as a complete architecture** |

## The Measured Facts (why the combination is worth testing)

1. The chaotic mixer has ~12 learnable scalars per layer; the rest of the
   model (90K) is embedding + readout. **The compute channel is nearly
   parameter-free.**
2. Mixer + local readout + β-table: **PPL 11.24, top-1 43.0%** (β=0.972) —
   competes with a same-mass transformer (11.9 / 42.4%).
3. β=0.972: the model LEARNS to trust memory almost entirely.
4. β-table (1.8 MB) beats kNN-LM at equal memory (1.84×) and beats it even
   at 20× kNN memory.
5. Depth does not help (blocks have no weights) — the mixer does not need
   to grow to add value.

⇒ **The architecture inverts the transformer's assumption**: memory lives in
a compact corpus table (scales with CORPUS), not in parameters (scales with
MODEL). Compute is cheap and nearly fixed.

## The Hypothesis (falsifiable)

> **β-Architecture (Compute-Memory Split):** An LLM where computation is a
> cheap, nearly-parameter-free reversible structured mixer (generalization
> channel), combined multiplicatively in log-space with a compact
> corpus-conditional statistical prior (memory channel), the split controlled
> by a learned per-position β-gate.

$$
P(y|x) = (1-\beta_t)\cdot P_{\text{compute}}(y|z_t) + \beta_t\cdot P_{\text{memory}}(y|\text{ctx}_t)
$$

Architecture (attention-free):

```
Input (BPE)
  → Embedding + Position
  → Cheap Reversible Structured Mixer (L layers, ~12 params each)   [O(W log W)]
  → Local Readout (query position + global mean → MLP)              [O(d)]
  → β-gate (learned per-position, sigmoid)                          [O(1)]
      P = (1-β)·P_compute + β·P_memory(ctx)                         [P_memory = n-gram table]
  → Logits
```

## Falsifiable Predictions

1. **Memory-scaling (new axis):** accuracy grows with TABLE size at FIXED
   (tiny) mixer. Corpus size becomes a scaling axis independent of parameters.
2. **β-law:** β increases with corpus size, decreases with model size
   (trust memory more when compute is weak; trust compute more when it is strong).
3. **Parameter paradox:** a 12-param/layer mixer + table competes with a
   transformer of the same total mass.
4. If instead accuracy REQUIRES the mixer to grow (needs transformer-scale
   compute), the hypothesis reduces to "transformer + n-gram interpolation"
   (known) — and we say so.

## What would kill it (failure modes)

1. Mixer must grow with model → just a transformer variant.
2. β collapses to 0 on natural language (memory irrelevant) → table unnecessary.
3. Table does not scale (saturates fast) → no memory-scaling axis.
4. Wall-clock: mixer slower than attention on GPU (exp16) → need fused impl.

## Next experiment (exp22)

Test Memory-scaling: fix a tiny mixer (12 params/layer), vary the β-table
(order-1/order-2, table size 10K→1M entries), measure PPL/top-1.
If accuracy rises with table size while mixer stays tiny → the new axis
exists → the hypothesis is a real architecture direction.
Then exp23: natural-language test of the β-law.
Then exp24: fused implementation for wall-clock.

## exp22 RESULT — Memory-scaling CONFIRMED

Fixed tiny mixer (PPL 32.83 alone, 49K mixer params / 90K total). Vary ONLY
the β-table:

| Table | Contexts | Memory | PPL+β | top-1+β |
|---|---|---|---|---|
| order-1 | 507 | 0.23 MB | 28.34 | 23.2% |
| order-2 cap 3K | 3,000 | 0.45 MB | 20.14 | 32.4% |
| order-2 cap 10K | 10,000 | 0.86 MB | 15.39 | 38.4% |
| order-2 full | 28,176 | 1.27 MB | 13.79 | 40.7% |
| **order-3 full** | **130,856** | **3.54 MB** | **10.35** | **52.4%** |

**Prediction #1 CONFIRMED**: accuracy scales monotonically with TABLE size at
fixed compute. PPL 28.3 → 10.35 (2.7×), top-1 23.2% → 52.4%. The corpus is a
scaling axis independent of parameters. **PPL 10.35 / top-1 52.4% is the best
result in the entire project** — beats the transformer (11.9 / 42.4%).

**Channel complementarity (new finding):**
- Table alone: PPL = inf (zero-probability on unseen contexts — no smoothing)
- Mixer alone: PPL = 32.8 (coverage, no precision)
- Together: PPL = 10.35 — **compute = coverage (backoff/smoothing),
  memory = precision**. The channels are complementary, not redundant.

**Next:** exp23 — β-law on natural language (does β drop when n-gram coverage
is sparser?); exp24 — fused implementation for wall-clock.
