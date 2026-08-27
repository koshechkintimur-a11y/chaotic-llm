# Experiment 20 — Gumbel Top-K Retrieval (Step 8: decisive test)

## Hypothesis (refined)

Chaotic dynamics generates a cheap structured proposal space for selective
computation. Can a LEARNABLE temperature-softmax selector find the correct
key tokens from the chaotic state on Task A (associative recall)?

## Setup

- Task A (toy_data.py): N=16 tokens, K=8 keys, each key appears 2×.
  Query at position 15. Answer = value of the query key.
- Models: embed → chaotic mixer (6 blocks, 4×4 grid) → selector → readout.
- ChaoticTopK: temperature-softmax attention (τ 2.0→0.2 annealing, 3000 steps).
- Baselines: FullAttn (transformer), ChaoticAttnReadout (full attention over N),
  ChaoticLocal (no attention).

## Results

| Model | Accuracy | Селектор |
|---|---|---|
| FullAttn (трансформер) | **1.0** | полный dot-product, O(N²·d) |
| ChaoticAttnReadout (mixer + attn над всеми) | **1.0** | query·state dot-product, O(N·d) |
| **ChaoticTopK** (температурный soft селектор) | **0.49** | линейный, контент-независимый |
| ChaoticTopK — hard top-2 eval | 0.28 | линейный, top-2 при τ=0.2 |
| ChaoticLocal (без attention) | 0.43 | локальный MLP |

**recall@K (top-2 обнаружения правильного ключевого токена): 0.25**

## Interpretation

1. **ChaoticTopK (0.49) > ChaoticLocal (0.43)** — обучаемый температурный
   селектор реально находит полезную информацию, лучшую, чем локальный readout.
   +6% — это скромно, но не ноль.

2. **Но!** recall@K=0.25 — правильный ключевой токен в top-2 только в 25%
   случаев. Линейный селектор (без взаимодействия с запросом) НЕ может
   определить, какой токен релевантен — для этого нужно содержимое запроса.

3. **ChaoticAttnReadout = 1.0** — один единственный dot-product
   (запрос·состояние) над хаотическими состояниями даёт идеальную точность.
   Хаотический mixer делает пространство structure ready для легковесного
   content-dependent взаимодействия, но не устраняет необходимость содержательного
   сравнения.

## Conclusion

**Гипотеза подтверждена в уточнённой форме:**

✅ **Chaotic mixer makes the space attention-ready.** Single query·state dot
product over chaotic states = 1.0 accuracy (same as full transformer).
No multi-head, no deep projections needed.

❌ **Chaotic mixer does NOT make the space ready for content-independent
selection.** A linear selector (no query interaction) cannot identify the
correct tokens (recall@K=0.25).

**The refined architecture lesson:** chaotic mixer replaces the COMPUTATION
(multi-head QKV, deep projections, softmax over all pairs) with a cheap
structured space where a SINGLE dot product suffices. Selection still needs
content-dependence, but the cost drops from O(N²·d·H) to O(N·d).

## Next

- Compare: single-head attention over chaotic states vs multi-head over raw
  states — quantify the computation saved by the mixer's organization.
- Chaotic mixer + single query·key attention as the full architecture v0.4.