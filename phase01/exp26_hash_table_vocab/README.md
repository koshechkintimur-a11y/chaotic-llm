# Experiment 26 — Hash Table + Vocab Scaling (Phase 5)

## Hypothesis

The memory-scaling advantage survives (a) hash-based storage (probing table,
KenLM-style) and (b) a 4× larger vocabulary (BPE-2048).

## Setup

- Corpus: code, 2M train chars.
- Part A (vocab 512): reuse exp22 mixer; hash-KN order-3 vs dict.
- Part B (vocab 2048): retrain tiny mixer (6000 steps, 289K params) + hash-KN order-3.
- Probing hash table: linear probing, splitmix hash, packed 64-bit keys.

## Results

### Part A — hash vs dict (vocab 512, β=0.3)

| Storage | PPL fused | top-1 |
|---|---|---|
| dict-MLE order-3 | 14.24 | 40.2% |
| hash-MLE order-3 | 14.24 | 40.2% (0 disagreements — faithful) |
| **hash-KN order-3** | **12.10** | **42.3%** |

### Part B — vocab 2048 (β=0.5)

| Model | PPL | top-1 | Memory |
|---|---|---|---|
| Mixer alone | 77.6 | 21.0% | — |
| **Mixer + hash-KN order-3** | **20.18** | **40.7%** | **8.9 MB** |

### Bug fixed along the way

Initial hash results were wrong (25.91 / 35.99): `ProbeTable.insert` was called
once per distinct n-gram → all counts = 1. Fixed with a `count` parameter.
After fix: 0 disagreements with the dict, and KN results are correct.

## Interpretation

1. **Hash storage is FAITHFUL**: hash-MLE ≡ dict-MLE exactly (PPL 14.24 both).
   The probing table (4.71-5.7 MB) is a valid replacement for the dict —
   same quality, compact, scales with the vocab.

2. **Hash-KN order-3 beats raw MLE** (12.10 vs 14.24): with correct counts,
   KN smoothing wins at order-3 on code too (consistent with exp24 on NL).

3. **Vocab scaling works**: BPE-2048 + hash-KN gives PPL 20.18 / top-1 40.7%
   at 8.9 MB. Per-token efficiency: vocab-2048 fused (0.0099 PPL/token) is
   2.4× better than vocab-512 (0.0236). The memory channel boosts top-1
   +19.7 п.п. over the mixer alone.

4. The β-Architecture scales toward real vocabularies: hash storage + KN +
   larger BPE all work; memory grows sub-linearly with vocab (8.9 MB at 2048).

## Conclusion

**Подтверждено**: memory channel scales with hash storage and larger vocab.
The probing table is faithful to exact storage; KN smoothing + vocab growth
improve per-token efficiency. Hash table ≈ 8.9 MB at BPE-2048 — the
architecture's memory axis holds at scale.

## Next

exp27 — end-to-end text GENERATION from the β-Architecture (autoregressive
sampling with mixer + KN table + β-gate), not just next-token PPL.