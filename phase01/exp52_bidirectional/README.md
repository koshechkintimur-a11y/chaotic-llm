# Experiment 52 — Bidirectional Gears + Learnable Projection: BREAKTHROUGH

## Result

| Model (1.2M, same protocol) | mixer-only PPL | +sparse β(c_h) |
|---|---|---|
| scalar (exp41b) | 97.4 | 9.00 |
| vector (exp42) | 159.3 | 9.08 |
| dual, no learnable params (exp50) | 133.5 | 9.04 |
| **bidirectional + Linear(2d→d) (exp52)** | **13.04** | **8.475** |
| Transformer control (exp45) | 8.65 | 7.97 |

Train loss trajectory: 6.25 → 4.37 (4K) → 2.97 (6K) → 2.51 (8K) → 2.34 (10K)
→ 2.11 (12K) → 1.98 (14K). Params: 1.71M (+262K for the projection).

## The insight

The d=512 scaling ceiling was NOT the chaotic mixing itself — it was the
**absence of learnable mixing capacity**. The fixed permutations + scalar
gates can't combine directions. The learnable Linear(2d→d) projection over
forward + backward mixer outputs gives the model a learnable way to
combine them — and the ceiling dissolves.

- exp50 (dual, no learnable params): FAIL (133.5)
- exp52 (bidirectional + learnable projection): **13.04 mixer-only,
  8.475 fused — best mixer result of the project**

## Position

- Bidirectional mixer alone (13.04) is within 1.5× of the Transformer
  (8.65) — while keeping O(W·d) instead of O(W²·d) compute.
- Fused 8.475 beats every previous architecture variant (best was 8.59).
- The "gears" hypothesis is PARTIALLY supported: not multiple mixers per
  se, but the LEARNABLE COMBINATION of mixing directions is what unlocks
  width scaling.

## Next

- Apply the learnable-projection principle at smaller scale: does a
  forward+backward + Linear(2d→d) at d=64/256 also improve (is it a
  general architecture upgrade, not just a d=512 fix)?
- Test bidirectional on NL.
- Measure wall-clock: bidirectional is 2× the mixer cost (two passes) —
  still O(W·d), verify it beats attention at large W.