# Experiment 40 — Larger Mixer (d=256, 460K params)

## Question

Does Architecture v0.7 hold when the mixer is scaled up 4× (d=64 → d=256,
90K → 460K params, 12+6 chaotic blocks)?

## Result (code, V=512)

| Mixer | mixer-only | +sparse β(c_h) | +full KN |
|---|---|---|---|
| d=64 (90K, exp36) | 32.78 | **8.59** | 10.94 |
| **d=256 (460K)** | **16.73** | **8.70** | 10.29 |

## Interpretation

1. **Bigger mixer = 2× better alone**: mixer-only drops 32.78 → 16.73
   (the chaotic compute channel scales with width/block count).

2. **Fused result is stable**: +sparse β(c_h) ≈ 8.6-8.7 on both model sizes.
   The memory channel dominates the final quality — the mixer's 2× gain
   mostly cancels because the memory already covers the common patterns.

3. **The architecture still beats KN at scale**: 8.70 vs 10.29 (d=256),
   same as 8.59 vs 10.94 (d=64). Cheap memory + confidence gate beats full
   Kneser-Ney regardless of mixer size.

4. **Generation is structurally richer**: d=256 produces more real code
   scaffolding (useState, Prisma types, export type, api calls) than d=64.
   Still noisy — corpus is mixed-language (Russian comments, multiple
   languages).

## Verdict

**Architecture v0.7 scales with compute.** Bigger mixer → stronger standalone
model; the cheap memory + β(c_h) advantage over KN is preserved at both
model sizes. The compute channel's scaling is real (2× PPL gain), and the
memory channel keeps quality flat near the best achievable.