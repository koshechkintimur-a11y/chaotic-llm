# Final Benchmark Results
**Commit:** `47877a6f5e6603b1d66a3e615b2b34a58e5bf7f0`
**Protocol:** FINAL_BENCHMARK.md
**Seeds:** [0, 1, 2]

## Parameter-matched benchmark
| Model | Params | PPL |
| --- | --- | --- |
| STS-Prog | 900,353 | 17.936±0.295 |
| STS-Prog (no-PC) | 900,353 | 19.678±0.233 |
| Transformer (D=88) | 865,904 | 21.569±0.261 |

## Retrieval
| Model | L | Trials | Hits | Accuracy |
| --- | --- | --- | --- | --- |
| STS-Prog seed=0 | L=16 | 200 | 84 | 0.420 |
| STS-Prog seed=0 | L=32 | 200 | 73 | 0.365 |
| STS-Prog seed=0 | L=64 | 200 | 73 | 0.365 |
| STS-Prog seed=0 | L=128 | 200 | 68 | 0.340 |
| STS-Prog seed=0 | L=256 | 200 | 56 | 0.280 |
| STS-Prog seed=1 | L=16 | 200 | 82 | 0.410 |
| STS-Prog seed=1 | L=32 | 200 | 90 | 0.450 |
| STS-Prog seed=1 | L=64 | 200 | 67 | 0.335 |
| STS-Prog seed=1 | L=128 | 200 | 70 | 0.350 |
| STS-Prog seed=1 | L=256 | 200 | 63 | 0.315 |
| STS-Prog seed=2 | L=16 | 200 | 63 | 0.315 |
| STS-Prog seed=2 | L=32 | 200 | 71 | 0.355 |
| STS-Prog seed=2 | L=64 | 200 | 72 | 0.360 |
| STS-Prog seed=2 | L=128 | 200 | 79 | 0.395 |
| STS-Prog seed=2 | L=256 | 200 | 73 | 0.365 |
| STS-Prog (no-PC) seed=0 | L=16 | 200 | 44 | 0.220 |
| STS-Prog (no-PC) seed=0 | L=32 | 200 | 28 | 0.140 |
| STS-Prog (no-PC) seed=0 | L=64 | 200 | 33 | 0.165 |
| STS-Prog (no-PC) seed=0 | L=128 | 200 | 30 | 0.150 |
| STS-Prog (no-PC) seed=0 | L=256 | 200 | 33 | 0.165 |
| STS-Prog (no-PC) seed=1 | L=16 | 200 | 35 | 0.175 |
| STS-Prog (no-PC) seed=1 | L=32 | 200 | 47 | 0.235 |
| STS-Prog (no-PC) seed=1 | L=64 | 200 | 29 | 0.145 |
| STS-Prog (no-PC) seed=1 | L=128 | 200 | 44 | 0.220 |
| STS-Prog (no-PC) seed=1 | L=256 | 200 | 31 | 0.155 |
| STS-Prog (no-PC) seed=2 | L=16 | 200 | 35 | 0.175 |
| STS-Prog (no-PC) seed=2 | L=32 | 200 | 23 | 0.115 |
| STS-Prog (no-PC) seed=2 | L=64 | 200 | 29 | 0.145 |
| STS-Prog (no-PC) seed=2 | L=128 | 200 | 34 | 0.170 |
| STS-Prog (no-PC) seed=2 | L=256 | 200 | 44 | 0.220 |
| Transformer (D=88) seed=0 | L=16 | 200 | 51 | 0.255 |
| Transformer (D=88) seed=0 | L=32 | 200 | 43 | 0.215 |
| Transformer (D=88) seed=0 | L=64 | 200 | 43 | 0.215 |
| Transformer (D=88) seed=0 | L=128 | 200 | 35 | 0.175 |
| Transformer (D=88) seed=0 | L=256 | 200 | 34 | 0.170 |
| Transformer (D=88) seed=1 | L=16 | 200 | 51 | 0.255 |
| Transformer (D=88) seed=1 | L=32 | 200 | 50 | 0.250 |
| Transformer (D=88) seed=1 | L=64 | 200 | 38 | 0.190 |
| Transformer (D=88) seed=1 | L=128 | 200 | 43 | 0.215 |
| Transformer (D=88) seed=1 | L=256 | 200 | 47 | 0.235 |
| Transformer (D=88) seed=2 | L=16 | 200 | 44 | 0.220 |
| Transformer (D=88) seed=2 | L=32 | 200 | 30 | 0.150 |
| Transformer (D=88) seed=2 | L=64 | 200 | 27 | 0.135 |
| Transformer (D=88) seed=2 | L=128 | 200 | 35 | 0.175 |
| Transformer (D=88) seed=2 | L=256 | 200 | 36 | 0.180 |

## Scaling
| Model | W | VRAM MB | Time ms | Tok/s |
| --- | --- | --- | --- | --- |