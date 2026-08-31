"""pc_probe2.py — Пекоры–Кэрролл с ДИССИПАТИВНОЙ системой.

Ключ из pc_probe: Арнольд консервативен (det=1, нет диссипации) ->
синхронизация невозможна. PC требует устойчивое многообразие (отрицательные
условные показатели Ляпунова). Проверяем с диссипативным хаосом.

Система: диссипативная версия карты на торе (умножение + сжатие по y),
или классический диссипативный Hénon с параметрами в зоне аттрактора.
"""
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

W = 256
T = 60
N_TRIALS = 300
V = 512


def dissipative_map(p, a=1.4, b=0.3):
    """Диссипативный Hénon (классический хаос с аттрактором, det = -b)."""
    x, y = p[..., 0].clone(), p[..., 1].clone()
    nx = y + 1 - a * x * x
    ny = b * x
    return torch.stack([nx, ny], dim=-1)


def pc_step(y, driver, k, a=1.4, b=0.3):
    """y' = f(y) + k*(d - y): ответ подтягивается к драйверу."""
    fy = dissipative_map(y, a, b)
    return fy + k * (driver - y)


def run_pc(ids, L, k):
    """Синхронизация ответа к драйверу L; возвращает (sync_err, decode_acc)."""
    aux = (torch.arange(W, dtype=torch.float64) * 0.6180339887) % 1.0
    # идентичность: x = токен/V в [0,1), y = aux в [0,1) — старт на аттракторе-ish
    drivers = torch.stack([(ids.float() / V) % 1.0, aux.unsqueeze(0)], dim=-1)[0]  # [W,2]

    hist = [drivers]
    s = drivers.clone()
    for t in range(T):
        s = dissipative_map(s)
        hist.append(s)
    hist = torch.stack(hist)          # [T+1,W,2]

    y = torch.rand(2) * 1.0
    driver_L = hist[:, L, :]          # [T+1,2]
    for t in range(T):
        y = pc_step(y, driver_L[t], k)

    sync_err = (y - hist[-1, L]).abs().max().item()
    final = hist[-1]                  # [W,2]
    dists = (y.unsqueeze(0) - final).pow(2).sum(-1)
    est_L = int(dists.argmin().item())
    return sync_err, 1.0 if est_L == L else 0.0


rng = np.random.default_rng(0)
ids0 = torch.randint(0, V, (1, W))

print("=== Q1: подбор k (ошибка синхронизации, диссипативный Hénon) ===")
for k in [0.1, 0.3, 0.6, 0.9, 1.2]:
    errs = [run_pc(ids0, 64, k)[0] for _ in range(20)]
    print(f"  k={k:.1f}: sync_err={np.mean(errs):.4f}")

best_k = 0.6
print(f"\n=== Q2: декодирование с k={best_k} ===")
pc_acc = {}
for L in [16, 64, 128, 240]:
    accs = [run_pc(ids0, L, best_k)[1] for _ in range(N_TRIALS)]
    pc_acc[L] = float(np.mean(accs))
    print(f"  L={L}: decode={np.mean(accs):.3f}")

print(f"\nPC (k={best_k}): { {k: round(v,3) for k,v in pc_acc.items()} }")
