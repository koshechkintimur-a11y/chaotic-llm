"""pc_probe.py — Пекоры–Кэрролл как замена Арнольда: сохраняет ли синхронизация
точную идентичность? Честная проба: НЕ предполагаем, что PC работает —
ИЗМЕРЯЕМ ошибку синхронизации и точность декодирования, подбирая k.

Карта: Арнольд на непрерывном торе [0,1)^2 (как в нашей архитектуре, det=1,
обратима). Драйверы = токены (идентичность = точка на торе).

Вопросы:
  Q1. Сходится ли ответ к драйверу L (ошибка синхронизации -> 0)?
  Q2. Можно ли декодировать «какой токен на L» из ответа (ближайший сосед)?
  Q3. Арнольд-блендер (перестановка+coupling) как контроль: идентичность теряется?
"""
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

W = 256
T = 40
N_TRIALS = 200
V = 512
A = np.array([[1, 1], [1, 2]], dtype=np.float64)   # Arnold cat matrix (det=1)


def arnold_torus(p):
    """p: [.., 2] в [0,1)^2 -> A p mod 1 (обратимая хаотическая карта)."""
    p = p @ A.T
    return p % 1.0


def pc_step(y, driver, k):
    """Один шаг PC-синхронизации: y' = A y + k (d - y) mod 1."""
    fy = arnold_torus(y)
    return (fy + k * (driver - y)) % 1.0


def run_pc(ids, L, k):
    """Синхронизируем ответ к драйверу позиции L; возвращаем (sync_err, decode_acc)."""
    # драйверы: идентичность токена -> точка на торе (x=token/V, y=aux)
    aux = (torch.arange(W, dtype=torch.float64) * 0.6180339887) % 1.0   # золотое сечение
    drivers = torch.stack([(ids.float() / V) % 1.0, aux.unsqueeze(0)], dim=-1)  # [1,W,2]
    drivers = drivers[0]                                                   # [W,2]

    # эволюция всех драйверов под Арнольдом
    hist = [drivers]
    s = drivers.clone()
    for t in range(T):
        s = arnold_torus(s)
        hist.append(s)
    hist = torch.stack(hist)                                               # [T+1,W,2]

    # ответ: случайное начало
    y = torch.rand(2)
    driver_L = hist[:, L, :]                                               # [T+1,2]
    for t in range(T):
        y = pc_step(y, driver_L[t], k)

    # Q1: ошибка синхронизации (расстояние ответа до драйвера L в конце)
    sync_err = (y - hist[-1, L]).abs().max().item()

    # Q2: декодирование — ближайший драйвер по конечному состоянию
    final = hist[-1]                                                       # [W,2]
    dists = (y.unsqueeze(0) - final).pow(2).sum(-1)
    est_L = int(dists.argmin().item())
    decode_acc = 1.0 if est_L == L else 0.0
    return sync_err, decode_acc


def arnold_blend_decode(ids, L):
    """Контроль: Арнольд-блендер (перестановка + coupling значений) — декодируем L."""
    N = int(np.sqrt(W))
    idx = np.arange(W)
    states = (ids.float() / V) % 1.0                                       # [1,W]
    h = states.clone()
    for t in range(1, 9):
        pos = np.stack([idx // N, idx % N], -1)
        xp, yp = pos[..., 0], pos[..., 1]
        for _ in range(t):
            xp, yp = (2 * xp - yp) % N, (-xp + yp) % N
        perm = torch.as_tensor(xp * N + yp, dtype=torch.long)
        h = h[:, perm]
        even, odd = h[:, 0::2], h[:, 1::2]
        g = 0.5
        h = torch.stack([even + g * odd, odd + g * even], dim=1).reshape(1, W)
    final = h[0]
    # декодирование: ближайший к ОРИГИНАЛЬНОМУ токену L по конечному состоянию
    dists = (final.unsqueeze(0) - states[0].unsqueeze(1)).abs()            # [1,W]
    est = int(dists.argmin().item())
    return 1.0 if est == L else 0.0


# ===================== ПРОБА =====================
rng = np.random.default_rng(0)

# подбор k: какое k даёт синхронизацию?
print("=== Q1: подбор k (ошибка синхронизации) ===")
ids0 = torch.randint(0, V, (1, W))
for k in [0.3, 0.6, 0.9, 1.2, 1.5, 2.0]:
    errs = [run_pc(ids0, 64, k)[0] for _ in range(20)]
    print(f"  k={k:.1f}: sync_err={np.mean(errs):.4f}")

# лучший k из стабильных
best_k = 1.5
print(f"\n=== Q2: декодирование с k={best_k} ===")
pc_acc = {}
for L in [16, 64, 128, 240]:
    accs = [run_pc(ids0, L, best_k)[1] for _ in range(N_TRIALS)]
    pc_acc[L] = np.mean(accs)
    print(f"  L={L}: decode={np.mean(accs):.3f}")

print("\n=== Q3: Арнольд-блендер (контроль) ===")
blend_acc = {}
for L in [16, 64, 128, 240]:
    accs = [arnold_blend_decode(ids0, L) for _ in range(100)]
    blend_acc[L] = np.mean(accs)
    print(f"  L={L}: decode={np.mean(accs):.3f}")

print("\n=== ИТОГ ===")
print(f"PC (k={best_k}):     { {k: round(v,3) for k,v in pc_acc.items()} }")
print(f"Arnold-blend: { {k: round(v,3) for k,v in blend_acc.items()} }")
