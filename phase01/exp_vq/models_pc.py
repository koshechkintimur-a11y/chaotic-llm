"""models_pc.py — ЧИСТЫЙ PC-микшер (Пекоры–Кэрролл, полностью без Арнольда).

Архитектор: «убираем Арнольда из уравнения, ставим синхронизацию Пекоры–Кэрролла».
НЕТ permute_indices, НЕТ even/odd coupling, НЕТ Arnold-перестановок.

Вся динамика:
  1. Хаотическая диссипативная карта: h = h + tanh(h @ W + b) (спектр.радиус W > 1)
  2. PC-синхронизация: h = h + k * (driver - h) (однонаправленная связь к драйверу)
  Смешивание по ПОЗИЦИЯМ — только через выбор драйвера (КТО-адрес или mean).
"""
import math
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

W = 256  # window size

def count_params(m):
    return sum(p.numel() for p in m.parameters())


class PurePCBlock(nn.Module):
    """Чистый PC-блок: НЕТ Arnold, НЕТ even/odd coupling.
    Хаотическая динамика (residual + tanh) + PC-синхронизация к драйверу."""
    def __init__(self, d, k=1.2):
        super().__init__()
        # матрица со спектральным радиусом >1 (хаос) + tanh сжимает (диссипация)
        self.W = nn.Parameter(torch.eye(d) * 1.5 + torch.randn(d, d) * 0.05)
        self.b = nn.Parameter(torch.zeros(d))
        self.k = k

    def forward(self, h, driver):
        # хаотическая динамика: растяжение (W) + нелинейность (tanh) = диссипативный хаос
        h = h + torch.tanh(h @ self.W + self.b) * 0.3
        # PC-синхронизация к драйверу (однонаправленная)
        h = h + self.k * (driver - h)
        return h


class PurePCLM(nn.Module):
    """Чистый PC-микшер LM: без Арнольда, только хаотическая динамика + PC-синхронизация.
    driver_mode: 'mean' | 'crt' (КТО-селекция по позиции запроса)
    """
    def __init__(self, vocab=512, d=128, layers=4, k_init=1.2,
                 sync_steps=1, driver_mode="mean", primes=(3, 5, 7, 11)):
        super().__init__()
        self.d = d
        self.layers = layers
        self.sync_steps = sync_steps
        self.driver_mode = driver_mode
        self.primes = list(primes)
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.blocks = nn.ModuleList([PurePCBlock(d, k=k_init) for _ in range(layers)])
        self.k = nn.Parameter(torch.tensor([k_init]))
        if driver_mode == "crt":
            self.crt_proj = nn.ModuleList([nn.Linear(d, d) for _ in primes])
        self.readout = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, vocab))

    def _select_driver(self, h):
        """Выбор драйвера для readout-позиции (W-1)."""
        if self.driver_mode == "mean":
            return h.mean(dim=1, keepdim=True)  # [B,1,d]
        elif self.driver_mode == "crt":
            i = W - 1
            buckets = []
            for pi, p in enumerate(self.primes):
                mask = (torch.arange(W, device=h.device) % p) == (i % p)
                sel = h[:, mask]
                b = self.crt_proj[pi](sel).sum(dim=1, keepdim=True)
                buckets.append(b)
            return torch.cat(buckets, dim=1).mean(dim=1, keepdim=True)

    def forward(self, x, return_aux=False):
        h = self.embed(x) + self.pos
        # стек блоков чистой PC-синхронизации
        driver_seq = self._select_driver(h)  # общий драйвер на всю последовательность блоков
        for blk in self.blocks:
            h = blk(h, driver_seq)
        # финальная PC-синхронизация readout c T шагами
        driver = self._select_driver(h)
        k = torch.clamp(self.k, 0.0, 2.0)
        h_sync = h[:, -1:, :]
        for _ in range(self.sync_steps):
            h_sync = h_sync + k * (driver - h_sync)
        h_last = h_sync[:, -1, :]
        g = h.mean(dim=1)
        logits = self.readout(torch.cat([h_last, g], dim=-1))
        if return_aux:
            return logits, h_sync
        return logits


def build_pc_model(config, vocab=512, d=128, alpha=0.9, k_init=1.2,
                   sync_steps=1, driver_mode="mean"):
    if config == "pc":
        return PurePCLM(vocab=vocab, d=d, layers=4, k_init=k_init,
                        sync_steps=sync_steps, driver_mode=driver_mode)
    raise ValueError(f"unknown {config}")


if __name__ == "__main__":
    for cfg in ["pc"]:
        m = build_pc_model(cfg)
        x = torch.randint(0, 512, (2, W))
        y = m(x)
        print(f"{cfg}: params={count_params(m):,} out={tuple(y.shape)}")