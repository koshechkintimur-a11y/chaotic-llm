"""models_pc.py — ЧИСТЫЙ PC-микшер + Lightweight Address Selection (LAS).

Архитектор: «убираем Арнольда из уравнения, ставим синхронизацию Пекоры–Кэрролла».
НЕТ permute_indices, НЕТ even/odd coupling, НЕТ Arnold.

Динамика:
  1. Хаотическая диссипативная карта: h = h + tanh(h @ W + b) (спектр.радиус W > 1)
  2. PC-синхронизация: h = h + k * (driver - h)

Селекция драйвера (lightweight address, O(W·d)):
  mean  — глобальный пул (базовая линия, разбавляет KEY в 256×)
  last  — контроль: driver = последняя позиция (без селекции)
  top1  — query=последняя позиция, keys=все, cosine, берём 1 позицию argmax
  soft  — softmax(cosine/temp) по позициям, взвешенная сумма
  crt   — КТО-сумма (провалился ранее, оставлен для сравнения)
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
    """Чистый PC-блок: НЕТ Arnold. Хаотическая карта + PC-синхронизация к драйверу."""
    def __init__(self, d, k=1.2):
        super().__init__()
        self.W = nn.Parameter(torch.eye(d) * 1.5 + torch.randn(d, d) * 0.05)
        self.b = nn.Parameter(torch.zeros(d))
        self.k = k

    def forward(self, h, driver):
        h = h + torch.tanh(h @ self.W + self.b) * 0.3   # диссипативный хаос
        h = h + self.k * (driver - h)                    # PC-синхронизация
        return h


class PurePCLM(nn.Module):
    def __init__(self, vocab=512, d=128, layers=4, k_init=1.2,
                 sync_steps=1, driver_mode="mean", temp=0.3, primes=(3, 5, 7, 11)):
        super().__init__()
        self.d = d
        self.layers = layers
        self.sync_steps = sync_steps
        self.driver_mode = driver_mode
        self.temp = temp
        self.primes = list(primes)
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.blocks = nn.ModuleList([PurePCBlock(d, k=k_init) for _ in range(layers)])
        self.k = nn.Parameter(torch.tensor([k_init]))
        if driver_mode == "crt":
            self.crt_proj = nn.ModuleList([nn.Linear(d, d) for _ in primes])
        self.readout = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, vocab))
        self.last_driver_pos = None  # для анализа распределения выбранных позиций

    def _address_weights(self, h):
        """Query=последняя позиция, keys=все позиции, cosine.
        Маскируем позицию запроса и ближайшие 8 (самовыбор = тривиальный максимум,
        не даёт найти ДРУГОЙ драйвер)."""
        B = h.shape[0]
        q = h[:, -1, :]                                   # [B, d]
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6)    # [B, W, d]
        sim = (hn * qn.unsqueeze(1)).sum(-1)              # [B, W] cosine
        # маскируем последние 8 позиций (включая саму query) — запрещаем самовыбор
        sim[:, W - 8:] = -1e9
        return sim

    def _select_driver(self, h):
        """Выбор драйвера для readout-позиции (W-1). Записывает позиции для анализа."""
        B = h.shape[0]
        m = self.driver_mode
        if m == "mean":
            driver = h.mean(dim=1, keepdim=True)
            self.last_driver_pos = torch.full((B,), -1, dtype=torch.long, device=h.device)
        elif m == "last":
            driver = h[:, -1:, :]                          # контроль: сам последний токен
            self.last_driver_pos = torch.full((B,), W - 1, dtype=torch.long, device=h.device)
        elif m == "top1":
            sim = self._address_weights(h)                 # [B, W]
            idx = sim.argmax(dim=1)                        # [B]
            self.last_driver_pos = idx
            driver = h[torch.arange(B, device=h.device), idx][:, None, :]
        elif m == "soft":
            sim = self._address_weights(h) / self.temp
            w = torch.softmax(sim, dim=1)                  # [B, W]
            self.last_driver_pos = w.argmax(dim=1)
            driver = (w.unsqueeze(-1) * h).sum(dim=1, keepdim=True)
        elif m == "crt":
            i = W - 1
            buckets = []
            for pi, p in enumerate(self.primes):
                mask = (torch.arange(W, device=h.device) % p) == (i % p)
                sel = h[:, mask]
                b = self.crt_proj[pi](sel).sum(dim=1, keepdim=True)
                buckets.append(b)
            driver = torch.cat(buckets, dim=1).mean(dim=1, keepdim=True)
            self.last_driver_pos = torch.full((B,), -2, dtype=torch.long, device=h.device)
        else:
            raise ValueError(f"unknown driver_mode {m}")
        return driver

    def forward(self, x, return_aux=False):
        h = self.embed(x) + self.pos
        # динамическая селекция на каждом блоке (итеративная синхронизация)
        for blk in self.blocks:
            driver = self._select_driver(h)
            h = blk(h, driver)
        # финальная синхронизация readout c T шагами
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
                   sync_steps=1, driver_mode="mean", temp=0.3):
    if config == "pc":
        return PurePCLM(vocab=vocab, d=d, layers=4, k_init=k_init,
                        sync_steps=sync_steps, driver_mode=driver_mode, temp=temp)
    raise ValueError(f"unknown {config}")


if __name__ == "__main__":
    for mode in ["mean", "last", "top1", "soft", "crt"]:
        m = build_pc_model("pc", driver_mode=mode)
        x = torch.randint(0, 512, (2, W))
        y = m(x)
        pos = m.last_driver_pos
        print(f"{mode}: params={count_params(m):,} out={tuple(y.shape)} last_driver_pos={pos.tolist()}")
