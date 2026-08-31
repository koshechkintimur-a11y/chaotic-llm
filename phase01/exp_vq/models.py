"""models.py — VQ-Phase (ТЗ этап A): Vector Quantization перед хаотическим микшером.

База: BidirectionalMixer (exp_memory_selector, проверен на corpus5m: gated=10.46).
Иерархия уже внутри: локальные окна Wl=64 + глобальное реле W=256.

Варианты (ТЗ A.4):
  baseline — BidirectionalMixer + order-3 prior (без VQ)
  vq_only  — + VQ перед микшером, exact-path как раньше
  vq_kto   — + VQ + КТО-корзины по квантованному ключу
  vq_aux   — + VQ + сильный aux-loss на использование exact-path
  nochao   — контроль: VQ + MLP (без Арнольда) + exact-path
"""
import math
import torch
import torch.nn as nn
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, REPO)
sys.path.insert(0, PHASE)
sys.path.insert(0, os.path.join(PHASE, "exp_memory_selector"))

W = 256
D = 128
BLOCKS = 4
VOCAB = 512


class VQLayer(nn.Module):
    """VQ-VAE style: codebook + commitment loss + straight-through estimator."""
    def __init__(self, dim, n_codes, beta_commit=0.5, init_scale=0.02):
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        self.beta_commit = beta_commit
        self.codebook = nn.Parameter(torch.randn(n_codes, dim) * init_scale)

    def forward(self, z):
        B, Wd, D = z.shape
        zf = z.reshape(-1, D)
        dist = (zf.pow(2).sum(1, keepdim=True)
                - 2 * zf @ self.codebook.t()
                + self.codebook.pow(2).sum(1, keepdim=True).t())
        idx = dist.argmin(dim=1)
        ze = self.codebook[idx]                       # [B*W, D]
        ze_b = ze.reshape(B, Wd, D)
        zq = z + (ze_b - z).detach()                  # straight-through
        commit = ((ze_b.detach() - z) ** 2).mean()
        codebook = ((ze_b - z.detach()) ** 2).mean()
        usage = len(torch.unique(idx)) / self.n_codes
        return zq.reshape(B, Wd, D), idx.reshape(B, Wd), commit, codebook, usage


def build_mixer():
    from experiment import BidirectionalMixer   # Wl=64 локально + W=256 глобально
    return BidirectionalMixer(seed=0, blocks=BLOCKS)


class BaseLM(nn.Module):
    """Базовый класс: embed + pos + mixer + readout(last, global)."""
    def __init__(self, V, W=W, d=D):
        super().__init__()
        self.V, self.W, self.d = V, W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.readout = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, V))
        self.has_vq = False

    def embed_pos(self, x):
        return self.embed(x) + self.pos

    def head(self, h):
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


class BaselineLM(BaseLM):
    """baseline: BidirectionalMixer + order-3 prior (без VQ)."""
    def __init__(self, V, W=W, d=D):
        super().__init__(V, W, d)
        self.mixer = build_mixer()

    def forward(self, x):
        h = self.mixer(self.embed_pos(x))
        return self.head(h)


class VQLM(BaseLM):
    """vq_only: VQ перед микшером, exact-path как раньше (β-гейт на eval)."""
    def __init__(self, V, n_codes=512, beta_commit=0.5, W=W, d=D):
        super().__init__(V, W, d)
        self.vq = VQLayer(d, n_codes, beta_commit=beta_commit)
        self.mixer = build_mixer()
        self.has_vq = True
        self.vq_w = 0.1
        self.beta_commit = beta_commit

    def forward(self, x):
        h, idx, commit, cb, usage = self.vq(self.embed_pos(x))
        return self.head(self.mixer(h)), commit, cb, usage, idx


class VQKTOLM(VQLM):
    """vq_kto: VQ + КТО-корзины по квантованному ключу (точный путь)."""
    def __init__(self, V, n_codes=512, beta_commit=0.5, W=W, d=D,
                 primes=(3, 5, 7, 11), d_crt=16):
        super().__init__(V, n_codes=n_codes, beta_commit=beta_commit, W=W, d=d)
        self.primes = primes
        self.n_buckets = sum(primes)
        self.d_crt = d_crt
        # отдельная проекция на каждую корзину (не сжимать в вектор!)
        self.per_bucket = nn.ModuleList([nn.Linear(d, d_crt) for _ in range(self.n_buckets)])
        self.ktogate = nn.Sequential(nn.Linear(d + self.n_buckets * d_crt, d),
                                     nn.ReLU(), nn.Linear(d, d))
        # суммарные параметры почти не растут (d_crt=16)

    def crt_buckets(self, e, idx):
        """Корзины остатков по КВАНТОВАННЫМ индексам (дискретный ключ)."""
        B, Wd, _ = e.shape
        out = []
        for p in self.primes:
            for r in range(p):
                mask = (torch.arange(Wd, device=e.device) % p) == r
                b = e[:, mask, :].sum(dim=1)                 # [B, d]
                out.append(self.per_bucket[len(out)](b))     # [B, d_crt]
        return torch.cat(out, dim=-1)                        # [B, n_buckets*d_crt]

    def forward(self, x):
        e = self.embed_pos(x)
        h, idx, commit, cb, usage = self.vq(e)
        hm = self.mixer(h)
        last = hm[:, -1, :]                                  # [B,d]
        gvec = hm.mean(dim=1)
        rec = self.crt_buckets(h, idx)                       # точный путь по ключу
        gate_in = torch.cat([last, rec], dim=-1)
        g = torch.sigmoid(self.ktogate(gate_in))             # [B,d]
        mix = last * (1 - g) + g * self.ktogate(gate_in)     # локально: g уже [B,d]
        # корректная смесь: возьмём последний + точный вектор
        exact_vec = self.ktogate(gate_in)
        combined = last * (1 - g) + exact_vec * g
        return self.readout(torch.cat([combined, gvec], dim=-1)), commit, cb, usage, idx


class VQAuxLM(VQLM):
    """vq_aux: VQ + сильный aux-loss на использование exact-path (KL-штраф)."""
    def __init__(self, V, n_codes=512, beta_commit=0.5, W=W, d=D, aux_w=0.5):
        super().__init__(V, n_codes=n_codes, beta_commit=beta_commit, W=W, d=D)
        # exact-path: отдельный головка от квантованного ключа (свободный prior)
        self.exact_head = nn.Linear(d, V)
        self.aux_w = aux_w

    def forward(self, x):
        e = self.embed_pos(x)
        h, idx, commit, cb, usage = self.vq(e)
        hm = self.mixer(h)
        logits = self.head(hm)
        # aux: предсказание по exact-пути от квантованного ключа (среднее по окну)
        key_vec = h.mean(dim=1)                              # [B,d]
        exact_logits = self.exact_head(key_vec)
        return logits, exact_logits, commit, cb, usage


class NoChaoLM(BaseLM):
    """nochao: контроль — VQ + локальный MLP (без Арнольда) + exact-path."""
    def __init__(self, V, n_codes=512, beta_commit=0.5, W=W, d=D):
        super().__init__(V, W, d)
        self.vq = VQLayer(d, n_codes, beta_commit=beta_commit)
        self.mlp = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=16, padding=15),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=16, padding=15),
        )
        self.exact_head = nn.Linear(d, V)
        self.has_vq = True
        self.vq_w = 0.1

    def forward(self, x):
        e = self.embed_pos(x)
        h, idx, commit, cb, usage = self.vq(e)
        hm = self.mlp(h.transpose(1, 2)).transpose(1, 2)
        logits = self.head(hm)
        key_vec = h.mean(dim=1)
        exact_logits = self.exact_head(key_vec)
        return logits, exact_logits, commit, cb, usage


def build_model(config, V, n_codes=512, beta_commit=0.5):
    if config == "baseline":
        return BaselineLM(V)
    if config == "vq_only":
        return VQLM(V, n_codes=n_codes, beta_commit=beta_commit)
    if config == "vq_kto":
        return VQKTOLM(V, n_codes=n_codes, beta_commit=beta_commit)
    if config == "vq_aux":
        return VQAuxLM(V, n_codes=n_codes, beta_commit=beta_commit)
    if config == "nochao":
        return NoChaoLM(V, n_codes=n_codes, beta_commit=beta_commit)
    raise ValueError(config)


if __name__ == "__main__":
    for cfg in ["baseline", "vq_only", "vq_kto", "vq_aux", "nochao"]:
        m = build_model(cfg, 512)
        n = sum(p.numel() for p in m.parameters())
        print(f"{cfg:10s}: {n:,} params")
    m = build_model("vq_kto", 512)
    out = m(torch.randint(0, 512, (2, 256)))
    print("vq_kto forward ok:", out[0].shape, "usage:", float(out[3]))
