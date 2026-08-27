"""exp19_chaotic_topk.py — Step 7: Chaotic Top-K Readout.

The hypothesis: chaotic mixing organizes the state space so that a cheap
linear selector can identify the K most relevant tokens. Attention then
only processes those K tokens — O(W log W) + O(K²·d) vs O(W²·d).

Variants:
  V1: linear selector → top-16 → attention
  V2: linear selector → top-64 → query-content refinement → top-16 → attention
  V3: linear selector → top-64 → refine → top-16 → refine → top-4 → attention

Baselines (same budget, 6000 steps):
  B1: Full attention (AttnLM, W=256)
  B2: Chaotic + full attention readout (v0.2, all W)
  B3: Chaotic + local readout (v0.3, no attention)
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp19_chaotic_topk")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W_LOCAL = 64
W = 256
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
BATCH = 64
MAX_TRAIN_BYTES = 2_000_000
ORDER = 2
BETA = 0.3
TRAIN_STEPS = 6000
K1 = 64
K2 = 16
K3 = 4
TEMP = 0.5


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))


def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)],
                            trainer=trainer)
    return tok


print("Training BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids
print(f"BPE vocab {V}, train tokens {len(train_ids):,}, test tokens {len(test_ids):,}")


# ============ beta corpus-prior ============
print("Building beta prior...")
prior = defaultdict(lambda: defaultdict(int))
for i in range(ORDER, len(train_ids)):
    prior[tuple(train_ids[i - ORDER:i])][train_ids[i]] += 1
prior = {k: dict(v) for k, v in prior.items()}

def prior_logp(ctx):
    for back in range(ORDER, 0, -1):
        table = prior.get(tuple(ctx[-back:]))
        if table:
            tot = sum(table.values())
            out = np.full(V, -1e9, dtype=np.float64)
            for t, c in table.items():
                out[t] = np.log(c / tot)
            return out
    return None


# ============ chaotic mixer ============
class ChaoticBlock(nn.Module):
    def __init__(self, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long)
                       for t in range(1, bl + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long)
                       for t in range(1, bg + 1)}

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, h):
        B, W, d = h.shape
        hw = h.view(B, self.Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l)
                           for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        gvec = glob.mean(dim=1, keepdim=True)
        return loc.reshape(B, W, d) + gvec


# ============ models ============
class ChaoticBase(nn.Module):
    def __init__(self, V, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.block = ChaoticBlock(W, Wl, d, bl, bg)
        self.norm = nn.LayerNorm(d)

    def mix(self, x):
        h = self.embed(x) + self.pos
        return self.norm(h + self.block(h))


def differentiable_topk(scores, K, temp=TEMP):
    """Differentiable top-K: mask non-top-K scores to -inf, softmax with temperature."""
    threshold = scores.topk(K, dim=-1)[0][:, -1:]  # (B, 1) K-th largest
    mask = (scores >= threshold).float()
    masked = scores - (1 - mask) * 1e9
    weights = F.softmax(masked / temp, dim=-1)
    return weights, mask


# V1: linear selector → top-K → attention
class ChaoticTopK(nn.Module):
    """linear selector → top-K → attention over K → output."""
    def __init__(self, base, V, K=K2):
        super().__init__()
        self.base = base
        self.K = K
        self.selector = nn.Linear(D_MODEL, 1, bias=False)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.kv = nn.Linear(D_MODEL, D_MODEL * 2)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.readout = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))

    def forward(self, x):
        B, W = x.shape
        h = self.base.mix(x)                       # (B, W, d)
        scores = self.selector(h).squeeze(-1)       # (B, W)
        weights, mask = differentiable_topk(scores, self.K)  # (B, W)
        # query token
        q = self.q(h[:, -1:, :])                   # (B, 1, d)
        kv = self.kv(h).reshape(B, W, 2, self.base.d).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]                       # (B, W, d)
        # weighted attention over K (weights are softmax over top-K, zeros elsewhere)
        # effective: k and v weighted by weights
        wk = k * weights.unsqueeze(-1)             # (B, W, d)
        wv = v * weights.unsqueeze(-1)
        # attention over weighted tokens
        scores_attn = (q @ wk.transpose(-2, -1)) / (D_MODEL ** 0.5)  # (B, 1, W)
        # mask out non-top-K before softmax
        scores_attn = scores_attn - (1 - mask.unsqueeze(1)) * 1e9
        attn_weights = F.softmax(scores_attn, dim=-1)
        out = self.proj(attn_weights @ wv)          # (B, 1, d)
        qh = self.norm(h[:, -1:, :] + out).squeeze(1)
        return self.readout(qh)


# V2: cascade: linear → top-64 → dot-product refinement → top-16 → attention
class ChaoticCascadeTopK(nn.Module):
    def __init__(self, base, V, K1=K1, K2=K2):
        super().__init__()
        self.base = base
        self.K1, self.K2 = K1, K2
        self.selector = nn.Linear(D_MODEL, 1, bias=False)  # stage 1
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)           # stage 2 query
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)           # stage 2 key
        self.q = nn.Linear(D_MODEL, D_MODEL)                # attention
        self.kv = nn.Linear(D_MODEL, D_MODEL * 2)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.readout = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))

    def forward(self, x):
        B, W = x.shape
        h = self.base.mix(x)                       # (B, W, d)
        # stage 1: cheap linear selector → top-K1
        s1 = self.selector(h).squeeze(-1)           # (B, W)
        _, mask1 = differentiable_topk(s1, self.K1, 0.5)
        # stage 2: content-based refinement on K1 survivors
        q = self.q_proj(h[:, -1:, :])               # (B, 1, d)
        k = self.k_proj(h)                          # (B, W, d)
        s2 = (q @ k.transpose(-2, -1)).squeeze(1)   # (B, W)
        s2 = s2 - (1 - mask1) * 1e9                 # zero out non-survivors
        weights2, mask2 = differentiable_topk(s2, self.K2, 0.5)
        # attention over K2 survivors
        kv = self.kv(h).reshape(B, W, 2, self.base.d).permute(2, 0, 1, 3)
        k_full, v_full = kv[0], kv[1]
        wk = k_full * weights2.unsqueeze(-1)
        wv = v_full * weights2.unsqueeze(-1)
        scores_attn = (q @ wk.transpose(-2, -1)) / (D_MODEL ** 0.5)
        scores_attn = scores_attn - (1 - mask2.unsqueeze(1)) * 1e9
        attn_weights = F.softmax(scores_attn, dim=-1)
        out = self.proj(attn_weights @ wv)
        qh = self.norm(h[:, -1:, :] + out).squeeze(1)
        return self.readout(qh)


# V3: triple cascade 256→64→16→4→attention
class ChaoticTripleCascade(nn.Module):
    def __init__(self, base, V, K1=K1, K2=K2, K3=K3):
        super().__init__()
        self.base = base
        self.K1, self.K2, self.K3 = K1, K2, K3
        self.selector = nn.Linear(D_MODEL, 1, bias=False)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.q_proj2 = nn.Linear(D_MODEL, D_MODEL)  # second refinement
        self.k_proj2 = nn.Linear(D_MODEL, D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.kv = nn.Linear(D_MODEL, D_MODEL * 2)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.readout = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))

    def forward(self, x):
        B, W = x.shape
        h = self.base.mix(x)
        # stage 1: linear → top-K1
        s1 = self.selector(h).squeeze(-1)
        _, m1 = differentiable_topk(s1, self.K1, 0.5)
        # stage 2: content → top-K2
        q1 = self.q_proj(h[:, -1:, :])
        k1 = self.k_proj(h)
        s2 = (q1 @ k1.transpose(-2, -1)).squeeze(1)
        s2 = s2 - (1 - m1) * 1e9
        _, m2 = differentiable_topk(s2, self.K2, 0.5)
        # stage 3: content → top-K3
        q2 = self.q_proj2(h[:, -1:, :])
        k2 = self.k_proj2(h)
        s3 = (q2 @ k2.transpose(-2, -1)).squeeze(1)
        s3 = s3 - (1 - m2) * 1e9
        w3, m3 = differentiable_topk(s3, self.K3, 0.5)
        # attention over K3
        kv = self.kv(h).reshape(B, W, 2, self.base.d).permute(2, 0, 1, 3)
        k_full, v_full = kv[0], kv[1]
        wk = k_full * w3.unsqueeze(-1)
        wv = v_full * w3.unsqueeze(-1)
        scores_attn = (q2 @ wk.transpose(-2, -1)) / (D_MODEL ** 0.5)
        scores_attn = scores_attn - (1 - m3.unsqueeze(1)) * 1e9
        attn_weights = F.softmax(scores_attn, dim=-1)
        out = self.proj(attn_weights @ wv)
        qh = self.norm(h[:, -1:, :] + out).squeeze(1)
        return self.readout(qh)


# ============ evaluate ============
def evaluate(model, W, n_samples=20000, beta=BETA):
    model.eval()
    nll_m, nll_p, acc_m, acc_p = [], [], 0, 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            logits = model(torch.tensor([ctx], dtype=torch.long))
            logp = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
            nll_m.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc_m += 1
            cp = prior_logp(ctx)
            if cp is not None:
                logp_c = np.logaddexp(np.log1p(-beta) + logp, np.log(beta) + cp)
            else:
                logp_c = logp
            nll_p.append(-logp_c[y])
            if int(np.argmax(logp_c)) == y:
                acc_p += 1
            if len(nll_m) >= n_samples:
                break
    n = len(nll_m)
    model.train()
    return {"n": n,
            "ppl_model": float(np.exp(np.mean(nll_m))),
            "ppl_prior": float(np.exp(np.mean(nll_p))),
            "ppl_gain": float(np.exp(np.mean(nll_m)) / np.exp(np.mean(nll_p))),
            "acc_model": acc_m / n, "acc_prior": acc_p / n}


def train(model, steps=TRAIN_STEPS, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    n = len(train_ids) - W - 1
    s = 0
    while s < steps:
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
        Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
        opt.zero_grad()
        loss = lossf(model(X), Y)
        loss.backward()
        opt.step()
        s += 1
        if s % 3000 == 0:
            print(f"  [{tag} {s:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    return loss.item()


results = {"V": V, "W": W, "d": D_MODEL, "order": ORDER, "beta": BETA,
           "train_steps": TRAIN_STEPS, "K1": K1, "K2": K2, "K3": K3,
           "train_tokens": len(train_ids), "test_tokens": len(test_ids)}

models = [
    ("V1_top16", lambda: ChaoticTopK(ChaoticBase(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL), V, K=K2)),
    ("V2_cascade_64_16", lambda: ChaoticCascadeTopK(ChaoticBase(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL), V, K1=K1, K2=K2)),
    ("V3_triple_64_16_4", lambda: ChaoticTripleCascade(ChaoticBase(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL), V, K1=K1, K2=K2, K3=K3)),
]

for name, make in models:
    print(f"\n=== {name} ===")
    model = make()
    n_params = sum(p.numel() for p in model.parameters())
    ckpt = os.path.join(OUT, f"{name}.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"loaded checkpoint ({n_params:,} params)")
    else:
        print(f"params: {n_params:,}")
        train(model, tag=name)
        torch.save(model.state_dict(), ckpt)
    r = evaluate(model, W)
    print(f"{name}: {r}")
    results[name] = {**r, "params": n_params}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)