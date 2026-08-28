"""exp40_larger_mixer.py — scale up the mixer (d=256, ~350K params).

v0.7 used a tiny d=64 mixer (90K params). "А что если" a larger mixer
(d=256) produces better generation while keeping the sparse MLE + β(c_h)
advantage? Tests the architecture at a more realistic model scale.

Train: 8000 steps, code corpus, CE loss (standalone — no memory in training).
Evaluate: mixer-only, +sparse β(c_h), +KN. Generate: compare with exp37.
"""
import os
import sys
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp40_larger_mixer")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
D_MODEL = 256          # 4× wider
BLOCKS_LOCAL = 12      # more blocks
BLOCKS_GLOBAL = 6
ORDER = 3
MAX_TRAIN_BYTES = 2_000_000
BATCH = 64
TRAIN_STEPS = 8000
N_TR, N_EVAL = 80000, 12000
EVAL_BATCH = 512       # smaller for larger model


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))


def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i+100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok


print("BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids
print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}")

# sparse MLE memory
ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1
print(f"contexts: {len(ctx_counts):,}")


# Larger model
class ChaoticBlock(nn.Module):
    def __init__(self, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long) for t in range(1, bl + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long) for t in range(1, bg + 1)}

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even, odd = h[:, 0::2, :], h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, h):
        B, W, d = h.shape
        hw = h.view(B, self.Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l) for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        return loc.reshape(B, W, d) + glob.mean(dim=1, keepdim=True)


class ChaoticBase(nn.Module):
    def __init__(self, V, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.block = ChaoticBlock(W, Wl, d, bl, bg)
        self.norm = nn.LayerNorm(d)

    def mix(self, x):
        return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))


class ModelV1(nn.Module):
    def __init__(self, base, V):
        super().__init__()
        self.base = base
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


base = ChaoticBase(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
model = ModelV1(base, V)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# ============ training ============
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
n = len(train_ids) - W - 1
t0 = time.time()
print("training...")
for s in range(TRAIN_STEPS):
    bi = np.random.randint(0, n, size=BATCH)
    X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
    Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
    opt.zero_grad()
    loss = lossf(model(X), Y)
    loss.backward()
    opt.step()
    if s % 2000 == 0:
        print(f"  [{s:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
torch.save(model.state_dict(), os.path.join(OUT, "mixer_d256.pt"))
print("trained + saved")

# ============ eval ============
model.eval()
rng = np.random.default_rng(42)
maxstart_te = len(test_ids) - W - 1
te_starts = np.sort(rng.choice(maxstart_te, size=N_EVAL, replace=False))

logits_te, y_te = np.zeros((N_EVAL, V), dtype=np.float32), np.zeros(N_EVAL, dtype=np.int64)
with torch.no_grad():
    for s0 in range(0, N_EVAL, EVAL_BATCH):
        e = min(s0 + EVAL_BATCH, N_EVAL)
        X = np.zeros((e - s0, W), dtype=np.int64)
        for k, i in enumerate(te_starts[s0:e]):
            X[k] = test_ids[i:i + W]
            y_te[s0 + k] = test_ids[i + W]
        logits_te[s0:e] = model(torch.tensor(X, dtype=torch.long)).numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()

mixer_only = float(np.exp(np.mean([-lpmix_te[k, y_te[k]] for k in range(N_EVAL)])))
print(f"\nmixer-only (d=256): PPL {mixer_only:.3f}")

# sparse β(c_h) and KN
def eval_mem(k_beta, beta_max=1.0):
    N = N_EVAL
    nll = np.zeros(N)
    for k in range(N):
        i = te_starts[k]
        pos = i + W
        ctx = tuple(test_ids[pos - ORDER + 1:pos])
        e = ctx_counts.get(ctx)
        pm = lpmix_te[k, y_te[k]]
        if e:
            tot = sum(e.values())
            c = e.get(int(y_te[k]), 0)
            if c > 0:
                lp_mem = np.log(c / tot)
                beta = beta_max * tot / (tot + k_beta)
            else:
                nll[k] = -pm
                continue
        else:
            nll[k] = -pm
            continue
        nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + lp_mem)
    return float(np.exp(np.mean(nll)))

print("  sparse β(c_h):")
for k_beta in [0.5, 1.0, 2.0, 5.0]:
    p = eval_mem(k_beta)
    print(f"    k={k_beta}: PPL {p:.3f}")

# KN baseline
# Full KN with memo
cnt = [defaultdict(int) for _ in range(ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1
cont = defaultdict(int)
for (x, w), c in cnt[2].items():
    if c > 0: cont[w] += 1
total_cont = sum(cont.values())
P_UNI = np.zeros(V, dtype=np.float64)
for w in range(V): P_UNI[w] = max(cont.get(w, 0) - 0.75, 0) / total_cont
P_UNI += (0.75 * len(cont)) / total_cont / V
P_UNI /= P_UNI.sum()
memo = {(): P_UNI}


def kn_dist(ctx):
    if ctx in memo: return memo[ctx]
    n = len(ctx) + 1
    c_h = cnt[n-1].get(ctx, 0)
    if c_h == 0:
        res = kn_dist(ctx[1:]); memo[ctx] = res; return res
    p = np.zeros(V, dtype=np.float64)
    n1 = 0; row = cnt[n]
    for w in range(V):
        c_hw = row.get(ctx + (w,), 0)
        if c_hw > 0: n1 += 1; p[w] = max(c_hw - 0.75, 0) / c_h
    p_lower = kn_dist(ctx[1:])
    p += (0.75 / c_h) * n1 * p_lower
    p /= p.sum()
    memo[ctx] = p; return p


nll_kn = np.zeros(N_EVAL)
for k in range(N_EVAL):
    i = te_starts[k]
    pos = i + W
    ctx = tuple(test_ids[pos - ORDER + 1:pos])
    pm = lpmix_te[k, y_te[k]]
    lp_mem = np.log(np.maximum(kn_dist(ctx), 1e-12))[y_te[k]]
    nll_kn[k] = -np.logaddexp(np.log1p(-0.9) + pm, np.log(0.9) + lp_mem)
print(f"  +KN β=0.9: PPL {np.exp(np.mean(nll_kn)):.3f}")

# comparison with d=64 results
print(f"\nComparison (d=64 from exp36: 8.59 sparse β(c_h), 10.94 KN, 32.78 mixer-only)")

# ============ generation ============
print("\n=== generation (d=256) ===")
def generate(seed, k_beta=1.0, length=120):
    model.eval()
    ids = tok.encode(seed).ids
    out = seed
    for _ in range(length):
        ctx = ids[-W:] if len(ids) >= W else ids
        fill = ctx[0] if ctx else 0
        full = [fill] * (W - len(ctx)) + ctx
        x = torch.tensor([full], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        lp_mix = torch.log_softmax(logits[0], -1).double().numpy()
        pm = np.exp(lp_mix)
        mem_ctx = tuple(ids[-ORDER+1:]) if len(ids) >= ORDER-1 else ()
        e = ctx_counts.get(mem_ctx)
        if e:
            c_h = sum(e.values())
            beta = c_h / (c_h + k_beta)
            pf = (1 - beta) * pm
            for w, c in e.items():
                pf[w] += beta * c / c_h
        else:
            pf = pm
        # sample
        p = np.maximum(pf, 1e-12) ** (1/0.8)
        p /= p.sum()
        idx = np.argsort(-p)
        cum = np.cumsum(p[idx])
        keep = idx[cum <= 0.9]
        if len(keep) == 0: keep = idx[:1]
        mask = np.zeros(V, dtype=bool); mask[keep] = True
        p[~mask] = 0; p /= p.sum()
        nxt = int(np.random.choice(V, p=p))
        ids.append(nxt)
        out += tok.decode([nxt])
    return out

for seed_name, seed in [("fn", "def fibonacci(n):"), ("api", "app.get('/users', async (req, res) =>")]:
    torch.manual_seed(0); np.random.seed(0)
    g = generate(seed, k_beta=1.0)
    print(f"\n--- {seed_name} ---\n{g}")

json.dump({"mixer_only": mixer_only, "n_params": sum(p.numel() for p in model.parameters())},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("saved", OUT)