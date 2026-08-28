"""exp34_joint_training.py — train the mixer TOGETHER with sparse MLE memory.

"А что если": the exp32 gap (sparse 28.28 vs KN 23.21) exists because the
mixer was trained WITHOUT memory — it never learned to cover sparse MLE's
blind spots (unseen continuations, no backoff). If we train the mixer with
the FUSED loss -log((1-β)·p_mix + β·p_sparse), it learns to compensate:
put mass where memory has none.

Compare (NL, WikiText-2):
  exp23 plain mixer + sparse MLE:  PPL 28.28  (baseline)
  exp34 joint-trained mixer + sparse MLE:  PPL ?
  exp34 joint-trained mixer + KN:           PPL ?  (generalization check)
"""
import os
import sys
import time
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp34_joint_training")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
ORDER = 3
MAX_TRAIN_CHARS = 4_000_000
BATCH = 64
TRAIN_STEPS = 6000
BETA_TRAIN = 0.5     # fusion weight during training


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_train.txt"), MAX_TRAIN_CHARS)
test_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_test.txt"))


def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok


print("BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids
print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}")

# sparse MLE memory: ctx(2) -> {w: count}
ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1
print(f"contexts: {len(ctx_counts):,}")


def sparse_logp_y(ctx, y):
    """log MLE probability of y given ctx (order-3), -inf if unseen."""
    e = ctx_counts.get(ctx)
    if e:
        c = e.get(y, 0)
        if c > 0:
            return np.log(c / sum(e.values()))
    return -1e9


# model (same as exp23)
class ChaoticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nw = W // W_LOCAL
        self.gates_l = nn.Parameter(torch.zeros(BLOCKS_LOCAL))
        self.gates_g = nn.Parameter(torch.zeros(BLOCKS_GLOBAL))
        self._sig_l = {t: torch.as_tensor(permute_indices(W_LOCAL, t), dtype=torch.long) for t in range(1, BLOCKS_LOCAL + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long) for t in range(1, BLOCKS_GLOBAL + 1)}

    def _chaotic(self, h, sigmas, gates):
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even, odd = h[:, 0::2, :], h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(h.shape[0], h.shape[1], D_MODEL)
        return h

    def forward(self, h):
        B, Wd, d = h.shape
        hw = h.view(B, self.Nw, W_LOCAL, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l) for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        return loc.reshape(B, Wd, d) + glob.mean(dim=1, keepdim=True)


class ChaoticBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, W, D_MODEL) * 0.02)
        self.block = ChaoticBlock()
        self.norm = nn.LayerNorm(D_MODEL)

    def mix(self, x):
        return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))


class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ChaoticBase()
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


model = ModelV1()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============ joint training ============
print(f"training with fused loss (β={BETA_TRAIN})...")
n = len(train_ids) - W - 1
t0 = time.time()
for s in range(TRAIN_STEPS):
    bi = np.random.randint(0, n, size=BATCH)
    X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
    Y = [train_ids[i + W] for i in bi]
    opt.zero_grad()
    logits = model(X)
    logp_mix = torch.log_softmax(logits, -1)  # (B, V)
    loss_sum = 0.0
    for b in range(BATCH):
        i = bi[b]
        ctx = tuple(train_ids[i + W - ORDER + 1:i + W])
        lp_mem = sparse_logp_y(ctx, Y[b])
        lp_mix_y = logp_mix[b, Y[b]]
        fused = torch.logaddexp(torch.log1p(-torch.tensor(BETA_TRAIN)) + lp_mix_y,
                                torch.log(torch.tensor(BETA_TRAIN)) + lp_mem)
        loss_sum += -fused
    loss = loss_sum / BATCH
    loss.backward()
    opt.step()
    if s % 1500 == 0:
        print(f"  [{s:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
torch.save(model.state_dict(), os.path.join(OUT, "joint_mixer.pt"))
print("trained + saved")

# ============ evaluate ============
model.eval()
N_EVAL = 12000
rng = np.random.default_rng(42)
maxstart_te = len(test_ids) - W - 1
te_starts = np.sort(rng.choice(maxstart_te, size=N_EVAL, replace=False))


def extract_logits(starts, seq):
    N = len(starts)
    logits_all = np.zeros((N, V), dtype=np.float32)
    ys = np.zeros(N, dtype=np.int64)
    with torch.no_grad():
        for s0 in range(0, N, 1024):
            e = min(s0 + 1024, N)
            X = np.zeros((e - s0, W), dtype=np.int64)
            for k, i in enumerate(starts[s0:e]):
                X[k] = seq[i:i + W]
                ys[s0 + k] = seq[i + W]
            logits_all[s0:e] = model(torch.tensor(X, dtype=torch.long)).numpy()
    return logits_all, ys


logits_te, y_te = extract_logits(te_starts, test_ids)
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()

mixer_only = float(np.exp(np.mean([-lpmix_te[k, y_te[k]] for k in range(N_EVAL)])))
print(f"\nmixer-only (joint-trained): PPL {mixer_only:.3f}")

# fused with sparse MLE / KN at various β
# KN table
cnt = [defaultdict(int) for _ in range(ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1
cont = defaultdict(int)
for (x, w), c in cnt[2].items():
    if c > 0:
        cont[w] += 1
total_cont = sum(cont.values())
P_UNI = np.zeros(V, dtype=np.float64)
for w in range(V):
    cw = cont.get(w, 0)
    P_UNI[w] = max(cw - 0.75, 0) / total_cont
P_UNI += (0.75 * len(cont)) / total_cont / V
P_UNI /= P_UNI.sum()
memo = {(): P_UNI}


def kn_dist(ctx):
    if ctx in memo:
        return memo[ctx]
    n = len(ctx) + 1
    c_h = cnt[n - 1].get(ctx, 0)
    if c_h == 0:
        res = kn_dist(ctx[1:])
        memo[ctx] = res
        return res
    p = np.zeros(V, dtype=np.float64)
    n1 = 0
    row = cnt[n]
    for w in range(V):
        c_hw = row.get(ctx + (w,), 0)
        if c_hw > 0:
            n1 += 1
            p[w] = max(c_hw - 0.75, 0) / c_h
    p_lower = kn_dist(ctx[1:])
    p += (0.75 / c_h) * n1 * p_lower
    p /= p.sum()
    memo[ctx] = p
    return p


for kind, lp_fn in [("sparse", sparse_logp_y),
                    ("kn", lambda ctx, y: np.log(np.maximum(kn_dist(ctx), 1e-12))[y])]:
    for beta in [0.3, 0.5, 0.7, 0.9]:
        nll = np.zeros(N_EVAL)
        for k in range(N_EVAL):
            tpos = te_starts[k] + W
            ctx = tuple(test_ids[tpos - ORDER + 1:tpos])
            fused = np.logaddexp(np.log1p(-beta) + lpmix_te[k, y_te[k]],
                                 np.log(beta) + lp_fn(ctx, y_te[k]))
            nll[k] = -fused
        print(f"  +{kind} β={beta}: PPL {np.exp(np.mean(nll)):.3f}")

json.dump({"mixer_only": mixer_only, "train_beta": BETA_TRAIN},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("saved", OUT)
