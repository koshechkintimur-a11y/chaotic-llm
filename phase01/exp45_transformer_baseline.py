"""exp45_transformer_baseline.py — 1.2M Transformer CONTROL.

The decisive control: same corpus, same protocol (lr 5e-4, warmup 1000,
cosine 16K, batch 64, clip 1.0), ~1.2M params — but standard causal
Transformer instead of the chaotic mixer.

- If Transformer converges (train loss ~2.0-2.5): the mixer's 4.01 at
  1.2M is an ARCHITECTURAL ceiling. Problem is in the mixer.
- If Transformer also plateaus (~4+): problem is data/optimization.

Architecture: tiny GPT — d=128, 4 heads, 6 blocks, learned pos, causal
mask. Params ~1.35M (same scale as mixer 1.18M).
"""
import os
import sys
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp45_transformer_baseline")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
D = 128
HEADS = 4
LAYERS = 6
ORDER = 3
MAX_TRAIN_BYTES = 2_000_000
BATCH = 64
TRAIN_STEPS = 16000
LR = 5e-4
WARMUP = 1000
N_EVAL = 12000
EVAL_BATCH = 256


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

ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1


class Block(nn.Module):
    def __init__(self, d, heads, W):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        mask = torch.triu(torch.full((W, W), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self.mask)
        x = x + a
        return x + self.ffn(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, V, W, d, heads, layers):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, W) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        h = self.embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h[:, -1, :])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(V, W, D, HEADS, LAYERS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Transformer params: {n_params:,}  device: {DEVICE}")

opt = torch.optim.Adam(model.parameters(), lr=LR)
def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    p = (step - WARMUP) / max(1, TRAIN_STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * p))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
lossf = nn.CrossEntropyLoss()

n = len(train_ids) - W - 1
t0 = time.time()
print("training (16K, lr 5e-4, warmup+cosine)...")
for s in range(TRAIN_STEPS):
    bi = np.random.randint(0, n, size=BATCH)
    X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long, device=DEVICE)
    Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long, device=DEVICE)
    opt.zero_grad()
    loss = lossf(model(X), Y)
    loss.backward()
    if s % 2000 == 0:
        g_total = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
        print(f"  [{s:,}] loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e} |g|={g_total:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if s % 4000 == 0 and s > 0:
        torch.save(model.state_dict(), os.path.join(OUT, f"ckpt_{s}.pt"))

torch.save(model.state_dict(), os.path.join(OUT, "transformer_d128.pt"))
print(f"trained + saved ({time.time()-t0:.0f}s)")

# ============ eval (control: transformer alone; + sparse β(c_h)) ============
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
        logits_te[s0:e] = model(torch.tensor(X, dtype=torch.long, device=DEVICE)).cpu().numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()
mixer_only = float(np.exp(np.mean([-lpmix_te[k, y_te[k]] for k in range(N_EVAL)])))
print(f"\nCONTROL (transformer alone): PPL {mixer_only:.3f}")
print(f"  [mixer 1.2M: 97.4 | mixer 460K: 16.73 | mixer 90K: 32.78]")


def eval_gated(k_beta):
    nll = np.zeros(N_EVAL)
    for k in range(N_EVAL):
        i = te_starts[k]
        pos = i + W
        ctx = tuple(test_ids[pos - ORDER + 1:pos])
        e = ctx_counts.get(ctx)
        pm = lpmix_te[k, y_te[k]]
        if e:
            tot = sum(e.values())
            c = e.get(int(y_te[k]), 0)
            if c > 0:
                beta = tot / (tot + k_beta)
                nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + np.log(c / tot))
                continue
        nll[k] = -pm
    return float(np.exp(np.mean(nll)))


print("  +sparse β(c_h):")
gated = {}
for k_beta in [0.5, 1.0, 2.0]:
    p = eval_gated(k_beta)
    gated[k_beta] = p
    print(f"    k={k_beta}: PPL {p:.3f}")

json.dump({"params": n_params, "mixer_only": mixer_only, "gated": gated},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("saved", OUT)
