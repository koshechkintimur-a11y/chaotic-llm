"""exp_eye_main.py — main Eye experiment: 4 systems, sweep, noise test, cost.

Методология: сравниваем Attention, Chaotic, Chaotic+Eye, RandomEye
на одинаковых данных (корпус кода, BPE-512, d=128, 6K шагов).
После — шум/retrieval тест, FLOPs, визуализация.
"""
import os
import sys
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent          # exp_eye_selector/
PHASE = HERE.parent                   # phase01/ (корпус здесь)
OUT = HERE / "exp_eye_selector"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PHASE))
sys.path.insert(0, str(PHASE.parent))  # repo root with chaos_lib

from chaotic_gears import ChaoticBlock
from eye import SelectiveChaoticLM, Eye

torch.manual_seed(0)
np.random.seed(0)

# ============ config ============
VOCAB = 512
W = 256
D = 128
ORDER = 3
MAX_TRAIN = 2_000_000
BATCH = 64
STEPS = 6000
LR = 1e-3
N_EVAL = 6000
EVAL_BATCH = 128
# Eye defaults
EYE_VARIANT = "C"
EYE_MODE = "soft"
EYE_T = 1.0
EYE_K = max(2, W // 6)

def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()

train_text = load_chars(PHASE / "corpus_train.txt", MAX_TRAIN)
test_text = load_chars(PHASE / "corpus_test.txt")

def make_bpe(text, vs=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vs, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i+100000] for i in range(0, len(text), 100000)], trainer=tr)
    return tok

print("BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids

# ctx counts for sparse MLE eval
ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============ model builders ============
def build_attention(d=D, heads=4, layers=6, W=W):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
            mask = torch.triu(torch.full((W, W), float("-inf")), diagonal=1)
            self.register_buffer("mask", mask)
        def forward(self, x):
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self.mask)
            x = x + a
            return x + self.ffn(self.ln2(x))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.pos = nn.Parameter(torch.randn(1, W, d)*0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.ln_f = nn.LayerNorm(d)
            self.head = nn.Linear(d, V)
        def forward(self, x):
            h = self.embed(x) + self.pos
            for b in self.blocks:
                h = b(h)
            return self.head(self.ln_f(h[:, -1, :]))
    return Model()

def build_chaotic(d=D, blocks=8, bg=4, W=W):
    block = ChaoticBlock(W, W//4, d, blocks, bg)
    class ChaoticBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.pos = nn.Parameter(torch.randn(1, W, d)*0.02)
            self.block = block
            self.norm = nn.LayerNorm(d)
        def mix(self, x):
            return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = ChaoticBase()
            self.readout = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d, V))
        def forward(self, x):
            h = self.base.mix(x)
            gvec = h.mean(dim=1)
            return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))
    return Model()

def build_eye(d=D, variant="C", mode="soft", T=1.0, k=EYE_K, blocks=8, bg=4, W=W):
    block = ChaoticBlock(W, W//4, d, blocks, bg)
    class ChaoticBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.pos = nn.Parameter(torch.randn(1, W, d)*0.02)
            self.block = block
            self.norm = nn.LayerNorm(d)
        def mix(self, x):
            return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))
    return SelectiveChaoticLM(ChaoticBase(), V, d, variant, mode, T, k)

# ============ training ============
def train_model(model, name, steps=STEPS, lr=LR, entropy_lambda=0, verbose=True):
    params = sum(p.numel() for p in model.parameters())
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    t0 = time.time()
    losses = []
    for s in range(steps):
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i+W] for i in bi]), dtype=torch.long, device=DEVICE)
        Y = torch.tensor([train_ids[i+W] for i in bi], dtype=torch.long, device=DEVICE)
        opt.zero_grad()
        if isinstance(model, SelectiveChaoticLM):
            logits, w = model(X)
        else:
            logits = model(X)
        loss = lossf(logits, Y)
        if entropy_lambda > 0 and isinstance(model, SelectiveChaoticLM):
            ent = model.eye.entropy(model.base.mix(X))
            loss = loss - entropy_lambda * ent
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 2000 == 0 and verbose:
            print(f"  [{s}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    return params

# ============ evaluation ============
def eval_model(model, name):
    model.eval()
    rng = np.random.default_rng(42)
    maxstart = len(test_ids) - W - 1
    te_starts = np.sort(rng.choice(maxstart, size=N_EVAL, replace=False))
    logits_te = np.zeros((N_EVAL, V), dtype=np.float32)
    y_te = np.zeros(N_EVAL, dtype=np.int64)
    with torch.no_grad():
        for s0 in range(0, N_EVAL, EVAL_BATCH):
            e = min(s0+EVAL_BATCH, N_EVAL)
            X = np.zeros((e-s0, W), dtype=np.int64)
            for k,i in enumerate(te_starts[s0:e]):
                X[k] = test_ids[i:i+W]
                y_te[s0+k] = test_ids[i+W]
            if isinstance(model, SelectiveChaoticLM):
                logits, _ = model(torch.tensor(X, dtype=torch.long, device=DEVICE))
            else:
                logits = model(torch.tensor(X, dtype=torch.long, device=DEVICE))
            logits_te[s0:e] = logits.cpu().numpy()
    lpmix = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()
    mixer_only = float(np.exp(np.mean([-lpmix[k, y_te[k]] for k in range(N_EVAL)])))
    gated = {}
    for kb in [0.5, 1.0, 2.0]:
        nll = np.zeros(N_EVAL)
        for k in range(N_EVAL):
            i = te_starts[k]
            ctx = tuple(test_ids[i+W-ORDER+1:i+W])
            e = ctx_counts.get(ctx)
            pm = lpmix[k, y_te[k]]
            if e:
                tot = sum(e.values())
                c = e.get(int(y_te[k]), 0)
                if c > 0:
                    beta = tot/(tot+kb)
                    nll[k] = -np.logaddexp(np.log1p(-beta)+pm, np.log(beta)+np.log(c/tot))
                    continue
            nll[k] = -pm
        gated[kb] = float(np.exp(np.mean(nll)))
    return {"mixer_only": mixer_only, "gated": gated}

# ============ main comparison ============
results = {}
print("\n=== TRAINING ATTENTION ===")
m_att = build_attention()
p = train_model(m_att, "attention")
results["attention"] = {"params": p}
results["attention"].update(eval_model(m_att, "attention"))
print(f"  attention: {results['attention']}")

print("\n=== TRAINING CHAOTIC ===")
m_chaos = build_chaotic()
p = train_model(m_chaos, "chaotic")
results["chaotic"] = {"params": p}
results["chaotic"].update(eval_model(m_chaos, "chaotic"))
print(f"  chaotic: {results['chaotic']}")

print("\n=== TRAINING CHAOTIC+EYE (C, soft, T=1) ===")
m_eye = build_eye()
p = train_model(m_eye, "chaotic+eye")
results["chaotic+eye"] = {"params": p}
results["chaotic+eye"].update(eval_model(m_eye, "chaotic+eye"))
print(f"  chaotic+eye: {results['chaotic+eye']}")

# Random eye baseline
print("\n=== TRAINING CHAOTIC+RANDOM EYE ===")
m_rand = build_eye(variant="C", mode="soft")
m_rand.eye.global_lin.weight.requires_grad = False
m_rand.eye.score_proj.weight.requires_grad = False
if hasattr(m_rand.eye, 'local_conv'):
    m_rand.eye.local_conv.weight.requires_grad = False
p = train_model(m_rand, "chaotic+randomeye")
results["chaotic+randomeye"] = {"params": p}
results["chaotic+randomeye"].update(eval_model(m_rand, "chaotic+randomeye"))
print(f"  chaotic+randomeye: {results['chaotic+randomeye']}")

json.dump(results, open(OUT / "main_results.json", "w"), indent=2)
print("\nmain results saved")

# ============ Eye hyperparameter sweep ============
print("\n=== EYE SWEEP (d=128, 6K steps) ===")
sweep = []
for variant in ["A", "B", "C"]:
    for mode in ["soft", "topk", "hard"]:
        for T in [1.0, 0.5]:
            try:
                m = build_eye(variant=variant, mode=mode, T=T)
                p = train_model(m, f"eye-{variant}-{mode}-T{T}", steps=3000, verbose=False)
                r = eval_model(m, "")
                print(f"  {variant} {mode} T={T}: mixer={r['mixer_only']:.2f}", flush=True)
                sweep.append({"variant": variant, "mode": mode, "T": T, **r})
            except Exception as e:
                print(f"  {variant} {mode} T={T}: FAILED {e}", flush=True)
                sweep.append({"variant": variant, "mode": mode, "T": T, "error": str(e)})
json.dump(sweep, open(OUT / "sweep.json", "w"), indent=2)
print("sweep saved")

# ============ noise retrieval test (TRAIN on task, variable KEY) ============
print("\n=== NOISE RETRIEVAL TEST (train on task) ===")
# Task: [noise ... KEY ... noise] -> next token = KEY's VALUE.
# KEY value K ~ rand[100..511], placed at random position in [5, W-5].
# Model must retrieve K (content-addressing), NOT a constant.
# Train each architecture fresh on this task; measure final accuracy + the
# accuracy-vs-W scaling (noise fraction grows with W).

def make_noise_batch(n, W, noise_frac=0.95):
    """Batch of (seq, target): seq has a KEY at random pos with random value."""
    X = np.zeros((n, W), dtype=np.int64)
    Y = np.zeros(n, dtype=np.int64)
    n_key = max(1, int(W * (1 - noise_frac)))  # useful tokens
    for b in range(n):
        keys = np.random.randint(100, 512, size=n_key)
        pos = np.random.choice(W, size=n_key, replace=False)
        X[b] = np.random.randint(100, 512, size=W)
        X[b, pos] = keys
        Y[b] = keys[0]  # retrieve the FIRST key's value
    return X, Y


def train_noise_model(model, name, W, noise_frac, steps=4000, n=64):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    accs = []
    for s in range(steps):
        Xb, Yb = make_noise_batch(n, W, noise_frac)
        X = torch.tensor(Xb, dtype=torch.long, device=DEVICE)
        Y = torch.tensor(Yb, dtype=torch.long, device=DEVICE)
        opt.zero_grad()
        if isinstance(model, SelectiveChaoticLM):
            logits, _ = model(X)
        else:
            logits = model(X)
        loss = lossf(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 1000 == 0:
            acc = (logits.argmax(-1) == Y).float().mean().item()
            accs.append(acc)
    # final eval
    Xe, Ye = make_noise_batch(1000, W, noise_frac)
    with torch.no_grad():
        if isinstance(model, SelectiveChaoticLM):
            logits, _ = model(torch.tensor(Xe, dtype=torch.long, device=DEVICE))
        else:
            logits = model(torch.tensor(Xe, dtype=torch.long, device=DEVICE))
    final_acc = (logits.argmax(-1) == torch.tensor(Ye, device=DEVICE)).float().mean().item()
    return final_acc, accs


def noise_scaling(builder, name, noise_frac, Ws=[64, 128, 256]):
    """Retrieval accuracy at growing W (growing noise fraction)."""
    results = {}
    for W in Ws:
        model = builder(W=W)  # rebuild with the right context length
        acc, _ = train_noise_model(model, name, W, noise_frac, steps=3000)
        results[W] = round(acc * 100, 1)
        print(f"  {name} W={W} noise={noise_frac*100:.0f}%: retrieval acc {results[W]:.1f}%", flush=True)
    return results


noise = {}
for name, builder in [("attention", lambda W=256: build_attention(W=W)),
                      ("chaotic", lambda W=256: build_chaotic(W=W)),
                      ("chaotic+eye", lambda W=256: build_eye(W=W, variant="C", mode="soft", T=1.0))]:
    print(f"\n noise test: {name}")
    noise[name] = {}
    for nf in [0.95, 0.99]:
        m = builder()
        noise[name][f"noise_{nf}"] = noise_scaling(m, name, nf, Ws=[64, 128, 256])
json.dump(noise, open(OUT / "noise_test.json", "w"), indent=2)
print("noise test saved")

# ============ FLOPs estimate ============
print("\n=== FLOPs ESTIMATE ===")
def flops_chaotic(W, d, blocks=8, bg=4):
    Wl = W // 4
    # per block: local (Wl*bl*d*2) + global (1*bg*d*2) → per token
    local = Wl * blocks * d * 2.0
    global_ = 1.0 * bg * d * 2.0
    per_block = (local + global_) * 1.0 / W  # per token
    return per_block * (blocks + bg)

def flops_attention(W, d):
    return 2.0 * W * d * 2.0  # QK (W²*d) simplified per token

def flops_eye(W, d):
    return 2.0 * d * 2.0  # ~2 multiply-adds per dimension

flops = {}
for w in [256, 512, 1024, 2048]:
    flops[w] = {"chaotic": flops_chaotic(w, D),
                "attention": flops_attention(w, D),
                "eye": flops_eye(w, D)}
json.dump(flops, open(OUT / "flops.json", "w"), indent=2)
print(f"flops saved: {flops}")

print("\n=== DONE ===")