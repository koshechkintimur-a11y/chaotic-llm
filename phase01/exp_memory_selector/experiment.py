"""exp_memory_selector/experiment.py — memory as content selector (fail-safe MEM).

M1: Bidirectional Mixer unchanged (exp52-style, blocks=BLOCKS).
M2: memory is a lookup table built on TRAIN, used on test (no gradients).
M3: O(1) lookup, O(N) build, <=100MB footprint.
M4: controls MEM-R (random), MEM-NoMixer (no mixer).

Usage:  python experiment.py MEM0|MEM_A|MEM_B|MEM_C|MEM_R|MEM_NM [steps]
"""
import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, REPO)

# ----------------------------------------------------------------- config
VOCAB = 512
W = 256
D = 128
BLOCKS = 8
BATCH = 64
STEPS = 8000
LR = 5e-4
WARMUP = 1000
MAX_TRAIN = 990_000
ORDER = 3                      # baseline n-gram order (MEM-0)
KB = 1.0                      # adaptive-beta strength (exp09-style)
N_VAL = 2000
N_EVAL = 5000
EVAL_BATCH = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from chaotic_gears import ChaoticBlock
from chaos_lib import permute_indices

# ----------------------------------------------------------------- data helpers
def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()

def make_bpe(text, vs=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vs, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i+100000] for i in range(0, len(text), 100000)], trainer=tr)
    return tok

def build_tokenizer():
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    return make_bpe(train_text)

def seeded_block(W, Wl, d, bl, bg, seed):
    block = ChaoticBlock(W, Wl, d, bl, bg)
    block._sig_l = {t: torch.as_tensor(permute_indices(Wl, t + seed * (bl + 3)),
                                       dtype=torch.long) for t in range(1, bl + 1)}
    block._sig_g = {t: torch.as_tensor(permute_indices(block.Nw, t + seed * (bg + 3)),
                                       dtype=torch.long) for t in range(1, bg + 1)}
    return block

class BidirectionalMixer(nn.Module):
    """exp52-style bidirectional mixer with a perm seed."""
    def __init__(self, seed=0, blocks=BLOCKS):
        super().__init__()
        Wl = W // 4
        bl, bg = blocks, max(1, blocks // 2)
        self.fwd = seeded_block(W, Wl, D, bl, bg, seed * 2)
        self.bwd = seeded_block(W, Wl, D, bl, bg, seed * 2 + 1)
        self.proj = nn.Linear(2 * D, D)

    def forward(self, x):
        xf = self.fwd(x)
        xb = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.proj(torch.cat([xf, xb], dim=-1))

from memory import (BaseMemory, Order8Memory, ContentAddressableMemory,
                    HybridMemory, RandomMemory)

# ----------------------------------------------------------------- model
class LMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.pos = nn.Parameter(torch.randn(1, W, D) * 0.02)
        self.readout = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, VOCAB))

    def head(self, h):
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))

    def embed_pos(self, x):
        return self.embed(x) + self.pos


class MemMixerLM(nn.Module):
    """Tokens -> BidirectionalMixer -> Memory (selector) -> Output. (M1)"""
    def __init__(self, use_mixer=True):
        super().__init__()
        self.head = LMHead()
        self.use_mixer = use_mixer
        if use_mixer:
            self.mixer = BidirectionalMixer(seed=0)

    def forward(self, x):
        e = self.head.embed_pos(x)
        if self.use_mixer:
            h = self.mixer(e)
        else:
            h = e                      # no mixer: identity (memory-only ablation)
        return self.head.head(h)


# ----------------------------------------------------------------- data
print("BPE...", flush=True)
tok = build_tokenizer()
V = tok.get_vocab_size()
train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids

# Build memory on TRAIN (M2: lookup, no gradients)
def build_memory(kind):
    if kind == "MEM0":
        return BaseMemory(order=ORDER, min_count=2).build(train_ids)
    if kind == "MEM_A":
        return Order8Memory(order=8, min_count=3).build(train_ids)
    if kind == "MEM_B":
        return ContentAddressableMemory(min_len=8, max_len=16, min_count=2).build(train_ids)
    if kind == "MEM_C":
        return HybridMemory(local_order=3, min_len=8, max_len=16, alpha=0.5).build(train_ids)
    if kind == "MEM_R":
        return RandomMemory(order=8, min_count=3).build(train_ids)
    return None   # MEM_NM uses no memory in forward (identity), eval uses order-3 prior anyway


# ----------------------------------------------------------------- train
def train(model, name):
    params = sum(p.numel() for p in model.parameters())
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    def lr_lambda(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        p = (step - WARMUP) / max(1, STEPS - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    rng = np.random.default_rng(1)
    maxstart_v = len(train_ids) - W - 1
    val_starts = np.sort(rng.choice(maxstart_v, size=N_VAL, replace=False))
    t0 = time.time()
    gap_log = []
    for s in range(STEPS):
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i+W] for i in bi]), dtype=torch.long, device=DEVICE)
        Y = torch.tensor([train_ids[i+W] for i in bi], dtype=torch.long, device=DEVICE)
        opt.zero_grad()
        logits = model(X)
        loss = lossf(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 2000 == 0:
            tr_loss = loss.item()
            model.eval()
            with torch.no_grad():
                vnll = 0.0
                for s0 in range(0, N_VAL, EVAL_BATCH):
                    e = min(s0 + EVAL_BATCH, N_VAL)
                    Xv = np.zeros((e - s0, W), dtype=np.int64)
                    Yv = np.zeros(e - s0, dtype=np.int64)
                    for kk, i in enumerate(val_starts[s0:e]):
                        Xv[kk] = train_ids[i:i+W]; Yv[kk] = train_ids[i+W]
                    lo = model(torch.tensor(Xv, dtype=torch.long, device=DEVICE))
                    vnll += float(lossf(lo, torch.tensor(Yv, dtype=torch.long, device=DEVICE)).item()) * (e - s0)
                val_loss = vnll / N_VAL
            model.train()
            gap = tr_loss - val_loss
            gap_log.append({"step": s, "train": tr_loss, "val": val_loss, "gap": gap})
            print(f"[{name}] [{s}] tr={tr_loss:.3f} val={val_loss:.3f} gap={gap:+.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
    return params, gap_log


def eval_model(model, memory, name):
    model.eval()
    rng = np.random.default_rng(42)
    maxstart = len(test_ids) - W - 1
    te_starts = np.sort(rng.choice(maxstart, size=N_EVAL, replace=False))
    logits_te = np.zeros((N_EVAL, V), dtype=np.float32)
    y_te = np.zeros(N_EVAL, dtype=np.int64)
    with torch.no_grad():
        for s0 in range(0, N_EVAL, EVAL_BATCH):
            e = min(s0 + EVAL_BATCH, N_EVAL)
            X = np.zeros((e - s0, W), dtype=np.int64)
            for k, i in enumerate(te_starts[s0:e]):
                X[k] = test_ids[i:i+W]; y_te[s0+k] = test_ids[i+W]
            logits_te[s0:e] = model(torch.tensor(X, dtype=torch.long, device=DEVICE)).cpu().numpy()
    lpmix = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()

    # mixer-only PPL (no memory)
    mixer_only = float(np.exp(np.mean([-lpmix[k, y_te[k]] for k in range(N_EVAL)])))

    # memory-gated PPL (adaptive beta, exp09-style)
    ctx_len = memory.ctx_len if memory is not None else ORDER
    gated = {}
    for kb in [0.5, 1.0, 2.0]:
        nll = np.zeros(N_EVAL)
        miss = 0
        for k in range(N_EVAL):
            i = te_starts[k]
            ctx = tuple(test_ids[i+W-ctx_len:i+W])
            pm = lpmix[k, y_te[k]]
            if memory is not None:
                r = memory.query(ctx)
                if r is not None:
                    counts, tot = r
                    c = counts.get(int(y_te[k]), 0)
                    if c > 0:
                        beta = tot / (tot + kb)
                        nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + np.log(c/tot))
                        continue
            miss += 1
            nll[k] = -pm
        gated[kb] = float(np.exp(np.mean(nll)))
    return {"mixer_only": mixer_only, "gated": gated, "miss_rate": miss / N_EVAL}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["MEM0", "MEM_A", "MEM_B", "MEM_C", "MEM_R", "MEM_NM"])
    ap.add_argument("steps", nargs="?", type=int, default=STEPS)
    args = ap.parse_args()

    use_mixer = (args.config != "MEM_NM")
    model = MemMixerLM(use_mixer=use_mixer)
    memory = build_memory(args.config)

    params, gap_log = train(model, args.config)
    res = eval_model(model, memory, args.config)
    res.update({"params": params, "config": args.config, "gap_log": gap_log,
                "mem_entries": memory.n_entries() if memory else 0,
                "mem_bytes": memory.size_bytes() if memory else 0})
    if memory is not None:
        print(f"[{args.config}] mem_entries={memory.n_entries():,} bytes={memory.size_bytes():,}")
    print(f"[{args.config}] params={params:,} mixer_only={res['mixer_only']:.3f} "
          f"gated(k=1)={res['gated'][1.0]:.3f} miss={res['miss_rate']:.3f}", flush=True)
    json.dump(res, open(os.path.join(HERE, f"results_{args.config}.json"), "w"), indent=2)
