"""exp56_parallel_eye.py — ParallelEye: R routes computed fully, Eye weights.

Ablations (ТЗ): A=uniform 1/R, B=random frozen, C=learned (Око).
Usage: python exp56_parallel_eye.py A | B | C
"""
import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))   # phase01/
REPO = os.path.dirname(HERE)                          # chaotic-llm/ (chaos_lib)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from chaotic_gears import ChaoticBlock

VOCAB = 512
W = 256
D = 128
R = 4
BLOCKS = 8
BATCH = 64
STEPS = 8000
LR = 5e-4
WARMUP = 500
N_EVAL = 4000
EVAL_BATCH = 128
ORDER = 3
MAX_TRAIN = 2_000_000


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))


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

ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_block(W, d, blocks, seed=0):
    """ChaoticBlock with permutation offset by seed (same Arnold map)."""
    Wl = W // 4
    bl, bg = blocks, blocks // 2
    block = ChaoticBlock(W, Wl, d, bl, bg)
    # offset the local permutations by seed
    if seed > 0:
        from chaos_lib import permute_indices
        block._sig_l = {t: torch.as_tensor(permute_indices(Wl, t + seed * (bl + 3)),
                                           dtype=torch.long) for t in range(1, bl + 1)}
        block._sig_g = {t: torch.as_tensor(permute_indices(block.Nw, t + seed * (bg + 3)),
                                           dtype=torch.long) for t in range(1, bg + 1)}
    return block


class RouteMixer(nn.Module):
    """One route: just a chaotic block on shared embedding (no own embed)."""
    def __init__(self, seed=0):
        super().__init__()
        Wl = W // 4
        bl, bg = BLOCKS, BLOCKS // 2
        self.block = ChaoticBlock(W, Wl, D, bl, bg)
        if seed > 0:
            from chaos_lib import permute_indices
            self.block._sig_l = {t: torch.as_tensor(permute_indices(Wl, t + seed * (bl + 3)),
                                                    dtype=torch.long) for t in range(1, bl + 1)}
            self.block._sig_g = {t: torch.as_tensor(permute_indices(self.block.Nw, t + seed * (bg + 3)),
                                                    dtype=torch.long) for t in range(1, bg + 1)}
        self.norm = nn.LayerNorm(D)

    def forward(self, e):
        return self.norm(e + self.block(e))


class ParallelEyeLM(nn.Module):
    """One shared embed, R routes (different chaotic permutations), Eye weights."""
    def __init__(self, mode="C"):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, W, D) * 0.02)
        self.routes = nn.ModuleList([RouteMixer(seed=i) for i in range(R)])
        self.mode = mode
        self.eye = nn.Sequential(nn.Linear(D, D // 4), nn.ReLU(), nn.Linear(D // 4, R))
        if mode == "B":
            self.eye.requires_grad_(False)
            for p in self.eye.parameters():
                torch.nn.init.normal_(p, 0, 0.5)
        self.readout = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, V))
        self._route_w = None

    def forward(self, x):
        e = self.embed(x) + self.pos                        # shared embedding once
        outs = torch.stack([r(e) for r in self.routes], dim=-2)  # (B,W,R,D)
        if self.mode == "A":
            w = torch.full((outs.shape[0], outs.shape[1], R), 1.0 / R, device=outs.device)
        elif self.mode == "B":
            w = torch.softmax(self.eye(x), dim=-1)
        else:
            w = torch.softmax(self.eye(x), dim=-1)
        self._route_w = w
        h = (outs * w.unsqueeze(-1)).sum(-2)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))

    def route_entropy(self):
        if self._route_w is None:
            return torch.tensor(0.0)
        return -(self._route_w * torch.log(self._route_w.clamp_min(1e-9))).sum(-1).mean()


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
    t0 = time.time()
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
        if s % 4000 == 0:
            ent = model.route_entropy().item() if hasattr(model, 'route_entropy') else 0
            print(f"[{name}] [{s}] loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e} "
                  f"ent={ent:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return params


def eval_model(model, name):
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
                X[k] = test_ids[i:i+W]
                y_te[s0+k] = test_ids[i+W]
            logits_te[s0:e] = model(torch.tensor(X, dtype=torch.long, device=DEVICE)).cpu().numpy()
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
                    beta = tot / (tot + kb)
                    nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + np.log(c/tot))
                    continue
            nll[k] = -pm
        gated[kb] = float(np.exp(np.mean(nll)))
    return {"mixer_only": mixer_only, "gated": gated}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["A", "B", "C"])
    args = ap.parse_args()

    name = f"exp56-{args.mode}"
    model = ParallelEyeLM(mode=args.mode)
    params = train(model, name)
    res = eval_model(model, name)
    res["params"] = params
    res["mode"] = args.mode
    print(f"[{name}] params={params:,} mixer_only={res['mixer_only']:.3f} "
          f"gated(k=1)={res['gated'][1.0]:.3f}", flush=True)

    out_dir = os.path.join(HERE, "exp56_parallel_eye")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{args.mode}.json")
    json.dump(res, open(path, "w"), indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, f"model_{args.mode}.pt"))
    print("saved", path)
