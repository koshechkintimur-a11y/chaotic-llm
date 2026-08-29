"""exp_eye_failsafe/experiment.py — Corrected "Eye" (fail-safe).

Constitution (И1-И4):
  И1 full coverage always — Eye is a parallel branch (weights/residual), never gates
  И2 Eye starts as baseline — zero-init gates / uniform weights at step 0
  И3 no throwing away — convex weights with eps-floor, residual additions only
  И4 cheap & observable — <=2% params, uniform/random controls, train-eval gap

Forms:
  E1 EyeModulator: R routes all computed, Eye weights over them (eps-floor)
  E2 EyeGroupResidual: base mixing always + residual group mixing (gated from 0)

Modes per form: u=uniform(eps=1), r=random frozen, l=learned(eps=0.25).
Usage: python experiment.py E1 u | E1 r | E1 l | E2 r | E2 l
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

HERE = os.path.dirname(os.path.abspath(__file__))   # exp_eye_failsafe/
PHASE = os.path.dirname(HERE)                        # phase01/
REPO = os.path.dirname(PHASE)                        # chaotic-llm/
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, REPO)

from chaotic_gears import ChaoticBlock

# ---------------- config ----------------
VOCAB = 512
W = 256
D = 128
R = 4          # E1 routes
K = 4          # E2 groups
BLOCKS = 4     # per-direction blocks in each bidirectional mixer
EPS = 0.25     # eps-floor for learned Eye
BATCH = 64
STEPS = 8000
LR = 5e-4
WARMUP = 500
N_EVAL = 3000
EVAL_BATCH = 128
ORDER = 3
MAX_TRAIN = 2_000_000
N_VAL = 2000    # train-eval gap probe


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
    tok = make_bpe(train_text)
    return tok


def seeded_block(W, Wl, d, bl, bg, seed):
    """ChaoticBlock with permutation offset by seed (same Arnold map)."""
    from chaos_lib import permute_indices
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


# ---------------- E0: plain baseline (no eye) ----------------
class BaselineLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = LMHead()
        self.mixer = BidirectionalMixer(seed=0)

    def forward(self, x):
        e = self.head.embed_pos(x)
        h = self.mixer(e)
        return self.head.head(h)


# ---------------- E1: EyeModulator ----------------
class EyeModulatorLM(nn.Module):
    def __init__(self, mode="l"):
        super().__init__()
        self.head = LMHead()
        self.routes = nn.ModuleList([BidirectionalMixer(seed=i) for i in range(R)])
        self.mode = mode
        self.eye = nn.Sequential(nn.Linear(D, D // 8), nn.ReLU(), nn.Linear(D // 8, R))
        if mode == "u":
            self.eps = 1.0                     # uniform control (nested baseline)
        else:
            self.eps = EPS
            if mode == "l":                    # И2: zero-init → starts as baseline
                nn.init.zeros_(self.eye[-1].weight)
                nn.init.zeros_(self.eye[-1].bias)
            else:                              # r: random frozen
                nn.init.normal_(self.eye[-1].weight, 0, 0.3)
                nn.init.normal_(self.eye[-1].bias, 0, 0.3)
                self.eye.requires_grad_(False)
        self._w = None

    def forward(self, x):
        e = self.head.embed_pos(x)
        outs = torch.stack([r(e) for r in self.routes], dim=-2)   # (B,W,R,D)
        if self.mode == "u":
            w = torch.full((outs.shape[0], outs.shape[1], R), 1.0 / R, device=outs.device)
        else:
            logits = self.eye(e)
            w = (1 - self.eps) * torch.softmax(logits, dim=-1) + self.eps / R   # И3 floor
        self._w = w
        h = (outs * w.unsqueeze(-1)).sum(-2)
        return self.head.head(h)

    def route_entropy(self):
        if self._w is None:
            return torch.tensor(0.0)
        return -(self._w * torch.log(self._w.clamp_min(1e-9))).sum(-1).mean()


# ---------------- E2: EyeGroupResidual ----------------
def top1_with_capacity(logits, C):
    """Assign each token to a cluster, capacity C per cluster (Switch-style)."""
    B, W, K = logits.shape
    assign = torch.full((B, W), -1, dtype=torch.long, device=logits.device)
    for k in range(K):
        best = logits.argmax(-1)                                  # (B,W)
        mask = (best == k)
        scores = logits[:, :, k].masked_fill(~mask, -1e9)
        topc = scores.topk(min(C, W), dim=-1).indices              # (B, C)
        assign.scatter_(1, topc, k)
    unassigned = (assign == -1)
    if unassigned.any():
        assign = torch.where(unassigned, logits.argmax(-1), assign)
    return assign


class EyeGroupLM(nn.Module):
    def __init__(self, mode="l"):
        super().__init__()
        self.head = LMHead()
        self.base = BidirectionalMixer(seed=0)                     # И1: base always
        self.groups = nn.ModuleList([BidirectionalMixer(seed=i + 1) for i in range(K)])
        self.mode = mode
        self.eye = nn.Linear(D, K)
        self.gate = nn.Parameter(torch.zeros(1))                   # И2: start = base
        self.C = W // K
        if mode == "r":
            nn.init.normal_(self.eye.weight, 0, 0.3)
            self.eye.requires_grad_(False)
        else:
            nn.init.zeros_(self.eye.weight)                        # И2
            nn.init.zeros_(self.eye.bias)
        self._assign = None
        self._logits = None

    def _group_mix(self, e, assign):
        B, W, D = e.shape
        out = torch.zeros_like(e)
        for k in range(self.K):
            idx = (assign == k)                                    # (B,W) bool
            for b in range(B):
                pos = idx[b].nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                chunk = e[b, pos].unsqueeze(0)
                mixed = self.groups[k](chunk)
                out[b, pos] = mixed[0, :pos.numel()]
        return out

    def forward(self, x):
        e = self.head.embed_pos(x)
        base = self.base(e)
        if self.mode == "r":
            logits = self.eye(e)                                    # frozen random
        else:
            logits = self.eye(e)
        self._logits = logits
        assign = top1_with_capacity(logits, self.C)
        self._assign = assign
        extra = self._group_mix(e, assign)                          # residual mixing
        h = base + torch.sigmoid(self.gate) * extra                 # И3 residual, not replace
        return self.head.head(h)

    def balance_loss(self):
        if self._logits is None:
            return torch.tensor(0.0)
        P = torch.softmax(self._logits, dim=-1)                     # (B,W,K)
        f = P.mean(dim=1)                                           # (B,K) avg prob per cluster
        return K * (f * P.mean(dim=0)).sum()


# ---------------- data ----------------
print("BPE...")
tok = build_tokenizer()
V = tok.get_vocab_size()
train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids

ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, name, form):
    params = sum(p.numel() for p in model.parameters())
    eye_params = 0
    for n, m in model.named_modules():
        if 'eye' in n:
            eye_params += sum(p.numel() for p in m.parameters())
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

    # val positions for train-eval gap
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
        if isinstance(model, EyeGroupLM):
            loss = loss + 0.01 * model.balance_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 2000 == 0:
            # train loss on a fresh batch + val loss
            tr_loss = loss.item()
            model.eval()
            with torch.no_grad():
                vnll = 0.0
                for s0 in range(0, N_VAL, EVAL_BATCH):
                    e = min(s0 + EVAL_BATCH, N_VAL)
                    Xv = np.zeros((e - s0, W), dtype=np.int64)
                    Yv = np.zeros(e - s0, dtype=np.int64)
                    for kk, i in enumerate(val_starts[s0:e]):
                        Xv[kk] = train_ids[i:i+W]
                        Yv[kk] = train_ids[i+W]
                    lo = model(torch.tensor(Xv, dtype=torch.long, device=DEVICE))
                    vnll += float(lossf(lo, torch.tensor(Yv, dtype=torch.long, device=DEVICE)).item()) * (e - s0)
                val_loss = vnll / N_VAL
            model.train()
            gap = tr_loss - val_loss
            gap_log.append({"step": s, "train": tr_loss, "val": val_loss, "gap": gap})
            extra = ""
            if isinstance(model, EyeModulatorLM):
                extra = f" ent={model.route_entropy().item():.3f}"
            print(f"[{name}] [{s}] tr={tr_loss:.3f} val={val_loss:.3f} gap={gap:+.3f}"
                  f" lr={sched.get_last_lr()[0]:.2e}{extra} ({time.time()-t0:.0f}s)", flush=True)
    return params, eye_params, gap_log


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
    ap.add_argument("form", choices=["E0", "E1", "E2"])
    ap.add_argument("mode", choices=["u", "r", "l"], nargs="?", default=None)
    args = ap.parse_args()

    if args.form == "E0":
        name = "E0"
        model = BaselineLM()
    elif args.form == "E1":
        name = f"E1-{args.mode}"
        model = EyeModulatorLM(mode=args.mode)
    else:
        name = f"E2-{args.mode}"
        model = EyeGroupLM(mode=args.mode)
    params, eye_params, gap_log = train(model, name, args.form)
    res = eval_model(model, name)
    res.update({"params": params, "eye_params": eye_params,
                "eye_frac": eye_params / params if params else 0,
                "gap_log": gap_log, "form": args.form, "mode": args.mode})
    print(f"[{name}] params={params:,} eye={eye_params:,} ({eye_params/params*100:.1f}%) "
          f"mixer_only={res['mixer_only']:.3f} gated(k=1)={res['gated'][1.0]:.3f}", flush=True)

    os.makedirs(HERE, exist_ok=True)
    json.dump(res, open(os.path.join(HERE, f"results_{args.form}_{args.mode}.json"), "w"), indent=2)
    torch.save(model.state_dict(), os.path.join(HERE, f"model_{args.form}_{args.mode}.pt"))
    print("saved", os.path.join(HERE, f"results_{args.form}_{args.mode}.json"))
