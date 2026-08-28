"""exp29_memory_cutoff.py — REAL memory cutoff: physical compute savings.

Q: when the controller says "skip", do we PHYSICALLY save compute by NOT
computing the KN distribution at all — and does quality hold?

This is the honest version of exp28: avg β was cosmetic there (the full
V-dim KN distribution was computed for EVERY token, then blended).
Here, for β < τ the KN lookup is SKIPPED entirely: the mixer's logits are
the answer (β_eff = 0). The KN distribution is computed ONLY for tokens
where the controller flags memory.

Metrics:
  - skip rate (fraction of tokens with real memory OFF)
  - PPL / top-1 with real cutoff (vs always-on fixed-β baseline)
  - measured wall-clock: T_mixer per token vs T_kn per token (no memo —
    production cost: V hash/dict probes), speedup = (Tm+Tk)/(Tm+(1-s)·Tk)

Controller: MLP on mixer features (h_last ⊕ gvec), trained with λ sweep
(from exp28). Plus a pure RULE baseline: skip when the order-3 context is
NOT in the table (c_h == 0) — a two-stage "cheap probe → selective full
lookup" architecture, no controller needed.
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp29_memory_cutoff")
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
MAX_TRAIN_BYTES = 2_000_000

N_TRAIN = 80_000
N_EVAL = 12_000
BATCH = 1024

# ============ data (reuse exp28 convention: windows [i, i+W) → seq[i+W]) ============
def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)


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


print("BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(load_chars(os.path.join(HERE, "corpus_test.txt"))).ids
print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}")

# ============ KN counts (order-3, for both table and cheap probe) ============
print("KN counts...")
ctx_counts = defaultdict(dict)      # ctx(2) -> {w: count}
cnt = [defaultdict(int) for _ in range(ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1

# KN dist (memoized for exactness, but TIME is measured without memo)
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


def kn_logp(ctx):
    return np.log(np.maximum(kn_dist(ctx), 1e-12))


# ============ model ============
class ChaoticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nw = W // W_LOCAL
        self.gates_l = nn.Parameter(torch.zeros(BLOCKS_LOCAL))
        self.gates_g = nn.Parameter(torch.zeros(BLOCKS_GLOBAL))
        self._sig_l = {t: torch.as_tensor(permute_indices(W_LOCAL, t), dtype=torch.long)
                       for t in range(1, BLOCKS_LOCAL + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long)
                       for t in range(1, BLOCKS_GLOBAL + 1)}

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
        B, Wd, d = h.shape
        hw = h.view(B, self.Nw, W_LOCAL, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l)
                           for wi in range(self.Nw)], dim=1)
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
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(),
                                     nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


model = ModelV1()
model.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"),
                                 weights_only=True))
model.eval()


# ============ eval positions + logits + features ============
rng = np.random.default_rng(42)
maxstart_tr = len(train_ids) - W - 1
maxstart_ev = len(test_ids) - W - 1
train_starts = np.sort(rng.choice(maxstart_tr, size=N_TRAIN, replace=False))
eval_starts = np.sort(rng.choice(maxstart_ev, size=N_EVAL, replace=False))


def extract(starts, seq):
    N = len(starts)
    X = np.zeros((N, W), dtype=np.int64)
    ys = np.zeros(N, dtype=np.int64)
    for k, i in enumerate(starts):
        X[k] = seq[i:i + W]
        ys[k] = seq[i + W]
    feats = np.zeros((N, D_MODEL * 2), dtype=np.float32)
    logits_all = np.zeros((N, V), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, BATCH):
            e = min(s + BATCH, N)
            h = model.base.mix(torch.tensor(X[s:e], dtype=torch.long))
            gvec = h.mean(dim=1)
            feat = torch.cat([h[:, -1, :], gvec], dim=-1)
            logits_all[s:e] = model.readout(feat).numpy()
            feats[s:e] = feat.numpy()
    return feats, logits_all, ys


print("extracting...")
feats_tr, logits_tr, y_tr = extract(train_starts, train_ids)
feats_ev, logits_ev, y_ev = extract(eval_starts, test_ids)
lpmix_full_tr = torch.log_softmax(torch.tensor(logits_tr), -1).double().numpy()
lpmix_full_ev = torch.log_softmax(torch.tensor(logits_ev), -1).double().numpy()
lpmix_tr = np.take_along_axis(lpmix_full_tr, y_tr[:, None], 1)[:, 0]
lpmix_ev = np.take_along_axis(lpmix_full_ev, y_ev[:, None], 1)[:, 0]

# KN logp of true token per position (for training the controller)
lpmem_tr = np.zeros(N_TRAIN)
lpmem_ev = np.zeros(N_EVAL)
for k, i in enumerate(train_starts):
    tpos = i + W
    lpmem_tr[k] = kn_logp(tuple(train_ids[tpos - ORDER + 1:tpos]))[y_tr[k]]
for k, i in enumerate(eval_starts):
    tpos = i + W
    lpmem_ev[k] = kn_logp(tuple(test_ids[tpos - ORDER + 1:tpos]))[y_ev[k]]
print("features ready")

# ============ controller (trained like exp28, λ sweep) ============
class BetaController(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(D_MODEL * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, f):
        return torch.sigmoid(self.mlp(f))


def train_controller(lmbda, steps=3000, bs=512):
    ctl = BetaController()
    opt = torch.optim.Adam(ctl.parameters(), lr=1e-3)
    F = torch.tensor(feats_tr, dtype=torch.float32)
    lpm = torch.tensor(lpmix_tr, dtype=torch.float64)
    lpk = torch.tensor(lpmem_tr, dtype=torch.float64)
    for it in range(steps):
        idx = torch.randint(0, N_TRAIN, (bs,))
        beta = ctl(F[idx]).double().squeeze(-1)
        pm, pk = torch.exp(lpm[idx]), torch.exp(lpk[idx])
        fused = (1 - beta) * pm + beta * pk
        loss = -torch.log(fused + 1e-12).mean() + lmbda * beta.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ctl


# ============ REAL cutoff evaluation ============
def eval_cutoff(beta_pred, tau):
    """Physical memory cutoff: β < τ → KN NOT computed, mixer logits only."""
    nll = np.zeros(N_EVAL)
    acc = 0
    skip = 0
    for k in range(N_EVAL):
        b = beta_pred[k]
        if b < tau:                      # MEMORY OFF — no KN computation at all
            fused = lpmix_full_ev[k]
            skip += 1
        else:                            # MEMORY ON — full KN distribution
            tpos = eval_starts[k] + W
            lp_mem = kn_logp(tuple(test_ids[tpos - ORDER + 1:tpos]))
            fused = np.logaddexp(np.log1p(-b) + lpmix_full_ev[k], np.log(b) + lp_mem)
        nll[k] = -fused[y_ev[k]]
        acc += int(fused.argmax() == y_ev[k])
    return {"ppl": float(np.exp(np.mean(nll))), "acc": acc / N_EVAL,
            "skip": skip / N_EVAL}


def eval_fixed(beta):
    nll = np.zeros(N_EVAL)
    acc = 0
    for k in range(N_EVAL):
        tpos = eval_starts[k] + W
        lp_mem = kn_logp(tuple(test_ids[tpos - ORDER + 1:tpos]))
        fused = np.logaddexp(np.log1p(-beta) + lpmix_full_ev[k], np.log(beta) + lp_mem)
        nll[k] = -fused[y_ev[k]]
        acc += int(fused.argmax() == y_ev[k])
    return {"ppl": float(np.exp(np.mean(nll))), "acc": acc / N_EVAL}


print("\n=== baselines (always memory ON) ===")
for beta in [0.3, 0.5, 0.9, 1.0]:
    r = eval_fixed(beta)
    print(f"  fixed β={beta}: PPL {r['ppl']:.3f} top1 {r['acc']*100:.1f}%")

print("\n=== REAL CUTOFF: controller β < τ → memory physically OFF ===")
results = {}
for lmbda in [0.06, 0.2, 0.5, 1.0]:
    ctl = train_controller(lmbda)
    with torch.no_grad():
        beta_pred = torch.sigmoid(ctl.mlp(torch.tensor(feats_ev, dtype=torch.float32))
                                  ).numpy().reshape(-1)
    print(f"\nλ={lmbda} (avg β {beta_pred.mean():.3f}):")
    for tau in [0.05, 0.1, 0.2, 0.3, 0.5]:
        r = eval_cutoff(beta_pred, tau)
        print(f"  τ={tau}: skip {r['skip']*100:4.1f}%  PPL {r['ppl']:6.3f}  top1 {r['acc']*100:.1f}%")
        results[f"λ{lmbda}_τ{tau}"] = r

# ============ RULE baseline: two-stage cheap probe (no controller) ============
# skip when the order-3 context is absent from the table (c_h == 0)
print("\n=== RULE baseline: skip when context NOT in table (c_h==0) ===")
rule_skip = np.zeros(N_EVAL, dtype=bool)
for k, i in enumerate(eval_starts):
    tpos = i + W
    rule_skip[k] = tuple(test_ids[tpos - ORDER + 1:tpos]) not in ctx_counts
print(f"  contexts absent: {rule_skip.mean()*100:.1f}%")
for tau_mode in ["rule"]:
    nll = np.zeros(N_EVAL)
    acc = 0
    for k in range(N_EVAL):
        if rule_skip[k]:
            fused = lpmix_full_ev[k]
        else:
            tpos = eval_starts[k] + W
            lp_mem = kn_logp(tuple(test_ids[tpos - ORDER + 1:tpos]))
            b = 0.9
            fused = np.logaddexp(np.log1p(-b) + lpmix_full_ev[k], np.log(b) + lp_mem)
        nll[k] = -fused[y_ev[k]]
        acc += int(fused.argmax() == y_ev[k])
    print(f"  rule c_h==0: skip {rule_skip.mean()*100:.1f}%  PPL {np.exp(np.mean(nll)):.3f}  "
          f"top1 {acc/N_EVAL*100:.1f}%")

# ============ physical timing ============
print("\n=== wall-clock (production cost: KN WITHOUT memo) ===")
# T_mixer per token: batched forward
Xb = torch.tensor(np.random.randint(0, V, (1024, W)), dtype=torch.long)
with torch.no_grad():
    model.base.mix(Xb)  # warmup
    t0 = time.perf_counter()
    for _ in range(3):
        model.base.mix(Xb)
    t1 = time.perf_counter()
t_mix_batch = (t1 - t0) / 3
t_mix_per_token = t_mix_batch / (1024 * W)
print(f"mixer: {t_mix_batch*1e3:.1f} ms / batch(1024×{W}) = {t_mix_per_token*1e6:.2f} µs/token")

# T_kn per token WITHOUT memo (fresh recursion each call)
def kn_dist_nomemo(ctx):
    n = len(ctx) + 1
    c_h = cnt[n - 1].get(ctx, 0)
    if c_h == 0:
        return None
    p = np.zeros(V, dtype=np.float64)
    n1 = 0
    row = cnt[n]
    for w in range(V):
        c_hw = row.get(ctx + (w,), 0)
        if c_hw > 0:
            n1 += 1
            p[w] = max(c_hw - 0.75, 0) / c_h
    return p


sample_ctx = [tuple(test_ids[tpos - ORDER + 1:tpos]) for tpos in
              (eval_starts[:500] + W)]
t0 = time.perf_counter()
for c in sample_ctx:
    kn_dist_nomemo(c)
t1 = time.perf_counter()
t_kn_per_token = (t1 - t0) / len(sample_ctx)
print(f"KN full dist: {t_kn_per_token*1e6:.2f} µs/token (V={V} probes, no memo)")

# speedup formula
for skip in [0.25, 0.5, 0.75, 0.9]:
    speedup = (t_mix_per_token + t_kn_per_token) / (t_mix_per_token + (1 - skip) * t_kn_per_token)
    print(f"  skip {skip*100:.0f}% → speedup {speedup:.2f}× (this config)")

# extrapolation to real vocab (T_kn ∝ V)
print("\n=== extrapolation: T_kn ∝ V (real vocab, hash table) ===")
for Vv, tkn in [(512, t_kn_per_token), (4096, t_kn_per_token * 8), (50000, t_kn_per_token * 100)]:
    for skip in [0.5, 0.9]:
        sp = (t_mix_per_token + tkn) / (t_mix_per_token + (1 - skip) * tkn)
        print(f"  V={Vv:6d} T_kn={tkn*1e6:7.1f}µs  skip {skip*100:.0f}% → {sp:.2f}×")

import json
json.dump({"results": {k: v for k, v in results.items()},
           "rule_skip_frac": float(rule_skip.mean())},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("\nSaved to", OUT)
