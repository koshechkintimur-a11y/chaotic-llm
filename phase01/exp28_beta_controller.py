"""exp28_beta_controller.py — THE central experiment of the project.

Can a learned β-CONTROLLER route each token to memory or compute
per-token, and CUT the average β without losing PPL / generation quality?

Architecture claim: chaotic compute is the DEFAULT path (always cheap);
memory is a SELECTIVE correction invoked only when the controller flags
the token. If the controller keeps quality at low average β, the
β-Architecture becomes a genuinely new compute regime:
    O(W log W) chaotic compute (default) + sparse exact memory (on demand).

Method:
  - Frozen mixer V1 (vocab 512, code) + KN order-3 table.
  - Per-position windows [pos-W, pos) → batched mixer forward →
    features = concat(h_last, gvec)  (exact training convention).
  - Controller: MLP(feats) → β ∈ (0,1).  Zero extra compute —
    a head on the mixer's existing state.
  - Loss:  -log((1-β)·p_mix(y) + β·p_mem(y)) + λ·β
    (β raised when memory gives the true token MORE probability — exact
    routing signal; λ pushes average β down).
  - Oracle: per-token optimal β* = 1 iff p_mem(y) > p_mix(y) (binary).
  - Compare: fixed-β curve vs per-token controller vs oracle.
  - Metrics: PPL, avg β, sparsity (% β < 0.2 — memory SKIPPED),
    corr(controller, oracle), generation with β-trace.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp28_beta_controller")
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

N_TRAIN = 80_000        # positions for controller training
N_EVAL = 12_000         # positions for full-dist PPL
BATCH = 1024
GEN_LEN = 140
TEMPERATURE = 0.8
TOPP = 0.9


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


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


print("Training BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids

test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))
test_ids = tok.encode(test_text).ids
print(f"V={V}, train tokens={len(train_ids):,}, test tokens={len(test_ids):,}")


# ============ KN table (dict, order-3, memoized) ============
print("building KN order-3 table...")
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


def kn_logp(ctx):
    return np.log(np.maximum(kn_dist(ctx), 1e-12))


# ============ model ============
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


class ModelV1(nn.Module):
    def __init__(self, base, V):
        super().__init__()
        self.base = base
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(),
                                     nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


model = ModelV1(ChaoticBase(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL), V)
model.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"),
                                 weights_only=True))
model.eval()
print(f"model loaded ({sum(p.numel() for p in model.parameters()):,} params)")


# ============ batched per-position feature extraction ============
def extract(starts, seq):
    """Windows [i, i+W) from `seq` → feats (N,2d), logits (N,V), targets seq[i+W].
    Exactly exp18/22's eval convention — no padding needed."""
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
            xb = torch.tensor(X[s:e], dtype=torch.long)
            h = model.base.mix(xb)
            gvec = h.mean(dim=1)
            hlast = h[:, -1, :]
            feat = torch.cat([hlast, gvec], dim=-1)
            logits = model.readout(feat)
            feats[s:e] = feat.numpy()
            logits_all[s:e] = logits.numpy()
    return feats, logits_all, ys


def kn_logp_y(starts, seq, ys):
    """KN logp of target seq[i+W] given the last ORDER-1 tokens before it."""
    N = len(starts)
    out = np.zeros(N, dtype=np.float64)
    for k, i in enumerate(starts):
        tpos = i + W
        ctx = tuple(seq[tpos - ORDER + 1: tpos])  # last 2 tokens before target (order-3)
        out[k] = kn_logp(ctx)[ys[k]]
    return out


rng = np.random.default_rng(42)
maxstart_tr = len(train_ids) - W - 1
maxstart_ev = len(test_ids) - W - 1
train_starts = np.sort(rng.choice(maxstart_tr, size=N_TRAIN, replace=False))
eval_starts = np.sort(rng.choice(maxstart_ev, size=N_EVAL, replace=False))

print("extracting train features...")
feats_tr, logits_tr, y_tr = extract(train_starts, train_ids)
lpmix_tr = np.take_along_axis(torch.log_softmax(torch.tensor(logits_tr), -1).numpy(),
                              y_tr[:, None], 1)[:, 0]
print("extracting eval features...")
feats_ev, logits_ev, y_ev = extract(eval_starts, test_ids)
lpmix_ev = np.take_along_axis(torch.log_softmax(torch.tensor(logits_ev), -1).numpy(),
                              y_ev[:, None], 1)[:, 0]
print("KN logp...")
lpmem_tr = kn_logp_y(train_starts, train_ids, y_tr)
lpmem_ev = kn_logp_y(eval_starts, test_ids, y_ev)
np.savez_compressed(os.path.join(OUT, "features.npz"),
                    feats_tr=feats_tr, lpmix_tr=lpmix_tr, lpmem_tr=lpmem_tr,
                    feats_ev=feats_ev, lpmix_ev=lpmix_ev, lpmem_ev=lpmem_ev)

# ============ oracle stats ============
mem_better_tr = lpmem_tr > lpmix_tr
mem_better_ev = lpmem_ev > lpmix_ev
print(f"\nORACLE (per-token binary β*):")
print(f"  train: {mem_better_tr.mean()*100:.1f}% tokens prefer memory  (avg β*={mem_better_tr.mean():.3f})")
print(f"  eval : {mem_better_ev.mean()*100:.1f}% tokens prefer memory  (avg β*={mem_better_ev.mean():.3f})")


# ============ controller ============
class BetaController(nn.Module):
    def __init__(self, d_in=128, d_h=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, 1))

    def forward(self, f):
        return torch.sigmoid(self.mlp(f))


def train_controller(lmbda, steps=3000, bs=512, lr=1e-3):
    ctl = BetaController()
    opt = torch.optim.Adam(ctl.parameters(), lr=lr)
    F = torch.tensor(feats_tr, dtype=torch.float32)
    lpm = torch.tensor(lpmix_tr, dtype=torch.float64)
    lpk = torch.tensor(lpmem_tr, dtype=torch.float64)
    for it in range(steps):
        idx = torch.randint(0, N_TRAIN, (bs,))
        beta = ctl(F[idx]).double().squeeze(-1)
        pm = torch.exp(lpm[idx])
        pk = torch.exp(lpk[idx])
        fused = (1 - beta) * pm + beta * pk
        loss = -torch.log(fused + 1e-12).mean() + lmbda * beta.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ctl


# ============ eval: full-dist PPL + avg β + sparsity ============
def eval_controller(ctl, hard=False, thr=0.2):
    """Full-dist PPL on eval positions with the controller's β."""
    lpmix_full = torch.log_softmax(torch.tensor(logits_ev), -1).double().numpy()
    with torch.no_grad():
        beta_all = torch.sigmoid(ctl.mlp(torch.tensor(feats_ev, dtype=torch.float32)))
        beta_all = beta_all.numpy().reshape(-1)
    if hard:
        beta_all = np.where(beta_all >= thr, 1.0, 0.0)
    nll = np.zeros(N_EVAL)
    acc = 0
    for k, i in enumerate(eval_starts):
        tpos = i + W
        lp_mem = kn_logp(tuple(test_ids[tpos - ORDER + 1: tpos]))
        b = beta_all[k]
        fused = np.logaddexp(np.log1p(-b) + lpmix_full[k], np.log(b) + lp_mem)
        nll[k] = -fused[y_ev[k]]
        acc += int(fused.argmax() == y_ev[k])
    return {"ppl": float(np.exp(np.mean(nll))), "acc": acc / N_EVAL,
            "avg_beta": float(beta_all.mean()),
            "sparsity": float((beta_all < 0.2).mean())}


def eval_fixed(beta):
    lpmix_full = torch.log_softmax(torch.tensor(logits_ev), -1).double().numpy()
    nll = np.zeros(N_EVAL)
    acc = 0
    for k, i in enumerate(eval_starts):
        tpos = i + W
        lp_mem = kn_logp(tuple(test_ids[tpos - ORDER + 1: tpos]))
        fused = np.logaddexp(np.log1p(-beta) + lpmix_full[k], np.log(beta) + lp_mem)
        nll[k] = -fused[y_ev[k]]
        acc += int(fused.argmax() == y_ev[k])
    return {"ppl": float(np.exp(np.mean(nll))), "acc": acc / N_EVAL, "avg_beta": beta}


def eval_oracle():
    lpmix_full = torch.log_softmax(torch.tensor(logits_ev), -1).double().numpy()
    nll = np.zeros(N_EVAL)
    acc = 0
    avg = 0.0
    for k, i in enumerate(eval_starts):
        tpos = i + W
        lp_mem = kn_logp(tuple(test_ids[tpos - ORDER + 1: tpos]))
        b = 1.0 if lp_mem[y_ev[k]] > lpmix_full[k][y_ev[k]] else 0.0
        fused = np.logaddexp(np.log1p(-b) + lpmix_full[k], np.log(b) + lp_mem)
        nll[k] = -fused[y_ev[k]]
        acc += int(fused.argmax() == y_ev[k])
        avg += b
    return {"ppl": float(np.exp(np.mean(nll))), "acc": acc / N_EVAL, "avg_beta": avg / N_EVAL}


print("\n=== fixed-β baseline curve (eval, full dist) ===")
fixed_curve = []
for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    r = eval_fixed(beta)
    fixed_curve.append((beta, r["ppl"], r["acc"]))
    print(f"  β={beta:.1f}: PPL {r['ppl']:.3f}  top1 {r['acc']*100:.1f}%")
best_fixed = min(fixed_curve, key=lambda x: x[1])
print(f"  BEST fixed β={best_fixed[0]}: PPL {best_fixed[1]:.3f} top1 {best_fixed[2]*100:.1f}%")


print("\n=== oracle (per-token binary β*) ===")
oracle = eval_oracle()
print(f"  oracle: PPL {oracle['ppl']:.3f} top1 {oracle['acc']*100:.1f}% "
      f"avg β {oracle['avg_beta']:.3f}")


print("\n=== β-controller (per-token), λ sweep ===")
ctrl_results = {}
for lmbda in [0.0, 0.01, 0.03, 0.06, 0.1, 0.2]:
    ctl = train_controller(lmbda)
    r = eval_controller(ctl)
    rh = eval_controller(ctl, hard=True)
    ctrl_results[lmbda] = {"soft": r, "hard": rh}
    print(f"  λ={lmbda}: PPL {r['ppl']:.3f} top1 {r['acc']*100:.1f}% "
          f"avg β {r['avg_beta']:.3f} sparsity {r['sparsity']*100:.0f}%  | "
          f"HARD: PPL {rh['ppl']:.3f} avg β {rh['avg_beta']:.3f} sparsity {rh['sparsity']*100:.0f}%")
    torch.save(ctl.state_dict(), os.path.join(OUT, f"ctl_{lmbda}.pt"))


# corr(controller β, oracle) with a mid-λ controller
ctl_best = train_controller(0.06)
with torch.no_grad():
    beta_pred = torch.sigmoid(ctl_best.mlp(torch.tensor(feats_ev, dtype=torch.float32))).numpy().reshape(-1)
corr = float(np.corrcoef(beta_pred, (lpmem_ev > lpmix_ev).astype(float))[0, 1])
print(f"\n  corr(controller β, oracle) = {corr:.3f}")

import json
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump({
        "oracle": oracle, "best_fixed": {"beta": best_fixed[0], "ppl": best_fixed[1],
                                         "acc": best_fixed[2]},
        "fixed_curve": fixed_curve, "controller": ctrl_results,
        "corr": corr,
        "mem_better_frac_eval": float(mem_better_ev.mean()),
        "mixer_alone_ppl": fixed_curve[0][1],
    }, f, indent=2)
print("saved results.json")
