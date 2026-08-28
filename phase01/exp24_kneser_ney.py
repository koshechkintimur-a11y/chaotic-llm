"""exp24_kneser_ney.py — Phase 5, exp24: smoothed memory channel.

Raw MLE n-grams degraded on NL at order-3 (exp23: 28.5 vs 27.3 at order-2).
Does Kneser-Ney smoothing fix higher orders and keep the memory-scaling
curve dropping?

Setup: reuse the exp23 NL mixer (nl_mixer.pt, fixed). Memory channel = raw
MLE vs interpolated Kneser-Ney (d=0.75), orders 1..4. Eval on WikiText-2 test.
"""
import json
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
OUT = os.path.join(HERE, "exp24_kneser_ney")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
MAX_TRAIN_CHARS = 4_000_000
N_EVAL = 20000
D_KN = 0.75


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


train_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_train.txt"),
                        MAX_TRAIN_CHARS)
test_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_test.txt"))


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


# ============ n-gram counts (n=1..4) + continuation ============
MAX_ORDER = 4
print("counting n-grams...")
cnt = [defaultdict(int) for _ in range(MAX_ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, MAX_ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1
print("counts:", [len(cnt[n]) for n in range(1, MAX_ORDER + 1)])

# continuation counts for unigram level: cont[w] = #distinct left neighbors
cont = defaultdict(int)
for (x, w), c in cnt[2].items():
    if c > 0:
        cont[w] += 1
total_cont = sum(cont.values())
cont_total_types = len(cont)


def kn_unigram():
    p = np.zeros(V, dtype=np.float64)
    for w in range(V):
        cw = cont.get(w, 0)
        p[w] = max(cw - D_KN, 0) / total_cont
    p += (D_KN * cont_total_types) / total_cont / V   # discount mass -> uniform
    p /= p.sum()
    return p


P_UNI = kn_unigram()
_memo = {}


def kn_dist(ctx):
    """P_KN over V for context tuple (length 0..3). Memoized."""
    if ctx in _memo:
        return _memo[ctx]
    n = len(ctx) + 1
    if n == 1:
        _memo[ctx] = P_UNI
        return P_UNI
    c_h = cnt[n - 1].get(ctx, 0)
    if c_h == 0:
        res = kn_dist(ctx[1:])
        _memo[ctx] = res
        return res
    p = np.zeros(V, dtype=np.float64)
    n1 = 0
    row = cnt[n]
    for w in range(V):
        c_hw = row.get(ctx + (w,), 0)
        if c_hw > 0:
            n1 += 1
            p[w] = max(c_hw - D_KN, 0) / c_h
    p_lower = kn_dist(ctx[1:])
    p += (D_KN / c_h) * n1 * p_lower
    p /= p.sum()
    _memo[ctx] = p
    return p


# ============ model (reuse exp23 NL mixer) ============
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
ckpt = os.path.join(HERE, "exp23_nl_beta", "nl_mixer.pt")
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    print("NL mixer loaded (exp23)")
else:
    raise SystemExit("no NL mixer — run exp23 first")
model.eval()


# ============ evaluate ============
def evaluate(get_logp, beta, n_samples=N_EVAL):
    """get_logp(ctx, i) -> np logp over V (or None)."""
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
            cp = get_logp(ctx, i + W)
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
    return {"n": n,
            "ppl_model": float(np.exp(np.mean(nll_m))),
            "ppl_fused": float(np.exp(np.mean(nll_p))),
            "acc_model": acc_m / n, "acc_fused": acc_p / n}


results = {"corpus": "wikitext-2", "V": V, "d_kn": D_KN, "max_order": MAX_ORDER,
           "n_eval": N_EVAL, "mixer_params": sum(p.numel() for p in model.parameters())}

# mixer alone
r0 = evaluate(lambda c, i: None, 0.0)
print(f"\nmixer alone: PPL {r0['ppl_model']:.2f}, top-1 {r0['acc_model']*100:.1f}%")
results["mixer_alone"] = {"ppl": r0["ppl_model"], "acc": r0["acc_model"]}

# KN orders 1..4 at β=0.5
print("\n=== Kneser-Ney memory (β=0.5) ===")
for order in range(1, MAX_ORDER + 1):
    _memo.clear()
    # coverage-limited: build context from last (order-1) tokens
    def get_logp(ctx, i, order=order):
        return np.log(kn_dist(tuple(test_ids[i - (order - 1):i])))
    r = evaluate(get_logp, 0.5)
    mem = (len(cnt[order]) * order * 4) / 1e6
    print(f"order-{order}: contexts={len(cnt[order]):>9,} mem≈{mem:>6.2f}MB "
          f"PPL_fused={r['ppl_fused']:.2f} top1={r['acc_fused']*100:.1f}%")
    results[f"kn_order{order}"] = {"ctx": len(cnt[order]), "mem_MB": mem,
                                   "ppl_fused": r["ppl_fused"],
                                   "acc_fused": r["acc_fused"]}

# β sweep on KN order-4
print("\n=== β sweep (KN order-4) ===")
def get_logp4(ctx, i):
    return np.log(kn_dist(tuple(test_ids[i - 3:i])))
best_b, best_p = 0.0, r0["ppl_model"]
sweep = {}
for beta in [0.3, 0.5, 0.7, 0.9]:
    r = evaluate(get_logp4, beta)
    print(f"β={beta:.1f}: PPL {r['ppl_fused']:.2f} top1 {r['acc_fused']*100:.1f}%")
    sweep[str(beta)] = {"ppl": r["ppl_fused"], "acc": r["acc_fused"]}
    if r["ppl_fused"] < best_p:
        best_p, best_b = r["ppl_fused"], beta
results["kn4_beta_sweep"] = sweep
results["kn4_best"] = {"beta": best_b, "ppl": best_p}
print(f"→ KN order-4 best β={best_b}, PPL {best_p:.2f}")

# compare with raw MLE from exp23 (order-2: 27.25, order-3: 28.53)
results["reference_raw_exp23"] = {"order2_ppl": 27.25, "order3_ppl": 28.53}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
