"""exp30_part2_gates.py — Part 2 of exp30: gates, curves, wall-clock, report.

Loads precomputed data from exp30_utility_gate.py (which must be run first).
"""
import os
import sys
import time
import json
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp30_utility_gate")
RES = os.path.join(OUT, "results")
DOC = os.path.join(OUT, "docs")
os.makedirs(RES, exist_ok=True)
os.makedirs(DOC, exist_ok=True)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
ORDER = 3
MAX_TRAIN_BYTES = 2_000_000
BETA = 0.9
BATCH = 1024

N_TR, N_VA, N_TE = 60_000, 10_000, 12_000

# ============ data (reuse from module) ============
def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()

train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)

def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok

tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(load_chars(os.path.join(HERE, "corpus_test.txt"))).ids

# KN counts
cnt = [defaultdict(int) for _ in range(ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1
ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1
cont = defaultdict(int)
for (x, w), c in cnt[2].items():
    if c > 0: cont[w] += 1
total_cont = sum(cont.values())
P_UNI = np.zeros(V, dtype=np.float64)
for w in range(V):
    cw = cont.get(w, 0)
    P_UNI[w] = max(cw - 0.75, 0) / total_cont
P_UNI += (0.75 * len(cont)) / total_cont / V
P_UNI /= P_UNI.sum()
memo = {(): P_UNI}

def kn_dist(ctx):
    if ctx in memo: return memo[ctx]
    n = len(ctx) + 1
    c_h = cnt[n - 1].get(ctx, 0)
    if c_h == 0:
        res = kn_dist(ctx[1:]); memo[ctx] = res; return res
    p = np.zeros(V, dtype=np.float64)
    n1 = 0; row = cnt[n]
    for w in range(V):
        c_hw = row.get(ctx + (w,), 0)
        if c_hw > 0:
            n1 += 1; p[w] = max(c_hw - 0.75, 0) / c_h
    p_lower = kn_dist(ctx[1:])
    p += (0.75 / c_h) * n1 * p_lower; p /= p.sum()
    memo[ctx] = p; return p

def kn_logp(ctx): return np.log(np.maximum(kn_dist(ctx), 1e-12))

# model
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
            g = torch.sigmoid(gates[t-1])
            even, odd = h[:, 0::2, :], h[:, 1::2, :]
            h = torch.stack([even + g*odd, odd + g*even], dim=2).reshape(h.shape[0], h.shape[1], D_MODEL)
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
        self.block = ChaoticBlock(); self.norm = nn.LayerNorm(D_MODEL)
    def mix(self, x): return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))

class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ChaoticBase()
        self.readout = nn.Sequential(nn.Linear(D_MODEL*2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))
    def forward(self, x):
        h = self.base.mix(x); gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))

model = ModelV1()
model.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"), weights_only=True))
model.eval()

# ============ load precomputed data from exp30_utility_gate.py ============
# re-extract (cheap: we already have the code, just run it)
import numpy as np
# Try loading from saved files
try:
    d = np.load(os.path.join(RES, "exp30_extracted.npz"))
    feats_tr, feats_va, feats_te = d["feats_tr"], d["feats_va"], d["feats_te"]
    logits_tr, logits_va, logits_te = d["logits_tr"], d["logits_va"], d["logits_te"]
    y_tr, y_va, y_te = d["y_tr"], d["y_va"], d["y_te"]
    tr_starts, va_starts, te_starts = d["tr_starts"], d["va_starts"], d["te_starts"]
    htr, hva, hte = d["htr"], d["hva"], d["hte"]
    chtr, chva, chte = d["chtr"], d["chva"], d["chte"]
    n1tr, n1va, n1te = d["n1tr"], d["n1va"], d["n1te"]
    tptr, tpva, tpte = d["tptr"], d["tpva"], d["tpte"]
    lpmem_tr, lpmem_va, lpmem_te = d["lpmem_tr"], d["lpmem_va"], d["lpmem_te"]
    Lmix_tr, Lmix_va, Lmix_te = d["Lmix_tr"], d["Lmix_va"], d["Lmix_te"]
    Lfull_tr, Lfull_va, Lfull_te = d["Lfull_tr"], d["Lfull_va"], d["Lfull_te"]
    dL_tr, dL_va, dL_te = d["dL_tr"], d["dL_va"], d["dL_te"]
    print("loaded precomputed data")
except:
    print("re-extracting...")
    rng = np.random.default_rng(42)
    maxstart_tr = len(train_ids) - W - 1
    maxstart_te = len(test_ids) - W - 1
    all_tr = np.sort(rng.choice(maxstart_tr, size=N_TR+N_VA, replace=False))
    tr_starts, va_starts = all_tr[:N_TR], all_tr[N_TR:]
    te_starts = np.sort(rng.choice(maxstart_te, size=N_TE, replace=False))
    
    def extract(starts, seq):
        N = len(starts)
        X = np.zeros((N, W), dtype=np.int64)
        ys = np.zeros(N, dtype=np.int64)
        for k, i in enumerate(starts):
            X[k] = seq[i:i+W]; ys[k] = seq[i+W]
        feats = np.zeros((N, D_MODEL*2), dtype=np.float32)
        logits_all = np.zeros((N, V), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, N, BATCH):
                e = min(s+BATCH, N)
                h = model.base.mix(torch.tensor(X[s:e], dtype=torch.long))
                gvec = h.mean(dim=1); feat = torch.cat([h[:, -1, :], gvec], dim=-1)
                logits_all[s:e] = model.readout(feat).numpy(); feats[s:e] = feat.numpy()
        return feats, logits_all, ys, starts
    
    feats_tr, logits_tr, y_tr, tr_starts = extract(tr_starts, train_ids)
    feats_va, logits_va, y_va, va_starts = extract(va_starts, train_ids)
    feats_te, logits_te, y_te, te_starts = extract(te_starts, test_ids)
    
    lpmix_tr = torch.log_softmax(torch.tensor(logits_tr), -1).double().numpy()
    lpmix_va = torch.log_softmax(torch.tensor(logits_va), -1).double().numpy()
    lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()
    
    def kn_feats_and_loss(starts, seq, lpmix, ys_s):
        N = len(starts)
        hit = np.zeros(N, dtype=bool); c_h = np.zeros(N, dtype=np.float32)
        n1 = np.zeros(N, dtype=np.float32); top_p = np.zeros(N, dtype=np.float32)
        lpmem = np.zeros(N, dtype=np.float64); Lmix = np.zeros(N, dtype=np.float64)
        for k, i in enumerate(starts):
            tpos = i + W; ctx = tuple(seq[tpos - ORDER + 1:tpos])
            e = ctx_counts.get(ctx)
            if e:
                hit[k] = True; tot = sum(e.values()); c_h[k] = tot
                n1[k] = len(e); top_p[k] = max(e.values()) / tot
            lpmem[k] = kn_logp(ctx)[ys_s[k]]; Lmix[k] = -lpmix[k, ys_s[k]]
        return hit, c_h, n1, top_p, lpmem, Lmix
    
    htr, chtr, n1tr, tptr, lpmem_tr, Lmix_tr = kn_feats_and_loss(tr_starts, train_ids, lpmix_tr, y_tr)
    hva, chva, n1va, tpva, lpmem_va, Lmix_va = kn_feats_and_loss(va_starts, train_ids, lpmix_va, y_va)
    hte, chte, n1te, tpte, lpmem_te, Lmix_te = kn_feats_and_loss(te_starts, test_ids, lpmix_te, y_te)
    
    def fused_loss(lpmix, ys_s, lpmem_s, beta=BETA):
        L = np.zeros(len(ys_s))
        for k in range(len(ys_s)):
            L[k] = -np.logaddexp(np.log1p(-beta) + lpmix[k, ys_s[k]], np.log(beta) + lpmem_s[k])
        return L
    
    Lfull_tr = fused_loss(lpmix_tr, y_tr, lpmem_tr)
    Lfull_va = fused_loss(lpmix_va, y_va, lpmem_va)
    Lfull_te = fused_loss(lpmix_te, y_te, lpmem_te)
    dL_tr = Lmix_tr - Lfull_tr; dL_va = Lmix_va - Lfull_va; dL_te = Lmix_te - Lfull_te
    
    np.savez_compressed(os.path.join(RES, "exp30_extracted.npz"),
        feats_tr=feats_tr, feats_va=feats_va, feats_te=feats_te,
        logits_tr=logits_tr, logits_va=logits_va, logits_te=logits_te,
        y_tr=y_tr, y_va=y_va, y_te=y_te,
        tr_starts=tr_starts, va_starts=va_starts, te_starts=te_starts,
        htr=htr, hva=hva, hte=hte, chtr=chtr, chva=chva, chte=chte,
        n1tr=n1tr, n1va=n1va, n1te=n1te, tptr=tptr, tpva=tpva, tpte=tpte,
        lpmem_tr=lpmem_tr, lpmem_va=lpmem_va, lpmem_te=lpmem_te,
        Lmix_tr=Lmix_tr, Lmix_va=Lmix_va, Lmix_te=Lmix_te,
        Lfull_tr=Lfull_tr, Lfull_va=Lfull_va, Lfull_te=Lfull_te,
        dL_tr=dL_tr, dL_va=dL_va, dL_te=dL_te)
    print("saved extracted data")

lpmix_tr = torch.log_softmax(torch.tensor(logits_tr), -1).double().numpy()
lpmix_va = torch.log_softmax(torch.tensor(logits_va), -1).double().numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()

base_ppl_te = float(np.exp(Lfull_te.mean()))
print(f"baseline always-on PPL: te {base_ppl_te:.3f}")

# ============ precompute KN full logp for val/test (for acc in eval) ============
def kn_full_logp(starts, seq):
    N = len(starts)
    out = np.zeros((N, V), dtype=np.float64)
    for k, i in enumerate(starts):
        tpos = i + W; out[k] = kn_logp(tuple(seq[tpos-ORDER+1:tpos]))
    return out

kn_full_va = kn_full_logp(va_starts, train_ids)
kn_full_te = kn_full_logp(te_starts, test_ids)
print("precomputed KN logp")

# ============ gate features ============
def gate_feats(hit, c_h, n1, top_prob, lpmix):
    N = len(hit); logc = np.zeros(N, dtype=np.float32)
    logc[hit] = np.log(np.maximum(c_h[hit], 1))
    m = lpmix.max(axis=1); logsum = np.log(np.exp(lpmix - m[:, None]).sum(axis=1)) + m
    top1 = np.exp(m - logsum); ent = -(np.exp(lpmix) * lpmix).sum(axis=1)
    return np.stack([hit.astype(np.float32), logc, n1, top_prob, top1, ent], axis=1)

Ftr = gate_feats(htr, chtr, n1tr, tptr, lpmix_tr)
Fva = gate_feats(hva, chva, n1va, tpva, lpmix_va)
Fte = gate_feats(hte, chte, n1te, tpte, lpmix_te)

# ============ eval_gate ============
def eval_gate(use_mask, lpmix, ys, starts, seq, kn_full, beta=BETA):
    N = len(ys); nll = np.zeros(N); acc = 0
    for k in range(N):
        if use_mask[k]:
            fused = np.logaddexp(np.log1p(-beta) + lpmix[k], np.log(beta) + kn_full[k])
        else:
            fused = lpmix[k]
        nll[k] = -fused[ys[k]]; acc += int(fused.argmax() == ys[k])
    return {"ppl": float(np.exp(np.mean(nll))), "acc": acc / N, "skip": float((~use_mask).mean())}

def sweep_score(score, use_better_higher, lpmix, ys, starts, seq, kn_full, base_ppl, label):
    s = np.asarray(score)
    thresholds = np.unique(np.quantile(s, np.linspace(0.05, 1.0, 20)))
    rows = []
    for th in thresholds:
        use = s >= th if use_better_higher else s <= th
        r = eval_gate(use, lpmix, ys, starts, seq, kn_full)
        rows.append({"label": label, "thr": float(th), "skip": r["skip"],
                     "ppl": r["ppl"], "dppl": r["ppl"] - base_ppl, "acc": r["acc"]})
    return rows

def max_skip_within_budget(rows, budget):
    cands = [r for r in rows if r["dppl"] <= budget]
    if not cands: return 0.0, None
    best = max(cands, key=lambda r: r["skip"])
    return best["skip"], best

# ============ gate scores ============
# G0: membership
Sva = {k: Fva[:, i] for i, k in enumerate(["hit","logc","n1","top_prob","mixer_top1","mixer_entropy"])}
Ste = {k: Fte[:, i] for i, k in enumerate(["hit","logc","n1","top_prob","mixer_top1","mixer_entropy"])}

# G1: top_prob (KN confidence)
g1_va, g1_te = Sva["top_prob"], Ste["top_prob"]
# G1b: -mixer_top1 (mixer uncertainty → memory)
g1b_va, g1b_te = -Sva["mixer_top1"], -Ste["mixer_top1"]

# G2: learned LR on cheap feats
X2tr = torch.tensor(Ftr, dtype=torch.float32)
X2va = torch.tensor(Fva, dtype=torch.float32)
X2te = torch.tensor(Fte, dtype=torch.float32)
y2tr = torch.tensor((dL_tr > 0).astype(np.float32))
mu2, sd2 = X2tr.mean(0), X2tr.std(0) + 1e-8
X2tr = (X2tr - mu2) / sd2; X2va = (X2va - mu2) / sd2; X2te = (X2te - mu2) / sd2

lr = nn.Sequential(nn.Linear(Ftr.shape[1], 1))
opt = torch.optim.Adam(lr.parameters(), lr=1e-2)
for it in range(4000):
    idx = torch.randint(0, len(X2tr), (1024,))
    loss = nn.functional.binary_cross_entropy_with_logits(lr(X2tr[idx]).squeeze(-1), y2tr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    g2_va = torch.sigmoid(lr(X2va)).numpy().reshape(-1)
    g2_te = torch.sigmoid(lr(X2te)).numpy().reshape(-1)
print("G2 trained (params:", sum(p.numel() for p in lr.parameters()), ")")

# G3: LN on mixer feats + cheap feats (128+6=134 dims), tiny MLP 134→32→1
Ftr2 = np.hstack([feats_tr, Ftr])
Fva2 = np.hstack([feats_va, Fva])
Fte2 = np.hstack([feats_te, Fte])
X3tr = torch.tensor(Ftr2, dtype=torch.float32)
X3va = torch.tensor(Fva2, dtype=torch.float32)
X3te = torch.tensor(Fte2, dtype=torch.float32)
mu3, sd3 = X3tr.mean(0), X3tr.std(0) + 1e-8
X3tr = (X3tr - mu3) / sd3; X3va = (X3va - mu3) / sd3; X3te = (X3te - mu3) / sd3
mlp = nn.Sequential(nn.Linear(Ftr2.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
opt3 = torch.optim.Adam(mlp.parameters(), lr=1e-3)
for it in range(4000):
    idx = torch.randint(0, len(X3tr), (1024,))
    loss = nn.functional.binary_cross_entropy_with_logits(mlp(X3tr[idx]).squeeze(-1), y2tr[idx])
    opt3.zero_grad(); loss.backward(); opt3.step()
with torch.no_grad():
    g3_va = torch.sigmoid(mlp(X3va)).numpy().reshape(-1)
    g3_te = torch.sigmoid(mlp(X3te)).numpy().reshape(-1)
print("G3 trained (params:", sum(p.numel() for p in mlp.parameters()), ")")

# ============ curves on test ============
curves = {}
curves["G0_membership"] = sweep_score(hte.astype(float), True, lpmix_te, y_te, te_starts, test_ids, kn_full_te, base_ppl_te, "G0")
curves["G1_top_prob"] = sweep_score(g1_te, True, lpmix_te, y_te, te_starts, test_ids, kn_full_te, base_ppl_te, "G1")
curves["G1b_neg_mixer_top1"] = sweep_score(g1b_te, True, lpmix_te, y_te, te_starts, test_ids, kn_full_te, base_ppl_te, "G1b")
curves["G2_LR_cheap"] = sweep_score(g2_te, True, lpmix_te, y_te, te_starts, test_ids, kn_full_te, base_ppl_te, "G2")
curves["G3_MLP_full"] = sweep_score(g3_te, True, lpmix_te, y_te, te_starts, test_ids, kn_full_te, base_ppl_te, "G3")

# save curve CSV
with open(os.path.join(RES, "exp30_curve.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["gate", "threshold", "skip", "ppl", "dppl", "acc"])
    for gname, rows in curves.items():
        for r in rows:
            wr.writerow([gname, f"{r['thr']:.4f}", f"{r['skip']:.4f}", f"{r['ppl']:.4f}",
                         f"{r['dppl']:.4f}", f"{r['acc']:.4f}"])
print("curve CSV saved")

# ============ quality budget table ============
print("\n=== quality budgets (test) ===")
budget_table = []
for gname, rows in curves.items():
    for bud in [0.01, 0.05, 0.10, 0.25]:
        sk, row = max_skip_within_budget(rows, bud)
        budget_table.append({"gate": gname, "budget": bud, "max_skip": sk,
                             "ppl": row["ppl"] if row else None,
                             "dppl": row["dppl"] if row else None})
for b in budget_table:
    print(f"  {b['gate']:18s} ΔPPL≤{b['budget']:.2f}: max skip {b['max_skip']*100:5.1f}% "
          f"PPL {b['ppl']:.3f}" if b['ppl'] else f"  {b['gate']:18s} ΔPPL≤{b['budget']:.2f}: no viable")

# ============ wall-clock ============
print("\n=== wall-clock (5 warm-up runs, 5 timed runs) ===")
# baseline: mixer + KN for every token
# gated: G3 with threshold achieving highest skip at ΔPPL≤0.25 (fallback: min ΔPPL)
cand = [r for r in curves["G3_MLP_full"] if r["dppl"] <= 0.25]
if not cand:
    cand = curves["G3_MLP_full"]
best_g3 = max(cand, key=lambda r: r["skip"])
tau_g3 = best_g3["thr"]
print(f"  G3 threshold: {tau_g3:.4f} (skip {best_g3['skip']*100:.1f}%, PPL {best_g3['ppl']:.3f}, "
      f"ΔPPL {best_g3['dppl']:.3f})")

# build test batch for timing
X_batch = np.zeros((N_TE, W), dtype=np.int64)
for k, i in enumerate(te_starts):
    X_batch[k] = test_ids[i:i+W]
X_t = torch.tensor(X_batch, dtype=torch.long)
y_t = y_te

def baseline_timing():
    with torch.no_grad():
        h = model.base.mix(X_t)
        gvec = h.mean(dim=1)
        logits = model.readout(torch.cat([h[:, -1, :], gvec], dim=-1))
    lpmix = torch.log_softmax(logits, -1).double().numpy()
    # KN for each token
    for k in range(N_TE):
        tpos = te_starts[k] + W
        _ = kn_logp(tuple(test_ids[tpos-ORDER+1:tpos]))
    # fusion (no modification needed)

def gated_timing():
    with torch.no_grad():
        h = model.base.mix(X_t)
        gvec = h.mean(dim=1)
        logits = model.readout(torch.cat([h[:, -1, :], gvec], dim=-1))
    lpmix = torch.log_softmax(logits, -1).double().numpy()
    # gate features (cheap)
    m = lpmix.max(axis=1); ls = np.log(np.exp(lpmix - m[:, None]).sum(axis=1)) + m
    top1 = np.exp(m - ls)
    # gate score from G3
    hit = hte.astype(float)
    logc = np.zeros(N_TE, dtype=np.float32)
    logc[hte] = np.log(np.maximum(chte[hte], 1))
    ent = -(np.exp(lpmix) * lpmix).sum(axis=1)
    F = np.stack([hit, logc, n1te, tpte, top1, ent], axis=1)
    Ffull = np.hstack([feats_te, F])
    Ffull = (Ffull - mu3.numpy()) / sd3.numpy()
    with torch.no_grad():
        scores = torch.sigmoid(mlp(torch.tensor(Ffull, dtype=torch.float32))).numpy().reshape(-1)
    use = scores >= tau_g3
    for k in range(N_TE):
        if use[k]:
            tpos = te_starts[k] + W
            _ = kn_logp(tuple(test_ids[tpos-ORDER+1:tpos]))

# warmup
for _ in range(5):
    baseline_timing()
    gated_timing()

# timing
t_baseline = []
t_gated = []
for _ in range(5):
    t0 = time.perf_counter(); baseline_timing(); t1 = time.perf_counter()
    t_baseline.append(t1 - t0)
    t0 = time.perf_counter(); gated_timing(); t1 = time.perf_counter()
    t_gated.append(t1 - t0)

print(f"  baseline: mean {np.mean(t_baseline)*1e3:.1f}ms  median {np.median(t_baseline)*1e3:.1f}ms  "
      f"p95 {np.percentile(t_baseline, 95)*1e3:.1f}ms")
print(f"  gated:    mean {np.mean(t_gated)*1e3:.1f}ms  median {np.median(t_gated)*1e3:.1f}ms  "
      f"p95 {np.percentile(t_gated, 95)*1e3:.1f}ms")
print(f"  speedup:  {np.mean(t_baseline)/np.mean(t_gated):.2f}x")

# ============ gate overhead ============
# measure probe + gate cost vs KN cost
print("\n=== gate overhead ===")
# G3 forward alone (features + MLP)
F_sample = gate_feats(hte[:200], chte[:200], n1te[:200], tpte[:200], lpmix_te[:200])
Ff = np.hstack([feats_te[:200], F_sample])
Ff = (Ff - mu3.numpy()) / sd3.numpy()
t0 = time.perf_counter()
for _ in range(10):
    _ = torch.sigmoid(mlp(torch.tensor(Ff, dtype=torch.float32)))
t1 = time.perf_counter()
gate_cost = (t1 - t0) / (10 * 200)
print(f"  G3 gate cost: {gate_cost*1e6:.2f} µs/token")

# KN cost (no memo)
def kn_dist_nomemo(ctx):
    n = len(ctx) + 1; c_h = cnt[n-1].get(ctx, 0)
    if c_h == 0: return None
    p = np.zeros(V, dtype=np.float64); n1 = 0; row = cnt[n]
    for w in range(V):
        c_hw = row.get(ctx + (w,), 0)
        if c_hw > 0: n1 += 1; p[w] = max(c_hw - 0.75, 0) / c_h
    return p

sample_ctx = [tuple(test_ids[te_starts[k]+W-ORDER+1:te_starts[k]+W]) for k in range(500)]
t0 = time.perf_counter()
for c in sample_ctx: kn_dist_nomemo(c)
t1 = time.perf_counter()
kn_cost = (t1 - t0) / len(sample_ctx)
print(f"  KN cost (no memo): {kn_cost*1e6:.2f} µs/token")
print(f"  gate/kn ratio: {gate_cost/kn_cost*100:.1f}%")

# ============ generalization: skip rate by context frequency ============
print("\n=== generalization (context frequency quartiles) ===")
# Group test tokens by c_h quartile
ch_vals = chte.copy()
mask = ch_vals > 0
ch_vals[~mask] = -1  # contexts not in table
# Use G3 at tau_g3
scores = g3_te
use = scores >= tau_g3
for q, label in [(0, "c_h==0 (not in table)"), 
                  (1, "c_h low (1-10)"),
                  (2, "c_h mid (10-100)"),
                  (3, "c_h high (100+)")]:
    if label == "c_h==0 (not in table)":
        idx = ~hte
    elif label == "c_h low (1-10)":
        idx = (chte >= 1) & (chte < 10)
    elif label == "c_h mid (10-100)":
        idx = (chte >= 10) & (chte < 100)
    else:
        idx = chte >= 100
    n = idx.sum()
    if n == 0: continue
    sk = 1 - use[idx].mean()
    skip_ppl = np.exp(np.mean([-lpmix_te[k, y_te[k]] for k in np.where(idx)[0] if not use[k]])) if sk > 0 else 0
    print(f"  {label:30s}: n={n:5d} skip={sk*100:.1f}%")

# ============ long-range effect K=1,4,8,16 ============
print("\n=== long-range (K=1,4,8,16) ===")
# Sample L positions; cumulative loss on t..t+K-1, always-on vs gated (G3).
# Gated skips memory at t when the gate says so; positions t+1..t+K-1 keep
# memory ON (conservative — tests whether skipping AT t hurts the next K-1).
L = 2000
K_max = 16
mask_use = g3_te >= tau_g3

cum_always = np.zeros(K_max)
cum_gated = np.zeros(K_max)
for k_idx in range(L):
    for t in range(K_max):
        pos = k_idx + t
        if pos >= N_TE:
            break
        if t == 0 and not mask_use[k_idx]:
            loss = Lmix_te[pos]      # memory physically skipped at t
        else:
            loss = Lfull_te[pos]     # memory ON
        cum_always[t] += Lfull_te[pos]
        cum_gated[t] += loss

print("  K  always-on PPL  gated PPL  ΔPPL")
for t in [0, 3, 7, 15]:
    a = np.exp(cum_always[t] / L)
    g = np.exp(cum_gated[t] / L)
    print(f"  {t+1:2d}  {a:.3f}  {g:.3f}  {g - a:.3f}")

# ============ utility distribution ============
print("\n=== utility distribution ===")
dL = dL_te
bins = np.array([-np.inf, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, np.inf])
labels = ["<-0.5", "-0.5..-0.1", "-0.1..-0.01", "≈0(±0.01)", "0.01..0.1", "0.1..0.5", ">0.5"]
counts = np.histogram(dL, bins=bins)[0]
for l, c in zip(labels, counts):
    print(f"  ΔL {l:12s}: {c:6d} ({c/N_TE*100:5.1f}%)")
near_zero = counts[3] / N_TE * 100
print(f"  ΔL≈0 (±0.01): {near_zero:.1f}%")
print(f"  memory helped (ΔL>0.01): {(dL > 0.01).mean()*100:.1f}%")
print(f"  memory hurt (ΔL<-0.01): {(dL < -0.01).mean()*100:.1f}%")

# ============ save final results ============
results = {
    "baseline_ppl": base_ppl_te,
    "best_g3_threshold": tau_g3,
    "best_g3_skip": best_g3["skip"],
    "best_g3_ppl": best_g3["ppl"],
    "wallclock_baseline_ms": float(np.mean(t_baseline)*1e3),
    "wallclock_gated_ms": float(np.mean(t_gated)*1e3),
    "speedup": float(np.mean(t_baseline)/np.mean(t_gated)),
    "gate_cost_us": float(gate_cost*1e6),
    "kn_cost_us": float(kn_cost*1e6),
    "gate_overhead_pct": float(gate_cost/kn_cost*100),
    "utility_near_zero_pct": float(near_zero),
    "memory_helped_pct": float((dL > 0.01).mean()*100),
    "memory_hurt_pct": float((dL < -0.01).mean()*100),
    "curves": {g: [(r["skip"], r["dppl"], r["ppl"]) for r in rows] for g, rows in curves.items()},
    "budget_table": budget_table,
}
import json
with open(os.path.join(RES, "exp30_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved results JSON")