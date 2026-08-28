"""Diagnostic 2: richer cheap features for routing — what's the AUC ceiling?

Cheap per-token features (O(1), no full V-dim distribution):
  - mixer h_last ⊕ gvec           (compute state, free)
  - mixer top-1 prob / entropy    (compute confidence, free)
  - KN context count c_h          (1 hash lookup — is the pattern common?)
  - KN context distinct continuations n1 (same lookup)
  - KN top-1 candidate prob       (cheap: max over the context's continuations)
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

d = np.load(os.path.join(HERE, "exp28_beta_controller", "features.npz"))
feats_tr = d["feats_tr"]
feats_ev = d["feats_ev"]
lpmix_tr, lpmem_tr = d["lpmix_tr"], d["lpmem_tr"]
lpmix_ev, lpmem_ev = d["lpmix_ev"], d["lpmem_ev"]
y_tr = (lpmem_tr > lpmix_tr).astype(np.float32)
y_ev = (lpmem_ev > lpmix_ev).astype(np.float32)


def auc(labels, scores):
    order = np.argsort(scores)
    ranks = np.empty(len(labels))
    ranks[order] = np.arange(1, len(labels) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), len(labels) - pos.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_eval(feats_tr, feats_ev, name):
    mu = feats_tr.mean(0, keepdims=True)
    sd = feats_tr.std(0, keepdims=True) + 1e-8
    Xtr = torch.tensor((feats_tr - mu) / sd, dtype=torch.float32)
    Xev = torch.tensor((feats_ev - mu) / sd, dtype=torch.float32)
    Ytr = torch.tensor(y_tr)
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(feats_tr.shape[1], 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for it in range(5000):
        idx = torch.randint(0, len(Xtr), (1024,))
        loss = nn.functional.binary_cross_entropy_with_logits(
            m(Xtr[idx]).squeeze(-1), Ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s_ev = torch.sigmoid(m(Xev)).numpy().reshape(-1)
    a = auc(y_ev, s_ev)
    print(f"{name}: AUC eval {a:.3f}  corr {np.corrcoef(s_ev, y_ev)[0,1]:.3f}")
    return s_ev


# ---- cheap KN-context features ----
# rebuild KN counts on train (order-3)
W = 256
ORDER = 3
V = 512
train_ids = None
# load train ids via BPE (same as exp28)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), 2_000_000)
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=V, special_tokens=[], show_progress=False)
tok.train_from_iterator([train_text[i:i + 100000] for i in range(0, len(train_text), 100000)],
                        trainer=trainer)
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(load_chars(os.path.join(HERE, "corpus_test.txt"))).ids

# build context -> {w: count} for order-3
ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1


def kn_ctx_feats(seq, starts):
    N = len(starts)
    F = np.zeros((N, 3), dtype=np.float32)
    for k, i in enumerate(starts):
        tpos = i + W
        ctx = tuple(seq[tpos - ORDER + 1:tpos])
        e = ctx_counts.get(ctx)
        if e:
            tot = sum(e.values())
            n1 = len(e)
            top_c = max(e.values())
            F[k] = [tot, n1, top_c / tot]
        else:
            F[k] = [0, 0, 0]
    return F


# reconstruct starts (same seed as exp28)
rng = np.random.default_rng(42)
maxstart_tr = len(train_ids) - W - 1
maxstart_ev = len(test_ids) - W - 1
train_starts = np.sort(rng.choice(maxstart_tr, size=80000, replace=False))
eval_starts = np.sort(rng.choice(maxstart_ev, size=12000, replace=False))

Ftr_kn = kn_ctx_feats(train_ids, train_starts)
Fev_kn = kn_ctx_feats(test_ids, eval_starts)
print("KN-ctx features built:", Ftr_kn.shape)

# ---- mixer confidence features (free) ----
lpmix_ev_full = np.log(1e-12 + np.exp(np.clip(lpmix_ev, -30, 30)))
# we only have logp of true y; approximate confidence via ... we need full logits.
# skip mixer-confidence features for now (they'd need re-extraction).

# ---- test feature sets ----
print("\nFeature-set AUC ceilings (eval):")
fit_eval(feats_tr[:, :], feats_ev[:, :], "mixer h+gvec only")
fit_eval(Ftr_kn, Fev_kn, "KN-ctx cheap feats only")
fit_eval(np.hstack([feats_tr, Ftr_kn]), np.hstack([feats_ev, Fev_kn]),
         "mixer + KN-ctx")
