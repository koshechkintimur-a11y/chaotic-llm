"""Diagnostic: can ANY classifier predict memory-vs-compute routing from mixer features?"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "exp28_beta_controller", "features.npz"))
feats_tr = d["feats_tr"]
feats_ev = d["feats_ev"]
lpmix_tr, lpmem_tr = d["lpmix_tr"], d["lpmem_tr"]
lpmix_ev, lpmem_ev = d["lpmix_ev"], d["lpmem_ev"]

y_tr = (lpmem_tr > lpmix_tr).astype(np.float32)
y_ev = (lpmem_ev > lpmix_ev).astype(np.float32)
print(f"labels: train {y_tr.mean():.3f}, eval {y_ev.mean():.3f}")


def auc(labels, scores):
    """Rank-based AUC."""
    order = np.argsort(-scores)
    ranks = np.empty(len(labels))
    ranks[order] = np.arange(1, len(labels) + 1)
    pos = labels == 1
    n_pos = pos.sum()
    n_neg = len(labels) - n_pos
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.ReLU())
        layers.pop()  # remove last ReLU
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# normalize features
mu = feats_tr.mean(0, keepdims=True)
sd = feats_tr.std(0, keepdims=True) + 1e-8
Xtr = torch.tensor((feats_tr - mu) / sd, dtype=torch.float32)
Xev = torch.tensor((feats_ev - mu) / sd, dtype=torch.float32)
Ytr = torch.tensor(y_tr)
Yev = torch.tensor(y_ev)

for dims in [[128, 64, 1], [128, 256, 256, 1]]:
    torch.manual_seed(0)
    m = MLP(dims)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for it in range(6000):
        idx = torch.randint(0, len(Xtr), (1024,))
        logit = m(Xtr[idx]).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, Ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s_tr = torch.sigmoid(m(Xtr)).numpy().reshape(-1)
        s_ev = torch.sigmoid(m(Xev)).numpy().reshape(-1)
    print(f"MLP {dims}: AUC train {auc(y_tr, s_tr):.3f} eval {auc(y_ev, s_ev):.3f}  "
          f"corr eval {np.corrcoef(s_ev, y_ev)[0,1]:.3f}")

# also: how informative is a SINGLE cheap feature — KN hit (context in table)?
# approximate: use the KN logp margin (lpmem - lpmix) as an oracle-ish score
margin = (lpmem_ev - lpmix_ev)
print(f"margin score AUC eval: {auc(y_ev, margin):.3f}")

# oracle PPL attainable if we threshold the margin (calibrated routing)
def ppl_at_threshold(margin, th):
    nll, avg = [], 0.0
    for k in range(len(y_ev)):
        b = 1.0 if margin[k] > th else 0.0
        pm, pk = np.exp(lpmix_ev[k]), np.exp(lpmem_ev[k])
        fused = (1 - b) * pm + b * pk
        nll.append(-np.log(max(fused, 1e-12)))
        avg += b
    return float(np.exp(np.mean(nll))), avg / len(y_ev)

for th in [-1.0, 0.0, 1.0, 2.0, 3.0]:
    p, avg = ppl_at_threshold(margin, th)
    print(f"  margin>={th}: PPL {p:.3f} avg β {avg:.3f}")
