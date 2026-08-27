"""exp11_bpe_scaling.py — Phase 0.1, Experiment 11 (Step 2: BPE + scaling).

BPE vocabulary, full corpus, matched parameter budgets for ChaoticLM vs AttnLM.
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp11_bpe_scaling")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W_LOCAL = 64
W = 256
N_WINDOWS = W // W_LOCAL
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
BATCH = 64
MAX_TRAIN_BYTES = 2_000_000
EPOCHS = 1


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


# ============ BPE tokenizer (via tokenizers lib) ============
def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    # train on in-memory text
    tok.train_from_iterator([text[i:i+100000] for i in range(0, len(text), 100000)],
                            trainer=trainer)
    return tok


def make_batches(ids, W, bs, rng=np.random.default_rng(0)):
    n = len(ids) - W - 1
    idx = np.random.permutation(n)[: (n // bs) * bs].reshape(-1, bs)
    for bi in idx:
        X = np.stack([ids[i:i + W] for i in bi])
        Y = [ids[i + W] for i in bi]
        yield torch.tensor(X), torch.tensor(Y)


# ============ models ============
class HierChaoticLM(nn.Module):
    def __init__(self, V, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long)
                       for t in range(1, bl + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long)
                       for t in range(1, bg + 1)}
        self.readout = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(), nn.Linear(d, V))

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, x):
        B, W = x.shape
        d = self.d
        h = self.embed(x) + self.pos
        hw = h.view(B, self.Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l)
                           for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        gvec = glob.mean(dim=1, keepdim=True)
        last = loc[:, -1, -1, :]
        return self.readout(torch.cat([last, gvec.squeeze(1)], dim=-1))


class AttnLM(nn.Module):
    def __init__(self, V, W, d):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, x, use_global=True):
        B, W = x.shape
        h = self.embed(x) + self.pos
        qkv = self.qkv(h).reshape(B, W, 3, self.d).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        attn = attn.softmax(-1)
        h = self.norm(h + self.proj(attn @ v))
        return self.readout(h[:, -1, :])


# ============ main ============
train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))

print("Training BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
print(f"BPE vocab: {V}")

train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids
print(f"train tokens: {len(train_ids):,}, test tokens: {len(test_ids):,}")

def evaluate(model, ids, W, n_samples=20000):
    model.eval()
    nll, acc = [], 0
    with torch.no_grad():
        for i in range(0, len(ids) - W - 1, 32):
            ctx = ids[i:i + W]
            y = ids[i + W]
            logits = model(torch.tensor(ctx, dtype=torch.long).unsqueeze(0))
            logp = torch.log_softmax(logits[0], -1).numpy()
            nll.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc += 1
            if len(nll) >= n_samples:
                break
    n = len(nll)
    model.train()
    return {"n": n, "ppl": float(np.exp(np.mean(nll))), "acc": acc / n}

results = {"V": V, "W": W, "W_local": W_LOCAL, "d": D_MODEL,
           "train_tokens": len(train_ids), "test_tokens": len(test_ids)}

print("\n=== Hierarchical ChaoticLM ===")
hcm = HierChaoticLM(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
print(f"params: {sum(p.numel() for p in hcm.parameters()):,}")
ckpt_h = os.path.join(OUT, "hier.pt")
if os.path.exists(ckpt_h):
    hcm.load_state_dict(torch.load(ckpt_h))
    print("loaded checkpoint")
else:
    opt = torch.optim.Adam(hcm.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for step, (X, Y) in enumerate(make_batches(train_ids, W, BATCH)):
        opt.zero_grad()
        loss = lossf(hcm(X), Y)
        loss.backward()
        opt.step()
        if step % 2000 == 0 and step > 0:
            print(f"  [hier {step:,} steps] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    torch.save(hcm.state_dict(), ckpt_h)
r_hcm = evaluate(hcm, test_ids, W)
print(f"HierChaoticLM: {r_hcm}")
results["hierarchical"] = r_hcm

print("\n=== AttnLM ===")
am = AttnLM(V, W, D_MODEL)
print(f"params: {sum(p.numel() for p in am.parameters()):,}")
opt = torch.optim.Adam(am.parameters(), lr=1e-3)
for step, (X, Y) in enumerate(make_batches(train_ids, W, BATCH)):
    opt.zero_grad()
    loss = lossf(am(X), Y)
    loss.backward()
    opt.step()
    if step % 2000 == 0 and step > 0:
        print(f"  [attn {step:,} steps] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
r_am = evaluate(am, test_ids, W)
print(f"AttnLM: {r_am}")
results["attention"] = r_am

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)