"""exp13_two_selectors.py — Phase 0.1, Experiment 13 (Step 4).

TWO SELECTORS TOGETHER: the full architecture v0.2 final form.

  Hierarchical chaotic mixing (cheap proposer, O(W log W))
    + attention readout at the query (content selector, O(W·d))
    + beta corpus-prior (statistical selector, O(1))

vs the same selectors on a full-attention backbone.

Also re-evaluates the exp11/12 HierChaoticLM on 20k samples for a fair table.
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
OUT = os.path.join(HERE, "exp13_two_selectors")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W_LOCAL = 64
W = 256
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
BATCH = 64
MAX_TRAIN_BYTES = 2_000_000
ORDER = 2
BETA = 0.3
TRAIN_STEPS = 12000


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))


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


# ============ beta corpus-prior ============
print("Building beta prior...")
prior = defaultdict(lambda: defaultdict(int))
for i in range(ORDER, len(train_ids)):
    prior[tuple(train_ids[i - ORDER:i])][train_ids[i]] += 1
prior = {k: dict(v) for k, v in prior.items()}
print(f"prior contexts: {len(prior):,}")

def prior_logp(ctx):
    for back in range(ORDER, 0, -1):
        table = prior.get(tuple(ctx[-back:]))
        if table:
            tot = sum(table.values())
            out = np.full(V, -1e9, dtype=np.float64)
            for t, c in table.items():
                out[t] = np.log(c / tot)
            return out
    return None


# ============ models ============
class HierChaoticLM(nn.Module):
    """Returns final states h (B, W, d) — readout is external."""

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
        # broadcast global context back into each local state
        h = loc.reshape(B, self.Nw * self.Wl, d) + gvec
        return h


class PlainReadout(nn.Module):
    """Last position + global mean -> logits (exp11/12 readout)."""

    def __init__(self, V, W, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, h):
        B, W, d = h.shape
        gvec = h.mean(dim=1)
        return self.net(torch.cat([h[:, -1, :], gvec], dim=-1))


class AttnReadout(nn.Module):
    """Content selector: query attends over all W post-mixing states. O(W·d)."""

    def __init__(self, V, W, d):
        super().__init__()
        self.d = d
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, h):
        B, W, d = h.shape
        q = self.q(h[:, -1:, :])                       # (B, 1, d)
        k = self.kv(h)                                 # (B, W, d)
        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # (B, 1, W)
        out = self.proj(scores.softmax(-1) @ h)        # (B, 1, d)
        qh = h[:, -1:, :] + out
        return self.net(self.norm(qh).squeeze(1))


# ============ evaluate with/without prior ============
def evaluate(mixer, readout, W, n_samples=20000, beta=BETA):
    mixer.eval()
    readout.eval()
    nll_m, nll_p, acc_m, acc_p = [], [], 0, 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            h = mixer(torch.tensor(ctx, dtype=torch.long).unsqueeze(0))
            logits = readout(h)
            logp = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
            nll_m.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc_m += 1
            cp = prior_logp(ctx)
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
    mixer.train()
    readout.train()
    return {"n": n,
            "ppl_model": float(np.exp(np.mean(nll_m))),
            "ppl_prior": float(np.exp(np.mean(nll_p))),
            "ppl_gain": float(np.exp(np.mean(nll_m)) / np.exp(np.mean(nll_p))),
            "acc_model": acc_m / n, "acc_prior": acc_p / n}


def train(mixer, readout, steps=TRAIN_STEPS, tag=""):
    params = list(mixer.parameters()) + list(readout.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    n = len(train_ids) - W - 1
    s = 0
    while s < steps:
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
        Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
        opt.zero_grad()
        h = mixer(X)
        loss = lossf(readout(h), Y)
        loss.backward()
        opt.step()
        s += 1
        if s % 4000 == 0:
            print(f"  [{tag} {s:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")


results = {"V": V, "W": W, "W_local": W_LOCAL, "d": D_MODEL, "order": ORDER,
           "beta": BETA, "train_tokens": len(train_ids), "test_tokens": len(test_ids),
           "train_steps": TRAIN_STEPS}

# ============ A) HierChaoticLM + plain readout (fair 20k eval, reuse exp11 ckpt) ============
mixer_p = HierChaoticLM(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
sd = torch.load(os.path.join(HERE, "exp11_bpe_scaling", "hier.pt"))
sd = {k: v for k, v in sd.items() if not k.startswith("readout.")}
mixer_p.load_state_dict(sd)
ro_plain = PlainReadout(V, W, D_MODEL)
ckpt_plain = os.path.join(OUT, "plain_ro.pt")
if os.path.exists(ckpt_plain):
    ro_plain.load_state_dict(torch.load(ckpt_plain))
    print("plain readout: loaded checkpoint")
else:
    print("\n=== A) HierChaoticLM + plain readout (train readout head) ===")
    train(mixer_p, ro_plain, tag="plain")
    torch.save(ro_plain.state_dict(), ckpt_plain)
r_plain = evaluate(mixer_p, ro_plain, W, n_samples=20000)
print(f"Hier+plain: {r_plain}")
results["hier_plain"] = r_plain

# ============ B) HierChaoticLM + attention readout (two selectors) ============
mixer_a = HierChaoticLM(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
mixer_a.load_state_dict(sd)
ro_attn = AttnReadout(V, W, D_MODEL)
ckpt_attn = os.path.join(OUT, "attn_ro.pt")
if os.path.exists(ckpt_attn):
    ro_attn.load_state_dict(torch.load(ckpt_attn))
    print("attn readout: loaded checkpoint")
else:
    print("\n=== B) HierChaoticLM + attention readout (train readout head) ===")
    train(mixer_a, ro_attn, tag="attn-ro")
    torch.save(ro_attn.state_dict(), ckpt_attn)
r_attn = evaluate(mixer_a, ro_attn, W, n_samples=20000)
print(f"Hier+attn-readout: {r_attn}")
results["hier_attn_readout"] = r_attn

# ============ C) reference: AttnLM (full attention, 1 layer) + both ============
class AttnLM(nn.Module):
    def __init__(self, V, W, d):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, W = x.shape
        h = self.embed(x) + self.pos
        qkv = self.qkv(h).reshape(B, W, 3, self.d).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        attn = attn.softmax(-1)
        return self.norm(h + self.proj(attn @ v))


am = AttnLM(V, W, D_MODEL)
ckpt_attnlm = os.path.join(OUT, "attnlm.pt")
if os.path.exists(ckpt_attnlm):
    am.load_state_dict(torch.load(ckpt_attnlm))
    print("AttnLM: loaded checkpoint")
else:
    print("\n=== C) AttnLM + plain readout (train) ===")
    train(am, ro_plain, tag="attnlm")
    torch.save(am.state_dict(), ckpt_attnlm)
r_al = evaluate(am, ro_plain, W, n_samples=20000)
print(f"AttnLM+plain: {r_al}")
results["attn_plain"] = r_al

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
