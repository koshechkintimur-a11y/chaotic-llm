"""night_transformer.py — НОЧНОЙ МАТЧ: TransformerLM на той же подвыборке Stack,
что и night_sts_prog.py (100M токенов), тот же протокол. Для честного сравнения.

Автономный, копирует модели/данные. Подбор D под 1.4M (как у STS-Prog), 8 слоёв.
"""
import os, sys, time, json, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
STACK_PATH = os.path.join(PHASE, "corpus_stack_train.txt")
W = 256
VOCAB = 512
BATCH = 32
STEPS = 10000
LR = 5e-4
WARMUP = 1000
N_EVAL = 2000
SUB = 100_000_000
BUDGET = 1_446_145  # как у STS-Prog night

def load_stack_text(path, max_bytes=None):
    with open(path, "r", encoding="utf-8") as f:
        return f.read(max_bytes)

def make_bpe(text, vocab=VOCAB):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["[PAD]"])
    tok.train_from_iterator([text], trainer)
    tok.enable_padding(length=None)
    return tok

def build_order3(ids, ORDER=3):
    from collections import defaultdict
    prior = defaultdict(dict)
    for i in range(ORDER - 1, len(ids)):
        ctx = tuple(int(t) for t in ids[i - ORDER + 1:i])
        prior[ctx][int(ids[i])] = prior[ctx].get(int(ids[i]), 0) + 1
    return prior

def gated_ppl(lpm, targets, prior, ctx_tokens):
    N = len(lpm)
    nll = np.zeros(N)
    for k in range(N):
        ctx = ctx_tokens[k]
        table = prior.get(ctx)
        if table:
            tot = sum(table.values())
            c = table.get(int(targets[k]), 0)
            if c > 0:
                beta = tot / (tot + 1.0)
                nll[k] = -np.logaddexp(np.log1p(-beta) + lpm[k, int(targets[k])], np.log(beta) + np.log(c / tot))
                continue
        nll[k] = -lpm[k, int(targets[k])]
    return float(np.exp(np.mean(nll)))

def induction_retrieval(model, test_ids, distances=(16, 64, 128, 256), n_trials=200):
    model.eval()
    rng = np.random.default_rng(42)
    res = {}
    for L in distances:
        hits, miss = 0, 0
        for _ in range(n_trials):
            i = int(rng.integers(L + 2, len(test_ids) - L - 3))
            A = int(test_ids[i]); B = int(test_ids[i + 1])
            j = i + L
            if test_ids[j - 1] != A:
                continue
            window = test_ids[j - W:j]
            X = torch.tensor([window], dtype=torch.long, device="cuda")
            with torch.no_grad():
                logits = model(X)
            if int(logits[0].argmax().item()) == B:
                hits += 1
            miss += 1
        res[f"L{L}"] = hits / max(1, miss)
    return res

# ============ TransformerLM (из parametric_models.py, автономно) ============
class TFBlock(nn.Module):
    def __init__(self, D, H, W):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.norm1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.norm2 = nn.LayerNorm(D)
        self.register_buffer("mask", torch.triu(torch.full((W, W), float("-inf")), diagonal=1))
    def forward(self, x):
        a, _ = self.attn(x, x, x, attn_mask=self.mask)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab, W, D, HEADS=4, LAYERS=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, D)
        self.pos = nn.Parameter(torch.randn(1, W, D) * 0.02)
        self.blocks = nn.ModuleList([TFBlock(D, HEADS, W) for _ in range(LAYERS)])
        self.head = nn.Linear(D, vocab)
    def forward(self, x):
        h = self.embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, -1, :])

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def tf_params_analytic(V, W, D, layers=8, heads=4):
    embed = V * D
    pos = W * D
    attn = 4 * D * D + 4 * D
    ln = 4 * D
    ffn = 8 * D * D + 5 * D
    per_block = attn + ln + ffn
    head = D * V
    return embed + pos + layers * per_block + head

def pick_tf_dims(budget, V, W, layers=8, heads=4):
    lo, hi = 16, 1024
    while lo < hi:
        mid = (lo + hi) // 2
        mid = max(heads, (mid // heads) * heads)
        mid = max(mid, lo)
        if tf_params_analytic(V, W, mid, layers, heads) < budget:
            lo = mid + 1
        else:
            hi = mid
    best = max(heads, (lo // heads) * heads)
    return best

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("Loading corpus (подвыборка ~215MB)...", flush=True)
    max_bytes = int(SUB * 2.1) + 5_000_000
    text = load_stack_text(STACK_PATH, max_bytes)
    tok = make_bpe(text[:10_000_000])
    V = tok.get_vocab_size()
    CH = 5_000_000
    chunks = []
    for i in range(0, len(text), CH):
        chunks.append(np.array(tok.encode(text[i:i + CH]).ids, dtype=np.int32))
    ids = np.concatenate(chunks)
    del text, chunks
    train_ids = ids[:SUB]
    test_ids = ids[SUB:SUB + SUB // 10]
    print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}", flush=True)

    D = pick_tf_dims(BUDGET, V, W, layers=8, heads=4)
    model = TransformerLM(V, W, D=D, HEADS=4, LAYERS=8).to("cuda")
    nparam = count_params(model)
    print(f"Transformer: D={D} L=8 H=4 params={nparam:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(1)
    N = len(train_ids) - W - 1
    t0 = time.time()
    for step in range(1, STEPS + 1):
        if step < WARMUP:
            for pg in opt.param_groups:
                pg["lr"] = LR * step / WARMUP
        else:
            for pg in opt.param_groups:
                pg["lr"] = LR
        s = rng.integers(0, N, size=BATCH)
        X = np.stack([train_ids[i:i + W] for i in s])
        Y = np.array([train_ids[i + W] for i in s])
        Xt = torch.tensor(X, dtype=torch.long, device="cuda")
        Yt = torch.tensor(Y, dtype=torch.long, device="cuda")
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(Xt)
            loss = lossf(logits, Yt)
        if not torch.isfinite(loss):
            print(f"NaN at {step}, abort", flush=True)
            break
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 2000 == 0:
            print(f"  [{step}/{STEPS}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"Training done in {dt:.0f}s", flush=True)

    model.eval()
    rng2 = np.random.default_rng(42)
    te_starts = np.sort(rng2.choice(len(test_ids) - W - 1, size=N_EVAL, replace=False))
    logits_te = np.zeros((N_EVAL, V), dtype=np.float32)
    y_te = np.zeros(N_EVAL, dtype=np.int64)
    ctx_tokens = []
    with torch.no_grad():
        for k, s in enumerate(te_starts):
            Xt = torch.tensor([test_ids[s:s + W]], dtype=torch.long, device="cuda")
            logits_te[k] = model(Xt)[0].cpu().numpy()
            y_te[k] = test_ids[s + W]
            ctx_tokens.append(tuple(test_ids[s + W - 2:s + W]))
    lpm = torch.log_softmax(torch.tensor(logits_te), -1).numpy()
    mixer_ppl = float(np.exp(np.mean([-lpm[k, y_te[k]] for k in range(N_EVAL)])))
    prior = build_order3(train_ids[:500_000])
    gated = gated_ppl(lpm, y_te, prior, ctx_tokens)
    retrieval = induction_retrieval(model, test_ids)
    print(f"[night-tf] mixer={mixer_ppl:.3f} gated={gated:.3f} retrieval={retrieval} params={nparam:,} time={dt:.0f}s", flush=True)
    res = {"kind": "transformer_night", "d": D, "layers": 8, "params": nparam,
           "mixer_ppl": round(mixer_ppl, 3), "gated_ppl": round(gated, 3),
           "retrieval": {str(k): round(v, 3) for k, v in retrieval.items()},
           "time_s": round(dt, 1)}
    with open(os.path.join(HERE, "results_night_tf.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("saved results_night_tf.json", flush=True)

if __name__ == "__main__":
    main()