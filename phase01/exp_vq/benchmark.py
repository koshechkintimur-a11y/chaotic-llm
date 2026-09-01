"""benchmark.py — канонический прогон STS-Prog. Каждый конфиг пишет СВОЙ файл.

Почему этот файл существует
---------------------------
`train()` в experiment_pc.py сохраняет результат в `results_{config}.json`,
а `config` для всех режимов равен `"pc"`. Поэтому `sts_prog` и его абляция
`sts_prog_nopc` писали в один и тот же файл, и последний запуск затирал
предыдущий. Этим объясняется, почему заявленные 19.22 / retrieval 47% не имеют
файла-источника: файл занимает абляция.

Здесь каждый конфиг пишет `benchmark/<name>.json` и в него кладётся всё,
что нужно, чтобы число можно было проверить: режим драйвера, размерность,
глубина, число параметров, сиды, корпус с хэшем, коммит гита, железо, время
и — отдельно — знаменатель retrieval-теста.

Запуск
------
    python benchmark.py --list
    python benchmark.py --dry-run
    python benchmark.py --only sts_prog_d192_l8,nopc_d192_l8
    python benchmark.py --all --device cuda
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, os.path.join(PHASE, "exp_memory_selector"))
sys.path.insert(0, HERE)

from experiment_pc import (  # переиспользуем проверенный код
    load_chars, make_bpe, build_order3, gated_ppl, build_pc_model,
    MAX_TRAIN, W, STEPS, BATCH, LR, WARMUP, N_EVAL,
)

OUT_DIR = os.path.join(HERE, "benchmark")
RETRIEVAL_DISTANCES = (16, 64, 128, 256)

# ---------------------------------------------------------------- матрица прогонов
# Имена — стабильные идентификаторы. Менять нельзя: на них ссылаются таблицы.
MATRIX = [
    dict(name="sts_prog_d192_l8", driver="sts_prog", d=192, layers=8,
         k=1.2, alpha=0.3, aux_w=0.5, aux_mode="multibead",
         corpus="small", steps=STEPS),
    dict(name="nopc_d192_l8", driver="sts_prog_nopc", d=192, layers=8,
         k=1.2, alpha=0.3, aux_w=0.5, aux_mode="multibead",
         corpus="small", steps=STEPS),
    dict(name="transformer_d88_l8", driver="__transformer__", d=88, layers=8,
         heads=4, budget=900_000, corpus="small", steps=STEPS),
    dict(name="sts_prog_stack_d256_l12", driver="sts_prog", d=256, layers=12,
         k=1.2, alpha=0.3, aux_w=0.0, aux_mode="multibead",
         corpus="stack", sub=100_000_000, steps=10000, batch=32, n_eval=2000),
    dict(name="transformer_stack_d116_l8", driver="__transformer__", d=116, layers=8,
         heads=4, budget=1_446_145, corpus="stack", sub=100_000_000,
         steps=10000, batch=32, n_eval=2000),
]

REQUIRED_KEYS = ("name", "driver_mode", "params", "results", "provenance", "retrieval")


# ---------------------------------------------------------------- служебное
def sha256_file(path, limit_bytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            if limit_bytes and read + len(chunk) > limit_bytes:
                h.update(chunk[: limit_bytes - read])
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def git_meta():
    def run(args):
        try:
            out = subprocess.run(args, cwd=PHASE, capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip()
        except Exception:
            return ""
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def env_meta(device):
    meta = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if device != "cpu" and torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["cuda"] = torch.version.cuda
    return meta


# ---------------------------------------------------------------- retrieval с честным знаменателем
def induction_retrieval(model, test_ids, distances=RETRIEVAL_DISTANCES,
                        n_valid_target=200, max_attempts=200, device="cuda"):
    """Индукционный тест A→B. Возвращает по каждой дистанции hits, n_valid и долю.

    Отличие от night_sts_prog.py: там неподошедшие попытки просто отбрасывались
    (continue), из-за чего знаменатель падал до единиц, а 67% оказывалось «2 из 3».
    Здесь есть цикл повтора, как в experiment_pc.py, и знаменатель попадает
    в результат — иначе число нельзя интерпретировать.
    """
    model.eval()
    rng = np.random.default_rng(0)
    out = {}
    for L in distances:
        hits = 0
        n_valid = 0
        for _ in range(n_valid_target):
            found = False
            for _try in range(max_attempts):
                i = int(rng.integers(L + 2, len(test_ids) - L - 2))
                A = int(test_ids[i])
                B = int(test_ids[i + 1])
                j = i + L
                if j < len(test_ids) and test_ids[j - 1] == A:
                    found = True
                    break
            if not found:
                continue
            window = test_ids[j - W:j]
            X = torch.tensor([window], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(X)
            if int(logits[0].argmax().item()) == B:
                hits += 1
            n_valid += 1
        out[f"L{L}"] = {
            "hits": hits,
            "n_valid": n_valid,
            "rate": round(hits / n_valid, 4) if n_valid else None,
        }
    return out


# ---------------------------------------------------------------- данные
def build_data_small():
    path = os.path.join(PHASE, "corpus_train.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} не найден. Корпус в .gitignore — перед прогоном его нужно "
            f"положить локально или собрать build_corpus.py."
        )
    train_text = load_chars(path, MAX_TRAIN)
    test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
    tok = make_bpe(train_text)
    V = tok.get_vocab_size()
    train_ids = np.array(tok.encode(train_text).ids, dtype=np.int64)
    test_ids = np.array(tok.encode(test_text).ids, dtype=np.int64)
    meta = {"name": "corpus_train.txt", "path": os.path.relpath(path, PHASE),
            "sha256_8mb": sha256_file(path, 8 << 20),
            "n_train_tokens": int(len(train_ids)),
            "n_test_tokens": int(len(test_ids)), "vocab": int(V)}
    return tok, V, train_ids, test_ids, meta


def build_data_stack(sub):
    path = os.path.join(PHASE, "corpus_stack_train.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} не найден (2 ГБ, в .gitignore). Скачай Stack или "
            f"ограничься конфигами на малом корпусе."
        )
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    max_bytes = int(sub * 2.1) + 5_000_000
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(max_bytes)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=["[PAD]"])
    tok.train_from_iterator([text[:10_000_000]], trainer)
    V = tok.get_vocab_size()
    chunks = []
    for i in range(0, len(text), 5_000_000):
        chunks.append(np.array(tok.encode(text[i:i + 5_000_000]).ids, dtype=np.int64))
    ids = np.concatenate(chunks)
    train_ids = ids[:sub]
    test_ids = ids[sub:sub + sub // 10]
    meta = {"name": "corpus_stack_train.txt", "path": os.path.relpath(path, PHASE),
            "sha256_8mb": sha256_file(path, 8 << 20),
            "n_train_tokens": int(len(train_ids)),
            "n_test_tokens": int(len(test_ids)), "vocab": int(V),
            "subsample_tokens": int(sub)}
    return tok, V, train_ids, test_ids, meta


# ---------------------------------------------------------------- модели
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


def pick_tf_dims(budget, V, W, layers=8, heads=4):
    def params(D):
        per_block = (4 * D * D + 4 * D) + (4 * D) + (8 * D * D + 5 * D)
        return V * D + W * D + layers * per_block + D * V
    best, best_gap = heads, None
    for D in range(heads, 1024, heads):
        gap = budget - params(D)
        if gap >= 0 and (best_gap is None or gap < best_gap):
            best, best_gap = D, gap
    return best


# ---------------------------------------------------------------- обучение
def aux_term(model, X, aux_mode):
    """Вспомогательная потеря локализации повтора (копия логики experiment_pc)."""
    sim = model._last_sim
    if sim is None:
        return None, None
    last_tok = X[:, -1, None]
    pos_mask = (X == last_tok).float()
    pos_mask[:, W - 8:] = 0.0
    pos_sum = pos_mask.sum(dim=1, keepdim=True)
    valid = (pos_sum > 0).squeeze(1)
    if not bool(valid.any()):
        return None, None
    pos_idx = torch.arange(W, device=X.device).float().unsqueeze(0)
    dist = (pos_idx - (W - 9)).abs() + (1.0 - pos_mask) * 1e4
    nearest = dist.argmin(dim=1)
    gauss = torch.exp(-(pos_idx - nearest.unsqueeze(1)).pow(2) / (2 * 2.0 ** 2)) * pos_mask
    uniform = pos_mask / pos_sum.clamp(min=1e-6)
    log_sm = torch.log_softmax(sim, dim=1)
    if aux_mode == "multibead":
        g = gauss / gauss.sum(dim=1, keepdim=True).clamp(min=1e-6)
        local = -(g * log_sm).sum(dim=1)[valid].mean()
        glob = -(uniform * log_sm).sum(dim=1)[valid].mean()
        return 0.5 * local + 0.5 * glob, valid
    if aux_mode == "uniform":
        tgt = uniform
    else:
        tgt = gauss / gauss.sum(dim=1, keepdim=True).clamp(min=1e-6)
    tgt = tgt * pos_mask
    tgt = tgt / tgt.sum(dim=1, keepdim=True).clamp(min=1e-6)
    tgt[~valid] = 0.0
    return -(tgt * log_sm).sum(dim=1)[valid].mean(), valid


def train_one(cfg, device, out_dir, save_ckpt=True):
    name = cfg["name"]
    steps = cfg.get("steps", STEPS)
    batch = cfg.get("batch", BATCH)
    n_eval = cfg.get("n_eval", N_EVAL)
    is_tf = cfg["driver"] == "__transformer__"

    if cfg.get("corpus") == "stack":
        tok, V, train_ids, test_ids, corpus_meta = build_data_stack(cfg["sub"])
    else:
        tok, V, train_ids, test_ids, corpus_meta = build_data_small()

    if is_tf:
        D = cfg.get("d") or pick_tf_dims(cfg["budget"], V, W, cfg["layers"], cfg.get("heads", 4))
        model = TransformerLM(V, W, D=D, HEADS=cfg.get("heads", 4), LAYERS=cfg["layers"]).to(device)
    else:
        D = cfg["d"]
        model = build_pc_model("pc", V, d=D, alpha=cfg.get("alpha", 0.3),
                               k_init=cfg["k"], sync_steps=cfg.get("sync_steps", 1),
                               driver_mode=cfg["driver"], temp=cfg.get("temp", 0.3),
                               layers=cfg["layers"]).to(device)

    nparam = sum(p.numel() for p in model.parameters())
    print(f"[{name}] driver={cfg['driver']} d={D} L={cfg['layers']} "
          f"params={nparam:,} V={V} steps={steps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1)
    n = len(train_ids) - W - 1
    t0 = time.time()

    for step in range(1, steps + 1):
        if step < WARMUP:
            for pg in opt.param_groups:
                pg["lr"] = LR * step / WARMUP
        model.train()
        s = rng.integers(0, n, size=batch)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in s]), dtype=torch.long, device=device)
        Y = torch.tensor(np.array([train_ids[i + W] for i in s]), dtype=torch.long, device=device)
        logits = model(X)
        loss = lossf(logits, Y)
        if not is_tf and cfg.get("aux_w", 0.0) > 0:
            aux, _ = aux_term(model, X, cfg.get("aux_mode", "multibead"))
            if aux is not None:
                loss = loss + cfg["aux_w"] * aux
        if not torch.isfinite(loss):
            print(f"[{name}] нечисловая потеря на шаге {step}, прерываю", flush=True)
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0:
            print(f"  [{step}/{steps}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    time_s = round(time.time() - t0, 1)

    model.eval()
    rng2 = np.random.default_rng(42)
    te_starts = np.sort(rng2.choice(len(test_ids) - W - 1, size=n_eval, replace=False))
    logits_te = np.zeros((n_eval, V), dtype=np.float32)
    y_te = np.zeros(n_eval, dtype=np.int64)
    with torch.no_grad():
        for k, s in enumerate(te_starts):
            Xt = torch.tensor([test_ids[s:s + W]], dtype=torch.long, device=device)
            logits_te[k] = model(Xt)[0].float().cpu().numpy()
            y_te[k] = test_ids[s + W]
    lpm = torch.log_softmax(torch.tensor(logits_te), -1).numpy()
    mixer_ppl = float(np.exp(np.mean([-lpm[k, y_te[k]] for k in range(n_eval)])))
    prior = build_order3(train_ids[:500_000])
    ctx_tokens = [tuple(test_ids[s + W - 2:s + W]) for s in te_starts]
    gated = gated_ppl(lpm, y_te, prior, V, ctx_tokens)
    retrieval = induction_retrieval(model, test_ids, device=device)

    if save_ckpt:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"{name}.pt"))

    payload = {
        "name": name,
        "driver_mode": cfg["driver"],
        "model": "TransformerLM" if is_tf else "PurePCLM",
        "d": int(D),
        "layers": int(cfg["layers"]),
        "vocab": int(V),
        "params": int(nparam),
        "results": {
            "mixer_ppl": round(mixer_ppl, 3),
            "gated_ppl": round(gated, 3),
            "n_eval": int(n_eval),
        },
        "retrieval": retrieval,
        "provenance": {
            "steps": int(steps),
            "batch": int(batch),
            "lr": LR,
            "warmup": WARMUP,
            "grad_clip": 1.0,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "seed_data": 1,
            "seed_eval": 42,
            "seed_retrieval": 0,
            "aux_w": cfg.get("aux_w", 0.0),
            "aux_mode": cfg.get("aux_mode"),
            "time_s": time_s,
            "corpus": corpus_meta,
            "git": git_meta(),
            "env": env_meta(device),
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[{name}] mixer={mixer_ppl:.3f} gated={gated:.3f} "
          f"retrieval={ {k: v['rate'] for k, v in retrieval.items()} }", flush=True)
    print(f"[{name}] записано {path}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--no-ckpt", action="store_true")
    args = ap.parse_args()

    if args.list:
        for c in MATRIX:
            print(f"  {c['name']:32s} driver={c['driver']:16s} "
                  f"d={c.get('d','auto'):>4} L={c['layers']} corpus={c.get('corpus')}")
        return

    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        selected = [c for c in MATRIX if c["name"] in want]
        missing = set(want) - {c["name"] for c in selected}
        if missing:
            sys.exit(f"нет таких конфигов: {sorted(missing)}")
    elif args.all:
        selected = MATRIX
    else:
        sys.exit("нужно --all, --only <имена> или --list")

    if args.dry_run:
        print(f"устройство: {args.device}")
        for c in selected:
            print(f"  будет выполнен {c['name']}: {c}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for c in selected:
        train_one(c, args.device, args.out_dir, save_ckpt=not args.no_ckpt)


if __name__ == "__main__":
    main()
