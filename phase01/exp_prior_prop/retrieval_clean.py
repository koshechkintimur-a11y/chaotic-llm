"""retrieval_clean.py — accuracy(L) WITHOUT noise (честный тест сохранения связей).
Исправляет перебор: ТЗ делит accuracy(L) и шум-тест на разные группы.

Usage: python retrieval_clean.py DP [--noise 0.0] [--n-trials 300]
"""
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
for p in (HERE, PHASE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import build_model, VOCAB, W
from exp_memory_selector.experiment import load_chars, build_tokenizer, MAX_TRAIN
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["DP", "DP-fix", "DP-noprop", "DP-rand", "C-cap", "H1", "H2", "PM", "SP"])
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--n-trials", type=int, default=300)
    args = ap.parse_args()

    tok = build_tokenizer()
    V = tok.get_vocab_size()
    train_ids = tok.encode(load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)).ids
    test_ids = tok.encode(load_chars(os.path.join(PHASE, "corpus_test.txt"))).ids

    model = build_model(args.config, vocab=V).to(DEVICE)
    pt = os.path.join(HERE, f"model_{args.config}.pt")
    model.load_state_dict(torch.load(pt, map_location=DEVICE))
    model.eval()

    tr_counts = Counter(train_ids)
    cands = [t for t in range(150, V) if tr_counts.get(t, 0) >= 20]
    rng = np.random.default_rng(7)

    print(f"[{args.config}] noise={args.noise} n_trials={args.n_trials}", flush=True)
    out = {}
    for L in (16, 64, 256, 1024):
        acc, n_t = 0.0, 0
        for _ in range(args.n_trials):
            key = int(rng.choice(cands))
            i = int(rng.integers(W + L, len(test_ids) - 1))
            ctx = list(test_ids[i - W:i])
            kp = W - L
            if kp >= 0:
                ctx[kp] = key
            if args.noise > 0:
                n_noise = int(W * args.noise)
                for ni in rng.choice(W, size=n_noise, replace=False):
                    if ni != kp:
                        ctx[ni] = int(rng.integers(1, 100))
            with torch.no_grad():
                logits = model(torch.tensor([ctx], dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
            n_t += 1
            if int(np.argmax(logits)) == key:
                acc += 1.0
        out[L] = round(acc / max(1, n_t), 4)
        print(f"  L={L}: acc={out[L]}", flush=True)

    json.dump({"config": args.config, "noise": args.noise, "retrieval": out},
              open(os.path.join(HERE, f"retrieval_clean_{args.config}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
