"""analyze_eye.py — Q3: does the Eye actually LOOK? (visualize on KEY example)

Loads trained E1/E2 models and, on a synthetic KEY example
[noise... KEY ...noise] (noise = small ids 1..10, KEY = standout id 100..511),
shows where the Eye points.

E1 (modulator): prints route-weight vector per position + per-position entropy;
                 KEY should get a peaked (low-entropy) weight vs flat noise.
E2 (grouping):   prints cluster assignment; KEY should land in a cluster with
                 other standout tokens, not with noise.

Saves analyze_key.json. No plotting deps (Windows-safe text tables).

Usage:  python analyze_eye.py
"""
import os, sys, json, random
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import (EyeModulatorLM, EyeGroupLM, D, W, R, K)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEY_ID = 420  # standout id, outside noise range 1..10 (vocab 512)


def make_key_example(key_id=KEY_ID, seed=0, n_standout=4):
    """Deterministic KEY example: [NOISE ... KEY ... NOISE].
    Fixed structure: 40% noise, the KEY at a FIXED position, then more noise,
    plus a few other standout 'question' tokens near the KEY.
    Noise ids in 1..10; KEY and questions in 100..511 (standout)."""
    rng = random.Random(seed)
    n_pre = W // 3                      # fixed-ish pre-buffer
    n_post = W - n_pre - 1 - n_standout
    pre = [rng.randint(1, 10) for _ in range(n_pre)]
    post = [rng.randint(1, 10) for _ in range(n_post)]
    questions = [rng.randint(100, 200) for _ in range(n_standout)]
    body = pre + [key_id] + questions + post
    body = body[:W]
    body = [1] * (W - len(body)) + body     # left-pad to W
    keypos = body.index(key_id)
    return body, keypos, key_id


def _print_window(ids, keypos, rows, header):
    """Print a readable window: 8 before KEY, KEY row, 8 after KEY."""
    lo = max(0, keypos - 8); hi = min(W, keypos + 9)
    print(f"\n=== {header} (window around KEY, pos {keypos}, tok {ids[keypos]}) ===")
    print("  ".join(f"{h:>7}" for h in header.split()))
    for i in range(lo, hi):
        mark = "K" if i == keypos else " "
        print(f"{i:>4} {ids[i]:>4} {mark:>3}  " + rows[i])
    print(f"(omitted {lo} leading + {W-hi} trailing positions of W={W})")


def viz_E1(model, ids, keypos, keyid):
    model.eval()
    x = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        model(x)
        w = model._w[0].cpu().numpy()            # [W, R]
    ent = -(w * np.log(w + 1e-9)).sum(-1)
    rows = []
    for i in range(W):
        rows.append("  ".join(f"{w[i,r]:.2f}" for r in range(R)) + f"   {ent[i]:.2f}")
    _print_window(ids, keypos, rows, "pos  tok KEY  " + "  ".join(f"r{r}" for r in range(R)) + "   ent")
    peak = w[keypos].argmax()
    noise_ent = ent[[i for i in range(W) if i != keypos]].mean()
    print(f"\nKEY pos {keypos} (tok {keyid}) -> peaked route r{peak} "
          f"(w={w[keypos,peak]:.2f}, ent={ent[keypos]:.2f})")
    print(f"avg entropy over NOISE positions: {noise_ent:.2f}")
    return {"key_pos": keypos, "key_weights": w[keypos].tolist(),
            "key_entropy": float(ent[keypos]), "avg_noise_entropy": float(noise_ent)}


def viz_E2(model, ids, keypos, keyid):
    model.eval()
    x = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        model(x)
        assign = model._assign[0].cpu().numpy()
        logits = model._logits[0].cpu().numpy()
    rows = []
    clusters = {}
    for i in range(W):
        c = int(assign[i]); clusters.setdefault(c, []).append(i)
        rows.append(f"  {c:>3}  {logits[i].max():.2f}")
    _print_window(ids, keypos, rows, "pos  tok KEY   clus  lmax")
    kc = int(assign[keypos]); members = [ids[m] for m in clusters.get(kc, [])]
    n_stand = sum(1 for t in members if 100 <= t <= 511)
    n_noise = sum(1 for t in members if 1 <= t <= 10)
    print(f"\nKEY (pos {keypos}, tok {keyid}) -> cluster {kc}")
    print(f"cluster {kc} size={len(members)}; standout members={n_stand}, noise members={n_noise}")
    return {"key_pos": keypos, "key_cluster": kc, "cluster_members": members,
            "cluster_size": len(members), "standout": n_stand, "noise": n_noise}


def _tok_type(t):
    if 100 <= t <= 511:
        return "S"          # standout (KEY or question)
    return "."              # noise (1..10)


def viz_E2_clusters(model, ids, keypos, keyid):
    """Show cluster composition: which tokens (KEY/question/noise) land together."""
    model.eval()
    x = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        model(x)
        assign = model._assign[0].cpu().numpy()
    clusters = {}
    for i in range(W):
        clusters.setdefault(int(assign[i]), []).append(i)
    print(f"\n=== E2 clusters (KEY id={keyid} pos {keypos}) ===")
    print(f"{'cl':>3} {'size':>4}  {'S':>2} {'n':>3}  members (tok[type])")
    for c in sorted(clusters):
        mem = clusters[c]
        toks = [ids[m] for m in mem]
        n_s = sum(1 for t in toks if _tok_type(t) == "S")
        n_n = len(mem) - n_s
        # show up to 12 members; mark KEY position
        shown = []
        for m in mem[:12]:
            tag = "K" if m == keypos else _tok_type(ids[m])
            shown.append(f"{ids[m]}{tag}")
        tail = f" ...+{len(mem)-12}" if len(mem) > 12 else ""
        mark = " <-- KEY" if keypos in mem else ""
        print(f"{c:>3} {len(mem):>4}  {n_s:>2} {n_n:>3}  " + ", ".join(shown) + tail + mark)
    # detailed KEY cluster
    kc = int(assign[keypos])
    mem = clusters[kc]
    key_with = [(ids[m], "K" if m == keypos else _tok_type(ids[m])) for m in mem]
    print(f"\nKEY cluster {kc} ({len(mem)} tokens): " +
          ", ".join(f"{t}{tag}" for t, tag in key_with))
    return {"key_cluster": kc, "cluster_sizes": {c: len(clusters[c]) for c in clusters},
            "key_cluster_members": key_with}


def main():
    ids, keypos, keyid = make_key_example()
    print(f"KEY example: KEY id={keyid} at pos {keypos}, W={W} "
          f"(noise=ids 1..10, KEY/question=100..511)")
    res = {}
    for cfg, cls, kw in [("E1_l", EyeModulatorLM, {"mode": "l"}),
                         ("E1_u", EyeModulatorLM, {"mode": "u"}),
                         ("E2_l", EyeGroupLM, {"mode": "l"}),
                         ("E2_r", EyeGroupLM, {"mode": "r"})]:
        fn = os.path.join(HERE, f"model_{cfg}.pt")
        if not os.path.exists(fn):
            print(f"\n[skip] {cfg}: {fn} not found"); continue
        m = cls(**kw).to(DEVICE)
        try:
            m.load_state_dict(torch.load(fn, map_location=DEVICE, weights_only=False))
        except Exception as ex:
            print(f"\n[skip] {cfg}: load failed ({ex.__class__.__name__}); "
                  f"likely architecture mismatch (re-run that config with current models.py)")
            continue
        res[cfg] = viz_E1(m, ids, keypos, keyid) if cfg.startswith("E1") else viz_E2(m, ids, keypos, keyid)
        if cfg.startswith("E2"):
            res[cfg + "_clusters"] = viz_E2_clusters(m, ids, keypos, keyid)
    with open(os.path.join(HERE, "analyze_key.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved analyze_key.json")


if __name__ == "__main__":
    main()
