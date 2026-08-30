"""download_corpus.py — build big Python-code corpus from codeparrot/github-code.

load_dataset() is broken for this dataset (scripts no longer supported), so we
download the raw parquet shards (data/train-*.parquet, ~290MB each) directly via
hf_hub_download, read with pyarrow, keep only rows where language == 'Python',
and write the source text out until target_mb is reached.

Usage:  python download_corpus.py [target_mb] [out_prefix] [max_shards]
  default: target 400MB (~100M BPE tokens), out 'corpus_stack', up to 60 shards
"""
import os
import sys
import argparse
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "codeparrot/github-code"
N_SHARDS = 1126


def write_shard(idx, target_bytes, f_train, f_test, counters):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rfilename = f"data/train-{idx:05d}-of-{N_SHARDS:05d}.parquet"
    print(f"  shard {idx}/{N_SHARDS} downloading...", flush=True)
    t0 = time.time()
    path = hf_hub_download(REPO_ID, rfilename, repo_type="dataset")
    t1 = time.time()
    tbl = pq.read_table(path, columns=["path", "content"])
    t2 = time.time()
    paths = tbl.column("path").to_pylist()
    content = tbl.column("content").to_pylist()
    done = False
    n_py = 0
    for p, code in zip(paths, content):
        if not p or not str(p).lower().endswith(".py"):
            continue
        if not code or len(code) < 200 or len(code) > 500_000:
            continue
        counters["files"] += 1
        f = f_train if counters["files"] % 100 >= 2 else f_test  # 2% holdout
        f.write(code)
        if not code.endswith("\n"):
            f.write("\n")
        counters["bytes"] += len(code)
        n_py += 1
        if counters["bytes"] >= target_bytes:
            done = True
            break
    print(f"    py_rows={n_py} total={counters['bytes']/1e6:.0f}MB "
          f"dl={t1-t0:.0f}s read={t2-t1:.0f}s", flush=True)
    os.remove(path)  # free disk after each shard
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_mb", nargs="?", type=int, default=400)
    ap.add_argument("out_prefix", nargs="?", default="corpus_stack")
    ap.add_argument("max_shards", nargs="?", type=int, default=60)
    args = ap.parse_args()

    target = args.target_mb * 1_000_000
    train_path = os.path.join(HERE, f"{args.out_prefix}_train.txt")
    test_path = os.path.join(HERE, f"{args.out_prefix}_test.txt")
    f_train = open(train_path, "w", encoding="utf-8", errors="ignore")
    f_test = open(test_path, "w", encoding="utf-8", errors="ignore")
    counters = {"bytes": 0, "files": 0}
    t0 = time.time()
    try:
        for idx in range(args.max_shards):
            done = write_shard(idx, target, f_train, f_test, counters)
            if done:
                break
    finally:
        f_train.close()
        f_test.close()
    mb = counters["bytes"] / 1e6
    print(f"DONE: {counters['files']} python files, {mb:.0f}MB in "
          f"{time.time()-t0:.0f}s", flush=True)
    print(f"  train: {train_path}")
    print(f"  test:  {test_path}")


if __name__ == "__main__":
    main()
