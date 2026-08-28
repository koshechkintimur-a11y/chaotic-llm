"""exp23_fetch_nl_corpus.py — download WikiText-2 (natural language corpus)
for the β-law test. Saves nl_corpus_train.txt / nl_corpus_test.txt.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nl_corpus")
os.makedirs(OUT, exist_ok=True)

from datasets import load_dataset

print("loading wikitext-2-raw...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join(ds["text"])
print(f"train chars: {len(text):,}")

# validation split from the dataset
dsv = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
tval = "\n".join(dsv["text"])
print(f"val chars: {len(tval):,}")

with open(os.path.join(OUT, "nl_corpus_train.txt"), "w", encoding="utf-8") as f:
    f.write(text)
with open(os.path.join(OUT, "nl_corpus_test.txt"), "w", encoding="utf-8") as f:
    f.write(tval)
print("saved to", OUT)
