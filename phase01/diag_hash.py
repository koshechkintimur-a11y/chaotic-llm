"""Quick diagnostic: is ProbeTable faithful on real code tokens?"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


def make_bpe(text, vocab_size):
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


class ProbeTable:
    def __init__(self, n_slots, order, bits_per_tok):
        self.n = int(n_slots)
        self.order = order
        self.bpt = bits_per_tok
        self.keys = np.full(self.n, -1, dtype=np.int64)
        self.counts = np.zeros(self.n, dtype=np.int64)
        self.n_distinct = 0

    def _pack(self, ngram):
        k = 0
        for t in ngram:
            k = (k << self.bpt) | int(t)
        return k

    @staticmethod
    def _mix(k):
        k = (k ^ (k >> 33)) & 0xFFFFFFFFFFFFFFFF
        k = (k * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        return (k ^ (k >> 33))

    def insert(self, ngram, count=1):
        key = self._pack(ngram)
        s = self._mix(key) % self.n
        while self.keys[s] != -1 and self.keys[s] != key:
            s = (s + 1) % self.n
        if self.keys[s] == -1:
            if self.n_distinct >= self.n:
                return False
            self.keys[s] = key
            self.n_distinct += 1
        self.counts[s] += count
        return True

    def lookup(self, ngram):
        key = self._pack(ngram)
        s = self._mix(key) % self.n
        while self.keys[s] != -1:
            if self.keys[s] == key:
                return self.counts[s]
            s = (s + 1) % self.n
        return 0


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), 2_000_000)
tok = make_bpe(train_text, 512)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
print("V", V, "tokens", len(train_ids))

ORDER = 3
dict_cnt = defaultdict(int)
for i in range(ORDER, len(train_ids)):
    dict_cnt[tuple(train_ids[i - ORDER:i])] += 1
print("dict distinct:", len(dict_cnt))

# subset for quick test: every 4th context
subset = {}
for i, (g, c) in enumerate(dict_cnt.items()):
    if i % 4 == 0:
        subset[g] = c
print("subset size:", len(subset))

tab = ProbeTable(len(subset) * 3 + 10, ORDER, 9)
for g, c in subset.items():
    ok = tab.insert(g, c)
    assert ok

bad = 0
for g, c in subset.items():
    l = tab.lookup(g)
    if l != c:
        bad += 1
        if bad <= 8:
            print("MISMATCH", g, "dict", c, "hash", l)
print("mismatches:", bad, "/", len(subset))
