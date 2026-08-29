"""memory.py — MEM experiment: memory as content selector (lookup, no gradients).

Forms:
  MEM-0  baseline order-3 n-gram (current beta-prior)
  MEM-A  order-8 + hashing
  MEM-B  content-addressable (patterns length 8..16, hashed, k-nearest vote)
  MEM-C  hybrid (local order-3 + global content-addressable)
  MEM-R  random memory (control: random dist at each ctx)
  MEM-NoMixer  memory only (no mixer) -- ablation of mixer contribution

All memories expose:
  build(train_ids)   -> constructs lookup tables (O(N) on train)
  query(ctx)        -> (counts: dict[tok->count], total: int) or None
  size_bytes()      -> memory footprint
  n_entries()       -> number of stored patterns

The mixing with the neural mixer follows experiment.py's adaptive-beta NLL:
  beta = tot/(tot + kb);  nll = -logaddexp(log1p(-beta)+pm, log(beta)+log(c/tot))
"""
import hashlib
import math
import random
from collections import defaultdict, Counter


def _h(ctx_tuple):
    """Stable hash of a context tuple -> hex string (tokens up to 65535)."""
    return hashlib.blake2b(b"".join(t.to_bytes(2, "little") for t in ctx_tuple),
                           digest_size=8).hexdigest()


class BaseMemory:
    def __init__(self, order=3, min_count=2):
        self.order = order
        self.min_count = min_count
        self.ctx_len = order            # context length needed for query
        self.table = {}          # hash -> Counter(next_tok)
        self._size = 0

    def build(self, train_ids):
        self.table.clear()
        n = len(train_ids)
        for i in range(self.order, n):
            ctx = tuple(train_ids[i - self.order:i])
            w = train_ids[i]
            h = _h(ctx)
            c = self.table.get(h)
            if c is None:
                c = Counter()
                self.table[h] = c
            c[w] += 1
        # filter rare
        self.table = {h: c for h, c in self.table.items()
                      if sum(c.values()) >= self.min_count}
        self._size = sum(len(c) for c in self.table.values())
        return self

    def query(self, ctx):
        if len(ctx) < self.order:
            return None
        h = _h(tuple(ctx[-self.order:]))
        c = self.table.get(h)
        if c is None:
            return None
        return c, sum(c.values())

    def size_bytes(self):
        # rough: hash(8B) + entries * (tok 2B + count 4B)
        return 8 * len(self.table) + 6 * self._size

    def n_entries(self):
        return len(self.table)


class Order8Memory(BaseMemory):
    """MEM-A: order-8 with hashing."""
    def __init__(self, order=8, min_count=3):
        super().__init__(order=order, min_count=min_count)


class ContentAddressableMemory:
    """MEM-B: store patterns length 8..16, query k-nearest by hamming-ish hash vote.

    Practical approximation (O(1) lookup, no vector index):
      - build: for each length L in [min_len, max_len], store (hash(ctx), dist over next)
      - query: gather candidate dists from ALL stored lengths whose context is a
        SUFFIX of query (i.e. ctx[-L:] == stored pattern). Then weighted vote by
        length (longer match = more specific = higher weight).
    This keeps O(1) lookup (hashing) while using variable-length patterns.
    """
    def __init__(self, min_len=8, max_len=16, min_count=2):
        self.min_len = min_len
        self.max_len = max_len
        self.min_count = min_count
        self.ctx_len = max_len            # query needs last max_len tokens
        # length -> {hash(ctx) -> Counter(next)}
        self.tables = {L: {} for L in range(min_len, max_len + 1)}
        self._size = 0
        self._patterns = []               # [(pattern_tuple, Counter(next), total)] for analysis

    def build(self, train_ids):
        self._patterns = []
        for L in range(self.min_len, self.max_len + 1):
            tab = self.tables[L]
            for i in range(L, len(train_ids)):
                ctx = tuple(train_ids[i - L:i])
                w = train_ids[i]
                h = _h(ctx)
                c = tab.get(h)
                if c is None:
                    c = Counter(); tab[h] = c
                c[w] += 1
        # filter rare per length; store actual pattern tuples (≤16 tokens, cheap)
        for L in self.tables:
            kept = {}
            for h, c in self.tables[L].items():
                tot = sum(c.values())
                if tot >= self.min_count:
                    kept[h] = c
                    # (L, ctx_tuple, Counter(next), total) — recover ctx via identity map below
                    self._patterns.append((L, h, c, tot))
            self.tables[L] = kept
        # fast ctx recovery: map (L,hash) -> ctx using a single train scan
        want = {(L, h) for (L, h, _, _) in self._patterns}
        ctx_map = {}
        for L in range(self.min_len, self.max_len + 1):
            for i in range(L, len(train_ids)):
                ctx = tuple(train_ids[i - L:i])
                h = _h(ctx)
                key = (L, h)
                if key in want and key not in ctx_map:
                    ctx_map[key] = ctx
        self._patterns = [(L, ctx_map.get((L, h), None), c, tot)
                          for (L, h, c, tot) in self._patterns]
        self._size = sum(len(c) for tab in self.tables.values() for c in tab.values())
        return self

    def top_patterns(self, k=20):
        """Top-k patterns by total count. Returns list of (L, ctx_tuple, Counter(next), total)."""
        return sorted(self._patterns, key=lambda x: x[3], reverse=True)[:k]

    def query(self, ctx):
        if len(ctx) < self.min_len:
            return None
        votes = Counter()
        vtot = 0.0
        for L in range(self.min_len, min(self.max_len, len(ctx)) + 1):
            h = _h(tuple(ctx[-L:]))
            c = self.tables[L].get(h)
            if c is None:
                continue
            tot = sum(c.values())
            wlen = L / self.max_len          # longer match -> higher weight
            for tok, cnt in c.items():
                votes[tok] += cnt * wlen
                vtot += cnt * wlen
        if vtot == 0:
            return None
        return votes, int(vtot)

    def size_bytes(self):
        return 8 * sum(len(t) for t in self.tables.values()) + 6 * self._size

    def n_entries(self):
        return sum(len(t) for t in self.tables.values())


class HybridMemory:
    """MEM-C: local order-3 (fast) + global content-addressable (powerful).

    query returns combined (counts, total) where local and global dists are
    merged by weight alpha (fixed at build, not learned).
    """
    def __init__(self, local_order=3, min_len=8, max_len=16, alpha=0.5, min_count=2):
        self.local = BaseMemory(order=local_order, min_count=min_count)
        self.global_mem = ContentAddressableMemory(min_len, max_len, min_count)
        self.alpha = alpha

    def build(self, train_ids):
        self.local.build(train_ids)
        self.global_mem.build(train_ids)
        return self

    def query(self, ctx):
        r_loc = self.local.query(ctx)
        r_glob = self.global_mem.query(ctx)
        if r_loc is None and r_glob is None:
            return None
        if r_loc is None:
            return r_glob
        if r_glob is None:
            return r_loc
        lc, lt = r_loc
        gc, gt = r_glob
        merged = Counter()
        for tok, cnt in lc.items():
            merged[tok] += cnt * self.alpha
        for tok, cnt in gc.items():
            merged[tok] += cnt * (1 - self.alpha)
        return merged, int(sum(merged.values()))

    def size_bytes(self):
        return self.local.size_bytes() + self.global_mem.size_bytes()

    def n_entries(self):
        return self.local.n_entries() + self.global_mem.n_entries()


class RandomMemory:
    """MEM-R: control. Builds from train but at query time returns a RANDOM
    dist (shuffled) to prove the result isn't just 'lookup exists'."""
    def __init__(self, order=8, min_count=3, seed=0):
        self.base = BaseMemory(order=order, min_count=min_count)
        self.rng = random.Random(seed)

    def build(self, train_ids):
        self.base.build(train_ids)
        # collect vocabulary seen
        self.vocab = list({tok for c in self.base.table.values() for tok in c})
        return self

    def query(self, ctx):
        r = self.base.query(ctx)
        if r is None:
            return None
        # return a RANDOM permutation of the same counts (control)
        c, tot = r
        toks = list(c.keys())
        self.rng.shuffle(toks)
        rc = Counter({toks[i]: cnt for i, cnt in enumerate(c.values())})
        return rc, tot

    def size_bytes(self):
        return self.base.size_bytes()

    def n_entries(self):
        return self.base.n_entries()
