"""exp52_bidirectional.py — Gears: bidirectional mixers (fwd + bwd).

Exp52 of the "gears" hypothesis. Adds a Linear(2d,d) projection
(~262K params) — the only gear variant with learnable extra params.
See ТЗ / chaotic_gears.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gear_protocol import run

if __name__ == "__main__":
    run(kind="bi", name="exp52-bidirectional",
        out_dir="exp52_bidirectional",
        blocks=12, window=64)
