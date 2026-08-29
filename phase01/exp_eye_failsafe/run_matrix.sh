#!/bin/bash
# Run the required Eye failsafe matrix sequentially.
cd /c/Users/Geroin/chaotic-llm/phase01/exp_eye_failsafe
echo "=== E1-u (uniform) ==="
python experiment.py E1 u 2>&1 | tail -3
echo "=== E1-r (random frozen) ==="
python experiment.py E1 r 2>&1 | tail -3
echo "=== E1-l (learned) ==="
python experiment.py E1 l 2>&1 | tail -3
echo "=== E2-r (random frozen) ==="
python experiment.py E2 r 2>&1 | tail -3
echo "=== E2-l (learned) ==="
python experiment.py E2 l 2>&1 | tail -3
echo "=== MATRIX DONE ==="
