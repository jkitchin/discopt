#!/bin/bash
set -u
# Wait for the baseline probe to finish so nothing overlaps (CLAUDE.md §9).
while pgrep -f "repro_probe.py kriging_peaks-full200" >/dev/null; do sleep 20; done
echo "=== E3: single-threaded arm (rayon + BLAS pinned to 1) ==="
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -u scratchpad/issue1116/repro_probe.py kriging_peaks-full200 300 3
echo "=== E2: clock-call-site trace, default threading ==="
python -u scratchpad/issue1116/clock_trace.py kriging_peaks-full200 300 2
