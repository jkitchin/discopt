#!/bin/bash
# Execute notebooks sequentially, one log per notebook. Never silences stderr.
set -u
PY=/home/user/.venv312/bin/python
OUT=${OUT:-/tmp/claude-0/-home-user-discopt/e3e50092-94db-558b-b339-ae36f8cbc9c3/scratchpad/runs}
mkdir -p "$OUT"
export RUST_LOG=${RUST_LOG:-warn}
for nb in "$@"; do
  echo "=== START $nb $(date -u +%H:%M:%S)"
  timeout 1800 $PY -u scratchpad/nbreview/run_nb.py "docs/notebooks/$nb.ipynb" 900 \
      > "$OUT/$nb.log" 2>&1
  rc=$?
  echo "=== DONE  $nb rc=$rc $(date -u +%H:%M:%S) :: $(grep -m1 '^RESULT\|^PROBE' "$OUT/$nb.log" || echo '(no RESULT line)')"
done
