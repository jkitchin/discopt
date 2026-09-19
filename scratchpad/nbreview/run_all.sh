#!/bin/bash
# Re-execute every notebook in docs/notebooks, one log each, sequentially.
# Sequential on purpose: the notebooks report wall times, and parallel runs on a
# shared box make those numbers meaningless (CLAUDE.md §9).
set -u
cd /home/user/discopt
PY=/home/user/.venv312/bin/python
OUT=/tmp/claude-0/-home-user-discopt/e3e50092-94db-558b-b339-ae36f8cbc9c3/scratchpad/runs
mkdir -p "$OUT"
export RUST_LOG=warn
n=0
for nb in docs/notebooks/*.ipynb; do
  name=$(basename "$nb" .ipynb)
  n=$((n+1))
  timeout 2400 $PY -u scratchpad/nbreview/run_nb.py "$nb" 1800 > "$OUT/$name.log" 2>&1
  rc=$?
  printf '=== DONE %-32s rc=%d %s :: %s\n' "$name" "$rc" "$(date -u +%H:%M:%S)" \
     "$(grep -m1 '^RESULT\|^PROBE' "$OUT/$name.log" || echo '(no RESULT line)')"
done
echo "=== ALL DONE: $n notebooks"
