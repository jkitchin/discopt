#!/bin/bash
# usage: dumpfast.sh <outdir> <instance files...>: top-level presolve dump only; the
# worker is killed once the top-level presolve has logged (the dump is written first).
out=$1; shift; mkdir -p $out
for f in "$@"; do
  nm=$(basename $f .mps.gz)
  [ -s $out/$nm.done ] && continue
  rm -f $out/$nm.mps
  DISCOPT_MILP_PRESOLVE_LOG=1 DISCOPT_MILP_PRESOLVE_DUMP=$out/$nm.mps DISCOPT_MILP_SUBMIP=0 DISCOPT_HIGHS_SUBMIP_PRESOLVE=0 DISCOPT_PANEL_DETERMINISTIC=1 RAYON_NUM_THREADS=1 \
    python -u /home/user/discopt/discopt_benchmarks/scripts/lp_milp_panel.py worker $f R 1000 1e-4 1 > $out/$nm.log 2>&1 &
  p=$!
  for i in $(seq 1 1800); do
    if grep -q "^MILP presolve:" $out/$nm.log || ! kill -0 $p 2>/dev/null; then break; fi
    read -t 0.2 < /dev/zero 2>/dev/null || true
  done
  kill $p 2>/dev/null; wait $p 2>/dev/null
  h=$( [ -f $out/$nm.mps ] && md5sum < $out/$nm.mps | cut -c1-12 || echo none)
  echo "$nm $h $(grep -m1 '^MILP presolve:' $out/$nm.log | cut -c1-60)" | tee $out/$nm.done
done
