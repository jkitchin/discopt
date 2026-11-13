#!/bin/bash
# usage: run.sh <tag> <TL> <max_nodes> [extra env...]
cd /home/user/discopt
SP=/tmp/claude-0/-home-user-discopt/622858fd-8692-539a-a137-f41f6cd43f5f/scratchpad/gpool
TAG=$1; TL=$2; MN=$3; shift 3
OUT=$SP/$TAG.log; : > $OUT
i=0
for name in mik-250-20-75-1 mik-250-20-75-2 mik-250-20-75-3 mik-250-20-75-5 neos17 neos-911970; do
  if [ $((i%2)) = 0 ]; then arms="0 1"; else arms="1 0"; fi
  for a in $arms; do
    echo "### $name gpool=$a $(uptime)" >> $OUT
    env "$@" DISCOPT_HIGHS_SEARCH_GLOBAL_POOL=$a DISCOPT_PANEL_DETERMINISTIC=1 timeout 400 python -u discopt_benchmarks/scripts/lp_milp_panel.py worker ref/miplib2017/$name.mps.gz R $TL 1e-4 $MN >> $OUT 2>$SP/$TAG.$name.$a.err
    echo "### rc=$?" >> $OUT
  done
  i=$((i+1))
done
echo DONE >> $OUT
