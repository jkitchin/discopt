#!/bin/bash
# Interleaved A/B: DISCOPT_HIGHS_SEARCH_GLOBAL_POOL off vs on, TL 20, one solve at a time.
SP=${SP:?set SP to a scratch directory}
OUT=$SP/gp/panel20.jsonl
: > $OUT
cd /home/user/discopt
i=0
for name in mik-250-20-75-1 mik-250-20-75-2 mik-250-20-75-3 mik-250-20-75-5 neos17 neos-911970 b-ball enlight_hard gsvm2rl3; do
  if [ $((i % 2)) -eq 0 ]; then order="0 1"; else order="1 0"; fi
  i=$((i+1))
  for f in $order; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    res=$(DISCOPT_HIGHS_SEARCH_GLOBAL_POOL=$f DISCOPT_PANEL_DETERMINISTIC=1 timeout 300 python -u discopt_benchmarks/scripts/lp_milp_panel.py worker ref/miplib2017/$name.mps.gz R 20 1e-4 100000000 2>/dev/null | grep -o 'RESULT .*' | cut -c8-)
    echo "{\"name\": \"$name\", \"flag\": $f, \"load\": $load, \"res\": $res}" >> $OUT
    echo "$name flag=$f load=$load $(echo $res | cut -c1-160)"
  done
done
