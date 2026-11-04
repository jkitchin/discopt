#!/bin/bash
# 5-hour plan: rep 1 = correctness + solved counts; then the both-solved pairs
# x2 more reps at jobs=4 (3 timing reps with rep 1), then a serial control sample.
S=/private/tmp/claude-501/-Users-jkitchin-projects-discopt/5793622b-ea73-47f3-a307-5878f6abb68d/scratchpad/g61
WT=/Users/jkitchin/projects/discopt-g61
PY=/Users/jkitchin/Dropbox/uv/.venv/bin/python
export GDP_BASELINE_GLOBAL=scip PYTHONPATH=$WT/python
cd $WT
while kill -0 30899 2>/dev/null; do sleep 30; done
echo "=== rep1 done $(date) rows=$(wc -l < $S/rep1.jsonl)"
$PY - <<'PY' > $S/pairs.txt
import json
from collections import defaultdict
F={"big-m":("discopt-big-m","pyomo-bigm"),"hull":("discopt-hull","pyomo-hull"),"loa":("discopt-loa","pyomo-loa"),"gloa":("discopt-gloa","pyomo-gloa"),"ric":("discopt-ric","pyomo-ric"),"enumerate":("discopt-enumerate","pyomo-enumerate"),"lbb":("discopt-lbb","pyomo-lbb")}
v={}
for l in open("/private/tmp/claude-501/-Users-jkitchin-projects-discopt/5793622b-ea73-47f3-a307-5878f6abb68d/scratchpad/g61/rep1.jsonl"):
    r=json.loads(l); v[(r["instance"],r["arm"])]=r["verdict"]
for fam,(d,p) in F.items():
    inst=sorted({i for (i,a) in v if v.get((i,d))=="solved" and v.get((i,p))=="solved"})
    if inst: print(fam, d+","+p, ",".join(inst))
PY
cat $S/pairs.txt
for rep in 2 3; do
  echo "=== timing rep $rep start $(date)"
  while read fam arms inst; do
    $PY -u discopt_benchmarks/scripts/gdp_baseline.py --out $S/timing_rep$rep.jsonl \
      --arms $arms --include $inst --jobs 4 --time-limit 300 --grace 120
  done < $S/pairs.txt
  echo "=== timing rep $rep end $(date)"
done
echo "=== serial control start $(date)"
while read fam arms inst; do
  sub=$(echo $inst | tr ',' '\n' | head -2 | paste -sd, -)
  $PY -u discopt_benchmarks/scripts/gdp_baseline.py --out $S/serial.jsonl \
    --arms $arms --include $sub --jobs 1 --time-limit 300 --grace 120
done < $S/pairs.txt
echo "=== serial control end $(date)"
echo TIMING_DONE
