#!/bin/bash
# Interleaved OFF/ON, arm order alternating per (rep, instance, method). Pinned to core 3.
S=/tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad
PY=$S/venv312/bin/python
OUT=$S/h2_ab.jsonl
INST="pyomo.eight_process pyomo.eight_process_logical pyomo.ex_633_trespalacios pyomo.nonconvex_hen pyomo.constrained_layout_clay0203 pyomo.strip_packing_concrete pyomo.jobshop_small small_batch ex1_linan_2023 med_term_purchasing positioning modprodnet disease_model spectralog syngas"
k=0
for rep in 1 2 3; do for i in $INST; do for m in big-m hull; do
  if [ $((k % 2)) -eq 0 ]; then arms="OFF ON"; else arms="ON OFF"; fi; k=$((k+1))
  for a in $arms; do
    l=$(cut -d' ' -f1 /proc/loadavg)
    row=$(timeout 500 taskset -c 3 $PY -u $S/h2_ab.py $i $m $a 2>/dev/null | grep '^{')
    [ -z "$row" ] && row="{\"instance\":\"$i\",\"method\":\"$m\",\"arm\":\"$a\",\"error\":true}"
    echo "${row%\}}, \"rep\": $rep, \"load1\": $l}" >> $OUT
    echo "rep$rep $i $m $a $(echo $row | cut -c1-140)"
  done
done; done; done
echo H2_DONE
