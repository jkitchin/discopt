#!/bin/bash
# #928 binding-subset settling experiment: 3 reps of the OFF/ON panel at a 20 s
# budget over the instances whose OFF arm did not certify in the full panel
# (the deadline-binding subset). One rep at a time, load printed per rep (§9).
set -euo pipefail
cd /home/user/discopt

SUBSET=$(python - <<'EOF'
import json
d = json.load(open("discopt_benchmarks/results/issue928_full_panel.json"))
print(",".join(c["instance"] for c in d["cells"] if not c["off"]["gap_certified"]))
EOF
)
echo "binding subset: $SUBSET"

for rep in 1 2 3; do
  echo "=== rep $rep $(uptime) ==="
  python -u discopt_benchmarks/scripts/issue917_lp_warm_deadline_panel.py \
    --budget 20 --instances "$SUBSET" \
    --out "discopt_benchmarks/results/issue928_binding_rep${rep}.json"
done
echo "ALL_REPS_DONE"
