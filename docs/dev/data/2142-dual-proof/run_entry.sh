#!/bin/bash
# #2142 dual-proof entry experiment: cstr big-M, deterministic 20k nodes, 4 seeds x ON/OFF.
cd "$(dirname "$0")"
jobs_=()
for s in 0 1 2 3; do for f in 0 1; do jobs_+=("$s $f"); done; done
printf '%s\n' "${jobs_[@]}" | xargs -P 3 -L 1 bash -c 'PERM_SEED=$0 DISCOPT_DUAL_PROOF_PROPAGATION=$1 python -u e1_arm.py 20000 > entry_s$0_f$1.log 2>&1'
