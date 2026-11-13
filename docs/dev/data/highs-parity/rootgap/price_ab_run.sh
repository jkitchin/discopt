#!/bin/bash
# usage: run.sh <so> <out.jsonl> <instances...>: deterministic 300-node solves, 2 at a time
so=$1; out=$2; shift 2
cp $so /home/user/discopt/python/discopt/_rust.abi3.so
: > $out
printf "%s\n" "$@" | xargs -P 2 -I{} bash -c 'r=$(DISCOPT_PANEL_DETERMINISTIC=1 RAYON_NUM_THREADS=1 timeout 900 python -u /home/user/discopt/discopt_benchmarks/scripts/lp_milp_panel.py worker {} R 600 1e-4 300 2>/dev/null | grep "^RESULT " | tail -1); echo "{\"path\": \"{}\", \"r\": ${r#RESULT }}" >> '"$out"
