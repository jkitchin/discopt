#!/bin/bash
# #2142: SCIP presolve-component ablations on the homog spec (baseline 7,470 nodes).
# Runs 2 at a time; each -> ../2142-scip-hull-ablation/ablate.sh NAME default --param ... into this dir.
D=/home/user/discopt/docs/dev/data/2142-scip-presolved
A=/home/user/discopt/docs/dev/data/2142-scip-hull-ablation/ablate.sh
run() { NAME=$1; shift; PYTHONHASHSEED=0 python -u /home/user/discopt/docs/dev/data/2142-scip-expert/run.py --run /home/user/discopt/docs/dev/data/2142-exact-hull/scip/cstr_homog_spec.npz --arm default --tl 600 --node-limit 100000 --param "$@" > $D/abl_$NAME.log 2>&1; echo "DONE $NAME $(date +%T)"; }
run nlpre0 constraints/nonlinear/maxprerounds=0 &
run linpre0 constraints/linear/maxprerounds=0 &
wait
run linaggr_off constraints/linear/aggregatevariables=false &
run vbpre0 constraints/varbound/maxprerounds=0 &
wait
run probing0 propagating/probing/maxprerounds=0 &
run pairwise_off constraints/linear/presolpairwise=false &
wait
run setppc0 constraints/setppc/maxprerounds=0 &
run linhash_off constraints/linear/presolusehashing=false &
wait
