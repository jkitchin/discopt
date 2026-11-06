#!/bin/bash
# #2142 SCIP single-component ablations on the homog-hull kernel spec.
# usage: ablate.sh NAME ARM [--param k=v ...]   -> NAME.log (json line + executed-assertion count)
# Node limit 100k (well above discopt's 30k reach), 600 s backstop.
set -u
D=/home/user/discopt/docs/dev/data/2142-scip-hull-ablation
SPEC=/home/user/discopt/docs/dev/data/2142-exact-hull/scip/cstr_homog_spec.npz
NAME=$1; ARM=$2; shift 2
cd $D
PYTHONHASHSEED=0 python -u /home/user/discopt/docs/dev/data/2142-scip-expert/run.py --run $SPEC --arm $ARM --tl 600 --node-limit 100000 "$@" > $D/$NAME.log 2>&1
echo "DONE $NAME"
