#!/bin/bash
# usage: SCRATCH=<dir with s2_arms.py> DISCOPT_TREE=<worktree> runarms.sh arm:nodes ...
# Runs at most 3 arms at once; each writes arm_<arm>_<nodes>.log.
export DISCOPT_TREE=${DISCOPT_TREE:?} PYTHONPATH=${DISCOPT_TREE}/python
cd ${SCRATCH:?}
printf '%s\n' "$@" | xargs -P 3 -I{} bash -c 'a=${1%%:*}; n=${1##*:}; python -u s2_arms.py $a $n > arm_${a}_${n}.log 2>&1; echo "done $1 rc=$?"' _ {}
