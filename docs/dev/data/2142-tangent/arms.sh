#!/bin/bash
S=${SCRATCH:?set SCRATCH to the directory holding lp20k.npz and the t*.py scripts}
cd $S
export DISCOPT_TREE=${DISCOPT_TREE:?} PYTHONPATH=${DISCOPT_TREE}/python PROPDIR=${PROPDIR:?}
run() { python -u t2_tangent.py lp20k.npz --no-tangents "$@" > "arm_$(echo "$*" | tr '/=,' '___').log" 2>&1; echo "done $* rc=$?"; }
export -f run
printf '%s\n' nlhdlr/bilinear/enabled=False nlhdlr/convex/enabled=False nlhdlr/concave/enabled=False \
  nlhdlr/quadratic/enabled=False nlhdlr/perspective/enabled=False nlhdlr/soc/enabled=False \
  nlhdlr/quotient/enabled=False nlhdlr/signomial/enabled=False propagating/maxroundsroot=0 \
  constraints/nonlinear/propfreq=-1 constraints/linear/propfreq=-1 lp/solvefreq=-1 \
  | xargs -P 3 -I{} bash -c 'run {}'
