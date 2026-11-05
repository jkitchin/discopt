#!/bin/bash
export DISCOPT_PROFILE=1 PYTHONHASHSEED=0 PYTHONPATH=/home/user/discopt/python
cd /home/user/discopt/docs/dev/data/2142-exact-hull
( python -u arm.py homog 0 300 nodet > arm_homog_default300.log 2>&1 ) &
( DISCOPT_DUAL_PROOF_PROPAGATION=0 python -u arm.py homog 20000 7200 > arm_homog_20000_dpoff.log 2>&1 ) &
wait
echo done > run_c.done
