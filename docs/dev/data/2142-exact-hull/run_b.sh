#!/bin/bash
export DISCOPT_PROFILE=1 PYTHONHASHSEED=0 PYTHONPATH=/home/user/discopt/python
cd /home/user/discopt/docs/dev/data/2142-exact-hull
( python -u arm.py both 1 7200 > arm_both_1.log 2>&1; python -u arm.py both 10000 7200 > arm_both_10000.log 2>&1 ) &
( python -u arm.py homog 20000 7200 > arm_homog_20000.log 2>&1 ) &
wait
echo done > run_b.done
