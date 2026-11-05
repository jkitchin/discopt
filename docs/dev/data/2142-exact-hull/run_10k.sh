#!/bin/bash
# 4 concurrent arms on 4 cores; node-bounds are deterministic, walls are not comparable.
export DISCOPT_PROFILE=1 PYTHONHASHSEED=0 PYTHONPATH=/home/user/discopt/python
cd /home/user/discopt/docs/dev/data/2142-exact-hull
( python -u arm.py bigm 10000 7200 > arm_bigm_10000.log 2>&1; python -u arm.py bigm 20000 7200 > arm_bigm_20000.log 2>&1 ) &
( python -u arm.py homog 10000 7200 > arm_homog_10000.log 2>&1 ) &
( python -u arm.py lifted 10000 7200 > arm_lifted_10000.log 2>&1 ) &
( python -u arm.py ratio 10000 7200 > arm_ratio_10000.log 2>&1 ) &
wait
echo done > run_10k.done
