#!/bin/bash
# #2168 entry arms on the final presolve code, seeded, 2 at a time.
cd "$(dirname "$0")"
run() { DISCOPT_PROFILE=1 PYTHONHASHSEED=0 python -u kernel_arm.py $1 $2 seeded > karm_$1_$2_seeded.log 2>&1; echo "$1 $2 seeded exit $?" >> runs.out; }
(run pre 10000; run pre 20000) &
(run multi 10000; run multi 20000) &
wait
(run orig 20000; ) &
(run scip 20000; ) &
wait
echo QUEUE_DONE >> runs.out
