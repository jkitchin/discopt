set -x
cd /home/user/discopt
O=/tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad/p66
python -u discopt_benchmarks/scripts/lp_milp_panel.py run --manifest discopt_benchmarks/data/lp_milp/manifest.json --panel L --panel M --panel INF --arms R,H --rounds 3 --time-limit 20 --load-gate 4 --out $O/lmi.json > $O/lmi.log 2>&1; echo "STEP lmi $?"
python -u discopt_benchmarks/scripts/miplib_panel.py run --arms R,H --rounds 3 --time-limit 20 --load-gate 4 --out $O/miplib.json > $O/miplib.log 2>&1; echo "STEP miplib $?"
python -u discopt_benchmarks/scripts/p51_route_master_panel.py --population route --out $O/route.json > $O/route.log 2>&1; echo "STEP route $?"
python -u discopt_benchmarks/scripts/p51_route_master_panel.py --population syn --kernel on --out $O/syn_on.json > $O/syn_on.log 2>&1; echo "STEP syn_on $?"
python -u discopt_benchmarks/scripts/master_parity_panel.py replay /tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad/p52/decomp /tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad/p52/gdp --rounds 3 --out $O/replay.json > $O/replay.log 2>&1; echo "STEP replay $?"
python -u discopt_benchmarks/scripts/iis_engine_panel.py --reps 1 --out $O/iis.json > $O/iis.log 2>&1; echo "STEP iis $?"
python -m pytest python/tests/test_highs_optional.py -q -s -m "" -p no:cacheprovider > $O/guard.log 2>&1; echo "STEP guard $?"
echo ALLDONE
