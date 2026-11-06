#!/bin/bash
# #2184 review finding 1: does ROOT_SKI lose an incumbent at short time limits?
cd /tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/wt_feat
export PYTHONPATH=/tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/wt_feat/python
/tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/venv312/bin/python -c "import discopt,os,discopt.solver as S; assert discopt.__file__.startswith(os.getcwd()); assert hasattr(S,'_gdp_config_root_ski_enabled')" || exit 3
TL=$1; CORE=$2
echo "=== TL=$TL start $(date)"
taskset -c $CORE /tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/venv312/bin/python -u discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py --flag DISCOPT_GDP_CONFIG_ROOT_SKI --phase solve --all --reps 1 --time-limit $TL --grace 60 --out /tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/shortlim/gdp_tl$TL.jsonl
echo "=== TL=$TL end $(date) exit $?"
