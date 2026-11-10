#!/bin/bash
# bell5, bare HiGHS tree (no presolve/restart/symmetry/conflict/separators/SB/heuristics); prints RC fixing counts.
S=$(dirname $0); HS_RC_COUNT=1 HS_NO_CONFLICT=1 HS_NO_TABLEAU=1 HS_NO_PATH=1 HS_NO_MODK=1 HS_NO_IMPLB=1 HS_NO_CLIQUE=1 \
  $S/hsab/build/bin/highs --options_file $S/hsab/opt_tree_bare.txt /home/user/discopt/ref/HiGHS/check/instances/bell5.mps 2>&1 | grep -E "HS_RC|^ *Nodes +[0-9]|Status"
