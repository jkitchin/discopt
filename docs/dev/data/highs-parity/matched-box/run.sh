#!/usr/bin/env bash
# Matched-box node comparison on mik-250-20-75-5 (docs/dev/highs-mip-replacement-plan.md,
# entry 2026-11-13 "matched boxes"). Regenerates mik5_*.jsonl and summary_mik5_*.txt.
#
# HiGHS: a 1.15.1 build carrying the discopt debug dumps (HS_DUMP_SEARCH etc.) plus
# highs_nodebox.patch (HS_DUMP_NODEBOX, env-gated: unset, the build behaves as before).
# discopt: the extension built from this tree (DISCOPT_NODEBOX_FILE harness,
# crates/discopt-core/src/bnb/milp_driver/highs_env/nodebox.rs).
#
# usage: run.sh <highs binary> <mik5.sol (HiGHS solution file)> <mik5.seed> <workdir>
set -euo pipefail
HIGHS=$1 SOL=$2 SEED=$3 W=$4
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
MPS=$REPO/ref/miplib2017/mik-250-20-75-5.mps.gz
mkdir -p "$W"
printf "threads=1\nmip_allow_restart=false\n" > "$W/opts.txt"

# HiGHS, presolve off, seeded with the optimum; every node-LP solve and separation.
"$HIGHS" --presolve off --options_file "$W/opts.txt" --read_solution_file "$SOL" "$MPS" \
    > "$W/hs_plain.log"
HS_DUMP_NODEBOX_MAX=5000 HS_DUMP_NODEBOX="$W/hs_nodebox.txt" \
    "$HIGHS" --presolve off --options_file "$W/opts.txt" --read_solution_file "$SOL" "$MPS" \
    > "$W/hs_dump.log"
# The dump must not change the search (measured: 397 nodes, 47,399 LP iterations both).
diff <(grep -E "^  (Nodes|LP iterations)" "$W/hs_plain.log") \
     <(grep -E "^  (Nodes|LP iterations)" "$W/hs_dump.log")

# discopt, presolve and sub-MIPs off, seeded; max_nodes 2 so the search starts (the
# harness runs at its start) and stops right after.
ours() {  # <tag> [extra env...]
    local tag=$1; shift
    (cd "$REPO" && env "$@" DISCOPT_NODEBOX_FILE="$W/hs_nodebox.txt" \
        DISCOPT_MILP_PRESOLVE=0 DISCOPT_MILP_SUBMIP=0 DISCOPT_PANEL_DETERMINISTIC=1 \
        DISCOPT_MILP_SEED_POINT="$SEED" \
        python -u discopt_benchmarks/scripts/lp_milp_panel.py worker "$MPS" R 120 1e-4 2 \
        > "$W/ours_$tag.out" 2> "$W/ours_$tag.err")
    python -I "$HERE/compare.py" "$W/hs_nodebox.txt" "$W/ours_$tag.err" \
        --jsonl "$HERE/mik5_$tag.jsonl" > "$HERE/summary_mik5_$tag.txt"
}
ours fresh DISCOPT_NODEBOX_MODE=fresh
ours cumulative DISCOPT_NODEBOX_MODE=cumulative
ours fresh_sepa_propagate DISCOPT_NODEBOX_MODE=fresh DISCOPT_HIGHS_SEPA_PROPAGATE=1
