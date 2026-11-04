#!/bin/bash
# Capture the first Rust-MILP input (solve_milp_csc_py arguments) of every GDP-parity
# instance under LOA and big-M into ./captures/. Run from this directory.
set -u
export DISCOPT_LP_MILP_BACKEND=rust
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${G32_CAPTURES:-$HERE/captures}
mkdir -p "$OUT"
for meth in loa big-m; do
for i in batch_processing biofuel cstr disease_model ex1_linan_2023 gdp_col hda kaibel med_term_purchasing methanol mod_hens modprodnet positioning pyomo.constrained_layout_clay0203 pyomo.eight_process pyomo.eight_process_logical pyomo.ex_633_trespalacios pyomo.jobshop_small pyomo.nonconvex_hen pyomo.strip_packing_concrete small_batch spectralog stranded_gas syngas; do
  echo "== $meth $i"
  timeout 150 python -u "$HERE/caprust.py" "$i" "$meth" "$OUT/${meth}__$i.pkl" 2>&1 | grep -E "captured|no Rust|Error" | tail -2
done; done
echo CAP_DONE
