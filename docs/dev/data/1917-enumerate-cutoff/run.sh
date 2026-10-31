cd /home/user/discopt
for rep in 0 1 2; do
 for n in pyomo.strip_packing_concrete pyomo.eight_process pyomo.eight_process_logical pyomo.constrained_layout_clay0203 pyomo.ex_633_trespalacios pyomo.nonconvex_hen pyomo.jobshop_small; do
  if [ $((rep % 2)) = 0 ]; then arms="0 1"; else arms="1 0"; fi
  for f in $arms; do
   l=$(cut -d' ' -f1 /proc/loadavg)
   out=$(DISCOPT_GDP_ENUMERATE_CUTOFF=$f timeout 400 python -u discopt_benchmarks/scripts/gdp_baseline.py --worker $n discopt-enumerate 300 2>&1 | grep "@@RESULT")
   echo "rep=$rep flag=$f load=$l ${out}" | cut -c1-900
  done
 done
done
echo ALLDONE
