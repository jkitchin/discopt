cd /home/user/discopt
SP=/tmp/claude-0/-home-user-discopt/622858fd-8692-539a-a137-f41f6cd43f5f/scratchpad
C="DISCOPT_PROFILE=1 DISCOPT_PANEL_DETERMINISTIC=1 RAYON_NUM_THREADS=1 DISCOPT_LP_MILP_BACKEND=rust"
for arm in combo config default; do
  case $arm in
    combo) E="DISCOPT_HIGHS_SEARCH_CUT_LP=1 DISCOPT_HIGHS_SEPA_NODE=1 DISCOPT_HIGHS_SEPA_CUTOFF_BOX=1";;
    config) E="DISCOPT_HIGHS_CONFIG=1";;
    default) E="";;
  esac
  echo "start $arm $(date +%T)"
  env $E $C timeout 900 python -u discopt_benchmarks/scripts/lp_milp_panel.py worker ref/miplib2017/neos17.mps.gz R 600 1e-4 5000 > $SP/farkas/neos17_$arm.txt 2>&1
  echo "done $arm $(date +%T) rc=$?"
done
