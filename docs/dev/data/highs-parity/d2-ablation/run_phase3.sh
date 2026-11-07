#!/bin/bash
# Phase 3: queued behind phase 2 (one solve at a time on a loaded host).
cd /tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad
until grep -q "executed solves:" d2_highs_ablation_phase2.log || ! pgrep -f "d2_highs_ablation.py run" >/dev/null; do sleep 20; done
HEUR="2122,bppc8-02,gesa2,gt2,khb05250,mik-250-20-75-2,mik-250-20-75-5,nexp-50-20-1-1,nexp-50-20-4-2,p0548,sp150x300d,gsvm2rl3"
PRES="b-ball,bppc8-02,gesa2,issue-2446,mik-250-20-75-5,neos-3610040-iskar,gsvm2rl3"
CUTS="b-ball,neos-911970,bppc8-02,gsvm2rl3,mik-250-20-75-1,mik-250-20-75-5"
SLOW="gsvm2rl3,neos-911970,mik-250-20-75-1,neos17,mik-250-20-75-3"
python -I d2_highs_ablation.py run --out d2_highs_ablation_phase3.jsonl --jobs 2 --instances "$HEUR" --arms fj_off,rins_off,rens_off,rootredcost_off,heur_effort_0
python -I d2_highs_ablation.py run --out d2_highs_ablation_phase3.jsonl --jobs 2 --instances "$PRES" --arms parallel_rc_off,aggregator_off,doubleton_off,sparsify_off,dualfix_off,presolve_reduce_only
python -I d2_highs_ablation.py run --out d2_highs_ablation_phase3.jsonl --jobs 2 --instances "$CUTS" --arms pool_soft_1,lp_age_0
python -I d2_highs_ablation.py run --out d2_highs_ablation_base40.jsonl --jobs 2 --time-limit 40 --instances "$SLOW" --arms base,all_off
echo PHASE3_DONE
