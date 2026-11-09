#!/bin/bash
SP=${SP:?set SP to a scratch dir holding so/A.so and so/B.so}
OUT=$SP/eng_rowprice_entry.jsonl
cd /home/user/discopt
names="bppc8-02 sp150x300d beavma 2122 b-ball neos-3610051-istra enlight_hard p0548 neos-3118745-obra"
i=0
for rep in 1 2; do
 for n in $names; do
  i=$((i+1))
  if [ $((i % 2)) -eq 0 ]; then arms="A B"; else arms="B A"; fi
  for a in $arms; do
   cp $SP/so/$a.so python/discopt/_rust.abi3.so
   echo "rep $rep $n arm $a load $(cut -d' ' -f1 /proc/loadavg)"
   python -u docs/dev/data/highs-parity/d2-analysis/d2_probe.py --out $OUT --time-limit 20 --jobs 1 "$n:DISCOPT_LP_PRIMAL_INCR_PRICING=1,D2_ARM=$a" 2>&1 | tail -1
  done
 done
done
cp $SP/so/B.so python/discopt/_rust.abi3.so
echo ALLDONE
