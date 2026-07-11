import sys, numpy as np
import discopt.modeling as dm
NL_DIR="/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
ORACLE={
 'ex6_2_5':-70.7520778300,'ex6_2_9':-0.0340661841,'ex6_2_10':-3.0519761260,
 'ex6_2_11':-0.0000026724,'ex6_2_12':0.2891947485,'ex6_2_13':-0.2162094369,
}
inst=sys.argv[1]
tl=float(sys.argv[2]) if len(sys.argv)>2 else 5.0
m=dm.from_nl(f"{NL_DIR}/{inst}.nl")
r=m.solve(time_limit=tl)
rb=getattr(r,'root_bound',None)
rg=getattr(r,'root_gap',None)
orc=ORACLE[inst]
print(f"{inst}: status={r.status} obj={getattr(r,'objective',None)} bound={getattr(r,'bound',None)} root_bound={rb} root_gap={rg} oracle={orc}")
