import json,sys,re
OPT={"mik-250-20-75-1":-49716,"mik-250-20-75-2":-50768,"mik-250-20-75-3":-52242,"mik-250-20-75-5":-51532,"neos17":0.1500025774,"neos-911970":54.76}
cur=None; n=0; bad=0
for line in open(sys.argv[1]):
    m=re.match(r"### (\S+) gpool=(\d)",line)
    if m: cur=m.groups(); continue
    if line.startswith("RESULT "):
        r=json.loads(line[7:]); n+=1
        o=OPT[cur[0]]; b=r["bound"]
        flag="" if b is None or b<=o+1e-6*(1+abs(o)) else "  **BOUND PAST OPTIMUM**"
        if flag: bad+=1
        print(f"{cur[0]:18s} gpool={cur[1]} {r['status']:10s} obj={r['objective']} bound={b} cert={r['gap_certified']} nodes={r['nodes']} wall={r['wall']:.1f}{flag}")
print(f"rows={n} soundness_failures={bad}")
if n==0: sys.exit(1)
