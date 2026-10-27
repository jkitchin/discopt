import json,glob,sys
S=sys.argv[1]
runs=[f"{S}/panel_bundle2.jsonl"]+sorted(glob.glob(f"{S}/repc_*.jsonl"))
data={}
for i,f in enumerate(runs):
    for l in open(f):
        r=json.loads(l); data.setdefault(r["name"],[]).append(r)
n=0
def fmt(a):
    b=a.get("bound"); return ("C" if a.get("cert") else "")+("None" if b is None else f"{b:.6g}")
for name,rs in sorted(data.items()):
    if len(rs)<2: continue
    probs=[p for r in rs for p in r["problems"]]
    print(f"{name:24s} sense={rs[0]['off']['sense']:+d} OFF[{' '.join(fmt(r['off']) for r in rs)}] ON[{' '.join(fmt(r['on']) for r in rs)}] {probs if probs else ''}")
    n+=1
print("instances aggregated", n, "runs", len(runs))
assert n>0
