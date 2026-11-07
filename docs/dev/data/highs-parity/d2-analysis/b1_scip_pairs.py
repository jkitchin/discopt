import sys, pyscipopt
sys.path.insert(0,"discopt_benchmarks/scripts"); import p42_rens_panel as R
pop={m["name"]:m for m in R.population()}
m0=pyscipopt.Model(); fams=sorted({k.split("/")[1] for k in m0.getParams() if k.startswith("separating/") and k.endswith("/freq")})
arms={
 "all, gomory root-only": ("all", {"separating/gomory/freq":0}),
 "all, gomory tree-only": ("all", {"separating/gomory/maxroundsroot":0}),
 "gomory+aggregation": (["gomory","aggregation"], {}),
 "gomory+intobj": (["gomory","intobj"], {}),
 "gomory+zerohalf": (["gomory","zerohalf"], {}),
 "gomory+strongcg+cmir": (["gomory","strongcg","cmir"], {}),
 "gomory+gomorymi": (["gomory","gomorymi"], {}),
}
n=0
for nm in ["neos-3610040-iskar","neos-3610051-istra"]:
  for arm,(keep,extra) in arms.items():
    m=pyscipopt.Model(); m.hideOutput(); m.readProblem(pop[nm]["path"])
    m.setParam("limits/time",30.0); m.setParam("limits/gap",1e-4); m.setParam("presolving/maxrounds",0)
    if keep!="all":
        for f in fams:
            if f not in keep: m.setParam(f"separating/{f}/freq",-1)
    for k,v in extra.items(): m.setParam(k,v)
    m.optimize(); n+=1
    print(f"{nm:20s} {arm:24s} {m.getStatus():10s} nodes={m.getNNodes():6d} t={m.getSolvingTime():5.1f} bound={m.getDualbound():.4f}", flush=True)
print("executed", n)
