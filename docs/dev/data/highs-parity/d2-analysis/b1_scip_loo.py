import sys, pyscipopt
sys.path.insert(0,"discopt_benchmarks/scripts"); import p42_rens_panel as R
pop={m["name"]:m for m in R.population()}
drops=["(none)","aggregation","gomory","zerohalf","impliedbounds","clique","intobj","rapidlearning","cmir","flowcover","mcf","knapsackcover","strongcg","oddcycle","conflict"]
n=0
for nm in ["neos-3610040-iskar","neos-3610051-istra"]:
  for d in drops:
    m=pyscipopt.Model(); m.hideOutput(); m.readProblem(pop[nm]["path"])
    m.setParam("limits/time",30.0); m.setParam("limits/gap",1e-4); m.setParam("presolving/maxrounds",0)
    if d=="conflict": m.setParam("conflict/enable", False)
    elif d!="(none)": m.setParam(f"separating/{d}/freq",-1)
    m.optimize(); n+=1
    print(f"{nm:20s} drop={d:14s} {m.getStatus():10s} nodes={m.getNNodes():6d} t={m.getSolvingTime():5.1f} bound={m.getDualbound():.4f}", flush=True)
print("executed", n)
