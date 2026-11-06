import pyscipopt as ps, json
out={}
for nm in ["neos-3610051-istra","neos-3610040-iskar"]:
    m=ps.Model(); m.hideOutput(); m.readProblem(f"/home/user/discopt/ref/miplib2017/{nm}.mps.gz"); m.setParam("limits/time",120)
    m.optimize(); s=m.getBestSol()
    out[nm]={v.name:m.getSolVal(s,v) for v in m.getVars()}; print(nm,m.getStatus(),m.getObjVal(),m.getNNodes(),flush=True)
json.dump(out,open("optsols.json","w"))
