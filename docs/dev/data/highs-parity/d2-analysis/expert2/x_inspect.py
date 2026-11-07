import pyscipopt as ps, sys
for nm in ["neos-3610051-istra","neos-3610040-iskar"]:
    m=ps.Model(); m.hideOutput(); m.readProblem(f"/home/user/discopt/ref/miplib2017/{nm}.mps.gz")
    vs=m.getVars(); cs=m.getConss()
    from collections import Counter
    print(nm, len(vs), Counter(v.vtype() for v in vs), len(cs))
    obj=[(v.name,v.getObj()) for v in vs if v.getObj()!=0]; print(" obj nz",len(obj),obj[:5])
    # constraint shapes
    shp=Counter()
    for c in cs:
        lhs=m.getLhs(c); rhs=m.getRhs(c); co=m.getValsLinear(c)
        types=Counter(m.getVars()[0].vtype() for _ in [0])
        shp[(len(co), 'eq' if lhs==rhs else ('<=' if lhs<-1e19 else ('>=' if rhs>1e19 else 'rng')))]+=1
    print(" row shapes", shp.most_common(15))
    print(" bounds sample", Counter((v.getLbOriginal(), v.getUbOriginal(), v.vtype()) for v in vs).most_common(8))
