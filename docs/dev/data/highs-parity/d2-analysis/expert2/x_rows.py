import pyscipopt as ps
m=ps.Model(); m.hideOutput(); m.readProblem("/home/user/discopt/ref/miplib2017/neos-3610051-istra.mps.gz")
cs=m.getConss()
seen=0
for c in cs:
    co=m.getValsLinear(c)
    if len(co)<=3 and seen<14:
        print(m.getLhs(c), {k:v for k,v in co.items()}, m.getRhs(c)); seen+=1
v={x.name:x for x in m.getVars()}
c=[c for c in cs if len(m.getValsLinear(c))==77][0]; co=m.getValsLinear(c); print(m.getLhs(c),m.getRhs(c), set(co.values()))
