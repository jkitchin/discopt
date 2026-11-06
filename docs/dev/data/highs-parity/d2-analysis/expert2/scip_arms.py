import sys, json, pyscipopt as ps
from pyscipopt import SCIP_EVENTTYPE
P="/home/user/discopt/ref/miplib2017/{}.mps.gz"
class RowSpy(ps.Eventhdlr):
    def eventinit(self): self.model.catchEvent(SCIP_EVENTTYPE.ROWADDEDLP, self)
    def eventexit(self): self.model.dropEvent(SCIP_EVENTTYPE.ROWADDEDLP, self)
    def eventexec(self, ev):
        r=ev.getRow(); d=self.model.getDepth()
        nm=r.name
        if not nm.startswith(("gom","scg")) and "gom" not in nm: 
            self.c["other"]=self.c.get("other",0)+1; return
        k=("root" if d==0 else "tree")+("_local" if r.isLocal() else "_global")
        self.c[k]=self.c.get(k,0)+1
def run(nm, extra, tl=30.0, keep=("gomory","gomorymi")):
    m=ps.Model(); m.hideOutput(); m.readProblem(P.format(nm))
    m.setParam("limits/time",tl); m.setParam("limits/gap",1e-4); m.setParam("presolving/maxrounds",0)
    fams=sorted({k.split("/")[1] for k in m.getParams() if k.startswith("separating/") and k.endswith("/freq")})
    for f in fams:
        if f not in keep: m.setParam(f"separating/{f}/freq",-1)
    for k,v in extra.items(): m.setParam(k,v)
    spy=RowSpy(); spy.c={}; m.includeEventhdlr(spy,"rowspy","")
    m.optimize()
    return dict(status=m.getStatus(), nodes=m.getNNodes(), t=round(m.getSolvingTime(),1), bound=round(m.getDualbound(),4), rows=spy.c)
if __name__=="__main__":
    arms=json.loads(sys.argv[2]); n=0
    for nm in sys.argv[1].split(","):
        for name,extra in arms.items():
            r=run(nm, extra); n+=1
            print(f"{nm[-5:]} {name:34s} {r}", flush=True)
    print("executed",n); sys.exit(0 if n else 1)
