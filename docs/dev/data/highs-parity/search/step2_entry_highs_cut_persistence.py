import sys, time
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R, highspy
tgt=sys.argv[1].split(",")
arms={"default":{}, "no_persist":{"mip_lp_age_limit":0,"mip_pool_age_limit":0,"mip_pool_soft_limit":1}}
for mem in [m for m in R.population() if m["name"] in tgt]:
    o=mem["oracle_objective"]
    out=[]
    for a,opt in arms.items():
        h=highspy.Highs(); h.setOptionValue("output_flag",False); h.setOptionValue("threads",1); h.setOptionValue("time_limit",20.0)
        for k,v in opt.items(): h.setOptionValue(k,v)
        h.readModel(f"/home/user/discopt/{mem['path']}"); t=time.time(); h.run(); i=h.getInfo()
        out.append(f"{a}: {h.modelStatusToString(h.getModelStatus())[:8]} gap {abs(o-i.mip_dual_bound)/max(1,abs(o))*100:.3f}% nodes {i.mip_node_count} {time.time()-t:.1f}s")
    print(f"{mem['name']:20}", " | ".join(out), flush=True)
