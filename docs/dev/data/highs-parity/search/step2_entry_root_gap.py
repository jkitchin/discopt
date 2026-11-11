import json, os, subprocess, sys
REPO="/home/user/discopt"; sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R, highspy
tgt=sys.argv[1].split(",")
for mem in [m for m in R.population() if m["name"] in tgt]:
    o=mem["oracle_objective"]; g=lambda b: None if b is None else round(abs(o-b)/max(1,abs(o))*100,3)
    h=highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads",1)
    h.setOptionValue("mip_max_nodes",1); h.setOptionValue("time_limit",60.0)
    h.readModel(f"{REPO}/{mem['path']}"); h.run(); info=h.getInfo()
    hb=info.mip_dual_bound
    env=dict(os.environ, PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="rust")
    p=subprocess.run([sys.executable,f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py","worker",f"{REPO}/{mem['path']}","R","60","1e-4","1"],capture_output=True,text=True,env=env,timeout=400)
    line=[l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r=json.loads(line[-1][7:]) if line else {}
    print(f"{mem['name']:22} HiGHS root gap {g(hb)}%  ours root gap {g(r.get('bound'))}%  (ours status {r.get('status')}, nodes {r.get('nodes')})", flush=True)
