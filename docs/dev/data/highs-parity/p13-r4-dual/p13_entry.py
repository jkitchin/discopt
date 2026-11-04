import json, sys, numpy as np, importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('cap','discopt_benchmarks/scripts/capture_highs_parity_fixtures.py'); cap=importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
import discopt.solvers.lp_milp_highs as H
from discopt.solvers.lp_std_engine import HighsEngine, RustEngine
D=Path('python/tests/data/highs_parity'); man=json.loads((D/'manifest.json').read_text())
ents=[e for e in man['fixtures'] if e['role']=='R4']
rows=[]; ok=0; lost=0; worse=0
for e in ents:
    fx=cap.load_fixture(D/e['file']); rf=fx['rf']
    sf=H.StdForm.from_arrays(rf['c'], rf['A'].tocsc(), rf['row_lo'], rf['col_lo'], rf['col_hi'], rf['obj_const'])
    res={}
    for name,eng in (('highs',HighsEngine()),('rust',RustEngine())):
        raw=eng.solve(sf, time_limit=60, presolve=False, tight_tolerances=False)
        if raw.status!='optimal': res[name]=(None,raw.name); continue
        v=float(sf.c@raw.x)+sf.obj_const
        ns,prov,why=H.certify_lp_dual(raw.y, sf, v)
        res[name]=(ns,prov or 'declared-box')
    h,r=res['highs'][0],res['rust'][0]
    if h is None: verdict='highs-none'
    elif r is None: verdict='LOST'; lost+=1
    else:
        tol=H.CERT_ABS+H.CERT_REL*abs(h)
        verdict='ok' if r>=h-tol else 'WORSE'; worse+= verdict=='WORSE'
    ok+= verdict in ('ok','highs-none')
    rows.append({'file':e['file'],'instance':e['instance'],'highs':res['highs'],'rust':res['rust'],'verdict':verdict})
print('compared',len(rows),'ok',ok,'lost',lost,'worse',worse)
for x in rows:
    if x['verdict'] not in ('ok',): print(x['verdict'], x['instance'][:50], x['highs'], x['rust'])
Path(sys.argv[1]).write_text(json.dumps(rows,indent=1,default=str))
