"""P1.3's entry experiment (../p13-r4-dual/p13_entry.py), re-run with the candidate
pipeline `convex_node_bound._safe_bound` uses since P6: the cheap NS bound on the
engine's dual, and on the zero dual, always (the larger is kept), then, if
not closed, the exact correction on both (the larger is kept). Both arms run that
pipeline; only the engine differs. Bar (P1.3, pre-registered): 0 bounds lost and none looser than
CERT_ABS + CERT_REL*|v|. Prints the executed count; exits 1 when nothing compared."""
import json, sys, numpy as np, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("cap", "discopt_benchmarks/scripts/capture_highs_parity_fixtures.py")
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
import discopt.solvers.lp_milp_highs as H
from discopt.solvers.lp_std_engine import HighsEngine, RustEngine

def pipeline(y, sf, v):
    b, _, _ = H.certify_lp_dual(y, sf, v, exact=False)
    tol = H.CERT_ABS + H.CERT_REL * abs(v)
    b0, _, _ = H.certify_lp_dual(np.zeros(sf.m), sf, v, exact=False)
    if b0 is not None and (b is None or b0 > b):
        b = b0
    if b is None or v - b > tol:
        for cand in (y, np.zeros(sf.m)):
            be, _, _ = H.certify_lp_dual(cand, sf, v)
            if be is not None and (b is None or be > b):
                b = be
    return b

D = Path("python/tests/data/highs_parity"); man = json.loads((D / "manifest.json").read_text())
rows = []; n = lost = worse = 0
for e in [e for e in man["fixtures"] if e["role"] == "R4"]:
    rf = cap.load_fixture(D / e["file"])["rf"]
    sf = H.StdForm.from_arrays(rf["c"], rf["A"].tocsc(), rf["row_lo"], rf["col_lo"], rf["col_hi"], rf["obj_const"])
    res = {}
    for name, eng in (("highs", HighsEngine()), ("rust", RustEngine())):
        raw = eng.solve(sf, time_limit=60, presolve=False, tight_tolerances=False)
        if raw.status != "optimal":
            res[name] = None; continue
        v = float(sf.c @ raw.x) + sf.obj_const
        res[name] = pipeline(raw.y, sf, v)
    h, r = res["highs"], res["rust"]
    if h is None: verdict = "highs-none"
    elif r is None: verdict = "LOST"; lost += 1
    else:
        verdict = "ok" if r >= h - (H.CERT_ABS + H.CERT_REL * abs(h)) else "WORSE"
        worse += verdict == "WORSE"
    n += 1
    rows.append({"file": e["file"], "highs": h, "rust": r, "verdict": verdict})
print("compared", n, "lost", lost, "worse", worse, "rust tighter", sum(1 for x in rows if x["verdict"] == "ok" and x["rust"] is not None and x["highs"] is not None and x["rust"] > x["highs"] + 1e-12))
for x in rows:
    if x["verdict"] != "ok": print(x)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
sys.exit(0 if n else 1)
