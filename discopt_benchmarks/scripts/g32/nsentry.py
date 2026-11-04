"""NS-vs-vertex entry probe. Run once with DISCOPT_MILP_NS_NODE_BOUND unset (NS) and
once with =0 (vertex); per pkl and per coef-tighten arm, record the 2-node bound.
usage: nsentry.py out.jsonl pkl..."""
import os, sys, json, pickle
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import discopt._rust as R
assert b"DISCOPT_RUST_MILP_COEF_TIGHTEN" in open(R.__file__, "rb").read(), "marker absent"
import numpy as np
ns_mode = os.environ.get("DISCOPT_MILP_NS_NODE_BOUND", "default")
out = open(sys.argv[1], "a"); k = 0
for p in sys.argv[2:]:
    d = pickle.load(open(p, "rb")); d.pop("had_debug_hook", None)
    kw = dict(d); kw["max_nodes"] = 2; kw["time_limit_s"] = 60.0
    if kw.get("initial_incumbent") is not None: kw["initial_incumbent"] = np.ascontiguousarray(kw["initial_incumbent"])
    row = {"inst": os.path.basename(p)[:-4], "ns": ns_mode}
    for arm in ("0", "1"):
        os.environ["DISCOPT_RUST_MILP_COEF_TIGHTEN"] = arm
        r = R.solve_milp_csc_py(**kw); row[f"b{arm}"] = r[3]; k += 1
    out.write(json.dumps(row) + "\n"); out.flush(); print(row, flush=True)
print("PROBE_DONE", k, flush=True)
if k == 0: sys.exit(1)
