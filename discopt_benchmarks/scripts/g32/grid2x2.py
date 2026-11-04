"""#1917 G3.2 step 2: MTP root bound, coef OFF/ON x tableau c-MIR OFF/ON, by cut profile."""
import os, sys, pickle, time
os.environ["DISCOPT_PROFILE"] = "1"
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import discopt._rust as R
assert b"DISCOPT_RUST_MILP_TABLEAU_CMIR" in open(R.__file__, "rb").read()
OPT = -6797.5397
d = pickle.load(open(CAP + "/loa__med_term_purchasing.pkl", "rb")); d.pop("had_debug_hook", None)
d["initial_incumbent"] = None
profs = [(16, 1, False), (200, 10, True), (2000, 30, False)]
n = 0
for rc, cr, sel in profs:
    for coef in ("0", "1"):
        for cm in ("0", "1"):
            os.environ["DISCOPT_RUST_MILP_COEF_TIGHTEN"] = coef
            os.environ["DISCOPT_RUST_MILP_TABLEAU_CMIR"] = cm
            k = dict(d); k.update(root_cuts=rc, cut_rounds=cr, cut_select=sel, max_nodes=2, time_limit_s=300.0)
            R.profile_reset_py()
            t = time.perf_counter(); r = R.solve_milp_csc_py(**k); w = time.perf_counter() - t
            pc = R.profile_counters_py()
            assert r[3] <= OPT + 1e-6 * abs(OPT), ("root bound above the optimum", r[3])
            c = {kk.replace("SepCmir", ""): v for kk, v in pc.items() if kk.startswith("SepCmir")}
            print(f"cuts={rc:5d} rounds={cr:3d} sel={sel!s:5} coef={coef} cmir={cm} root={r[3]:.3f} "
                  f"wall={w:.2f} rounds_run={pc.get('RootCutRounds',0)} gmi={pc.get('SepGomoryCutsEmitted',0)} cmir={c}", flush=True)
            if cm == "1":
                assert c.get("Calls", 0) > 0 and c.get("Rows", 0) > 0, "c-MIR arm never separated"
            n += 1
print("PROBE_DONE", n); assert n == 4 * len(profs)
