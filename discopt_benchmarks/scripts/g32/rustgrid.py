"""Rust-driver root bound on the MTP master: coef OFF/ON x cut profile. max_nodes=2 (root)."""
import os, sys, pickle, time, numpy as np
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import discopt._rust as R
assert b"MilpSeedRefused" in open(R.__file__, "rb").read()
d = pickle.load(open(CAP + "/loa__med_term_purchasing.pkl", "rb")); d.pop("had_debug_hook", None)
d["initial_incumbent"] = None
profs = [(16, 1, False), (200, 10, True), (200, 15, True), (2000, 15, False), (2000, 30, False)]
n = 0
for rc, cr, sel in profs:
    for arm in ("0", "1"):
        os.environ["DISCOPT_RUST_MILP_COEF_TIGHTEN"] = arm
        k = dict(d); k.update(root_cuts=rc, cut_rounds=cr, cut_select=sel, max_nodes=2, time_limit_s=120.0)
        t = time.perf_counter(); r = R.solve_milp_csc_py(**k); w = time.perf_counter() - t
        print(f"cuts={rc:5d} rounds={cr:3d} sel={sel!s:5} coef={arm} root={r[3]:.3f} wall={w:.2f}", flush=True); n += 1
print("PROBE_DONE", n); assert n == 2 * len(profs)
