"""#2168 entry experiment at the kernel level: the captured cstr homog-hull kernel spec
(``../2142-exact-hull/scip/cstr_homog_spec.npz``, the spec SCIP certifies in 7,470 nodes
and whose SCIP-presolved translation discopt's tree certifies at 12,931), solved by the
native kernel with the solver's own main-call options, ORIGINAL vs PRESOLVED
(``presolve_spec``). No pricer, primal hook or NLP seed (they need the Pyomo model, and
gdplib's cstr is not installable in this container), the same in both arms.

  DISCOPT_PROFILE=1 PYTHONHASHSEED=0 python -u kernel_arm.py {orig|pre|multi|scip} MAX_NODES [seeded]

``pre`` is stages 1-3 (``presolve_spec``), ``multi`` adds stage 4 (``multi_aggregate=True``).

``scip`` is the positive control: SCIP's presolved model translated back to a kernel spec
(``../2142-scip-presolved/cstr_homog_scippre_spec.npz``); its incumbent is not expanded
(``translate.py`` owns that map), so only the oracle gates apply to it. ``seeded`` hands
every arm ``initial_incumbent = 3.0620145766187727``, the value of the verified incumbent
the full solve's NLP seed finds (``../2142-scip-presolved/arm_*_summary.log``): a real
feasible value, so a sound cutoff, standing in for the seed this harness cannot run.

Asserts the loaded tree (CLAUDE.md section 8), the presolve fired in ``pre``, the expanded
incumbent satisfies the ORIGINAL spec, soundness against the oracle, and prints an
executed-assertion count (section 6)."""
import os, sys, time
import numpy as np
import discopt, discopt._rust as R
import discopt.solver as S
from discopt._relax.spatial_presolve import presolve_spec

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
assert hasattr(S, "_kernel_presolve_main"), "presolve marker absent: wrong tree"
n_assert = 2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "python", "tests"))
from test_kernel_presolve import residual  # noqa: E402

arm, max_nodes = sys.argv[1], int(sys.argv[2])
seeded = len(sys.argv) > 3 and sys.argv[3] == "seeded"
src = ("2142-scip-presolved", "cstr_homog_scippre_spec.npz") if arm == "scip" else ("2142-exact-hull", "scip", "cstr_homog_spec.npz")
d = dict(np.load(os.path.join(HERE, "..", *src)))
sign, off = float(d["meta_obj_sense_sign"]), float(d["meta_obj_offset"])
spec = {k: (int(v) if v.ndim == 0 else v) for k, v in d.items() if not k.startswith("meta_")}
# the main call's options, from the solver's own helpers (defaults, no env overrides)
kw = dict(max_nodes=max_nodes, gap_tol=1e300, rel_gap_tol=1e-4, abs_gap_tol=S._DEFAULT_ABS_GAP_TOL,
          obj_offset=off, time_limit_s=None, cold_dual_start=bool(S._tuning().lp_cold_dual_start))
if S._scale_free_branching_enabled():
    kw["scale_free_branching"] = True
if S._kernel_pseudocost_branching_enabled():
    kw["pseudocost_branching"] = True
    kw["pseudocost_reliability"] = S._kernel_pseudocost_reliability()
kw.update(S._kernel_spatial_branching_kwargs())
wm = S._native_warm_start_mode()
if wm >= 1:
    kw["warm_start_children"] = True
if wm >= 2:
    kw["warm_flip_boxed"] = True
cr = S._kernel_convex_cut_rounds()
if cr is not None:
    kw["cut_rounds_root"], kw["cut_rounds_node"] = cr
if S._kernel_int_probe_enabled():
    kw["int_fix_probe"] = True
if S._kernel_undecided_retry_enabled():
    kw["undecided_unpropagated_retry"] = True
if S._kernel_dual_proof_enabled():
    kw["dual_proof_propagation"] = True
if seeded:
    kw["initial_incumbent"] = 3.0620145766187727
print("options:", {k: v for k, v in kw.items()}, flush=True)
expand = None
run_spec = spec
if arm in ("pre", "multi"):
    t = time.perf_counter()
    out = presolve_spec(spec, multi_aggregate=arm == "multi")
    assert out is not None, "presolve declined"; n_assert += 1
    run_spec, expand, info = out
    print(f"presolve: {time.perf_counter() - t:.2f}s", {k: v for k, v in info.items() if k not in ("compress", "eliminated")}, flush=True)
t0 = time.perf_counter()
res = R.solve_spatial_tree_py(**run_spec, **kw)
wall = time.perf_counter() - t0
x = res.get("incumbent_x")
if x is not None and len(x) and arm != "scip":
    x = np.asarray(x)
    if expand is not None:
        x = expand(x)
    v = residual(spec, x)
    assert v <= 1e-5, f"incumbent violates the ORIGINAL spec by {v}"; n_assert += 1
bound = sign * (float(res["bound"]) + off)
inc = None if res.get("incumbent") is None else sign * (float(res["incumbent"]) + off)
ctr = dict(R.profile_counters_py()) if os.environ.get("DISCOPT_PROFILE") == "1" else {}
print("counters:", {k: v for k, v in ctr.items() if k in ("LpVerdictNumerical", "LpInfeasUncertified", "DualProofLearned", "DualProofFathoms")})
assert bound <= 3.0620073 + 1e-6, "bound above the oracle"; n_assert += 1
if inc is not None:
    assert inc >= 3.0620073 - 1e-5, "incumbent below the oracle"; n_assert += 1
print(f"RESULT arm={arm} seeded={seeded} max_nodes={max_nodes} status={res['status']} bound={bound!r} incumbent={inc!r} "
      f"nodes={res['node_count']} wall={wall:.1f}s", flush=True)
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
