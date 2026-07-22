"""discopt-vs-SCIP head-to-head on the small GDPlib subset (issue #823).

Reproduces the table in ``docs/dev/gdplib-benchmarking.md``. For each model it
reformulates with big-M, times **discopt** and **SCIP** on the *same* ``.nl`` at a
fixed wall budget, and classifies discopt's outcome against the SCIP-certified
:func:`benchmarks.gdplib_runner.reference_optima`. Requires the ``[gdplib]`` extra
plus ``pyscipopt``; run from ``discopt_benchmarks/``::

    python scripts/reeval_gdplib.py                 # 60 s/solve (default)
    REEVAL_LIMIT=120 REEVAL_METHOD=hull python scripts/reeval_gdplib.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Allow `python scripts/reeval_gdplib.py` from the discopt_benchmarks/ root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import gdplib_runner as gr  # noqa: E402
from benchmarks.metrics import SolveStatus  # noqa: E402

TIME_LIMIT = float(os.environ.get("REEVAL_LIMIT", "60"))
METHOD = os.environ.get("REEVAL_METHOD", "bigm")

# The small subset with (mostly) SCIP-certified optima; the last two are the pair
# SCIP cannot certify in the budget, kept to compare incumbents.
MODELS = [
    "jobshop", "ex1_linan_2023", "positioning", "small_batch", "cstr",
    "spectralog", "syngas", "water_network", "modprodnet",
    "methanol", "batch_processing", "gdp_col",
]

REF = gr.reference_optima()


def scip_solve(spec, method, time_limit):
    """Full SCIP stats (status/obj/gap/wall), not just the certified-only oracle."""
    from pyomo.core import TransformationFactory
    from pyomo.repn.plugins.nl_writer import NLWriter

    try:
        import pyscipopt
    except ImportError:
        return None
    try:
        m = spec.builder()
        TransformationFactory(f"gdp.{method}").apply_to(m)
        with tempfile.TemporaryDirectory(prefix="reeval_scip_") as d:
            nl = os.path.join(d, "m.nl")
            with open(nl, "w") as s:
                NLWriter().write(
                    m, s, linear_presolve=False, scale_model=False,
                    skip_trivial_constraints=False,
                )
            sm = pyscipopt.Model()
            sm.hideOutput()
            sm.setParam("limits/time", float(time_limit))
            sm.readProblem(nl)
            t0 = time.time()
            sm.optimize()
            wall = time.time() - t0
            status = sm.getStatus()
            obj = float(sm.getObjVal()) if sm.getNSols() > 0 else None
            try:
                gap = abs(sm.getGap())
            except Exception:
                gap = None
            proved = status == "optimal" and gap is not None and gap <= 1e-6
            return {"status": status, "obj": obj, "gap": gap, "wall": wall, "proved": proved}
    except Exception as exc:  # noqa: BLE001
        return {"status": f"error:{type(exc).__name__}", "obj": None, "gap": None,
                "wall": None, "proved": False}


def classify(name, run):
    ref = REF.get(name)
    r = run.discopt
    if run.false_optimum or run.bound_crosses:
        return "UNSOUND"
    if r.status == SolveStatus.ERROR:
        return "error"
    if r.objective is None:
        return "no-incumbent"
    if ref is None:
        return "feasible(no-ref)"
    tol = 1e-4 + 1e-3 * abs(ref)
    if abs(r.objective - ref) <= tol and r.is_solved:
        return "solved-optimal"
    if abs(r.objective - ref) <= tol:
        return "optimal-not-proven"
    return "feasible-loose"


def main():
    rows = []
    print(f"# discopt vs SCIP on GDPlib ({METHOD}, {TIME_LIMIT:.0f}s/solve)\n")
    hdr = f"{'model':18s} {'certified':>12s} | {'discopt':>28s} | {'SCIP':>26s}"
    print(hdr)
    print("-" * len(hdr))
    for name in MODELS:
        specs = gr.discover_models(include=[name])
        if not specs:
            print(f"{name:18s}  (not discovered)")
            continue
        spec = specs[0]
        run = gr.solve_model(spec, method=METHOD, time_limit=TIME_LIMIT, oracle=False)
        d = run.discopt
        cls = classify(name, run)
        scip = scip_solve(spec, METHOD, TIME_LIMIT)

        ref = REF.get(name)
        ref_s = f"{ref:.6g}" if ref is not None else "—"
        d_obj = f"{d.objective:.6g}" if d.objective is not None else "—"
        d_cell = f"{cls:18s} {d_obj:>9s} {d.wall_time:5.1f}s"
        if scip:
            s_obj = f"{scip['obj']:.6g}" if scip["obj"] is not None else "—"
            s_flag = "opt" if scip["proved"] else scip["status"][:8]
            s_wall = f"{scip['wall']:.1f}s" if scip["wall"] is not None else "—"
            s_cell = f"{s_flag:8s} {s_obj:>9s} {s_wall:>6s}"
        else:
            s_cell = "n/a"
        print(f"{name:18s} {ref_s:>12s} | {d_cell:>28s} | {s_cell:>26s}")
        rows.append({
            "model": name, "certified": ref, "discopt_class": cls,
            "discopt_status": d.status.value, "discopt_obj": d.objective,
            "discopt_wall": d.wall_time, "discopt_nodes": d.node_count,
            "false_optimum": run.false_optimum, "bound_crosses": run.bound_crosses,
            "scip": scip,
        })

    n = len(rows)
    solved = sum(1 for r in rows if r["discopt_class"] == "solved-optimal")
    opt_np = sum(1 for r in rows if r["discopt_class"] == "optimal-not-proven")
    loose = sum(1 for r in rows if r["discopt_class"] == "feasible-loose")
    noinc = sum(1 for r in rows if r["discopt_class"] == "no-incumbent")
    unsound = sum(1 for r in rows if r["discopt_class"] == "UNSOUND")
    scip_proved = sum(1 for r in rows if r["scip"] and r["scip"]["proved"])
    print("\n" + "=" * 60)
    print(f"discopt: solved-optimal={solved} optimal-not-proven={opt_np} "
          f"feasible-loose={loose} no-incumbent={noinc} UNSOUND={unsound} / {n}")
    print(f"SCIP proved optimal: {scip_proved}/{n}")
    print("=" * 60)

    if os.environ.get("REEVAL_JSON"):
        Path(os.environ["REEVAL_JSON"]).write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nwrote {os.environ['REEVAL_JSON']}")


if __name__ == "__main__":
    main()
