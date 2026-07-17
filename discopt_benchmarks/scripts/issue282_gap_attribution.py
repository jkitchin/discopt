r"""#282 — attribute the syn*/rsyn* global-search gap to dual, primal, or throughput.

Runs the issue's seven process-synthesis instances against the MINLPLib snapshot
oracle at several wall-clock budgets and records, per run, the telemetry needed to
decide *why* the gap is open rather than merely that it is:

  * ``root_bound`` / ``root_time``  — is the budget even reaching the tree?
  * ``bound`` vs ``=opt=``          — dual excess (bound looser than the true optimum)
  * ``objective`` vs ``=opt=``      — primal deficit (incumbent worse than the optimum)
  * ``node_count`` / ``wall_time``  — node throughput
  * first-incumbent time            — derived from a ``node_callback`` trace

All seven instances carry an ``=opt=`` tag, i.e. a proven optimum that is both the
primal and the dual fence. That makes the decomposition exact rather than heuristic:
for a maximization problem,

    reported_gap = bound - objective = (bound - opt) + (opt - objective)
                                       \_dual excess_/   \_primal deficit_/

Both terms are non-negative for a sound solver, so a soundness violation shows up
directly as a negative term. Note the repo's ``generality_sweep.load_solu`` reads
only ``=best=``/``=bestdual=`` and returns nothing for these instances; this script
keys on ``=opt=``.

Usage:
    python discopt_benchmarks/scripts/issue282_gap_attribution.py --budgets 5,60
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
NL_DIR = SNAPSHOT / "minlplib" / "nl"
SOLU = SNAPSHOT / "minlplib.solu"

# The panel from issue #282.
INSTANCES = [
    "rsyn0805m",
    "rsyn0810m",
    "rsyn0815m",
    "syn15m02hfsg",
    "syn30hfsg",
    "syn40hfsg",
    "syn40m",
]


def load_opt(names: set[str]) -> dict[str, float]:
    """Read proven optima (``=opt=``) for ``names`` from the snapshot's .solu."""
    out: dict[str, float] = {}
    with open(SOLU) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "=opt=" and parts[1] in names:
                with contextlib.suppress(ValueError):
                    out[parts[1]] = float(parts[2])
    return out


def _f(x):
    """Scrub non-finite / missing values to None so the JSON stays clean."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if abs(xf) < 1e29 else None


def solve_one(name: str, budget: float) -> dict:
    from discopt._jax.convexity import classify_model
    from discopt.modeling.core import ObjectiveSense, from_nl

    rec: dict = {"instance": name, "budget": budget}
    nl = NL_DIR / f"{name}.nl"
    if not nl.exists():
        rec["error"] = "missing_nl"
        return rec

    t0 = time.perf_counter()
    model = from_nl(str(nl))
    rec["parse_time"] = time.perf_counter() - t0

    # The sense lives on the Objective, not the Model. Resolve it explicitly and
    # refuse to guess: a wrong sense silently flips every attribution term and
    # manufactures phantom soundness violations.
    objective = getattr(model, "_objective", None)
    sense = getattr(objective, "sense", None)
    if sense not in (ObjectiveSense.MAXIMIZE, ObjectiveSense.MINIMIZE):
        raise RuntimeError(
            f"{name}: could not resolve objective sense (got {sense!r}); "
            "refusing to attribute with an assumed sense"
        )
    is_max = sense == ObjectiveSense.MAXIMIZE
    rec["is_max"] = bool(is_max)
    rec["n_vars"] = len(getattr(model, "_variables", []))
    rec["n_cons"] = len(getattr(model, "_constraints", []))

    try:
        conv, _ = classify_model(model, use_certificate=True)
        rec["model_is_convex"] = bool(conv)
    except Exception as exc:  # pragma: no cover - diagnostic only
        rec["model_is_convex"] = None
        rec["convexity_error"] = f"{type(exc).__name__}: {exc}"

    # Trace incumbent/bound over time. node_callback does not disable the NLP-BB
    # auto-select (only lazy_constraints / incumbent_callback do), so this still
    # measures the default path.
    traj: list[list] = []

    def _cb(ctx, _model, _sink=traj):
        # The solver swallows callback exceptions (solver.py:10404), so a broken
        # recorder would look like an empty trace rather than an error.
        with contextlib.suppress(Exception):
            _sink.append(
                [
                    float(ctx.elapsed_time),
                    int(ctx.node_count),
                    None if ctx.best_bound is None else float(ctx.best_bound),
                    None if ctx.incumbent_obj is None else float(ctx.incumbent_obj),
                ]
            )

    t0 = time.perf_counter()
    res = model.solve(time_limit=budget, gap_tolerance=1e-4, node_callback=_cb)
    rec["harness_wall"] = time.perf_counter() - t0

    for field in (
        "status",
        "objective",
        "bound",
        "gap",
        "node_count",
        "root_bound",
        "root_gap",
        "root_time",
        "wall_time",
        "nlp_bb",
        "gap_certified",
        "convex_fast_path",
        "rust_time",
        "jax_time",
        "python_time",
    ):
        val = getattr(res, field, None)
        rec[field] = val if isinstance(val, (bool, str, int)) or val is None else _f(val)
    rec["objective"] = _f(getattr(res, "objective", None))
    rec["bound"] = _f(getattr(res, "bound", None))
    rec["root_bound"] = _f(getattr(res, "root_bound", None))

    # Time to first incumbent, derived from the trace (no built-in field exists).
    tfi = None
    for elapsed, _n, _b, inc in traj:
        if inc is not None:
            tfi = elapsed
            break
    rec["time_to_first_incumbent"] = tfi
    rec["trace_points"] = len(traj)
    rec["trace"] = traj[:400]
    return rec


def attribute(rec: dict, opt: float | None) -> dict:
    """Split the gap into dual excess and primal deficit against a proven optimum."""
    rec["opt"] = opt
    obj, bnd = rec.get("objective"), rec.get("bound")
    if opt is None:
        return rec
    if "is_max" not in rec:
        raise RuntimeError(f"{rec.get('instance')}: no sense recorded; cannot attribute")
    is_max = rec["is_max"]

    # Orient so that "excess" and "deficit" are non-negative for a sound solver.
    if is_max:
        rec["dual_excess"] = None if bnd is None else bnd - opt
        rec["primal_deficit"] = None if obj is None else opt - obj
        rec["root_dual_excess"] = None if rec.get("root_bound") is None else rec["root_bound"] - opt
    else:
        rec["dual_excess"] = None if bnd is None else opt - bnd
        rec["primal_deficit"] = None if obj is None else obj - opt
        rec["root_dual_excess"] = None if rec.get("root_bound") is None else opt - rec["root_bound"]

    denom = max(1.0, abs(opt))
    for key in ("dual_excess", "primal_deficit", "root_dual_excess"):
        val = rec.get(key)
        rec[key + "_rel"] = None if val is None else val / denom

    # Soundness: a valid bound never lands on the wrong side of a proven optimum,
    # and a feasible incumbent never beats it. Tolerance mirrors conftest rel=1e-4.
    tol = 1e-4 * denom
    rec["bound_unsound"] = bool(rec.get("dual_excess") is not None and rec["dual_excess"] < -tol)
    rec["incumbent_unsound"] = bool(
        rec.get("primal_deficit") is not None and rec["primal_deficit"] < -tol
    )

    # Which bucket dominates?
    if obj is None:
        rec["verdict"] = "no_incumbent"
    elif rec.get("dual_excess") is None:
        rec["verdict"] = "no_bound"
    else:
        de, pd = rec["dual_excess"], rec["primal_deficit"]
        total = de + pd
        rec["dual_share"] = None if total <= 0 else de / total
        if total <= tol:
            rec["verdict"] = "closed"
        elif de > 2 * pd:
            rec["verdict"] = "dual_dominated"
        elif pd > 2 * de:
            rec["verdict"] = "primal_dominated"
        else:
            rec["verdict"] = "mixed"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="5,60", help="comma-separated seconds")
    ap.add_argument("--instances", default=",".join(INSTANCES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    budgets = [float(b) for b in args.budgets.split(",") if b.strip()]
    names = [s for s in args.instances.split(",") if s.strip()]
    opts = load_opt(set(names))

    rows = []
    for budget in budgets:
        for name in names:
            print(f"=== {name} @ {budget}s ===", flush=True)
            try:
                rec = solve_one(name, budget)
            except Exception as exc:
                rec = {
                    "instance": name,
                    "budget": budget,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rec = attribute(rec, opts.get(name))
            rows.append(rec)
            print(
                "    status={status} obj={objective} bound={bound} nodes={node_count} "
                "root_t={root_time} verdict={verdict}".format(
                    status=rec.get("status"),
                    objective=rec.get("objective"),
                    bound=rec.get("bound"),
                    node_count=rec.get("node_count"),
                    root_time=(
                        None if rec.get("root_time") is None else round(rec["root_time"], 2)
                    ),
                    verdict=rec.get("verdict"),
                ),
                flush=True,
            )

    out = args.out or (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"issue282_attribution_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"rows": rows, "opts": opts}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
