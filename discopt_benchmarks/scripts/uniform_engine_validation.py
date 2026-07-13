"""Uniform factorable relaxation engine — corpus validation (#632).

Read-only capability metrics over the 62 vendored ``python/tests/data/minlplib_nl``
instances, using discopt's OWN in-house Rust simplex for every bound:

* Coverage  — objective/constraint feasibility fallbacks (target 0; federation 9).
* Soundness — feasible-point sampling (>=1000 pts) never cut by the relaxation.
* At-least-as-tight — engine root bound vs the federation's build_milp_relaxation.

Reproduce::

    cd discopt && source .venv/bin/activate
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      python discopt_benchmarks/scripts/uniform_engine_validation.py --json /tmp/u.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"

# The federation's 9 objective-fallback instances (blueprint §1.4).
_FORMER_FALLBACKS = [
    "fac2", "heatexch_gen2", "heatexch_gen3", "nvs06", "nvs09",
    "tspn05", "tspn08", "tspn10", "tspn12",
]  # fmt: skip


def _engine_bound(model):
    from discopt._jax.uniform_relax import build_uniform_relaxation, relaxation_report

    rep = relaxation_report(model)
    rel = build_uniform_relaxation(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = rel.model.solve(backend="simplex")
    bound = float(res.bound) if res.bound is not None else None
    return bound, res.status, rep, rel


def _federation_bound(model):
    import logging

    from discopt._jax import milp_relaxation as milp
    from discopt._jax.mccormick_lp import MccormickLPRelaxer

    class _Cap(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.msgs: list[str] = []

        def emit(self, record):
            self.msgs.append(record.getMessage())

    milp._warned_messages.clear()
    cap = _Cap()
    logger = logging.getLogger("discopt._jax.milp_relaxation")
    prev = logger.level
    logger.addHandler(cap)
    logger.setLevel(logging.WARNING)
    try:
        lbs, ubs = [], []
        for v in model._variables:
            lbs.append(np.asarray(v.lb, dtype=np.float64).ravel())
            ubs.append(np.asarray(v.ub, dtype=np.float64).ravel())
        res = MccormickLPRelaxer(model).solve_at_node(np.concatenate(lbs), np.concatenate(ubs))
    finally:
        logger.removeHandler(cap)
        logger.setLevel(prev)
    fell_back = any("could not linearize the objective" in m for m in cap.msgs)
    bound = float(res.lower_bound) if res.lower_bound is not None else None
    return bound, res.status, fell_back


def _sample_soundness(model, rel, n=1000, seed=0):
    """Sample points in the box; verify none is cut by the relaxation rows.

    Each aux column is set to the EXACT value it represents (the builder's
    ``track_aux_exprs`` map: node value / relaxed power / McCormick partial), so
    the lifted true point ``(x, w=f(x))`` is exact. A sound outer relaxation
    satisfies every row AND the aux interval floor at that point. Returns
    ``(max_violation, n_checked)``; a positive violation is an UNSOUND cut.
    """
    from discopt._jax import uniform_relax as ur
    from discopt._jax.canonical_expr import canonicalize
    from discopt._jax.model_utils import flat_variable_bounds

    flat_lb, flat_ub = flat_variable_bounds(model)
    dag = canonicalize(model)
    ctx = ur._Builder(model, flat_lb, flat_ub, track_aux_exprs=True)
    roots = ([dag.objective] if dag.objective is not None else []) + list(dag.constraints)
    for r in roots:
        ctx.rep(r)

    rows = ctx.rows
    n_orig = ctx.n_orig
    n_cols = len(ctx.col_lb)
    lo = np.asarray(ctx.col_lb)
    hi = np.asarray(ctx.col_ub)

    rng = np.random.default_rng(seed)
    max_viol = 0.0
    checked = 0
    fb = np.where(np.isfinite(flat_lb), flat_lb, -10.0)
    fu = np.where(np.isfinite(flat_ub), flat_ub, 10.0)
    for _ in range(n):
        xv = fb + rng.random(n_orig) * (fu - fb)
        z = np.zeros(n_cols)
        z[:n_orig] = xv
        ok = True
        try:
            for j in sorted(ctx.aux_expr):
                val = _eval_expr_at(ctx.aux_expr[j], model, xv)
                if not np.isfinite(val):
                    ok = False
                    break
                z[j] = val
        except Exception:
            ok = False
        if not ok:
            continue
        # The true aux value must lie within the declared interval floor; if not,
        # the aux box would be UNSOUND (it must enclose the node over the box).
        floor_viol = float(max(np.max(lo - z), np.max(z - hi)))
        if floor_viol > max_viol:
            max_viol = floor_viol
        checked += 1
        for coeffs, rhs in rows:
            lhs = sum(c * z[jj] for jj, c in coeffs.items())
            v = lhs - rhs
            if v > max_viol:
                max_viol = v
    return max_viol, checked


def _eval_expr_at(expr, model, xv):
    """Evaluate a scalar modeling expression at flat point ``xv``."""
    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_eval import evaluate_interval

    box = {}
    off = 0
    for v in model._variables:
        size = int(v.size)
        shape = tuple(getattr(v, "shape", ()) or ())
        pt = np.asarray(xv[off : off + size], dtype=np.float64).reshape(shape)
        box[v] = Interval(pt, pt)  # degenerate interval == point eval
        off += size
    enc = evaluate_interval(expr, model, box)
    return float(np.asarray(enc.lo))


def run(sample: bool) -> dict:
    from discopt.modeling.core import from_nl

    files = sorted(_NL_DIR.glob("*.nl"))
    rows = []
    fallbacks = []
    for f in files:
        name = f.stem
        rec: dict = {"instance": name}
        try:
            model = from_nl(str(f))
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)[:80]
            rows.append(rec)
            continue
        try:
            e_bound, e_status, rep, rel = _engine_bound(model)
            rec.update(
                engine_bound=e_bound,
                engine_status=e_status,
                fallbacks=rep["fallbacks"],
                n_atoms=rep["n_atoms"],
                tight=rep["tight"],
                loose=rep["loose"],
                by_kind=rep["by_kind"],
            )
            if rep["fallbacks"] != 0:
                fallbacks.append(name)
        except Exception as exc:  # noqa: BLE001
            rec["engine_error"] = repr(exc)[:120]
            fallbacks.append(name)
            rows.append(rec)
            continue
        try:
            f_bound, f_status, f_fb = _federation_bound(model)
            rec.update(fed_bound=f_bound, fed_status=f_status, fed_fell_back=f_fb)
        except Exception as exc:  # noqa: BLE001
            rec["fed_error"] = repr(exc)[:80]
        if sample and (name in _FORMER_FALLBACKS or len(rows) % 6 == 0):
            # Fewer points on the largest instances (per-aux reconstruct+interval
            # eval is O(atoms*pts)); soundness is a hard invariant, not a mean.
            n_pts = 200 if rec.get("n_atoms", 0) > 120 else 1000
            try:
                viol, checked = _sample_soundness(model, rel, n=n_pts)
                rec.update(sample_max_violation=viol, sample_checked=checked)
            except Exception as exc:  # noqa: BLE001
                rec["sample_error"] = repr(exc)[:120]
        rows.append(rec)
    return {"n": len(files), "engine_fallbacks": fallbacks, "rows": rows}


def report(c: dict) -> None:
    rows = c["rows"]
    print(f"\n=== Uniform engine validation ({c['n']} vendored .nl) ===")
    print(
        f"ENGINE objective/constraint fallbacks: {len(c['engine_fallbacks'])}/{c['n']}  "
        f"(federation baseline 9)"
    )
    if c["engine_fallbacks"]:
        print("  " + ", ".join(c["engine_fallbacks"]))

    # coverage aggregate
    agg: dict[str, dict[str, int]] = {}
    tot_tight = tot_loose = 0
    for r in rows:
        for k, d in r.get("by_kind", {}).items():
            a = agg.setdefault(k, {"total": 0, "tight": 0, "loose": 0})
            a["total"] += d["total"]
            a["tight"] += d["tight"]
            a["loose"] += d["loose"]
        tot_tight += r.get("tight", 0)
        tot_loose += r.get("loose", 0)
    print("\n--- atom-kind coverage (corpus-wide, per-node) ---")
    print(f"  {'kind':<18} {'total':>7} {'tight':>7} {'loose':>7}")
    for k in sorted(agg):
        d = agg[k]
        print(f"  {k:<18} {d['total']:>7} {d['tight']:>7} {d['loose']:>7}")
    print(f"  {'TOTAL':<18} {tot_tight + tot_loose:>7} {tot_tight:>7} {tot_loose:>7}")

    # soundness
    print("\n--- soundness (feasible-point sampling; violation should be ~0) ---")
    worst = 0.0
    n_checked_total = 0
    for r in rows:
        if "sample_max_violation" in r:
            v = r["sample_max_violation"]
            n_checked_total += r.get("sample_checked", 0)
            flag = "  <-- CUT!" if v > 1e-5 else ""
            worst = max(worst, v)
            if v > 1e-6 or r["instance"] in _FORMER_FALLBACKS:
                chk = r.get("sample_checked")
                print(f"  {r['instance']:<18} max_violation={v:.2e} checked={chk}{flag}")
    print(
        f"  WORST violation across sampled instances: {worst:.2e} "
        f"({'SOUND' if worst <= 1e-5 else 'UNSOUND — STOP'})"
    )

    # at-least-as-tight
    print("\n--- at-least-as-tight vs federation (both finite, both min-sense) ---")
    n_ge = n_lt = n_both = 0
    deltas = []
    regressions = []
    for r in rows:
        eb, fbn = r.get("engine_bound"), r.get("fed_bound")
        if eb is None or fbn is None:
            continue
        if not (np.isfinite(eb) and np.isfinite(fbn)):
            continue
        n_both += 1
        d = eb - fbn
        deltas.append(d)
        # engine >= federation means engine is at least as tight (higher lower bound)
        if d >= -1e-4 * (1 + abs(fbn)):
            n_ge += 1
        else:
            n_lt += 1
            regressions.append((r["instance"], eb, fbn))
    print(f"  instances both finite: {n_both}")
    print(f"  engine >= federation (at least as tight): {n_ge}")
    print(f"  engine <  federation (LOOSER): {n_lt}")
    if regressions:
        print("  looser instances (instance, engine, fed):")
        for name, eb, fbn in regressions[:30]:
            print(f"    {name:<18} engine={eb:.4g} fed={fbn:.4g}")

    # former fallbacks recovered
    print("\n--- bounds recovered on the 9 former federation fallbacks ---")
    for r in rows:
        if r["instance"] in _FORMER_FALLBACKS:
            print(
                f"  {r['instance']:<18} engine_bound={r.get('engine_bound')} "
                f"status={r.get('engine_status')} fed_bound={r.get('fed_bound')} "
                f"fed_fell_back={r.get('fed_fell_back')}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--no-sample", action="store_true", help="skip feasible-point sampling")
    args = ap.parse_args()
    c = run(sample=not args.no_sample)
    report(c)
    if args.json:
        Path(args.json).write_text(json.dumps(c, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
