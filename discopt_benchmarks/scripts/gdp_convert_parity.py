"""Pyomo GDP -> native discopt converter parity over GDPlib (#1775 G0.4).

For each GDPlib model this runs spec §8's three checks on
:func:`discopt.pyomo.gdp.convert_pyomo_gdp`:

1. **evaluation** -- every converted row and the objective at random points,
   Pyomo vs discopt's interval evaluator (:func:`check_evaluation_parity`);
2. **structure** -- active Pyomo disjunctions / disjuncts / logical constraints
   vs the discopt GDP tree;
3. **optimum** (``--optimum``) -- Pyomo ``gdp.bigm`` -> ``.nl`` -> discopt vs the
   native model under discopt big-M, both under the same time limit; compared only
   where both certify optimality. The ``.nl`` side is a *reference*, not an
   oracle, on nested disjunctions: Pyomo's ``gdp.bigm`` can compute a negative M
   for a row that holds on the whole box, and a nested row's relaxation sums one
   such term per enclosing indicator, so the transformed model can cut points the
   GDP admits (seen in #1800's differential probe). The native incumbent is also
   loaded into the original Pyomo GDP and checked there; on a mismatch, that check
   says which side is wrong.

A model whose conversion is *refused* is reported with the refusal, never skipped
silently. Prints one line per model (flushed), then the totals, and exits non-zero
when any check disagrees or when no check ran at all (CLAUDE.md §6).

    python -u discopt_benchmarks/scripts/gdp_convert_parity.py --suite gdplib_small
    python -u discopt_benchmarks/scripts/gdp_convert_parity.py --all --optimum --time-limit 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _structure(pm, conv) -> dict:
    import pyomo.environ as pyo
    from discopt.gdp import gdp_tree
    from pyomo.core.base.logical_constraint import LogicalConstraint
    from pyomo.gdp import Disjunct, Disjunction

    descend = (pyo.Block, Disjunct)
    # Pyomo-side counts, over what a solve would see: disjunctions reachable
    # through active disjuncts (a disjunction inside a deactivated disjunct is moot).
    nodes = [n for n in gdp_tree(conv.model).walk() if n.kind != "implication"]
    want = {
        "disjunctions": len(conv.disjunctions),
        "disjuncts": sum(len(dj.disjuncts) for dj in conv.disjunctions),
        "logical": sum(
            1
            for _ in pm.component_data_objects(LogicalConstraint, active=True, descend_into=descend)
        ),
        "pyomo_disjunctions_all": sum(
            1 for _ in pm.component_data_objects(Disjunction, active=True, descend_into=descend)
        ),
    }
    got = {
        "disjunctions": len(nodes),
        "disjuncts": sum(len(n.disjuncts) for n in nodes),
        "logical": len(gdp_tree(conv.model).logic),
    }
    ok = (
        all(want[k] == got[k] for k in got)
        and want["disjunctions"] == want["pyomo_disjunctions_all"]
    )
    return {"ok": ok, "pyomo": want, "discopt": got}


def _pyomo_violation(pm, conv, result) -> dict:
    """Load discopt's incumbent into the Pyomo model and measure it against the
    *original GDP* there, in Pyomo's semantics: global rows; each disjunction's
    selection count (one for XOR, at least one for OR) wherever every enclosing
    disjunct is selected -- under a deselected parent a nested disjunction and
    its indicators are free; the rows of each disjunct selected along its whole
    chain; and every logical constraint. Pyomo evaluates everything, so this
    shares no code with discopt's own verifier."""
    import numpy as np
    import pyomo.environ as pyo
    from pyomo.core.base.logical_constraint import LogicalConstraint
    from pyomo.gdp import Disjunct

    def val(col) -> float:
        return float(np.asarray(result.x[col.name]).reshape(-1)[0])

    for v, col in conv.variables.items():
        v.set_value(val(col), skip_validation=True)
    for b, boolean in conv.booleans.items():
        b.set_value(bool(round(val(boolean.variable))))
        binary = b.get_associated_binary()
        if binary is not None and not binary.fixed:
            binary.set_value(round(val(boolean.variable)), skip_validation=True)

    def row_viol(c) -> float:
        body = pyo.value(c.body)
        lo = 0.0 if c.lb is None else float(pyo.value(c.lb)) - body
        hi = 0.0 if c.ub is None else body - float(pyo.value(c.ub))
        return max(0.0, lo, hi) / max(1.0, abs(body))

    def on(d) -> int:
        return int(round(pyo.value(d.binary_indicator_var)))

    def parent_of(c):
        p = c.parent_block()
        while p is not None and p.ctype is not Disjunct:
            p = p.parent_block()
        return p

    def selected(d) -> bool:
        while d is not None:
            if not on(d):
                return False
            d = parent_of(d)
        return True

    worst, bad_logic, bad_count, checked = 0.0, 0, 0, 0
    for c in pm.component_data_objects(pyo.Constraint, active=True, descend_into=pyo.Block):
        worst = max(worst, row_viol(c))
        checked += 1
    for dj in conv.disjunctions:
        parent = parent_of(dj)
        if parent is not None and not selected(parent):
            continue
        n = sum(on(d) for d in dj.disjuncts)
        bad_count += int((n != 1) if dj.xor else (n < 1))
        checked += 1
        for d in dj.disjuncts:
            if d.active and on(d):
                for c in d.component_data_objects(
                    pyo.Constraint, active=True, descend_into=pyo.Block
                ):
                    worst = max(worst, row_viol(c))
                    checked += 1
    for lc in pm.component_data_objects(LogicalConstraint, active=True, descend_into=pyo.Block):
        bad_logic += int(not bool(pyo.value(lc.expr)))
        checked += 1
    return {
        "max_rel_violation": worst,
        "bad_counts": bad_count,
        "bad_logic": bad_logic,
        "checked": checked,
    }


def _optimum(spec, time_limit: float) -> dict:
    import discopt.pyomo  # noqa: F401  (registers the solver)
    import pyomo.environ as pyo
    from discopt.pyomo.gdp import convert_pyomo_gdp

    out: dict = {}
    pm = spec.builder()
    conv = convert_pyomo_gdp(pm)
    t0 = time.perf_counter()
    r = conv.model.solve(gdp_method="big-m", time_limit=time_limit)
    out["native"] = {
        "status": r.status,
        "objective": r.objective,
        "bound": r.bound,
        "wall": round(time.perf_counter() - t0, 2),
    }
    if r.x is not None and r.objective is not None:
        out["native"]["in_pyomo"] = _pyomo_violation(pm, conv, r)
    pm = spec.builder()
    pyo.TransformationFactory("gdp.bigm").apply_to(pm)
    obj = next(pm.component_data_objects(pyo.Objective, active=True))
    t0 = time.perf_counter()
    res = pyo.SolverFactory("discopt").solve(pm, options={"time_limit": time_limit})
    tc = str(res.solver.termination_condition)
    val = pyo.value(obj, exception=False) if tc in ("optimal", "feasible") else None
    out["bigm_nl"] = {"status": tc, "objective": val, "wall": round(time.perf_counter() - t0, 2)}
    both = out["native"]["status"] == "optimal" and tc == "optimal"
    out["compared"] = both
    # The native dual bound may never cross an optimum the .nl route certified.
    if tc == "optimal" and r.bound is not None:
        minimize = obj.sense == pyo.minimize
        slack = 1e-6 * max(1.0, abs(float(val)))
        crossed = r.bound > float(val) + slack if minimize else r.bound < float(val) - slack
        out["bound_crosses_nl_optimum"] = bool(crossed)
    if both:
        a, b = float(out["native"]["objective"]), float(val)
        out["ok"] = abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))
    return out


def main(argv: list[str] | None = None) -> int:
    import discopt
    from discopt.pyomo.gdp import (
        PyomoGDPConversionError,
        check_evaluation_parity,
        convert_pyomo_gdp,
    )

    from benchmarks import gdplib_runner

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="gdplib_small", choices=sorted(gdplib_runner.SUITES))
    ap.add_argument("--all", action="store_true", help="every discoverable GDPlib model")
    ap.add_argument("--include", default="", help="comma-separated model names")
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--optimum", action="store_true")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    print("discopt:", discopt.__file__, flush=True)

    if args.include:
        names = [n for n in args.include.split(",") if n]
        specs = gdplib_runner.discover_models(include=names)
    elif args.all:
        specs = gdplib_runner.discover_models()
    else:
        specs = gdplib_runner.discover_models(include=list(gdplib_runner.SUITES[args.suite]))

    records = []
    totals = {
        "models": 0,
        "converted": 0,
        "refused": 0,
        "build_error": 0,
        "row_checks": 0,
        "objective_checks": 0,
        "structure_ok": 0,
        "optimum_compared": 0,
        "incumbents_checked": 0,
        "incumbent_bad": 0,
        "bound_crossings": 0,
        "optimum_ok": 0,
        "failures": 0,
    }
    for spec in specs:
        totals["models"] += 1
        rec: dict = {"model": spec.name}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pm = spec.builder()
            except (ImportError, OSError) as e:  # an external tool or data file missing
                rec["build_error"] = f"{type(e).__name__}: {str(e)[:120]}"
                totals["build_error"] += 1
                print(f"{spec.name:32s} BUILD-ERROR {rec['build_error']}", flush=True)
                records.append(rec)
                continue
            try:
                t0 = time.perf_counter()
                conv = convert_pyomo_gdp(pm)
                rec["convert_s"] = round(time.perf_counter() - t0, 2)
            except PyomoGDPConversionError as e:
                rec["refused"] = str(e)
                totals["refused"] += 1
                print(f"{spec.name:32s} REFUSED {e}", flush=True)
                records.append(rec)
                continue
            totals["converted"] += 1
            try:
                rep = check_evaluation_parity(conv, n_points=args.points)
                rec["evaluation"] = {
                    "ok": True,
                    "rows": rep.row_checks,
                    "objective": rep.objective_checks,
                    "max_abs_error": rep.max_abs_error,
                }
                totals["row_checks"] += rep.row_checks
                totals["objective_checks"] += rep.objective_checks
            except AssertionError as e:
                rec["evaluation"] = {"ok": False, "error": str(e)}
                totals["failures"] += 1
            st = _structure(pm, conv)
            rec["structure"] = st
            totals["structure_ok"] += int(st["ok"])
            totals["failures"] += int(not st["ok"])
            if args.optimum:
                opt = _optimum(spec, args.time_limit)
                rec["optimum"] = opt
                if opt["compared"]:
                    totals["optimum_compared"] += 1
                    totals["optimum_ok"] += int(opt["ok"])
                    totals["failures"] += int(not opt["ok"])
                if opt.get("bound_crosses_nl_optimum"):
                    totals["failures"] += 1
                    totals["bound_crossings"] += 1
                inp = opt["native"].get("in_pyomo")
                if inp is not None:
                    totals["incumbents_checked"] += 1
                    bad = inp["max_rel_violation"] > 1e-5 or inp["bad_counts"] or inp["bad_logic"]
                    totals["incumbent_bad"] += int(bool(bad))
                    totals["failures"] += int(bool(bad))
        ev = rec["evaluation"]
        line = (
            f"{spec.name:32s} eval={'ok' if ev['ok'] else 'FAIL'}"
            f"({ev.get('rows', 0)} rows, {ev.get('objective', 0)} obj)"
            f" structure={'ok' if st['ok'] else 'FAIL'} {st['discopt']}"
        )
        if args.optimum:
            o = rec["optimum"]
            line += (
                f" native={o['native']['status']}:{o['native']['objective']}"
                f"[bound {o['native']['bound']}]"
                f"({o['native']['wall']}s) bigm.nl={o['bigm_nl']['status']}:"
                f"{o['bigm_nl']['objective']}({o['bigm_nl']['wall']}s)"
                f" {'MATCH' if o.get('ok') else ('MISMATCH' if o['compared'] else 'not-compared')}"
            )
            if "in_pyomo" in o["native"]:
                line += f" in_pyomo={o['native']['in_pyomo']}"
        if not ev["ok"]:
            line += f" :: {ev['error']}"
        print(line, flush=True)
        records.append(rec)

    print("totals:", json.dumps(totals), flush=True)
    if args.out:
        Path(args.out).write_text(
            json.dumps({"totals": totals, "records": records}, indent=1, default=str)
        )
    executed = totals["row_checks"] + totals["objective_checks"]
    if executed == 0:
        print("no check executed", flush=True)
        return 2
    return 1 if totals["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
