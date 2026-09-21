#!/usr/bin/env python
"""#1414 entry experiment: how far does big-M coefficient tightening (cure 1) close
the lifted model's scale hazard?

## What this measures, and why it is the deciding experiment

`reformulate_integer_bilinear` lifts `x_i * x_j` (integer factor) into binary
expansion plus per-bit big-M rows `v <= U*e`, `v >= x - U*(1-e)`. #1414's defect is
that the exit feasibility gate is an **absolute** `abs=1e-6` row violation, while
those rows' objective sensitivity is set by the big-M weights the lift introduced.
#1380 measured the conversion rate: an integer column inside `INT_TOL` carries up to
`|a_ij| * INT_TOL` of slack on row `i`.

So the hazard is a function of **`max |a_ij|` over the lifted linear rows**, and
cure 1's whole claim is that it shrinks that number. This script measures it before
and after, on real corpus instances (§2/§4 — the #727 lesson: a synthetic proxy can
show a gain that is 0.0 on the real class).

Cure 1's rule, for a *negative* binary coefficient (which is what every big-M row
has, and which `coefficient_strengthening.rs:230`'s `if coeff <= 0.0 { continue; }`
currently skips), is the existing Savelsbergh rule under the complementation
`x_k = 1 - z`. That composite collapses to:

    new_coeff_k = b - U_minus_k          (RHS unchanged)
    fires iff    a_k < b - U_minus_k < 0

which is what `_tighten_row` below applies. This script does **not** implement the
Rust pass; it simulates the rule on the lifted model so the decision to build it
rests on a measurement rather than on the algebra alone.

## Kill criterion

If the simulated rule does not materially reduce `max |a_ij|` on real lifted
instances -- specifically, if the median reduction factor is < 2x -- then cure 1 is
not the fix for this class and cure 2 (column-scaled tolerance) must be attempted
instead. That outcome is a pass for the experiment and a stop for the plan.

Usage:
    python -u discopt_benchmarks/scripts/issue1414_bigm_entry.py
    python -u discopt_benchmarks/scripts/issue1414_bigm_entry.py --solve-nvs07
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import traceback

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
CORPUS = REPO / "python" / "tests" / "data" / "minlplib_nl"

# `branching.rs`'s global integrality tolerance -- the multiplier in #1380's
# `|a_ij| * INT_TOL` slack mechanism.
INT_TOL = 1e-5
# The exit feasibility gate this slack is measured against (conftest.py abs=1e-6).
FEAS_ABS = 1e-6


def _rows(model):
    """Yield ``(coeffs, b)`` for every *linear* row, normalized to ``coeffs @ x <= b``.

    Constraints are stored normalized as ``body sense 0``, so ``body = c@x + off``
    gives ``b = -off``. A ``>=`` row is reflected. Nonlinear rows yield nothing --
    they carry no ``a_ij`` and are not what the big-M hazard lives on.
    """
    from discopt._relax.gdp_reformulate import _extract_body_coeffs

    n = sum(v.size for v in model._variables)
    for con in model._constraints:
        res = _extract_body_coeffs(con.body, model, n)
        if res is None:
            continue
        c_vec, off = res
        c_vec = np.asarray(c_vec, dtype=float)
        if con.sense == ">=":
            yield -c_vec, off
        elif con.sense == "<=":
            yield c_vec, -off
        else:  # "==" -- both directions; the rule needs an inequality
            continue


def _eq_rows(model):
    """Yield the coefficient vector of every linear **equality** row.

    These are the rows coefficient strengthening cannot touch
    (`coefficient_strengthening.rs` returns `None` for `ConstraintSense::Eq`,
    correctly -- the "slack at x_k = 1" semantics the rule rests on do not exist
    for an equality). On a lifted model they are the binary-expansion rows
    ``x_i = lo + sum 2^k e_k`` and the product expansions
    ``x_i*x_j = lo*x_j + sum 2^k v_k``, which is where the ``2^k`` weights live.
    """
    from discopt._relax.gdp_reformulate import _extract_body_coeffs

    n = sum(v.size for v in model._variables)
    for con in model._constraints:
        if con.sense != "==":
            continue
        res = _extract_body_coeffs(con.body, model, n)
        if res is None:
            continue
        yield np.asarray(res[0], dtype=float)


def _is_binary_flags(model) -> np.ndarray:
    """Flat per-slot boolean: is this column a binary variable?"""
    from discopt.modeling.core import VarType

    flags: list[bool] = []
    for v in model._variables:
        is_bin = v.var_type == VarType.BINARY
        flags.extend([is_bin] * v.size)
    return np.array(flags, dtype=bool)


def _bounds(model) -> tuple[np.ndarray, np.ndarray]:
    lo: list[float] = []
    hi: list[float] = []
    for v in model._variables:
        lb = np.broadcast_to(np.asarray(v.lb, dtype=float), (v.size,))
        ub = np.broadcast_to(np.asarray(v.ub, dtype=float), (v.size,))
        lo.extend(lb.tolist())
        hi.extend(ub.tolist())
    return np.array(lo), np.array(hi)


def _tighten_row(coeffs, b, is_bin, lo, hi):
    """Apply cure 1 to one ``coeffs @ x <= b`` row. Returns the new coefficients.

    Positive binary coefficients use the shipped Savelsbergh rule
    (``a' = a - slack``, ``b' = b - slack``); this function reports only the
    *negative* case, which is the one `coefficient_strengthening.rs` skips and the
    one every big-M row has. For ``a_k < 0`` the complemented rule collapses to
    ``new_coeff = b - U_minus_k`` with the RHS unchanged.
    """
    nz = np.nonzero(coeffs)[0]
    if nz.size == 0:
        return coeffs
    contrib = np.maximum(coeffs[nz] * lo[nz], coeffs[nz] * hi[nz])
    if not np.all(np.isfinite(contrib)):
        return coeffs  # an unbounded column makes U_minus_k meaningless
    total = float(contrib.sum())
    out = coeffs.copy()
    for pos, j in enumerate(nz):
        a_k = float(coeffs[j])
        if a_k >= 0.0 or not is_bin[j]:
            continue
        u_minus_k = total - float(contrib[pos])
        if not math.isfinite(u_minus_k):
            continue
        new_c = b - u_minus_k
        if a_k < new_c < 0.0:
            out[j] = new_c
    return out


def _max_abs(rows) -> float:
    m = 0.0
    for coeffs, _b in rows:
        if coeffs.size:
            m = max(m, float(np.max(np.abs(coeffs))))
    return m


def analyze(path: pathlib.Path) -> dict:
    """Lift one instance and measure max|a_ij| before and after cure 1."""
    from discopt._relax.integer_product_reform import (
        has_reformulation_work,
        reformulate_integer_bilinear,
    )
    from discopt.modeling.core import from_nl

    model = from_nl(str(path))
    if not has_reformulation_work(model):
        return {"name": path.stem, "lifted": False}

    lifted = reformulate_integer_bilinear(model)
    is_bin = _is_binary_flags(lifted)
    lo, hi = _bounds(lifted)

    rows = [(c.copy(), b) for c, b in _rows(lifted)]
    if not rows:
        return {"name": path.stem, "lifted": True, "linear_rows": 0}

    before = _max_abs(rows)
    tightened_rows = []
    n_tightened = 0
    for coeffs, b in rows:
        new_c = _tighten_row(coeffs, b, is_bin, lo, hi)
        n_tightened += int(np.count_nonzero(new_c != coeffs))
        tightened_rows.append((new_c, b))
    after = _max_abs(tightened_rows)

    # The weights cure 1 structurally cannot reach: linear equality rows.
    eq_max = 0.0
    n_eq = 0
    for c in _eq_rows(lifted):
        n_eq += 1
        if c.size:
            eq_max = max(eq_max, float(np.max(np.abs(c))))

    return {
        "name": path.stem,
        "lifted": True,
        "linear_rows": len(rows),
        "eq_rows": n_eq,
        "n_coeffs_tightened": n_tightened,
        "max_abs_before": before,
        "max_abs_after": after,
        "max_abs_eq": eq_max,
        # The hazard, defined identically before and after so the two are
        # comparable: the larger of the inequality weights (which cure 1 shrinks)
        # and the equality weights (which it cannot touch). Reporting
        # ``max_abs_before`` against ``hazard_after`` would compare an
        # inequality-only figure against one that includes equalities, and would
        # read as cure 1 having made a coefficient *larger* -- which it never does.
        "hazard_before": max(before, eq_max),
        "hazard_after": max(after, eq_max),
        "reduction_factor": (before / after) if after > 0 else float("inf"),
        # #1380's conversion: slack an in-tolerance integer column buys on a row.
        "buyable_slack_before": before * INT_TOL,
        "buyable_slack_after": after * INT_TOL,
    }


def solve_nvs07() -> dict:
    """Reproduce #1414's headline nvs07 numbers on the *shipped* code path."""
    from discopt._relax.integer_product_reform import reformulate_integer_bilinear
    from discopt.modeling.core import from_nl

    path = CORPUS / "nvs07.nl"
    model = from_nl(str(path))
    lifted = reformulate_integer_bilinear(model)
    res = model.solve(time_limit=60)
    return {
        "user_objective": None if res.objective is None else float(res.objective),
        "status": str(res.status),
        "minlplib_optimum": 4.0,
        "lifted_cols": sum(v.size for v in lifted._variables),
        "lifted_rows": len(lifted._constraints),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--solve-nvs07", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    corpus = pathlib.Path(args.corpus)
    files = sorted(corpus.glob("*.nl"))
    print(f"scanning {len(files)} instances from {corpus}", flush=True)

    results: list[dict] = []
    errors: list[tuple[str, str]] = []
    executed = 0

    for i, path in enumerate(files, 1):
        try:
            r = analyze(path)
        except Exception:
            # §7: never swallow. Record verbatim and keep going so one bad
            # instance cannot hide the population; re-raised below if ALL failed.
            errors.append((path.stem, traceback.format_exc()))
            print(f"  [{i}/{len(files)}] {path.stem:<22} ERROR", flush=True)
            continue
        results.append(r)
        if r.get("lifted") and r.get("linear_rows"):
            executed += 1
            print(
                f"  [{i}/{len(files)}] {path.stem:<22} ineq={r['linear_rows']:<5} "
                f"max|a|: {r['max_abs_before']:.4g} -> {r['max_abs_after']:.4g} "
                f"({r['n_coeffs_tightened']} coeffs) | eq={r['eq_rows']:<4} "
                f"max|a|_eq={r['max_abs_eq']:.4g} -> hazard {r['hazard_after']:.4g}",
                flush=True,
            )
        else:
            print(f"  [{i}/{len(files)}] {path.stem:<22} no lift work", flush=True)

    for name, tb in errors:
        print(f"\n--- ERROR on {name} ---\n{tb}", flush=True)

    lifted = [r for r in results if r.get("lifted") and r.get("linear_rows")]

    print("\n" + "=" * 72, flush=True)
    print(f"instances scanned         : {len(files)}", flush=True)
    print(f"instances with lift work  : {len(lifted)}", flush=True)
    print(f"instances erroring        : {len(errors)}", flush=True)
    print(f"EXECUTED MEASUREMENTS     : {executed}", flush=True)

    verdict = "NO DATA"
    if lifted:
        # Overall hazard reduction: what cure 1 can shrink, against what remains
        # once it has done everything it can (equality weights included).
        hazard_factors = sorted(
            (r["hazard_before"] / r["hazard_after"]) if r["hazard_after"] > 0 else float("inf")
            for r in lifted
        )
        med = hazard_factors[len(hazard_factors) // 2]
        # The worst instance, and *its own* before/after -- not two maxima taken
        # over different instances, which would not be a before/after pair.
        worst = max(lifted, key=lambda r: r["hazard_after"])
        worst_before = worst["hazard_before"]
        worst_after = worst["hazard_after"]
        n_helped = sum(1 for r in lifted if r["n_coeffs_tightened"] > 0)
        n_eq_dominates = sum(1 for r in lifted if r["max_abs_eq"] >= r["max_abs_after"])
        print(f"median hazard reduction   : {med:.4g}", flush=True)
        print(f"instances where cure 1 fired : {n_helped}/{len(lifted)}", flush=True)
        print(
            f"instances where equality weights dominate : "
            f"{n_eq_dominates}/{len(lifted)}   <- cure 1 cannot reach these",
            flush=True,
        )
        print(
            f"worst instance ({worst['name']}) hazard before/after : "
            f"{worst_before:.6g} / {worst_after:.6g}",
            flush=True,
        )
        print(
            f"worst buyable slack       : {worst_before * INT_TOL:.4g} -> "
            f"{worst_after * INT_TOL:.4g}   (gate abs={FEAS_ABS:g})",
            flush=True,
        )
        # Kill criterion, stated in the docstring, evaluated here.
        verdict = "CURE 1 VIABLE" if med >= 2.0 else "CURE 1 INSUFFICIENT -> need cure 2"
        print(f"\nKILL CRITERION (median hazard reduction >= 2x): {verdict}", flush=True)

    if args.solve_nvs07:
        print("\n--- nvs07 shipped-path solve ---", flush=True)
        info = solve_nvs07()
        for k, v in info.items():
            print(f"  {k:<22}: {v}", flush=True)
        results.append({"name": "nvs07_solve", **info})

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps({"results": results, "verdict": verdict}, indent=2)
        )
        print(f"\nwrote {args.out}", flush=True)

    if executed == 0:
        # §6: a probe that measured nothing must not read as a pass.
        print("\nFAIL: zero measurements executed -- the probe did not fire.", flush=True)
        if errors:
            raise RuntimeError(f"probe measured nothing; {len(errors)} instances errored")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
