"""Per-node LP-value parity panel for the incremental McCormick path (issue #861).

``IncrementalMcCormickLP._validate`` compares the patched rows against a cold
build on six synthetic regime boxes at construction. This panel is its per-node
analogue on REAL instances and REALISTIC boxes: for each named instance it samples
branching-style sub-boxes of the true root box (integer pins, sign-regime splits,
interior shrinks) and asserts that the patched LP and the cold-build LP agree on

  * the objective value (to 1e-9 relative), and
  * the infeasibility verdict (both infeasible or both feasible).

A newly admitted instance must pass this before its admission is trusted: the
whole premise of the fast path is that it changes speed, never the bound.

Usage::

    python -u discopt_benchmarks/scripts/incremental_node_parity_panel.py \\
        --instances prob02,prob03,st_e01,st_e08,st_e09 --boxes 24

Prints an executed-comparison count and exits non-zero if it is zero (a panel
that compared nothing must never read as a pass — CLAUDE.md rule 6) or if any
comparison disagreed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
    "tests",
    "data",
    "minlplib",
)


def _solve_dense(a_ub, b, bounds, c):
    """Solve ``min c·x  s.t. A x <= b, bounds`` with scipy HiGHS. Returns
    ``(status, objective)`` where status is 'optimal' | 'infeasible' | 'other'."""
    from scipy.optimize import linprog

    lo = np.asarray(bounds, dtype=float)[:, 0]
    hi = np.asarray(bounds, dtype=float)[:, 1]
    res = linprog(
        c=np.asarray(c, dtype=float),
        A_ub=a_ub,
        b_ub=np.asarray(b, dtype=float),
        bounds=list(zip(lo, hi, strict=True)),
        method="highs",
    )
    if res.status == 0:
        return "optimal", float(res.fun)
    if res.status == 2:
        return "infeasible", None
    return "other", None


def sample_boxes(lb, ub, is_int, count, seed=12345):
    """Branching-style reachable sub-boxes of the root: interior shrinks, one-sided
    splits, and integer pins — the shapes a real B&B tree produces."""
    rng = np.random.default_rng(seed)
    n = lb.size
    out = [(lb.copy(), ub.copy())]
    for t in range(count - 1):
        lo = lb.copy()
        hi = ub.copy()
        for i in range(n):
            w = hi[i] - lo[i]
            if w <= 0 or not np.isfinite(w):
                continue
            mode = int(rng.integers(0, 4))
            if mode == 0:  # left half (branch down)
                hi[i] = lo[i] + 0.5 * w
            elif mode == 1:  # right half (branch up)
                lo[i] = hi[i] - 0.5 * w
            elif mode == 2:  # interior shrink
                lo[i] = lo[i] + 0.25 * w
                hi[i] = hi[i] - 0.25 * w
            # mode 3: leave the dimension alone
            if is_int[i]:
                lo[i] = np.ceil(lo[i] - 1e-9)
                hi[i] = np.floor(hi[i] + 1e-9)
                if lo[i] > hi[i]:
                    lo[i] = hi[i] = np.floor(0.5 * (lo[i] + hi[i]))
                # every few boxes, PIN an integer (the degenerate regime)
                if t % 3 == 0 and rng.random() < 0.4:
                    v = float(rng.integers(int(lo[i]), int(hi[i]) + 1)) if hi[i] > lo[i] else lo[i]
                    lo[i] = hi[i] = v
        if np.all(lo <= hi):
            out.append((lo, hi))
    return out


def run_instance(name, n_boxes):
    from discopt._jax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.modeling.core import VarType, from_nl

    path = os.path.join(_CORPUS, f"{name}.nl")
    if not os.path.exists(path):
        print(f"{name}: NOT IN CORPUS — skipped")
        return 0, 0
    model = from_nl(path)
    terms = classify_nonlinear_terms(model)
    inc = IncrementalMcCormickLP(model, terms, deadline=None)
    if not inc.ok:
        print(f"{name}: structure declined ({inc.decline_reason}) — nothing to compare")
        return 0, 0
    lb = np.array([float(np.min(v.lb)) for v in model._variables])
    ub = np.array([float(np.max(v.ub)) for v in model._variables])
    is_int = np.array([v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables])
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        print(f"{name}: infinite root box — skipped (needs a presolved box, T4)")
        return 0, 0

    compared = 0
    bad = 0
    for k, (lo, hi) in enumerate(sample_boxes(lb, ub, is_int, n_boxes)):
        a_pat, b_pat, bd_pat = inc._patch(lo, hi)
        a_cold, b_cold, bd_cold, c_cold, _info, _relax = inc._full_build(lo, hi)
        sp_status, sp_obj = _solve_dense(a_pat, b_pat, bd_pat, inc.c)
        cf_status, cf_obj = _solve_dense(a_cold, b_cold, bd_cold, c_cold)
        compared += 1
        if sp_status != cf_status:
            bad += 1
            print(f"  {name} box[{k}]: STATUS patched={sp_status} cold={cf_status}")
            continue
        if sp_status == "optimal" and not np.isclose(sp_obj, cf_obj, rtol=1e-9, atol=1e-9):
            bad += 1
            print(
                f"  {name} box[{k}]: OBJECTIVE patched={sp_obj!r} cold={cf_obj!r} "
                f"delta={sp_obj - cf_obj:.3e}"
            )
    print(f"{name}: {compared} boxes compared, {bad} disagreements")
    return compared, bad


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", required=True, help="comma-separated instance names")
    ap.add_argument("--boxes", type=int, default=24)
    args = ap.parse_args(argv)
    total = 0
    total_bad = 0
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        c, b = run_instance(name, args.boxes)
        total += c
        total_bad += b
    print(f"\nexecuted LP comparisons: {total}")
    print(f"disagreements: {total_bad}")
    if not total:
        print("PANEL MEASURED NOTHING", file=sys.stderr)
        return 2
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
