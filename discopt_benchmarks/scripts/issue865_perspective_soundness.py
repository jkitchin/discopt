#!/usr/bin/env python
"""Issue #865 soundness evidence for the perspective term.

Two independent checks per routed instance, over random points in the variable box:

1. EXACTNESS — the marshaled row `g_spec(x)` (evaluated with the same formula the
   Rust kernel uses) must equal the PRISTINE model's constraint residual `g(x)`.
   Any drift means the lift `s·h(·/s) → affine + perspective` is not an identity.
2. CONVEXITY — each routed nonlinear row must satisfy the midpoint inequality
   `g(λa+(1-λ)b) <= λ g(a) + (1-λ) g(b)`. A violation means the row is not convex
   and the OA tangent would be an invalid (unsound) relaxation.
"""

import os
import sys

os.environ.setdefault("DISCOPT_CONVEX_KERNEL", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
from discopt._relax.gdp_reformulate import reformulate_gdp  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt._relax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt.solvers import _convex_kernel as ck  # noqa: E402

_FUNC_NP = {
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log1p": np.log1p,
    "sqr": np.square,
}


def row_value(d, x):
    """g(x) for a `_Decomp`, using the Rust kernel's term semantics."""
    v = d.const + sum(k * x[c] for c, k in d.aff.items())
    for t in d.terms:
        a = t["arg_const"] + sum(k * x[c] for c, k in t["arg_aff"].items())
        f = _FUNC_NP[t["func"]]
        if t["sc_aff"] is None:
            v += t["coeff"] * f(a)
        else:
            s = t["sc_const"] + sum(k * x[c] for c, k in t["sc_aff"].items())
            v += t["coeff"] * s * f(a / s)
    return float(v)


def decomps_for(model):
    """Re-run _build's row loop, returning [(row_index, sign, _Decomp)] for nl rows."""
    m = reformulate_gdp(model, method="big-m")
    lb, ub = flat_variable_bounds(m)
    lb, ub = lb.astype(float), ub.astype(float)
    ev = NLPEvaluator(m)
    n = len(lb)
    rng = np.random.default_rng(0)
    lo = np.where(np.isfinite(lb), lb, 0.0)
    hi = np.where(np.isfinite(ub), ub, lo + 5.0)
    xa = lo + rng.random(n) * (hi - lo)
    xb = lo + rng.random(n) * (hi - lo)
    ja, jb = ev.evaluate_jacobian(xa), ev.evaluate_jacobian(xb)
    lin = np.all(np.isclose(ja, jb, atol=1e-9), axis=1)
    offsets = ck._flat_offsets(m)
    out = []
    for i, con in enumerate(m._constraints):
        if lin[i]:
            continue
        s = con.sense if isinstance(con.sense, str) else con.sense.value
        d = ck._decompose(ck._constraint_expr(m, i), offsets)
        sign = -1.0 if s == ">=" else 1.0
        if sign < 0:
            d.scale(-1.0)
        out.append((i, sign, d))
    return m, lb, ub, ev, out


def sample(lb, ub, rng, n_pts):
    """Random points in the box; unbounded sides are capped to keep samples finite."""
    lo = np.where(np.isfinite(lb), lb, -10.0)
    hi = np.where(np.isfinite(ub), ub, lo + 20.0)
    return lo + rng.random((n_pts, len(lb))) * (hi - lo)


def check(name, path, n_pts=400):
    model = dm.from_nl(path)
    if ck.build_convex_spec(model) is None:
        print(f"{name}: NOT ROUTED (skipped)")
        return True
    m, lb, ub, ev, rows = decomps_for(model)
    if not rows:
        print(f"{name}: routed, no nonlinear rows (skipped)")
        return True
    kinds = sorted({t["func"] for _i, _s, d in rows for t in d.terms})
    n_persp = sum(1 for _i, _s, d in rows for t in d.terms if t["sc_aff"] is not None)
    rng = np.random.default_rng(12345)
    pts = sample(lb, ub, rng, n_pts)

    # 1. exactness vs the pristine model. `n_exact` counts comparisons ACTUALLY
    # made: the finiteness guard below would otherwise let this degrade to a
    # silent no-op that reports `worst=0.0` and reads as a pass (CLAUDE.md
    # "Measurement & instrumentation discipline" rule 6).
    worst = 0.0
    n_exact = 0
    for x in pts:
        g = np.asarray(ev.evaluate_constraints(x), float)
        for i, sign, d in rows:
            ref = sign * g[i]
            got = row_value(d, x)
            if np.isfinite(ref) and np.isfinite(got):
                worst = max(worst, abs(ref - got) / max(1.0, abs(ref)))
                n_exact += 1
    ok_exact = worst < 1e-9 and n_exact > 0

    # 2. midpoint convexity of every routed row
    worst_cx = 0.0
    n_cx = 0
    left, right = sample(lb, ub, rng, n_pts), sample(lb, ub, rng, n_pts)
    for a, b in zip(left, right, strict=True):
        for lam in (0.25, 0.5, 0.75):
            mid = lam * a + (1 - lam) * b
            for _i, _s, d in rows:
                gm, ga, gb = row_value(d, mid), row_value(d, a), row_value(d, b)
                if not all(np.isfinite(v) for v in (gm, ga, gb)):
                    continue
                viol = gm - (lam * ga + (1 - lam) * gb)
                worst_cx = max(worst_cx, viol / max(1.0, abs(gm)))
                n_cx += 1
    ok_cx = worst_cx < 1e-9 and n_cx > 0

    print(
        f"{name}: rows={len(rows)} persp_terms={n_persp} funcs={kinds} "
        f"exactness worst_rel_err={worst:.3e} over {n_exact} cmps "
        f"{'OK' if ok_exact else 'FAIL'} | "
        f"convexity worst_violation={worst_cx:.3e} over {n_cx} cmps "
        f"{'OK' if ok_cx else 'FAIL'}"
    )
    return ok_exact and ok_cx


if __name__ == "__main__":
    targets = sys.argv[1:] or ["python/tests/data/minlplib_nl/syn05hfsg.nl"]
    all_ok = all(check(os.path.basename(t)[:-3], t) for t in targets)
    print("\nALL OK" if all_ok else "\nFAILURES PRESENT")
    sys.exit(0 if all_ok else 1)
