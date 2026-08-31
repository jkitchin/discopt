"""Feasible-point sampling for the #1141 fractional-node cuts (CLAUDE.md §5).

Every row the node separator returns is a claim: "no feasible point of the MINLP
violates me". This wraps the separator, keeps every row, and tests each against a
reference point that IS feasible (the OFF arm's incumbent, independently
feasibility-checked here against the model's own evaluator).

Prints an executed-check count and exits non-zero if it is zero (§6).
"""
import os, sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2
from discopt._tape_nlp_evaluator import make_evaluator
from discopt._relax.model_utils import flat_variable_bounds

KW = dict(n=40, K=6, spread=0.001, cap_scale=0.7)


def flat_point(model, xdict):
    lb, _ub = flat_variable_bounds(model)
    out = np.zeros(len(lb))
    k = 0
    for v in model._variables:
        val = xdict[v.name]
        arr = np.atleast_1d(np.asarray(val, float)).ravel()
        out[k:k + v.size] = arr
        k += v.size
    return out


def feasible(model, x, tol=1e-6):
    ev = make_evaluator(model)
    g = np.asarray(ev.evaluate_constraints(x), float)
    senses = [c.sense if isinstance(c.sense, str) else c.sense.value for c in model._constraints]
    worst = 0.0
    for gi, s in zip(g, senses):
        if s == "<=":
            worst = max(worst, gi)
        elif s == ">=":
            worst = max(worst, -gi)
        else:
            worst = max(worst, abs(gi))
    return worst


# --- reference: OFF arm ------------------------------------------------------
os.environ["DISCOPT_OA_NODE_CUTS"] = "0"
m_off = portfolio2.build(**KW)
r_off = m_off.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                    time_limit=120, gap_tolerance=1e-4)
x_ref = flat_point(m_off, r_off.x)
viol = feasible(m_off, x_ref)
print(f"OFF arm: obj={r_off.objective!r} bound={r_off.bound!r} max constraint violation={viol:.3e}")
assert viol <= 1e-6, f"reference point is NOT feasible ({viol:.3e}); cannot judge cuts against it"

# --- ON arm, with every node row recorded ------------------------------------
import discopt.solvers.milp_simplex as ms

recorded = []
_orig = ms.solve_milp_with_lazy_cuts


def wrapped(*a, **kw):
    for key, tag in (("node_callback", "node"), ("lazy_callback", "lazy")):
        cb = kw.get(key)
        if cb is None:
            continue

        def spy(x, _cb=cb, _tag=tag):
            rows = _cb(x)
            for coeffs, rhs in rows or []:
                recorded.append((_tag, np.asarray(coeffs, float).copy(), float(rhs)))
            return rows

        kw[key] = spy
    return _orig(*a, **kw)


ms.solve_milp_with_lazy_cuts = wrapped
import discopt.solvers.oa as oa
oa.solve_milp_with_lazy_cuts = wrapped  # in case of a module-level binding

os.environ["DISCOPT_OA_NODE_CUTS"] = "1"
m_on = portfolio2.build(**KW)
r_on = m_on.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                  time_limit=120, gap_tolerance=1e-4)
print(f"ON  arm: obj={r_on.objective!r} bound={r_on.bound!r}")
print(f"recorded rows: {len(recorded)}")

checks = 0
bad = 0
nvars = len(x_ref)
by_tag = {}
for tag, coeffs, rhs in recorded:
    if coeffs.shape[0] < nvars:
        continue
    lhs = float(coeffs[:nvars] @ x_ref)
    # Master rows may be longer than the model's variable vector (epigraph /
    # slack columns). Those extra columns are zero at a point that sets no
    # slack, so this is the honest test only when the tail is all zero.
    if coeffs.shape[0] > nvars and np.any(np.abs(coeffs[nvars:]) > 1e-12):
        continue
    checks += 1
    by_tag[tag] = by_tag.get(tag, 0) + 1
    if lhs > rhs + 1e-6:
        bad += 1
        if bad <= 8:
            print(f"  INVALID {tag} ROW: lhs={lhs:.10g} > rhs={rhs:.10g} (by {lhs-rhs:.3e})")

print(f"\nEXECUTED CUT CHECKS: {checks} {by_tag}   INVALID: {bad}")
if checks == 0:
    print("PROBE MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if bad else 0)
