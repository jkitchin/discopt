"""#1151 review finding 1+2: the defective floored row scale in two more places.

The examiner is the user-facing "is my solution feasible" tool and `_dual_recovery`
decides the active set for the KKT check. Both carry the pre-fix formula verbatim,
so both still vouch for the exact point the fixed incumbent gate rejects.

Prints an executed-assertion count and exits non-zero if it is zero (§6).
"""

import sys

import numpy as np

import discopt.modeling as dm
from discopt import Model
from discopt.validation import feasibility as F

CHECKS = 0


def quotient_aux_model():
    """The row `factorable_reform` emits for `minimize x/y`: the cleared
    bilinear `w*y - x == 0`, multiplied by `1/dmin = 1000`."""
    m = Model("quotient_aux")
    m.continuous("x", lb=1e-3, ub=1e3)
    m.continuous("y", lb=1e-3, ub=1e3)
    m.continuous("w", lb=1e-6, ub=1e6)
    x, y, w = m._variables
    m.subject_to(1000.0 * (w * y) - 1000.0 * x == 0.0)
    m.minimize(w)
    return m


# The point the solver actually returned on `min x/y + y/x` before the fix.
PT = np.array([0.0014052502011193727, 0.0014073586395206353, 0.9978427215251631])


def main():
    global CHECKS
    m = quotient_aux_model()

    res = F.verify_point(m, PT)
    CHECKS += 1
    print(f"verify_point           -> ok={res.ok}  {res.reason}")

    from types import SimpleNamespace

    from discopt.validation.examiner import examine

    pt = SimpleNamespace(
        x={v.name: float(PT[i]) for i, v in enumerate(m._variables)},
        objective=None,
        bound=None,
        status="optimal",
    )
    rep = examine(pt, m, recover_duals=False)
    for c in rep.checks:
        if "primal_con_feas" in c.name:
            CHECKS += 1
            print(f"examiner {c.name:<32} -> passed={c.passed}")

    # `_dual_recovery`'s active-set test, evaluated directly on this row.
    from discopt._relax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(m)
    J = np.asarray(ev.evaluate_jacobian(PT), dtype=float)
    signed = float(np.asarray(ev.evaluate_constraints(PT))[0])
    floored = float((np.abs(J[0]) * np.maximum(1.0, np.abs(PT))).max())
    term = float((np.abs(J[0]) * np.abs(PT)).max())
    active_tol = 1e-6
    CHECKS += 1
    print(
        f"_dual_recovery near-test  -> |signed|={abs(signed):.3e}  "
        f"floored scale={floored:.3e} (admits <= {active_tol * max(1.0, floored):.3e}) "
        f"-> near={abs(signed) <= active_tol * max(1.0, floored)}; "
        f"term scale={term:.3e} (admits <= {active_tol * max(1.0, term):.3e}) "
        f"-> near={abs(signed) <= active_tol * max(1.0, term)}"
    )

    CHECKS += nonfinite_jacobian_check()

    print(f"executed checks: {CHECKS}")
    if CHECKS == 0:
        sys.exit("PROBE MEASURED NOTHING")



def nonfinite_jacobian_check():
    """Second-review finding: `inf * 0` at a variable pinned to zero.

    `d/dx log(x)` at `x = 0` is unbounded, so the term product is NaN. Before the
    guard moved into the helper this raised a numpy RuntimeWarning and produced
    `scale=nan`, which compares False against every tolerance — a spurious
    `[FAIL] primal_con_feas (scaled)` on a possibly-feasible point.
    """
    import warnings as _w
    from types import SimpleNamespace

    from discopt.validation.examiner import examine
    from discopt.validation.feasibility import jacobian_row_scales

    n = 0
    J = np.array([[np.inf, 1.0]])
    x = np.array([0.0, 2.0])
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)  # a warning here FAILS the probe
        scales = jacobian_row_scales(J, x)
    n += 1
    print(f"jacobian_row_scales(inf*0) -> {scales}  (floored form gave inf)")
    assert np.all(np.isfinite(scales)), "helper still leaks a non-finite scale"

    m = Model("logmodel")
    m.continuous("a", lb=0.0, ub=10.0)
    m.continuous("b", lb=0.0, ub=10.0)
    a, b = m._variables
    m.subject_to(dm.log(a) + b <= 100.0)
    m.minimize(b)
    pt = SimpleNamespace(x={"a": 0.0, "b": 2.0}, objective=None, bound=None, status="optimal")
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        rep = examine(pt, m, recover_duals=False)
        rt = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    for c in rep.checks:
        if "primal_con_feas" in c.name:
            n += 1
            print(f"examiner {c.name:<32} -> passed={c.passed}")
    print(f"numpy RuntimeWarnings from the examiner: {len(rt)}")
    return n




main()
