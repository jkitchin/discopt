"""NLP-BB's exit refine must not strand a point on a large-coefficient row.

portfol_roundlot links each weight to an integer lot count through
``c_i x_i = n_i`` (c_i up to 1e5) and sums the weights to 1. At the lot counts
NLP-BB settles on, those rows disagree by 3.35e-7. The terminal refine solves
with POUNCE's gradient scaling, which moved that residual onto the linking rows:
``-78000 x2 + x11`` read 2.6e-6 at ``x11 = 0``. The #954 exit gate refused that
point, so ``solve(nlp_bb=True)`` raised instead of returning the incumbent it
had. The #1059 auto-route reaches the same path as its fallback when OA has not
certified in its share of the budget, which is how this surfaced in CI.

The fix re-solves the refine once without scaling when the point about to leave
fails the gate, and adopts that output only when it clears the gate.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S
from discopt.modeling.core import from_nl

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")
DECLARED_ABS_TOL = 1e-6
# minlplib.solu
PORTFOL_ROUNDLOT_OPT = 0.0282906349


def _worst_violation(model, x_dict):
    """Raw max violation over declared rows and bounds, with a comparison count.

    Evaluated through ``NLPEvaluator`` against the declared model, with no
    term-scale forgiveness, so it cannot inherit the solver's own arbiter.
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator

    x = np.concatenate(
        [np.asarray(x_dict[v.name], dtype=np.float64).ravel() for v in model._variables]
    )
    ev = NLPEvaluator(model)
    cl, cu = (np.asarray(b, dtype=np.float64) for b in S._infer_constraint_bounds(model, ev))
    g = np.asarray(ev.evaluate_constraints(x), dtype=np.float64)
    n = min(len(g), len(cl))
    assert n > 0, "no constraint rows evaluated: this probe would measure nothing"
    lo, hi = (np.asarray(b, dtype=np.float64) for b in ev.variable_bounds)
    viols = np.concatenate([g[:n] - cu[:n], cl[:n] - g[:n], x - hi, lo - x])
    return float(np.max(viols)), int(viols.size)


def _count_unscaled_refines(monkeypatch):
    calls = []
    orig = S._solve_node_nlp_kkt

    def spy(evaluator, x0, lb, ub, constraint_bounds, opts):
        if opts.get("nlp_scaling_method") == "none":
            calls.append(1)
        return orig(evaluator, x0, lb, ub, constraint_bounds, opts)

    monkeypatch.setattr(S, "_solve_node_nlp_kkt", spy)
    return calls


@pytest.mark.correctness
def test_large_coefficient_rows_leave_through_the_gate(monkeypatch):
    """Raised RuntimeError from the #954 exit gate before the fix."""
    calls = _count_unscaled_refines(monkeypatch)
    m = from_nl(os.path.join(DATA, "portfol_roundlot.nl"))
    r = m.solve(time_limit=60, nlp_bb=True)

    assert r.nlp_bb, "did not dispatch to NLP-BB; this test covers that path"
    assert calls, "the unscaled refine never ran, so this test measured nothing"
    assert r.status in ("optimal", "feasible"), r.status
    assert r.x is not None
    worst, n_checked = _worst_violation(m, r.x)
    assert n_checked == 2 * 11 + 2 * 17
    assert worst <= DECLARED_ABS_TOL, f"returned point is {worst:.3e} outside a declared row"
    assert abs(r.objective - PORTFOL_ROUNDLOT_OPT) <= 1e-6 + 1e-4 * PORTFOL_ROUNDLOT_OPT


@pytest.mark.correctness
def test_unscaled_refine_stays_off_when_the_point_clears_the_gate(monkeypatch):
    """Control: an incumbent that already clears the gate never triggers the retry."""
    calls = _count_unscaled_refines(monkeypatch)
    m = dm.Model("unscaled_refine_control")
    y = m.binary("y", shape=(2,))
    x = m.continuous("x", shape=(3,), lb=0.0, ub=4.0)
    m.minimize(sum(dm.exp(0.4 * x[j]) for j in range(3)) + 1.5 * y[0] + 2.5 * y[1])
    m.subject_to(sum(dm.log(1.0 + x[j]) for j in range(3)) >= 1.4)
    for j in range(3):
        m.subject_to(x[j] <= 4 * y[j % 2])
    r = m.solve(time_limit=60, nlp_bb=True)

    assert r.nlp_bb and r.status == "optimal", r.status
    worst, n_checked = _worst_violation(m, r.x)
    assert n_checked == 2 * 4 + 2 * 5
    assert worst <= DECLARED_ABS_TOL
    assert calls == [], "the unscaled refine ran on a point that already cleared the gate"
