"""The reported objective must be THIS model's objective at the returned point.

A solve may run on a REFORMULATED model -- ``factorable_reformulate`` lifts
subexpressions into ``_fr_aux_*`` columns, ``reformulate_integer_*``
binary-expands integer factors into ``_ipx_*`` columns -- and reports that
model's objective at the lifted point. The lift is exact in exact arithmetic,
but its big-M rows carry the binary-expansion weights, so a violation INSIDE the
absolute feasibility tolerance buys objective in proportion to those weights.

Measured on ``nvs07`` through the NLP-BB route (minlplib optimum 4.0), lifted
model ``factorable -> integer-bilinear``, 84 columns / 261 rows::

    reported objective                        3.9992714241881955  cert=True
    lifted objective at the reported point    3.9992714241881955  (agrees)
    worst lifted row violation                9.82e-09  (gate 1e-06)
    THIS model's objective at that point      4.000005454149035

64 lifted rows sit violated by ~1e-9..1e-8, each a big-M row whose objective
sensitivity reaches 3.3e+04. The slack they can buy totals 7.345e-04 against an
observed super-optimality of 7.286e-04 -- so the solve certified a value 7.3e-04
below an attainable optimum while every check it ran agreed with it, because all
of them ran in lifted space.

These tests drive the reconciliation directly. The corpus trigger needs the
lift, the route and a particular search path; the invariant does not.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult


def _model(sense: str = "min") -> dm.Model:
    m = dm.Model("recon")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.integer("y", lb=0, ub=4)
    expr = x + 2.0 * y
    (m.minimize if sense == "min" else m.maximize)(expr)
    m.subject_to(x + y >= 3)
    return m


def _result(m: dm.Model, point: dict, objective: float, **kw) -> SolveResult:
    base = dict(
        status="optimal",
        objective=objective,
        bound=objective,
        gap=0.0,
        x={k: np.atleast_1d(np.asarray(v, float)) for k, v in point.items()},
        wall_time=0.0,
        node_count=1,
        gap_certified=True,
    )
    base.update(kw)
    return SolveResult(**base)


def test_an_objective_that_does_not_match_the_point_is_replaced():
    """The defect's shape: a number better than the point actually achieves."""
    m = _model()
    point = {"x": 1.0, "y": 2.0}  # feasible; true objective 1 + 4 = 5
    r = _result(m, point, objective=4.9992714)  # 7.3e-4 too good, as nvs07 was
    m._reconcile_objective_with_model(r)
    assert r.objective == pytest.approx(5.0, abs=1e-9), (
        f"objective {r.objective!r} is still not this model's objective at the point"
    )


def test_a_correct_objective_is_left_alone():
    """No spurious movement: a result that already agrees must not be touched."""
    m = _model()
    r = _result(m, {"x": 1.0, "y": 2.0}, objective=5.0)
    before = (r.objective, r.status, r.gap, r.gap_certified, r.bound)
    m._reconcile_objective_with_model(r)
    assert (r.objective, r.status, r.gap, r.gap_certified, r.bound) == before


def test_the_certificate_is_re_tested_against_the_replaced_number():
    """A gap that existed only in the solver's own space is withdrawn.

    The bound stays where it was (a lower bound below the optimum is still a
    valid lower bound); what cannot stand is ``optimal`` at a gap the corrected
    pair does not close.
    """
    m = _model()
    # bound 4.9992714 is a valid lower bound for a true optimum of 5.0, but the
    # incumbent's real value is 5.0, so the real gap is 1.46e-4 > 1e-4.
    r = _result(m, {"x": 1.0, "y": 2.0}, objective=4.9992714)
    m._reconcile_objective_with_model(r)
    assert r.objective == pytest.approx(5.0, abs=1e-9)
    assert r.bound is not None and r.bound <= r.objective + 1e-9, (
        "bound must not sit above the corrected incumbent"
    )
    assert not r.gap_certified, (
        "a certificate granted against the replaced number must not survive it"
    )
    assert r.status != "optimal", f"status {r.status!r} still claims optimality"


def test_maximize_uses_the_declared_sense():
    """The evaluator is minimize-internal; the reported value is declared-sense."""
    m = _model("max")
    r = _result(m, {"x": 1.0, "y": 2.0}, objective=5.5)  # true value is 5.0
    m._reconcile_objective_with_model(r)
    assert r.objective == pytest.approx(5.0, abs=1e-9), (
        f"maximize reconciliation produced {r.objective!r}, expected 5.0"
    )


def test_a_point_missing_a_variable_is_skipped_LOUDLY(caplog):
    """When the check cannot be made it must say so, not look like a pass."""
    m = _model()
    r = _result(m, {"x": 1.0}, objective=4.9992714)  # no "y"
    with caplog.at_level("WARNING"):
        m._reconcile_objective_with_model(r)
    assert r.objective == pytest.approx(4.9992714), "result should be left untouched"
    assert any("objective reconciliation skipped" in rec.message for rec in caplog.records), (
        "a skipped check must be visible at WARNING"
    )


def test_an_ordinary_solve_still_reports_its_own_objective():
    """End to end: the reconciliation must not disturb a normal solve."""
    m = _model()
    r = m.solve(time_limit=30)
    assert r.objective is not None and r.x is not None
    flat = []
    for v in m._variables:
        flat.extend(np.atleast_1d(np.asarray(r.x[v.name], float)).ravel().tolist())
    from discopt._tape_nlp_evaluator import make_evaluator

    ev = make_evaluator(m)
    fi = float(ev.evaluate_objective(np.asarray(flat, float)))
    f = -fi if getattr(ev, "_negate", False) else fi
    assert r.objective == pytest.approx(f, abs=1e-9, rel=1e-12)
