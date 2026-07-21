"""Regression tests for #815: the pure-continuous single-NLP path must not report
an infeasible NLP iterate as an incumbent (a false primal).

``_solve_continuous`` (``solver.py``) solves a pure-continuous model with one
local NLP solve (no B&B tree). A local NLP that stalls at the time/iteration
limit — or, on a model with unbounded variables, terminates at a non-KKT point —
can hand back an INFEASIBLE point with a finite objective. Before the fix the
path reported that point as ``objective``/``x`` (emfl050_3_3: an all-10 iterate
violating the distance-defining equalities by ~9.5, reported as objective=594),
and only the final incumbent-verification guard (#772) caught it downstream.

The fix verifies feasibility AT THE SOURCE — with the same loose 1e-3 check the
#772 guard uses — and withholds an infeasible point there, so the bogus incumbent
never propagates.

The unit tests drive ``_solve_continuous`` directly with a monkeypatched NLP
backend (deterministic, CI-safe, no benchmark corpus needed); the slow test is
the real-instance guard when the MINLPLib corpus is present.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import discopt.modeling as dm
import discopt.solver as solver_mod
import discopt.solvers.nlp_ipopt as _nlp_ipopt
import discopt.solvers.nlp_pounce as _nlp_pounce
import numpy as np
import pytest
from discopt.solvers.nlp_ipopt import NLPResult, SolveStatus


def _patch_nlp(monkeypatch, x_value: float, status: SolveStatus, objective: float):
    """Force both NLP backends to return a fixed point/status/objective."""

    def _fake_solve(evaluator, x0, *args, **kwargs):
        n = int(np.asarray(x0).size)
        return NLPResult(
            status=status,
            x=np.full(n, x_value, dtype=np.float64),
            objective=objective,
            multipliers=None,
            bound_multipliers_lower=None,
            bound_multipliers_upper=None,
            iterations=1,
            wall_time=0.01,
        )

    monkeypatch.setattr(_nlp_pounce, "solve_nlp", _fake_solve)
    monkeypatch.setattr(_nlp_ipopt, "solve_nlp", _fake_solve)


def _nonlinear_continuous_model() -> dm.Model:
    """A pure-continuous, nonlinear, FEASIBLE model (optimum at x=1)."""
    m = dm.Model("fp815")
    m.continuous("x", lb=0.0, ub=2.0)
    m.minimize(m._variables[0])
    m.subject_to(m._variables[0] ** 2 == 1.0)
    return m


@pytest.mark.correctness
def test_815_solve_continuous_withholds_infeasible_incumbent(monkeypatch):
    """A stalled NLP returning an INFEASIBLE point (x=2 violates x**2==1 by 3)
    must NOT be reported as an incumbent — objective/x are withheld at the source.
    Before the fix this returned objective=2.0 with an infeasible x."""
    _patch_nlp(monkeypatch, x_value=2.0, status=SolveStatus.TIME_LIMIT, objective=2.0)
    m = _nonlinear_continuous_model()
    r = solver_mod._solve_continuous(m, 5.0, None, time.perf_counter(), nlp_solver="pounce")
    assert r.objective is None, f"#815: infeasible incumbent reported: obj={r.objective}"
    assert r.x is None, "#815: infeasible incumbent point reported"


@pytest.mark.correctness
def test_815_solve_continuous_keeps_feasible_incumbent(monkeypatch):
    """Positive control: a FEASIBLE returned point (x=1 satisfies x**2==1) must be
    kept — the feasibility gate must not over-withhold genuine incumbents."""
    _patch_nlp(monkeypatch, x_value=1.0, status=SolveStatus.OPTIMAL, objective=1.0)
    m = _nonlinear_continuous_model()
    r = solver_mod._solve_continuous(m, 5.0, None, time.perf_counter(), nlp_solver="pounce")
    assert r.objective is not None and abs(r.objective - 1.0) < 1e-6, (
        f"#815 over-withholding: a feasible incumbent was dropped (obj={r.objective})"
    )
    assert r.x is not None


# --- Real-instance guard (needs the MINLPLib benchmark corpus) ---------------

_EMFL = Path(
    os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/emfl050_3_3.nl")
)


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _EMFL.exists(), reason="emfl050_3_3.nl (benchmark corpus) absent")
def test_815_emfl_no_false_primal():
    """The real repro: emfl050_3_3's stalled NLP must not surface the ~594 false
    primal, and the #772 guard must never have to fire (the source withholds)."""
    model = dm.from_nl(str(_EMFL))
    result = model.solve(time_limit=5)
    assert not getattr(result, "incumbent_verification_failed", False), (
        "#815: the #772 guard had to catch a false primal the source should have withheld."
    )
    assert result.objective is None or abs(result.objective - 594.0) > 1.0, (
        f"#815 garbage incumbent 594 returned: {result.objective}"
    )
    if result.objective is not None and result.x is not None:
        from discopt._jax.nlp_evaluator import cached_evaluator
        from discopt._jax.primal_heuristics import _check_constraint_feasibility

        ev = cached_evaluator(model)
        flat = np.concatenate(
            [
                np.atleast_1d(np.asarray(result.x[v.name], dtype=np.float64)).ravel()
                for v in model._variables
            ]
        )
        assert _check_constraint_feasibility(ev, flat, tol=1e-3), (
            "#815: reported incumbent is infeasible in the original model"
        )
