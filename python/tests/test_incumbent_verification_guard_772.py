"""#772 — final-incumbent verification guard (defense-in-depth against false primals).

``Model.solve`` captures a COMPILED feasibility evaluator of the ORIGINAL model
before ``solve_model`` runs (presolve mutates the constraint DAG in place), then
verifies the returned incumbent against that immune snapshot. A reported incumbent
that is infeasible in the original problem is a false primal (the #770 class); the
guard withholds it and decertifies rather than return it.

These tests prove the three things that make the guard trustworthy:
  1. it is INERT on valid solves (never withholds a feasible incumbent),
  2. the snapshot is IMMUNE to a real presolve run (still reflects the original),
  3. it FIRES on an injected false primal (withhold + decertify + flag).
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import cached_evaluator
from discopt._jax.primal_heuristics import _check_constraint_feasibility
from discopt.modeling.core import SolveResult

_DATA = "python/tests/data/minlplib_nl"


def test_guard_inert_on_valid_solve():
    """A genuinely feasible incumbent is never withheld or decertified."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y >= 1.0)
    m.minimize(x * x + y * y)
    r = m.solve(time_limit=10)
    assert r.incumbent_verification_failed is False
    assert r.x is not None and r.objective is not None


@pytest.mark.parametrize("instance", ["syn05hfsg", "alan", "nvs03"])
def test_snapshot_immune_to_real_presolve(instance):
    """The entry-time compiled evaluator remains a faithful representation of the
    ORIGINAL problem across a real solve (which runs presolve mutation): it agrees
    with a freshly parsed model on the returned incumbent's feasibility."""
    m = dm.from_nl(f"{_DATA}/{instance}.nl")
    held_ev = cached_evaluator(m)  # snapshot BEFORE solve mutates the model
    names = [v.name for v in m._variables]
    r = m.solve(time_limit=15)  # runs presolve (in-place mutation)
    if r.x is None:
        pytest.skip(f"{instance}: no incumbent to compare")
    held_flat = np.array([np.asarray(r.x[n], dtype=np.float64).ravel()[0] for n in names])

    fresh = dm.from_nl(f"{_DATA}/{instance}.nl")  # pristine original
    fresh_ev = cached_evaluator(fresh)
    fresh_flat = np.concatenate(
        [np.atleast_1d(np.asarray(r.x[v.name], dtype=np.float64)).ravel() for v in fresh._variables]
    )
    assert _check_constraint_feasibility(held_ev, held_flat, tol=1e-3) == (
        _check_constraint_feasibility(fresh_ev, fresh_flat, tol=1e-3)
    ), "held snapshot disagrees with a fresh original model — not immune to presolve"


def test_guard_withholds_injected_false_primal(monkeypatch):
    """If solve_model ever returned an incumbent infeasible in the original problem
    (a false primal — the #770 failure mode), the guard must withhold it, null the
    objective, decertify, and flag. Injected via monkeypatch since correct solver
    code never produces one."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y <= 1.0)  # (3, 3) violates this by 5 — grossly infeasible
    m.minimize(x + y)

    def _fake_solve_model(model, **kwargs):
        return SolveResult(
            status="optimal",
            objective=6.0,
            bound=6.0,
            gap=0.0,
            x={"x": np.array(3.0), "y": np.array(3.0)},
            gap_certified=True,
        )

    monkeypatch.setattr(solver_mod, "solve_model", _fake_solve_model)
    r = m.solve(time_limit=5)

    assert r.incumbent_verification_failed is True
    assert r.x is None
    assert r.objective is None
    assert r.gap_certified is False


def test_verify_incumbent_false_disables_guard(monkeypatch):
    """``verify_incumbent=False`` restores the pre-guard behavior (the same injected
    false primal passes through unflagged) — the opt-out works."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y <= 1.0)
    m.minimize(x + y)

    def _fake_solve_model(model, **kwargs):
        return SolveResult(
            status="optimal",
            objective=6.0,
            bound=6.0,
            gap=0.0,
            x={"x": np.array(3.0), "y": np.array(3.0)},
            gap_certified=True,
        )

    monkeypatch.setattr(solver_mod, "solve_model", _fake_solve_model)
    r = m.solve(time_limit=5, verify_incumbent=False)
    assert r.incumbent_verification_failed is False
    assert r.x is not None  # not withheld — guard was off
