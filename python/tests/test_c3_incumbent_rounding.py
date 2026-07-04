"""C-3 (P2, solver.py incumbent) — a near-integral discrete coordinate must not
survive unrounded into the reported incumbent when the terminal polish throws.

Mechanism (from ``docs/dev/correctness-issues.md`` §C-3): the batched JAX IPM
leaves an integer column a few digits shy of integral (e.g. ``2.999997``). Such a
point still passes the ``1e-5`` integrality gate at ``tree.inject_incumbent`` and
is stored verbatim as the tree incumbent. The terminal KKT polish / dual-recovery
re-solve in the B&B finalizers normally rounds+re-solves it, but that step is
best-effort and guarded — if it *raises* (or is skipped / not adopted) the raw
fractional coordinate is reported as-is, yielding a certified "optimal" whose
integer variable is fractional.

Two layers, both fast:

1. Direct unit test of ``_round_incumbent_integers`` — the extracted round-and-
   verify helper wired into all four finalizers. Encodes the *class*: rounds
   in-tolerance integers, leaves genuinely-fractional coords alone, and reports
   ``feasible=False`` (so the caller must not adopt) when rounding breaks a
   nonlinear constraint.

2. End-to-end: force the terminal polish to raise and inject a near-integral
   incumbent, then assert the reported solution's integer variable is *exactly*
   integral. Fails-before / passes-after.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


class _FakeEvaluator:
    """Minimal evaluator: constraint  x[1] <= 2.5  (violated once x[1] rounds to 3)."""

    n_constraints = 1

    def evaluate_constraints(self, x):
        return np.array([x[1]], dtype=np.float64)


@pytest.mark.smoke
def test_c3_round_incumbent_snaps_near_integral():
    # y = 2.999997 is within 1e-5 of 3; x = 1.2 is continuous (offset excluded).
    sol = np.array([1.2, 2.999997])
    rounded, feasible = S._round_incumbent_integers(sol, int_offsets=[1], int_sizes=[1])
    assert feasible is True
    assert rounded[1] == 3.0  # exactly integral
    assert rounded[0] == 1.2  # continuous untouched
    assert sol[1] == 2.999997  # input not mutated


@pytest.mark.smoke
def test_c3_round_incumbent_leaves_genuinely_fractional_untouched():
    # 2.4 is NOT within 1e-5 of any integer: snapping it would fabricate a point
    # the search never proved feasible, so it must be left alone.
    sol = np.array([0.0, 2.4])
    rounded, feasible = S._round_incumbent_integers(sol, int_offsets=[1], int_sizes=[1])
    assert feasible is True
    assert rounded[1] == 2.4


@pytest.mark.smoke
def test_c3_round_incumbent_rejects_when_rounding_breaks_feasibility():
    # x[1] = 2.999997 -> rounds to 3.0, violating the fake constraint x[1] <= 2.5.
    # The helper must report feasible=False so the caller keeps the unrounded
    # point rather than certifying an infeasible "integral" solution.
    sol = np.array([0.0, 2.999997])
    rounded, feasible = S._round_incumbent_integers(
        sol,
        int_offsets=[1],
        int_sizes=[1],
        evaluator=_FakeEvaluator(),
        cl_list=[-1e20],
        cu_list=[2.5],
    )
    assert feasible is False
    assert rounded[1] == 3.0  # still rounded, but flagged infeasible


@pytest.mark.smoke
def test_c3_fractional_integer_does_not_survive_polish_failure():
    """End-to-end: terminal polish raises + near-integral incumbent injected ->
    the reported integer variable must be exactly integral (fails before fix)."""
    from discopt._rust import PyTreeManager

    orig_polish = S._solve_node_nlp_kkt
    orig_inc = PyTreeManager.incumbent

    def _boom(*a, **k):
        raise RuntimeError("forced terminal polish failure (C-3 repro)")

    def _frac_inc(self):
        inc = orig_inc(self)
        if inc is None:
            return inc
        sol, obj = inc
        sol = list(sol)
        sol[1] = 2.999997  # y: near-integral but not exact
        return (sol, obj)

    S._solve_node_nlp_kkt = _boom
    PyTreeManager.incumbent = _frac_inc
    try:
        m = dm.Model("c3_e2e")
        x = m.continuous("x", lb=0.0, ub=5.0)
        y = m.integer("y", lb=0, ub=5)
        m.subject_to(x * x + y >= 3.0)
        m.minimize((x - 1.2) ** 2 + (y - 2.7) ** 2)
        res = m.solve(time_limit=20)
    finally:
        S._solve_node_nlp_kkt = orig_polish
        PyTreeManager.incumbent = orig_inc

    assert res.x is not None
    yv = float(np.asarray(res.x["y"]))
    assert abs(yv - round(yv)) < 1e-9, (
        f"C-3: fractional integer {yv!r} survived into the reported incumbent"
    )
