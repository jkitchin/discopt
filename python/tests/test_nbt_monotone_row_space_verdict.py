"""Monotone-function bound tightening must decide infeasibility in ROW space.

Adversarial fuzz (seed 20047, badly-scaled pure-integer program vs an exact
enumeration oracle): the row

    -4.7e-05 * log(x2 + 10) + c == 0,   c = 4.7e-05 * log(11),   x2 binary

holds at ``x2 = 1`` to ~1e-16, but ``MonotoneFunctionBoundsRule`` divided it by
the function coefficient and compared ``log(11)`` with the required value against
an absolute 1e-12 in FUNCTION space. The division inflated the rounding residue
by ``1/4.7e-5`` past that yardstick and the whole model was certified
``infeasible`` in 0.3 s. The verdict now uses the row's own violation against the
feasibility tolerance plus round-off (the #1397 bar).
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
from discopt._relax.nonlinear_bound_tightening import (
    MonotoneFunctionBoundsRule,
    NonlinearBoundTighteningInfeasible,
    build_flat_variable_metadata,
)

C = 4.7e-05
# The fuzzer's construction: the constant is split as ``1.0 + shift`` with the shift
# chosen so the planted point z = 1 satisfies the row. ``1.0 + shift`` cancels and
# leaves a rounding residue that the division by C inflates.
SHIFT = -((-C * math.log(11.0)) + 1.0)


def _model():
    m = dm.Model("mono_row_space")
    z = m.binary("z")
    m.subject_to(-C * dm.log(z + 10) + 1.0 + SHIFT == 0)
    m.minimize(-z)
    return m


def test_fixture_rounds_past_the_old_function_space_yardstick():
    # Precondition (CLAUDE.md §6): the divided requirement really does exceed
    # log(11) by more than the old absolute 1e-12, so the tests below exercise the
    # defect -- while the ROW residual at z = 1 is ~1e-16.
    rhs = -(1.0 + SHIFT) / -C
    assert rhs - math.log(11.0) > 1e-12
    assert abs(-C * math.log(11.0) + 1.0 + SHIFT) < 1e-15


def test_rule_does_not_prove_a_tolerance_feasible_row_infeasible():
    m = _model()
    meta = build_flat_variable_metadata(m)
    lb, ub = np.array([0.0]), np.array([1.0])
    try:
        new_lb, new_ub = MonotoneFunctionBoundsRule().tighten(m, lb, ub, meta)
    except NonlinearBoundTighteningInfeasible as exc:  # pragma: no cover - the defect
        raise AssertionError(f"false infeasibility proof: {exc}") from exc
    # z = 1 satisfies the row; it must survive.
    assert new_ub[0] >= 1.0


def test_solve_finds_the_feasible_point():
    r = _model().solve(time_limit=20)
    assert r.status == "optimal"
    assert float(r.x["z"]) == 1.0


def test_a_genuinely_infeasible_row_is_still_refused():
    # Control: a real row-space miss (log(11) needs to reach log(12)) is far past the
    # feasibility tolerance and must still be proved infeasible.
    m = dm.Model("mono_really_infeasible")
    z = m.binary("z")
    m.subject_to(-1.0 * dm.log(z + 10) + math.log(12.0) <= 0)
    m.minimize(z)
    meta = build_flat_variable_metadata(m)
    raised = False
    try:
        MonotoneFunctionBoundsRule().tighten(m, np.array([0.0]), np.array([1.0]), meta)
    except NonlinearBoundTighteningInfeasible:
        raised = True
    assert raised
