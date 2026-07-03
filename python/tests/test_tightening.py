"""Tests for the public FBBT bound-tightening utility (discopt.tightening).

The correctness-critical property: the tightened box is always a sound *outer*
box — it contains every feasible point (hence the optimum) and only ever shrinks
the input box. These tests pin cross-constraint (row-activity) propagation to its
fixpoint, infeasibility detection, soundness on a known feasible point, and the
never-loosen invariant (including for vector variables).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt.tightening import fbbt_box


def test_cross_row_propagation_reaches_fixpoint():
    """a >= b+5, b >= c+3 on [0,10] -> lower bounds propagate to [8, 3, 0]."""
    m = dm.Model("chain")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    c = m.continuous("c", lb=0, ub=10)
    m.minimize(a + b + c)
    m.subject_to(a - b >= 5)
    m.subject_to(b - c >= 3)

    res = fbbt_box(m)
    assert not res.infeasible
    np.testing.assert_allclose(res.lb, [8.0, 3.0, 0.0], atol=1e-6)
    # Upper bounds propagate the other way: b <= 5, c <= 2.
    np.testing.assert_allclose(res.ub, [10.0, 5.0, 2.0], atol=1e-6)
    assert res.n_tightened == 3  # a (lb), b (lb+ub), c (ub)


def test_detects_infeasibility():
    m = dm.Model("inf")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.subject_to(x >= 5)
    m.subject_to(x <= 3)
    res = fbbt_box(m)
    assert res.infeasible


def test_tightened_box_contains_optimum():
    """Soundness: the tightened box must still contain the true optimum."""
    m = dm.Model("sound")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.maximize(x + y)
    m.subject_to(x + y <= 4)
    m.subject_to(x - y >= 1)
    res = fbbt_box(m)
    # True optimum is x=2.5, y=1.5 (x+y=4, x-y=1). It must remain in the box.
    assert res.lb[0] - 1e-9 <= 2.5 <= res.ub[0] + 1e-9
    assert res.lb[1] - 1e-9 <= 1.5 <= res.ub[1] + 1e-9
    # And the box only shrank.
    assert res.lb[0] >= 0 and res.ub[0] <= 10
    assert res.lb[1] >= 0 and res.ub[1] <= 10


def test_never_loosens_and_handles_vector_vars():
    m = dm.Model("vec")
    xv = m.continuous("xv", shape=(3,), lb=0, ub=10)
    s = m.continuous("s", lb=0, ub=100)
    m.minimize(s)
    m.subject_to(xv[0] + xv[1] + xv[2] == s)
    m.subject_to(s <= 6)
    res = fbbt_box(m)
    assert not res.infeasible
    # Never loosened beyond the original box (length 4: xv[0..2], s).
    assert np.all(res.lb >= np.array([0, 0, 0, 0]) - 1e-9)
    assert np.all(res.ub <= np.array([10, 10, 10, 100]) + 1e-9)
    # s is tightened to <= 6 by the demand constraint.
    assert res.ub[3] <= 6.0 + 1e-9


def test_idempotent():
    m = dm.Model("idem")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.minimize(a + b)
    m.subject_to(a - b >= 5)
    first = fbbt_box(m)
    # Re-tightening from the already-tight bounds yields the same box.
    second = fbbt_box(m, max_iter=first.lb.size + 5)
    np.testing.assert_allclose(first.lb, second.lb, atol=1e-9)
    np.testing.assert_allclose(first.ub, second.ub, atol=1e-9)


def test_no_constraints_returns_original_box():
    m = dm.Model("free")
    x = m.continuous("x", lb=-2, ub=7)
    m.minimize(x)
    res = fbbt_box(m)
    assert res.n_tightened == 0
    np.testing.assert_allclose(res.lb, [-2.0], atol=1e-12)
    np.testing.assert_allclose(res.ub, [7.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Reverse-division FBBT: bound a variable that appears only in v*affine == c
# (BilinearProductEqualityRule). This is the only mechanism that can bound a
# variable living solely in a nonlinear constraint, e.g. the mole-fraction
# ratios x = a/(affine) in ex6_1_4.
# ---------------------------------------------------------------------------


def _run_bilinear_rule(model):
    from discopt._jax.nonlinear_bound_tightening import (
        BilinearProductEqualityRule,
        build_flat_variable_metadata,
    )

    lb = np.array([float(v.lb) for v in model._variables], dtype=np.float64)
    ub = np.array([float(v.ub) for v in model._variables], dtype=np.float64)
    meta = build_flat_variable_metadata(model)
    return BilinearProductEqualityRule().tighten(model, lb.copy(), ub.copy(), meta)


def test_bilinear_product_equality_bounds_free_variable():
    """``t * a == 5`` with ``a in [1, 2]`` pins the unbounded ``t`` to ``[2.5, 5]``."""
    m = dm.Model("ratio")
    a = m.continuous("a", lb=1.0, ub=2.0)
    t = m.continuous("t", lb=0.0, ub=np.inf)
    m.minimize(t)
    m.subject_to(t * a == 5.0)

    tl, tu = _run_bilinear_rule(m)
    # t = 5 / a, a in [1, 2]  ->  t in [2.5, 5].
    assert tu[1] == pytest.approx(5.0, rel=1e-9)
    assert tl[1] == pytest.approx(2.5, rel=1e-9)
    # The bounding variable is untouched.
    assert tl[0] == pytest.approx(1.0)
    assert tu[0] == pytest.approx(2.0)


def test_bilinear_product_equality_multivariate_affine_factor():
    """``t * (x + 0.5*y) == x`` bounds ``t`` from the affine factor's interval."""
    m = dm.Model("ratio_multi")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=2.0, ub=4.0)
    t = m.continuous("t", lb=0.0, ub=np.inf)
    m.minimize(t)
    m.subject_to(t * (x + 0.5 * y) == x)

    tl, tu = _run_bilinear_rule(m)
    # F = x + 0.5y in [1+1, 2+2] = [2, 4]; numerator x in [1, 2].
    # t = x/F in [1/4, 2/2] = [0.25, 1.0].
    assert tu[2] == pytest.approx(1.0, rel=1e-9)
    assert tl[2] == pytest.approx(0.25, rel=1e-9)


def test_bilinear_product_equality_soundness_keeps_feasible_point():
    """The tightened box must still contain a genuine feasible solution."""
    m = dm.Model("ratio_sound")
    a = m.continuous("a", lb=1.0, ub=4.0)
    t = m.continuous("t", lb=0.0, ub=np.inf)
    m.minimize(t)
    m.subject_to(t * a == 6.0)

    tl, tu = _run_bilinear_rule(m)
    # Feasible point a=3, t=2 must survive the tightened box.
    assert tl[0] - 1e-9 <= 3.0 <= tu[0] + 1e-9
    assert tl[1] - 1e-9 <= 2.0 <= tu[1] + 1e-9
    # And the never-loosen invariant holds for the bounded var.
    assert tu[1] <= np.inf


def test_bilinear_product_equality_skips_sign_unstable_factor():
    """An affine factor whose interval straddles 0 cannot be divided: no change."""
    m = dm.Model("straddle")
    a = m.continuous("a", lb=-1.0, ub=2.0)  # interval contains 0
    t = m.continuous("t", lb=0.0, ub=np.inf)
    m.minimize(t)
    m.subject_to(t * a == 5.0)

    tl, tu = _run_bilinear_rule(m)
    assert tu[1] == np.inf  # left unbounded — division by a zero-straddling F is unsafe


def test_bilinear_product_equality_skips_when_var_in_remainder():
    """If the target variable also appears affinely, isolation is unsound: skip."""
    m = dm.Model("circular")
    a = m.continuous("a", lb=1.0, ub=2.0)
    t = m.continuous("t", lb=0.0, ub=np.inf)
    m.minimize(t)
    m.subject_to(t * a + t == 5.0)  # t appears in the product AND the remainder

    tl, tu = _run_bilinear_rule(m)
    assert tu[1] == np.inf  # not this rule's pattern; left for other machinery


# ─────────────────────────────────────────────────────────────
# C-31 (P0) — FBBT array-block collapse reaching the certified LP dual bound.
#
# The Rust FBBT engine carries one interval per variable BLOCK. Pre-fix it
# seeded that interval from element 0's bounds and stamped it onto every scalar
# slot, so on a HETEROGENEOUS per-element array block the box excluded the
# feasible region of the other elements. That collapsed box reaches the certified
# LP dual bound via `_fbbt_argument_box` (`milp_relaxation.py`), where an envelope
# built over a box that excludes feasible arguments is unsound. The fix seeds
# each block from the element-wise UNION [min lb, max ub] — a valid outer bound
# for every element — so no feasible point is ever cut. These tests fail on the
# element-0 seed and pass on the union seed.
# ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_c31_heterogeneous_array_block_not_overtightened():
    """A heterogeneous array block must keep every element's own feasible range."""
    m = dm.Model("c31_over")
    # x[0] ∈ [8,10] (tight), x[1] ∈ [0,10] (free). Element-0 collapse would stamp
    # [8,10] onto x[1], erasing the feasible region x[1] ∈ [0,8).
    x = m.continuous("x", shape=(2,), lb=np.array([8.0, 0.0]), ub=np.array([10.0, 10.0]))
    m.minimize(x[0])
    m.subject_to(x[1] >= 0.0)
    res = fbbt_box(m)
    assert not res.infeasible
    # The block interval must be a valid outer bound for EVERY element: x[1]'s
    # feasible point 0.0 must remain inside the box (pre-fix lb[1] was 8.0).
    feasible_pt = np.array([9.0, 0.0])
    assert np.all(res.lb - 1e-9 <= feasible_pt), f"cut feasible point, lb={res.lb}"
    assert np.all(feasible_pt <= res.ub + 1e-9), f"cut feasible point, ub={res.ub}"
    assert res.lb[1] <= 0.0 + 1e-9, f"C-31: x[1] lb collapsed to element-0, got {res.lb[1]}"


@pytest.mark.smoke
def test_c31_heterogeneous_array_block_no_false_infeasible():
    """A feasible heterogeneous-block model must not be reported infeasible."""
    m = dm.Model("c31_infeas")
    # x[0] fixed at 5; x[1] ∈ [0,3]. x=[5, 0..3] is feasible. Element-0 collapse
    # stamps [5,5] onto x[1]; intersecting with x[1] <= 3 is empty → false infeasible.
    x = m.continuous("x", shape=(2,), lb=np.array([5.0, 0.0]), ub=np.array([5.0, 3.0]))
    m.minimize(x[0])
    m.subject_to(x[1] <= 3.0)
    res = fbbt_box(m)
    assert not res.infeasible, f"C-31: feasible model declared infeasible, lb={res.lb} ub={res.ub}"
    # The feasible point x=[5, 1] must remain inside the box.
    feasible_pt = np.array([5.0, 1.0])
    assert np.all(res.lb - 1e-9 <= feasible_pt)
    assert np.all(feasible_pt <= res.ub + 1e-9)


@pytest.mark.smoke
def test_c31_fbbt_argument_box_envelope_contains_feasible_arg():
    """The univariate-rescue envelope (certified LP path) must not exclude a
    feasible argument of a heterogeneous array block.

    `_collect_univariate_relaxations` builds a McCormick envelope over the FBBT
    argument box for out-of-domain univariate ops (issue #219 rescue). Pre-fix,
    a `log` on a heterogeneous array element got element-0's collapsed box, which
    excluded the element's true feasible arguments → an invalid envelope in the
    certified LP relaxation. Post-fix the union seed keeps the element's box, so
    the op either (a) has an envelope whose arg box CONTAINS every feasible
    argument, or (b) is soundly abstained (dropped) — never an envelope over a
    box that excludes a feasible argument.
    """
    import discopt._jax.milp_relaxation as mr
    from discopt.modeling import log as dlog

    m = dm.Model("c31_env")
    # x[0] fixed at [3,3] (in log domain); x[1] raw box [-1,5] straddles 0 → log
    # out of domain raw → FBBT-rescue path fires. Element-0 collapse would give
    # the log arg box [3,3], excluding x[1]'s feasible arguments in (0,5].
    x = m.continuous("x", shape=(2,), lb=np.array([3.0, -1.0]), ub=np.array([3.0, 5.0]))
    m.minimize(dlog(x[1]) + x[0])

    flat_lb = np.array([3.0, -1.0])
    flat_ub = np.array([3.0, 5.0])
    relax, _var_map, _bounds = mr._collect_univariate_relaxations(
        m, 2, flat_lb, flat_ub, start_col=2
    )
    # A feasible argument for x[1] under the model is any value in (0, 5], e.g. 4.0.
    for r in relax:
        assert r.arg_lb - 1e-9 <= 4.0 <= r.arg_ub + 1e-9, (
            f"C-31: univariate-rescue envelope arg box [{r.arg_lb}, {r.arg_ub}] "
            "excludes the feasible argument 4.0 → invalid envelope in the "
            "certified LP relaxation"
        )
