"""#2362 sweep: no Python bound pass derives a bound by dividing by a noise coefficient.

Every site here bounds a variable as ``(rhs - rest) / c``. ``rest`` is a float sum,
so it carries round-off, and dividing by a coefficient at or below HiGHS's
``small_matrix_value`` (1e-9, ``HighsOptions.h:787``) turns that round-off into the
bound itself, past every relative pad these passes apply.

The adversary is the same everywhere: ``0.1·x0 + 0.2·x1 + c·y (==|<=) 0.3`` with
``x0 = x1 = 1``. In floats the rest is ``0.30000000000000004``, so the derived
bound on ``y`` is ``-5.55e-17 / c``. The *exact* model (the stored doubles, summed
in rationals) puts ``y`` at ``(0.3 - 0.1 - 0.2) / c = -2.78e-17 / c``, so for
``c = 1e-12`` the float bound ``y <= -5.55e-5`` cuts the exactly feasible
``y = -2.78e-5``. Each test checks that exact point stays in the box, and that a
real coefficient (the control) is still propagated.

A second adversary covers the activity: a dropped ``1e-13·y`` term with ``y`` in
``[-1e13, 0]`` is worth 1, so ``x + 1e-13·y <= 1`` must not read as ``x <= 1``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from fractions import Fraction

import discopt.modeling as dm
import numpy as np
from discopt._relax.nonlinear_bound_tightening import (
    DefinedVariableForwardRule,
    _cached_flat_metadata,
    _cached_flat_terms,
)
from discopt._relax.obbt import propagate_equality_defined_bounds
from discopt._relax.uniform_relax import _fix_single_var_equalities
from discopt.solver import _extract_variable_info
from discopt.solvers._root_cuts import _RootLP
from discopt.solvers._root_presolve import _LinRow, _propagate_linear_rows

INF = float("inf")
NOISE = (1e-12, -1e-12)
CONTROL = 1e-3


def _exact_y(c: float) -> float:
    """The exact solution of ``0.1 + 0.2 + c·y = 0.3`` over the stored doubles."""
    return float((Fraction(0.3) - Fraction(0.1) - Fraction(0.2)) / Fraction(c))


def _contains(lo: float, hi: float, v: float) -> bool:
    return lo <= v + 1e-12 * max(1.0, abs(v)) and hi >= v - 1e-12 * max(1.0, abs(v))


def test_exact_point_is_strictly_inside_the_float_round_off():
    """The adversary is real: the float bound and the exact point differ by 2x."""
    y_exact = _exact_y(1e-12)
    y_float = (0.3 - (0.1 * 1.0 + 0.2 * 1.0)) / 1e-12
    assert y_float < 1.5 * y_exact < 0.0, (y_float, y_exact)


def test_root_presolve_propagation_never_divides_by_a_noise_coefficient():
    checked = 0
    for c in (*NOISE, CONTROL):
        for sense in ("<=", "=="):
            # Σ terms + const ⋈ 0  ->  0.1 x0 + 0.2 x1 + c y - 0.3 ⋈ 0
            terms = {0: 0.1, 1: 0.2, 2: c}
            if c < 0:  # keep y's side the bounded one: flip the row
                terms = {j: -a for j, a in terms.items()}
                row = _LinRow(0, 0, 1, terms, 0.3, ">=" if sense == "<=" else "==", "r")
            else:
                row = _LinRow(0, 0, 1, terms, -0.3, sense, "r")
            lb = np.array([1.0, 1.0, -1.0])
            ub = np.array([1.0, 1.0, 1.0])
            infeasible = _propagate_linear_rows([row], lb, ub, np.zeros(3, bool))
            assert not infeasible, (c, sense)
            y = _exact_y(c)
            assert _contains(lb[2], ub[2], y), f"c={c} {sense}: y in [{lb[2]}, {ub[2]}] cuts {y}"
            if c == CONTROL:
                assert ub[2] < 1.0, "the control is still propagated"
            checked += 1
    # A tiny term still counts in the activity: x + 1e-13 y <= 1, y in [-1e13, 0]
    # admits (2, -1e13); dropping the term would derive x <= 1.
    row = _LinRow(0, 0, 1, {0: 1.0, 1: 1e-13}, -1.0, "<=", "r")
    lb = np.array([0.0, -1e13])
    ub = np.array([10.0, 0.0])
    assert not _propagate_linear_rows([row], lb, ub, np.zeros(2, bool))
    assert ub[0] >= 2.0 - 1e-9, f"x <= {ub[0]} cuts (2, -1e13)"
    checked += 1
    assert checked == 7


def _root_lp(a_le: np.ndarray, b_le: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> _RootLP:
    lp = _RootLP.__new__(_RootLP)
    lp.n = len(lb)
    lp.lb = lb.copy()
    lp.ub = ub.copy()
    lp.is_int = np.zeros(len(lb), bool)
    lp.A_le = a_le
    lp.b_le = b_le
    lp.A_eq = np.zeros((0, len(lb)))
    lp.b_eq = np.zeros(0)
    lp._fbbt_separation_bounds()
    return lp


def test_root_cut_separation_bounds_never_divide_by_a_noise_coefficient():
    checked = 0
    for c in (*NOISE, CONTROL):
        s = 1.0 if c > 0 else -1.0  # orient so y's upper side is the one bounded
        lp = _root_lp(
            np.array([[0.1, 0.2, c]]) * s,
            np.array([0.3]) * s,
            np.array([1.0, 1.0, -1.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        y = _exact_y(c)
        lo, hi = lp.lb_sep[2], lp.ub_sep[2]
        assert _contains(lo, hi, y), f"c={c}: y in [{lo}, {hi}] cuts {y}"
        if c == CONTROL:
            assert hi < 1.0, "the control is still propagated"
        checked += 1
    lp = _root_lp(
        np.array([[1.0, 1e-13]]),
        np.array([1.0]),
        np.array([0.0, -1e13]),
        np.array([10.0, 0.0]),
    )
    assert lp.ub_sep[0] >= 2.0 - 1e-9, f"x <= {lp.ub_sep[0]} cuts (2, -1e13)"
    checked += 1
    assert checked == 4


def _defined_model(c: float | None, y_lb: float = -INF, y_ub: float = INF):
    m = dm.Model("noise_defined")
    x0 = m.continuous("x0", lb=1.0, ub=1.0)
    x1 = m.continuous("x1", lb=1.0, ub=1.0)
    y = m.continuous("y", lb=y_lb, ub=y_ub)
    m.minimize(x0 + x1)
    m.subject_to((y if c is None else c * y) + 0.1 * x0 + 0.2 * x1 == 0.3)
    return m


def test_defined_variable_forward_divides_only_by_a_path_sign_count():
    """Why ``DefinedVariableForwardRule`` needs no noise guard (classified safe).

    Its divisor is the scale ``_flatten_sum`` gives a scalar leaf: a sum of the
    ``±1`` signs on the ``+``/``-``/``neg`` paths to it, never a model constant (a
    ``c * y`` product is an opaque leaf). So the divisor is an integer or 0, and 0 is
    skipped. Pin that: ``c * y`` must not isolate ``y``, for any ``c``, while a bare
    ``y`` is isolated and bounded.
    """
    checked = 0
    for c in (*NOISE, CONTROL):
        m = _defined_model(c)
        _, lb, ub, _, _ = _extract_variable_info(m)
        md = _cached_flat_metadata(m)
        terms = _cached_flat_terms(m, m._constraints[0].body)
        assert all(float(s).is_integer() for s, _leaf in terms), terms
        assert DefinedVariableForwardRule._isolated_defined_var(terms, md, lb, ub) is None
        out_lb, out_ub = DefinedVariableForwardRule().tighten(m, lb.copy(), ub.copy(), md)
        assert _contains(out_lb[2], out_ub[2], _exact_y(c))
        checked += 1
    m = _defined_model(None)  # a bare ``y`` leaf
    _, lb, ub, _, _ = _extract_variable_info(m)
    out_lb, out_ub = DefinedVariableForwardRule().tighten(
        m, lb.copy(), ub.copy(), _cached_flat_metadata(m)
    )
    assert np.isfinite(out_lb[2]) and np.isfinite(out_ub[2]), "the bare leaf is still bounded"
    checked += 1
    assert checked == 4


def test_obbt_equality_defined_bounds_never_divide_by_a_noise_coefficient():
    checked = 0
    for c in (*NOISE, CONTROL):
        m = _defined_model(c)
        _, lb, ub, _, _ = _extract_variable_info(m)
        out_lb, out_ub, _n = propagate_equality_defined_bounds(m, lb.copy(), ub.copy())
        y = _exact_y(c)
        assert _contains(out_lb[2], out_ub[2], y), f"c={c}: [{out_lb[2]}, {out_ub[2]}] cuts {y}"
        if c == CONTROL:
            assert np.isfinite(out_lb[2]) and np.isfinite(out_ub[2]), "control still bounded"
        checked += 1
    assert checked == 3


def test_single_variable_pin_never_divides_by_a_noise_coefficient():
    checked = 0
    for c in (*NOISE, CONTROL):
        m = dm.Model("noise_pin")
        y = m.continuous("y", lb=-1.0, ub=1.0)
        m.minimize(y)
        m.subject_to(c * y + 0.1 + 0.2 == 0.3)
        _, lb, ub, _, _ = _extract_variable_info(m)
        out_lb, out_ub = _fix_single_var_equalities(m, lb.copy(), ub.copy())
        if c == CONTROL:
            assert out_ub[0] - out_lb[0] < 1e-9, "the control is still pinned"
        else:
            # The pin is the residual's round-off over c: -5.55e-5 or +5.55e-5, with
            # nothing in the row to say which of the stored constants was folded first.
            assert out_lb[0] == -1.0 and out_ub[0] == 1.0, f"c={c}: pinned at {out_lb[0]}"
        checked += 1
    # A dropped tiny term would turn x + 1e-13 y == 1 into a pin of x.
    m = dm.Model("noise_pin_term")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=-1e13, ub=0.0)
    m.minimize(x)
    m.subject_to(x + 1e-13 * y == 1.0)
    _, lb, ub, _, _ = _extract_variable_info(m)
    out_lb, out_ub = _fix_single_var_equalities(m, lb.copy(), ub.copy())
    assert out_ub[0] >= 2.0 - 1e-9, f"x pinned to [{out_lb[0]}, {out_ub[0]}] cuts (2, -1e13)"
    checked += 1
    assert checked == 4
