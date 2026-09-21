"""#1397: an absolute tolerance must not be asked a scale-dependent question.

``_EMPTY_INTERVAL_FEAS_TOL = 1e-6`` answers "is this violation inside the solver's
feasibility tolerance?" -- a scale-free question about the *model*. Every one of its
call sites in ``nonlinear_bound_tightening`` also asked it a second question it
cannot answer: "is this crossover real, or is it the round-off of the arithmetic
that produced it?" That one is not scale-free. Differencing floats of magnitude
``M`` carries an error of order ``n*u*M``, and at ``M = 1e10`` a single ulp is
``1.9e-6`` -- already past the whole tolerance. So on a large row an ordinary
rounding residual was read as a *proof* of infeasibility and the node was pruned.

The fix is an **additive** round-off bound (``_roundoff_slack``), never a
multiplicative inflation of the tolerance. Two things follow, and both are pinned
below:

* ``max(1, |result|)`` -- the shape the audit doc originally proposed -- is the
  wrong yardstick and was falsified before this landed. A cancellation residual is
  *small by construction*: terms of magnitude 1e10 that cancel to ``+3e-5`` give
  ``max(1, 3e-5) = 1``, which changes nothing. Worse, at ``M = 1e14`` the
  multiplicative form gives a threshold of ``1e8``, which would snap away a
  genuinely empty interval 1.4e6 wide. The scale lives in the *inputs*.
* The yardstick changes, never the tolerance. ``_EMPTY_INTERVAL_FEAS_TOL`` still
  reads 1e-6 and an O(1) problem keeps bit-identical behaviour, because the slack
  of O(1) terms is O(1e-15).

Each test below is a **sweep over magnitude**, not a single instance: the defect is
a class (an absolute constant compared against a scale-dependent quantity), so the
regression has to pin the class. Every sweep carries a matching *no-weakening* arm
-- a genuine violation at the same magnitude, which must still be proved -- because
the cheap way to pass the first arm is to stop proving infeasibility at all.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import nonlinear_bound_tightening as nbt
from discopt.solver import _declared_box_tightening

U = float(np.finfo(np.float64).eps)

#: Magnitudes at which one ulp reaches or exceeds the 1e-6 tolerance. Below ~1.5e10
#: the tolerance still dominates and the old code was right, which is why the sweep
#: starts here rather than at 1.0.
MAGNITUDES = (1e11, 1e12, 1e14, 1e16, 1e18)

#: Crossovers of this many ulps must survive as round-off, not be proved infeasible.
ULPS = (1, 2, 8)


def _genuine(magnitude: float) -> float:
    """A violation far beyond any round-off bound at this magnitude.

    ``_roundoff_slack`` over two terms of magnitude ``M`` is ``8*u*2M``; this is
    1e4 times that, so the no-weakening arms fail loudly if the slack is ever
    widened into a blanket amnesty.
    """
    return max(1.0, 1e4 * 16.0 * U * magnitude)


def test_the_sweep_constants_are_not_empty():
    """#6: a sweep that degenerates to nothing reports a pass having tested nothing."""
    assert len(MAGNITUDES) >= 4
    assert len(ULPS) >= 3
    assert min(MAGNITUDES) >= 1e10


# --------------------------------------------------------------------------- #
# The shared yardstick
# --------------------------------------------------------------------------- #


def test_an_o1_problem_keeps_todays_behaviour():
    """The slack must be negligible against 1e-6 wherever the old code was right."""
    assert nbt._roundoff_slack(0.0, 0.0) == 0.0
    assert nbt._roundoff_slack(1.0, -1.0) < 1e-14
    assert nbt._roundoff_slack(1e3, 1e3) < 1e-11


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_slack_covers_a_few_ulps_of_the_terms(magnitude):
    """It must exceed the worst rounding of a sum at this magnitude, with margin."""
    slack = nbt._roundoff_slack(magnitude, -magnitude)
    assert slack > float(np.spacing(magnitude))
    # ...and still be a round-off-scale quantity, not a licence to ignore the row.
    assert slack < 1e-13 * magnitude


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_a_term_with_no_magnitude_contributes_nothing(value):
    """An ``inf`` slack would widen every guard without limit.

    It would swallow the genuine ``+inf > ub`` emptiness along with the artifact and
    make the outward widenings unbounded, so a term with no magnitude falls back to
    today's behaviour.
    """
    assert nbt._roundoff_slack(value) == 0.0
    assert nbt._roundoff_slack(value, 1.0) == nbt._roundoff_slack(1.0)


@pytest.mark.parametrize("value", [1e19, -1e19, 9.999e19, 1e24, 1e30])
def test_a_large_finite_term_does_count(value):
    """The 1e19 "effectively infinite" cut is for *bounds*, not computed quantities.

    This helper is also handed row arithmetic, where ``b*b`` at ``|b| = 1e12`` is a
    genuine 1e24: filtering it as a sentinel returned a slack of zero and the
    discriminant guard reverted to the absolute tolerance. Including large finite
    terms is safe in both roles because every use widens a yardstick *outward*.
    """
    assert nbt._roundoff_slack(value) > 0.0
    assert nbt._roundoff_slack(value) == pytest.approx(
        nbt._CROSSOVER_OPS * U * abs(value), rel=1e-12
    )


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_vector_form_agrees_with_the_scalar_one_elementwise(magnitude):
    lo = np.array([-magnitude, -1.0, 0.0, -np.inf, -9.999e19])
    hi = np.array([magnitude, 1.0, 0.0, 1.0, 9.999e19])
    got = nbt._roundoff_slack_arr(lo, hi)
    for i in range(lo.size):
        assert got[i] == pytest.approx(nbt._roundoff_slack(float(lo[i]), float(hi[i])), rel=1e-12)
    # Per-column, so one huge variable does not inflate the O(1) ones.
    assert got[1] < 1e-14


# --------------------------------------------------------------------------- #
# Box crossovers: _snap_tolerant_crossovers and the empty-interval declarations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("ulps", ULPS)
def test_a_rounding_crossover_is_not_a_proof_of_emptiness(magnitude, ulps):
    """``lb`` a few ulps above ``ub`` at magnitude M is noise, not an empty box."""
    ub = np.array([magnitude])
    lb = ub.copy()
    for _ in range(ulps):
        lb = np.nextafter(lb, np.inf)
    assert lb[0] > ub[0], "the construction must actually cross"
    nbt._snap_tolerant_crossovers(lb, ub)
    assert lb[0] == ub[0], (
        f"a {ulps}-ulp crossover at {magnitude:.0e} "
        f"({lb[0] - ub[0]:.3e}) was left crossed and is pruned upstream"
    )


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_a_genuine_empty_box_is_still_empty_after_the_snap(magnitude):
    """The no-weakening arm: a real crossover must survive to be pruned."""
    ub = np.array([magnitude])
    lb = np.array([magnitude + _genuine(magnitude)])
    nbt._snap_tolerant_crossovers(lb, ub)
    assert lb[0] > ub[0], "a genuine empty interval was snapped away"


@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("ulps", ULPS)
def test_the_entry_point_does_not_declare_a_rounding_crossover_empty(magnitude, ulps):
    """Same class, at ``tighten_nonlinear_bounds``' own emptiness check."""
    m = dm.Model("nbt_entry")
    m.continuous("x", lb=-magnitude, ub=magnitude)
    m.minimize(0.0 * m._variables[0])

    ub = np.array([magnitude])
    lb = ub.copy()
    for _ in range(ulps):
        lb = np.nextafter(lb, np.inf)
    _, _, stats = nbt.tighten_nonlinear_bounds(m, lb, ub)
    assert not stats.infeasible, stats.infeasibility_reason


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_entry_point_still_declares_a_genuine_empty_box(magnitude):
    m = dm.Model("nbt_entry_genuine")
    m.continuous("x", lb=-magnitude, ub=magnitude)
    m.minimize(0.0 * m._variables[0])

    ub = np.array([magnitude])
    lb = np.array([magnitude + _genuine(magnitude)])
    _, _, stats = nbt.tighten_nonlinear_bounds(m, lb, ub)
    assert stats.infeasible


# --------------------------------------------------------------------------- #
# The discriminant: a cancellation built inside the rule, not by its caller
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("b", [1e6, 1e8, 1e10, 1e12])
@pytest.mark.parametrize("ulps", ULPS)
def test_a_rounding_negative_discriminant_is_not_infeasible(b, ulps):
    """``b*b + 4*a*rhs`` cancels; its error is ulps of ``b*b``, not of the result.

    The vertex violation of ``a*x^2 + b*x <= rhs`` is ``-discriminant/(4a)``, so the
    tolerance is correctly translated by ``4a`` -- but the *discriminant's own*
    round-off is not, and at ``|b| = 1e8`` one ulp of ``b*b`` is 2.2, against a
    translated tolerance of 4e-6.
    """
    a = 1.0
    # discriminant == -ulps * u * b*b, i.e. a few ulps below zero.
    rhs = -(b * b) * (1.0 + ulps * U) / (4.0 * a)
    got = nbt._tighten_univariate_quadratic_interval(a, b, rhs, -1e9, 1e9)
    assert got is not None, (
        f"|b|={b:.0e}, {ulps} ulps: discriminant "
        f"{b * b + 4.0 * a * rhs:.3e} read as proof of infeasibility"
    )


@pytest.mark.parametrize("b", [1e6, 1e8, 1e10, 1e12])
def test_a_genuinely_negative_discriminant_is_still_infeasible(b):
    a = 1.0
    rhs = -(b * b) * (1.0 + 1e4 * 16.0 * U) / (4.0 * a) - 1.0
    assert nbt._tighten_univariate_quadratic_interval(a, b, rhs, -1e9, 1e9) is None


# --------------------------------------------------------------------------- #
# Row residuals: the constant is a signed sum, so it cannot be its own yardstick
# --------------------------------------------------------------------------- #

#: Four row shapes, covering five rules. Each is built so that in EXACT arithmetic
#: the row is satisfied at ``x = 0``, while the float accumulation of its constant
#: leaves lands a ``spacing(M)``-scale residue on the wrong side of zero.
#:
#: The cancellation is ``+/-0.4e, +M, -M, -/+0.3e`` evaluated left to right: the
#: small leading term is absorbed when ``M`` is added (it is below half an ulp of
#: ``M``), so it is lost, and only the trailing one survives. Exact arithmetic keeps
#: both and gives ``0.1e`` with the opposite sign.
ROW_SHAPES = ("sum_of_squares", "sqrt_sum_of_squares", "quadratic_equality", "square_difference")


def _row_model(shape: str, magnitude: float, *, genuine: bool) -> dm.Model:
    spacing = float(np.spacing(magnitude))
    m = dm.Model(f"{shape}_{magnitude:.0e}")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=0.0)
    m.minimize(x)

    if shape in ("sum_of_squares", "sqrt_sum_of_squares"):
        head = x * x if shape == "sum_of_squares" else dm.sqrt(x * x)
        # `<=` rows: rhs = -constant_term, so the float residue must be POSITIVE.
        if genuine:
            m.subject_to(head + magnitude - magnitude + _genuine(magnitude) <= 0)
        else:
            m.subject_to(head - 0.4 * spacing + magnitude - magnitude + 0.3 * spacing <= 0)
    else:
        head = (y - x * x) if shape == "quadratic_equality" else (-(x * x) + y * y)
        # `==` rows against a fixed y = 0: the residue must be NEGATIVE.
        if genuine:
            m.subject_to(head + magnitude - magnitude - _genuine(magnitude) == 0)
        else:
            m.subject_to(head + 0.4 * spacing + magnitude - magnitude - 0.3 * spacing == 0)
    return m


@pytest.mark.parametrize("shape", ROW_SHAPES)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_a_cancellation_residue_is_not_proved_infeasible(shape, magnitude):
    """The row holds at ``x = 0`` in exact arithmetic; it must not be pruned."""
    lb, ub, stats = _declared_box_tightening(_row_model(shape, magnitude, genuine=False))
    assert not stats.infeasible, (
        f"{shape} at {magnitude:.0e}: a {np.spacing(magnitude):.3e}-scale rounding "
        f"residue was read as proof of infeasibility -- {stats.infeasibility_reason}"
    )
    assert lb[0] <= 0.0 <= ub[0], f"{shape} at {magnitude:.0e}: x = 0 was cut from the box"


@pytest.mark.parametrize("shape", ROW_SHAPES)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_a_genuine_row_violation_is_still_proved(shape, magnitude):
    """The no-weakening arm, at the same magnitude and through the same rules."""
    _, _, stats = _declared_box_tightening(_row_model(shape, magnitude, genuine=True))
    assert stats.infeasible, (
        f"{shape} at {magnitude:.0e}: a violation of {_genuine(magnitude):.3e} "
        f"stopped being proved -- the slack has become a blanket amnesty"
    )


def test_the_tolerance_constant_itself_did_not_move():
    """#1397 is not a tolerance-tuning exercise: the yardstick changed, not this."""
    assert nbt._EMPTY_INTERVAL_FEAS_TOL == 1e-6


# ---------------------------------------------------------------------------------
# #1397, second site: ``factorable_reform._ZERO_MARGIN`` -- the denominator sign gate
#
# ``_find_clearable_denominator`` licenses multiplying a whole constraint through by a
# denominator ``D`` once it believes ``D`` is sign-definite over the box, deciding that
# with ``lo > 1e-9`` where ``lo`` comes from ``_bound_expression`` -- plain float
# interval arithmetic with NO outward rounding, so ``lo`` is not a rigorous
# under-estimate of the true infimum. Clearing a sign-indefinite ``D`` is not a weaker
# bound: it FLIPS the inequality wherever ``D < 0``, so the rewritten model has a
# different feasible set than the one the user wrote -- a false-optimal generator.
# ---------------------------------------------------------------------------------


def _cancelling_denominator(y, magnitude: float, residue: float):
    """``y + M - M + residue``.

    ``dm`` does not fold constants, so the four leaves survive in source order and the
    float evaluation genuinely cancels: the exact infimum is ``y.lb + residue`` while
    ``_bound_expression`` reports ``residue`` once ``|y.lb|`` is absorbed by ``M``.
    """
    return y + magnitude - magnitude + residue


@pytest.mark.parametrize("ylb", [-0.5, -8.0])
@pytest.mark.parametrize("magnitude", [1e15, 1e16, 1e17, 1e18])
def test_a_cancelling_denominator_is_not_cleared(ylb, magnitude):
    """Exact infimum is negative; the float-folded lower bound clears 1e-9 anyway."""
    from discopt._relax import factorable_reform as fr

    m = dm.Model()
    y = m.continuous("y", lb=ylb, ub=1.0)
    x = m.continuous("x", lb=1.0, ub=2.0)
    d = _cancelling_denominator(y, magnitude, 1e-8)
    lo, _hi = fr._bound_expression(d, m)
    if not lo > fr._ZERO_MARGIN:
        pytest.skip(f"construction did not land at M={magnitude:.0e} (lo={lo:.3e})")
    assert fr._find_clearable_denominator(x / d - 1.0, m) is None, (
        f"M={magnitude:.0e}, exact infimum {ylb + 1e-8:.3e}: a denominator that is "
        f"negative over most of the box was cleared, which flips the constraint"
    )


@pytest.mark.parametrize("dlb,dub", [(1.0, 5.0), (0.5, 100.0), (-9.0, -0.25), (1e-3, 1.0)])
def test_a_genuinely_definite_denominator_is_still_cleared(dlb, dub):
    """No-weakening arm: the cheap way to pass the arm above is to clear nothing."""
    from discopt._relax import factorable_reform as fr

    m = dm.Model()
    d = m.continuous("d", lb=dlb, ub=dub)
    x = m.continuous("x", lb=1.0, ub=2.0)
    assert fr._find_clearable_denominator(x / d - 1.0, m) is not None, (
        f"D in [{dlb}, {dub}] is sign-definite by its own declared bounds; refusing "
        f"it means the slack has switched the pass off rather than made it rigorous"
    )


def test_an_unbounded_endpoint_does_not_poison_the_other_one():
    """The two endpoints fail separately and must be tracked separately.

    ``0.01 + x`` with ``x.ub = +inf`` has an exact lower endpoint and an unusable
    upper one, and the positive-sign test reads the *lower* one. Collapsing both into
    a single scalar error lost 92 of 303 denominator clears on the in-repo corpus
    (three ``heatexch_gen*`` instances to zero), which is why ``bound_expression_error``
    returns a pair.
    """
    from discopt._relax import factorable_reform as fr
    from discopt._relax.gdp_reformulate import bound_expression_error

    m = dm.Model()
    y = m.continuous("y", lb=0.0)  # ub defaults to the sentinel box
    x = m.continuous("x", lb=1.0, ub=2.0)
    d = 0.01 + y
    err_lo, err_hi = bound_expression_error(d, m)
    assert np.isfinite(err_lo), "the exact lower endpoint was reported as unbounded"
    assert fr._find_clearable_denominator(x / d - 1.0, m) is not None, (
        f"err=({err_lo!r}, {err_hi!r}): an unusable upper endpoint refused a clear "
        f"that only depends on the lower one"
    )


def test_the_zero_margin_constant_itself_did_not_move():
    """#1397 non-goal: the yardstick changed, not the constant."""
    from discopt._relax import factorable_reform as fr

    assert fr._ZERO_MARGIN == 1e-9


# ---------------------------------------------------------------------------------
# #1397, third site: the reduced-cost deadbands
#
# Both RC-fixing sites gate on an absolute 1e-7 and then DIVIDE the optimality gap by
# ``|d_j|``. A reduced cost is a *difference* -- ``c_j - (A^T y)_j`` at the simplex
# site, ``mult_x_L - mult_x_U`` at the POUNCE site -- so its own magnitude says nothing
# about how much of it is round-off. The unsound direction is specific: an over-stated
# ``|d_j|`` makes ``gap/|d_j|`` too small, which can fix the true optimum out of the
# box and certify a false ``optimal``.
# ---------------------------------------------------------------------------------

#: Magnitudes of the two terms the reduced cost is a difference of. A few ulps of these
#: clear the absolute 1e-7 deadband while carrying no information whatsoever.
RC_MAGNITUDES = (1e8, 1e9, 1e11, 1e14)


@pytest.mark.parametrize("magnitude", RC_MAGNITUDES)
def test_a_roundoff_reduced_cost_fixes_nothing_at_the_root(magnitude):
    from discopt.solver import _reduced_cost_fixing

    d = np.array([4.0 * U * magnitude])
    lb, ub = np.array([0.0]), np.array([1e12])
    _new_lb, new_ub, n = _reduced_cost_fixing(
        lb, ub, [0], d, 0.0, 1.0, rc_absum=np.array([2.0 * magnitude])
    )
    assert n == 0 and new_ub[0] == ub[0], (
        f"terms of magnitude {magnitude:.0e} differing by {d[0]:.3e} (a few ulps) "
        f"fixed ub to {new_ub[0]:g}; gap/|d| was divided by pure noise"
    )


@pytest.mark.parametrize("magnitude", RC_MAGNITUDES)
def test_a_roundoff_reduced_cost_fixes_nothing_at_a_node(magnitude):
    from discopt._relax.node_reduce import _dbbt_from_reduced_costs

    rc = np.array([4.0 * U * magnitude])
    lb, ub = np.array([0.0]), np.array([1e12])
    new_lb, new_ub, _nt, _infeas = _dbbt_from_reduced_costs(
        lb, ub, rc, 0.0, 1.0, np.array([True]), rc_absum=np.array([2.0 * magnitude])
    )
    assert new_ub[0] == ub[0] and new_lb[0] == lb[0], (
        f"node DBBT tightened [{new_lb[0]:g}, {new_ub[0]:g}] from a reduced cost of "
        f"{rc[0]:.3e} formed from {magnitude:.0e}-magnitude multipliers"
    )


@pytest.mark.parametrize(
    "d_val,absum_val,gap",
    [(0.5, 1.5, 10.0), (2.0, 5.0, 7.0), (1.0, 3.0, 100.0), (0.25, 0.75, 4.0)],
)
def test_an_honest_reduced_cost_still_fixes_exactly_as_far(d_val, absum_val, gap):
    """No-weakening arm: a well-scaled ``d_j`` must land on the same ``floor()``."""
    from discopt.solver import _reduced_cost_fixing

    lb, ub = np.array([0.0]), np.array([1e4])
    _new_lb, new_ub, n = _reduced_cost_fixing(
        lb, ub, [0], np.array([d_val]), 0.0, gap, rc_absum=np.array([absum_val])
    )
    inflated = gap + 1e-6 * (1.0 + abs(gap))
    expected = float(np.floor(inflated / d_val + 1e-9))
    assert n == 1 and new_ub[0] == expected, (
        f"d={d_val} at scale {absum_val}: ub={new_ub[0]:g} but the undeflated formula "
        f"gives {expected:g} -- a 1e-16-relative slack must not move a floor()"
    )


def test_reduced_cost_fixing_refuses_when_no_scale_is_reported():
    """Refusing is sound; guessing with an absolute deadband is not (CLAUDE.md §3)."""
    from discopt.solver import _reduced_cost_fixing

    lb, ub = np.array([0.0]), np.array([1e4])
    _new_lb, new_ub, n = _reduced_cost_fixing(
        lb, ub, [0], np.array([1.0]), 0.0, 10.0, rc_absum=None
    )
    assert n == 0 and new_ub[0] == ub[0]


def test_the_reduced_cost_deadbands_themselves_did_not_move():
    """#1397 non-goal, again: the yardstick changed, not the constants."""
    from discopt._relax.node_reduce import _RC_TOL
    from discopt.solver import _RCF_RC_TOL

    assert _RCF_RC_TOL == 1e-7
    assert _RC_TOL == 1e-7


# ---------------------------------------------------------------------------------
# #1397, fourth site: ``perspective._ZERO_TOL`` -- the separability gate
#
# ``find_candidates`` documents each of its gates as "a soundness condition, not a
# heuristic", and the separable one asked whether the off-diagonal mass of a Hessian
# row is zero -- by computing ``|Q[j,:]|.sum() - |Q[j,j]|``, a difference of two large
# numbers, and comparing it to an absolute 1e-12. At ``Q[j,j] = 1e12`` one ulp is
# 1.2e-4, so a genuine cross-term below that is absorbed by the row sum and the
# subtraction returns exactly 0.0: a coupled row certified separable, feeding a
# perspective strengthening of the OA master cut that is only valid when it is not.
# ---------------------------------------------------------------------------------

#: Diagonal magnitudes whose ulp exceeds the cross-term below.
PERSP_DIAGONALS = (1e10, 1e12, 1e14, 1e16)


def _coupled_semicontinuous_model(diagonal: float, cross: float):
    """``0.5*diagonal*x**2 + cross*x*z`` over a semicontinuous ``x``.

    The ``x`` row of the Hessian is ``[diagonal, cross]``: genuinely non-separable,
    so ``x`` must not be offered as a perspective candidate at any magnitude.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=5.0)
    z = m.continuous("z", lb=0.0, ub=5.0)
    y = m.binary("y")
    m.subject_to(x - 5.0 * y <= 0)
    m.minimize(0.5 * diagonal * x * x + cross * x * z + 0.5 * z * z)
    return m, x


@pytest.mark.parametrize("diagonal", PERSP_DIAGONALS)
def test_a_coupled_hessian_row_is_not_certified_separable(diagonal):
    from discopt._relax.perspective import find_candidates, perspective_objective_terms

    cross = 1e-6 * diagonal * U * 4.0  # a few ulps of the diagonal: invisible to the sum
    cross = max(cross, 1e-9)
    m, x = _coupled_semicontinuous_model(diagonal, cross)
    x_col = 0
    cands = [c for c in find_candidates(m) if c.flat == x_col]
    assert not cands, (
        f"diagonal {diagonal:.0e} with a genuine cross-term of {cross:.3e}: the row was "
        f"certified separable because the row sum absorbed the cross-term "
        f"(ulp = {np.spacing(diagonal):.3e}), so an invalid perspective cut is emitted"
    )
    assert all(t[0] != x_col for t in perspective_objective_terms(m))


@pytest.mark.parametrize("diagonal", PERSP_DIAGONALS)
def test_a_genuinely_separable_row_is_still_lifted(diagonal):
    """No-weakening arm: the same magnitudes, with no cross-term at all."""
    from discopt._relax.perspective import perspective_objective_terms

    m, _x = _coupled_semicontinuous_model(diagonal, 0.0)
    terms = perspective_objective_terms(m)
    assert any(t[0] == 0 for t in terms), (
        f"diagonal {diagonal:.0e}: a separable convex square over a semicontinuous x "
        f"stopped being lifted -- the gate has become a blanket refusal, not a test"
    )


def test_the_perspective_tolerance_itself_did_not_move():
    from discopt._relax import perspective

    assert perspective._ZERO_TOL == 1e-12


# ---------------------------------------------------------------------------
# signomial._ZERO_TOL -- the merged coefficient is an ACCUMULATED sum
#
# ``_merge_like_terms`` combined monomials sharing an exponent vector with a
# running ``buckets[key] += mono.coeff``, then compared the result against an
# absolute ``_ZERO_TOL = 1e-12`` to decide the term had cancelled, and let
# ``is_mixed_sign`` / ``has_negative_term`` read its *sign*. A running sum over
# terms of magnitude ``M`` carries an error of order ``n * u * M``, so both the
# zero test and the sign read were scale-exposed.
#
# The canonical witness needs THREE addends -- the float sum of two doubles
# always has the sign of their exact sum -- so the sweep is over the magnitude
# that swallows the middle term: with ``ulp(1e16) = 2.0``, adding 1.0 to 1e16
# changes nothing, and the subsequent ``-1e16`` then cancels to exactly 0.0
# while the exact sum is 1.0.
#
# Unsound consequences, both pinned below:
#   * the dropped term makes ``is_signomial`` return a form that EVALUATES
#     DIFFERENTLY from the expression it parsed, breaking the module's stated
#     contract that a non-``None`` return is a genuine signomial;
#   * when the lost term is the only negative one, the form reads as a pure
#     posynomial, so the signomial global engine (``solver="sgo"``, which
#     reports ``gap_certified=True``) builds its DC relaxation for the wrong
#     function.
# ---------------------------------------------------------------------------

# Magnitudes whose ulp exceeds the 1.0 middle coefficient (ulp(1e16) = 2.0).
SIGNOMIAL_SWAMPING_MAGNITUDES = (1e16, 1e17, 1e18, 1e20)


def _swamped_bucket_model(magnitude, middle):
    """``M*x*y + middle*x*y - M*x*y + 3*x``: exact ``x*y`` coefficient ``middle``.

    A running left-to-right sum loses ``middle`` entirely (it is below
    ``ulp(M)``) and then cancels the two ``M`` terms to exactly 0.0.
    """
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    expr = magnitude * x * y + middle * x * y - magnitude * x * y + 3.0 * x
    return m, expr


@pytest.mark.parametrize("magnitude", SIGNOMIAL_SWAMPING_MAGNITUDES)
def test_a_swamped_monomial_is_not_silently_dropped(magnitude):
    from discopt._relax.convexity.signomial import is_signomial

    # Establish that the construction really does swamp the middle term, so a
    # pass cannot come from the witness failing to land (CLAUDE.md §6).
    running = 0.0
    for c in (magnitude, 1.0, -magnitude):
        running += c
    assert running == 0.0, f"M={magnitude:.0e} did not swamp the 1.0; witness invalid"

    m, expr = _swamped_bucket_model(magnitude, 1.0)
    form = is_signomial(expr, m)
    assert form is not None
    # True body at x = y = 1 is 1.0*1*1 + 3.0*1 = 4.0.
    value = form.evaluate({0: 1.0, 1: 1.0})
    assert value == pytest.approx(4.0), (
        f"M={magnitude:.0e}: is_signomial returned a form evaluating to {value} where the "
        f"parsed expression is 4.0 -- the x*y term (exact coefficient 1.0) was dropped as "
        f"'cancelled' because the running sum reached exactly 0.0"
    )


@pytest.mark.parametrize("magnitude", SIGNOMIAL_SWAMPING_MAGNITUDES)
def test_a_swamped_negative_monomial_still_reads_as_negative(magnitude):
    from discopt._relax.convexity.signomial import is_signomial

    m, expr = _swamped_bucket_model(magnitude, -1.0)
    form = is_signomial(expr, m)
    assert form is not None
    assert form.has_negative_term, (
        f"M={magnitude:.0e}: the only negative monomial (exact coefficient -1.0) vanished, so "
        f"a mixed-sign signomial reads as a pure posynomial and the DC relaxation would be "
        f"built for the wrong function"
    )
    assert form.is_mixed_sign
    value = form.evaluate({0: 1.0, 1: 1.0})
    assert value == pytest.approx(2.0), f"M={magnitude:.0e}: body is 3.0 - 1.0 = 2.0, got {value}"


def test_a_genuinely_cancelling_bucket_is_still_collapsed():
    """No-weakening arm: an exact cancellation must still disappear.

    The fix makes the bucket sum exact; it must not make it *timid*. ``5*x*y``
    minus itself is exactly zero and has to collapse, or the canonical form stops
    being canonical and ``x - x`` masquerades as a two-term signomial.
    """
    from discopt._relax.convexity.signomial import is_signomial

    m = dm.Model()
    p = m.continuous("p", lb=1.0, ub=2.0)
    q = m.continuous("q", lb=1.0, ub=2.0)
    form = is_signomial(5.0 * p * q - 5.0 * p * q + 2.0 * p, m)
    assert form is not None
    assert len(form.monomials) == 1, (
        f"the exactly-cancelling p*q bucket survived as {[m.coeff for m in form.monomials]}"
    )
    assert form.monomials[0].coeff == pytest.approx(2.0)
    assert not form.has_negative_term


def test_an_ordinary_mixed_sign_signomial_is_unaffected():
    """No-weakening arm: the everyday case must parse exactly as before."""
    from discopt._relax.convexity.signomial import is_signomial

    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=4.0)
    y = m.continuous("y", lb=1.0, ub=4.0)
    form = is_signomial(2.0 * x**2 * y - 3.0 * x * y**0.5 + 1.5 * y, m)
    assert form is not None
    assert len(form.monomials) == 3
    assert form.is_mixed_sign
    assert form.evaluate({0: 1.0, 1: 1.0}) == pytest.approx(2.0 - 3.0 + 1.5)


def test_the_signomial_tolerances_themselves_did_not_move():
    from discopt._relax.convexity import signomial

    assert signomial._ZERO_TOL == 1e-12
    assert signomial._EXP_TOL == 1e-12
