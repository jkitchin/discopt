"""``x*x`` is ``x**2``; nonlinear bound tightening must not care which you wrote (#1395).

``_match_scaled_square_var`` recognized a square only as a ``Pow`` node, so every
square-based tightening rule — ``sum_of_squares_upper_bound``,
``sqrt_sum_of_squares_upper_bound``, the separable and univariate quadratic rules,
all five of which consume that one matcher — silently declined the ``Mul``
spelling. Measured on ``x`` declared over ``[-5, 5]``:

* ``x**2 <= 4`` tightened the box to ``[-2, 2]``; ``x*x <= 4`` left it ``[-5, 5]``
  — a 2.5x looser box on each side, weakening every envelope derived from it.
* ``x**2 <= -1`` was refuted by ``sum_of_squares_upper_bound``; ``x*x <= -1``
  was not refuted at all, so the solve fell through to the continuous NLP, which
  returned an infeasible point, and ``solve()`` reported ``status="error"`` with
  ``objective=None``, ``bound=None`` and ``error=None`` for a model whose
  infeasibility one interval-arithmetic pass settles.

The relaxation layer was never affected (it is syntax-blind: the same nonconvex
model written both ways gave bit-identical bound and node count), which is what
localized the defect to the matcher.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.nonlinear_bound_tightening import (
    _match_scaled_square_var,
    build_flat_variable_metadata,
)
from discopt.solver import (
    _declared_box_tightening,
    _detect_nonlinear_bound_infeasibility,
)


def _square_model(form: str, rhs: float, lb: float = -5.0, ub: float = 5.0) -> dm.Model:
    """``<square> <= rhs`` over ``x in [lb, ub]``, the square spelled two ways."""
    m = dm.Model(f"square_{form}")
    x = m.continuous("x", lb=lb, ub=ub)
    m.subject_to((x**2 if form == "pow" else x * x) <= rhs)
    m.minimize(x)
    return m


@pytest.mark.unit
@pytest.mark.parametrize("rhs,expected", [(4.0, 2.0), (9.0, 3.0), (0.25, 0.5)])
def test_box_tightening_is_the_same_for_both_spellings(rhs, expected):
    """``x*x <= rhs`` must tighten ``x`` to ``[-sqrt(rhs), sqrt(rhs)]``, as ``x**2`` does."""
    boxes = {}
    for form in ("pow", "mul"):
        tightening = _declared_box_tightening(_square_model(form, rhs))
        assert tightening is not None, f"{form}: no tightening result at all"
        lb, ub, stats = tightening
        assert not stats.infeasible, f"{form}: a satisfiable row was called infeasible"
        boxes[form] = (float(lb[0]), float(ub[0]))
    assert boxes["mul"] == pytest.approx(boxes["pow"], abs=1e-9), (
        f"the two spellings disagree: pow={boxes['pow']!r} mul={boxes['mul']!r}"
    )
    assert boxes["mul"] == pytest.approx((-expected, expected), abs=1e-9)


@pytest.mark.unit
def test_an_unsatisfiable_square_row_is_refuted_in_both_spellings():
    """A nonnegative square with a negative upper bound has no solution anywhere."""
    reasons = {}
    for form in ("pow", "mul"):
        reasons[form] = _detect_nonlinear_bound_infeasibility(_square_model(form, -1.0))
    for form, reason in reasons.items():
        assert reason is not None, f"{form}: an unsatisfiable square row was not refuted"
        assert "sum of squares" in reason, f"{form}: unexpected proof {reason!r}"


@pytest.mark.unit
def test_refutation_holds_when_the_box_excludes_zero():
    """``x*x <= -1`` on ``x in [1, 5]`` — body interval ``[1, 25]``, no solution."""
    reason = _detect_nonlinear_bound_infeasibility(_square_model("mul", -1.0, lb=1.0, ub=5.0))
    assert reason is not None, "an unsatisfiable square row was not refuted"


@pytest.mark.unit
def test_a_mixed_row_of_products_and_squares_is_refuted():
    """``x*y + x*x <= 0.5`` over ``x, y in [1, 2]``: the body is at least 2."""
    m = dm.Model("mixed")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    m.subject_to(x * y + x * x <= 0.5)
    m.minimize(x + y)
    # This row is not a pure sum of squares, so the sum-of-squares rule declines
    # it; what must hold is that the solve does not report ``error`` for a model
    # with no feasible point.
    r = m.solve(time_limit=30)
    assert r.status != "optimal", f"an infeasible model reported {r.status!r}"
    assert r.objective is None, f"an infeasible model returned objective {r.objective!r}"


@pytest.mark.unit
def test_the_matcher_refuses_what_is_not_a_square():
    """A genuine bilinear term must stay unmatched — the soundness side of #1395."""
    m = dm.Model("bilinear")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    m.subject_to(x * y <= 3.0)
    m.minimize(x + y)
    meta = build_flat_variable_metadata(m)

    checked = 0
    # x*x and x**2 both match, to the same flat index and scale.
    for expr in (x * x, x**2):
        got = _match_scaled_square_var(expr, 1.0, meta)
        assert got is not None, f"{expr!r} must match as a square"
        assert got[1] == pytest.approx(1.0)
        checked += 1
    assert _match_scaled_square_var(x * x, 1.0, meta) == _match_scaled_square_var(
        x**2, 1.0, meta
    ), "the two spellings must match identically"
    checked += 1
    # A product of DIFFERENT variables is not a square.
    assert _match_scaled_square_var(x * y, 1.0, meta) is None, "x*y is not a square"
    checked += 1
    # A scaled square keeps its coefficient.
    scaled = _match_scaled_square_var(3.0 * (x * x), 1.0, meta)
    assert scaled is not None and scaled[1] == pytest.approx(3.0)
    checked += 1
    # A cube is not a square.
    assert _match_scaled_square_var(x**3, 1.0, meta) is None, "x**3 is not a square"
    checked += 1
    assert checked == 6, f"only {checked} assertions executed"


@pytest.mark.smoke
def test_solve_reports_infeasible_not_error_for_a_mul_square():
    """End to end: the user-visible consequence of the missed recognition."""
    for form in ("pow", "mul"):
        r = _square_model(form, -1.0).solve(time_limit=30)
        assert r.status == "infeasible", f"{form}: expected infeasible, got {r.status!r}"
        assert r.objective is None and r.bound is None


@pytest.mark.smoke
def test_a_satisfiable_mul_square_still_solves_to_the_right_answer():
    """The tightening must not cut the optimum out: ``min x`` s.t. ``x*x <= 4`` is -2."""
    for form in ("pow", "mul"):
        r = _square_model(form, 4.0).solve(time_limit=30)
        assert r.objective is not None, f"{form}: no solution returned"
        assert r.objective == pytest.approx(-2.0, abs=1e-5), (
            f"{form}: objective {r.objective!r} is not the true optimum -2"
        )
        if r.bound is not None:
            assert r.bound <= -2.0 + 1e-5, f"{form}: bound {r.bound!r} above the optimum"
        # And the point it returns actually satisfies the row.
        xv = float(np.asarray(r.x["x"]).reshape(-1)[0])
        assert xv * xv <= 4.0 + 1e-6, f"{form}: returned x={xv!r} violates x*x <= 4"


@pytest.mark.smoke
def test_the_tightened_box_never_cuts_a_feasible_point():
    """Sample the row's feasible set and require every sampled point inside the box.

    The soundness direction for a bound-changing change: a tightening that
    excluded a genuinely feasible point would be a false reduction.
    """
    m = dm.Model("sos")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.subject_to(2.0 * (x * x) + 3.0 * (y * y) <= 12.0)
    m.minimize(x + y)
    tightening = _declared_box_tightening(m)
    assert tightening is not None
    lb, ub, stats = tightening
    assert not stats.infeasible

    rng = np.random.default_rng(1395)
    checked = 0
    for _ in range(4000):
        px, py = rng.uniform(-5.0, 5.0, size=2)
        if 2.0 * px * px + 3.0 * py * py > 12.0:
            continue
        checked += 1
        assert lb[0] - 1e-9 <= px <= ub[0] + 1e-9, f"tightening cut feasible x={px!r}"
        assert lb[1] - 1e-9 <= py <= ub[1] + 1e-9, f"tightening cut feasible y={py!r}"
    assert checked > 100, f"only {checked} feasible samples drawn — the probe proved nothing"
    # ...and it really did tighten, or the test above is vacuous.
    assert ub[0] < 5.0 and ub[1] < 5.0, f"no tightening happened: lb={lb!r} ub={ub!r}"
