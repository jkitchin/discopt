"""``dm.xlogx`` — the ideal-mixing atom ``x log x`` — end to end (#1242).

Every CALPHAD phase carries a :math:`\\sum_i y_i \\log y_i` term over site
fractions, and a site-fraction box *always* starts at ``lo = 0``. Before this
change the atom existed only as an internal ``FunctionCall("entropy", x)`` node
and three of the four layers that have to agree about it did not:

* the interval rule returned ``[-inf, +inf]`` for any box with ``lo <= 0``, so a
  convexity certificate abstained on every mixing term there is;
* the core IR had no ``MathFunc`` variant, so ``model_to_repr`` raised and the
  whole Rust presolve / FBBT layer was unreachable for such a model;
* ``symbolic_diff`` had no derivative rule.

The consequence was measured on the acceptance model below: ``min y log y +
(1-y) log(1-y) + 3 y (1-y)`` over ``y in [0, 1]`` terminated ``feasible`` after a
single node with ``bound=None`` — a correct incumbent that could never be
certified.

Soundness is the point of the tests, not speed: the interval enclosure is
checked against a dense sample of the function (it must CONTAIN it), and the
B&B bound is checked against the true optimum (it must never exceed it).
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval import entropy as iv_entropy
from discopt._relax.convexity.lattice import Curvature
from discopt._relax.convexity.rules import classify_model
from discopt.bilevel.symbolic_diff import diff
from discopt.modeling.core import FunctionCall

_INV_E = 1.0 / math.e


def _xlogx(t: np.ndarray) -> np.ndarray:
    """Reference ``t log t`` with the continuous extension ``f(0) = 0``."""
    t = np.asarray(t, dtype=float)
    return np.where(t == 0.0, 0.0, t * np.log(np.where(t == 0.0, 1.0, t)))


def _iv(lo: float, hi: float) -> Interval:
    return iv_entropy(Interval(np.array([lo]), np.array([hi])))


# ──────────────────────────────────────────────────────────────────────
# 1. Public surface
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_xlogx_is_public_and_builds_the_entropy_node():
    """``dm.xlogx`` / ``discopt.xlogx`` exist and keep the ``entropy`` node."""
    import discopt

    assert dm.xlogx is discopt.xlogx
    m = Model()
    y = m.continuous("y", lb=0.0, ub=1.0)
    e = dm.xlogx(y)
    assert isinstance(e, FunctionCall)
    assert e.func_name == "entropy"
    assert e.args[0] is y
    # Element-wise, so the shape flows into an enclosing broadcast (#816).
    ya = m.continuous("ya", shape=3, lb=0.0, ub=1.0)
    assert dm.xlogx(ya).shape == (3,)


# ──────────────────────────────────────────────────────────────────────
# 2. Interval rule: exact on [0, u], abstains below 0
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "lo,hi",
    [
        (0.0, 0.2),  # u < 1/e   -> decreasing branch
        (0.0, _INV_E),  # u == 1/e  -> minimizer at the right endpoint
        (0.0, 1.0),  # u > 1/e   -> interior minimum
        (0.0, 2.0),  # u > 1     -> positive upper end
        (0.0, 0.0),  # degenerate box at the extension point
        (0.1, 0.9),
        (0.5, 0.9),  # strictly increasing branch
    ],
)
def test_interval_encloses_and_is_tight_on_the_closed_domain(lo, hi):
    """The enclosure must contain the function AND be the exact hull."""
    got = _iv(lo, hi)
    assert np.isfinite(got.lo).all() and np.isfinite(got.hi).all(), (
        f"entropy([{lo}, {hi}]) is not finite: {got}"
    )
    sample = _xlogx(np.linspace(lo, hi, 20001))
    # Soundness: the enclosure contains every sampled value.
    assert got.lo[0] <= sample.min() + 1e-12
    assert got.hi[0] >= sample.max() - 1e-12
    # Tightness: it is the hull, not a loose superset (the sample is dense
    # enough to reach both extrema to ~1e-8 on these boxes).
    assert got.lo[0] >= sample.min() - 1e-6
    assert got.hi[0] <= sample.max() + 1e-6


@pytest.mark.unit
def test_interval_lower_end_is_minus_one_over_e_when_the_minimizer_is_inside():
    assert _iv(0.0, 1.0).lo[0] == pytest.approx(-_INV_E, abs=1e-12)
    assert _iv(0.0, _INV_E).lo[0] == pytest.approx(-_INV_E, abs=1e-12)
    # Just short of the minimizer: the endpoint value, not -1/e.
    u = _INV_E - 1e-3
    assert _iv(0.0, u).lo[0] == pytest.approx(u * math.log(u), abs=1e-12)
    assert _iv(0.0, u).lo[0] > -_INV_E


@pytest.mark.unit
def test_interval_abstains_below_zero():
    """``lo < 0`` is outside the domain: abstain, never guess."""
    got = _iv(-0.1, 1.0)
    assert got.lo[0] == -np.inf and got.hi[0] == np.inf
    got = _iv(-1e-12, 1.0)
    assert got.lo[0] == -np.inf and got.hi[0] == np.inf


# ──────────────────────────────────────────────────────────────────────
# 3. Convexity: an ideal-mixing term certifies
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_binary_entropy_is_proved_convex_on_the_unit_interval():
    """``y log y + (1-y) log(1-y)`` on ``[0, 1]``.

    The complement ``1 - y`` is nonnegative only by the box, never
    syntactically, which is why ``entropy`` joins the sign-refined atoms.
    """
    m = Model()
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.xlogx(y) + dm.xlogx(1 - y))
    obj_convex, _ = classify_model(m)
    assert obj_convex


@pytest.mark.unit
def test_simplex_entropy_is_proved_convex():
    """The CALPHAD shape: mixing entropy over a site-fraction simplex."""
    m = Model()
    y = m.continuous("y", shape=3, lb=0.0, ub=1.0)
    m.subject_to(dm.sum(y) == 1.0)
    m.minimize(dm.sum(dm.xlogx(y)))
    obj_convex, con_mask = classify_model(m)
    assert obj_convex
    assert all(con_mask)


@pytest.mark.unit
def test_entropy_of_a_nonaffine_argument_is_not_claimed_convex():
    """``entropy`` is not monotone, so ``entropy(convex)`` must abstain.

    The guard that keeps the certificate sound: a CONVEX-but-not-monotone atom
    licenses a verdict only for an AFFINE argument.
    """
    from discopt._relax.convexity.rules import classify_expr

    m = Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    # x**2 is convex and nonnegative, but entropy(x**2) is not convex on a box
    # that straddles the minimizer of the composition.
    assert classify_expr(dm.xlogx(x**2), m, {}) is Curvature.UNKNOWN


# ──────────────────────────────────────────────────────────────────────
# 4. Rust IR: the atom reaches the core (and so FBBT / presolve)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_entropy_lowers_into_the_core_arena():
    """``model_to_repr`` used to raise ``Unknown MathFunc: entropy``."""
    from discopt._rust import model_to_repr

    m = Model()
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.xlogx(y))
    repr_ = model_to_repr(m, getattr(m, "_builder", None))
    assert repr_ is not None


@pytest.mark.unit
def test_nl_export_of_an_entropy_model_refuses_loudly():
    """`.nl` has no entropy opcode and no exact rewrite: refuse, don't guess.

    ``x*log(x)`` is not a faithful substitute — it is ``nan`` at ``x = 0`` and
    its second derivative overflows near 0 — so writing it would hand an
    external solver a *different* function on exactly the boxes this atom is
    for. The refusal must name the atom, not report "unknown function", now
    that ``xlogx`` is public.
    """
    from discopt.export.nl import to_nl

    m = Model()
    y = m.continuous("y", lb=0.01, ub=1.0)
    m.minimize(dm.xlogx(y))
    with pytest.raises(ValueError, match="xlogx"):
        to_nl(m)


# ──────────────────────────────────────────────────────────────────────
# 5. Symbolic derivative
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_symbolic_diff_of_entropy_is_log_plus_one():
    m = Model()
    y = m.continuous("y", lb=0.1, ub=1.0)
    d = diff(dm.xlogx(y), y)
    # Evaluate the emitted expression against the analytic derivative.
    from discopt._relax.dag_compiler import compile_expression

    fn = compile_expression(d, m)
    checked = 0
    for val in (0.1, 0.25, _INV_E, 0.5, 1.0):
        got = float(fn(np.array([val])))
        assert got == pytest.approx(math.log(val) + 1.0, rel=1e-10, abs=1e-12)
        checked += 1
    assert checked == 5


@pytest.mark.unit
def test_symbolic_diff_chains_through_the_argument():
    m = Model()
    y = m.continuous("y", lb=0.1, ub=0.9)
    d = diff(dm.xlogx(1 - y), y)
    from discopt._relax.dag_compiler import compile_expression

    fn = compile_expression(d, m)
    for val in (0.2, 0.5, 0.8):
        # d/dy [ (1-y) log(1-y) ] = -(log(1-y) + 1)
        assert float(fn(np.array([val]))) == pytest.approx(-(math.log(1 - val) + 1.0), rel=1e-10)


# ──────────────────────────────────────────────────────────────────────
# 6. Acceptance: the model that could not certify
# ──────────────────────────────────────────────────────────────────────


def _acceptance_model() -> Model:
    m = Model()
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.xlogx(y) + dm.xlogx(1 - y) + 3 * y * (1 - y))
    return m


@pytest.mark.slow
@pytest.mark.correctness
def test_entropy_objective_solves_globally_with_a_valid_bound():
    """Certifies, with a finite bound that never exceeds the true optimum.

    Before #1242 this returned ``status='feasible'``, ``bound=None`` after one
    node: the interval rule abstained on ``[0, 1]``, so no relaxation bound was
    available at the root.
    """
    m = _acceptance_model()
    res = m.solve(solver="bb", time_limit=300)
    assert res.status == "optimal"
    assert res.bound is not None and math.isfinite(res.bound)
    # The certificate invariant: a lower bound never crosses the incumbent, and
    # never exceeds the true optimum of a model whose incumbent is feasible.
    assert res.bound <= res.objective + 1e-9
    assert res.objective == pytest.approx(-0.05834134944143926, abs=1e-6)


@pytest.mark.slow
@pytest.mark.correctness
def test_entropy_objective_certifies_to_an_absolute_gap_of_1e_9():
    """The issue's stated acceptance bar.

    Reaching it needs #1243: at the default the search stops on the *absolute*
    criterion at ~9e-7, because ``_DEFAULT_ABS_GAP_TOL = 1e-6`` used to be a
    module constant with no caller control. Tightening ``gap_tolerance`` alone
    does nothing — the absolute arm is what binds here.
    """
    res = _acceptance_model().solve(
        solver="bb", time_limit=300, gap_tolerance=1e-9, abs_gap_tolerance=1e-9
    )
    assert res.status == "optimal"
    assert res.bound is not None and math.isfinite(res.bound)
    assert abs(res.objective - res.bound) <= 1e-9
    assert res.bound <= res.objective + 1e-12
