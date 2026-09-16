"""A static index into a constant leaf is a constant (#1272, convexity layer).

Writing a coefficient vector as one array ``Parameter`` and using ``mu[k] * x[k]``
is the natural spelling — it is how discopt-calphad wants to carry chemical
potentials (#1249). The convexity classifier did not recognise ``mu[k]`` as a
constant, so the product rule saw two non-constants, fell through to UNKNOWN, and
the model lost its convexity certificate. Measured on the restricted-equilibrium
NLP below, before the fix:

    scalar parameters   convex=True   -> status=optimal   (single-NLP route, KKT reported)
    array parameter     convex=False  -> status=feasible  (spatial B&B, no certificate)

for two spellings of the *same* model, with the same optimum.

The fix resolves a **static** index into a ``Constant``/``Parameter`` to its
numeric value, using numpy's own indexing so it agrees with what the evaluator
computes. Since it manufactures curvature *proofs*, most of this file is about
the proofs it must NOT manufacture: the sign has to follow the indexed element,
a nonconvex model must stay unproven, and anything not statically resolvable must
be refused rather than guessed.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import Curvature, classify_expr
from discopt._relax.convexity.rules import _const_leaf_value, _is_static_index
from discopt.modeling.core import IndexExpression
from discopt.solver import _classify_model_convexity

pytestmark = [pytest.mark.smoke]


# --------------------------------------------------------------------------- #
# The resolver itself
# --------------------------------------------------------------------------- #
def test_const_leaf_value_resolves_scalars_arrays_and_indices():
    m = dm.Model("resolver")
    x = m.continuous("x", lb=0, ub=1)
    p = m.parameter("p", np.array([1.5, -2.0, 3.0]))
    q = m.parameter("q", 4.0)

    assert _const_leaf_value(q) == pytest.approx(4.0)
    np.testing.assert_allclose(_const_leaf_value(p), [1.5, -2.0, 3.0])
    assert _const_leaf_value(p[1]) == pytest.approx(-2.0)
    np.testing.assert_allclose(_const_leaf_value(p[0:2]), [1.5, -2.0])
    # A variable is not a constant leaf, indexed or not.
    assert _const_leaf_value(x) is None


def test_const_leaf_value_refuses_what_it_cannot_resolve():
    m = dm.Model("resolver_refusals")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    p = m.parameter("p", np.array([1.0, 2.0]))

    assert _const_leaf_value(x[0]) is None  # indexed VARIABLE
    assert _const_leaf_value(p * x[0]) is None  # not a leaf at all
    # ``p[7]`` raises at the operator (#816), but the expression-graph machinery
    # builds out-of-range IndexExpressions directly as lazy nodes. The resolver
    # must refuse those rather than propagate the IndexError into a
    # classification.
    assert _const_leaf_value(IndexExpression(p, 7)) is None
    assert _const_leaf_value(IndexExpression(p, "not-an-index")) is None


def test_symbolic_indices_are_not_static():
    m = dm.Model("static_index")
    i = m.continuous("i", lb=0, ub=2)
    assert _is_static_index(1)
    assert _is_static_index((0, 1))
    assert _is_static_index(slice(0, 2))
    assert _is_static_index(np.array([0, 2]))
    assert not _is_static_index(i)
    assert not _is_static_index((0, i))
    assert not _is_static_index(slice(0, i))


# --------------------------------------------------------------------------- #
# The proofs it should now make
# --------------------------------------------------------------------------- #
def test_indexed_parameter_times_variable_is_affine():
    m = dm.Model("affine")
    x = m.continuous("x", lb=0, ub=1)
    mu = m.parameter("mu", np.array([2.0, -3.0]))
    assert classify_expr(mu[0] * x, m) == Curvature.AFFINE
    assert classify_expr(mu[1] * x, m) == Curvature.AFFINE


def test_indexed_parameter_scales_curvature_by_its_own_sign():
    """The element's sign decides, not the array's."""
    m = dm.Model("scaling")
    x = m.continuous("x", lb=-1, ub=1)
    c = m.parameter("c", np.array([1.0, -1.0]))
    sq = x * x  # convex
    assert classify_expr(c[0] * sq, m) == Curvature.CONVEX
    assert classify_expr(c[1] * sq, m) == Curvature.CONCAVE


def test_the_restricted_equilibrium_model_is_convex_in_both_spellings():
    """The measurement from the module docstring, as a test."""
    n = 6
    g_vals = np.random.default_rng(0).normal(0.0, 1.0, size=n)
    a_rows = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 2.0, 1.0]])

    def build(array_spelling: bool):
        m = dm.Model("array" if array_spelling else "scalar")
        y = [m.continuous(f"y{k}", lb=1e-8, ub=10.0) for k in range(n)]
        if array_spelling:
            g = m.parameter("g", g_vals.copy())
            b = m.parameter("b", np.array([1.0, 2.0]))
            gs, bs = [g[k] for k in range(n)], [b[0], b[1]]
        else:
            gs = [m.parameter(f"g{k}", float(g_vals[k])) for k in range(n)]
            bs = [m.parameter("b0", 1.0), m.parameter("b1", 2.0)]
        m.minimize(sum(dm.xlogx(y[k]) + gs[k] * y[k] for k in range(n)))
        for row in range(2):
            m.subject_to(sum(float(a_rows[row, k]) * y[k] for k in range(n)) == bs[row])
        return m

    results = {}
    for array_spelling in (False, True):
        model = build(array_spelling)
        known, is_convex, _mask = _classify_model_convexity(model)
        assert known and is_convex, (array_spelling, known, is_convex)
        r = model.solve(time_limit=30)
        assert r.status == "optimal", (array_spelling, r.status)
        assert r.convex_fast_path is True
        assert r.kkt is not None and r.kkt["kkt_error"] < 1e-6
        results[array_spelling] = r.objective
    assert results[True] == pytest.approx(results[False], abs=1e-9)


def test_an_indexed_exponent_steers_the_power_rule_by_its_value():
    """The exponent path is the one where the resolved NUMBER — not just its sign
    — decides the verdict, so it is where a wrong resolution would fabricate a
    proof. ``x**2`` is convex, ``x**0.5`` is concave on ``x > 0``; both must match
    the scalar-parameter spelling exactly.
    """
    m = dm.Model("indexed_exponent")
    x = m.continuous("x", lb=0.1, ub=4.0)
    p = m.parameter("p", np.array([2.0, 0.5]))
    two = m.parameter("two", 2.0)
    half = m.parameter("half", 0.5)

    assert classify_expr(x ** p[0], m) == classify_expr(x**two, m) == Curvature.CONVEX
    assert classify_expr(x ** p[1], m) == classify_expr(x**half, m) == Curvature.CONCAVE


# --------------------------------------------------------------------------- #
# The proofs it must NOT make
# --------------------------------------------------------------------------- #
def test_a_nonconvex_model_with_array_parameters_is_not_claimed_convex():
    m = dm.Model("nonconvex_array_param")
    x = m.continuous("x", lb=0.1, ub=2.0)
    y = m.continuous("y", lb=0.1, ub=2.0)
    c = m.parameter("c", np.array([1.0, 1.0]))
    m.minimize(c[0] * x * y + c[1] * x)  # bilinear -> nonconvex
    m.subject_to(x + y == 2.0)
    known, is_convex, _mask = _classify_model_convexity(m)
    assert not (known and is_convex), "a bilinear objective must not read as convex"


def test_a_negative_indexed_coefficient_is_not_proven_convex():
    """``-1 * x^2`` is concave; proving it convex would be a false certificate."""
    m = dm.Model("concave_array_param")
    x = m.continuous("x", lb=-1, ub=1)
    c = m.parameter("c", np.array([-1.0]))
    m.minimize(c[0] * x * x)
    known, is_convex, _mask = _classify_model_convexity(m)
    assert not (known and is_convex)


def test_a_rebound_array_parameter_is_reclassified_not_remembered():
    """Flipping the sign of the element flips the verdict AND the answer.

    The convexity memo is per-solve, so a value change must be re-classified. A
    stale ``convex`` verdict here would send a concave objective down the
    single-NLP route and report a local point as a proven global optimum — the
    #742 failure class, in the shape this fix could introduce.
    """
    m = dm.Model("rebound_array_param")
    x = m.continuous("x", lb=-1, ub=1)
    c = m.parameter("c", np.array([1.0]))
    m.minimize(c[0] * x * x + x)

    expected = {1.0: (-0.25, -0.5), -1.0: (-2.0, -1.0)}
    checked = 0
    for value in (1.0, -1.0, 1.0, -1.0):
        c.value = np.array([value])
        r = m.solve(time_limit=30)
        want_obj, want_x = expected[value]
        assert r.status == "optimal", (value, r.status)
        assert r.objective == pytest.approx(want_obj, abs=1e-5), (value, r.objective)
        assert float(np.ravel(r.x["x"])[0]) == pytest.approx(want_x, abs=1e-4), value
        assert r.bound is not None and r.bound <= r.objective + 1e-6, (value, r.bound)
        checked += 1
    assert checked == 4


def test_an_indexed_variable_still_blocks_the_product_rule():
    """The resolver must not mistake an indexed VARIABLE for a constant — that
    would turn a bilinear product into a proof of affineness."""
    m = dm.Model("indexed_variable")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    assert classify_expr(x[0] * x[1], m) == Curvature.UNKNOWN


def test_a_non_finite_indexed_coefficient_is_refused():
    m = dm.Model("nonfinite")
    x = m.continuous("x", lb=0, ub=1)
    c = m.parameter("c", np.array([np.nan, np.inf]))
    assert classify_expr(c[0] * (x * x), m) == Curvature.UNKNOWN
    assert classify_expr(c[1] * (x * x), m) == Curvature.UNKNOWN
