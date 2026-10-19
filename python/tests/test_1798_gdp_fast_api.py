"""#1798: the GDP lowering dropped fast-API rows and the fast-API objective.

``add_linear_constraints`` (and the ``Model.constraint`` linear fast path) keep their
rows only on ``_builder_linear_blocks``; ``add_linear_objective`` /
``add_quadratic_objective`` keep the objective in the Rust builder and leave a zero
placeholder in ``_objective``. ``reformulate_gdp`` rebuilt the model from
``_objective`` and ``_constraints`` alone, so both were lost:

* repro A (a fast-API row): the lowered solve reported ``optimal 1.0``; the #1561
  backstop caught the violated row and downgraded it -- ``feasible 6.0``, value right
  but the certificate gone;
* repro B (a fast-API objective): the solver minimized 0 and returned an arbitrary
  point -- ``feasible 10.0`` against the optimum 1 (hull / auto raised instead).

After: the rows are carried as ordinary constraints, and the builder objective is
re-issued on the lowered model *after* its auxiliary columns exist (re-issuing it
first registered a builder without them, and the Rust repr then refused the model:
"Variable not found in model", FBBT/presolve off). LOA's master also read the
placeholder (``_objective.expression``) and reported bound 0; it now reads
``_objective_expression()``.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp

pytestmark = pytest.mark.smoke

#: ``"oa"`` is ``solver="mip-nlp", mip_nlp_method="oa"`` over the big-M lowering
#: (see :func:`_solve`): its decomposition read the placeholder as well (#1802 B1).
METHODS = ["big-m", "mbigm", "hull", "auto", "simplex", "loa", "oa"]


def _rows_model():
    """Repro A: the row ``x0 + x1 >= 6`` exists only in the builder; optimum 6."""
    m = dm.Model("a")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([6.0]))
    m.subject_to(x[1] <= 0)
    m.either_or([[x[0] >= 1], [x[0] >= 2]])
    m.minimize(x[0])
    return m


def _objective_model():
    """Repro B: ``min x0 + 2 x1`` exists only in the builder; optimum 1 at (1, 0).

    The x1 coefficient is nonzero on purpose: with ``c = [1, 0]`` a reader that saw
    the zero placeholder could still land on the optimum by accident (#1802 S1)."""
    m = dm.Model("b")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.either_or([[x[0] >= 1], [x[0] >= 3]])
    m.add_linear_objective(np.array([1.0, 2.0]), x)
    return m


def _coupled_objective_model():
    """#1802 S1: ``min x0 + 2 x1`` (builder) with ``x0 + x1 >= 5`` and
    ``x0 <= 1 or x0 >= 3``: optimum 5 at (5, 0). ``gdp_method="simplex"`` returned
    ``feasible 5, bound 0`` here while big-M certified it."""
    m = dm.Model("c")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.subject_to(x[0] + x[1] >= 5)
    m.either_or([[x[0] <= 1], [x[0] >= 3]])
    m.add_linear_objective(np.array([1.0, 2.0]), x)
    return m


def _quadratic_objective_model():
    """``min (x0 - 2)^2 + x1`` through ``add_quadratic_objective``, with a
    disjunction that keeps x0 out of (1, 3): optimum 1 at x0 = 1 or 3."""
    import scipy.sparse as sp

    m = dm.Model("q")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.either_or([[x[0] <= 1], [x[0] >= 3]])
    Q = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 0.0]]))  # 0.5 * 2 * x0^2
    m.add_quadratic_objective(Q, np.array([-4.0, 1.0]), x, constant=4.0)
    return m


def _rule_rows_model():
    """``Model.constraint`` linear fast path: x0 >= 6 - i for i = 0 (so x0 >= 6)."""
    m = dm.Model("r")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.constraint(m.set("S", [0]), lambda i: x[0] >= 6 - i, name="fast_rows")
    m.either_or([[x[1] >= 1], [x[1] >= 2]])
    m.minimize(x[0] + x[1])
    return m


def _solve(m, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if method == "oa":
            return m.solve(solver="mip-nlp", mip_nlp_method="oa", time_limit=60)
        return m.solve(gdp_method=method, time_limit=60)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "build, optimum",
    [
        (_rows_model, 6.0),
        (_objective_model, 1.0),
        (_coupled_objective_model, 5.0),
        (_quadratic_objective_model, 1.0),
        (_rule_rows_model, 7.0),
    ],
    ids=["rows", "objective", "coupled_objective", "quadratic_objective", "rule_rows"],
)
def test_fast_api_model_certifies_its_optimum(build, optimum, method):
    r = _solve(build(), method)
    assert r.status == "optimal", (method, r.status, r.objective)
    assert r.objective == pytest.approx(optimum, rel=1e-4, abs=1e-5)


def test_the_rule_model_really_uses_the_fast_path():
    """The ``rule_rows`` case only tests #1798 if its row is builder-resident."""
    m = _rule_rows_model()
    assert m._builder_linear_blocks and not any(
        getattr(c, "name", None) == "fast_rows" for c in m._constraints
    )


@pytest.mark.parametrize("method", ["big-m", "mbigm", "hull", "simplex"])
def test_lowered_model_carries_the_rows_and_a_buildable_repr(method):
    from discopt._rust import model_to_repr

    src = _rows_model()
    low = reformulate_gdp(src, method)
    # The builder row arrives as an ordinary row of the lowered model.
    (builder_row,) = src._builder_linear_constraints()
    assert str(builder_row) in {str(c) for c in low._constraints}
    low_b = reformulate_gdp(_objective_model(), method)
    assert low_b._builder_linear_objective is not None
    # The re-issued objective's builder knows every column, auxiliaries included.
    model_to_repr(low_b, low_b._builder)


def _oa_model(fast):
    """A non-GDP MINLP for ``mip_nlp_method="oa"``: ``min x0 + 2 x1`` subject to
    ``x0 + x1 >= 4``, ``x0 >= y + 1/2``, ``y >= 1`` integer, ``x0^2 + x1^2 <= 50``;
    optimum 4 at (4, 0). ``fast`` puts the objective and the coupling row on the
    builder (``add_linear_objective`` / ``add_linear_constraints``)."""
    m = dm.Model("oa")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    y = m.integer("y", lb=0, ub=5)
    m.subject_to(x[0] >= y + 0.5)
    m.subject_to(y >= 1)
    m.subject_to(x[0] ** 2 + x[1] ** 2 <= 50)
    if fast:
        m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([4.0]))
        m.add_linear_objective(np.array([1.0, 2.0]), x)
    else:
        m.subject_to(x[0] + x[1] >= 4)
        m.minimize(x[0] + 2 * x[1])
    return m


@pytest.mark.parametrize("fast", [False, True], ids=["modeling", "fast_api"])
def test_mip_nlp_oa_certifies_a_fast_api_objective(fast):
    """#1802 B1: OA's decomposition read ``_objective.expression`` -- the zero
    placeholder of a builder objective -- so its master minimized 0 and the solve
    came back ``feasible 4.0, bound 0`` against the modeling twin's ``optimal 4``."""
    m = _oa_model(fast)
    assert (m._builder_linear_objective is not None) == fast
    r = _solve(m, "oa")
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(4.0, rel=1e-4, abs=1e-5)
    assert r.bound == pytest.approx(4.0, rel=1e-4, abs=1e-5)


def test_placeholder_objective_carries_the_real_expression():
    """#1802: the class fix. Every reader of ``model._objective.expression`` (60-odd
    of them: relaxations, convexity, OA, interval bounds, exporters) sees the builder
    objective, not a zero. The Rust repr still takes the builder's own objective
    (the ``_is_placeholder`` flag), so this does not change what it compiles."""
    import scipy.sparse as sp
    from discopt._relax.convexity.interval import Interval
    from discopt._relax.convexity.interval_eval import evaluate_interval

    m = dm.Model("p")
    x = m.continuous("x", shape=(2,), lb=1, ub=2)
    m.add_linear_objective(np.array([1.0, 2.0]), x, constant=3.0)
    assert m._objective._is_placeholder
    iv = evaluate_interval(m._objective.expression, m, {x: Interval(x.lb, x.ub)})
    assert (float(np.min(iv.lo)), float(np.max(iv.hi))) == pytest.approx((6.0, 9.0))

    q = dm.Model("q")
    z = q.continuous("z", shape=(2,), lb=-1, ub=1)
    # 0.5 z'Qz with Q = [[0, -2], [-2, 0]]: the nonconvex -2 z0 z1.
    q.add_quadratic_objective(sp.csr_matrix([[0.0, -2.0], [-2.0, 0.0]]), np.zeros(2), z)
    iv = evaluate_interval(q._objective.expression, q, {z: Interval(z.lb, z.ub)})
    assert (float(np.min(iv.lo)), float(np.max(iv.hi))) == pytest.approx((-2.0, 2.0))


# The builder objective is now an array-gather expression (#1802), which reached
# readers that had only ever seen scalar indices. Each crash below was latent for a
# *pure* modeling-API model indexed by an integer array; these pin the fixes without
# the fast API, so a reader that regresses is caught on the class, not the instance.


def test_oa_accepts_a_gathered_objective():
    """Was: perspective's Hessian probe let ``_NotLinearError`` escape the solve."""
    m = dm.Model("oa_gather")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    y = m.binary("y")
    m.subject_to(x[0] + x[1] >= 4)
    m.subject_to(x[0] <= 10 * y)
    m.minimize(dm.sum(np.array([1.0, 2.0]) * x[np.array([0, 1])]) + y)
    r = m.solve(solver="mip-nlp", mip_nlp_method="oa", time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-6)


def test_integer_product_reform_declines_an_array_coefficient():
    """Was: ``float()`` of a shape-(1,) coefficient raised a bare TypeError."""
    m = dm.Model("intprod_gather")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    y = m.integer("y", lb=0, ub=3)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(y * z >= 2)
    m.subject_to(x[0] >= y + z - 3)
    m.subject_to(x[1] >= 0.5)
    g = x[np.array([0])]
    m.minimize(dm.sum(np.array([1.0]) * g * g) + x[1])
    r = m.solve(time_limit=60)
    assert r.objective == pytest.approx(0.5, abs=1e-3)


def test_convexity_struct_eq_compares_array_indices_exactly():
    """Was: ``index == index`` returned an array, whose truth value raised."""
    from discopt._relax.convexity.patterns import _expr_struct_eq

    m = dm.Model("structeq")
    x = m.continuous("x", shape=(3,))
    assert _expr_struct_eq(x[np.array([0, 1])], x[np.array([0, 1])]) is True
    assert _expr_struct_eq(x[np.array([0, 1])], x[np.array([0, 2])]) is False
    assert _expr_struct_eq(x[np.array([0, 1])], x[1]) is False


def test_nonconvex_fast_api_quadratic_is_not_certified_from_the_placeholder():
    """A concave builder objective was certified at a non-global point.

    The convexity classifier read the zero placeholder, called the model convex,
    and a single local NLP solve's point came back ``optimal`` with its own value
    as the bound: on this instance ``optimal -8.5604`` / bound ``-8.5604`` against
    the optimum ``sum(b - 1) = -9.8024`` at ``x = 1`` -- a false certificate, not a
    slow one (measured on the #1802 review head and on ``main`` 3be568ef).
    """
    import scipy.sparse as sp
    from discopt._relax.convexity import classify_model

    n = 20
    b = np.random.default_rng(0).uniform(0.1, 0.9, n)
    m = dm.Model("concave_fast_qp")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    m.add_linear_constraints(sp.csr_matrix(np.ones((1, n))), x, ">=", np.array([0.0]))
    m.add_quadratic_objective(sp.diags(-2.0 * np.ones(n), format="csr"), b, x)
    assert classify_model(m)[0] is False
    r = m.solve(time_limit=60)
    optimum = float(np.sum(b - 1.0))
    assert r.status == "optimal"
    assert r.objective == pytest.approx(optimum, abs=1e-5)
    assert r.bound <= optimum + 1e-6
