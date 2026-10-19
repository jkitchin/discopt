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

METHODS = ["big-m", "mbigm", "hull", "auto", "simplex", "loa"]


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
    """Repro B: ``min x0`` exists only in the builder; optimum 1."""
    m = dm.Model("b")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.either_or([[x[0] >= 1], [x[0] >= 3]])
    m.add_linear_objective(np.array([1.0, 0.0]), x)
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
        return m.solve(gdp_method=method, time_limit=60)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "build, optimum",
    [
        (_rows_model, 6.0),
        (_objective_model, 1.0),
        (_quadratic_objective_model, 1.0),
        (_rule_rows_model, 7.0),
    ],
    ids=["rows", "objective", "quadratic_objective", "rule_rows"],
)
def test_fast_api_model_certifies_its_optimum(build, optimum, method):
    if method in ("simplex", "loa") and build is _quadratic_objective_model:
        pytest.skip("quadratic objective: covered by the B&B methods")
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
