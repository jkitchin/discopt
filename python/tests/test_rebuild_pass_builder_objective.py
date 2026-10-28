"""Model-rebuilding reformulation passes dropped the fast-API objective's builder block.

``add_linear_objective`` / ``add_quadratic_objective`` keep the objective in the Rust
builder and flag ``_objective`` as a placeholder that ``_objective_expression()``
rebuilds from ``_builder_linear_objective`` / ``_builder_quadratic_objective``. The
GDP lowering re-issues that block on its rebuilt model (#1798), but five other
passes that rebuild a model -- the factorable reformulation, the entropy
canonicalization, the multilinear split, the nonsmooth epigraph lift and the
integer-product expansion -- copied the placeholder ``_objective`` without the
block. The rebuilt model then raised "objective is a builder placeholder but the
model retains no builder objective block to rebuild it from", so ``solve()`` of a
fast-API-objective model with a nonlinear row failed (found by the
``modeling_guide`` notebook). Where a pass rewrote the objective with
``copy.copy`` the copy also kept the placeholder flag, so the rewritten
expression was discarded in favour of a block the new model does not hold.

After: each pass re-issues the builder objective when it carried the placeholder
unchanged, and a rewritten objective is an ordinary one.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._point_eval import evaluate_at
from discopt._relax.factorable_reform import canonicalize_entropy, factorable_reformulate
from discopt._relax.integer_product_reform import expand_integer_products
from discopt._relax.multilinear_split import split_multilinear_products
from discopt._relax.nonsmooth_lift import lift_nonsmooth_atoms

pytestmark = pytest.mark.smoke

C = np.array([1.0, 2.0, 3.0])


def _check(src, dst, x_src):
    """``dst`` must rebuild its objective. Carried unchanged, it must still be the
    builder objective and equal ``src``'s at ``x_src``; rewritten by the pass (it
    then reads aux columns), it must be an ordinary objective holding the rewrite."""
    assert dst is not src, "the pass declined; the test exercises nothing"
    expr = dst._objective_expression()  # raised before the fix
    assert dst._objective.sense == src._objective.sense
    if not getattr(dst._objective, "_is_placeholder", False):
        assert expr is dst._objective.expression
        return
    n_extra = sum(v.size for v in dst._variables) - x_src.size
    want = float(np.asarray(evaluate_at(src._objective_expression(), src, x_src)))
    got = float(np.asarray(evaluate_at(expr, dst, np.concatenate([x_src, np.zeros(n_extra)]))))
    assert got == pytest.approx(want)


def _linear(m, x):
    m.add_linear_objective(C, x, constant=0.5)


def _quadratic(m, x):
    m.add_quadratic_objective(sp.eye(3, format="csr"), C, x, constant=0.5)


ADDERS = [_linear, _quadratic]


@pytest.mark.parametrize("add_obj", ADDERS, ids=["linear", "quadratic"])
def test_factorable_reformulate_carries_builder_objective(add_obj):
    m = dm.Model("f")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=4)
    m.subject_to((3.0 * x[0] + 5.0 * x[1]) / (x[0] + x[1]) >= 4.0)  # cleared
    add_obj(m, x)
    _check(m, factorable_reformulate(m), np.array([1.0, 2.0, 0.5]))


@pytest.mark.parametrize("add_obj", ADDERS, ids=["linear", "quadratic"])
def test_entropy_canonicalization_carries_builder_objective(add_obj):
    m = dm.Model("e")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=4)
    m.subject_to(x[0] * dm.log(x[0]) <= 2.0)  # becomes ``entropy(x0)``
    add_obj(m, x)
    _check(m, canonicalize_entropy(m), np.array([1.0, 2.0, 0.5]))


@pytest.mark.parametrize("add_obj", ADDERS, ids=["linear", "quadratic"])
def test_multilinear_split_carries_builder_objective(add_obj):
    m = dm.Model("m")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=4)
    m.subject_to(x[0] * x[1] * x[2] <= 5.0)
    add_obj(m, x)
    out = split_multilinear_products(m, np.full(3, 0.1), np.full(3, 4.0))
    assert out is not None, "the pass declined; the test exercises nothing"
    _check(m, out[0], np.array([1.0, 2.0, 0.5]))


@pytest.mark.parametrize("add_obj", ADDERS, ids=["linear", "quadratic"])
def test_nonsmooth_lift_carries_builder_objective(add_obj):
    m = dm.Model("n")
    x = m.continuous("x", shape=(3,), lb=-4, ub=4)
    m.subject_to(dm.abs(x[0] - x[1]) <= 1.0)
    add_obj(m, x)
    _check(m, lift_nonsmooth_atoms(m), np.array([1.0, 2.0, 0.5]))


@pytest.mark.parametrize("add_obj", ADDERS, ids=["linear", "quadratic"])
def test_integer_product_expansion_carries_builder_objective(add_obj):
    m = dm.Model("i")
    x = m.integer("x", shape=(3,), lb=0, ub=3)
    m.subject_to(x[0] * x[1] <= 4)
    add_obj(m, x)
    _check(m, expand_integer_products(m), np.array([1.0, 2.0, 0.0]))


def test_solve_fast_api_objective_with_nonlinear_rows():
    """The ``modeling_guide`` hybrid model in miniature: a builder row, a builder
    objective, a ratio row and an ``exp`` row. ``solve()`` raised the placeholder
    ``ValueError``.

    ``exp(-x2/10) <= 0.9`` gives ``x2 >= -10 ln 0.9``; the ratio row is
    ``x1 >= x0 + 0.04``; ``x0 + x1 + x2 >= 2`` then binds with x2 at its floor, so
    ``x0 = (2 - x2 - 0.04)/2`` and ``x1 = x0 + 0.04``."""
    m = dm.Model("hybrid")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.add_linear_constraints(np.ones((1, 3)), x, ">=", np.array([2.0]))
    m.add_linear_objective(C, x)
    m.subject_to((3.0 * x[0] + 5.0 * x[1]) / (x[0] + x[1] + 0.01) >= 4.0)
    m.subject_to(dm.exp(-x[2] / 10) <= 0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=30)
    x2 = -10 * np.log(0.9)
    x0 = (2 - x2 - 0.04) / 2
    assert r.status == "optimal"
    assert r.objective == pytest.approx(x0 + 2 * (x0 + 0.04) + 3 * x2, rel=1e-4)
