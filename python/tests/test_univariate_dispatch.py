"""Univariate dominance-dispatch decision tests (issue #632, R1.2).

``_univariate_dispatch_owner`` is the single decision point that replaces the
scattered ``_should_claim_composite(allow_general=…)`` + finite-domain-table
defers inside the composite-univariate collector. These tests pin the tightness
dominance order (table > exact > composed) on synthetic single-variable
atoms, and check the function is well-defined (never raises) over the whole
vendored corpus — the de-risking before it is wired into the build.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt._jax.milp_relaxation import (
    _UNI_OWNER_COMPOSED,
    _UNI_OWNER_EXACT,
    _UNI_OWNER_NONE,
    _UNI_OWNER_TABLE,
    _build_convexity_box,
    _flat_variable_types,
    _univariate_dispatch_owner,
)
from discopt._jax.model_utils import flat_variable_bounds
from discopt.modeling.core import from_nl

pytestmark = [pytest.mark.claim_boundary]

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def _owner(model, expr):
    n_orig = sum(v.size for v in model._variables)
    flat_lb, flat_ub = flat_variable_bounds(model)
    flat_types = _flat_variable_types(model)
    box = _build_convexity_box(model, flat_lb, flat_ub)
    return _univariate_dispatch_owner(expr, model, n_orig, flat_lb, flat_ub, flat_types, box)


def test_rule2_convex_composite_is_exact():
    # sqrt(x**2 + 1) is convex in x (non-affine arg) -> exact envelope owner.
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=5.0)
    e = dm.sqrt(x**2 + 1)
    m.minimize(e)
    assert _owner(m, e) == _UNI_OWNER_EXACT


def test_rule3_nonconvex_single_var_composite_is_composed():
    # (ln(x-2))**2 over a finite box is neither convex nor concave. The former
    # 1-D hull owner (H-UNI) is deleted; it now falls back to the composed path
    # (non-convex composites are recovered through the factorable AVM, #632).
    m = dm.Model()
    x = m.continuous("x", lb=2.1, ub=9.0)
    e = dm.log(x - 2) ** 2
    m.minimize(e)
    assert _owner(m, e) == _UNI_OWNER_COMPOSED


def test_rule1_integer_trig_square_is_table():
    # sin(x)**2 over a small integer domain -> exact finite-domain table.
    m = dm.Model()
    x = m.integer("x", lb=0, ub=5)
    e = dm.sin(x) ** 2
    m.minimize(e)
    assert _owner(m, e) == _UNI_OWNER_TABLE


def test_unbounded_box_is_composed():
    # A genuine composite over an effectively-unbounded box falls back to composed.
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1e20)
    e = dm.log(x + 3) ** 2
    m.minimize(e)
    assert _owner(m, e) == _UNI_OWNER_COMPOSED


def test_bare_monomial_owned_elsewhere_is_none():
    # x**2 is a bare monomial (monomial lift owns it) -> not a composite atom here.
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=3.0)
    e = x**2
    m.minimize(e)
    assert _owner(m, e) == _UNI_OWNER_NONE


def test_affine_of_var_call_owned_elsewhere_is_none():
    # exp(2*x - 1): a univariate call of an AFFINE arg is owned by the
    # univariate-of-affine collector, not the composite path.
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    e = dm.exp(2 * x - 1)
    m.minimize(e)
    assert _owner(m, e) == _UNI_OWNER_NONE


_VALID_OWNERS = {
    _UNI_OWNER_NONE,
    _UNI_OWNER_TABLE,
    _UNI_OWNER_EXACT,
    _UNI_OWNER_COMPOSED,
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(p.stem for p in _NL_DIR.glob("*.nl")))
def test_dispatch_is_well_defined_over_corpus(name):
    """The dispatch must return a valid owner (never raise) for every nonlinear
    node in every corpus instance."""
    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    n_orig = sum(v.size for v in model._variables)
    flat_lb, flat_ub = flat_variable_bounds(model)
    flat_types = _flat_variable_types(model)
    box = _build_convexity_box(model, flat_lb, flat_ub)
    from discopt.modeling.core import BinaryOp, FunctionCall, IndexExpression, UnaryOp

    seen = 0

    def visit(e):
        nonlocal seen
        owner = _univariate_dispatch_owner(e, model, n_orig, flat_lb, flat_ub, flat_types, box)
        assert owner in _VALID_OWNERS
        seen += 1
        if isinstance(e, BinaryOp):
            visit(e.left)
            visit(e.right)
        elif isinstance(e, UnaryOp):
            visit(e.operand)
        elif isinstance(e, FunctionCall):
            for a in e.args:
                visit(a)
        elif isinstance(e, IndexExpression) and not hasattr(e.base, "var_type"):
            visit(e.base)

    if model._objective is not None:
        visit(model._objective.expression)
    for c in model._constraints:
        visit(c.body)
    assert seen > 0
