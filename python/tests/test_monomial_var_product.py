"""Regression test for *mixed repeated-factor* product support (x**2 * y).

The term classifier dumps a product like ``x*x*y`` into ``general_nl`` and the
MILP linearizer raised ``"Mixed repeated-factor products are not supported"``,
so any model with a ``monomial * variable`` term fell back to a feasibility
objective and produced no dual bound (e.g. ``st_e38`` finished ``feasible`` with
``bound=None``).

:func:`_decompose_product` now collapses the repeated original-variable group
into its monomial aux column, leaving a lifted bilinear pair
``[col(x**2), y]``. The standard McCormick pipeline relaxes that with one
monomial envelope plus one bilinear envelope — a rigorous outer approximation —
so the term lifts cleanly and the model certifies.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_DATA = Path(__file__).parent / "data" / "minlplib"

# (instance, optimum) — models with monomial*variable terms that previously
# produced no certifying bound and now certify. Optima agree with MINLPLib.
_CERTIFY_CASES = [
    ("st_e38", 7197.727),
    ("ex1226", -17.0),
]


@pytest.mark.correctness
@pytest.mark.parametrize("instance, optimum", _CERTIFY_CASES)
def test_monomial_var_product_certifies(instance, optimum):
    """Each monomial*variable model reaches its optimum with a sound bound."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=60, gap_tolerance=1e-4, max_nodes=200_000)

    assert r.objective is not None
    assert abs(r.objective - optimum) <= 1e-2, f"[{instance}] obj={r.objective} != {optimum}"
    assert r.bound is not None, f"[{instance}] no dual bound produced"
    # Soundness: a valid lower bound never exceeds the incumbent.
    assert r.bound <= r.objective + 1e-3, f"[{instance}] unsound bound {r.bound} > {r.objective}"
    assert r.gap_certified, f"[{instance}] expected certified optimality"


@pytest.mark.correctness
def test_decompose_collapses_repeated_factor_to_monomial_col():
    """x*x*y decomposes to [monomial_aux(x**2), y], not a raw [x, x, y]."""
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import _decompose_product
    from discopt._jax.term_classifier import classify_nonlinear_terms

    m = dm.Model("xxy")
    x = m.continuous("x", lb=1.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    m.minimize(x * x * y)

    # Build the monomial aux map the relaxation would construct for x**2.
    classify_nonlinear_terms(m)
    DiscretizationState(partitions={})
    monomial_var_map = {(0, 2): 2}  # col 2 stands in for x**2 (cols 0,1 are x,y)
    expr = m._objective.expression
    decomp = _decompose_product(expr, m, monomial_var_map=monomial_var_map)
    assert decomp is not None
    _scalar, indices = decomp
    # x*x folds to the monomial column (2); y (index 1) stays. No bare repeat.
    assert sorted(indices) == [1, 2], f"expected [1, 2], got {indices}"


@pytest.mark.correctness
def test_monomial_var_relaxation_is_a_valid_lower_bound():
    """The lifted x**2*y relaxation never overestimates the true minimum."""
    # min x**2 * y over a positive box has minimum at the lower corner.
    m = dm.Model("xxy_bound")
    x = m.continuous("x", lb=1.0, ub=4.0)
    y = m.continuous("y", lb=2.0, ub=5.0)
    m.minimize(x * x * y)
    r = m.solve(time_limit=30, gap_tolerance=1e-4)
    assert r.objective is not None
    true_min = 1.0 * 1.0 * 2.0  # x=1, y=2
    assert r.objective == pytest.approx(true_min, abs=1e-3)
    if r.bound is not None:
        assert r.bound <= true_min + 1e-6, f"unsound bound {r.bound} > {true_min}"
