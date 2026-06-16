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

import math
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


@pytest.mark.correctness
def test_nvs22_objective_term_lifts_to_sound_root_bound():
    """nvs22's objective ``1.10471*x2**2*x3 + ...`` has a repeated-factor product
    (``x2**2 * x3``) that the term classifier punts to ``general_nl`` without
    recording the constituent monomial. The relaxation builder now also collects
    monomial sub-terms from the *regular* objective, so ``x2**2`` is lifted and the
    product relaxes via one monomial + one lifted-bilinear envelope. Previously the
    objective dropped (``objective_bound_valid=False``) and the McCormick LP root
    relaxation returned no bound (issue #139, bucket 2).

    nvs22 also carries free auxiliary variables (x4-x7, defined only by omitted
    division/sqrt constraints). The bilinear envelope is skipped for factors that
    lack finite bounds, so the LP no longer errors on infinite McCormick
    coefficients. The resulting root bound must be finite and sound (≤ the known
    MINLPLib optimum 6.0584); full certification is out of scope here because the
    division/sqrt constraints remain un-linearizable.
    """
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt._jax.model_utils import flat_variable_bounds

    nl = _DATA / "nvs22.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal", f"root LP status {res.status}"
    assert res.lower_bound is not None, "objective dropped — no root bound produced"
    assert math.isfinite(res.lower_bound), f"non-finite root bound {res.lower_bound}"
    # Soundness: a valid lower bound never exceeds the true optimum.
    assert res.lower_bound <= 6.0584 + 1e-4, f"unsound bound {res.lower_bound} > 6.0584"


@pytest.mark.correctness
def test_repeated_factor_objective_with_pinned_node_still_linearizes():
    """A trilinear objective term whose factor is pinned (lb==ub) at a node must
    still linearize. ``_linearize_expr`` folds the pinned factor into the
    coefficient, collapsing ``x*y*z`` to a bilinear in the two live factors; the
    builder pre-allocates that collapsed bilinear's aux + envelope so the objective
    does not drop with "Bilinear (i,j) not in map" (issue #139)."""
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.term_classifier import classify_nonlinear_terms

    m = dm.Model("xyz_pinned")
    x = m.continuous("x", lb=1.0, ub=4.0)
    y = m.continuous("y", lb=1.0, ub=3.0)
    z = m.continuous("z", lb=1.0, ub=2.0)
    m.minimize(x * y * z)

    terms = classify_nonlinear_terms(m)
    assert (0, 1, 2) in terms.trilinear
    lb, ub = flat_variable_bounds(m)
    # Pin x (column 0) to its midpoint, as branching/OBBT would at a node.
    pinned_lb, pinned_ub = lb.copy(), ub.copy()
    pinned_lb[0] = pinned_ub[0] = 2.5

    milp, _ = build_milp_relaxation(
        m, terms, DiscretizationState(partitions={}), bound_override=(pinned_lb, pinned_ub)
    )
    # Objective must still bound (no drop), and the relaxation must underestimate
    # the pinned minimum x*y*z = 2.5*1*1 = 2.5.
    assert milp._objective_bound_valid, "objective dropped at pinned node"
    res = milp.solve()
    assert res.bound is not None and math.isfinite(res.bound)
    assert res.bound <= 2.5 + 1e-6, f"unsound bound {res.bound} > 2.5"


@pytest.mark.correctness
def test_lifted_trilinear_with_monomial_factor_certifies_soundly():
    """A product whose repeated factor collapses to a lifted aux column —
    ``x**2 * y * z`` becomes the three-distinct-column term ``[col(x**2), y, z]``
    — was never recorded in ``terms.trilinear`` (the classifier dumps it into
    ``general_nl``), so the linearizer raised "Trilinear (i,j,k) not in map" and
    the objective dropped (issue #139, bucket 2).

    The builder now collects lifted trilinear/multilinear products and allocates
    their recursive bilinear chain (one monomial envelope + two McCormick
    envelopes), so the term lifts and the relaxation is a sound underestimator.
    Over the positive box the minimum sits at the lower corner x=y=z=1."""
    m = dm.Model("x2yz")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=1.0, ub=3.0)
    z = m.continuous("z", lb=1.0, ub=2.0)
    m.minimize((x**2) * y * z)

    r = m.solve(time_limit=30, gap_tolerance=1e-4)
    assert r.objective is not None
    true_min = 1.0**2 * 1.0 * 1.0  # x=y=z=1
    assert r.objective == pytest.approx(true_min, abs=1e-3)
    assert r.bound is not None, "objective dropped — no dual bound"
    assert r.bound <= true_min + 1e-6, f"unsound bound {r.bound} > {true_min}"


@pytest.mark.correctness
def test_lifted_trilinear_sound_over_mixed_sign_box():
    """Soundness of the lifted ``x**2 * y * z`` envelope when factors straddle
    zero: the true minimum is the negative corner (x**2 large, y negative).
    The recursive McCormick chain must never overestimate it (issue #139)."""
    m = dm.Model("x2yz_neg")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=3.0)
    z = m.continuous("z", lb=1.0, ub=2.0)
    m.minimize((x**2) * y * z)

    r = m.solve(time_limit=30, gap_tolerance=1e-4)
    assert r.objective is not None
    true_min = (2.0**2) * (-2.0) * 2.0  # x=2 (x**2=4), y=-2, z=2  →  -16
    assert r.objective == pytest.approx(true_min, abs=1e-2)
    assert r.bound is not None, "objective dropped — no dual bound"
    assert r.bound <= true_min + 1e-4, f"unsound bound {r.bound} > {true_min}"
