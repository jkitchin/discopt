"""Regression test for the spatial-relaxation *pinned-variable* fold.

The fractional-power / monomial aux-column builders intentionally skip a
variable whose domain is degenerate (``ub - lb`` within tolerance). Spatial
branching routinely drives a domain to a point, so at those nodes a term like
``x**0.6`` or ``x**2`` has no aux column. Previously the linearizer raised
``"Fractional power (i, p) has no aux column"`` / ``"Monomial ... not in map"``
there, which sank the node's *objective* to a feasibility-only relaxation and
silently dropped its dual bound — so the global lower bound stalled and the
model never certified (e.g. ``st_e11`` finished ``feasible`` with ``bound=None``
despite the root LP producing a perfectly valid bound).

A pinned variable has an exact value, so ``x**p`` is the constant ``v**p`` and a
partially-pinned product collapses to a lower-order term. Folding pinned factors
into constants is rigorous (no relaxation involved) and restores the node bound.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.term_classifier import classify_nonlinear_terms

_DATA = Path(__file__).parent / "data" / "minlplib"


@pytest.mark.correctness
def test_pinned_fractional_power_keeps_objective_bound():
    """Pinning a fractional-power base must not collapse the objective bound."""
    m = dm.from_nl(str(_DATA / "st_e11.nl"))
    terms = classify_nonlinear_terms(m)
    disc = DiscretizationState(partitions={})
    lb, ub = flat_variable_bounds(m)

    # Root box: objective linearizes (baseline).
    milp_root, _ = build_milp_relaxation(m, terms, disc, bound_override=(lb.copy(), ub.copy()))
    assert milp_root._objective_bound_valid

    # Pin each fractional-power base in turn (lb == ub) and confirm the
    # objective still linearizes (its term folds to an exact constant) rather
    # than degrading to the feasibility fallback.
    for var_idx, _p in terms.fractional_power:
        ub_pin = ub.copy()
        ub_pin[var_idx] = lb[var_idx]
        milp, _ = build_milp_relaxation(m, terms, disc, bound_override=(lb.copy(), ub_pin))
        assert milp._objective_bound_valid, f"pinning x{var_idx} dropped the objective bound"


@pytest.mark.correctness
def test_st_e11_certifies_with_sound_bound():
    """st_e11 must reach its optimum and certify with a valid dual bound."""
    r = dm.from_nl(str(_DATA / "st_e11.nl")).solve(time_limit=60, gap_tolerance=1e-4)

    assert r.objective is not None
    assert r.bound is not None, "no dual bound produced (pinned-variable fold)"
    # Soundness: a valid lower bound never exceeds the incumbent.
    assert r.bound <= r.objective + 1e-3, f"unsound bound {r.bound} > obj {r.objective}"
    assert r.gap_certified, "expected certified optimality"
    # Known MINLPLib optimum.
    assert abs(r.objective - 189.31157) <= 1e-2, f"obj={r.objective}"


@pytest.mark.correctness
@pytest.mark.parametrize(
    "arity,pins",
    [
        (4, 1),  # 4-way -> trilinear  (unpinned == 3)
        (4, 2),  # 4-way -> bilinear   (unpinned == 2)
        (5, 1),  # 5-way -> 4-way      (unpinned >= 4: collapsed subset still multilinear)
        (5, 2),  # 5-way -> trilinear
        (6, 2),  # 6-way -> 4-way      (unpinned >= 4)
    ],
)
def test_multilinear_pinned_collapse_keeps_objective_bound(arity, pins):
    """A k-way product that pins factors must still linearize at that node.

    During B&B a factor of a multilinear term can pin (lb==ub), and
    ``_linearize_expr`` folds it into the coefficient — collapsing the term to a
    lower-arity product *keyed by the unpinned subset*. Unless that collapsed
    bilinear/trilinear/multilinear key is pre-registered, the linearizer raises
    ``"... not in map"`` and the whole constraint (here the objective) drops,
    sinking the node's dual bound. The collapse target follows the linearizer's
    arity dispatch exactly: 2 -> bilinear, 3 -> trilinear, >=4 -> multilinear.
    """
    m = dm.Model(f"multi{arity}")
    xs = [m.continuous(f"x{i}", lb=0.5, ub=2.0) for i in range(arity)]
    prod = xs[0]
    for x in xs[1:]:
        prod = prod * x
    m.minimize(prod)
    terms = classify_nonlinear_terms(m)
    disc = DiscretizationState(partitions={})
    lb, ub = flat_variable_bounds(m)

    # Root box: the full k-way term linearizes (baseline).
    root, _ = build_milp_relaxation(m, terms, disc, bound_override=(lb.copy(), ub.copy()))
    assert root._objective_bound_valid

    # Pin the first ``pins`` factors (lb==ub) and confirm the collapsed
    # (k-pins)-ary product still linearizes rather than dropping the objective.
    lb_pin, ub_pin = lb.copy(), ub.copy()
    for i in range(pins):
        lb_pin[i] = 1.0
        ub_pin[i] = 1.0
    milp, _ = build_milp_relaxation(m, terms, disc, bound_override=(lb_pin, ub_pin))
    assert milp._objective_bound_valid, (
        f"pinning {pins} of {arity} factors dropped the objective bound"
    )


@pytest.mark.correctness
@pytest.mark.parametrize("pin_base", [True, False])
def test_bilinear_with_pinned_fractional_power_keeps_objective_bound(pin_base):
    """``y * x**p`` must still linearize when the power base ``x`` pins.

    A variable times a fractional power (the gas-network compressor objective
    ``w * beta**0.2857``) is relaxed via the bilinear-with-fractional-power path:
    an aux ``z = x**p`` plus a McCormick envelope on ``y * z``. When ``x`` pins
    (lb==ub from branching/OBBT) the builder skips the ``z`` aux for the
    degenerate domain, so the product reaches ``_decompose_product`` with no aux
    column. Without folding the pinned power to its exact constant the product is
    undecomposable and the whole objective drops to the feasibility fallback,
    sinking the node's dual bound. Pinning the *other* factor (``y``) must
    likewise keep the term (it collapses to the pure power ``x**p``).
    """
    m = dm.Model("bilinear_fp")
    y = m.continuous("y", lb=0.0, ub=10.0)
    x = m.continuous("x", lb=1.0, ub=2.0)
    m.minimize(y * (x**0.2857))
    terms = classify_nonlinear_terms(m)
    assert terms.bilinear_with_fp, "expected a bilinear-with-fractional-power term"
    disc = DiscretizationState(partitions={})
    lb, ub = flat_variable_bounds(m)

    root, _ = build_milp_relaxation(m, terms, disc, bound_override=(lb.copy(), ub.copy()))
    assert root._objective_bound_valid

    lb_pin, ub_pin = lb.copy(), ub.copy()
    idx = 1 if pin_base else 0  # x is var 1 (power base), y is var 0
    val = 1.5 if pin_base else 4.0
    lb_pin[idx] = ub_pin[idx] = val
    milp, _ = build_milp_relaxation(m, terms, disc, bound_override=(lb_pin, ub_pin))
    which = "power base x" if pin_base else "factor y"
    assert milp._objective_bound_valid, f"pinning {which} dropped the objective bound"


@pytest.mark.correctness
def test_pinned_value_is_exact_not_relaxed():
    """A fully pinned univariate-power objective term equals the exact value."""
    # min x**0.6 with x pinned to 16 → exactly 16**0.6, no relaxation slack.
    m = dm.Model("pinned")
    x = m.continuous("x", lb=16.0, ub=16.0)
    m.minimize(x**0.6)
    terms = classify_nonlinear_terms(m)
    disc = DiscretizationState(partitions={})
    lb, ub = flat_variable_bounds(m)
    milp, _ = build_milp_relaxation(m, terms, disc, bound_override=(lb, ub))
    assert milp._objective_bound_valid
    res = milp.solve()
    assert res.objective is not None
    assert res.objective == pytest.approx(16.0**0.6, abs=1e-6)
