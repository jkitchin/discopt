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
