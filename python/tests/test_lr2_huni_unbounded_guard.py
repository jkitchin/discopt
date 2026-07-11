"""cert:LR-3 — H-UNI soundness/quality guards, exercised with the flag ON.

The default-OFF tests elsewhere run the flag off, so they cannot catch a
regression in the H-UNI (``DISCOPT_UNIVARIATE_ENVELOPE``) path. These tests set
the flag ON explicitly and assert the three behaviours the LR-3 fix established:

1. **No false-infeasible on unbounded variables.** H-UNI must not build a 1-D
   hull over an *effectively unbounded* box (the ~±1e20 default of an unbounded
   variable); the sampled hull is numerically meaningless (aux magnitudes reach
   ~1e60) and drives the LP to a false ``infeasible``. The guard uses the
   solver-sense ``is_effectively_finite`` (|bound| < 1e19), not raw ``np.isfinite``.
2. **Defer to the exact finite-domain trig-square table.** ``sin/cos(int x)**2``
   (and its additive ``... + c`` form) is covered exactly by the table; H-UNI's
   continuous hull is looser, so it must defer.
3. **No redundant sub-expression composite.** H-UNI must not re-claim an inner
   composite of an already-claimed single-variable composite.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.modeling.core import Model

pytestmark = [pytest.mark.correctness]


def _build(model):
    from discopt._jax.discretization import initialize_partitions
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    terms = classify_nonlinear_terms(model)
    state = initialize_partitions([], lb=[], ub=[], n_init=2)
    return build_milp_relaxation(model, terms, state, incumbent=None)


def test_huni_unbounded_variable_not_false_infeasible(monkeypatch):
    """Unbounded x + a min/max of polynomial branches: H-UNI must abstain on the
    effectively-unbounded box and fall back to a sound relaxation (optimal), not
    certify a false ``infeasible``."""
    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "1")
    m = Model("huni_unbounded")
    x = m.continuous("x")  # unbounded → ~±1e20 default box
    m.maximize(dm.minimum(0.75 + (x - 0.5) ** 3, 0.75 - (x - 0.5) ** 2))
    milp_model, _ = _build(m)
    result = milp_model.solve()
    assert result.status != "infeasible", (
        "H-UNI built a hull over an unbounded box → false infeasible"
    )
    assert result.status == "optimal"


def test_huni_bounded_variable_still_fires(monkeypatch):
    """The guard is targeted: on a *finite* box H-UNI still engages soundly."""
    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "1")
    m = Model("huni_bounded")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.maximize(dm.minimum(0.75 + (x - 0.5) ** 3, 0.75 - (x - 0.5) ** 2))
    milp_model, _ = _build(m)
    assert milp_model.solve().status == "optimal"


def test_huni_defers_to_finite_domain_trig_table(monkeypatch):
    """sin(int x)**2 in a constraint RHS must go to the exact table, not H-UNI's
    looser hull: the relaxation bound must match the flag-OFF (exact-table) value."""
    import numpy as np

    def build_sin():
        m = Model("huni_trig")
        x = m.integer("x", lb=0, ub=4)
        y = m.continuous("y", lb=0, ub=4)
        m.maximize(10 * x + y)
        m.subject_to(y <= dm.sin(x) ** 2 + 2)
        return _build(m)[0].solve()

    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "0")
    off = build_sin()
    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "1")
    on = build_sin()
    exact = -(40.0 + np.sin(4.0) ** 2 + 2.0)
    assert on.status == "optimal" and off.status == "optimal"
    # ON must not be looser than the exact table (which OFF uses).
    assert on.objective == pytest.approx(off.objective, abs=1e-8)
    assert on.objective == pytest.approx(exact, abs=1e-8)


def test_huni_no_redundant_subexpression_composite(monkeypatch):
    """sqrt(x**2 + 1): the convex composite is claimed once; H-UNI must not also
    claim the inner x**2 + 1 (a redundant duplicate column)."""
    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "1")
    m = Model("huni_nested")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(dm.sqrt(x**2 + 1.0))
    milp_model, varmap = _build(m)
    result = milp_model.solve()
    assert len(varmap["composite_relaxations"]) == 1
    assert result.status == "optimal"
    assert result.bound is not None and result.bound <= 1.0 + 1e-6
