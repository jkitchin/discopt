"""cert:LR-2 (Path B) — alias-shape guard: the reform must keep emitting the
``aux == h(x)`` equality that H-UNI binds to.

Path B's H-UNI/H-LOG lifts are *additive* recognizers keyed on a specific piece of
the AMP reform's output: ``factorable_reformulate`` splits a single-variable
composite like nvs09's ``(ln(x-2))**2`` into

    aux == ln(x-2)          # an equality constraint  ``Variable(aux) - h == 0``
    ... + aux**2            # a monomial in the objective

``_alias_equality_defs`` recovers the ``aux -> h`` map from those equalities and
``_collect_aliased_monomial_hull_relaxations`` binds the ``aux**p`` monomial column
to the exact 1-D hull of ``h(x)**p``. If a future reform change stops emitting that
``aux == h(x)`` equality (or stops splitting the composite into an ``aux**p``
monomial), the recognizer silently finds nothing and the H-UNI flag goes *inert* —
no error, just a lost lever and a re-opened nvs09 dual gap.

These tests pin the contract so such a reform change trips a **loud** failure here
instead of going silently inert (CLAUDE.md §3 — no dead flags). They assert the
recognized *shape*, not any instance-specific numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from discopt._jax import milp_relaxation as mr
from discopt._jax.factorable_reform import factorable_reformulate
from discopt.modeling.core import from_nl

pytestmark = [pytest.mark.correctness, pytest.mark.claim_boundary]

_NL = Path(__file__).parent / "data" / "minlplib_nl" / "nvs09.nl"


def _reformed_nvs09():
    return factorable_reformulate(from_nl(str(_NL)))


def test_reform_emits_alias_equalities():
    """The reform must still emit ``aux == h(x)`` equalities the H-UNI recognizer
    reads. Empty here means a reform change silently disabled the lever."""
    rm = _reformed_nvs09()
    defs = mr._alias_equality_defs(rm)
    assert defs, (
        "reform emitted no aux==h(x) alias equalities — H-UNI recognizer would go "
        "silently inert; a reform change broke the contract H-UNI depends on"
    )


def test_alias_bodies_are_nonlinear_single_var_composites():
    """At least one recovered alias ``h`` must be a nonlinear function of a single
    ORIGINAL variable — exactly the shape H-UNI binds (``ln(x-2)`` / ``ln(10-x)``
    in nvs09). Guards against the reform aliasing something H-UNI cannot use."""
    rm = _reformed_nvs09()
    defs = mr._alias_equality_defs(rm)
    n_orig = len(rm._variables)
    single_var_composites = [
        h
        for aux_idx, h in defs.items()
        if (ref := mr._composite_referenced_var(h, rm)) is not None and ref[1] < n_orig
    ]
    assert single_var_composites, (
        "no alias body is a single-original-variable nonlinear composite — the "
        "H-UNI reform-split contract (aux == h(x), h single-var) is broken"
    )


def test_huni_collector_binds_when_flag_on(monkeypatch):
    """End-to-end contract: with the flag ON the aliased-hull collector must bind
    at least one ``aux**p`` monomial. Zero rows here means the reform split, the
    alias equalities, and the monomial map no longer line up — the exact silent
    failure this guard exists to catch."""
    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "1")
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    rm = _reformed_nvs09()
    terms = classify_nonlinear_terms(rm)
    flat_lb, flat_ub = mr.flat_variable_bounds(rm)

    # The builder exposes the assembled ``monomial_var_map`` under varmap["monomial"]
    # ({(var_idx, n): col}). Feed it back to the collector so the assertion is about
    # the recognizer's contract, not a full solve.
    _relax, info = build_milp_relaxation(rm, terms, DiscretizationState())
    monomial_var_map = info["monomial"]
    assert monomial_var_map, "reform produced no aux**n monomial columns for nvs09"
    rows = mr._collect_aliased_monomial_hull_relaxations(
        rm, len(rm._variables), flat_lb, flat_ub, monomial_var_map
    )
    assert rows, (
        "H-UNI collector produced no aliased-monomial hull rows on nvs09 with the "
        "flag ON — the reform no longer splits (ln(x-2))**2 into aux==ln(x-2) + "
        "aux**2, so the lever is inert (would silently re-open nvs09's dual gap)"
    )
