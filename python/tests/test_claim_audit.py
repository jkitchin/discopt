"""Claim-audit instrumentation tests (issue #632, R0.3 + R0.4).

Proves the audit machinery is (a) deterministic, (b) read-only — an audited build
is byte-identical to an un-audited one — and (c) that the defer-fire counter is a
genuine no-op unless a ``defer_audit`` context is active.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from discopt._jax.claim_audit import (
    AuditReport,
    audit_build,
    defer_audit,
    fingerprint_model,
    note_defer,
    relaxation_fingerprint,
)
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import from_nl

pytestmark = [pytest.mark.claim_boundary]

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
# A few instances spanning owner families: nvs09 (univariate + monomial),
# a pure-linear-ish and a product instance.
_PROBES = ["nvs09", "nvs01", "ex1221"]


def _available(name: str) -> bool:
    return (_NL_DIR / f"{name}.nl").exists()


def _fingerprint(name: str) -> str:
    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    terms = classify_nonlinear_terms(model)
    relax, _info = build_milp_relaxation(model, terms, DiscretizationState())
    return relaxation_fingerprint(relax)


@pytest.mark.parametrize("name", _PROBES)
def test_fingerprint_is_deterministic(name):
    if not _available(name):
        pytest.skip(f"{name}.nl not vendored")
    assert _fingerprint(name) == _fingerprint(name)


@pytest.mark.parametrize("name", _PROBES)
def test_audit_build_is_read_only(name):
    """An audited build must be byte-identical to a plain build (the R0.4 gate:
    the auditor changes no solver math)."""
    if not _available(name):
        pytest.skip(f"{name}.nl not vendored")
    plain = _fingerprint(name)
    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    report = audit_build(model)
    assert isinstance(report, AuditReport)
    assert report.fingerprint == plain
    # And a second convenience path agrees.
    assert fingerprint_model(from_nl(str(_NL_DIR / f"{name}.nl"))) == plain


@pytest.mark.parametrize("name", _PROBES)
def test_exactly_one_owner_per_column(name):
    """No aux column is claimed by two owner families (the exactly-one-owner
    invariant the cutover must preserve; today's federation already satisfies it
    on these probes)."""
    if not _available(name):
        pytest.skip(f"{name}.nl not vendored")
    report = audit_build(from_nl(str(_NL_DIR / f"{name}.nl")))
    assert report.conflicts == {}, f"columns double-claimed: {report.conflicts}"


def test_nvs09_ownership_has_univariate_family():
    """nvs09's fractional-power objective lifts univariate columns — the auditor
    must see the univariate owner family populated."""
    if not _available("nvs09"):
        pytest.skip("nvs09.nl not vendored")
    report = audit_build(from_nl(str(_NL_DIR / "nvs09.nl")))
    assert report.n_claims > 0
    assert "univariate_relaxations" in report.owners()


def test_defer_counter_is_noop_outside_context():
    """note_defer outside a defer_audit context must do nothing and not raise."""
    note_defer("some_site")  # must be a silent no-op


def test_defer_counter_counts_inside_context():
    with defer_audit() as fires:
        note_defer("a")
        note_defer("a")
        note_defer("b")
    assert dict(fires) == {"a": 2, "b": 1}


def test_defer_counter_resets_between_contexts():
    with defer_audit() as first:
        note_defer("x")
    with defer_audit() as second:
        note_defer("y")
    assert dict(first) == {"x": 1}
    assert dict(second) == {"y": 1}
    # After both contexts, note_defer is inert again.
    note_defer("z")
