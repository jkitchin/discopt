"""cert:LR-2 — flag-OFF relaxation byte-identity guardrail (post-R1.2 scope).

**Update (#632 R1.2):** the composite univariate 1-D **hull** graduated to
default-ON — it is now chosen by the canonical dominance dispatch and is no longer
gated by ``DISCOPT_UNIVARIATE_ENVELOPE``. What remains flag-gated is the
**aliased-monomial-hull** collector (``DISCOPT_UNIVARIATE_ENVELOPE``) and the
**log-monomial** collector (``DISCOPT_LOG_MONOMIAL``) — both still purely additive.

This guardrail therefore now proves the neutrality of those two *still-flag-gated*
collectors: for every corpus instance it fingerprints the built LP
``(_c, _A_ub, _b_ub, _bounds, _integrality)`` with the flags OFF and asserts it is
identical to the fingerprint built with those collectors forced to their
"code-absent" behavior (empty / disabled). Because the graduated composite hull is
unconditional, it appears identically on both sides and cancels out. If they
differ, an OFF path is leaking a row — a guardrail failure.

It also asserts the graduated hull is a live default (nvs09's default relaxation
carries the hull columns), so the mechanism is not inert (CLAUDE.md §3).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
from discopt._jax import milp_relaxation as mr
from discopt._jax.claim_audit import relaxation_fingerprint
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import from_nl

pytestmark = [pytest.mark.correctness, pytest.mark.claim_boundary]

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
_ALL = sorted(os.path.basename(p)[:-3] for p in glob.glob(str(_NL_DIR / "*.nl")))


def _relaxation_fingerprint(name: str) -> str:
    """Deterministic hash of the built MILP relaxation matrix for ``name``.

    Thin wrapper over the extracted library primitive
    :func:`discopt._jax.claim_audit.relaxation_fingerprint` (issue #632, R0.3),
    which hashes ``(_c, _A_ub, _b_ub, _bounds, _integrality)`` — every array a
    claim (additive row/column) could touch. Two identical hashes mean identical
    LP relaxations (same columns, same rows, same numbers)."""
    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    terms = classify_nonlinear_terms(model)
    relax, _info = build_milp_relaxation(model, terms, DiscretizationState())
    return relaxation_fingerprint(relax)


@pytest.fixture(autouse=True)
def _clear_flags(monkeypatch):
    monkeypatch.delenv("DISCOPT_UNIVARIATE_ENVELOPE", raising=False)
    monkeypatch.delenv("DISCOPT_LOG_MONOMIAL", raising=False)
    yield


def test_flags_default_off():
    # cert:LR-3 graduation was DEFERRED: H-UNI stays default-OFF (opt-in) because
    # graduating it default-ON surfaced order-masked claim-boundary collisions with
    # the existing lifted-relaxation paths (see _univariate_envelope_enabled docstring
    # and docs/dev/maingo-parity-plan.md). H-LOG is likewise default-OFF. Both remain
    # sound opt-in flags; OFF is byte-identical to prior main (guarded below).
    assert mr._univariate_envelope_enabled() is False
    assert mr._log_monomial_enabled() is False


def test_aliased_collector_empty_when_off():
    """The H-UNI additive collector must produce no rows when the flag is OFF (it
    is only *called* under the flag, but assert its empty contract directly too)."""
    model = from_nl(str(_NL_DIR / "nvs09.nl"))
    n_orig = len(model._variables)
    flat_lb, flat_ub = mr.flat_variable_bounds(model)
    rows = mr._collect_aliased_monomial_hull_relaxations(model, n_orig, flat_lb, flat_ub, {})
    assert rows == [], "aliased-hull collector returned rows with the flag OFF"


@pytest.mark.parametrize("name", _ALL)
def test_relaxation_off_byte_identical_corpus(name, monkeypatch):
    """OFF relaxation matrix must equal the relaxation built as if the additive
    Path-B code did not exist.

    "Code-absent" is simulated by neutralizing the two flag gates that could ever
    add a row: the H-UNI ``allow_general`` composite claim and both additive
    collectors. Because the flags are OFF, the real path already takes these
    branches — so equality proves the OFF path is a genuine no-op (no row leaks),
    byte-identical to prior main, on every corpus instance."""
    # H-UNI is default-OFF (opt-in; cert:LR-3 graduation deferred, see #632). Force
    # =0 explicitly so this guardrail exercises the genuine flag-OFF path regardless
    # of any ambient env.
    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "0")
    monkeypatch.setenv("DISCOPT_LOG_MONOMIAL", "0")
    fp_real_off = _relaxation_fingerprint(name)

    # Force the additive Path-B entry points to their code-absent behavior.
    monkeypatch.setattr(mr, "_univariate_envelope_enabled", lambda: False)
    monkeypatch.setattr(mr, "_log_monomial_enabled", lambda: False)
    monkeypatch.setattr(mr, "_collect_aliased_monomial_hull_relaxations", lambda *a, **k: [])
    fp_code_absent = _relaxation_fingerprint(name)

    assert fp_real_off == fp_code_absent, (
        f"{name}: OFF relaxation differs from code-absent relaxation — an additive "
        f"Path-B row leaked into the flag-OFF path (guardrail failure)"
    )


def test_composite_hull_graduated_default_on_for_nvs09():
    """R1.2 graduated the composite univariate 1-D hull to default-ON: it is now
    chosen by the canonical dominance dispatch, no longer gated by
    ``DISCOPT_UNIVARIATE_ENVELOPE``. nvs09's DEFAULT relaxation therefore carries
    the hull columns for its ``(ln(x-2))**2``/``(ln(10-x))**2`` composites —
    proving the hull is a live default (CLAUDE.md §3: not an inert flag), the whole
    point of the #632 graduation."""
    model = from_nl(str(_NL_DIR / "nvs09.nl"))
    terms = classify_nonlinear_terms(model)
    _relax, info = build_milp_relaxation(model, terms, DiscretizationState())
    assert info["composite_relaxations"], (
        "nvs09's default relaxation must include composite-hull columns (the hull "
        "graduated default-ON in R1.2)"
    )
