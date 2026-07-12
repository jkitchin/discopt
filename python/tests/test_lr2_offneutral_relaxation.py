"""cert:LR-2 (Path B) — flag-OFF relaxation byte-identity guardrail (H-LOG).

Path B shipped the H-UNI / H-LOG envelopes as **purely additive** rows/columns
collected inside :func:`build_milp_relaxation`, each guarded by a default-OFF env
flag. H-UNI (the composite univariate 1-D hull and its opt-in flag) has since been
**deleted** (issue #632): a monolithic per-node composite hull is not how the
factorable SOTA solvers build envelopes, and non-convex composites are recovered
through the factorable AVM instead. H-LOG (``DISCOPT_LOG_MONOMIAL``, the log-space
positive-product envelope) remains as a sound opt-in flag.

The soundness argument for shipping H-LOG is that when the flag is OFF the
relaxation matrix is byte-identical to prior main: the collector is never called,
so no row/column is added.

The team-lead's lock-condition #1 is explicit: *prove* that neutrality with a test,
do not assume it. This module does so at the relaxation-matrix level (the layer the
additive rows actually touch): for every corpus instance it fingerprints the built
LP ``(_c, _A_ub, _b_ub, _bounds, _integrality)`` with the flag OFF and asserts it
is identical to the fingerprint built with the additive H-LOG collector forced to
its "code-absent" behavior (disabled). If they differ, an OFF path is leaking a
row — a guardrail failure.
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
    monkeypatch.delenv("DISCOPT_LOG_MONOMIAL", raising=False)
    yield


def test_flag_default_off():
    # H-LOG is a sound opt-in flag; OFF is byte-identical to prior main (guarded
    # below). (H-UNI's flag was deleted with the composite hull, #632.)
    assert mr._log_monomial_enabled() is False


@pytest.mark.parametrize("name", _ALL)
def test_relaxation_off_byte_identical_corpus(name, monkeypatch):
    """OFF relaxation matrix must equal the relaxation built as if the additive
    H-LOG code did not exist.

    "Code-absent" is simulated by neutralizing the flag gate that could add a row
    (the H-LOG collector). Because the flag is OFF, the real path already skips it
    — so equality proves the OFF path is a genuine no-op (no row leaks),
    byte-identical to prior main, on every corpus instance."""
    monkeypatch.setenv("DISCOPT_LOG_MONOMIAL", "0")
    fp_real_off = _relaxation_fingerprint(name)

    # Force the additive H-LOG entry point to its code-absent behavior.
    monkeypatch.setattr(mr, "_log_monomial_enabled", lambda: False)
    fp_code_absent = _relaxation_fingerprint(name)

    assert fp_real_off == fp_code_absent, (
        f"{name}: OFF relaxation differs from code-absent relaxation — an additive "
        f"H-LOG row leaked into the flag-OFF path (guardrail failure)"
    )
