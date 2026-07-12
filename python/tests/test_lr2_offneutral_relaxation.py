"""cert:LR-2 (Path B) — flag-OFF relaxation byte-identity guardrail.

Path B ships the H-UNI / H-LOG envelopes as **purely additive** rows/columns
collected inside :func:`build_milp_relaxation`, each guarded by a default-OFF env
flag (``DISCOPT_UNIVARIATE_ENVELOPE`` / ``DISCOPT_LOG_MONOMIAL``). The soundness
argument for shipping them is that when the flags are OFF the relaxation matrix is
byte-identical to prior main: the collectors are never called, so no row/column is
added and the composite-claim path stays at ``allow_general=False``.

The team-lead's lock-condition #1 is explicit: *prove* that neutrality with a test,
do not assume it. This module does so at the relaxation-matrix level (the layer the
additive rows actually touch): for every corpus instance it fingerprints the built
LP ``(_c, _A_ub, _b_ub, _bounds, _integrality)`` with the flags OFF and asserts it
is identical to the fingerprint built with the additive collectors forced to their
"code-absent" behavior (empty / disabled). If they differ, an OFF path is leaking a
row — a guardrail failure.

It also asserts the lever is *real* (ON changes nvs09's relaxation), so the flag is
not inert (CLAUDE.md §3).
"""

from __future__ import annotations

import glob
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._jax import milp_relaxation as mr
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import from_nl

pytestmark = [pytest.mark.correctness, pytest.mark.claim_boundary]

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
_ALL = sorted(os.path.basename(p)[:-3] for p in glob.glob(str(_NL_DIR / "*.nl")))


def _relaxation_fingerprint(name: str) -> str:
    """Deterministic hash of the built MILP relaxation matrix for ``name``.

    Covers every array a Path-B additive row could touch: objective ``_c``, the
    inequality matrix ``_A_ub`` (densified in a stable order), the RHS ``_b_ub``,
    the variable ``_bounds``, and the ``_integrality`` mask. Two identical hashes
    mean identical LP relaxations (same columns, same rows, same numbers)."""
    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    terms = classify_nonlinear_terms(model)
    relax, _info = build_milp_relaxation(model, terms, DiscretizationState())

    h = hashlib.sha256()

    def _feed(label: str, arr) -> None:
        h.update(label.encode())
        if arr is None:
            h.update(b"None")
            return
        if sp.issparse(arr):
            arr = np.asarray(arr.todense())
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())

    _feed("c", relax._c)
    _feed("A_ub", relax._A_ub)
    _feed("b_ub", relax._b_ub)
    _feed("bounds", np.asarray(relax._bounds, dtype=np.float64) if relax._bounds else None)
    _feed(
        "integrality",
        np.asarray(relax._integrality, dtype=np.float64)
        if relax._integrality is not None
        else None,
    )
    return h.hexdigest()


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


def test_flag_on_changes_nvs09_relaxation(monkeypatch):
    """Flag ON must add rows to nvs09's relaxation — proves the envelope lever is
    real, not an inert flag (CLAUDE.md §3)."""
    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "0")
    fp_off = _relaxation_fingerprint("nvs09")
    monkeypatch.setenv("DISCOPT_UNIVARIATE_ENVELOPE", "1")
    fp_on = _relaxation_fingerprint("nvs09")
    assert fp_on != fp_off, "H-UNI ON must change nvs09's relaxation matrix"
