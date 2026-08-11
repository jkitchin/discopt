"""#861 — a declined incremental structure records WHY, as an attribute.

``IncrementalMcCormickLP`` recorded its decline reason only via ``logger.debug``.
Nothing was stored on the object, so a caller measuring coverage had to scrape a
log, and ``getattr(inc, "reason", None)`` returned ``None`` because the attribute
did not exist — which reads as "declined for no reason" and cost a wrong triage
pass while the issue was being written.

The reason is diagnostic ONLY: nothing in the solve path branches on it, and
declining stays exactly as sound as before (the caller falls back to the trusted
per-node cold build). What it buys is a measurable meter — the admission sweep
(``discopt_benchmarks/scripts/incremental_admission_sweep.py``) reads this
attribute instead of a DEBUG log to attribute the corpus's declines to buckets.

Pinned here: an ADMITTED structure carries ``None``; a DECLINED one carries a
non-empty ``"<ExcType>: <message>"``; and the two non-exception declines (the
#654/#844 deadline guards, which return early without raising) carry a reason
too — those were the paths most likely to look like "no reason at all".
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time

import discopt.modeling as dm
import pytest
from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
from discopt._relax.term_classifier import classify_nonlinear_terms


def _admitting_model():
    """Small all-integer QCQP (bilinear + affine squares) — in the fast-path scope."""
    m = dm.Model("iqcqp")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize((x - 3) ** 2 + (y - 2) ** 2 + x * y)
    m.subject_to(x + y >= 3)
    return m


def _declining_model():
    """An ODD power on a root box that spans zero — declined by design (the envelope
    switches between the 4-row secant/tangent hull and the 2-facet S-hull, which the
    fixed sparsity pattern cannot express). Chosen because its decline is a *stable*
    property of the mathematics, not of the current validation-box set."""
    m = dm.Model("odd_span")
    x = m.continuous("x", lb=-2.0, ub=3.0)  # straddles zero
    y = m.continuous("y", lb=0.5, ub=4.0)
    m.minimize(x**3 + y)
    m.subject_to(x + y >= 1)
    return m


def _build(model):
    return IncrementalMcCormickLP(model, classify_nonlinear_terms(model), deadline=None)


def test_admitted_structure_has_no_decline_reason():
    inc = _build(_admitting_model())
    assert inc.ok, "expected the integer QCQP to be admitted by the fast path"
    assert inc.decline_reason is None


def test_declined_structure_records_the_reason():
    inc = _build(_declining_model())
    assert not inc.ok, "expected an odd power on a straddling root box to decline"
    # The attribute exists, is non-empty, and names both the exception type and the
    # cause — enough for the sweep to bucket it without reading a log.
    assert isinstance(inc.decline_reason, str) and inc.decline_reason
    assert "ValueError" in inc.decline_reason
    assert "odd power on a root box spanning zero" in inc.decline_reason


def test_deadline_declines_carry_a_reason():
    """The two non-exception declines (#654/#844 budget guards) return early rather
    than raising, so they are the paths most at risk of reading as "no reason"."""
    model = _admitting_model()
    terms = classify_nonlinear_terms(model)
    # Deadline already in the past when the constructor is entered.
    inc = IncrementalMcCormickLP(model, terms, deadline=time.perf_counter() - 1.0)
    assert not inc.ok
    assert inc.decline_reason is not None
    assert "deadline" in inc.decline_reason.lower()


@pytest.mark.parametrize(
    "reason,expected",
    [
        (None, "admitted"),
        ("ValueError: bounds mismatch", "bounds mismatch"),
        ("ValueError: column-count mismatch", "column-count mismatch"),
        ("ValueError: bilinear (0,1) -> 5 rows, expected 4", "envelope row count != 4"),
        ("ValueError: relaxation has no valid bound / no rows", "no valid bound / no rows"),
        (
            "ValueError: monomial x_2^3: odd power on a root box spanning zero (…)",
            "odd power on straddling root",
        ),
        ("deadline spent during validation", "deadline spent"),
    ],
)
def test_sweep_bucketing_of_reasons(reason, expected):
    """The sweep's bucket mapping is part of the meter's contract: a renamed error
    message that silently falls into "other" would make a coverage change look like
    a bucket shift. Pin the mapping against the exact messages the code raises."""
    import importlib.util

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "discopt_benchmarks", "scripts", "incremental_admission_sweep.py")
    spec = importlib.util.spec_from_file_location("_admission_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.bucket(reason) == expected
