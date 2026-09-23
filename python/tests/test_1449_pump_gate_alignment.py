"""#1449: the pump's accept gate is ~1000x stricter than the judge it predicts.

THE DEFECT. ``_check_constraint_feasibility`` is the feasibility pump's accept
gate. It is a PRE-FILTER: every candidate it passes is re-verified by
``solver._native_kernel_verify_point`` -> ``validation.feasibility.verify_point``
before it may seed a cutoff or become an incumbent. But the two use different
row bounds:

  * gate:         ``tol + rtol*scale``          (tol=1e-6, rtol=1e-9)
  * verify_point: ``abs_tol*max(anchor, scale)`` (abs_tol=1e-6)

On a row whose terms are of magnitude 3.2e4 that is 3.26e-5 against 3.16e-2 --
a ~970x gap. A pre-filter stricter than its own judge discards work for nothing.

THE EVIDENCE (#1449 entry experiment, re-running ``verify_point`` on every
rejection the gate made during a real solve): on a 119-instance MINLPLib sample,
8 of 57 compared rejections (14.0 %) were points ``verify_point`` accepts
outright -- 7x the issue's 2 % kill criterion.

THE FLAG. ``DISCOPT_PUMP_GATE_ALIGN`` (default OFF) switches the loosening half
to ``abs_tol*max(1, scale)``. These tests pin the properties the §5 graduation
panel rests on, none of which depend on machine speed or on POUNCE:

  * OFF is bit-for-bit the legacy threshold (the opt-out stays real);
  * ON is never STRICTER than OFF -- it can only pass more candidates, so it
    cannot lose an incumbent the old gate would have produced;
  * ON tracks ``verify_point``'s own row bound wherever the two forms diverge
    (scale > 1.001), so the filter predicts its judge instead of second-guessing
    it;
  * the #1254 distance cap still binds in both arms. THIS IS THE SOUNDNESS PIN:
    the alignment must not become a back door around the cap.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax import primal_heuristics as ph  # noqa: E402

pytestmark = pytest.mark.unit

ABS_TOL = 1e-6
# Row term magnitudes spanning the regime where the two forms diverge: at
# scale=0 they agree exactly, by 3.2e4 they are ~1000x apart.
SCALES = np.array([0.0, 1.0, 1e2, 3.16e4, 1e6, 1e9])


def _tol(scale, *, grad=None, **kw):
    return ph.combined_tolerance(np.asarray(scale, dtype=float), grad_inf=grad, **kw)


def test_off_is_the_legacy_threshold_exactly(monkeypatch):
    """The opt-out must be real: OFF reproduces ``tol + rtol*scale`` bit-for-bit."""
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "0")
    got = _tol(SCALES)
    np.testing.assert_array_equal(got, 1e-6 + 1e-9 * SCALES)


def test_on_matches_verify_points_row_bound_where_it_matters(monkeypatch):
    """Wherever the two forms diverge (scale > 1.001), ON is exactly the judge's
    row bound at anchor 1. Below that the legacy additive form is fractionally
    larger and is kept, so "align" never tightens a small row."""
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "1")
    big = SCALES[SCALES > 1.001]
    assert big.size >= 3, "no diverging scales exercised (vacuous pass)"
    np.testing.assert_allclose(_tol(big), ABS_TOL * big, rtol=0, atol=0)


def test_on_is_never_stricter_than_off(monkeypatch):
    """THE REGRESSION. Fails before the change (the flag does nothing, so the two
    arms are equal and the strict inequality below never holds).

    A pre-filter that only loosens cannot lose a candidate the old one passed,
    which is what licenses reading a panel's primal deltas as gains.
    """
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "0")
    off = _tol(SCALES)
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "1")
    on = _tol(SCALES)
    assert np.all(on >= off), f"alignment TIGHTENED the gate: {on} < {off}"
    # ...and it is a real widening, not a no-op dressed as one.
    widened = int(np.sum(on > off))
    assert widened >= 3, f"only {widened} of {SCALES.size} scales widened (vacuous pass)"
    # The gasnet row that raised #1449: scale 3.16e4 -> ~970x.
    i = int(np.argmin(np.abs(SCALES - 3.16e4)))
    assert on[i] / off[i] > 100.0, f"expected ~1000x at scale 3.16e4, got {on[i] / off[i]:.1f}x"


def test_on_never_outruns_the_judge_by_more_than_the_legacy_term(monkeypatch):
    """The filter should not be materially looser than the judge -- a candidate
    it passes that the judge refuses is a wasted ``verify_point`` call. The only
    excess is the legacy ``rtol*scale`` term retained for small rows, which is
    1e-3 of the judge's own bound."""
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "1")
    on = _tol(SCALES)
    for a in (1.0, 5.0, 1e3, 1e7):
        judge = ABS_TOL * np.maximum(a, SCALES)
        assert np.all(on <= judge * 1.001 + 1e-18), f"gate materially looser at anchor {a}"


@pytest.mark.parametrize("flag", ["0", "1"])
def test_the_1254_distance_cap_still_binds_in_both_arms(monkeypatch, flag):
    """SOUNDNESS PIN. #1254's first-order distance cap is what stops a nearly
    flat row vouching for a point no small move makes feasible. The alignment
    touches the loosening half ONLY; if it ever becomes a way around the cap,
    this fails loudly.
    """
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", flag)
    scale = np.array([3.16e4, 1e6])
    # A steep row: the cap is generous, so the loosening half decides.
    steep = np.array([1e4, 1e4])
    # A nearly flat row: the cap must decide, in BOTH arms.
    flat = np.array([1e-10, 1e-10])
    capped = _tol(scale, grad=flat)
    uncapped = _tol(scale, grad=steep)
    expected_cap = ph.feasible_distance_cap(flat, scale)
    np.testing.assert_allclose(capped, np.minimum(_tol(scale), expected_cap))
    assert np.all(capped <= uncapped), "the cap failed to bind on the flat row"
    assert np.all(capped < _tol(scale)), "the cap was a no-op (vacuous pass)"


def test_flag_is_read_from_the_environment_not_frozen_at_import(monkeypatch):
    """A flag cached at import cannot be A/B'd in one tree, which is how the
    graduation panel is run."""
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "0")
    assert ph._pump_gate_align_enabled() is False
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "1")
    assert ph._pump_gate_align_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", off)
        assert ph._pump_gate_align_enabled() is False, off


def test_gate_accepts_a_point_the_judge_accepts(monkeypatch):
    """End-to-end on the shape that raised #1449: a row violated by 3.5e-5 whose
    terms are of magnitude 3.16e4. The judge allows 3.16e-2; the legacy gate
    allows 3.26e-5 and refuses. Uses the real threshold function, so it pins the
    decision rather than restating the arithmetic.
    """
    scale = np.array([3.16e4])
    grad = np.array([2.0119e4])
    viol = 3.47555e-05

    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "0")
    legacy = float(_tol(scale, grad=grad)[0])
    monkeypatch.setenv("DISCOPT_PUMP_GATE_ALIGN", "1")
    aligned = float(_tol(scale, grad=grad)[0])

    assert viol > legacy, f"legacy gate would have accepted {viol:.3e} (tol {legacy:.3e})"
    assert viol <= aligned, f"aligned gate still rejects {viol:.3e} (tol {aligned:.3e})"
