"""#1449: the pump's accept gate is deliberately stricter than ``verify_point``.

``_check_constraint_feasibility`` is the feasibility pump's accept gate. It is a
PRE-FILTER: every candidate it passes is re-verified by
``solver._native_kernel_verify_point`` -> ``validation.feasibility.verify_point``
before it may seed a cutoff or become an incumbent. The two use different row
bounds, and the gate's is far tighter:

  * gate:         ``tol + rtol*scale``           (tol=1e-6, rtol=1e-9)
  * verify_point: ``abs_tol*max(anchor, scale)`` (abs_tol=1e-6)

On a row whose terms are of magnitude 3.16e4 that is 3.26e-5 against 3.16e-2 --
a ~970x gap. #1449 asked whether closing it recovers lost incumbents.

THE ANSWER IS NO, AND THESE TESTS PIN THE CURRENT FORM SO IT IS NOT RE-OPENED BY
ACCIDENT. The discarded work is real: re-running ``verify_point`` on every
rejection the gate made during real solves, 8 of 57 compared rejections (14.0 %)
on a 119-instance MINLPLib sample were points ``verify_point`` accepts outright
(1 of 62 on the vendored corpus). But an aligned gate, run as a 117-instance
differential panel at 20 s/instance, was cert-clean and NEUTRAL: 30/30 certified,
61/61 with an incumbent, total wall +0.15 %, dual bound 6 tighter / 8 looser. Its
one objective difference did not reproduce over 8 interleaved repeats per arm
(OFF -50223.5 +/- 158.0, ON -50236.0 +/- 117.8, best of all 16 runs in the OFF
arm). The alignment was retired under CLAUDE.md §5's three-outcome rule.

So the divergence below is a measured decision, not an oversight. A future change
that closes it needs evidence of a primal LOSS attributable to the threshold, not
of a rejection count.
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


def test_gate_is_the_legacy_additive_threshold():
    """CHARACTERIZATION. The gate is ``tol + rtol*scale``, bit-for-bit. If this
    fails, someone has changed the pump's accept threshold -- read #1449 before
    deciding that is an improvement."""
    np.testing.assert_array_equal(_tol(SCALES), 1e-6 + 1e-9 * SCALES)


def test_the_gap_to_verify_point_is_large_and_deliberate():
    """Pins the ~970x divergence #1449 measured, so the number in the docstring
    cannot silently stop being true."""
    judge = ABS_TOL * np.maximum(1.0, SCALES)
    gate = _tol(SCALES)
    # Wherever the two MATERIALLY diverge the gate is the stricter one. Below
    # scale ~1.001 the additive rtol term makes it fractionally (<=0.1 %) looser
    # than the judge, which is why an "alignment" had to keep it in a max().
    big = SCALES > 1.001
    assert big.sum() >= 3, "no diverging scales exercised (vacuous pass)"
    assert np.all(gate[big] < judge[big]), "the gate is no longer the stricter of the two"
    assert np.all(gate <= judge * 1.001 + 1e-18), "the gate is materially looser somewhere"
    i = int(np.argmin(np.abs(SCALES - 3.16e4)))
    ratio = judge[i] / gate[i]
    assert ratio > 100.0, f"expected ~1000x at scale 3.16e4, got {ratio:.1f}x"


def test_gate_rejects_a_point_the_judge_accepts():
    """The concrete shape that raised #1449: a row violated by 3.5e-5 whose terms
    are of magnitude 3.16e4. The judge allows 3.16e-2; the gate allows 3.26e-5 and
    refuses. Measured to cost nothing (see module docstring), but it IS a refusal
    and this test says so out loud rather than leaving it implicit."""
    scale = np.array([3.16e4])
    grad = np.array([2.0119e4])
    viol = 3.47555e-05

    gate = float(_tol(scale, grad=grad)[0])
    judge = ABS_TOL * max(1.0, float(scale[0]))
    assert viol > gate, f"gate would accept {viol:.3e} (tol {gate:.3e}) -- premise gone"
    assert viol <= judge, f"judge would also reject {viol:.3e} (tol {judge:.3e}) -- premise gone"


def test_the_1254_distance_cap_still_binds():
    """SOUNDNESS PIN. #1254's first-order distance cap is what stops a nearly flat
    row vouching for a point no small move makes feasible. It must keep binding
    whatever happens to the loosening half."""
    scale = np.array([3.16e4, 1e6])
    # A steep row: the cap is generous, so the loosening half decides.
    steep = np.array([1e4, 1e4])
    # A nearly flat row: the cap must decide.
    flat = np.array([1e-10, 1e-10])
    capped = _tol(scale, grad=flat)
    uncapped = _tol(scale, grad=steep)
    expected_cap = ph.feasible_distance_cap(flat, scale)
    np.testing.assert_allclose(capped, np.minimum(_tol(scale), expected_cap))
    assert np.all(capped <= uncapped), "the cap failed to bind on the flat row"
    assert np.all(capped < _tol(scale)), "the cap was a no-op (vacuous pass)"


def test_the_retired_alignment_flag_is_gone():
    """#1449 was retired, not shipped default-off. A flag left behind would read
    as a stalled graduation to the next person (CLAUDE.md §5, #1345)."""
    assert not hasattr(ph, "_pump_gate_align_enabled")
    os.environ["DISCOPT_PUMP_GATE_ALIGN"] = "1"
    try:
        np.testing.assert_array_equal(_tol(SCALES), 1e-6 + 1e-9 * SCALES)
    finally:
        del os.environ["DISCOPT_PUMP_GATE_ALIGN"]
