"""#1397 site 9: the edge-concave ``sense`` must not be read off a cancelling sum.

``collect_edge_concave_quadratics`` decides ``sense`` from the sign of the accumulated
``x_i^2`` coefficients, and ``sense`` selects the *direction* of the inequality that
``_separate_edge_concave`` appends to the node LP (``mccormick_lp.py``) with no
downstream validity check. For an edge-concave block the *minimum* is at a box vertex,
so vertex data yields a valid under-estimator; for an edge-convex block the *maximum*
is at a vertex. Flip the sign and the vertex-derived hyperplane is emitted on the side
where it is not a bound, cutting off true points.

The old test was absolute (``all(v <= 1e-12)`` / ``any(v < -1e-9)``) while the residue
of a cancelling accumulation scales with the terms that cancelled. Measured: a true
curvature of -3.95e-10 accumulates to +1.49e-09 from addends of magnitude 5.2e7 —
clearing the 1e-9 "edge-convex" threshold with the wrong sign. Invalidity of the
resulting cut over a width-10 box, as a function of the cancelling magnitude M:
M=1e8 -> 6.4e-7, M=1e10 -> 9.3e-5, M=1e12 -> 5.6e-3, M=1e14 -> 0.69, M=1e16 -> 93.
Everything from 1e10 up exceeds the 1e-6 feasibility tolerance.

These tests pin the *class*: the sign of an accumulated coefficient is declared only
when it clears the round-off bound of its own addends. They are not tied to the
particular instance below, which is only the smallest reproducer of it.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import pytest
from discopt._relax.edge_concave import _definite_sign, collect_edge_concave_quadratics

# The measured reproducer: addends [M, t, -M, -u] with t < u, so the true sum is
# NEGATIVE (edge-concave), but t exceeds half an ulp of M while u does not, so the
# running sum lands at +1.49e-09 and reads as edge-convex.
M = 51903611.79275842
T = 5.569955805899157e-09
U = 5.9652512542035745e-09
CANCELLING = [M, T, -M, -U]


def _block_model(addends, *, bilin=3.0, ub=10.0):
    """One quadratic body whose ``x^2`` coefficient is the given accumulation."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=ub)
    y = m.continuous("y", lb=0.0, ub=ub)
    body = addends[0] * x * x
    for c in addends[1:]:
        body = body + c * x * x
    m.minimize(body + bilin * x * y)
    return m


def test_the_reproducer_really_does_accumulate_to_the_wrong_sign():
    """Guard the premise: without this, every assertion below is vacuous."""
    assert math.fsum(CANCELLING) < 0.0, "true curvature must be negative (edge-concave)"
    running = 0.0
    for c in CANCELLING:
        running += c
    assert running > 1e-9, "running sum must clear the old 1e-9 edge-convex threshold"
    assert running > 0.0 > math.fsum(CANCELLING), "the sign must actually be inverted"


def test_no_over_sense_on_a_truly_edge_concave_block():
    """The bug: ``sense='over'`` on a block whose true curvature is negative.

    Fails before the fix (the block is collected with ``sense='over'``).
    """
    blocks = collect_edge_concave_quadratics(_block_model(CANCELLING))
    senses = [b.sense for b in blocks]
    assert "over" not in senses, (
        f"block classified edge-convex though its true curvature is "
        f"{math.fsum(CANCELLING):.6e} < 0; senses={senses}"
    )


@pytest.mark.parametrize("magnitude", [1e6, 1e8, 1e10, 1e12, 1e14, 1e16])
def test_sign_is_never_declared_from_a_residue_below_its_round_off_bound(magnitude):
    """The class, swept over the scale that drives it (#1397 parameter sweep).

    At every magnitude, an accumulation whose residue is below its own round-off
    bound must yield no sign -- and hence no block -- rather than a sign whose
    correctness is set by the rounding.
    """
    step = math.ulp(magnitude)
    addends = [magnitude, 0.75 * step, -magnitude, -0.80 * step]
    total = 0.0
    for c in addends:
        total += c
    assert _definite_sign(total, addends) == 0, (
        f"declared a sign for residue {total:.6e} accumulated from addends of "
        f"magnitude {magnitude:.0e} (round-off bound scales with the addends)"
    )
    assert not [b for b in collect_edge_concave_quadratics(_block_model(addends))]


def test_an_uncancelled_coefficient_keeps_its_sign():
    """No capability loss: a coefficient assembled without cancellation still reads."""
    assert _definite_sign(-2.0, [-2.0]) == -1
    assert _definite_sign(3.0, [3.0]) == 1
    assert _definite_sign(1000001.0, [1e6, 1.0]) == 1
    # Tiny but honest: a single addend carries its own scale, so the relative test
    # keeps a curvature the old absolute ``< -1e-9`` threshold threw away.
    assert _definite_sign(-1e-30, [-1e-30]) == -1


@pytest.mark.parametrize(
    ("addends", "want"),
    [
        ([-2.0], "under"),
        ([2.0], "over"),
        ([1e6, 1.0], "over"),
        ([-1e-30], "under"),
    ],
)
def test_well_conditioned_blocks_are_still_collected(addends, want):
    senses = [b.sense for b in collect_edge_concave_quadratics(_block_model(addends))]
    assert want in senses, f"lost a legitimate {want!r} block for addends={addends}"


def test_a_purely_bilinear_variable_is_an_exact_zero_not_an_undetermined_one():
    """A variable with no square term at all is exactly 0 and blocks neither sense.

    This is the distinction the old ``sq.get(i, 0.0)`` erased: a structurally absent
    coefficient is exact, while a computed one near zero is not.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(-2.0 * x * x + 4.0 * x * y + y)
    blocks = collect_edge_concave_quadratics(m)
    assert [b.sense for b in blocks] == ["under"]
    assert 1 not in blocks[0].sq, "y has no square term; it must not be stored as 0.0"


def test_infinite_accumulation_declares_no_sign():
    assert _definite_sign(float("inf"), [1e308, 1e308]) == 0
    assert _definite_sign(float("nan"), [1.0, -1.0]) == 0
