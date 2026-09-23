"""#1386: the REPORTED gap must be the gap of the pair that is reported.

``SolveResult.gap`` is documented as ``|objective - bound|`` over
``max(|objective|, |bound|, ...)``. Every route's exit recomputed it by hand as
``|obj - bound| / max(1.0, |obj|)`` -- the tree's own hybrid denominator, which is
right for the internal termination test and wrong for the reported number because
it **drops ``|bound|``**. The two disagree exactly when ``|bound| > max(1, |obj|)``.

Found by an adversarial probe comparing every solved instance's reported gap
against ``optimality_gap`` of its own published pair: 61 executed comparisons over
the vendored corpus plus 13 hand-built models, 1 violation -- the vendored
``nvs09`` at a 4 s limit (minimise, ``status="feasible"``, ``objective=-37.894800``,
``bound=-50.589648``) reporting **0.335002** where its pair gives **0.250938**.

The error is one-directional: ``max(1.0, |ub|) <= max(|lb|, |ub|, 1.0)``, so the
gap was never *understated* and no certificate was ever granted on it. It is a
reporting defect, not a soundness one -- which is why the fix is required to be
bound-neutral, and the first test here is what pins that.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from discopt.modeling import from_nl
from discopt.solver import _withhold_stale_certificate
from discopt.solvers._gap import optimality_gap

_NL = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "nvs09.nl")


def _call(obj, bound, *, certified=False, is_max=False, gap=None):
    """Drive the shared exit guard the four publishing routes all call."""
    return _withhold_stale_certificate(
        "feasible" if not certified else "optimal",
        obj,
        bound,
        gap,
        certified,
        is_max,
        1e-4,
        1e-6,
        "test",
    )


class TestTheReportedGapIsTheGapOfTheReportedPair:
    def test_the_bound_is_not_dropped_from_the_denominator(self):
        """The nvs09 shape: ``|bound|`` exceeds ``max(1, |objective|)``."""
        obj, bound = -37.89479964149548, -50.589648265739015
        hybrid = abs(obj - bound) / max(1.0, abs(obj))
        _, gap, _, _ = _call(obj, bound, gap=hybrid)
        want = optimality_gap(bound, obj)
        assert gap == pytest.approx(want, rel=1e-12), (
            f"reported {gap!r} against the contract {want!r} for the published pair "
            f"(objective={obj!r}, bound={bound!r})"
        )
        assert gap < hybrid, (
            "this fixture must exercise the disagreement; if the hybrid and the "
            "contract now agree here, the test has stopped measuring anything"
        )

    def test_the_sense_is_respected(self):
        """On a MAXIMIZE model ``bound`` is the UPPER bound.

        Passing the pair in the model's sense rather than minimisation sense would
        invert the ordering, and ``optimality_gap`` returns 1.0 ("nothing known")
        on a materially invalid ordering -- so a sign slip here shows up as a gap
        of 1.0 rather than as a small number, which is what this pins.
        """
        obj, bound = 37.89479964149548, 50.589648265739015  # maximize: lb=obj, ub=bound
        _, gap, _, _ = _call(obj, bound, is_max=True, gap=0.5)
        want = optimality_gap(-bound, -obj)
        assert gap == pytest.approx(want, rel=1e-12)
        assert gap < 1.0, "the pair was passed in the wrong sense (ordering inverted)"

    def test_a_withdrawn_bound_keeps_no_gap(self):
        """Normalisation must not invent a gap where the guard removed the bound."""
        _, gap, _, bound = _call(None, None, gap=None)
        assert gap is None and bound is None

    def test_a_closed_gap_still_reports_zero(self):
        obj = bound = 12.5
        _, gap, _, _ = _call(obj, bound, gap=0.0)
        assert gap == 0.0

    @pytest.mark.skipif(not os.path.exists(_NL), reason="vendored nvs09.nl not present")
    @pytest.mark.slow
    def test_end_to_end_on_the_instance_that_exposed_it(self):
        r = from_nl(_NL).solve(time_limit=4.0, gap_tolerance=1e-4)
        if r.objective is None or r.bound is None or not np.isfinite(r.bound):
            pytest.skip("no (objective, bound) pair on this machine at this budget")
        want = optimality_gap(r.bound, r.objective)
        assert r.gap == pytest.approx(want, rel=1e-9), (
            f"reported gap {r.gap!r} is not the gap of the reported pair "
            f"(objective={r.objective!r}, bound={r.bound!r}, contract={want!r})"
        )
