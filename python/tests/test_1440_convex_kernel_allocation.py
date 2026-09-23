"""#1440 -- the convex kernel's budget allocation, made visible to CI.

`try_convex_solve` gets ``min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)`` -- the
*whole* budget for any ``time_limit <= 120``. On a model it then DECLINES, the
default path gets what is left, which is ~0 s, and short-circuits at 0 nodes.
#1429 made that spend stop being a total loss by adopting the rigorous dual bound
the declined attempt proved; the allocation itself is #1440 and is open.

**Why this file exists.** #1422's panel comment recorded that the in-repo corpus
"cannot measure this class at all -- of 66 vendored instances, 3 are
kernel-eligible, and the single decliner among them declines in 0.000 s. Zero
members of the harmed class are present in this repository."

That is not correct, and the correction is the point of this file. Of the three
eligible instances, ``clay0303hfsg`` declines at every budget up to 12 s while
consuming essentially all of it -- the exact harmed-class signature -- and only
certifies at 16 s. So the class IS reproducible here, on the very instance #911
cites to foreclose a fractional cap.

These tests pin the **soundness invariants** that must hold however #1440 is
eventually settled, plus one characterization of the current allocation that is
labelled as such. They deliberately assert structure rather than wall-clock
seconds, so they do not become a speed test on a loaded runner.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from discopt.modeling import from_nl
from discopt.solvers import _convex_kernel as ck

_NL = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "clay0303hfsg.nl")

#: ``known_optima.toml``: MINLPLib ``=opt=``. The kernel certifies 26669.10957 at a
#: generous budget, agreeing to 8 significant figures.
REFERENCE_OPTIMUM = 26669.10955143

#: Budget at which the decline is robust. Measured: declines at 2, 4, 6, 8, 10 and
#: 12 s, consuming ~100% of each; certifies at 16 s using 99.6% of it.
TIGHT = 2.0


def _fresh():
    return from_nl(_NL)


pytestmark = pytest.mark.skipif(
    not os.path.exists(_NL), reason="vendored clay0303hfsg.nl not present"
)


class TestTheHarmedClassIsPresentInThisRepository:
    """Corrects #1422's "zero members of the harmed class" record."""

    def test_the_instance_is_kernel_eligible(self):
        spec = ck.build_convex_spec(_fresh())
        assert spec is not None, (
            "clay0303hfsg is no longer accepted by build_convex_spec; the #1440 "
            "fixture has lost its subject and these tests measure nothing"
        )

    @pytest.mark.slow
    def test_a_tight_budget_is_consumed_and_the_attempt_declines(self):
        """The harmed-class signature: eligible, spends the budget, declines.

        CHARACTERIZATION, not an endorsement -- this is the behaviour #1440 exists
        to change. A fix that makes the attempt cheaper, or that stops it taking
        the whole budget, SHOULD break this test, and the replacement should
        assert the new allocation.
        """
        result = ck.try_convex_solve(_fresh(), time_limit=TIGHT, gap_tolerance=1e-4)
        assert result is None, "expected a decline at a tight budget"
        spent = ck.last_attempt_seconds()
        assert spent >= 0.5 * TIGHT, (
            f"the attempt returned after {spent:.3f}s of a {TIGHT}s budget; the "
            f"harmed class is 'declines AFTER spending', and a cheap decline "
            f"(cvxnonsep_psig40r declines in ~0 s) is not this class"
        )


class TestTheSoundnessInvariantsAnyFixMustKeep:
    @pytest.mark.slow
    def test_a_recovered_declined_bound_never_exceeds_the_optimum(self):
        """#1429 adopts a declined attempt's bound; it must stay a valid bound."""
        ck.try_convex_solve(_fresh(), time_limit=TIGHT, gap_tolerance=1e-4)
        bound = ck.last_declined_bound()
        if bound is None:
            pytest.skip("no bound was published at this budget on this machine")
        assert bound <= REFERENCE_OPTIMUM + 1e-6, (
            f"recovered declined bound {bound!r} exceeds the reference optimum "
            f"{REFERENCE_OPTIMUM!r} -- this is a FALSE bound"
        )

    @pytest.mark.slow
    def test_the_attempt_wall_is_billed_not_hidden(self):
        """#1426: a solve that spends its budget in the kernel must say so.

        Before #1426 this read as a 4 ms solve that timed out, which poisoned the
        benchmark runner's median-time and total-wall columns.
        """
        m = _fresh()
        r = m.solve(time_limit=TIGHT, gap_tolerance=1e-4)
        assert r.wall_time >= 0.5 * TIGHT, (
            f"reported wall_time {r.wall_time:.3f}s is far below the {TIGHT}s "
            f"budget the solve actually consumed -- the kernel attempt is not "
            f"being billed (#1426 regression)"
        )

    @pytest.mark.slow
    def test_the_solve_never_reports_a_bound_above_the_optimum(self):
        """Whatever the allocation, the reported bound must remain sound."""
        for tl in (TIGHT, 2 * TIGHT):
            r = _fresh().solve(time_limit=tl, gap_tolerance=1e-4)
            if r.bound is not None and np.isfinite(r.bound):
                assert r.bound <= REFERENCE_OPTIMUM + 1e-6, (
                    f"tl={tl}: bound {r.bound!r} exceeds the reference optimum"
                )
            if r.objective is not None:
                assert r.objective >= REFERENCE_OPTIMUM - 1e-3, (
                    f"tl={tl}: incumbent {r.objective!r} is BELOW the reference "
                    f"optimum {REFERENCE_OPTIMUM!r} -- a false primal"
                )


class TestTheKernelStillEarnsItsKeepHere:
    """The other half: a fix must not stop this instance certifying.

    #911 foreclosed a fractional cap because the kernel needs the large majority of
    a tight budget where it wins. Measured on this instance: it certifies at 16 s
    using **99.6%** of the budget, so any cap below that gives the certification
    back. This test is the guard on that.
    """

    @pytest.mark.slow
    def test_a_generous_budget_certifies_the_reference_optimum(self):
        r = ck.try_convex_solve(_fresh(), time_limit=60.0, gap_tolerance=1e-4)
        if r is None:
            pytest.skip("did not certify within 60 s on this machine")
        assert r.gap_certified
        assert r.objective == pytest.approx(REFERENCE_OPTIMUM, rel=1e-6), (
            f"certified {r.objective!r} against the reference optimum {REFERENCE_OPTIMUM!r}"
        )
        assert r.bound <= REFERENCE_OPTIMUM + 1e-6
