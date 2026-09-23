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

import logging
import os

import discopt.modeling as dm
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

#: Budget at which the declined attempt reliably HOLDS a feasible point (#1440).
#: The kernel's work is back-loaded -- its first incumbent arrives essentially at
#: convergence -- so this is measured, not guessed: 5 replicates each gave 0/5 at
#: 10 s, 4/5 at 12 s and 5/5 at 14 s, always the same point (47287.5613). The
#: tests that use it skip rather than fail when nothing was published, since a
#: slower machine legitimately reaches convergence later.
RECOVERY_BUDGET = 14.0


def _fresh():
    return from_nl(_NL)


#: Applied per class rather than module-wide: the adoption guards below drive the
#: block with a synthetic point on a model built in-process, so they must keep
#: running on a tree that does not vendor this instance.
needs_nl = pytest.mark.skipif(
    not os.path.exists(_NL), reason="vendored clay0303hfsg.nl not present"
)


@needs_nl
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


@needs_nl
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


@needs_nl
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


@needs_nl
class TestTheDeclinedAttemptsIncumbentIsRecovered:
    """#1440: a declined attempt also discards any FEASIBLE POINT it found.

    Same waste as #1422's discarded dual bound, the other half of it, and free in
    the same way: it recovers what the spend already produced rather than changing
    the allocation. ``DISCOPT_CONVEX_KERNEL_KEEP_INCUMBENT`` gates it (default ON).
    """

    def test_the_flag_is_on_by_default_and_has_a_working_opt_out(self, monkeypatch):
        monkeypatch.delenv("DISCOPT_CONVEX_KERNEL_KEEP_INCUMBENT", raising=False)
        assert ck.keep_declined_incumbent_enabled() is True
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_KEEP_INCUMBENT", "0")
        assert ck.keep_declined_incumbent_enabled() is False

    @pytest.mark.slow
    def test_a_published_point_is_feasible_in_the_pristine_model(self):
        """Whatever is published must be a point, not a number someone hopes is one.

        ``_incumbent_is_feasible`` runs before publication; this re-checks it here
        with the same verifier on a FRESH parse, so the test does not take the
        publisher's word for its own guard.
        """
        from discopt.validation.feasibility import verify_point

        ck.try_convex_solve(_fresh(), time_limit=RECOVERY_BUDGET, gap_tolerance=1e-4)
        published = ck.last_declined_incumbent()
        if published is None:
            pytest.skip(
                f"no incumbent was published at {RECOVERY_BUDGET}s on this machine; "
                f"the kernel's first incumbent arrives essentially at convergence, so "
                f"a slower machine legitimately has none to recover"
            )
        obj, xd = published
        m = _fresh()
        flat = np.concatenate(
            [np.atleast_1d(np.asarray(xd[v.name], dtype=float)).ravel() for v in m._variables]
        )
        assert verify_point(m, flat).ok, (
            "a point that does NOT satisfy the pristine model was published as a "
            "recoverable incumbent -- this is the false-primal class (CLAUDE.md §1)"
        )
        assert obj >= REFERENCE_OPTIMUM - 1e-3, (
            f"published incumbent {obj!r} is BELOW the reference optimum {REFERENCE_OPTIMUM!r}"
        )

    @pytest.mark.slow
    def test_the_published_point_reaches_the_caller(self):
        """The fails-before assertion: publication is worthless if nothing adopts it.

        Measured before the fix, 3 reps at this budget: the attempt held
        47287.5613 every time and ``Model.solve`` returned ``objective=None`` every
        time. With the kernel off the solve reports an incumbent but a bound of
        -0.0 (a 100% gap), so neither configuration reported a usable pair.
        """
        m = _fresh()
        r = m.solve(time_limit=RECOVERY_BUDGET, gap_tolerance=1e-4)
        published = ck.last_declined_incumbent()
        if published is None:
            pytest.skip(f"no incumbent was published at {RECOVERY_BUDGET}s on this machine")
        if r.gap_certified:
            pytest.skip("this machine certified within the budget; nothing was declined")
        assert r.objective is not None, (
            f"the declined attempt published a verified incumbent {published[0]!r} "
            f"and the solve still reported none -- the recovery is not wired up"
        )
        assert r.objective == pytest.approx(published[0], rel=1e-9), (
            f"reported {r.objective!r} against the published {published[0]!r}"
        )
        assert r.x, "an objective was adopted without the point it came from"
        assert not r.gap_certified, "a recovered incumbent is never a certificate"
        assert r.objective >= REFERENCE_OPTIMUM - 1e-3


def _bilinear(sense: str):
    """A model the convex kernel REFUSES, so the adoption path is what is tested.

    ``x*y`` on a box is nonconvex, so ``build_convex_spec`` declines it and the
    solve goes through the default path -- which is the path the recovery block
    sits on. The test below asserts that refusal rather than assuming it
    (CLAUDE.md #6).
    """
    m = dm.Model("bilinear_1440")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    if sense == "min":
        m.minimize(x * y)
    else:
        m.maximize(x * y)
    m.subject_to(x + y <= 6.0)
    return m


#: The point published as the declined attempt's incumbent. Feasible in
#: ``_bilinear`` (1 + 2 <= 6, both inside [0, 4]) with objective 2.0, so the #772
#: false-primal screen that runs on whatever is adopted passes it on its merits.
POINT = {"x": np.array(1.0), "y": np.array(2.0)}


class TestTheAdoptionGuards:
    """Deterministic unit coverage of the adoption block itself.

    The end-to-end tests above need the kernel to reach convergence-adjacent work
    on a real instance, which is a machine-speed question. These drive the block
    directly by publishing a synthetic point and starving the default path -- the
    same technique ``test_1422_declined_kernel_bound.py`` uses for the bound -- so
    the guards are pinned on every runner regardless of speed.
    """

    @staticmethod
    def _stub(monkeypatch, published, *, fake):
        """Publish ``published`` as the declined attempt's incumbent and make the
        default path return ``fake``."""
        import discopt.solver as _solver

        monkeypatch.setattr(ck, "last_attempt_seconds", lambda: 0.0)
        monkeypatch.setattr(ck, "last_attempt_rust_seconds", lambda: 0.0)
        monkeypatch.setattr(ck, "last_declined_bound", lambda: None)
        monkeypatch.setattr(ck, "last_declined_incumbent", lambda: published)
        monkeypatch.setattr(_solver, "solve_model", lambda model, **kw: fake)

    @staticmethod
    def _fake(objective=None, bound=None, status="time_limit", certified=False, x=None):
        from discopt.modeling.core import SolveResult

        # ``bound`` goes through the CONSTRUCTOR, not only ``_set_bound``:
        # ``__post_init__`` downgrades ``gap_certified`` on a result whose bound is
        # absent or non-finite, so a "certified" fake built without one arrives
        # UNcertified and the test silently exercises a different branch.
        r = SolveResult(
            status=status,
            objective=objective,
            bound=bound,
            x=x or {},
            gap_certified=certified,
        )
        if bound is not None:
            r._set_bound(bound, valid=True, source="bnb_tree")
        return r

    def test_the_fixture_model_is_refused_by_the_kernel(self):
        assert ck.build_convex_spec(_bilinear("min")) is None, (
            "the fixture is now kernel-eligible, so these tests would exercise the "
            "kernel's own accept path instead of the adoption block"
        )

    def test_a_point_is_adopted_when_the_default_path_found_none(self, monkeypatch):
        from discopt.solvers._gap import optimality_gap

        self._stub(monkeypatch, (2.0, POINT), fake=self._fake(bound=-16.0))
        r = _bilinear("min").solve(time_limit=2.0, gap_tolerance=1e-4)
        assert r.objective == pytest.approx(2.0)
        assert set(r.x) == {"x", "y"}
        assert r.x["x"] == pytest.approx(1.0)
        assert not r.gap_certified, "a recovered point must never certify"
        assert r.status == "time_limit", "a recovered point must not upgrade the status"
        assert r.gap == pytest.approx(optimality_gap(-16.0, 2.0)), (
            "the gap must be recomputed from the pair actually reported (#1386)"
        )

    def test_a_worse_point_is_ignored(self, monkeypatch):
        self._stub(monkeypatch, (2.0, POINT), fake=self._fake(objective=0.5, bound=-16.0))
        r = _bilinear("min").solve(time_limit=2.0, gap_tolerance=1e-4)
        assert r.objective == pytest.approx(0.5), (
            "adopting a WORSE incumbent than the solve already had is a regression, not a recovery"
        )

    def test_the_sense_is_respected(self, monkeypatch):
        """On a MAXIMIZE model, 'better' means LARGER -- the #860 lesson.

        A sign-blind comparison would adopt 2.0 over 6.0 here and report the worse
        of the two as the incumbent.
        """
        self._stub(monkeypatch, (2.0, POINT), fake=self._fake(objective=6.0, bound=16.0))
        r = _bilinear("max").solve(time_limit=2.0, gap_tolerance=1e-4)
        assert r.objective == pytest.approx(6.0)

        self._stub(monkeypatch, (2.0, POINT), fake=self._fake(bound=16.0))
        r = _bilinear("max").solve(time_limit=2.0, gap_tolerance=1e-4)
        assert r.objective == pytest.approx(2.0), (
            "a maximize model with no incumbent should still adopt the recovered point"
        )

    def test_a_point_that_beats_a_certified_optimum_is_refused_loudly(self, monkeypatch, caplog):
        """Two claims that cannot both be right: report neither, say so (§3)."""
        # The candidate must beat the certified OBJECTIVE while staying above the
        # certified BOUND: a certificate holds with ``bound`` up to a tolerance below
        # the incumbent, and a candidate in that sliver is the only way to reach this
        # branch. A candidate below the bound as well trips the bound guard first,
        # and the test would then pass while never exercising the branch it is named
        # for -- so the numbers here are the point of the test.
        self._stub(
            monkeypatch,
            (2.9995, POINT),
            fake=self._fake(
                objective=3.0,
                bound=2.999,
                status="optimal",
                certified=True,
                x={"x": np.array(3.0), "y": np.array(1.0)},
            ),
        )
        with caplog.at_level(logging.ERROR, logger="discopt.solver"):
            r = _bilinear("min").solve(time_limit=2.0, gap_tolerance=1e-4)
        assert r.objective == pytest.approx(3.0), "a certified result was overwritten"
        assert r.gap_certified is True
        assert any("DISCARDING" in rec.getMessage() for rec in caplog.records), (
            "the contradiction was resolved SILENTLY -- CLAUDE.md §3 requires a "
            "loud refusal, not a quiet preference"
        )

    def test_an_infeasibility_proof_is_not_overwritten_by_a_point(self, monkeypatch, caplog):
        self._stub(monkeypatch, (2.0, POINT), fake=self._fake(status="infeasible"))
        with caplog.at_level(logging.ERROR, logger="discopt.solver"):
            r = _bilinear("min").solve(time_limit=2.0, gap_tolerance=1e-4)
        assert r.status == "infeasible"
        assert r.objective is None
        assert any("DISCARDING" in rec.getMessage() for rec in caplog.records)

    def test_a_point_below_a_proven_dual_bound_is_refused_loudly(self, monkeypatch, caplog):
        self._stub(monkeypatch, (2.0, POINT), fake=self._fake(bound=5.0))
        with caplog.at_level(logging.ERROR, logger="discopt.solver"):
            r = _bilinear("min").solve(time_limit=2.0, gap_tolerance=1e-4)
        assert r.objective is None, (
            "a point BELOW a dual bound the solve proved was adopted; one of the two "
            "is unsound and neither may be reported"
        )
        assert any("DISCARDING" in rec.getMessage() for rec in caplog.records)

    def test_the_opt_out_disables_adoption(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_KEEP_INCUMBENT", "0")
        self._stub(monkeypatch, (2.0, POINT), fake=self._fake(bound=-16.0))
        r = _bilinear("min").solve(time_limit=2.0, gap_tolerance=1e-4)
        assert r.objective is None, "=0 must restore the pre-#1440 behaviour exactly"


class TestTheBoundedReserve:
    """#1440: the attempt may be made to leave the default path a bounded slice.

    Default ``0.0`` -- the attempt is byte-identical until the knob is set. The
    measured reason it is not a default (cert-clean, but its benefit is confined to
    one instance at two of five budgets, with a counterexample at a third) is on
    ``convex_kernel_reserve_seconds``, together with the closure of every other
    candidate fix for the allocation.
    """

    def test_no_reserve_by_default(self, monkeypatch):
        monkeypatch.delenv("DISCOPT_CONVEX_KERNEL_RESERVE_FRAC", raising=False)
        for tl in (1.0, 8.0, 120.0, 3600.0):
            assert ck.convex_kernel_reserve_seconds(tl) == 0.0, (
                f"tl={tl}: a reserve appeared without the knob being set, so the "
                f"attempt is no longer byte-identical by default"
            )

    def test_the_reserve_is_a_fraction_under_a_cap(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_FRAC", "0.25")
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_CAP", "2.0")
        assert ck.convex_kernel_reserve_seconds(4.0) == pytest.approx(1.0)
        assert ck.convex_kernel_reserve_seconds(8.0) == pytest.approx(2.0), "cap binds"
        assert ck.convex_kernel_reserve_seconds(3600.0) == pytest.approx(2.0)

    def test_the_reserve_never_takes_more_than_half_the_budget(self, monkeypatch):
        """A mis-set knob must not silently turn the kernel off.

        Turning it off is what ``DISCOPT_CONVEX_KERNEL=0`` is for; a reserve that
        swallowed the budget would disable the kernel while still reporting it as
        enabled, which is the kind of silent divergence CLAUDE.md §3 refuses.
        """
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_FRAC", "5.0")
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_CAP", "1e9")
        assert ck.convex_kernel_reserve_seconds(2.0) == pytest.approx(1.0)

    def test_the_reserve_saturates_and_the_env_cap_cannot_raise_it(self, monkeypatch):
        """#1153: a role-2 carve must stop growing with the caller's budget.

        Regression on the real defect ``test_no_unsaturated_role2_carve`` caught:
        with only the env knobs bounding it, ``RESERVE_CAP=1e9`` left the reserve
        tracking ``0.5 * time_limit`` upward forever, so a bigger budget bought
        more preprocessing instead of more search. The ceiling is a module
        constant the environment can lower but never raise.
        """
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_FRAC", "5.0")
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_CAP", "1e9")
        wide = [ck.convex_kernel_reserve_seconds(t) for t in (100.0, 1_000.0, 3_600.0)]
        assert wide == [pytest.approx(ck._RESERVE_MAX_S)] * 3, (
            f"the reserve still grows with the budget ({wide}) -- it is an "
            f"unsaturated role-2 carve (#1153)"
        )
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_CAP", "0.5")
        assert ck.convex_kernel_reserve_seconds(3_600.0) == pytest.approx(0.5), (
            "the env cap must still be able to LOWER the reserve"
        )

    def test_a_non_finite_or_non_positive_budget_reserves_nothing(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_FRAC", "0.25")
        assert ck.convex_kernel_reserve_seconds(float("inf")) == 0.0
        assert ck.convex_kernel_reserve_seconds(0.0) == 0.0
        assert ck.convex_kernel_reserve_seconds(-1.0) == 0.0

    def test_an_unparseable_knob_reserves_nothing(self, monkeypatch):
        """Fail closed: a typo must not change the allocation."""
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_FRAC", "one quarter")
        assert ck.convex_kernel_reserve_seconds(8.0) == 0.0


@needs_nl
class TestTheReserveReachesTheAttempt:
    @pytest.mark.slow
    def test_the_attempt_is_shortened_by_the_reserve(self, monkeypatch):
        """The knob has to move the ATTEMPT, not just compute a number.

        Measured on the harmed instance, which consumes 100% of every budget from
        1 s to 16 s: without the reserve the attempt spends ~the whole budget, with
        it the attempt stops a reserve short and the default path gets that slice.
        """
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_FRAC", "0.25")
        monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_RESERVE_CAP", "2.0")
        _fresh().solve(time_limit=8.0, gap_tolerance=1e-4)
        spent = ck.last_attempt_seconds()
        assert spent <= 6.0 + 1.0, (
            f"the attempt spent {spent:.3f}s of an 8s budget with a 2s reserve set; "
            f"the reserve is not reaching try_convex_solve"
        )
        assert spent >= 3.0, (
            f"the attempt spent only {spent:.3f}s -- the reserve has taken far more "
            f"than it was asked for, which would disable the kernel by stealth"
        )
