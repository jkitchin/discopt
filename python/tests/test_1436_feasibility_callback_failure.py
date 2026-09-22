"""Regression suite for #1436: a feasibility-defining callback that RAISES must
not yield a certificate.

`lazy_constraints` and `incumbent_callback` do not advise the search — they
define which points are acceptable. Before this fix, `solver.py` caught a raising
callback, logged a warning, and let the node proceed "as normal (no cut, no
rejection)", so the tree searched the model WITHOUT the restriction the caller
asked for and then certified the result. Measured on `min -x - y` over
`{0..3}^2`, where the callback is the only thing imposing `x + y <= 2`
(true optimum **-2**):

    lazy_constraints working        -> status="unknown",  objective=None
    lazy_constraints raises KeyError-> status="optimal",  objective=-6.0,
                                       gap_certified=True, x = {x: 3, y: 3}

`x = y = 3` is a point the callback itself excludes. The only signal was a
`logger.warning`, and logging is unconfigured in most scripts.

The contrast arm is why the fix is narrow: `cut_callback` raising is SOUND and
keeps failing soft. A cut is a *strengthening*, so losing it can only loosen the
bound, never invalidate it — measured on the #1278 spatial model with all 22
invocations raising, objective and bound came back bit-identical to the
no-callback run. `node_callback` is an observer and is likewise untouched.

Structure
---------
* `TestFeasibilityCallbackFailureIsRefused` — the two reproducers. These FAIL
  before the change (they return the false certificate instead of raising).
* `TestTheRefusalIsScoped` — `cut_callback` and `node_callback` still fail soft,
  a working callback is unaffected, and a solve with no callback at all is
  untouched.
* `TestTheRecordDoesNotLeak` — the failure record is per-solve and per-thread, so
  one solve's broken callback cannot refuse an unrelated later result.
"""

from __future__ import annotations

import threading

import discopt.modeling as dm
import pytest
from discopt.callbacks import CutResult
from discopt.solver import FeasibilityCallbackError

# min -x - y over {0..3}^2. Unconstrained optimum -6 at (3,3); with the callback's
# x + y <= 2 enforced the optimum is -2. The gap between those two numbers is
# exactly what a swallowed callback failure used to certify.
_RELAXED_OPT = -6.0


def _milp():
    m = dm.Model("cb1436")
    x = m.integer("x", lb=0, ub=3)
    y = m.integer("y", lb=0, ub=3)
    m.minimize(-x - y)
    return m, x, y


class TestFeasibilityCallbackFailureIsRefused:
    def test_lazy_constraints_failure_is_refused(self):
        """Before #1436 this returned `optimal` at -6.0 with `gap_certified=True`."""
        m, x, y = _milp()
        calls = []

        def boom(ctx, model):
            calls.append(1)
            raise KeyError("a typo in the user's callback")

        with pytest.raises(FeasibilityCallbackError) as exc:
            m.solve(time_limit=25, lazy_constraints=boom)

        assert calls, "PROBE NEVER FIRED: the callback was never invoked"
        assert "lazy_constraints" in str(exc.value)
        # The original error must be chained, not replaced — it is what the user
        # needs to find the bug in their own code.
        assert isinstance(exc.value.__cause__, KeyError)

    def test_incumbent_callback_failure_is_refused(self):
        """A failed veto means the point was accepted unvetted."""
        m, x, y = _milp()
        calls = []

        def boom(ctx, model, solution):
            calls.append(1)
            raise AttributeError("a typo in the user's veto")

        with pytest.raises(FeasibilityCallbackError) as exc:
            m.solve(time_limit=25, incumbent_callback=boom)

        assert calls, "PROBE NEVER FIRED: the callback was never invoked"
        assert "incumbent_callback" in str(exc.value)
        assert isinstance(exc.value.__cause__, AttributeError)

    def test_the_message_counts_the_failures(self):
        """The count is the actionable part: it says whether this was one bad node
        or every node, which is the difference between a data-dependent bug and a
        callback that never worked."""
        m, x, y = _milp()

        def boom(ctx, model):
            raise ValueError("always")

        with pytest.raises(FeasibilityCallbackError) as exc:
            m.solve(time_limit=25, lazy_constraints=boom)
        msg = str(exc.value)
        assert "x)" in msg, msg  # "lazy_constraints (Nx)"
        assert "certify" in msg or "certif" in msg, msg

    def test_a_partially_failing_callback_is_still_refused(self):
        """The dangerous shape: the callback works at most nodes, so the solve
        looks normal, and fails at one — whose restriction is then the one missing."""
        m, x, y = _milp()
        state = {"n": 0}

        def flaky(ctx, model):
            state["n"] += 1
            if state["n"] == 2:
                raise RuntimeError("failed on the second node only")
            return [CutResult([(x, 1.0), (y, 1.0)], "<=", 2.0)]

        with pytest.raises(FeasibilityCallbackError):
            m.solve(time_limit=25, lazy_constraints=flaky)
        assert state["n"] >= 2, f"PROBE NEVER FIRED: only {state['n']} invocations"


class TestTheRefusalIsScoped:
    def test_cut_callback_still_fails_soft(self):
        """A cut is a strengthening, so losing one can only loosen the bound.
        Refusing here would break the #1278 contract for no soundness gain."""
        m = dm.Model("spatial1436")
        v = [m.continuous(f"v{i}", lb=0.1, ub=3.0) for i in range(3)]
        m.subject_to(v[0] + v[1] + v[2] == 4.0)
        m.minimize(v[0] * v[1] * v[2] - dm.exp(v[0]) - dm.sin(3 * v[1]))
        calls = []

        def boom(ctx, model):
            calls.append(1)
            raise RuntimeError("user bug")

        r = m.solve(time_limit=60, cut_callback=boom)
        assert calls, "PROBE NEVER FIRED: the cut callback was never invoked"
        assert r.status == "optimal", r.status

    def test_node_callback_still_fails_soft(self):
        """A node callback is an observer; it cannot change the feasible set."""
        m, x, y = _milp()
        calls = []

        def boom(ctx, model):
            calls.append(1)
            raise RuntimeError("user bug")

        r = m.solve(time_limit=25, node_callback=boom)
        assert r.status in ("optimal", "feasible", "node_limit", "unknown"), r.status

    def test_a_working_lazy_callback_is_unaffected(self):
        """The guard must not fire on a callback that never raises."""
        m, x, y = _milp()
        r = m.solve(
            time_limit=25,
            lazy_constraints=lambda ctx, model: [CutResult([(x, 1.0), (y, 1.0)], "<=", 2.0)],
        )
        # Whatever it concludes, it must not be the relaxed optimum certified.
        assert not (r.status == "optimal" and r.objective == pytest.approx(_RELAXED_OPT)), (
            f"the enforced model was certified at the RELAXED optimum {_RELAXED_OPT}"
        )

    def test_a_solve_with_no_callback_is_untouched(self):
        """The overwhelmingly common case pays nothing."""
        m, x, y = _milp()
        r = m.solve(time_limit=25)
        assert r.status == "optimal"
        assert r.objective == pytest.approx(_RELAXED_OPT, abs=1e-6)


class TestTheRecordDoesNotLeak:
    def test_a_later_solve_is_not_refused(self):
        """The record is reset on entry to every solve. Without that, one broken
        callback would refuse every later result on the thread — a guard that
        fires on innocent solves gets switched off."""
        m, x, y = _milp()
        with pytest.raises(FeasibilityCallbackError):
            m.solve(
                time_limit=25,
                lazy_constraints=lambda ctx, model: (_ for _ in ()).throw(RuntimeError("boom")),
            )
        m2, _, _ = _milp()
        r = m2.solve(time_limit=25)
        assert r.status == "optimal", (
            f"a later, callback-free solve was affected by the earlier failure: {r.status}"
        )
        assert r.objective == pytest.approx(_RELAXED_OPT, abs=1e-6)

    def test_a_failure_on_another_thread_does_not_refuse_this_one(self):
        """The record is `threading.local`: two concurrent solves must not read
        each other's failures."""
        done = {}

        def broken():
            m, x, y = _milp()
            try:
                m.solve(
                    time_limit=25,
                    lazy_constraints=lambda ctx, model: (_ for _ in ()).throw(RuntimeError("boom")),
                )
                done["broken"] = "not refused"
            except FeasibilityCallbackError:
                done["broken"] = "refused"

        t = threading.Thread(target=broken)
        t.start()
        t.join(timeout=120)
        assert done.get("broken") == "refused", done

        m2, _, _ = _milp()
        r = m2.solve(time_limit=25)
        assert r.status == "optimal", (
            f"a failure on another thread refused this thread's solve: {r.status}"
        )
