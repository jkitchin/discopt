"""Increment-2 regression: the warm-started Rust whole-search MILP engine
(``nlp_solver="simplex"``).

Two behaviors are pinned:

1. **Continuous-repair root dive.** On a weak-relaxation (big-M) MILP the LP
   relaxation is fractional and plain rounding finds no feasible point, so
   without a dive the search returns *no incumbent*. The Rust driver's dive
   fixes integers one at a time and re-solves the continuous variables between
   fixes (with a single-variable backtrack), so the engine now finds the
   optimal-valued incumbent. (Before, it returned ``node_limit``/no incumbent.)

2. **Feasibility gate.** ``_solve_milp_simplex`` verifies the returned point
   against the model's own rows/bounds/integrality and defers (returns ``None``)
   on violation, so a wrong "optimal" can never escape — the caller falls back
   to a sound engine. This guards against the whole-search returning an
   infeasible point on a zero-objective feasibility MILP.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402


def _bigM_jobshop(n_jobs: int, n_machines: int) -> dm.Model:
    rng = np.random.default_rng(0)
    proc = rng.integers(1, 5, size=(n_jobs, n_machines)).astype(float)
    M = float(proc.sum())
    m = dm.Model(f"js_{n_jobs}x{n_machines}")
    s = m.continuous("start", shape=(n_jobs, n_machines), lb=0, ub=M)
    C = m.continuous("makespan", lb=0, ub=M)
    n_pairs = n_jobs * (n_jobs - 1) // 2
    z = m.binary("order", shape=(n_pairs * n_machines,))
    m.minimize(C)
    for i in range(n_jobs):
        for k in range(n_machines):
            m.subject_to(C >= s[i, k] + float(proc[i, k]))
    pi = 0
    for i in range(n_jobs):
        for j in range(i + 1, n_jobs):
            for k in range(n_machines):
                idx = pi * n_machines + k
                m.subject_to(s[i, k] + float(proc[i, k]) <= s[j, k] + M * (1 - z[idx]))
                m.subject_to(s[j, k] + float(proc[j, k]) <= s[i, k] + M * z[idx])
            pi += 1
    return m


class TestSimplexRootDive:
    def test_finds_incumbent_on_bigM(self):
        """The Rust whole-search finds an incumbent on a moderate big-M jobshop
        via the continuous-repair dive — no incumbent (objective None) is the
        regression this guards against.

        Scope: this exercises the *opt-in* ``nlp_solver="simplex"`` whole-search.
        Its greedy dive is best-effort on big-M — on larger instances a greedy
        fix order can lock a cyclic precedence and dead-end (the chosen LP vertex
        influences whether it does), so the guarantee here is the moderate 4x3
        case. The universal default path (Python ``_solve_milp_bb`` simplex
        nodes) is what reliably finds big-M incumbents at larger sizes."""
        r = _bigM_jobshop(4, 3).solve(nlp_solver="simplex", time_limit=30)
        assert r.status in ("optimal", "feasible")
        assert r.objective is not None  # an incumbent was found (dive worked)

    def test_simplex_matches_highs(self):
        """The simplex engine's incumbent matches the HiGHS optimum."""
        pytest.importorskip("highspy")
        r_s = _bigM_jobshop(4, 3).solve(nlp_solver="simplex", time_limit=30)
        r_h = _bigM_jobshop(4, 3).solve(nlp_solver="ipm", time_limit=30)
        assert r_s.objective is not None and r_h.objective is not None
        assert abs(r_s.objective - r_h.objective) < 1e-3


class TestFeasibilityGate:
    def test_gate_defers_on_infeasible_result(self, monkeypatch):
        """If the Rust whole-search returns a point that violates the model's
        constraints, ``_solve_milp_simplex`` returns None (defers) rather than
        reporting a wrong 'optimal'."""
        import time

        import discopt._rust as _rust

        m = dm.Model("knap")
        x = m.binary("x", shape=(4,))
        m.minimize(-sum(2 * x[i] for i in range(4)))
        m.subject_to(sum(x[i] for i in range(4)) <= 2)

        # Force the Rust solver to claim "optimal" with an infeasible point
        # (all ones violates sum<=2). The gate must catch it and defer.
        def _lying(*a, **k):
            return ("optimal", np.ones(4, dtype=np.float64), -8.0, -8.0, 1, 1)

        monkeypatch.setattr(_rust, "solve_milp_py", _lying)
        res = S._solve_milp_simplex(m, 30.0, 1e-9, 1000, time.perf_counter())
        assert res is None  # infeasible "optimal" rejected by the gate

    def test_gate_accepts_feasible_result(self):
        """A genuinely feasible/optimal simplex result passes the gate."""
        m = dm.Model("knap_ok")
        x = m.binary("x", shape=(4,))
        m.minimize(-sum(2 * x[i] for i in range(4)))
        m.subject_to(sum(x[i] for i in range(4)) <= 2)
        r = m.solve(nlp_solver="simplex", time_limit=30)
        assert r.status == "optimal"
        assert abs(r.objective - (-4.0)) < 1e-4  # pick any 2 -> -4


class TestEqualityRowSoundness:
    """The rounding heuristic must respect *equality* rows.

    A zero-objective feasibility MILP ``Σx == k`` is the failure that motivated
    the fix: the old heuristic checked only ``act <= b`` (a `<=` test), so
    all-zeros (``0 <= k``) passed and was injected as a bogus "optimal" even
    though ``0 != k``. The check now tests the residual against each row's slack
    range, which is empty ([0,0]) for an equality row — so all-zeros is correctly
    rejected and the engine returns a genuinely feasible point.
    """

    def test_whole_search_sound_on_equality_feasibility(self):
        m = dm.Model("eqfeas")
        x = m.binary("x", shape=(8,))
        m.minimize(0.0 * x[0])  # pure feasibility (constant objective)
        m.subject_to(sum(x[i] for i in range(8)) == 4)
        r = m.solve(nlp_solver="simplex", time_limit=30)
        assert r.status in ("optimal", "feasible")
        xv = np.round(np.asarray(r.x["x"]).ravel())
        assert int(xv.sum()) == 4  # feasible — not the bogus all-zeros


class TestReentryOnUncertifiedFeasible:
    """Issue #698: on an *uncertified* ``feasible`` the engine used to return
    with the bounded first slice (a #291 stall guard) and DISCARD the rest of
    the caller's ``time_limit``. It now re-enters the same engine with the
    remaining budget, seeded with the best point so far, and keeps run 2 only
    when it certifies or is strictly better.

    Both runs are driven through a monkeypatched ``solve_milp_py`` so the
    behavior is pinned without depending on any particular instance's runtime.
    Call routing: the first non-empty-``int_idx`` call is run 1; the second is
    the re-entry; a call with an *empty* ``int_idx`` is the root-relaxation
    instrumentation LP (integers relaxed) and is answered separately.
    """

    @staticmethod
    def _knap() -> dm.Model:
        m = dm.Model("knap_reentry")
        x = m.binary("x", shape=(4,))
        m.minimize(-2 * sum(x[i] for i in range(4)))  # min -2 Σx  (opt = -4)
        m.subject_to(sum(x[i] for i in range(4)) <= 2)
        return m

    def test_reentry_adopts_certifying_run(self, monkeypatch):
        """Run 1 returns an uncertified ``feasible`` (one item, obj -2, loose
        bound); the re-entry certifies the true optimum (-4). The certified,
        strictly-better result is adopted and node counts are summed."""
        import time

        import discopt._rust as _rust

        state = {"n": 0}

        def mock(*a, **k):
            if np.asarray(a[5]).size == 0:  # root-relaxation LP (integers relaxed)
                return ("optimal", np.zeros(4), -8.0, -8.0, 1, 1)
            state["n"] += 1
            if state["n"] == 1:
                return ("feasible", np.array([1.0, 0, 0, 0]), -2.0, -8.0, 3, 5)
            return ("optimal", np.array([1.0, 1.0, 0, 0]), -4.0, -4.0, 100, 50)

        monkeypatch.setattr(_rust, "solve_milp_py", mock)
        res = S._solve_milp_simplex(self._knap(), 30.0, 1e-9, 1000, time.perf_counter())
        assert state["n"] == 2  # the re-entry actually ran
        assert res is not None
        assert res.status == "optimal"
        assert res.gap_certified is True
        assert abs(res.objective - (-4.0)) < 1e-6
        assert res.node_count == 103  # 3 (run 1) + 100 (re-entry)

    def test_reentry_skipped_without_more_budget(self, monkeypatch):
        """With no more time than run 1's own slice, a restart cannot beat it,
        so the re-entry is skipped and run 1's result is returned verbatim."""
        import time

        import discopt._rust as _rust

        state = {"n": 0}

        def mock(*a, **k):
            if np.asarray(a[5]).size == 0:
                return ("optimal", np.zeros(4), -8.0, -8.0, 1, 1)
            state["n"] += 1
            return ("feasible", np.array([1.0, 0, 0, 0]), -2.0, -8.0, 7, 5)

        monkeypatch.setattr(_rust, "solve_milp_py", mock)
        # time_limit == first-slice floor: _milp_budget == remaining, so re-entry
        # is skipped by the "no more time than run 1 had" guard.
        res = S._solve_milp_simplex(self._knap(), 0.5, 1e-9, 1000, time.perf_counter())
        assert state["n"] == 1  # no re-entry
        assert res is not None
        assert res.status == "feasible"
        assert res.node_count == 7

    def test_reentry_not_adopted_when_not_better(self, monkeypatch):
        """A re-entry that neither certifies nor strictly improves (same obj,
        same bound, still ``feasible``) is discarded — run 1's result stands,
        and the bound is never regressed."""
        import time

        import discopt._rust as _rust

        state = {"n": 0}

        def mock(*a, **k):
            if np.asarray(a[5]).size == 0:
                return ("optimal", np.zeros(4), -8.0, -8.0, 1, 1)
            state["n"] += 1
            # Both runs report the same feasible point / bound.
            return ("feasible", np.array([1.0, 1.0, 0, 0]), -4.0, -8.0, 4, 5)

        monkeypatch.setattr(_rust, "solve_milp_py", mock)
        res = S._solve_milp_simplex(self._knap(), 30.0, 1e-9, 1000, time.perf_counter())
        assert state["n"] == 2  # re-entry ran...
        assert res is not None
        assert res.status == "feasible"  # ...but was not adopted
        assert abs(res.objective - (-4.0)) < 1e-6
        assert res.node_count == 4  # run 1's nodes only; re-entry nodes not merged
