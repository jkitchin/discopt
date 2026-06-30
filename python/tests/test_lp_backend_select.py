"""Backend selection for matrix-form LP/QP solves (roadmap P4 / 'just POUNCE').

discopt must run with only POUNCE installed (no HiGHS). The shared selector
in ``discopt.solvers.lp_backend`` picks a signature-compatible engine and
falls back to whichever is importable, so consumers (OBBT, ...) work either
way; ``prefer_pounce`` flips the preference for POUNCE-only mode.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest
from discopt.solvers import lp_backend


def test_default_prefers_highs_when_present():
    pytest.importorskip("highspy")
    assert lp_backend.get_lp_solver().__module__ == "discopt.solvers.lp_highs"


def test_prefer_pounce_selects_pounce():
    pytest.importorskip("pounce")
    assert lp_backend.get_lp_solver(prefer_pounce=True).__module__ == "discopt.solvers.lp_pounce"
    assert lp_backend.get_qp_solver(prefer_pounce=True).__module__ == "discopt.solvers.qp_pounce"


def test_default_qp_prefers_pounce():
    """Issue #359: the QP seam is HiGHS-free by default — POUNCE-first even when
    HiGHS is installed (unlike the LP/MILP selectors, which stay HiGHS-first)."""
    pytest.importorskip("pounce")
    assert lp_backend.get_qp_solver().__module__ == "discopt.solvers.qp_pounce"


def test_qp_selector_is_pounce_only_no_highs_fallback(monkeypatch):
    """Issue #359: ``qp_highs`` was removed, so the QP seam is POUNCE-only — with
    POUNCE unavailable it raises rather than falling back to HiGHS."""
    real_import = builtins.__import__

    def no_pounce(name, *a, **k):
        if "qp_pounce" in name or name == "pounce":
            raise ImportError("POUNCE unavailable (simulated)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pounce)
    with pytest.raises(ImportError, match="No QP backend"):
        lp_backend.get_qp_solver()


def _without_highs(monkeypatch):
    """Make any HiGHS import raise, simulating a POUNCE-only install."""
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if any(s in name for s in ("lp_highs", "milp_highs")) or name == "highspy":
            raise ImportError("HiGHS unavailable (simulated POUNCE-only install)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_default_milp_prefers_highs_when_present():
    pytest.importorskip("highspy")
    assert lp_backend.get_milp_solver().__module__ == "discopt.solvers.milp_highs"


def test_prefer_pounce_milp_uses_self_hosted_bb():
    pytest.importorskip("pounce")
    assert (
        lp_backend.get_milp_solver(prefer_pounce=True).__module__ == "discopt.solvers.milp_pounce"
    )


class TestPounceOnlyInstall:
    def test_lp_selector_falls_back_to_pounce(self, monkeypatch):
        pytest.importorskip("pounce")
        _without_highs(monkeypatch)
        assert lp_backend.get_lp_solver().__module__ == "discopt.solvers.lp_pounce"

    def test_milp_selector_falls_back_to_pounce(self, monkeypatch):
        pytest.importorskip("pounce")
        _without_highs(monkeypatch)
        assert lp_backend.get_milp_solver().__module__ == "discopt.solvers.milp_pounce"

    def test_oa_master_is_highs_free(self, monkeypatch):
        """The OA convex-MINLP master MILP solves without HiGHS (via the
        POUNCE B&B adapter)."""
        pytest.importorskip("pounce")
        import discopt.modeling as dm
        from discopt.solvers import lp_backend as lb
        from discopt.solvers.milp_pounce import solve_milp as _pounce_milp

        monkeypatch.setattr(
            lb,
            "get_milp_solver",
            lambda prefer_pounce=False, backend="auto": _pounce_milp,
        )
        m = dm.Model("oa_highsfree")
        x = m.integer("x", lb=0, ub=3)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize((x - 1.5) ** 2 + y)
        m.subject_to(x + y >= 2)
        from discopt.solvers.oa import solve_oa

        r = solve_oa(m, time_limit=30)
        assert r.status in ("optimal", "feasible")
        assert abs(r.objective - 0.25) < 1e-3  # x=2, y=0

    def test_qp_selector_falls_back_to_pounce(self, monkeypatch):
        pytest.importorskip("pounce")
        _without_highs(monkeypatch)
        assert lp_backend.get_qp_solver().__module__ == "discopt.solvers.qp_pounce"

    def test_neither_backend_raises(self, monkeypatch):
        real_import = builtins.__import__

        def no_backend(name, *a, **k):
            if any(s in name for s in ("lp_highs", "lp_pounce", "highspy", "pounce")):
                raise ImportError("no backend")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_backend)
        with pytest.raises(ImportError, match="No LP backend"):
            lp_backend.get_lp_solver()


class TestHighspyConsumersRetired:
    """The remaining raw-``highspy`` LP/QP/MILP consumers now route through the
    backend selector, so they run with only POUNCE installed (roadmap Phase 4).
    """

    def test_partition_selection_milp_is_highs_free(self, monkeypatch):
        """The vertex-cover MILP in partition selection solves without HiGHS."""
        pytest.importorskip("pounce")
        _without_highs(monkeypatch)
        from discopt._jax import partition_selection as ps

        # Disable the greedy fallback so any returned cover must come from the
        # MILP — solved via the POUNCE B&B, since HiGHS is unavailable.
        def _no_greedy(*a, **k):
            raise AssertionError("greedy fallback used; MILP path did not run")

        monkeypatch.setattr(ps, "_greedy_vertex_cover", _no_greedy)

        terms = [(0, 1), (1, 2), (0, 2)]  # triangle: min vertex cover is 2
        cover = ps._solve_vertex_cover_milp([0, 1, 2], terms)
        cset = set(cover)
        assert all(any(v in cset for v in t) for t in terms)  # valid cover
        assert len(cover) == 2  # and minimal

    def test_strong_branch_lp_is_highs_free(self, monkeypatch):
        """Strong branching's LP probes run on POUNCE when HiGHS is absent."""
        pytest.importorskip("pounce")
        _without_highs(monkeypatch)
        import discopt.solver as S

        class _Eval:  # minimal evaluator: linear objective, no constraints
            n_constraints = 0

            def evaluate_gradient(self, x):
                return np.array([-1.0, -1.0])

        best = S._strong_branch_lp(
            _Eval(),
            solution=np.array([0.5, 0.5]),
            node_lb=np.array([0.0, 0.0]),
            node_ub=np.array([1.0, 1.0]),
            candidate_var_indices=np.array([0, 1]),
            parent_lb=-1.0,
            prefer_pounce=True,
        )
        assert best in (0, 1)  # picked a candidate via POUNCE LP probes

    def test_gdp_big_m_lp_is_highs_free(self, monkeypatch):
        """Multiple-big-M's LP-based tightening runs without HiGHS, via the
        exact (Rust simplex) oracle, and is tighter than the interval fallback.

        Big-M from an LP optimum must use an exact oracle, never the POUNCE IPM
        (#145): a too-small M cuts off the inactive disjunct's feasible points.
        With HiGHS absent the path must still produce the LP-tight M from the
        self-hosted simplex.
        """
        pytest.importorskip("pounce")
        from discopt.solvers.lp_backend import get_exact_lp_solver

        _without_highs(monkeypatch)
        if get_exact_lp_solver() is None:
            pytest.skip("no exact LP oracle (neither Rust simplex nor HiGHS)")
        import discopt.modeling as dm
        from discopt._jax.gdp_reformulate import (
            _compute_big_m,
            _compute_big_m_lp,
            _precompute_lp_relaxation,
        )
        from discopt.modeling.core import Constraint

        m = dm.Model("mbm")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.subject_to(x + y <= 8)
        lp_data = _precompute_lp_relaxation(m)
        assert lp_data is not None
        con = Constraint(body=x + y - dm.core.Constant(5.0), sense="<=", rhs=0.0)

        # body = x + y - 5, maximized over x + y <= 8 -> 3, so M = 3 * 1.01.
        m_lp = _compute_big_m_lp(con, m, lp_data)
        assert np.isfinite(m_lp)
        assert m_lp == pytest.approx(3.0 * 1.01, abs=1e-3)
        # The LP-tight M must be a sound under-bound of the interval fallback
        # (interval: |10 + 10 - 5| = 15) -> strictly tighter, never larger.
        assert m_lp < _compute_big_m(con, m)


class TestObbtRetired:
    """OBBT bound tightening requires an *exact* LP oracle.

    OBBT tightens a variable's bound to the optimum of ``min``/``max x_i`` over
    the relaxation polytope, so the subproblem LP must be solved to its true
    optimum to stay sound. The POUNCE IPM returns an analytic-center objective
    that can be wrong on ill-conditioned LPs while reporting ``OPTIMAL`` (#145),
    so OBBT routes through ``get_exact_lp_solver()`` — discopt's own pure-Rust
    simplex (or HiGHS) but never the IPM — regardless of ``prefer_pounce``, and
    is a sound no-op when no exact oracle is available. The Rust simplex needs no
    external HiGHS, so these run in a POUNCE-only install.
    """

    def _model(self):
        import discopt.modeling as dm

        m = dm.Model("obbt")
        m.continuous("x", lb=0, ub=10)
        m.continuous("y", lb=0, ub=10)
        xv, yv = m._variables[0], m._variables[1]
        m.minimize(xv + yv)
        m.subject_to(xv + yv >= 3)
        m.subject_to(xv <= 4)
        return m

    def test_obbt_tightens_via_exact_oracle(self):
        # Tightening requires the exact (self-hosted simplex) oracle;
        # ``prefer_pounce`` is a no-op for the solver selection (#145).
        from discopt.solvers.lp_backend import get_exact_lp_solver

        if get_exact_lp_solver() is None:
            pytest.skip("no exact LP oracle (neither Rust simplex nor HiGHS)")
        from discopt._jax.obbt import run_obbt

        res = run_obbt(self._model(), prefer_pounce=True, time_limit_per_lp=5.0)
        # x is capped at 4 by the constraint; OBBT must discover ub_x <= 4.
        assert res.tightened_ub[0] == pytest.approx(4.0, abs=1e-4)
        assert res.n_lp_solves >= 1

    def test_obbt_ignores_prefer_pounce(self):
        # Both calls must route through the exact oracle, so the tightened box
        # is identical regardless of ``prefer_pounce`` (it no longer selects the
        # IPM backend for OBBT — #145).
        from discopt.solvers.lp_backend import get_exact_lp_solver

        if get_exact_lp_solver() is None:
            pytest.skip("no exact LP oracle (neither Rust simplex nor HiGHS)")
        from discopt._jax.obbt import run_obbt

        rp = run_obbt(self._model(), prefer_pounce=True, time_limit_per_lp=5.0)
        rh = run_obbt(self._model(), prefer_pounce=False, time_limit_per_lp=5.0)
        np.testing.assert_allclose(rp.tightened_ub, rh.tightened_ub, atol=1e-4)
        np.testing.assert_allclose(rp.tightened_lb, rh.tightened_lb, atol=1e-4)
