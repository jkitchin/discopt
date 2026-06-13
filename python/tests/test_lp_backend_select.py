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


def _without_highs(monkeypatch):
    """Make any HiGHS import raise, simulating a POUNCE-only install."""
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if any(s in name for s in ("lp_highs", "qp_highs", "milp_highs")) or name == "highspy":
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

        monkeypatch.setattr(lb, "get_milp_solver", lambda prefer_pounce=False: _pounce_milp)
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


class TestObbtRetired:
    """OBBT runs through the backend seam, so it works POUNCE-only."""

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

    def test_obbt_via_pounce_tightens(self):
        pytest.importorskip("pounce")
        from discopt._jax.obbt import run_obbt

        res = run_obbt(self._model(), prefer_pounce=True, time_limit_per_lp=5.0)
        # x is capped at 4 by the constraint; OBBT must discover ub_x <= 4.
        assert res.tightened_ub[0] == pytest.approx(4.0, abs=1e-4)
        assert res.n_lp_solves >= 1

    def test_obbt_matches_across_backends(self):
        pytest.importorskip("pounce")
        pytest.importorskip("highspy")
        from discopt._jax.obbt import run_obbt

        rp = run_obbt(self._model(), prefer_pounce=True, time_limit_per_lp=5.0)
        rh = run_obbt(self._model(), prefer_pounce=False, time_limit_per_lp=5.0)
        np.testing.assert_allclose(rp.tightened_ub, rh.tightened_ub, atol=1e-4)
        np.testing.assert_allclose(rp.tightened_lb, rh.tightened_lb, atol=1e-4)
