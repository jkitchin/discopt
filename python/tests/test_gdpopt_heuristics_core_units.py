"""Coverage tests for the GDPopt LOA solver, primal heuristics, and modeling core.

Issue #87 (coverage restoration), final round. Targets:

* ``discopt/solvers/gdpopt_loa.py`` — end-to-end LOA solves on tiny disjunctive
  models with hand-verifiable optima, plus unit tests of the gap/cut/master
  helpers and the rigorous-infeasibility split (C-35).
* ``discopt/_jax/primal_heuristics.py`` — feasibility of every returned point is
  re-verified against the model, and returned objectives are never better than
  the (enumerable) true optimum.
* ``discopt/modeling/core.py`` — documented validation/error branches raise with
  their documented messages; behavior-preserving paths are checked by value.

Two genuine-bug probes are marked ``xfail(strict=False)`` and assert the CORRECT
behavior (see the test docstrings): the general-integer no-good cut and the
timeout-as-infeasible LOA result.
"""

from __future__ import annotations

import importlib.util
import time
import types

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax import primal_heuristics as ph
from discopt._jax.nlp_evaluator import NLPEvaluator, cached_evaluator
from discopt.modeling import core
from discopt.modeling.core import SolveResult
from discopt.solvers import NLPResult, SolveStatus
from discopt.solvers import gdpopt_loa as loa

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _flat_point(model, result_x: dict) -> np.ndarray:
    """Flatten a SolveResult.x dict into the model's flat variable order."""
    chunks = []
    for v in model._variables:
        chunks.append(np.asarray(result_x[v.name], dtype=np.float64).reshape(-1))
    return np.concatenate(chunks)


def _assert_feasible(model, x_flat: np.ndarray, tol: float = 1e-5) -> None:
    """Independently verify constraint + bound feasibility of a flat point."""
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    ev = cached_evaluator(model)
    lb, ub = ph._get_variable_bounds(model)
    assert np.all(x_flat >= lb - tol)
    assert np.all(x_flat <= ub + tol)
    if ev.n_constraints:
        g = np.asarray(ev.evaluate_constraints(x_flat), dtype=np.float64)
        cl, cu = (np.asarray(b, dtype=np.float64) for b in _infer_constraint_bounds(ev))
        assert np.all(g >= cl - tol), f"constraint lower violated: {g} vs {cl}"
        assert np.all(g <= cu + tol), f"constraint upper violated: {g} vs {cu}"


def _raising_backend(evaluator, x0, options=None):
    raise RuntimeError("backend boom")


def _mk_backend(x, status=SolveStatus.OPTIMAL, objective=0.0):
    """A fake NLP backend that always returns the given point."""

    def backend(evaluator, x0, options=None):
        return NLPResult(status=status, x=None if x is None else np.asarray(x), objective=objective)

    return backend


# ─────────────────────────────────────────────────────────────────────────────
# GDPopt LOA: end-to-end solves on tiny, hand-verified disjunctive models
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
class TestGDPoptLOASolves:
    def test_simple_disjunction_optimal(self):
        """min x s.t. (x<=3) or (x>=7): optimum 0 in the first disjunct."""
        m = dm.Model("loa_simple")
        x = m.continuous("x", lb=0, ub=10)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.minimize(x)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status == "optimal"
        assert r.objective == pytest.approx(0.0, abs=1e-5)
        # Certificate invariant: dual bound never above the incumbent.
        assert r.bound is not None and r.bound <= r.objective + 1e-6
        assert float(np.asarray(r.x["x"])) == pytest.approx(0.0, abs=1e-5)

    def test_linear_eq_and_ge_rows_in_master(self):
        """Linear == and >= rows must reach the master MILP.

        x + u == 5 and x - u >= 1 force x >= 3; with the disjunct x <= 3 this
        pins x = 3, u = 2 (the x >= 7 disjunct is infeasible since u >= 0
        forces x <= 5). min u -> 2.
        """
        m = dm.Model("loa_rows")
        x = m.continuous("x", lb=0, ub=10)
        u = m.continuous("u", lb=0, ub=10)
        m.subject_to(x + u == 5)
        m.subject_to(x - u >= 1)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.minimize(u)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status in ("optimal", "feasible")
        assert r.objective == pytest.approx(2.0, abs=1e-4)
        assert float(np.asarray(r.x["x"])) == pytest.approx(3.0, abs=1e-4)

    def test_nonlinear_convex_objective_epigraph(self):
        """min (x-5)^2 over (x<=3) or (x>=7): both disjuncts give 4."""
        m = dm.Model("loa_nlobj")
        x = m.continuous("x", lb=0, ub=10)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.minimize((x - 5) ** 2)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status in ("optimal", "feasible")
        assert r.objective == pytest.approx(4.0, abs=1e-3)
        if r.bound is not None:
            assert r.bound <= r.objective + 1e-6

    def test_nonconvex_objective_disables_bound(self):
        """A nonconvex objective must not produce a master lower bound."""
        m = dm.Model("loa_nonconvex")
        x = m.continuous("x", lb=0, ub=2)
        m.either_or([[x <= 1], [x >= 2]], name="choice")
        m.minimize(-(x**2))
        r = m.solve(time_limit=30, gdp_method="loa", max_nodes=6)
        assert r.bound is None
        assert r.gap is None
        if r.objective is not None:
            # True optimum is -(2^2) = -4 at x = 2; never claim better.
            assert r.objective >= -4.0 - 1e-6

    def test_infeasible_by_master(self):
        """Linear rows 4 <= x <= 6 contradict both disjuncts: infeasible."""
        m = dm.Model("loa_infeas")
        x = m.continuous("x", lb=0, ub=10)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.subject_to(x >= 4)
        m.subject_to(x <= 6)
        m.minimize(x)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status == "infeasible"
        assert r.objective is None

    def test_binary_no_good_cut_reaches_optimum(self):
        """Bilinear binary y1*y2 >= 1: only (1,1) feasible; no-good cuts on the
        rigorously-infeasible binary configs must not cut it off."""
        m = dm.Model("loa_nogood_bin")
        y1 = m.binary("y1")
        y2 = m.binary("y2")
        m.subject_to(y1 * y2 >= 1)
        m.minimize(y1 + y2)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status in ("optimal", "feasible")
        assert r.objective == pytest.approx(2.0, abs=1e-5)
        assert float(np.asarray(r.x["y1"])) == pytest.approx(1.0, abs=1e-5)
        assert float(np.asarray(r.x["y2"])) == pytest.approx(1.0, abs=1e-5)

    def test_milp_solver_simplex_backend(self):
        """The in-house simplex master backend solves the simple disjunction."""
        m = dm.Model("loa_simplex")
        x = m.continuous("x", lb=0, ub=10)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.minimize(x)
        r = m.solve(time_limit=30, gdp_method="loa", milp_solver="simplex")
        assert r.status in ("optimal", "feasible")
        assert r.objective == pytest.approx(0.0, abs=1e-4)

    def test_unresolved_config_returns_unknown(self, monkeypatch):
        """C-35: a non-rigorous NLP failure must NOT certify infeasibility.

        With the fixed-integer subproblem failing (and a free continuous
        variable, so no rigorous verdict), the re-proposed configuration must
        end the solve as status 'unknown' with gap_certified=False.
        """
        monkeypatch.setattr(loa, "_solve_nlp_subproblem", lambda *a, **k: None)
        m = dm.Model("loa_unresolved")
        y = m.binary("y")
        x = m.continuous("x", lb=0, ub=10)
        m.subject_to((x - 1) ** 2 <= 4)
        m.minimize(x + y)
        r = loa.solve_gdpopt_loa(m, time_limit=30, max_iterations=10)
        assert r.status == "unknown"
        assert r.objective is None
        assert r.gap_certified is False

    @pytest.mark.xfail(
        strict=False,
        reason="#756: BUG: _add_no_good_cut uses the binary-only exclusion form for "
        "general integer variables; the cut for config (0,1) (-y + z <= 0) also "
        "cuts the feasible (1,2), so LOA certifies a FALSE 'infeasible' on "
        "min y+z s.t. y*z>=2 (optimum 3 at y=1,z=2).",
    )
    def test_general_integer_no_good_cut_soundness(self):
        """y binary, z integer in [0,3], y*z >= 2: optimum is 3 at (1, 2)."""
        m = dm.Model("loa_nogood_int")
        y = m.binary("y")
        z = m.integer("z", lb=0, ub=3)
        m.subject_to(y * z >= 2)
        m.minimize(y + z)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status in ("optimal", "feasible"), (
            f"feasible model declared {r.status!r} (false infeasibility)"
        )
        assert r.objective == pytest.approx(3.0, abs=1e-5)

    @pytest.mark.xfail(
        strict=False,
        reason="#756: BUG: solve_gdpopt_loa returns status='infeasible' (default "
        "gap_certified=True) when the LOA loop exits on the time limit with no "
        "iteration run — a timeout is not an infeasibility proof; the correct "
        "status is 'unknown' (or an uncertified result).",
    )
    def test_time_limit_exhausted_is_not_infeasible(self):
        m = dm.Model("loa_tl0")
        x = m.continuous("x", lb=0, ub=10)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.minimize(x)
        r = loa.solve_gdpopt_loa(m, time_limit=0.0)
        assert r.status != "infeasible", (
            "timeout with zero iterations must not certify infeasibility"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GDPopt LOA: unit tests of the helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGDPoptLOAHelpers:
    def test_compute_gap(self):
        assert loa._compute_gap(-1e20, 5.0) == 1.0
        assert loa._compute_gap(0.0, 1e20) == 1.0
        assert loa._compute_gap(3.0, 3.0) == 0.0
        # ub=4, lb=2 -> abs gap 2 over denom 4
        assert loa._compute_gap(2.0, 4.0) == pytest.approx(0.5)
        # gap never negative even if lb > ub numerically
        assert loa._compute_gap(4.0, 3.9999999999) == 0.0

    @staticmethod
    def _one_var_model():
        m = dm.Model("rig")
        x = m.continuous("x", lb=0, ub=2)
        m.subject_to(x >= 1)
        m.minimize(x)
        return m, NLPEvaluator(m)

    def test_rigorous_infeasibility_pinned_violated(self):
        _, ev = self._one_var_model()
        assert loa._fixed_subproblem_rigorously_infeasible(ev, np.array([0.2]), np.array([0.2]))

    def test_rigorous_infeasibility_pinned_feasible(self):
        _, ev = self._one_var_model()
        assert not loa._fixed_subproblem_rigorously_infeasible(ev, np.array([1.5]), np.array([1.5]))

    def test_rigorous_infeasibility_not_pinned(self):
        _, ev = self._one_var_model()
        assert not loa._fixed_subproblem_rigorously_infeasible(ev, np.array([0.0]), np.array([2.0]))

    def test_rigorous_infeasibility_unconstrained(self):
        m = dm.Model("rig0")
        x = m.continuous("x", lb=0, ub=2)
        m.minimize(x)
        ev = NLPEvaluator(m)
        assert not loa._fixed_subproblem_rigorously_infeasible(ev, np.array([0.5]), np.array([0.5]))

    def test_bounds_proxy_delegates(self):
        _, ev = self._one_var_model()
        new_lb = np.array([0.5])
        new_ub = np.array([1.5])
        proxy = loa._BoundsProxy(ev, new_lb, new_ub)
        assert proxy.n_variables == ev.n_variables
        assert proxy.n_constraints == ev.n_constraints
        plb, pub = proxy.variable_bounds
        np.testing.assert_allclose(plb, new_lb)
        np.testing.assert_allclose(pub, new_ub)
        assert proxy._model is ev._model
        assert proxy._obj_fn is ev._obj_fn
        assert proxy._cons_fn is ev._cons_fn
        assert proxy._source_constraints is ev._source_constraints
        assert proxy._constraint_flat_sizes == ev._constraint_flat_sizes
        x_pt = np.array([1.2])
        assert float(proxy.evaluate_objective(x_pt)) == pytest.approx(
            float(ev.evaluate_objective(x_pt))
        )
        np.testing.assert_allclose(proxy.evaluate_gradient(x_pt), ev.evaluate_gradient(x_pt))
        np.testing.assert_allclose(
            np.asarray(proxy.evaluate_constraints(x_pt)),
            np.asarray(ev.evaluate_constraints(x_pt)),
        )
        np.testing.assert_allclose(
            np.asarray(proxy.evaluate_jacobian(x_pt)), np.asarray(ev.evaluate_jacobian(x_pt))
        )
        np.testing.assert_allclose(
            np.asarray(proxy.evaluate_hessian(x_pt)), np.asarray(ev.evaluate_hessian(x_pt))
        )
        lam = np.zeros(ev.n_constraints)
        np.testing.assert_allclose(
            np.asarray(proxy.evaluate_lagrangian_hessian(x_pt, 1.0, lam)),
            np.asarray(ev.evaluate_lagrangian_hessian(x_pt, 1.0, lam)),
        )

    def test_add_oa_cuts_flips_ge_cuts(self, monkeypatch):
        """A '>=' OA cut must be negated into the '<=' master form."""
        from discopt._jax import cutting_planes

        fake_cuts = [
            types.SimpleNamespace(sense="<=", coeffs=np.array([1.0, 2.0]), rhs=3.0),
            types.SimpleNamespace(sense=">=", coeffs=np.array([4.0, 5.0]), rhs=6.0),
        ]
        monkeypatch.setattr(
            cutting_planes, "generate_oa_cuts_from_evaluator", lambda *a, **k: fake_cuts
        )
        A_rows: list = []
        b_rows: list = []
        loa._add_oa_cuts(
            evaluator=None,
            x_star=np.zeros(2),
            n_vars=2,
            n_cons=1,
            oa_A_rows=A_rows,
            oa_b_rows=b_rows,
            obj_is_linear=True,
            constraint_convex_mask=[True],
            objective_is_convex=False,
        )
        assert len(A_rows) == 2
        np.testing.assert_allclose(A_rows[0], [1.0, 2.0])
        assert b_rows[0] == 3.0
        np.testing.assert_allclose(A_rows[1], [-4.0, -5.0])
        assert b_rows[1] == -6.0

    def test_add_no_good_cut_binary_form(self):
        """For x* = (1, 0): cut y1 - y2 <= 0 (excludes exactly (1,0))."""
        A_rows: list = []
        b_rows: list = []
        loa._add_no_good_cut(np.array([1.0, 0.0]), [0, 1], A_rows, b_rows, 2)
        np.testing.assert_allclose(A_rows[0], [1.0, -1.0])
        assert b_rows[0] == 0.0
        # The cut must NOT exclude the other binary configs.
        for cfg in ([0.0, 0.0], [0.0, 1.0], [1.0, 1.0]):
            assert float(np.dot(A_rows[0], cfg)) <= b_rows[0] + 1e-12
        # And must exclude (1, 0) itself.
        assert float(np.dot(A_rows[0], [1.0, 0.0])) > b_rows[0]

    def test_int_config_key(self):
        assert loa._int_config_key(np.array([0.9, 2.2, 0.1]), [0, 1]) == (1, 2)

    def test_build_x_dict_shapes(self):
        m = dm.Model("bxd")
        m.continuous("a", lb=0, ub=1)
        m.continuous("B", shape=(2, 2), lb=0, ub=1)
        x = np.arange(5.0)
        out = loa._build_x_dict(x, m)
        assert out["a"].shape == ()
        assert out["B"].shape == (2, 2)
        np.testing.assert_allclose(out["B"], np.arange(1.0, 5.0).reshape(2, 2))

    def test_master_milp_import_error(self, monkeypatch):
        import discopt.solvers.lp_backend as lp_backend

        def raise_import(**kwargs):
            raise ImportError("no backend")

        monkeypatch.setattr(lp_backend, "get_milp_solver", raise_import)
        with pytest.raises(ImportError, match="LOA solver requires a MILP backend"):
            loa._solve_master_milp(
                [],
                [],
                [],
                [],
                [],
                n_vars=1,
                integrality=np.zeros(1, dtype=np.int32),
                lb=np.zeros(1),
                ub=np.ones(1),
                obj_coeffs=(np.ones(1), 0.0),
                obj_is_linear=True,
                objective_bound_valid=True,
                time_limit=1.0,
                gap_tolerance=1e-4,
            )

    def test_nlp_subproblem_iteration_limit_accept_reject(self, monkeypatch):
        """ITERATION_LIMIT points are accepted only when primal-feasible."""
        import discopt.solvers.nlp_pounce as nlp_pounce

        m, ev = self._one_var_model()
        sub_lb, sub_ub = np.array([0.0]), np.array([2.0])

        monkeypatch.setattr(
            nlp_pounce,
            "solve_nlp",
            lambda *a, **k: NLPResult(
                status=SolveStatus.ITERATION_LIMIT, x=np.array([1.5]), objective=1.5
            ),
        )
        out = loa._solve_nlp_subproblem(ev, sub_lb, sub_ub, "ipm")
        assert out is not None
        np.testing.assert_allclose(out, [1.5])

        monkeypatch.setattr(
            nlp_pounce,
            "solve_nlp",
            lambda *a, **k: NLPResult(
                status=SolveStatus.ITERATION_LIMIT, x=np.array([0.2]), objective=0.2
            ),
        )
        assert loa._solve_nlp_subproblem(ev, sub_lb, sub_ub, "ipm") is None

    def test_nlp_relaxation_backend_failure_returns_none(self, monkeypatch):
        import discopt.solvers.nlp_pounce as nlp_pounce

        _, ev = self._one_var_model()
        monkeypatch.setattr(
            nlp_pounce, "solve_nlp", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        assert loa._solve_nlp_relaxation(ev, np.array([0.0]), np.array([2.0]), "ipm") is None

    def test_nlp_subproblem_backend_failure_returns_none(self, monkeypatch):
        import discopt.solvers.nlp_pounce as nlp_pounce

        _, ev = self._one_var_model()
        monkeypatch.setattr(
            nlp_pounce, "solve_nlp", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        assert loa._solve_nlp_subproblem(ev, np.array([0.0]), np.array([2.0]), "ipm") is None

    def test_ipopt_solver_selection_branches(self, monkeypatch):
        """nlp_solver='ipopt' must route to the nlp_ipopt backend."""
        import discopt.solvers.nlp_ipopt as nlp_ipopt

        _, ev = self._one_var_model()
        monkeypatch.setattr(
            nlp_ipopt,
            "solve_nlp",
            lambda *a, **k: NLPResult(status=SolveStatus.OPTIMAL, x=np.array([1.0]), objective=1.0),
            raising=False,
        )
        out = loa._solve_nlp_relaxation(ev, np.array([0.0]), np.array([2.0]), "ipopt")
        assert out is not None and out[0] == pytest.approx(1.0)
        out = loa._solve_nlp_subproblem(ev, np.array([0.0]), np.array([2.0]), "ipopt")
        assert out is not None and out[0] == pytest.approx(1.0)

    def test_rigorous_infeasibility_nonfinite_and_error_paths(self):
        _, ev = self._one_var_model()

        class NanCons:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, k):
                return getattr(self._inner, k)

            def evaluate_constraints(self, x):
                return np.array([np.nan])

        # Non-finite constraint value: never claim rigorous infeasibility.
        assert not loa._fixed_subproblem_rigorously_infeasible(
            NanCons(ev), np.array([0.2]), np.array([0.2])
        )

        class BrokenModel:
            n_constraints = 1
            _model = None

            def evaluate_constraints(self, x):
                return np.array([0.0])

        # Bound inference failing: never claim rigorous infeasibility.
        assert not loa._fixed_subproblem_rigorously_infeasible(
            BrokenModel(), np.array([0.2]), np.array([0.2])
        )


# ─────────────────────────────────────────────────────────────────────────────
# Primal heuristics: MultiStartNLP
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
class TestMultiStartNLP:
    def test_continuous_model_finds_optimum(self):
        """min (x-1)^2 s.t. x >= 0.5: optimum 0 at x = 1."""
        m = dm.Model("ms_cont")
        x = m.continuous("x", lb=-2, ub=3)
        m.subject_to(x >= 0.5)
        m.minimize((x - 1) ** 2)
        res = ph.MultiStartNLP(m, n_starts=4, seed=1).solve()
        assert res.n_starts == 4
        assert res.n_feasible > 0
        assert res.best_solution is not None
        _assert_feasible(m, res.best_solution)
        # Never better than the true optimum (0), and should reach it.
        assert res.best_objective >= -1e-9
        assert res.best_objective == pytest.approx(0.0, abs=1e-6)
        assert len(res.all_objectives) == res.n_feasible

    def test_integer_model_updates_only_on_integral_points(self):
        """Integer var driven to its bound is integral: incumbent recorded."""
        m = dm.Model("ms_int_bound")
        xc = m.continuous("xc", lb=0, ub=4)
        yi = m.integer("yi", lb=0, ub=3)
        m.minimize((xc - 0.7) ** 2 + yi)
        res = ph.MultiStartNLP(m, n_starts=3, seed=1).solve()
        assert res.n_integer_feasible > 0
        assert res.best_solution is not None
        _assert_feasible(m, res.best_solution)
        assert ph._is_integer_feasible(res.best_solution, ph._get_integer_mask(m))
        # True optimum 0 at (0.7, 0): never claim better (up to solver abs tol).
        assert res.best_objective >= -1e-6

    def test_infeasible_backend_records_no_solutions(self):
        m = dm.Model("ms_nofeas")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        res = ph.MultiStartNLP(m, n_starts=2, seed=0).solve(
            backend=_mk_backend([0.5], status=SolveStatus.INFEASIBLE)
        )
        assert res.n_feasible == 0
        assert res.best_objective is None
        assert res.best_solution is None

    def test_integer_model_fractional_relaxation_yields_no_incumbent(self):
        """When the NLP parks the integer at a fractional value, no incumbent."""
        m = dm.Model("ms_int_frac")
        xc = m.continuous("xc", lb=0, ub=4)
        yi = m.integer("yi", lb=0, ub=3)
        m.subject_to(xc + yi >= 1.2)
        m.minimize((xc - 0.7) ** 2 + 0.1 * yi)
        res = ph.MultiStartNLP(m, n_starts=4, seed=1).solve()
        assert res.best_objective is None
        assert res.best_solution is None
        assert res.n_integer_feasible == 0


# ─────────────────────────────────────────────────────────────────────────────
# Primal heuristics: feasibility pump / subnlp / continuous multistart
# ─────────────────────────────────────────────────────────────────────────────


def _pump_model():
    m = dm.Model("fp_model")
    xb = m.binary("xb")
    xc = m.continuous("xc", lb=0, ub=2)
    m.subject_to(xc + xb >= 0.5)
    m.minimize(xc + xb)
    return m


@pytest.mark.unit
class TestFeasibilityPumpEdges:
    def test_backend_failure_then_deadline_stops(self):
        m = _pump_model()
        out = ph.feasibility_pump(
            m,
            np.array([0.4, 0.3]),
            backend=_raising_backend,
            deadline=time.perf_counter() - 1.0,
        )
        assert out is None

    def test_nan_solution_rejected(self):
        m = _pump_model()
        out = ph.feasibility_pump(
            m, np.array([0.4, 0.3]), max_rounds=1, backend=_mk_backend([np.nan, np.nan])
        )
        assert out is None

    def test_constraint_infeasible_solution_rejected(self):
        m = _pump_model()
        out = ph.feasibility_pump(
            m, np.array([0.4, 0.3]), max_rounds=1, backend=_mk_backend([0.0, 0.0])
        )
        assert out is None

    def test_bounds_restored_after_pump(self):
        m = _pump_model()
        lb0, ub0 = ph._get_variable_bounds(m)
        ph.feasibility_pump(m, np.array([0.4, 0.3]), max_rounds=1, backend=_raising_backend)
        lb1, ub1 = ph._get_variable_bounds(m)
        np.testing.assert_allclose(lb0, lb1)
        np.testing.assert_allclose(ub0, ub1)


@pytest.mark.smoke
class TestFeasibilityPumpReal:
    def test_pump_returns_feasible_integer_point(self):
        m = _pump_model()
        out = ph.feasibility_pump(m, np.array([0.4, 0.3]))
        assert out is not None
        _assert_feasible(m, out)
        assert ph._is_integer_feasible(out, ph._get_integer_mask(m))
        ev = cached_evaluator(m)
        obj = float(ev.evaluate_objective(out))
        # True optimum is 0.5 (xb=0, xc=0.5): never claim better.
        assert obj >= 0.5 - 1e-6


@pytest.mark.unit
class TestCheckConstraintFeasibility:
    def test_jacobian_failure_is_infeasible(self):
        m = _pump_model()
        ev = cached_evaluator(m)

        class JacRaising:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, k):
                return getattr(self._inner, k)

            def evaluate_jacobian(self, x):
                raise RuntimeError("no jacobian")

        assert ph._check_constraint_feasibility(JacRaising(ev), np.array([0.0, 0.0])) is False

    def test_unconstrained_model_is_feasible(self):
        m = dm.Model("ccf0")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        assert ph._check_constraint_feasibility(cached_evaluator(m), np.array([0.5])) is True


@pytest.mark.unit
class TestSubNLPEdges:
    def test_backend_raising_returns_none(self):
        m = _pump_model()
        assert ph.subnlp(m, np.array([0.4, 0.3]), backend=_raising_backend) is None

    def test_missing_x_returns_none(self):
        m = _pump_model()
        assert ph.subnlp(m, np.array([0.4, 0.3]), backend=_mk_backend(None, objective=None)) is None

    def test_nan_integer_slot_returns_none(self):
        m = _pump_model()
        assert ph.subnlp(m, np.array([0.4, 0.3]), backend=_mk_backend([np.nan, np.nan])) is None

    def test_constraint_infeasible_returns_none(self):
        m = _pump_model()
        assert ph.subnlp(m, np.array([0.4, 0.3]), backend=_mk_backend([0.0, 0.0])) is None

    def test_bounds_restored_after_subnlp(self):
        m = _pump_model()
        lb0, ub0 = ph._get_variable_bounds(m)
        ph.subnlp(m, np.array([1.0, 0.3]), backend=_raising_backend)
        lb1, ub1 = ph._get_variable_bounds(m)
        np.testing.assert_allclose(lb0, lb1)
        np.testing.assert_allclose(ub0, ub1)


@pytest.mark.smoke
class TestSubNLPReal:
    def test_default_evaluator_and_backend(self):
        """subnlp with no evaluator/backend: fixes xb=1, resolves xc -> obj 1."""
        m = _pump_model()
        out = ph.subnlp(m, np.array([1.0, 0.3]))
        assert out is not None
        x_out, obj = out
        _assert_feasible(m, x_out)
        assert x_out[0] == pytest.approx(1.0, abs=1e-9)
        assert obj == pytest.approx(1.0, abs=1e-4)


def _cms_model():
    m = dm.Model("cms_model")
    u = m.continuous("u", lb=-1, ub=2)
    m.subject_to(u >= 0.25)
    m.minimize((u - 1) ** 2)
    return m


@pytest.mark.unit
class TestContinuousMultistartEdges:
    def test_integer_model_is_skipped(self):
        m = _pump_model()
        assert ph.continuous_multistart(m, n_starts=2) is None

    def test_zero_starts(self):
        assert ph.continuous_multistart(_cms_model(), n_starts=0) is None

    def test_backend_raising(self):
        assert ph.continuous_multistart(_cms_model(), n_starts=2, backend=_raising_backend) is None

    def test_missing_x(self):
        out = ph.continuous_multistart(
            _cms_model(), n_starts=2, backend=_mk_backend(None, objective=None)
        )
        assert out is None

    def test_infeasible_point_rejected(self):
        # u = -0.5 violates u >= 0.25 but has a "good" objective claim.
        out = ph.continuous_multistart(
            _cms_model(), n_starts=2, backend=_mk_backend([-0.5], objective=0.0)
        )
        assert out is None


@pytest.mark.smoke
class TestContinuousMultistartReal:
    def test_finds_verified_optimum(self):
        m = _cms_model()
        out = ph.continuous_multistart(m, n_starts=4)
        assert out is not None
        x_out, obj = out
        _assert_feasible(m, x_out)
        # True optimum 0 at u = 1: reached but never beaten.
        assert obj >= -1e-9
        assert obj == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Primal heuristics: integer lattice searches
# ─────────────────────────────────────────────────────────────────────────────


def _ab_model():
    """min a+b s.t. a*b >= 4, a,b integer in [1,4]. Optimum 4 at (2,2)."""
    m = dm.Model("ab_model")
    a = m.integer("a", lb=1, ub=4)
    b = m.integer("b", lb=1, ub=4)
    m.subject_to(a * b >= 4)
    m.minimize(a + b)
    return m


@pytest.mark.smoke
class TestIntegerLocalSearch:
    def test_finds_true_optimum_with_defaults(self):
        m = _ab_model()
        out = ph.integer_local_search(m, np.array([1.0, 1.0]), time_budget=3.0)
        assert out is not None
        x_out, obj = out
        _assert_feasible(m, x_out)
        assert ph._is_integer_feasible(x_out, ph._get_integer_mask(m))
        # Enumerated optimum is 4 at (2,2): reached, never beaten.
        assert obj >= 4.0 - 1e-9
        assert obj == pytest.approx(4.0, abs=1e-6)

    def test_pure_continuous_model_is_noop(self):
        m = _cms_model()
        assert ph.integer_local_search(m, np.array([0.5])) is None

    def test_backend_raising_yields_none(self):
        m = _ab_model()
        out = ph.integer_local_search(
            m, np.array([1.0, 1.0]), backend=_raising_backend, time_budget=0.5
        )
        assert out is None


@pytest.mark.unit
class TestIntegerBoxSearchEdges:
    def test_short_incumbent_vector(self):
        assert ph.integer_box_search(_ab_model(), np.array([2.0])) is None

    def test_center_outside_bounds_gives_empty_axis(self):
        assert ph.integer_box_search(_ab_model(), np.array([10.0, 2.0]), radius=1) is None

    def test_combo_cap(self):
        assert ph.integer_box_search(_ab_model(), np.array([2.0, 2.0]), max_combos=2) is None

    def test_zero_time_budget(self):
        assert ph.integer_box_search(_ab_model(), np.array([4.0, 1.0]), time_budget=0.0) is None

    def test_too_many_integers(self):
        assert ph.integer_box_search(_ab_model(), np.array([2.0, 2.0]), max_int_vars=1) is None


@pytest.mark.smoke
class TestIntegerBoxSearchReal:
    def test_improves_suboptimal_incumbent(self):
        """From (4,1) obj 5 the radius-2 box reaches the optimum (2,2) obj 4."""
        m = _ab_model()
        out = ph.integer_box_search(m, np.array([4.0, 1.0]), radius=2, time_budget=5.0)
        assert out is not None
        x_out, obj = out
        _assert_feasible(m, x_out)
        assert obj >= 4.0 - 1e-9
        assert obj == pytest.approx(4.0, abs=1e-6)


@pytest.mark.unit
class TestSmallHeuristicHelpers:
    def test_enumerate_binary_seeds_skips_general_integers(self):
        assert ph.enumerate_binary_seeds_subnlp(_ab_model(), np.array([2.0, 2.0])) == []

    def test_finalize_candidate_rejects_nan_integers(self):
        m = _ab_model()
        out = ph._finalize_candidate(
            cached_evaluator(m), np.array([np.nan, 1.0]), ph._get_integer_mask(m), 1e-5, 1e-6
        )
        assert out is None

    def test_binary_slot_term_matrix_variable(self):
        m = dm.Model("slot_mtx")
        B = m.binary("B", shape=(2, 2))
        term = ph._binary_slot_term(m, 3)
        assert isinstance(term, core.IndexExpression)
        assert term.base is B

    def test_rens_short_relaxation_vector(self):
        assert ph.rens(_ab_model(), np.array([0.5]), sub_solver=lambda mm: None) is None


@pytest.mark.unit
class TestDivingEdges:
    @staticmethod
    def _mixed_model():
        m = dm.Model("dive_model")
        zb = m.integer("zb", lb=0, ub=3)
        w = m.continuous("w", lb=0, ub=2)
        m.subject_to(zb + w >= 1.0)
        m.minimize(zb + w)
        return m

    def test_backend_raising(self):
        m = self._mixed_model()
        assert ph.diving(m, np.array([0.5, 0.5]), backend=_raising_backend) is None

    def test_zero_dive_budget(self):
        m = self._mixed_model()
        assert ph.diving(m, np.array([0.5, 0.5]), max_dives=0, backend=_raising_backend) is None

    def test_objective_mode_gradient_failure_falls_back_to_round(self):
        m = self._mixed_model()
        ev = cached_evaluator(m)

        class GradRaising:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, k):
                return getattr(self._inner, k)

            def evaluate_gradient(self, x):
                raise RuntimeError("no gradient")

        calls = {"n": 0}

        def backend(evaluator, x0, options=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return NLPResult(status=SolveStatus.OPTIMAL, x=np.array([0.6, 0.4]), objective=1.0)
            return NLPResult(status=SolveStatus.OPTIMAL, x=np.array([1.0, 0.0]), objective=1.0)

        out = ph.diving(
            m, np.array([0.5, 0.5]), mode="objective", backend=backend, evaluator=GradRaising(ev)
        )
        assert out is not None
        x_out, obj = out
        _assert_feasible(m, x_out)
        assert obj == pytest.approx(1.0, abs=1e-9)
        # Bounds restored after the dive.
        lb1, ub1 = ph._get_variable_bounds(m)
        np.testing.assert_allclose(lb1, [0.0, 0.0])
        np.testing.assert_allclose(ub1, [3.0, 2.0])


# ─────────────────────────────────────────────────────────────────────────────
# Primal heuristics: local branching
# ─────────────────────────────────────────────────────────────────────────────


def _two_binary_model():
    m = dm.Model("lb2")
    y1 = m.binary("y1")
    y2 = m.binary("y2")
    m.subject_to(y1 + y2 >= 1)
    m.minimize(y1 + y2)
    return m


class _FakeSolveResult:
    def __init__(self, x):
        self.x = x
        self.status = "optimal"


@pytest.mark.unit
class TestLocalBranchingSubmip:
    @staticmethod
    def _call(m, monkeypatch, fake_solve):
        import discopt.solver as solver_mod

        monkeypatch.setattr(solver_mod, "solve_model", fake_solve)
        ev = cached_evaluator(m)
        n_before = len(m._constraints)
        out = ph._local_branching_submip(
            m,
            np.array([1.0, 1.0]),
            [0, 1],
            k=1,
            backend=None,
            nlp_options=None,
            integer_tol=1e-5,
            feas_tol=1e-6,
            evaluator=ev,
            time_limit=1.0,
            max_nodes=10,
            gap_tolerance=1e-4,
        )
        # The Hamming cut must always be removed again.
        assert len(m._constraints) == n_before
        return out

    def test_solver_exception_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("solver crashed")

        assert self._call(_two_binary_model(), monkeypatch, boom) is None

    def test_missing_x_dict_returns_none(self, monkeypatch):
        assert (
            self._call(_two_binary_model(), monkeypatch, lambda *a, **k: _FakeSolveResult(None))
            is None
        )

    def test_missing_variable_name_returns_none(self, monkeypatch):
        fake = lambda *a, **k: _FakeSolveResult({"y1": np.array(1.0)})  # noqa: E731
        assert self._call(_two_binary_model(), monkeypatch, fake) is None

    def test_wrong_size_returns_none(self, monkeypatch):
        fake = lambda *a, **k: _FakeSolveResult(  # noqa: E731
            {"y1": np.array([1.0, 0.0]), "y2": np.array([0.0, 1.0])}
        )
        assert self._call(_two_binary_model(), monkeypatch, fake) is None

    def test_infeasible_point_returns_none(self, monkeypatch):
        # (0, 0) violates y1 + y2 >= 1.
        fake = lambda *a, **k: _FakeSolveResult(  # noqa: E731
            {"y1": np.array(0.0), "y2": np.array(0.0)}
        )
        assert self._call(_two_binary_model(), monkeypatch, fake) is None

    def test_improving_point_is_returned(self, monkeypatch):
        # (1, 0): feasible, obj 1 < incumbent obj 2.
        fake = lambda *a, **k: _FakeSolveResult(  # noqa: E731
            {"y1": np.array(1.0), "y2": np.array(0.0)}
        )
        out = self._call(_two_binary_model(), monkeypatch, fake)
        assert out is not None
        x_out, obj = out
        np.testing.assert_allclose(x_out, [1.0, 0.0])
        assert obj == pytest.approx(1.0)


@pytest.mark.unit
class TestLocalBranchingBudget:
    def test_no_binaries_returns_none(self):
        assert ph.local_branching(_ab_model(), np.array([2.0, 2.0])) is None

    def test_zero_slice_truncates_before_radius_zero(self):
        m = _two_binary_model()
        out = ph.local_branching(m, np.array([1.0, 1.0]), submip_time_limit=0.0)
        assert out is None

    def test_deadline_expiring_mid_round_truncates(self):
        m = dm.Model("lb3")
        b = m.binary("b", shape=(3,))
        m.minimize(dm.sum(b))
        calls = {"n": 0}

        def slow_backend(evaluator, x0, options=None):
            calls["n"] += 1
            if calls["n"] >= 2:
                time.sleep(0.3)
            raise RuntimeError("no point")

        out = ph.local_branching(
            m,
            np.zeros(3),
            k=1,
            backend=slow_backend,
            submip_time_limit=0.25,
        )
        assert out is None
        # radius 0 ran, radius 1 started and was cut off by the deadline poll.
        assert calls["n"] >= 2

    def test_truncation_dispatches_bounded_submip(self, monkeypatch):
        """When the enumeration cannot fit the budget but >= 2.5s remain, the
        unexplored neighbourhood goes to the bounded sub-MIP."""
        m = dm.Model("lb24")
        b = m.binary("b", shape=(24,))
        m.minimize(dm.sum(b))
        x_better = np.zeros(24)

        submip_calls = {"n": 0}

        def fake_submip(*args, **kwargs):
            submip_calls["n"] += 1
            return x_better, 0.0

        monkeypatch.setattr(ph, "_local_branching_submip", fake_submip)

        def slow_backend(evaluator, x0, options=None):
            time.sleep(0.6)  # inflate the measured per-sub-NLP mean
            raise RuntimeError("no point")

        out = ph.local_branching(
            m,
            np.ones(24),
            k=2,
            backend=slow_backend,
            submip_time_limit=8.0,
            max_binaries=30,  # force the enumeration branch (not the >12 dispatch)
        )
        assert submip_calls["n"] == 1
        assert out is not None
        np.testing.assert_allclose(out[0], x_better)
        assert out[1] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Primal heuristics: one-hot swap search
# ─────────────────────────────────────────────────────────────────────────────


def _assignment_model(extra_constraint: bool = False):
    """2 items x 2 slots; swap of the identity assignment drops cost 10 -> 2."""
    m = dm.Model("ohs_model")
    X = m.binary("X", shape=(2, 2))
    m.subject_to(X[0, 0] + X[0, 1] == 1)
    m.subject_to(X[1, 0] + X[1, 1] == 1)
    if extra_constraint:
        m.subject_to(X[0, 1] + X[1, 0] <= 1)
    m.minimize(5 * X[0, 0] + 1 * X[0, 1] + 1 * X[1, 0] + 5 * X[1, 1])
    return m


@pytest.mark.smoke
class TestOneHotSwapSearch:
    def test_swap_reaches_cheaper_assignment(self):
        m = _assignment_model()
        inc = np.array([1.0, 0.0, 0.0, 1.0])  # cost 10
        out = ph.one_hot_swap_search(m, inc, time_budget=1.0)
        assert out is not None
        x_out, obj = out
        _assert_feasible(m, x_out)
        np.testing.assert_allclose(x_out, [0.0, 1.0, 1.0, 0.0])
        assert obj == pytest.approx(2.0)

    def test_swap_blocked_by_side_constraint_returns_none(self):
        m = _assignment_model(extra_constraint=True)
        inc = np.array([1.0, 0.0, 0.0, 1.0])
        assert ph.one_hot_swap_search(m, inc, time_budget=1.0) is None

    def test_no_integer_variables(self):
        assert ph.one_hot_swap_search(_cms_model(), np.array([1.0])) is None

    def test_zero_budget_returns_none(self):
        m = _assignment_model()
        inc = np.array([1.0, 0.0, 0.0, 1.0])
        assert ph.one_hot_swap_search(m, inc, time_budget=0.0) is None


@pytest.mark.unit
class TestDetectOneHotGroups:
    @staticmethod
    def _mask(m):
        lb, ub = ph._get_variable_bounds(m)
        return ph._get_integer_mask(m) & (lb >= -1e-9) & (ub <= 1.0 + 1e-9)

    def test_rejects_non_unit_non_binary_and_overlapping_rows(self):
        m = dm.Model("oh_mixed")
        X = m.binary("X", shape=(2, 2))
        c1 = m.continuous("c1", lb=0, ub=1)
        m.subject_to(X[0, 0] + X[0, 1] == 1)
        m.subject_to(X[1, 0] + X[1, 1] == 1)
        m.subject_to(2 * X[0, 0] + 2 * X[1, 1] == 1)  # non-unit coefficients
        m.subject_to(X[0, 0] + c1 == 1)  # non-binary support
        m.subject_to(X[0, 1] + X[1, 0] == 1)  # overlaps the first group
        m.minimize(dm.sum(X) + c1)
        mask = self._mask(m)
        groups = ph._detect_one_hot_groups(m, mask, mask.size)
        assert groups == [[0, 1], [2, 3]]

    def test_unequal_group_sizes_rejected(self):
        m = dm.Model("oh_uneq")
        Y = m.binary("Y", shape=(5,))
        m.subject_to(Y[0] + Y[1] == 1)
        m.subject_to(Y[2] + Y[3] + Y[4] == 1)
        m.minimize(dm.sum(Y))
        mask = self._mask(m)
        assert ph._detect_one_hot_groups(m, mask, mask.size) == []

    def test_single_group_rejected(self):
        m = dm.Model("oh_one")
        Y = m.binary("Y", shape=(2,))
        m.subject_to(Y[0] + Y[1] == 1)
        m.minimize(dm.sum(Y))
        mask = self._mask(m)
        assert ph._detect_one_hot_groups(m, mask, mask.size) == []


# ─────────────────────────────────────────────────────────────────────────────
# modeling/core.py: expression-layer units
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCoreExpressionUnits:
    def test_rpow_builds_power_with_constant_base(self):
        m = dm.Model("rpow")
        x = m.continuous("x", lb=0, ub=1)
        e = 2**x
        assert isinstance(e, core.BinaryOp)
        assert e.op == "**"
        assert isinstance(e.left, core.Constant)
        assert e.left.value == 2.0
        assert e.right is x

    def test_repr_latex_falls_back_on_renderer_failure(self, monkeypatch):
        from discopt.modeling import latex as latex_mod

        def boom(expr):
            raise RuntimeError("renderer down")

        monkeypatch.setattr(latex_mod, "expr_to_latex", boom)
        m = dm.Model("latex")
        x = m.continuous("x", lb=0, ub=1)
        out = (x + 1)._repr_latex_()
        assert out.startswith("$") and out.endswith("$")
        assert "renderer down" not in out

    def test_sum_repr_with_axis(self):
        m = dm.Model("sumrepr")
        v = m.continuous("v", shape=(3,), lb=0, ub=1)
        assert "axis=0" in repr(dm.sum(v, axis=0))

    def test_is_term_iterable_excludes_indexed_containers(self):
        class FakeIndexed:
            _is_indexed_container = True

            def __iter__(self):
                return iter(["a", "b"])

        assert core._is_term_iterable(FakeIndexed()) is False
        assert core._is_term_iterable([1, 2]) is True
        assert core._is_term_iterable("abc") is False

    def test_find_owning_model_walks_all_node_types(self):
        m = dm.Model("fom")
        x = m.continuous("x", lb=0, ub=1)
        v = m.continuous("v", shape=(3,), lb=0, ub=1)
        assert core._find_owning_model(v[0]) is m  # IndexExpression
        assert core._find_owning_model(1.0 * x) is m  # BinaryOp (var on right)
        assert core._find_owning_model(-x) is m  # UnaryOp
        assert core._find_owning_model(dm.exp(x)) is m  # FunctionCall
        assert core._find_owning_model(np.eye(3) @ v) is m  # MatMulExpression
        assert core._find_owning_model(dm.sum(v)) is m  # SumExpression
        assert core._find_owning_model(core.SumOverExpression([x])) is m
        assert core._find_owning_model(core._wrap(1.0)) is None

    def test_if_else_free_function_without_model_raises(self):
        with pytest.raises(ValueError, match="could not determine the owning Model"):
            core.if_else(core._wrap(1.0) >= 0, 1.0, 2.0)

    def test_if_else_free_function_requires_constraint(self):
        with pytest.raises(TypeError, match="condition must be a Constraint"):
            core.if_else(True, 1.0, 2.0)

    def test_norm_inf_suffix(self):
        m = dm.Model("norminf")
        v = m.continuous("v", shape=(2,), lb=0, ub=1)
        e = core.norm(v, ord="inf")
        assert isinstance(e, core.FunctionCall)
        assert e.func_name == "norminf"

    def test_constraint_list_len(self):
        m = dm.Model("clist")
        x = m.continuous("x", lb=0, ub=1)
        cl = core.ConstraintList([x <= 1, x >= 0])
        assert len(cl) == 2

    def test_logical_operator_wrapping_errors(self):
        m = dm.Model("logic_ops")
        b1 = m.boolean("b1")
        with pytest.raises(TypeError, match="Expected LogicalExpression, got bool"):
            True & b1  # __rand__ -> _wrap_logical
        with pytest.raises(TypeError, match="Expected LogicalExpression, got bool"):
            True | b1  # __ror__ -> _wrap_logical

    def test_land_and_lnot_constructors(self):
        m = dm.Model("logic_fns")
        b1 = m.boolean("b1")
        b2 = m.boolean("b2")
        b3 = m.boolean("b3")
        e = core.land(b1, b2, b3)
        assert isinstance(e, core.LogicalAnd)
        assert isinstance(e.left, core.LogicalAnd)
        n = core.lnot(b1)
        assert isinstance(n, core.LogicalNot)
        assert n.operand is b1

    def test_rebalance_deep_sum_preserves_terms(self):
        m = dm.Model("rebal")
        x = m.continuous("x", lb=0, ub=1)
        n_terms = core._SUM_REBALANCE_DEPTH + 60
        expr = x
        for _ in range(n_terms):
            expr = expr + 1.0
        out = core._rebalance_deep_sum(expr)

        def spine_depth(e):
            d = 0
            while isinstance(e, core.BinaryOp) and e.op == "+":
                e = e.left
                d += 1
            return d

        assert spine_depth(expr) == n_terms
        assert spine_depth(out) < 32  # balanced: O(log n)

        # Same multiset of leaves: the constant total and the single variable.
        def leaf_totals(e):
            total, n_vars = 0.0, 0
            stack = [e]
            while stack:
                node = stack.pop()
                if isinstance(node, core.BinaryOp):
                    stack.extend((node.left, node.right))
                elif isinstance(node, core.Constant):
                    total += float(node.value)
                elif isinstance(node, core.Variable):
                    n_vars += 1
            return total, n_vars

        assert leaf_totals(out) == leaf_totals(expr) == (float(n_terms), 1)

    def test_shallow_sum_is_untouched(self):
        m = dm.Model("rebal2")
        x = m.continuous("x", lb=0, ub=1)
        expr = x + 1.0 + 2.0
        assert core._rebalance_deep_sum(expr) is expr


# ─────────────────────────────────────────────────────────────────────────────
# modeling/core.py: Model validation and error branches
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestModelValidationErrors:
    def test_subject_to_non_iterable(self):
        m = dm.Model("st_bad")
        with pytest.raises(TypeError, match="Expected Constraint or an iterable"):
            m.subject_to(42)

    def test_decomp_var_name_resolution(self):
        m = dm.Model("decomp")
        x = m.continuous("x", lb=0, ub=1)
        v = m.continuous("v", shape=(3,), lb=0, ub=1)
        y = m.integer("y", lb=0, ub=1)
        assert m._decomp_var_name("v") == "v"
        assert m._decomp_var_name(v) == "v"
        assert m._decomp_var_name(v[1]) == "v"
        assert m._decomp_var_name(x + 1) == "x"
        with pytest.raises(TypeError, match="Cannot resolve a decomposition variable"):
            m._decomp_var_name(x + y)
        # Non-expression input: variable collection fails, ambiguous -> TypeError.
        with pytest.raises(TypeError, match="Cannot resolve a decomposition variable"):
            m._decomp_var_name(object())

        # A malformed expression node (DAG walk raises) also resolves to TypeError.
        class BrokenNode(core.UnaryOp):
            def __init__(self):
                pass

            @property
            def operand(self):
                raise RuntimeError("broken operand")

            def __repr__(self):
                return "<broken>"

        with pytest.raises(TypeError, match="Cannot resolve a decomposition variable"):
            m._decomp_var_name(BrokenNode())

    def test_constraint_family_mixed_variables_uses_general_path(self):
        """A rule family over two different backing variables cannot use the
        single-variable fast linear path."""
        m = dm.Model("fam_mixed")
        idx = m.set("I", ["a", "b"])
        x = m.continuous("x", shape=(2,), lb=0, ub=1)
        y = m.continuous("y", shape=(2,), lb=0, ub=1)
        vars_ = {"a": x, "b": y}
        ic = m.constraint(idx, lambda i: vars_[i][0] <= 1.0, name="fam")
        assert ic.fast is False
        assert len(m._constraints) == 2

    def test_add_linear_constraints_invalid_sense(self):
        m = dm.Model("alc_sense")
        v = m.continuous("v", shape=(3,), lb=0, ub=1)
        with pytest.raises(ValueError, match="Invalid sense"):
            m.add_linear_constraints(np.eye(3), v, "!!", np.zeros(3))

    def test_add_linear_constraints_unregistered_variable(self):
        m = dm.Model("alc_a")
        m_other = dm.Model("alc_b")
        w = m_other.continuous("w", shape=(2,), lb=0, ub=1)
        with pytest.raises(ValueError, match="not registered in the builder"):
            m.add_linear_constraints(np.eye(2), w, "<=", np.zeros(2))

    def test_add_linear_objective_unregistered_variable(self):
        m = dm.Model("alo_a")
        m_other = dm.Model("alo_b")
        w = m_other.continuous("w", shape=(2,), lb=0, ub=1)
        with pytest.raises(ValueError, match="not registered in the builder"):
            m.add_linear_objective(np.ones(2), w)

    def test_add_quadratic_objective_errors(self):
        import scipy.sparse as sp

        m = dm.Model("aqo_a")
        m_other = dm.Model("aqo_b")
        w = m_other.continuous("w", shape=(2,), lb=0, ub=1)
        with pytest.raises(ValueError, match="c has 3 elements"):
            m.add_quadratic_objective(np.eye(2), np.ones(3), w)
        # A COO input exercises the tocsr() conversion before the shape check.
        with pytest.raises(ValueError, match=r"Q has shape \(3, 3\)"):
            m.add_quadratic_objective(sp.coo_matrix(np.eye(3)), np.ones(2), w)
        with pytest.raises(ValueError, match="not registered in the builder"):
            m.add_quadratic_objective(np.eye(2), np.ones(2), w)

    def test_validate_binaries_via_at_least(self):
        m = dm.Model("vb")
        x = m.continuous("x", lb=0, ub=1)
        cc = m.continuous("cc", shape=(2,), lb=0, ub=1)
        bb = m.binary("bb", shape=(2,))
        with pytest.raises(ValueError, match="requires binary variables, but 'x'"):
            m.at_least(1, [x])
        with pytest.raises(ValueError, match="requires binary variables, but 'cc'"):
            m.at_least(1, [cc[0]])
        with pytest.raises(TypeError, match="IndexExpression with non-Variable base"):
            m.at_least(1, [(x + x)[0]])
        with pytest.raises(TypeError, match="requires Variable or IndexExpression"):
            m.at_least(1, [3.14])
        # Valid binaries pass.
        m.at_least(1, [bb[0], bb[1]])

    def test_boolean_int_shape_and_logical_type_error(self):
        m = dm.Model("bool_shape")
        bv = m.boolean("bv", shape=2)
        assert type(bv).__name__ == "BooleanVarArray"
        with pytest.raises(TypeError, match="Expected LogicalExpression, got int"):
            m.logical(42)

    def test_validate_duplicate_variable_name(self):
        m = dm.Model("dupname")
        a = m.continuous("a", lb=0, ub=1)
        m._variables.append(a)  # simulate a duplicate registration
        m.minimize(a)
        with pytest.raises(ValueError, match="Duplicate variable name: 'a'"):
            m.validate()

    def test_validate_lb_greater_than_ub(self):
        m = dm.Model("lbub")
        a = m.continuous("a", lb=2, ub=1)
        m.minimize(a)
        with pytest.raises(ValueError, match="lb > ub"):
            m.validate()

    def test_validate_no_objective(self):
        m = dm.Model("noobj")
        m.continuous("a", lb=0, ub=1)
        with pytest.raises(ValueError, match="No objective set"):
            m.validate()

    def test_validate_walks_logical_and_unknown_bodied_constraints(self):
        m = dm.Model("walk")
        x = m.continuous("x", lb=0, ub=1)
        b1 = m.boolean("b1")
        b2 = m.boolean("b2")
        m.logical(b1 & b2)

        class OddConstraint:
            def __init__(self, body):
                self.body = body

        m._constraints.append(OddConstraint(x * 2.0))
        m.minimize(x)
        m.validate()  # must walk both without error (all leaves owned)

    def test_materialize_builder_linear_rows(self):
        # Scalar variable: the component is the variable itself.
        m1 = dm.Model("mat_scalar")
        s = m1.continuous("s", lb=0, ub=5)
        m1.add_linear_constraints(np.array([[1.0]]), s, "<=", np.array([2.0]))
        assert m1._materialize_builder_linear_rows() == 1
        assert len(m1._constraints) == 1

        # Matrix variable + an all-zero row (materialized as constant 0 body).
        m2 = dm.Model("mat_matrix")
        M = m2.continuous("M", shape=(2, 2), lb=0, ub=5)
        A = np.zeros((2, 4))
        A[0, 3] = 2.0
        m2.add_linear_constraints(A, M, ">=", np.array([1.0, 0.0]))
        assert m2._materialize_builder_linear_rows() == 2
        assert len(m2._constraints) == 2

    def test_if_else_with_unbounded_branch_gets_wide_aux_bounds(self):
        m = dm.Model("ife_wide")
        x = m.continuous("x", lb=0, ub=1)
        free = m.continuous("free")  # unbounded
        w = m.if_else(x >= 0.5, free, 1.0)
        assert float(np.asarray(w.lb)) <= -9e19
        assert float(np.asarray(w.ub)) >= 9e19

    def test_if_else_with_uninterpretable_branch_gets_wide_aux_bounds(self):
        m = dm.Model("ife_bogus")
        x = m.continuous("x", lb=0, ub=1)
        bogus = core.FunctionCall("no_such_function", core._wrap(x))
        w = m.if_else(x >= 0.5, bogus, 1.0)
        assert float(np.asarray(w.lb)) <= -9e19
        assert float(np.asarray(w.ub)) >= 9e19

    def test_if_else_with_erroring_interval_branch_gets_wide_aux_bounds(self):
        # Shape-mismatched matmul makes the interval evaluation raise.
        m = dm.Model("ife_err")
        x = m.continuous("x", lb=0, ub=1)
        bad = core.MatMulExpression(core._wrap(np.eye(3)), core._wrap(np.ones(2)))
        w = m.if_else(x >= 0.5, bad, 1.0)
        assert float(np.asarray(w.lb)) <= -9e19
        assert float(np.asarray(w.ub)) >= 9e19

    def test_boolean_var_and_disjunct_reprs(self):
        m = dm.Model("reprs")
        b = m.boolean("b")
        assert repr(b) == "BooleanVar(b)"
        d = m.make_disjunct("dj")
        assert repr(d) == "Disjunct('dj', 0 constraints)"


# ─────────────────────────────────────────────────────────────────────────────
# modeling/core.py: SolveResult, solve() option paths, LLM hooks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSolveResultUnits:
    def test_value_without_solution_raises(self):
        m = dm.Model("srv")
        x = m.continuous("x", lb=0, ub=1)
        r = SolveResult(
            status="infeasible", objective=None, bound=None, gap=None, x=None, wall_time=0.0
        )
        with pytest.raises(ValueError, match="No solution available"):
            r.value(x)

    def test_gradient_without_model_raises(self):
        r = SolveResult(status="optimal", objective=0.0, bound=None, gap=None, x={}, wall_time=0.0)
        with pytest.raises(ValueError, match="No model attached"):
            r.gradient(None)

    def test_repr_and_template_explain(self):
        r = SolveResult(status="optimal", objective=1.5, bound=1.5, gap=0.0, x={}, wall_time=0.25)
        assert "status='optimal'" in repr(r)
        assert "Solved to optimal" in r.explain()

    def test_explain_llm_failure_falls_back_to_stored_explanation(self, monkeypatch):
        from discopt.llm import provider

        def boom(**kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(provider, "complete", boom)
        r = SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0, x={}, wall_time=0.1)
        r._explanation = "stored explanation"
        assert r.explain(llm=True) == "stored explanation"

    def test_explain_with_llm_uses_provider(self, monkeypatch):
        from discopt.llm import provider

        monkeypatch.setattr(provider, "complete", lambda **kwargs: "The solve reached the optimum.")
        m = dm.Model("llm_explain")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        r = SolveResult(
            status="optimal", objective=0.0, bound=0.0, gap=0.0, x={"x": 0.0}, wall_time=0.1
        )
        r._model = m
        assert "optimum" in r.explain(llm=True)


@pytest.mark.smoke
class TestSolveOptionPaths:
    def test_solve_with_llm_and_failing_validator(self, monkeypatch):
        """llm=True is advisory-only and a crashing examiner leaves report=None."""
        from discopt.llm import advisor
        from discopt.validation import examiner

        def boom(result, model):
            raise RuntimeError("examiner crashed")

        monkeypatch.setattr(examiner, "examine", boom)
        monkeypatch.setattr(advisor, "presolve_analysis", lambda model: ["check bounds"])
        m = dm.Model("solve_opts")
        x = m.continuous("x", lb=-1, ub=2)
        m.minimize((x - 1) ** 2)
        r = m.solve(time_limit=10, llm=True, validate=True)
        assert r.status == "optimal"
        assert r.objective == pytest.approx(0.0, abs=1e-6)
        assert r.validation_report is None

    def test_streaming_solve_not_implemented(self, monkeypatch):
        from discopt.llm import advisor

        def boom(model):
            raise RuntimeError("advisor crashed")

        # A crashing LLM advisor must never block solving (safety invariant).
        monkeypatch.setattr(advisor, "presolve_analysis", boom)
        m = dm.Model("stream")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        with pytest.raises(NotImplementedError, match="Streaming solve"):
            m.solve(stream=True, llm=True)

    def test_gradient_vector_parameter_reshape(self):
        m = dm.Model("grad_vec")
        p = m.parameter("p", value=np.array([1.0, 2.0]))
        xv = m.continuous("xv", shape=(2,), lb=-5, ub=5)
        m.minimize(dm.sum((xv - p) ** 2))
        r = m.solve(time_limit=10)
        g = r.gradient(p)
        assert isinstance(g, np.ndarray)
        assert g.shape == (2,)
        # At the optimum xv = p the sensitivity of the optimal value is 0.
        np.testing.assert_allclose(g, np.zeros(2), atol=1e-6)

    def test_repr_markdown_emits_display_math(self):
        m = dm.Model("md")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        md = m._repr_markdown_()
        assert md.startswith("$$")
        assert md.endswith("$$")


@pytest.mark.unit
class TestFromDescription:
    # from_description guards on litellm's presence before the mocked
    # completion layer is reached; skip cleanly where litellm isn't
    # installed (e.g. the python-fast CI job) and run in the coverage job.
    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("litellm") is None,
        reason="requires litellm (installed in the coverage CI job)",
    )

    class _Msg:
        def __init__(self, content, tool_calls=None):
            self._content = content
            self.tool_calls = tool_calls

        def model_dump(self, exclude_none=True):
            return {"role": "assistant", "content": self._content}

    class _Call:
        def __init__(self, call_id, name, arguments):
            self.id = call_id
            self.function = types.SimpleNamespace(name=name, arguments=arguments)

    def test_builds_model_from_mocked_tool_calls(self, monkeypatch, capsys):
        from discopt.llm import provider

        responses = iter(
            [
                self._Msg(
                    "building",
                    tool_calls=[
                        self._Call("1", "create_model", '{"name": "fd_model"}'),
                        self._Call("2", "add_continuous", '{"name": "u", "lb": 0.0, "ub": 1.0}'),
                        self._Call(
                            "3",
                            "set_objective",
                            '{"expression": "u", "sense": "minimize"}',
                        ),
                    ],
                ),
                self._Msg("Done: minimize u over [0, 1]."),
            ]
        )
        monkeypatch.setattr(provider, "complete_with_tools", lambda **kw: next(responses))
        model = core.from_description("minimize u in [0,1]", data={"demand": [1.0, 2.0]})
        assert model.name == "fd_model"
        assert [v.name for v in model._variables] == ["u"]
        model.validate()
        assert "LLM explanation" in capsys.readouterr().out

    def test_no_model_created_raises(self, monkeypatch):
        from discopt.llm import provider

        monkeypatch.setattr(
            provider, "complete_with_tools", lambda **kw: self._Msg("no tools used")
        )
        with pytest.raises(ValueError, match="LLM did not create a model"):
            core.from_description("do nothing")

    def test_unavailable_llm_raises_import_error(self, monkeypatch):
        import discopt.llm as llm_mod

        monkeypatch.setattr(llm_mod, "is_available", lambda: False)
        with pytest.raises(ImportError, match="requires litellm"):
            core.from_description("anything")
