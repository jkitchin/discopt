"""Tests for the general-purpose Outer Approximation (OA) solver.

Requires highspy for the MILP master problem.
"""

import discopt.modeling as dm
import numpy as np
import pytest

try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")

ABS_TOL = 1e-3
REL_TOL = 1e-3
INTEGRALITY_TOL = 1e-5


# ── Helper ────────────────────────────────────────────────────


def _solve_oa(model, **kwargs):
    """Solve model with OA and return result."""
    mip_nlp_method = "ecp" if kwargs.pop("ecp_mode", False) else "oa"
    defaults = dict(solver="mip-nlp", mip_nlp_method=mip_nlp_method, time_limit=60)
    defaults.update(kwargs)
    return model.solve(**defaults)


def _assert_optimal(result, expected_obj, abs_tol=ABS_TOL):
    assert result.status in ("optimal", "feasible"), (
        f"Expected optimal/feasible, got {result.status}"
    )
    assert result.objective == pytest.approx(expected_obj, abs=abs_tol)


def _assert_integer_feasible(result, int_var_names, model):
    """Check that integer/binary variables are integral."""
    for name in int_var_names:
        vals = np.atleast_1d(result.x[name])
        for v in vals.flat:
            assert abs(v - round(v)) < INTEGRALITY_TOL, f"Variable {name} = {v} is not integral"


def _assert_complete_optimal_result(result, expected_obj, var_names, abs_tol=ABS_TOL):
    assert result.status == "optimal"
    assert result.objective == pytest.approx(expected_obj, abs=abs_tol)
    assert result.bound is not None
    assert result.bound == pytest.approx(expected_obj, abs=abs_tol)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    for name in var_names:
        assert name in result.x


def _mindtpy_simple_minlp():
    """Native version of Pyomo MindtPy's small APSE convex MINLP baseline."""
    m = dm.Model("mindtpy_simple_minlp")
    x = m.continuous("x", shape=(2,), lb=0, ub=4)
    y = m.binary("y", shape=(3,))

    m.subject_to((x[0] - 2) ** 2 - x[1] <= 0)
    m.subject_to(x[0] - 2 * y[0] >= 0)
    m.subject_to(x[0] - x[1] - 4 * (1 - y[1]) <= 0)
    m.subject_to(x[0] - (1 - y[0]) >= 0)
    m.subject_to(x[1] - y[1] >= 0)
    m.subject_to(x[0] + x[1] >= 3 * y[2])
    m.subject_to(y[0] + y[1] + y[2] >= 1)
    m.minimize(y[0] + 1.5 * y[1] + 0.5 * y[2] + x[0] ** 2 + x[1] ** 2)
    return m


def _mindtpy_duran_grossmann_minlp():
    """Native version of Pyomo MindtPy's Duran-Grossmann OA/ECP baseline."""
    m = dm.Model("mindtpy_duran_grossmann")
    x = m.continuous("x", shape=(4,), lb=0, ub=[2, 2, 1, 100])
    y = m.binary("y", shape=(3,))

    m.subject_to(0.8 * dm.log(x[1] + 1) + 0.96 * dm.log(x[0] - x[1] + 1) - 0.8 * x[2] >= 0)
    m.subject_to(dm.log(x[1] + 1) + 1.2 * dm.log(x[0] - x[1] + 1) - x[2] - 2 * y[2] >= -2)
    m.subject_to(
        10 * x[0] - 7 * x[2] - 18 * dm.log(x[1] + 1) - 19.2 * dm.log(x[0] - x[1] + 1) + 10 - x[3]
        <= 0
    )
    m.subject_to(x[1] - x[0] <= 0)
    m.subject_to(x[1] - 2 * y[0] <= 0)
    m.subject_to(x[0] - x[1] - 2 * y[1] <= 0)
    m.subject_to(y[0] + y[1] <= 1)
    m.minimize(5 * y[0] + 6 * y[1] + 8 * y[2] + x[3])
    return m


def test_compute_gap_uses_absolute_scale_near_zero():
    from discopt.solvers.oa import _compute_gap

    assert _compute_gap(-1.99e-8, 9e-18) == pytest.approx(1.99e-8)


# ── Convex MINLP ─────────────────────────────────────────────


class TestOAConvexMINLP:
    """Convex MINLP problems where OA should find the global optimum."""

    @pytest.mark.slow
    def test_simple_quadratic_binary(self):
        """min x^2 + y, y in {0,1}, x + y >= 1, x in [0, 2].

        y=0 → x >= 1 → obj = 1
        y=1 → x >= 0 → obj = 1
        Both give obj=1; OA should find optimal=1.
        """
        m = dm.Model("simple_qb")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        result = _solve_oa(m)
        _assert_optimal(result, 1.0)

    @pytest.mark.slow
    def test_simple_minlp_from_examples(self):
        """Example simple MINLP: min x1^2 + x2^2 + x3, x3 binary.

        With x1 + x2 >= 1 and x1^2 + x2 <= 3.
        Optimal: x1=x2=0.5, x3=0, obj=0.5.
        """
        from discopt.modeling.examples import example_simple_minlp

        m = example_simple_minlp()
        result = _solve_oa(m)
        _assert_optimal(result, 0.5, abs_tol=0.05)
        _assert_integer_feasible(result, ["x3"], m)

    @pytest.mark.slow
    def test_convex_with_multiple_binaries(self):
        """min (x-3)^2 + 2*y1 + 3*y2, y1+y2 <= 1, x <= 2*y1 + 4*y2.

        y1=0,y2=0 → x <= 0 → obj = 9
        y1=1,y2=0 → x <= 2 → obj = 1 + 2 = 3
        y1=0,y2=1 → x <= 4 → obj = (3-3)^2 + 3 = 3
        Optimal: y1=1,y2=0,x=2 or y1=0,y2=1,x=3, both obj=3.
        """
        m = dm.Model("multi_binary")
        x = m.continuous("x", lb=0, ub=5)
        y1 = m.binary("y1")
        y2 = m.binary("y2")
        m.minimize((x - 3) ** 2 + 2 * y1 + 3 * y2)
        m.subject_to(y1 + y2 <= 1)
        m.subject_to(x <= 2 * y1 + 4 * y2)

        result = _solve_oa(m)
        _assert_optimal(result, 3.0, abs_tol=0.1)

    @pytest.mark.slow
    def test_linear_objective_nonlinear_constraints(self):
        """min x + 10*y, x^2 <= 4*y, x >= 0.5, y in {0,1}.

        y=0 → x^2 <= 0 → infeasible (x >= 0.5)
        y=1 → x^2 <= 4, x >= 0.5 → x=0.5, obj = 0.5 + 10 = 10.5
        """
        m = dm.Model("lin_obj")
        x = m.continuous("x", lb=0.5, ub=3)
        y = m.binary("y")
        m.minimize(x + 10 * y)
        m.subject_to(x**2 - 4 * y <= 0)

        result = _solve_oa(m)
        _assert_optimal(result, 10.5, abs_tol=0.5)
        assert result.x["y"] == pytest.approx(1.0, abs=INTEGRALITY_TOL)


class TestMindtPyBaselineParity:
    """Native discopt coverage for the small Pyomo MindtPy OA/ECP baselines."""

    @pytest.mark.parametrize(
        ("builder", "expected_obj", "abs_tol"),
        [
            (_mindtpy_simple_minlp, 3.5, 1e-3),
            (_mindtpy_duran_grossmann_minlp, 6.00976, 1e-3),
        ],
    )
    @pytest.mark.parametrize("method", ["oa", "ecp"])
    def test_small_mindtpy_baselines_report_complete_results(
        self, builder, expected_obj, abs_tol, method
    ):
        model = builder()
        result = model.solve(
            solver="mip-nlp",
            mip_nlp_method=method,
            time_limit=60,
            max_nodes=100,
        )

        _assert_complete_optimal_result(result, expected_obj, ["x", "y"], abs_tol=abs_tol)
        _assert_integer_feasible(result, ["y"], model)
        assert np.asarray(result.x["y"]).tolist() == pytest.approx([0.0, 1.0, 0.0])


# ── Non-convex MINLP ─────────────────────────────────────────


class TestOANonConvex:
    """Non-convex problems: OA may find local optimum, not global."""

    @pytest.mark.slow
    def test_nonconvex_finds_feasible(self):
        """min -x*y_bin, x in [0,2], y_bin in {0,1}, x <= 1 + y_bin.

        y=0 → x <= 1 → obj = 0 (x*0)
        y=1 → x <= 2 → obj = -2 (x=2, y=1)
        OA should at least find a feasible solution.
        """
        m = dm.Model("nonconvex")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.minimize(-(x * y))
        m.subject_to(x - y <= 1)

        result = _solve_oa(m)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None

    @pytest.mark.slow
    def test_nonconvex_objective_skips_objective_oa_cuts(self, monkeypatch):
        """A nonconvex objective must not produce OA objective cuts or certified bounds."""
        from discopt._jax import cutting_planes

        calls = []
        real_generate = cutting_planes.generate_objective_oa_cut

        def wrapped_generate(*args, **kwargs):
            calls.append((args, kwargs))
            return real_generate(*args, **kwargs)

        monkeypatch.setattr(cutting_planes, "generate_objective_oa_cut", wrapped_generate)

        m = dm.Model("oa_nonconvex_objective")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.subject_to(x <= 1 + y)
        m.minimize(-(x * y))

        result = _solve_oa(m, max_nodes=6)

        assert calls == []
        assert result.status == "feasible"
        assert result.objective is not None
        assert result.bound is None
        assert result.gap is None


# ── Edge Cases ────────────────────────────────────────────────


class TestOAEdgeCases:
    """Edge cases and degenerate problems."""

    def test_no_discrete_short_circuit(self):
        """No integer variables: MIP-NLP should solve one continuous NLP."""
        m = dm.Model("no_discrete_short_circuit")
        x = m.continuous("x", lb=0, ub=10)
        m.subject_to(x**2 >= 1)
        m.minimize(x)

        result = _solve_oa(m)
        _assert_complete_optimal_result(result, 1.0, ["x"], abs_tol=0.01)
        assert result.x["x"] == pytest.approx(1.0, abs=0.01)

    def test_pure_milp_all_linear(self):
        """All-linear MINLP: OA should converge in one iteration."""
        m = dm.Model("pure_milp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x + 5 * y)
        m.subject_to(x + y >= 1)

        result = _solve_oa(m)
        _assert_optimal(result, 1.0, abs_tol=0.1)

    @pytest.mark.slow
    def test_infeasible_model(self):
        """Infeasible MINLP: contradictory constraints."""
        m = dm.Model("infeasible")
        x = m.continuous("x", lb=0, ub=1)
        y = m.binary("y")
        m.minimize(x + y)
        m.subject_to(x >= 2)  # infeasible: x in [0,1] but x >= 2

        result = _solve_oa(m)
        assert result.status == "infeasible"
        assert result.objective is None
        assert result.gap is None
        assert result.x == {}

    @pytest.mark.slow
    def test_single_iteration_optimal(self):
        """NLP relaxation is already integer-feasible → immediate convergence."""
        m = dm.Model("trivial")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        # Optimal at y=1, x=0 → obj=1. Relaxation likely finds this.
        m.minimize(x**2 + y)
        m.subject_to(y >= 1)

        result = _solve_oa(m)
        _assert_optimal(result, 1.0, abs_tol=0.1)


# ── Infeasible NLP Handling ───────────────────────────────────


class TestOAInfeasibleNLP:
    """Tests for handling infeasible NLP subproblems."""

    @pytest.mark.slow
    def test_some_assignments_infeasible(self):
        """Problem where one binary assignment makes NLP infeasible.

        y=0: x^2 <= -1 (infeasible)
        y=1: x^2 <= 3, min x^2 + 1 → x=0, obj=1
        """
        m = dm.Model("partial_infeas")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x**2 - 4 * y + 1 <= 0)

        result = _solve_oa(m)
        _assert_optimal(result, 1.0, abs_tol=0.5)
        assert result.x["y"] == pytest.approx(1.0, abs=INTEGRALITY_TOL)


# ── ECP Mode ─────────────────────────────────────────────────


class TestECPMode:
    """Extended Cutting Plane mode (no NLP subproblem solves)."""

    def test_ecp_convex_minlp(self):
        """ECP should converge on a convex MINLP."""
        m = dm.Model("ecp_test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        result = _solve_oa(m, ecp_mode=True)
        _assert_optimal(result, 1.0, abs_tol=0.2)

    def test_ecp_linear_objective(self):
        """ECP with linear objective + nonlinear constraints."""
        m = dm.Model("ecp_lin")
        x = m.continuous("x", lb=0, ub=3)
        y = m.binary("y")
        m.minimize(x + 10 * y)
        m.subject_to(x**2 - 4 * y <= 0)
        m.subject_to(x >= 0.5)

        result = _solve_oa(m, ecp_mode=True)
        assert result.status in ("optimal", "feasible")


# ── Equality Relaxation ──────────────────────────────────────


class TestEqualityRelaxation:
    """Tests for equality relaxation (ER) strategy."""

    def test_er_helps_nonlinear_equality(self):
        """Nonlinear equality that may cause master infeasibility.

        min x^2 + y, x^2 == y (nonlinear equality), x in [0,2], y in {0,1}.
        y=0 → x=0, obj=0
        y=1 → x=1, obj=2

        With ER, the equality is relaxed to x^2 <= y in OA cuts.
        """
        m = dm.Model("er_test")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x**2 - y == 0)

        result = _solve_oa(m, equality_relaxation=True)
        assert result.status in ("optimal", "feasible")


class TestOARobustnessOptions:
    """MindtPy-style OA robustness controls."""

    def test_feasibility_norm_merit_values(self):
        from discopt.solvers.oa import (
            _constraint_violation_merit,
            _decompose_model,
            _normalize_feasibility_norm,
        )

        m = dm.Model("feasibility_norm_merit")
        x = m.continuous("x", shape=(2,), lb=-10, ub=10)
        m.subject_to(x[0] <= 0)
        m.subject_to(x[1] >= 0)
        m.minimize(x[0])

        evaluator = _decompose_model(m).evaluator
        point = np.array([3.0, -4.0], dtype=float)

        assert _constraint_violation_merit(evaluator, point, "L1") == pytest.approx(7.0)
        assert _constraint_violation_merit(evaluator, point, "L2") == pytest.approx(25.0)
        assert _constraint_violation_merit(evaluator, point, "L_infinity") == pytest.approx(4.0)
        assert _normalize_feasibility_norm("l-inf") == "L_infinity"
        with pytest.raises(ValueError, match="Unknown feasibility_norm"):
            _normalize_feasibility_norm("L0")

    def test_master_slack_penalizes_and_relaxes_constraint_cuts(self, monkeypatch):
        from discopt.solvers import MILPResult, SolveStatus, lp_backend
        from discopt.solvers.oa import _solve_master_milp

        captured = {}

        def fake_solve_milp(**kwargs):
            captured.update(kwargs)
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.zeros(2),
                objective=0.0,
                bound=0.0,
            )

        monkeypatch.setattr(
            lp_backend,
            "get_milp_solver",
            lambda backend="auto": fake_solve_milp,
        )

        _solve_master_milp(
            linear_A_rows=[],
            linear_b_rows=[],
            linear_senses=[],
            oa_A_rows=[np.array([1.0])],
            oa_b_rows=[2.0],
            n_vars=1,
            integrality=np.array([0], dtype=np.int32),
            lb=np.array([0.0]),
            ub=np.array([10.0]),
            obj_coeffs=(np.array([0.0]), 0.0),
            obj_is_linear=True,
            objective_bound_valid=True,
            time_limit=10,
            gap_tolerance=1e-4,
            add_slack=True,
            max_slack=5.0,
            oa_penalty_factor=17.0,
        )

        assert captured["c"].tolist() == pytest.approx([0.0, 17.0])
        np.testing.assert_allclose(captured["A_ub"], np.array([[1.0, -1.0]]))
        assert captured["b_ub"].tolist() == pytest.approx([2.0])
        assert captured["bounds"] == [(0.0, 10.0), (0.0, 5.0)]
        assert captured["integrality"].tolist() == [0, 0]

    def test_master_slack_does_not_relax_no_good_cuts(self, monkeypatch):
        from discopt.solvers import MILPResult, SolveStatus, lp_backend
        from discopt.solvers.oa import _add_no_good_cut, _solve_master_milp

        captured = {}

        def fake_solve_milp(**kwargs):
            captured.update(kwargs)
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.zeros(2),
                objective=0.0,
                bound=0.0,
            )

        monkeypatch.setattr(
            lp_backend,
            "get_milp_solver",
            lambda backend="auto": fake_solve_milp,
        )

        oa_A_rows = []
        oa_b_rows = []
        oa_cut_relaxable = []
        _add_no_good_cut(
            np.array([1.0]),
            [0],
            oa_A_rows,
            oa_b_rows,
            n_vars=1,
            oa_cut_relaxable=oa_cut_relaxable,
        )

        _solve_master_milp(
            linear_A_rows=[],
            linear_b_rows=[],
            linear_senses=[],
            oa_A_rows=oa_A_rows,
            oa_b_rows=oa_b_rows,
            n_vars=1,
            integrality=np.array([1], dtype=np.int32),
            lb=np.array([0.0]),
            ub=np.array([1.0]),
            obj_coeffs=(np.array([0.0]), 0.0),
            obj_is_linear=True,
            objective_bound_valid=True,
            time_limit=10,
            gap_tolerance=1e-4,
            add_slack=True,
            max_slack=5.0,
            oa_penalty_factor=17.0,
            oa_cut_relaxable=oa_cut_relaxable,
        )

        np.testing.assert_allclose(captured["A_ub"], np.array([[1.0, 0.0]]))
        assert captured["b_ub"].tolist() == pytest.approx([0.0])

    def test_no_good_cut_uses_binary_indices_only_for_mixed_assignment(self):
        from discopt.solvers.oa import _add_no_good_cut

        oa_A_rows = []
        oa_b_rows = []
        oa_cut_relaxable = []

        added = _add_no_good_cut(
            np.array([1.0, 2.0, 0.0]),
            [0, 2],
            oa_A_rows,
            oa_b_rows,
            n_vars=3,
            oa_cut_relaxable=oa_cut_relaxable,
        )

        assert added is True
        np.testing.assert_allclose(oa_A_rows[0], np.array([1.0, 0.0, -1.0]))
        assert oa_b_rows == pytest.approx([0.0])
        assert oa_cut_relaxable == [False]

    def test_no_good_cut_skips_when_no_binary_indices(self):
        from discopt.solvers.oa import _add_no_good_cut

        oa_A_rows = []
        oa_b_rows = []
        oa_cut_relaxable = []

        added = _add_no_good_cut(
            np.array([2.0]),
            [],
            oa_A_rows,
            oa_b_rows,
            n_vars=1,
            oa_cut_relaxable=oa_cut_relaxable,
        )

        assert added is False
        assert oa_A_rows == []
        assert oa_b_rows == []
        assert oa_cut_relaxable == []

    def test_projection_no_good_cut_uses_binary_indices_in_mixed_model(self):
        from discopt.solvers.oa import (
            _append_binary_no_good_projection_cut,
            _decompose_model,
        )

        m = dm.Model("mixed_projection_no_good")
        y = m.binary("y")
        z = m.integer("z", lb=0, ub=3)
        m.minimize(y + z)
        decomp = _decompose_model(m)
        a_rows = []
        b_rows = []

        added = _append_binary_no_good_projection_cut(
            decomp,
            assignment=(1.0, 2.0),
            n_master=2,
            a_rows=a_rows,
            b_rows=b_rows,
        )

        assert added is True
        np.testing.assert_allclose(a_rows[0], np.array([1.0, 0.0]))
        assert b_rows == pytest.approx([0.0])

    @pytest.mark.parametrize(("enabled", "expected_calls"), [(False, 0), (True, 1)])
    def test_no_good_cut_option_controls_infeasible_assignment(
        self,
        monkeypatch,
        enabled,
        expected_calls,
    ):
        import discopt.solvers.oa as oa_module
        from discopt.solvers import MILPResult, SolveStatus

        m = dm.Model("no_good_option")
        y = m.binary("y")
        m.minimize(y)

        no_good_calls = []

        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_relaxation",
            lambda *args, **kwargs: (np.array([0.5]), 0.5),
        )
        monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            oa_module,
            "_solve_master_milp",
            lambda *args, **kwargs: MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([1.0]),
                objective=1.0,
                bound=0.0,
            ),
        )
        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_subproblem",
            lambda *args, **kwargs: (None, None),
        )
        monkeypatch.setattr(
            oa_module,
            "_add_no_good_cut",
            lambda *args, **kwargs: no_good_calls.append(args),
        )

        result = oa_module.solve_oa(
            m,
            max_iterations=1,
            feasibility_cuts=False,
            add_no_good_cuts=enabled,
            cycling_check=False,
        )

        assert result.status == "infeasible"
        assert len(no_good_calls) == expected_calls

    def test_multitree_no_good_cut_passes_only_binary_indices_for_mixed_model(
        self,
        monkeypatch,
    ):
        import discopt.solvers.oa as oa_module
        from discopt.solvers import MILPResult, SolveStatus

        m = dm.Model("mixed_no_good_option")
        y = m.binary("y")
        z = m.integer("z", lb=0, ub=3)
        m.minimize(y + z)

        no_good_indices = []

        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_relaxation",
            lambda *args, **kwargs: (np.array([0.5, 1.5]), 2.0),
        )
        monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            oa_module,
            "_solve_master_milp",
            lambda *args, **kwargs: MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([1.0, 2.0]),
                objective=3.0,
                bound=0.0,
            ),
        )
        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_subproblem",
            lambda *args, **kwargs: (None, None),
        )
        monkeypatch.setattr(
            oa_module,
            "_add_no_good_cut",
            lambda _x_master, binary_indices, *args, **kwargs: no_good_indices.append(
                list(binary_indices)
            ),
        )

        result = oa_module.solve_oa(
            m,
            max_iterations=1,
            feasibility_cuts=False,
            add_no_good_cuts=True,
            cycling_check=False,
        )

        assert result.status == "infeasible"
        assert no_good_indices == [[0]]

    def test_cycling_check_stops_repeated_integer_assignment(self, monkeypatch):
        import discopt.solvers.oa as oa_module
        from discopt.solvers import MILPResult, SolveStatus

        m = dm.Model("cycling_check")
        y = m.binary("y")
        m.minimize(y)

        master_calls = 0
        nlp_calls = 0

        def fake_master(*args, **kwargs):
            nonlocal master_calls
            master_calls += 1
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([1.0]),
                objective=1.0,
                bound=0.0,
            )

        def fake_nlp(*args, **kwargs):
            nonlocal nlp_calls
            nlp_calls += 1
            return np.array([1.0]), 1.0

        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_relaxation",
            lambda *args, **kwargs: (np.array([0.5]), 0.5),
        )
        monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
        monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
        monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_nlp)

        result = oa_module.solve_oa(
            m,
            max_iterations=5,
            add_no_good_cuts=False,
            cycling_check=True,
        )

        assert result.status == "feasible"
        assert master_calls == 2
        assert nlp_calls == 1

    def test_stalling_limit_stops_without_incumbent_progress(self, monkeypatch):
        import discopt.solvers.oa as oa_module
        from discopt.solvers import MILPResult, SolveStatus

        m = dm.Model("stalling_limit")
        y = m.binary("y")
        m.minimize(y)

        nlp_calls = 0

        def fake_nlp(*args, **kwargs):
            nonlocal nlp_calls
            nlp_calls += 1
            return np.array([1.0]), 1.0

        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_relaxation",
            lambda *args, **kwargs: (np.array([0.5]), 0.5),
        )
        monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            oa_module,
            "_solve_master_milp",
            lambda *args, **kwargs: MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([1.0]),
                objective=1.0,
                bound=0.0,
            ),
        )
        monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_nlp)

        result = oa_module.solve_oa(
            m,
            max_iterations=5,
            add_no_good_cuts=False,
            cycling_check=False,
            stalling_limit=2,
        )

        assert result.status == "feasible"
        assert nlp_calls == 2

    def test_heuristic_nonconvex_enables_slack_and_uncertified_result(self, monkeypatch):
        import discopt.solvers.oa as oa_module
        from discopt.solvers import MILPResult, SolveStatus

        m = dm.Model("heuristic_nonconvex_controls")
        y = m.binary("y")
        m.minimize(y)

        cut_kwargs = []
        master_kwargs = []

        def fake_add_oa_cuts(*args, **kwargs):
            cut_kwargs.append(kwargs)

        def fake_master(*args, **kwargs):
            master_kwargs.append(kwargs)
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([1.0]),
                objective=1.0,
                bound=1.0,
            )

        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_relaxation",
            lambda *args, **kwargs: (np.array([0.5]), 0.5),
        )
        monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)
        monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_subproblem",
            lambda *args, **kwargs: (np.array([1.0]), 1.0),
        )

        result = oa_module.solve_oa(
            m,
            max_iterations=1,
            equality_relaxation=False,
            add_slack=False,
            heuristic_nonconvex=True,
        )

        assert result.status == "feasible"
        assert result.bound is None
        assert result.gap is None
        assert cut_kwargs[0]["equality_relaxation"] is True
        assert master_kwargs[0]["add_slack"] is True

    def test_heuristic_nonconvex_nonlinear_objective_is_uncertified_end_to_end(self):
        result = _solve_oa(
            _mindtpy_simple_minlp(),
            heuristic_nonconvex=True,
            max_nodes=10,
        )

        assert result.status == "feasible"
        assert result.objective == pytest.approx(3.5, abs=1e-3)
        assert result.bound is None
        assert result.gap is None

    def test_no_good_cuts_preserve_certified_convex_bound(self, monkeypatch):
        import discopt.solvers.oa as oa_module
        from discopt.solvers import MILPResult, SolveStatus

        m = dm.Model("no_good_certified_bound")
        y = m.binary("y")
        m.minimize(y)

        master_points = [np.array([1.0]), np.array([0.0])]
        master_bounds = [0.0, 0.0]

        def fake_master(*args, **kwargs):
            idx = min(fake_master.calls, len(master_points) - 1)
            fake_master.calls += 1
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=master_points[idx],
                objective=float(master_points[idx][0]),
                bound=master_bounds[idx],
            )

        fake_master.calls = 0

        def fake_nlp(*args, **kwargs):
            x_master = np.asarray(args[4], dtype=float)
            if x_master[0] > 0.5:
                return None, None
            return np.array([0.0]), 0.0

        monkeypatch.setattr(
            oa_module,
            "_solve_nlp_relaxation",
            lambda *args, **kwargs: (np.array([0.5]), 0.5),
        )
        monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
        monkeypatch.setattr(oa_module, "_add_feasibility_cuts", lambda *args, **kwargs: None)
        monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
        monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_nlp)

        result = oa_module.solve_oa(
            m,
            max_iterations=2,
            feasibility_cuts=False,
            add_no_good_cuts=True,
        )

        assert result.status == "optimal"
        assert result.objective == pytest.approx(0.0)
        assert result.bound == pytest.approx(0.0)
        assert result.gap == pytest.approx(0.0)


# ── Regression vs B&B ────────────────────────────────────────


@pytest.mark.slow
class TestOAMatchesBnB:
    """OA results should be close to B&B on shared test problems."""

    def test_simple_minlp_matches(self):
        """Compare OA and default B&B on simple MINLP."""
        from discopt.modeling.examples import example_simple_minlp

        m_oa = example_simple_minlp()
        m_bb = example_simple_minlp()

        result_oa = _solve_oa(m_oa)
        result_bb = m_bb.solve(time_limit=60)

        # Both should find feasible solutions
        assert result_oa.status in ("optimal", "feasible")
        assert result_bb.status in ("optimal", "feasible")

        # Objectives should be close (within tolerance)
        if result_oa.objective is not None and result_bb.objective is not None:
            assert result_oa.objective == pytest.approx(result_bb.objective, abs=0.5)


# ── Maximize objective (sense handling) ───────────────────────
# Regression for the OA minimization-convention bug: the master MILP minimized
# the raw objective coefficients while the NLP subproblems and epigraph cuts ran
# in the evaluator's minimization convention (which negates a MAXIMIZE
# objective). The two disagreed on direction, so OA converged to — and reported
# as "optimal" — the *minimum* of a maximize problem (syn05m: -831 vs the true
# maximum 837.73). These guard that OA optimizes the correct direction.


def test_oa_maximize_linear_objective_is_not_the_minimum():
    """A MAXIMIZE model must return the maximum, not the (certified-wrong) min."""
    m = dm.Model("oa_max")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.maximize(x + y)
    m.subject_to(x**2 + y**2 <= 10)
    r = _solve_oa(m)
    assert r.status == "optimal"
    _assert_optimal(r, 4.0)  # max x+y over integers with x^2+y^2 <= 10
    _assert_integer_feasible(r, ["x", "y"], m)
    # For a maximization the dual bound sits at/above the optimum.
    assert r.bound is None or r.bound >= r.objective - ABS_TOL


def test_oa_maximize_scalar_reaches_true_max():
    """The pre-fix bug returned 0 (the minimum) as 'optimal'; the true max is 9."""
    m = dm.Model("oa_max2")
    x = m.integer("x", lb=0, ub=4)
    m.maximize(3 * x)
    m.subject_to(x**2 <= 9)  # x <= 3
    r = _solve_oa(m)
    assert r.objective == pytest.approx(9.0, abs=ABS_TOL)
