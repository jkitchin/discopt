import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult, _DisjunctiveConstraint

try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False


def _require_gurobi():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        env.dispose()
    except Exception as exc:
        pytest.skip(f"Gurobi is installed but no usable license is available: {exc}")


def _binary_model(name="mip_nlp_route"):
    m = dm.Model(name)
    x = m.binary("x")
    m.minimize(x)
    return m


def _continuous_model(name="mip_nlp_continuous"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=4)
    m.minimize((x - 2) ** 2)
    return m


def _convex_general_nlp_model(name="shot_direct_nlp"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize(dm.exp(x))
    return m


def _gdp_model(name="mip_nlp_gdp_route"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.either_or([[x <= 3], [x >= 7]], name="mode")
    return m


def _mixed_discrete_model(name="mip_nlp_init_strategy"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2, ub=2)
    y = m.binary("y", shape=(2,))
    z = m.integer("z", lb=-2, ub=3)
    m.minimize(x + y[0] + y[1] + z)
    return m


def _convex_binary_nonlinear_model(name="mip_nlp_cut_trace"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=4)
    y = m.binary("y")
    m.subject_to((x - 2) ** 2 + y - 1 <= 0)
    m.minimize((x - 1) ** 2 + y)
    return m


def _objective_epigraph_model(name="shot_objective_epigraph"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2, ub=2)
    z = m.continuous("z", lb=-1e20, ub=1e20)
    m.subject_to(z - (x - 1) ** 2 == 0.0)
    m.minimize(z)
    return m


def _anti_epigraph_model(name="shot_anti_epigraph"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2, ub=2)
    z = m.continuous("z", lb=-1e20, ub=1e20)
    m.subject_to(z + (x - 1) ** 2 == 0.0)
    m.maximize(z)
    return m


def _bilinear_partition_model(name="shot_bilinear_partition"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=2)
    m.subject_to(x * y <= 1.0)
    m.minimize(x + y)
    return m


def _quadratic_partition_model(name="shot_quadratic_partition"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=0, ub=2)
    m.subject_to(x**2 + y <= 3.0)
    m.minimize(y)
    return m


def _absolute_value_model(name="shot_abs_aux"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2, ub=2)
    m.subject_to(dm.abs(x) <= 1.0)
    m.minimize(x)
    return m


def _monomial_model(name="shot_monomial"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=2)
    m.subject_to(x**3 + y <= 4.0)
    m.minimize(y)
    return m


def _signomial_model(name="shot_signomial"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.subject_to((x**1.5) * (y**-0.5) <= 4.0)
    m.minimize(x + y)
    return m


def _integer_bilinear_model(name="shot_integer_bilinear"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=5)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y)
    return m


def _miqp_style_model(name="shot_miqp_style"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2, ub=2)
    y = m.binary("y")
    m.minimize((x - 1) ** 2 + y)
    return m


def _miqcqp_style_model(name="shot_miqcqp_style"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2, ub=2)
    y = m.binary("y")
    m.subject_to(x**2 + y <= 3.0)
    m.minimize((x - 1) ** 2 + y)
    return m


def _initial_poa_fixture(name="shot_initial_poa_seed"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.binary("y")
    m.subject_to((x - 2.0) ** 2 - y <= 0.0)
    m.minimize(x + y)
    return m


def _esh_convex_fixture(name="shot_esh_convex"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.binary("y")
    m.subject_to(x**2 - 1.0 <= 0.0)
    m.minimize(-x - 0.1 * y)
    return m


def _esh_objective_fixture(name="shot_esh_objective"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.binary("y")
    m.subject_to(x**2 - 1.0 <= 0.0)
    m.minimize(x**2 - 0.1 * y)
    return m


def _esh_candidate_feasible_objective_fixture(name="shot_esh_candidate_feasible_objective"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.binary("y")
    m.subject_to(x**2 - 4.0 <= 0.0)
    m.minimize(x**2 + x - 0.1 * y)
    return m


def _esh_nonconvex_fixture(name="shot_esh_nonconvex"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.binary("y")
    m.subject_to(1.0 - x**2 <= 0.0)
    m.minimize(y)
    return m


def _esh_nonconvex_objective_fixture(name="shot_esh_nonconvex_objective"):
    m = dm.Model(name)
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.binary("y")
    m.subject_to(x**2 - 1.0 <= 0.0)
    m.minimize((x - 1.0) ** 3 - 0.1 * y)
    return m


def test_mip_nlp_cut_record_construction_and_dedup():
    from discopt.solvers.oa import MIPNLPCutProvenance, MIPNLPCutRecord, _append_master_cut

    record = MIPNLPCutRecord.from_row(
        "oa",
        np.array([1.0, -1.0]),
        0.5,
        global_valid=True,
        supporting_point=np.array([2.0, 1.0]),
        constraint_id=3,
    )

    assert record.source == "oa"
    assert record.coefficients == (1.0, -1.0)
    assert record.rhs == pytest.approx(0.5)
    assert record.violation == pytest.approx(0.5)
    assert record.constraint_id == 3

    provenance = MIPNLPCutProvenance()
    oa_A_rows = []
    oa_b_rows = []
    _append_master_cut(
        oa_A_rows,
        oa_b_rows,
        np.array([1.0, -1.0]),
        0.5,
        cut_provenance=provenance,
        source="oa",
        global_valid=True,
    )
    _append_master_cut(
        oa_A_rows,
        oa_b_rows,
        np.array([1.0, -1.0]),
        0.5,
        cut_provenance=provenance,
        source="ecp",
        global_valid=True,
    )
    _append_master_cut(
        oa_A_rows,
        oa_b_rows,
        np.array([1.0, 1.0]),
        1.0,
        cut_provenance=provenance,
        source="integer",
        global_valid=True,
    )

    assert len(oa_A_rows) == 3
    assert len(provenance.records) == 2
    assert provenance.source_counts()["oa"] == 1
    assert provenance.source_counts()["ecp"] == 0
    assert provenance.source_counts()["integer"] == 1


def test_mip_nlp_trace_exposes_cut_source_counts(monkeypatch):
    import discopt.solvers.oa as oa_module

    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_relaxation",
        lambda *args, **kwargs: (np.array([1.5, 0.0]), 0.25),
    )

    result = oa_module.solve_oa(
        _convex_binary_nonlinear_model(),
        max_iterations=0,
        init_strategy="rNLP",
    )

    assert result.mip_nlp_trace is not None
    summary = result.mip_nlp_trace["summary"]
    counts = summary["cut_source_counts"]
    assert summary["cut_count"] == 2
    assert summary["provenance_cut_count"] == 2
    assert counts["oa"] == 1
    assert counts["objective"] == 1
    assert counts["ecp"] == 0
    assert counts["feasibility"] == 0
    assert counts["integer"] == 0


def test_mip_nlp_shot_initial_poa_imports_cuts_with_provenance(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    cut_sources = []

    def fake_solve_nlp_subproblem(*args, **kwargs):
        x_master = np.asarray(args[4], dtype=float)
        return x_master, 0.0

    def fake_solve_initial_poa_master(*args, **kwargs):
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0, 0.0], dtype=float),
            objective=-2.0,
            bound=-2.0,
            node_count=3,
        )

    def fake_add_oa_cuts(
        evaluator,
        x_star,
        n_vars,
        n_cons,
        constraint_senses,
        oa_A_rows,
        oa_b_rows,
        *args,
        **kwargs,
    ):
        source = kwargs.get("constraint_source", "oa")
        cut_sources.append(source)
        coeffs = np.zeros(n_vars, dtype=float)
        coeffs[0] = -1.0 if source == "initial_poa" else 1.0
        rhs = -1.0 if source == "initial_poa" else 10.0
        oa_module._append_master_cut(
            oa_A_rows,
            oa_b_rows,
            coeffs,
            rhs,
            kwargs.get("oa_cut_relaxable"),
            cut_provenance=kwargs.get("cut_provenance"),
            source=source,
            global_valid=True,
            supporting_point=x_star,
        )

    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_solve_nlp_subproblem)
    monkeypatch.setattr(oa_module, "_solve_initial_poa_master", fake_solve_initial_poa_master)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)

    result = oa_module.solve_oa(
        _initial_poa_fixture("initial_poa_unit_import"),
        init_strategy="initial_binary",
        max_iterations=0,
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(relaxation_phase="initial"),
    )

    trace = result.mip_nlp_trace
    assert "initial_poa" in cut_sources
    assert trace["initial_poa"]["status"] == "seeded"
    assert trace["initial_poa"]["objective_bound"] == pytest.approx(-2.0)
    assert trace["initial_poa"]["objective_bound_valid"] is True
    assert trace["initial_poa"]["node_count"] == 3
    assert trace["summary"]["initial_poa_cuts"] == 1
    assert trace["summary"]["cut_source_counts"]["initial_poa"] == 1


def test_mip_nlp_shot_initial_poa_fallback_preserves_initialization(monkeypatch):
    import discopt.solvers.oa as oa_module

    def fake_solve_nlp_subproblem(*args, **kwargs):
        x_master = np.asarray(args[4], dtype=float)
        return x_master, 0.0

    def fake_solve_initial_poa_master(*args, **kwargs):
        raise RuntimeError("poa unavailable")

    def fake_add_oa_cuts(
        evaluator,
        x_star,
        n_vars,
        n_cons,
        constraint_senses,
        oa_A_rows,
        oa_b_rows,
        *args,
        **kwargs,
    ):
        coeffs = np.zeros(n_vars, dtype=float)
        coeffs[0] = 1.0
        oa_module._append_master_cut(
            oa_A_rows,
            oa_b_rows,
            coeffs,
            10.0,
            kwargs.get("oa_cut_relaxable"),
            cut_provenance=kwargs.get("cut_provenance"),
            source=kwargs.get("constraint_source", "oa"),
            global_valid=True,
            supporting_point=x_star,
        )

    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_solve_nlp_subproblem)
    monkeypatch.setattr(oa_module, "_solve_initial_poa_master", fake_solve_initial_poa_master)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)

    result = oa_module.solve_oa(
        _initial_poa_fixture("initial_poa_unit_fallback"),
        init_strategy="initial_binary",
        max_iterations=0,
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(relaxation_phase="initial"),
    )

    assert result.status == "feasible"
    trace = result.mip_nlp_trace
    assert trace["initial_poa"]["attempted"] is True
    assert trace["initial_poa"]["status"] == "fallback"
    assert "RuntimeError: poa unavailable" in trace["initial_poa"]["fallback_reason"]
    assert trace["summary"]["cut_source_counts"]["initial_poa"] == 0


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
def test_mip_nlp_shot_initial_poa_adds_initial_cuts_integration():
    import discopt.solvers.oa as oa_module

    disabled = oa_module.solve_oa(
        _initial_poa_fixture("initial_poa_disabled"),
        init_strategy="initial_binary",
        ecp_mode=True,
        max_iterations=0,
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(relaxation_phase="off"),
    )
    enabled = oa_module.solve_oa(
        _initial_poa_fixture("initial_poa_enabled"),
        init_strategy="initial_binary",
        ecp_mode=True,
        max_iterations=0,
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(relaxation_phase="initial"),
    )

    disabled_trace = disabled.mip_nlp_trace
    enabled_trace = enabled.mip_nlp_trace
    assert disabled_trace["initial_poa"]["status"] == "disabled"
    assert enabled_trace["initial_poa"]["status"] == "seeded"
    assert enabled_trace["summary"]["cut_count"] > disabled_trace["summary"]["cut_count"]
    assert enabled_trace["summary"]["cut_source_counts"]["initial_poa"] >= 1


def test_mip_nlp_shot_adaptive_solution_limit_state_transitions():
    import discopt.solvers.oa as oa_module

    state = oa_module._ShotMIPSolutionLimitState(
        strategy="adaptive",
        capacity=3,
        backend="gurobi",
    )

    assert state.as_trace_dict()["limit"] == 1
    update = state.observe_iteration(
        incumbent_improved=False,
        cuts_added=0,
        master_status="optimal",
    )
    assert update["raw_limit"] == 2
    assert update["update_reason"] == "no_new_cuts"

    update = state.observe_iteration(
        incumbent_improved=False,
        cuts_added=0,
        master_status="optimal",
    )
    assert update["raw_limit"] == 3

    update = state.observe_iteration(
        incumbent_improved=True,
        cuts_added=0,
        master_status="optimal",
    )
    assert update["raw_limit"] == 1
    assert update["update_reason"] == "incumbent_improved"

    unsupported = oa_module._ShotMIPSolutionLimitState(
        strategy="static",
        capacity=2,
        backend="highs",
    )
    assert unsupported.as_trace_dict()["limit"] is None
    assert "gurobi" in unsupported.as_trace_dict()["degraded_reason"]


def test_mip_nlp_shot_master_controls_forward_to_gurobi(monkeypatch):
    import discopt.solvers.gurobi as gurobi_module
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    captured = {}

    def fake_solve_milp(**kwargs):
        captured.update(kwargs)
        return MILPResult(status=SolveStatus.OPTIMAL, x=np.array([1.0]), bound=1.0)

    monkeypatch.setattr(gurobi_module, "solve_milp", fake_solve_milp)

    oa_module._solve_master_milp(
        linear_A_rows=[],
        linear_b_rows=[],
        linear_senses=[],
        oa_A_rows=[],
        oa_b_rows=[],
        n_vars=1,
        integrality=np.array([1], dtype=np.int32),
        lb=np.array([0.0]),
        ub=np.array([1.0]),
        obj_coeffs=(np.array([1.0]), 0.0),
        obj_is_linear=True,
        objective_bound_valid=True,
        time_limit=10.0,
        gap_tolerance=1e-4,
        milp_solver="gurobi",
        mip_start=np.array([1.0]),
        mip_start_objective=1.0,
        objective_cutoff=1.1,
        mip_solution_limit=2,
    )

    assert captured["options"]["Cutoff"] == pytest.approx(1.1)
    assert captured["options"]["SolutionLimit"] == 2
    assert captured["mip_start"].tolist() == pytest.approx([1.0])


def test_mip_nlp_shot_objective_cutoff_uses_master_objective_units(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    m = dm.Model("shot_cutoff_linear_offset")
    y = m.binary("y")
    m.minimize(y - 100.0)

    master_calls = []

    def fake_master(*args, **kwargs):
        master_calls.append(kwargs)
        return MILPResult(
            status=SolveStatus.CUTOFF,
            x=None,
            objective=None,
            bound=1.0,
        )

    def fake_nlp(*args, **kwargs):
        x_master = np.asarray(args[4], dtype=float)
        return x_master.copy(), float(x_master[0] - 100.0)

    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_nlp)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    result = oa_module.solve_oa(
        m,
        init_strategy="initial_binary",
        max_iterations=1,
        milp_solver="gurobi",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            relaxation_phase="off",
            mip_solution_limit_strategy="none",
        ),
    )

    assert master_calls[0]["objective_cutoff"] == pytest.approx(1.0 + 2e-8)
    assert master_calls[0]["mip_start_objective"] == pytest.approx(1.0)
    assert result.objective == pytest.approx(-99.0)
    assert result.bound == pytest.approx(-99.0)


def test_mip_nlp_shot_master_repair_resets_controls_and_continues(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    master_calls = []

    def fake_relaxation(*args, **kwargs):
        del args, kwargs
        return np.array([0.0], dtype=float), 0.0

    def fake_master(*args, **kwargs):
        del args
        master_calls.append(kwargs)
        if len(master_calls) == 1:
            return MILPResult(status=SolveStatus.INFEASIBLE, node_count=2)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0], dtype=float),
            objective=0.0,
            bound=0.0,
            node_count=3,
        )

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_relaxation)
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(
        oa_module,
        "_solve_fixed_nlp_subproblem_attempt",
        lambda *args, **kwargs: oa_module._NLPAttempt(
            x=np.array([0.0], dtype=float),
            objective=0.0,
            multipliers=None,
            status=SolveStatus.OPTIMAL,
        ),
    )
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    result = oa_module.solve_oa(
        _binary_model("shot_master_repair_success"),
        init_strategy="rNLP",
        max_iterations=1,
        milp_solver="gurobi",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            master_repair=True,
            relaxation_phase="off",
        ),
    )

    assert result.status == "optimal"
    assert master_calls[0]["objective_cutoff"] is not None
    assert master_calls[0]["mip_solution_limit"] == 1
    assert master_calls[1]["objective_cutoff"] is None
    assert master_calls[1]["mip_solution_limit"] is None
    assert master_calls[1]["add_slack"] is True

    repair = result.mip_nlp_trace["iterations"][0]["repair_actions"][0]
    assert repair["status"] == "repaired"
    assert repair["reset_objective_cutoff"] is True
    assert repair["reset_mip_solution_limit"] is True
    assert repair["master_status"] == "optimal"
    assert result.mip_nlp_trace["summary"]["master_repair_success_count"] == 1


def test_mip_nlp_shot_master_repair_failure_records_diagnostic(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    def fake_master(*args, **kwargs):
        del args, kwargs
        return MILPResult(status=SolveStatus.INFEASIBLE, node_count=4)

    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_relaxation",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    result = oa_module.solve_oa(
        _binary_model("shot_master_repair_failure"),
        init_strategy="rNLP",
        max_iterations=1,
        milp_solver="gurobi",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            master_repair=True,
            relaxation_phase="off",
        ),
    )

    assert result.status == "infeasible"
    assert result.mip_nlp_trace["termination_reason"] == "master_infeasible_unrepaired"
    repair = result.mip_nlp_trace["iterations"][0]["repair_actions"][0]
    assert repair["status"] == "failed"
    assert repair["reason"] == "master_status=infeasible"
    assert repair["reset_mip_solution_limit"] is True
    assert result.mip_nlp_trace["summary"]["master_repair_failure_count"] == 1


def test_mip_nlp_shot_master_repair_loop_is_detected(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    master_calls = []

    def fake_master(*args, **kwargs):
        del args
        master_calls.append(kwargs)
        if kwargs["add_slack"]:
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([0.0], dtype=float),
                objective=0.0,
                bound=None,
            )
        return MILPResult(status=SolveStatus.INFEASIBLE)

    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_relaxation",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(
        oa_module,
        "_solve_fixed_nlp_subproblem_attempt",
        lambda *args, **kwargs: oa_module._NLPAttempt(
            x=None,
            objective=None,
            multipliers=None,
            status=SolveStatus.INFEASIBLE,
        ),
    )
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
    monkeypatch.setattr(oa_module, "_solve_feasibility_subproblem", lambda *args, **kwargs: None)

    result = oa_module.solve_oa(
        _binary_model("shot_master_repair_loop"),
        init_strategy="rNLP",
        max_iterations=2,
        milp_solver="gurobi",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            master_repair=True,
            relaxation_phase="off",
        ),
    )

    assert len(master_calls) == 4
    assert result.mip_nlp_trace["termination_reason"] == "master_repair_loop"
    repair = result.mip_nlp_trace["iterations"][1]["repair_actions"][0]
    assert repair["status"] == "loop_detected"
    assert repair["reason"] == "repaired_integer_assignment_repeated"
    assert result.mip_nlp_trace["summary"]["master_repair_loop_count"] == 1


def test_mip_nlp_shot_reduction_cut_is_local_only_for_nonconvex_heuristic(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    captured_rows = []
    captured_rhs = []

    def fake_master(*args, **kwargs):
        del kwargs
        captured_rows.extend(np.asarray(row, dtype=float).copy() for row in args[3])
        captured_rhs.extend(float(rhs) for rhs in args[4])
        return MILPResult(status=SolveStatus.INFEASIBLE)

    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_relaxation",
        lambda *args, **kwargs: (np.array([0.0], dtype=float), 0.0),
    )
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    result = oa_module.solve_oa(
        _binary_model("shot_reduction_cut_local_only"),
        init_strategy="rNLP",
        max_iterations=1,
        heuristic_nonconvex=True,
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            reduction_cuts=True,
            relaxation_phase="off",
        ),
    )

    assert len(captured_rows) == 1
    np.testing.assert_allclose(captured_rows[0], np.array([1.0]))
    assert captured_rhs[0] < 0.0

    event = result.mip_nlp_trace["iterations"][0]["reduction_cuts"][0]
    assert event["status"] == "added"
    assert event["global_valid"] is False
    assert result.mip_nlp_trace["summary"]["cut_source_counts"]["reduction"] == 1
    assert result.mip_nlp_trace["summary"]["local_cut_count"] == 1
    assert result.mip_nlp_trace["gap_certified"] is False
    assert result.mip_nlp_trace["bound_validity"] == "unavailable"


def test_mip_nlp_shot_convex_bounding_filters_local_cuts(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    master_calls = []

    def fake_master(*args, **kwargs):
        rows = [np.asarray(row, dtype=float).copy() for row in args[3]]
        master_calls.append({"rows": rows, "kwargs": kwargs})
        if kwargs["add_slack"]:
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([0.0], dtype=float),
                objective=0.0,
                bound=-1.0,
                node_count=2,
            )
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0], dtype=float),
            objective=-2.0,
            bound=-2.0,
            node_count=3,
        )

    def fake_add_oa_cuts(
        evaluator,
        x_star,
        n_vars,
        n_cons,
        constraint_senses,
        oa_A_rows,
        oa_b_rows,
        *args,
        **kwargs,
    ):
        del evaluator, x_star, n_cons, constraint_senses, args
        oa_module._append_master_cut(
            oa_A_rows,
            oa_b_rows,
            np.ones(n_vars),
            10.0,
            kwargs.get("oa_cut_relaxable"),
            cut_provenance=kwargs.get("cut_provenance"),
            source="oa",
            global_valid=True,
        )
        oa_module._append_master_cut(
            oa_A_rows,
            oa_b_rows,
            -np.ones(n_vars),
            -1.0,
            kwargs.get("oa_cut_relaxable"),
            cut_provenance=kwargs.get("cut_provenance"),
            source="ecp",
            global_valid=False,
        )

    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_relaxation",
        lambda *args, **kwargs: (np.array([0.0], dtype=float), 0.0),
    )
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)
    monkeypatch.setattr(
        oa_module,
        "_solve_fixed_nlp_subproblem_attempt",
        lambda *args, **kwargs: oa_module._NLPAttempt(
            x=np.array([0.0], dtype=float),
            objective=0.0,
            multipliers=None,
            status=SolveStatus.OPTIMAL,
        ),
    )

    result = oa_module.solve_oa(
        _binary_model("shot_convex_bounding_filters_local"),
        init_strategy="rNLP",
        max_iterations=1,
        heuristic_nonconvex=True,
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            relaxation_phase="off",
            mip_solution_limit_strategy="none",
        ),
    )

    convex_calls = [call for call in master_calls if not call["kwargs"]["add_slack"]]
    assert len(convex_calls) == 1
    assert len(convex_calls[0]["rows"]) == 1
    np.testing.assert_allclose(convex_calls[0]["rows"][0], np.array([1.0]))

    trace = result.mip_nlp_trace
    convex_trace = trace["iterations"][0]["convex_bounding"]
    assert convex_trace["status"] == "bound_updated"
    assert convex_trace["global_cut_count"] == 1
    assert convex_trace["local_cut_excluded_count"] == 1
    assert trace["summary"]["convex_bounding_solve_count"] == 1
    assert trace["summary"]["convex_bounding_bound_update_count"] == 1
    assert trace["bound_validity"] == "global"
    assert trace["final_lb"] == pytest.approx(-2.0)
    assert trace["heuristic_lb"] == pytest.approx(-1.0)
    assert result.bound == pytest.approx(-2.0)
    assert result.gap_certified is True


def test_mip_nlp_shot_convex_bounding_excludes_integer_no_good_cuts(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    primary_calls = 0
    convex_rows_by_call = []

    def has_no_good_for_y_zero(rows):
        return any(
            np.asarray(row, dtype=float).shape == (2,)
            and np.allclose(np.asarray(row, dtype=float), np.array([0.0, -1.0]))
            for row in rows
        )

    def fake_master(*args, **kwargs):
        nonlocal primary_calls
        rows = [np.asarray(row, dtype=float).copy() for row in args[3]]
        if kwargs["add_slack"]:
            primary_calls += 1
            if has_no_good_for_y_zero(rows):
                return MILPResult(
                    status=SolveStatus.OPTIMAL,
                    x=np.array([1.0, 1.0], dtype=float),
                    objective=1.0,
                    bound=1.0,
                    node_count=2,
                )
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([1.0, 0.0], dtype=float),
                objective=0.0,
                bound=0.0,
                node_count=1,
            )

        convex_rows_by_call.append(rows)
        bound = 1.0 if has_no_good_for_y_zero(rows) else 0.0
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([1.0, 1.0], dtype=float),
            objective=bound,
            bound=bound,
            node_count=3,
        )

    def fake_fixed_nlp_attempt(*args, **kwargs):
        x_master = np.asarray(args[4], dtype=float)
        if x_master[1] < 0.5:
            return oa_module._NLPAttempt(
                x=None,
                objective=None,
                multipliers=None,
                status=SolveStatus.INFEASIBLE,
            )
        return oa_module._NLPAttempt(
            x=x_master.copy(),
            objective=1.0,
            multipliers=None,
            status=SolveStatus.OPTIMAL,
        )

    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_subproblem",
        lambda *args, **kwargs: (np.array([1.0, 0.0], dtype=float), 1.0),
    )
    monkeypatch.setattr(oa_module, "_solve_fixed_nlp_subproblem_attempt", fake_fixed_nlp_attempt)

    result = oa_module.solve_oa(
        _esh_nonconvex_fixture("shot_convex_bounding_excludes_no_good"),
        init_strategy="initial_binary",
        initial_point=np.array([1.0, 0.0], dtype=float),
        max_iterations=2,
        heuristic_nonconvex=True,
        add_no_good_cuts=True,
        feasibility_cuts=False,
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            relaxation_phase="off",
            mip_solution_limit_strategy="none",
        ),
    )

    assert primary_calls == 2
    assert len(convex_rows_by_call) == 2
    assert not any(has_no_good_for_y_zero(rows) for rows in convex_rows_by_call)

    trace = result.mip_nlp_trace
    assert trace["iterations"][1]["convex_bounding"]["integer_cut_excluded_count"] == 1
    assert trace["iterations"][1]["convex_bounding"]["global_cut_count"] == 0
    assert trace["bound_validity"] == "global"
    assert trace["final_lb"] == pytest.approx(0.0)
    assert trace["heuristic_lb"] == pytest.approx(1.0)
    assert result.status == "feasible"
    assert result.bound == pytest.approx(0.0)
    assert result.objective == pytest.approx(1.0)


def test_mip_nlp_shot_nonconvex_objective_keeps_heuristic_bound_uncertified(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    master_calls = []

    def fake_master(*args, **kwargs):
        del args, kwargs
        master_calls.append(True)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0, 0.0], dtype=float),
            objective=-4.0,
            bound=-4.0,
            node_count=2,
        )

    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_subproblem",
        lambda *args, **kwargs: (np.asarray(args[4], dtype=float), 0.0),
    )
    monkeypatch.setattr(
        oa_module,
        "_solve_fixed_nlp_subproblem_attempt",
        lambda *args, **kwargs: oa_module._NLPAttempt(
            x=np.array([0.0, 0.0], dtype=float),
            objective=0.0,
            multipliers=None,
            status=SolveStatus.OPTIMAL,
        ),
    )

    result = oa_module.solve_oa(
        _esh_nonconvex_objective_fixture("shot_convex_bounding_unavailable"),
        init_strategy="initial_binary",
        max_iterations=1,
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            relaxation_phase="off",
            mip_solution_limit_strategy="none",
        ),
    )

    assert len(master_calls) == 1
    assert result.bound is None
    assert result.gap is None
    assert result.gap_certified is False

    trace = result.mip_nlp_trace
    assert trace["bound_validity"] == "heuristic"
    assert trace["final_lb"] is None
    assert trace["heuristic_lb"] == pytest.approx(-4.0)
    assert trace["summary"]["convex_bounding_solve_count"] == 0
    assert trace["iterations"][0]["convex_bounding"]["status"] == "unavailable"
    assert trace["iterations"][0]["convex_bounding"]["reason"] == "objective_not_globally_boundable"


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
def test_mip_nlp_shot_reduction_cut_infeasible_master_repairs_integration():
    import discopt.solvers.oa as oa_module

    m = dm.Model("shot_reduction_cut_repair_integration")
    x = m.binary("x")
    m.subject_to(x >= 1.0)
    m.minimize(1000.0 * x)

    result = oa_module.solve_oa(
        m,
        init_strategy="initial_binary",
        initial_point=np.array([1.0], dtype=float),
        max_iterations=1,
        heuristic_nonconvex=True,
        milp_solver="highs",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            master_repair=True,
            reduction_cuts=True,
            relaxation_phase="off",
            mip_solution_limit_strategy="none",
        ),
    )

    assert result.status == "optimal"
    assert result.bound == pytest.approx(1000.0)
    assert result.gap_certified is True
    iteration = result.mip_nlp_trace["iterations"][0]
    assert iteration["reduction_cuts"][0]["status"] == "added"
    repair = iteration["repair_actions"][0]
    assert repair["status"] == "repaired"
    assert repair["dropped_reduction_cuts"] == 1
    assert iteration["convex_bounding"]["status"] == "bound_updated"
    assert result.mip_nlp_trace["bound_validity"] == "global"
    assert result.mip_nlp_trace["termination_reason"] != "master_infeasible"


def test_mip_nlp_shot_master_controls_trace_unsupported_backend(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    master_calls = []

    def fake_master(*args, **kwargs):
        master_calls.append(kwargs)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0]),
            objective=0.0,
            bound=0.0,
        )

    def fake_nlp(*args, **kwargs):
        x_master = np.asarray(args[4], dtype=float)
        return x_master.copy(), float(x_master[0])

    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_nlp)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    result = oa_module.solve_oa(
        _binary_model("shot_unsupported_master_controls"),
        init_strategy="initial_binary",
        max_iterations=1,
        milp_solver="highs",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            relaxation_phase="off",
            mip_solution_limit_strategy="static",
        ),
    )

    assert result.status == "optimal"
    assert master_calls[0]["mip_start"] is None
    assert master_calls[0]["objective_cutoff"] is None
    assert master_calls[0]["mip_solution_limit"] is None
    controls = result.mip_nlp_trace["iterations"][0]["master_controls"]
    assert controls["backend_supported"] is False
    assert controls["mip_start"]["requested"] is True
    assert "gurobi" in controls["mip_start"]["degraded_reason"]
    assert "mip_solution_limit" in result.mip_nlp_trace["summary"]["unsupported_backend_features"]


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
def test_mip_nlp_shot_periodic_relaxation_phase_runs_before_integer_master():
    import discopt.solvers.oa as oa_module

    result = oa_module.solve_oa(
        _initial_poa_fixture("shot_periodic_relaxation_phase"),
        init_strategy="initial_binary",
        ecp_mode=True,
        max_iterations=1,
        milp_solver="highs",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(relaxation_phase="periodic"),
    )

    trace = result.mip_nlp_trace
    phase = trace["iterations"][0]["relaxation_phase"]
    assert phase["enabled"] is True
    assert phase["attempted"] is True
    assert phase["status"] in {"seeded", "no_new_cuts"}
    assert trace["summary"]["mip_count"] == 2
    assert trace["summary"]["cut_source_counts"]["relaxation_phase"] >= 0


def test_mip_nlp_shot_esh_generates_rootsearch_cut_with_provenance():
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

    decomp = oa_module._decompose_model(_esh_convex_fixture("esh_unit_convex"))
    store = MIPNLPInteriorPointStore(
        decomp.n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
    )
    store.add(
        np.array([0.0, 1.0]),
        source="unit",
        evaluator=decomp.evaluator,
        constraint_senses=decomp.constraint_senses,
        require_feasible=True,
    )
    provenance = oa_module.MIPNLPCutProvenance()
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    relaxable: list[bool] = []

    added, trace = oa_module._add_esh_cuts(
        decomp.evaluator,
        np.array([2.0, 1.0]),
        decomp.n_vars,
        decomp.constraint_senses,
        rows,
        rhs,
        decomp.obj_is_linear,
        decomp.oa_constraint_mask,
        decomp.oa_objective_is_convex,
        store,
        rootsearch_strategy="bisection",
        oa_cut_relaxable=relaxable,
        cut_provenance=provenance,
    )

    assert added == 1
    assert trace["fallback_used"] is False
    assert trace["rootsearch"]["status"] == "converged"
    assert rows[0] == pytest.approx(np.array([2.0, 0.0]), abs=1e-6)
    assert rhs[0] == pytest.approx(2.0, abs=1e-6)
    assert relaxable == [True]
    record = provenance.records[0]
    assert record.source == "esh"
    assert record.global_valid is True
    assert record.local_valid is True
    assert record.supporting_point == pytest.approx((1.0, 1.0), abs=1e-6)


def test_mip_nlp_shot_esh_falls_back_to_ecp_without_interior_point():
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

    decomp = oa_module._decompose_model(_esh_convex_fixture("esh_unit_fallback"))
    store = MIPNLPInteriorPointStore(
        decomp.n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
    )
    provenance = oa_module.MIPNLPCutProvenance()
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    added, trace = oa_module._add_esh_cuts(
        decomp.evaluator,
        np.array([2.0, 1.0]),
        decomp.n_vars,
        decomp.constraint_senses,
        rows,
        rhs,
        decomp.obj_is_linear,
        decomp.oa_constraint_mask,
        decomp.oa_objective_is_convex,
        store,
        rootsearch_strategy="bisection",
        cut_provenance=provenance,
    )

    assert added == 1
    assert trace["fallback_used"] is True
    assert trace["fallback_reason"] == "missing_interior_point"
    assert provenance.records[0].source == "ecp"
    assert rows[0] == pytest.approx(np.array([4.0, 0.0]), abs=1e-6)
    assert rhs[0] == pytest.approx(5.0, abs=1e-6)


def test_mip_nlp_shot_esh_adds_objective_rootsearch_hyperplane():
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

    decomp = oa_module._decompose_model(_esh_objective_fixture("esh_unit_objective"))
    store = MIPNLPInteriorPointStore(
        decomp.n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
    )
    store.add(
        np.array([0.0, 1.0]),
        source="unit",
        evaluator=decomp.evaluator,
        constraint_senses=decomp.constraint_senses,
        require_feasible=True,
    )
    provenance = oa_module.MIPNLPCutProvenance()
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    relaxable: list[bool] = []

    added, trace = oa_module._add_esh_cuts(
        decomp.evaluator,
        np.array([2.0, 1.0]),
        decomp.n_vars,
        decomp.constraint_senses,
        rows,
        rhs,
        decomp.obj_is_linear,
        decomp.oa_constraint_mask,
        decomp.oa_objective_is_convex,
        store,
        rootsearch_strategy="bisection",
        oa_cut_relaxable=relaxable,
        cut_provenance=provenance,
    )

    assert added == 2
    assert trace["candidate_hyperplanes"] == 2
    assert relaxable == [True, False]
    sources = [record.source for record in provenance.records]
    assert sources == ["esh", "objective_rootsearch"]
    assert provenance.records[1].objective_id == "objective"
    assert provenance.records[1].global_valid is True
    assert len(rows[1]) == decomp.n_vars + 1


def test_mip_nlp_shot_esh_candidate_feasible_adds_objective_fallback_cut():
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

    decomp = oa_module._decompose_model(
        _esh_candidate_feasible_objective_fixture("esh_unit_candidate_feasible_objective")
    )
    store = MIPNLPInteriorPointStore(
        decomp.n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
    )
    store.add(
        np.array([0.0, 1.0]),
        source="unit",
        evaluator=decomp.evaluator,
        constraint_senses=decomp.constraint_senses,
        require_feasible=True,
    )
    provenance = oa_module.MIPNLPCutProvenance()
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    relaxable: list[bool] = []

    added, trace = oa_module._add_esh_cuts(
        decomp.evaluator,
        np.array([0.0, 1.0]),
        decomp.n_vars,
        decomp.constraint_senses,
        rows,
        rhs,
        decomp.obj_is_linear,
        decomp.oa_constraint_mask,
        decomp.oa_objective_is_convex,
        store,
        rootsearch_strategy="bisection",
        oa_cut_relaxable=relaxable,
        cut_provenance=provenance,
    )

    assert added == 1
    assert trace["rootsearch"]["status"] == "candidate_feasible"
    assert trace["fallback_used"] is True
    assert trace["fallback_reason"] == "candidate_feasible"
    assert relaxable == [False]
    assert provenance.records[0].source == "objective"
    assert provenance.records[0].objective_id == "objective"
    assert len(rows[0]) == decomp.n_vars + 1


def test_mip_nlp_shot_esh_respects_hyperplane_selection_controls():
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

    decomp = oa_module._decompose_model(_esh_objective_fixture("esh_unit_selection"))
    store = MIPNLPInteriorPointStore(
        decomp.n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
    )
    store.add(
        np.array([0.0, 1.0]),
        source="unit",
        evaluator=decomp.evaluator,
        constraint_senses=decomp.constraint_senses,
        require_feasible=True,
    )
    provenance = oa_module.MIPNLPCutProvenance()

    added, trace = oa_module._add_esh_cuts(
        decomp.evaluator,
        np.array([2.0, 1.0]),
        decomp.n_vars,
        decomp.constraint_senses,
        [],
        [],
        decomp.obj_is_linear,
        decomp.oa_constraint_mask,
        decomp.oa_objective_is_convex,
        store,
        rootsearch_strategy="bisection",
        cut_provenance=provenance,
        hyperplane_max_per_iter=1,
        hyperplane_selection_factor=0.5,
    )

    assert added == 1
    assert trace["candidate_hyperplanes"] == 2
    assert trace["selected_hyperplanes"] == 1
    assert [record.source for record in provenance.records] == ["esh"]


def test_mip_nlp_shot_esh_protects_incumbent_from_local_cut():
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

    decomp = oa_module._decompose_model(_esh_nonconvex_fixture("esh_unit_local"))
    store = MIPNLPInteriorPointStore(
        decomp.n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
    )
    store.add(
        np.array([2.0, 1.0]),
        source="unit",
        evaluator=decomp.evaluator,
        constraint_senses=decomp.constraint_senses,
        require_feasible=True,
    )
    provenance = oa_module.MIPNLPCutProvenance()

    added, trace = oa_module._add_esh_cuts(
        decomp.evaluator,
        np.array([0.0, 1.0]),
        decomp.n_vars,
        decomp.constraint_senses,
        [],
        [],
        decomp.obj_is_linear,
        decomp.oa_constraint_mask,
        decomp.oa_objective_is_convex,
        store,
        rootsearch_strategy="bisection",
        cut_provenance=provenance,
        incumbent=np.array([-2.0, 1.0]),
    )

    assert added == 0
    assert trace["fallback_used"] is False
    assert trace["local_cuts_rejected"] == 1
    assert provenance.records == []


def test_mip_nlp_shot_esh_ecp_fallback_protects_incumbent_from_local_cut():
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

    decomp = oa_module._decompose_model(_esh_nonconvex_fixture("esh_unit_fallback_local"))
    store = MIPNLPInteriorPointStore(
        decomp.n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
    )
    provenance = oa_module.MIPNLPCutProvenance()

    added, trace = oa_module._add_esh_cuts(
        decomp.evaluator,
        np.array([0.5, 1.0]),
        decomp.n_vars,
        decomp.constraint_senses,
        [],
        [],
        decomp.obj_is_linear,
        decomp.oa_constraint_mask,
        decomp.oa_objective_is_convex,
        store,
        rootsearch_strategy="bisection",
        cut_provenance=provenance,
        incumbent=np.array([-2.0, 1.0]),
    )

    assert added == 0
    assert trace["fallback_used"] is True
    assert trace["fallback_reason"] == "missing_interior_point"
    assert trace["local_cuts_rejected"] == 1
    assert provenance.records == []


def test_mip_nlp_shot_esh_objective_rootsearch_protects_incumbent_from_local_cut():
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

    decomp = oa_module._decompose_model(
        _esh_nonconvex_objective_fixture("esh_unit_objective_local")
    )
    store = MIPNLPInteriorPointStore(
        decomp.n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
    )
    store.add(
        np.array([0.0, 1.0]),
        source="unit",
        evaluator=decomp.evaluator,
        constraint_senses=decomp.constraint_senses,
        require_feasible=True,
    )
    provenance = oa_module.MIPNLPCutProvenance()

    added, trace = oa_module._add_esh_cuts(
        decomp.evaluator,
        np.array([2.0, 1.0]),
        decomp.n_vars,
        decomp.constraint_senses,
        [],
        [],
        decomp.obj_is_linear,
        decomp.oa_constraint_mask,
        decomp.oa_objective_is_convex,
        store,
        rootsearch_strategy="bisection",
        cut_provenance=provenance,
        incumbent=np.array([0.0, 1.0]),
        objective_epigraph_available=True,
    )

    assert decomp.oa_objective_is_convex is False
    assert added == 1
    assert trace["fallback_used"] is False
    assert trace["local_cuts_rejected"] == 1
    assert [record.source for record in provenance.records] == ["esh"]


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
def test_mip_nlp_shot_esh_integration_uses_rootsearch_cuts():
    import discopt.solvers.oa as oa_module

    result = oa_module.solve_oa(
        _esh_convex_fixture("esh_integration"),
        init_strategy="initial_binary",
        initial_point=np.array([0.0, 1.0]),
        ecp_mode=True,
        max_iterations=1,
        milp_solver="highs",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            cut_strategy="esh",
            rootsearch_strategy="bisection",
            relaxation_phase="off",
        ),
    )

    trace = result.mip_nlp_trace
    assert trace["summary"]["cut_source_counts"]["esh"] >= 1
    assert trace["summary"]["cut_source_counts"]["ecp"] == 0
    assert trace["iterations"][0]["esh"][0]["fallback_used"] is False
    assert trace["iterations"][0]["esh"][0]["rootsearch"]["status"] == "converged"


def _mindtpy_simple_minlp(name="mindtpy_init_strategy"):
    """Native port of Pyomo MindtPy's simple MINLP baseline fixture."""
    m = dm.Model(name)
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


def _mindtpy_simple3_minlp(name="mindtpy_simple3"):
    """Native port of Pyomo MindtPy's MINLP3_simple fixture."""
    m = dm.Model(name)
    x = m.continuous("x", shape=(2,), lb=-0.9, ub=50.0)
    y = m.binary("y")

    m.subject_to(-x[1] + 5.0 * dm.log(x[0] + 1.0) + 3.0 * y >= 0.0)
    m.subject_to(-x[1] + x[0] ** 2 - y <= 1.0)
    m.subject_to(x[0] + x[1] + 20.0 * y <= 24.0)
    m.subject_to(2.0 * x[1] + 3.0 * x[0] <= 10.0)
    m.minimize(10.0 * x[0] ** 2 - x[1] + 5.0 * (y - 1.0))
    return m


def _mindtpy_simple5_minlp(name="mindtpy_simple5"):
    """Native port of Pyomo MindtPy's MINLP5_simple convex MINLP fixture."""
    m = dm.Model(name)
    x = m.continuous("x", lb=1.0, ub=20.0)
    y = m.integer("y", lb=1, ub=20)

    m.subject_to(6.0 * x + y <= 60.0)
    m.subject_to(1.0 / x + 1.0 / x - (x**0.5) * (y**0.5) <= -1.0)
    m.subject_to(2.0 * x - 5.0 * y <= -1.0)
    m.minimize(0.3 * (x - 8.0) ** 2 + 0.04 * (y - 6.0) ** 4 + 0.1 * dm.exp(2.0 * x) * (y ** (-4.0)))
    return m


def _mindtpy_constraint_qualification_example(name="mindtpy_constraint_qualification"):
    """Native port of Pyomo MindtPy's constraint-qualification fixture."""
    m = dm.Model(name)
    x = m.continuous("x", lb=1.0, ub=10.0)
    y = m.binary("y")

    m.subject_to((x - 3.0) ** 2 <= 50.0 * (1 - y))
    m.subject_to(x * dm.log(x) + 5.0 <= 50.0 * y)
    m.minimize(x)
    return m


def _mindtpy_eight_process_flowsheet(name="mindtpy_eight_process", *, convex=True):
    """Native port of Pyomo MindtPy's eight-process flowsheet fixture."""
    m = dm.Model(name)
    stream_ubs = np.full(24, 10.0, dtype=float)
    for stream, ub in {
        3: 2.0,
        5: 2.0,
        9: 2.0,
        10: 1.0,
        14: 1.0,
        17: 2.0,
        19: 2.0,
        21: 2.0,
        25: 3.0,
    }.items():
        stream_ubs[stream - 2] = ub

    x = m.continuous("x", shape=(24,), lb=0.0, ub=stream_ubs)
    y = m.binary("y", shape=(8,))

    def x_stream(stream):
        """Return the flowsheet stream variable using one-based stream labels."""
        return x[stream - 2]

    def y_unit(unit):
        """Return the unit-selection binary using one-based unit labels."""
        return y[unit - 1]

    m.subject_to(1.5 * x_stream(9) + x_stream(10) == x_stream(8))
    m.subject_to(1.25 * (x_stream(12) + x_stream(14)) == x_stream(13))
    m.subject_to(x_stream(15) == 2 * x_stream(16))

    if convex:
        m.subject_to(dm.exp(x_stream(3)) - 1 <= x_stream(2))
        m.subject_to(dm.exp(x_stream(5) / 1.2) - 1 <= x_stream(4))
        m.subject_to(dm.exp(x_stream(22)) - 1 <= x_stream(21))
        m.subject_to(dm.exp(x_stream(18)) - 1 <= x_stream(10) + x_stream(17))
        m.subject_to(dm.exp(x_stream(20) / 1.5) - 1 <= x_stream(19))
    else:
        m.subject_to(dm.exp(x_stream(3)) - 1 == x_stream(2))
        m.subject_to(dm.exp(x_stream(5) / 1.2) - 1 == x_stream(4))
        m.subject_to(dm.exp(x_stream(22)) - 1 == x_stream(21))
        m.subject_to(dm.exp(x_stream(18)) - 1 == x_stream(10) + x_stream(17))
        m.subject_to(dm.exp(x_stream(20) / 1.5) - 1 == x_stream(19))

    m.subject_to(x_stream(13) == x_stream(19) + x_stream(21))
    m.subject_to(x_stream(17) == x_stream(9) + x_stream(16) + x_stream(25))
    m.subject_to(x_stream(11) == x_stream(12) + x_stream(15))
    m.subject_to(x_stream(3) + x_stream(5) == x_stream(6) + x_stream(11))
    m.subject_to(x_stream(6) == x_stream(7) + x_stream(8))
    m.subject_to(x_stream(23) == x_stream(20) + x_stream(22))
    m.subject_to(x_stream(23) == x_stream(14) + x_stream(24))

    m.subject_to(x_stream(10) <= 0.8 * x_stream(17))
    m.subject_to(x_stream(10) >= 0.4 * x_stream(17))
    m.subject_to(x_stream(12) <= 5 * x_stream(14))
    m.subject_to(x_stream(12) >= 2 * x_stream(14))

    m.subject_to(x_stream(2) <= 10 * y_unit(1))
    m.subject_to(x_stream(4) <= 10 * y_unit(2))
    m.subject_to(x_stream(9) <= 10 * y_unit(3))
    m.subject_to(x_stream(12) + x_stream(14) <= 10 * y_unit(4))
    m.subject_to(x_stream(15) <= 10 * y_unit(5))
    m.subject_to(x_stream(19) <= 10 * y_unit(6))
    m.subject_to(x_stream(21) <= 10 * y_unit(7))
    m.subject_to(x_stream(10) + x_stream(17) <= 10 * y_unit(8))

    m.subject_to(y_unit(1) + y_unit(2) == 1)
    m.subject_to(y_unit(4) + y_unit(5) <= 1)
    m.subject_to(y_unit(6) + y_unit(7) - y_unit(4) == 0)
    m.subject_to(y_unit(3) - y_unit(8) <= 0)

    fixed_cost = np.array([5.0, 8.0, 6.0, 10.0, 6.0, 7.0, 4.0, 5.0])
    variable_cost = {
        2: 1.0,
        3: -10.0,
        4: 1.0,
        5: -15.0,
        9: -40.0,
        10: 15.0,
        14: 15.0,
        17: 80.0,
        18: -65.0,
        19: 25.0,
        20: -60.0,
        21: 35.0,
        22: -80.0,
        25: -35.0,
    }
    m.minimize(
        122.0
        + sum(fixed_cost[unit] * y[unit] for unit in range(8))
        + sum(variable_cost.get(stream, 0.0) * x_stream(stream) for stream in range(2, 26))
    )
    return m


def _has_disjunctions(model):
    return any(isinstance(c, _DisjunctiveConstraint) for c in model._constraints)


def _expr_has_function(expr, func_name):
    if getattr(expr, "func_name", None) == func_name:
        return True
    for child_name in ("left", "right", "operand"):
        child = getattr(expr, child_name, None)
        if child is not None and _expr_has_function(child, func_name):
            return True
    return any(
        _expr_has_function(child, func_name)
        for child in tuple(getattr(expr, "args", ())) + tuple(getattr(expr, "terms", ()))
    )


def test_model_solve_routes_mip_nlp_options(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls["model"] = model
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    with pytest.warns(
        UserWarning,
        match="MIP-NLP solver ignores solve_model options: skip_convex_check",
    ):
        result = _binary_model().solve(
            solver="mip-nlp",
            mip_nlp_method="ecp",
            equality_relaxation=True,
            add_slack=True,
            max_slack=12.0,
            oa_penalty_factor=34.0,
            add_no_good_cuts=False,
            feasibility_norm="L2",
            add_regularization="level_L1",
            level_coef=0.4,
            stalling_limit=3,
            cycling_check=False,
            milp_solver="gurobi",
            solution_pool=True,
            num_solution_iteration=7,
            fp_iteration_limit=4,
            fp_main_norm="L1",
            fp_projcuts=False,
            fp_discrete_only=False,
            fp_projzerotol=1e-7,
            fp_mipgap=0.05,
            skip_convex_check=True,
        )

    assert result.status == "optimal"
    assert calls["method"] == "ecp"
    assert calls["equality_relaxation"] is True
    assert calls["add_slack"] is True
    assert calls["max_slack"] == pytest.approx(12.0)
    assert calls["oa_penalty_factor"] == pytest.approx(34.0)
    assert calls["add_no_good_cuts"] is False
    assert calls["feasibility_norm"] == "L2"
    assert calls["add_regularization"] == "level_L1"
    assert calls["level_coef"] == pytest.approx(0.4)
    assert calls["stalling_limit"] == 3
    assert calls["cycling_check"] is False
    assert calls["milp_solver"] == "gurobi"
    assert calls["solution_pool"] is True
    assert calls["num_solution_iteration"] == 7
    assert calls["fp_iteration_limit"] == 4
    assert calls["fp_main_norm"] == "L1"
    assert calls["fp_projcuts"] is False
    assert calls["fp_discrete_only"] is False
    assert calls["fp_projzerotol"] == pytest.approx(1e-7)
    assert calls["fp_mipgap"] == pytest.approx(0.05)


def test_model_solve_mip_nlp_path_runs_entropy_canonicalization(monkeypatch):
    import discopt._jax.factorable_reform as reform_module
    import discopt.solvers.oa as oa_module

    real_canonicalize_entropy = reform_module.canonicalize_entropy
    calls = {}

    def spy_canonicalize_entropy(model):
        out = real_canonicalize_entropy(model)
        calls["canonicalize_called"] = True
        calls["input_had_entropy"] = _expr_has_function(model._objective.expression, "entropy")
        calls["output_had_entropy"] = _expr_has_function(out._objective.expression, "entropy")
        return out

    def fake_solve_oa(model, **kwargs):
        calls["solve_oa_had_entropy"] = _expr_has_function(model._objective.expression, "entropy")
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(reform_module, "canonicalize_entropy", spy_canonicalize_entropy)
    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    m = dm.Model("mip_nlp_entropy_canonicalization")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.binary("y")
    m.subject_to(x >= y)
    m.minimize(x * dm.log(x) + y)

    result = m.solve(solver="mip-nlp", mip_nlp_method="oa")

    assert result.status == "optimal"
    assert calls == {
        "canonicalize_called": True,
        "input_had_entropy": False,
        "output_had_entropy": True,
        "solve_oa_had_entropy": True,
    }


def test_model_solve_ecp_alias_derives_method(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    result = _binary_model("ecp_alias").solve(solver="mip-nlp", ecp_mode=True)

    assert result.status == "optimal"
    assert calls["method"] == "ecp"
    assert calls["ecp_mode"] is True


def test_mip_nlp_options_precedence(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_oa(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        _binary_model("option_precedence"),
        method="oa",
        mip_nlp_options={
            "equality_relaxation": False,
            "feasibility_cuts": False,
            "ecp_mode": True,
        },
        equality_relaxation=True,
        feasibility_cuts=True,
    )

    assert result.status == "optimal"
    assert calls["equality_relaxation"] is True
    assert calls["feasibility_cuts"] is True
    assert calls["ecp_mode"] is False


def test_mip_nlp_init_strategy_precedence(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_oa(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        _binary_model("init_strategy_precedence"),
        method="oa",
        mip_nlp_options={"init_strategy": "max_binary"},
        init_strategy="initial_binary",
    )

    assert result.status == "optimal"
    assert calls["init_strategy"] == "initial_binary"


def test_mip_nlp_new_oa_options_precedence_and_alias(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_oa(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        _binary_model("new_options_precedence"),
        method="oa",
        mip_nlp_options={
            "add_slack": False,
            "OA_penalty_factor": 11.0,
            "feasibility_norm": "L1",
            "add_regularization": "level_L1",
            "level_coef": 0.4,
            "cycling_check": True,
            "solution_pool": False,
            "num_solution_iteration": 2,
        },
        add_slack=True,
        oa_penalty_factor=17.0,
        feasibility_norm="L_infinity",
        add_regularization="level_L_infinity",
        level_coef=0.6,
        add_no_good_cuts=False,
        stalling_limit=4,
        heuristic_nonconvex=True,
        cycling_check=False,
        solution_pool=True,
        num_solution_iteration=4,
    )

    assert result.status == "optimal"
    assert calls["add_slack"] is True
    assert calls["oa_penalty_factor"] == pytest.approx(17.0)
    assert calls["feasibility_norm"] == "L_infinity"
    assert calls["add_regularization"] == "level_L_infinity"
    assert calls["level_coef"] == pytest.approx(0.6)
    assert calls["add_no_good_cuts"] is False
    assert calls["stalling_limit"] == 4
    assert calls["heuristic_nonconvex"] is True
    assert calls["cycling_check"] is False
    assert calls["solution_pool"] is True
    assert calls["num_solution_iteration"] == 4


def test_mip_nlp_shot_profile_options_validate_and_attach_trace(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_oa(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        _convex_binary_nonlinear_model("shot_profile_options"),
        method="oa",
        mip_nlp_options={
            "mip_nlp_profile": "shot",
            "tree_strategy": "multi-tree",
            "cut_strategy": "esh",
            "objective_epigraph": "on",
            "anti_epigraph": "off",
            "nonlinear_partitioning": "adaptive",
            "quadratic_partitioning": "static",
            "absolute_value_auxiliaries": "on",
            "monomial_extraction": "off",
            "signomial_extraction": "on",
            "integer_bilinear_strategy": "binary-expansion",
            "integer_bilinear_max_bits": 8,
            "quadratic_extraction": "native",
            "direct_quadratic_routing": "auto",
            "rootsearch_strategy": "toms748",
            "fixed_nlp_strategy": "solution-pool",
            "solution_pool_capacity": 5,
            "hyperplane_max_per_iter": 8,
            "hyperplane_selection_factor": 0.75,
            "relaxation_phase": "initial",
            "mip_solution_limit_strategy": "force-optimal",
            "convex_bounding": True,
            "master_repair": "true",
            "reduction_cuts": False,
        },
    )

    cfg = calls["mip_nlp_shot_config"]
    assert calls["mip_nlp_profile"] == "shot"
    assert cfg.tree_strategy == "multi_tree"
    assert cfg.cut_strategy == "esh"
    assert cfg.objective_epigraph == "on"
    assert cfg.anti_epigraph == "off"
    assert cfg.nonlinear_partitioning == "adaptive"
    assert cfg.quadratic_partitioning == "static"
    assert cfg.absolute_value_auxiliaries == "on"
    assert cfg.monomial_extraction == "off"
    assert cfg.signomial_extraction == "on"
    assert cfg.integer_bilinear_strategy == "binary_expansion"
    assert cfg.integer_bilinear_max_bits == 8
    assert cfg.quadratic_extraction == "native"
    assert cfg.direct_quadratic_routing == "auto"
    assert cfg.rootsearch_strategy == "toms748"
    assert cfg.fixed_nlp_strategy == "solution_pool"
    assert cfg.solution_pool_capacity == 5
    assert cfg.hyperplane_max_per_iter == 8
    assert cfg.hyperplane_selection_factor == pytest.approx(0.75)
    assert cfg.relaxation_phase == "initial"
    assert cfg.mip_solution_limit_strategy == "force_optimal"
    assert cfg.convex_bounding is True
    assert cfg.master_repair is True
    assert cfg.reduction_cuts is False

    assert result.mip_nlp_trace["schema_version"] == 1
    assert result.mip_nlp_trace["profile"] == "shot"
    assert result.mip_nlp_trace["shot_options"]["cut_strategy"] == "esh"
    assert result.mip_nlp_trace["shot_options"]["integer_bilinear_strategy"] == ("binary_expansion")


def test_mip_nlp_shot_single_tree_routes_to_callback_solver(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_lp_nlp_bb(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_lp_nlp_bb", fake_solve_lp_nlp_bb)

    result = solve_mip_nlp(
        _binary_model("shot_single_tree_route"),
        method="oa",
        mip_nlp_options={
            "mip_nlp_profile": "shot",
            "tree_strategy": "single_tree",
        },
    )

    assert result.status == "optimal"
    assert calls["milp_solver"] == "gurobi"
    assert calls["mip_nlp_profile"] == "shot"
    assert calls["mip_nlp_shot_config"].tree_strategy == "single_tree"
    assert calls["add_no_good_cuts"] is True
    assert result.mip_nlp_trace["method"] == "lp_nlp_bb"


def test_mip_nlp_shot_single_tree_rejects_non_gurobi_backend():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(RuntimeError, match="tree_strategy='single_tree'.*milp_solver='gurobi'"):
        solve_mip_nlp(
            _binary_model("shot_single_tree_highs"),
            method="oa",
            mip_nlp_options={
                "mip_nlp_profile": "shot",
                "tree_strategy": "single_tree",
                "milp_solver": "highs",
            },
        )


def test_mip_nlp_shot_direct_nlp_routes_to_continuous_solver(monkeypatch):
    import discopt.solver as solver_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    monkeypatch.setattr(
        solver_module,
        "_classify_model_convexity",
        lambda *args, **kwargs: (True, True, []),
    )

    def fake_solve_continuous(
        model,
        time_limit,
        ipopt_options,
        t_start,
        nlp_solver,
        initial_point=None,
    ):
        del model, ipopt_options, t_start, initial_point
        calls["time_limit"] = time_limit
        calls["nlp_solver"] = nlp_solver
        return SolveResult(status="optimal", objective=1.0, bound=1.0, gap=0.0)

    monkeypatch.setattr(solver_module, "_solve_continuous", fake_solve_continuous)

    result = solve_mip_nlp(
        _convex_general_nlp_model("shot_direct_nlp_route"),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot"},
        time_limit=12.0,
        nlp_solver="pounce",
    )

    assert calls == {"time_limit": 12.0, "nlp_solver": "pounce"}
    assert result.convex_fast_path is True
    assert result.mip_nlp_trace["method"] == "direct"
    assert result.mip_nlp_trace["selected_strategy"] == "direct_nlp"
    assert result.mip_nlp_trace["strategy_selection"]["problem_class"] == "nlp"


def test_mip_nlp_shot_direct_milp_routes_to_requested_backend(monkeypatch):
    import discopt.solver as solver_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_milp_highs(model, t_start, time_limit=None, gap_tolerance=1e-4):
        del model, t_start
        calls["backend"] = "highs"
        calls["time_limit"] = time_limit
        calls["gap_tolerance"] = gap_tolerance
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(solver_module, "_solve_milp_highs", fake_solve_milp_highs)

    result = solve_mip_nlp(
        _binary_model("shot_direct_milp_route"),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", "milp_solver": "highs"},
        time_limit=7.0,
        gap_tolerance=5e-5,
    )

    assert calls == {"backend": "highs", "time_limit": 7.0, "gap_tolerance": 5e-5}
    assert result.mip_nlp_trace["selected_strategy"] == "direct_milp"
    assert result.mip_nlp_trace["strategy_selection"]["backend"] == "highs"
    assert result.mip_nlp_trace["summary"]["selected_strategy"] == "direct_milp"


def test_mip_nlp_external_hooks_skip_shot_direct_routing(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def hook(ctx):
        del ctx
        return []

    def fake_solve_oa(model, **kwargs):
        del model
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        _binary_model("shot_hook_direct_fallback"),
        method="oa",
        mip_nlp_options={
            "mip_nlp_profile": "shot",
            "external_primal_candidate_hook": hook,
        },
    )

    assert calls["external_primal_candidate_hook"] is hook
    assert result.mip_nlp_trace["selected_strategy"] == "oa"
    attempt = result.mip_nlp_trace["strategy_selection"]["direct_attempt"]
    assert attempt["fallback_reason"] == "external_hooks_requested"
    assert attempt["external_hooks"] == ["external_primal_candidate_hook"]


def test_mip_nlp_shot_direct_miqp_routes_when_convex(monkeypatch):
    import discopt.solver as solver_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    monkeypatch.setattr(
        solver_module,
        "_classify_model_convexity",
        lambda *args, **kwargs: (True, True, []),
    )

    def fake_solve_qp_highs(model, t_start, time_limit=None):
        del model, t_start
        calls["backend"] = "highs"
        calls["time_limit"] = time_limit
        return SolveResult(status="optimal", objective=1.0, bound=1.0, gap=0.0)

    monkeypatch.setattr(solver_module, "_solve_qp_highs", fake_solve_qp_highs)

    result = solve_mip_nlp(
        _miqp_style_model("shot_direct_miqp_route"),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", "milp_solver": "highs"},
        time_limit=9.0,
    )

    assert calls == {"backend": "highs", "time_limit": 9.0}
    assert result.mip_nlp_trace["selected_strategy"] == "direct_miqp"
    assert result.mip_nlp_trace["strategy_selection"]["problem_class"] == "miqp"


def test_mip_nlp_shot_direct_qcp_routes_to_gurobi_when_requested(monkeypatch):
    import discopt.solver as solver_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    monkeypatch.setattr(
        solver_module,
        "_classify_model_convexity",
        lambda *args, **kwargs: (True, True, []),
    )

    def fake_solve_qcp_gurobi(model, t_start, time_limit=None, gap_tolerance=1e-4):
        del model, t_start
        calls["backend"] = "gurobi"
        calls["time_limit"] = time_limit
        calls["gap_tolerance"] = gap_tolerance
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(solver_module, "_solve_qcp_gurobi", fake_solve_qcp_gurobi)

    result = solve_mip_nlp(
        _quadratic_partition_model("shot_direct_qcp_route"),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", "milp_solver": "gurobi"},
        time_limit=8.0,
        gap_tolerance=1e-5,
    )

    assert calls == {"backend": "gurobi", "time_limit": 8.0, "gap_tolerance": 1e-5}
    assert result.mip_nlp_trace["selected_strategy"] == "direct_qcp"
    assert result.mip_nlp_trace["strategy_selection"]["backend"] == "gurobi"


def test_mip_nlp_shot_direct_miqcqp_routes_to_gurobi_when_requested(monkeypatch):
    import discopt.solver as solver_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    monkeypatch.setattr(
        solver_module,
        "_classify_model_convexity",
        lambda *args, **kwargs: (True, True, []),
    )

    def fake_solve_qcp_gurobi(model, t_start, time_limit=None, gap_tolerance=1e-4):
        del model, t_start
        calls["backend"] = "gurobi"
        calls["time_limit"] = time_limit
        calls["gap_tolerance"] = gap_tolerance
        return SolveResult(status="optimal", objective=1.0, bound=1.0, gap=0.0)

    monkeypatch.setattr(solver_module, "_solve_qcp_gurobi", fake_solve_qcp_gurobi)

    result = solve_mip_nlp(
        _miqcqp_style_model("shot_direct_miqcqp_route"),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", "milp_solver": "gurobi"},
        time_limit=8.0,
        gap_tolerance=1e-5,
    )

    assert calls == {"backend": "gurobi", "time_limit": 8.0, "gap_tolerance": 1e-5}
    assert result.mip_nlp_trace["selected_strategy"] == "direct_miqcqp"
    assert result.mip_nlp_trace["strategy_selection"]["problem_class"] == "miqcqp"


def test_mip_nlp_shot_direct_qcp_falls_back_without_gurobi(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_oa(model, **kwargs):
        del model
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        _quadratic_partition_model("shot_direct_qcp_fallback"),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", "milp_solver": "highs"},
    )

    assert calls["milp_solver"] == "highs"
    assert result.mip_nlp_trace["selected_strategy"] == "oa"
    attempt = result.mip_nlp_trace["strategy_selection"]["direct_attempt"]
    assert attempt["candidate_strategy"] == "direct_qcp"
    assert attempt["fallback_reason"] == "requires_milp_solver_gurobi"


@pytest.mark.parametrize(
    ("convexity_result", "fallback_reason"),
    [
        ((True, False, []), "nonconvex_model"),
        ((False, False, None), "convexity_unknown"),
    ],
)
def test_mip_nlp_shot_direct_miqp_falls_back_when_convexity_not_certified(
    monkeypatch,
    convexity_result,
    fallback_reason,
):
    import discopt.solver as solver_module
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    monkeypatch.setattr(
        solver_module,
        "_classify_model_convexity",
        lambda *args, **kwargs: convexity_result,
    )

    def fail_direct(*args, **kwargs):
        raise AssertionError("direct QP backend should not run without convexity proof")

    def fake_solve_oa(model, **kwargs):
        del model
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(solver_module, "_solve_qp_highs", fail_direct)
    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        _miqp_style_model("shot_direct_miqp_convexity_fallback"),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", "milp_solver": "highs"},
    )

    assert calls["milp_solver"] == "highs"
    assert result.mip_nlp_trace["selected_strategy"] == "oa"
    attempt = result.mip_nlp_trace["strategy_selection"]["direct_attempt"]
    assert attempt["candidate_strategy"] == "direct_miqp"
    assert attempt["fallback_reason"] == fallback_reason


def test_mip_nlp_shot_rejects_unimplemented_safe_direct_routing_mode():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(ValueError, match="direct_quadratic_routing"):
        solve_mip_nlp(
            _miqp_style_model("shot_direct_safe_rejected"),
            method="oa",
            mip_nlp_options={
                "mip_nlp_profile": "shot",
                "direct_quadratic_routing": "safe",
            },
        )


@pytest.mark.parametrize(
    ("model_factory", "options", "trace_key", "trace_value"),
    [
        (_objective_epigraph_model, {"objective_epigraph": "on"}, "objective_epigraph", "on"),
        (_anti_epigraph_model, {"anti_epigraph": "on"}, "anti_epigraph", "on"),
        (
            _bilinear_partition_model,
            {"nonlinear_partitioning": "adaptive"},
            "nonlinear_partitioning",
            "adaptive",
        ),
        (
            _quadratic_partition_model,
            {"quadratic_partitioning": "static"},
            "quadratic_partitioning",
            "static",
        ),
        (
            _absolute_value_model,
            {"absolute_value_auxiliaries": "on"},
            "absolute_value_auxiliaries",
            "on",
        ),
        (_monomial_model, {"monomial_extraction": "off"}, "monomial_extraction", "off"),
        (_signomial_model, {"signomial_extraction": "on"}, "signomial_extraction", "on"),
        (
            _integer_bilinear_model,
            {"integer_bilinear_strategy": "mccormick", "integer_bilinear_max_bits": 6},
            "integer_bilinear_strategy",
            "mccormick",
        ),
        (_miqp_style_model, {"quadratic_extraction": "native"}, "quadratic_extraction", "native"),
        (
            _miqp_style_model,
            {"direct_quadratic_routing": "off"},
            "direct_quadratic_routing",
            "off",
        ),
    ],
)
def test_mip_nlp_shot_reformulation_controls_validate_targeted_models(
    monkeypatch,
    model_factory,
    options,
    trace_key,
    trace_value,
):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    def fake_solve_oa(model, **kwargs):
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        model_factory(),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", **options},
    )

    assert result.mip_nlp_trace["profile"] == "shot"
    assert result.mip_nlp_trace["shot_options"][trace_key] == trace_value
    if "integer_bilinear_max_bits" in options:
        assert result.mip_nlp_trace["shot_options"]["integer_bilinear_max_bits"] == 6


def test_mip_nlp_shot_objective_epigraph_reforms_before_oa_and_stays_convex(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_oa(model, **kwargs):
        calls["model"] = model
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    solve_mip_nlp(
        _objective_epigraph_model(),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", "objective_epigraph": "on"},
    )

    transformed = calls["model"]
    assert transformed._constraints[0].sense == ">="
    convexity = classify_oa_cut_convexity(transformed)
    assert convexity.objective_is_convex
    assert convexity.constraint_mask == [True]


def test_mip_nlp_shot_objective_epigraph_off_leaves_defining_equality(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_oa(model, **kwargs):
        calls["model"] = model
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    solve_mip_nlp(
        _objective_epigraph_model(),
        method="oa",
        mip_nlp_options={"mip_nlp_profile": "shot", "objective_epigraph": "off"},
    )

    assert calls["model"]._constraints[0].sense == "=="


def test_mip_nlp_trace_gap_certification_requires_finite_bound():
    result = _binary_model("trace_no_bound_time_limit").solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        time_limit=0.0,
    )

    assert result.bound is None
    assert result.gap_certified is False
    assert result.mip_nlp_trace is not None
    assert result.mip_nlp_trace["final_lb"] is None
    assert result.mip_nlp_trace["final_gap"] is None
    assert result.mip_nlp_trace["gap_certified"] is False
    assert result.mip_nlp_trace["bound_validity"] == "unavailable"


def test_mip_nlp_shot_options_require_shot_profile():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(ValueError, match="require mip_nlp_profile='shot'"):
        solve_mip_nlp(
            _binary_model("shot_option_without_profile"),
            method="oa",
            mip_nlp_options={"tree_strategy": "single_tree"},
        )


def test_model_solve_forwards_shot_profile_options(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    result = _binary_model("model_solve_shot_profile").solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        mip_nlp_profile="shot",
        cut_strategy="esh",
        solution_pool_capacity=3,
        master_repair=True,
    )

    assert result.status == "optimal"
    assert calls["mip_nlp_profile"] == "shot"
    assert calls["cut_strategy"] == "esh"
    assert calls["solution_pool_capacity"] == 3
    assert calls["master_repair"] is True


def test_model_solve_passes_initial_solution_to_mip_nlp(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    model = _binary_model("initial_solution_forwarding")
    y = model._variables[0]
    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    result = model.solve(
        solver="mip-nlp",
        init_strategy="initial_binary",
        initial_solution={y: 1.0},
    )

    assert result.status == "optimal"
    assert calls["init_strategy"] == "initial_binary"
    assert calls["initial_point"] is not None
    assert calls["initial_point"].tolist() == pytest.approx([1.0])


def test_mip_nlp_method_ecp_overrides_nested_ecp_mode(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_oa(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    result = solve_mip_nlp(
        _binary_model("ecp_overrides_nested"),
        method="ecp",
        mip_nlp_options={"ecp_mode": False},
    )

    assert result.status == "optimal"
    assert calls["ecp_mode"] is True


def test_gdp_method_oa_deprecated_alias_routes_to_mip_nlp(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    with pytest.deprecated_call(match="gdp_method='oa' is deprecated"):
        result = _binary_model("oa_alias").solve(
            gdp_method="oa",
            equality_relaxation=True,
            skip_convex_check=True,
        )

    assert result.status == "optimal"
    assert calls["method"] == "oa"
    assert calls["equality_relaxation"] is True


def test_mip_nlp_rejects_native_gdp_solver_method(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    called = False

    def fake_solve_mip_nlp(model, **kwargs):
        nonlocal called
        called = True
        return SolveResult(status="optimal")

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    with pytest.raises(ValueError, match="conflicts with solver='mip-nlp'"):
        _binary_model("native_gdp_method").solve(solver="mip-nlp", gdp_method="loa")

    assert called is False


def test_mip_nlp_and_deprecated_oa_alias_reformulate_gdp(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    captured = []

    def fake_solve_mip_nlp(model, **kwargs):
        captured.append(model)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    _gdp_model("mip_nlp_gdp").solve(solver="mip-nlp")
    with pytest.deprecated_call(match="gdp_method='oa' is deprecated"):
        _gdp_model("oa_alias_gdp").solve(gdp_method="oa")

    assert len(captured) == 2
    assert not _has_disjunctions(captured[0])
    assert not _has_disjunctions(captured[1])


@pytest.mark.parametrize(
    ("method", "issue"),
    [
        ("roa", "#116/#117"),
    ],
)
def test_mip_nlp_reserved_methods_raise(method, issue):
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(NotImplementedError, match=issue):
        solve_mip_nlp(_binary_model(f"{method}_reserved"), method=method)


def test_mip_nlp_method_lp_nlp_bb_requires_gurobi_backend():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(RuntimeError, match="requires milp_solver='gurobi'"):
        solve_mip_nlp(_binary_model("lp_nlp_bb_backend"), method="lp_nlp_bb", milp_solver="highs")


def test_mip_nlp_method_lp_nlp_bb_alias_routes_to_single_tree_solver(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_lp_nlp_bb(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(oa_module, "solve_lp_nlp_bb", fake_solve_lp_nlp_bb)

    result = solve_mip_nlp(
        _binary_model("lp_nlp_bb_route"),
        method="lp/nlp-bb",
        milp_solver="gurobi",
        init_strategy="initial_binary",
        feasibility_norm="L1",
    )

    assert result.status == "optimal"
    assert calls["milp_solver"] == "gurobi"
    assert calls["init_strategy"] == "initial_binary"
    assert calls["feasibility_norm"] == "L1"


def test_lp_nlp_bb_lazy_callback_solves_fixed_integer_nlp(monkeypatch):
    import discopt.solvers.gurobi as gurobi_module
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {"lazy": 0, "subproblem": 0, "oa": 0}

    def fake_relaxation(*_args, **_kwargs):
        return None, None

    def fake_subproblem(
        _evaluator,
        _lb,
        _ub,
        _int_indices,
        x_master,
        _nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        assert initial_point is not None
        calls["subproblem"] += 1
        x = np.asarray(x_master, dtype=np.float64).copy()
        return x, 0.0

    def fake_add_oa_cuts(
        _evaluator,
        _x_star,
        n_vars,
        _n_cons,
        _constraint_senses,
        oa_A_rows,
        oa_b_rows,
        _obj_is_linear,
        constraint_convex_mask,
        _objective_is_convex,
        equality_relaxation=False,
        oa_cut_relaxable=None,
        cut_provenance=None,
    ):
        calls["oa"] += 1
        assert constraint_convex_mask is None or all(
            isinstance(v, bool) for v in constraint_convex_mask
        )
        row = np.zeros(n_vars, dtype=np.float64)
        row[0] = 1.0
        oa_A_rows.append(row)
        oa_b_rows.append(-1.0)
        if oa_cut_relaxable is not None:
            oa_cut_relaxable.append(True)

    def fake_lazy_milp(**kwargs):
        candidate = np.zeros_like(kwargs["c"], dtype=np.float64)
        cuts = kwargs["lazy_callback"](candidate)
        calls["lazy"] += 1
        assert cuts
        row, rhs = cuts[0]
        assert row.shape == candidate.shape
        assert float(row @ candidate) > rhs
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=candidate,
            objective=0.0,
            bound=0.0,
            gap=0.0,
            node_count=1,
        )

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_relaxation)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_subproblem)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)
    monkeypatch.setattr(gurobi_module, "solve_milp_with_lazy_cuts", fake_lazy_milp)

    result = solve_mip_nlp(
        _binary_model("lp_nlp_bb_lazy"),
        method="lp_nlp_bb",
        milp_solver="gurobi",
    )

    assert result.status == "optimal"
    assert result.subnlp_calls == 1
    assert calls["lazy"] == 1
    assert calls["subproblem"] == 1
    assert calls["oa"] >= 2


def test_lp_nlp_bb_shot_callback_trace_records_node_and_incumbent_cuts(monkeypatch):
    import discopt.solvers.gurobi as gurobi_module
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {"lazy": 0, "node": 0, "subproblem": 0}

    def fake_relaxation(*_args, **_kwargs):
        return None, None

    def fake_subproblem(
        _evaluator,
        _lb,
        _ub,
        _int_indices,
        x_master,
        _nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        assert initial_point is not None
        calls["subproblem"] += 1
        return np.asarray(x_master, dtype=np.float64), 0.0

    def fake_add_oa_cuts(
        _evaluator,
        _x_star,
        n_vars,
        _n_cons,
        _constraint_senses,
        oa_A_rows,
        oa_b_rows,
        *_args,
        oa_cut_relaxable=None,
        cut_provenance=None,
        **_kwargs,
    ):
        row = np.zeros(n_vars, dtype=np.float64)
        row[0] = 1.0
        oa_module._append_master_cut(
            oa_A_rows,
            oa_b_rows,
            row,
            -0.5,
            oa_cut_relaxable,
            cut_provenance=cut_provenance,
            source="oa",
            global_valid=True,
            supporting_point=np.zeros(n_vars, dtype=np.float64),
        )

    def fake_add_ecp_cuts(
        _evaluator,
        _x_master,
        n_vars,
        _constraint_senses,
        oa_A_rows,
        oa_b_rows,
        *_args,
        oa_cut_relaxable=None,
        cut_provenance=None,
        **_kwargs,
    ):
        row = np.zeros(n_vars, dtype=np.float64)
        row[0] = 1.0
        oa_module._append_master_cut(
            oa_A_rows,
            oa_b_rows,
            row,
            -0.25,
            oa_cut_relaxable,
            cut_provenance=cut_provenance,
            source="ecp",
            global_valid=True,
            supporting_point=np.zeros(n_vars, dtype=np.float64),
        )
        return 1

    def fake_lazy_milp(**kwargs):
        candidate = np.zeros_like(kwargs["c"], dtype=np.float64)
        node_cuts = kwargs["node_callback"](candidate)
        calls["node"] += 1
        lazy_cuts = kwargs["lazy_callback"](candidate)
        calls["lazy"] += 1
        assert node_cuts
        assert lazy_cuts
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=candidate,
            objective=0.0,
            bound=0.0,
            gap=0.0,
            node_count=2,
            callback_stats={
                "mipsol_calls": 1,
                "mipnode_calls": 1,
                "lazy_cuts": len(lazy_cuts),
                "node_cuts": len(node_cuts),
            },
        )

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_relaxation)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_subproblem)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)
    monkeypatch.setattr(oa_module, "_add_ecp_cuts", fake_add_ecp_cuts)
    monkeypatch.setattr(gurobi_module, "solve_milp_with_lazy_cuts", fake_lazy_milp)

    result = solve_mip_nlp(
        _binary_model("lp_nlp_bb_shot_callback_trace"),
        method="lp_nlp_bb",
        mip_nlp_options={
            "mip_nlp_profile": "shot",
            "tree_strategy": "single_tree",
            "cut_strategy": "oa",
        },
        milp_solver="gurobi",
    )

    assert result.status == "optimal"
    assert calls == {"lazy": 1, "node": 1, "subproblem": 1}
    trace = result.mip_nlp_trace
    assert trace["profile"] == "shot"
    assert trace["summary"]["callback_stats"]["mipnode_calls"] == 1
    assert trace["summary"]["callback_stats"]["mipsol_calls"] == 1
    assert trace["summary"]["cut_source_counts"]["oa"] >= 1
    assert trace["summary"]["cut_source_counts"]["ecp"] >= 1
    contexts = [event["context"] for event in trace["iterations"][0]["callback_events"]]
    assert contexts == ["mipnode", "mipsol"]


def test_lp_nlp_bb_callback_termination_reports_time_limit(monkeypatch):
    import discopt.solvers.gurobi as gurobi_module
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus
    from discopt.solvers.mip_nlp import solve_mip_nlp

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    def fake_lazy_milp(**kwargs):
        assert kwargs["terminate_callback"]({}) is True
        return MILPResult(
            status=SolveStatus.ERROR,
            x=None,
            objective=None,
            bound=None,
            gap=None,
            node_count=0,
            callback_stats={
                "terminated": True,
                "terminate_context": "callback",
                "mipsol_calls": 0,
                "mipnode_calls": 0,
                "lazy_cuts": 0,
                "node_cuts": 0,
            },
        )

    monkeypatch.setattr(gurobi_module, "solve_milp_with_lazy_cuts", fake_lazy_milp)

    result = solve_mip_nlp(
        _binary_model("lp_nlp_bb_callback_timeout"),
        method="lp_nlp_bb",
        milp_solver="gurobi",
        time_limit=0.0,
    )

    assert result.status == "time_limit"
    assert result.mip_nlp_trace["termination_reason"] == "time_limit"
    assert result.mip_nlp_trace["summary"]["callback_stats"]["terminated"] is True


def test_lp_nlp_bb_no_good_cut_skips_mixed_integer_model(monkeypatch):
    import discopt.solvers.gurobi as gurobi_module
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus
    from discopt.solvers.mip_nlp import solve_mip_nlp

    no_good_calls = []

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_subproblem",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        oa_module,
        "_solve_feasibility_subproblem",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        oa_module,
        "_add_no_good_cut",
        lambda *args, **kwargs: no_good_calls.append(args),
    )

    def fake_lazy_milp(**kwargs):
        candidate = np.zeros_like(kwargs["c"], dtype=np.float64)
        kwargs["lazy_callback"](candidate)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=candidate,
            objective=0.0,
            bound=0.0,
            gap=0.0,
            node_count=1,
        )

    monkeypatch.setattr(gurobi_module, "solve_milp_with_lazy_cuts", fake_lazy_milp)

    result = solve_mip_nlp(
        _mixed_discrete_model("lp_nlp_bb_mixed_no_good"),
        method="lp_nlp_bb",
        milp_solver="gurobi",
        add_no_good_cuts=True,
        feasibility_cuts=False,
    )

    assert result.status == "no_feasible_point"
    assert no_good_calls == []


def test_lp_nlp_bb_linear_objective_constant_offsets_certified_bound(monkeypatch):
    import discopt.solvers.gurobi as gurobi_module
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus
    from discopt.solvers.mip_nlp import solve_mip_nlp

    m = dm.Model("lp_nlp_bb_constant_bound")
    y = m.binary("y")
    m.minimize(y + 100.0)

    def fake_relaxation(*_args, **_kwargs):
        return None, None

    def fake_subproblem(
        _evaluator,
        _lb,
        _ub,
        _int_indices,
        x_master,
        _nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        return np.asarray(x_master, dtype=np.float64), 100.0

    def fake_lazy_milp(**kwargs):
        candidate = np.zeros_like(kwargs["c"], dtype=np.float64)
        kwargs["lazy_callback"](candidate)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=candidate,
            objective=0.0,
            bound=0.0,
            gap=0.0,
            node_count=1,
        )

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_relaxation)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_subproblem)
    monkeypatch.setattr(gurobi_module, "solve_milp_with_lazy_cuts", fake_lazy_milp)

    result = solve_mip_nlp(m, method="lp_nlp_bb", milp_solver="gurobi")

    assert result.status == "optimal"
    assert result.objective == pytest.approx(100.0)
    assert result.bound == pytest.approx(100.0)
    assert result.gap == pytest.approx(0.0)
    assert result.gap_certified is True


def test_lp_nlp_bb_gurobi_solves_mindtpy_fixture_if_available():
    """Verify Gurobi LP/NLP-BB solves the MindtPy simple fixture."""
    _require_gurobi()

    result = _mindtpy_simple_minlp("lp_nlp_bb_gurobi_mindtpy").solve(
        solver="mip-nlp",
        mip_nlp_method="lp_nlp_bb",
        milp_solver="gurobi",
        time_limit=30.0,
        gap_tolerance=1e-5,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(3.5, abs=1e-3)
    assert result.bound == pytest.approx(3.5, abs=1e-3)
    assert result.gap == pytest.approx(0.0, abs=1e-6)
    assert result.gap_certified is True


def test_lp_nlp_bb_gurobi_reports_constant_offset_bound_if_available():
    _require_gurobi()

    m = dm.Model("lp_nlp_bb_gurobi_constant_bound")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.binary("y")
    m.subject_to((x - 1.0) ** 2 <= 1.0)
    m.subject_to(x >= y)
    m.minimize(2.0 * x + 3.0 * y + 100.0)

    result = m.solve(
        solver="mip-nlp",
        mip_nlp_method="lp_nlp_bb",
        milp_solver="gurobi",
        time_limit=30.0,
        gap_tolerance=1e-5,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(100.0, abs=1e-6)
    assert result.bound == pytest.approx(100.0, abs=1e-6)
    assert result.gap == pytest.approx(0.0, abs=1e-8)
    assert result.gap_certified is True


def test_mip_nlp_method_gloa_is_reserved_for_gdp_axis():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(ValueError, match="GDP logic-based global outer approximation"):
        solve_mip_nlp(_binary_model("gloa_reserved"), method="gloa")


def test_mip_nlp_method_fp_routes_to_standalone_feasibility_pump(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_fp(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="feasible", objective=1.0, x={"x": np.array(1.0)})

    monkeypatch.setattr(oa_module, "solve_feasibility_pump", fake_solve_fp)

    result = solve_mip_nlp(
        _binary_model("fp_route"),
        method="fp",
        init_strategy="fp",
        feasibility_norm="L1",
        add_no_good_cuts=False,
        fp_iteration_limit=3,
        fp_main_norm="L_infinity",
        fp_projcuts=True,
        fp_projzerotol=1e-8,
        fp_mipgap=0.2,
        fp_discrete_only=False,
    )

    assert result.status == "feasible"
    assert calls["feasibility_norm"] == "L1"
    assert calls["add_no_good_cuts"] is False
    assert calls["fp_iteration_limit"] == 3
    assert calls["fp_main_norm"] == "L_infinity"
    assert calls["fp_projcuts"] is True
    assert calls["fp_projzerotol"] == pytest.approx(1e-8)
    assert calls["fp_mipgap"] == pytest.approx(0.2)
    assert calls["fp_discrete_only"] is False


@pytest.mark.parametrize(
    ("option", "value", "match"),
    [
        ("fp_transfercuts", True, "fp_transfercuts=True"),
        ("fp_norm_constraint", True, "fp_norm_constraint=True"),
        ("fp_norm_constraint_coef", 0.5, "fp_norm_constraint_coef"),
        ("fp_cutoffdecr", 0.1, "fp_cutoffdecr"),
    ],
)
def test_mip_nlp_method_fp_rejects_unsupported_mindtpy_options(option, value, match):
    """Verify unsupported MindtPy FP options fail with explicit messages."""
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(ValueError, match=match):
        solve_mip_nlp(
            _binary_model(f"fp_unsupported_{option}"),
            method="fp",
            **{option: value},
        )


def test_mip_nlp_method_goa_routes_to_global_relaxation(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = {}

    def fake_solve_goa(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(
            status="feasible",
            objective=1.0,
            bound=0.0,
            gap=1.0,
            x={"x": np.array(1.0)},
            gap_certified=False,
        )

    monkeypatch.setattr(oa_module, "solve_goa", fake_solve_goa)

    result = solve_mip_nlp(
        _binary_model("goa_route"),
        method="goa",
        add_no_good_cuts=False,
        n_init_partitions=3,
        rel_gap=0.05,
    )

    assert result.status == "feasible"
    assert calls["add_no_good_cuts"] is False
    assert calls["n_init_partitions"] == 3
    assert calls["rel_gap"] == pytest.approx(0.05)


def test_goa_warns_when_amp_only_options_ignored_on_convex_handoff(monkeypatch):
    import discopt.solvers.oa as oa_module

    calls = {}

    def fake_solve_oa(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(
            status="optimal",
            objective=0.0,
            bound=0.0,
            gap=0.0,
            x={"x": np.array(0.0)},
            gap_certified=True,
        )

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    with pytest.warns(
        UserWarning,
        match="AMP-only GOA option\\(s\\).*n_init_partitions.*presolve_bt",
    ):
        result = oa_module.solve_goa(
            _binary_model("goa_convex_ignores_amp_options"),
            rel_gap=0.05,
            n_init_partitions=3,
            presolve_bt=False,
        )

    assert result.status == "optimal"
    assert calls["gap_tolerance"] == pytest.approx(0.05)


def test_goa_fp_seed_uses_no_good_cuts_without_certifying_seed_bound(monkeypatch):
    import discopt.solvers.amp as amp_module
    import discopt.solvers.oa as oa_module

    calls = {}
    m = dm.Model("goa_fp_seed_nonconvex")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.binary("y")
    m.minimize(x * y - x)

    def fake_run_fp(model, decomp, **kwargs):
        calls["fp"] = kwargs
        return oa_module._FeasibilityPumpResult(
            best_x=np.array([1.0, 1.0]),
            best_obj=1.0,
            best_near_x=None,
            best_near_merit=0.0,
            mip_count=2,
        )

    def fake_solve_amp(model, **kwargs):
        calls["amp"] = kwargs
        return SolveResult(
            status="feasible",
            objective=1.0,
            bound=0.0,
            gap=1.0,
            x={"x": np.array(1.0)},
            wall_time=0.1,
            mip_count=3,
            gap_certified=False,
        )

    monkeypatch.setattr(oa_module, "_run_feasibility_pump", fake_run_fp)
    monkeypatch.setattr(amp_module, "solve_amp", fake_solve_amp)

    result = oa_module.solve_goa(m, time_limit=10, max_iterations=5)

    assert calls["fp"]["add_no_good_cuts"] is True
    assert calls["amp"]["initial_point"].tolist() == pytest.approx([1.0, 1.0])
    assert calls["amp"]["use_start_as_incumbent"] is True
    assert result.mip_count == 5
    assert result.gap_certified is False


def test_mip_nlp_feasibility_pump_continuous_only_is_uncertified():
    result = _continuous_model("fp_continuous_uncertified").solve(
        solver="mip-nlp",
        mip_nlp_method="fp",
    )

    assert result.status == "feasible"
    assert result.objective == pytest.approx(0.0, abs=1e-6)
    assert result.bound is None
    assert result.gap is None
    assert result.gap_certified is False
    assert result.x["x"] == pytest.approx(2.0, abs=1e-3)


def test_mip_nlp_unknown_method_fails_before_oa(monkeypatch):
    import discopt.solvers.oa as oa_module

    called = False

    def fake_solve_oa(model, **kwargs):
        nonlocal called
        called = True
        return SolveResult(status="optimal")

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    with pytest.raises(ValueError, match="Unknown mip_nlp_method"):
        _binary_model("unknown_method").solve(solver="mip-nlp", mip_nlp_method="not-a-method")

    assert called is False


def test_mip_nlp_conflicting_ecp_alias_fails_before_oa(monkeypatch):
    import discopt.solvers.oa as oa_module

    called = False

    def fake_solve_oa(model, **kwargs):
        nonlocal called
        called = True
        return SolveResult(status="optimal")

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    with pytest.raises(ValueError, match="Conflicting MIP-NLP method selectors"):
        _binary_model("conflicting_ecp_alias").solve(
            solver="mip-nlp",
            mip_nlp_method="oa",
            ecp_mode=True,
        )

    assert called is False


def test_mip_nlp_invalid_init_strategy_fails_before_oa(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers.mip_nlp import solve_mip_nlp

    called = False

    def fake_solve_oa(model, **kwargs):
        nonlocal called
        called = True
        return SolveResult(status="optimal")

    monkeypatch.setattr(oa_module, "solve_oa", fake_solve_oa)

    with pytest.raises(ValueError, match="Unknown init_strategy"):
        solve_mip_nlp(_binary_model("bad_init_strategy"), method="oa", init_strategy="bad")

    assert called is False


def test_mip_nlp_rejects_unsupported_oa_options():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(ValueError, match="Unsupported MIP-NLP oa option"):
        solve_mip_nlp(
            _binary_model("unsupported_oa_option"),
            method="oa",
            mip_nlp_options={"not_an_oa_option": True},
        )


def test_oa_solution_pool_requires_gurobi_backend():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(RuntimeError, match="solution_pool=True requires milp_solver='gurobi'"):
        solve_mip_nlp(
            _binary_model("solution_pool_non_gurobi"),
            method="oa",
            solution_pool=True,
            milp_solver="highs",
        )


def test_mip_nlp_options_must_be_dict():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(TypeError, match="mip_nlp_options must be a dict"):
        solve_mip_nlp(
            _binary_model("bad_options_type"),
            method="oa",
            mip_nlp_options=[("ecp_mode", True)],
        )


def test_oa_solution_pool_processes_multiple_master_candidates(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    nlp_master_points = []
    cut_points = []
    master_calls = []

    def fake_add_oa_cuts(*args, **kwargs):
        x_star = np.asarray(args[1], dtype=float).copy()
        n_vars = int(args[2])
        oa_A_rows = args[5]
        oa_b_rows = args[6]
        cut_points.append(x_star)
        oa_A_rows.append(np.zeros(n_vars, dtype=float))
        oa_b_rows.append(0.0)
        if kwargs.get("oa_cut_relaxable") is not None:
            kwargs["oa_cut_relaxable"].append(True)

    def fake_solve_nlp_subproblem(
        _evaluator,
        _lb,
        _ub,
        _int_indices,
        x_master,
        _nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        del initial_point
        x = np.asarray(x_master, dtype=float).copy()
        nlp_master_points.append(x)
        obj = 10.0 - float(x[0])
        if return_attempt:
            return oa_module._NLPAttempt(x=x, objective=obj, multipliers=None)
        return x, obj

    def fake_solve_master_milp(*args, **kwargs):
        master_calls.append(kwargs)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0]),
            objective=0.0,
            bound=-100.0,
            solution_pool=[np.array([0.0]), np.array([1.0])],
            solution_pool_objectives=[0.0, 1.0],
        )

    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_solve_nlp_subproblem)
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_solve_master_milp)

    result = oa_module.solve_oa(
        _binary_model("solution_pool_rounds"),
        init_strategy="initial_binary",
        max_iterations=1,
        gap_tolerance=0.0,
        solution_pool=True,
        num_solution_iteration=2,
        milp_solver="gurobi",
    )

    assert result.status == "feasible"
    assert len(master_calls) == 1
    assert master_calls[0]["solution_pool"] is True
    assert master_calls[0]["num_solution_iteration"] == 2
    assert [point.tolist() for point in nlp_master_points[1:]] == [[0.0], [1.0]]
    assert len(cut_points) == 3
    assert result.mip_nlp_trace is not None
    assert result.mip_nlp_trace["profile"] == "default"
    assert result.mip_nlp_trace["summary"]["mip_count"] == 1
    assert result.mip_nlp_trace["summary"]["nlp_subproblem_count"] == 3
    assert result.mip_nlp_trace["summary"]["solution_pool_candidates"] == 2
    assert result.mip_nlp_trace["iterations"][0]["solution_pool_candidates"] == 2
    assert result.mip_nlp_trace["iterations"][0]["cuts_added"] == 2


def test_mip_nlp_shot_solution_pool_strategy_uses_candidate_pool(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    nlp_master_points = []
    master_calls = []

    def fake_add_oa_cuts(*args, **kwargs):
        n_vars = int(args[2])
        oa_A_rows = args[5]
        oa_b_rows = args[6]
        oa_A_rows.append(np.zeros(n_vars, dtype=float))
        oa_b_rows.append(0.0)
        if kwargs.get("oa_cut_relaxable") is not None:
            kwargs["oa_cut_relaxable"].append(True)

    def fake_solve_nlp_subproblem(
        _evaluator,
        _lb,
        _ub,
        _int_indices,
        x_master,
        _nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        del initial_point
        x = np.asarray(x_master, dtype=float).copy()
        nlp_master_points.append(x)
        obj = 10.0 - float(x[0])
        if return_attempt:
            return oa_module._NLPAttempt(x=x, objective=obj, multipliers=None)
        return x, obj

    def fake_solve_master_milp(*args, **kwargs):
        master_calls.append(kwargs)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0]),
            objective=0.0,
            bound=-100.0,
            solution_pool=[np.array([0.0]), np.array([1.0])],
            solution_pool_objectives=[0.0, 1.0],
        )

    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_solve_nlp_subproblem)
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_solve_master_milp)

    result = oa_module.solve_oa(
        _binary_model("shot_solution_pool_strategy"),
        init_strategy="initial_binary",
        max_iterations=1,
        gap_tolerance=0.0,
        milp_solver="gurobi",
        mip_nlp_profile="shot",
        mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
            fixed_nlp_strategy="solution_pool",
            solution_pool_capacity=2,
            relaxation_phase="off",
            mip_solution_limit_strategy="none",
        ),
    )

    assert master_calls[0]["solution_pool"] is True
    assert master_calls[0]["num_solution_iteration"] == 2
    assert [point.tolist() for point in nlp_master_points[1:]] == [[0.0], [1.0]]
    assert result.mip_nlp_trace["iterations"][0]["solution_pool_candidates"] == 2
    assert result.mip_nlp_trace["summary"]["solution_pool_candidates"] == 2


def test_mip_nlp_shot_solution_pool_capacity_warns_on_unsupported_backend(
    monkeypatch,
    caplog,
):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    master_calls = []

    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_relaxation",
        lambda *args, **kwargs: (np.array([0.0]), 0.0),
    )
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        oa_module,
        "_solve_nlp_subproblem",
        lambda *args, **kwargs: (np.array([0.0]), 0.0),
    )

    def fake_solve_master_milp(*args, **kwargs):
        master_calls.append(kwargs)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0]),
            objective=0.0,
            bound=0.0,
            gap=0.0,
        )

    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_solve_master_milp)

    with caplog.at_level(logging.WARNING, logger="discopt.solvers.oa"):
        result = oa_module.solve_oa(
            _binary_model("shot_solution_pool_unsupported_backend"),
            init_strategy="rNLP",
            max_iterations=1,
            gap_tolerance=0.0,
            milp_solver="highs",
            mip_nlp_profile="shot",
            mip_nlp_shot_config=oa_module.MIPNLPShotConfig(
                solution_pool_capacity=2,
                relaxation_phase="off",
                mip_solution_limit_strategy="none",
            ),
        )

    assert master_calls[0]["solution_pool"] is False
    assert master_calls[0]["num_solution_iteration"] == 2
    assert "SHOT solution-pool request ignored" in caplog.text
    assert "solution_pool" in result.mip_nlp_trace["summary"]["unsupported_backend_features"]


def test_fixed_nlp_candidate_manager_orders_and_deduplicates_assignments():
    from discopt.solvers.mip_nlp_candidates import FixedNLPCandidateManager

    manager = FixedNLPCandidateManager(
        n_vars=3,
        int_indices=[1],
        lb=np.array([0.0, 0.0, 0.0]),
        ub=np.array([10.0, 4.0, 10.0]),
        strategy="always",
    )

    manager.add([0.0, 2.0, 0.0], source="solution_pool", objective=20.0, iteration=0)
    manager.add([0.0, 2.1, 0.0], source="solution_pool", objective=10.0, iteration=0)
    manager.add([0.0, 0.0, 0.0], source="mip_optimum", objective=5.0, iteration=0)
    manager.add([0.0, 1.0, 0.0], source="lp_relaxation", objective=3.0, iteration=0)
    manager.add([0.0, 3.0, 0.0], source="rootsearch", objective=1.0, iteration=0)
    manager.add_external_candidates(
        [{"point": [0.0, 4.0, 0.0], "objective": 0.5, "provider": "unit"}],
        iteration=0,
    )

    ready = manager.take_ready(iteration=0, elapsed=0.0)

    assert [candidate.source for candidate in ready] == [
        "mip_optimum",
        "lp_relaxation",
        "solution_pool",
        "rootsearch",
        "external",
    ]
    assert [candidate.integer_assignment for candidate in ready] == [
        (0.0,),
        (1.0,),
        (2.0,),
        (3.0,),
        (4.0,),
    ]
    assert ready[2].objective == pytest.approx(10.0)


def test_fixed_nlp_candidate_manager_trace_counts_and_external_defaults():
    from discopt.solvers.mip_nlp_candidates import FixedNLPCandidateManager

    kwargs = {
        "n_vars": 2,
        "int_indices": [1],
        "lb": np.array([0.0, 0.0]),
        "ub": np.array([10.0, 5.0]),
        "strategy": "always",
    }

    replaced = FixedNLPCandidateManager(**kwargs)
    replaced.add([0.0, 2.0], source="solution_pool", objective=20.0, iteration=0)
    replaced.add([0.0, 2.1], source="mip_optimum", objective=5.0, iteration=0)

    assert replaced.scheduler_trace()["added_source_counts"] == {"mip_optimum": 1}
    assert replaced.scheduler_trace()["pending"] == 1

    external = FixedNLPCandidateManager(**kwargs)
    external.add_external_candidates(
        [
            {
                "point": [0.0, 1.0],
                "provider": "candidate-provider",
                "nlp_source": "original",
            },
            {"point": [0.0, 2.0]},
        ],
        iteration=0,
        provider="default-provider",
        nlp_source="active",
    )

    ready = external.take_ready(iteration=0, elapsed=0.0)
    assert [(candidate.provider, candidate.nlp_source) for candidate in ready] == [
        ("candidate-provider", "original"),
        ("default-provider", "active"),
    ]


def test_fixed_nlp_candidate_manager_scheduling_modes():
    from discopt.solvers.mip_nlp_candidates import FixedNLPCandidateManager

    kwargs = {
        "n_vars": 1,
        "int_indices": [0],
        "lb": np.array([0.0]),
        "ub": np.array([5.0]),
    }
    by_iteration = FixedNLPCandidateManager(
        **kwargs,
        strategy="iteration",
        iteration_frequency=2,
    )
    by_iteration.add([0.0], source="mip_optimum", iteration=0)
    first = by_iteration.take_ready(iteration=0, elapsed=0.0)
    assert first
    by_iteration.record_call_result(
        first[0],
        iteration=0,
        elapsed=0.0,
        success=False,
    )

    by_iteration.add([1.0], source="mip_optimum", iteration=1)
    assert by_iteration.take_ready(iteration=1, elapsed=0.1) == []
    assert by_iteration.take_ready(iteration=2, elapsed=0.2)

    by_time = FixedNLPCandidateManager(**kwargs, strategy="time", time_frequency=5.0)
    by_time.add([0.0], source="mip_optimum", iteration=0)
    first = by_time.take_ready(iteration=0, elapsed=0.0)
    assert first
    by_time.record_call_result(first[0], iteration=0, elapsed=0.0, success=False)
    by_time.add([1.0], source="mip_optimum", iteration=1)
    assert by_time.take_ready(iteration=1, elapsed=4.9) == []
    assert by_time.take_ready(iteration=1, elapsed=5.0)

    pool_driven = FixedNLPCandidateManager(**kwargs, strategy="solution_pool")
    pool_driven.add([0.0], source="mip_optimum", iteration=0)
    assert pool_driven.take_ready(iteration=0, elapsed=0.0) == []
    assert pool_driven.take_ready(
        iteration=0,
        elapsed=0.0,
        has_solution_pool_candidate=True,
    )


def test_external_primal_candidate_hook_validation_accepts_and_rejects():
    from discopt.solvers.oa import _validate_external_primal_candidates

    assert _validate_external_primal_candidates([], n_vars=2) == []

    valid = _validate_external_primal_candidates(
        [{"point": [1.0, 0.0], "objective": 2.5, "provider": "unit"}],
        n_vars=2,
    )

    assert valid[0]["source"] == "external"
    assert valid[0]["objective"] == pytest.approx(2.5)
    assert valid[0]["provider"] == "unit"
    assert np.asarray(valid[0]["point"]).tolist() == [1.0, 0.0]

    with pytest.raises(ValueError, match="candidate 0 point has length 1; expected 2"):
        _validate_external_primal_candidates([[1.0]], n_vars=2)
    with pytest.raises(ValueError, match="candidate 0 point must contain only finite"):
        _validate_external_primal_candidates([[np.inf, 0.0]], n_vars=2)


def test_external_hyperplane_hook_validation_accepts_and_rejects():
    from discopt.solvers.oa import _validate_external_hyperplanes

    valid = _validate_external_hyperplanes(
        [
            {
                "coefficients": [1.0, -1.0],
                "rhs": 0.5,
                "global_valid": False,
                "supporting_point": [2.0, 1.0],
            }
        ],
        n_vars=2,
    )

    assert np.asarray(valid[0]["coefficients"]).tolist() == [1.0, -1.0]
    assert valid[0]["rhs"] == pytest.approx(0.5)
    assert valid[0]["global_valid"] is False

    with pytest.raises(ValueError, match="must include 'coefficients' or 'coeffs'"):
        _validate_external_hyperplanes([{"rhs": 1.0}], n_vars=2)
    with pytest.raises(ValueError, match="coefficients must be nonzero"):
        _validate_external_hyperplanes([{"coefficients": [0.0, 0.0], "rhs": 1.0}], n_vars=2)


def test_external_dual_bound_hook_validation_accepts_and_rejects():
    from discopt.solvers.oa import _validate_external_dual_bound

    valid = _validate_external_dual_bound(
        {"bound": 1.25, "global_valid": False, "provider": "unit"}
    )

    assert valid["bound"] == pytest.approx(1.25)
    assert valid["global_valid"] is False
    assert valid["provider"] == "unit"

    with pytest.raises(ValueError, match="bound must be finite"):
        _validate_external_dual_bound(np.nan)
    with pytest.raises(ValueError, match="global_valid must be a boolean"):
        _validate_external_dual_bound({"bound": 1.0, "global_valid": "yes"})


def test_external_termination_hook_validation_accepts_and_rejects():
    from discopt.solvers.oa import _validate_external_termination

    assert _validate_external_termination(True) is True
    assert _validate_external_termination(False) is False

    with pytest.raises(ValueError, match="return value must be a boolean"):
        _validate_external_termination("stop")


def test_oa_external_hooks_add_candidate_cut_and_bound(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    m = _binary_model("external_hooks")
    seen_events = []

    def fake_master(*args, **kwargs):
        del args, kwargs
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0], dtype=float),
            objective=0.0,
            bound=-100.0,
        )

    def fake_nlp_subproblem(
        evaluator,
        lb,
        ub,
        int_indices,
        x_master,
        nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        del evaluator, lb, ub, int_indices, nlp_solver, initial_point
        point = np.asarray(x_master, dtype=float).copy()
        if point[0] < 0.5:
            attempt = oa_module._NLPAttempt(
                x=None,
                objective=None,
                multipliers=None,
                status=SolveStatus.INFEASIBLE,
            )
        else:
            attempt = oa_module._NLPAttempt(
                x=point,
                objective=0.25,
                multipliers=None,
                status=SolveStatus.OPTIMAL,
            )
        if return_attempt:
            return attempt
        return attempt.x, attempt.objective

    def primal_hook(ctx):
        seen_events.append(ctx["event"])
        assert ctx["solution_points"][0].tolist() == [0.0]
        return [{"point": [1.0], "objective": 0.25, "provider": "unit"}]

    def hyperplane_hook(ctx):
        seen_events.append(ctx["event"])
        assert ctx["master_point"].tolist() == [0.0]
        return [{"coefficients": [1.0], "rhs": 0.75, "provider": "unit"}]

    def dual_bound_hook(ctx):
        seen_events.append(ctx["event"])
        assert ctx["current_dual_bound"] == pytest.approx(-100.0)
        return {"bound": -10.0, "provider": "unit"}

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_nlp_subproblem)
    monkeypatch.setattr(oa_module, "_solve_feasibility_subproblem", lambda *args, **kwargs: None)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    result = oa_module.solve_oa(
        m,
        init_strategy="rNLP",
        max_iterations=1,
        gap_tolerance=0.0,
        external_primal_candidate_hook=primal_hook,
        external_hyperplane_hook=hyperplane_hook,
        external_dual_bound_hook=dual_bound_hook,
    )

    assert seen_events == [
        "external_dual_bound",
        "external_hyperplane",
        "external_primal_candidate",
    ]
    trace = result.mip_nlp_trace
    summary = trace["summary"]
    assert trace["certified_bound_source"] == "external"
    assert summary["cut_source_counts"]["external"] == 1
    assert summary["fixed_nlp_candidate_source_counts"]["external"] == 1
    assert summary["fixed_nlp_call_source_counts"]["external"] == 1
    assert summary["external_hooks"]["accepted_counts"] == {
        "external_dual_bound": 1,
        "external_hyperplane": 1,
        "external_primal_candidate": 1,
    }
    calls = trace["iterations"][0]["fixed_nlp_calls"]
    assert [call["source"] for call in calls] == ["mip_optimum", "external"]
    assert trace["iterations"][0]["external_hooks"][1]["cuts_added"] == 1


def test_oa_external_primal_hook_orders_maximize_objective_hints(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    m = dm.Model("external_hook_maximize_objective")
    x = m.binary("x", shape=(2,))
    m.maximize(x[0] + 2.0 * x[1])
    fixed_nlp_points = []

    def fake_master(*args, **kwargs):
        del args, kwargs
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0, 0.0], dtype=float),
            objective=0.0,
            bound=-10.0,
        )

    def fake_nlp_subproblem(
        evaluator,
        lb,
        ub,
        int_indices,
        x_master,
        nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        del evaluator, lb, ub, int_indices, nlp_solver, initial_point
        point = np.asarray(x_master, dtype=float).copy()
        fixed_nlp_points.append(point.tolist())
        attempt = oa_module._NLPAttempt(
            x=point,
            objective=float(-(point[0] + 2.0 * point[1])),
            multipliers=None,
            status=SolveStatus.OPTIMAL,
        )
        if return_attempt:
            return attempt
        return attempt.x, attempt.objective

    def primal_hook(ctx):
        assert ctx["is_minimization"] is False
        return [
            {"point": [1.0, 0.0], "objective": 1.0},
            {"point": [0.0, 1.0], "objective": 2.0},
        ]

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_nlp_subproblem)
    monkeypatch.setattr(oa_module, "_solve_feasibility_subproblem", lambda *args, **kwargs: None)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    oa_module.solve_oa(
        m,
        init_strategy="rNLP",
        max_iterations=1,
        gap_tolerance=0.0,
        external_primal_candidate_hook=primal_hook,
    )

    assert fixed_nlp_points == [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]


def test_oa_termination_hook_stops_before_master(monkeypatch):
    import discopt.solvers.oa as oa_module

    m = _binary_model("external_termination")
    contexts = []

    def fail_master(*args, **kwargs):
        del args, kwargs
        raise AssertionError("master solve should not run after user termination")

    def termination_hook(ctx):
        contexts.append(ctx)
        return True

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(oa_module, "_solve_master_milp", fail_master)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    result = oa_module.solve_oa(
        m,
        init_strategy="rNLP",
        max_iterations=3,
        termination_hook=termination_hook,
    )

    assert contexts[0]["event"] == "termination"
    trace = result.mip_nlp_trace
    assert trace["termination_reason"] == "user_termination"
    assert trace["iterations"][0]["master_status"] == "not_run"
    assert trace["iterations"][0]["external_hooks"] == [
        {"hook": "termination", "status": "terminate", "requested": True}
    ]
    assert trace["summary"]["external_hooks"]["accepted_counts"] == {"termination": 1}


def test_oa_fixed_nlp_candidate_trace_and_safe_failed_cuts(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    m = dm.Model("fixed_nlp_candidate_trace")
    x = m.continuous("x", lb=0, ub=5)
    y = m.binary("y", shape=(2,))
    m.minimize(x + y[0] + y[1])

    warm_starts = []
    no_good_points = []

    def fake_relaxation(*args, **kwargs):
        if kwargs.get("return_attempt"):
            return oa_module._NLPAttempt(x=None, objective=None, multipliers=None)
        return None, None

    def fake_master(*args, **kwargs):
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.0, 0.0, 0.0], dtype=float),
            objective=0.0,
            bound=-100.0,
            solution_pool=[
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 1.0, 0.0], dtype=float),
                np.array([2.0, 0.0, 1.0], dtype=float),
                np.array([3.0, 1.0, 0.0], dtype=float),
            ],
            solution_pool_objectives=[0.0, 1.0, 2.0, 3.0],
        )

    def fake_nlp_subproblem(
        evaluator,
        lb,
        ub,
        int_indices,
        x_master,
        nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        del evaluator, lb, ub, int_indices, nlp_solver
        warm_starts.append(np.asarray(initial_point, dtype=float).copy())
        key = tuple(np.asarray(x_master, dtype=float)[1:].astype(int).tolist())
        if key == (0, 0):
            attempt = oa_module._NLPAttempt(
                x=np.asarray(x_master, dtype=float).copy(),
                objective=10.0,
                multipliers=None,
                status=SolveStatus.OPTIMAL,
            )
        elif key == (1, 0):
            attempt = oa_module._NLPAttempt(
                x=None,
                objective=None,
                multipliers=None,
                status=SolveStatus.INFEASIBLE,
            )
        else:
            attempt = oa_module._NLPAttempt(
                x=None,
                objective=None,
                multipliers=None,
                status=SolveStatus.TIME_LIMIT,
            )
        if return_attempt:
            return attempt
        return attempt.x, attempt.objective

    def fake_no_good(x_master, *args, **kwargs):
        del args, kwargs
        no_good_points.append(np.asarray(x_master, dtype=float).copy())
        return True

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_relaxation)
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_master)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_nlp_subproblem)
    monkeypatch.setattr(oa_module, "_solve_feasibility_subproblem", lambda *args, **kwargs: None)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)
    monkeypatch.setattr(oa_module, "_add_no_good_cut", fake_no_good)

    result = oa_module.solve_oa(
        m,
        init_strategy="rNLP",
        max_iterations=1,
        solution_pool=True,
        num_solution_iteration=3,
        milp_solver="gurobi",
        add_no_good_cuts=True,
        gap_tolerance=0.0,
    )

    calls = result.mip_nlp_trace["iterations"][0]["fixed_nlp_calls"]
    assert [call["source"] for call in calls] == [
        "mip_optimum",
        "solution_pool",
        "solution_pool",
    ]
    assert [call["status"] for call in calls] == ["optimal", "infeasible", "time_limit"]
    assert [call["incumbent_update"] for call in calls] == [
        "improved",
        "not_feasible",
        "not_feasible",
    ]
    assert [point.tolist() for point in warm_starts] == [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
    ]
    assert [point.tolist() for point in no_good_points] == [[1.0, 1.0, 0.0]]
    assert result.mip_nlp_trace["summary"]["fixed_nlp_call_count"] == 3
    assert result.mip_nlp_trace["summary"]["solution_pool_candidates"] == 3


def test_oa_initial_binary_seed_rounds_and_clamps_discrete_values():
    from discopt.solvers.oa import _build_initial_strategy_point, _decompose_model

    decomp = _decompose_model(_mixed_discrete_model("initial_binary_seed"))
    seed = _build_initial_strategy_point(
        decomp,
        "initial_binary",
        np.array([1.25, 1.8, -0.2, 2.6], dtype=float),
    )

    assert seed.tolist() == pytest.approx([1.25, 1.0, 0.0, 3.0])


def test_oa_max_binary_seed_sets_binaries_and_general_integer_fallback():
    from discopt.solvers.oa import _build_initial_strategy_point, _decompose_model

    decomp = _decompose_model(_mixed_discrete_model("max_binary_seed"))
    seed = _build_initial_strategy_point(
        decomp,
        "max_binary",
        np.array([1.25, 0.0, 0.0, -1.0], dtype=float),
    )

    assert seed.tolist() == pytest.approx([1.25, 1.0, 1.0, 3.0])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("level_L1", "level_L1"),
        ("level-l2", "level_L2"),
        ("level_Linf", "level_L_infinity"),
        ("level_L_infinity", "level_L_infinity"),
        ("grad-lag", "grad_lag"),
        ("hess_lag", "hess_lag"),
        ("hess-only-lag", "hess_only_lag"),
        ("sqp_lag", "sqp_lag"),
    ],
)
def test_oa_regularization_normalizes_supported_modes(raw, expected):
    from discopt.solvers.oa import _normalize_regularization

    assert _normalize_regularization(raw) == expected


def test_oa_regularization_rejects_unknown_modes():
    from discopt.solvers.oa import _normalize_regularization

    with pytest.raises(ValueError, match="Unknown add_regularization"):
        _normalize_regularization("not_a_regularization")


def test_oa_regularization_rejects_ecp_mode():
    from discopt.solvers.oa import solve_oa

    with pytest.raises(ValueError, match="only supported for OA"):
        solve_oa(
            _binary_model("regularized_ecp"),
            ecp_mode=True,
            add_regularization="level_L1",
            max_iterations=0,
        )


def test_oa_derivative_regularization_rejects_fp_init_on_constrained_models():
    """Verify derivative regularization rejects FP init on MindtPy constraints."""
    from discopt.solvers.oa import solve_oa

    with pytest.raises(ValueError, match="init_strategy='fp'.*does not provide"):
        solve_oa(
            _mindtpy_simple_minlp("regularized_grad_lag_fp_init"),
            add_regularization="grad_lag",
            init_strategy="fp",
            max_iterations=0,
        )


@pytest.mark.parametrize("level_coef", [0.0, 1.0, 1.5])
def test_oa_regularization_level_coef_requires_open_unit_interval(level_coef):
    from discopt.solvers.oa import solve_oa

    with pytest.raises(ValueError, match="level_coef must be a finite number"):
        solve_oa(
            _binary_model(f"bad_level_coef_{level_coef}"),
            add_regularization="level_L1",
            level_coef=level_coef,
            max_iterations=0,
        )


@pytest.mark.parametrize(
    ("mode", "expected_aux"),
    [
        ("level_L1", 4),
        ("level_L_infinity", 1),
    ],
)
def test_oa_regularized_master_builds_linear_distance_objective(monkeypatch, mode, expected_aux):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.solvers import MILPResult, SolveStatus
    from discopt.solvers.oa import _decompose_model, _solve_regularized_master

    decomp = _decompose_model(_mixed_discrete_model(f"regularized_{mode}"))
    captured = {}
    fake_x = np.concatenate([np.array([0.0, 1.0, 0.0, 0.0], dtype=float), np.zeros(expected_aux)])

    def fake_milp(**kwargs):
        captured.update(kwargs)
        return MILPResult(status=SolveStatus.OPTIMAL, x=fake_x)

    monkeypatch.setattr(lp_backend, "get_milp_solver", lambda backend="auto": fake_milp)

    x_regularized = _solve_regularized_master(
        decomp,
        [],
        [],
        add_regularization=mode,
        target=np.array([0.5, 1.0, 0.0, 2.0], dtype=float),
        objective_level=2.0,
        time_limit=10.0,
        gap_tolerance=1e-4,
    )

    assert x_regularized.tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert len(captured["c"]) == 4 + expected_aux
    assert captured["c"][4:].tolist() == pytest.approx([1.0] * expected_aux)
    assert captured["integrality"][:4].tolist() == decomp.integrality.tolist()
    assert captured["A_ub"][0, :4].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert captured["b_ub"][0] == pytest.approx(2.0)


def test_oa_regularized_master_builds_l2_qp_distance_objective(monkeypatch):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.solvers import QPResult, SolveStatus
    from discopt.solvers.oa import _decompose_model, _solve_regularized_master

    decomp = _decompose_model(_mixed_discrete_model("regularized_level_l2"))
    captured = {}

    def fake_qp(**kwargs):
        captured.update(kwargs)
        return QPResult(status=SolveStatus.OPTIMAL, x=np.array([0.0, 1.0, 0.0, 0.0]))

    monkeypatch.setattr(lp_backend, "get_qp_solver", lambda: fake_qp)

    x_regularized = _solve_regularized_master(
        decomp,
        [],
        [],
        add_regularization="level_L2",
        target=np.array([0.5, 1.0, 0.0, 2.0], dtype=float),
        objective_level=2.0,
        time_limit=10.0,
        gap_tolerance=1e-4,
    )

    assert x_regularized.tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert captured["Q"].shape == (4, 4)
    assert np.diag(captured["Q"]).tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert captured["c"].tolist() == pytest.approx([-1.0, -2.0, 0.0, -4.0])
    assert captured["integrality"].tolist() == decomp.integrality.tolist()
    assert captured["A_ub"][0, :4].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert captured["b_ub"][0] == pytest.approx(2.0)


def test_oa_regularized_master_builds_grad_lag_objective(monkeypatch):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.solvers import MILPResult, SolveStatus
    from discopt.solvers.oa import (
        _decompose_model,
        _DerivativeRegularizationData,
        _solve_regularized_master,
    )

    decomp = _decompose_model(_mixed_discrete_model("regularized_grad_lag"))
    captured = {}
    data = _DerivativeRegularizationData(
        target=np.array([0.5, 1.0, 0.0, 2.0], dtype=float),
        gradient=np.array([1.0, -2.0, 3.0, -4.0], dtype=float),
    )

    def fake_milp(**kwargs):
        captured.update(kwargs)
        return MILPResult(status=SolveStatus.OPTIMAL, x=np.array([0.0, 1.0, 0.0, 0.0]))

    monkeypatch.setattr(lp_backend, "get_milp_solver", lambda backend="auto": fake_milp)

    x_regularized = _solve_regularized_master(
        decomp,
        [],
        [],
        add_regularization="grad_lag",
        target=data.target,
        objective_level=2.0,
        time_limit=10.0,
        gap_tolerance=1e-4,
        derivative_data=data,
    )

    assert x_regularized.tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert captured["c"].tolist() == pytest.approx([1.0, -2.0, 3.0, -4.0])
    assert captured["integrality"].tolist() == decomp.integrality.tolist()
    assert captured["A_ub"][0, :4].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert captured["b_ub"][0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("mode", "expected_diag", "expected_c"),
    [
        ("hess_lag", [2.0, 4.0, 6.0, 8.0], [0.0, -6.0, 3.0, -20.0]),
        ("hess_only_lag", [2.0, 4.0, 6.0, 8.0], [-1.0, -4.0, 0.0, -16.0]),
        ("sqp_lag", [2.0, 2.0, 2.0, 2.0], [0.0, -4.0, 3.0, -8.0]),
    ],
)
def test_oa_regularized_master_builds_derivative_qp_objectives(
    monkeypatch, mode, expected_diag, expected_c
):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.solvers import QPResult, SolveStatus
    from discopt.solvers.oa import (
        _decompose_model,
        _DerivativeRegularizationData,
        _solve_regularized_master,
    )

    decomp = _decompose_model(_mixed_discrete_model(f"regularized_{mode}"))
    captured = {}
    data = _DerivativeRegularizationData(
        target=np.array([0.5, 1.0, 0.0, 2.0], dtype=float),
        gradient=np.array([1.0, -2.0, 3.0, -4.0], dtype=float),
        hessian=np.diag([2.0, 4.0, 6.0, 8.0]),
    )

    def fake_qp(**kwargs):
        captured.update(kwargs)
        return QPResult(status=SolveStatus.OPTIMAL, x=np.array([0.0, 1.0, 0.0, 0.0]))

    monkeypatch.setattr(lp_backend, "get_qp_solver", lambda: fake_qp)

    x_regularized = _solve_regularized_master(
        decomp,
        [],
        [],
        add_regularization=mode,
        target=data.target,
        objective_level=2.0,
        time_limit=10.0,
        gap_tolerance=1e-4,
        derivative_data=data,
    )

    assert x_regularized.tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert captured["Q"].shape == (4, 4)
    assert np.diag(captured["Q"]).tolist() == pytest.approx(expected_diag)
    assert captured["c"].tolist() == pytest.approx(expected_c)
    assert captured["integrality"].tolist() == decomp.integrality.tolist()


def test_oa_derivative_regularization_data_assembles_lagrangian_terms():
    from discopt.solvers.oa import (
        _build_derivative_regularization_data,
        _DecomposedProblem,
    )

    class FakeEvaluator:
        def evaluate_gradient(self, x):
            return np.array([1.0, 2.0, 3.0, 4.0])

        def evaluate_jacobian(self, x):
            return np.array(
                [
                    [1.0, 0.0, 2.0, -1.0],
                    [0.5, 3.0, 0.0, 2.0],
                ]
            )

        def evaluate_lagrangian_hessian(self, x, obj_factor, multipliers):
            return np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [0.0, 5.0, 6.0, 7.0],
                    [0.0, 0.0, 9.0, 8.0],
                    [0.0, 0.0, 0.0, 13.0],
                ]
            )

    decomp = _DecomposedProblem(
        evaluator=FakeEvaluator(),
        n_vars=4,
        n_cons=2,
        lb=np.zeros(4),
        ub=np.ones(4),
        int_indices=[1, 3],
        binary_indices=[1],
        general_integer_indices=[3],
        integrality=np.array([0, 1, 0, 1], dtype=np.int32),
        linear_A_rows=[],
        linear_b_rows=[],
        linear_senses=[],
        nonlinear_indices=[0, 1],
        constraint_senses=["<=", "<="],
    )

    data = _build_derivative_regularization_data(
        decomp,
        "hess_lag",
        np.array([0.25, 1.0, 0.5, 2.0], dtype=float),
        np.array([10.0, -2.0], dtype=float),
    )

    assert data.target.tolist() == pytest.approx([0.25, 1.0, 0.5, 2.0])
    assert data.gradient.tolist() == pytest.approx([0.0, -4.0, 0.0, -10.0])
    np.testing.assert_allclose(
        data.hessian,
        np.array(
            [
                [1.0, 1.0, 1.5, 2.0],
                [1.0, 5.0, 3.0, 3.5],
                [1.5, 3.0, 9.0, 4.0],
                [2.0, 3.5, 4.0, 13.0],
            ]
        ),
    )


def test_oa_derivative_regularization_requires_duals_from_initial_incumbent(monkeypatch):
    import discopt.solvers.oa as oa_module

    model = _binary_model("regularized_missing_duals")
    y = model._variables[0]
    model.subject_to(y >= 0)

    def fake_solve_nlp_relaxation(
        evaluator,
        lb,
        ub,
        nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        attempt = oa_module._NLPAttempt(
            x=np.array([0.0], dtype=float),
            objective=0.0,
            multipliers=None,
        )
        if return_attempt:
            return attempt
        return attempt.x, attempt.objective

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_solve_nlp_relaxation)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="requires NLP dual multipliers"):
        oa_module.solve_oa(
            model,
            add_regularization="grad_lag",
            max_iterations=0,
        )


def test_oa_hessian_regularization_requires_hessian_access(monkeypatch):
    from discopt.solvers.oa import _build_derivative_regularization_data, _decompose_model

    decomp = _decompose_model(_mixed_discrete_model("regularized_missing_hessian"))

    def missing_hessian(*args, **kwargs):
        raise AttributeError("no hessian")

    monkeypatch.setattr(decomp.evaluator, "evaluate_lagrangian_hessian", missing_hessian)

    with pytest.raises(RuntimeError, match="requires NLP Hessian access"):
        _build_derivative_regularization_data(
            decomp,
            "hess_only_lag",
            np.array([0.5, 1.0, 0.0, 2.0], dtype=float),
            np.empty(0, dtype=float),
        )


def test_oa_l2_regularization_requires_miqp_backend(monkeypatch):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.solvers.oa import _decompose_model, _solve_regularized_master

    decomp = _decompose_model(_mixed_discrete_model("regularized_l2_no_backend"))

    def missing_qp():
        raise ImportError("no qp backend")

    monkeypatch.setattr(lp_backend, "get_qp_solver", missing_qp)

    with pytest.raises(RuntimeError, match="QP/MIQP-capable backend"):
        _solve_regularized_master(
            decomp,
            [],
            [],
            add_regularization="level_L2",
            target=np.array([0.5, 1.0, 0.0, 2.0], dtype=float),
            objective_level=2.0,
            time_limit=10.0,
            gap_tolerance=1e-4,
        )


def test_oa_qp_regularization_reports_backend_rejection_separately(monkeypatch):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.solvers.oa import (
        _decompose_model,
        _DerivativeRegularizationData,
        _solve_regularized_master,
    )

    decomp = _decompose_model(_mixed_discrete_model("regularized_hess_rejected"))
    data = _DerivativeRegularizationData(
        target=np.array([0.5, 1.0, 0.0, 2.0], dtype=float),
        gradient=np.zeros(4),
        hessian=np.diag([1.0, -1.0, 1.0, 1.0]),
    )

    def rejected_qp(**kwargs):
        raise ValueError("Hessian is not positive semidefinite")

    monkeypatch.setattr(lp_backend, "get_qp_solver", lambda: rejected_qp)

    with pytest.raises(RuntimeError, match="rejected by the QP/MIQP backend") as excinfo:
        _solve_regularized_master(
            decomp,
            [],
            [],
            add_regularization="hess_lag",
            target=data.target,
            objective_level=2.0,
            time_limit=10.0,
            gap_tolerance=1e-4,
            derivative_data=data,
        )

    assert "requires a QP/MIQP-capable backend" not in str(excinfo.value)


def test_oa_l2_regularization_checks_backend_before_iterations(monkeypatch):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.solvers.oa import solve_oa

    def missing_qp():
        raise ImportError("no qp backend")

    monkeypatch.setattr(lp_backend, "get_qp_solver", missing_qp)

    with pytest.raises(RuntimeError, match="QP/MIQP-capable backend"):
        solve_oa(
            _binary_model("regularized_l2_no_backend_solve"),
            add_regularization="level_L2",
            max_iterations=0,
        )


@pytest.mark.parametrize("mode", ["hess_lag", "hess_only_lag", "sqp_lag"])
def test_oa_derivative_qp_regularization_checks_backend_before_iterations(monkeypatch, mode):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.solvers.oa import solve_oa

    def missing_qp():
        raise ImportError("no qp backend")

    monkeypatch.setattr(lp_backend, "get_qp_solver", missing_qp)

    with pytest.raises(RuntimeError, match="QP/MIQP-capable backend"):
        solve_oa(
            _binary_model(f"regularized_{mode}_no_backend_solve"),
            add_regularization=mode,
            max_iterations=0,
        )


def test_oa_regularization_seeds_nlp_without_replacing_master_assignment(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    model = _mixed_discrete_model("regularized_seed_only")
    relaxation_point = np.array([0.0, 1.0, 0.0, 3.0], dtype=float)
    master_point = np.array([0.25, 0.0, 1.0, -1.0], dtype=float)
    regularized_point = np.array([1.25, 1.0, 0.0, 2.0], dtype=float)
    subproblem_calls = []

    def fake_solve_nlp_relaxation(evaluator, lb, ub, nlp_solver, initial_point=None):
        return relaxation_point, 4.0

    def fake_add_oa_cuts(*args, **kwargs):
        return None

    def fake_solve_master_milp(*args, **kwargs):
        return MILPResult(status=SolveStatus.OPTIMAL, x=master_point, bound=1.0)

    def fake_solve_regularized_master(*args, **kwargs):
        return regularized_point

    def fake_solve_nlp_subproblem(
        evaluator,
        lb,
        ub,
        int_indices,
        x_master,
        nlp_solver,
        initial_point=None,
    ):
        subproblem_calls.append(
            (
                np.asarray(x_master, dtype=float).copy(),
                np.asarray(initial_point, dtype=float).copy(),
            )
        )
        return np.asarray(x_master, dtype=float), 4.0

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_solve_nlp_relaxation)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_solve_master_milp)
    monkeypatch.setattr(oa_module, "_solve_regularized_master", fake_solve_regularized_master)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_solve_nlp_subproblem)

    result = oa_module.solve_oa(
        model,
        add_regularization="level_L1",
        max_iterations=1,
    )

    assert result.status == "feasible"
    assert len(subproblem_calls) == 1
    assert subproblem_calls[0][0].tolist() == pytest.approx(master_point.tolist())
    assert subproblem_calls[0][1].tolist() == pytest.approx(regularized_point.tolist())


def test_oa_derivative_regularization_seeds_nlp_from_regularized_point(monkeypatch):
    import discopt.solvers.oa as oa_module
    from discopt.solvers import MILPResult, SolveStatus

    model = _mixed_discrete_model("derivative_regularized_seed_only")
    relaxation_point = np.array([0.0, 1.0, 0.0, 3.0], dtype=float)
    master_point = np.array([0.25, 0.0, 1.0, -1.0], dtype=float)
    regularized_point = np.array([1.25, 1.0, 0.0, 2.0], dtype=float)
    regularized_calls = []
    subproblem_calls = []

    def fake_solve_nlp_relaxation(
        evaluator,
        lb,
        ub,
        nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        attempt = oa_module._NLPAttempt(
            x=relaxation_point,
            objective=4.0,
            multipliers=np.empty(0, dtype=float),
        )
        if return_attempt:
            return attempt
        return attempt.x, attempt.objective

    def fake_add_oa_cuts(*args, **kwargs):
        return None

    def fake_solve_master_milp(*args, **kwargs):
        return MILPResult(status=SolveStatus.OPTIMAL, x=master_point, bound=1.0)

    def fake_solve_regularized_master(*args, **kwargs):
        regularized_calls.append(kwargs)
        return regularized_point

    def fake_solve_nlp_subproblem(
        evaluator,
        lb,
        ub,
        int_indices,
        x_master,
        nlp_solver,
        initial_point=None,
        return_attempt=False,
    ):
        subproblem_calls.append(
            (
                np.asarray(x_master, dtype=float).copy(),
                np.asarray(initial_point, dtype=float).copy(),
            )
        )
        attempt = oa_module._NLPAttempt(
            x=np.asarray(x_master, dtype=float),
            objective=3.0,
            multipliers=np.empty(0, dtype=float),
        )
        if return_attempt:
            return attempt
        return attempt.x, attempt.objective

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_solve_nlp_relaxation)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)
    monkeypatch.setattr(oa_module, "_solve_master_milp", fake_solve_master_milp)
    monkeypatch.setattr(oa_module, "_solve_regularized_master", fake_solve_regularized_master)
    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_solve_nlp_subproblem)

    result = oa_module.solve_oa(
        model,
        add_regularization="grad_lag",
        max_iterations=1,
    )

    assert result.status == "feasible"
    assert len(regularized_calls) == 1
    assert regularized_calls[0]["derivative_data"].gradient.tolist() == pytest.approx(
        [0.0, 1.0, 1.0, 1.0]
    )
    assert len(subproblem_calls) == 1
    assert subproblem_calls[0][0].tolist() == pytest.approx(master_point.tolist())
    assert subproblem_calls[0][1].tolist() == pytest.approx(regularized_point.tolist())


def test_oa_rnlp_initialization_adds_cuts_at_relaxation_point(monkeypatch):
    import discopt.solvers.oa as oa_module

    model = _mixed_discrete_model("rnlp_init_point")
    initial_point = np.array([1.25, 1.0, 0.0, 2.0], dtype=float)
    relaxation_point = np.array([0.25, 0.5, 0.5, 1.5], dtype=float)
    cut_points = []

    def fake_solve_nlp_relaxation(evaluator, lb, ub, nlp_solver, initial_point=None):
        assert np.asarray(initial_point, dtype=float).tolist() == pytest.approx(
            [1.25, 1.0, 0.0, 2.0]
        )
        return relaxation_point, 0.0

    def fake_add_oa_cuts(evaluator, x_star, *args, **kwargs):
        cut_points.append(np.asarray(x_star, dtype=float).copy())

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_solve_nlp_relaxation)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)

    result = oa_module.solve_oa(
        model,
        init_strategy="rNLP",
        initial_point=initial_point,
        max_iterations=0,
    )

    assert result.status == "infeasible"
    assert cut_points[0].tolist() == pytest.approx(relaxation_point.tolist())


def test_oa_fp_initialization_adds_cuts_at_best_pump_point(monkeypatch):
    import discopt.solvers.oa as oa_module

    model = _mixed_discrete_model("fp_init_point")
    pump_point = np.array([0.5, 1.0, 0.0, 2.0], dtype=float)
    cut_points = []
    fp_kwargs = {}

    def fake_run_fp(*args, **kwargs):
        fp_kwargs.update(kwargs)
        return oa_module._FeasibilityPumpResult(
            best_x=pump_point,
            best_obj=3.5,
            best_near_x=pump_point,
            best_near_merit=0.0,
            iterations=1,
            mip_count=1,
        )

    def fake_add_oa_cuts(evaluator, x_star, *args, **kwargs):
        cut_points.append(np.asarray(x_star, dtype=float).copy())

    monkeypatch.setattr(oa_module, "_run_feasibility_pump", fake_run_fp)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)

    result = oa_module.solve_oa(
        model,
        init_strategy="fp",
        max_iterations=0,
        fp_iteration_limit=3,
        fp_main_norm="L1",
        fp_projcuts=False,
        fp_discrete_only=False,
        fp_projzerotol=1e-8,
        fp_mipgap=0.2,
    )

    assert result.status == "feasible"
    assert result.objective == pytest.approx(3.5)
    assert cut_points[0].tolist() == pytest.approx(pump_point.tolist())
    assert fp_kwargs["max_iterations"] == 3
    assert fp_kwargs["fp_main_norm"] == "L1"
    assert fp_kwargs["add_no_good_cuts"] is False
    assert fp_kwargs["fp_discrete_only"] is False
    assert fp_kwargs["fp_projzerotol"] == pytest.approx(1e-8)
    assert fp_kwargs["fp_mipgap"] == pytest.approx(0.2)


def test_oa_no_discrete_relaxation_uses_initial_point(monkeypatch):
    import discopt.solvers.oa as oa_module

    initial_point = np.array([3.0], dtype=float)

    def fake_solve_nlp_relaxation(evaluator, lb, ub, nlp_solver, initial_point=None):
        assert np.asarray(initial_point, dtype=float).tolist() == pytest.approx([3.0])
        return np.asarray(initial_point, dtype=float), 1.0

    monkeypatch.setattr(oa_module, "_solve_nlp_relaxation", fake_solve_nlp_relaxation)

    result = oa_module.solve_oa(
        _continuous_model("no_discrete_initial_point"),
        initial_point=initial_point,
    )

    assert result.status == "optimal"
    assert result.x["x"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("strategy", "start", "expected_fixed"),
    [
        ("initial_binary", [1.25, 1.8, -0.2, 2.6], [1.25, 1.0, 0.0, 3.0]),
        ("max_binary", [1.25, 0.0, 0.0, -1.0], [1.25, 1.0, 1.0, 3.0]),
    ],
)
def test_oa_fixed_integer_initializers_seed_first_nlp(monkeypatch, strategy, start, expected_fixed):
    import discopt.solvers.oa as oa_module

    model = _mixed_discrete_model(f"{strategy}_first_nlp")
    fixed_points = []
    cut_points = []

    def fake_solve_nlp_subproblem(
        evaluator,
        lb,
        ub,
        int_indices,
        x_master,
        nlp_solver,
        initial_point=None,
    ):
        fixed_points.append(np.asarray(x_master, dtype=float).copy())
        assert np.asarray(initial_point, dtype=float).tolist() == pytest.approx(expected_fixed)
        return np.asarray(x_master, dtype=float), 0.0

    def fake_add_oa_cuts(evaluator, x_star, *args, **kwargs):
        cut_points.append(np.asarray(x_star, dtype=float).copy())

    monkeypatch.setattr(oa_module, "_solve_nlp_subproblem", fake_solve_nlp_subproblem)
    monkeypatch.setattr(oa_module, "_add_oa_cuts", fake_add_oa_cuts)

    result = oa_module.solve_oa(
        model,
        init_strategy=strategy,
        initial_point=np.array(start, dtype=float),
        max_iterations=0,
    )

    assert result.status == "feasible"
    assert fixed_points[0].tolist() == pytest.approx(expected_fixed)
    assert cut_points[0].tolist() == pytest.approx(expected_fixed)


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
@pytest.mark.parametrize("method", ["oa", "ecp"])
@pytest.mark.parametrize("strategy", ["rNLP", "initial_binary", "max_binary"])
def test_mip_nlp_init_strategies_solve_mindtpy_baseline_to_optimum(method, strategy):
    """Verify OA/ECP init strategies solve the MindtPy simple baseline."""
    result = _mindtpy_simple_minlp(f"mindtpy_{method}_{strategy}").solve(
        solver="mip-nlp",
        mip_nlp_method=method,
        init_strategy=strategy,
        time_limit=60,
        max_nodes=100,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(3.5, abs=1e-3)
    assert result.bound == pytest.approx(3.5, abs=1e-3)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert np.asarray(result.x["y"]).tolist() == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.smoke
@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
def test_mip_nlp_goa_certifies_native_bilinear_minlp():
    m = dm.Model("goa_bilinear_minlp")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.binary("y")

    m.minimize(x * y - 2.0 * x)

    result = m.solve(
        solver="mip-nlp",
        mip_nlp_method="goa",
        time_limit=30,
        max_nodes=20,
        rel_gap=1e-6,
        presolve_bt=False,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(-4.0, abs=1e-6)
    assert result.bound == pytest.approx(-4.0, abs=1e-6)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert result.gap_certified is True
    assert result.x["x"] == pytest.approx(2.0, abs=1e-6)
    assert result.x["y"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
def test_mip_nlp_goa_certifies_mindtpy_minlp5_convex_fixture():
    """Verify GOA certifies the convex MindtPy MINLP5 fixture."""
    result = _mindtpy_simple5_minlp("goa_mindtpy_minlp5").solve(
        solver="mip-nlp",
        mip_nlp_method="goa",
        time_limit=30,
        rel_gap=1e-4,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(3.6572, abs=5e-4)
    assert result.bound == pytest.approx(result.objective, abs=5e-4)
    assert result.gap == pytest.approx(0.0, abs=1e-4)
    assert result.gap_certified is True
    assert result.x["x"] == pytest.approx(4.99179784, abs=1e-4)
    assert result.x["y"] == pytest.approx(7.0, abs=1e-6)


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
@pytest.mark.parametrize("feasibility_norm", ["L_infinity", "L1", "L2"])
def test_mip_nlp_feasibility_pump_solves_mindtpy_baseline(feasibility_norm):
    """Verify feasibility pump finds the MindtPy simple incumbent."""
    result = _mindtpy_simple_minlp(f"mindtpy_fp_{feasibility_norm}").solve(
        solver="mip-nlp",
        mip_nlp_method="fp",
        feasibility_norm=feasibility_norm,
        time_limit=60,
        max_nodes=20,
    )

    assert result.status == "feasible"
    assert result.objective == pytest.approx(3.5, abs=1e-3)
    assert result.bound is None
    assert result.gap is None
    assert result.gap_certified is False
    assert np.asarray(result.x["y"]).tolist() == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
def test_mip_nlp_oa_fp_init_strategy_solves_mindtpy_baseline_to_optimum():
    """Verify OA seeded by feasibility pump solves the MindtPy simple fixture."""
    result = _mindtpy_simple_minlp("mindtpy_oa_fp_init").solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        init_strategy="fp",
        time_limit=60,
        max_nodes=100,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(3.5, abs=1e-3)
    assert result.bound == pytest.approx(3.5, abs=1e-3)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert np.asarray(result.x["y"]).tolist() == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
@pytest.mark.parametrize("add_regularization", ["level_L1", "level_L2", "level_L_infinity"])
def test_mip_nlp_regularized_oa_level_variants_solve_mindtpy_baseline(add_regularization):
    """Verify level-regularized OA solves the MindtPy simple fixture."""
    result = _mindtpy_simple_minlp(f"mindtpy_roa_{add_regularization}").solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        add_regularization=add_regularization,
        time_limit=60,
        max_nodes=100,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(3.5, abs=1e-3)
    assert result.bound == pytest.approx(3.5, abs=1e-3)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert np.asarray(result.x["y"]).tolist() == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
def test_mip_nlp_regularized_oa_grad_lag_solves_mindtpy_baseline():
    """Verify gradient-Lagrangian regularized OA solves the simple fixture."""
    result = _mindtpy_simple_minlp("mindtpy_roa_grad_lag").solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        add_regularization="grad_lag",
        time_limit=60,
        max_nodes=100,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(3.5, abs=1e-3)
    assert result.bound == pytest.approx(3.5, abs=1e-3)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert np.asarray(result.x["y"]).tolist() == pytest.approx([0.0, 1.0, 0.0])


_MINDTPY_REGULARIZATION_MODES = [
    "level_L1",
    "level_L2",
    "level_L_infinity",
    "grad_lag",
    "hess_lag",
    "hess_only_lag",
    "sqp_lag",
]


def test_mip_nlp_mindtpy_eight_process_convex_flag_controls_oa_guarantee():
    """Verify convex flag selection controls eight-process OA certification."""
    from discopt._jax.convexity import classify_oa_cut_convexity

    convex = _mindtpy_eight_process_flowsheet("mindtpy_eight_process_convex", convex=True)
    equality = _mindtpy_eight_process_flowsheet("mindtpy_eight_process_eq", convex=False)

    convex_oa = classify_oa_cut_convexity(convex)
    equality_oa = classify_oa_cut_convexity(equality)

    assert all(convex_oa.constraint_mask)
    assert sum(equality_oa.constraint_mask) == len(equality_oa.constraint_mask) - 5
    assert [equality_oa.constraint_mask[i] for i in range(3, 8)] == [False] * 5


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
@pytest.mark.parametrize("add_regularization", _MINDTPY_REGULARIZATION_MODES)
def test_mip_nlp_regularized_oa_matches_mindtpy_constraint_qualification(
    add_regularization,
):
    """Verify regularized OA matches the MindtPy constraint-qualification case."""
    result = _mindtpy_constraint_qualification_example(f"mindtpy_cq_{add_regularization}").solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        add_regularization=add_regularization,
        time_limit=60,
        max_nodes=100,
    )

    assert result.status in ("optimal", "feasible")
    assert result.objective == pytest.approx(3.0, abs=1e-3)
    if result.status == "optimal":
        assert result.bound == pytest.approx(3.0, abs=1e-3)
        assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert result.x["x"] == pytest.approx(3.0, abs=1e-3)
    assert result.x["y"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
@pytest.mark.parametrize("add_regularization", _MINDTPY_REGULARIZATION_MODES)
def test_mip_nlp_regularized_oa_matches_mindtpy_minlp3_simple(add_regularization):
    """Verify regularized OA matches the MindtPy MINLP3 simple fixture."""
    result = _mindtpy_simple3_minlp(f"mindtpy_simple3_{add_regularization}").solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        add_regularization=add_regularization,
        time_limit=60,
        max_nodes=100,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(-5.5122, abs=1e-3)
    assert result.bound == pytest.approx(result.objective, abs=1e-3)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert np.asarray(result.x["x"]).tolist() == pytest.approx(
        [0.2071068, 0.9411321],
        abs=1e-3,
    )
    assert result.x["y"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
@pytest.mark.parametrize("add_regularization", _MINDTPY_REGULARIZATION_MODES)
def test_mip_nlp_regularized_oa_matches_mindtpy_eight_process_flowsheet(
    add_regularization,
):
    """Verify regularized OA matches the MindtPy eight-process fixture."""
    result = _mindtpy_eight_process_flowsheet(f"mindtpy_eight_process_{add_regularization}").solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        add_regularization=add_regularization,
        time_limit=120,
        max_nodes=200,
    )

    assert result.status in ("optimal", "feasible")
    assert result.objective == pytest.approx(68.0097, abs=2e-2)
    if result.status == "optimal":
        assert result.bound == pytest.approx(result.objective, abs=2e-2)
        assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert np.asarray(result.x["y"]).tolist() == pytest.approx(
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        abs=1e-5,
    )
