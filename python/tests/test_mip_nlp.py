import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult, _DisjunctiveConstraint

try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False


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


def _mindtpy_simple_minlp(name="mindtpy_init_strategy"):
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


def _has_disjunctions(model):
    return any(isinstance(c, _DisjunctiveConstraint) for c in model._constraints)


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
        ("goa", "#118"),
        ("lp_nlp_bb", "#119"),
        ("lp/nlp-bb", "#119"),
    ],
)
def test_mip_nlp_reserved_methods_raise(method, issue):
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(NotImplementedError, match=issue):
        solve_mip_nlp(_binary_model(f"{method}_reserved"), method=method)


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
    )

    assert result.status == "feasible"
    assert calls["feasibility_norm"] == "L1"
    assert calls["add_no_good_cuts"] is False


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


def test_mip_nlp_options_must_be_dict():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(TypeError, match="mip_nlp_options must be a dict"):
        solve_mip_nlp(
            _binary_model("bad_options_type"),
            method="oa",
            mip_nlp_options=[("ecp_mode", True)],
        )


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
    ],
)
def test_oa_regularization_normalizes_supported_level_modes(raw, expected):
    from discopt.solvers.oa import _normalize_regularization

    assert _normalize_regularization(raw) == expected


def test_oa_regularization_rejects_reserved_future_modes():
    from discopt.solvers.oa import _normalize_regularization

    with pytest.raises(ValueError, match="Unknown add_regularization"):
        _normalize_regularization("hess_lag")


def test_oa_regularization_rejects_ecp_mode():
    from discopt.solvers.oa import solve_oa

    with pytest.raises(ValueError, match="only supported for OA"):
        solve_oa(
            _binary_model("regularized_ecp"),
            ecp_mode=True,
            add_regularization="level_L1",
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

    monkeypatch.setattr(lp_backend, "get_milp_solver", lambda: fake_milp)

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

    def fake_run_fp(*args, **kwargs):
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
    )

    assert result.status == "feasible"
    assert result.objective == pytest.approx(3.5)
    assert cut_points[0].tolist() == pytest.approx(pump_point.tolist())


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


@pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")
@pytest.mark.parametrize("feasibility_norm", ["L_infinity", "L1"])
def test_mip_nlp_feasibility_pump_solves_mindtpy_baseline(feasibility_norm):
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
