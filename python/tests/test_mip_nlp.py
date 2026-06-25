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
            "cycling_check": True,
        },
        add_slack=True,
        oa_penalty_factor=17.0,
        feasibility_norm="L_infinity",
        add_no_good_cuts=False,
        stalling_limit=4,
        heuristic_nonconvex=True,
        cycling_check=False,
    )

    assert result.status == "optimal"
    assert calls["add_slack"] is True
    assert calls["oa_penalty_factor"] == pytest.approx(17.0)
    assert calls["feasibility_norm"] == "L_infinity"
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
        ("fp", "#115"),
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

    with pytest.raises(ValueError, match="Unsupported MIP-NLP OA/ECP option"):
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
