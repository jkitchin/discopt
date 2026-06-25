import discopt.modeling as dm
import pytest
from discopt.modeling.core import SolveResult, _DisjunctiveConstraint


def _binary_model(name="mip_nlp_route"):
    m = dm.Model(name)
    x = m.binary("x")
    m.minimize(x)
    return m


def _gdp_model(name="mip_nlp_gdp_route"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.either_or([[x <= 3], [x >= 7]], name="mode")
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
            skip_convex_check=True,
        )

    assert result.status == "optimal"
    assert calls["method"] == "ecp"
    assert calls["equality_relaxation"] is True


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


def test_mip_nlp_rejects_unsupported_oa_options():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(ValueError, match="Unsupported MIP-NLP OA/ECP option"):
        solve_mip_nlp(
            _binary_model("unsupported_oa_option"),
            method="oa",
            mip_nlp_options={"add_slack": True},
        )


def test_mip_nlp_options_must_be_dict():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(TypeError, match="mip_nlp_options must be a dict"):
        solve_mip_nlp(
            _binary_model("bad_options_type"),
            method="oa",
            mip_nlp_options=[("ecp_mode", True)],
        )
