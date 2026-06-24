import discopt.modeling as dm
import pytest
from discopt.modeling.core import SolveResult


def _binary_model(name="mip_nlp_route"):
    m = dm.Model(name)
    x = m.binary("x")
    m.minimize(x)
    return m


def test_model_solve_routes_mip_nlp_options(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls["model"] = model
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    result = _binary_model().solve(
        solver="mip-nlp",
        mip_nlp_method="ecp",
        equality_relaxation=True,
        skip_convex_check=True,
    )

    assert result.status == "optimal"
    assert calls["method"] == "ecp"
    assert calls["equality_relaxation"] is True


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


def test_mip_nlp_reserved_methods_raise():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(NotImplementedError, match="mip_nlp_method='fp'"):
        solve_mip_nlp(_binary_model("fp_reserved"), method="fp")


def test_mip_nlp_rejects_unsupported_oa_options():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(ValueError, match="Unsupported MIP-NLP OA/ECP option"):
        solve_mip_nlp(
            _binary_model("unsupported_oa_option"),
            method="oa",
            mip_nlp_options={"add_slack": True},
        )

