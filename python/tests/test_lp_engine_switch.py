"""The LP/MILP engine switch: ``DISCOPT_LP_ENGINE=auto|highs|rust``.

``auto`` (the default) runs HiGHS where ``highspy`` is importable and re-solves on the
in-house Rust engine when the HiGHS route returns ``error``. ``highs`` and ``rust`` pin
one engine with no fallback. The per-seam variables still override the switch.

The fallback cases are the two classes the 2026-11-17 measurement found, where the HiGHS
route gives up and the Rust engine certifies: a coefficient or bound near ``1e15``
(HiGHS refuses the model at ``passModel``), and a HiGHS ``kOptimal`` incumbent that
fails discopt's own row check (the X-32 ``1e-13*y >= 5`` MILP).
"""

import sys

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import (
    _ENGINE_FALLBACK_NOTE,
    _lp_backend,
    _lp_engine,
    _lp_milp_backend,
)

pytest.importorskip("highspy")

_SEAM_VARS = (
    "DISCOPT_LP_ENGINE",
    "DISCOPT_LP_BACKEND",
    "DISCOPT_LP_MILP_BACKEND",
    "DISCOPT_IIS_ENGINE",
)


@pytest.fixture
def clean_env(monkeypatch):
    for v in _SEAM_VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _huge_box_lp():
    """Feasible LP on a ``[1e15, 3e15]`` box (#1335's satisfiable twin)."""
    big = 1.0e15
    m = dm.Model("huge_box_lp")
    x = m.continuous("x", lb=big, ub=3 * big)
    y = m.continuous("y", lb=big, ub=3 * big)
    m.subject_to(x - y <= 10.0)
    m.subject_to(x - y >= 1.0)
    m.minimize(x + y)
    return m


def _tiny_coef_milp():
    """``1e-13*y >= 5``, ``y <= 1e14`` integer: optimum ``y = 5e13`` (X-32)."""
    m = dm.Model("tiny_int")
    y = m.integer("y", lb=0, ub=1e14)
    m.subject_to(1e-13 * y >= 5.0)
    m.minimize(y)
    return m


def test_default_is_auto_and_resolves_to_highs(clean_env):
    assert _lp_engine() == "auto"
    assert _lp_backend() == "highs"
    assert _lp_milp_backend() == "highs"


def test_auto_without_highspy_is_rust(clean_env):
    clean_env.setitem(sys.modules, "highspy", None)
    assert _lp_backend() == "rust"
    assert _lp_milp_backend() == "rust"


def test_rust_switch_pins_every_seam(clean_env):
    from discopt._iis_milp import _resolve_iis_engine

    clean_env.setenv("DISCOPT_LP_ENGINE", "rust")
    assert (_lp_backend(), _lp_milp_backend(), _resolve_iis_engine(None)) == ("rust",) * 3
    clean_env.setenv("DISCOPT_LP_ENGINE", "highs")
    assert (_lp_backend(), _lp_milp_backend(), _resolve_iis_engine(None)) == ("highs",) * 3


def test_per_seam_variable_overrides_the_switch(clean_env):
    clean_env.setenv("DISCOPT_LP_ENGINE", "rust")
    clean_env.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    assert _lp_milp_backend() == "highs"
    assert _lp_backend() == "highs"  # LP falls back to the joint variable
    clean_env.setenv("DISCOPT_LP_BACKEND", "rust")
    assert _lp_backend() == "rust"


def test_a_typo_is_refused(clean_env):
    clean_env.setenv("DISCOPT_LP_ENGINE", "higs")
    with pytest.raises(ValueError, match="DISCOPT_LP_ENGINE"):
        _lp_engine()


def _check_huge_box(r):
    assert r.status == "optimal", (r.status, r.algorithm_route)
    d = float(np.asarray(r.x["x"])) - float(np.asarray(r.x["y"]))
    assert 1.0 - 1e-6 <= d <= 10.0 + 1e-6


def test_lp_auto_falls_back_to_rust_when_highs_refuses_the_model(clean_env):
    r = _huge_box_lp().solve(time_limit=60)
    _check_huge_box(r)
    assert _ENGINE_FALLBACK_NOTE in r.algorithm_route, r.algorithm_route
    assert r.solver_stats["lp/engine_fallback"] == 1


def test_lp_pinned_highs_reports_the_error_without_fallback(clean_env):
    clean_env.setenv("DISCOPT_LP_ENGINE", "highs")
    r = _huge_box_lp().solve(time_limit=60)
    # The premise of the fallback test: HiGHS really does give up here. If a HiGHS
    # release starts accepting the model, this fails and the fallback test above
    # no longer exercises the fallback.
    assert r.status == "error", (r.status, r.algorithm_route)


def test_lp_pinned_rust_certifies_without_a_fallback_note(clean_env):
    clean_env.setenv("DISCOPT_LP_ENGINE", "rust")
    r = _huge_box_lp().solve(time_limit=60)
    _check_huge_box(r)
    assert _ENGINE_FALLBACK_NOTE not in (r.algorithm_route or "")


def test_milp_auto_falls_back_to_rust_when_highs_incumbent_fails_verification(clean_env):
    r = _tiny_coef_milp().solve(time_limit=30)
    assert r.status in ("optimal", "feasible"), (r.status, r.error)
    assert r.objective == pytest.approx(5e13, rel=1e-6)
    assert _ENGINE_FALLBACK_NOTE in r.algorithm_route, r.algorithm_route
    assert r.algorithm_route.startswith("native-milp"), r.algorithm_route


def test_milp_pinned_highs_reports_the_error_without_fallback(clean_env):
    clean_env.setenv("DISCOPT_LP_ENGINE", "highs")
    r = _tiny_coef_milp().solve(time_limit=30)
    assert r.status == "error", (r.status, r.algorithm_route)


def test_explicit_milp_backend_keyword_gets_no_fallback(clean_env):
    r = _tiny_coef_milp().solve(time_limit=30, milp_backend="highs")
    assert r.status == "error", (r.status, r.algorithm_route)


def test_ordinary_models_take_highs_without_fallback(clean_env):
    m = dm.Model("plain")
    x = m.integer("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y <= 7.5)
    m.maximize(2 * x + y)
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.objective == pytest.approx(14.5)
    assert r.algorithm_route.startswith("highs-milp"), r.algorithm_route
    assert _ENGINE_FALLBACK_NOTE not in r.algorithm_route


# --- review round (PR #2233) -------------------------------------------------------


def _forced_highs_error(*_a, **_k):
    from discopt.modeling.core import SolveResult

    return SolveResult(status="error", algorithm_route="highs-lp: forced error (test)")


def test_empty_selector_values_mean_unset(clean_env):
    """An exported-but-empty variable is "not set" for every selector: it neither
    breaks the solve nor counts as a per-seam override that blocks the fallback."""
    from discopt._iis_milp import _resolve_iis_engine
    from discopt.solver import _lp_engine_falls_back

    for v in _SEAM_VARS:
        clean_env.setenv(v, "  ")
    assert _lp_engine() == "auto"
    assert (_lp_backend(), _lp_milp_backend(), _resolve_iis_engine(None)) == ("highs",) * 3
    assert _lp_engine_falls_back("DISCOPT_LP_BACKEND", "DISCOPT_LP_MILP_BACKEND")
    r = _huge_box_lp().solve(time_limit=60)
    _check_huge_box(r)
    assert _ENGINE_FALLBACK_NOTE in r.algorithm_route


def test_lp_fallback_passes_the_caller_gap_tolerances(clean_env):
    """The Rust fallback gets the same gap tolerances the pinned-Rust route gets, and
    names its engine like the main route does."""
    import discopt.solver as S

    seen = []
    real = S._solve_lp

    def spy(model, t_start, time_limit=None, gap_tolerance=None, abs_gap_tolerance=None):
        seen.append((gap_tolerance, abs_gap_tolerance))
        return real(model, t_start, time_limit, gap_tolerance, abs_gap_tolerance)

    clean_env.setattr(S, "_solve_lp", spy)
    r = _huge_box_lp().solve(time_limit=60, gap_tolerance=1e-3, abs_gap_tolerance=1e-5)
    _check_huge_box(r)
    assert seen == [(1e-3, 1e-5)], seen
    assert r.algorithm_route.startswith("rust-lp: discopt LP solve, "), r.algorithm_route
    assert _ENGINE_FALLBACK_NOTE in r.algorithm_route


def test_highs_options_block_the_fallback(clean_env):
    """``highs_options`` are a request for HiGHS: a HiGHS error under them is reported,
    not answered by an engine that never saw the options."""
    r = _huge_box_lp().solve(time_limit=60, highs_options={"output_flag": False})
    assert r.status == "error", (r.status, r.algorithm_route)
    assert _ENGINE_FALLBACK_NOTE not in (r.algorithm_route or "")
    r = _tiny_coef_milp().solve(time_limit=30, highs_options={"output_flag": False})
    assert r.status == "error", (r.status, r.algorithm_route)


def test_relu_native_cut_milp_falls_back_too(clean_env):
    """The #2157 ReLU native-cut MILP branch shares the auto fallback: a HiGHS error
    there is re-solved on the Rust engine with node separation, not returned."""
    import discopt.solver as S
    from discopt.ml import DenseLayer, NetworkDefinition, NNFormulation

    rng = np.random.default_rng(0)
    layers = [
        DenseLayer(rng.normal(0, 1, (2, 4)), rng.normal(0, 0.3, 4), activation="relu"),
        DenseLayer(rng.normal(0, 1, (4, 1)), rng.normal(0, 0.3, 1), activation="linear"),
    ]
    net = NetworkDefinition(layers, input_bounds=(np.full(2, -1.0), np.full(2, 1.0)))

    def build():
        m = dm.Model("relu")
        nn = NNFormulation(m, net, strategy="relu_bigm", prefix="p", ideal_cuts=True)
        nn.formulate()
        m.minimize(nn.outputs[0])
        return m, nn

    m, nn = build()
    ref = m.solve(time_limit=60, cut_callback=nn.cut_callback)
    assert ref.status == "optimal" and ref.algorithm_route.startswith("native-relu-cuts")
    assert "HiGHS" in ref.algorithm_route

    clean_env.setattr(S, "_solve_milp_highs", _forced_highs_error)
    m, nn = build()
    r = m.solve(time_limit=60, cut_callback=nn.cut_callback)
    assert r.status == "optimal", (r.status, r.algorithm_route)
    assert r.objective == pytest.approx(ref.objective, abs=1e-6)
    assert "Rust MILP engine" in r.algorithm_route, r.algorithm_route
    assert _ENGINE_FALLBACK_NOTE in r.algorithm_route
    assert r.solver_stats["lp/engine_fallback"] == 1

    clean_env.setenv("DISCOPT_LP_ENGINE", "highs")  # pinned: no fallback
    m, nn = build()
    assert m.solve(time_limit=60, cut_callback=nn.cut_callback).status == "error"


def test_mip_nlp_trace_names_the_engine_that_answered(clean_env):
    import discopt.solver as S
    from discopt.modeling.core import SolveResult
    from discopt.solvers import mip_nlp

    fell_back = SolveResult(status="optimal", solver_stats={"lp/engine_fallback": 1})
    assert mip_nlp._answering_lp_engine(fell_back, S) == "rust"
    assert mip_nlp._answering_lp_engine(SolveResult(status="optimal"), S) == "highs"
