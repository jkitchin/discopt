"""``solver="scip"``: the whole model handed to SCIP through PySCIPOpt (issue #1784).

Three things are pinned here:

1. **Fidelity.** The in-memory translation is the same model SCIP reads from the
   ``.nl`` file: solving a corpus instance both ways with the same SCIP binary
   reaches the same optimum. That is a differential test with SCIP held fixed, so
   a disagreement can only be the translation.
2. **Refusals.** Every node without an exact SCIP spelling is refused by name,
   never approximated -- including the domain-restricted rewrites (``x**y`` and
   ``xlogx``), which are only exact where their argument is proven positive.
3. **Certificates.** SCIP's point is judged by discopt's ``verify_point`` and its
   objective recomputed; a contradiction between the two raises, and a point
   discopt rejects is reported ``feasible``/uncertified.
"""

from __future__ import annotations

import pathlib
import sys

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt.modeling.core import from_nl

_NL_DIR = pathlib.Path(__file__).parent / "data" / "minlplib_nl"


def _require_scip():
    return pytest.importorskip("pyscipopt")


def _rel_close(a: float, b: float, rtol: float = 1e-4, atol: float = 1e-6) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


# ── import contract ──────────────────────────────────────────────────────────


def test_missing_pyscipopt_names_the_extra(monkeypatch):
    from discopt.solvers import scip as scip_backend

    monkeypatch.setitem(sys.modules, "pyscipopt", None)
    with pytest.raises(ImportError, match=r"discopt\[scip\]"):
        scip_backend._load_scip()


def test_unknown_solver_message_lists_scip():
    m = Model("u")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    with pytest.raises(ValueError, match="'scip'"):
        m.solve(solver="no-such-solver")


# ── end-to-end ───────────────────────────────────────────────────────────────


def _vector_miqp() -> Model:
    m = Model("vec")
    A = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    x = m.continuous("x", shape=(3,), lb=0, ub=5)
    y = m.binary("y", shape=(3,))
    m.subject_to(A @ x <= np.array([6.0, 7.0]))
    m.subject_to(x <= 5 * y)
    m.subject_to(dm.sum(y) <= 2)
    m.minimize(dm.sum((x - 2) ** 2) - dm.sum(x * x[::-1]) + dm.sum(y))
    return m


@pytest.mark.smoke
def test_vector_model_matches_discopt_certified_optimum():
    _require_scip()
    ref = _vector_miqp().solve(time_limit=30)
    assert ref.status == "optimal" and ref.gap_certified
    res = _vector_miqp().solve(solver="scip", time_limit=30)
    assert res.status == "optimal"
    assert res.gap_certified and res.bound_valid
    assert res.algorithm_route.startswith("scip:")
    assert _rel_close(res.objective, ref.objective)
    assert res.bound <= res.objective + 1e-9
    assert set(res.x) == {"x", "y"} and res.x["x"].shape == (3,)


@pytest.mark.smoke
def test_maximize_reports_model_sense_bound():
    _require_scip()
    m = Model("max")
    x = m.continuous("x", lb=-2, ub=2)
    m.maximize(dm.sin(3 * x) - 0.1 * x**2)
    res = m.solve(solver="scip", time_limit=30)
    assert res.status == "optimal"
    # For a maximization the dual bound is an UPPER bound on the objective.
    assert res.bound >= res.objective - 1e-9
    xs = np.linspace(-2, 2, 200001)
    assert _rel_close(res.objective, float(np.max(np.sin(3 * xs) - 0.1 * xs**2)), rtol=1e-6)


@pytest.mark.smoke
def test_disjunction_is_lowered_and_solved():
    _require_scip()
    m = Model("g")
    x = m.continuous("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=3)
    m.either_or([[x <= 2], [x >= 6]])
    m.minimize((x - 4) ** 2 + y)
    res = m.solve(solver="scip", time_limit=30)
    assert res.status == "optimal"
    assert _rel_close(res.objective, 4.0)
    assert float(res.x["x"]) <= 2 + 1e-6 or float(res.x["x"]) >= 6 - 1e-6
    # The lowering's selector binaries are not reported as user variables.
    assert set(res.x) == {"x", "y"}


@pytest.mark.smoke
def test_integer_infeasibility_is_scips_proof():
    _require_scip()
    m = Model("inf")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.subject_to(x + y == 1.5)
    m.minimize(x + y)
    res = m.solve(solver="scip", time_limit=30)
    assert res.status == "infeasible"
    assert res.algorithm_route.startswith("scip:")


@pytest.mark.smoke
def test_unbounded():
    _require_scip()
    m = Model("unb")
    x = m.continuous("x", lb=0)
    y = m.integer("y", lb=0, ub=3)
    m.subject_to(x >= y)
    m.maximize(x + y)
    res = m.solve(solver="scip", time_limit=30)
    assert res.status in ("unbounded", "unknown")
    assert not res.gap_certified or res.status == "unbounded"


@pytest.mark.smoke
def test_fast_api_rows_and_objective_reach_scip():
    _require_scip()
    m = Model("fast")
    x = m.integer("x", shape=(3,), lb=0, ub=10)
    m.add_linear_constraints(np.array([[2.0, 3.0, 4.0]]), x, "<=", np.array([12.0]))
    m.add_linear_objective(np.array([-3.0, -4.0, -5.0]), x, constant=1.0)
    res = m.solve(solver="scip", time_limit=30)
    assert res.status == "optimal"
    # Exhaustive optimum of max 3a+4b+5c s.t. 2a+3b+4c <= 12, integer in [0, 10].
    best = min(
        -3 * a - 4 * b - 5 * c + 1.0
        for a in range(11)
        for b in range(11)
        for c in range(11)
        if 2 * a + 3 * b + 4 * c <= 12
    )
    assert _rel_close(res.objective, best)


@pytest.mark.smoke
def test_node_limit_is_uncertified_with_incumbent():
    _require_scip()
    rng = np.random.default_rng(7)
    n = 40
    w = rng.integers(10, 60, n).astype(float)
    v = w + rng.integers(1, 10, n)
    m = Model("knap")
    x = m.binary("x", shape=(n,))
    m.subject_to(dm.sum(w * x) <= float(w.sum()) / 2 + 0.5)
    m.maximize(dm.sum(v * x))
    res = m.solve(
        solver="scip",
        time_limit=30,
        max_nodes=1,
        scip_options={"presolving/maxrounds": 0, "separating/maxroundsroot": 0},
    )
    if res.termination == "optimal":
        pytest.skip("SCIP closed this instance at the root; nothing to limit")
    assert res.status == "iteration_limit"
    assert not res.gap_certified
    assert res.objective is not None and res.bound >= res.objective - 1e-9


@pytest.mark.smoke
@pytest.mark.parametrize("rhs, status", [(1.0, "optimal"), (5.0, "infeasible")])
def test_constant_only_row_is_kept_honest(rhs, status):
    """A row whose terms all fold to constants is checked, never dropped blindly."""
    _require_scip()
    m = Model("c")
    x = m.continuous("x", lb=0, ub=1)
    p = m.parameter("p", value=3.0)
    m.subject_to(p * 1.0 >= rhs)
    m.minimize(x)
    res = m.solve(solver="scip")
    assert res.status == status
    assert res.algorithm_route.startswith("scip:")


def test_scip_option_typo_is_an_error():
    _require_scip()
    m = Model("t")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    with pytest.raises(KeyError):
        m.solve(solver="scip", scip_options={"limits/tiem": 1.0})


def test_discopt_search_options_are_declared_ignored():
    _require_scip()
    m = Model("t")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize((x - 0.3) ** 2)
    with pytest.warns(UserWarning, match="solver='scip' ignores threads"):
        res = m.solve(solver="scip", threads=2)
    assert res.status == "optimal"


def test_node_callback_is_refused():
    _require_scip()
    m = Model("t")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    with pytest.raises(ValueError, match="node_callback"):
        m.solve(solver="scip", node_callback=lambda *a, **k: None)


@pytest.mark.smoke
def test_initial_solution_reaches_scip(monkeypatch):
    _require_scip()
    import discopt.solvers.scip as scip_backend

    seen = []
    real = scip_backend.solve_scip

    def spy(*args, **kwargs):
        seen.append(kwargs["initial_point"])
        return real(*args, **kwargs)

    monkeypatch.setattr(scip_backend, "solve_scip", spy)
    m = _vector_miqp()
    x, y = m._variables
    hint = {x: np.array([4.0, 0.0, 2.0]), y: np.array([1.0, 0.0, 1.0])}
    res = m.solve(solver="scip", initial_solution=hint, time_limit=30)
    assert res.status == "optimal"
    assert len(seen) == 1 and seen[0] is not None
    np.testing.assert_allclose(seen[0][:6], [4.0, 0.0, 2.0, 1.0, 0.0, 1.0])


# ── refusals and exact rewrites ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda x: dm.atan(x), id="atan"),
        pytest.param(lambda x: dm.erf(x), id="erf"),
        pytest.param(lambda x: dm.maximum(x, 1.0 - x), id="max"),
        pytest.param(lambda x: (x - 1.5) ** x, id="var-exponent-base-not-positive"),
        pytest.param(lambda x: dm.xlogx(x - 0.5), id="xlogx-at-zero"),
    ],
)
def test_unrepresentable_nodes_are_refused(build):
    _require_scip()
    from discopt.solvers.scip import ScipUnsupportedError, build_scip_model

    m = Model("r")
    x = m.continuous("x", lb=0.5, ub=2.0)
    m.minimize(build(x))
    with pytest.raises(ScipUnsupportedError):
        build_scip_model(m)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build, f",
    [
        pytest.param(lambda x: x**x, lambda v: v**v, id="x**x"),
        pytest.param(lambda x: dm.xlogx(x), lambda v: v * np.log(v), id="xlogx"),
        pytest.param(
            lambda x: dm.tanh(x - 1) + dm.log10(x) + dm.sigmoid(-x),
            lambda v: np.tanh(v - 1) + np.log10(v) + 1 / (1 + np.exp(v)),
            id="tanh-log10-sigmoid",
        ),
        pytest.param(
            lambda x: dm.cosh(x - 1) + dm.log1p(x) - dm.softplus(x - 2),
            lambda v: np.cosh(v - 1) + np.log1p(v) - np.log1p(np.exp(v - 2)),
            id="cosh-log1p-softplus",
        ),
    ],
)
def test_exact_rewrites_reach_the_true_minimum(build, f):
    """A rewrite that changed the function would move the minimum."""
    _require_scip()
    m = Model("w")
    x = m.continuous("x", lb=0.2, ub=2.0)
    m.minimize(build(x))
    res = m.solve(solver="scip", time_limit=30)
    assert res.status == "optimal"
    xs = np.linspace(0.2, 2.0, 400001)
    assert _rel_close(res.objective, float(np.min(f(xs))), rtol=1e-6, atol=1e-6)


# ── fidelity: same SCIP, two ingest paths ────────────────────────────────────

_FIDELITY = [
    "alan",
    "chance",
    "dispatch",
    "ex1221",
    "ex1222",
    "ex1224",
    "ex1225",
    "ex1226",
    "gbd",
    "gkocis",
    "nvs01",
    "nvs03",
    "nvs06",
    "nvs10",
    "oaer",
    "st_e13",
    "st_e38",
    "st_miqp1",
    "st_testgr3",
    "syn05hfsg",
]


@pytest.mark.parametrize("name", _FIDELITY)
def test_translation_matches_scip_nl_reader(name):
    ps = _require_scip()
    path = _NL_DIR / f"{name}.nl"
    if not path.exists():
        pytest.skip(f"{path} not in checkout")
    ref = ps.Model()
    ref.hideOutput()
    ref.readProblem(str(path))
    ref.setParam("limits/time", 60.0)
    ref.optimize()
    assert ref.getStatus() == "optimal", ref.getStatus()
    ref_obj = float(ref.getObjVal())

    res = from_nl(str(path)).solve(solver="scip", time_limit=60)
    assert res.status == "optimal", (res.status, res.termination)
    assert res.gap_certified
    assert _rel_close(res.objective, ref_obj), (res.objective, ref_obj)


# ── the certificate guards ───────────────────────────────────────────────────


def _fake_raw(x, scip_objective, dual_bound, status="optimal"):
    return {
        "scip_status": status,
        "x": None if x is None else np.asarray(x, dtype=float),
        "scip_objective": scip_objective,
        "dual_bound": dual_bound,
        "gap": 0.0,
        "node_count": 1,
        "n_scip_vars": 1,
        "n_scip_conss": 1,
        "version": "SCIP fake",
    }


def _box_model() -> Model:
    m = Model("b")
    x = m.continuous("x", lb=0, ub=4)
    m.subject_to(x >= 1)
    m.minimize((x - 2) ** 2)
    return m


def test_objective_disagreement_raises(monkeypatch):
    import discopt.solvers.scip as scip_backend

    monkeypatch.setattr(
        scip_backend,
        "solve_scip",
        lambda *a, **k: _fake_raw([2.0], scip_objective=-1.0, dual_bound=-1.0),
    )
    with pytest.raises(RuntimeError, match="not faithful"):
        _box_model().solve(solver="scip")


def test_bound_past_verified_objective_raises(monkeypatch):
    import discopt.solvers.scip as scip_backend

    monkeypatch.setattr(
        scip_backend,
        "solve_scip",
        lambda *a, **k: _fake_raw([2.0], scip_objective=0.0, dual_bound=0.5),
    )
    with pytest.raises(RuntimeError, match="dual bound"):
        _box_model().solve(solver="scip")


def test_point_discopt_rejects_is_reported_uncertified(monkeypatch):
    import discopt.solvers.scip as scip_backend

    # x = 0.5 violates x >= 1 by 0.5; SCIP "claims" it optimal.
    monkeypatch.setattr(
        scip_backend,
        "solve_scip",
        lambda *a, **k: _fake_raw([0.5], scip_objective=2.25, dual_bound=0.0),
    )
    res = _box_model().solve(solver="scip")
    assert res.status == "feasible"
    assert not res.gap_certified
    assert res.solver_stats["certificate/incumbent_unverified"] == 1.0
