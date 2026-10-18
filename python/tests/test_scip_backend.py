"""``solver="scip"``: the whole model handed to SCIP through PySCIPOpt (issue #1784).

Three things are pinned here:

1. **Fidelity.** The in-memory translation is the same model SCIP reads from the
   ``.nl`` file: solving a corpus instance both ways with the same SCIP binary
   reaches the same optimum. That is a differential test with SCIP held fixed, so
   a disagreement can only be the translation.
2. **Refusals.** Every node without an exact SCIP spelling is refused by name,
   never approximated -- including the domain-restricted rewrites (``x**y`` and
   ``xlogx``), which are only exact where their argument is proven positive.
3. **Certificates.** Nothing this route returns is certified (#1784 review B2:
   SCIP's presolve certified a bound past the true optimum of ``ex1252a``).
   SCIP's point is judged by discopt's ``verify_point`` and its objective
   recomputed; a contradiction between the two raises unless SCIP's numerics
   explain it, and an answer resting on SCIP's numerical limits (a pole, a value
   past ``numerics/hugeval``) carries no bound (review B1).
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


def _assert_scip_claimed_optimal(res) -> None:
    """SCIP proved optimality; discopt reports it, verified, but never certified."""
    assert res.status == "feasible", (res.status, res.termination)
    assert res.termination in ("optimal", "gaplimit"), res.termination
    assert not res.gap_certified and not res.bound_valid
    assert res.solver_stats["scip/claimed_optimal"] == 1.0
    assert res.bound_source == "external_solver"
    assert res.algorithm_route.startswith("scip:")


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
    _assert_scip_claimed_optimal(res)
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
    _assert_scip_claimed_optimal(res)
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
    _assert_scip_claimed_optimal(res)
    assert _rel_close(res.objective, 4.0)
    assert float(res.x["x"]) <= 2 + 1e-6 or float(res.x["x"]) >= 6 - 1e-6
    # The lowering's selector binaries are not reported as user variables.
    assert set(res.x) == {"x", "y"}


@pytest.mark.smoke
def test_integer_infeasibility_is_scips_claim_not_a_certificate():
    """Review B3: SCIP's ``infeasible`` comes from the presolve that certified a
    bound past the optimum of ex1252a, so it is reported, not certified."""
    _require_scip()
    m = Model("inf")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.subject_to(x + y == 1.5)
    m.minimize(x + y)
    res = m.solve(solver="scip", time_limit=30)
    assert res.status == "unknown" and res.termination == "infeasible"
    assert not res.gap_certified
    assert res.solver_stats["scip/claimed_infeasible"] == 1.0
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
    assert res.status == "unknown"
    assert res.termination in ("unbounded", "inforunbd")
    assert not res.gap_certified


@pytest.mark.smoke
def test_fast_api_rows_and_objective_reach_scip():
    _require_scip()
    m = Model("fast")
    x = m.integer("x", shape=(3,), lb=0, ub=10)
    m.add_linear_constraints(np.array([[2.0, 3.0, 4.0]]), x, "<=", np.array([12.0]))
    m.add_linear_objective(np.array([-3.0, -4.0, -5.0]), x, constant=1.0)
    res = m.solve(solver="scip", time_limit=30)
    _assert_scip_claimed_optimal(res)
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
    assert res.status == "node_limit"
    assert not res.gap_certified
    assert res.objective is not None and res.bound >= res.objective - 1e-9


@pytest.mark.smoke
@pytest.mark.parametrize("rhs, status", [(1.0, "feasible"), (5.0, "unknown")])
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
    _assert_scip_claimed_optimal(res)


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
    _assert_scip_claimed_optimal(res)
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
    _assert_scip_claimed_optimal(res)
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
    _assert_scip_claimed_optimal(res)
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
        "numerics": [],
        "singular_sites": [],
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
    with pytest.raises(RuntimeError, match="does not match this one"):
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


def test_contradiction_on_a_singular_box_is_numerics_not_a_raise(monkeypatch):
    """Review B1: ``min log(x)`` on [0, 1] -- SCIP called 1e5 optimal at x = 1."""
    import discopt.solvers.scip as scip_backend

    raw = _fake_raw([2.0], scip_objective=-1.0, dual_bound=-1.0)
    raw["singular_sites"] = ["the log argument in log(x)"]
    monkeypatch.setattr(scip_backend, "solve_scip", lambda *a, **k: raw)
    res = _box_model().solve(solver="scip")
    assert res.status == "feasible"
    assert res.bound is None and not res.gap_certified
    assert "admits the log argument" in res.solver_stats["scip/numerical_limit"]


# ── review findings (#1784) ──────────────────────────────────────────────────


def _inv(m):
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize(1.0 / x)


def _neg_pow(m):
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize(x**-1)


def _tan_pole_max(m):
    x = m.continuous("x", lb=0, ub=3)
    m.maximize(dm.tan(x))


def _tan_pole_min(m):
    x = m.continuous("x", lb=1, ub=3)
    m.minimize(dm.tan(x))


def _log_unbounded_domain(m):
    x = m.continuous("x", lb=1)
    y = m.continuous("y", lb=-10)
    m.subject_to(y <= dm.log(x))
    m.maximize(y)


def _log_at_zero(m):
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(dm.log(x))


def _free_bilinear(m):
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x * y)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build",
    [
        _inv,
        _neg_pow,
        _tan_pole_max,
        _tan_pole_min,
        _log_unbounded_domain,
        _log_at_zero,
        _free_bilinear,
    ],
)
def test_unbounded_problems_are_never_certified(build):
    """Review B1: each of these is unbounded. SCIP stopped against its own
    numerical limits and called a finite value optimal; before the fix four
    came back ``optimal``/``gap_certified=True`` and three raised a misleading
    "translation is not faithful"."""
    _require_scip()
    m = Model(build.__name__)
    build(m)
    res = m.solve(solver="scip", time_limit=5)
    assert not res.gap_certified and not res.bound_valid
    assert res.status in ("feasible", "time_limit")
    assert res.bound is None
    assert res.solver_stats["scip/numerical_limit"]


def test_ex1252a_presolve_bound_is_not_certified():
    """Review B2: SCIP's presolve certifies 131452.97 with a bound past the true
    optimum 128893.741 (MINLPLib ``.solu``) on the faithfully translated model. The
    point is real, the bound is not, so nothing is certified."""
    _require_scip()
    path = pathlib.Path(__file__).parent / "data" / "minlplib" / "ex1252a.nl"
    if not path.exists():
        pytest.skip(f"{path} not in checkout")
    res = from_nl(str(path)).solve(solver="scip", time_limit=60)
    assert not res.gap_certified and not res.bound_valid
    assert res.status in ("feasible", "time_limit")
    # The incumbent is a verified feasible point, so it can never beat the optimum.
    assert res.objective >= 128893.741 - 1e-3


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda x: x**1.5 <= 100.0, id="x**1.5"),
        pytest.param(lambda x: x ** (1.0 / 3.0) <= 1.0, id="cube-root"),
        pytest.param(lambda x: dm.sqrt(x) <= 1.0, id="sqrt"),
    ],
)
def test_fractional_power_of_a_possibly_negative_base_gets_its_domain(build):
    """Review minor 1: discopt evaluates these to NaN for x < 0, so x >= 0 is part
    of the feasible set. Before the fix SCIP returned x = -8 and the route ended
    in ``error``; with the domain row it reaches the true optimum x = 0."""
    _require_scip()
    m = Model("frac")
    x = m.continuous("x", lb=-8, ub=8)
    m.subject_to(build(x))
    m.minimize(x)
    res = m.solve(solver="scip", time_limit=20)
    _assert_scip_claimed_optimal(res)
    assert abs(res.objective) <= 1e-6


def test_negative_constant_to_a_fractional_power_is_refused():
    """Review minor 1: ``float(a) ** float(b)`` returned a Python complex here."""
    _require_scip()
    from discopt.solvers.scip import build_scip_model

    m = Model("cfold")
    x = m.continuous("x", lb=0, ub=1)
    p = m.parameter("p", value=-8.0)
    m.minimize(x + p ** (1.0 / 3.0))
    with pytest.raises(ValueError, match="not a real number"):
        build_scip_model(m)


def test_limit_exits_never_claim_a_point_they_do_not_have():
    """Review minor 2."""
    from discopt.solvers.scip import limit_status

    assert limit_status("sollimit", has_point=False) == "unknown"
    assert limit_status("sollimit", has_point=True) == "feasible"
    assert limit_status("memlimit", has_point=False) == "error"
    assert limit_status("memlimit", has_point=True) == "feasible"
    assert limit_status("nodelimit", has_point=True) == "node_limit"
    assert limit_status("optimal", has_point=True) is None


def test_explicit_default_max_nodes_is_forwarded(monkeypatch):
    """Review minor 3: max_nodes=100_000 was mistaken for "not given"."""
    _require_scip()
    import discopt.solvers.scip as scip_backend

    seen = []
    real = scip_backend.solve_scip

    def spy(*args, **kwargs):
        seen.append(dict(kwargs["options"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(scip_backend, "solve_scip", spy)
    m = Model("n")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    m.solve(solver="scip", max_nodes=100_000)
    assert len(seen) == 1 and seen[0]["limits/totalnodes"] == 100_000


def test_a_relation_the_lowering_left_behind_is_refused(monkeypatch):
    """Review minor 4: the row iteration would silently drop it (a relaxed model)."""
    _require_scip()
    import discopt.transformations as transformations

    class _Identity:
        def apply(self, model, **_kw):
            return model

    m = Model("g")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 6]])
    m.minimize((x - 4) ** 2)
    monkeypatch.setattr(transformations, "get", lambda name: _Identity())
    with pytest.raises(ValueError, match="disjunctive"):
        m.solve(solver="scip")


_SCIP_EXITS = [
    "optimal",
    "gaplimit",
    "infeasible",
    "unbounded",
    "inforunbd",
    "timelimit",
    "nodelimit",
    "totalnodelimit",
    "stallnodelimit",
    "restartlimit",
    "memlimit",
    "userinterrupt",
    "sollimit",
    "bestsollimit",
    "primallimit",
    "duallimit",
    "unknown",
]


@pytest.mark.parametrize("with_point", [True, False], ids=["point", "no-point"])
@pytest.mark.parametrize("scip_status", _SCIP_EXITS)
def test_no_scip_exit_is_ever_certified(monkeypatch, scip_status, with_point):
    """Review B3: pin ``gap_certified is False`` (and no proof status) for every
    SCIP exit, with and without a point -- the route reports, never certifies."""
    import discopt.solvers.scip as scip_backend

    raw = _fake_raw([2.0] if with_point else None, 0.0 if with_point else None, 0.0, scip_status)
    monkeypatch.setattr(scip_backend, "solve_scip", lambda *a, **k: raw)
    res = _box_model().solve(solver="scip")
    assert res.gap_certified is False
    assert res.bound_valid is False
    assert res.status not in ("optimal", "infeasible", "unbounded")
    assert res.termination == scip_status
