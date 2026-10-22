"""#1775 G3.2: the framework's master MILP and fixed-configuration subproblem.

Before: LOA's master lived inside ``solvers/gdpopt_loa.py`` and there was no
reusable subproblem; ``discopt.gdp.algorithms.master`` / ``.subproblem`` did not
exist.

After (NC): ``MasterMILP`` (linear rows, OA cut pool, exact no-goods; LOA's
master functions moved unchanged and imported back by LOA) and
``solve_subproblem`` (the reduced model of spec §5.2 solved locally or globally
within the caller's deadline; certificates read from ``gap_certified``).
"""

from __future__ import annotations

import itertools
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.gdp.algorithms import GDPAlgorithm
from discopt.gdp.algorithms.base import Deadline
from discopt.gdp.algorithms.master import DEFAULT_MASTER_BACKEND, MasterMILP
from discopt.gdp.algorithms.subproblem import solve_subproblem
from discopt.gdp.tree import gdp_tree

pytestmark = pytest.mark.smoke


# ── master ───────────────────────────────────────────────────────────────────


def _binary_master(backend=DEFAULT_MASTER_BACKEND):
    """min 3 y1 + 2 y2 + 10  s.t.  y1 + y2 >= 1, y binary."""
    m = MasterMILP(
        2, [0, 0], [1, 1], [1, 1], objective=(np.array([3.0, 2.0]), 10.0), backend=backend
    )
    m.add_row([1.0, 1.0], ">=", 1.0)
    return m


def test_master_solves_with_the_objective_constant():
    r = _binary_master().solve(time_limit=10)
    assert r.config == (0, 1) and r.bound == pytest.approx(12.0)
    assert not r.proven_infeasible


def test_master_binary_no_goods_walk_every_configuration_then_prove_infeasible():
    m = _binary_master()
    seen = []
    for _ in range(3):
        r = m.solve(time_limit=10)
        seen.append((r.config, round(r.bound, 6)))
        m.exclude(r.config)
    assert seen == [((0, 1), 12.0), ((1, 0), 13.0), ((1, 1), 15.0)]
    last = m.solve(time_limit=10)
    assert last.x is None and last.proven_infeasible and m.n_solves == 4


def test_master_general_integer_exclusion_is_exact():
    """#756: the binary cut would also remove 2 and 3 here."""
    m = MasterMILP(1, [0], [3], [1], objective=(np.array([1.0]), 0.0))
    m.add_row([1.0], ">=", 1.0)
    assert not m.all_binary
    walk = []
    for _ in range(3):
        r = m.solve(time_limit=10)
        walk.append(r.config)
        m.exclude(r.config)
    assert walk == [(1,), (2,), (3,)]
    assert m.solve(time_limit=10).proven_infeasible


def test_master_epigraph_bound_needs_a_convex_objective():
    # min eta, eta >= 2 - y  (a cut on (y, eta)), y binary -> y = 1, eta = 1
    m = MasterMILP(1, [0], [1], [1], objective_convex=True)
    m.add_cut([-1.0, -1.0], -2.0)
    r = m.solve(time_limit=10)
    assert r.config == (1,) and r.bound == pytest.approx(1.0)
    flat = MasterMILP(1, [0], [1], [1])
    assert not flat.bound_valid
    with pytest.raises(ValueError, match="epigraph cut needs objective_convex"):
        flat.add_cut([-1.0, -1.0], -2.0)
    assert flat.solve(time_limit=10).bound is None


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda m: m.add_row([1.0], "<=", 0.0), "expected 2 coefficients"),
        (lambda m: m.add_row([1.0, 1.0], "<", 0.0), "unknown sense"),
        (lambda m: m.add_cut([1.0, np.inf], 0.0), "non-finite cut"),
        (lambda m: m.exclude((1,)), "does not match"),
    ],
)
def test_master_refuses_malformed_input(call, match):
    with pytest.raises(ValueError, match=match):
        call(_binary_master())


def test_master_refuses_an_exact_no_good_on_an_unbounded_integer():
    """Review of #1864: the #756 augmentation's coefficients are u - v + 1, v + 1 - l."""
    for ub in (np.inf, 1e20):
        m = MasterMILP(1, [0], [ub], [1], objective=(np.array([1.0]), 0.0))
        with pytest.raises(ValueError, match="needs finite bounds"):
            m.exclude((1,))
        assert m.no_good_configs == []


def test_master_refuses_a_linear_objective_with_the_epigraph_flag():
    with pytest.raises(ValueError, match="needs none"):
        MasterMILP(1, [0], [1], [1], objective=(np.array([1.0]), 0.0), objective_convex=True)


def test_master_ignores_a_non_finite_incumbent(monkeypatch):
    """No incumbent yet (inf) must not reach the master's gap rule (LOA: UB < 1e20)."""
    import discopt.solvers._gap as gap

    seen = []

    orig = gap.master_gap_tolerance
    monkeypatch.setattr(
        gap, "master_gap_tolerance", lambda tol, inc: seen.append(inc) or orig(tol, inc)
    )
    _binary_master().solve(time_limit=10, incumbent=float("inf"))
    _binary_master().solve(time_limit=10, incumbent=15.0)
    assert seen == [None, 5.0]


def test_master_backends_agree():
    pytest.importorskip("highspy")
    for backend in ("simplex", "highs"):
        m = _binary_master(backend)
        r = m.solve(time_limit=10)
        assert (r.config, round(r.bound, 6)) == ((0, 1), 12.0), backend


def test_loa_uses_the_framework_master():
    import discopt.gdp.algorithms.master as master
    import discopt.solvers.gdpopt_loa as loa

    for name in (
        "_solve_master_milp",
        "_add_no_good_cut",
        "_build_no_good_augmentation",
        "_discrete_vars_all_binary",
        "_int_config_key",
    ):
        assert getattr(loa, name) is getattr(master, name)


# ── subproblem ───────────────────────────────────────────────────────────────


def _gdp(with_integer=True, sense="min"):
    """[x <= 2] v [x >= 5] on x in [0, 10]; (x - 3)^2 (+ (y - 0.4)^2, y integer)."""
    m = dm.Model("g32")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    obj = (x - 3) ** 2
    if with_integer:
        y = m.integer("y", lb=0, ub=3)
        m.subject_to(x + y >= 1.5)
        obj = obj + (y - 0.4) ** 2
    if sense == "min":
        m.minimize(obj)
    else:
        m.maximize(-obj)
    return m


def _disjuncts(m):
    (node,) = list(gdp_tree(m).walk())
    return node.disjuncts


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


@pytest.mark.parametrize("j, value", [(0, 1.16), (1, 4.16)])
def test_global_subproblem_certifies_each_configuration(j, value):
    m = _gdp()
    r = _quiet(solve_subproblem, m, [_disjuncts(m)[j]], Deadline(60), mode="global")
    assert r.status == "optimal" and r.certified
    assert r.objective == pytest.approx(value, abs=1e-6)
    assert r.bound == pytest.approx(value, abs=1e-6) and r.bound <= r.objective + 1e-9
    assert len(r.x) == sum(v.size for v in m._variables)  # the original layout


def test_global_subproblem_maximize_is_in_model_units():
    m = _gdp(sense="max")
    r = _quiet(solve_subproblem, m, [_disjuncts(m)[0]], Deadline(60), mode="global")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.16, abs=1e-6) and r.bound >= r.objective - 1e-9


def _infeasible_first_disjunct():
    m = dm.Model("inf")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    m.subject_to(x >= 3)
    m.minimize(x)
    return m


def test_global_subproblem_proves_infeasibility_local_never_does():
    m = _infeasible_first_disjunct()
    d0 = _disjuncts(m)[0]
    g = _quiet(solve_subproblem, m, [d0], Deadline(60), mode="global")
    assert g.status == "infeasible" and g.certified
    loc = _quiet(solve_subproblem, m, [d0], Deadline(60), mode="local")
    assert loc.status == "unresolved" and not loc.certified


def test_an_uncertified_infeasible_is_unresolved(monkeypatch):
    """``infeasible`` proves nothing without ``gap_certified`` (the SolveResult
    contract); LOA would exclude the configuration on it, and with every one
    excluded report the GDP infeasible."""
    from types import SimpleNamespace

    from discopt.modeling.core import Model

    def heuristic(self, **k):
        return SimpleNamespace(
            status="infeasible", gap_certified=False, bound=None, bound_valid=False, x=None
        )

    m = _infeasible_first_disjunct()
    d0 = _disjuncts(m)[0]
    monkeypatch.setattr(Model, "solve", heuristic)
    r = solve_subproblem(m, [d0], Deadline(60), mode="global")
    assert r.status == "unresolved" and not r.certified


def test_local_subproblem_returns_a_verified_point():
    m = _gdp(with_integer=False)
    r = _quiet(solve_subproblem, m, [_disjuncts(m)[1]], Deadline(60), mode="local")
    assert r.status == "feasible" and not r.certified and r.bound is None
    assert r.objective == pytest.approx(4.0, abs=1e-6) and float(r.x[0]) == pytest.approx(5.0)


def test_local_subproblem_refuses_a_fractional_integer():
    """The local NLP relaxes y; its point (y = 0.4) is not an incumbent."""
    m = _gdp()
    r = _quiet(solve_subproblem, m, [_disjuncts(m)[0]], Deadline(60), mode="local")
    assert r.status == "unresolved" and "not integral" in r.reason


def test_an_expired_deadline_solves_nothing(monkeypatch):
    import discopt.gdp.algorithms.subproblem as sp

    t = [0.0]
    d = Deadline(1.0, clock=lambda: t[0])
    t[0] = 2.0
    monkeypatch.setattr(sp, "_global", lambda *a: pytest.fail("solved past the deadline"))
    m = _gdp()
    r = solve_subproblem(m, [_disjuncts(m)[0]], d, mode="global")
    assert r.status == "unresolved" and r.reason == "deadline expired"


def test_the_deadline_caps_the_subsolve(monkeypatch):
    import discopt.gdp.algorithms.subproblem as sp

    seen = []
    monkeypatch.setattr(sp, "_global", lambda sub, limit, tol: seen.append(limit) or None)
    m = _gdp()
    solve_subproblem(m, [_disjuncts(m)[0]], Deadline(100.0), mode="global", time_cap=7.0)
    solve_subproblem(m, [_disjuncts(m)[0]], Deadline(3.0), mode="global", time_cap=7.0)
    assert seen[0] == pytest.approx(7.0) and 0 < seen[1] <= 3.0


def test_unknown_mode_is_refused():
    m = _gdp()
    with pytest.raises(ValueError, match="mode must be one of"):
        solve_subproblem(m, [_disjuncts(m)[0]], Deadline(1.0), mode="globl")


def test_unknown_nlp_solver_is_refused_even_past_the_deadline():
    t = [0.0]
    d = Deadline(1.0, clock=lambda: t[0])
    t[0] = 2.0
    m = _gdp()
    with pytest.raises(ValueError, match="nlp_solver must be"):
        solve_subproblem(m, [_disjuncts(m)[0]], d, nlp_solver="ipotp")


# ── the pieces together ──────────────────────────────────────────────────────


class _EnumerateEveryConfiguration(GDPAlgorithm):
    """Test harness: global subproblem per configuration; LB = min over certified
    configuration bounds once every configuration is closed."""

    route = "gdp:test-enumerate"

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        nodes = list(gdp_tree(model).walk())
        self.todo = list(itertools.product(*(n.disjuncts for n in nodes)))
        self.config_bounds = []

    def iterate(self):
        sel = self.todo.pop(0)
        key = tuple(id(d) for d in sel)
        r = solve_subproblem(self.model, sel, self.deadline, mode="global")
        self.n_subproblems += 1
        if r.status == "unresolved":
            self.mark_unresolved(key)
        else:
            self.mark_solved(key)
            if r.x is not None:
                self.offer_incumbent(r.x)
            if r.status == "optimal":
                self.config_bounds.append(self.obj_sign * r.bound)
        if not self.todo and not self.unresolved:
            if self.config_bounds:
                self.offer_bound(min(self.config_bounds))
            else:
                self.prove_infeasible()
        return bool(self.todo)


def test_framework_certifies_by_enumeration():
    m = _gdp()
    r = _quiet(_EnumerateEveryConfiguration(m, time_limit=120).run)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(1.16, abs=1e-6) and r.bound <= r.objective + 1e-9
    assert r.solver_stats["n_subproblems"] == 2


def test_framework_proves_infeasibility_by_enumeration():
    m = dm.Model("allinf")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    m.subject_to(x >= 3)
    m.subject_to(x <= 4)
    m.minimize(x)
    r = _quiet(_EnumerateEveryConfiguration(m, time_limit=120).run)
    assert r.status == "infeasible" and r.gap_certified
