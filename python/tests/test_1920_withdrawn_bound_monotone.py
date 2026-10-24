"""#1920: the HiGHS MILP route on a ReLU network: bound, speed, incumbent.

1. A withdrawn certificate fell back to the root LP bound. When HiGHS proves a MILP
   optimal late in the budget, the #1634 presolve-free cross-solve reaches no verdict
   and the certificate is withdrawn; ``_withdraw`` then published the NS-safe ROOT
   bound, so a longer run returned a looser bound than a shorter one (issue: -15.30 at
   600 s against -4.38 at 180 s on a 2-64-64-1 ``relu_bigm`` net). The bound is now
   ``max(root, min(primary tree bound, cross tree bound))`` -- the ``min`` rule
   ``_weaker_bound`` already certifies on -- and the same rule holds on a budgeted exit
   (``_merge_uncertified_cross``), so the bound no longer depends on which exit a run
   happened to take.
2. The cross-solve ran after the primary, doubling the wall time; it now runs
   concurrently (``DISCOPT_MILP_CONCURRENT_CROSS``).
3. HiGHS found no incumbent within a minute on the issue's class; the route now seeds
   the primary with a verified fix-and-propagate point (``DISCOPT_MILP_FIXPROP_SEED``).

The cross-solve's "no verdict" is made deterministic here by capping it at one node
instead of racing a clock.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ml import add_predictor
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition
from discopt.solver import _highs_std_form
from discopt.solvers import lp_milp_highs


def _relu_model(seed: int = 0, width: int = 16) -> dm.Model:
    rng = np.random.default_rng(seed)
    lb, ub = np.array([-2.0, -1.0]), np.array([2.0, 1.0])
    layers = [
        DenseLayer(rng.normal(size=(2, width)), rng.normal(size=width) * 0.5, Activation.RELU),
        DenseLayer(
            rng.normal(size=(width, width)) / np.sqrt(width),
            rng.normal(size=width) * 0.5,
            Activation.RELU,
        ),
        DenseLayer(
            rng.normal(size=(width, 1)) / np.sqrt(width),
            rng.normal(size=1) * 0.1,
            Activation.LINEAR,
        ),
    ]
    m = dm.Model("i1920")
    x = m.continuous("x", shape=(2,), lb=lb, ub=ub)
    y, _ = add_predictor(m, x, NetworkDefinition(layers, input_bounds=(lb, ub)), method="relu_bigm")
    m.minimize(y[0])
    return m


def _one_node_cross(monkeypatch):
    """Cap the presolve-free cross-solve at one node: it returns a tree bound and no
    verdict, deterministically."""
    real = lp_milp_highs._solve_milp_scaled
    calls = []

    def capped(sf, *, presolve=True, **kw):
        if not presolve:
            kw = dict(kw, max_nodes=1)
            calls.append(1)
        return real(sf, presolve=presolve, **kw)

    monkeypatch.setattr(lp_milp_highs, "_solve_milp_scaled", capped)
    return calls


def test_withdrawn_certificate_keeps_the_tree_bounds(monkeypatch):
    truth = _relu_model().solve()
    assert truth.status == "optimal" and truth.gap_certified
    opt = float(truth.objective)

    calls = _one_node_cross(monkeypatch)
    r = _relu_model().solve()
    assert calls, "the presolve-free cross-solve never ran"
    stats = r.solver_stats
    assert stats.get("milp/presolve_cross_check_no_verdict") == 1.0, stats
    assert r.status == "feasible" and not r.gap_certified
    root = stats["milp/root_ns_bound"]
    # Before #1920 the bound was exactly the root bound.
    assert stats.get("milp/withdrawn_bound_from_trees") == 1.0
    assert r.bound > root + 1e-6 * (1.0 + abs(root)), (r.bound, root)
    assert r.bound <= opt + 1e-9 * (1.0 + abs(opt)), (r.bound, opt)
    assert r.bound_valid and r.bound_source == "bnb_tree"


def _sf():
    m = dm.Model("tiny")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(1.37 * x + 2.11 * y)
    m.subject_to(0.7 * x + 1.3 * y >= 2.9)
    return _highs_std_form(m)[2]


def _primary(sf, bound=4.0, root=1.0):
    out = lp_milp_highs.HighsOutcome(
        "optimal", x=np.zeros(sf.n), objective=4.5, bound=bound, gap_certified=True,
        root_bound=root,
    )  # fmt: skip
    out.wall_time = 1.0
    return out


@pytest.mark.parametrize(
    ("cross_bound", "expected", "provenance"),
    [
        (3.0, 3.0, "min(presolved, presolve-free)"),  # the weaker tree bound
        (5.0, 4.0, "min(presolved, presolve-free)"),  # the primary is the weaker one
        (0.5, 1.0, "root-ns"),  # both trees weaker than the root: the root stands
        (None, 1.0, "root-ns"),  # no cross bound: the root stands
    ],
)
def test_withdraw_bound_rule(cross_bound, expected, provenance):
    sf = _sf()
    cross = lp_milp_highs.HighsOutcome("time_limit", bound=cross_bound)
    got = lp_milp_highs._weaker_bound(sf, _primary(sf), cross, {"gap_tolerance": 1e-4})
    assert got.status == "feasible" and not got.gap_certified
    assert got.labels["milp/certificate"] == "declined"
    assert got.bound == pytest.approx(expected)
    assert got.labels["milp/bound_provenance"] == provenance


def test_withdraw_bound_never_exceeds_the_incumbent():
    sf = _sf()
    cross = lp_milp_highs.HighsOutcome("time_limit", bound=9.0)
    got = lp_milp_highs._weaker_bound(sf, _primary(sf, bound=9.0), cross, {"gap_tolerance": 1e-4})
    assert got.bound == pytest.approx(4.5)


def test_no_budget_withdrawal_keeps_the_root_bound():
    """The cross-solve never ran: there is no second configuration to hold the presolved
    bound against, so only the root bound is published (unchanged by #1920)."""
    sf = _sf()
    assert not lp_milp_highs.presolve_exact_class(sf)
    got = lp_milp_highs._cross_check_presolve(
        sf, _primary(sf), {"time_limit": 1.0, "gap_tolerance": 1e-4}
    )
    assert got.stats["milp/presolve_cross_check_skipped"] == 1.0
    assert got.bound == pytest.approx(1.0)
    assert got.labels["milp/bound_provenance"] == "root-ns"


# ── The concurrent cross-solve and the uncertified merge ──


def test_cross_solve_runs_concurrently_and_certifies():
    r = _relu_model(width=8).solve()
    assert r.status == "optimal" and r.gap_certified
    assert r.solver_stats.get("milp/presolve_cross_concurrent") == 1.0
    assert r.solver_stats.get("milp/presolve_cross_check_ran") == 1.0


def test_concurrent_cross_opt_out_restores_the_sequential_check(monkeypatch):
    monkeypatch.setenv("DISCOPT_MILP_CONCURRENT_CROSS", "0")
    r = _relu_model(width=8).solve()
    assert r.status == "optimal" and r.gap_certified
    assert "milp/presolve_cross_concurrent" not in r.solver_stats
    assert r.solver_stats.get("milp/presolve_cross_check_ran") == 1.0


def test_a_node_limited_solve_publishes_the_weaker_tree_bound():
    """A budgeted exit no longer publishes the presolved tree bound alone: the stopped
    cross-solve's bound joins it, so the same rule holds on every exit."""
    r = _relu_model(width=24).solve(max_nodes=2)
    stats = r.solver_stats
    assert r.status in ("feasible", "node_limit"), r.status
    assert stats.get("milp/presolve_cross_concurrent") == 1.0
    assert "milp/presolve_cross_bound" in stats, stats
    assert r.bound is not None and r.bound <= stats["milp/presolve_cross_bound"] + 1e-12


def _merge(sf, out, cross):
    return lp_milp_highs._merge_uncertified_cross(sf, out, cross, None)


def _uncertified(sf, *, status="time_limit", x=None, objective=None, bound=-2.0, root=None):
    return lp_milp_highs.HighsOutcome(
        status, x=x, objective=objective, bound=bound, root_bound=root
    )


def test_merge_takes_the_weaker_tree_bound():
    sf = _sf()
    got = _merge(
        sf, _uncertified(sf, bound=3.0), lp_milp_highs.HighsOutcome("node_limit", bound=2.5)
    )
    assert got.bound == pytest.approx(2.5)
    assert got.labels["milp/bound_provenance"] == "min(presolved, presolve-free)"


def test_merge_never_raises_the_primary_bound():
    sf = _sf()
    got = _merge(
        sf, _uncertified(sf, bound=2.0), lp_milp_highs.HighsOutcome("node_limit", bound=3.5)
    )
    assert got.bound == pytest.approx(2.0)


def test_merge_keeps_a_stronger_root_bound():
    sf = _sf()
    got = _merge(
        sf,
        _uncertified(sf, bound=3.0, root=2.8),
        lp_milp_highs.HighsOutcome("node_limit", bound=1.0),
    )
    assert got.bound == pytest.approx(2.8)
    assert got.labels["milp/bound_provenance"] == "root-ns"


def _point(sf, x, y):
    """A verified point of :func:`_sf` (``x``, ``y`` integral, row satisfied)."""
    pt = lp_milp_highs._fixed_integer_lp(sf, np.r_[x, y, np.zeros(sf.n - 2)], None)
    got = lp_milp_highs._verified_mip_point(sf, pt.x)
    assert got is not None
    return got


def test_merge_adopts_a_verified_cross_point_when_the_primary_has_none():
    sf = _sf()
    x, obj = _point(sf, 0, 3)  # 6.33
    got = _merge(
        sf, _uncertified(sf, bound=4.0), lp_milp_highs.HighsOutcome("node_limit", x=x, bound=4.0)
    )
    assert got.status == "feasible" and got.objective == pytest.approx(obj)
    assert got.stats["milp/presolve_cross_incumbent_adopted"] == 1.0


def test_merge_refuted_primary_bound_falls_back_to_the_root():
    """A verified cross point below the primary's tree bound proves that bound false."""
    sf = _sf()
    x, obj = _point(sf, 0, 3)
    got = _merge(
        sf,
        _uncertified(sf, bound=obj + 1.0, root=1.0),
        lp_milp_highs.HighsOutcome("node_limit", x=x, bound=0.5),
    )
    assert got.stats["milp/presolve_bound_refuted"] == 1.0
    assert got.bound == pytest.approx(1.0)
    assert got.labels["milp/bound_provenance"] == "root-ns"


def test_merge_leaves_boundless_statuses_alone():
    sf = _sf()
    out = lp_milp_highs.HighsOutcome("error", message="x")
    got = _merge(sf, out, lp_milp_highs.HighsOutcome("node_limit", bound=1.0))
    assert got.status == "error" and got.bound is None


def _sequencing(seed: int, big_m: float) -> dm.Model:
    """The #1654 panel's job-sequencing family (6 jobs, disjunctive big-M pairs)."""
    import itertools

    rng = np.random.default_rng(seed)
    n = 6
    p = rng.integers(2, 10, n)
    r = rng.integers(0, 9, n)
    m = dm.Model(f"seq{seed}")
    s = [m.continuous(f"s{i}", lb=float(r[i]), ub=100) for i in range(n)]
    c = m.continuous("C", lb=0, ub=100)
    for i, j in itertools.combinations(range(n), 2):
        y = m.binary(f"y{i}_{j}")
        m.subject_to(s[i] + float(p[i]) <= s[j] + big_m * (1 - y))
        m.subject_to(s[j] + float(p[j]) <= s[i] + big_m * y)
    for i in range(n):
        m.subject_to(s[i] + float(p[i]) <= c)
    m.minimize(c)
    return m


def test_merge_never_adopts_a_tolerance_point_on_huge_big_m():
    """Panel witness: at M = 1e10 the raw presolve-free point verifies at 11.0 on the
    standard form while violating the model by 8 (the verifier's relative row
    tolerance). Adopted, it was published and outranked the #1654 rescue's certified
    31.0. Only the LP over its integers may be taken, and that gives 31.0."""
    m = _sequencing(0, 1e10)
    r = m.solve(time_limit=60)
    from discopt.warm_start import check_feasibility

    assert r.objective == pytest.approx(31.0, abs=1e-6), (r.status, r.objective)
    assert r.bound is None or r.bound <= 31.0 + 1e-6
    x = np.concatenate([np.atleast_1d(np.asarray(r.x[v.name], float)) for v in m._variables])
    ok, why = check_feasibility(m, x, tol=1e-5)
    assert ok, why


# ── The fix-and-propagate start ──


def test_fix_and_propagate_completes_a_relu_network():
    """Fixing the inputs lets propagation decide every neuron; the point verifies."""
    m = _relu_model(width=32)
    _, n_orig, sf = _highs_std_form(m)
    x = lp_milp_highs.fix_and_propagate_point(sf, n_orig, None)
    assert x is not None
    stats: dict = {}
    pt = lp_milp_highs._fixprop_start(sf, n_orig, None, stats)
    assert pt is not None and stats["milp/fixprop_found"] == 1.0
    assert lp_milp_highs._verified_mip_point(sf, pt[0]) is not None


def test_fix_and_propagate_gives_up_on_a_dead_end():
    """``2x == 1``, ``x`` integer: propagation empties the domain; no point, no raise."""
    sf = lp_milp_highs.StdForm.from_arrays(
        c=np.array([1.0]),
        A=np.array([[2.0]]),
        b=np.array([1.0]),
        xl=np.array([0.0]),
        xu=np.array([10.0]),
        int_idx=np.array([0]),
    )
    assert lp_milp_highs.fix_and_propagate_point(sf, 1, None) is None


def test_start_is_published_when_highs_ends_without_an_incumbent(monkeypatch):
    """HiGHS's limit exit with no point of its own: the verified start is the incumbent."""
    m = _relu_model(width=16)
    _, n_orig, sf = _highs_std_form(m)

    def starved(sf_, kw):
        assert kw["initial_point"] is not None, "the start never reached the primary"
        return lp_milp_highs.HighsOutcome("time_limit", bound=-50.0)

    monkeypatch.setattr(lp_milp_highs, "_certified_milp", starved)
    out = lp_milp_highs.solve_milp_std(
        sf, time_limit=60.0, gap_tolerance=1e-4, max_nodes=10**6, n_struct=n_orig
    )
    assert out.status == "feasible" and out.x is not None
    assert out.stats["milp/fixprop_incumbent_published"] == 1.0
    assert lp_milp_highs._verified_mip_point(sf, out.x) is not None


def test_start_opt_out(monkeypatch):
    monkeypatch.setenv("DISCOPT_MILP_FIXPROP_SEED", "0")
    r = _relu_model(width=8).solve()
    assert "milp/fixprop_ran" not in r.solver_stats


def _fixed_input_pwl():
    """``nonlinear_to_pwl`` of ``sin(3x) + x**2`` with ``x`` fixed at -1.5: HiGHS's
    presolve, handed a start, stops ``kOptimal`` at 0 nodes with NO dual bound."""
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(dm.sin(3 * x) + x**2)
    tm = dm.nonlinear_to_pwl(m, segments=5).model
    xv = next(v for v in tm._variables if v.name == "x")
    wv = next(v for v in tm._variables if v.name.endswith("_w"))
    xv.lb = np.float64(-1.5)
    xv.ub = np.float64(-1.5)
    tm.minimize(wv)
    return tm


def test_a_seeded_optimum_without_a_bound_is_re_solved_cold():
    tm = _fixed_input_pwl()
    _, n_orig, sf = _highs_std_form(tm)
    start = lp_milp_highs._fixprop_start(sf, n_orig, None, {})
    assert start is not None
    out = lp_milp_highs._solve_milp_std(
        sf, time_limit=30.0, gap_tolerance=1e-4, max_nodes=10**6, n_struct=n_orig,
        initial_point=start[0][:n_orig], root_check=False,
    )  # fmt: skip
    assert out.stats.get("milp/seeded_optimal_without_bound") == 1.0
    # Never a certificate with nothing behind it.
    assert not (out.gap_certified and out.status == "optimal" and out.bound is None)


def test_the_default_route_keeps_a_bound_on_the_fixed_input_pwl():
    r = _fixed_input_pwl().solve(time_limit=30)
    g = float(np.sin(3 * -1.5) + 2.25)
    assert r.x is not None and r.bound is not None, r.status
    assert r.bound <= g + 1e-7
