"""#1920: a withdrawn HiGHS MILP certificate fell back to the root LP bound.

When HiGHS proves a MILP optimal late in the budget, the #1634 presolve-free cross-solve
gets what is left, reaches no verdict, and the certificate is withdrawn. ``_withdraw``
then published the NS-safe ROOT bound, discarding both tree bounds, so a longer run
returned a looser bound than a shorter one (issue: -15.30 at 600 s against -4.38 at
180 s on a 2-64-64-1 ``relu_bigm`` net). The published bound is now
``max(root, min(primary tree bound, cross tree bound))`` -- the ``min`` rule
``_weaker_bound`` already certifies on.

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
