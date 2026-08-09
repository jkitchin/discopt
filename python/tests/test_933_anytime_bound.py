"""Regression tests for #933: anytime dual bound — root seeding + one report.

#933 diagnosed that 27% of a 200-instance MINLPLib sweep reported NO dual bound
at ``time_limit=8`` because (a) the root relaxation bound proved during setup was
never installed into the tree (``global_lower_bound`` stayed ``-inf`` until the
first fully-processed batch), and (b) the reported bound was re-derived
independently in five exit paths, four of which had no recovery when the tree
bound was unusable — including silently discarding a perfectly VALID finite tree
bound on every no-incumbent limit exit, by conflating "the exit is not
certified" with "the tree bound is invalid".

Covered here:

* ``_finalize_reported_bound`` — the shared exit chokepoint (taint rule,
  independent-bound fallback, ``1e20`` sentinel refusal, sense mapping,
  ``bound <= incumbent`` cap).
* ``PyTreeManager.seed_root_bound`` — the #933 part (a) seeding entry point
  (the Rust-side unit tests live in ``tree_manager.rs``).
* The MILP-BB exit: an uncertified (budget-limited) exit must report the tree's
  live frontier bound, not the stale root-LP fallback. Fails before #933: both
  runs below reported the identical root LP value and ``gap=None``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import PyTreeManager
from discopt.solver import _finalize_reported_bound

# ---------------------------------------------------------------------------
# _finalize_reported_bound (part (b) chokepoint)
# ---------------------------------------------------------------------------


def test_valid_tree_bound_is_reported_with_gap():
    bound, gap = _finalize_reported_bound(
        tree_bound_internal=5.0,
        tree_bound_valid=True,
        is_maximize=False,
        obj_val=10.0,
        independent_bounds_internal=(),
    )
    assert bound == 5.0
    assert gap == pytest.approx(0.5)


def test_valid_tree_bound_survives_missing_incumbent():
    # The #933 headline defect: a valid finite tree bound with NO incumbent
    # (a time/node-limited exit) must be reported, with gap None.
    bound, gap = _finalize_reported_bound(
        tree_bound_internal=5.0,
        tree_bound_valid=True,
        is_maximize=False,
        obj_val=None,
    )
    assert bound == 5.0
    assert gap is None


def test_tainted_tree_bound_is_discarded_wholesale():
    bound, gap = _finalize_reported_bound(
        tree_bound_internal=5.0,
        tree_bound_valid=False,
        is_maximize=False,
        obj_val=10.0,
    )
    assert bound is None and gap is None


def test_tainted_tree_bound_falls_back_to_independent_bound():
    # A tainted tree discards its own bound but the independently-proved root
    # bound (untainted snapshot / root LP) still surfaces — the tightest wins.
    bound, _ = _finalize_reported_bound(
        tree_bound_internal=5.0,
        tree_bound_valid=False,
        is_maximize=False,
        obj_val=None,
        independent_bounds_internal=(1.0, None, 2.5, float("-inf")),
    )
    assert bound == 2.5


def test_valid_tree_bound_composes_with_tighter_independent_bound():
    bound, _ = _finalize_reported_bound(
        tree_bound_internal=2.0,
        tree_bound_valid=True,
        is_maximize=False,
        obj_val=None,
        independent_bounds_internal=(3.0,),
    )
    assert bound == 3.0  # larger internal lower bound = tighter


def test_sentinel_magnitude_bounds_are_refused():
    # np.isfinite(1e20) is True — the #930 hole. Neither the tree bound nor a
    # candidate may surface at/beyond the effective-infinity sentinel.
    bound, _ = _finalize_reported_bound(
        tree_bound_internal=1e20,
        tree_bound_valid=True,
        is_maximize=False,
        obj_val=None,
        independent_bounds_internal=(-1e20,),
    )
    assert bound is None


def test_maximize_sense_mapping():
    # Internal min space tracks -obj: an internal lower bound of -7 is the user
    # upper bound 7, and must sit ABOVE the incumbent objective.
    bound, gap = _finalize_reported_bound(
        tree_bound_internal=-7.0,
        tree_bound_valid=True,
        is_maximize=True,
        obj_val=6.0,
        independent_bounds_internal=(),
    )
    assert bound == 7.0
    assert bound >= 6.0
    assert gap == pytest.approx(1.0 / 6.0)


def test_bound_never_crosses_the_incumbent():
    # Certificate invariant: bound <= incumbent (min sense). A candidate that
    # numerically exceeds the incumbent is capped at it, mirroring the Rust
    # tree's own cap; the gap then closes to exactly 0.
    bound, gap = _finalize_reported_bound(
        tree_bound_internal=None,
        tree_bound_valid=False,
        is_maximize=False,
        obj_val=10.0,
        independent_bounds_internal=(10.0 + 1e-9,),
    )
    assert bound == 10.0
    assert gap == 0.0


def test_nothing_to_report_returns_none_pair():
    bound, gap = _finalize_reported_bound(
        tree_bound_internal=float("-inf"),
        tree_bound_valid=True,
        is_maximize=False,
        obj_val=None,
    )
    assert bound is None and gap is None


# ---------------------------------------------------------------------------
# PyTreeManager.seed_root_bound (part (a) entry point)
# ---------------------------------------------------------------------------


def _fresh_tree() -> PyTreeManager:
    t = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
    t.initialize()
    return t


def test_seed_root_bound_makes_glb_finite_before_any_batch():
    t = _fresh_tree()
    assert t.stats()["global_lower_bound"] == float("-inf")
    t.seed_root_bound(-3.25)
    assert t.stats()["global_lower_bound"] == -3.25
    # Non-finite seeds are ignored; a looser seed never replaces a tighter one.
    t.seed_root_bound(float("-inf"))
    t.seed_root_bound(float("nan"))
    t.seed_root_bound(-9.0)
    assert t.stats()["global_lower_bound"] == -3.25


def test_seed_root_bound_floor_survives_a_failed_root_relaxation():
    # The root LP fails (raw -inf import). Unseeded, the tree bound would be
    # pinned at -inf; seeded, the children inherit the floored root bound and
    # the frontier minimum stays at the seed.
    t = _fresh_tree()
    t.seed_root_bound(1.0)
    _lb, _ub, ids, _psols = t.export_batch(1)
    t.import_results(
        np.asarray(ids, dtype=np.int64),
        np.array([-np.inf]),
        np.array([[0.5, 0.7]]),
        np.array([False]),
    )
    t.process_evaluated()
    stats = t.stats()
    assert stats["open_nodes"] >= 2, "root must have branched"
    assert stats["global_lower_bound"] == 1.0


# ---------------------------------------------------------------------------
# MILP-BB exit: uncertified exits keep the live tree bound (part (b) wiring)
# ---------------------------------------------------------------------------


def _knapsack(n: int = 14) -> dm.Model:
    m = dm.Model("kp933")
    rng = np.random.default_rng(0)
    c = rng.uniform(1, 10, n)
    w = rng.uniform(1, 10, n)
    xs = [m.binary(f"x{i}") for i in range(n)]
    m.subject_to(sum(float(w[i]) * xs[i] for i in range(n)) <= float(w.sum()) * 0.35)
    for i in range(0, n - 1, 2):
        m.subject_to(xs[i] + xs[i + 1] <= 1)
    m.maximize(sum(float(c[i]) * xs[i] for i in range(n)))
    return m


def test_milp_budget_limited_exit_reports_live_tree_bound():
    """Before #933 the MILP path dropped its (valid) tree bound on every
    uncertified exit and re-reported the STALE root LP value, so the reported
    bound never moved past the root no matter how far the search got — and the
    gap was reported as None next to a finite bound. After: the frontier bound
    is reported (strictly tighter once the tree has processed batches beyond
    the root), with the gap computed against it."""
    shallow = _knapsack().solve(time_limit=60, max_nodes=3)
    deep = _knapsack().solve(time_limit=60, max_nodes=9)

    for r in (shallow, deep):
        assert r.status == "feasible"
        assert r.objective is not None
        assert r.bound is not None and np.isfinite(r.bound)
        # MAXIMIZE: the dual bound is an upper bound, at/above the incumbent.
        assert r.bound >= r.objective - 1e-9
        assert r.gap is not None
        assert r.gap == pytest.approx(
            abs(r.objective - r.bound) / max(1.0, abs(r.objective)), abs=1e-12
        )

    # The deeper search's reported bound must reflect its lifted frontier, not
    # the shallow run's root-level value (pre-#933 both reported the identical
    # root LP bound: 42.4745...).
    assert deep.bound < shallow.bound - 1e-9, (
        f"reported bound pinned at the root value: shallow={shallow.bound} "
        f"deep={deep.bound} — the live tree bound is being discarded again"
    )
