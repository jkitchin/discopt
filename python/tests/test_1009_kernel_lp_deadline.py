"""#1009 / #1014 — the spatial kernel's node LP honors the tree's wall-clock deadline.

`solve_spatial_tree` checks its clock only BETWEEN nodes, so before #1014 the
wall-clock floor of any kernel solve was the cost of **one node LP**, however small
the caller's budget: `spatial_bindings.rs` built `SimplexOptions` with no `deadline`
and the pivot loops polled nothing. #1014's `node_lp_opts` composes the tree's live
deadline into the LP's options.

That fix shipped "sound by construction, not by measurement" — and the instance #1009
was opened for turned out to overrun for an unrelated reason (the Python incremental
McCormick fast path, fixed separately in #1015), so the kernel-side change has no
end-to-end coverage. `crates/discopt-core/src/bnb/spatial_tree.rs` unit-tests the
composition and the `IterLimit -> Undecided` verdict; nothing exercises an LP that
genuinely outlives its budget and is cut *mid-solve*, through the PyO3 boundary,
where the wall-clock behavior either materializes or does not.

These tests close that gap. They are calibrated against an **uncapped single-node
run** on the same build rather than against a reverted binary: that time is precisely
what the pre-#1014 path spent on node 1 regardless of the budget, so it is the floor
the fix removes. `scripts/spatial_kernel_lp_deadline_scaling.py` runs the same
comparison across sizes; at a fixed 1 s budget it measures a floor of 2.7x the budget
at 2 415 lifted terms, 7.7x at 4 005 and **25.9x** at 7 140, against 1.0-1.2x capped.

Soundness rides along on every assertion: an interrupted LP bails as `IterLimit`,
`solve_spatial_node` certifies a Neumaier-Shcherbina safe bound only on `Optimal`, so
the node contributes no bound and is branched rather than fathomed. A partial simplex
iterate must never be readable as a dual bound.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import pytest
from discopt._relax.spatial_producer import build_spatial_kernel_spec

_rust = pytest.importorskip("discopt._rust")

# Big enough that one node LP costs several times the budget below on ordinary CI
# hardware (4 005 lifted terms), small enough to keep the file under ~15 s.
_N = 90
_BUDGET = 1.0


def _dense_bilinear(n: int):
    """``min sum_{i<j} x_i x_j  s.t.  sum x >= n/2,  x in [0,1]^n``.

    ``n(n-1)/2`` lifted bilinear terms over a trivial box, so the *node LP* is large
    while the search is a single root node — which isolates the cost of one LP, the
    quantity the overshoot is bounded by. Deliberately a general model rather than a
    named instance (CLAUDE.md §2): the defect is a property of LP size, not of any
    corpus entry.

    Its global minimum is available in closed form, so the soundness assertions have a
    real oracle: ``sum_{i<j} x_i x_j = ((sum x)^2 - sum x^2) / 2``, and for
    ``s = sum x >= n/2`` that is minimized by maximizing ``sum x^2``, i.e. at
    ``s = n/2`` with ``n/2`` variables at 1 — giving ``C(n/2, 2)``.
    """
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    m.constraint(dm.RangeSet(1), lambda _k: sum(xs) >= n / 2.0, name="half", fast=False)
    terms = [xs[i] * xs[j] for i in range(n) for j in range(i + 1, n)]
    # Balanced reduction: a left-deep `sum()` chain exceeds the canonicalizer's
    # recursion limit long before the LP is big enough to matter here.
    while len(terms) > 1:
        terms = [
            terms[k] + terms[k + 1] if k + 1 < len(terms) else terms[k]
            for k in range(0, len(terms), 2)
        ]
    m.minimize(terms[0])
    return m


def _tiny_bilinear():
    """``min x*y  s.t.  x + y >= 3,  x,y in [0,2]`` — global minimum 2.0.

    Converges by *proof* in milliseconds, which is what the neutrality test needs: a
    solve that stops on the clock has a legitimately run-dependent bound and could not
    be compared exactly against anything.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.constraint(dm.RangeSet(1), lambda _k: x + y >= 3.0, name="sum", fast=False)
    m.minimize(x * y)
    return m


def _optimum(n: int) -> float:
    half = n // 2
    return half * (half - 1) / 2.0


def _spec(model):
    spec = build_spatial_kernel_spec(model)
    assert spec is not None, "the producer declined the model these tests are built on"
    for key in [k for k in spec if k.startswith("meta_")]:
        spec.pop(key)
    return spec


def _solve(spec, **kwargs):
    t0 = time.perf_counter()
    res = _rust.solve_spatial_tree_py(**spec, **kwargs)
    return res, time.perf_counter() - t0


def _one_node_lp_cost(spec):
    """Wall time of a single node LP run to completion — the floor the pre-#1014 path
    could not go below, measured on this machine and this build."""
    _, wall = _solve(spec, time_limit_s=None, max_nodes=1)
    return wall


@pytest.mark.slow
def test_a_node_lp_costlier_than_the_budget_does_not_set_the_wall_clock_floor():
    """The defect itself, end to end through the binding.

    Self-calibrating: the uncapped single-node run measures what one LP costs *here*,
    so the assertion is a ratio rather than a wall-clock threshold. If one LP is
    already cheaper than the budget on this machine there is no floor to remove, and
    the test skips instead of passing vacuously (CLAUDE.md §6).
    """
    spec = _spec(_dense_bilinear(_N))
    floor = _one_node_lp_cost(spec)
    if floor < 2.5 * _BUDGET:
        pytest.skip(
            f"one node LP costs {floor:.2f}s here, under 2.5x the {_BUDGET}s budget — "
            "no wall-clock floor for the deadline to remove"
        )

    res, wall = _solve(spec, time_limit_s=_BUDGET, max_nodes=10**9)

    assert wall < 0.5 * floor, (
        f"the budget did not bound the node LP: {wall:.2f}s against a {_BUDGET}s "
        f"budget, with one uninterrupted LP costing {floor:.2f}s"
    )
    assert res["status"] == "time_limit", f"expected a clock-limited exit, got {res['status']}"
    opt = _optimum(_N)
    assert res["bound"] <= opt + 1e-6 * (1.0 + abs(opt)), (
        f"bound {res['bound']} exceeds the true optimum {opt}"
    )


@pytest.mark.slow
def test_an_interrupted_node_lp_contributes_no_bound_and_no_fathom():
    """What a deadline-cut LP is allowed to say.

    A budget far below the cost of the root LP cuts it mid-solve. The kernel must then
    report ``time_limit`` with NO dual bound (``-inf``) and NO incumbent: an
    interrupted simplex iterate is not a certified dual value, and reading one as a
    bound is the shape of bug that yields a certified-``optimal`` false bound. The
    node must also be recorded as *undecided* — branched, never fathomed.

    The core's `an_interrupted_node_lp_is_undecided_never_fathomed` pins the same
    contract with a deadline that expired before pivot 0; this one cuts a real LP in
    flight, across the PyO3 boundary.
    """
    spec = _spec(_dense_bilinear(_N))
    res, wall = _solve(spec, time_limit_s=0.05, max_nodes=10**9)

    assert res["status"] == "time_limit"
    assert res["bound"] == float("-inf"), (
        f"a cut root LP produced a bound ({res['bound']}) it never certified"
    )
    assert res["incumbent"] is None, "a cut root LP invented an incumbent"
    assert res["n_undecided"] >= 1, (
        "the interrupted LP was not recorded as undecided — it was either never cut "
        "(the probe did not fire) or it was fathomed, which would be unsound"
    )
    assert wall < 1.0, f"a 0.05s budget took {wall:.2f}s"


def test_a_non_binding_budget_matches_the_uncapped_search():
    """A deadline with time to spare must not perturb the search.

    The per-LP cap fires only on an LP that outlives the tree's deadline, so on
    everything else a budgeted solve must agree exactly with an unbudgeted one —
    status, bound, incumbent, node count, LP count. This is the bound-neutral half of
    the CLAUDE.md §5 regime made observable at the binding.
    """
    spec = _spec(_tiny_bilinear())
    uncapped, _ = _solve(spec, time_limit_s=None, max_nodes=10**9)
    capped, _ = _solve(spec, time_limit_s=30.0, max_nodes=10**9)

    # Non-vacuity: a clock-limited solve has a legitimately run-dependent bound, so the
    # exact comparison is only meaningful once the search terminates by proof.
    assert uncapped["status"] == "optimal", (
        f"the control instance did not solve: {uncapped['status']}"
    )
    for key in ("status", "bound", "incumbent", "node_count", "n_lp_solves", "n_undecided"):
        assert uncapped[key] == capped[key], (
            f"{key} drifted under a non-binding budget: {uncapped[key]!r} != {capped[key]!r}"
        )
    assert uncapped["bound"] <= 2.0 + 1e-6, f"bound {uncapped['bound']} above the optimum 2.0"
    assert capped["incumbent"] is not None and abs(capped["incumbent"] - 2.0) < 1e-3
