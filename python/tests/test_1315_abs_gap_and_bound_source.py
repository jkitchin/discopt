"""#1315: two broken public contracts found against #1260.

1. ``abs_gap_tolerance`` was a **no-op** for every MILP that reached the
   monolithic Rust simplex B&B engine -- the public ``nlp_solver="simplex"``
   option always, and the default engine whenever ``DISCOPT_LP_MILP_BACKEND=rust``
   opts out of the HiGHS route. ``_solve_milp_simplex`` had no ``abs_gap_tol``
   parameter at all, and the one reroute guard that existed only covered the
   *tighter*-than-relative case. CLAUDE.md §3 forbids dead flags; the documented
   contract is "the search stops when EITHER the relative or the absolute
   criterion holds", unconditionally.
2. ``bound_source`` stayed ``None`` on the general convex-NLP fast path
   (``_solve_continuous``), contradicting the ``SolveResult.bound_valid``
   docstring's unconditional claim that ``convex_fast_path=True`` reports
   ``bound = objective`` with ``bound_source="convex_proof"``. Its two siblings
   (``_solve_lp_matrix``, ``_solve_qp_matrix``) were updated by #1260; the most
   general of the three -- every non-LP/QP convex NLP -- was missed.

The knapsack below is the issue's own repro shape: a root LP gap in the
thousands, so an absolute tolerance an order of magnitude above it is satisfied
at the root if -- and only if -- it is actually honored.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

_N = 100


def _knapsack() -> dm.Model:
    """A 100-item knapsack whose root LP gap is ~4e3, rebuilt fresh per solve."""
    rng = np.random.default_rng(0)
    values = rng.integers(10, 1000, size=_N).astype(float)
    weights = rng.integers(10, 1000, size=_N).astype(float)
    capacity = float(weights.sum() * 0.5)

    m = dm.Model("i1315_knapsack")
    xs = [m.binary(f"x{i}") for i in range(_N)]
    m.maximize(sum(float(values[i]) * xs[i] for i in range(_N)))
    m.subject_to(sum(float(weights[i]) * xs[i] for i in range(_N)) <= capacity)
    return m


# ── part 1: abs_gap_tolerance on the Rust MILP engine ───────────────────


@pytest.mark.smoke
def test_abs_gap_tolerance_stops_the_rust_milp_engine_early():
    """A loose absolute tolerance must end the search, not be dropped on the floor."""
    baseline = _knapsack().solve(nlp_solver="simplex", time_limit=120)
    assert baseline.status == "optimal"
    assert baseline.bound is not None

    # 10x the root gap: satisfiable immediately if the criterion is applied.
    loose = _knapsack().solve(nlp_solver="simplex", time_limit=120, abs_gap_tolerance=10.0 * 4074.0)
    assert loose.status == "optimal"
    assert loose.objective is not None and loose.bound is not None

    assert loose.node_count < baseline.node_count, (
        "abs_gap_tolerance did not shorten the search: "
        f"{loose.node_count} nodes vs {baseline.node_count} baseline — the "
        "parameter is still inert on this route"
    )
    # And the stop was legitimate: the absolute gap it stopped on really is
    # within what was asked for.
    assert abs(loose.bound - loose.objective) <= 10.0 * 4074.0


@pytest.mark.smoke
def test_abs_gap_tolerance_is_honored_on_the_default_engine_route(monkeypatch):
    """The other reachable entry: the default engine under DISCOPT_LP_MILP_BACKEND=rust."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")

    baseline = _knapsack().solve(time_limit=120)
    loose = _knapsack().solve(time_limit=120, abs_gap_tolerance=10.0 * 4074.0)

    assert baseline.status == "optimal"
    assert loose.status == "optimal"
    assert loose.node_count < baseline.node_count, (
        f"{loose.node_count} nodes vs {baseline.node_count} baseline on the default route"
    )


@pytest.mark.smoke
def test_omitting_abs_gap_tolerance_changes_nothing():
    """``None`` keeps the route's established default -- for this engine, no absolute arm.

    Two solves of the same model with the argument omitted and passed as ``None``
    must be identical in node count and objective: the fix encodes "absent" as
    ``MilpOptions::abs_gap_tol = None``, so a default solve takes the same code
    path it always did (CLAUDE.md §5, bound-neutral).
    """
    omitted = _knapsack().solve(nlp_solver="simplex", time_limit=120)
    explicit_none = _knapsack().solve(nlp_solver="simplex", time_limit=120, abs_gap_tolerance=None)

    assert omitted.node_count == explicit_none.node_count
    assert omitted.objective == explicit_none.objective
    assert omitted.bound == explicit_none.bound
    # An unqualified solve still closes the gap exactly.
    assert omitted.bound == pytest.approx(omitted.objective, rel=1e-6)


@pytest.mark.smoke
def test_a_tolerance_tighter_than_the_relative_one_still_reroutes():
    """The #1243 reroute (away from this engine, to the Python tree) must survive.

    The criterion is a disjunction, so the relative arm can still stop the search
    first -- what the tighter absolute request buys is the Python tree's
    *unfloored* relative gap, which is the regime #1243 routes here for. The
    assertion is therefore that the result is certified and its gap meets the
    relative tolerance, not that the gap is exactly zero.
    """
    tight = _knapsack().solve(nlp_solver="simplex", time_limit=120, abs_gap_tolerance=1e-9)
    assert tight.status == "optimal"
    assert tight.objective is not None and tight.bound is not None
    rel_gap = abs(tight.bound - tight.objective) / max(
        abs(tight.objective), abs(tight.bound), 1e-10
    )
    assert rel_gap <= 1e-4


# ── part 2: bound_source on the general convex-NLP fast path ────────────


def _convex_models():
    """Convex NLPs that route through ``_solve_continuous``, not the LP/QP matrix paths."""
    binary_entropy = dm.Model("i1315_entropy")
    y = binary_entropy.continuous("y", lb=0.0, ub=1.0)
    binary_entropy.minimize(dm.xlogx(y) + dm.xlogx(1 - y))

    exponential = dm.Model("i1315_exp")
    z = exponential.continuous("z", lb=-5.0, ub=5.0)
    exponential.minimize(dm.exp(z) - z)

    return [("xlogx", binary_entropy), ("exp", exponential)]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "label,model", _convex_models(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_convex_fast_path_names_its_bound_provenance(label, model):
    """``convex_fast_path=True`` must carry ``bound_source="convex_proof"``, as documented."""
    r = model.solve()

    assert r.status == "optimal"
    assert r.convex_fast_path is True
    assert r.bound is not None
    assert r.bound_valid is True
    assert r.bound_source == "convex_proof", (
        f"{label}: convex fast path reported bound_source={r.bound_source!r}; the "
        "SolveResult.bound_valid docstring promises 'convex_proof' for every "
        "convex_fast_path=True result"
    )


@pytest.mark.smoke
def test_the_documented_value_is_the_one_the_siblings_use():
    """A QP goes through ``_solve_qp_matrix``; the two routes must not disagree."""
    qp = dm.Model("i1315_qp")
    x = qp.continuous("x", lb=-10.0, ub=10.0)
    qp.minimize((x - 2.0) ** 2)
    r = qp.solve()

    assert r.convex_fast_path is True
    assert r.bound_source == "convex_proof"
