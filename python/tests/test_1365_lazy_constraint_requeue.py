"""Issue #1365: a lazy-constraint callback that returns a cut lost the solution.

Found re-running ``docs/notebooks/callbacks.ipynb`` for #1362. The notebook's TSP
section is the documented showcase for ``Model.solve(lazy_constraints=...)`` and
its committed output looked healthy -- but on 4 cities with degree-2 constraints
a subtour is structurally impossible, so the callback returned ``[]`` every time
and the feature was never exercised. On any instance where it *did* return a
``CutResult``, the solve came back ``status="unknown"``, ``objective=None``.

Cause: ``_invoke_pre_import_callbacks`` marked the node with
``_INFEASIBILITY_SENTINEL``. That is the #748 non-rigorous fathom -- it removes a
box that was never proven free of acceptable points -- and the removed box is
exactly where the corrected answer lives. A lazy cut says "this *point* is not
acceptable", never "this *box* is empty".

Fix: the node is returned to the open frontier (``TreeManager::requeue_node``,
already written for the single-tree OA path in #1060, now bound into Python) and
re-solved against the cut-augmented relaxation, which is what every MIP solver
does with a lazy cut.
"""

from __future__ import annotations

import discopt
import numpy as np
import pytest
from discopt.callbacks import CutResult


def _two_binaries():
    m = discopt.Model("two_binaries")
    x = [m.binary(f"x{i}") for i in range(2)]
    m.maximize(x[0] + x[1])
    return m, x


def test_control_the_cut_as_a_static_constraint():
    """Bracket the repro: the cut itself is satisfiable and gives 1.0."""
    m, x = _two_binaries()
    m.subject_to(x[0] + x[1] <= 1)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0)


def test_control_a_callback_that_returns_no_cut():
    """The callback plumbing alone must not disturb the solve."""
    m, _ = _two_binaries()
    calls = []
    r = m.solve(lazy_constraints=lambda ctx, model: calls.append(1) or [], time_limit=30)
    assert calls, "the lazy callback was never invoked"
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0)


@pytest.mark.parametrize("kwargs", [{}, {"solver": "bb"}])
def test_a_returned_cut_still_yields_the_correct_optimum(kwargs):
    m, x = _two_binaries()
    stats = {"calls": 0, "cuts": 0}

    def cb(ctx, model):
        stats["calls"] += 1
        if ctx.x_relaxation[0] > 0.5 and ctx.x_relaxation[1] > 0.5:
            stats["cuts"] += 1
            return [CutResult(terms=[(x[0], 1.0), (x[1], 1.0)], sense="<=", rhs=1.0)]
        return []

    r = m.solve(lazy_constraints=cb, time_limit=30, **kwargs)

    assert stats["cuts"] > 0, "the probe did not fire: no cut was ever returned"
    assert r.status == "optimal", f"status={r.status} obj={r.objective}"
    assert r.objective == pytest.approx(1.0)
    assert r.x is not None
    assert int(r.x["x0"]) + int(r.x["x1"]) == 1


@pytest.mark.slow
def test_subtour_elimination_on_an_instance_that_has_a_subtour():
    """Six cities as two tight triangles -- the notebook's example, made real."""
    n = 6
    dist = np.full((n, n), 5.0)
    np.fill_diagonal(dist, 0.0)
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        dist[a, b] = dist[b, a] = 1.0

    m = discopt.Model("tsp6")
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, m.binary(f"x_{i}_{j}")))
    m.minimize(sum(dist[i, j] * v for i, j, v in edges))
    for c in range(n):
        m.subject_to(sum(v for i, j, v in edges if c in (i, j)) == 2, name=f"deg{c}")

    def components(sol):
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for k, (i, j, _) in enumerate(edges):
            if sol[k] > 0.5:
                adj[i].append(j)
                adj[j].append(i)
        seen: set[int] = set()
        out = []
        for s in range(n):
            if s in seen:
                continue
            comp: set[int] = set()
            stack = [s]
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                stack.extend(b for b in adj[node] if b not in comp)
            seen |= comp
            out.append(comp)
        return out

    # Without the callback the "tour" is the two triangles -- a genuine subtour.
    plain = m.solve(time_limit=60)
    flat = np.array([plain.x[f"x_{i}_{j}"] for i, j, _ in edges]).ravel()
    assert len(components(flat)) == 2, "precondition: this instance must have a subtour"

    stats = {"cuts": 0}

    def sec(ctx, model):
        comps = components(ctx.x_relaxation)
        if len(comps) <= 1:
            return []
        cuts = []
        for comp in comps:
            if len(comp) == n:
                continue
            terms = [(v, 1.0) for i, j, v in edges if i in comp and j in comp]
            if terms:
                cuts.append(CutResult(terms=terms, sense="<=", rhs=float(len(comp) - 1)))
        stats["cuts"] += len(cuts)
        return cuts

    r = m.solve(lazy_constraints=sec, time_limit=120)
    assert stats["cuts"] > 0, "the probe did not fire: no subtour cut was returned"
    assert r.status == "optimal", f"status={r.status} obj={r.objective}"
    # A Hamiltonian cycle must cross between the triangles twice: four cost-1
    # triangle edges plus two cost-5 crossings.
    assert r.objective == pytest.approx(14.0, abs=1e-6)


def test_a_non_separating_cut_terminates_instead_of_requeueing_forever():
    """The loop guard: a cut the point already satisfies cannot make progress."""
    m, x = _two_binaries()
    calls = {"n": 0}

    def cb(ctx, model):
        # Always returns a cut, and one the incumbent candidate satisfies, so the
        # re-solve can return the same point indefinitely.
        calls["n"] += 1
        return [CutResult(terms=[(x[0], 1.0), (x[1], 1.0)], sense="<=", rhs=99.0)]

    r = m.solve(lazy_constraints=cb, time_limit=60)
    assert calls["n"] > 0
    # It must TERMINATE. Which terminal state is reached is the pre-#1365
    # exclusion behaviour; hanging is the failure this guards against.
    assert r.status in {"optimal", "feasible", "unknown", "time_limit"}


def test_incumbent_callback_rejection_is_unchanged():
    """A bare veto changes no relaxation, so it stays a non-rigorous fathom."""
    m = discopt.Model("filter")
    x = m.binary("x", shape=(4,))
    profits = np.array([10.0, 6.0, 4.0, 2.0])
    m.maximize(sum(profits[i] * x[i] for i in range(4)))
    m.subject_to(sum(3 * x[i] for i in range(4)) <= 7, name="capacity")

    seen = []

    def reject_everything(ctx, model, solution):
        seen.append(int(np.sum(solution["x"] > 0.5)))
        return False

    r = m.solve(incumbent_callback=reject_everything, time_limit=30)
    assert seen, "the incumbent callback was never invoked"
    assert r.objective is None, "no solution may survive an always-reject rule"
