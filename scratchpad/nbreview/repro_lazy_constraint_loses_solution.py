#!/usr/bin/env python3
"""Repro: a lazy-constraint callback that returns a cut loses the solution.

Found while re-running ``docs/notebooks/callbacks.ipynb`` for issue #1362.

The notebook's TSP section is the documented showcase for
``Model.solve(lazy_constraints=...)``. Its committed output looks healthy
(``optimal``, tour cost 80.0) -- but only because on a 4-city undirected TSP
with degree-2 constraints **no subtour is possible**, so the callback returned
an empty list every time and the feature was never exercised.

On any instance where the callback actually returns a ``CutResult``, the solve
comes back ``status="unknown"``, ``objective=None``, ``x=None``.

Run: python scratchpad/nbreview/repro_lazy_constraint_loses_solution.py
Exits non-zero while the defect is present.
"""

from __future__ import annotations

import numpy as np

import discopt
from discopt.callbacks import CutResult

CHECKS = 0
FAILURES: list[str] = []


def record(name: str, ok: bool, detail: str) -> None:
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok ' if ok else 'BAD'}] {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def two_binaries():
    m = discopt.Model("two_binaries")
    x = [m.binary(f"x{i}") for i in range(2)]
    m.maximize(x[0] + x[1])
    return m, x


def case_controls() -> None:
    """Both controls must pass, or the repro proves nothing."""
    print("\n1. Controls --- max x0+x1 over {0,1}^2")

    m, x = two_binaries()
    m.subject_to(x[0] + x[1] <= 1)
    r = m.solve(time_limit=30)
    record(
        "the cut as a STATIC constraint",
        r.status == "optimal" and abs(r.objective - 1.0) < 1e-9,
        f"status={r.status} obj={r.objective}",
    )

    m, _ = two_binaries()
    r = m.solve(lazy_constraints=lambda ctx, model: [], time_limit=30)
    record(
        "a lazy callback that returns NO cut",
        r.status == "optimal" and abs(r.objective - 2.0) < 1e-9,
        f"status={r.status} obj={r.objective}",
    )


def case_defect() -> None:
    """The same cut, returned lazily, loses the solution entirely."""
    print("\n2. The same cut returned by the lazy callback (expected: optimal 1.0)")
    for label, kwargs in (("default", {}), ("solver='bb'", {"solver": "bb"})):
        m, x = two_binaries()
        stats = {"calls": 0, "cuts": 0}

        def cb(ctx, model, _x=x, _stats=stats):
            _stats["calls"] += 1
            if ctx.x_relaxation[0] > 0.5 and ctx.x_relaxation[1] > 0.5:
                _stats["cuts"] += 1
                return [
                    CutResult(terms=[(_x[0], 1.0), (_x[1], 1.0)], sense="<=", rhs=1.0)
                ]
            return []

        r = m.solve(lazy_constraints=cb, time_limit=30, **kwargs)
        print(f"     {label}: calls={stats['calls']} cuts_returned={stats['cuts']}")
        record(
            f"lazy cut, {label}",
            r.status == "optimal" and r.objective is not None,
            f"status={r.status} obj={r.objective} x={r.x}",
        )


def case_tsp() -> None:
    """The notebook's own example, on an instance that HAS a subtour."""
    print("\n3. A 6-city TSP with two tight triangles (the notebook's SEC example)")
    n = 6
    dist = np.full((n, n), 5.0)
    np.fill_diagonal(dist, 0.0)
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        dist[a, b] = dist[b, a] = 1.0

    m = discopt.Model("tsp")
    edge_vars = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_vars.append((i, j, m.binary(f"x_{i}_{j}")))
    m.minimize(sum(dist[i, j] * v for i, j, v in edge_vars))
    for c in range(n):
        m.subject_to(sum(v for i, j, v in edge_vars if c in (i, j)) == 2, name=f"d{c}")

    def components(sol):
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for k, (i, j, _) in enumerate(edge_vars):
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

    # Without the callback the "tour" is two disjoint triangles -- a real subtour.
    r0 = m.solve(time_limit=60)
    comps0 = components(np.array([r0.x[f"x_{i}_{j}"] for i, j, _ in edge_vars]).ravel())
    print(f"     no callback: obj={r0.objective} components={[sorted(c) for c in comps0]}")

    stats = {"calls": 0, "cuts": 0}

    def cb(ctx, model):
        stats["calls"] += 1
        comps = components(ctx.x_relaxation)
        if len(comps) <= 1:
            return []
        cuts = []
        for comp in comps:
            if len(comp) == n:
                continue
            terms = [(v, 1.0) for i, j, v in edge_vars if i in comp and j in comp]
            if terms:
                cuts.append(
                    CutResult(terms=terms, sense="<=", rhs=float(len(comp) - 1))
                )
        stats["cuts"] += len(cuts)
        return cuts

    r = m.solve(lazy_constraints=cb, time_limit=60)
    print(f"     with SECs: calls={stats['calls']} cuts_returned={stats['cuts']}")
    record(
        "TSP with real subtour elimination",
        r.status == "optimal" and r.objective is not None,
        f"status={r.status} obj={r.objective}",
    )


def main() -> int:
    case_controls()
    case_defect()
    case_tsp()
    print(f"\n[executed {CHECKS} assertion(s)]")
    if CHECKS == 0:
        print("PROBE DID NOT FIRE: zero assertions executed")
        return 2
    if FAILURES:
        print(f"DEFECT PRESENT: {len(FAILURES)} failure(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("No failures: the defect appears fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
