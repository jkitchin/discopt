"""Seeded convex-quadratic GDP instances for the ``DISCOPT_HULL_CONIC_QUADRATIC`` panel (#1917).

The flag hulls a *separable convex quadratic* disjunct row -- a ball or an
axis-aligned ellipsoid, every variable squared, no cross term -- as a smoothed
cone instead of the eps-perspective. GDPlib has one instance of that class that
lowers (``pyomo.constrained_layout_clay0203``; ``syngas`` also fires, neutrally),
so the flag's audit row asked for "a wider panel with more convex-quadratic GDP
models". This module generates that panel as a *class*: three families from the
convex-GDP literature, each a seeded random draw at several sizes. No instance
is tuned by hand, and every instance is built with a constructive witness, so it
is feasible by construction.

Families (names are ``cgdp.<family>_<size>_s<seed>``; the size string is parsed,
so any size can be requested, not only the panel's):

``clay_r<R>c<C>``
    Constrained layout {cite}`Sawaya2005`: ``R`` rectangles each placed in one of
    ``C`` disjoint circles (all four corners inside the chosen circle: four ball
    rows per disjunct), pairwise non-overlap disjunctions, l1 connection cost. The
    model is Pyomo's own ``build_constrained_layout_model`` (vendored, see
    :mod:`benchmarks.pyomo_gdp_examples`) with seeded parameters; the circles are
    sized so a round-robin assignment of rectangles laid side by side fits.
``cover_n<N>k<K>``
    Coverage clustering (the disjunctive clustering class of
    {cite}`Kronqvist2021`, with ball rows on the centres): ``N`` points in the
    plane, ``K`` movable cluster centres. Each point is assigned to one centre and
    must lie within the radius ``rho`` of it (``||c_k - p_i||^2 <= rho^2``, a ball
    in the centre's coordinates); the objective is an assignment cost plus the l1
    distance each centre moves from its home. The points are drawn within
    ``0.8 rho`` of a home, so ``c = home`` is feasible. (The k-means form, ``d_i >=
    ||c_k - p_i||^2``, has an epigraph variable with no square term and is outside
    the flag's class, so it would not measure the flag.)
``route_t<T>k<K>d<D>``
    Waypoints through ellipsoidal regions (the convex-region route class, as in
    Pyomo's ``circles`` example of {cite}`Lee2000`, which picks one point in one
    of several balls): ``T+1`` waypoints in ``R^D`` from a fixed start to a fixed
    goal, each in one of ``K`` axis-aligned ellipsoids, minimizing the l1 path
    length plus a per-region cost. The ellipsoids are a random overlapping chain
    from start to goal plus distractors; ``T + 1`` is at least the chain length,
    so walking the chain centres is feasible.

All bounds are finite (the hull needs them) and every disjunct row is in the
flag's class. Random state is ``numpy.random.default_rng(seed)`` and nothing
else, so a name always builds the same model.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

PREFIX = "cgdp."

#: The panel corpus: three families x a size ladder x seeds. The sizes were fixed
#: from a one-seed (``s1``) OFF-arm calibration at 120 s (2026-11-01, load1 30-110,
#: so walls are inflated): clay r3c2 solved 37 s, r3c3 feasible at the limit (gap
#: open), r4c2 / r4c3 / r5c2 far from closed; cover n8k2 solved 14 s, n10k3 /
#: n14k3 / n20k3 no incumbent; route t4k5d2 22 s, t6k6d2 39 s, t5k6d3 38 s,
#: t8k8d2 119 s. The ladder keeps the sizes OFF solves or nearly solves; every
#: instance of it is in the panel, none is dropped on its result.
PANEL: tuple[str, ...] = tuple(
    [f"{PREFIX}clay_r{r}c{c}_s{s}" for (r, c) in ((3, 2), (3, 3)) for s in (1, 2, 3)]
    + [f"{PREFIX}cover_n{n}k{k}_s{s}" for (n, k) in ((6, 2), (8, 2), (8, 3)) for s in (1, 2)]
    + [
        f"{PREFIX}route_t{t}k{k}d{d}_s{s}"
        for (t, k, d) in ((4, 5, 2), (6, 6, 2), (5, 6, 3))
        for s in (1, 2)
    ]
)

_NAME = re.compile(r"^(clay|cover|route)_([a-z0-9]+)_s(\d+)$")


def _ints(spec: str, keys: str) -> dict[str, int]:
    pattern = "".join(f"{k}(\\d+)" for k in keys)
    m = re.fullmatch(pattern, spec)
    if m is None:
        raise ValueError(f"size {spec!r} does not match {pattern!r}")
    return {k: int(v) for k, v in zip(keys, m.groups(), strict=True)}


def parse(name: str) -> tuple[str, dict[str, int], int]:
    """``cgdp.clay_r4c2_s1`` -> ``("clay", {"r": 4, "c": 2}, 1)``."""
    key = name[len(PREFIX) :] if name.startswith(PREFIX) else name
    m = _NAME.match(key)
    if m is None:
        raise ValueError(f"not a generated convex-GDP name: {name!r}")
    family, size, seed = m.group(1), m.group(2), int(m.group(3))
    keys = {"clay": "rc", "cover": "nk", "route": "tkd"}[family]
    return family, _ints(size, keys), seed


# ---------------------------------------------------------------------------
# (a) constrained layout
# ---------------------------------------------------------------------------


def clay_params(n_rect: int, n_circ: int, seed: int) -> dict[str, Any]:
    """Seeded parameters in the shape ``build_constrained_layout_model`` takes."""
    import numpy as np

    rng = np.random.default_rng(seed)
    lengths = rng.integers(2, 9, size=n_rect)
    heights = rng.integers(2, 9, size=n_rect)
    radii = []
    for c in range(n_circ):
        group = list(range(c, n_rect, n_circ))  # round-robin witness assignment
        if group:
            w = float(sum(lengths[i] for i in group))
            h = float(max(heights[i] for i in group))
            need = math.hypot(w / 2.0, h / 2.0)
        else:
            need = 3.0
        radii.append(round(need * float(rng.uniform(1.02, 1.25)), 2))
    centers: list[tuple[float, float]] = []
    while len(centers) < n_circ:  # disjoint circles in a 100 x 100 field
        i = len(centers)
        x, y = (round(float(v), 1) for v in rng.uniform(radii[i], 100.0 - radii[i], size=2))
        if all(
            math.hypot(x - cx, y - cy) > radii[i] + radii[j] + 1.0
            for j, (cx, cy) in enumerate(centers)
        ):
            centers.append((x, y))
    pen = rng.integers(10, 301, size=(n_rect, n_rect))
    return {
        "rect_lengths": {i + 1: int(lengths[i]) for i in range(n_rect)},
        "rect_heights": {i + 1: int(heights[i]) for i in range(n_rect)},
        "circ_xvals": {c + 1: centers[c][0] for c in range(n_circ)},
        "circ_yvals": {c + 1: centers[c][1] for c in range(n_circ)},
        "circ_rvals": {c + 1: radii[c] for c in range(n_circ)},
        "sep_penalty_matrix": [
            [int(pen[i, j]) if j > i else 0 for j in range(n_rect)] for i in range(n_rect - 1)
        ],
    }


def build_clay(r: int, c: int, seed: int):
    from benchmarks import pyomo_gdp_examples as px

    mod = px._module("constrained_layout/cons_layout_model.py")
    return mod.build_constrained_layout_model(params=clay_params(r, c, seed))


# ---------------------------------------------------------------------------
# (b) coverage clustering
# ---------------------------------------------------------------------------


def build_cover(n: int, k: int, seed: int):
    import numpy as np
    import pyomo.environ as pyo
    from pyomo.gdp import Disjunction

    rng = np.random.default_rng(seed)
    rho = 2.0
    homes = rng.uniform(2.0, 8.0, size=(k, 2))
    owner = rng.integers(0, k, size=n)
    ang = rng.uniform(0.0, 2.0 * math.pi, size=n)
    rad = 0.8 * rho * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    pts = homes[owner] + np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    cost = rng.integers(1, 11, size=(n, k)).astype(float)
    move_w = rng.uniform(1.0, 5.0, size=k)

    m = pyo.ConcreteModel(name=f"cover_n{n}k{k}_s{seed}")
    m.P, m.K, m.J = range(n), range(k), range(2)
    lo, hi = float(pts.min()) - rho, float(pts.max()) + rho
    m.c = pyo.Var(m.K, m.J, bounds=(lo, hi), initialize=lambda _, kk, j: float(homes[kk, j]))
    m.move = pyo.Var(m.K, m.J, bounds=(0.0, hi - lo))
    m.acost = pyo.Var(m.P, bounds=(0.0, float(cost.max())))
    m.mv1 = pyo.Constraint(
        m.K, m.J, rule=lambda m, kk, j: m.move[kk, j] >= m.c[kk, j] - homes[kk, j]
    )
    m.mv2 = pyo.Constraint(
        m.K, m.J, rule=lambda m, kk, j: m.move[kk, j] >= homes[kk, j] - m.c[kk, j]
    )

    def assign(m, i):
        return [
            [
                sum((m.c[kk, j] - float(pts[i, j])) ** 2 for j in m.J) <= rho**2,
                m.acost[i] == float(cost[i, kk]),
            ]
            for kk in m.K
        ]

    m.assign = Disjunction(m.P, rule=assign)
    m.obj = pyo.Objective(
        expr=sum(m.acost[i] for i in m.P)
        + sum(float(move_w[kk]) * m.move[kk, j] for kk in m.K for j in m.J)
    )
    return m


# ---------------------------------------------------------------------------
# (c) waypoints through ellipsoids
# ---------------------------------------------------------------------------


def build_route(t: int, k: int, d: int, seed: int):
    import numpy as np
    import pyomo.environ as pyo
    from pyomo.gdp import Disjunction

    rng = np.random.default_rng(seed)
    chain = max(2, min(k - 1, t + 1))  # chain ellipsoids, walkable in t+1 waypoints
    axes = rng.uniform(1.0, 3.0, size=(k, d))
    centers = np.zeros((k, d))
    for e in range(1, chain):
        step = rng.normal(size=d)
        step /= np.linalg.norm(step)
        # overlap: consecutive centres closer than the smaller of their shortest axes
        reach = 0.9 * min(axes[e - 1].min(), axes[e].min())
        centers[e] = centers[e - 1] + reach * step
    span_lo, span_hi = centers[:chain].min(axis=0) - 3.0, centers[:chain].max(axis=0) + 3.0
    for e in range(chain, k):  # distractors anywhere in the chain's box
        centers[e] = rng.uniform(span_lo, span_hi)
    region_cost = rng.uniform(0.0, 2.0, size=k)
    start, goal = centers[0].copy(), centers[chain - 1].copy()
    lo = float((centers - axes).min()) - 1.0
    hi = float((centers + axes).max()) + 1.0

    m = pyo.ConcreteModel(name=f"route_t{t}k{k}d{d}_s{seed}")
    m.T, m.S, m.E, m.J = range(t + 1), range(t), range(k), range(d)
    m.p = pyo.Var(m.T, m.J, bounds=(lo, hi))
    m.u = pyo.Var(m.S, m.J, bounds=(0.0, hi - lo))
    m.rc = pyo.Var(m.T, bounds=(0.0, 2.0))
    m.fix0 = pyo.Constraint(m.J, rule=lambda m, j: m.p[0, j] == float(start[j]))
    m.fixT = pyo.Constraint(m.J, rule=lambda m, j: m.p[t, j] == float(goal[j]))
    m.l1a = pyo.Constraint(m.S, m.J, rule=lambda m, s, j: m.u[s, j] >= m.p[s + 1, j] - m.p[s, j])
    m.l1b = pyo.Constraint(m.S, m.J, rule=lambda m, s, j: m.u[s, j] >= m.p[s, j] - m.p[s + 1, j])

    def region(m, s):
        return [
            [
                sum(((m.p[s, j] - float(centers[e, j])) / float(axes[e, j])) ** 2 for j in m.J)
                <= 1.0,
                m.rc[s] == float(region_cost[e]),
            ]
            for e in m.E
        ]

    m.region = Disjunction(m.T, rule=region)
    m.obj = pyo.Objective(expr=sum(m.u[s, j] for s in m.S for j in m.J) + sum(m.rc[s] for s in m.T))
    return m


_BUILDERS: dict[str, Callable[..., Any]] = {
    "clay": lambda p, s: build_clay(p["r"], p["c"], s),
    "cover": lambda p, s: build_cover(p["n"], p["k"], s),
    "route": lambda p, s: build_route(p["t"], p["k"], p["d"], s),
}


def builder(name: str) -> Callable[[], Any]:
    """A zero-argument builder for ``name``; a fresh Pyomo model on every call."""
    family, size, seed = parse(name)

    def build() -> Any:
        return _BUILDERS[family](size, seed)

    build.__name__ = f"build_{name[len(PREFIX) :] if name.startswith(PREFIX) else name}"
    return build


def is_name(name: str) -> bool:
    if not name.startswith(PREFIX):
        return False
    try:
        parse(name)
    except ValueError:
        return False
    return True


__all__ = ["PANEL", "PREFIX", "builder", "clay_params", "is_name", "parse"]
