"""#1619 D-05 graduation panel: ``DISCOPT_POLY_HESSIAN_ALPHA`` OFF vs ON.

RETIRED 2026-10-05: the panel ran (28 rows, 30 s): cert-clean, and exactly
neutral -- identical status, bound and certificate on every row, nodes 6536 vs
6578 (pcubic at its time limit) -- although the flag shrank alpha by ~17% on the
node boxes it reached (camel6: sum of alpha over 87 calls 2600 -> 2165). The
alphaBB bound never wins the per-node ``max`` against the other bounds. The flag
and ``_relax/convexity/poly_hessian.py`` were deleted; this script is kept as the
benchmark record (it now measures nothing: the flag is no longer read).

ON intersects the alphaBB interval Hessian with the exact enclosure of the
expanded polynomial's Hessian (``_relax/convexity/poly_hessian.py``). alphaBB is
the node bound only when the McCormick LP relaxer is not (``solver.py``: the
alpha estimate is skipped while the LP relaxer supplies node bounds), so the
in-repo ``.nl`` corpus cannot exercise it; this panel uses the standard
polynomial global-optimization test functions, each with its published global
minimum as the oracle, solved twice per arm: default settings and
``mccormick_bounds="none"`` (the alphaBB route). Arms run back to back per row.

CLAUDE.md §5 bars, per arm: no certified objective off the oracle, no bound above
it (minimize), no certificate lost; then net-positive on nodes / wall / bound.
Prints an executed-comparison count and exits non-zero when it is zero (§6).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import discopt.modeling as dm

FLAG = "DISCOPT_POLY_HESSIAN_ALPHA"


def _vars(m, n, lo, hi):
    return [m.continuous(f"x{i}", lb=lo, ub=hi) for i in range(n)]


def camel6():
    m = dm.Model("camel6")
    x, y = m.continuous("x", lb=-3, ub=3), m.continuous("y", lb=-2, ub=2)
    m.minimize((4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2)
    return m, -1.0316284535


def camel3():
    m = dm.Model("camel3")
    x, y = m.continuous("x", lb=-5, ub=5), m.continuous("y", lb=-5, ub=5)
    m.minimize(2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2)
    return m, 0.0


def rosenbrock(n):
    m = dm.Model(f"rosen{n}")
    x = _vars(m, n, -2.048, 2.048)
    m.minimize(sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(n - 1)))
    return m, 0.0


def styblinski(n):
    m = dm.Model(f"styb{n}")
    x = _vars(m, n, -5, 5)
    m.minimize(sum(0.5 * (xi**4 - 16 * xi**2 + 5 * xi) for xi in x))
    return m, -39.16616570377142 * n


def beale():
    m = dm.Model("beale")
    x, y = m.continuous("x", lb=-4.5, ub=4.5), m.continuous("y", lb=-4.5, ub=4.5)
    m.minimize((1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2)
    return m, 0.0


def goldstein_price():
    m = dm.Model("gp")
    x, y = m.continuous("x", lb=-2, ub=2), m.continuous("y", lb=-2, ub=2)
    a = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    b = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    m.minimize(a * b)
    return m, 3.0


def dixon_price(n):
    m = dm.Model(f"dixon{n}")
    x = _vars(m, n, -10, 10)
    m.minimize(
        (x[0] - 1) ** 2 + sum((i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, n))
    )
    return m, 0.0


def powell4():
    m = dm.Model("powell")
    x = _vars(m, 4, -4, 5)
    m.minimize(
        (x[0] + 10 * x[1]) ** 2
        + 5 * (x[2] - x[3]) ** 2
        + (x[1] - 2 * x[2]) ** 4
        + 10 * (x[0] - x[3]) ** 4
    )
    return m, 0.0


def zakharov(n):
    m = dm.Model(f"zak{n}")
    x = _vars(m, n, -5, 10)
    s = sum(0.5 * (i + 1) * x[i] for i in range(n))
    m.minimize(sum(xi**2 for xi in x) + s**2 + s**4)
    return m, 0.0


def price_cubic():
    """A constrained polynomial: min (x-1)^3 * y + x*y^2 s.t. x^2 + y^2 <= 4."""
    m = dm.Model("pcubic")
    x, y = m.continuous("x", lb=-2, ub=2), m.continuous("y", lb=-2, ub=2)
    m.minimize((x - 1) ** 3 * y + x * y**2)
    m.subject_to(x**2 + y**2 <= 4)
    return m, None  # no published optimum: cross-checked between arms only


INSTANCES = [
    camel6,
    camel3,
    lambda: rosenbrock(2),
    lambda: rosenbrock(3),
    lambda: styblinski(2),
    lambda: styblinski(3),
    beale,
    goldstein_price,
    lambda: dixon_price(2),
    lambda: dixon_price(3),
    powell4,
    lambda: zakharov(2),
    lambda: zakharov(3),
    price_cubic,
]


def run(build, on, tl, kw):
    os.environ[FLAG] = "1" if on else "0"
    m, opt = build()
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl, **kw)
    return m.name, opt, r, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()
    import discopt

    print("discopt", discopt.__file__, "load", os.getloadavg(), flush=True)
    tally = {a: dict(false=0, cert=0, wall=0.0, nodes=0) for a in ("OFF", "ON")}
    lost = gained = disagree = compared = 0
    for kw in ({}, {"mccormick_bounds": "none"}):
        for build in INSTANCES:
            out = {}
            for arm, on in (("OFF", False), ("ON", True)):
                name, opt, r, wall = run(build, on, args.time_limit, kw)
                t = tally[arm]
                t["wall"] += wall
                t["nodes"] += int(r.node_count or 0)
                t["cert"] += bool(r.gap_certified)
                if opt is not None:
                    tol = 1e-4 * (1 + abs(opt))
                    if r.gap_certified and abs(r.objective - opt) > tol:
                        t["false"] += 1
                        print(f"FALSE {arm} {name}: objective {r.objective} vs {opt}", flush=True)
                    if r.bound is not None and r.bound > opt + tol:
                        t["false"] += 1
                        print(f"FALSE {arm} {name}: bound {r.bound} above {opt}", flush=True)
                out[arm] = r
                print(
                    f"{name:9s} {str(kw):30s} {arm:3s} {r.status:10s} obj={r.objective} "
                    f"bound={r.bound} n={r.node_count} wall={wall:.2f}",
                    flush=True,
                )
            compared += 1
            c0, c1 = out["OFF"].gap_certified, out["ON"].gap_certified
            lost += bool(c0 and not c1)
            gained += bool(c1 and not c0)
            if (
                c0
                and c1
                and abs(out["OFF"].objective - out["ON"].objective)
                > 1e-4 * (1 + abs(out["OFF"].objective))
            ):
                disagree += 1
    for arm, t in tally.items():
        print(
            f"{arm}: false={t['false']} certified={t['cert']} nodes={t['nodes']} "
            f"wall={t['wall']:.1f}s",
            flush=True,
        )
    print(f"compared={compared} lost={lost} gained={gained} disagree={disagree}", flush=True)
    return 0 if compared else 1


if __name__ == "__main__":
    sys.exit(main())
