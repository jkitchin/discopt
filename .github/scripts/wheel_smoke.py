#!/usr/bin/env python3
"""Solve a tiny LP and MILP from an *installed* wheel, with and without HiGHS.

Run by ``release.yml`` in a fresh venv after ``pip install dist/<wheel>`` (``--no-highs``)
and again after ``pip install "<wheel>[highs]"`` (``--highs``). A green build job is not
evidence that an installed wheel solves; this script is.

* ``--no-highs`` -- ``highspy`` is the optional ``discopt[highs]`` extra (highs-parity
  P6.3), so the bare wheel must not pull it in, and must still solve both models on the
  in-house engines: the LP on the Rust LP route (the default since highs-parity D1), the
  MILP on the in-house MILP engine, whose route records that ``highspy`` is not installed.
* ``--highs`` -- with the extra, a wheel whose ``highspy`` requirement does not resolve on
  its platform must fail the release, not the first user who tries it. The LP stays on
  the Rust LP route; the MILP must go to the verified HiGHS route (its default where
  ``highspy`` is installed) and report HiGHS's version.

Exits non-zero unless every check ran and passed (the executed count is printed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if len(sys.argv) != 2 or sys.argv[1] not in ("--highs", "--no-highs"):
    sys.exit("usage: wheel_smoke.py --highs | --no-highs")
WANT_HIGHS = sys.argv[1] == "--highs"

# The shipped defaults are under test, so nothing may pre-select a route.
for var in ("DISCOPT_LP_MILP_BACKEND", "DISCOPT_LP_BACKEND", "DISCOPT_LP_STD_ENGINE"):
    os.environ.pop(var, None)

import discopt  # noqa: E402
from discopt import Model  # noqa: E402
from discopt._optional_highs import highs_available  # noqa: E402

# The installed wheel, not a source checkout that happens to be on sys.path.
repo = Path(__file__).resolve().parents[2]
if Path(discopt.__file__).resolve().is_relative_to(repo):
    sys.exit(f"FAIL: imported discopt from the checkout ({discopt.__file__}), not the wheel")
if highs_available() != WANT_HIGHS:
    sys.exit(f"FAIL: highspy installed={highs_available()}, expected {WANT_HIGHS}")
if WANT_HIGHS:
    import highspy

    print("discopt", discopt.__file__, "highspy", highspy.Highs().version(), flush=True)
else:
    print("discopt", discopt.__file__, "highspy not installed", flush=True)

checked = 0


def check(res, want: float, label: str, route_prefix: str, highs_route: bool) -> None:
    global checked
    stats = res.solver_stats or {}
    route = res.algorithm_route or ""
    problems = []
    if res.status != "optimal" or not res.gap_certified:
        problems.append(f"status {res.status!r}, gap_certified={res.gap_certified}")
    if res.objective is None or abs(res.objective - want) > 1e-6 * (1 + abs(want)):
        problems.append(f"objective {res.objective} != {want}")
    if not route.startswith(route_prefix):
        problems.append(f"route {route!r} does not start with {route_prefix!r}")
    if highs_route:
        if stats.get("route/lp_milp_backend") != 1.0:
            problems.append(f"route/lp_milp_backend={stats.get('route/lp_milp_backend')}")
        if not stats.get("highs/version"):
            problems.append("highs/version missing")
    elif label == "MILP" and "highspy is not installed" not in route:
        problems.append(f"route {route!r} does not say highspy is not installed")
    if problems:
        sys.exit(f"FAIL {label}: " + "; ".join(problems))
    checked += 1
    print(f"ok {label}: objective {res.objective}, route {route!r}", flush=True)


# LP: max 3x + 2y s.t. x + y <= 4, x + 3y <= 6, x, y >= 0  ->  x = 4, y = 0, obj 12.
m = Model("wheel_smoke_lp")
x = m.continuous("x", lb=0, ub=10)
y = m.continuous("y", lb=0, ub=10)
m.maximize(3 * x + 2 * y)
m.subject_to(x + y <= 4)
m.subject_to(x + 3 * y <= 6)
check(m.solve(), 12.0, "LP", "rust-lp", highs_route=False)

# MILP: max 5a + 4b + 3c s.t. 2a + 3b + c <= 5, 4a + b + 2c <= 11, 3a + 4b + 2c <= 8,
# a, b, c integer in [0, 5]  ->  a = 2, b = 0, c = 1, obj 13.
m = Model("wheel_smoke_milp")
a = m.integer("a", lb=0, ub=5)
b = m.integer("b", lb=0, ub=5)
c = m.integer("c", lb=0, ub=5)
m.maximize(5 * a + 4 * b + 3 * c)
m.subject_to(2 * a + 3 * b + c <= 5)
m.subject_to(4 * a + b + 2 * c <= 11)
m.subject_to(3 * a + 4 * b + 2 * c <= 8)
milp_prefix = "highs-milp" if WANT_HIGHS else "native-milp"
check(m.solve(), 13.0, "MILP", milp_prefix, highs_route=WANT_HIGHS)

if checked != 2:
    sys.exit(f"FAIL: {checked} of 2 checks executed")
print(f"wheel smoke ({sys.argv[1]}): 2 of 2 checks passed")
