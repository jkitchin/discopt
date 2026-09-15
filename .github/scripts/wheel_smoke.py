#!/usr/bin/env python3
"""Solve a tiny LP and MILP from an *installed* wheel, on the HiGHS route.

Run by ``release.yml`` in a fresh venv after ``pip install dist/<wheel>``. highspy is a
core dependency (issue #1229, ``docs/dev/lp-milp-highs-routing-plan.md`` §5): a wheel
whose highspy requirement does not resolve on its platform must fail the release, not
the first user who tries it. A green build job is not that evidence -- this script is.

Exits non-zero unless both solves ran, returned the known optimum, and reported that
HiGHS (not the Rust driver) produced the answer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The shipped default must be the HiGHS route, so nothing may pre-select it.
os.environ.pop("DISCOPT_LP_MILP_BACKEND", None)

import discopt  # noqa: E402
import highspy  # noqa: E402
from discopt import Model  # noqa: E402

# The installed wheel, not a source checkout that happens to be on sys.path.
repo = Path(__file__).resolve().parents[2]
if Path(discopt.__file__).resolve().is_relative_to(repo):
    sys.exit(f"FAIL: imported discopt from the checkout ({discopt.__file__}), not the wheel")
print("discopt", discopt.__file__, "highspy", highspy.Highs().version(), flush=True)

checked = 0


def check(res, want: float, label: str) -> None:
    global checked
    stats = res.solver_stats or {}
    problems = []
    if res.status != "optimal":
        problems.append(f"status {res.status!r}")
    if res.objective is None or abs(res.objective - want) > 1e-6 * (1 + abs(want)):
        problems.append(f"objective {res.objective} != {want}")
    if stats.get("route/lp_milp_backend") != 1.0:
        problems.append(f"route/lp_milp_backend={stats.get('route/lp_milp_backend')}")
    if not stats.get("highs/version"):
        problems.append("highs/version missing")
    if problems:
        sys.exit(f"FAIL {label}: " + "; ".join(problems))
    checked += 1
    print(f"ok {label}: objective {res.objective}, highs/version {stats['highs/version']:.0f}")


# LP: max 3x + 2y s.t. x + y <= 4, x + 3y <= 6, x, y >= 0  ->  x = 4, y = 0, obj 12.
m = Model("wheel_smoke_lp")
x = m.continuous("x", lb=0, ub=10)
y = m.continuous("y", lb=0, ub=10)
m.maximize(3 * x + 2 * y)
m.subject_to(x + y <= 4)
m.subject_to(x + 3 * y <= 6)
check(m.solve(), 12.0, "LP")

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
check(m.solve(), 13.0, "MILP")

if checked != 2:
    sys.exit(f"FAIL: {checked} of 2 checks executed")
print("wheel smoke: 2 of 2 checks passed")
