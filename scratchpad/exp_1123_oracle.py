#!/usr/bin/env python -u
"""Independent oracle for the #1123 entry experiment's reference optima.

Brute-forces each bilevel instance WITHOUT discopt: grid the leader variable,
solve the follower exactly with scipy.linprog at each point, evaluate the leader
objective on the follower's optimistic response, take the best. This is what the
hand-derived optima in exp_1123_bilevel_bigm_free.py are checked against, so a
reference value is a measurement rather than an assertion (CLAUDE.md §4).
"""

import sys

import numpy as np
import scipy.optimize as opt

CHECKS = 0


def check(label, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return ok


def _sweep(x_lo, x_hi, n, follower, leader_obj):
    """Best (obj, x, y) over a grid, skipping x where the follower is INFEASIBLE.

    A leader x that makes the follower infeasible is not a candidate — the
    follower has no response there. Counted so an all-infeasible sweep cannot
    masquerade as 'no better point found' (CLAUDE.md §6).
    """
    best, n_feas = (np.inf, None, None), 0
    for xv in np.linspace(x_lo, x_hi, n):
        yv = follower(float(xv))
        if yv is None:
            continue
        n_feas += 1
        f = leader_obj(float(xv), yv)
        if f < best[0]:
            best = (f, float(xv), yv)
    return best, n_feas


def scan(name, x_lo, x_hi, follower, leader_obj, claimed, n=2001, refine=4001):
    """Coarse grid, then refine locally around the incumbent."""
    (obj, xs, ys), n_feas = _sweep(x_lo, x_hi, n, follower, leader_obj)
    if not check(f"{name}: follower feasible somewhere on the grid", n_feas > 0, f"{n_feas}/{n}"):
        return None
    h = (x_hi - x_lo) / (n - 1)
    (obj2, xs2, ys2), _ = _sweep(
        max(x_lo, xs - 2 * h), min(x_hi, xs + 2 * h), refine, follower, leader_obj
    )
    if obj2 < obj:
        obj, xs, ys = obj2, xs2, ys2
    ok = abs(obj - claimed) <= 1e-3
    check(
        f"{name}: brute-force optimum matches hand derivation",
        ok,
        f"grid={obj:+.6g} at x={xs:.6g}, y={ys:.6g} | claimed={claimed:+.6g}",
    )
    return obj


def _lp(c, A_ub, b_ub, bounds=(0.0, 10.0)):
    """Follower LP. Returns None when the follower is infeasible at this leader x."""
    r = opt.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[bounds], method="highs")
    if not r.success:
        if "infeasible" in (r.message or "").lower():
            return None
        raise RuntimeError(f"follower LP failed for a non-infeasibility reason: {r.message}")
    return float(r.x[0])


print("=" * 78)
print("ORACLE — independent reference optima for the #1123 entry experiment")
print("=" * 78)

# bard_lp: leader min x-4y; follower min y s.t. x+y>=3, y<=2x
scan(
    "bard_lp",
    0.0,
    10.0,
    lambda xv: _lp([1.0], [[-1.0], [1.0]], [xv - 3.0, 2.0 * xv]),
    lambda xv, yv: xv - 4.0 * yv,
    -7.0,
)

# follower_pushes_up: leader min x+y; follower MAX y s.t. y<=x, y<=4-x
scan(
    "follower_pushes_up",
    0.0,
    4.0,
    lambda xv: _lp([-1.0], [[1.0], [1.0]], [xv, 4.0 - xv]),
    lambda xv, yv: xv + yv,
    0.0,
)

# two_row_active: leader min x-2y; follower min -y s.t. y<=1+x, y<=3-x
scan(
    "two_row_active",
    0.0,
    3.0,
    lambda xv: _lp([-1.0], [[1.0], [1.0]], [1.0 + xv, 3.0 - xv]),
    lambda xv, yv: xv - 2.0 * yv,
    -3.0,
)

print(f"\nEXECUTED CHECKS: {CHECKS}")
if CHECKS == 0:
    sys.exit("FATAL: zero checks executed (CLAUDE.md §6)")
