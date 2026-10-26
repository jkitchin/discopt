"""#1953: two tolerance tests that certified a vertex the exact model contradicts.

1. **Objective lattice.** ``objective_granularity`` accepted a coefficient within a
   rational tolerance of a "nice" value, so ``x - (1 - 2^-32) y`` over integers in
   ``[2^30, 2^31]`` was given unit spacing. Every attainable objective differs from a
   lattice point by up to ``2^-32 * 2^31 = 0.5``, and the cutoff ``U - 1`` fathomed the
   optimum (0.25) under a 0.5 incumbent. A lattice now carries its *drift* -- the
   largest distance from an attainable value to a lattice point, bounded with directed
   rounding -- and every consumer widens by it; a lattice whose drift is not small
   against its spacing is declined.
2. **Sub-tolerance noise filter.** The primal simplex dropped a wrong-signed reduced
   cost below ``SUBTOL_NOISE_REL * (|c| + |A^T y|)`` as rounding. On
   ``min -x + (1 - 2^-32) y``, ``x = y``, ``x, y in [2^30, 2^31]`` it is exactly
   ``-2^-32`` and the column can move ``2^30``: the vertex at -0.25 was certified
   against an optimum of -0.5. Such a column is now confirmed or rejected with
   refined duals whenever the NS bound leaves the vertex uncertified.

Zero tolerance against the exact optima: a certified result must *be* the optimum and
every reported bound must lie below it.
"""

from __future__ import annotations

import discopt
import pytest

N, EPS = 2**30, 2.0**-32


def _lattice():
    m = discopt.Model("lattice_1953")
    x = m.integer("x", lb=N, ub=2 * N)
    y = m.integer("y", lb=N, ub=2 * N)
    m.subject_to(x - y == 0)
    m.minimize(x - (1 - EPS) * y)
    return m, (x, y), 0.25


def _subtol(integer):
    m = discopt.Model("subtol_1953")
    var = m.integer if integer else m.continuous
    x = var("x", lb=N, ub=2 * N)
    y = var("y", lb=N, ub=2 * N)
    m.subject_to(x - y == 0)
    m.minimize(-x + (1 - EPS) * y)
    return m, (x, y), -0.5


def _check(r, opt):
    if r.bound_valid and r.bound is not None:
        assert r.bound <= opt, (r.status, r.objective, r.bound)
    if r.gap_certified:
        assert r.objective == opt, (r.status, r.objective, r.bound)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build, opts, start",
    [
        (_lattice, {"nlp_solver": "simplex"}, False),
        (_lattice, {"nlp_solver": "simplex"}, True),
        (_lattice, {"milp_backend": "native"}, True),
        (lambda: _subtol(False), {"milp_backend": "native"}, False),
        (lambda: _subtol(True), {"milp_backend": "native"}, False),
        (lambda: _subtol(True), {"nlp_solver": "simplex"}, False),
        (lambda: _subtol(False), {}, False),
    ],
    ids=[
        "lattice-mono",
        "lattice-mono-start",
        "lattice-native-start",
        "subtol-lp-native",
        "subtol-milp-native",
        "subtol-milp-mono",
        "subtol-lp-default",
    ],
)
def test_no_certificate_contradicts_exact_optimum(build, opts, start):
    m, (x, y), opt = build()
    if start:
        # The pre-#1953 failure needed the 0.5 incumbent to arm the lattice cutoff.
        opts = {**opts, "initial_solution": {x: 2.0 * N, y: 2.0 * N}}
    r = m.solve(time_limit=30, threads=1, **opts)
    _check(r, opt)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build, opts",
    [
        (_lattice, {"nlp_solver": "simplex"}),
        (lambda: _subtol(False), {"milp_backend": "native"}),
        (lambda: _subtol(True), {"milp_backend": "native"}),
    ],
    ids=["lattice", "subtol-lp", "subtol-milp"],
)
def test_native_routes_certify_the_exact_optimum(build, opts):
    m, _, opt = build()
    r = m.solve(time_limit=30, threads=1, **opts)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.objective == opt
