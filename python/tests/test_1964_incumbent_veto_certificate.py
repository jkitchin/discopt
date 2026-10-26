"""#1964: an ``incumbent_callback`` veto must not leave a certificate over the pruned box.

A veto (``return False``) sentinels the WHOLE node: re-solving would return the same
point forever, so the node is fathomed. That fathom is non-rigorous -- it rejects one
point and proves nothing about the rest of the box. When another incumbent was
accepted, the solve still reported ``optimal`` / ``gap_certified=True`` at -4.03 while
the pruned box held an accepted point at -8.47.

The fix treats a veto exclusion like the #1954 lazy-cap exclusion: decertify and floor
the reported bound at the node's pop-time bound, so certification is re-earned only if
the accepted incumbent is within tolerance of every pruned box.
"""

from __future__ import annotations

import numpy as np
from discopt import Model

ORACLE = -8.47  # every y != 5 point; the y <= 4 model solves to this


def _build():
    m = Model("cb")
    y = m.integer("y", lb=0, ub=5)
    x = m.continuous("x", lb=0, ub=10)
    u = m.continuous("u", lb=-3, ub=3)
    m.subject_to(x <= y + 0.5 * u * u)
    m.subject_to(x * u <= 1)
    m.minimize(-x - 0.01 * u)
    return m, y


def _veto_y5(calls):
    def cb(ctx, model, sol):
        calls.append(1)
        return abs(float(np.asarray(sol["y"]).ravel()[0]) - 5) > 0.5

    return cb


def test_oracle_restricted_model():
    m, y = _build()
    m.subject_to(y <= 4)
    r = m.solve(time_limit=30, threads=1)
    assert r.status == "optimal"
    assert abs(r.objective - ORACLE) < 1e-3


def test_veto_never_certifies_a_worse_point_than_an_accepted_one():
    m, _ = _build()
    calls: list[int] = []
    r = m.solve(time_limit=30, threads=1, incumbent_callback=_veto_y5(calls))
    assert calls, "the callback never ran: the probe measured nothing"
    assert r.objective is not None
    y = float(np.asarray(r.x["y"]).ravel()[0])
    assert abs(y - 5) > 0.5, "a vetoed point became the incumbent"
    # Either the search found the accepted optimum, or it must not claim one.
    if r.gap_certified:
        assert r.objective <= ORACLE + 1e-3, (r.status, r.objective, r.bound)
    else:
        assert r.status != "optimal"
    # The reported dual bound never crosses the accepted optimum.
    if r.bound is not None:
        assert r.bound <= ORACLE + 1e-3, r.bound


def test_no_veto_still_certifies():
    """A callback that accepts everything changes nothing (no exclusion, no taint)."""
    m, _ = _build()
    r = m.solve(time_limit=30, threads=1, incumbent_callback=lambda ctx, model, sol: True)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective < ORACLE + 1e-3  # unrestricted optimum is at y = 5, below ORACLE
