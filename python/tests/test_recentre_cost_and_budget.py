"""Recentring is linear in the model and charged to the caller's ``time_limit``.

Found re-measuring #1929 on MINLPLib ``gams04`` (36,766 vars, 46,137 rows): the
#1537 recentring pass spent 48.7 s before ``solve_model`` started its clock, and
the recentred twin was then solved with the full ``time_limit`` again, so the solve
returned at 105-114 s on a 60 s limit.

* The rewriter built (and so canonicalised) the rebuilt node of every affine
  subtree, which its parent then discarded: a sum chain of n terms canonicalised
  every prefix, O(n^2). Now an affine subtree is canonicalised once, where it is
  used; the recentred model is the same, object sharing included (47,282 rows and
  objectives compared on gams04, the in-repo corpus and synthetic offset models).
* The nested solve gets what is left of ``time_limit``.
"""

import time

import discopt.modeling as dm
import pytest
from discopt.modeling import _recentre


def _offset_chain(n):
    m = dm.Model("chain")
    x = m.continuous("x", shape=(n,), lb=1e6, ub=1e6 + 1)
    y = m.continuous("y", lb=0, ub=1)
    s = x[0]
    for i in range(1, n):
        s = s + x[i]
    m.subject_to(s * y <= 5e8)  # the affine chain sits under a non-affine parent
    m.minimize(s + y)
    return m


@pytest.mark.unit
def test_an_affine_chain_is_canonicalised_once(monkeypatch):
    calls = []
    real = _recentre._canonical

    def spy(aff):
        calls.append(len(aff[0]))
        return real(aff)

    monkeypatch.setattr(_recentre, "_canonical", spy)
    n = 400
    rc = _recentre.recentre(_offset_chain(n))
    assert rc is not None and "x" in rc.shifts
    # One canonical form per affine boundary (the chain under ``*``, the chain as
    # the objective term), not one per prefix: before, sum(calls) was ~n^2/2.
    assert sum(calls) <= 4 * n, (len(calls), sum(calls))


@pytest.mark.unit
def test_the_nested_solve_gets_what_is_left_of_the_limit(monkeypatch):
    monkeypatch.setenv(_recentre.FLAG, "1")
    seen = []
    real_recentre = _recentre.recentre
    real_solve = _recentre.solve_recentred

    def slow_recentre(model, *a, **k):
        time.sleep(1.0)  # stands in for a costly rewrite
        return real_recentre(model, *a, **k)

    def spy(outer, rc, solve_args):
        seen.append(solve_args["time_limit"])
        return real_solve(outer, rc, solve_args)

    monkeypatch.setattr(_recentre, "recentre", slow_recentre)
    monkeypatch.setattr(_recentre, "solve_recentred", spy)
    m = _offset_chain(3)
    m.solve(time_limit=20)
    assert seen, "the model was not recentred, so the assertion below would be vacuous"
    assert seen[0] <= 19.0, seen
