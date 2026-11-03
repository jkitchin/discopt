"""Staged-oracle ladder for the IIS engine (portfolio/order flag ``l``).

Before the ordinary filter, the ladder runs the deletion filter with cheap,
incomplete oracles: HiGHS presolve alone, then a MIP capped at a multiple of the
measured base proof. It drops an element only on a proven Infeasible. The
ordinary filter then runs on what is left, so the result is decided, confirmed
and certified exactly as without the ladder. These tests pin that contract: the
same kind of answer, fewer probes where presolve can do the work, and a valid
result when the ladder's budget runs out midway.
"""

import discopt.modeling as dm
import pytest
from discopt import _iis_milp
from discopt._iis_milp import _parse_order, milp_iis

highspy = pytest.importorskip("highspy")


def _kedron_like_with_noise(n_noise=25):
    """kedron's 12-element big-M structure plus redundant big-M rows and binaries."""
    m = dm.Model("kedron_noise")
    x1 = m.continuous("x1", lb=0, ub=0.34375)
    x2 = m.continuous("x2", lb=0, ub=0.6875)
    z1 = m.binary("z1")
    z2 = m.binary("z2")
    a = m.continuous("a", lb=0, ub=1e7)
    b = m.continuous("b", lb=0, ub=1e7)
    c = m.continuous("c", lb=0, ub=1e7)
    d = m.continuous("d", lb=0, ub=1e7)
    w = m.continuous("w", lb=-1e7, ub=1e7)
    m.subject_to(x1 - 8e6 * z1 >= -7999999.656, name="R2415")
    m.subject_to(x2 - 8e6 * z2 >= -7999999.312, name="R3701")
    m.subject_to(a <= 8e6 * z1, name="R2322")
    m.subject_to(c <= 8e6 * z2, name="R3608")
    m.subject_to(w == a - b + 4000, name="R0757")
    m.subject_to(w == 6000 - c + d, name="R0993")
    y = m.continuous("y", shape=(n_noise,), lb=0, ub=100)
    u = m.binary("u", shape=(n_noise,))
    for k in range(n_noise):
        m.subject_to(y[k] <= 8e6 * u[k], name=f"noise_bigm{k}")
        m.subject_to(y[k] + w <= 1e7, name=f"noise_link{k}")
    m.minimize(0 * w)
    return m


def _size(r):
    return len(r.constraints) + len(r.bounds) + len(r.integrality)


@pytest.mark.smoke
def test_ladder_returns_a_complete_certified_iis():
    m = _kedron_like_with_noise()
    res = milp_iis(m, order="CIB-l", total_time_limit=120.0)
    assert res.ladder_size is not None and res.ladder_calls > 0
    assert res.complete and res.proven_irreducible
    assert _size(res) == 12
    # Every constraint and bound is certified. An integrality's probe point may sit
    # at the big-M vertex (z = 1 - 4e-8): a true witness, but one that violates
    # integrality by less than the 1e-6 certificate tolerance, so it is
    # (correctly) not reported as a certificate.
    assert {k for k in res.certificates if k[0] != "integrality"} == {
        ("constraint", i) for i in res.constraints
    } | {("bound", j, s) for j, s in res.bounds}
    names = sorted(m._constraints[i].name for i in res.constraints)
    assert names == ["R0757", "R0993", "R2322", "R2415", "R3608", "R3701"]
    plain = milp_iis(m, order="CIB", total_time_limit=120.0)
    assert plain.ladder_size is None
    assert _size(plain) == 12


def test_ladder_out_of_time_midway_still_returns_a_valid_iis(monkeypatch):
    real = _iis_milp._Engine.presolve_proves
    calls = {"n": 0}

    def presolve_proves(self, *a, **k):
        calls["n"] += 1
        if calls["n"] > 5:
            raise _iis_milp._OutOfTime  # the ladder's budget runs out
        return real(self, *a, **k)

    monkeypatch.setattr(_iis_milp._Engine, "presolve_proves", presolve_proves)
    m = _kedron_like_with_noise(10)
    res = milp_iis(m, order="CIB-l", total_time_limit=120.0)
    assert calls["n"] > 5
    assert res.complete and res.proven_irreducible and _size(res) == 12


def test_ladder_flag_parses():
    assert _parse_order("CIB-l") == ("CIB", False, True, True)
    assert _parse_order("ICB-rnl") == ("ICB", True, False, True)


def _ladder_lie_model():
    m = dm.Model("ladder_lie")
    x = m.binary("x", shape=(3,))
    m.subject_to(x[0] + x[1] >= 2, name="r0")
    m.subject_to(x[0] + x[2] <= 0, name="r1")
    m.subject_to(x[1] + x[2] <= 2, name="r2")
    m.minimize(0 * x[0])
    return m


@pytest.mark.parametrize("confirm", ["final", "each"])
def test_false_presolve_claim_cannot_shrink_the_certified_set(monkeypatch, confirm):
    """A ladder drop rests on one presolve-on claim; it must be cross-checked.

    PR #2114 review: presolve is stubbed to claim {r0} (feasible: x0=x1=1) is
    infeasible -- the #1634 failure mode. Before the fix the ladder dropped r1 on
    that claim and {r0} came back ``proven_irreducible=True`` in both confirm modes.
    The ladder's set must now be confirmed under both presolve configurations, and
    its drops discarded when it is not.
    """
    import discopt._iis_milp as im

    real = im._Engine.presolve_proves
    lies = {"n": 0}

    def lying(self, cons, bounds, ints, stop_at=None):
        if 0 in cons and 1 not in cons:
            lies["n"] += 1
            return True
        return real(self, cons, bounds, ints, stop_at)

    monkeypatch.setattr(im._Engine, "presolve_proves", lying)
    m = _ladder_lie_model()
    r = im.milp_iis(m, order="CIB-l", confirm=confirm)
    assert lies["n"] > 0  # the stub fired: the ladder did act on the false claim
    names = {m._constraints[i].name for i in r.constraints}
    assert names == {"r0", "r1"}  # the true IIS, not the feasible {r0}
    assert r.proven_irreducible and r.complete
    assert r.ladder_rejected
