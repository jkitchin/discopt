"""Mod-2 (GF(2)) certificates in the MILP IIS engine.

Integer equality rows whose sum is ``even.x = odd`` mod 2 have no integer
solution, whatever the bounds -- an exact proof in integer arithmetic. The engine
uses it twice:

* as a candidate (``_parity_core``): a minimal certificate's rows plus the
  integrality and bounds of the columns they touch, proven by HiGHS like the SAT
  core before use (MIPcc27 neos-807454: Gurobi's 9 rows + 56 integralities in
  ~6 s, was 46 s; enlight11: the 197-element certificate set instead of the full
  363-element model);
* in the second confirmation route (``_Engine._parity_infeasible``): a set that
  contains a certificate is confirmed without a presolve-free solve -- which
  needed 47 s on enlight11's set and stopped checkpoint 0 inside a 60 s budget.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _iis_milp
from discopt._iis_milp import _gf2_certificates, _parity_core, milp_iis

highspy = pytest.importorskip("highspy")


def _odd_cycle(n, noise=10):
    """x_i + x_{i+1} = 1 around an odd cycle of binaries: LP-feasible (all 1/2),
    integer-infeasible by parity. Plus redundant rows and columns."""
    m = dm.Model("cycle")
    x = m.binary("x", shape=(n,))
    for i in range(n):
        m.subject_to(x[i] + x[(i + 1) % n] == 1, name=f"cyc{i}")
    y = m.integer("y", shape=(noise,), lb=0, ub=5)
    for k in range(noise):
        m.subject_to(y[k] + x[k % n] <= 6, name=f"noise{k}")
    m.minimize(0 * x[0])
    return m


def test_gf2_elimination_finds_exactly_the_contradictions():
    # rows over bits: x0+x1=1, x1+x2=1, x0+x2=1 -> sum: 0 = 1 (mod 2)
    rows = [(0b011, 1), (0b110, 1), (0b101, 1)]
    certs = _gf2_certificates(rows)
    assert certs == [0b111]
    # make the third rhs even: consistent, no certificate
    assert _gf2_certificates([(0b011, 1), (0b110, 1), (0b101, 0)]) == []


@pytest.mark.smoke
def test_odd_cycle_candidate_gives_a_certified_iis():
    n = 5
    m = _odd_cycle(n)
    res = milp_iis(m)
    assert res.parity_rows == n
    names = sorted(m._constraints[i].name for i in res.constraints)
    assert names == sorted(f"cyc{i}" for i in range(n))
    assert res.complete and res.proven_irreducible
    # One integral variable already forces x = 1 - x around an odd cycle.
    assert len(res.integrality) == 1 and res.bounds == []
    assert len(res.certificates) == len(res.constraints) + len(res.integrality)


def test_parity_core_returns_a_minimal_certificate():
    # Two disjoint odd cycles (5 and 3): the core is the smaller, alone.
    m = dm.Model("two")
    a = m.binary("a", shape=(5,))
    b = m.binary("b", shape=(3,))
    for i in range(5):
        m.subject_to(a[i] + a[(i + 1) % 5] == 1, name=f"a{i}")
    for i in range(3):
        m.subject_to(b[i] + b[(i + 1) % 3] == 1, name=f"b{i}")
    m.minimize(0 * a[0])
    import scipy.sparse as sp
    from discopt.decomposition._linear import extract_linear

    lin = extract_linear(m)
    A = sp.csr_matrix(lin.A)
    b_ = np.asarray(lin.b, float)
    lo = np.where(np.isin(lin.sense, [">=", "=="]), b_, -1e20)
    hi = np.where(np.isin(lin.sense, ["<=", "=="]), b_, 1e20)
    groups = {}
    for r, src in enumerate(lin.row_source):
        groups.setdefault(src, []).append(r)
    groups = {k: np.asarray(v) for k, v in groups.items()}
    cons, cols = _parity_core(A, lo, hi, np.ones(8, dtype=bool), groups)
    assert sorted(m._constraints[i].name for i in cons) == ["b0", "b1", "b2"]
    assert cols == {5, 6, 7}


def test_parity_confirms_a_set_beyond_the_exact_caps_without_presolve_off(monkeypatch):
    """Checkpoint 0 on a 45-binary odd cycle (past EXACT_MAX_INTS, so only
    parity can decide it exactly) is confirmed by the certificate alone."""
    m = _odd_cycle(45, noise=0)
    solves: list[str] = []
    real_solve = _iis_milp._Engine._solve

    def solve(self, presolve, stop_at=None):
        solves.append(presolve)
        return real_solve(self, presolve, stop_at)

    first: dict = {}
    real_confirm = _iis_milp._Engine.confirm_infeasible

    def confirm(self, *a, **k):
        n0 = len(solves)
        verdict = real_confirm(self, *a, **k)
        first.setdefault("verdict", verdict)
        first.setdefault("solves", solves[n0:])
        return verdict

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    monkeypatch.setattr(_iis_milp._Engine, "confirm_infeasible", confirm)
    res = milp_iis(m, total_time_limit=60.0)
    assert first["verdict"] is True and first["solves"] == []  # checkpoint 0
    assert res.complete and res.n_exact_confirmed >= 1


def test_non_integer_data_is_not_eligible():
    m = dm.Model("frac")
    x = m.integer("x", shape=(2,), lb=0, ub=3)
    m.subject_to(1.5 * x[0] + x[1] == 1, name="frac")  # fractional coefficient
    m.subject_to(x[0] + x[1] == 0.5, name="half_rhs")  # fractional rhs
    m.minimize(0 * x[0])
    assert milp_iis(m).parity_rows is None
