"""Exact second route for the IIS engine's confirmation (``discopt._iis_exact``).

On big-M sets HiGHS with presolve off cannot confirm infeasibility. On MIPcc27's
``neos-5115982-kedron`` the 12-element reference IIS is two binaries forced to 0
by ``x - 8e6*z >= -7999999.656`` with ``ub(x) = 0.34375`` (2.5e-4 of slack), then
a 4-row LP contradiction. Presolve-off branch-and-bound returns ``Optimal`` with
the binaries at ``1 - 3e-11``, and rounding breaks the rows. The engine therefore
raised ``TimeoutError`` 0.08 s into a 60 s budget, or ``ValueError`` without one.
The set is now decided in exact rational arithmetic instead.
"""

import itertools
import random
import time
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._iis_exact import exact_feasible
from discopt._iis_milp import milp_iis

highspy = pytest.importorskip("highspy")


def _kedron_like():
    m = dm.Model("kedron_like")
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
    m.minimize(0 * w)
    return m


def _presolve_off_status(model):
    from discopt.decomposition._linear import extract_linear

    lin = extract_linear(model)
    n = lin.n
    lb = np.concatenate([np.atleast_1d(np.asarray(v.lb, float)) for v in model._variables])
    ub = np.concatenate([np.atleast_1d(np.asarray(v.ub, float)) for v in model._variables])
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("presolve", "off")
    h.addVars(n, lb, ub)
    integ = [v.var_type.name in ("BINARY", "INTEGER") for v in model._variables]
    h.changeColsIntegrality(
        n,
        np.arange(n, dtype=np.int32),
        np.array(
            [
                highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
                for i in integ
            ]
        ),
    )
    A = sp.csr_matrix(lin.A)
    inf = highspy.kHighsInf
    for r in range(lin.m):
        s, e = A.indptr[r], A.indptr[r + 1]
        lo = lin.b[r] if lin.sense[r] in (">=", "==") else -inf
        hi = lin.b[r] if lin.sense[r] in ("<=", "==") else inf
        h.addRow(lo, hi, e - s, A.indices[s:e].astype(np.int32), A.data[s:e])
    h.run()
    return h.getModelStatus()


@pytest.mark.smoke
def test_big_m_set_presolve_off_cannot_confirm_is_decided_exactly():
    m = _kedron_like()
    # The premise: presolve-off HiGHS calls this infeasible set "Optimal".
    assert _presolve_off_status(m) == highspy.HighsModelStatus.kOptimal
    for tl in (None, 60.0):
        res = milp_iis(m, total_time_limit=tl)
        assert res.complete and res.proven_irreducible and not res.fallback
        assert (len(res.constraints), len(res.bounds), len(res.integrality)) == (6, 4, 2)
        assert len(res.certificates) == 12
        assert res.n_exact_confirmed >= 1 and res.n_exact_refuted == 0


def _brute(A, rlo, rhi, lb, ub, held):
    """Independent oracle for pure-integer sets: enumerate every point exactly."""
    doms = [range(int(np.ceil(lb[j])), int(np.floor(ub[j])) + 1) for j in range(lb.size)]
    Ad = A.toarray()
    for x in itertools.product(*doms):
        act = [
            sum(Fraction(float(Ad[r, j])) * x[j] for j in range(lb.size))
            for r in range(Ad.shape[0])
        ]
        if all(
            (not np.isfinite(rlo[r]) or act[r] >= Fraction(float(rlo[r])))
            and (not np.isfinite(rhi[r]) or act[r] <= Fraction(float(rhi[r])))
            for r in range(Ad.shape[0])
        ):
            return False
    return True


def _random_set(rng, n_int, n_cont, m):
    n = n_int + n_cont
    A = sp.csr_matrix(np.array([[rng.randint(-3, 3) for _ in range(n)] for _ in range(m)], float))
    rlo = np.array([rng.choice([-np.inf, rng.randint(-4, 4)]) for _ in range(m)], float)
    rhi = np.array([rng.choice([np.inf, rng.randint(-4, 4)]) for _ in range(m)], float)
    lb = np.array([rng.randint(-2, 0) for _ in range(n)], float)
    ub = np.array([rng.randint(0, 2) for _ in range(n)], float)
    return A, rlo, rhi, lb, ub, np.arange(n_int)


@pytest.mark.smoke
def test_pure_integer_sets_agree_with_exhaustive_enumeration():
    rng = random.Random(3)
    checked = infeasible = 0
    for _ in range(150):
        A, rlo, rhi, lb, ub, held = _random_set(rng, rng.randint(1, 4), 0, rng.randint(1, 4))
        got = exact_feasible(A, rlo, rhi, lb, ub, held)
        assert got == _brute(A, rlo, rhi, lb, ub, held), (A.toarray(), rlo, rhi, lb, ub)
        checked += 1
        infeasible += got
    assert checked == 150 and 20 <= infeasible <= 130  # both verdicts exercised


def test_mixed_sets_agree_with_highs_on_integer_data():
    # Integer data with small coefficients: every infeasibility margin is a
    # rational with a small denominator, far above HiGHS's tolerances.
    rng = random.Random(5)
    checked = infeasible = 0
    for _ in range(150):
        n_int, n_cont = rng.randint(1, 3), rng.randint(1, 3)
        A, rlo, rhi, lb, ub, held = _random_set(rng, n_int, n_cont, rng.randint(1, 4))
        n = lb.size
        h = highspy.Highs()
        h.silent()
        h.addVars(n, lb, ub)
        vt = highspy.HighsVarType
        h.changeColsIntegrality(
            n,
            np.arange(n, dtype=np.int32),
            np.array([vt.kInteger] * n_int + [vt.kContinuous] * n_cont),
        )
        inf = highspy.kHighsInf
        for r in range(A.shape[0]):
            s, e = A.indptr[r], A.indptr[r + 1]
            h.addRow(
                rlo[r] if np.isfinite(rlo[r]) else -inf,
                rhi[r] if np.isfinite(rhi[r]) else inf,
                e - s,
                A.indices[s:e].astype(np.int32),
                A.data[s:e],
            )
        h.run()
        want = h.getModelStatus() == highspy.HighsModelStatus.kInfeasible
        got = exact_feasible(A, rlo, rhi, lb, ub, held)
        assert got == want, (A.toarray(), rlo, rhi, lb, ub, n_int)
        checked += 1
        infeasible += got
    assert checked == 150 and 10 <= infeasible <= 140


def test_rational_feasible_point_is_found():
    # 3x = 1 with x continuous in [0, 1] and 2z + 3x = 1 for an integer z:
    # feasible only at x = 1/3, z = 0 -- exact, not within a tolerance.
    A = sp.csr_matrix(np.array([[0.0, 3.0], [2.0, 3.0]]))
    one = np.array([1.0, 1.0])
    assert (
        exact_feasible(A, one, one, np.array([-3.0, 0.0]), np.array([3.0, 1.0]), np.array([0]))
        is False
    )


def test_caps_and_deadline_give_undecided():
    n = 20  # more continuous columns than Fourier-Motzkin is allowed
    A = sp.csr_matrix(np.ones((1, n)))
    lb, ub = np.zeros(n), np.ones(n)
    assert (
        exact_feasible(A, np.array([30.0]), np.array([np.inf]), lb, ub, np.array([], int)) is None
    )
    # A held integer with a large domain and a deadline already passed.
    A = sp.csr_matrix(np.array([[1.0, 1.0]]))
    got = exact_feasible(
        A,
        np.array([0.5]),
        np.array([0.5]),
        np.array([0.0, 0.0]),
        np.array([100.0, 100.0]),
        np.array([0, 1]),
        stop_at=time.perf_counter() - 1,
    )
    assert got is None
