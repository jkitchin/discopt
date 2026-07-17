"""Pin the issue #673 falsification: strengthening the lifted binary-product
(Boolean quadric) ``z``-polytope does NOT move the reformed-autocorr root LP
bound off the parity floor.

Issue #673 proposed BQP triangle inequalities, PSD moment cuts on ``[1 b; bᵀ Z]``
(``Z_ii = b_i``), and square-linkage RLT to certify the ``autocorr_bern*`` class
on the binary-multilinear MILP route. The entry experiment
(``discopt_benchmarks/scripts/bqp673_zpolytope_falsification.py``, recorded in
``docs/dev/performance-plan.md`` §6, 2026-07-17) measured the *full closure* of
the two cheapest families and found the bound unchanged at the parity floor —
the looseness is the decoupling across the squared correlations, not the
pairwise ``z``-polytope. These tests lock that result so the falsified direction
is not silently re-attempted, and double as a soundness pin (the triangle
inequalities are valid — they never cut a genuine 0/1 point).
"""

import itertools
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt import Model
from discopt._jax import binary_multilinear_reform as B
from discopt._jax.problem_classifier import extract_lp_data

pytestmark = pytest.mark.relaxation

linprog = pytest.importorskip("scipy.optimize").linprog


def _build_autocorr(n, K):
    m = Model(name=f"autocorr{n}_{K}")
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(n)]
    s = [2 * bi - 1 for bi in b]
    E = None
    for k in range(1, K + 1):
        Ck = None
        for i in range(n - k):
            t = s[i] * s[i + k]
            Ck = t if Ck is None else Ck + t
        term = Ck * Ck
        E = term if E is None else E + term
    m.minimize(E)
    return m, b


def _parity_floor(n, K):
    # |C_k| >= 1 whenever (n - k) is odd; the bound is the count of such lags.
    return sum(1 for k in range(1, K + 1) if (n - k) % 2 == 1)


def _pair_columns(reformed):
    spec = getattr(reformed, "_bml_aux_spec")
    n0 = getattr(reformed, "_bml_n_orig_flat")
    return {
        frozenset(entry[1]): n0 + k
        for k, entry in enumerate(spec)
        if entry[0] == "z" and len(entry[1]) == 2
    }


def _bounds(lp):
    return [
        (None if lo <= -1e19 else lo, None if hi >= 1e19 else hi)
        for lo, hi in zip(np.asarray(lp.x_l, float), np.asarray(lp.x_u, float))
    ]


def _lp_bound(lp, A_ub=None, b_ub=None):
    kw = dict(
        c=np.asarray(lp.c, float),
        A_eq=np.asarray(lp.A_eq, float),
        b_eq=np.asarray(lp.b_eq, float),
        bounds=_bounds(lp),
        method="highs",
    )
    if A_ub is not None and len(A_ub):
        kw["A_ub"] = np.asarray(A_ub, float)
        kw["b_ub"] = np.asarray(b_ub, float)
    res = linprog(**kw)
    assert res.success, res.message
    return res.fun + float(getattr(lp, "obj_const", 0.0)), res.x


def _triangle_rows(reformed, ncols):
    pair = _pair_columns(reformed)
    bits = sorted(getattr(reformed, "_bml_binary_flat"))
    A, b = [], []

    def row(coefs, rhs):
        r = np.zeros(ncols)
        for c, v in coefs.items():
            r[c] += v
        A.append(r)
        b.append(rhs)

    for i, j, k in itertools.combinations(bits, 3):
        pij, pik, pjk = (
            pair.get(frozenset((i, j))),
            pair.get(frozenset((i, k))),
            pair.get(frozenset((j, k))),
        )
        if pij is None or pik is None or pjk is None:
            continue
        row({pij: 1, pik: 1, pjk: -1, i: -1}, 0.0)
        row({pij: 1, pjk: 1, pik: -1, j: -1}, 0.0)
        row({pik: 1, pjk: 1, pij: -1, k: -1}, 0.0)
        row({i: 1, j: 1, k: 1, pij: -1, pik: -1, pjk: -1}, 1.0)
    return (np.array(A), np.array(b)) if A else (None, None)


@pytest.mark.parametrize("n,K", [(6, 5), (8, 7), (10, 9)])
def test_reformed_autocorr_root_bound_is_the_parity_floor(n, K):
    """The binary-multilinear-reformed autocorr root LP bound equals the analytic
    parity floor — the phenomenon issue #673 set out to defeat."""
    m, _ = _build_autocorr(n, K)
    reformed = B.reformulate_binary_multilinear(m)
    assert reformed is not m
    lp = extract_lp_data(reformed)
    base, _ = _lp_bound(lp)
    assert base == pytest.approx(_parity_floor(n, K), abs=1e-6)


@pytest.mark.parametrize("n,K", [(6, 5), (8, 7), (10, 9)])
def test_bqp_triangle_closure_does_not_move_the_bound(n, K):
    """Issue #673 direction #1 is falsified: adding the FULL family of Padberg
    triangle inequalities over the recorded pairwise-product columns and
    re-solving to closure leaves the root bound exactly at the parity floor.
    (Bound-changing gate, reversed outcome: new bound == old bound.)"""
    m, _ = _build_autocorr(n, K)
    reformed = B.reformulate_binary_multilinear(m)
    lp = extract_lp_data(reformed)
    N = len(lp.c)
    base, _ = _lp_bound(lp)
    A, b = _triangle_rows(reformed, N)
    assert A is not None and len(A) > 0, "expected pairwise columns to form triangles"
    tightened, _ = _lp_bound(lp, A, b)
    # The differential-bound invariant still holds (never weaker) ...
    assert tightened >= base - 1e-6
    # ... but the closure is inert here: no movement off the parity floor.
    assert tightened == pytest.approx(base, abs=1e-6)


def test_triangle_inequalities_are_valid_no_integer_point_cut():
    """Soundness pin: the triangle inequalities never cut a genuine 0/1 point
    (z = the true product), so if a future change DOES adopt them the guard here
    proves they are valid cuts on the Boolean quadric polytope."""
    n, K = 6, 5
    m, _ = _build_autocorr(n, K)
    reformed = B.reformulate_binary_multilinear(m)
    lp = extract_lp_data(reformed)
    N = len(lp.c)
    A, b = _triangle_rows(reformed, N)
    assert A is not None
    spec = getattr(reformed, "_bml_aux_spec")
    n0 = getattr(reformed, "_bml_n_orig_flat")
    bits = sorted(getattr(reformed, "_bml_binary_flat"))
    for assignment in itertools.product([0, 1], repeat=len(bits)):
        x = np.zeros(N)
        val = {col: v for col, v in zip(bits, assignment)}
        for col, v in val.items():
            x[col] = v
        # fill every recorded aux column with its true value at this vertex
        for k, entry in enumerate(spec):
            if entry[0] == "z":
                x[n0 + k] = float(np.prod([val[c] for c in entry[1]]))
        # each row is  coeffs · x <= rhs  ; a valid cut never violates it
        assert np.all(A @ x <= np.asarray(b) + 1e-9)
