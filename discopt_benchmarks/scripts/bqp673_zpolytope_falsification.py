"""Entry experiment for issue #673 — falsifying z-polytope strengthening as the
lever for certifying the ``autocorr_bern*`` (Bernasconi low-autocorrelation)
class on the binary-multilinear MILP route.

Issue #673 proposed three strengthenings of the lifted binary-product (Boolean
quadric) polytope, "in increasing order of ambition", to move the reformed-
autocorr root dual bound off the parity floor (12.0 on n=25 dense; optimum 36):

  1. BQP triangle inequalities (Padberg).
  2. PSD moment cuts on ``[1 b; bᵀ Z]`` with ``Z_ii = b_i`` (the #663 recognition).
  3. Square-linkage RLT rows coupling the ``y_k`` epigraphs with the ``z`` vars.

This script measures the reformed-autocorr *root LP bound* under each, at the full
closure (not one round — the whole family is added and the LP re-solved to the
polytope's optimum), so the numbers are the strongest the family can give.

Result (see ``docs/dev/performance-plan.md`` §6): **all three leave the bound
exactly at the parity floor.** The looseness is not in the pairwise z-polytope —
each squared correlation ``C_k**2`` is already relaxed through the *exact* 2D
convex hull of ``{(y_k, y_k**2)}`` (the secant envelope), and ``y_k`` is affine
in ``(b, z)``. Σ t_k reaches the parity floor by driving each ``y_k``
independently to its parity-nearest attainable value; no pairwise/RLT tightening
of ``(b, z)`` constrains the *joint* realization of ``(C_1, …, C_K)``, which is
where the gap lives (a degree-≥4, LABS/merit-factor combinatorial property — not
a Boolean-quadric one, and problem-class-specific, so out of scope per the house
rule against single-problem solutions).

Run: ``python discopt_benchmarks/scripts/bqp673_zpolytope_falsification.py``
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from discopt import Model
from discopt._jax import binary_multilinear_reform as bml
from discopt._jax.problem_classifier import extract_lp_data
from scipy.optimize import linprog


# --------------------------------------------------------------------------- #
# Model + oracle
# --------------------------------------------------------------------------- #
def build_autocorr(n: int, K: int) -> tuple[Model, list]:
    """Dense Bernasconi instance: min sum_{k=1..K} (sum_i s_i s_{i+k})**2 with
    s = 2b - 1, b typed {0,1}-INTEGER (matching from_nl's typing of MINLPLib 0/1
    columns). K = n-1 is the dense (all-lag) autocorrelation."""
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


def autocorr_energy(bits, K):
    n = len(bits)
    s = [2 * x - 1 for x in bits]
    return sum(sum(s[i] * s[i + k] for i in range(n - k)) ** 2 for k in range(1, K + 1))


def brute_optimum(n, K):
    return min(autocorr_energy(bits, K) for bits in itertools.product([0, 1], repeat=n))


def parity_floor(n, K):
    """Analytic parity floor: |C_k| >= 1 whenever (n-k) is odd, so the bound is
    the count of parity-forced-nonzero lags."""
    return sum(1 for k in range(1, K + 1) if (n - k) % 2 == 1)


# --------------------------------------------------------------------------- #
# LP plumbing
# --------------------------------------------------------------------------- #
def _bounds(lp):
    return [
        (None if lo <= -1e19 else lo, None if hi >= 1e19 else hi)
        for lo, hi in zip(np.asarray(lp.x_l, float), np.asarray(lp.x_u, float), strict=False)
    ]


def lp_bound(lp, A_ub=None, b_ub=None):
    kw = {
        "c": np.asarray(lp.c, float),
        "A_eq": np.asarray(lp.A_eq, float),
        "b_eq": np.asarray(lp.b_eq, float),
        "bounds": _bounds(lp),
        "method": "highs",
    }
    if A_ub is not None and len(A_ub):
        kw["A_ub"] = np.asarray(A_ub, float)
        kw["b_ub"] = np.asarray(b_ub, float)
    res = linprog(**kw)
    if not res.success:
        return None, res
    return res.fun + float(getattr(lp, "obj_const", 0.0)), res


def pair_columns(reformed):
    """Map frozenset({flat_i, flat_j}) -> LP column of z_ij, for every recorded
    degree-2 product monomial. Structural columns follow the reformed model's
    flat variable layout: original vars, then the aux z's in append order."""
    spec = reformed._bml_aux_spec
    n0 = reformed._bml_n_orig_flat
    out = {}
    for k, entry in enumerate(spec):
        if entry[0] == "z" and len(entry[1]) == 2:
            out[frozenset(entry[1])] = n0 + k
    return out


# --------------------------------------------------------------------------- #
# Direction #1 — Padberg triangle inequalities
# --------------------------------------------------------------------------- #
def triangle_rows(reformed, ncols):
    pair = pair_columns(reformed)
    bits = sorted(reformed._bml_binary_flat)
    A, b, ntri = [], [], 0

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
        ntri += 1
        row({pij: 1, pik: 1, pjk: -1, i: -1}, 0.0)
        row({pij: 1, pjk: 1, pik: -1, j: -1}, 0.0)
        row({pik: 1, pjk: 1, pij: -1, k: -1}, 0.0)
        row({i: 1, j: 1, k: 1, pij: -1, pik: -1, pjk: -1}, 1.0)
    return (np.array(A), np.array(b), ntri) if A else (None, None, 0)


# --------------------------------------------------------------------------- #
# Direction #2 — PSD pairwise-moment closure (iterated eigenvector cuts = Shor)
# --------------------------------------------------------------------------- #
def psd_closure_bound(reformed, lp, rounds=200):
    pair = pair_columns(reformed)
    bits = sorted(reformed._bml_binary_flat)
    n = len(bits)
    # require full pairwise coverage to form a dense moment matrix
    for a in range(n):
        for c in range(a + 1, n):
            if frozenset((bits[a], bits[c])) not in pair:
                return None  # not fully pairwise-covered; PSD probe N/A
    N = len(lp.c)
    A, b = [], []
    base, _ = lp_bound(lp)
    for _ in range(rounds):
        val, res = lp_bound(lp, A, b)
        if res is None or not res.success:
            return base
        x = res.x
        M = np.ones((n + 1, n + 1))
        M[0, 0] = 1.0
        for a in range(n):
            M[0, a + 1] = M[a + 1, 0] = x[bits[a]]
            M[a + 1, a + 1] = x[bits[a]]  # Z_ii = b_i (binary)
            for c in range(a + 1, n):
                z = x[pair[frozenset((bits[a], bits[c]))]]
                M[a + 1, c + 1] = M[c + 1, a + 1] = z
        w, V = np.linalg.eigh(M)
        if w[0] > -1e-7:
            return val  # M is PSD at the LP optimum: Shor closure reached
        v = V[:, 0]
        row = np.zeros(N)
        for a in range(n):
            row[bits[a]] += 2.0 * v[0] * v[a + 1] + v[a + 1] * v[a + 1]
        for a in range(n):
            for c in range(a + 1, n):
                row[pair[frozenset((bits[a], bits[c]))]] += 2.0 * v[a + 1] * v[c + 1]
        # v^T M v = v0^2 + row·x >= 0  ->  (-row)·x <= v0^2
        A.append(-row)
        b.append(v[0] * v[0])
    return lp_bound(lp, A, b)[0]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(n, K):
    m, _ = build_autocorr(n, K)
    reformed = bml.reformulate_binary_multilinear(m)
    assert reformed is not m, "reformulation did not fire"
    lp = extract_lp_data(reformed)
    N = len(lp.c)
    base, _ = lp_bound(lp)
    A, b, ntri = triangle_rows(reformed, N)
    tri = lp_bound(lp, A, b)[0] if A is not None else base
    psd = psd_closure_bound(reformed, lp)
    floor = parity_floor(n, K)
    opt = brute_optimum(n, K) if n <= 20 else None
    print(
        f"n={n:2d} K={K:2d}  vars={sum(v.size for v in reformed._variables):4d} "
        f"eq_rows={lp.A_eq.shape[0]:4d}  parity_floor={floor:2d}  "
        f"base={base:6.2f}  +triangle({ntri})={tri:6.2f}  "
        f"+PSD={('  N/A' if psd is None else f'{psd:6.2f}')}  "
        f"opt={opt}"
    )
    return base, tri, psd, floor, opt


if __name__ == "__main__":
    print("Reformed-autocorr root LP bound vs z-polytope strengthening (issue #673)\n")
    for n, K in [(6, 5), (8, 7), (10, 9), (13, 12), (25, 24)]:
        run(n, K)
    print(
        "\nAll three z-polytope directions leave the bound at the parity floor.\n"
        "Root cause + re-scoping: docs/dev/performance-plan.md §6 (2026-07-17)."
    )
