"""Entry experiment for issue #677 — falsifying the joint-correlation
(cross-square) lever for certifying the autocorr/LABS class.

#673 showed that pairwise `z`-polytope cuts (triangle/PSD/RLT) leave the
reformed-autocorr root bound at the parity floor. #677 re-scoped to the only
remaining lever: coupling the *joint* realization of the correlation vector
`(C_1,…,C_K)`, via the degree-4 moment (Lasserre level-2) relaxation.

This probe solves the **combined** relaxation, which is the strongest form of the
idea — the level-2 moment matrix ``M(y) ⪰ 0`` over ``s in {±1}`` (which sees the
cross-square degree-4 couplings) PLUS the per-square parity secant cuts
``t_k ≥ (u+v)y_k − uv`` (the integrality structure that gives the reformed model
its floor). Pseudo-moments need not have ``(E[C_k], E[C_k²])`` inside the integer
hull, so the secant cuts genuinely add to the SDP.

  y_S     : moment of squarefree monomial  s_S = ∏_{i∈S} s_i  (|S| ≤ 4), y_∅ = 1
  M[S,T]  = y_{S△T}  over degree-≤2 monomials S,T  (the level-2 moment matrix)
  y_k = E[C_k]   = Σ_i y_{{i,i+k}}
  t_k = E[C_k²]  = Σ_{i,i'} y_{{i,i+k}△{i',i'+k}}
  minimize Σ_k t_k  s.t.  M ⪰ 0,  secant(t_k, y_k),  |y_S| ≤ 1

Result (recorded in ``docs/dev/performance-plan.md`` §6, 2026-07-17): the bound
moves above the floor only at small n and the movement **decays to zero by the
target scale n=25** (11.9999 vs floor 12; optimum 36), while the dense SDP
(326×326, ~2 min for a single root relaxation) is intractable in-solver. The
LABS Fourier sum-rule (#677 direction 2) is a single ``vᵀMv ≥ 0`` cut subsumed by
the level-1 PSD closure #673 already found inert.

Requires ``cvxpy`` + an SDP solver (``pip install cvxpy clarabel``); prints a note
and exits if unavailable. Small n solve in seconds; n≥20 take minutes (SCS).
Run: ``python discopt_benchmarks/scripts/joint_correlation_moment_probe.py``
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import scipy.sparse as sp


def parity_floor(n: int, K: int) -> int:
    """|C_k| ≥ 1 whenever (n-k) is odd; the floor is the count of such lags."""
    return sum(1 for k in range(1, K + 1) if (n - k) % 2 == 1)


def brute_optimum(n: int, K: int) -> int:
    def energy(bits):
        s = [2 * x - 1 for x in bits]
        return sum(sum(s[i] * s[i + k] for i in range(n - k)) ** 2 for k in range(1, K + 1))

    return min(energy(bits) for bits in itertools.product([0, 1], repeat=n))


def combined_level2_secant(n: int, K: int, solver: str = "CLARABEL"):
    """Combined level-2 moment SDP + parity secant cuts; returns (value, status,
    solve_seconds, matrix_dim, n_moments) or a reason string if cvxpy absent."""
    try:
        import cvxpy as cp
    except ImportError:
        return None

    # degree-≤2 squarefree monomials index the moment matrix
    B = [frozenset()]
    B += [frozenset((i,)) for i in range(n)]
    B += [frozenset((i, j)) for i, j in itertools.combinations(range(n), 2)]
    nb = len(B)

    mom = sorted(
        {B[a] ^ B[b] for a in range(nb) for b in range(nb)} - {frozenset()},
        key=lambda s: (len(s), tuple(sorted(s))),
    )
    idx = {U: j for j, U in enumerate(mom)}
    n_mom = len(mom)

    # sparse moment -> matrix-entry map (empty diff -> identity constant)
    rows, cols = [], []
    for a in range(nb):
        for b in range(nb):
            U = B[a] ^ B[b]
            if U:
                rows.append(a * nb + b)
                cols.append(idx[U])
    amap = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nb * nb, n_mom))
    i_flat = np.eye(nb).reshape(-1)

    y = cp.Variable(n_mom)
    matrix = cp.reshape(amap @ y + i_flat, (nb, nb), order="C")
    cons = [matrix >> 0, y <= 1, y >= -1]

    obj_coef = np.zeros(n_mom)
    obj_const = 0.0
    for k in range(1, K + 1):
        yk = np.zeros(n_mom)
        for i in range(n - k):
            yk[idx[frozenset((i, i + k))]] += 1.0
        tk = np.zeros(n_mom)
        tk_const = 0.0
        for i in range(n - k):
            for ip in range(n - k):
                U = frozenset((i, i + k)) ^ frozenset((ip, ip + k))
                if U:
                    tk[idx[U]] += 1.0
                else:
                    tk_const += 1.0
        obj_coef += tk
        obj_const += tk_const
        rng = n - k
        c0 = (n - k) % 2  # parity of the C_k value
        g = 2
        lo = -rng + ((c0 + rng) % g)
        hi = rng - ((rng - c0) % g)
        for u in range(lo, hi, g):
            v = u + g
            cons.append((tk - (u + v) * yk) @ y + (tk_const + u * v) >= 0)

    prob = cp.Problem(cp.Minimize(obj_coef @ y + obj_const), cons)
    t0 = time.time()
    prob.solve(solver=getattr(cp, solver))
    return prob.value, prob.status, time.time() - t0, nb, n_mom


def main():
    print("Combined level-2 moment + parity-secant bound vs parity floor (issue #677)\n")
    # small n with the exact solver; larger n need SCS and are slow (documented).
    plan = [(6, "CLARABEL"), (8, "CLARABEL"), (10, "CLARABEL"), (13, "CLARABEL"), (20, "SCS")]
    for n, solver in plan:
        K = n - 1
        out = combined_level2_secant(n, K, solver)
        if out is None:
            print("cvxpy not installed — `pip install cvxpy clarabel` to run this probe.")
            return
        val, status, dt, nb, n_mom = out
        pf = parity_floor(n, K)
        opt = brute_optimum(n, K) if n <= 18 else None
        moved = "MOVED" if val is not None and val > pf + 1e-3 else "flat"
        shown = None if val is None else round(val, 3)
        print(
            f"n={n:2d}: parity_floor={pf:2d}  level2+secant={shown!s:>7}"
            f"  opt={opt}  [{moved}]  ({status}, {nb}x{nb}, {dt:.1f}s, {solver})"
        )
    print(
        "\nn=25 (the target, SCS eps=1e-6, ~2 min): "
        "parity_floor=12, level2+secant=11.9999, opt=36.\n"
        "The movement above the floor decays to zero by n≈13 and is nil at n=25, while the\n"
        "dense 326x326 SDP is intractable in-solver. Both #677 directions fail the kill\n"
        "criterion — see docs/dev/performance-plan.md §6 (2026-07-17)."
    )


if __name__ == "__main__":
    main()
