"""Strong (ZKRW-style) Shor SDP root bound for a constrained **binary** quadratic
program (issue #661).

McCormick envelopes on an indefinite ``x'Qx`` are trivially loose (~0 on qap vs an
optimum of 388214), and the local (<=6-var) moment cuts cannot enforce the *global*
moment-matrix PSD constraint that actually binds there. This module builds the
**strong Shor semidefinite relaxation** over the moment matrix
``M = [[1, x'], [x, X]]``:

    min  <Q, X> + c·x + offset
    s.t. M >= 0 (PSD),  M_00 = 1,  diag(X) = x           (binary diagonal),
         A_eq x = b_eq,  A_ub x <= b_ub                  (model linear rows),
         A_eq_r · X[:, p] = b_r x_p   for every equality row r and variable p
                                                          (lifted-equality RLT),
         0 <= X_ij <= min(x_i, x_j),  X_ij >= x_i + x_j - 1   (McCormick box on X),
         X_ij = 0 for mutually exclusive pairs             (gangster rows),
         0 <= x <= 1.

The **plain** Shor relaxation (PSD + diagonal + assignment on ``x`` only) is
*falsified* on this class — the indefinite objective is unbounded below over the
cone ``X >= x x'`` (measured: qap unbounded, synthetic QAPs negative, i.e. worse
than McCormick; ``docs/dev/issue-661-qap-sdp-entry-experiment-2026-07-17.md``).
The bound comes from the SDP *plus* the lifted-equality RLT rows, the McCormick
box on ``X``, and the gangster rows — all derived from structure the RLT-1 path
already extracts, stated generally over linear equalities (no problem-name keying,
CLAUDE.md §2). Measured entry experiment: qap root 377098 = 97.1 % of the optimum
(vs McCormick ~0, RLT-1 LP 352891, published dual 149106); exact on brute-forced
synthetic Koopmans–Beckmann QAPs.

Solved with a first-order conic solver (SCS — interior-point does not converge at
qap's 226-dim moment matrix; optional dependency ``discopt[sdp]``). The SCS value
is a first-order *approximate* optimum, *not* a certificate, so the surfaced bound
is never the solver objective: it is the **safe dual bound** recomputed from the
returned dual multipliers (:func:`shor_sdp_safe_dual_bound`) — the SDP analogue of
the Neumaier–Shcherbina safe LP bound already used on the RLT-1 LP. For *any*
multipliers ``y1`` (free, on equality rows) and ``y2 >= 0`` (on ``<=`` rows), weak
duality gives, with ``S = C + sum_k y1_k A_k + sum_l y2_l G_l``,

    <C, M> >= -(b_eq·y1 + h·y2) + min(0, lambda_min(S)) * tr_ub(M)

for every feasible lifted point (``M >= 0``, ``tr(M) = 1 + sum_i x_i <= n+1``), so
a clamped, margin-padded evaluation is a rigorous global lower bound at any solver
accuracy — no convergence is needed for *soundness*, only for tightness. Gated by
:class:`~discopt.solver_tuning.SolverTuning` (``DISCOPT_SHOR_SDP_ROOT_BOUND``,
default off; root-only, §5 bound-changing) and a moment-dimension size guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

__all__ = [
    "ShorSDPProblem",
    "build_shor_sdp",
    "shor_sdp_lower_bound",
    "shor_sdp_safe_dual_bound",
]

_SQRT2 = float(np.sqrt(2.0))


def _svec_index(d: int, i: int, j: int) -> int:
    """Index of entry ``(i, j)`` in the lower-triangular column-major stacking of a
    ``d x d`` symmetric matrix (the SCS PSD-cone vectorization order)."""
    c, r = (i, j) if i <= j else (j, i)
    return c * d - (c * (c - 1)) // 2 + (r - c)


class _RowAccum:
    """Accumulate sparse constraint rows over the svec coordinates of ``M``.

    ``add(entries, rhs)`` takes ``entries = [(i, j, w), ...]`` meaning the row
    functional ``sum w * M_ij`` (each symmetric pair named once, either order) and
    encodes it in the scaled-lower-triangular (svec) basis SCS uses: a diagonal
    entry keeps coefficient ``w``; an off-diagonal svec coordinate stores
    ``sqrt(2) * M_ij``, so the row coefficient there is ``w / sqrt(2)``. Duplicate
    coordinates are summed by the COO->CSR conversion.
    """

    def __init__(self, d: int):
        self.d = d
        self.m = d * (d + 1) // 2
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.rhs: list[float] = []

    def add(self, entries: list[tuple[int, int, float]], rhs: float) -> None:
        r = len(self.rhs)
        for i, j, w in entries:
            self.rows.append(r)
            self.cols.append(_svec_index(self.d, i, j))
            self.data.append(w if i == j else w / _SQRT2)
        self.rhs.append(rhs)

    def build(self) -> tuple["sp.csr_matrix", np.ndarray]:
        A = sp.csr_matrix((self.data, (self.rows, self.cols)), shape=(len(self.rhs), self.m))
        A.sort_indices()
        return A, np.asarray(self.rhs, dtype=np.float64)


@dataclass
class ShorSDPProblem:
    """The assembled strong-Shor SDP in svec coordinates of the moment matrix.

    The conic variable is ``u = svec(M)`` (lower-triangular column-major,
    off-diagonals scaled by ``sqrt(2)``) of the ``dim x dim`` moment matrix
    ``M = [[1, x'], [x, X]]`` with ``dim = n + 1``. The relaxation is

        min c_svec·u + offset  s.t.  A_eq u = b_eq,  A_in u <= h,  M(u) >= 0.

    ``offset`` recovers the user objective scale.
    """

    c_svec: np.ndarray
    A_eq: "sp.csr_matrix"
    b_eq: np.ndarray
    A_in: "sp.csr_matrix"
    h: np.ndarray
    dim: int
    n: int
    offset: float

    @property
    def m(self) -> int:
        """Length of the svec variable vector."""
        return self.dim * (self.dim + 1) // 2

    def pack_point(self, x: np.ndarray) -> np.ndarray:
        """Lift an original-variable point ``x`` to ``u = svec(M)`` with
        ``M = [1; x][1; x]'`` — the (rank-1, PSD) value at a genuinely feasible
        point."""
        v = np.concatenate(([1.0], np.asarray(x, dtype=np.float64)))
        u = np.zeros(self.m, dtype=np.float64)
        for c in range(self.dim):
            for r in range(c, self.dim):
                val = v[r] * v[c]
                u[_svec_index(self.dim, r, c)] = val if r == c else _SQRT2 * val
        return u

    def unsvec(self, s: np.ndarray) -> np.ndarray:
        """Symmetric ``dim x dim`` matrix whose svec is ``s`` (exact inverse of the
        scaled-lower-triangular stacking)."""
        S = np.zeros((self.dim, self.dim), dtype=np.float64)
        for c in range(self.dim):
            for r in range(c, self.dim):
                val = float(s[_svec_index(self.dim, r, c)])
                if r == c:
                    S[r, c] = val
                else:
                    S[r, c] = S[c, r] = val / _SQRT2
        return S


def build_shor_sdp(
    model,
    relax,
    info: dict,
    *,
    binary_vars: frozenset,
    max_dim: int = 400,
) -> Optional[ShorSDPProblem]:
    """Assemble the strong-Shor SDP for a constrained binary QP, or ``None``.

    ``None`` is returned (a sound no-op upstream) on any ineligibility: no binary
    variables, no quadratic objective terms, an objective that is not a pure
    quadratic in the original variables, no linear equality constraints (the
    lifted-equality RLT rows — the load-bearing part of the *strong* Shor bound,
    without which the relaxation is unbounded on this class — need equalities), a
    moment dimension ``n + 1`` above ``max_dim``, or **any non-binary variable**.
    The all-binary gate is stricter than the RLT-1 LP's (which only requires the
    objective quadratic's support to be binary) because three ingredients here
    lean on every lifted variable being 0/1: the moment diagonal ``X_ii = x_i``,
    the ``X_pp = x_p`` substitution in every lifted-equality row, and the trace
    cap ``tr(M) <= n + 1`` used by the safe dual bound.
    """
    from discopt._jax.obbt import _extract_linear_constraints
    from discopt._jax.rlt import (
        _mutually_exclusive_pairs,
        _reconstruct_quadratic_objective,
    )

    if not binary_vars:
        return None
    if not info.get("bilinear"):
        return None  # no quadratic objective terms -> nothing to strengthen
    if not getattr(relax, "_objective_bound_valid", False):
        return None

    A_ub_m, b_ub_m, A_eq_m, b_eq_m, n = _extract_linear_constraints(model)
    if A_eq_m is None or A_eq_m.shape[0] == 0:
        return None
    if n + 1 > max_dim:
        return None
    if any(v not in binary_vars for v in range(n)):
        return None

    recon = _reconstruct_quadratic_objective(relax, info, n)
    if recon is None:
        return None
    Q, c_lin, offset = recon

    d = n + 1
    excluded = _mutually_exclusive_pairs(A_eq_m, b_eq_m, binary_vars, n)

    eq = _RowAccum(d)
    ineq = _RowAccum(d)

    # Homogenization corner and binary moment diagonal.
    eq.add([(0, 0, 1.0)], 1.0)  # M_00 = 1
    for i in range(n):
        eq.add([(i + 1, i + 1, 1.0), (0, i + 1, -1.0)], 0.0)  # X_ii = x_i

    # Model linear rows on x (x_i = M[0, i+1]).
    A_eq_c = sp.csr_matrix(A_eq_m)
    b_eq_c = np.asarray(b_eq_m, dtype=np.float64)
    for r in range(A_eq_c.shape[0]):
        s, e = A_eq_c.indptr[r], A_eq_c.indptr[r + 1]
        terms = [(0, int(A_eq_c.indices[t]) + 1, float(A_eq_c.data[t])) for t in range(s, e)]
        if terms:
            eq.add(terms, float(b_eq_c[r]))
    if A_ub_m is not None and b_ub_m is not None and A_ub_m.shape[0] > 0:
        A_ub_c = sp.csr_matrix(A_ub_m)
        b_ub_c = np.asarray(b_ub_m, dtype=np.float64)
        for r in range(A_ub_c.shape[0]):
            s, e = A_ub_c.indptr[r], A_ub_c.indptr[r + 1]
            terms = [(0, int(A_ub_c.indices[t]) + 1, float(A_ub_c.data[t])) for t in range(s, e)]
            if terms:
                ineq.add(terms, float(b_ub_c[r]))

    # Gangster rows: X_ij identically 0 for mutually exclusive pairs (set-partitioning
    # exclusivity, computed generally from the equality rows — see
    # ``rlt._mutually_exclusive_pairs``).
    for i, j in sorted(excluded):
        eq.add([(i + 1, j + 1, 1.0)], 0.0)

    # Lifted-equality RLT rows: (a·x = beta) * x_p -> sum_k a_k X_{p,k} = beta x_p,
    # using X_pp = x_p. Terms on gangster-pinned pairs drop out (their product is
    # identically 0 and the gangster row pins the entry). A row that reduces to
    # ``coef·x_p = 0`` with ``coef ~ 0`` is a trivial identity — dropped.
    for r in range(A_eq_c.shape[0]):
        s, e = A_eq_c.indptr[r], A_eq_c.indptr[r + 1]
        supp = [(int(A_eq_c.indices[t]), float(A_eq_c.data[t])) for t in range(s, e)]
        beta = float(b_eq_c[r])
        if not supp:
            continue
        for p in range(n):
            lift_terms: list[tuple[int, int, float]] = []
            coef_xp = -beta
            for k, a_k in supp:
                if k == p:
                    coef_xp += a_k  # X_pp = x_p
                elif (min(p, k), max(p, k)) not in excluded:
                    lift_terms.append((p + 1, k + 1, a_k))
            if not lift_terms and abs(coef_xp) <= 1e-12:
                continue
            lift_terms.append((0, p + 1, coef_xp))
            eq.add(lift_terms, 0.0)

    # Bounds 0 <= x <= 1 and the McCormick box on X. For an excluded pair the box
    # collapses to its one non-redundant face x_i + x_j <= 1 (from
    # X_ij >= x_i + x_j - 1 with X_ij = 0); the other faces are implied by x >= 0.
    for i in range(n):
        ineq.add([(0, i + 1, -1.0)], 0.0)  # x_i >= 0
        ineq.add([(0, i + 1, 1.0)], 1.0)  # x_i <= 1
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                ineq.add([(0, i + 1, 1.0), (0, j + 1, 1.0)], 1.0)
                continue
            ineq.add([(i + 1, j + 1, -1.0)], 0.0)  # X_ij >= 0
            ineq.add([(i + 1, j + 1, 1.0), (0, i + 1, -1.0)], 0.0)  # X_ij <= x_i
            ineq.add([(i + 1, j + 1, 1.0), (0, j + 1, -1.0)], 0.0)  # X_ij <= x_j
            ineq.add(
                [(i + 1, j + 1, -1.0), (0, i + 1, 1.0), (0, j + 1, 1.0)], 1.0
            )  # X_ij >= x_i + x_j - 1

    # Objective <C, M> = c_lin·x + <Q, X> in the same svec basis.
    obj = _RowAccum(d)
    obj_terms: list[tuple[int, int, float]] = []
    for i in range(n):
        w = Q[i, i]
        if w != 0.0:
            obj_terms.append((i + 1, i + 1, float(w)))
        if c_lin[i] != 0.0:
            obj_terms.append((0, i + 1, float(c_lin[i])))
        for j in range(i + 1, n):
            w = 2.0 * Q[i, j]
            if w != 0.0:
                obj_terms.append((i + 1, j + 1, float(w)))
    obj.add(obj_terms, 0.0)
    A_obj, _ = obj.build()
    c_svec = np.asarray(A_obj.todense()).ravel()

    A_eq_s, b_eq_s = eq.build()
    A_in_s, h_s = ineq.build()
    return ShorSDPProblem(
        c_svec=c_svec,
        A_eq=A_eq_s,
        b_eq=b_eq_s,
        A_in=A_in_s,
        h=h_s,
        dim=d,
        n=n,
        offset=offset,
    )


def shor_sdp_safe_dual_bound(prob: ShorSDPProblem, y: np.ndarray) -> Optional[float]:
    """Rigorous global lower bound from (arbitrary) dual multipliers ``y``.

    ``y`` is the leading ``n_eq + n_in`` block of a conic dual vector: ``y1``
    (equality rows, free sign) then ``y2`` (``<=`` rows, clamped to ``>= 0`` here
    — clamping picks a different valid multiplier, it never breaks validity).
    With ``S = C + sum_k y1_k A_k + sum_l y2_l G_l`` (assembled exactly in the
    shared svec basis), weak duality gives for every feasible lifted point
    (``M >= 0``, all rows satisfied, ``tr(M) = 1 + sum x_i <= dim``):

        <C, M> = <S, M> - sum y1_k <A_k, M> - sum y2_l <G_l, M>
              >= <S, M> - b_eq·y1 - h·y2
              >= min(0, lambda_min(S)) * dim - b_eq·y1 - h·y2 ,

    using ``tr(S M) >= lambda_min(S) tr(M)`` for ``M >= 0``. So the value holds
    for *any* ``y`` — solver convergence and solver dual conventions affect only
    tightness, never soundness (the Neumaier–Shcherbina property, lifted to SDP).

    Numerics are plain float64 with magnitude-scaled safety margins (the same
    epistemic class as ``obbt._ns_safe_lp_lower_bound``): the eigenvalue margin
    dominates both the accumulation error of assembling ``S`` (bounded via the
    absolute-value accumulation ``S_abs``) and the symmetric-eigensolver backward
    error (``O(dim * eps * ||S||)``); a second margin covers the dual constant.
    Returns ``None`` when the evaluation is non-finite.
    """
    n_eq = int(prob.b_eq.shape[0])
    n_in = int(prob.h.shape[0])
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.shape[0] < n_eq + n_in or not np.all(np.isfinite(y[: n_eq + n_in])):
        return None
    y1 = y[:n_eq]
    y2 = np.clip(y[n_eq : n_eq + n_in], 0.0, None)

    s_svec = prob.c_svec + prob.A_eq.T @ y1 + prob.A_in.T @ y2
    S = prob.unsvec(s_svec)
    # Magnitude gauge for the float margins: the absolute-value accumulation is an
    # upper bound on every |contribution| summed into any entry of S, so
    # ``k * eps * max(S_abs)`` dominates the dot-product rounding error at any
    # cancellation, with k <= the number of rows. 1e-9 * dim >> k * eps here.
    s_abs = np.abs(prob.c_svec) + abs(prob.A_eq).T @ np.abs(y1) + abs(prob.A_in).T @ y2
    margin_eig = 1e-9 * prob.dim * (1.0 + float(s_abs.max(initial=0.0)))
    lam_min = float(np.linalg.eigvalsh(S)[0])
    lam_safe = min(0.0, lam_min - margin_eig)

    const = -(float(prob.b_eq @ y1) + float(prob.h @ y2))
    margin_c = 1e-10 * (
        1.0 + float(np.abs(prob.b_eq) @ np.abs(y1)) + float(np.abs(prob.h) @ y2) + abs(const)
    )
    g = const + lam_safe * prob.dim - margin_c
    if not np.isfinite(g):
        return None
    return float(g) + prob.offset


def shor_sdp_lower_bound(
    model,
    relax,
    info: dict,
    *,
    binary_vars: frozenset,
    time_limit: Optional[float] = 120.0,
    max_dim: int = 400,
    eps: float = 1e-5,
    max_iters: int = 100_000,
) -> tuple[Optional[float], int]:
    """Rigorous strong-Shor SDP lower bound for a constrained binary QP, or a
    sound no-op.

    Builds the strong-Shor SDP (:func:`build_shor_sdp`), solves it with the
    first-order conic solver SCS (optional dependency ``discopt[sdp]``; a missing
    or failing solver is a sound no-op), and returns the **safe dual bound**
    recomputed from the returned multipliers (:func:`shor_sdp_safe_dual_bound`)
    — never the solver's approximate objective. Returns ``(bound, dim)`` where
    ``dim = n + 1`` is the moment-matrix dimension (``0`` when ineligible);
    ``bound`` is ``None`` on any ineligibility or failure.
    """
    prob = build_shor_sdp(model, relax, info, binary_vars=binary_vars, max_dim=max_dim)
    if prob is None:
        return (None, 0)
    try:
        import scs
    except ImportError:
        return (None, prob.dim)

    m = prob.m
    A = sp.vstack([prob.A_eq, prob.A_in, -sp.identity(m, format="csr")], format="csc")
    b = np.concatenate([prob.b_eq, prob.h, np.zeros(m)])
    cone = {"z": int(prob.b_eq.shape[0]), "l": int(prob.h.shape[0]), "s": [prob.dim]}
    settings: dict = {
        "verbose": False,
        "eps_abs": float(eps),
        "eps_rel": float(eps),
        "max_iters": int(max_iters),
    }
    if time_limit is not None and np.isfinite(time_limit):
        settings["time_limit_secs"] = max(1.0, float(time_limit))
    try:
        solver = scs.SCS({"A": A, "b": b, "c": prob.c_svec}, cone, **settings)
        sol = solver.solve()
    except Exception:
        return (None, prob.dim)
    status = str(sol.get("info", {}).get("status", "")).lower()
    y = sol.get("y")
    # An infeasible/unbounded verdict returns a certificate ray, not a dual
    # iterate; the safe bound would still be *valid* for it, but it carries no
    # information — decline. (The relaxation is infeasible only if the model is.)
    if y is None or not status.startswith("solved"):
        return (None, prob.dim)
    g = shor_sdp_safe_dual_bound(prob, np.asarray(y, dtype=np.float64))
    if g is None or not np.isfinite(g):
        return (None, prob.dim)
    return (float(g), prob.dim)
