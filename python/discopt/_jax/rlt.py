"""Reformulation-Linearization Technique (RLT level-1) lower bound for the
lifted relaxation of an indefinite **binary** quadratic program.

Term-wise McCormick / bound-factor relaxations lift each product ``x_i x_j`` to
an independent variable ``X_ij`` bounded only by the four McCormick inequalities

    X_ij >= 0,  X_ij <= x_i,  X_ij <= x_j,  X_ij >= x_i + x_j - 1     (x in [0,1]).

On an **indefinite** ``x'Qx`` those envelopes are trivially loose: the LP simply
drops every ``X_ij`` to its independent lower face, so the relaxation minimum is
~0 and fathoms nothing (the qap phenomenon, issue #661 / ``sparse-milp-plan`` T7).

RLT-1 adds the *constraint-factor* products that McCormick omits. For every
linear **equality** ``a·x = beta`` of the model and every variable ``x_p``,
multiplying the equality by ``x_p`` gives a valid linear identity on the lifted
variables

    sum_k a_k X_{p,k} = beta * x_p ,

(using the binary diagonal ``X_pp = x_p``, since ``x_p**2 = x_p``). These rows
couple the ``X_ij`` across a whole constraint and are exactly what tightens a
constrained binary QP toward its Shor/SDP bound — for the assignment-constrained
QAP they close the gap almost entirely (measured: qap root 0 -> ~352891 vs true
optimum 388214; small synthetic Koopmans-Beckmann QAPs close to the exact
optimum). Every added row is a product of valid model constraints, so the LP
minimum is a **rigorous lower bound** on the original objective — the certificate
is preserved (issue #661).

Purely LP-based: no SDP solver. Solved with the exact (vertex) simplex oracle;
the returned bound is the **Neumaier–Shcherbina safe dual bound** built from the
simplex's own exposed row duals, *not* the raw vertex objective. On an indefinite
``x'Qx`` the RLT-1 objective has a wide coefficient spread (qap: eig ∈
[−330k, +953k]), and the reported vertex objective from such an ill-conditioned
basis can drift a few ``ulp·cond`` *above* the true LP minimum — an over-estimate
that, surfaced as a lower bound, could prune the global optimum (the nvs22
false-certificate class, issue #145). The NS bound ``g(y) = −bᵀy + Σⱼ minₓ (c+Aᵀy)ⱼxⱼ``
satisfies ``g(y) ≤ true LP min`` for *any* ``y ≥ 0`` by weak duality, so it is a
rigorous under-estimate at **any** conditioning while reproducing the vertex
objective to within its float margin on well-conditioned solves. Gated by
:class:`~discopt.solver_tuning.SolverTuning` (``DISCOPT_RLT1_ROOT_BOUND``, default
off) and a size guard.

Distinct from ``rlt_cuts.py``: that module *separates* individual violated
constraint×bound-factor RLT cuts per node over the **already-lifted** columns (the
targeted, non-exhaustive half of RLT-1). This module builds the **exhaustive**
root RLT-1 LP — it introduces a lifted column for *every* pair (not only the
objective's) and adds *all* equality×variable product rows — which is what
actually binds on qap, where the targeted per-node separator leaves the bound ~0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

__all__ = ["rlt1_lower_bound", "build_rlt1_lp", "RLT1Problem"]


@dataclass
class RLT1Problem:
    """The assembled exhaustive RLT-1 LP (all in the ``A_ub <= b_ub`` form).

    ``pair_index[(i, j)]`` (``i < j``) is the column of the lifted product
    ``X_ij``; the first ``n`` columns are the original variables ``x``. ``offset``
    is added to the LP objective to recover the user objective scale.
    """

    cobj: np.ndarray
    A_ub: "sp.csr_matrix"
    b_ub: np.ndarray
    bounds: list
    offset: float
    n: int
    pair_index: dict
    n_rlt_rows: int

    def pack_point(self, x: np.ndarray) -> np.ndarray:
        """Lift an original-variable point ``x`` to the full LP vector with
        ``X_ij = x_i x_j`` — the value taken at a genuinely feasible point."""
        z = np.zeros(self.cobj.shape[0], dtype=np.float64)
        z[: self.n] = x
        for (i, j), p in self.pair_index.items():
            z[p] = x[i] * x[j]
        return z


def _reconstruct_quadratic_objective(
    relax, info: dict, n_orig: int
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Reconstruct ``(Q, c_lin, offset)`` from the relaxation objective, or ``None``.

    The lifted objective is ``sum_col c[col] * z[col] + offset``. For a pure
    quadratic it is supported only on ``original`` columns (the linear part
    ``c_lin``) and ``bilinear`` columns ``X_ij`` (the coefficient there is
    ``Q_ij + Q_ji`` for the symmetric ``Q``). If **any** other lifted column
    carries an objective coefficient the objective is not a pure quadratic in the
    original variables and we must refuse (return ``None``) rather than emit a
    bound against a wrong objective.
    """
    c = np.asarray(relax._c, dtype=np.float64)
    orig = info.get("original", {})
    bil = info.get("bilinear", {})
    offset = float(relax._obj_offset)

    orig_cols = {int(col): int(i) for i, col in orig.items()}
    bil_cols = {int(col) for col in bil.values()}
    allowed = set(orig_cols) | bil_cols

    # Refuse if the objective touches any non-(original|bilinear) lifted column.
    nz = np.flatnonzero(np.abs(c) > 0.0)
    if any(int(col) not in allowed for col in nz):
        return None

    c_lin = np.zeros(n_orig, dtype=np.float64)
    for col, i in orig_cols.items():
        if i < n_orig:
            c_lin[i] = c[col]
    Q = np.zeros((n_orig, n_orig), dtype=np.float64)
    for (i, j), col in bil.items():
        i, j = int(i), int(j)
        if i >= n_orig or j >= n_orig:
            return None
        coeff = float(c[int(col)])
        # objective coeff on X_ij equals Q_ij + Q_ji = 2 * Q_sym_ij
        Q[i, j] += coeff / 2.0
        Q[j, i] += coeff / 2.0
    return Q, c_lin, offset


def _mutually_exclusive_pairs(A_eq, b_eq, binary_vars: frozenset, n: int) -> set:
    """Pairs ``(i, j)`` (``i < j``) whose product ``x_i x_j`` is identically 0.

    A **bound-neutral** structural presolve for the RLT-1 LP. For a linear equality
    ``sum_k a_k x_k = beta`` with two binaries ``x_i, x_j`` in its support, setting
    both to 1 forces ``LHS >= a_i + a_j + sum_{k != i,j} min(0, a_k)`` (each other
    binary term is ``>= min(0, a_k)``); if that lower bound already exceeds
    ``beta`` the assignment is infeasible, so ``x_i x_j = 0`` at *every* feasible
    point. This is exactly the set-partitioning / packing exclusivity (assignment
    ``sum_k x_k = 1`` is the unit-coefficient special case: ``1 + 1 > 1``), stated
    generally so it is not keyed to a problem name (§2).

    The lifted column ``X_ij`` of such a pair is already pinned to 0 by the RLT
    product of that same equality (``sum_k X_{i,k} = x_i`` with ``X_{i,k} >= 0`` and
    ``X_ii = x_i`` forces every off-diagonal ``X_{i,k} = 0``), so dropping the
    column, its three McCormick rows, and its now-trivial RLT rows leaves the LP
    optimum **exactly unchanged** while shrinking the system the exact simplex must
    factor. Measured: ~1.3–1.5x fewer rows and ~2.4x faster on synthetic QAPs at an
    identical bound (docs/dev/sparse-milp-plan.md §RLT1).
    """
    if A_eq is None or getattr(A_eq, "shape", (0,))[0] == 0:
        return set()
    A = sp.csr_matrix(A_eq)
    b = np.asarray(b_eq, dtype=np.float64)
    excl: set[tuple[int, int]] = set()
    tol = 1e-9
    for r in range(A.shape[0]):
        s, e = A.indptr[r], A.indptr[r + 1]
        idx = [int(c) for c in A.indices[s:e]]
        val = [float(v) for v in A.data[s:e]]
        beta = float(b[r])
        # Lower bound on the sum of all *other* binary terms when two members are 1.
        neg_rest = sum(min(0.0, v) for v in val)
        for a in range(len(idx)):
            ia, va = idx[a], val[a]
            if ia not in binary_vars or ia >= n:
                continue
            for c in range(a + 1, len(idx)):
                jb, vb = idx[c], val[c]
                if jb not in binary_vars or jb >= n:
                    continue
                rest = neg_rest - min(0.0, va) - min(0.0, vb)
                if va + vb + rest > beta + tol:
                    excl.add((min(ia, jb), max(ia, jb)))
    return excl


def build_rlt1_lp(
    model,
    relax,
    info: dict,
    *,
    binary_vars: frozenset,
    max_pairs: int = 60_000,
) -> Optional[RLT1Problem]:
    """Assemble the exhaustive RLT-1 LP for a constrained binary QP, or ``None``.

    ``None`` is returned (a sound no-op upstream) on any ineligibility: no binary
    variables, no quadratic objective terms, an objective that is not a pure
    quadratic in the original variables, no linear equality constraints (RLT-1's
    constraint factors need equalities), a non-binary variable in the objective
    quadratic, or an all-pairs lift larger than ``max_pairs``.

    Parameters
    ----------
    model:
        The user model (for its linear constraints).
    relax, info:
        A built McCormick relaxation and its column map (from
        ``build_milp_relaxation``) — used only to reconstruct ``Q``/``c`` and to
        confirm the objective is a pure quadratic.
    binary_vars:
        Flat indices of the binary variables (``binary_flat_cols(model)``). The
        RLT diagonal ``X_ii = x_i`` and the whole construction are sound **only**
        for binaries.
    max_pairs:
        Size guard: skip when the all-pairs lift ``n(n-1)/2`` exceeds this, so a
        huge model never builds a giant LP.
    """
    from discopt._jax.obbt import _extract_linear_constraints

    # -- eligibility ---------------------------------------------------------
    if not binary_vars:
        return None
    if not info.get("bilinear"):
        return None  # no quadratic objective terms -> nothing to strengthen
    if not getattr(relax, "_objective_bound_valid", False):
        return None

    A_ub_m, b_ub_m, A_eq_m, b_eq_m, n = _extract_linear_constraints(model)
    if A_eq_m is None or A_eq_m.shape[0] == 0:
        # RLT-1 constraint-factor products need equalities; without them the
        # relaxation is just McCormick and there is nothing to add.
        return None

    # Every variable that appears in the objective quadratic must be binary — the
    # X_ii = x_i diagonal and the bound-factor rows are unsound otherwise.
    bil = info.get("bilinear", {})
    involved: set[int] = set()
    for i, j in bil:
        involved.add(int(i))
        involved.add(int(j))
    if any(v not in binary_vars for v in involved):
        return None

    if n * (n - 1) // 2 > max_pairs:
        return None

    recon = _reconstruct_quadratic_objective(relax, info, n)
    if recon is None:
        return None
    Q, c_lin, offset = recon

    # -- build the RLT-1 LP ---------------------------------------------------
    # Bound-neutral presolve: pairs whose product is identically 0 (set-partitioning
    # exclusivity) are already pinned to 0 by the RLT rows, so we do not lift them —
    # dropping the column, its McCormick rows, and its trivial RLT rows shrinks the
    # LP without changing its optimum (see ``_mutually_exclusive_pairs``).
    excluded = _mutually_exclusive_pairs(A_eq_m, b_eq_m, binary_vars, n)

    # Variables: x (n) then X_ij for every non-excluded i<j pair.
    pair_index: dict[tuple[int, int], int] = {}
    nv = n
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            pair_index[(i, j)] = nv
            nv += 1

    def pcol(i: int, j: int) -> int:
        return pair_index[(i, j)] if i < j else pair_index[(j, i)]

    # objective: sum_i Q_ii x_i + sum_{i<j} 2 Q_ij X_ij + c_lin·x + offset
    cobj = np.zeros(nv, dtype=np.float64)
    for i in range(n):
        cobj[i] = Q[i, i] + c_lin[i]
    for (i, j), p in pair_index.items():
        cobj[p] = 2.0 * Q[i, j]

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_ub: list[float] = []
    ru = 0

    def add_ub(terms: list[tuple[int, float]], rhs: float) -> None:
        nonlocal ru
        for col, val in terms:
            rows.append(ru)
            cols.append(col)
            data.append(val)
        b_ub.append(rhs)
        ru += 1

    def add_eq(terms: list[tuple[int, float]], rhs: float) -> None:
        # exact oracle takes only A_ub; encode a·z = r as a·z <= r and -a·z <= -r
        add_ub(terms, rhs)
        add_ub([(col, -val) for col, val in terms], -rhs)

    # McCormick bound-factor rows for every pair (X_ij >= 0 handled by bounds).
    for (i, j), p in pair_index.items():
        add_ub([(p, 1.0), (i, -1.0)], 0.0)  # X_ij <= x_i
        add_ub([(p, 1.0), (j, -1.0)], 0.0)  # X_ij <= x_j
        add_ub([(p, -1.0), (i, 1.0), (j, 1.0)], 1.0)  # X_ij >= x_i + x_j - 1

    # Model linear constraints on the original x (keep the relaxation's linear
    # feasible set — assignment etc.). ``A_eq_m``/``b_eq_m`` are non-None here
    # (guarded above); narrow into local names for the type checker.
    A_eq = sp.csr_matrix(A_eq_m)
    b_eq = np.asarray(b_eq_m, dtype=np.float64)
    for r in range(A_eq.shape[0]):
        s, e = A_eq.indptr[r], A_eq.indptr[r + 1]
        eq_terms = [(int(A_eq.indices[t]), float(A_eq.data[t])) for t in range(s, e)]
        if eq_terms:
            add_eq(eq_terms, float(b_eq[r]))
    if A_ub_m is not None and b_ub_m is not None and A_ub_m.shape[0] > 0:
        A_ubm = sp.csr_matrix(A_ub_m)
        b_ubm = np.asarray(b_ub_m, dtype=np.float64)
        for r in range(A_ubm.shape[0]):
            s, e = A_ubm.indptr[r], A_ubm.indptr[r + 1]
            ub_terms = [(int(A_ubm.indices[t]), float(A_ubm.data[t])) for t in range(s, e)]
            if ub_terms:
                add_ub(ub_terms, float(b_ubm[r]))

    # RLT-1: (a·x = beta) * x_p  ->  sum_k a_k X_{p,k} = beta x_p, using X_pp = x_p.
    # Terms whose pair (p, k) is exclusion-pinned to 0 are dropped (sound: the
    # product is identically 0). A row that reduces to only ``coef·x_p = 0`` with
    # ``coef ~ 0`` is a trivial identity — dropped — but ``coef·x_p = 0`` with a
    # nonzero ``coef`` is the genuine constraint ``x_p = 0`` and is kept.
    n_rlt = 0
    for r in range(A_eq.shape[0]):
        s, e = A_eq.indptr[r], A_eq.indptr[r + 1]
        supp = [(int(A_eq.indices[t]), float(A_eq.data[t])) for t in range(s, e)]
        beta = float(b_eq[r])
        if not supp:
            continue
        for p in range(n):
            rlt_terms: list[tuple[int, float]] = []
            coef_xp = -beta  # the -beta x_p on the RHS side
            for k, a_k in supp:
                if k == p:
                    coef_xp += a_k  # X_pp = x_p
                elif (min(p, k), max(p, k)) in pair_index:
                    rlt_terms.append((pcol(p, k), a_k))
                # else: pair (p, k) exclusion-pinned to 0 -> term drops out.
            if not rlt_terms and abs(coef_xp) <= 1e-12:
                continue  # trivial 0 = 0 identity after presolve
            rlt_terms.append((p, coef_xp))
            add_eq(rlt_terms, 0.0)
            n_rlt += 1

    A_ub = sp.csr_matrix((data, (rows, cols)), shape=(ru, nv))
    A_ub.sort_indices()
    b_ub_arr = np.asarray(b_ub, dtype=np.float64)
    bounds = [(0.0, 1.0)] * nv

    return RLT1Problem(
        cobj=cobj,
        A_ub=A_ub,
        b_ub=b_ub_arr,
        bounds=bounds,
        offset=offset,
        n=n,
        pair_index=pair_index,
        n_rlt_rows=n_rlt,
    )


def rlt1_lower_bound(
    model,
    relax,
    info: dict,
    *,
    binary_vars: frozenset,
    time_limit: Optional[float] = 30.0,
    max_pairs: int = 60_000,
) -> tuple[Optional[float], int]:
    """Rigorous RLT-1 lower bound for a constrained binary QP, or a sound no-op.

    Builds the exhaustive RLT-1 LP (:func:`build_rlt1_lp`) and solves it once with
    the **exact vertex simplex** oracle, then returns the **Neumaier–Shcherbina
    safe dual bound** computed from that solve's exposed row duals — a rigorous
    global lower bound on the minimize objective (``<=`` the true optimum) at *any*
    conditioning, rather than the raw vertex objective which can drift above the
    true LP minimum on the wide-coefficient RLT LP (issue #145 / #661). Returns
    ``(bound, n_rlt_rows)``; ``bound`` is ``None`` on any ineligibility/failure —
    including a solve that exposes no usable duals, in which case we decline rather
    than surface the un-certified vertex objective (a sound no-op) — and
    ``n_rlt_rows`` is the number of RLT product rows added.
    """
    from discopt._jax.obbt import _ns_safe_lp_lower_bound, get_exact_dual_lp_solver
    from discopt.solvers import SolveStatus

    prob = build_rlt1_lp(model, relax, info, binary_vars=binary_vars, max_pairs=max_pairs)
    if prob is None:
        return (None, 0)
    # Prefer the dual-exposing exact oracle: the RLT-1 bound we surface is the
    # rigorous NS-safe value built from its row duals, not the raw vertex value.
    _lp = get_exact_dual_lp_solver()
    if _lp is None:
        return (None, 0)
    res = _lp(
        c=prob.cobj, A_ub=prob.A_ub, b_ub=prob.b_ub, bounds=prob.bounds, time_limit=time_limit
    )
    if res.status != SolveStatus.OPTIMAL or res.objective is None:
        return (None, prob.n_rlt_rows)
    # The RLT-1 LP is assembled entirely as ``A_ub z <= b_ub`` (equalities are split
    # into two ``<=`` rows by ``build_rlt1_lp``), so ``n_eq = 0`` — every row's dual
    # clamps to ``>= 0`` in the NS bound. All columns carry finite ``[0, 1]`` bounds,
    # so no free-column reduced-cost snap is needed (``rc_snap_tol = 0``).
    lo = np.array([b[0] for b in prob.bounds], dtype=np.float64)
    hi = np.array([b[1] for b in prob.bounds], dtype=np.float64)
    g = _ns_safe_lp_lower_bound(
        prob.cobj, getattr(res, "dual_values", None), prob.A_ub, prob.b_ub, lo, hi, n_eq=0
    )
    if g is None or not np.isfinite(g):
        # No usable duals -> the vertex objective is not certified rigorous on an
        # ill-conditioned RLT LP; decline rather than risk an over-estimate.
        return (None, prob.n_rlt_rows)
    return (float(g) + prob.offset, prob.n_rlt_rows)
