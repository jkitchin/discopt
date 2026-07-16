"""Warm-started-simplex MILP backend (Rust ``solve_milp_py``).

A ``solve_milp(c, A_ub, b_ub, ..., integrality, ...)`` adapter, signature- and
``MILPResult``-compatible with :mod:`discopt.solvers.milp_pounce` and
:mod:`discopt.solvers.milp_pounce`, so it can be selected through
:func:`discopt.solvers.lp_backend.get_milp_solver`. It marshals the ``A_ub x <= b_ub``
form into the engine's standard form ``A_eq z = b`` (one explicit slack per row) and runs
the pure-Rust warm-started-simplex branch-and-bound.

Soundness: ``MILPResult.objective`` is the incumbent (an upper bound on a non-optimal
exit) and ``MILPResult.bound`` is the engine's **dual lower bound** — equal to the
incumbent once the solve is proven optimal, and a valid lower bound otherwise. Callers
that need a lower bound (AMP/OA/GDP-LOA) must read ``bound``, never ``objective``. If the
Rust binding is unavailable it raises :class:`SimplexBackendUnavailable` so the selector
can fall back.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union, cast

import numpy as np
import scipy.sparse as sp

from discopt.solvers import MILPResult, SolveStatus


class SimplexBackendUnavailable(RuntimeError):
    """Raised when the Rust ``solve_milp_py`` binding cannot be imported."""


_NS_MARGIN_REL = 1e-9
"""Magnitude-scaled relative margin for the safe-bound / Farkas-ray evaluations.

The two dot products below run in plain float64 (not directed-rounding interval
arithmetic), so a margin proportional to the operands' magnitude is subtracted to
dominate their rounding error and keep the returned bound a *rigorous*
under-estimate (and the Farkas test a rigorous proof). Mirrors the constant in
:func:`discopt._jax.obbt._ns_safe_lp_lower_bound`."""

_INF = 1e20  # discopt's effective-infinity sentinel for free variable bounds.


class LpWarmCert(NamedTuple):
    """Verified-certificate side-channel from :func:`solve_lp_warm_std`.

    * ``safe_bound`` — on an ``optimal`` solve, a Neumaier–Shcherbina safe lower
      bound computed from the simplex's own row duals: ``<=`` the true LP optimum
      at *any* conditioning, so a caller can use it as a rigorous bound without an
      independent second solve. ``None`` when unavailable.
    * ``farkas_certified`` — on an ``infeasible`` solve, ``True`` iff the
      simplex's Farkas dual-ray candidate was independently verified to prove the
      feasible set empty (a rigorous fathoming proof). ``False`` otherwise — the
      caller must then fall back rather than trust the bare verdict.
    * ``dual`` — on an ``optimal`` solve, the simplex's row-dual vector ``y`` (one
      entry per constraint row of the *standard-form* ``[A_ub | I] z = b`` system,
      so ``len(dual) == m``). ``None`` when unavailable. Additive side-channel for
      duality-based bound tightening (cert:T2.4a); it never changes the reported
      objective/bound (those are computed identically whether or not it is read).
    * ``col_status`` — on an ``optimal`` solve, the final column-status vector for
      the standard-form columns (structural first, then slacks) — the warm-start
      basis to thread into a downstream re-solve. ``None`` when unavailable.
    """

    safe_bound: Optional[float]
    farkas_certified: bool
    dual: Optional[np.ndarray] = None
    col_status: Optional[np.ndarray] = None


def _fbbt_eq_bounds(
    a_std: "object",
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    rounds: int = 3,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Feasibility-based bound tightening on the equality system ``A_std z = b``.

    Returns ``(lb, ub)`` tightened so that every derived bound is *implied* by the
    equalities plus the incoming box — i.e. the result still contains the whole
    feasible set ``{z : A_std z = b, lb <= z <= ub}``. Used to give the unbounded
    lifted/slack columns a finite, **valid** box for the Neumaier–Shcherbina safe
    bound: a superset-preserving tightening keeps ``g(y) <= p*`` sound while making
    the box-min term finite (an open side whose reduced cost selects it would
    otherwise collapse the whole bound to ``-inf``).

    Each equality ``Σ_j a_ij z_j = b_i`` bounds column ``k`` two-sidedly from the
    min/max activity of the *other* columns, with an explicit per-row infinity
    tally so a single open bound still propagates (vectorised over the sparse
    matrix; the per-element Python loop is too slow for this per-solve path).
    """
    coo = sp.csr_matrix(a_std).tocoo()
    rows = coo.row
    cols = coo.col
    vals = coo.data
    n_rows = coo.shape[0]
    lb = np.array(lb, dtype=np.float64)
    ub = np.array(ub, dtype=np.float64)
    if vals.size == 0:
        return lb, ub
    pos = vals > 0.0
    b_r = b[rows]

    for _ in range(rounds):
        # Min activity uses lb where coeff>0, ub where coeff<0; max activity swaps.
        min_used = np.where(pos, lb[cols], ub[cols])
        max_used = np.where(pos, ub[cols], lb[cols])
        min_term = vals * min_used  # may be -inf
        max_term = vals * max_used  # may be +inf
        min_inf = ~np.isfinite(min_term)
        max_inf = ~np.isfinite(max_term)
        n_min_inf = np.zeros(n_rows)
        np.add.at(n_min_inf, rows, min_inf.astype(np.float64))
        n_max_inf = np.zeros(n_rows)
        np.add.at(n_max_inf, rows, max_inf.astype(np.float64))
        sum_min = np.zeros(n_rows)
        np.add.at(sum_min, rows, np.where(min_inf, 0.0, min_term))
        sum_max = np.zeros(n_rows)
        np.add.at(sum_max, rows, np.where(max_inf, 0.0, max_term))
        # Activity of the row excluding column j (finite iff every *other* term is).
        minrest_finite = (n_min_inf[rows] - min_inf.astype(np.float64)) == 0
        maxrest_finite = (n_max_inf[rows] - max_inf.astype(np.float64)) == 0
        minrest = sum_min[rows] - np.where(min_inf, 0.0, min_term)
        maxrest = sum_max[rows] - np.where(max_inf, 0.0, max_term)
        # z_j = (b_i - rest)/a_ij; the rest-interval endpoints give z_j's bounds.
        lo_cand = np.where(pos, (b_r - maxrest) / vals, (b_r - minrest) / vals)
        lo_valid = np.where(pos, maxrest_finite, minrest_finite)
        hi_cand = np.where(pos, (b_r - minrest) / vals, (b_r - maxrest) / vals)
        hi_valid = np.where(pos, minrest_finite, maxrest_finite)

        new_lo = np.full(lb.shape[0], -np.inf)
        sel = lo_valid & np.isfinite(lo_cand)
        if sel.any():
            np.maximum.at(new_lo, cols[sel], lo_cand[sel])
        upd_lo = new_lo > lb + tol
        if upd_lo.any():
            lb = np.where(upd_lo, np.maximum(lb, new_lo), lb)

        new_hi = np.full(ub.shape[0], np.inf)
        sel = hi_valid & np.isfinite(hi_cand)
        if sel.any():
            np.minimum.at(new_hi, cols[sel], hi_cand[sel])
        upd_hi = new_hi < ub - tol
        if upd_hi.any():
            ub = np.where(upd_hi, np.minimum(ub, new_hi), ub)

        if not (upd_lo.any() or upd_hi.any()):
            break
    return lb, ub


def _safe_lp_lower_bound_std(
    y: np.ndarray,
    c: np.ndarray,
    a_std: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Optional[float]:
    """Neumaier–Shcherbina safe lower bound on ``min cᵀz s.t. A z = b, lb<=z<=ub``
    from *free-sign* equality multipliers ``y`` (length m).

    Weak duality gives, for ANY ``y``,

        g(y) = bᵀy + Σ_k min_{z_k∈[lb_k,ub_k]} (c − Aᵀy)_k z_k  ≤  min cᵀz,

    so ``g(y)`` is a valid lower bound regardless of how ``y`` was obtained — it
    stays sound even when an ill-conditioned basis makes the reported vertex
    objective drift *above* the true optimum (the nvs22 false-certificate class).
    A magnitude-scaled margin is subtracted so the float64 evaluation error cannot
    push ``g`` above the true optimum. Returns ``None`` when no usable (finite)
    bound exists (e.g. an unbounded box term)."""
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0 or not np.all(np.isfinite(y)):
        return None
    # Map the ±1e20 sentinels to true infinities so an infinite box side with a
    # nonzero reduced cost yields −inf (an unusable bound → None), not a spurious
    # large-finite contribution.
    lb = np.where(np.asarray(lb, dtype=np.float64) <= -_INF, -np.inf, lb)
    ub = np.where(np.asarray(ub, dtype=np.float64) >= _INF, np.inf, ub)
    c = np.asarray(c, dtype=np.float64)
    at_y = a_std.T @ y if not sp.issparse(a_std) else (a_std.T @ y)
    rc = c - np.asarray(at_y).ravel()
    pos = rc > 0.0
    neg = rc < 0.0
    # A box-min term is -inf when the reduced cost selects an open side. The
    # lifted relaxation's objective-epigraph / sqrt-/division-lift aux columns and
    # the row slacks carry +/-inf bounds, and a roundoff-flipped tiny reduced cost
    # on such a column would otherwise collapse the whole safe bound to -inf (the
    # nvs05/nvs22/st_e36/chance root-bound drop). Recover a finite, *valid* box for
    # exactly those columns by feasibility-based bound tightening: FBBT bounds still
    # contain the feasible set, so g(y) stays <= p* (sound) while becoming finite.
    # Gated on actually needing it, so well-bounded LPs keep the cheap path.
    if (pos & ~np.isfinite(lb)).any() or (neg & ~np.isfinite(ub)).any():
        # FBBT's float64 division roundoff (~ulp·|bound|) is dominated by the
        # magnitude-scaled ``margin`` subtracted from g below, so the derived box
        # needs no extra outward loosening — and adding one perturbs the (large)
        # slack/aux bounds enough to break the safe bound's rescaling invariance
        # without improving soundness. Use the FBBT bounds directly.
        lb, ub = _fbbt_eq_bounds(a_std, np.asarray(b, dtype=np.float64), lb, ub)
    # min_{z_k∈[lb,ub]} rc_k z_k = lb_k if rc_k>0, ub_k if rc_k<0, else 0 (the
    # rc_k==0 case contributes 0 even when that bound is infinite).
    contrib = np.zeros_like(rc)
    contrib[pos] = rc[pos] * lb[pos]
    contrib[neg] = rc[neg] * ub[neg]
    # Any term still open after FBBT (a genuinely unbounded selected side) leaves
    # no usable bound — abstain rather than return a spurious value.
    if not np.all(np.isfinite(contrib)):
        return None
    by = float(np.asarray(b, dtype=np.float64) @ y)
    g = by + float(contrib.sum())
    if not np.isfinite(g):
        return None
    margin = _NS_MARGIN_REL * (1.0 + abs(by) + float(np.abs(contrib).sum()))
    return g - margin


def _farkas_certified_std(
    ray: np.ndarray,
    a_std: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> bool:
    """Verify a Farkas dual-ray candidate proves ``A z = b, lb<=z<=ub`` is empty.

    The system is infeasible iff some free-sign ``y`` has ``bᵀy`` exceeding the
    box-maximum of ``(Aᵀy)ᵀz`` — i.e. the ``c=0`` safe bound ``g₀(y) > 0`` (the
    margin inside :func:`_safe_lp_lower_bound_std` already makes the strict
    inequality rigorous). The simplex hands us a ray up to an overall sign, so we
    try ``±ray``; a candidate that fails to verify simply returns ``False`` and
    the caller falls back — it can never produce an unsound fathom."""
    ray = np.asarray(ray, dtype=np.float64)
    if ray.size == 0 or not np.all(np.isfinite(ray)):
        return False
    zeros_c = np.zeros(a_std.shape[1], dtype=np.float64)
    for sign in (1.0, -1.0):
        g0 = _safe_lp_lower_bound_std(sign * ray, zeros_c, a_std, b, lb, ub)
        if g0 is not None and g0 > 0.0:
            return True
    return False


def solve_milp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 1_000_000,
) -> MILPResult:
    """Solve ``min c^T x  s.t.  A_ub x <= b_ub, A_eq x == b_eq, bounds, integrality``
    with the Rust warm-started-simplex B&B.

    Mirrors :func:`discopt.solvers.milp_pounce.solve_milp`. The returned
    ``objective`` is the engine's dual lower bound (see module docstring).
    """
    try:
        from discopt._rust import solve_milp_csc_py
    except ImportError as err:  # pragma: no cover - exercised via the selector
        raise SimplexBackendUnavailable(
            "discopt._rust.solve_milp_csc_py is unavailable; build the Rust extension"
        ) from err

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]

    # Assemble all rows as `<=` (A_eq becomes a pair of `<=` rows) then slack.
    # SPARSE throughout: a dense `[A_ub | I]` would materialize an m×(n+m) matrix
    # (~73 GB for qap's 85k×21k McCormick relaxation) that the Rust driver never
    # needs — it consumes CSC. We keep every block sparse and hand the driver the
    # CSC triplets of the standard-form matrix; nothing is ever densified here.
    blocks: list[sp.spmatrix] = []
    rhs: list[float] = []
    if A_ub is not None and b_ub is not None and np.size(b_ub) > 0:
        au = (
            sp.csr_matrix(cast("sp.spmatrix", A_ub))
            if sp.issparse(A_ub)
            else sp.csr_matrix(np.asarray(A_ub, dtype=np.float64).reshape(-1, n))
        )
        blocks.append(au)
        rhs.extend(np.asarray(b_ub, dtype=np.float64).ravel().tolist())
    if A_eq is not None and b_eq is not None and np.size(b_eq) > 0:
        ae = (
            sp.csr_matrix(cast("sp.spmatrix", A_eq))
            if sp.issparse(A_eq)
            else sp.csr_matrix(np.asarray(A_eq, dtype=np.float64).reshape(-1, n))
        )
        be = np.asarray(b_eq, dtype=np.float64).ravel()
        blocks.append(ae)
        rhs.extend(be.tolist())
        blocks.append(-ae)
        rhs.extend((-be).tolist())

    if blocks:
        a_ub_sp = sp.vstack(blocks, format="csr")
    else:
        a_ub_sp = sp.csr_matrix((0, n), dtype=np.float64)
    b_vec = np.asarray(rhs, dtype=np.float64)
    m = a_ub_sp.shape[0]

    # Standard form A_eq z = b with one slack per row: [A_ub | I] z = b_ub, built
    # directly as CSC (never densified). ``sort_indices`` gives ascending row
    # order within each column, which ``SparseCols::from_csc`` requires.
    a_std_sp = sp.hstack([a_ub_sp, sp.identity(m, dtype=np.float64, format="csr")], format="csc")
    a_std_sp.sum_duplicates()
    a_std_sp.sort_indices()
    csc_col_ptr = np.ascontiguousarray(a_std_sp.indptr, dtype=np.int64)
    csc_row_idx = np.ascontiguousarray(a_std_sp.indices, dtype=np.int64)
    csc_vals = np.ascontiguousarray(a_std_sp.data, dtype=np.float64)

    if bounds is not None:
        lb = np.array([lo for lo, _ in bounds], dtype=np.float64)
        ub = np.array([hi for _, hi in bounds], dtype=np.float64)
    else:
        lb = np.zeros(n, dtype=np.float64)
        ub = np.full(n, 1e20, dtype=np.float64)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m, 1e20)])
    c_std = np.concatenate([c_arr, np.zeros(m)])

    if integrality is not None:
        int_mask = np.asarray(integrality, dtype=np.int64).ravel()
        int_cols = np.flatnonzero(int_mask != 0).astype(np.int64)
    else:
        int_cols = np.zeros(0, dtype=np.int64)

    # Interactive debugger: install the Rust checkpoint hook only when a debugger
    # is attached now, so the pure-Rust search stays bound-neutral otherwise.
    from discopt import debug as _debug

    # Pure-LP short-circuit (THRU-2b): when there are no integer columns this is a
    # plain LP, yet the MILP driver still runs its integer-search machinery — root
    # cut rounds, GMI, primal heuristics, strong branching. With no integer
    # variables none of that can fire (GMI needs a fractional integer; the
    # heuristics round nothing; there is no candidate to branch on), so it is pure
    # overhead on the root LP whose optimum/infeasibility is the whole answer. This
    # path is the fallback the McCormick node relaxer reaches when the warm sparse
    # simplex breaks down numerically on a hard, ill-conditioned lifted LP
    # (``solve_lp_warm_csc_py`` -> ``numerical`` at iters=0); the driver's LP
    # presolve then decides it, but the wasted cut/heuristic passes inflate the
    # solve (nvs24 node LP 10.9 s -> 5.5 s with the machinery off). Turning the
    # machinery off on a genuine LP is bound-neutral by construction: the root LP
    # optimum and the infeasibility verdict are unchanged — only inert integer-side
    # work is skipped. It never triggers when ``int_cols`` is non-empty.
    _pure_lp = int(int_cols.size) == 0
    _lp_kwargs: dict = (
        dict(
            root_cuts=0,
            cut_rounds=0,
            gmi_cuts=False,
            heuristics=False,
            strong_branch=False,
        )
        if _pure_lp
        else {}
    )

    status, x_full, obj, bound, nodes, _iters = solve_milp_csc_py(
        np.ascontiguousarray(c_std),
        m,
        n + m,  # total columns: structural + one slack per row
        csc_col_ptr,
        csc_row_idx,
        csc_vals,
        np.ascontiguousarray(b_vec),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
        np.ascontiguousarray(int_cols),
        n,  # n_struct: structural columns precede the slacks
        0.0,  # obj_const: caller (MilpRelaxationModel) applies its own offset
        int(max_nodes),
        float(gap_tolerance),
        time_limit_s=0.0 if time_limit is None else max(0.0, float(time_limit)),
        debug_hook=_debug.rust_hook(),
        **_lp_kwargs,
    )

    if status == "infeasible":
        return MILPResult(status=SolveStatus.INFEASIBLE, node_count=int(nodes))
    if status == "unbounded":
        return MILPResult(status=SolveStatus.UNBOUNDED, node_count=int(nodes))

    x_struct = np.asarray(x_full, dtype=np.float64)[:n]
    if status == "optimal":
        # Proven optimum: incumbent == dual bound, a tight valid lower bound.
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=x_struct,
            objective=float(obj),
            bound=float(obj),
            node_count=int(nodes),
        )

    # node_limit / feasible: ``objective`` is the incumbent (upper bound) and
    # ``bound`` is the engine's dual lower bound (sound) if finite. Callers that
    # need a lower bound must read ``bound``, never ``objective``.
    return MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=x_struct,
        objective=float(obj) if np.isfinite(obj) else None,
        bound=float(bound) if np.isfinite(bound) else None,
        node_count=int(nodes),
    )


def solve_lp_warm_std(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    bounds: Optional[list[tuple[float, float]]],
    in_basis: Optional[tuple[np.ndarray, np.ndarray]] = None,
    *,
    return_cert: bool = False,
):
    """Warm-startable **pure-LP** solve of ``min c^T x s.t. A_ub x <= b_ub, bounds``.

    Marshals the ``A_ub x <= b_ub`` form into standard form ``[A_ub | I] z = b_ub``
    (one explicit slack per row, structural columns first) and calls the Rust
    ``solve_lp_warm_py``. ``in_basis`` is a ``(col_status, basic_vars)`` pair from a
    previous solve of a *prefix* of the same column set (rows since appended); Rust
    extends it by making the appended slacks basic and dual-simplex re-optimizes.

    Returns ``(result, out_basis)``. ``out_basis`` is the final ``(col_status,
    basic_vars)`` to thread into the next re-solve (``None`` when the LP is not
    optimal). ``result`` is ``None`` for ``iter_limit``/``numerical`` exits so the
    caller can fall back to a cold/HiGHS path. Soundness: the dual simplex
    converges to the LP optimum exactly as a cold solve (a bad basis is ignored
    inside Rust), so the returned objective/bound is unchanged — only the speed is.

    When ``return_cert`` is set, returns ``(result, out_basis, cert)`` with a
    :class:`LpWarmCert` built from the simplex's own duals / Farkas ray: a
    rigorous safe lower bound on an ``optimal`` solve, and an independently
    verified infeasibility proof on an ``infeasible`` one — both without a second
    external solve (issue #356).
    """
    from discopt._rust import solve_lp_warm_csc_py

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]

    if A_ub is not None and b_ub is not None and np.size(b_ub) > 0:
        a_struct = (
            sp.csc_matrix(A_ub)
            if sp.issparse(A_ub)
            else sp.csc_matrix(np.asarray(A_ub, dtype=np.float64).reshape(-1, n))
        )
        b_vec = np.asarray(b_ub, dtype=np.float64).ravel()
    else:
        a_struct = sp.csc_matrix((0, n), dtype=np.float64)
        b_vec = np.zeros(0, dtype=np.float64)
    m = a_struct.shape[0]

    # Standard form ``[A_ub | I_m] z = b`` (one slack per row) assembled directly in
    # CSC, with the slack identity left implicit-sparse — the lifted relaxations are
    # ~0.3% dense, so the old dense ``a_std`` (np.zeros((m, n+m)) + np.eye(m)) was a
    # 431MB / O(m^2) allocation per solve that the Rust side then re-scanned. The
    # sparse matrix flows straight to the CSC-native simplex and the safe-bound /
    # Farkas helpers (both accept a SciPy sparse matrix). (issue #356)
    if m > 0:
        a_std = sp.hstack(
            [a_struct, sp.identity(m, format="csc", dtype=np.float64)], format="csc"
        ).tocsc()
    else:
        a_std = a_struct.tocsc()

    if bounds is not None:
        lb = np.array([lo for lo, _ in bounds], dtype=np.float64)
        ub = np.array([hi for _, hi in bounds], dtype=np.float64)
    else:
        lb = np.zeros(n, dtype=np.float64)
        ub = np.full(n, 1e20, dtype=np.float64)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m, 1e20)])
    c_std = np.concatenate([c_arr, np.zeros(m)])

    cs0 = None if in_basis is None else np.ascontiguousarray(in_basis[0], dtype=np.int8)
    bv0 = None if in_basis is None else np.ascontiguousarray(in_basis[1], dtype=np.int64)

    status, x_full, obj, _iters, cs, bv, dual, ray = solve_lp_warm_csc_py(
        np.ascontiguousarray(c_std),
        m,
        n + m,
        np.ascontiguousarray(a_std.indptr, dtype=np.int64),
        np.ascontiguousarray(a_std.indices, dtype=np.int64),
        np.ascontiguousarray(a_std.data, dtype=np.float64),
        np.ascontiguousarray(b_vec),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
        cs0,
        bv0,
    )

    def _result_basis_cert():
        x_struct = np.asarray(x_full, dtype=np.float64)[:n]
        if status == "optimal":
            # Rigorous safe lower bound from the simplex's own row duals (sound at
            # any conditioning — never above the true optimum), reported as the
            # ``bound`` so a caller can fathom on it without an independent solve.
            # ``objective`` stays the raw vertex value; ``bound`` is the certified
            # one (the safe bound, clamped to never exceed the raw value, since a
            # well-conditioned raw value <= safe bound is itself sound and tighter).
            safe = _safe_lp_lower_bound_std(dual, c_std, a_std, b_vec, lb_std, ub_std)
            bound = float(obj) if safe is None else min(float(obj), float(safe))
            return (
                MILPResult(
                    status=SolveStatus.OPTIMAL,
                    x=x_struct,
                    objective=float(obj),
                    bound=bound,
                    node_count=0,
                ),
                (np.asarray(cs), np.asarray(bv)),
                LpWarmCert(
                    safe_bound=(None if safe is None else float(safe)),
                    farkas_certified=False,
                    # Additive marginals (cert:T2.4a): row duals ``y`` and the final
                    # column status. Reduced costs ``d = c - A^T y`` are derived by
                    # the consumer from ``dual``; exposing the raw dual keeps this a
                    # pure plumbing change (no new math here).
                    dual=np.asarray(dual, dtype=np.float64) if dual is not None else None,
                    col_status=np.asarray(cs) if cs is not None else None,
                ),
            )
        if status == "infeasible":
            certified = _farkas_certified_std(dual, a_std, b_vec, lb_std, ub_std)
            return (
                MILPResult(status=SolveStatus.INFEASIBLE, node_count=0),
                None,
                LpWarmCert(safe_bound=None, farkas_certified=certified),
            )
        if status == "unbounded":
            return (
                MILPResult(status=SolveStatus.UNBOUNDED, node_count=0),
                None,
                LpWarmCert(safe_bound=None, farkas_certified=False),
            )
        # iter_limit / numerical: no clean optimum — signal fallback (result None).
        # But if the engine exported a dual candidate from the broken basis (#517),
        # carry a Neumaier–Shcherbina safe lower bound in the cert. It is valid for
        # ANY multiplier vector, so a drifted-basis dual only loosens it — never
        # lifts it above the optimum. The caller (behind a default-OFF flag) uses it
        # only as a last-resort floor when nothing else produced a bound.
        safe = None
        if dual is not None and np.size(dual):
            safe = _safe_lp_lower_bound_std(dual, c_std, a_std, b_vec, lb_std, ub_std)
        return (
            None,
            None,
            LpWarmCert(
                safe_bound=(None if safe is None else float(safe)),
                farkas_certified=False,
            ),
        )

    result, out_basis, cert = _result_basis_cert()
    if return_cert:
        return result, out_basis, cert
    return result, out_basis
