"""LP-form McCormick relaxation: bilinears lifted to aux columns, solved by HiGHS.

This is the "spatial-BB" relaxation: a polyhedral underestimator of the
nonconvex feasible set. Unlike the McCormick-NLP path in
:mod:`discopt._jax.mccormick_nlp`, it returns a globally optimal value of
the linear relaxation in one LP solve — no local minima, no warm-start
sensitivity. The trade-off is that bound information per node is loose
until spatial branching tightens variable domains.

The heavy lifting (term classification, McCormick row construction, aux
column bookkeeping) lives in :func:`build_milp_relaxation`. This module
is a thin wrapper that strips integrality on aux columns so the result is
a pure LP and exposes a simple ``compute(model, node_lb, node_ub)`` API
that fits the per-node call shape in :mod:`discopt.solver`.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import (
    _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE,
    _RELAX_NUMERIC_CAP,
    _collect_affine_powers,
    _linear_constraint_forms,
    _quadratic_constraint_forms,
    build_milp_relaxation,
)
from discopt._jax.term_classifier import NonlinearTerms, classify_nonlinear_terms
from discopt.modeling.core import Model, VarType
from discopt.solver_tuning import current as _tuning

logger = logging.getLogger(__name__)
# Dedup key set so the oversize-lift fallback warns once per (cols, rows) shape.
_oversize_warned: set[int] = set()


# ── Cut-pool row helpers (P1: global cut pool) ──────────────────────────────
def _as_csr(A):
    import scipy.sparse as sp

    return A if sp.issparse(A) else sp.csr_matrix(A)


def _sparse_rows(A) -> int:
    return int(_as_csr(A).shape[0]) if A is not None else 0


def _sparse_cols(A) -> int:
    return int(_as_csr(A).shape[1]) if A is not None else 0


def _pool_has_rows(inherited_cuts: Optional[tuple]) -> bool:
    """True iff ``inherited_cuts`` is a non-empty ``(A_rows, b_rows[, idents])`` pool.

    Sparse-safe row count: the pool ``A_rows`` is a scipy CSR matrix, whose
    ``len()`` is *ambiguous and raises* — so a naive ``len(A_rows) > 0`` check
    throws and (when swallowed by the caller's node-solve guard) silently skips
    the node. Use ``.shape[0]`` via :func:`_sparse_rows` instead. Used to gate
    the C-43 pool-infeasible re-verification (only re-verify when a pool that
    could have been appended was actually on offer)."""
    if inherited_cuts is None:
        return False
    a_rows = inherited_cuts[0]
    return a_rows is not None and _sparse_rows(a_rows) > 0


# ── Column-identity-safe cut inheritance (C-44 / #567) ──────────────────────
#
# A root-pool row is stated by *column position* over the ROOT build's lifted
# column layout. Per node the relaxation is re-built (cold path) or re-lifted by
# lifted-FBBT, which can produce the *same column count* with *different column
# semantics* — a given position that was ``x_2·x_5·x_8`` at the root can become
# ``x_3·x_4`` at a node (measured on nvs22: 16–24 of 69 columns remap while the
# count is unchanged). Appending a pool row by position onto such a node then
# constrains the WRONG lifted variables → an invalid cut → a node holding the
# true optimum can be falsely Farkas-fathomed (the C-43 mechanism). The
# count-only guard (``sparse_cols == n_total``) cannot see this.
#
# The fix tags every pool row with the *identities* of the lifted columns it
# references (the structural term key — the ``(i,j)`` bilinear pair, the
# ``(i,p)`` monomial, the trilinear/multilinear tuple, the fractional power, the
# univariate-square base) and, per node, remaps each row's coefficients from
# root-column-identity → the node's current position for that SAME identity. A
# row that references a column whose identity is absent or changed at the node is
# INAPPLICABLE and is skipped (skipping an optional cut is always sound).


def column_identities(varmap: dict, n_total: int, n_orig: int) -> tuple:
    """Stable per-column identity keys for a lifted McCormick column layout.

    Returns a tuple of length ``n_total``. Position ``k`` carries a hashable
    identity that is *stable across builds* iff the column is either an original
    model variable (``("orig", k)``, always fixed) or an aux lifted by a
    structurally-keyed term map (bilinear ``(i,j)``, monomial ``(i,p)``,
    trilinear/multilinear tuple, fractional power, or univariate square). Any
    aux column not so claimed is tagged ``("opaque", k)`` — an identity that is
    only ever equal to itself at the *same* position, so a pool row that
    references it never remaps across a re-lifted layout and is skipped (sound).

    The univariate-square key ``(base_col, 2)`` is resolved through the *base*
    column's own identity, so a square of a lifted aux carries the aux's
    structural identity rather than a raw (unstable) column index.
    """
    ident: list = [("orig", k) if k < n_orig else None for k in range(n_total)]
    # Structurally-keyed aux maps: value is a column index, key is a stable
    # structural term identity (independent of build order / Python ``id``).
    for mapname in (
        "bilinear",
        "monomial",
        "trilinear",
        "multilinear",
        "fractional_power",
    ):
        m = varmap.get(mapname) or {}
        for key, col in m.items():
            if isinstance(col, (int, np.integer)) and 0 <= int(col) < n_total:
                ident[int(col)] = (mapname, key)
    # Univariate square: key ``(base_col, 2)``; resolve the base to its identity
    # (which may itself be a lifted aux) so the square is stably identified.
    usq = varmap.get("univariate_square") or {}
    for key, col in usq.items():
        if not (isinstance(col, (int, np.integer)) and 0 <= int(col) < n_total):
            continue
        base = key[0] if isinstance(key, tuple) and key else None
        base_id: object
        if isinstance(base, (int, np.integer)) and 0 <= int(base) < n_total:
            base_id = ident[int(base)]  # nested identity of the squared column
        else:
            base_id = base
        ident[int(col)] = ("univariate_square", (base_id, key[1] if len(key) > 1 else 2))
    # Any aux column left unclaimed: opaque, position-locked (never remaps).
    for k in range(n_total):
        if ident[k] is None:
            ident[k] = ("opaque", k)
    return tuple(ident)


def _remap_pool_rows(a_rows, b_rows, root_idents, node_idents, ncol: int):
    """Remap root-pool rows onto a node's column layout by column identity.

    ``a_rows`` (dense 2-D array, one row per pooled cut over the ROOT layout) and
    ``b_rows`` are remapped so each root column position is moved to the node
    position that carries the SAME identity (``root_idents``/``node_idents`` are
    the identity tuples from :func:`column_identities`). A row is emitted only if
    EVERY column it references with a nonzero coefficient has its identity present
    at the node; otherwise the row is inapplicable and is dropped (sound — fewer
    cuts only loosen the relaxation).

    Returns ``(A_remapped, b_remapped, n_kept, n_skipped)`` where ``A_remapped``
    is an ``(n_kept, ncol)`` dense array in the NODE column space (or ``None`` if
    no row survives)."""
    a = np.atleast_2d(np.asarray(a_rows, dtype=np.float64))
    b = np.asarray(b_rows, dtype=np.float64).ravel()
    if a.size == 0 or a.shape[0] == 0:
        return None, None, 0, 0
    # node identity -> node column position (first wins; identities are unique
    # per build by construction — orig cols are unique, structural keys unique,
    # opaque keys carry their own position).
    node_pos: dict = {}
    for pos, key in enumerate(node_idents):
        if key not in node_pos:
            node_pos[key] = pos
    n_root_cols = len(root_idents)
    out_rows = []
    out_b = []
    n_skipped = 0
    for r in range(a.shape[0]):
        row = a[r]
        nz = np.nonzero(np.abs(row) > 0.0)[0]
        new_row = np.zeros(ncol, dtype=np.float64)
        ok = True
        for c in nz:
            if c >= n_root_cols:
                ok = False  # references a column absent from the root identity map
                break
            rid = root_idents[c]
            # An opaque root column has no verifiable stable identity across builds
            # (only its own position), so a nonzero coefficient on one cannot be
            # soundly remapped — refuse the row. (In practice pool rows never touch
            # opaque columns; this is defense-in-depth.)
            if isinstance(rid, tuple) and rid and rid[0] == "opaque":
                ok = False
                break
            tgt = node_pos.get(rid)
            if tgt is None:
                ok = False  # this lifted term is not present at the node → skip row
                break
            new_row[tgt] += row[c]
        if ok:
            out_rows.append(new_row)
            out_b.append(b[r])
        else:
            n_skipped += 1
    if not out_rows:
        return None, None, 0, n_skipped
    A_out = np.array(out_rows, dtype=np.float64)
    b_out = np.array(out_b, dtype=np.float64)
    return A_out, b_out, len(out_rows), n_skipped


def _append_relax_rows(milp, A_rows, b_rows) -> None:
    """Append ``A_rows z <= b_rows`` to a relaxation model's inequality block."""
    import scipy.sparse as sp

    R = _as_csr(A_rows)
    b = np.asarray(b_rows, dtype=np.float64)
    if milp._A_ub is None:
        milp._A_ub, milp._b_ub = R, b
    elif sp.issparse(milp._A_ub):
        milp._A_ub = sp.vstack([milp._A_ub, R], format="csr")
        milp._b_ub = np.concatenate([np.asarray(milp._b_ub), b])
    else:
        milp._A_ub = np.vstack([np.asarray(milp._A_ub), R.toarray()])
        milp._b_ub = np.concatenate([np.asarray(milp._b_ub), b])


# When a per-node wall-clock budget is threaded through solve_at_node, no single
# internal re-solve is handed less than this floor (keeps a node that straddles
# the deadline from receiving a zero/negative budget the backend would reject).
_SOLVE_DEADLINE_FLOOR_S = 0.05

# Densification cap for the lifted relaxation. The matrix-form MILP backends
# (warm-started Rust simplex, POUNCE) materialize a DENSE ``(m, n+m)`` constraint
# array plus a dense ``m×m`` slack identity, so their footprint scales as
# ``(n_cols + n_rows) * n_rows`` cells. A binary-multilinear lift explodes the row
# count -- e.g. autocorr_bern's degree-4 objective lifts to ~3.7k cols × ~85k rows,
# whose dense form is ~7.5e9 cells (~60 GB): the allocation thrashes swap and never
# returns (a hang, not a bound). Above this cap the LP relaxer declines the node
# (returns no bound) and the spatial/integer B&B falls back to the rigorous
# alphaBB/interval underestimator -- a valid, if weaker, lower bound -- so the
# solve still returns its incumbent within the time limit instead of hanging.
# ~1e8 cells ≈ 800 MB dense: far above any well-posed relaxation in practice
# (typical MINLP lifts are < 1e6 cells) and far below the multi-GB blowup.
_MAX_RELAX_DENSE_CELLS = 1.0e8

# Lifted-LP FBBT (issue #184): number of feasibility-propagation sweeps over the
# relaxation rows, and the width left around a factor that propagation pins to a
# point so the build path keeps the multilinear term at full arity (see
# ``solve_at_node`` for why un-pinning is needed and why it stays sound).
_LIFTED_FBBT_ROUNDS = 30
_LIFTED_FBBT_TOL = 1e-9
_LIFTED_FBBT_UNPIN_EPS = 1e-6

# Lazy re-separation, relaxer-side arm (C-42, the THRU-4 follow-on). Under
# root-pool inheritance (``skip_pool_separators``) a node normally skips the
# per-node square/PSD point-separation loops — sound (skipping only loosens)
# but not free on every class: on tspn05-shape spatial models the inherited
# pool alone leaves each node too loose to fathom, the tree's bound stalls,
# and the certificate is lost at budget (THRU-4 graduation, #552). The full
# trigger is split between two layers:
#   (a) the DRIVER's global-bound-stall governor (``solver.py``,
#       ``_LAZY_RESEP_STALL_WINDOW`` there): when the tree's global lower
#       bound stagnates, it stops passing ``skip_pool_separators`` for a
#       bounded probe of node solves, and mutes again if the probe does not
#       move the bound;
#   (b) this module's stride safety net: every ``_LAZY_RESEP_STRIDE``-th
#       skip-eligible node solve runs the full separation pass regardless, so
#       inheritance can never fully starve a class the governor misjudges.
#
# Two per-NODE signals were tried first and FALSIFIED by measurement
# (2026-07-07, this branch — kept as a negative result per CLAUDE.md §4):
#   * "node LP value fails to improve on the parent's bound" fires on 185/191
#     nvs19 nodes — on the dense integer-QP class the node LP sitting at the
#     parent bound is the NORMAL state (closure comes from branching depth),
#     and the resulting re-separation destroys the 2–5× throughput win of
#     #551 (nvs19 loses its 53 s certificate again);
#   * "LP-value gain of the re-separation pass below a threshold" does not
#     discriminate either: on nvs19 the per-fire LP gain is LARGE (median
#     rel. gain 0.117) yet worthless for the 60 s certificate — separation
#     there is bound-productive but wall-unproductive.
# The discriminating signal is the GLOBAL bound's progress per wall spent,
# which only the driver can see — hence the split above.
#
# The stride is GLOBAL — never tuned per instance. Firing more only tightens
# a node's relaxation (every emitted cut is valid) and firing less only
# loosens it, so the net affects performance, never soundness. Sized by
# measurement: one separation pass on the dense integer-QP class costs
# seconds (THRU-3: the loops are 73%+12% of the nvs24 solve wall), and a
# 1-in-16 net measurably degraded nvs24's node throughput (49 -> 29 nodes at
# the same 60 s budget); 1-in-64 keeps the starvation guarantee at a bounded
# ~1.5 % worst-case node overhead.
_LAZY_RESEP_STRIDE = 64


def _lifted_lp_fbbt(
    A_ub: "object",
    b_ub: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    rounds: int = _LIFTED_FBBT_ROUNDS,
    tol: float = _LIFTED_FBBT_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Feasibility-based bound tightening on the relaxation's own rows.

    Propagates the lifted relaxation constraints ``A_ub · z <= b_ub`` (where ``z``
    spans the original variables *and* every McCormick product/monomial column)
    to tighten the column box ``[lo, hi]``. Because the McCormick rows of a
    product column ``w = x_i·x_j`` include facets such as ``w <= ub_j·x_i``,
    propagating a tightened ``w`` back through them recovers the *bilinear-implied*
    factor bounds that purely linear FBBT on the original model misses (issue
    #184: ``x18 = 0.0025·x9·x3`` with ``x18`` pinned to 1 forces ``x3 >= 1``).

    Soundness: every row is a valid relaxation inequality, so each derived bound
    is implied by valid constraints plus the incoming box — the result only ever
    *tightens* and never excludes a point feasible for the node. Returns the
    tightened ``(lo, hi)``; an empty result (``lo > hi`` on some column) is a
    rigorous proof the node's relaxation is infeasible.

    Vectorised over the sparse matrix (the per-row/per-element Python loop in the
    issue prototype is too slow for the per-node path). Each sweep computes, for
    every nonzero ``a_{r,j}``, the minimum activity of row ``r`` excluding column
    ``j`` and back-solves the tightest implied bound on ``z_j``; infinite
    contributions are handled by an explicit per-row infinity count so a single
    open bound still propagates.
    """
    import scipy.sparse as sp

    coo = sp.csr_matrix(A_ub).tocoo()
    rows = coo.row
    cols = coo.col
    vals = coo.data
    n_rows = coo.shape[0]
    lo = lo.copy()
    hi = hi.copy()
    pos = vals > 0
    neg = vals < 0

    for _ in range(rounds):
        # Bound each nonzero uses for the row's *minimum* activity: the lower
        # endpoint where the coefficient is positive, the upper where negative.
        bound_used = np.where(pos, lo[cols], hi[cols])
        term = vals * bound_used  # may be -inf when a bound is open in that direction
        is_ninf = ~np.isfinite(term)
        # Per-row tally of infinite terms and the finite partial sum.
        n_inf = np.zeros(n_rows)
        np.add.at(n_inf, rows, is_ninf.astype(np.float64))
        sum_fin = np.zeros(n_rows)
        np.add.at(sum_fin, rows, np.where(is_ninf, 0.0, term))
        # The activity excluding column j is finite iff every *other* term is
        # finite, i.e. the row's infinity count drops to zero once j is removed.
        valid = (n_inf[rows] - is_ninf.astype(np.float64)) == 0
        min_others = sum_fin[rows] - np.where(is_ninf, 0.0, term)
        nb = (b_ub[rows] - min_others) / vals
        changed = False

        sel = valid & pos
        if sel.any():
            new_hi = np.full(lo.shape[0], np.inf)
            np.minimum.at(new_hi, cols[sel], nb[sel])
            upd = new_hi < hi - tol
            if upd.any():
                hi = np.where(upd, np.minimum(hi, new_hi), hi)
                changed = True
        sel = valid & neg
        if sel.any():
            new_lo = np.full(lo.shape[0], -np.inf)
            np.maximum.at(new_lo, cols[sel], nb[sel])
            upd = new_lo > lo + tol
            if upd.any():
                lo = np.where(upd, np.maximum(lo, new_lo), lo)
                changed = True
        if not changed:
            break

    return lo, hi


@dataclass
class MccormickLPResult:
    """Outcome of one LP-form McCormick relaxation solve.

    The ``dual`` / ``col_status`` / ``safe_bound`` / ``reduced_costs`` fields are
    additive node-LP marginals (cert:T2.4a), populated only on the incremental
    fast path (``_try_incremental_node``) and only when the caller requests them
    (``want_marginals=True``). They are a pure side-channel for per-node
    duality-based reduction (cert:T2.4b ``reduce_node``); they never influence the
    reported ``lower_bound``/``x`` (those are byte-identical whether or not the
    marginals are computed), so a solve that does not request them is unchanged.
    """

    status: str
    lower_bound: Optional[float] = None
    x: Optional[np.ndarray] = None  # first ``n_orig`` columns of the LP solution
    # cert:T2.4a marginals (incremental fast path only; None otherwise).
    dual: Optional[np.ndarray] = None  # row duals y of the std-form node LP
    col_status: Optional[np.ndarray] = None  # final std-form column status (warm basis)
    safe_bound: Optional[float] = None  # Neumaier-Shcherbina safe LP lower bound (== lower_bound)
    reduced_costs: Optional[np.ndarray] = None  # d_j = c_j - (A^T y)_j for the ORIGINAL columns


class MccormickLPRelaxer:
    """Reusable per-node LP-form McCormick relaxation.

    Term classification and the empty :class:`DiscretizationState` are built
    once at construction time. Each call to :meth:`solve_at_node` rebuilds
    the lifted LP with the node's bound box and solves it via HiGHS.
    """

    def __init__(
        self,
        model: Model,
        *,
        superposition: bool = False,
        backend: str = "simplex",
        psd_cuts: bool = False,
        rlt_cuts: bool = False,
        rlt_level1: bool = False,
        build_incremental: bool = True,
    ) -> None:
        self._model = model
        self._terms: NonlinearTerms = classify_nonlinear_terms(model)
        # Opt-in M8 superposition cuts for bilinear-of-nonlinear products.
        self._superposition = superposition
        # Opt-in per-node PSD (moment) cut separation (W2e): enforce the lifted
        # moment matrix M = [[1, x^T],[x, X]] >= 0 by separating dense clique cuts
        # at each node, tightening the bound toward the SDP relaxation.
        self._psd_cuts = psd_cuts
        # Opt-in per-node targeted RLT (constraint-factor x bound-factor) cuts:
        # for each linear constraint and variable bound factor, separate the
        # violated product cut linearized over the lifted columns.
        self._rlt_cuts = rlt_cuts
        # LP backend for the per-node relaxation solve: "simplex" routes to the
        # pure-Rust warm-started simplex (no JAX in the spatial-B&B relaxation
        # path); "auto" keeps HiGHS->POUNCE. Falls back automatically if the
        # Rust binding is unavailable.
        self._backend = backend
        # Per-family separation timers (cert:T0.3). Accumulated across every
        # ``solve_at_node`` call for this relaxer instance; surfaced on the final
        # SolveResult.solver_stats. Pure instrumentation — never affects control
        # flow or the emitted cuts.
        self._sep_timers: dict[str, float] = {
            "multilinear": 0.0,
            "edge_concave": 0.0,
            "univariate_square": 0.0,
            "convex": 0.0,
            "psd": 0.0,
            "rlt": 0.0,
        }
        # THRU-3: count of nodes where the DISCOPT_SQUARE_COST_GATE gate fired
        # (budget or diminishing-returns early-exit). Pure instrumentation; used to
        # prove the default-OFF gate actually engages. Zero on the default path.
        self._square_gate_fires: int = 0
        # Root-cut-pool inheritance counters (THRU-4, ``DISCOPT_CUT_INHERIT``).
        # Pure instrumentation — never affects control flow or the emitted cuts:
        #   * inherited_nodes / inherited_rows: node solves that appended the
        #     root pool (and how many rows in total), across the cold AND fast
        #     paths;
        #   * skipped_separations: node solves where the square/PSD point
        #     separators were skipped in favour of the inherited pool;
        #   * dropped_nodes: cold node solves where the pool-augmented system
        #     produced no certified verdict and the pool rows were stripped for
        #     a no-pool retry (C-42 — the pool is an accelerator, never a
        #     dependency).
        #   * lazy_reseparations: skip-eligible node solves where the lazy
        #     trigger (bound stall vs the parent, or the stride safety net)
        #     re-enabled the square/PSD separation pass (C-42 Part 2).
        self._pool_stats: dict[str, int] = {
            "inherited_nodes": 0,
            "inherited_rows": 0,
            # C-44: pool rows dropped at a node because a lifted term they
            # reference is absent/remapped there (column-identity skip — sound).
            "inherited_rows_skipped": 0,
            "skipped_separations": 0,
            "dropped_nodes": 0,
            "lazy_reseparations": 0,
        }
        # Skip-eligible node-solve counter for the lazy-trigger stride net
        # (see the module-level ``_LAZY_RESEP_STRIDE`` rationale).
        self._lazy_skip_ctr: int = 0
        # Spatial-BB uses standard McCormick globally — no partitioning here.
        self._disc = DiscretizationState(partitions={})
        self._n_orig = sum(v.size for v in model._variables)
        # Pre-compute which original columns are integer/binary so that
        # integrality is preserved (only aux columns get relaxed).
        flags: list[int] = []
        for v in model._variables:
            flag = 1 if v.var_type in (VarType.BINARY, VarType.INTEGER) else 0
            flags.extend([flag] * v.size)
        self._orig_integrality = np.asarray(flags, dtype=np.int32)
        # A scaled affine power ``(c*x)**n`` (n>=3) is lifted by the build path but
        # is not catalogued in :class:`NonlinearTerms`; record its presence so the
        # rigorous LP relaxer engages on a model whose *only* nonlinearity is such a
        # power (issue #175) instead of falling back to an unsound nonconvex bound.
        self._has_affine_power = bool(_collect_affine_powers(model, set()))
        # Level-1 RLT (issue #175): multiply the model's linear constraints by
        # variable bound factors and lift the products, tightening the root bound
        # for high-degree-product instances (nvs20: 87.35 -> 91.74). It ~doubles
        # the relaxation size, so it is a root-bound tightener rather than a
        # per-node B&B default. Enabled via the first-class ``rlt_level1=True``
        # constructor argument (threaded from ``Model.solve(rlt=...)``), with the
        # legacy ``DISCOPT_RLT=1`` environment variable kept as a force-on
        # override for benchmarking. Applicable when the model has any constraint
        # to multiply by a bound factor — a *linear* constraint (Phase-1 RLT) OR a
        # *quadratic* one (Phase-2 quadratic-constraint RLT, issue #15). The old
        # gate required a linear constraint, which silently excluded pure-quadratic
        # models (e.g. the indefinite integer QPs nvs17/nvs24, whose constraints
        # are all quadratic) from RLT entirely — exactly the dense-QP instances
        # whose McCormick bound is hopelessly loose without it.
        self._rlt_applicable = (rlt_level1 or _tuning().rlt) and (
            bool(_linear_constraint_forms(model, self._n_orig))
            or bool(_quadratic_constraint_forms(model, self._n_orig))
        )
        # Lever 1 (issue #194): solve each spatial-B&B node's relaxation as a pure
        # LP for the dual bound, rather than as a nested integer MILP B&B. The LP
        # relaxation of the lifted McCormick MILP is still a valid (if looser)
        # lower bound, is ~5-10x cheaper than re-running an integer branch-and-bound
        # at every node, and returns a *fractional* solution the OUTER spatial/
        # integer tree can branch on (the old MILP node solve returned integer
        # values, which dead-ended the spatial path). Integrality is enforced by
        # the outer tree, not redundantly at every node. Set
        # ``DISCOPT_NODE_BOUND_MODE=milp`` to restore the legacy nested-MILP node
        # bound (for A/B comparison).
        self._lp_node_bound = _tuning().node_bound_mode == "lp"
        # Per-node lifted-LP FBBT (issue #184): propagate the relaxation's own
        # McCormick rows to recover bilinear-implied factor bounds (e.g. a pinned
        # binary forcing a continuous factor >= 1 *through* a bilinear constraint),
        # then rebuild the relaxation on the tightened box so its envelopes no
        # longer underestimate the product to zero. This is what lets the global
        # dual bound climb off a structural 0 on ``ex1252``. Opt-in via
        # ``DISCOPT_LIFTED_FBBT=1`` — it adds an FBBT sweep plus a conditional
        # relaxation rebuild per node, so it stays off the default B&B path.
        self._lifted_fbbt = _tuning().lifted_fbbt
        # Original columns that participate in a nonlinear (product/power) term.
        # A McCormick/RLT envelope for such a term is only valid when its
        # variables have FINITE bounds: over an unbounded box the lifted aux is
        # effectively free, so the relaxation has no valid finite lower bound for
        # that term and is genuinely unbounded. The fast Rust simplex nonetheless
        # fabricates a finite "optimal" there (himmel16: simplex says 0.0 / RLT
        # cuts -0.6749 where HiGHS correctly says "unbounded"), which would be
        # trusted as a too-high lower bound and certify a suboptimal incumbent.
        # We record these columns to gate a HiGHS unboundedness cross-check in
        # ``solve_at_node`` whenever any of them is still unbounded at a node.
        nl_cols: set[int] = set()
        t = self._terms
        for i, j in t.bilinear:
            nl_cols.update((int(i), int(j)))
        for i, j, k in t.trilinear:
            nl_cols.update((int(i), int(j), int(k)))
        for term in t.multilinear:
            nl_cols.update(int(c) for c in term)
        for base, _power in t.monomial:
            nl_cols.add(int(base))
        for base, _exp in t.fractional_power:
            nl_cols.add(int(base))
        self._nonlinear_cols = nl_cols

        # Incremental McCormick fast path (Phase B throughput). The default
        # ``solve_at_node`` cold-rebuilds the relaxation (``build_milp_relaxation``)
        # and re-equilibrates it every node — together ~half the spatial-B&B wall
        # clock. ``IncrementalMcCormickLP`` builds the LP *structure* once and per
        # node patches only the box-dependent product rows in closed form (numpy),
        # then warm-starts the Rust simplex (no equilibration). Its patched matrix
        # is validated row-for-row against the cold build at construction
        # (``IncrementalMcCormickLP._validate``), so the LP bound is sound; the fast
        # path is a strict valid relaxation and ``solve_at_node`` falls back to the
        # cold build for any node out of scope (near-unbounded box, unsupported
        # term, build/solve failure). Disable with ``DISCOPT_INCREMENTAL_MC=0``.
        self._inc = None
        self._inc_warm_basis = None
        self._inc_basis_nrows = -1
        # ``build_incremental=False`` skips this (the structure build + its
        # row-for-row validation cold-build the relaxation a handful of times) for
        # callers that only want the relaxer's model/terms/disc and never invoke
        # ``solve_at_node`` (e.g. structural OBBT), avoiding wasted cold builds.
        if build_incremental and os.environ.get("DISCOPT_INCREMENTAL_MC", "1") != "0":
            try:
                from discopt._jax.incremental_mccormick import IncrementalMcCormickLP

                # Scope gate (cert:T1.3): gate ONLY on the constructor's row-for-row
                # self-validation (`ok`), for ANY model — any variable mix, any
                # objective sense. The prior `_is_in_scope` (pure-integer, minimize)
                # was a conservative rollout limit inherited from the opt-in
                # lp_spatial engine (#355), not a soundness boundary: the fast path
                # solves the *McCormick LP relaxation* (a valid lower bound for
                # continuous, mixed, and integer models alike), and `_validate`
                # proves the patched rows reproduce the cold `build_milp_relaxation`
                # exactly; any uncovered term (univariate/NN-embedding smooth
                # activations, RLT-lifted rows, …) makes `_validate` fail → `ok=False`
                # → the trusted cold build runs unchanged.
                #
                # A first attempt at this widening collapsed the spatial bound
                # (``dispatch`` 3 → 9843 nodes) because the fast path returns before
                # the per-node separation chain and no root cut pool was built off
                # the PSD path. That is fixed by the *general* root-cut-pool capture
                # in solver.py (built whenever ``_inc`` is set) which the fast path
                # inherits; and by skipping the fast path during pool capture
                # (``out_cuts``) so the pool actually separates. Bound-changing
                # behaviour is verified by the differential-neutrality check.
                _inc = IncrementalMcCormickLP(model, self._terms)
                if _inc.ok:
                    self._inc = _inc
            except Exception:
                self._inc = None

        # Composite convex/concave OA lift detection is LAZY (see
        # ``_model_has_composite_lift``): computed at most once, and only the first
        # time a ``solve_at_node(separate=True)`` on the fast path with no inherited
        # pool actually needs it. Most relaxers (structural OBBT, pooled spatial
        # nodes, non-separating solves) never trigger it, so the extra cold build it
        # costs stays off their hot path.
        self._has_composite_lift_cache: Optional[bool] = None

    def _model_has_composite_lift(self) -> bool:
        """Whether the model carries a composite convex/concave OA lift whose
        Kelley tightening the incremental fast path would drop. Computed once and
        cached; only ever called from the fast-path bound-neutrality guard, so its
        one probe build stays off every other relaxer's hot path."""
        if self._has_composite_lift_cache is None:
            self._has_composite_lift_cache = False
            try:
                from discopt._jax.model_utils import flat_variable_bounds
                from discopt._jax.uniform_relax import build_uniform_relaxation

                _flb, _fub = flat_variable_bounds(self._model)
                _probe = build_uniform_relaxation(self._model, box=(_flb, _fub))
                self._has_composite_lift_cache = bool(_probe.composite_multivar_specs)
            except Exception:
                self._has_composite_lift_cache = False
        return self._has_composite_lift_cache

    @property
    def nonlinear_columns(self) -> frozenset[int]:
        """Original-variable flat columns in any nonlinear term (product,
        monomial, fractional power). Used by the spatial B&B driver to flag
        integer variables eligible for spatial domain-partition branching when no
        continuous variable remains to bisect (issue #194)."""
        return frozenset(self._nonlinear_cols)

    @property
    def has_bilinear(self) -> bool:
        """True if the model has any bilinear / trilinear / multilinear product."""
        return bool(self._terms.bilinear or self._terms.trilinear or self._terms.multilinear)

    @property
    def has_relaxable_nonlinearity(self) -> bool:
        """True if the model has any nonlinear term the LP relaxer can bound.

        :func:`build_milp_relaxation` emits valid outer-approximation cuts not
        only for products (bilinear/trilinear/multilinear) but also for
        ``monomial`` (x^n, n≥2) and ``fractional_power`` (x^p) terms. Gating the
        spatial LP relaxer on :attr:`has_bilinear` alone routes purely
        univariate-power nonconvex models to the McCormick "nlp" bound, which is
        not a valid dual bound for nonconvex models (issue #120). The LP
        relaxation is a rigorous polyhedral outer approximation for every term
        type here, so engaging it on these models yields a valid lower bound
        (and any per-node term it cannot relax safely errors out to "no bound"
        rather than an unsound one).
        """
        t = self._terms
        return bool(
            t.bilinear
            or t.trilinear
            or t.multilinear
            or t.monomial
            or t.fractional_power
            or t.general_nl
            or self._has_affine_power
        )

    def _try_incremental_node(
        self,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
        inherited_cuts: Optional[tuple],
        *,
        want_marginals: bool = False,
    ) -> Optional["MccormickLPResult"]:
        """Incremental McCormick node solve: patch the cached structure + warm-start,
        instead of a cold ``build_milp_relaxation`` + equilibration. Returns a
        :class:`MccormickLPResult` on success or ``None`` to fall back to the cold
        build. Sound: the patched matrix is validated equal to the cold build at
        construction, so the pure-LP value is a valid lower bound (integrality is
        branched by the outer tree); the inherited root cut pool is appended only
        when its column layout matches exactly (a mismatch is skipped — sound).

        When ``want_marginals`` is set, the returned result additionally carries the
        node LP's row duals, safe bound, and the ORIGINAL-column reduced costs
        (cert:T2.4a). Those are computed from the same solve — no extra LP — and
        never change ``lower_bound``/``x``."""
        inc = self._inc
        if inc is None:
            return None
        lb = np.asarray(node_lb, dtype=np.float64).ravel()
        ub = np.asarray(node_ub, dtype=np.float64).ravel()
        if lb.size != inc.n or ub.size != inc.n:
            return None
        # No valid finite McCormick envelope over a near-unbounded box on a
        # nonlinear column: route to the cold path (clamp + HiGHS unboundedness
        # cross-check). Mirrors the cold-path conditioning guard.
        if self._nonlinear_cols:
            idx = np.fromiter(self._nonlinear_cols, dtype=int)
            if np.any(np.abs(lb[idx]) >= _RELAX_NUMERIC_CAP) or np.any(
                np.abs(ub[idx]) >= _RELAX_NUMERIC_CAP
            ):
                return None
        try:
            cut_rows = None
            if inherited_cuts is not None:
                a_rows = inherited_cuts[0]
                b_rows = inherited_cuts[1]
                root_idents = inherited_cuts[2] if len(inherited_cuts) > 2 else None
                if a_rows is not None and _sparse_rows(a_rows) > 0:
                    a_rows = _as_csr(a_rows).toarray()
                    b_rows = np.asarray(b_rows, dtype=np.float64).ravel()
                    node_idents = getattr(inc, "col_identities", None)
                    # A pool captured over a WIDER column layout than this incremental
                    # structure carries (``a_rows.shape[1] != inc.ncol``) cannot be
                    # applied here — decline the incremental path (fall to the cold
                    # build, which re-lifts to the pool's own layout and can inherit
                    # it there). This preserves the pre-C-44 routing (the count check
                    # gated inheritance to the cold path on such models, e.g.
                    # dispatch: pool 23 cols vs incremental 10 cols) while C-44 makes
                    # the equal-count case IDENTITY-safe below.
                    _count_ok = (
                        a_rows.ndim == 2
                        and a_rows.shape[1] == inc.ncol
                        and len(b_rows) == a_rows.shape[0]
                    )
                    if not _count_ok:
                        return None
                    # C-44: at the SAME column count, remap the root pool from its
                    # ROOT column identities onto this structure's layout by matching
                    # identity — the old count-only gate appended a row by position
                    # even when a re-lift remapped which lifted variable each position
                    # means (same count, different semantics → the C-43 false-fathom).
                    # A row referencing a lifted term this layout does not carry at
                    # that count is skipped (sound — fewer cuts only loosen).
                    if root_idents is not None and node_idents is not None:
                        A_rm, b_rm, n_kept, n_skip = _remap_pool_rows(
                            a_rows, b_rows, root_idents, node_idents, inc.ncol
                        )
                        self._pool_stats["inherited_rows_skipped"] += n_skip
                        if A_rm is not None and n_kept > 0:
                            cut_rows = list(zip(A_rm, b_rm))
                            self._pool_stats["inherited_nodes"] += 1
                            self._pool_stats["inherited_rows"] += n_kept
                    elif root_idents is None or node_idents is None:
                        # Untagged pool or unavailable node identities (defensive
                        # fallback): legacy count-gated positional append (we already
                        # passed the equal-count precondition). Should not occur once
                        # every capture tags idents and the structure builds them.
                        cut_rows = list(zip(a_rows, b_rows))
                        self._pool_stats["inherited_nodes"] += 1
                        self._pool_stats["inherited_rows"] += len(cut_rows)
            A, b, bounds = inc.assemble(lb, ub, cut_rows=cut_rows)
            nrows = int(A.shape[0])
            # Densification guard (Issue #20), same cap as the cold path below.
            # T1.3 widened the fast path to any model, so a large multilinear lift
            # can now reach it; decline an oversize node here too rather than force
            # a multi-GB dense solve. Sound: no bound is returned, so the caller
            # keeps the rigorous alphaBB/interval underestimator (identical to the
            # cold-path decline). Falling back to the cold build would only hit the
            # same guard, so return the oversize verdict directly.
            if (inc.ncol + nrows) * nrows > _MAX_RELAX_DENSE_CELLS:
                return MccormickLPResult(status="skipped_oversize")
            in_basis = (
                self._inc_warm_basis
                if (self._inc_warm_basis is not None and self._inc_basis_nrows == nrows)
                else None
            )
            _solved = inc.solve_assembled_full(
                A, b, bounds, in_basis=in_basis, return_cert=want_marginals
            )
            if want_marginals:
                status, bound, x_full, basis, farkas_certified, cert = _solved
            else:
                status, bound, x_full, basis, farkas_certified = _solved
                cert = None
        except Exception:
            logger.debug("incremental McCormick node failed; cold fallback", exc_info=True)
            return None

        if status == "optimal" and bound is not None and np.isfinite(bound):
            self._inc_warm_basis = basis
            self._inc_basis_nrows = nrows
            x_orig = np.asarray(x_full, dtype=np.float64)[: self._n_orig]
            res = MccormickLPResult(status="optimal", lower_bound=float(bound), x=x_orig)
            if want_marginals and cert is not None and cert.dual is not None:
                # Original-column reduced costs d_j = c_j - (A^T y)_j from THIS solve
                # (no extra LP). ``A`` is the constraint matrix (m x ncol) and ``y``
                # the row duals of the std-form ``[A | I] z = b`` system, so it has
                # one entry per row of ``A``. Only the first ``n_orig`` structural
                # columns are needed by DBBT / RC-fixing (aux columns are branched
                # by the tree, never reduced here).
                try:
                    y = np.asarray(cert.dual, dtype=np.float64)
                    n0 = self._n_orig
                    A_arr = np.asarray(A, dtype=np.float64)
                    if A_arr.ndim == 2 and A_arr.shape[0] == y.shape[0] and A_arr.shape[1] >= n0:
                        c_full = np.asarray(inc.c, dtype=np.float64)
                        rc = c_full[:n0] - (A_arr[:, :n0].T @ y)
                        res.reduced_costs = rc
                        res.dual = y
                        res.col_status = cert.col_status
                        res.safe_bound = (
                            float(cert.safe_bound) if cert.safe_bound is not None else float(bound)
                        )
                except Exception:
                    logger.debug("node-LP marginal extraction failed (non-fatal)", exc_info=True)
            return res

        if status == "infeasible":
            # An empty McCormick polytope over a FINITE box is a rigorous
            # infeasibility proof for this node's subtree (the relaxation is a valid
            # outer approximation), so the node is fathomed WITHOUT a cold rebuild.
            # A *verified Farkas dual ray* (issue #356) makes that verdict rigorous
            # with no second solve at all: the ray is an independent certificate
            # that the lifted LP's feasible set is empty, so when it checks out we
            # fathom directly — closing the #355-review gap where the incremental
            # infeasible-trust path dropped the cold path's independent cross-check.
            if farkas_certified:
                return MccormickLPResult(status="infeasible")
            # No verified Farkas ray. When the solve was WARM-STARTED, the verdict is
            # untrustworthy: the in-house dual simplex can converge from a stale
            # cross-node basis to a *false* ``infeasible`` on a feasible LP whose ray
            # does not certify emptiness (C-38: kall_circles_c8a — the warm basis of
            # one node's box, reused on a sibling's, drove a spurious infeasible that
            # fathomed the sub-box holding the true optimum, both prematurely
            # terminating the search AND certifying a false-optimal dual bound). A
            # cold solve of the IDENTICAL assembled system (``in_basis=None``) is
            # authoritative — the warm-start artifact vanishes — so re-solve cold once
            # before concluding. A genuine emptiness re-emerges as a Farkas-certified
            # infeasible (fathom); a false one re-solves to ``optimal`` (keep the node
            # with its valid bound). This restores both soundness and completeness.
            if in_basis is not None:
                try:
                    (
                        c_status,
                        c_bound,
                        c_x,
                        _c_basis,
                        c_farkas,
                    ) = inc.solve_assembled_full(A, b, bounds, in_basis=None)
                except Exception:
                    c_status = None
                    c_bound = None
                    c_x = None
                    c_farkas = False
                if c_status == "optimal" and c_bound is not None and np.isfinite(c_bound):
                    x_orig = np.asarray(c_x, dtype=np.float64)[: self._n_orig]
                    return MccormickLPResult(status="optimal", lower_bound=float(c_bound), x=x_orig)
                if c_status == "infeasible" and c_farkas:
                    return MccormickLPResult(status="infeasible")
                # Cold solve inconclusive (uncertified infeasible / numerical): fall
                # through to the equilibration re-verify below.
            # The ray did not verify (an ill-conditioned candidate): fall back to
            # the equilibration re-verify, which fathoms ONLY on a verified Farkas
            # ray and otherwise defers to the cold rebuild (never trusts an
            # uncertified infeasible — see :meth:`_reverify_incremental_infeasible`).
            return self._reverify_incremental_infeasible(inc, A, b, bounds)

        # time_limit / unbounded / numerical error: no certified verdict — fall back
        # to the trusted cold build.
        return None

    def _reverify_incremental_infeasible(
        self, inc, A: np.ndarray, b: np.ndarray, bounds: np.ndarray
    ) -> Optional["MccormickLPResult"]:
        """Confirm an incremental ``infeasible`` verdict soundly, without a cold
        rebuild, when the simplex's Farkas ray did not already certify it. A node
        fathom on ``infeasible`` is rigorous ONLY when a *verified Farkas dual ray*
        proves the lifted polytope empty; the raw simplex verdict is not itself a
        proof. So this re-solves once with exact geometric-mean equilibration (a
        feasible-set-preserving rescale) and accepts ``infeasible`` **only** when
        that solve returns a verified Farkas ray. If the equilibrated solve recovers
        a feasible point, that was a false infeasible (return it as ``optimal``).
        Any other outcome — infeasible with no Farkas ray, or a solver failure —
        yields ``None`` (cold fallback), NOT a fathom.

        C-38 rationale: the in-house simplex returns *numerical false* ``infeasible``
        on ill-conditioned lifted McCormick LPs that HiGHS proves feasible, and it
        does so with no Farkas ray both cold AND after equilibration (coefficient
        spread as low as ~1e2 — well under any conditioning heuristic). The old
        ``if not ill: return infeasible`` and ``if status=="infeasible": return
        infeasible`` both trusted such an uncertified verdict and fathomed the
        sub-box containing the true optimum, certifying a false-optimal dual bound
        (kall_circles_c8a: 3.6142 > true opt 2.5409). A conditioning heuristic is not
        a soundness proof — only a Farkas ray is — so an uncertified infeasible must
        never fathom.

        Returns ``MccormickLPResult(status="infeasible")`` (Farkas-certified) to
        fathom, an ``"optimal"`` result if equilibration recovers a feasible point,
        or ``None`` (cold fallback) otherwise."""
        import scipy.sparse as sp

        from discopt._jax.milp_relaxation import equilibrate_relaxation_lp

        try:
            a_csr = sp.csr_matrix(A)
        except Exception:
            return None

        try:
            bl = [(float(bounds[i, 0]), float(bounds[i, 1])) for i in range(bounds.shape[0])]
            c2, a2, b2, bd2, col_scale = equilibrate_relaxation_lp(inc.c, a_csr, b, bl, None)
            status, bound, x_s, _, farkas = inc.solve_assembled_full(
                a2, b2, np.asarray(bd2, dtype=np.float64), in_basis=None, c_override=c2
            )
        except Exception:
            return None  # re-verify failed -> trusted cold rebuild
        if status == "infeasible":
            # Fathom ONLY on a verified Farkas ray; an uncertified infeasible is not a
            # rigorous emptiness proof and must not prune (C-38). Cold fallback lets
            # the driver keep the node open on its inherited (valid) parent bound.
            if farkas:
                return MccormickLPResult(status="infeasible")
            logger.debug(
                "incremental McCormick infeasible not Farkas-certified after "
                "equilibration; declining fathom (cold fallback) to avoid a "
                "false-infeasible prune"
            )
            return None
        if status == "optimal" and bound is not None and np.isfinite(bound):
            # False infeasible recovered: map the scaled point back (x = D·x'). The
            # equilibrated basis has different scaling, so don't carry it as a warm
            # start (the ``nrows`` guard would reject it anyway).
            x_orig = (np.asarray(x_s, dtype=np.float64) * col_scale)[: self._n_orig]
            return MccormickLPResult(status="optimal", lower_bound=float(bound), x=x_orig)
        return None

    def solve_at_node(
        self,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
        time_limit: Optional[float] = None,
        *,
        inherited_cuts: Optional[tuple] = None,
        separate: bool = True,
        out_cuts: Optional[list] = None,
        psd_max_rounds: int = 8,
        want_marginals: bool = False,
        skip_pool_separators: bool = False,
    ) -> MccormickLPResult:
        """Solve the McCormick LP relaxation restricted to the given bound box.

        Thin wrapper around :meth:`_solve_at_node_impl` that enforces the C-43
        pool-soundness invariant: **an inherited pool row may only accelerate the
        node solve; it may never be the sole reason a node is fathomed.** When a
        non-empty pool was on offer (``inherited_cuts`` present) and the
        pool-augmented solve returns ``infeasible``, the node is re-solved WITHOUT
        the pool; the ``infeasible`` fathom is kept only if the pool-free
        relaxation is *also* infeasible.

        Why (C-43, nvs22): the code assumed a cut separated at the ROOT box is
        valid on every sub-box (so an inherited row could only tighten, never cut
        a feasible point). That premise is **false for at least one captured cut
        family** — measured on nvs22: a node whose box contains the true optimum
        ``(x0,x1,x2,x3)=(5,1,1,2)`` becomes *Farkas-certified infeasible* once the
        pool is appended, though the pool-free relaxation of the same box solves
        to ``optimal``. The Farkas ray certifies the *pool-augmented* polytope is
        empty, which is a real proof only if every pool row is valid on the box;
        when a pool row is invalid there, it is a **false fathom** that closes the
        tree around a suboptimal incumbent and certifies a false optimum
        (nvs22: ``optimal 33.55`` vs oracle ``6.0582``). C-42 (#553) extended the
        same "pool is an accelerator, never a dependency" guarantee to the
        *no-certified-verdict* branch of the cold path; C-43 (#564) extends it to
        the *Farkas-infeasible* branch across BOTH the incremental fast path and
        the cold path — the branch C-42's cold-path-only, verdict-gated retry did
        not cover. Re-solving pool-free only *loosens* the relaxation, so this can
        never introduce a false fathom of its own; it only forgoes a possible (and
        possibly-invalid) prune, then keeps the node open on its valid parent
        bound. See ``docs/dev/c43-nvs22-fix-graduate-2026-07-08.md``.

        See :meth:`_solve_at_node_impl` for the full parameter contract.
        """
        res = self._solve_at_node_impl(
            node_lb,
            node_ub,
            time_limit,
            inherited_cuts=inherited_cuts,
            separate=separate,
            out_cuts=out_cuts,
            psd_max_rounds=psd_max_rounds,
            want_marginals=want_marginals,
            skip_pool_separators=skip_pool_separators,
        )
        # C-43 pool-infeasible re-verification. Only relevant when (a) this was a
        # regular node solve (not a root pool-capture call, which passes no pool),
        # (b) inherited pool rows were on offer, and (c) the augmented solve
        # fathomed the node as infeasible. Re-solve the IDENTICAL box with no pool
        # (and no pool-separator skip — a pool-free node must see the full chain,
        # exactly like the default path); trust the ``infeasible`` fathom only when
        # the pool-free relaxation confirms it. A pool row that is genuinely valid
        # on this box cannot flip a feasible relaxation to empty, so on the common
        # (sound-pool) case the re-solve re-confirms ``infeasible`` and the verdict
        # stands — the cost is one extra LP on the *rare* pool-infeasible node, not
        # on the hot feasible path. Not pre-gated on the pool's column width: a
        # column-mismatched pool that was never appended yields an identical
        # pool-free re-solve, so at worst this spends one extra LP re-confirming a
        # genuine infeasible — it can never *keep* a false fathom, which is the
        # invariant that matters.
        #
        # This is a runtime GUARD, not a root fix: the underlying defect is that a
        # pool row stated over the root's lifted column layout can be applied at a
        # node whose re-lifted/pinned layout changed the columns' meaning (same
        # count, different semantics — column remapping), so the row is no longer
        # valid there. #567 tracks the source fix (column-identity-safe inheritance)
        # that would make this guard inert; until then it keeps the certificate
        # sound. Part of the #396 backlog.
        if out_cuts is None and res.status == "infeasible" and _pool_has_rows(inherited_cuts):
            # Cheap first: the BASE McCormick relaxation (no separators, no pool) is
            # the loosest valid outer approximation. If IT is already infeasible, the
            # node's subtree is genuinely empty (a valid relaxation with an empty
            # feasible set is a rigorous fathom) and the pool did not cause it — keep
            # the original ``infeasible`` verdict without a full re-separated re-solve.
            # This is the hot path for a *sound* pool (its infeasibles re-confirm here
            # in one loose LP), so inheritance keeps its throughput where the pool is
            # valid. Only when the base relaxation is FEASIBLE was the fathom
            # pool-induced (a pool row cut the non-empty node empty) — then pay the
            # full pool-free re-solve to recover the node with the tightest valid
            # bound the default path would have produced.
            base_free = self._solve_at_node_impl(
                node_lb,
                node_ub,
                time_limit,
                inherited_cuts=None,
                separate=False,
                out_cuts=None,
                psd_max_rounds=psd_max_rounds,
                want_marginals=False,
                skip_pool_separators=False,
            )
            if base_free.status != "infeasible":
                # The pool row(s) caused a FALSE fathom of a non-empty node. Re-solve
                # pool-free WITH the full separation chain (byte-identical to the
                # default path's node solve) so the recovered node carries a bound as
                # tight as the pool-free tree would give, then trust that valid
                # relaxation. Count the recovery — same role as C-42's
                # ``dropped_nodes``.
                pool_free = self._solve_at_node_impl(
                    node_lb,
                    node_ub,
                    time_limit,
                    inherited_cuts=None,
                    separate=separate,
                    out_cuts=None,
                    psd_max_rounds=psd_max_rounds,
                    want_marginals=want_marginals,
                    skip_pool_separators=False,
                )
                self._pool_stats["dropped_nodes"] += 1
                # Adopt the pool-free result unconditionally: it is a valid relaxation
                # of the identical box (optimal → recovered node with a sound bound;
                # infeasible → the *default-path* separators tightened the loose base
                # to empty, a rigorous fathom, since every separated cut is valid).
                return pool_free
        return res

    def _solve_at_node_impl(
        self,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
        time_limit: Optional[float] = None,
        *,
        inherited_cuts: Optional[tuple] = None,
        separate: bool = True,
        out_cuts: Optional[list] = None,
        psd_max_rounds: int = 8,
        want_marginals: bool = False,
        skip_pool_separators: bool = False,
    ) -> MccormickLPResult:
        """Solve the McCormick LP relaxation restricted to the given bound box.

        Returns a :class:`MccormickLPResult`. ``lower_bound`` is a valid lower
        bound on the original problem within this box (for minimization).
        ``x`` is the LP solution projected to the original variable columns.
        On any solver failure, ``status != "optimal"`` and the LB is ``None``.

        Global cut pool (P1, see ``docs/design/global-cut-pool.md``):

        * ``inherited_cuts``: ``(A_rows, b_rows)`` of cut rows separated once at
          the root, appended to this node's relaxation before solving. Applied
          only when the lifted column layout matches the pool's (a tightened child
          box can pin/fold columns); a mismatch skips them, which is sound (fewer
          cuts only loosens the bound). NOTE: a pool row is NOT assumed valid on
          every sub-box — see :meth:`solve_at_node`, which re-verifies any
          pool-augmented ``infeasible`` against a pool-free solve (C-43).
        * ``separate``: when ``False`` skip the per-node cut-separation chain (use
          the inherited pool instead of re-deriving cuts every node).
        * ``out_cuts``: when a list is supplied, the rows the separation chain
          appended at THIS call are pushed onto it as ``(A_rows, b_rows)`` — used
          to capture the root pool once and replay it at every node.
        * ``skip_pool_separators``: skip ONLY the two point-separation loops the
          root pool already covers — the univariate-square tangent loop and the
          PSD (moment) loop — while keeping the rest of the chain. Set by the
          driver at non-pool node solves under ``DISCOPT_CUT_INHERIT`` (THRU-4):
          those loops each re-derive the pool's cut families via up to 8 full
          MILP re-solves per node (73% + 12% of the nvs24 solve wall, THRU-3),
          and every cut they emit is valid independent of the node box (a square
          tangent under-estimates ``x**2`` everywhere; a PSD eigencut
          ``vᵀMv ≥ 0`` holds wherever ``X = x xᵀ``), so the inherited root pool
          already supplies the family and skipping re-separation only *loosens*
          this node's relaxation — never cuts a feasible point. The driver's
          global-stall governor and this module's stride net (C-42, see
          ``_LAZY_RESEP_STRIDE``) selectively re-enable the pass.
        """
        # Phase-B fast path: incremental patch + warm-started solve, reusing the
        # structure built once at construction instead of a per-node cold rebuild +
        # equilibration. Sound (validated patch == cold build; a valid relaxation),
        # and it carries the inherited (root RLT/PSD) cut pool so it keeps the
        # default path's bound strength. Returns ``None`` for any out-of-scope node,
        # falling through to the trusted cold build below.
        #
        # cert:T1.3: skip the fast path when *capturing* a cut pool (``out_cuts``
        # set). The fast path returns before the per-node separation chain, so it
        # would capture nothing; a pool-building call must run the cold, separating
        # path. Regular nodes (``out_cuts is None``) still take the fast path and
        # inherit the pool this captures. Without this, the root cut pool is never
        # populated once the incremental engine is active — exactly why the T1.3
        # gate flip collapsed the spatial bound (dispatch 3 → 9843).
        # Bound-neutrality: with a composite convex lift present, a FEASIBLE fast-path
        # bound omits the lift's Kelley tightening unless it can inherit those cuts
        # from the pool (``inherited_cuts``). When separation is requested with no
        # inherited pool, fall back to the cold, separating build for an OPTIMAL
        # result so the lift actually tightens the node (fixes test_convex_claimer's
        # -204 -> -350 on a direct solve_at_node call). An ``infeasible`` fast verdict
        # is bound-independent (an empty McCormick polytope over a finite box is a
        # rigorous infeasibility proof), so it is always trusted; pooled spatial
        # nodes, which DO inherit cuts, keep the fast path for both verdicts.
        _skip_fast_for_lift = (
            separate
            and self._inc is not None
            and not inherited_cuts
            and self._model_has_composite_lift()
        )
        if out_cuts is None:
            _fast = self._try_incremental_node(
                node_lb, node_ub, inherited_cuts, want_marginals=want_marginals
            )
            if _fast is not None and not (_skip_fast_for_lift and _fast.status == "optimal"):
                return _fast

        try:
            milp, varmap = build_milp_relaxation(
                self._model,
                self._terms,
                self._disc,
                bound_override=(
                    np.asarray(node_lb, dtype=np.float64),
                    np.asarray(node_ub, dtype=np.float64),
                ),
                superposition=self._superposition,
                rlt_level1=self._rlt_applicable,
            )
        except Exception:
            # Build failures here are otherwise invisible: the result silently
            # becomes status="error", which propagates up to a top-level
            # status="error" SolveResult with no diagnostic. Log the full
            # traceback at DEBUG (this is a per-node hot path, so keep it off the
            # default log level) so the underlying cause is recoverable.
            logger.debug("McCormick LP relaxation build failed", exc_info=True)
            return MccormickLPResult(status="error")

        # Densification guard: decline nodes whose lifted relaxation would force a
        # multi-GB dense allocation in the matrix-form backend (see
        # ``_MAX_RELAX_DENSE_CELLS``). Returning no bound here is sound -- it only
        # forgoes this node's LP underestimator; the caller keeps the rigorous
        # alphaBB/interval bound and the incumbent search, so the solve respects
        # its time limit instead of hanging on the dense solve.
        n_cols = int(np.size(milp._c))
        _a_ub = milp._A_ub
        n_rows = 0 if _a_ub is None else int(_a_ub.shape[0])
        if (n_cols + n_rows) * n_rows > _MAX_RELAX_DENSE_CELLS:
            key = (n_cols, n_rows).__hash__()
            if key not in _oversize_warned:
                _oversize_warned.add(key)
                logger.warning(
                    "McCormick LP relaxation too large to solve densely "
                    "(%d cols x %d rows ~ %.1e dense cells > cap %.0e); declining "
                    "the per-node LP bound and falling back to the rigorous "
                    "alphaBB/interval underestimator.",
                    n_cols,
                    n_rows,
                    float((n_cols + n_rows) * n_rows),
                    _MAX_RELAX_DENSE_CELLS,
                )
            return MccormickLPResult(status="skipped_oversize")

        # Per-node lifted-LP FBBT (issue #184): tighten the box by propagating the
        # relaxation's own rows, then rebuild on the tightened original-variable
        # box. The rebuild is what realises the bound improvement — tightening a
        # factor's domain (``x0,x3 in [1,3]``) regenerates the McCormick envelope
        # of ``x0·x3`` so it no longer underestimates the product to 0 in the box
        # interior. Linear FBBT on the original model cannot reach this because the
        # forcing runs through *bilinear* constraints; the relaxation's product
        # rows expose it. Sound at every step (see :func:`_lifted_lp_fbbt`).
        if self._lifted_fbbt:
            try:
                fb = self._lifted_fbbt_rebuild(milp, node_lb, node_ub)
                if fb is not None:
                    # Adopt the rebuilt relaxation *and* its varmap together: the
                    # downstream separation routines index ``milp``'s product
                    # columns through ``varmap``, so a stale map would address the
                    # wrong columns of the rebuilt LP.
                    milp, varmap = fb
            except Exception:
                # Tightening is a best-effort accelerator; on any failure keep the
                # original (already valid) relaxation rather than losing the node.
                pass

        # Pad original integrality flags to the lifted column count; aux cols
        # remain continuous (flag 0). If the model has no integers this is a
        # pure LP anyway.
        #
        # Under level-1 RLT the relaxation is solved as a pure LP (integrality
        # dropped): RLT is an opt-in root-bound tightener whose LP bound — made
        # tighter than the integer-aware relaxation without it (nvs20: 87.35 ->
        # 91.74) by the product cuts — is far cheaper than the RLT-augmented MILP's
        # branch-and-bound. The bound stays a valid lower bound either way.
        n_total = len(milp._c)
        if self._rlt_applicable or self._lp_node_bound:
            # Pure-LP node bound (lever 1, issue #194): drop integrality entirely.
            # A valid lower bound for the outer tree; integrality is branched there.
            milp._integrality = None
        elif n_total > self._n_orig:
            pad = np.zeros(n_total - self._n_orig, dtype=np.int32)
            milp._integrality = np.concatenate([self._orig_integrality, pad])
        else:
            milp._integrality = self._orig_integrality
        if milp._integrality is not None and not int(milp._integrality.sum()):
            milp._integrality = None

        # Soundness/conditioning guard (issue #15): the warm-started Rust simplex
        # mishandles effectively-infinite *finite* variable bounds -- the ``1e20``
        # sentinel discopt assigns to unbounded variables -- and declares a
        # premature "optimal" at a wrong value. On ex9_2_6's root McCormick LP the
        # simplex returns 0.0 where HiGHS returns -406.0; the bogus 0.0 (which
        # exceeds the true optimum) then prunes the root and certifies a
        # suboptimal incumbent -- a false-"optimal", the worst failure class.
        # Clamp any bound whose magnitude reaches the effective-infinity cap to a
        # true +/-inf before solving. This only WIDENS a variable's box, so it
        # merely enlarges the relaxed feasible set and the LP optimum stays a
        # valid (never-too-high) lower bound; it is a no-op on well-scaled boxes
        # and routes genuine unbounded variables through the simplex's correct
        # free-variable handling. Mirrors the root path's
        # ``sanitize_relaxation_for_conditioning``.
        _cap = _RELAX_NUMERIC_CAP
        milp._bounds = [
            (
                lo if abs(lo) < _cap else (-np.inf if lo < 0 else np.inf),
                hi if abs(hi) < _cap else (np.inf if hi > 0 else -np.inf),
            )
            for (lo, hi) in milp._bounds
        ]

        # The caller's ``time_limit`` is the budget for the WHOLE node, but this
        # method re-solves the relaxation many times (the initial solve, up to
        # 8+6 cut-separation rounds, and up to two HiGHS soundness re-verifies).
        # Passing the same duration to each gave every re-solve the full budget,
        # so one node could run (1 + rounds + verifies) x time_limit — e.g.
        # du-opt's hard relaxation overshot a 2s budget to 7.3s wall, and the
        # spatial B&B (one such node per step) blew a 25s limit out to ~75s.
        # Convert the duration to a single deadline and hand each internal solve
        # only the time that remains, so the node's TOTAL respects the budget.
        _deadline = None if time_limit is None else time.perf_counter() + time_limit

        def _remaining() -> Optional[float]:
            if _deadline is None:
                return None
            return max(_deadline - time.perf_counter(), _SOLVE_DEADLINE_FLOOR_S)

        # Root cut-pool inheritance (P1 + C-44): append globally-valid cut rows
        # separated once at the root instead of re-separating them every node.
        # Each pool row is a valid inequality at every feasible point regardless
        # of the node box — but it is stated over the ROOT's lifted column
        # positions, and this node's re-built / re-lifted layout can carry
        # *different* lifted variables at the same positions (same count,
        # different semantics — the C-43 remapping). So instead of the old
        # count-only gate (which appended a positionally-wrong row), remap each
        # row from its ROOT column identities onto THIS build's column positions
        # by :func:`column_identities`; a row referencing a lifted term this node
        # does not carry is skipped (sound — fewer cuts only loosen the bound).
        _n_pre_pool_rows = 0 if milp._A_ub is None else _sparse_rows(milp._A_ub)
        _pool_rows_appended = 0
        if inherited_cuts is not None:
            _ia = inherited_cuts[0]
            _ib = inherited_cuts[1]
            _root_idents = inherited_cuts[2] if len(inherited_cuts) > 2 else None
            if _ia is not None and _sparse_rows(_ia) > 0:
                _node_idents = None
                try:
                    _node_idents = column_identities(varmap, n_total, self._n_orig)
                except Exception:
                    logger.debug("node column-identity build failed; skipping pool", exc_info=True)
                # Equal-count precondition (matches the legacy ``sparse_cols ==
                # n_total`` gate): only inherit when the pool was captured over the
                # same column count as this build, so C-44 stays behaviour-neutral on
                # the layouts the legacy gate rejected and only *remaps* (never newly
                # inherits) at equal count — exactly where the C-43 false-fathom lived.
                _count_ok = _sparse_cols(_ia) == n_total
                if _root_idents is not None and _node_idents is not None and _count_ok:
                    _ia_dense = _as_csr(_ia).toarray()
                    _A_rm, _b_rm, _n_kept, _n_skip = _remap_pool_rows(
                        _ia_dense, _ib, _root_idents, _node_idents, n_total
                    )
                    self._pool_stats["inherited_rows_skipped"] += _n_skip
                    if _A_rm is not None and _n_kept > 0:
                        _append_relax_rows(milp, _A_rm, _b_rm)
                        _pool_rows_appended = _n_kept
                        self._pool_stats["inherited_nodes"] += 1
                        self._pool_stats["inherited_rows"] += _pool_rows_appended
                elif (_root_idents is None or _node_idents is None) and _count_ok:
                    # Untagged pool or unavailable node identities (defensive
                    # fallback): legacy count-gated positional append. Should not
                    # occur once every capture site tags identities and the node
                    # build produces them; kept so it degrades safely.
                    _append_relax_rows(milp, _ia, _ib)
                    _pool_rows_appended = _sparse_rows(_ia)
                    self._pool_stats["inherited_nodes"] += 1
                    self._pool_stats["inherited_rows"] += _pool_rows_appended
        _n_base_rows = 0 if milp._A_ub is None else _sparse_rows(milp._A_ub)

        res = milp.solve(time_limit=_remaining(), backend=self._backend)

        # C-42: the inherited pool is an ACCELERATOR, never a dependency — a node
        # solve must be no worse with it than without it. The warm/equilibrated
        # in-house simplex can fail numerically on the pool-augmented system even
        # though the pool rows are valid (nvs06: one valid root PSD eigencut flips
        # the root node's integer-aware relaxation solve from ``optimal`` to an
        # uncertified ``infeasible``/``iteration_limit`` — the C-38 failure class,
        # now triggered by the extra row instead of a stale basis). Losing the
        # node bound is what truncated the flag-ON B&B loop at the root (the
        # driver's deliberately-skipped node NLP leaves the failure sentinel in
        # place, the sentinel prunes the root non-rigorously, and the tree
        # exhausts after one node). So: when pool rows were appended and the
        # solve produced no certified verdict — not ``optimal`` and not a
        # Farkas-certified ``infeasible`` — strip the pool rows and re-solve.
        # Dropping valid rows only loosens the relaxation (sound), and the retry
        # is byte-identical to the no-pool solve the default path performs, so
        # the pool can perturb neither the incumbent search nor loop termination.
        # The per-node square/PSD skip is likewise lifted for this node: its
        # justification ("the pool already supplies the family") no longer holds
        # once the pool is dropped here.
        if _pool_rows_appended and not (
            res.status == "optimal"
            or (res.status == "infeasible" and getattr(res, "farkas_certified", False))
        ):
            _a_ub_cur = milp._A_ub
            if _n_pre_pool_rows == 0 or _a_ub_cur is None:
                milp._A_ub, milp._b_ub = None, None
            else:
                milp._A_ub = _a_ub_cur[:_n_pre_pool_rows]
                milp._b_ub = np.asarray(milp._b_ub)[:_n_pre_pool_rows]
            _n_base_rows = _n_pre_pool_rows
            skip_pool_separators = False
            self._pool_stats["dropped_nodes"] += 1
            res = milp.solve(time_limit=_remaining(), backend=self._backend)

        # Lazy re-separation, stride safety net (C-42): every
        # ``_LAZY_RESEP_STRIDE``-th skip-eligible node solve runs the full
        # square/PSD separation pass regardless of the driver's global-stall
        # governor, so pool inheritance can never fully starve a class the
        # governor misjudges. Separating more only tightens (each emitted cut
        # is valid), so the net is performance-only — see the constants above.
        if skip_pool_separators and separate:
            self._lazy_skip_ctr += 1
            if self._lazy_skip_ctr % _LAZY_RESEP_STRIDE == 0:
                skip_pool_separators = False
                self._pool_stats["lazy_reseparations"] += 1

        # Snapshot the pre-separation solve. Every separated cut is a valid
        # inequality, so the pre-separation LP optimum is a valid lower bound on the
        # separated (tighter) system too. On a wide/ill-conditioned box a valid cut
        # can nonetheless flip the warm simplex to an uncertifiable re-solve (safe
        # bound lost AND the integer-aware dual bound not recomputed → no finite
        # bound), which would make separation *degrade* the node to no-bound. Keep
        # this snapshot as a floor so separation can never make a node less
        # certifiable than it was before (sound: the floor is a valid lower bound).
        _presep_res = res
        if separate:
            _st = self._sep_timers  # cert:T0.3 per-family separation timers
            # On-demand separation of the exact multilinear hull for products with
            # more factors than the dense RLT cap (those carry only the loose
            # recursive chain). Every separated cut is a supporting hyperplane of
            # the convex/concave envelope, hence valid; adding them only tightens
            # the bound, so the loop is sound at any round.
            _t = time.perf_counter()
            res = self._separate_multilinear(milp, varmap, res, _deadline)
            _st["multilinear"] += time.perf_counter() - _t
            # Edge-concave / edge-convex quadratic blocks: tighten the joint
            # vertex-polyhedral envelope (cuts on bilinear/square auxes).
            _t = time.perf_counter()
            res = self._separate_edge_concave(milp, varmap, res, _deadline)
            _st["edge_concave"] += time.perf_counter() - _t
            # Univariate squares ``s = x**2``: the static envelope cuts only at the
            # box endpoints, so deep inside a wide box the convex underestimator is
            # slack. Add the exact supporting tangent at the LP point each round.
            # Under root-pool inheritance (THRU-4, ``skip_pool_separators``) the
            # inherited pool already carries the root's tangents and this loop —
            # up to 8 full MILP re-solves — is skipped; every tangent is a global
            # underestimator of ``x**2``, so the pool rows stay valid on any
            # sub-box and skipping only loosens the node bound (sound).
            if skip_pool_separators:
                self._pool_stats["skipped_separations"] += 1
            else:
                _t = time.perf_counter()
                res = self._separate_univariate_square(milp, varmap, res, _deadline)
                _st["univariate_square"] += time.perf_counter() - _t
            # Convex/concave composite lifts (#358): add the exact supporting
            # hyperplane at the LP point, closing the gap the fixed reference-point
            # gradient cuts leave. Inert unless the convex claimer lifted a node.
            _t = time.perf_counter()
            res = self._separate_convex(milp, varmap, res, _deadline)
            _st["convex"] += time.perf_counter() - _t
            # PSD (moment) cuts: enforce M = [[1,x^T],[x,X]] >= 0 over fully-lifted
            # cliques. Each cut v^T M v >= 0 is valid for every feasible point
            # (X = x x^T), so adding them only tightens the bound. Skipped under
            # root-pool inheritance (THRU-4): the eigencuts are valid wherever the
            # lifting relations hold — independent of the node box — so the
            # inherited root pool already supplies the family.
            if not skip_pool_separators:
                _t = time.perf_counter()
                res = self._separate_psd(milp, varmap, res, _deadline, max_rounds=psd_max_rounds)
                _st["psd"] += time.perf_counter() - _t
            # Targeted RLT (constraint-factor x bound-factor) cuts.
            _t = time.perf_counter()
            res = self._separate_rlt(milp, varmap, res, _deadline)
            _st["rlt"] += time.perf_counter() - _t

        # Capture the rows the separation chain just appended, for the root cut
        # pool. Stated over this node's lifted column space (``n_total`` columns).
        # C-44: tag the chunk with the per-column IDENTITIES of this build's
        # layout (``column_identities``), so a node inheriting the pool can remap
        # each row from these root-column identities onto its own (possibly
        # re-lifted) column positions — or skip a row whose lifted term the node
        # does not carry. The identity vector is captured from the SAME
        # ``varmap`` the separated rows are stated over (post-FBBT-rebuild if that
        # fired, since ``milp``/``varmap`` are swapped together above).
        if out_cuts is not None and milp._A_ub is not None:
            _A = _as_csr(milp._A_ub)
            if _A.shape[0] > _n_base_rows:
                _n_total_cap = int(np.size(milp._c))
                try:
                    _idents = column_identities(varmap, _n_total_cap, self._n_orig)
                except Exception:
                    logger.debug(
                        "column-identity capture failed; pool chunk untagged", exc_info=True
                    )
                    _idents = None
                out_cuts.append(
                    (
                        _A[_n_base_rows:].copy(),
                        np.asarray(milp._b_ub)[_n_base_rows:].copy(),
                        _idents,
                    )
                )

        # Rigorous bound / fathom from pure-Rust certificates (issue #356) — no
        # HiGHS cross-check. The four failure modes the old ``milp.solve(
        # backend="auto")`` guards protected against (false infeasible, unconverged
        # non-optimal, too-high optimal, fabricated-finite unbounded) are now
        # handled by the certificate the warm simplex itself produces:
        #
        #  * infeasible  -> ``milp.solve`` already re-verifies the verdict with an
        #    exact, feasible-set-preserving equilibration re-solve (pure Rust) — the
        #    same soundness bar the incremental path's ``_reverify`` accepts — and
        #    surfaces ``farkas_certified`` when a verified Farkas dual ray
        #    additionally proves the lifted polytope empty. Either is sound to
        #    fathom (the McCormick relaxation is a valid outer approximation); this
        #    is the pure-Rust replacement for the old HiGHS cross-check.
        #  * unbounded / iteration_limit / numerical -> no certified finite bound,
        #    so report the status with no lower bound and let the driver branch.
        #  * optimal -> use the Neumaier–Shcherbina *safe* lower bound built from
        #    the simplex's own row duals (``res.safe_bound``), which is ``<=`` the
        #    true LP optimum at *any* conditioning — a drifted vertex objective can
        #    never be reported as the bound, so the too-high failure class is
        #    eliminated by construction rather than caught by a second solve. When
        #    no safe bound is computable (the lifted LP has a free variable — e.g.
        #    the objective epigraph — whose reduced cost makes the safe-bound box
        #    term unbounded), fall back to the warm simplex's vertex objective: it
        #    equilibrates internally (so a wide coefficient spread is handled in the
        #    factorization, not left to drift the vertex) and verifies dual
        #    feasibility on exact reduced costs before declaring optimal, so its
        #    reported optimum is the converged LP value. A genuinely unbounded
        #    relaxation returns "unbounded" above (not "optimal"), so this is never
        #    a fabricated finite bound.
        if res.status == "infeasible":
            # A node fathom on ``infeasible`` is rigorous ONLY when the verdict is
            # backed by a *verified Farkas dual ray* (``res.farkas_certified``): the
            # ray is an independent certificate that the lifted McCormick polytope is
            # empty. The warm/equilibrated in-house simplex can otherwise return a
            # *numerical false* ``infeasible`` on an ill-conditioned lifted relaxation
            # that is in fact feasible — and it does so with NO Farkas ray (C-38:
            # kall_circles_c8a's reverse-convex non-overlap relaxation, coefficient
            # spread only ~1e2–1e4, is declared infeasible cold AND after
            # equilibration though HiGHS proves it feasible; trusting that fathomed
            # the sub-box holding the true optimum and certified a false-optimal dual
            # bound 3.6142 > true opt 2.5409). ``milp.solve``'s equilibration
            # re-verify only fires above a 1e3 spread and, even when it does, the same
            # simplex can re-confirm the false infeasible — so equilibration is not a
            # sufficient guard. A verified Farkas ray is the only rigorous proof of LP
            # emptiness, so without it the verdict is not a proof and must NOT fathom:
            # report a non-fathoming status so the driver keeps the node open on its
            # inherited (valid) parent bound and branches, exactly as for an
            # ``iteration_limit``/``numerical`` exit. This forgoes a *possible* prune,
            # never a valid bound — sound by construction.
            if res.farkas_certified:
                return MccormickLPResult(status="infeasible")
            logger.debug(
                "McCormick LP node reported infeasible without a Farkas certificate; "
                "treating as inconclusive (no fathom) to avoid a false-infeasible prune"
            )
            return MccormickLPResult(status="numerical")
        if res.status != "optimal" or res.x is None:
            # Objective-floor fallback (issue #640 Bucket 2, nvs22). The conditioning
            # clamp above widens a free non-cost column to true +/-inf, which can make
            # the fast simplex spuriously report ``unbounded`` on a relaxation whose
            # OBJECTIVE is provably bounded below (its cost columns are never
            # near-inf, so they are never clamped). The relaxation's rigorous
            # box-interval objective floor is a valid global lower bound in that case,
            # so report it rather than declining — a sound (never-too-high) bound that
            # keeps the node from being dropped for want of a certificate. Only fires
            # for a finite floor (a genuinely unbounded-below objective has none).
            floor = getattr(milp, "_objective_floor", None)
            if res.status == "unbounded" and floor is not None and np.isfinite(floor):
                return MccormickLPResult(status="optimal", lower_bound=float(floor))
            return MccormickLPResult(status=res.status)

        def _certify(r) -> Optional[float]:
            """The valid lower bound this optimal solve certifies (or None)."""
            if r is None or r.status != "optimal" or r.x is None:
                return None
            if self._backend == "auto":
                # HiGHS/POUNCE already returns a trustworthy optimum; keep the
                # legacy behaviour (no certificate is produced on that path).
                b: Optional[float] = r.objective
            elif r.safe_bound is not None:
                # The common pure-LP warm-simplex path: rigorous safe bound.
                b = r.safe_bound
            elif milp._integrality is not None:
                # Integer-aware node bound (non-default ``node_bound_mode="milp"``):
                # the engine's own B&B dual bound is the valid lower bound here.
                b = r.bound
            elif self._nonlinear_cols and self._has_unbounded_nonlinear_col(milp):
                # A nonlinear-participating variable is still unbounded at this node,
                # so the McCormick/RLT envelope may be genuinely UNBOUNDED — and the
                # fast simplex can fabricate a finite "optimal" there (himmel16 with
                # RLT cuts). With no computable safe bound (free variable) we cannot
                # certify the vertex is not too high, so decline: report no bound and
                # let the driver branch, rather than trust a fabricated value that
                # would fathom the optimal region. (Pure-Rust replacement of the old
                # HiGHS unbounded-relaxation cross-check.)
                b = None
            elif self._max_finite_magnitude(milp) <= _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE:
                # Pure-LP with a free variable but a bounded relaxation: trust the
                # internally-equilibrated, dual-feasibility-verified vertex objective
                # only when well-conditioned (mirrors the old guard's conditioning
                # gate — below it the vertex carries no meaningful drift).
                b = r.objective
            else:
                # Free variable AND ill-conditioned beyond where the fast simplex is
                # reliable, with no computable safe bound: decline (branch) rather
                # than risk a too-high vertex value (pure-Rust replacement of the old
                # too-high-optimal HiGHS cross-check).
                b = None
            return b if (b is not None and np.isfinite(b)) else None

        bound = _certify(res)
        x_source = res
        if bound is None and _presep_res is not res:
            # Separation left the node uncertifiable (a valid cut flipped the warm
            # simplex on a wide box); fall back to the pre-separation certified bound
            # — still a valid lower bound on the separated system, so sound.
            presep_bound = _certify(_presep_res)
            if presep_bound is not None:
                bound, x_source = presep_bound, _presep_res

        if bound is None or not np.isfinite(bound):
            return MccormickLPResult(status=res.status)
        x_orig = np.asarray(x_source.x)[: self._n_orig].copy()
        return MccormickLPResult(status="optimal", lower_bound=float(bound), x=x_orig)

    def _lifted_fbbt_rebuild(self, milp, node_lb, node_ub):
        """Tighten the node box via lifted-LP FBBT and rebuild the relaxation.

        Returns ``(MilpRelaxationModel, varmap)`` built on the FBBT-tightened
        original-variable box, or ``None`` to keep the input relaxation unchanged
        (no original bound tightened, an empty box, a rebuild failure, or a
        rebuild that would *lose* a previously valid objective bound).
        """
        if milp._A_ub is None:
            return None

        bnds = np.asarray(milp._bounds, dtype=np.float64)
        lo = bnds[:, 0].copy()
        hi = bnds[:, 1].copy()
        node_lb = np.asarray(node_lb, dtype=np.float64)
        node_ub = np.asarray(node_ub, dtype=np.float64)

        lo, hi = _lifted_lp_fbbt(milp._A_ub, milp._b_ub, lo, hi)

        n = self._n_orig
        new_lb = lo[:n].copy()
        new_ub = hi[:n].copy()

        # An empty box anywhere is a rigorous infeasibility proof, but routing that
        # verdict through a non-LP path would bypass the HiGHS re-verification the
        # caller relies on for "infeasible". Conservatively skip the rebuild and let
        # the (still valid) original relaxation solve report the node's status.
        if np.any(new_lb > new_ub + _LIFTED_FBBT_TOL):
            return None

        # Only rebuild if an *original* variable's domain actually tightened — a
        # rebuild that reproduces the same box is wasted work, and at shallow nodes
        # (binaries still relaxed) FBBT typically tightens nothing useful.
        tol = 1e-7
        tightened = np.any(new_lb > node_lb[:n] + tol) or np.any(new_ub < node_ub[:n] - tol)
        if not tightened:
            return None

        # Un-pin factors FBBT drove to a point. When a multilinear factor (e.g.
        # ``x18``) is pinned to ``[1, 1]``, ``build_milp_relaxation`` folds the
        # degree-4 objective term to a lower-arity product whose aux column it never
        # allocated, and drops ``objective_bound_valid`` ("Trilinear (0,3,15) not in
        # map"). Leaving a hair of width keeps the term at full arity. Widening a
        # bound only *enlarges* the relaxation box (a superset of the proven point),
        # so it stays a valid relaxation — never unsound. Branch-pinned variables
        # (already a point on entry) are left pinned.
        for i in range(n):
            entry_pinned = (node_ub[i] - node_lb[i]) <= _LIFTED_FBBT_TOL
            now_pinned = (new_ub[i] - new_lb[i]) <= _LIFTED_FBBT_TOL
            if now_pinned and not entry_pinned:
                w = max(1.0, abs(new_lb[i])) * _LIFTED_FBBT_UNPIN_EPS
                if new_lb[i] > node_lb[i] + _LIFTED_FBBT_TOL:
                    new_lb[i] -= w
                else:
                    new_ub[i] += w

        rebuilt, rebuilt_varmap = build_milp_relaxation(
            self._model,
            self._terms,
            self._disc,
            bound_override=(new_lb, new_ub),
            superposition=self._superposition,
            rlt_level1=self._rlt_applicable,
        )

        # Never let tightening regress a node: if the rebuild would drop the
        # objective bound that the original relaxation had, keep the original.
        if milp._objective_bound_valid and not rebuilt._objective_bound_valid:
            return None
        return rebuilt, rebuilt_varmap

    def _has_unbounded_nonlinear_col(self, milp) -> bool:
        """True if any nonlinear-term original column has a non-finite bound.

        After the ``solve_at_node`` clamp, a variable that was effectively
        unbounded (``|bound| >= _RELAX_NUMERIC_CAP``) carries a true ``+/-inf``
        bound. A McCormick/RLT envelope over such a column is not a valid finite
        relaxation, so the LP is genuinely unbounded and the fast simplex's
        finite "optimal" cannot be trusted.
        """
        bounds = milp._bounds
        for col in self._nonlinear_cols:
            if col >= len(bounds):
                continue
            lo, hi = bounds[col]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return True
        return False

    @staticmethod
    def _max_finite_magnitude(milp) -> float:
        """Largest finite magnitude across the LP's data (cost, rows, RHS, bounds).

        A cheap conditioning proxy used only on the rare free-variable LPs where
        the Neumaier–Shcherbina safe bound is not computable: below the threshold a
        dual-feasible optimal vertex carries no meaningful drift and is trusted;
        above it the bound is declined. Non-finite entries (the +/-inf bounds left
        by the clamp) are ignored.
        """
        import scipy.sparse as sp

        worst = 0.0
        A = milp._A_ub
        if A is not None:
            data = A.data if sp.issparse(A) else np.asarray(A).ravel()
            if np.size(data):
                fin = np.abs(data[np.isfinite(data)])
                if fin.size:
                    worst = max(worst, float(fin.max()))
        for arr in (milp._b_ub, milp._c):
            if arr is not None and np.size(arr):
                a = np.abs(np.asarray(arr, dtype=np.float64).ravel())
                fin = a[np.isfinite(a)]
                if fin.size:
                    worst = max(worst, float(fin.max()))
        for lo, hi in milp._bounds:
            for v in (lo, hi):
                if np.isfinite(v):
                    worst = max(worst, abs(float(v)))
        return worst

    def _separate_multilinear(self, milp, varmap, res, deadline):
        """Tighten products beyond the dense RLT cap by on-demand hull separation.

        For each multilinear/trilinear product with more factors than
        ``DISCOPT_MULTILINEAR_RLT_MAX`` (the dense-cut cap), repeatedly solve the
        relaxation, separate the convex/concave envelope cuts the current point
        violates (``multilinear_separation``), append them, and re-solve — up to
        a small round limit. Each cut is a valid supporting hyperplane, so the
        returned bound is always sound; on any failure the input ``res`` is
        returned unchanged.
        """
        if not _tuning().multilinear_separate:
            return res
        if res is None or res.status != "optimal" or res.x is None:
            return res
        try:
            import scipy.sparse as sp

            from discopt._jax.multilinear_separation import separate_multilinear_envelope

            cap = _tuning().multilinear_rlt_max
            n_total = len(milp._c)
            # Build (factor-columns, product-column) specs for over-cap products.
            specs: list[tuple[list[int], int]] = []
            for src in ("multilinear", "trilinear"):
                for term, prod_col in (varmap.get(src) or {}).items():
                    cols = sorted(set(int(c) for c in term))
                    if len(cols) > cap and len(cols) == len(set(term)):
                        specs.append((cols, int(prod_col)))
            if not specs:
                return res

            def _append(rows: list[np.ndarray], rhs: list[float]) -> None:
                R = np.asarray(rows, dtype=np.float64)
                b = np.asarray(rhs, dtype=np.float64)
                if milp._A_ub is None:
                    milp._A_ub = R
                    milp._b_ub = b
                elif sp.issparse(milp._A_ub):
                    milp._A_ub = sp.vstack([milp._A_ub, sp.csr_matrix(R)], format="csr")
                    milp._b_ub = np.concatenate([milp._b_ub, b])
                else:
                    milp._A_ub = np.vstack([np.asarray(milp._A_ub), R])
                    milp._b_ub = np.concatenate([np.asarray(milp._b_ub), b])

            for _round in range(8):
                # Stop separating once the node's wall-clock budget is spent;
                # each round costs a full MILP re-solve.
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                x = np.asarray(res.x, dtype=np.float64)
                rows: list[np.ndarray] = []
                rhs: list[float] = []
                for cols, prod_col in specs:
                    lb = np.array([milp._bounds[c][0] for c in cols], dtype=np.float64)
                    ub = np.array([milp._bounds[c][1] for c in cols], dtype=np.float64)
                    xs = x[cols]
                    ws = float(x[prod_col])
                    for cut in separate_multilinear_envelope(lb, ub, xs, ws):
                        row = np.zeros(n_total)
                        if cut.sense == "under":
                            # w >= a.x + b  ->  a.x - w <= -b
                            for d, c in enumerate(cols):
                                row[c] += float(cut.a[d])
                            row[prod_col] += -1.0
                            rows.append(row)
                            rhs.append(-float(cut.b))
                        else:
                            # w <= a.x + b  ->  w - a.x <= b
                            for d, c in enumerate(cols):
                                row[c] += -float(cut.a[d])
                            row[prod_col] += 1.0
                            rows.append(row)
                            rhs.append(float(cut.b))
                if not rows:
                    break
                _append(rows, rhs)
                _tl = (
                    None
                    if deadline is None
                    else max(deadline - time.perf_counter(), _SOLVE_DEADLINE_FLOOR_S)
                )
                new_res = milp.solve(time_limit=_tl, backend=self._backend)
                if new_res.status != "optimal" or new_res.objective is None:
                    break
                res = new_res
            return res
        except Exception:
            return res

    def _separate_univariate_square(self, milp, varmap, res, deadline):
        """Tighten lifted univariate squares ``s = x**2`` by tangent separation.

        The static relaxation pins ``s`` with tangents only at the box endpoints
        (and 0), so deep inside a wide box the convex underestimator ``s >= x**2``
        is slack: on ex9_2_6 each ``s_i - 2 x_i`` is driven down to ~``-|box|``,
        giving a root LP bound of ~-200 against a true optimum of -1. Each round
        we add the *exact* supporting tangent at the current LP point ``x0``:
        ``s >= 2 x0 x - x0**2``. A tangent of a convex function is a global
        underestimator and every original point has ``s = x**2 >= 2 x0 x - x0**2``,
        so no feasible point is ever cut -- the bound stays sound at any round.
        On any failure the input ``res`` is returned unchanged.
        """
        if not _tuning().square_separate:
            return res
        if res is None or res.status != "optimal" or res.x is None:
            return res
        try:
            import scipy.sparse as sp

            monomial = varmap.get("monomial") or {}
            # (base_col, aux_col) for every lifted univariate square ``x**2``.
            specs: list[tuple[int, int]] = []
            for key, aux in monomial.items():
                base, power = key
                if int(power) == 2:
                    specs.append((int(base), int(aux)))
            if not specs:
                return res
            n_total = len(milp._c)

            def _append(rows: list[np.ndarray], rhs: list[float]) -> None:
                R = np.asarray(rows, dtype=np.float64)
                b = np.asarray(rhs, dtype=np.float64)
                if milp._A_ub is None:
                    milp._A_ub, milp._b_ub = R, b
                elif sp.issparse(milp._A_ub):
                    milp._A_ub = sp.vstack([milp._A_ub, sp.csr_matrix(R)], format="csr")
                    milp._b_ub = np.concatenate([milp._b_ub, b])
                else:
                    milp._A_ub = np.vstack([np.asarray(milp._A_ub), R])
                    milp._b_ub = np.concatenate([milp._b_ub, b])

            tol = 1e-7
            # Cost-aware gate (THRU-3, DISCOPT_SQUARE_COST_GATE, default OFF). This
            # per-node loop is the ungated twin of the PSD loop (THRU-2a): both run
            # up to 8 full MILP re-solves per node, and THRU-3 measured *this* one —
            # not PSD — as the dominant per-node cost on dense integer-product QPs
            # (nvs24/nvs19). When on, cap this node's square-separation wall to
            # ``budget × base_solve_wall`` and abandon on per-round diminishing
            # returns. SOUND: dropping tangent cuts can only loosen the relaxation —
            # never cut a feasible point or cross the optimum. The gate only ever
            # *shortens* the loop, so the default (gate-off) path is bit-identical to
            # the pre-THRU-3 code. Keys purely on observed per-node cost/bound-delta
            # (§0.2, general). See docs/dev/thru3-node-decomp-2026-07-07.md.
            _tun = _tuning()
            _gate = _tun.square_cost_gate
            _gate_budget = _tun.square_cost_gate_budget
            _gate_tau = _tun.square_cost_gate_tau
            _sq_t0 = time.perf_counter()
            _base_solve_wall: Optional[float] = None
            for _round in range(8):
                # Each round costs a full re-solve; stop once the budget is spent.
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                if (
                    _gate
                    and _base_solve_wall is not None
                    and (time.perf_counter() - _sq_t0) > _gate_budget * _base_solve_wall
                ):
                    # Per-node square-separation wall budget spent — stop.
                    self._square_gate_fires += 1
                    break
                x = np.asarray(res.x, dtype=np.float64)
                rows: list[np.ndarray] = []
                rhs: list[float] = []
                for base, aux in specs:
                    if base >= x.size or aux >= x.size:
                        continue
                    x0 = float(x[base])
                    s = float(x[aux])
                    if not (np.isfinite(x0) and np.isfinite(s)):
                        continue
                    # Conditioning guard (mirrors ``_separate_convex``): on a very
                    # wide box the tangent ``s >= 2 x0 x - x0**2`` has coefficient
                    # ``2 x0`` and intercept ``x0**2`` that blow up (``x0`` ~ 1e15
                    # on st_miqp4's ``[0,1e15]`` square), and a cut with ~1e30
                    # entries fools the fast simplex into an uncertifiable basis —
                    # the node then reports no finite bound at all. Skip it; the
                    # static envelope still bounds ``s`` and the relaxation stays
                    # sound, just looser at this point (dropping a cut only loosens).
                    if abs(x0) > _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE:
                        continue
                    # Separate only where the convex underestimator is slack:
                    # the LP put ``s`` below the true parabola at ``x0``.
                    if x0 * x0 - s > tol * max(1.0, abs(x0 * x0)):
                        row = np.zeros(n_total)
                        row[base] += 2.0 * x0
                        row[aux] += -1.0
                        rows.append(row)
                        rhs.append(x0 * x0)
                if not rows:
                    break
                _lb_before = res.objective if _gate else None
                _append(rows, rhs)
                _tl = (
                    None
                    if deadline is None
                    else max(deadline - time.perf_counter(), _SOLVE_DEADLINE_FLOOR_S)
                )
                _solve_t0 = time.perf_counter()
                new_res = milp.solve(time_limit=_tl, backend=self._backend)
                if _gate and _base_solve_wall is None:
                    _base_solve_wall = max(time.perf_counter() - _solve_t0, 1e-4)
                if new_res.status != "optimal" or new_res.objective is None:
                    break
                res = new_res
                if _gate and _lb_before is not None:
                    _delta = new_res.objective - _lb_before
                    if _delta <= _gate_tau * (1.0 + abs(_lb_before)):
                        # Diminishing returns — abandon the remaining rounds.
                        self._square_gate_fires += 1
                        break
            return res
        except Exception:
            return res

    def _separate_convex(self, milp, varmap, res, deadline):
        """Tighten lifted convex/concave composite nodes by supporting-hyperplane
        separation at the LP point (issue #358 Phase 2).

        A convex subexpression ``g`` lifted to aux ``d`` (the #358 claimer) carries
        only a few static gradient cuts at fixed reference points, so deep inside
        the box ``d >= g(x)`` is slack. Each round add the EXACT supporting tangent
        at the current LP point ``x0``:

            convex:  d >= g(x0) + ∇g(x0)·(x − x0)
            concave: d <= g(x0) + ∇g(x0)·(x − x0)

        A tangent of a convex (resp. secant-free concave) function is a global
        under- (over-) estimator, so no feasible point is ever cut — the bound
        stays sound at every round. Sound no-op on any failure.
        """
        relaxations = varmap.get("composite_multivar_relaxations") or []
        specs = [
            r
            for r in relaxations
            if getattr(r, "value_fn", None) is not None
            and getattr(r, "grad_fn", None) is not None
            and r.idxs
        ]
        if not specs:
            return res
        if res is None or res.status != "optimal" or res.x is None:
            return res
        try:
            import jax.numpy as jnp
            import scipy.sparse as sp

            n_total = len(milp._c)
            n_orig = self._n_orig

            def _append(rows: list[np.ndarray], rhs: list[float]) -> None:
                R = np.asarray(rows, dtype=np.float64)
                b = np.asarray(rhs, dtype=np.float64)
                if milp._A_ub is None:
                    milp._A_ub, milp._b_ub = R, b
                elif sp.issparse(milp._A_ub):
                    milp._A_ub = sp.vstack([milp._A_ub, sp.csr_matrix(R)], format="csr")
                    milp._b_ub = np.concatenate([milp._b_ub, b])
                else:
                    milp._A_ub = np.vstack([np.asarray(milp._A_ub), R])
                    milp._b_ub = np.concatenate([milp._b_ub, b])

            tol = 1e-7
            for _round in range(8):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                x = np.asarray(res.x, dtype=np.float64)
                xv = jnp.asarray(x[:n_orig], dtype=jnp.float64)
                rows: list[np.ndarray] = []
                rhs: list[float] = []
                for r in specs:
                    aux = r.aux_col
                    if aux >= x.size:
                        continue
                    d = float(x[aux])
                    try:
                        gval = float(jnp.reshape(r.value_fn(xv), ()))
                        grad = np.asarray(r.grad_fn(xv), dtype=np.float64).ravel()
                    except Exception:
                        continue
                    if not np.isfinite(gval) or not all(
                        j < grad.size and np.isfinite(grad[j]) for j in r.idxs
                    ):
                        continue
                    # Conditioning guard (#358): a cut whose coefficients have blown
                    # up (steep gradient on a wide box) fools the fast simplex into a
                    # garbage bound. Skip it — the static cuts still bound ``d`` and
                    # the relaxation stays sound, just looser at this point.
                    if any(
                        abs(float(grad[j])) > _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE for j in r.idxs
                    ):
                        continue
                    # ``slope·x0`` with x0 the LP point restricted to dependent cols.
                    slope_dot_x0 = float(sum(float(grad[j]) * float(x[j]) for j in r.idxs))
                    if r.curvature == "convex":
                        # slack iff the LP put d below g at x0; cut d >= g(x0)+∇g·(x−x0)
                        # i.e. ∇g·x − d <= ∇g·x0 − g(x0).
                        if gval - d > tol * max(1.0, abs(gval)):
                            row = np.zeros(n_total)
                            for j in r.idxs:
                                row[j] += float(grad[j])
                            row[aux] += -1.0
                            rows.append(row)
                            rhs.append(slope_dot_x0 - gval)
                    else:  # concave: d <= g(x0)+∇g·(x−x0)  ->  −∇g·x + d <= g(x0)−∇g·x0
                        if d - gval > tol * max(1.0, abs(gval)):
                            row = np.zeros(n_total)
                            for j in r.idxs:
                                row[j] += -float(grad[j])
                            row[aux] += 1.0
                            rows.append(row)
                            rhs.append(gval - slope_dot_x0)
                if not rows:
                    break
                _append(rows, rhs)
                _tl = (
                    None
                    if deadline is None
                    else max(deadline - time.perf_counter(), _SOLVE_DEADLINE_FLOOR_S)
                )
                new_res = milp.solve(time_limit=_tl, backend=self._backend)
                if new_res.status != "optimal" or new_res.objective is None:
                    break
                res = new_res
            return res
        except Exception:
            return res

    def _separate_rlt(self, milp, varmap, res, deadline):
        """Separate targeted RLT (constraint-factor x bound-factor) cuts.

        For each linear constraint ``a^T x <= b`` and variable bound factor
        ``x_j - l >= 0`` / ``u - x_j >= 0``, the product ``(b - a^T x)(factor)``
        is non-negative; linearized over the lifted columns it is a valid cut.
        Separates only violated ones (targeted) and re-solves. Each cut is valid
        for every feasible point, so the bound only tightens; on any failure the
        input ``res`` is returned unchanged. Off unless ``rlt_cuts=True``.
        """
        if not self._rlt_cuts:
            return res
        if res is None or res.status != "optimal" or res.x is None:
            return res
        try:
            import scipy.sparse as sp

            from discopt._jax.milp_relaxation import _linear_constraint_forms
            from discopt._jax.rlt_cuts import rlt_constraint_bound_cut

            forms = _linear_constraint_forms(self._model, self._n_orig)
            if not forms:
                return res
            n_total = len(milp._c)
            bounds = milp._bounds  # original-variable bounds live in cols 0..n_orig-1
            max_cuts_per_round = 128

            def _append(rows: list[np.ndarray], rhs: list[float]) -> None:
                R = np.asarray(rows, dtype=np.float64)
                bb = np.asarray(rhs, dtype=np.float64)
                if milp._A_ub is None:
                    milp._A_ub, milp._b_ub = R, bb
                elif sp.issparse(milp._A_ub):
                    milp._A_ub = sp.vstack([milp._A_ub, sp.csr_matrix(R)], format="csr")
                    milp._b_ub = np.concatenate([milp._b_ub, bb])
                else:
                    milp._A_ub = np.vstack([np.asarray(milp._A_ub), R])
                    milp._b_ub = np.concatenate([milp._b_ub, bb])

            for _round in range(6):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                x = np.asarray(res.x, dtype=np.float64)
                rows: list[np.ndarray] = []
                rhs: list[float] = []
                for coeff, const in forms:
                    # coeff . x + const <= 0  ==>  a^T x <= b with a=coeff, b=-const.
                    a = {i: float(coeff[i]) for i in range(self._n_orig) if coeff[i] != 0.0}
                    if not a:
                        continue
                    b = -float(const)
                    for j in range(self._n_orig):
                        if j >= len(bounds):
                            break
                        lo, hi = bounds[j]
                        for lower, bnd in ((True, lo), (False, hi)):
                            if not np.isfinite(bnd):
                                continue
                            cut = rlt_constraint_bound_cut(
                                a, b, j, float(bnd), lower, varmap, x, n_total
                            )
                            if cut is not None:
                                rows.append(cut.coeffs)
                                rhs.append(cut.rhs)
                                if len(rows) >= max_cuts_per_round:
                                    break
                        if len(rows) >= max_cuts_per_round:
                            break
                    if len(rows) >= max_cuts_per_round:
                        break
                if not rows:
                    break
                # cut is ``coeffs . z >= rhs`` -> ``(-coeffs) . z <= -rhs``.
                _append([-r for r in rows], [-v for v in rhs])
                _tl = (
                    None
                    if deadline is None
                    else max(deadline - time.perf_counter(), _SOLVE_DEADLINE_FLOOR_S)
                )
                new_res = milp.solve(time_limit=_tl, backend=self._backend)
                if new_res.status != "optimal" or new_res.objective is None:
                    break
                res = new_res
            return res
        except Exception:
            return res

    def _separate_psd(self, milp, varmap, res, deadline, max_rounds: int = 8):
        """Separate moment (PSD) cuts on the lifted relaxation at the LP point.

        Enforces ``M = [[1, x^T], [x, X]] >= 0`` over cliques of variables whose
        pairwise products and squares are all lifted. Each separated cut
        ``v^T M v >= 0`` is valid for every feasible point (``X = x x^T`` makes
        ``M`` rank-1 PSD), so the bound only tightens; on any failure the input
        ``res`` is returned unchanged. Off unless ``psd_cuts=True``.

        The spectral cutting plane converges to the Shor SDP bound only with many
        rounds (each adds a few eigenvector cuts), so ``max_rounds`` is large for a
        one-shot *root* separation (P2): on nvs17 ~150 rounds reaches -1221 (Shor
        is -1104.7) in ~1.4 s, vs -2453 at the old 8-round cap. Per-node callers
        keep the small default and instead inherit the root pool.
        """
        if not self._psd_cuts:
            return res
        if res is None or res.status != "optimal" or res.x is None:
            return res
        try:
            import scipy.sparse as sp

            from discopt._jax.psd_cuts import separate_psd_cuts_on_relaxation

            n_total = len(milp._c)

            def _append(rows: list[np.ndarray], rhs: list[float]) -> None:
                R = np.asarray(rows, dtype=np.float64)
                b = np.asarray(rhs, dtype=np.float64)
                if milp._A_ub is None:
                    milp._A_ub, milp._b_ub = R, b
                elif sp.issparse(milp._A_ub):
                    milp._A_ub = sp.vstack([milp._A_ub, sp.csr_matrix(R)], format="csr")
                    milp._b_ub = np.concatenate([milp._b_ub, b])
                else:
                    milp._A_ub = np.vstack([np.asarray(milp._A_ub), R])
                    milp._b_ub = np.concatenate([milp._b_ub, b])

            # Cost-aware gate (THRU-2a, DISCOPT_PSD_COST_GATE, default OFF). PSD
            # separation dominates the QCQP node wall (~60% on nvs17/19/24) while
            # the certified bound is set by McCormick+RLT and recovered by branching
            # when PSD is absent, so unbudgeted PSD starves the tree search. When on,
            # cap this node's PSD wall to ``budget × base_solve_wall`` and abandon on
            # per-round diminishing returns. SOUND: dropping cuts can only loosen the
            # relaxation — never cut a feasible point or cross the optimum. Keying is
            # purely on observed per-node cost/bound-delta (§0.2, general). The gate
            # only ever *shortens* the loop, so the default (gate-off) path below is
            # bit-identical to the pre-THRU-2a code.
            _tun = _tuning()
            _gate = _tun.psd_cost_gate
            _gate_budget = _tun.psd_cost_gate_budget
            _gate_tau = _tun.psd_cost_gate_tau
            _psd_t0 = time.perf_counter()
            _base_solve_wall: Optional[float] = None
            for _round in range(max_rounds):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                if (
                    _gate
                    and _base_solve_wall is not None
                    and (time.perf_counter() - _psd_t0) > _gate_budget * _base_solve_wall
                ):
                    # Per-node PSD wall budget spent — stop feeding the search.
                    break
                cuts = separate_psd_cuts_on_relaxation(
                    varmap, np.asarray(res.x, dtype=np.float64), n_total
                )
                if not cuts:
                    break
                _lb_before = res.objective if _gate else None
                # ``coeffs . z >= rhs``  ->  ``(-coeffs) . z <= -rhs`` for A_ub<=b_ub.
                _append([-c.coeffs for c in cuts], [-c.rhs for c in cuts])
                _tl = (
                    None
                    if deadline is None
                    else max(deadline - time.perf_counter(), _SOLVE_DEADLINE_FLOOR_S)
                )
                _solve_t0 = time.perf_counter()
                new_res = milp.solve(time_limit=_tl, backend=self._backend)
                if _gate and _base_solve_wall is None:
                    _base_solve_wall = max(time.perf_counter() - _solve_t0, 1e-4)
                if new_res.status != "optimal" or new_res.objective is None:
                    break
                res = new_res
                if _gate and _lb_before is not None:
                    _delta = new_res.objective - _lb_before
                    if _delta <= _gate_tau * (1.0 + abs(_lb_before)):
                        # Diminishing returns — abandon the remaining PSD rounds.
                        break
            return res
        except Exception:
            return res

    def _separate_edge_concave(self, milp, varmap, res, deadline):
        """Tighten edge-concave/edge-convex quadratic blocks by hull separation.

        Each block's vertex-hull supporting hyperplane gives a valid cut on the
        existing ``x_i^2`` / ``x_i x_j`` auxiliary columns (no lifting). Sound at
        any round; any failure returns the input ``res`` unchanged.
        """
        if not _tuning().edge_concave:
            return res
        if res is None or res.status != "optimal" or res.x is None:
            return res
        try:
            import scipy.sparse as sp

            from discopt._jax.edge_concave import (
                collect_edge_concave_quadratics,
                separate_edge_concave_quadratic,
            )

            if getattr(self, "_ec_blocks", None) is None:
                self._ec_blocks = collect_edge_concave_quadratics(self._model)
            blocks = self._ec_blocks
            if not blocks:
                return res

            bilinear = varmap.get("bilinear") or {}
            monomial = varmap.get("monomial") or {}
            n_total = len(milp._c)
            lb = np.array([b[0] for b in milp._bounds], dtype=np.float64)
            ub = np.array([b[1] for b in milp._bounds], dtype=np.float64)

            # Resolve each block's aux columns once; drop blocks missing any aux.
            specs = []
            for blk in blocks:
                cols_sq = {}
                cols_bl = {}
                ok = True
                for i in blk.sq:
                    col = monomial.get((i, 2))
                    if col is None:
                        ok = False
                        break
                    cols_sq[i] = col
                if ok:
                    for key in blk.bilin:
                        col = bilinear.get(key)
                        if col is None:
                            ok = False
                            break
                        cols_bl[key] = col
                if ok:
                    specs.append((blk, cols_sq, cols_bl))
            if not specs:
                return res

            def _append(rows, rhs):
                R = np.asarray(rows, dtype=np.float64)
                b = np.asarray(rhs, dtype=np.float64)
                if milp._A_ub is None:
                    milp._A_ub, milp._b_ub = R, b
                elif sp.issparse(milp._A_ub):
                    milp._A_ub = sp.vstack([milp._A_ub, sp.csr_matrix(R)], format="csr")
                    milp._b_ub = np.concatenate([milp._b_ub, b])
                else:
                    milp._A_ub = np.vstack([np.asarray(milp._A_ub), R])
                    milp._b_ub = np.concatenate([np.asarray(milp._b_ub), b])

            for _round in range(6):
                # Stop separating once the node's wall-clock budget is spent;
                # each round costs a full MILP re-solve.
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                x = np.asarray(res.x, dtype=np.float64)
                rows, rhs = [], []
                for blk, cols_sq, cols_bl in specs:
                    vi = list(blk.var_idxs)
                    blk_lb = lb[vi]
                    blk_ub = ub[vi]
                    x_star = x[vi]
                    q_star = blk.const
                    for i, coeff in blk.sq.items():
                        q_star += coeff * x[cols_sq[i]]
                    for key, coeff in blk.bilin.items():
                        q_star += coeff * x[cols_bl[key]]
                    for i, coeff in blk.lin.items():
                        q_star += coeff * x[i]
                    cut = separate_edge_concave_quadratic(blk, blk_lb, blk_ub, x_star, q_star)
                    if cut is None:
                        continue
                    A, B = cut
                    row = np.zeros(n_total)
                    if blk.sense == "under":
                        # q >= A.x + B  ->  A.x - q <= -B
                        for d, v in enumerate(vi):
                            row[v] += float(A[d])
                        for i, coeff in blk.sq.items():
                            row[cols_sq[i]] += -coeff
                        for key, coeff in blk.bilin.items():
                            row[cols_bl[key]] += -coeff
                        for i, coeff in blk.lin.items():
                            row[i] += -coeff
                        rows.append(row)
                        rhs.append(blk.const - float(B))
                    else:
                        # q <= A.x + B  ->  q - A.x <= B
                        for i, coeff in blk.sq.items():
                            row[cols_sq[i]] += coeff
                        for key, coeff in blk.bilin.items():
                            row[cols_bl[key]] += coeff
                        for i, coeff in blk.lin.items():
                            row[i] += coeff
                        for d, v in enumerate(vi):
                            row[v] += -float(A[d])
                        rows.append(row)
                        rhs.append(float(B) - blk.const)
                if not rows:
                    break
                _append(rows, rhs)
                _tl = (
                    None
                    if deadline is None
                    else max(deadline - time.perf_counter(), _SOLVE_DEADLINE_FLOOR_S)
                )
                new_res = milp.solve(time_limit=_tl, backend=self._backend)
                if new_res.status != "optimal" or new_res.objective is None:
                    break
                res = new_res
            return res
        except Exception:
            return res
