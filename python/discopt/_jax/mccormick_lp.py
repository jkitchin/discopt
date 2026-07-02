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
    """Outcome of one LP-form McCormick relaxation solve."""

    status: str
    lower_bound: Optional[float] = None
    x: Optional[np.ndarray] = None  # first ``n_orig`` columns of the LP solution


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
                from discopt._jax.lp_spatial_bb import _is_in_scope as _inc_in_scope

                # Restrict to the lp_spatial engine's *validated-safe* domain
                # (pure-integer, minimize): there the incremental patch is the same
                # construction the opt-in LP-node engine ships and trusts. Outside
                # it (e.g. mixed-integer NN/tree embeddings), the dense patch can be
                # an unsound relaxation, so we keep those on the cold build.
                if _inc_in_scope(model):
                    _inc = IncrementalMcCormickLP(model, self._terms)
                    if _inc.ok:
                        self._inc = _inc
            except Exception:
                self._inc = None

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
    ) -> Optional["MccormickLPResult"]:
        """Incremental McCormick node solve: patch the cached structure + warm-start,
        instead of a cold ``build_milp_relaxation`` + equilibration. Returns a
        :class:`MccormickLPResult` on success or ``None`` to fall back to the cold
        build. Sound: the patched matrix is validated equal to the cold build at
        construction, so the pure-LP value is a valid lower bound (integrality is
        branched by the outer tree); the inherited root cut pool is appended only
        when its column layout matches exactly (a mismatch is skipped — sound)."""
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
                a_rows, b_rows = inherited_cuts
                if a_rows is not None and len(a_rows) > 0:
                    a_rows = np.asarray(a_rows, dtype=np.float64)
                    b_rows = np.asarray(b_rows, dtype=np.float64).ravel()
                    # Apply the root cut pool only when its column layout matches the
                    # incremental structure exactly; a mismatch would address the
                    # wrong columns (skipping is sound — fewer cuts only loosen).
                    if (
                        a_rows.ndim == 2
                        and a_rows.shape[1] == inc.ncol
                        and len(b_rows) == a_rows.shape[0]
                    ):
                        cut_rows = list(zip(a_rows, b_rows))
            A, b, bounds = inc.assemble(lb, ub, cut_rows=cut_rows)
            nrows = int(A.shape[0])
            in_basis = (
                self._inc_warm_basis
                if (self._inc_warm_basis is not None and self._inc_basis_nrows == nrows)
                else None
            )
            status, bound, x_full, basis, farkas_certified = inc.solve_assembled_full(
                A, b, bounds, in_basis=in_basis
            )
        except Exception:
            logger.debug("incremental McCormick node failed; cold fallback", exc_info=True)
            return None

        if status == "optimal" and bound is not None and np.isfinite(bound):
            self._inc_warm_basis = basis
            self._inc_basis_nrows = nrows
            x_orig = np.asarray(x_full, dtype=np.float64)[: self._n_orig]
            return MccormickLPResult(status="optimal", lower_bound=float(bound), x=x_orig)

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
            # The ray did not verify (an ill-conditioned candidate): fall back to
            # the equilibration re-verify, mirroring the cold path's
            # false-infeasible guard (trust a raw infeasible only after an exact,
            # feasible-set-preserving rescale recovers no feasible point).
            return self._reverify_incremental_infeasible(inc, A, b, bounds)

        # time_limit / unbounded / numerical error: no certified verdict — fall back
        # to the trusted cold build.
        return None

    def _reverify_incremental_infeasible(
        self, inc, A: np.ndarray, b: np.ndarray, bounds: np.ndarray
    ) -> Optional["MccormickLPResult"]:
        """Confirm an incremental ``infeasible`` verdict soundly, without a cold
        rebuild, when the simplex's Farkas ray did not already certify it. Mirrors
        the cold path's false-infeasible guard: a well-scaled LP's infeasible
        verdict is trusted directly; an ill-conditioned one (coefficient spread
        ``> _RELAX_FALSE_INFEAS_TRIGGER``) is re-solved once with exact
        geometric-mean equilibration (feasible-set-preserving) before being
        accepted. The equilibrated re-solve also yields a fresh Farkas ray, so a
        recovered infeasible is preferentially confirmed by that certificate.

        Returns ``MccormickLPResult(status="infeasible")`` to fathom, an
        ``"optimal"`` result if equilibration recovers a feasible point (a false
        infeasible), or ``None`` (cold fallback) if the re-verify itself fails."""
        import scipy.sparse as sp

        from discopt._jax.milp_relaxation import (
            _RELAX_FALSE_INFEAS_TRIGGER,
            equilibrate_relaxation_lp,
        )

        try:
            a_csr = sp.csr_matrix(A)
            nz = np.abs(a_csr.data)
            nz = nz[nz != 0.0]
            ill = bool(
                nz.size
                and np.isfinite(nz).all()
                and nz.max() / nz.min() > _RELAX_FALSE_INFEAS_TRIGGER
            )
        except Exception:
            ill = True  # be conservative: re-verify rather than trust
        if not ill:
            return MccormickLPResult(status="infeasible")

        try:
            bl = [(float(bounds[i, 0]), float(bounds[i, 1])) for i in range(bounds.shape[0])]
            c2, a2, b2, bd2, col_scale = equilibrate_relaxation_lp(inc.c, a_csr, b, bl, None)
            status, bound, x_s, _, _farkas = inc.solve_assembled_full(
                a2, b2, np.asarray(bd2, dtype=np.float64), in_basis=None, c_override=c2
            )
        except Exception:
            return None  # re-verify failed -> trusted cold rebuild
        if status == "infeasible":
            return MccormickLPResult(status="infeasible")
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
    ) -> MccormickLPResult:
        """Solve the McCormick LP relaxation restricted to the given bound box.

        Returns a :class:`MccormickLPResult`. ``lower_bound`` is a valid lower
        bound on the original problem within this box (for minimization).
        ``x`` is the LP solution projected to the original variable columns.
        On any solver failure, ``status != "optimal"`` and the LB is ``None``.

        Global cut pool (P1, see ``docs/design/global-cut-pool.md``):

        * ``inherited_cuts``: ``(A_rows, b_rows)`` of globally-valid cut rows
          separated once at the root, appended to this node's relaxation before
          solving. Applied only when the lifted column layout matches the pool's
          (a tightened child box can pin/fold columns); a mismatch skips them,
          which is sound (fewer cuts only loosens the bound).
        * ``separate``: when ``False`` skip the per-node cut-separation chain (use
          the inherited pool instead of re-deriving cuts every node).
        * ``out_cuts``: when a list is supplied, the rows the separation chain
          appended at THIS call are pushed onto it as ``(A_rows, b_rows)`` — used
          to capture the root pool once and replay it at every node.
        """
        # Phase-B fast path: incremental patch + warm-started solve, reusing the
        # structure built once at construction instead of a per-node cold rebuild +
        # equilibration. Sound (validated patch == cold build; a valid relaxation),
        # and it carries the inherited (root RLT/PSD) cut pool so it keeps the
        # default path's bound strength. Returns ``None`` for any out-of-scope node,
        # falling through to the trusted cold build below.
        _fast = self._try_incremental_node(node_lb, node_ub, inherited_cuts)
        if _fast is not None:
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

        # Root cut-pool inheritance (P1): append globally-valid cut rows separated
        # once at the root instead of re-separating them every node. Sound — each
        # pool row is a valid inequality at every feasible point regardless of the
        # node box — but only column-aligned when the lifted layout is unchanged
        # (a pinned child variable can fold columns), so gate on a matching
        # ``n_total`` and otherwise skip (fewer cuts only loosens the bound).
        if inherited_cuts is not None:
            _ia, _ib = inherited_cuts
            if _ia is not None and _sparse_cols(_ia) == n_total:
                _append_relax_rows(milp, _ia, _ib)
        _n_base_rows = 0 if milp._A_ub is None else _sparse_rows(milp._A_ub)

        res = milp.solve(time_limit=_remaining(), backend=self._backend)

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
            # (X = x x^T), so adding them only tightens the bound.
            _t = time.perf_counter()
            res = self._separate_psd(milp, varmap, res, _deadline, max_rounds=psd_max_rounds)
            _st["psd"] += time.perf_counter() - _t
            # Targeted RLT (constraint-factor x bound-factor) cuts.
            _t = time.perf_counter()
            res = self._separate_rlt(milp, varmap, res, _deadline)
            _st["rlt"] += time.perf_counter() - _t

        # Capture the rows the separation chain just appended, for the root cut
        # pool. Stated over this node's lifted column space (``n_total`` columns).
        if out_cuts is not None and milp._A_ub is not None:
            _A = _as_csr(milp._A_ub)
            if _A.shape[0] > _n_base_rows:
                out_cuts.append(
                    (_A[_n_base_rows:].copy(), np.asarray(milp._b_ub)[_n_base_rows:].copy())
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
            # Sound: ``milp.solve`` equilibration-verified it (and Farkas-certified
            # it when ``res.farkas_certified``).
            return MccormickLPResult(status="infeasible")
        if res.status != "optimal" or res.x is None:
            return MccormickLPResult(status=res.status)

        if self._backend == "auto":
            # HiGHS/POUNCE already returns a trustworthy optimum; keep the legacy
            # behaviour (no certificate is produced on that path).
            bound: Optional[float] = res.objective
        elif res.safe_bound is not None:
            # The common pure-LP warm-simplex path: rigorous safe bound.
            bound = res.safe_bound
        elif milp._integrality is not None:
            # Integer-aware node bound (non-default ``node_bound_mode="milp"``):
            # the engine's own B&B dual bound is the valid lower bound here.
            bound = res.bound
        elif self._nonlinear_cols and self._has_unbounded_nonlinear_col(milp):
            # A nonlinear-participating variable is still unbounded at this node, so
            # the McCormick/RLT envelope may be genuinely UNBOUNDED — and the fast
            # simplex can fabricate a finite "optimal" there (himmel16 with RLT
            # cuts). With no computable safe bound (free variable) we cannot certify
            # the vertex is not too high, so decline: report no bound and let the
            # driver branch, rather than trust a fabricated value that would fathom
            # the optimal region. (Pure-Rust replacement of the old HiGHS
            # unbounded-relaxation cross-check.)
            bound = None
        elif self._max_finite_magnitude(milp) <= _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE:
            # Pure-LP with a free variable but a bounded relaxation: trust the
            # internally-equilibrated, dual-feasibility-verified vertex objective
            # only when well-conditioned (mirrors the old guard's conditioning gate
            # — below it the vertex carries no meaningful drift).
            bound = res.objective
        else:
            # Free variable AND ill-conditioned beyond where the fast simplex is
            # reliable, with no computable safe bound: decline (branch) rather than
            # risk a too-high vertex value (pure-Rust replacement of the old
            # too-high-optimal HiGHS cross-check).
            bound = None

        if bound is None or not np.isfinite(bound):
            return MccormickLPResult(status=res.status)
        x_orig = np.asarray(res.x)[: self._n_orig].copy()
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
            for _round in range(8):
                # Each round costs a full re-solve; stop once the budget is spent.
                if deadline is not None and time.perf_counter() >= deadline:
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

            for _round in range(max_rounds):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                cuts = separate_psd_cuts_on_relaxation(
                    varmap, np.asarray(res.x, dtype=np.float64), n_total
                )
                if not cuts:
                    break
                # ``coeffs . z >= rhs``  ->  ``(-coeffs) . z <= -rhs`` for A_ub<=b_ub.
                _append([-c.coeffs for c in cuts], [-c.rhs for c in cuts])
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
