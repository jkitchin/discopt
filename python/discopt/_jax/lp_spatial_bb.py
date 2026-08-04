"""LP-node spatial branch-and-bound for integer-product MINLPs.

This is the engine SCIP uses on dense all-integer polynomial problems (the
``nvs17/19/24`` family) and that discopt's NLP-per-node spatial B&B cannot keep
up with. The diagnosis (``docs/dev/scip-gap-nvs-diagnosis.md``) showed discopt's
default path solves a *continuous NLP relaxation* at every node (~0.2 s) and
freezes its dual bound; SCIP solves a *pure LP* per node, branches on the integer
variables (which drives the products exact), and separates integer cuts.

This module implements the LP side of that: at each node it solves the **McCormick
LP relaxation** (``build_milp_relaxation`` — one LP, no NLP, globally valid lower
bound for a minimize), and branches either on a fractional integer variable or, when
the integer assignment is integral but a lifted product ``w_ij`` disagrees with
``x_i*x_j``, spatially on the worst-violated product's variable. A rounding
heuristic produces incumbents, verified *exactly* against a ground-truth point
evaluator (true objective + constraint feasibility) — never the relaxation bound.

**Scope.** Models with at least one integer/binary variable, in either objective
sense, with any mix of continuous variables (issue #860; the original step was
pure-integer MINIMIZE only). Three things make the wider scope sound:

* *Sense.* The engine works entirely in **minimize-equivalent** space, and both of
  its inputs already live there: ``uniform_relax`` negates a MAXIMIZE objective when
  it builds the LP cost, and ``NLPEvaluator`` negates it when it evaluates a point.
  So every LP bound is a valid LOWER bound on ``sgn * f``, every verified incumbent
  is ``sgn * f`` at a feasible point, and nothing inside the loop needs a sign at
  all. ``sgn`` is applied exactly once, to the reported objective/bound. The
  ``bound <= incumbent`` invariant therefore holds in min-equivalent space exactly as
  before, and becomes ``bound >= incumbent`` (a valid upper bound) once reported for
  a maximize.
* *Continuous variables.* Branching generalizes: integer variables are branched
  integrally (``floor``/``ceil``) and spatially on the integer midpoint; continuous
  variables are bisected at the box midpoint. Both only ever *shrink* a box, so
  every child's McCormick relaxation stays a valid outer approximation of its
  subtree. A node whose remaining candidates are all narrower than
  ``_MIN_BRANCH_WIDTH`` is not branchable and folds into ``unresolved_lb``.
* *Primal.* An incumbent is never read off the relaxation. Integer coordinates are
  rounded, continuous coordinates are completed by re-solving the node LP with the
  integers fixed, and the resulting point is verified *exactly* against a
  ground-truth evaluator (true objective + constraint feasibility). A completion
  that is not genuinely feasible is discarded, so ``bound <= incumbent`` cannot be
  broken by the continuous half.

Optimality is declared only when the valid dual bound (frontier + a floor for nodes
the engine cannot branch — see ``unresolved_lb``) closes the gap: a node whose
products are lifted outside this engine's ``info`` map (e.g. univariate-square
bilinear post-#636) is *not* treated as an exact leaf, so a loose relaxation can
never masquerade as a proof. Returns ``None`` (caller falls back to the default
path) whenever the model is out of scope or anything fails — it can never make a
solve unsound.

Pure-continuous models stay out of scope: the engine's convergence argument is that
branching the *integers* drives the products exact, and a model with no integer
variable gives it nothing the default NLP-per-node spatial path does not already do
better.
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import NamedTuple, Optional

import numpy as np

from discopt.modeling.core import Model, ObjectiveSense, VarType

logger = logging.getLogger(__name__)

_INT_TOL = 1e-6
_PROD_TOL = 1e-5
# Narrowest box a variable may still be branched on. Bisecting below this buys no
# relaxation tightness and only spawns nodes, so a node whose every candidate is
# this narrow is treated as unbranchable (its bound becomes an ``unresolved_lb``
# floor, never an optimality proof). Deliberately the SAME threshold as the
# collapsed-box exactness test below, so the two agree: a variable that cannot be
# branched any further is exactly one the leaf test counts as collapsed.
_MIN_BRANCH_WIDTH = 1e-9

# #862: how deep a single plunge may run before the search returns to best-first.
# A plunge trades bound quality for depth in order to reach an EXACT LEAF, which is
# the only place this engine's node loop can produce an incumbent (a fully-fixed box
# determines every nonlinear term). Best-first on tln6/nvs17 reaches depth 32 with
# ``fully_fixed = 0`` -- no exact leaf in 2000+ nodes -- so no incumbent ever arrives
# from the loop and the primal is left entirely to the rounding heuristics, which on
# this family are reading a relaxation that is ~250x loose (2562/2562 roundings
# infeasible; see docs/dev/lp-node-primal-quality.md).
_PLUNGE_MAX_DEPTH = 64

# Relative gap below which plunging STOPS. A plunge is a primal device: it buys depth
# (hence exact leaves, hence incumbents) at the cost of dual progress. Once the gap is
# nearly closed the remaining work is proving the bound, which is exactly what
# best-first does and what a plunge starves. Measured on the in-repo corpus: without
# this guard ``gear2`` -- whose best-first incumbent is 1.4e-06 against an optimum of
# 0.0, i.e. already converged -- lost ``status=optimal`` entirely, the one
# certification regression in the panel. With it, gear2 is untouched and every gain
# (tln6 +132.7% -> +7.8%, nvs19, nvs23, ex1265) is kept: those all sit at gaps of
# tens of percent when the plunge decision is made.
_PLUNGE_MIN_GAP = 1e-2


def _plunge_enabled(*, require_incremental: bool = False) -> bool:
    """#862: depth-first plunging in the node loop.

    **Default: ON for the #844 no-incumbent fallback, OFF for the general engine.**

    The scoping is the point, not a compromise. The fallback exists *only* to find a
    primal on a model the default path left with no incumbent — it is invoked with
    ``require_incremental=True`` and its output is an incumbent, never a certificate.
    There, trading dual progress for depth is the whole job. The general engine, by
    contrast, is asked to *prove* optimality, and plunging costs it: on ``gear2``
    best-first certifies in 657 nodes / 7.2 s while an unconditional plunge needs 6017
    nodes / 61.3 s for the same answer, which shows up as a certification regression at
    a 20 s panel budget. Enabling it everywhere would buy #862's incumbents at the
    price of a §5 cert-clean violation; enabling it where it belongs buys them for free.

    Measured on the in-repo corpus at 20 s (engine-direct, 58 in-scope), plunge ON vs
    OFF *everywhere*: 0 oracle crossings, incumbents 46 -> 49 (``ex1252a``, ``ex1263``,
    ``nvs24`` gained, none lost), quality better=7 worse=2, wall +7.0% — and exactly
    one certification regression, ``gear2``. Restricting the default to the fallback
    keeps the primal gains on the path #862 is about and leaves ``gear2``'s path
    bit-identical.

    ``DISCOPT_LP_SPATIAL_PLUNGE=1`` forces it on everywhere (the measurement above);
    ``=0`` forces it off everywhere, fallback included.

    Node ORDER cannot change the set of nodes explored, so plunging cannot make a
    bound unsound *by itself* -- but the in-loop global lower bound used to be read
    off the popped node, which is the frontier minimum only under best-first. That
    read is generalized to a true minimum over all live nodes (see ``glb`` below), so
    the two orders share one sound accounting rather than plunging relying on the
    best-first assumption. With this flag off the generalized form is bit-identical to
    the old one, which is what keeps the default path bound-neutral.

    Opt in with ``DISCOPT_LP_SPATIAL_PLUNGE=1``.
    """
    import os as _os

    _raw = _os.environ.get("DISCOPT_LP_SPATIAL_PLUNGE")
    if _raw is not None:
        return _raw not in ("0", "", "false", "False")
    return bool(require_incremental)


class LpSpatialResult(NamedTuple):
    status: str  # "optimal" | "feasible" | "infeasible" | "time_limit"
    objective: Optional[float]
    bound: Optional[float]
    gap: Optional[float]
    x: Optional[np.ndarray]
    node_count: int


def _is_in_scope(model: Model, *, mixed: bool = False) -> bool:
    """Model this engine can serve: an objective, at least one integer variable, and
    only ordinary algebraic rows. Either objective sense; any continuous mix.

    ``mixed=False`` restores the pre-#860 gate (pure-integer, MINIMIZE only) for
    callers rolling the widening out behind a flag.
    """
    if model._objective is None:
        return False
    if not model._variables:
        return False
    # At least one integer variable: the engine converges by branching the integers
    # (which drives the lifted products exact). With none, it degenerates to plain
    # spatial bisection, which the default NLP-per-node path already does.
    if not any(v.var_type in (VarType.INTEGER, VarType.BINARY) for v in model._variables):
        return False
    if not mixed:
        if model._objective.sense != ObjectiveSense.MINIMIZE:
            return False
        if not all(v.var_type in (VarType.INTEGER, VarType.BINARY) for v in model._variables):
            return False
    # Every row must be an ordinary algebraic constraint. A GDP/logical row
    # (``_LogicalConstraint``) carries no ``.body``, and the relaxation builder
    # walks ``.body`` unconditionally -- admitting one raises deep inside the
    # engine instead of being declined here.
    return all(hasattr(c, "body") for c in model._constraints)


def _integer_mask(model: Model) -> np.ndarray:
    """Boolean mask over flat columns: True where the variable is integer/binary."""
    flags: list[bool] = []
    for v in model._variables:
        flags.extend([v.var_type in (VarType.INTEGER, VarType.BINARY)] * int(v.size))
    return np.array(flags, dtype=bool)


def _relax_bound(model, terms, lb, ub, deadline=None):
    """McCormick LP over [lb,ub]; return (bound, full_x, info) or None.

    ``deadline`` (absolute ``time.perf_counter()``) bounds the two uninterruptible
    halves of a cold node: the DAG re-walk (``build_deadline``, which stops adding
    constraint rows and keeps the prefix — a weaker but still valid outer
    relaxation) and the LP solve itself. Without it a single cold node can outlive
    the whole engine budget, which the widened scope (#860) makes reachable: the
    mixed class is served largely by this path, on models an order of magnitude
    bigger than the pure-integer instances the engine started on."""
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation

    _remaining = None
    if deadline is not None:
        _remaining = deadline - time.perf_counter()
        if _remaining <= 0.0:
            return None
    try:
        relax, info = build_milp_relaxation(
            model,
            terms,
            DiscretizationState(),
            bound_override=(lb, ub),
            build_deadline=deadline,
        )
        if not relax._objective_bound_valid:
            return None
        # Solve the LP RELAXATION, not the MILP. ``build_milp_relaxation`` hands back
        # a model that still carries integrality (30 integer columns on ball_mk2_30,
        # 48 on tln6), so ``solve()`` was running a full branch-and-bound at every
        # node of *this* engine's own branch-and-bound -- duplicating the integer
        # branching the engine exists to perform. Measured: 32.85 s -> 0.00 s on
        # ball_mk2_30 (21,220x) and 4.24 s -> 0.00 s on tln6 (2,340x).
        #
        # Sound: dropping integrality only enlarges the feasible set, so the LP
        # optimum is <= the MILP optimum and remains a valid lower bound for a
        # minimize (verified on both instances: -29.88 <= -24.90 and 3.22 <= 4.60).
        # The bound is weaker per node, which is precisely the trade this engine is
        # built to make -- it recovers the tightness by branching on the integers
        # itself, which is what the module docstring means by "one LP, no NLP" per
        # node. The incremental path already solves a pure LP; this aligns the cold
        # path with it.
        relax._integrality = None
        if _remaining is not None:
            _remaining = deadline - time.perf_counter()
            if _remaining <= 0.0:
                return None
        res = relax.solve(time_limit=_remaining)
    except Exception:
        return None
    if res is None or res.bound is None or res.x is None:
        return None
    return float(res.bound), np.asarray(res.x, dtype=float), info


def _spatially_branchable(k, widths, is_int) -> bool:
    """Can variable ``k`` still be split by a spatial (domain-bisection) branch?

    An integer needs at least two values left (``width >= 1``); a continuous variable
    needs a finite box wider than ``_MIN_BRANCH_WIDTH``. An infinite width is never
    branchable — there is no midpoint to bisect at, and the McCormick envelope over
    such a box is vacuous anyway."""
    w = widths[k]
    if not np.isfinite(w):
        return False
    return bool(w >= 1.0) if is_int[k] else bool(w > _MIN_BRANCH_WIDTH)


def _worst_product_var(x, info, widths, is_int):
    """Branchable variable in the most-violated product (weighted by box width),
    or None if every lifted product matches its defining product.

    The weight ``viol * (width + 1)`` prefers a violation sitting on a wide box (more
    envelope to gain by splitting). For a continuous factor the width can be < 1, so
    the ``+1`` keeps the score positive and monotone in the violation — the same
    ordering the pure-integer version had, extended rather than replaced."""
    best, best_var = _PROD_TOL, None
    for (i, j), col in info.get("bilinear", {}).items():
        viol = abs(x[col] - x[i] * x[j])
        for k in (i, j):
            if _spatially_branchable(k, widths, is_int) and viol * (widths[k] + 1) > best:
                best, best_var = viol * (widths[k] + 1), k
    for (i, _p), col in info.get("monomial", {}).items():
        viol = abs(x[col] - x[i] * x[i])
        if _spatially_branchable(i, widths, is_int) and viol * (widths[i] + 1) > best:
            best, best_var = viol * (widths[i] + 1), i
    return best_var


def _set(a, i, v):
    b = a.copy()
    b[i] = v
    return b


def _separate_node_cuts(A, b, bounds, x, ncol, c, max_cuts=12):
    """Separate integer cuts from the assembled node LP at solution ``x``: GMI from
    the optimal basis (via crossover) plus complemented-MIR. Every structural AND
    product-aux column is marked integer — ``w_ij = x_i*x_j`` is integer-valued when
    the factors are, so the fractional envelope values of ``w`` become cut targets
    (the key to separating the McCormick optimum at all). Each returned cut
    ``coeffs·x <= rhs`` is a valid MIR/GMI inequality of the node relaxation, hence
    valid for every integer-feasible point in the node's box (and its subtree)."""
    cuts: list = []
    try:
        from discopt._jax.cmir_cuts import separate_cmir
        from discopt._jax.crossover import crossover_to_vertex
        from discopt._jax.problem_classifier import LPData
        from discopt.solver import _separate_gomory_cuts
    except Exception:
        return cuts

    # This GMI/crossover cut separator is dense by construction (``np.hstack([A,
    # np.eye])``, ``LPData``, the crossover vertex solve), and its dense ``A_eq`` is
    # ``m x (ncol+m)`` regardless — it was never viable for a large lift. ``inc.assemble``
    # now returns a SPARSE ``A``, so densify once here to preserve the exact prior
    # contract (only this bounded per-node cut path densifies; the node LP solve stays
    # sparse).
    import scipy.sparse as sp

    if sp.issparse(A):
        A = A.toarray()

    lb = bounds[:, 0]
    ub = bounds[:, 1]
    is_int = np.ones(ncol, dtype=bool)  # original + product aux are integer-valued
    # GMI from the optimal basis (equality standard form with explicit slacks)
    try:
        m = A.shape[0]
        A_eq = np.hstack([A, np.eye(m)])
        b_eq = b.copy()
        cc = np.concatenate([np.asarray(c, dtype=np.float64)[:ncol], np.zeros(m)])
        xl = np.concatenate([lb, np.zeros(m)])
        xu = np.concatenate([ub, np.full(m, 1e20)])
        xrelax = np.concatenate([x, b - A @ x])
        xv = crossover_to_vertex(xrelax, A_eq, b_eq, cc, xl, xu)
        lp = LPData(cc, A_eq, b_eq, xl, xu, 0.0)
        gc = _separate_gomory_cuts(lp, xv, ncol, list(range(ncol)), max_cuts=max_cuts)
        if gc is not None:
            for i in range(len(gc[1])):  # GMI returns coeffs·x >= rhs -> negate to <=
                row = -np.asarray(gc[0][i])[:ncol]
                # GMI validity holds only up to machine precision (gomory.rs:31); the
                # raw crossover vertex the cut separates carries ~1e-12 float error, so
                # a cut whose boundary passes through a feasible integer point could
                # shave it. Relax the <= rhs outward by the same safe margin every
                # other GMI consumer uses (solver.py _augment_lpdata_with_gomory_cuts,
                # cmir_cuts.py) — C-10. Sound: it only ever moves the cut AWAY from the
                # feasible region, never removing a feasible point.
                margin = 1e-7 * (1.0 + float(np.abs(row).sum()))
                cuts.append((row, -float(gc[1][i]) + margin))
    except Exception as exc:  # cuts are optional: a missing cut is always safe
        logger.debug("lp_spatial GMI separation skipped: %s: %s", type(exc).__name__, exc)
    # complemented-MIR (multi-row aggregation)
    try:
        mc = separate_cmir(A, b, x, lb, ub, is_int, max_cuts=max_cuts)
        cuts.extend(mc)
    except Exception as exc:  # cuts are optional: a missing cut is always safe
        logger.debug("lp_spatial c-MIR separation skipped: %s: %s", type(exc).__name__, exc)
    # Native Marchand–Wolsey aggregation c-MIR (cert:P3). DEFAULT-OFF, gated by
    # DISCOPT_CMIR_AGGREGATION. Pairs <= rows with nonnegative weights to cancel a
    # column, then applies the native Rust complemented MIR to the aggregate —
    # valid by construction (nonnegative row combo + valid MIR; proven by the Rust
    # aggregation_validity_random_systems property test). Every column here is an
    # integer-valued (structural or product-aux) column, so the separator's
    # fractional-column fallback picks the cancel target. It only ADDS valid cuts.
    try:
        from discopt.solver import _cmir_aggregation_enabled

        if _cmir_aggregation_enabled():
            from discopt._rust import aggregation_mir_cuts_py

            res = aggregation_mir_cuts_py(
                np.ascontiguousarray(np.asarray(A, dtype=np.float64)),
                np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel()),
                np.ascontiguousarray(lb.astype(np.float64)),
                np.ascontiguousarray(ub.astype(np.float64)),
                np.ascontiguousarray(is_int),
                np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel()),
            )
            if res is not None:
                acoef, arhs = np.asarray(res[0]), np.asarray(res[1])
                for i in range(min(acoef.shape[0], max_cuts)):
                    cuts.append((acoef[i][:ncol], float(arhs[i])))
    except Exception as exc:  # cuts are optional: a missing cut is always safe
        logger.debug("lp_spatial aggregated cuts skipped: %s: %s", type(exc).__name__, exc)
    return cuts


def solve_lp_spatial_bb(
    model: Model,
    *,
    time_limit: float = 300.0,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 500_000,
    use_obbt: bool = True,
    root_cut_rounds: int = 0,
    require_incremental: bool = False,
    mixed: bool = False,
) -> Optional[LpSpatialResult]:
    """LP-node spatial branch-and-bound. Returns ``None`` if out of scope.

    ``require_incremental`` declines the solve (returns ``None``) when the incremental
    McCormick structure cannot be built. Set it when the caller has a *bounded* budget
    and wants a primal: without that structure the engine has no cuts, no feasibility
    pump, and rebuilds the whole relaxation per node. Measured on ball_mk2_30 at a
    21 s budget, back when its ``x_0**2`` monomial still declined for spanning a sign
    change, the cold path spent 61 s on the *root* LP alone — 0 nodes, no incumbent,
    2.91x over budget. Declining costs nothing there and cannot overrun.

    #861 has since admitted even powers on a straddling root, so ball_mk2_30 itself
    now passes this guard and runs the fast path — and, measured, still returns NO
    incumbent (objective ``None``, sound bound, 208k nodes, budget honoured). So for
    that instance the guard no longer buys the early decline it was added for: it
    spends the reserve instead of skipping it. That is a real trade — a sound dual
    bound where there was nothing, in exchange for a fallback that no longer exits in
    0.5 s — and it is a *primal* gap (the #844 family), not something this predicate
    can detect: "the structure builds" is only a proxy for "this path can produce a
    primal", and #861 widened the gap between the two. An ODD power on a straddling
    root still declines here, which is the case the guard now mainly serves.

    ``root_cut_rounds`` enables GMI + complemented-MIR separation at the root (cuts
    inherited by all nodes). Default 0 (off): with discopt's current Python-level
    separators the per-round crossover/GMI cost and the larger inherited LP at every
    node outweigh the modest tightening — measured net-negative on nvs17/19/24. The
    machinery is sound and kept opt-in for when a fast native separator exists.

    ``mixed`` (#860) admits mixed-integer and MAXIMIZE models; ``False`` is the
    pre-#860 pure-integer/MINIMIZE gate. **Default False**: the widening is a real
    capability but it is not net-positive on the default path (CLAUDE.md §5 bar 2 —
    see ``_lp_spatial_mixed_fallback_enabled``), so both production call sites pass
    ``mixed=_lp_spatial_mixed_fallback_enabled()`` rather than relying on this
    default. Defaulting to ``False`` means a *new* call site inherits the
    conservative gate instead of silently shipping the widening, which is the same
    reason ``row_scan_is_anytime`` defaults to ``False`` in the tightening rules."""
    if not _is_in_scope(model, mixed=mixed):
        return None

    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.term_classifier import classify_nonlinear_terms

    terms = classify_nonlinear_terms(model)
    lb0, ub0 = flat_variable_bounds(model)
    lb0 = lb0.astype(float, copy=True)
    ub0 = ub0.astype(float, copy=True)
    n = int(lb0.size)
    is_int = _integer_mask(model)
    if is_int.size != n:  # defensive: flat layouts must agree
        return None
    INT = [i for i in range(n) if is_int[i]]
    _has_cont = bool(np.any(~is_int))
    # Minimize-equivalent sign. Both the relaxation builder and ``NLPEvaluator``
    # already negate a MAXIMIZE objective, so every LP bound below is a valid LOWER
    # bound on ``sgn * f``, every verified incumbent is ``sgn * f``, and the whole
    # engine (heap order, incumbent comparison, fathoming, gap) runs unchanged in that
    # space. ``sgn`` appears in exactly ONE place: the reported objective/bound at the
    # exit. Applying it anywhere else would double-negate. (``_objective`` is not None
    # here — ``_is_in_scope`` above rejects a model without one.)
    _obj = model._objective
    sgn = -1.0 if (_obj is not None and _obj.sense == ObjectiveSense.MAXIMIZE) else 1.0

    t0 = time.perf_counter()

    # Ground-truth point evaluator for exact incumbent verification. An incumbent's
    # objective MUST be the true objective at a verified-feasible integer point,
    # never a McCormick relaxation value: post-uniform-relaxation (#636) a bilinear
    # product ``x_i*x_j`` is lifted via univariate squares, so it no longer appears
    # in this engine's ``info`` product map -- the "collapsed box is exact" argument
    # (and ``_worst_product_var``'s "all products tight" check) silently fail, and
    # trusting the relaxation bound as a primal produced *certified false optima*
    # (nvs17: reported optimal -1836.2 vs true -1100.4, at an infeasible point).
    # Verifying against the evaluator restores ``bound <= incumbent`` unconditionally.
    # If we cannot build a verifier we cannot safely accept any incumbent, so bail to
    # the sound default path (return None) rather than risk an unverified certificate.
    try:
        from discopt._tape_nlp_evaluator import make_evaluator
        from discopt.solver import _check_constraint_feasibility, _infer_constraint_bounds

        # #75: dispatcher, not a direct NLPEvaluator -- this incumbent verifier
        # runs on the spatial B&B path and imported JAX unconditionally.
        _ev = make_evaluator(model)
        _cl, _cu = _infer_constraint_bounds(model, _ev)
    except Exception:
        return None
    _FEAS_TOL = 1e-6

    def _pt_feasible(xr: np.ndarray) -> bool:
        return bool(_check_constraint_feasibility(_ev, xr, _cl, _cu, tol=_FEAS_TOL))

    # Root box with an infinite endpoint. Pre-#860 this was an immediate decline
    # ("unbounded integer box: out of scope"). It is the single biggest reason the
    # engine could not *accept* the mixed class: 31 of the 71 mixed instances in the
    # in-repo corpus carry at least one infinite bound (real MINLPs leave continuous
    # columns unbounded above), including the syn/rsyn maximize family this issue
    # names. It is not, however, a soundness boundary — see the two paths below.
    _inf_box = not (np.all(np.isfinite(lb0)) and np.all(np.isfinite(ub0)))

    # Root OBBT. Runs when the caller asked for it OR when the box is infinite: on an
    # unbounded box it is the difference between a bound and nothing at all. Measured
    # on the two in-repo maximize instances, whose relaxation has NO valid objective
    # bound over the raw box (``_objective_bound_valid=False``): OBBT finitizes every
    # infinite upper bound in ~0.2-0.4 s and the McCormick LP then returns a valid
    # bound (syn05m -1165.78, syn05hfsg -1335.50 minimize-equivalent).
    if use_obbt or _inf_box:
        try:
            from discopt._jax.obbt import obbt_tighten_root

            # Budget the root OBBT against the caller's deadline. Without this it
            # runs |vars| x 2 x rounds LPs at up to time_limit_per_lp each, entirely
            # outside time_limit -- on ball_mk2_30 (30 integers) that alone is up to
            # 150 s against a 30 s budget. Cap it at a third of the remaining time so
            # the node loop always gets the majority of the budget.
            _obbt_budget = max(0.0, time_limit - (time.perf_counter() - t0)) / 3.0
            r = obbt_tighten_root(
                model,
                lb0,
                ub0,
                rounds=5,
                deadline=time.perf_counter() + _obbt_budget,
                time_limit_per_lp=min(0.5, max(0.05, _obbt_budget / 10.0)),
            )
            if not r.infeasible:
                # Integrality rounding applies to INTEGER columns only. Flooring a
                # continuous variable's tightened lower bound would push it BELOW the
                # value OBBT proved (widening, merely wasteful) while ceiling its upper
                # bound would do the same — but rounding a continuous bound the wrong
                # way on a narrow box can also erase the tightening entirely. Keep the
                # exact continuous bounds and round only where integrality permits.
                _rlb = np.asarray(r.lb, dtype=float)
                _rub = np.asarray(r.ub, dtype=float)
                lb0 = np.maximum(lb0, np.where(is_int, np.floor(_rlb + 1e-9), _rlb))
                ub0 = np.minimum(ub0, np.where(is_int, np.ceil(_rub - 1e-9), _rub))
                _inf_box = not (np.all(np.isfinite(lb0)) and np.all(np.isfinite(ub0)))
        except Exception as exc:
            # Never let root tightening break the solve -- but never hide it either.
            # A silently-skipped capability is precisely the failure mode that cost
            # #844 several wrong conclusions (a swallowed TypeError made a whole
            # fallback an invisible no-op).
            logger.debug("lp_spatial root OBBT skipped: %s: %s", type(exc).__name__, exc)

    # Fast path: incremental McCormick LP (structure built once, box-dependent rows
    # patched per node, warm-started). Guarded by its own validation against
    # build_milp_relaxation; on any failure fall back to the trusted per-node
    # builder (correct, ~30x slower). This is what gives the throughput to close by
    # branching (the no-cut-SCIP regime).
    from discopt._jax.incremental_mccormick import IncrementalMcCormickLP

    # Budget the structure build against THIS engine's deadline, not whatever
    # ``model._solve_deadline`` a previous ``solve_model`` left behind. That stash is
    # written once per solve and never cleared, so when this engine runs as the #844
    # no-incumbent fallback -- i.e. *after* a primary solve that used its whole budget
    # -- the ambient deadline is already in the PAST and the incremental structure
    # declined to build at all (``ok=False``). The engine then silently degraded to
    # the trusted-but-~30x-slower per-node cold build, which also disables cuts and
    # the feasibility pump, and whose nodes are slow enough that a single one runs
    # past the top-of-loop deadline poll. Measured on tln5 at a 21 s budget: ok=False
    # gave 5 nodes in 43.8 s (2.08x, slowest node 42.4 s) where ok=True gives 12643
    # nodes in 21.0 s (1.00x, slowest node 0.04 s). Take the tighter of the two when
    # an enclosing deadline is still live, so the #654 guard is never weakened.
    _own_deadline = t0 + time_limit
    _ambient = getattr(model, "_solve_deadline", None)
    if _ambient is not None and float(_ambient) > t0:
        _own_deadline = min(_own_deadline, float(_ambient))

    # Pass the ROOT box this engine will actually branch in — post-FBBT/OBBT, not the
    # model's declared bounds. The structure's probe box and ``_validate``'s comparison
    # boxes are generated inside it (#861), so anchoring them here is what makes them
    # reachable; and for a model whose declared box is unbounded (ex1233: 28 infinite
    # bounds) or whose raw-box relaxation has no valid objective bound (st_e04), it is
    # the difference between a structure that can be judged at all and one that fails
    # at the probe build.
    _inc = IncrementalMcCormickLP(model, terms, deadline=_own_deadline, box=(lb0, ub0))
    # The incremental patch writes closed-form envelope coefficients straight from the
    # box endpoints, so an infinite bound on a column it patches would inject
    # ``inf``/``nan`` into the node LP with no row-level guard to catch it (the cold
    # builder discards such rows and merely loosens). Decline the fast path there and
    # let the cold build serve the model: sound either way, and it is what lets the
    # engine accept a partially infinite box at all. Branching only shrinks boxes, so
    # a root box that is patchable stays patchable at every node.
    if _inc.ok and not _inc.box_is_patchable(lb0, ub0):
        logger.debug(
            "lp_spatial: incremental structure declined on this box (an infinite "
            "endpoint on a patched product column); using the per-node cold build"
        )
        _inc.ok = False
    if require_incremental and not _inc.ok:
        logger.debug(
            "lp_spatial declined: no incremental McCormick structure and the caller "
            "requires it (the per-node cold build cannot serve a bounded budget)"
        )
        return None
    # ``relax`` returns ``(bound, x, basis, info)``. The product map travels WITH the
    # solution it describes: on the incremental path the lifted layout is fixed once
    # and ``info`` is a constant, but the cold path re-runs ``build_milp_relaxation``
    # at every node and the lift is box-dependent, so a node's column count can differ
    # from the root's. Reusing the root map against a node's ``x`` indexes past the
    # end of that node's solution vector (measured on st_e38: root aux column 18 vs a
    # 17-column node LP -> IndexError, which the caller swallowed as "engine failed"
    # and silently fell back). A product map is only meaningful for the exact LP it
    # came from.
    if _inc.ok:
        info = {"bilinear": _inc.bilinear, "monomial": _inc.monomial}

        def relax(lb, ub, basis):
            b_, x_, bas = _inc.solve(lb, ub, in_basis=basis)
            return b_, x_, bas, info
    else:
        _r0 = _relax_bound(model, terms, lb0, ub0, deadline=t0 + time_limit)
        if _r0 is None:
            return None
        info = _r0[2]

        def relax(lb, ub, basis):
            c = _relax_bound(model, terms, lb, ub, deadline=t0 + time_limit)
            return (c[0], c[1], None, c[2]) if c is not None else (None, None, None, None)

    # Branch-and-cut: separate integer cuts (GMI + complemented-MIR, product aux
    # vars marked integer) at each node and re-solve, tightening the node bound
    # before branching. Cuts derived over a node's box are valid for its whole
    # subtree, so children inherit them; their cumulative effect across the tree is
    # what converges the McCormick bound (the no-cut engine stalls). Only available
    # on the incremental path (needs the explicit row system).
    cut_enabled = _inc.ok
    _MAX_INHERITED_CUTS = 400

    def node_relax(lb, ub, basis, inherited, rounds):
        """Solve the node LP with inherited cuts, then run ``rounds`` of cut
        separation (add only bound-improving cuts), returning
        ``(bound, x, basis, cuts, verdict, info)``.

        ``info`` is the product map of the LP that produced ``x`` — see the note on
        ``relax`` above; it is carried through the frontier so the spatial-branching
        test always reads the map belonging to the solution in hand.

        ``verdict`` distinguishes the two very different reasons a node LP can come
        back with no bound, which the caller MUST NOT conflate:

        * ``"fathom"`` — the LP feasible set over this box is provably empty, proven
          by a verified Farkas dual ray. Since the McCormick polytope is a valid
          outer approximation, an empty relaxation means the subtree contains no
          feasible point: dropping it is rigorous.
        * ``"unresolved"`` — no certified verdict (numerical failure, time limit, or
          an ``infeasible`` claim without a Farkas proof). The subtree is **not**
          ruled out, so silently dropping it would remove live space from the search
          and a later heap exhaustion could certify optimality over it. The caller
          must fold such a node into ``unresolved_lb`` instead.
        """
        if not cut_enabled:
            b_, x_, bas, info_ = relax(lb, ub, basis)
            # Cold path: ``_relax_bound`` collapses every failure mode into ``None``
            # and cannot prove infeasibility, so the only sound reading is
            # "unresolved". This is conservative in the safe direction — it can cost
            # an optimality certificate, never create a false one.
            verdict = "optimal" if b_ is not None else "unresolved"
            return b_, x_, bas, (), verdict, info_
        cuts = list(inherited)
        A, b, bounds = _inc.assemble(lb, ub, cuts)
        _st, b_, x_, bas, _farkas = _inc.solve_assembled_full(A, b, bounds, in_basis=basis)
        if b_ is None:
            _verdict = "fathom" if (_st == "infeasible" and _farkas) else "unresolved"
            return None, None, None, tuple(cuts), _verdict, info
        for _r in range(rounds):
            if len(cuts) >= _MAX_INHERITED_CUTS:
                break
            new = _separate_node_cuts(A, b, bounds, x_, _inc.ncol, _inc.c)
            if not new:
                break
            cuts.extend(new)
            A, b, bounds = _inc.assemble(lb, ub, cuts)
            nb, nx, nbas = _inc.solve_assembled(A, b, bounds, in_basis=bas)
            if nb is None or nx is None:
                break
            improved = nb > b_ + 1e-7 * (1 + abs(b_))
            b_, x_, bas = nb, nx, nbas
            if not improved:
                break
        return b_, x_, bas, tuple(cuts), "optimal", info

    root_b, root_x, root_basis, root_cuts, _root_verdict, root_info = node_relax(
        lb0, ub0, None, (), root_cut_rounds
    )
    if root_b is None:
        return None

    inc_val = float("inf")
    inc_x: Optional[np.ndarray] = None
    # frontier entries:
    #   (bound, tiebreak, lb, ub, x, warm_basis, inherited_cuts, info)
    # ``info`` is the product map of the LP that produced this node's ``x`` — see
    # ``relax``; it must travel with the solution, not be read from the root.
    heap = [(root_b, 0, lb0, ub0, root_x, root_basis, root_cuts, root_info)]
    counter = 1
    nodes = 0

    # pseudocosts: average objective gain per unit of branched fractionality, for
    # up/down branches on each variable (Achterberg). Score = product of the two
    # estimated gains -> a reliable variable-selection rule that converges far
    # faster than most-fractional. Uninitialized entries use the running average.
    psi_d = np.zeros(n)
    psi_u = np.zeros(n)
    cnt_d = np.zeros(n, dtype=int)
    cnt_u = np.zeros(n, dtype=int)

    def _avg_psi(arr, cnt):
        m = cnt > 0
        return float(arr[m].mean()) if m.any() else 1.0

    def _branch_var(x, lb, ub):
        """Pseudocost-scored fractional integer variable, or None if all integral."""
        cand = [i for i in INT if abs(x[i] - round(x[i])) > _INT_TOL and ub[i] - lb[i] > 0.5]
        if not cand:
            return None
        ad, au = _avg_psi(psi_d, cnt_d), _avg_psi(psi_u, cnt_u)
        best_s, best_i = -1.0, cand[0]
        for i in cand:
            fd = x[i] - np.floor(x[i])
            fu = np.ceil(x[i]) - x[i]
            sd = (psi_d[i] if cnt_d[i] else ad) * fd
            su = (psi_u[i] if cnt_u[i] else au) * fu
            s = max(sd, 1e-6) * max(su, 1e-6)
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    def _update_pc(i, direction, parent_b, child_b, frac):
        if child_b is None or frac < 1e-6:
            return
        gain = max(0.0, child_b - parent_b) / frac
        if direction == "d":
            psi_d[i] = (psi_d[i] * cnt_d[i] + gain) / (cnt_d[i] + 1)
            cnt_d[i] += 1
        else:
            psi_u[i] = (psi_u[i] * cnt_u[i] + gain) / (cnt_u[i] + 1)
            cnt_u[i] += 1

    def _round_integers(xhat):
        """Integer coordinates rounded, continuous coordinates untouched, clipped to
        the root box. Rounding a CONTINUOUS coordinate would be wrong twice over: it
        is not required to be integral, and moving it off the LP value throws away the
        only continuous information the node has."""
        xr = np.asarray(xhat, dtype=float).copy()
        xr[is_int] = np.round(xr[is_int])
        return np.minimum(np.maximum(xr, lb0), ub0)

    def verify(xhat):
        """Exact minimize-equivalent objective at the candidate point, or None if it
        is not feasible.

        Ground truth only: evaluates the true objective and checks constraint
        feasibility with the point evaluator (never the McCormick relaxation bound,
        which is not exact once a product is lifted outside ``info`` -- see the
        verifier note above). Guarantees any accepted incumbent is a genuinely
        feasible point whose reported objective is its true objective, so the
        frontier's valid lower bound can never exceed it.

        The value is MINIMIZE-EQUIVALENT, the same space every LP bound here lives in:
        ``NLPEvaluator`` already negates a MAXIMIZE objective (``_negate``), exactly as
        the relaxation builder negates the LP cost, so the two agree without any
        further adjustment. ``sgn`` is applied once, at the exit."""
        xr = _round_integers(xhat)
        if not _pt_feasible(xr):
            return None
        try:
            return float(_ev.evaluate_objective(xr)), xr
        except Exception:
            return None

    def complete(lb_c, ub_c, xhat):
        """Mixed-integer primal: fix the integers at their rounded values, re-solve the
        node LP for the continuous coordinates, and verify the completed point.

        This is the mixed generalization of the pure-integer engine's collapsed-box
        primal. On a pure-integer model fixing the integers collapses the box to a
        point and this degenerates to :func:`verify`; with continuous variables
        present, the collapse leaves an LP whose optimum supplies them. That LP is
        still a *relaxation* of the continuous restriction, so its point is only a
        candidate -- which is exactly why the result goes through ``verify`` and is
        discarded unless the ground-truth evaluator accepts it."""
        xr = _round_integers(xhat)
        lo, hi = np.asarray(lb_c, float).copy(), np.asarray(ub_c, float).copy()
        # Only fix integers the node box still admits; a rounded value outside it
        # would make the child box empty and the LP trivially infeasible.
        fix = is_int & (xr >= lo - 1e-9) & (xr <= hi + 1e-9)
        lo[fix] = xr[fix]
        hi[fix] = xr[fix]
        if np.any(lo > hi + 1e-9):
            return None
        _b, xx, _bas, _inf = relax(lo, hi, None)
        if xx is None:
            return None
        xc = np.asarray(xx[:n], dtype=float).copy()
        xc[fix] = xr[fix]
        return verify(xc)

    def dive(lb_d, ub_d):
        """Fix-and-dive: repeatedly fix the most-fractional free integer to its
        rounded LP value and re-solve, until all are fixed (a feasible candidate via
        the collapsed box) or the LP turns infeasible. Cheap, found nvs17's primal."""
        lo, hi = lb_d.copy(), ub_d.copy()
        for _ in range(2 * n + 2):
            # Poll the deadline: this loop solves an LP per iteration (2n+2 of them)
            # and is invoked at the root AND at every node, so without a check here
            # the engine can blow past ``time_limit`` by an order of magnitude.
            if (time.perf_counter() - t0) >= time_limit:
                return None
            b_, xx, _bas, _inf = relax(lo, hi, None)
            if b_ is None:
                return None
            free = [(abs(xx[i] - round(xx[i])), i) for i in INT if hi[i] - lo[i] > 0.5]
            if not free:
                # Every integer is fixed. On a pure-integer model the box is now a
                # single point and ``xx`` IS that point; with continuous variables it
                # is the LP's continuous completion of the fixed integer assignment.
                # Either way the candidate is verified exactly, never trusted.
                return verify(xx[:n])
            _, bi = max(free)
            v = min(max(round(xx[bi]), lo[bi]), hi[bi])
            lo[bi] = hi[bi] = v
        return None

    def feasibility_pump(lb, ub, x_seed, max_iter=30):
        """Objective feasibility pump (Fischetti-Glover-Lodi): alternate between the
        relaxation and rounding, each step re-solving the McCormick LP with a linear
        objective that pushes it toward the current rounded point, until the rounded
        integers are feasible (verified by the collapsed box). Finds incumbents where
        one-shot rounding / diving fail (e.g. nvs19/24).

        On a mixed model the pump acts on the INTEGER coordinates only -- those are the
        ones that have to be rounded to something. The continuous coordinates carry no
        rounding distance to close, and pushing them toward a rounded value would fight
        the LP for no reason; they are left to the relaxation, and the candidate point
        goes through :func:`complete` so they are re-solved against the fixed integer
        assignment before verification."""
        if not _inc.ok:
            return None
        x = np.asarray(x_seed, dtype=float)
        xhat = np.minimum(np.maximum(np.round(x[:n]), lb), ub)
        xhat[~is_int] = np.minimum(np.maximum(x[:n][~is_int], lb[~is_int]), ub[~is_int])
        seen: set = set()
        for _ in range(max_iter):
            # Same deadline poll as ``dive``: one LP per iteration, run at the root
            # and at every node.
            if (time.perf_counter() - t0) >= time_limit:
                return None
            h = verify(xhat)
            if h is None and _has_cont:
                h = complete(lb, ub, xhat)
            if h is not None:
                return h
            key = tuple(np.asarray(xhat)[is_int].tolist())
            if key in seen:  # cycle -> perturb the most-fractional integer coordinates
                frac = np.where(is_int, np.abs(x[:n] - xhat), -np.inf)
                for j in np.argsort(-frac)[: max(1, len(INT) // 4)]:
                    if not is_int[j]:
                        continue
                    step = 1.0 if x[j] > xhat[j] else -1.0
                    xhat[j] = min(max(xhat[j] + step, lb[j]), ub[j])
            seen.add(key)
            c_fp = np.zeros(_inc.ncol)
            # minimize -> pull the integer coordinates toward xhat; leave the
            # continuous ones with zero cost so the LP places them freely.
            c_fp[:n] = np.where(is_int, np.where(x[:n] > xhat, 1.0, -1.0), 0.0)
            _b, x_, _bas = _inc.solve(lb, ub, c_override=c_fp)
            if x_ is None:
                return None
            x = x_
            xhat = np.minimum(np.maximum(np.round(x[:n]), lb), ub)
            xhat[~is_int] = np.minimum(np.maximum(x[:n][~is_int], lb[~is_int]), ub[~is_int])
        return None

    def consider(cand):
        nonlocal inc_val, inc_x
        if cand is not None and cand[0] < inc_val:
            inc_val, inc_x = cand[0], cand[1].copy()

    def child(lb, ub, parent_basis, parent_cuts, rounds, parent_bound, collect=None):
        """Solve a child node (inheriting parent cuts, separating ``rounds`` more);
        push if promising. Returns its bound (or None).

        ``parent_bound`` is a valid lower bound for this child (its box is contained
        in the parent's), and is what gets recorded if the child's own relaxation
        cannot be resolved -- see below.

        ``collect`` (#862) receives the surviving node instead of the global heap, so
        the caller can decide between plunging into it and returning it to the
        best-first frontier. It changes only WHERE a live node is parked, never
        whether it stays live: ``_route`` below places every collected node in exactly
        one of the two containers, and both are counted by ``glb``.
        """
        nonlocal counter, unresolved_lb
        if np.any(lb > ub + 1e-9):
            return None  # empty integer box: genuinely nothing here, safe to drop
        b_, x_, basis_, cuts_, verdict, info_ = node_relax(
            lb, ub, parent_basis, parent_cuts, rounds
        )
        if b_ is not None and b_ < inc_val - 1e-9:
            _node = (b_, counter, lb, ub, x_, basis_, cuts_, info_)
            counter += 1
            if collect is None:
                heapq.heappush(heap, _node)
            else:
                collect.append(_node)
        elif b_ is None and verdict != "fathom":
            # The child's relaxation gave no certified verdict, so its subtree is NOT
            # ruled out. Dropping it silently (the pre-#844 behaviour) removes live
            # space from the search, and a later heap exhaustion would then declare
            # "optimal" over a region the engine never examined -- a false optimality
            # certificate (CLAUDE.md §1). Record the parent's bound, which is a valid
            # lower bound over the child's box, so the global bound can never close
            # above unexamined space.
            unresolved_lb = min(unresolved_lb, parent_bound)
        return b_

    # seed an incumbent: root dive (cheap) then a root feasibility pump (catches
    # cases diving misses). Both are rate-limited below so they never dominate. On a
    # mixed model the one-shot continuous completion of the root LP point goes first:
    # it costs a single LP and, measured over the in-repo corpus, is on its own enough
    # on 13 of the 46 mixed instances whose root relaxation is bounded (#860 entry
    # experiment) -- before any diving, pumping or branching.
    if _has_cont:
        consider(complete(lb0, ub0, root_x[:n]))
    consider(dive(lb0, ub0))
    consider(feasibility_pump(lb0, ub0, root_x))

    # Valid global lower bound floor from nodes the engine popped but could NOT
    # branch or fathom exactly (see the unbranchable-node handling below). The true
    # global lower bound is min(best frontier bound, this floor); optimality may be
    # declared only when that closes the gap to the verified incumbent.
    unresolved_lb = float("inf")

    # #862 plunging: a LIFO of nodes to descend into depth-first. Empty (and every
    # branch below routes straight to ``heap``) unless the flag is on, so the default
    # path keeps pure best-first.
    plunge: list = []
    plunge_depth = 0
    _plunge_on = _plunge_enabled(require_incremental=require_incremental)

    def _route(kids, node_bound):
        """Send the most promising child down the plunge, the rest to the frontier.

        "Most promising" is the smallest bound, i.e. the child best-first would have
        chosen next anyway -- so a plunge is best-first *committed to for a while*
        rather than a different search. Every child lands in exactly one container, so
        the live-node set is identical to what pure best-first would hold; only the
        ORDER of exploration differs.

        Plunging stops once the gap is nearly closed (``_PLUNGE_MIN_GAP``): past that
        point the work left is proving the bound, and depth-first starves exactly the
        dual progress that does it. With no incumbent yet the gap is infinite, so the
        search plunges freely -- which is when it is most needed, since the node loop
        has no other way to reach an exact leaf.
        """
        if not kids:
            return
        kids.sort(key=lambda k: k[0])
        for k in kids[1:]:
            heapq.heappush(heap, k)
        _gap_now = (
            float("inf") if inc_x is None else abs(inc_val - node_bound) / (1.0 + abs(inc_val))
        )
        if plunge_depth < _PLUNGE_MAX_DEPTH and _gap_now > _PLUNGE_MIN_GAP:
            plunge.append(kids[0])
        else:
            heapq.heappush(heap, kids[0])

    status = "infeasible"
    while heap or plunge:
        if (time.perf_counter() - t0) >= time_limit or nodes >= max_nodes:
            status = "time_limit"
            break
        if plunge:
            bound, _, lb, ub, x, basis, ncuts, ninfo = plunge.pop()
            plunge_depth += 1
        else:
            bound, _, lb, ub, x, basis, ncuts, ninfo = heapq.heappop(heap)
            plunge_depth = 0
        nodes += 1
        # Global lower bound: the minimum over ALL live nodes -- this popped one, the
        # best-first frontier, and anything parked on the plunge stack -- capped by the
        # unresolved-node floor. Fathoming/gap tests use this, never the popped node's
        # bound alone, so an unresolved node below the incumbent keeps the gap open
        # instead of yielding a false optimality proof.
        #
        # The frontier/plunge terms are what make plunging sound. Under pure best-first
        # the popped node IS the frontier minimum, so ``min(bound, unresolved_lb)`` was
        # correct; a plunge pops a node that may be far from minimal, and reading the
        # bound off it would report a global lower bound HIGHER than the truth -- which
        # closes the gap early and certifies optimality over space still on the heap
        # (CLAUDE.md §1). With the flag off ``plunge`` is empty and ``heap[0][0] >=
        # bound`` by the heap invariant, so this expression is bit-identical to the old
        # one and the default path stays bound-neutral.
        glb = min(bound, unresolved_lb)
        if heap:
            glb = min(glb, heap[0][0])
        for _p in plunge:
            glb = min(glb, _p[0])
        if bound >= inc_val - 1e-9 * (1 + abs(inc_val)):
            continue
        if inc_x is not None and abs(inc_val - glb) <= gap_tolerance * (1 + abs(inc_val)):
            status = "optimal"
            break
        # primal: cheap one-shot rounding every node; the continuous completion (one
        # extra LP, mixed models only) and dive / pump rate-limited so they never
        # dominate the node throughput (the earlier every-node bug).
        consider(verify(x[:n]))
        if _has_cont and nodes % 16 == 0:
            consider(complete(lb, ub, x[:n]))
        if nodes % 64 == 0:
            consider(dive(lb, ub))
        if nodes % 512 == 0:
            consider(feasibility_pump(lb, ub, x))
        # Per-node separation (crossover + GMI + c-MIR) costs ~seconds/node and
        # crashes throughput; the cuts are too weak to pay for it. Cut only at the
        # root (the main locus in SCIP too) — every node inherits those globally
        # valid root cuts, tightening its LP at no per-node separation cost.
        _rounds = 0
        # branch: pseudocost-scored integer-fractional variable
        bi = _branch_var(x, lb, ub)
        if bi is not None:
            fd = x[bi] - np.floor(x[bi])
            fu = np.ceil(x[bi]) - x[bi]
            _kids: Optional[list] = [] if _plunge_on else None
            bd = child(lb, _set(ub, bi, np.floor(x[bi])), basis, ncuts, _rounds, bound, _kids)
            bu = child(_set(lb, bi, np.ceil(x[bi])), ub, basis, ncuts, _rounds, bound, _kids)
            _update_pc(bi, "d", bound, bd, fd)
            _update_pc(bi, "u", bound, bu, fu)
            if _kids is not None:
                _route(_kids, bound)
            continue
        # Integral assignment: spatial-bisect the worst-violated product variable.
        # ``_worst_product_var`` can only see products present in ``info``; once a
        # product is lifted elsewhere (univariate-square bilinear post-#636) it
        # returns None even though the relaxation is NOT tight, so "no branchable
        # product" is NOT a proof that ``bound`` is exact here. Record a verified
        # incumbent (exact objective, never the loose ``bound``); the node is a true,
        # fathomable leaf only when the box is fully fixed (a single point, where every
        # nonlinear term is determined). Otherwise it is unresolved: keep its valid
        # lower bound as a global-bound floor so optimality is never claimed over
        # branching the engine could not perform. The collapse test spans EVERY
        # variable, continuous ones included -- a mixed node with a live continuous
        # dimension is never an exact leaf, however tight its products look.
        bv = _worst_product_var(x, ninfo, ub - lb, is_int)
        if bv is None:
            consider(verify(x[:n]))
            if _has_cont:
                consider(complete(lb, ub, x[:n]))
            if not bool(np.all(ub[:n] - lb[:n] <= _MIN_BRANCH_WIDTH)):
                unresolved_lb = min(unresolved_lb, bound)
            continue
        if is_int[bv]:
            # Integer domain split: [lb, mid] and [mid+1, ub] partition the integers
            # in the box exactly, so no integer point is lost.
            mid = np.floor((lb[bv] + ub[bv]) / 2)
            _kids = [] if _plunge_on else None
            child(lb, _set(ub, bv, mid), basis, ncuts, _rounds, bound, _kids)
            child(_set(lb, bv, mid + 1.0), ub, basis, ncuts, _rounds, bound, _kids)
            if _kids is not None:
                _route(_kids, bound)
        else:
            # Continuous bisection: the children SHARE the midpoint, so their union is
            # the parent box and no feasible point can fall between them. (The integer
            # split above can use disjoint halves only because there is nothing between
            # ``mid`` and ``mid+1``.)
            mid = 0.5 * (lb[bv] + ub[bv])
            _kids = [] if _plunge_on else None
            child(lb, _set(ub, bv, mid), basis, ncuts, _rounds, bound, _kids)
            child(_set(lb, bv, mid), ub, basis, ncuts, _rounds, bound, _kids)
            if _kids is not None:
                _route(_kids, bound)
    else:
        # Heap exhausted. Optimal only if the unresolved-node floor does not sit below
        # the incumbent (else there is space the engine could not rule out -> feasible
        # with an honest gap, never a false optimality certificate).
        if inc_x is not None and (
            not np.isfinite(unresolved_lb)
            or abs(inc_val - unresolved_lb) <= gap_tolerance * (1 + abs(inc_val))
        ):
            status = "optimal"
        elif inc_x is not None:
            status = "feasible"
        elif np.isfinite(unresolved_lb):
            # Nodes the engine could neither branch nor find a feasible point in were
            # left unresolved: cannot certify infeasibility.
            status = "time_limit"
        else:
            status = "infeasible"

    # Every live node counts, on the frontier OR parked mid-plunge (#862).
    gbound = min([h[0] for h in heap] + [p[0] for p in plunge], default=float("inf"))
    gbound = min(gbound, unresolved_lb)
    if inc_x is not None:
        gbound = min(gbound, inc_val)
    if not np.isfinite(gbound):
        gbound = inc_val if inc_x is not None else None
    obj = inc_val if inc_x is not None else None
    gap = None
    if obj is not None and gbound is not None and np.isfinite(gbound):
        gap = abs(obj - gbound) / (1 + abs(obj))
    if status == "time_limit" and inc_x is None:
        obj = None
    # Back to the model's own objective sense. Everything above is
    # minimize-equivalent; for a MAXIMIZE model ``gbound`` is a valid lower bound on
    # ``-f``, hence ``-gbound`` is a valid UPPER bound on ``f``, and the
    # ``gbound <= inc_val`` invariant established above becomes
    # ``bound >= objective`` — the maximize form of ``bound`` never crossing the
    # incumbent. ``gap`` is a ratio of absolute differences and is sign-invariant.
    if obj is not None:
        obj = sgn * obj
    if gbound is not None and np.isfinite(gbound):
        gbound = sgn * gbound
    return LpSpatialResult(
        status=status,
        objective=obj,
        bound=(gbound if (gbound is not None and np.isfinite(gbound)) else None),
        gap=gap,
        x=inc_x,
        node_count=nodes,
    )
