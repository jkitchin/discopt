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

**Scope (this step).** Pure-integer models with a MINIMIZE objective. The frontier
McCormick bound is always a valid lower bound; every incumbent is a genuinely
feasible point whose reported objective is its true objective, so ``bound <=
incumbent`` holds unconditionally. Optimality is declared only when the valid dual
bound (frontier + a floor for nodes the engine cannot branch — see
``unresolved_lb``) closes the gap: a node whose products are lifted outside this
engine's ``info`` map (e.g. univariate-square bilinear post-#636) is *not* treated
as an exact leaf, so a loose relaxation can never masquerade as a proof. Cuts
(Gomory/MIR) and warm-started incremental LPs are added in later steps;
``build_milp_relaxation`` is currently rebuilt per node. Returns ``None`` (caller
falls back to the default path) whenever the model is out of scope or anything
fails — it can never make a solve unsound.
"""

from __future__ import annotations

import heapq
import time
from typing import NamedTuple, Optional

import numpy as np

from discopt.modeling.core import Model, ObjectiveSense, VarType

_INT_TOL = 1e-6
_PROD_TOL = 1e-5


class LpSpatialResult(NamedTuple):
    status: str  # "optimal" | "feasible" | "infeasible" | "time_limit"
    objective: Optional[float]
    bound: Optional[float]
    gap: Optional[float]
    x: Optional[np.ndarray]
    node_count: int


def _is_in_scope(model: Model) -> bool:
    """Pure-integer model with a minimize objective and at least one var."""
    if model._objective is None or model._objective.sense != ObjectiveSense.MINIMIZE:
        return False
    if not model._variables:
        return False
    return all(v.var_type in (VarType.INTEGER, VarType.BINARY) for v in model._variables)


def _relax_bound(model, terms, lb, ub):
    """McCormick LP over [lb,ub]; return (bound, full_x, info) or None."""
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation

    try:
        relax, info = build_milp_relaxation(
            model, terms, DiscretizationState(), bound_override=(lb, ub)
        )
        if not relax._objective_bound_valid:
            return None
        res = relax.solve()
    except Exception:
        return None
    if res is None or res.bound is None or res.x is None:
        return None
    return float(res.bound), np.asarray(res.x, dtype=float), info


def _worst_product_var(x, info, widths):
    """Branchable variable in the most-violated product (weighted by box width),
    or None if every lifted product matches its defining product."""
    best, best_var = _PROD_TOL, None
    for (i, j), col in info.get("bilinear", {}).items():
        viol = abs(x[col] - x[i] * x[j])
        for k in (i, j):
            if widths[k] >= 1 and viol * (widths[k] + 1) > best:
                best, best_var = viol * (widths[k] + 1), k
    for (i, _p), col in info.get("monomial", {}).items():
        viol = abs(x[col] - x[i] * x[i])
        if widths[i] >= 1 and viol * (widths[i] + 1) > best:
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
    except Exception:
        pass
    # complemented-MIR (multi-row aggregation)
    try:
        mc = separate_cmir(A, b, x, lb, ub, is_int, max_cuts=max_cuts)
        cuts.extend(mc)
    except Exception:
        pass
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
    except Exception:
        pass
    return cuts


def solve_lp_spatial_bb(
    model: Model,
    *,
    time_limit: float = 300.0,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 500_000,
    use_obbt: bool = True,
    root_cut_rounds: int = 0,
) -> Optional[LpSpatialResult]:
    """LP-node spatial branch-and-bound. Returns ``None`` if out of scope.

    ``root_cut_rounds`` enables GMI + complemented-MIR separation at the root (cuts
    inherited by all nodes). Default 0 (off): with discopt's current Python-level
    separators the per-round crossover/GMI cost and the larger inherited LP at every
    node outweigh the modest tightening — measured net-negative on nvs17/19/24. The
    machinery is sound and kept opt-in for when a fast native separator exists."""
    if not _is_in_scope(model):
        return None

    from discopt._jax.term_classifier import classify_nonlinear_terms

    terms = classify_nonlinear_terms(model)
    n = len(model._variables)
    INT = list(range(n))  # scope: all variables integer
    lb0 = np.array([float(v.lb) for v in model._variables])
    ub0 = np.array([float(v.ub) for v in model._variables])
    if not (np.all(np.isfinite(lb0)) and np.all(np.isfinite(ub0))):
        return None  # unbounded integer box: out of scope for this step

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
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.solver import _check_constraint_feasibility, _infer_constraint_bounds

        _ev = NLPEvaluator(model)
        _cl, _cu = _infer_constraint_bounds(model, _ev)
    except Exception:
        return None
    _FEAS_TOL = 1e-6

    def _pt_feasible(xr: np.ndarray) -> bool:
        return bool(_check_constraint_feasibility(_ev, xr, _cl, _cu, tol=_FEAS_TOL))

    if use_obbt:
        try:
            from discopt._jax.obbt import obbt_tighten_root

            r = obbt_tighten_root(model, lb0, ub0, rounds=5, time_limit_per_lp=0.5)
            if not r.infeasible:
                lb0 = np.maximum(lb0, np.floor(np.asarray(r.lb) + 1e-9))
                ub0 = np.minimum(ub0, np.ceil(np.asarray(r.ub) - 1e-9))
        except Exception:
            pass

    # Fast path: incremental McCormick LP (structure built once, box-dependent rows
    # patched per node, warm-started). Guarded by its own validation against
    # build_milp_relaxation; on any failure fall back to the trusted per-node
    # builder (correct, ~30x slower). This is what gives the throughput to close by
    # branching (the no-cut-SCIP regime).
    from discopt._jax.incremental_mccormick import IncrementalMcCormickLP

    _inc = IncrementalMcCormickLP(model, terms)
    if _inc.ok:
        info = {"bilinear": _inc.bilinear, "monomial": _inc.monomial}

        def relax(lb, ub, basis):
            return _inc.solve(lb, ub, in_basis=basis)
    else:
        _r0 = _relax_bound(model, terms, lb0, ub0)
        if _r0 is None:
            return None
        info = _r0[2]

        def relax(lb, ub, basis):
            c = _relax_bound(model, terms, lb, ub)
            return (c[0], c[1], None) if c is not None else (None, None, None)

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
        (bound, x, basis, cuts)."""
        if not cut_enabled:
            b_, x_, bas = relax(lb, ub, basis)
            return b_, x_, bas, ()
        cuts = list(inherited)
        A, b, bounds = _inc.assemble(lb, ub, cuts)
        b_, x_, bas = _inc.solve_assembled(A, b, bounds, in_basis=basis)
        if b_ is None:
            return None, None, None, tuple(cuts)
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
        return b_, x_, bas, tuple(cuts)

    root_b, root_x, root_basis, root_cuts = node_relax(lb0, ub0, None, (), root_cut_rounds)
    if root_b is None:
        return None

    inc_val = float("inf")
    inc_x: Optional[np.ndarray] = None
    # frontier entries: (bound, tiebreak, lb, ub, x, warm_basis, inherited_cuts)
    heap = [(root_b, 0, lb0, ub0, root_x, root_basis, root_cuts)]
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

    def verify(xhat):
        """Exact objective at the rounded integer point, or None if infeasible.

        Ground truth only: evaluates the true objective and checks constraint
        feasibility with the point evaluator (never the McCormick relaxation bound,
        which is not exact once a product is lifted outside ``info`` -- see the
        verifier note above). Guarantees any accepted incumbent is a genuinely
        feasible point whose reported objective is its true objective, so the
        frontier's valid lower bound can never exceed it."""
        xr = np.minimum(np.maximum(np.round(xhat), lb0), ub0)
        if not _pt_feasible(xr):
            return None
        try:
            return float(_ev.evaluate_objective(xr)), xr
        except Exception:
            return None

    def dive(lb_d, ub_d):
        """Fix-and-dive: repeatedly fix the most-fractional free integer to its
        rounded LP value and re-solve, until all are fixed (a feasible candidate via
        the collapsed box) or the LP turns infeasible. Cheap, found nvs17's primal."""
        lo, hi = lb_d.copy(), ub_d.copy()
        for _ in range(2 * n + 2):
            b_, xx, _bas = relax(lo, hi, None)
            if b_ is None:
                return None
            free = [(abs(xx[i] - round(xx[i])), i) for i in INT if hi[i] - lo[i] > 0.5]
            if not free:
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
        one-shot rounding / diving fail (e.g. nvs19/24)."""
        if not _inc.ok:
            return None
        x = np.asarray(x_seed, dtype=float)
        xhat = np.minimum(np.maximum(np.round(x[:n]), lb), ub)
        seen: set = set()
        for _ in range(max_iter):
            h = verify(xhat)
            if h is not None:
                return h
            key = tuple(xhat.tolist())
            if key in seen:  # cycle -> perturb the most-fractional coordinates
                order = np.argsort(-np.abs(x[:n] - xhat))
                for j in order[: max(1, n // 4)]:
                    step = 1.0 if x[j] > xhat[j] else -1.0
                    xhat[j] = min(max(xhat[j] + step, lb[j]), ub[j])
            seen.add(key)
            c_fp = np.zeros(_inc.ncol)
            c_fp[:n] = np.where(x[:n] > xhat, 1.0, -1.0)  # minimize -> pull x to xhat
            _b, x_, _bas = _inc.solve(lb, ub, c_override=c_fp)
            if x_ is None:
                return None
            x = x_
            xhat = np.minimum(np.maximum(np.round(x[:n]), lb), ub)
        return None

    def consider(cand):
        nonlocal inc_val, inc_x
        if cand is not None and cand[0] < inc_val:
            inc_val, inc_x = cand[0], cand[1].copy()

    def child(lb, ub, parent_basis, parent_cuts, rounds):
        """Solve a child node (inheriting parent cuts, separating ``rounds`` more);
        push if promising. Returns its bound (or None)."""
        nonlocal counter
        if np.any(lb > ub + 1e-9):
            return None
        b_, x_, basis_, cuts_ = node_relax(lb, ub, parent_basis, parent_cuts, rounds)
        if b_ is not None and b_ < inc_val - 1e-9:
            heapq.heappush(heap, (b_, counter, lb, ub, x_, basis_, cuts_))
            counter += 1
        return b_

    # seed an incumbent: root dive (cheap) then a root feasibility pump (catches
    # cases diving misses). Both are rate-limited below so they never dominate.
    consider(dive(lb0, ub0))
    consider(feasibility_pump(lb0, ub0, root_x))

    # Valid global lower bound floor from nodes the engine popped but could NOT
    # branch or fathom exactly (see the unbranchable-node handling below). The true
    # global lower bound is min(best frontier bound, this floor); optimality may be
    # declared only when that closes the gap to the verified incumbent.
    unresolved_lb = float("inf")

    status = "infeasible"
    while heap:
        if (time.perf_counter() - t0) >= time_limit or nodes >= max_nodes:
            status = "time_limit"
            break
        bound, _, lb, ub, x, basis, ncuts = heapq.heappop(heap)
        nodes += 1
        # Global lower bound = smallest frontier bound (this popped node, best-first)
        # capped by any unresolved-node floor. Fathoming/gap tests use this, never the
        # popped node's bound alone, so an unresolved node below the incumbent keeps
        # the gap open instead of yielding a false optimality proof.
        glb = min(bound, unresolved_lb)
        if bound >= inc_val - 1e-9 * (1 + abs(inc_val)):
            continue
        if inc_x is not None and abs(inc_val - glb) <= gap_tolerance * (1 + abs(inc_val)):
            status = "optimal"
            break
        # primal: cheap one-shot rounding every node; dive / pump rate-limited so
        # they never dominate the node throughput (the earlier every-node bug).
        consider(verify(x[:n]))
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
            bd = child(lb, _set(ub, bi, np.floor(x[bi])), basis, ncuts, _rounds)
            bu = child(_set(lb, bi, np.ceil(x[bi])), ub, basis, ncuts, _rounds)
            _update_pc(bi, "d", bound, bd, fd)
            _update_pc(bi, "u", bound, bu, fu)
            continue
        # Integral assignment: spatial-bisect the worst-violated product variable.
        # ``_worst_product_var`` can only see products present in ``info``; once a
        # product is lifted elsewhere (univariate-square bilinear post-#636) it
        # returns None even though the relaxation is NOT tight, so "no branchable
        # product" is NOT a proof that ``bound`` is exact here. Record a verified
        # incumbent (exact objective, never the loose ``bound``); the node is a true,
        # fathomable leaf only when the integer box is fully fixed (a single point,
        # where every nonlinear term is determined). Otherwise it is unresolved: keep
        # its valid lower bound as a global-bound floor so optimality is never claimed
        # over branching the engine could not perform.
        bv = _worst_product_var(x, info, ub - lb)
        if bv is None:
            consider(verify(x[:n]))
            if not bool(np.all(ub[:n] - lb[:n] <= 1e-9)):
                unresolved_lb = min(unresolved_lb, bound)
            continue
        mid = np.floor((lb[bv] + ub[bv]) / 2)
        child(lb, _set(ub, bv, mid), basis, ncuts, _rounds)
        child(_set(lb, bv, mid + 1.0), ub, basis, ncuts, _rounds)
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

    gbound = min([h[0] for h in heap], default=float("inf"))
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
    return LpSpatialResult(
        status=status,
        objective=obj,
        bound=(gbound if (gbound is not None and np.isfinite(gbound)) else None),
        gap=gap,
        x=inc_x,
        node_count=nodes,
    )
