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
heuristic produces incumbents, verified *exactly* by collapsing the integer box to
the rounded point (where McCormick is exact).

**Scope (this step).** Pure-integer models with a MINIMIZE objective. With every
variable integer, fixing the integers at a leaf determines every nonlinear term, so
the collapsed-box LP value is the true objective and fathoming is exact. Cuts
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


def solve_lp_spatial_bb(
    model: Model,
    *,
    time_limit: float = 300.0,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 500_000,
    use_obbt: bool = True,
) -> Optional[LpSpatialResult]:
    """LP-node spatial branch-and-bound. Returns ``None`` if out of scope."""
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

    root_b, root_x, root_basis = relax(lb0, ub0, None)
    if root_b is None:
        return None

    inc_val = float("inf")
    inc_x: Optional[np.ndarray] = None
    # frontier entries: (bound, tiebreak, lb, ub, x, warm_basis)
    heap = [(root_b, 0, lb0, ub0, root_x, root_basis)]
    counter = 1
    nodes = 0

    def collapsed_incumbent(x):
        """Round integers, verify exactly via the collapsed (singleton) box (where
        the products are fixed, so the McCormick LP value is the true objective)."""
        xr = np.minimum(np.maximum(np.round(x[:n]), lb0), ub0)
        b_, _x, _bas = relax(xr, xr, None)
        return (b_, xr) if b_ is not None else None

    def dive(lb_d, ub_d):
        """Fix-and-dive rounding heuristic: from the node box, repeatedly fix the
        most-fractional integer to its rounded LP value and re-solve, until all
        integers are fixed (a feasible candidate verified by the collapsed box) or
        the LP turns infeasible. Robust where one-shot rounding misses (e.g. nvs19/
        24, where no single rounding of the relaxation optimum is feasible)."""
        lo, hi = lb_d.copy(), ub_d.copy()
        for _ in range(2 * n + 2):
            b_, xx, _bas = relax(lo, hi, None)
            if b_ is None:
                return None
            free = [(abs(xx[i] - round(xx[i])), i) for i in INT if hi[i] - lo[i] > 0.5]
            if not free:
                return collapsed_incumbent(xx)
            # fix the most-fractional still-free integer to its rounded value
            _, bi = max(free)
            v = min(max(round(xx[bi]), lo[bi]), hi[bi])
            lo[bi] = hi[bi] = v
        return None

    def push(lb, ub, parent_basis):
        nonlocal counter
        if np.any(lb > ub + 1e-9):
            return
        b_, x_, basis_ = relax(lb, ub, parent_basis)
        if b_ is not None and b_ < inc_val - 1e-9:
            heapq.heappush(heap, (b_, counter, lb, ub, x_, basis_))
            counter += 1

    # seed an incumbent with a root dive (robust primal)
    seed = dive(lb0, ub0)
    if seed is not None:
        inc_val, inc_x = seed[0], seed[1].copy()

    status = "infeasible"
    while heap:
        if (time.perf_counter() - t0) >= time_limit or nodes >= max_nodes:
            status = "time_limit"
            break
        bound, _, lb, ub, x, basis = heapq.heappop(heap)
        nodes += 1
        # fathom by bound (the frontier is a valid global lower bound)
        if bound >= inc_val - 1e-9 * (1 + abs(inc_val)):
            continue
        # gap check against the best open bound (this node has the min bound)
        if inc_x is not None and abs(inc_val - bound) <= gap_tolerance * (1 + abs(inc_val)):
            status = "optimal"
            break
        # primal: cheap one-shot rounding every node, a deeper dive periodically
        h = collapsed_incumbent(x)
        if h is not None and h[0] < inc_val:
            inc_val, inc_x = h[0], h[1].copy()
        if nodes % 64 == 0:
            hd = dive(lb, ub)
            if hd is not None and hd[0] < inc_val:
                inc_val, inc_x = hd[0], hd[1].copy()
        widths = ub - lb
        # branch: integer-fractional first (children warm-start from this basis)
        frac = [(abs(x[i] - round(x[i])), i) for i in INT]
        frac = [(f, i) for f, i in frac if f > _INT_TOL]
        if frac:
            _, bi = max(frac)
            push(lb, _set(ub, bi, np.floor(x[bi])), basis)
            push(_set(lb, bi, np.ceil(x[bi])), ub, basis)
            continue
        # integral assignment: spatial-bisect the worst-violated product var
        bv = _worst_product_var(x, info, widths)
        if bv is None:
            # all products tight at an integral point => true feasible solution
            if bound < inc_val:
                inc_val, inc_x = bound, x[:n].copy()
            continue
        mid = np.floor((lb[bv] + ub[bv]) / 2)
        push(lb, _set(ub, bv, mid), basis)
        push(_set(lb, bv, mid + 1.0), ub, basis)
    else:
        # frontier exhausted: incumbent is proven optimal
        status = "optimal" if inc_x is not None else "infeasible"

    gbound = min([h[0] for h in heap], default=inc_val)
    gbound = min(gbound, inc_val) if inc_x is not None else gbound
    obj = inc_val if inc_x is not None else None
    gap = None
    if obj is not None and gbound is not None and np.isfinite(gbound):
        gap = abs(obj - gbound) / (1 + abs(obj))
    if status == "time_limit" and inc_x is None:
        obj = None
    return LpSpatialResult(
        status=status,
        objective=obj,
        bound=(gbound if np.isfinite(gbound) else None),
        gap=gap,
        x=inc_x,
        node_count=nodes,
    )
