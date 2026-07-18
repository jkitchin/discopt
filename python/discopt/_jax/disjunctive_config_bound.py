"""Disjunctive configuration bound for gated-configuration MINLPs (#732 Stage 2).

For models whose objective couples through integer-multilinear products (the #707
reform class), the tree's global dual is dominated by shallow nodes where the
configuration selectors are fractional and every cost coupling relaxes to ~0
(the L3 wall of ``docs/dev/ex1252-certification-plan.md``). But the configuration
space itself is tiny: the reform records which integer factors gate the products
(``model._ipx_config_indicators`` — range-{0,1} selectors — and
``model._ipx_config_counts`` — small-range counts, e.g. pump multiplicities).

This pass computes a valid global lower bound by **partitioning on the
configuration variables** instead of relaxing across them:

1. Enumerate the ``2^k`` indicator patterns (every feasible point has integral
   indicators; the patterns partition the box).
2. Per configuration box: interval **FBBT** (fresh Rust repr per box — proves the
   degenerate patterns crossed/empty in milliseconds, which the ill-conditioned
   LP cannot Farkas-certify), then budgeted **OBBT** with the incumbent cutoff,
   then the node **LP** bound.
3. **Unit-peel** the weakest surviving leaf on a configuration count variable
   (``{lo}`` vs ``[lo+1, ub]`` — a partition of the integer domain, the same
   implied-integrality trust the #707 reform itself relies on), best-first,
   under a leaf/wall budget.

Anytime-valid by construction: every child inherits its parent's bound until its
own is certified, so ``min over leaves`` is a valid lower bound at any budget
cutoff; leaves pruned infeasible contribute nothing and leaves pruned by the
incumbent cutoff contribute their certified bound. Soundness: FBBT/OBBT/LP are
each sound per box, the indicator patterns partition the box exactly, and the
unit-peel partitions the integer domain of a reform-expanded (declared or
implied integer) variable.

Entry experiments (recorded in the plan doc): per-config bounds 71644–115466 on
the single-line ex1252 patterns (one pruned outright by the incumbent cutoff),
12/16 pump-dichotomy children of a multi-line pattern pruned, and the FBBT step
closing the ``numerical`` leaves that poisoned the min.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import product as _iproduct
from typing import Optional

import numpy as np

from discopt.modeling.core import Model

logger = logging.getLogger(__name__)

# A leaf whose certified LP bound sits within this of the incumbent is pruned by
# the cutoff (mirrors the solver's absolute gap tolerance).
_CUTOFF_EPS = 1e-6
# FBBT iteration cap per box (the per-box repr build + sweep measures ~4 ms).
_FBBT_MAX_ITER = 20
_FBBT_TOL = 1e-9


@dataclass
class DisjunctiveConfigResult:
    """Outcome of the disjunctive configuration pass.

    ``bound`` is a valid global lower bound for the model over the input box, or
    ``None`` when the pass declined / could not certify anything above ``-inf``.
    ``infeasible`` is True only when EVERY configuration pattern was pruned
    infeasible — a rigorous proof the input box contains no feasible point.
    """

    bound: Optional[float] = None
    infeasible: bool = False
    n_processed: int = 0
    n_pruned_infeasible: int = 0
    n_pruned_cutoff: int = 0
    n_leaves: int = 0
    wall: float = 0.0


@dataclass
class _Leaf:
    lb: np.ndarray
    ub: np.ndarray
    bound: float  # inherited from the parent until certified — always valid
    certified: bool = False
    attempts: int = 0
    depth: int = 0
    terminal: bool = False  # no splittable variable remains / budget-terminal
    key: tuple = field(default_factory=tuple)


def _box_fbbt(model: Model, lb: np.ndarray, ub: np.ndarray):
    """Interval FBBT restricted to ``[lb, ub]`` via a fresh Rust repr.

    Returns ``(tight_lb, tight_ub, crossed)``; on any failure returns the input
    box with ``crossed=False`` (abstaining is sound — FBBT is a tightener).
    """
    try:
        from discopt._rust import model_to_repr

        rep = model_to_repr(model, getattr(model, "_builder", None))
        off = 0
        for bi, v in enumerate(model._variables):
            rep.tighten_var_bounds(bi, list(lb[off : off + v.size]), list(ub[off : off + v.size]))
            off += v.size
        fl, fu = rep.fbbt(max_iter=_FBBT_MAX_ITER, tol=_FBBT_TOL)
        tl = np.concatenate([np.atleast_1d(np.asarray(b, dtype=np.float64)) for b in fl])
        tu = np.concatenate([np.atleast_1d(np.asarray(b, dtype=np.float64)) for b in fu])
        if tl.shape != lb.shape:
            return lb, ub, False
        # Intersect defensively (FBBT starts from the written box, so this is a
        # no-op unless the binding returned something unexpected).
        tl = np.maximum(tl, lb)
        tu = np.minimum(tu, ub)
        crossed = bool(np.any(tl > tu + 1e-9))
        return tl, tu, crossed
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("disjunctive pass: per-box FBBT abstained: %s", exc)
        return lb, ub, False


def _split_var(leaf: _Leaf, count_flats: list[int]) -> Optional[int]:
    """First configuration count variable with integer width >= 1 in the leaf box."""
    for j in count_flats:
        lo = np.ceil(leaf.lb[j] - 1e-9)
        hi = np.floor(leaf.ub[j] + 1e-9)
        if hi - lo >= 1.0:
            return j
    return None


def compute_disjunctive_config_bound(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    incumbent: Optional[float] = None,
    deadline: Optional[float] = None,
    max_leaf_solves: int = 48,
    max_indicators: int = 4,
    obbt_rounds: int = 3,
    obbt_lp_time: float = 0.2,
    root_floor: float = -np.inf,
) -> DisjunctiveConfigResult:
    """Compute the disjunctive configuration bound over ``[lb, ub]``.

    ``root_floor`` is the caller's existing valid bound for the box (used as the
    inherited bound of unprocessed leaves, keeping the pass anytime-valid).
    Declines (``bound=None``) when the model carries no configuration metadata,
    the indicator count exceeds ``max_indicators``, or nothing above ``-inf``
    could be certified within budget.
    """
    t0 = time.perf_counter()
    res = DisjunctiveConfigResult()

    indicators = sorted(getattr(model, "_ipx_config_indicators", None) or ())
    count_flats = sorted(getattr(model, "_ipx_config_counts", None) or ())
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
    # Only indicators still free in this box need enumeration.
    free_ind = [j for j in indicators if ub[j] - lb[j] > 0.5]
    if not indicators or len(free_ind) > max_indicators:
        return res

    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt._jax.obbt import obbt_tighten_root

    relaxer = MccormickLPRelaxer(model)

    leaves: list[_Leaf] = []
    for pattern in _iproduct((0.0, 1.0), repeat=len(free_ind)):
        clb, cub = lb.copy(), ub.copy()
        for j, v in zip(free_ind, pattern, strict=True):
            clb[j] = cub[j] = v
        leaves.append(_Leaf(clb, cub, root_floor, key=pattern))
    terminal_bounds: list[float] = []

    def _out_of_budget() -> bool:
        if res.n_processed >= max_leaf_solves:
            return True
        return deadline is not None and time.perf_counter() >= deadline

    def _process(leaf: _Leaf) -> Optional[_Leaf]:
        """FBBT -> OBBT(cutoff) -> LP on the leaf. Returns the surviving leaf
        (certified or not), or ``None`` when pruned (stats updated)."""
        res.n_processed += 1
        tl, tu, crossed = _box_fbbt(model, leaf.lb, leaf.ub)
        if crossed:
            res.n_pruned_infeasible += 1
            return None
        leaf.lb, leaf.ub = tl, tu
        try:
            ob = obbt_tighten_root(
                model,
                leaf.lb.copy(),
                leaf.ub.copy(),
                rounds=obbt_rounds,
                time_limit_per_lp=obbt_lp_time,
                incumbent_cutoff=incumbent,
                deadline=deadline,
            )
            if ob.infeasible:
                res.n_pruned_infeasible += 1
                return None
            leaf.lb, leaf.ub = ob.lb.copy(), ob.ub.copy()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("disjunctive pass: OBBT abstained: %s", exc)
        node = relaxer.solve_at_node(leaf.lb.copy(), leaf.ub.copy())
        if node.status == "infeasible":
            res.n_pruned_infeasible += 1
            return None
        leaf.attempts += 1
        if node.status == "optimal" and node.lower_bound is not None:
            b = float(node.lower_bound)
            leaf.bound = max(leaf.bound, b)
            leaf.certified = True
            if incumbent is not None and b >= incumbent - _CUTOFF_EPS:
                # The leaf holds nothing better than the incumbent: its box
                # contributes at least b to the min — terminal, kept as a bound.
                res.n_pruned_cutoff += 1
                terminal_bounds.append(b)
                return None
        return leaf

    # Pass 1: process every indicator pattern once.
    survivors: list[_Leaf] = []
    for leaf in leaves:
        if _out_of_budget():
            survivors.append(leaf)  # unprocessed: keeps its inherited bound
            continue
        kept = _process(leaf)
        if kept is not None:
            survivors.append(kept)

    if not survivors and not terminal_bounds:
        # Every configuration pattern pruned infeasible: the box is empty.
        res.infeasible = res.n_pruned_cutoff == 0
        res.bound = incumbent if res.n_pruned_cutoff else None
        res.n_leaves = 0
        res.wall = time.perf_counter() - t0
        return res

    # Pass 2: best-first unit-peeling — always work the weakest leaf, because the
    # result is min over leaves and only lifting the min helps.
    while not _out_of_budget():
        survivors.sort(key=lambda le: (le.terminal, le.certified, le.bound))
        leaf = survivors[0]
        if leaf.terminal:
            break  # weakest leaf can no longer improve
        if leaf.certified or leaf.attempts > 0:
            j = _split_var(leaf, count_flats)
            if j is None:
                leaf.terminal = True
                continue
            lo = float(np.ceil(leaf.lb[j] - 1e-9))
            left = _Leaf(leaf.lb.copy(), leaf.ub.copy(), leaf.bound, depth=leaf.depth + 1)
            left.ub[j] = lo
            left.lb[j] = min(left.lb[j], lo)
            right = _Leaf(leaf.lb.copy(), leaf.ub.copy(), leaf.bound, depth=leaf.depth + 1)
            right.lb[j] = lo + 1.0
            survivors.remove(leaf)
            for child in (left, right):
                if _out_of_budget():
                    survivors.append(child)  # inherited bound keeps validity
                    continue
                kept = _process(child)
                if kept is not None:
                    survivors.append(kept)
            if not survivors and not terminal_bounds:
                res.infeasible = res.n_pruned_cutoff == 0
                res.bound = incumbent if res.n_pruned_cutoff else None
                res.n_leaves = 0
                res.wall = time.perf_counter() - t0
                return res
        else:
            kept = _process(leaf)
            if kept is None:
                survivors.remove(leaf)
                if not survivors and not terminal_bounds:
                    res.infeasible = res.n_pruned_cutoff == 0
                    res.bound = incumbent if res.n_pruned_cutoff else None
                    res.n_leaves = 0
                    res.wall = time.perf_counter() - t0
                    return res

    all_bounds = [le.bound for le in survivors] + terminal_bounds
    res.n_leaves = len(survivors)
    res.wall = time.perf_counter() - t0
    if not all_bounds:
        return res
    best = float(min(all_bounds))
    res.bound = best if np.isfinite(best) else None
    return res
