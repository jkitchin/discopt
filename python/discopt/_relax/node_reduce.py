"""Per-node cheap reduction: one ``reduce_node()`` call (cert:T2.4b).

BARON's signature in-tree move is to reduce the node box using information the node
LP *already produced* — no extra LP solves. This unifies the three cheap, sound
reductions the plan (cert-gap-plan §14 T2.4) names, run after each spatial node LP
solve:

  (i)   Rust ``fbbt_with_cutoff`` on the node box — cutoff-aware interval FBBT
        (today it fires only on incumbent *improvement*; here it runs at the node);
  (ii)  free DBBT from the just-solved node LP's reduced costs — ``d = c - A^T y``
        (from the returned ``dual``), with **z = safe_bound, NEVER the raw LP
        objective** (the C-15 rule: the nvs22 #277 / st_ph10 #306 false-certificate
        class). Zero extra LP solves;
  (iii) integer reduced-cost fixing via the ``duality.rs:85`` kernel semantics
        (inward rounding, positive gap slack — mirrors ``milp_driver.rs:1249``).

(ii) and (iii) are the same reduced-cost inequality — LP duality gives, for every
feasible point with objective ``<= z_inc``,

    d_j > 0:  x_j <= lb_j + gap / d_j       (gap = z_inc - z_lp >= 0)
    d_j < 0:  x_j >= ub_j - gap / |d_j|,

with the *only* difference that an integer variable's tightened endpoint is rounded
**inward** (floor an upper, ceil a lower). So they are implemented as one loop keyed
on integrality. Soundness: the McCormick LP is a valid OUTER relaxation and both
``z_lp`` (the NS-safe bound) and ``z_inc`` (the incumbent) are valid bounds, so the
true optimum satisfies these inequalities — the reduction never removes it. Every
tightening is an intersection (never loosens); any failure returns the box unchanged.

Behind the ``node_reduce`` / ``DISCOPT_NODE_REDUCE`` flag at the solver integration
point (default OFF until T2.6); this module is pure and flag-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.modeling.core import Model

logger = logging.getLogger(__name__)

# Guard tolerances mirror ``dbbt_on_relaxation`` (obbt.py) and the Rust
# ``reduced_cost_fixing`` kernel (duality.rs) so the reduction is the same math.
_RC_TOL = 1e-7  # a reduced cost below this magnitude does not press its bound
_EPS = 1e-7  # minimum improvement to record a tightening


@dataclass
class NodeReduceResult:
    """Outcome of :func:`reduce_node`. ``lb``/``ub`` is the tightened node box
    (always a subset of the input). ``infeasible`` is True iff the reduction proved
    the node's subtree cannot improve on the cutoff (a rigorous fathom)."""

    lb: np.ndarray
    ub: np.ndarray
    infeasible: bool = False
    n_tightened: int = 0


def _is_int_mask(model: Model, n: int) -> np.ndarray:
    from discopt.modeling.core import VarType

    is_int = np.zeros(n, dtype=bool)
    k = 0
    for v in model._variables:
        flag = v.var_type in (VarType.BINARY, VarType.INTEGER)
        for _ in range(v.size):
            if k < n:
                is_int[k] = flag
            k += 1
    return is_int


def _dbbt_from_reduced_costs(
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    reduced_costs: np.ndarray,
    z_lp: float,
    cutoff: float,
    is_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """DBBT + integer RC-fixing from the node LP's reduced costs (moves (ii)+(iii)).

    ``z_lp`` MUST be the NS-safe bound (never the raw LP vertex objective). Mirrors
    ``dbbt_on_relaxation`` (obbt.py:1202) and the ``reduced_cost_fixing`` kernel
    (duality.rs:85): gap = cutoff - z_lp (>= 0), and an integer endpoint is rounded
    inward. Tighten-only; returns ``(lb, ub, n_tightened, infeasible)``."""
    lb = np.asarray(node_lb, dtype=np.float64).copy()
    ub = np.asarray(node_ub, dtype=np.float64).copy()
    d = np.asarray(reduced_costs, dtype=np.float64)

    if not np.isfinite(cutoff) or not np.isfinite(z_lp):
        return lb, ub, 0, False
    gap = float(cutoff) - float(z_lp)
    if gap < -1e-9:
        # LP already proves the cutoff infeasible for this node -> fathom.
        return lb, ub, 0, True
    gap = max(gap, 0.0)
    # Safety margin so residual dual tolerance cannot over-tighten (matches
    # dbbt_on_relaxation).
    gap += 1e-6 * (1.0 + abs(float(cutoff)))

    n = min(lb.size, d.shape[0])
    n_tight = 0
    for j in range(n):
        dj = float(d[j])
        if not np.isfinite(dj):
            continue
        if lb[j] > ub[j]:
            continue
        if dj > _RC_TOL and np.isfinite(lb[j]):
            cand = lb[j] + gap / dj
            if is_int[j]:
                cand = np.floor(cand + 1e-9)
            if cand < ub[j] - _EPS:
                ub[j] = max(lb[j], cand)
                n_tight += 1
        elif dj < -_RC_TOL and np.isfinite(ub[j]):
            cand = ub[j] - gap / (-dj)
            if is_int[j]:
                cand = np.ceil(cand - 1e-9)
            if cand > lb[j] + _EPS:
                lb[j] = min(ub[j], cand)
                n_tight += 1
        if lb[j] > ub[j] + 1e-9:
            return lb, ub, n_tight, True
    return lb, ub, n_tight, False


def _fbbt_on_node(
    model: Model,
    model_repr,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    cutoff: Optional[float],
    *,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Move (i): Rust ``fbbt_with_cutoff`` on the node box.

    ``model_repr`` is the persistent Rust repr (built once). ``fbbt_with_cutoff``
    reads the repr's declared per-block bounds, so the node box is applied by
    temporarily setting ``v.lb``/``v.ub`` on the model and rebuilding a repr from it
    — the same save/restore pattern the root loop and obbt.py use. Only scalar
    (size-1) blocks are mapped (matches solver.py:7142). Tighten-only."""
    lb = np.asarray(node_lb, dtype=np.float64).copy()
    ub = np.asarray(node_ub, dtype=np.float64).copy()
    n = lb.size
    try:
        from discopt._rust import model_to_repr
    except Exception:
        return lb, ub, 0, False

    saved = [(v.lb, v.ub) for v in model._variables]
    try:
        off = 0
        for v in model._variables:
            sz = v.size
            if off + sz <= n:
                v.lb = lb[off : off + sz].reshape(v.lb.shape)
                v.ub = ub[off : off + sz].reshape(v.ub.shape)
            off += sz
        repr_ = model_to_repr(model, getattr(model, "_builder", None))
        fbbt_lbs, fbbt_ubs = repr_.fbbt_with_cutoff(
            max_iter=max_iter,
            tol=tol,
            incumbent_bound=(float(cutoff) if cutoff is not None and np.isfinite(cutoff) else None),
        )
    except Exception as exc:
        # C-41: surface, never silently swallow — a swallowed error here is the
        # exact compounding smell behind C-40 (a misaligned map that corrupts a
        # box, then eats the resulting IndexError). Tighten-only: on any failure
        # keep the node box unchanged (a valid, looser box).
        logger.debug("node cutoff-FBBT skipped (build/solve failed): %s", exc)
        return lb, ub, 0, False
    finally:
        for v, (olb, oub) in zip(model._variables, saved):
            v.lb = olb
            v.ub = oub

    fbbt_lbs = np.asarray(fbbt_lbs, dtype=np.float64)
    fbbt_ubs = np.asarray(fbbt_ubs, dtype=np.float64)
    # C-41: the block->flat map below reads ``fbbt_lbs[bi]`` (BLOCK-indexed: the
    # Rust ``fbbt_with_cutoff`` returns one interval per ``model.variables`` block)
    # and writes ``lb[flat]`` (FLAT scalar column). That is sound ONLY when the
    # repr's block layout aligns 1:1 with ``model._variables`` — i.e. the returned
    # array has exactly ``len(model._variables)`` entries. A builder-mode /
    # reformulated repr can return a DIFFERENT block count (C-40: 144 for a
    # 145-column model); ``fbbt_lbs[bi]`` would then read a *misaligned* variable's
    # bound and write a crossed ``lb>ub`` box, wrongly fathoming the node. The old
    # ``bi >= shape[0]`` check only guarded OOB, not this semantic misalignment.
    # On a misaligned repr, forgo this *optional* tightening (a valid, looser box);
    # mirrors solver.py:7443 and solvers/_root_presolve.py:43 (CLAUDE.md §3).
    if fbbt_lbs.shape[0] != len(model._variables) or fbbt_ubs.shape[0] != len(model._variables):
        logger.debug(
            "node cutoff-FBBT skipped: repr layout misaligned "
            "(returned %d/%d intervals, n_blocks=%d)",
            fbbt_lbs.shape[0],
            fbbt_ubs.shape[0],
            len(model._variables),
        )
        return lb, ub, 0, False
    is_int = _is_int_mask(model, n)
    n_tight = 0
    flat = 0
    infeasible = False
    for bi, v in enumerate(model._variables):
        if v.size != 1:
            flat += v.size
            continue
        new_lo = float(fbbt_lbs[bi])
        new_hi = float(fbbt_ubs[bi])
        if is_int[flat]:
            new_lo = np.ceil(new_lo - 1e-9)
            new_hi = np.floor(new_hi + 1e-9)
        if np.isfinite(new_lo) and new_lo > lb[flat] + 1e-10:
            lb[flat] = new_lo
            n_tight += 1
        if np.isfinite(new_hi) and new_hi < ub[flat] - 1e-10:
            ub[flat] = new_hi
            n_tight += 1
        if lb[flat] > ub[flat] + 1e-9:
            infeasible = True
        flat += 1
    return lb, ub, n_tight, infeasible


def reduce_node(
    model: Model,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    lp_result,
    cutoff: Optional[float],
    *,
    model_repr=None,
    do_fbbt: bool = True,
    fbbt_max_iter: int = 8,
    tol: float = 1e-8,
) -> NodeReduceResult:
    """Cheap, sound per-node reduction unifying (i) cutoff-FBBT, (ii) free DBBT from
    the node LP reduced costs, and (iii) integer RC-fixing.

    Parameters
    ----------
    model
        The (reformulated) model.
    node_lb, node_ub
        The current node box.
    lp_result
        The :class:`MccormickLPResult` from the node LP *just solved* with
        ``want_marginals=True`` — supplies ``reduced_costs`` (original columns) and
        ``safe_bound`` (the NS-safe LP bound = ``z_lp``, the C-15 rule). If it lacks
        marginals, moves (ii)/(iii) are skipped (still sound).
    cutoff
        A valid incumbent objective (upper bound on the optimum). With no finite
        cutoff, moves (ii)/(iii) no-op (DBBT needs a gap) and (i) degrades to
        structural FBBT.
    model_repr
        Optional persistent Rust repr (unused directly today; the FBBT stage rebuilds
        a boxed repr — kept for signature stability / future reuse).

    Returns a :class:`NodeReduceResult` (tighten-only; a subset box or an infeasible
    fathom verdict). Never loosens a bound; any failure returns the box unchanged."""
    lb = np.asarray(node_lb, dtype=np.float64).copy()
    ub = np.asarray(node_ub, dtype=np.float64).copy()
    n = lb.size
    is_int = _is_int_mask(model, n)
    total_tight = 0

    cutoff_f = float(cutoff) if cutoff is not None and np.isfinite(cutoff) else None

    # (ii)+(iii): free DBBT / integer RC-fixing from the node LP reduced costs.
    # z_lp = safe_bound (NEVER the raw LP objective) — the C-15 rule.
    rc = getattr(lp_result, "reduced_costs", None)
    safe_bound = getattr(lp_result, "safe_bound", None)
    if (
        rc is not None
        and safe_bound is not None
        and cutoff_f is not None
        and np.isfinite(safe_bound)
    ):
        lb, ub, nt, infeas = _dbbt_from_reduced_costs(
            lb, ub, rc, float(safe_bound), cutoff_f, is_int
        )
        total_tight += nt
        if infeas:
            return NodeReduceResult(lb=lb, ub=ub, infeasible=True, n_tightened=total_tight)

    # (i): cutoff-aware FBBT on the (possibly already DBBT-tightened) node box.
    if do_fbbt:
        lb, ub, nt, infeas = _fbbt_on_node(
            model, model_repr, lb, ub, cutoff_f, max_iter=fbbt_max_iter, tol=tol
        )
        total_tight += nt
        if infeas or np.any(lb > ub + 1e-9):
            return NodeReduceResult(lb=lb, ub=ub, infeasible=True, n_tightened=total_tight)

    return NodeReduceResult(lb=lb, ub=ub, infeasible=False, n_tightened=total_tight)
