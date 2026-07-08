"""Root branch-and-reduce fixpoint loop (cert:T2.3).

BARON closes most of the global50 set at 0–9 nodes because it iterates *reduction*
(FBBT / OBBT / DBBT with the incumbent as a cutoff) against a re-derived relaxation
until the root box stops moving. discopt has every reduction component but ran them
once, without an incumbent cutoff at the root. This module is the missing loop: a
deterministic, deadline-aware, tighten-only fixpoint over the two stages the R1
entry experiment (cert-gap-plan §14 "T2.1-revisit RESULTS / VERDICT (2026-07-06)")
measured to carry the win —

    S2  ``fbbt_with_cutoff``          (Rust FBBT + the incumbent objective cutoff),
    S3  ``obbt_tighten_root(cutoff)`` (LP-form OBBT/DBBT over the McCormick envelope),

dropping the two stages R1 measured dead here (S1 root presolve: 0% marginal, the
Rust root presolve already ran; S4 envelope-rebuild + re-separation: 0%, consistent
with the Phase-3 1c/zerohalf NO-GOs). The loop is *general* (no instance-keyed code,
per §0.2): its value is broad small-MINLP root closure, not the hard uncertified
tail (st_e36/tspn05/casctanks are R4/relaxation-strength, out of scope).

Soundness (the load-bearing invariant, §0.2/§0.3):

  * **Tighten-only.** Every stage intersects its result into the running box
    (``lb = max(lb, new_lb)``, ``ub = min(ub, new_ub)`` — the solver.py:3914
    pattern). A stage NEVER loosens a bound; a stage failure returns the box
    unchanged. The reduced box is therefore always a subset of the input box, so no
    feasible point is ever removed by the loop itself.
  * **NS-safe bounds only (the C-15 rule).** S3 uses ``obbt_tighten_root``, whose
    tightenings go through the Neumaier–Shcherbina safe vertex clamp; S2 is pure
    interval FBBT. Neither uses a raw LP vertex objective as a bound.
  * **Cutoff-optional.** With no finite incumbent the cutoff-using stages degrade to
    their structural (no-cutoff) subset — FBBT still propagates constraints, OBBT
    still projects — so the loop is always sound to call, incumbent or not.

The loop is behind a flag (``root_fixpoint`` / ``DISCOPT_ROOT_FIXPOINT``,
default OFF until T2.6) at its solver integration point; this module is pure and
flag-agnostic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from discopt.modeling.core import Model

logger = logging.getLogger(__name__)


@dataclass
class RootReduceResult:
    """Outcome of :func:`run_root_fixpoint`.

    ``lb`` / ``ub`` are the final tightened box (always a subset of the input box).
    ``infeasible`` is True iff a stage proved the box empty (a rigorous fathom).
    ``n_tightened`` counts half-bounds tightened across the whole loop, ``n_rounds``
    the number of fixpoint iterations actually run, and ``stage_time`` records the
    wall spent per stage (for the §14 build-results table).
    """

    lb: np.ndarray
    ub: np.ndarray
    infeasible: bool = False
    n_tightened: int = 0
    n_rounds: int = 0
    root_bound_before: Optional[float] = None
    root_bound_after: Optional[float] = None
    stage_time: dict = field(default_factory=lambda: {"fbbt": 0.0, "obbt": 0.0})


# R1's include list, in deterministic order (no adaptive reordering — determinism).
DEFAULT_STAGES = ("fbbt", "obbt")


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


def _stage_fbbt_with_cutoff(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    incumbent_cutoff: Optional[float],
    *,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """S2: Rust ``fbbt_with_cutoff`` over the *current box*, cutoff-aware.

    ``fbbt_with_cutoff`` reads the model's declared per-block bounds, so the box is
    applied by temporarily setting ``v.lb``/``v.ub`` on the model (the obbt.py:1496
    save/restore pattern), building a fresh Rust repr, and restoring afterwards.
    Returns ``(lb, ub, n_tightened, infeasible)``; any failure returns the box
    unchanged (tightening-only, never unsound). Only scalar (size-1) variable
    blocks are mapped, mirroring the existing Phase-C3 FBBT site (solver.py:7142)."""
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
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
            incumbent_bound=(
                float(incumbent_cutoff)
                if incumbent_cutoff is not None and np.isfinite(incumbent_cutoff)
                else None
            ),
        )
    except Exception as exc:
        # C-41: surface, never silently swallow — a swallowed error here is the
        # exact compounding smell behind C-40 (a misaligned map that corrupts a
        # box, then eats the resulting IndexError). Tighten-only: keep the box.
        logger.debug("root cutoff-FBBT skipped (build/solve failed): %s", exc)
        return lb, ub, 0, False
    finally:
        for v, (olb, oub) in zip(model._variables, saved):
            v.lb = olb
            v.ub = oub

    fbbt_lbs = np.asarray(fbbt_lbs, dtype=np.float64)
    fbbt_ubs = np.asarray(fbbt_ubs, dtype=np.float64)
    # C-41: ``fbbt_lbs[bi]`` is BLOCK-indexed (one interval per ``model.variables``
    # block) while ``lb[flat]`` is a FLAT scalar column. The map is sound ONLY when
    # the repr returns exactly ``len(model._variables)`` intervals; a builder-mode /
    # reformulated repr can diverge (C-40: 144 for a 145-column model), and the old
    # ``bi >= shape[0]`` check guarded OOB, not this semantic misalignment — reading
    # the wrong variable's bound and writing a crossed box. On a misaligned repr,
    # forgo this *optional* tightening (a valid, looser box); mirrors solver.py:7443
    # and solvers/_root_presolve.py:43 (CLAUDE.md §3).
    if fbbt_lbs.shape[0] != len(model._variables) or fbbt_ubs.shape[0] != len(model._variables):
        logger.debug(
            "root cutoff-FBBT skipped: repr layout misaligned "
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
        # Round integer bounds inward (FBBT already does, but be defensive).
        if is_int[flat]:
            new_lo = np.ceil(new_lo - 1e-9)
            new_hi = np.floor(new_hi + 1e-9)
        # Tighten-only intersection.
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


def _root_lp_bound(model: Model, lb: np.ndarray, ub: np.ndarray) -> Optional[float]:
    """The McCormick-LP root dual bound over the box, for fixpoint convergence
    detection only (never used as a certificate here). Returns None if no bound."""
    try:
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        relaxer = MccormickLPRelaxer(model, build_incremental=False)
        if not relaxer.has_relaxable_nonlinearity:
            return None
        res = relaxer.solve_at_node(
            np.asarray(lb, dtype=np.float64),
            np.asarray(ub, dtype=np.float64),
            separate=True,
        )
        if res.status == "optimal" and res.lower_bound is not None:
            return float(res.lower_bound)
    except Exception:
        return None
    return None


def run_root_fixpoint(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    incumbent_cutoff: Optional[float] = None,
    deadline: Optional[float] = None,
    tol: float = 1e-6,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    max_rounds: int = 2,
    fbbt_max_iter: int = 10,
    obbt_rounds: int = 3,
    obbt_stage_frac: float = 0.85,
    prefer_pounce: bool = True,
    superposition: bool = False,
    measure_bound: bool = True,
) -> RootReduceResult:
    """Iterate the R1 reduce stages {S2 cutoff-FBBT, S3 cutoff-OBBT} to a fixpoint.

    Parameters
    ----------
    model, lb, ub
        The (already reformulated) model and the current root box.
    incumbent_cutoff
        A valid upper bound on the optimum (an incumbent objective). When None the
        cutoff-using stages degrade to their structural subset (still sound).
    deadline
        Absolute ``time.perf_counter()`` budget for the whole loop; each stage is
        additionally deadline-clamped. The loop stops the moment it is reached.
    tol
        Convergence tolerance on the root-bound move between rounds.
    stages
        Deterministic stage order; a subset/reorder of ``("fbbt", "obbt")``.
    max_rounds
        Fixpoint iteration cap (R1: converges in ≤2 iters).
    obbt_stage_frac
        Fraction of the *remaining* per-round budget handed to S3 OBBT (R1: S3≈85%,
        S2≈15%).

    Returns a :class:`RootReduceResult`. Tighten-only and deadline-safe: any failure
    or timeout returns the best box tightened so far, never a loosened one.
    """
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()

    result = RootReduceResult(lb=lb, ub=ub)

    cutoff = (
        float(incumbent_cutoff)
        if incumbent_cutoff is not None and np.isfinite(incumbent_cutoff)
        else None
    )

    if measure_bound:
        result.root_bound_before = _root_lp_bound(model, lb, ub)

    prev_bound = result.root_bound_before

    for _round in range(max(1, max_rounds)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        round_tightened = 0

        for stage in stages:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())

            if stage == "fbbt":
                t0 = time.perf_counter()
                lb2, ub2, nt, infeas = _stage_fbbt_with_cutoff(
                    model, lb, ub, cutoff, max_iter=fbbt_max_iter, tol=tol
                )
                result.stage_time["fbbt"] += time.perf_counter() - t0
                if infeas:
                    result.infeasible = True
                    result.lb, result.ub = lb, ub
                    return result
                # Intersect (defensive: the stage already returns tighten-only).
                new_lb = np.maximum(lb, lb2)
                new_ub = np.minimum(ub, ub2)
                round_tightened += nt
                lb, ub = new_lb, new_ub

            elif stage == "obbt":
                # S3 gets the lion's share of the per-round budget (R1: ≈85%).
                obbt_budget = None
                if remaining is not None:
                    obbt_budget = remaining * float(obbt_stage_frac)
                obbt_deadline = None if obbt_budget is None else time.perf_counter() + obbt_budget
                t0 = time.perf_counter()
                lb, ub, obbt_infeas, nt = _stage_obbt(
                    model,
                    lb,
                    ub,
                    cutoff,
                    rounds=obbt_rounds,
                    deadline=obbt_deadline,
                    prefer_pounce=prefer_pounce,
                    superposition=superposition,
                )
                result.stage_time["obbt"] += time.perf_counter() - t0
                if obbt_infeas:
                    result.infeasible = True
                    result.lb, result.ub = lb, ub
                    return result
                round_tightened += nt

            if np.any(lb > ub + 1e-9):
                result.infeasible = True
                result.lb, result.ub = lb, ub
                return result

        result.n_tightened += round_tightened
        result.n_rounds = _round + 1

        # Convergence: no bounds moved this round, or the root bound stalled.
        if round_tightened == 0:
            break
        if measure_bound:
            cur_bound = _root_lp_bound(model, lb, ub)
            if (
                cur_bound is not None
                and prev_bound is not None
                and np.isfinite(cur_bound)
                and np.isfinite(prev_bound)
                and abs(cur_bound - prev_bound) < tol * (1.0 + abs(prev_bound))
            ):
                prev_bound = cur_bound
                break
            prev_bound = cur_bound if cur_bound is not None else prev_bound

    if measure_bound:
        result.root_bound_after = (
            _root_lp_bound(model, lb, ub) if prev_bound is None else prev_bound
        )

    result.lb = lb
    result.ub = ub
    return result


def _stage_obbt(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    cutoff: Optional[float],
    *,
    rounds: int,
    deadline: Optional[float],
    prefer_pounce: bool,
    superposition: bool,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """S3: cutoff-aware OBBT/DBBT over the McCormick envelope (the R1 primary lever).

    Delegates to :func:`obbt_tighten_root`, which already runs a ≤``rounds`` internal
    reduce↔rebuild fixpoint, DBBT-first (one objective LP → reduced costs tighten all
    vars) then OBBT (2n min/max probes), all through the NS-safe vertex clamp. It is
    tighten-only and returns the input box on any failure. ``cascade_aux`` stays off
    (measured-dead, §4/§14)."""
    try:
        from discopt._jax.obbt import obbt_tighten_root
    except Exception:
        return lb, ub, False, 0

    try:
        res = obbt_tighten_root(
            model,
            np.asarray(lb, dtype=np.float64),
            np.asarray(ub, dtype=np.float64),
            rounds=rounds,
            deadline=deadline,
            incumbent_cutoff=cutoff,
            superposition=superposition,
            prefer_pounce=prefer_pounce,
            cascade_aux=False,
        )
    except Exception:
        return lb, ub, False, 0

    if res.infeasible:
        return lb, ub, True, 0
    if res.n_tightened > 0:
        new_lb = np.maximum(lb, np.asarray(res.lb, dtype=np.float64))
        new_ub = np.minimum(ub, np.asarray(res.ub, dtype=np.float64))
        return new_lb, new_ub, False, int(res.n_tightened)
    return lb, ub, False, 0
