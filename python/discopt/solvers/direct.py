"""DIRECT — DIviding RECTangles: deterministic derivative-free global search.

Selected with ``Model.solve(solver="direct")``. Minimizes a bounded model by
partitioning its box into hyperrectangles, sampling each rectangle's centre, and
repeatedly subdividing the rectangles that could hold the optimum for *some*
Lipschitz constant. It needs only a finite box and the ability to evaluate the
model at a point — no derivatives, no relaxation, no Lipschitz constant, and no
convexity.

**It returns no certificate, and must never claim one.** ``bound`` and ``gap`` are
``None``, ``gap_certified`` is ``False``, and the status is never ``"optimal"``
(nor ``"infeasible"`` — DIRECT cannot prove infeasibility). This is stricter than
the local-NLP path on purpose: a sampling method has no dual information at all,
so reporting the incumbent as a bound would be a false certificate (CLAUDE.md §1).

Its niche in discopt is the model whose objective or constraints contain an
opaque ``dm.custom`` (``CustomCall``) body outside the reduced-space MCBox scope
— a simulator, a table lookup, a body written with raw ``jnp`` intrinsics. Such a
model has no algebraic relaxation, so the default path degrades to a single local
NLP with no global search at all (see ``docs/global_optimization.md``). A model
that *can* be written algebraically should not use this backend: spatial
branch-and-bound will certify it, and DIRECT will not.

Where this sits in derivative-free optimization
-----------------------------------------------
**This is a strong baseline, not the state of the art**, and it is worth being
plain about that. DIRECT dates from 1993. What it offers is a good set of
properties rather than best-in-class performance: it is deterministic (same
answer every run), has essentially one hyperparameter, needs no surrogate model
and no fitting cost, and its sampling is dense in the limit. That makes it a
dependable default and a fair yardstick.

What it is not: competitive with modern model-based methods on expensive
objectives. For a genuinely costly evaluation, a surrogate method (Bayesian
optimization / RBF surrogates) reaches a comparable answer in far fewer calls,
because it spends real computation deciding *where* to sample. For local
refinement, trust-region methods (BOBYQA, DFO-LS) beat DIRECT badly — which is
precisely why the local-refinement hybrid here exists. For constrained blackbox
work at scale, MADS-family solvers (NOMAD) are the mature choice.

The one caveat against writing the family off: DIRECT *hybridized with a local
solver* remains competitive. The survey notes that in a recent comparison of
derivative-free algorithms on problems up to 300 variables, the DIRECT variant
``glcCluster`` was among the top performers. The variant matters, and the winners
are all DIRECT-plus-local-search — which is the shape implemented here.

Algorithm
---------
Original DIRECT is Jones, Perttunen & Stuckman (1993). This implementation adds
the modifications that Jones & Martins (2021) conclude are *generally* beneficial,
all default-on:

1. **Trisect one long side**, not all of them (Jones 2001) — ``divide="one"``.
   The side is the long dimension trisected the fewest times so far, ties broken
   by lower index (survey p. 17 fn. 3), which breaks the tie deterministically
   without favouring a dimension and cuts sampling from ``2|M|`` points to 2.
2. **Break ties**: select one rectangle per size level rather than every
   rectangle tied for potentially optimal — ``break_ties=True``.
3. **An ``epsilon`` floor**, so vanishingly small rectangles stop being selected.
4. **Hybridize with a local solve** — the survey's most-endorsed change, since
   DIRECT finds the basin quickly but refines slowly.

Measured on the survey's own ``1+x₁+…+x₅`` (evaluations to 1% accuracy), 1 and 2
together are worth ~86×: 16,555 → 479 → 193, against the published 14,492 → 470 →
192 (``docs/dev/direct-entry-2026-08-12.md``).

Parallel evaluation
-------------------
Within an iteration every sample point is independent, so ``n_jobs`` evaluates
them together. This is the lever that matters when an evaluation is slow: on a
50 ms objective, 8 threads cut a 200-evaluation run from 10.0 s to 1.9 s.

It changes nothing about the answer — a run with ``n_jobs=8`` is *identical* to
``n_jobs=1``, not merely equivalent. Two things make that true and both are easy
to lose: results are collected in input order, and the incumbent is updated at
apply time in selection order rather than as evaluations land. Two more
subtleties were found by differential-testing against the pre-batching code on 44
configurations — the budget guard must charge a pessimistic 2 evaluations per
dimension before sampling, and ``split_counts`` must be charged at *plan* time,
because the ``divide="one"`` tie-break reads it.

Threads, not processes: a ``dm.custom`` model is not picklable, so a process pool
cannot carry the evaluator. The speedup therefore depends on the evaluation
releasing the GIL — numpy/JAX compute and any subprocess or I/O-bound simulator
do; a pure-Python arithmetic body does not.

Local refinement launches from the **best of** the caller's start and DIRECT's
incumbent, keeping the better result, so the backend is no worse than the local
path by construction. That is not cosmetic: on ``griewank_3`` DIRECT's incumbent
sat in a worse basin than the default start, and refining from it alone lost to
today's behaviour.

References
----------
Jones, Perttunen & Stuckman, *JOTA* 79(1), 1993 — the original algorithm.
Jones, *Encyclopedia of Optimization*, 2001 — the revision (1, 2, 4 above).
Jones & Martins, *J. Global Optim.* 79, 2021 — the 25-year survey.
Stripinis, Paulavičius & Žilinskas, *Struct. Multidisc. Optim.* 59, 2019 —
DIRECT-GL / GLce (the ``variant="gl"`` selection and the constraint handling).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from discopt.modeling.core import Model, SolveResult

logger = logging.getLogger(__name__)

__all__ = [
    "solve_direct",
    "select_potentially_optimal",
    "DirectStats",
]

#: Rectangles whose centre-vertex distance falls below this are never selected,
#: independent of ``epsilon``. At 3**-34 the side length is already far below
#: double precision's ability to distinguish neighbouring centres.
_MIN_CENTER_VERTEX_DISTANCE = 3.0**-34

#: Above this dimension DIRECT degrades badly (the survey documents good
#: behaviour to ~6 variables). Warn, never refuse — refusing a legal problem
#: would overreach.
_DIMENSION_WARN_THRESHOLD = 10


@dataclass
class DirectStats:
    """Counters reported through ``SolveResult.solver_stats``."""

    evals: int = 0
    cache_hits: int = 0
    rectangles: int = 0
    levels: int = 0
    iterations: int = 0
    local_solves: int = 0
    local_improvements: int = 0
    local_failures: int = 0
    derivative_free_refines: int = 0
    feasible_found: int = 0
    batches: int = 0
    max_batch_size: int = 0

    def as_dict(self) -> dict[str, float]:
        return {f"direct/{k}": float(v) for k, v in vars(self).items()}


# ══════════════════════════════════════════════════════════════════════════════
# Selection — pure geometry, unit-testable without a Model
# ══════════════════════════════════════════════════════════════════════════════


def _lower_right_hull(pts: list[tuple[float, float, int]]) -> list[int]:
    """Indices minimizing ``f - K*d`` for SOME ``K > 0`` (DIRECT's Eq. 3).

    ``pts`` is ``[(d, f, index)]``, one entry per size level, sorted by ``d``
    ascending. Two steps, and the second is the one that is easy to miss:

    1. Monotone-chain **lower** convex hull; its vertices have monotonically
       increasing supporting slopes left to right.
    2. Keep only the suffix from the minimum-``f`` vertex, which is exactly where
       the supporting slope turns positive. Without it the leftmost vertex is
       always returned even though no ``K > 0`` makes it the minimizer — it is
       optimal only for ``K < 0``, i.e. preferring *smaller* rectangles with
       *worse* values.

    Cross-checked against a brute-force sweep over ``K`` in the unit tests.
    """
    hull: list[tuple[float, float, int]] = []
    for d, fv, idx in pts:
        while len(hull) >= 2:
            (d1, f1, _), (d2, f2, _) = hull[-2], hull[-1]
            # Drop hull[-1] when it sits on/above the segment hull[-2] -> current.
            if (f2 - f1) * (d - d1) >= (fv - f1) * (d2 - d1):
                hull.pop()
            else:
                break
        hull.append((d, fv, idx))
    if not hull:
        return []
    start = min(range(len(hull)), key=lambda i: (hull[i][1], -hull[i][0]))
    return [idx for _, _, idx in hull[start:]]


def select_potentially_optimal(
    sizes: np.ndarray,
    values: np.ndarray,
    epsilon: float = 1e-4,
    f_min: Optional[float] = None,
) -> list[int]:
    """Indices of the potentially optimal rectangles.

    Rectangle ``j`` is potentially optimal when there is a ``K > 0`` with

    * ``f(c_j) - K d_j <= f(c_i) - K d_i`` for all ``i``  (Eq. 3), and
    * ``f(c_j) - K d_j <= f_min - epsilon*|f_min|``       (Eq. 4).

    Eq. 3 is the lower-right convex hull. For Eq. 4, ``f - K d`` decreases in
    ``K``, so the best chance is the LARGEST admissible ``K`` — the slope to the
    next hull point, ``(f_j - f_i) / (d_j - d_i)``. For the rightmost (largest)
    rectangle ``K`` is unbounded above, so ``f - K d -> -inf`` and Eq. 4 always
    holds: **the biggest rectangle is always potentially optimal.** Reading Eq. 4
    with ``K = 0`` there instead all but excludes it and starves global search.

    Parameters
    ----------
    sizes, values
        Centre-vertex distance and centre value, one entry per candidate. Pass
        the best rectangle per size level; duplicates at one size only add work.
    epsilon
        Eq. 4's relative floor on the improvement a selection must promise.
    f_min
        Current incumbent; defaults to ``values.min()``.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if sizes.shape != values.shape or sizes.ndim != 1:
        raise ValueError(
            f"sizes and values must be matching 1-D arrays, got {sizes.shape} / {values.shape}"
        )
    if sizes.size == 0:
        return []
    if f_min is None:
        f_min = float(values.min())

    order = np.argsort(sizes, kind="stable")
    pts = [(float(sizes[i]), float(values[i]), int(i)) for i in order]
    hull = _lower_right_hull(pts)

    lookup = {i: (float(sizes[i]), float(values[i])) for i in hull}
    selected: list[int] = []
    for pos, i in enumerate(hull):
        di, fi = lookup[i]
        if di < _MIN_CENTER_VERTEX_DISTANCE:
            continue
        if pos + 1 == len(hull):
            selected.append(i)  # largest rectangle: K unbounded, Eq. 4 always holds
            continue
        dj, fj = lookup[hull[pos + 1]]
        if dj <= di:
            continue
        k_max = (fj - fi) / (dj - di)
        if fi - k_max * di <= f_min - epsilon * abs(f_min) + 1e-12:
            selected.append(i)
    if not selected and hull:
        selected = [hull[-1]]
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Partition
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class _Partition:
    """Hyperrectangles of the unit cube.

    Each rectangle is a centre ``c`` plus a split-count vector ``t``: its side in
    dimension ``k`` is ``3**-t[k]``. Keeping ``t`` as integers rather than float
    side lengths makes the centre-vertex distance exact and lets rectangles bucket
    into discrete size *levels* by ``t``'s multiset — no float comparison of sizes
    anywhere, which is what keeps selection deterministic.
    """

    centers: list[np.ndarray] = field(default_factory=list)
    t: list[np.ndarray] = field(default_factory=list)
    fvals: list[float] = field(default_factory=list)  # objective at the centre
    viols: list[float] = field(default_factory=list)  # total constraint violation

    def __len__(self) -> int:
        return len(self.fvals)

    def add(self, c: np.ndarray, t: np.ndarray, fval: float, viol: float) -> int:
        self.centers.append(c)
        self.t.append(t)
        self.fvals.append(fval)
        self.viols.append(viol)
        return len(self.fvals) - 1

    def distance(self, i: int) -> float:
        """Centre-to-vertex distance of rectangle ``i`` in the unit cube."""
        sides = 3.0 ** (-self.t[i].astype(np.float64))
        return 0.5 * float(np.sqrt(np.sum(sides * sides)))

    def level_key(self, i: int) -> tuple[int, ...]:
        """Rectangles sharing this key have identical centre-vertex distance."""
        return tuple(sorted(self.t[i].tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# The search
# ══════════════════════════════════════════════════════════════════════════════


class _DirectSearch:
    """DIRECT over the unit cube, driven by a caller-supplied evaluation callable.

    The engine is deliberately independent of ``Model``: it takes a function from
    a unit-cube point to ``(objective, violation)`` and knows nothing about how
    that value is produced. That keeps the algorithm unit-testable on plain
    functions and keeps the model plumbing in :func:`solve_direct`.
    """

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        *,
        integer_mask: Optional[np.ndarray] = None,
        epsilon: float = 1e-4,
        divide: str = "one",
        break_ties: bool = True,
        variant: str = "classic",
    ) -> None:
        if divide not in ("one", "all"):
            raise ValueError(f"divide must be 'one' or 'all', got {divide!r}")
        if variant not in ("classic", "gl"):
            raise ValueError(f"variant must be 'classic' or 'gl', got {variant!r}")
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        # Checked here as well as in solve_direct: the engine is usable directly
        # on a plain callable, and an infinite side silently produces NaN centres
        # (``-inf + 0.5*inf``) that all hash to the same cache key.
        if not (np.all(np.isfinite(self.lb)) and np.all(np.isfinite(self.ub))):
            raise ValueError(
                "DIRECT requires a finite box on every variable; got "
                f"lb={self.lb!r}, ub={self.ub!r}"
            )
        self.width = self.ub - self.lb
        self.n = self.lb.size
        self.integer_mask = (
            np.zeros(self.n, dtype=bool)
            if integer_mask is None
            else np.asarray(integer_mask, dtype=bool)
        )
        self.epsilon = float(epsilon)
        self.divide_rule = divide  # NOT `self.divide`: that would shadow divide()
        self.break_ties = bool(break_ties)
        self.variant = variant

        self.part = _Partition()
        self.split_counts = np.zeros(self.n, dtype=np.int64)
        self.stats = DirectStats()
        self._cache: dict[bytes, tuple[float, float]] = {}

        # Incumbent bookkeeping. ``best_feasible_*`` is the reportable answer;
        # ``best_any_*`` tracks the least-infeasible point for phase A.
        self.best_feasible_value: Optional[float] = None
        self.best_feasible_point: Optional[np.ndarray] = None
        self.best_violation = np.inf
        self.best_violation_point: Optional[np.ndarray] = None
        self.eps_cons = 1e-6

    # -- coordinates -------------------------------------------------------
    def to_model_point(self, u: np.ndarray) -> np.ndarray:
        """Map a unit-cube point to model coordinates, rounding integer dims.

        Rounding here (rather than reshaping the partition) is what implements
        Jones 2001's "the centre always takes an integer value in an integer
        coordinate" without disturbing the unit-cube geometry that makes the
        centre-vertex distance scale-free.
        """
        x = self.lb + u * self.width
        if self.integer_mask.any():
            x = x.copy()
            x[self.integer_mask] = np.clip(
                np.round(x[self.integer_mask]),
                np.ceil(self.lb[self.integer_mask] - 1e-9),
                np.floor(self.ub[self.integer_mask] + 1e-9),
            )
        return x

    def _divisible(self, t: np.ndarray) -> np.ndarray:
        """Dimensions still worth splitting.

        An integer dimension stops once its side spans less than one unit — every
        integer value inside it is then already represented by the centre, so
        splitting further only re-evaluates the same rounded point.
        """
        sides = 3.0 ** (-t.astype(np.float64))
        ok = np.ones(self.n, dtype=bool)
        if self.integer_mask.any():
            ok[self.integer_mask] = (
                sides[self.integer_mask] * self.width[self.integer_mask]
            ) >= 1.0
        # A dimension of zero width is never worth splitting.
        ok &= self.width > 0.0
        return ok

    # -- ranking (DIRECT-GLce) ---------------------------------------------
    def rank_values(self) -> np.ndarray:
        """Per-rectangle ranking value, following DIRECT-GLce.

        * **Phase A** (no feasible point yet): rank by total constraint violation,
          so the search first hunts for feasibility.
        * **Phase B** (a feasible point exists): a feasible centre ranks by its
          objective; an infeasible one is penalized by its violation *plus*
          ``|f - f_min|``. That last term is what stops an infeasible point from
          earning credit for an objective below the incumbent, and it needs no
          penalty weight to be tuned.
        * The ``ce`` refinement: a violation within ``eps_cons`` is treated as
          feasible, so the ranking is not discontinuous exactly at the boundary
          where optima usually sit (DIRECT has no convergence guarantee across a
          discontinuity).
        """
        f = np.asarray(self.part.fvals, dtype=np.float64)
        v = np.asarray(self.part.viols, dtype=np.float64)
        if self.best_feasible_value is None:
            return v
        near_feasible = v <= self.eps_cons
        return np.where(near_feasible, f, f + v + np.abs(f - self.best_feasible_value))

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self, u: np.ndarray, oracle: Callable[[np.ndarray], tuple[float, float]]
    ) -> tuple[float, float]:
        """Evaluate at unit-cube point ``u``, memoized on the model point.

        Integer rounding can map distinct rectangle centres onto the same model
        point; the cache makes that free rather than a wasted call. It matters
        whenever an evaluation is expensive, which is the regime this backend
        exists for.
        """
        x = self.to_model_point(u)
        key = x.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            self.stats.cache_hits += 1
            return hit
        fval, viol = oracle(x)
        self._cache[key] = (fval, viol)
        self.stats.evals += 1
        self._offer(x, fval, viol)
        return fval, viol

    def _offer(self, x: np.ndarray, fval: float, viol: float) -> None:
        """Update the incumbent from a freshly evaluated model point."""
        if viol < self.best_violation:
            self.best_violation = viol
            self.best_violation_point = x.copy()
        if viol <= self.eps_cons and np.isfinite(fval):
            if self.best_feasible_value is None:
                self.stats.feasible_found += 1
            if self.best_feasible_value is None or fval < self.best_feasible_value:
                self.best_feasible_value = float(fval)
                self.best_feasible_point = x.copy()

    # -- selection ---------------------------------------------------------
    def _select(self) -> list[int]:
        ranks = self.rank_values()
        by_level: dict[tuple[int, ...], int] = {}
        for i in range(len(self.part)):
            key = self.part.level_key(i)
            cur = by_level.get(key)
            if cur is None or ranks[i] < ranks[cur]:
                by_level[key] = i
        self.stats.levels = len(by_level)
        idx = list(by_level.values())
        sizes = np.array([self.part.distance(i) for i in idx])
        values = ranks[idx]
        f_min = float(ranks.min())

        chosen_positions = select_potentially_optimal(sizes, values, self.epsilon, f_min)
        selected = [idx[p] for p in chosen_positions]

        if self.variant == "gl":
            selected = sorted(set(selected) | set(self._gl_local_step()))

        if not self.break_ties:
            selected = self._expand_ties(selected, ranks)
        return selected

    def _gl_local_step(self) -> list[int]:
        """DIRECT-GL's second, 'local' selection step.

        Rectangles nondominated on (distance from the current best point: low;
        rectangle size: large). Focuses effort near the incumbent while still
        preferring the larger — i.e. less explored — of the near rectangles.
        """
        anchor = self.best_feasible_point
        if anchor is None:
            anchor = self.best_violation_point
        if anchor is None:
            return []
        anchor_u = (anchor - self.lb) / np.where(self.width > 0.0, self.width, 1.0)
        by_level: dict[tuple[int, ...], int] = {}
        dist: dict[int, float] = {}
        for i in range(len(self.part)):
            di = float(np.linalg.norm(self.part.centers[i] - anchor_u))
            dist[i] = di
            key = self.part.level_key(i)
            cur = by_level.get(key)
            if cur is None or di < dist[cur]:
                by_level[key] = i
        idx = list(by_level.values())
        # Pareto front on (small distance, large size): sort by size descending and
        # keep entries that strictly improve on the best distance seen so far.
        idx.sort(key=lambda i: (-self.part.distance(i), dist[i]))
        front: list[int] = []
        best = np.inf
        for i in idx:
            if dist[i] < best - 1e-15:
                front.append(i)
                best = dist[i]
        return front

    def _expand_ties(self, selected: list[int], ranks: np.ndarray) -> list[int]:
        """Original behaviour: also take every rectangle tied at a selected level."""
        out: set[int] = set()
        for i in selected:
            key = self.part.level_key(i)
            for j in range(len(self.part)):
                if self.part.level_key(j) == key and ranks[j] <= ranks[i] + 1e-12:
                    out.add(j)
        return sorted(out)

    # -- division ----------------------------------------------------------
    def _split_dims(self, i: int) -> list[int]:
        t_i = self.part.t[i]
        divisible = self._divisible(t_i)
        if not divisible.any():
            return []
        masked = np.where(divisible, t_i, np.iinfo(np.int64).max)
        t_min = int(masked.min())
        longest = [k for k in range(self.n) if divisible[k] and int(t_i[k]) == t_min]
        if self.divide_rule == "all":
            return longest
        # Jones 2001: the long side trisected the fewest times over the whole
        # search, ties to the lower index. Deterministic, and it does not let one
        # dimension run away with the splitting budget.
        return [min(longest, key=lambda k: (int(self.split_counts[k]), k))]

    def _plan_division(self, i: int, budget_used: int, max_evals: int):
        """Sample points rectangle ``i`` needs, without evaluating any of them.

        Splitting planning from evaluation is what makes an iteration batchable:
        every point the whole iteration needs can be gathered first, evaluated
        together, and only then applied. Returns ``(entry, new_points, cost)``
        where ``entry`` maps a split dimension to its ``(u_minus, u_plus)`` pair.
        """
        dims = self._split_dims(i)
        if not dims:
            return {}, [], 0
        t_i = self.part.t[i]
        c_i = self.part.centers[i]
        entry: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        new_points: list[np.ndarray] = []
        cost = 0
        for k in dims:
            # Budget guard, deliberately identical to the pre-batching serial
            # code: charge the PESSIMISTIC 2 evaluations per dimension up front,
            # even though a cached point turns out to be free. Charging only the
            # uncached points shifts where the budget cuts off mid-iteration,
            # which changes the partition and so the answer.
            if budget_used + cost + 2 > max_evals:
                break
            delta = 3.0 ** (-(int(t_i[k]) + 1))
            u_minus = c_i.copy()
            u_minus[k] = max(0.0, c_i[k] - delta)
            u_plus = c_i.copy()
            u_plus[k] = min(1.0, c_i[k] + delta)
            # ...but only uncached points are actually *spent*, matching how the
            # serial path's counter moved.
            cost += sum(
                1 for u in (u_minus, u_plus) if self.to_model_point(u).tobytes() not in self._cache
            )
            new_points.extend((u_minus, u_plus))
            entry[k] = (u_minus, u_plus)
            # Charge the split HERE, not at apply time. The `divide="one"`
            # tie-break picks the long dimension split fewest times so far, and
            # the serial code divided each rectangle before planning the next, so
            # a later rectangle saw the earlier ones' increments. Planning a whole
            # iteration up front and incrementing at apply time leaves the counter
            # stale and picks different dimensions.
            self.split_counts[k] += 1
        return entry, new_points, cost

    def _apply_division(self, i: int, entry: dict[int, tuple[np.ndarray, np.ndarray]]) -> bool:
        """Trisect rectangle ``i`` using already-evaluated sample points.

        Incumbent updates happen HERE, in the same order the serial path would
        make them, so batching cannot change the search trajectory: ``_w`` below
        reads ``_scalar_rank``, which depends on the incumbent, and offering a
        later rectangle's points early would reorder this rectangle's split.
        """
        if not entry:
            return False
        t_i = self.part.t[i]
        c_i = self.part.centers[i]

        samples: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
        for k, (u_minus, u_plus) in entry.items():
            pair = []
            for u in (u_minus, u_plus):
                x = self.to_model_point(u)
                fval, viol = self._cache[x.tobytes()]
                self._offer(x, fval, viol)
                pair.append((fval, viol))
            samples[k] = (pair[0], pair[1])

        # Split in increasing order of w_k = min(f-, f+) so the best values end up
        # in the biggest subrectangles (survey Algorithm 1), which is what biases
        # subsequent search toward promising regions.
        def _w(k: int) -> float:
            (fm, vm), (fp, vp) = samples[k]
            return min(self._scalar_rank(fm, vm), self._scalar_rank(fp, vp))

        t_child = t_i.copy()
        for k in sorted(samples, key=lambda k: (_w(k), k)):
            t_child[k] += 1  # split_counts was already charged in _plan_division
            delta = 3.0 ** (-int(t_child[k]))
            (fm, vm), (fp, vp) = samples[k]
            u_minus = c_i.copy()
            u_minus[k] = max(0.0, c_i[k] - delta)
            u_plus = c_i.copy()
            u_plus[k] = min(1.0, c_i[k] + delta)
            self.part.add(u_minus, t_child.copy(), fm, vm)
            self.part.add(u_plus, t_child.copy(), fp, vp)
        self.part.t[i] = t_child
        return True

    def _evaluate_points(
        self,
        points: list[np.ndarray],
        oracle: Callable[[np.ndarray], tuple[float, float]],
        executor,
    ) -> None:
        """Evaluate and cache ``points``; does NOT touch the incumbent.

        With an executor, evaluations run concurrently and results are collected
        in **input order** (``executor.map`` preserves it), so the cache ends up
        identical to the serial path regardless of completion order. The
        incumbent is deliberately left alone here — see :meth:`_apply_division`.
        """
        fresh: list[np.ndarray] = []
        seen: set[bytes] = set()
        for u in points:
            key = self.to_model_point(u).tobytes()
            if key in self._cache or key in seen:
                continue
            seen.add(key)
            fresh.append(u)
        if not fresh:
            return
        xs = [self.to_model_point(u) for u in fresh]
        if executor is None or len(xs) == 1:
            results = [oracle(x) for x in xs]
        else:
            results = list(executor.map(oracle, xs))
        for x, (fval, viol) in zip(xs, results):
            self._cache[x.tobytes()] = (fval, viol)
            self.stats.evals += 1

    def _scalar_rank(self, fval: float, viol: float) -> float:
        if self.best_feasible_value is None:
            return viol
        if viol <= self.eps_cons:
            return fval
        return fval + viol + abs(fval - self.best_feasible_value)

    # -- driver ------------------------------------------------------------
    def run(
        self,
        oracle: Callable[[np.ndarray], tuple[float, float]],
        max_evals: int,
        deadline: Optional[float] = None,
        on_iteration: Optional[Callable[["_DirectSearch"], None]] = None,
        executor=None,
    ) -> None:
        """Run until the evaluation budget or the deadline is exhausted.

        ``executor`` is an optional ``concurrent.futures`` executor used to
        evaluate each iteration's sample points concurrently. Results are still
        collected in input order and applied in selection order, so the search
        trajectory -- and the final answer -- do not depend on it.
        """
        if not self.part:
            c0 = np.full(self.n, 0.5)
            fval, viol = self.evaluate(c0, oracle)
            self.part.add(c0, np.zeros(self.n, dtype=np.int64), fval, viol)

        while self.stats.evals < max_evals:
            if deadline is not None and time.perf_counter() >= deadline:
                logger.info("DIRECT: time limit reached after %d evaluations", self.stats.evals)
                break
            selected = self._select()
            if not selected:
                break
            evals_before = self.stats.evals

            # An iteration in three steps, which is what makes it parallelizable:
            # plan every sample point the selected rectangles need, evaluate them
            # together, then apply the divisions in the original selection order.
            # Splitting it this way is also what keeps the result independent of
            # n_jobs -- see _evaluate_points and _apply_division.
            plans: list[tuple[int, dict[int, tuple[np.ndarray, np.ndarray]]]] = []
            batch: list[np.ndarray] = []
            spent = 0
            for i in selected:
                # The same two guards the pre-batching serial loop applied before
                # each rectangle; dropping them shifts the mid-iteration cutoff.
                if self.stats.evals + spent >= max_evals:
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                entry, points, cost = self._plan_division(i, self.stats.evals + spent, max_evals)
                if not entry:
                    continue
                plans.append((i, entry))
                batch.extend(points)
                spent += cost

            self._evaluate_points(batch, oracle, executor)
            self.stats.batches += 1
            self.stats.max_batch_size = max(self.stats.max_batch_size, len(batch))

            progressed = False
            for i, entry in plans:
                progressed |= self._apply_division(i, entry)
            self.stats.iterations += 1
            if on_iteration is not None:
                on_iteration(self)
            # A whole iteration that subdivided rectangles but produced no NEW
            # evaluation means every fresh centre collapsed onto a point already
            # in the cache — the partition is finer than the model's own
            # resolution (integer coordinates exhausted, or a degenerate box).
            # Without this the budget loop never advances, since a cache hit
            # deliberately does not spend budget, and the search spins forever
            # growing the partition. Found by test_evaluation_cache_absorbs_
            # rounded_duplicates, which hung rather than failed.
            if progressed and self.stats.evals == evals_before:
                logger.info(
                    "DIRECT: partition is finer than the model's resolution after "
                    "%d evaluations (every new centre repeats a sampled point); stopping",
                    self.stats.evals,
                )
                break
            if not progressed:
                logger.info(
                    "DIRECT: partition can no longer be refined after %d evaluations "
                    "(every selected rectangle is at the resolution floor)",
                    self.stats.evals,
                )
                break
        self.stats.rectangles = len(self.part)


# ══════════════════════════════════════════════════════════════════════════════
# Model-facing entry point
# ══════════════════════════════════════════════════════════════════════════════


def _build_oracle(model: Model, feas_tol: float):
    """``(evaluate, n_vars, integer_mask)`` for ``model``.

    ``evaluate`` maps a model point to ``(objective, total_violation)``. Both come
    from the one evaluator funnel the rest of the solver uses, so an opaque
    ``dm.custom`` body is evaluated exactly as the local NLP path would.
    """
    from discopt.solver import _extract_variable_info, _infer_constraint_bounds, _make_evaluator

    evaluator = _make_evaluator(model)
    logger.info("DIRECT: evaluator backend is %s", type(evaluator).__name__)

    n_vars, _lb, _ub, int_offsets, int_sizes = _extract_variable_info(model)
    integer_mask = np.zeros(n_vars, dtype=bool)
    for off, size in zip(int_offsets, int_sizes):
        integer_mask[off : off + size] = True

    n_cons = int(getattr(evaluator, "n_constraints", 0) or 0)
    if n_cons:
        cl, cu = _infer_constraint_bounds(model, evaluator)
        cl = np.asarray(cl, dtype=np.float64)
        cu = np.asarray(cu, dtype=np.float64)
    else:
        cl = cu = None

    def evaluate(x: np.ndarray) -> tuple[float, float]:
        fval = float(evaluator.evaluate_objective(x))
        if not np.isfinite(fval):
            # A black box may be undefined here. Treat it as an unusable point
            # rather than letting a NaN poison the ordering: +inf loses every
            # comparison, which is the honest ranking for "no value".
            fval = np.inf
        viol = 0.0
        if cl is not None:
            g = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64)
            g = np.where(np.isfinite(g), g, np.inf)
            viol = float(np.sum(np.maximum(0.0, g - cu)) + np.sum(np.maximum(0.0, cl - g)))
        return fval, viol

    del feas_tol  # the caller owns the tolerance; kept for signature stability
    return evaluate, n_vars, integer_mask


def _refine_derivative_free(
    search: "_DirectSearch",
    oracle: Callable[[np.ndarray], tuple[float, float]],
    start: np.ndarray,
    max_fev: int,
) -> None:
    """Derivative-free polish with Powell, integers held fixed.

    The gradient path (:func:`_refine_locally`) is the better refiner when the
    objective really is differentiable, and a ``dm.custom`` body is JAX-traceable
    by construction. But traceable is not the same as *usefully* differentiable:
    a body containing ``jnp.round``/``jnp.floor``, a table lookup, or a simulator
    behind ``jax.pure_callback`` hands back zero or meaningless gradients, and a
    gradient method will sit still while reporting success. Powell needs no
    derivatives at all.

    Powell rather than Nelder-Mead: it handles bounds properly in SciPy and
    behaves better on the mildly ill-conditioned valleys DIRECT tends to hand
    over. SciPy is already a core dependency, so this costs no new install.

    Points are scored with the same GLce auxiliary the search itself ranks by, so
    the polish optimizes exactly what the search optimizes and no new penalty
    weight is introduced. Every evaluation is routed through the oracle so it
    counts against the budget and updates the incumbent.
    """
    from scipy.optimize import minimize

    free = ~search.integer_mask
    if not free.any():
        return  # nothing continuous to polish
    lo = search.lb[free]
    hi = search.ub[free]
    if np.any(hi <= lo):
        return
    base = np.asarray(start, dtype=np.float64).copy()

    def objective(z: np.ndarray) -> float:
        x = base.copy()
        x[free] = np.clip(z, lo, hi)
        fval, viol = oracle(x)
        search.stats.evals += 1
        search._offer(x, fval, viol)
        return search._scalar_rank(fval, viol)

    try:
        minimize(
            objective,
            base[free],
            method="Powell",
            bounds=list(zip(lo, hi)),
            options={"maxfev": int(max(1, max_fev)), "xtol": 1e-8, "ftol": 1e-10},
        )
    except Exception as exc:
        # Reported, never swallowed: a polish that cannot run must not read as
        # "the hybrid ran and did not help" (CLAUDE.md §7).
        search.stats.local_failures += 1
        logger.warning("DIRECT: derivative-free refinement raised (%s)", exc)


def _refine_locally(
    model: Model,
    start: np.ndarray,
    integer_mask: np.ndarray,
    time_limit: float,
    nlp_solver: str,
    stats: DirectStats,
) -> Optional[tuple[float, np.ndarray]]:
    """Local NLP from ``start``; ``None`` when it yields nothing usable.

    Integer variables are pinned to ``start``'s values for the duration, so a
    mixed-integer model refines its continuous part instead of skipping
    refinement entirely. Bounds are restored unconditionally.
    """
    import time as _time

    from discopt.solver import _solve_continuous
    from discopt.solvers.amp import _restore_variable_bounds, _snapshot_variable_bounds

    saved = _snapshot_variable_bounds(model)
    try:
        if integer_mask.any():
            offset = 0
            for var in model._variables:
                size = var.size
                sl = slice(offset, offset + size)
                if integer_mask[sl].any():
                    fixed = np.round(np.asarray(start[sl], dtype=np.float64))
                    var.lb = fixed.reshape(var.shape).copy()
                    var.ub = fixed.reshape(var.shape).copy()
                offset += size
        stats.local_solves += 1
        result = _solve_continuous(
            model,
            time_limit,
            None,
            _time.perf_counter(),
            nlp_solver,
            initial_point=np.asarray(start, dtype=np.float64),
        )
    except Exception as exc:
        # Never swallowed silently: a refinement that cannot run is reported and
        # counted, because a quiet no-op here reads as "the hybrid ran and did
        # not help" (CLAUDE.md §7).
        stats.local_failures += 1
        logger.warning("DIRECT: local refinement raised (%s); keeping the sampled incumbent", exc)
        return None
    finally:
        _restore_variable_bounds(saved)

    if result.objective is None or result.x is None:
        stats.local_failures += 1
        logger.info(
            "DIRECT: local refinement returned no usable point (status=%s); "
            "keeping the sampled incumbent",
            result.status,
        )
        return None
    flat = np.concatenate(
        [np.asarray(result.x[v.name], dtype=np.float64).reshape(-1) for v in model._variables]
    )
    return float(result.objective), flat


def solve_direct(
    model: Model,
    *,
    time_limit: float = 3600.0,
    max_evals: int = 5000,
    epsilon: float = 1e-4,
    direct_variant: str = "classic",
    divide: str = "one",
    break_ties: bool = True,
    local_refine: bool = True,
    local_refine_after: int = 100,
    local_refine_time_limit: float = 30.0,
    local_refine_method: str = "auto",
    n_jobs: int = 1,
    feasibility_tolerance: float = 1e-6,
    nlp_solver: str = "pounce",
    initial_point: Optional[np.ndarray] = None,
) -> SolveResult:
    """Minimize ``model`` over its box with DIRECT. **Returns no certificate.**

    Parameters
    ----------
    max_evals
        Objective-evaluation budget. This is the primary control: DIRECT's cost is
        counted in evaluations, not nodes. Repeated points are served from a cache
        and do not consume budget.
    epsilon
        Eq. 4's relative floor. With ``local_refine`` on, a larger value (~1e-2) is
        the survey's advice — the local solve, not DIRECT, does the fine-tuning.
    direct_variant
        ``"classic"`` (default) or ``"gl"`` for DIRECT-GL's two-step selection.
        ``"gl"`` is better on multimodal problems and materially worse on unimodal
        ones (the survey reports Hartman-6 at 8793 evaluations versus 571), so it
        is opt-in rather than the default.
    divide, break_ties
        The Jones 2001 revisions; see the module docstring. Defaults measured.
    local_refine, local_refine_after
        Run a local solve once ``local_refine_after`` evaluations have been spent,
        then again whenever sampling improves on the last local solution. The
        refinement starts from the best of the caller's ``initial_point`` and
        DIRECT's incumbent, so this can only improve on the local-only path.
    local_refine_method
        ``"nlp"`` uses discopt's gradient-based local solver; ``"derivative-free"``
        uses Powell, which needs no gradients at all. ``"auto"`` (default) runs the
        NLP and falls back to Powell when it fails or fails to improve — the right
        default because a ``dm.custom`` body is JAX-*traceable* by construction but
        not necessarily usefully *differentiable* (``jnp.round``, a table lookup, a
        simulator behind ``jax.pure_callback`` all yield useless gradients, and a
        gradient method then stalls while reporting success).
    n_jobs
        Evaluate each iteration's sample points concurrently on this many threads
        (``1`` = serial, the default; ``-1`` = one per CPU). Within an iteration
        every sample is independent, so this is a near-linear wall-clock win when
        an evaluation is slow — and it changes nothing about the answer: results
        are collected in input order and applied in selection order, so a run with
        ``n_jobs=8`` is identical to ``n_jobs=1``, not merely equivalent.

        Threads, not processes: a ``dm.custom`` model is not picklable (the Rust
        model repr refuses), so a process pool cannot carry the evaluator. That
        makes the speedup depend on the evaluation releasing the GIL — numpy/JAX
        compute and any subprocess or I/O-bound simulator do; a pure-Python
        arithmetic body does not, and will see no gain.

    Raises
    ------
    ValueError
        If the model has no objective, or any variable bound is non-finite.
        DIRECT is defined only on a finite box; substituting a big-M box would be
        a silent approximation, and it is the caller who knows the real range.
    """
    from discopt.solver import _unpack_solution

    t_start = time.perf_counter()

    if model._objective is None:
        raise ValueError(
            "solver='direct' requires an objective; the model has none. "
            "DIRECT is a minimization method, not a feasibility search."
        )
    if direct_variant not in ("classic", "gl"):
        raise ValueError(f"direct_variant must be 'classic' or 'gl', got {direct_variant!r}")
    if max_evals < 1:
        raise ValueError(f"max_evals must be >= 1, got {max_evals}")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be >= 1 or -1 (all CPUs), got {n_jobs}")
    if local_refine_method not in ("auto", "nlp", "derivative-free"):
        raise ValueError(
            "local_refine_method must be 'auto', 'nlp' or 'derivative-free', "
            f"got {local_refine_method!r}"
        )

    from discopt.solver import _flat_var_box

    lb, ub = _flat_var_box(model)
    bad = ~(np.isfinite(lb) & np.isfinite(ub))
    if bad.any():
        names = _offending_variable_names(model, bad)
        raise ValueError(
            "solver='direct' requires a finite box on every variable; "
            f"non-finite bounds on: {', '.join(names)}. DIRECT partitions the box, "
            "so an infinite side has no midpoint and no centre-vertex distance. "
            "Add explicit bounds, e.g. m.continuous('x', lb=-10, ub=10)."
        )
    if np.any(ub < lb):
        names = _offending_variable_names(model, ub < lb)
        raise ValueError(f"lower bound exceeds upper bound on: {', '.join(names)}")

    n_vars = lb.size
    if n_vars > _DIMENSION_WARN_THRESHOLD:
        logger.warning(
            "DIRECT on %d variables: convergence degrades sharply with dimension "
            "(the literature reports good behaviour to ~6). Expect to need a large "
            "max_evals, and prefer an algebraic formulation if one exists.",
            n_vars,
        )

    oracle, n_from_model, integer_mask = _build_oracle(model, feasibility_tolerance)
    if n_from_model != n_vars:
        raise ValueError(f"variable-count mismatch: box has {n_vars}, evaluator has {n_from_model}")

    search = _DirectSearch(
        lb,
        ub,
        integer_mask=integer_mask,
        epsilon=epsilon,
        divide=divide,
        break_ties=break_ties,
        variant=direct_variant,
    )
    search.eps_cons = float(feasibility_tolerance)

    # Seed with the caller's start so the hybrid can never do worse than the
    # local-only path: it is a refinement candidate even if sampling never
    # improves on it.
    seed_value: Optional[float] = None
    seed_point: Optional[np.ndarray] = None
    if initial_point is not None:
        seed_point = np.clip(np.asarray(initial_point, dtype=np.float64).reshape(-1), lb, ub)
        f_seed, v_seed = oracle(seed_point)
        search.stats.evals += 1
        search._offer(seed_point, f_seed, v_seed)
        if v_seed <= search.eps_cons:
            seed_value = f_seed

    deadline = t_start + float(time_limit)
    stats = search.stats
    next_refine_at = local_refine_after if local_refine else np.inf
    last_local_value: Optional[float] = None

    def _maybe_refine(_s: "_DirectSearch") -> None:
        nonlocal next_refine_at, last_local_value
        if not local_refine or stats.evals < next_refine_at:
            return
        if search.best_feasible_point is None:
            next_refine_at = stats.evals + local_refine_after
            return
        # Best of {caller's start, DIRECT's incumbent} — on griewank_3 DIRECT's
        # incumbent was in a worse basin than the default start, so refining from
        # it alone lost to today's local-only behaviour.
        start = search.best_feasible_point
        best_val = search.best_feasible_value
        if seed_value is not None and best_val is not None and seed_value < best_val:
            assert seed_point is not None  # set together with seed_value
            start = seed_point
        remaining = max(0.0, deadline - time.perf_counter())
        before = search.best_feasible_value

        if local_refine_method in ("nlp", "auto"):
            refined = _refine_locally(
                model,
                start,
                integer_mask,
                min(local_refine_time_limit, remaining),
                nlp_solver,
                stats,
            )
            if refined is not None:
                value, point = refined
                fval, viol = oracle(point)
                search._offer(point, fval, viol)
                if last_local_value is None or value < last_local_value:
                    stats.local_improvements += 1
                last_local_value = value

        # A gradient method that did not move is the signature of a body that is
        # JAX-traceable but not usefully differentiable (jnp.round, a table
        # lookup, a pure_callback simulator). Powell does not care.
        gradient_stalled = (
            local_refine_method == "auto"
            and before is not None
            and search.best_feasible_value is not None
            and search.best_feasible_value >= before - 1e-12
        )
        if local_refine_method == "derivative-free" or gradient_stalled:
            budget_left = max(0, max_evals - stats.evals)
            if budget_left > 0:
                stats.derivative_free_refines += 1
                _refine_derivative_free(
                    search,
                    oracle,
                    search.best_feasible_point,
                    max_fev=min(budget_left, 200 * max(1, int((~integer_mask).sum()))),
                )
                if (
                    before is not None
                    and search.best_feasible_value is not None
                    and search.best_feasible_value < before - 1e-12
                ):
                    stats.local_improvements += 1

        # Re-trigger once sampling beats the last local solution (Jones 2001).
        next_refine_at = stats.evals + local_refine_after

    workers = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    if workers > 1:
        logger.info("DIRECT: evaluating each iteration's samples on %d threads", workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            search.run(
                oracle, max_evals, deadline=deadline, on_iteration=_maybe_refine, executor=pool
            )
    else:
        search.run(oracle, max_evals, deadline=deadline, on_iteration=_maybe_refine)

    # A final refinement so the reported point is never a raw centre when a local
    # solve was available and affordable.
    if local_refine and search.best_feasible_point is not None and time.perf_counter() < deadline:
        next_refine_at = 0
        _maybe_refine(search)

    wall = time.perf_counter() - t_start
    stats.rectangles = len(search.part)

    # ---- the result contract (CLAUDE.md §1) -------------------------------
    # DIRECT is a sampling method: it has NO dual information, so there is no
    # bound and no gap, and the status must never read as a proof. In particular
    # a budget that ran out without a feasible point is a LIMIT, never
    # "infeasible" — DIRECT cannot prove infeasibility.
    if search.best_feasible_point is None:
        status = "time_limit" if time.perf_counter() >= deadline else "iteration_limit"
        logger.info("DIRECT: no feasible point found in %d evaluations", stats.evals)
        return SolveResult(
            status=status,
            objective=None,
            bound=None,
            gap=None,
            x=None,
            wall_time=wall,
            node_count=stats.rectangles,
            gap_certified=False,
            solver_stats=stats.as_dict(),
        )

    hit_deadline = time.perf_counter() >= deadline and stats.evals < max_evals
    status = "time_limit" if hit_deadline else "feasible"
    best_value = search.best_feasible_value
    assert best_value is not None  # the no-incumbent case returned above
    return SolveResult(
        status=status,
        objective=float(best_value),
        bound=None,
        gap=None,
        x=_unpack_solution(model, np.asarray(search.best_feasible_point, dtype=np.float64)),
        wall_time=wall,
        node_count=stats.rectangles,
        gap_certified=False,
        solver_stats=stats.as_dict(),
    )


def _offending_variable_names(model: Model, bad_mask: np.ndarray) -> list[str]:
    """Names of the variables covering the flagged flat positions."""
    names: list[str] = []
    offset = 0
    for var in model._variables:
        size = var.size
        if bad_mask[offset : offset + size].any():
            names.append(var.name)
        offset += size
    return names or ["<unknown>"]
