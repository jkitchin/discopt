"""Integer-ratio partition bound for ratio-of-integer-products structure (#309).

The gear4-class pathology: a model constraint couples the objective to a
quotient ``q = (Π x_i)/(Π y_j)`` of bounded INTEGER variables. Every convex
relaxation of the quotient admits a fractional point hitting any target ratio
inside the box, so the node dual bound freezes (gear4: bound 0 for ~2.5k nodes
— every cheap lever was measured out in #309's comments). But the *achievable*
values of ``q`` form a sparse, exactly-enumerable rational set: products of
integers in the node box. Around the LP optimum's ratio there is a *hole* —
an open interval containing no achievable value — and partitioning ``q`` on
that hole is a valid disjunction that forces the LP off its frozen point.

This module implements that as a per-node bound side-channel (the "dive"):

1. Enumerate the achievable numerator/denominator product sets for the node's
   integer boxes (numpy, ~1 ms at gear4's root; hard-capped).
2. Solve a small worklist of piece LPs, each the model's own lifted McCormick
   relaxation with the quotient column's bounds intersected with a
   hole-partition interval. Piece LPs are built on the PRE-REFORM model (the
   division kept), so the disjunction is a plain column-bound change and the
   deviation reaches the objective undiluted by the cleared form's trilinear
   envelopes (measured: 0.455 diluted vs 1.643 undiluted at gear4's root).
3. The node bound is the min over all pieces — sound, because every feasible
   point of the node has ``q`` in the achievable set and each piece LP is a
   valid outer relaxation of the node restricted to its interval.

Soundness invariants:
- The achievable set is computed from the node's integer bounds rounded
  OUTWARD (`ceil(lb - tol)`, `floor(ub + tol)`) — a superset of the truly
  feasible factor values, so every hole is genuinely empty of feasible ratios.
- Hole endpoints are achievable ratios evaluated in float; each piece interval
  is padded outward by a relative ``_PAD`` (1e-12), dwarfing the half-ulp
  rounding of the float division while costing ~1e-6 of bound on gear4.
- Any failure (missing quotient column, LP error, cap exceeded, budget out)
  abstains: the caller keeps the engine's own bound. The dive can only ever
  *raise* a node bound via ``max(engine, dive)`` on a valid relaxation; when
  every piece is infeasible it abstains too (conservative — the engine's own
  relaxation is the fathoming authority).

Feature flag: ``DISCOPT_INTEGER_RATIO_PARTITION=1`` (default OFF; wired in
``MccormickLPRelaxer.solve_at_node`` and ``solver.py``). Bound-changing per
CLAUDE.md regime 2, so it ships default-off pending nightly graduation.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Integer-domain rounding slack for node bounds (branching produces exact
# integers; FBBT can leave float noise). Outward rounding only widens the
# achievable set — sound.
_INT_TOL = 1e-6
# Relative outward padding of piece intervals against float rounding of the
# achievable-ratio endpoints (a/b evaluated in double).
_PAD = 1e-12
# Achievability tolerance: an LP ratio point within this relative distance of
# an achievable value has no usable hole (piece is settled).
_ACH_TOL = 1e-9
# Enumeration caps: factor-range pair count per side, and detected specs per model.
_ENUM_CAP = 250_000
_MAX_SPECS = 4
# Piece-LP budget per node dive.
_MAX_LPS = 8
_PIECE_LP_TIME_LIMIT = 10.0


def enabled() -> bool:
    """Feature flag (default OFF)."""
    return os.environ.get("DISCOPT_INTEGER_RATIO_PARTITION") == "1"


@dataclass(frozen=True)
class IntegerRatioSpec:
    """One ratio-of-integer-products occurrence, in ORIGINAL flat columns.

    ``num``/``den`` carry multiplicity (``x**2`` lists its column twice) and are
    sorted, matching the ``uniform_relax`` ratio_map registration key.
    """

    num: tuple[int, ...]
    den: tuple[int, ...]

    @property
    def key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self.num, self.den)


def _iter_subexpressions(expr):
    """Yield every node of a modeling Expression tree (defensive walk)."""
    stack = [expr]
    seen: set[int] = set()
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        for attr in ("left", "right", "operand"):
            child = getattr(e, attr, None)
            if child is not None:
                stack.append(child)
        for attr in ("args", "terms"):
            children = getattr(e, attr, None)
            if children:
                stack.extend(children)


def detect_integer_ratio_specs(model) -> list[IntegerRatioSpec]:
    """Find ratio-of-integer-products occurrences eligible for the partition bound.

    Eligibility (all must hold; anything else is skipped, never guessed):
    - the quotient matches ``(c·Π x_i)/(Π y_j)`` per ``extract_ratio_of_products``;
    - every factor is a scalar INTEGER/BINARY variable with finite bounds;
    - numerator and denominator each involve at most 2 factors (v1 scope: the
      bilinear-integer class of #309; wider products need staged enumeration);
    - the denominator box is strictly positive (sign-definite quotient; the
      negative-definite case is deferred);
    - the root-box enumeration is within ``_ENUM_CAP`` per side.
    """
    from discopt._jax.term_classifier import extract_ratio_of_products
    from discopt.modeling.core import BinaryOp, VarType

    n_flat = sum(int(v.size) for v in model._variables)
    flat_lb = np.empty(n_flat)
    flat_ub = np.empty(n_flat)
    flat_int = np.zeros(n_flat, dtype=bool)
    off = 0
    for v in model._variables:
        sz = int(v.size)
        flat_lb[off : off + sz] = np.asarray(v.lb, dtype=np.float64).ravel()
        flat_ub[off : off + sz] = np.asarray(v.ub, dtype=np.float64).ravel()
        flat_int[off : off + sz] = v.var_type in (VarType.INTEGER, VarType.BINARY)
        off += sz

    exprs = []
    if model._objective is not None:
        exprs.append(model._objective.expression)
    exprs.extend(c.body for c in model._constraints)

    specs: dict[tuple, IntegerRatioSpec] = {}
    for root in exprs:
        for e in _iter_subexpressions(root):
            if not (isinstance(e, BinaryOp) and e.op == "/"):
                continue
            ratio = extract_ratio_of_products(e, model)
            if ratio is None:
                continue
            num_idx, den_idx = ratio
            cols = list(num_idx) + list(den_idx)
            if len(num_idx) > 2 or len(den_idx) > 2 or not den_idx:
                continue
            if not all(0 <= c < n_flat and flat_int[c] for c in cols):
                continue
            if not all(math.isfinite(flat_lb[c]) and math.isfinite(flat_ub[c]) for c in cols):
                continue
            if not all(flat_lb[c] > 0.0 for c in den_idx):
                continue
            if _enum_count(num_idx, flat_lb, flat_ub) > _ENUM_CAP:
                continue
            if _enum_count(den_idx, flat_lb, flat_ub) > _ENUM_CAP:
                continue
            spec = IntegerRatioSpec(
                num=tuple(sorted(int(c) for c in num_idx)),
                den=tuple(sorted(int(c) for c in den_idx)),
            )
            specs.setdefault(spec.key, spec)
            if len(specs) >= _MAX_SPECS:
                return list(specs.values())
    return list(specs.values())


def _enum_count(cols, flat_lb, flat_ub) -> float:
    n = 1.0
    for c in set(cols):
        n *= max(0.0, math.floor(flat_ub[c] + _INT_TOL) - math.ceil(flat_lb[c] - _INT_TOL) + 1)
    return n


def _product_values(cols: tuple[int, ...], ilo: np.ndarray, ihi: np.ndarray):
    """Sorted distinct achievable values of ``Π x_c`` over integer boxes, or None.

    ``cols`` may repeat a column (a squared factor uses the SAME variable twice:
    the achievable set is ``{v**2}``, not ``{v·w}``).
    """
    distinct = sorted(set(cols))
    ranges = {}
    count = 1
    for c in distinct:
        lo, hi = int(ilo[c]), int(ihi[c])
        if lo > hi:
            return None
        count *= hi - lo + 1
        if count > _ENUM_CAP:
            return None
        ranges[c] = np.arange(lo, hi + 1, dtype=np.int64)
    vals = None
    for c in cols:
        base = ranges[c]
        if vals is None:
            vals = base.copy()
        else:
            vals = np.unique(np.outer(vals, base).ravel())
            if vals.size > _ENUM_CAP:
                return None
    if cols and len(distinct) == 1 and len(cols) > 1:
        # exact power set for a repeated single variable: {v**k}, not {v·w}
        k = len(cols)
        vals = np.unique(ranges[distinct[0]] ** k)
    return np.unique(vals) if vals is not None else None


def _hole_around(num_vals: np.ndarray, den_vals: np.ndarray, q_star: float):
    """Largest empty open interval of achievable ratios containing ``q_star``.

    Returns ``(v_below, v_above)`` (either may be ±inf when no achievable value
    exists on that side) or ``None`` when ``q_star`` is itself achievable within
    ``_ACH_TOL`` (no usable hole).
    """
    v_below, v_above = -np.inf, np.inf
    t = q_star * den_vals.astype(np.float64)
    idx = np.searchsorted(num_vals, t)
    tol = _ACH_TOL * max(1.0, abs(q_star))
    for j_off in (-1, 0, 1):
        jj = idx + j_off
        ok = (jj >= 0) & (jj < len(num_vals))
        if not ok.any():
            continue
        v = num_vals[jj[ok]].astype(np.float64) / den_vals[ok].astype(np.float64)
        if (np.abs(v - q_star) <= tol).any():
            return None
        lo = v[v < q_star]
        hi = v[v > q_star]
        if lo.size:
            v_below = max(v_below, float(lo.max()))
        if hi.size:
            v_above = min(v_above, float(hi.min()))
    if not np.isfinite(v_below) and not np.isfinite(v_above):
        return None
    return v_below, v_above


class IntegerRatioPartitioner:
    """Per-node integer-ratio partition bound over a fixed pre-reform model.

    Built once at solve setup (``solver.py``) and attached to the node LP
    relaxer; ``node_bound`` is called per node with the node's box (in the
    SOLVE-TIME variable space — original columns first, reform aux after; the
    dive uses only the original slice).
    """

    def __init__(self, prereform_model, specs: list[IntegerRatioSpec]):
        from discopt._jax.discretization import DiscretizationState
        from discopt._jax.term_classifier import classify_nonlinear_terms

        self._model = prereform_model
        self._specs = list(specs)
        self._n_pre = sum(int(v.size) for v in prereform_model._variables)
        self._terms = classify_nonlinear_terms(prereform_model)
        self._disc = DiscretizationState(partitions={})
        self._stats = {"nodes": 0, "lifted": 0, "lps": 0, "abstained": 0}

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def node_bound(
        self,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
        *,
        deadline: Optional[float] = None,
    ) -> Optional[float]:
        """Sound lower bound on the node's objective from the ratio partition.

        Returns the best bound across the registered specs, or ``None`` to
        abstain (the caller keeps the engine bound). Never raises.
        """
        self._stats["nodes"] += 1
        best: Optional[float] = None
        try:
            lb = np.asarray(node_lb, dtype=np.float64).ravel()[: self._n_pre]
            ub = np.asarray(node_ub, dtype=np.float64).ravel()[: self._n_pre]
            if lb.size < self._n_pre or ub.size < self._n_pre:
                return None
            for spec in self._specs:
                b = self._dive(spec, lb, ub, deadline)
                if b is not None and (best is None or b > best):
                    best = b
        except Exception:
            logger.debug("integer-ratio dive abstained", exc_info=True)
            self._stats["abstained"] += 1
            return None
        if best is not None:
            self._stats["lifted"] += 1
        return best

    # -- internals --------------------------------------------------------- #

    def _dive(self, spec, lb, ub, deadline) -> Optional[float]:
        ilo = np.ceil(lb - _INT_TOL)
        ihi = np.floor(ub + _INT_TOL)
        num_vals = _product_values(spec.num, ilo, ihi)
        den_vals = _product_values(spec.den, ilo, ihi)
        if num_vals is None or den_vals is None or den_vals.size == 0 or num_vals.size == 0:
            return None
        if den_vals[0] <= 0:
            return None  # node box violated the sign gate; abstain
        lp_used = [0]

        def out_of_budget() -> bool:
            if lp_used[0] >= _MAX_LPS:
                return True
            return deadline is not None and time.perf_counter() >= deadline

        first = self._solve_piece(spec, lb, ub, -np.inf, np.inf, lp_used, deadline)
        if first is None:
            return None
        tag, b0, q0 = first
        if tag == "infeasible":
            return None  # conservative: leave infeasibility proofs to the engine
        # worklist of open pieces: (bound, q_lo, q_hi, lp_ratio_point)
        work = [(b0, -np.inf, np.inf, q0)]
        while work:
            work.sort(key=lambda item: item[0])
            _bnd, ql, qh, qs = work[0]
            hole = _hole_around(num_vals, den_vals, qs)
            if hole is None or out_of_budget():
                break  # loosest piece settled (or budget out): bound is final
            work.pop(0)
            v_below, v_above = hole
            pieces = []
            if np.isfinite(v_below):
                pieces.append((ql, v_below))
            if np.isfinite(v_above):
                pieces.append((v_above, qh))
            for pql, pqh in pieces:
                out = self._solve_piece(spec, lb, ub, pql, pqh, lp_used, deadline)
                if out is None:
                    return None  # abstain wholesale on any piece failure
                ptag, pb, pq = out
                if ptag == "infeasible":
                    continue
                work.append((pb, pql, pqh, pq))
        if not work:
            return None  # every piece infeasible: abstain (see module docstring)
        return float(min(item[0] for item in work))

    def _solve_piece(self, spec, lb, ub, ql, qh, lp_used, deadline=None):
        """Solve one piece LP; returns (tag, bound, ratio_point) or None."""
        from discopt._jax.milp_relaxation import build_milp_relaxation

        lp_used[0] += 1
        self._stats["lps"] += 1
        milp, varmap = build_milp_relaxation(
            self._model, self._terms, self._disc, bound_override=(lb, ub)
        )
        qcol = (varmap.get("ratio") or {}).get(spec.key)
        if qcol is None:
            return None
        bounds = np.asarray(milp._bounds, dtype=np.float64)
        if np.isfinite(ql):
            bounds[qcol, 0] = max(bounds[qcol, 0], ql - _PAD * max(1.0, abs(ql)))
        if np.isfinite(qh):
            bounds[qcol, 1] = min(bounds[qcol, 1], qh + _PAD * max(1.0, abs(qh)))
        if bounds[qcol, 0] > bounds[qcol, 1]:
            return ("infeasible", None, None)
        milp._bounds = bounds
        milp._integrality = None
        lp_limit = _PIECE_LP_TIME_LIMIT
        if deadline is not None:
            lp_limit = min(lp_limit, max(0.05, deadline - time.perf_counter()))
        res = milp.solve(time_limit=lp_limit, backend="simplex")
        if res.status == "infeasible":
            return ("infeasible", None, None)
        bound = getattr(res, "bound", None)
        x = getattr(res, "x", None)
        if res.status != "optimal" or bound is None or x is None or not math.isfinite(bound):
            return None
        return ("open", float(bound), float(np.asarray(x)[qcol]))
