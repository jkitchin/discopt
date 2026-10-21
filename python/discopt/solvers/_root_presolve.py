"""Root-node presolve helpers shared by global solvers."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from discopt.modeling.core import Constraint, Model

logger = logging.getLogger(__name__)


def coef_tighten_enabled() -> bool:
    """Whether ``DISCOPT_COEF_TIGHTEN`` opt-in flag is set (default OFF, #282).

    Bound-changing presolve (CLAUDE.md §5): the strengthened LP relaxation is
    only smaller than the original, never larger, so it is sound.

    **§5 state (#1414, 2026-09-21): KEPT AS A DOCUMENTED OPT-OUT.** The
    graduation panel was run, and the three-outcome rule needs an answer, so here
    it is with the numbers.

    *Why it is not the default.* BAR 1 (cert-clean) passes; BAR 2 (net-positive)
    does not. Measured on the in-repo corpus over the 92-instance population where
    ``tighten_bigm_coefficients`` actually fires (flag ON vs OFF, 60 s/instance):

      * node count — 95.0 % of scored instances neutral (±2 %), pooled ON/OFF
        ratio 1.0542, i.e. very slightly *worse*;
      * answer quality — 0 of 92 status transitions, no ``error`` rows in either
        arm; bound 4 tighter / 5 looser / 7 same, incumbent 3 better / 2 worse /
        2 equal. The bound/incumbent rows are single-rep and time-limited, so
        they measure progress at 60 s rather than relaxation strength; the status
        column is the robust part, and it is flat.

    A cert-clean but neutral flag does not graduate (the ``DISCOPT_CUT_INHERIT``
    lesson: sound is not the same as helpful).

    *Why it is not retired either.* It is the only mechanism in the tree that
    removes the #1380/#1414 big-M weakness at the source rather than coping with
    it: on ``min -x + 3z`` s.t. ``x <= M z``, ``x in [0, 10]``, ``z`` binary, it
    rewrites the row to ``x <= 10 z``, which makes the relaxation exact and leaves
    the search nothing to branch on (0 extra nodes at every ``M``). The default
    path now certifies that class on its own — #1414 fixed the root dive, the
    fathom-and-promote and the pinned-column trust that made it fail — but it
    does so by *branching*, and a user whose model is a wall of big-M indicator
    rows is exactly the case where paying presolve to delete that branching is
    the better trade. The corpus cannot see this because MINLPLib's big-M rows
    arrive already tightened by the modeller.

    *What would change it.* A differential panel on a population of
    **user-authored** big-M models (rows whose indicator coefficient exceeds the
    implied bound), showing the node-count win the repro shows. Wiring exists for
    the run: add an ``ARMS`` entry in ``generality_sweep.GRADUATION_ARMS`` so
    ``graduation_gate.py`` drives it.

    *Scope note (#1610 C-14, 2026-10-04).* The panel above predates two scope
    changes: the pass now reads rows over flat slots (array blocks and
    vector-valued bodies, which it previously declined outright), and it writes
    the discrete bounds its propagation proves (implied fixings) instead of
    discarding them. The in-repo corpus is all scalar ``.nl``, so the first
    change is invisible to it; a re-run of the graduation panel must include
    array-built models.

    Its Stage-2 verdict document (``docs/dev/issue-282-stage2-verdict.md``)
    reports ``syn40m`` +2608 → +1145 % and 52-62 rows tightened on the ``rsyn*``
    family, where the root barely moves. What that document FALSIFIES is
    **Stage 2** (Marchand-Wolsey VUB substitution plus a sustained aggregation
    loop), not Stage 1, which is this flag; do not read its banner as a verdict on
    coefficient tightening.
    """
    val = os.environ.get("DISCOPT_COEF_TIGHTEN", "0").strip().lower()
    return val not in ("", "0", "false", "off", "no")


def _round_integral_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
    int_offsets: list[int],
    int_sizes: list[int],
) -> None:
    """Round integer/binary flat bounds in-place."""
    for offset, size in zip(int_offsets, int_sizes):
        sl = slice(offset, offset + size)
        finite_lb = np.isfinite(lb[sl])
        finite_ub = np.isfinite(ub[sl])
        lb_view = lb[sl]
        ub_view = ub[sl]
        lb_view[finite_lb] = np.ceil(lb_view[finite_lb] - 1e-9)
        ub_view[finite_ub] = np.floor(ub_view[finite_ub] + 1e-9)


def _flat_fbbt_bounds(
    model: Model,
    fbbt_lbs: np.ndarray,
    fbbt_ubs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map block-level FBBT bounds to flat solver bounds."""
    tightened_lb = lb.copy()
    tightened_ub = ub.copy()

    if len(fbbt_lbs) != len(model._variables) or len(fbbt_ubs) != len(model._variables):
        return tightened_lb, tightened_ub

    offset = 0
    for block_idx, var in enumerate(model._variables):
        size = var.size
        if size != 1:
            offset += size
            continue
        block_lb = float(fbbt_lbs[block_idx])
        block_ub = float(fbbt_ubs[block_idx])
        sl = slice(offset, offset + size)
        if np.isfinite(block_lb):
            tightened_lb[sl] = np.maximum(tightened_lb[sl], block_lb)
        if np.isfinite(block_ub):
            tightened_ub[sl] = np.minimum(tightened_ub[sl], block_ub)
        offset += size

    return tightened_lb, tightened_ub


def tighten_root_bounds_with_fbbt(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    int_offsets: list[int],
    int_sizes: list[int],
    *,
    model_repr: Any | None = None,
    max_iter: int = 20,
    tol: float = 1e-8,
    time_limit_ms: int | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Run root FBBT and integer-bound rounding for solver tree bounds.

    Returns ``(tightened_lb, tightened_ub, infeasible, changed)``. If the Rust
    FBBT binding is unavailable, this still performs sound integer rounding.

    ``time_limit_ms`` caps the FBBT call's wall time (``None`` / 0 = unlimited).
    FBBT is anytime — it only ever tightens, and each constraint's inference is
    independently valid — so on expiry it returns the bounds contracted so far,
    a valid but looser box, and the integer rounding below still runs.

    The cap matters because this is the last step before tree creation and it has
    no other escape: on ``watercontamination0202`` (106,711 vars / 107,209 rows)
    ``max_iter=20`` full sweeps ran for **>10 minutes** against a 30 s solve
    budget, with the deadline unreachable (issue #863).
    """
    orig_lb = np.asarray(lb, dtype=np.float64)
    orig_ub = np.asarray(ub, dtype=np.float64)
    tightened_lb = orig_lb.copy()
    tightened_ub = orig_ub.copy()

    if model_repr is None:
        try:
            from discopt._rust import model_to_repr

            model_repr = model_to_repr(model, getattr(model, "_builder", None))
        except Exception as exc:
            logger.debug("Root FBBT model conversion skipped: %s", exc)
            model_repr = None

    fbbt_proved_empty = False
    if model_repr is not None:
        try:
            fbbt_lbs, fbbt_ubs = model_repr.fbbt(
                max_iter=max_iter, tol=tol, time_limit_ms=time_limit_ms
            )
            fbbt_lbs = np.asarray(fbbt_lbs, dtype=np.float64)
            fbbt_ubs = np.asarray(fbbt_ubs, dtype=np.float64)
            # #1852: the Rust pass reports a proven-empty box as the
            # ``Interval::empty()`` sentinel ``[+inf, -inf]`` on every block
            # (``fbbt_until`` marks all bounds empty once a row is violated past
            # its feasibility tolerance). ``_flat_fbbt_bounds`` copies finite
            # values only, so the proof was dropped: an infeasible model went on
            # to an NLP that cannot certify infeasibility, and ended ``unknown``.
            fbbt_proved_empty = bool(np.any(fbbt_lbs - fbbt_ubs > tol))
            tightened_lb, tightened_ub = _flat_fbbt_bounds(
                model,
                fbbt_lbs,
                fbbt_ubs,
                tightened_lb,
                tightened_ub,
            )
        except Exception as exc:
            logger.debug("Root FBBT bound tightening skipped: %s", exc)

    _round_integral_bounds(tightened_lb, tightened_ub, int_offsets, int_sizes)

    infeasible = fbbt_proved_empty or bool(np.any(tightened_lb > tightened_ub + tol))
    close = (tightened_lb > tightened_ub) & (tightened_lb <= tightened_ub + tol)
    if np.any(close):
        midpoint = 0.5 * (tightened_lb[close] + tightened_ub[close])
        tightened_lb[close] = midpoint
        tightened_ub[close] = midpoint

    changed = bool(np.any(tightened_lb > orig_lb + tol) or np.any(tightened_ub < orig_ub - tol))
    return tightened_lb, tightened_ub, infeasible, changed


# ─────────────────────────────────────────────────────────────────────
# Activity-based big-M coefficient tightening (issue #282, Stage 1)
# ─────────────────────────────────────────────────────────────────────
#
# Standard MIP presolve (Savelsbergh 1994; Achterberg 2007): for a linear
# row ``Σ_{j≠k} a_j x_j + a_k y ⋈ b`` with ``y ∈ {0,1}`` binary and the rest
# of the activity bounded, the coefficient ``a_k`` can be shrunk toward the
# activity slack without removing any *integer-feasible* point, while the LP
# relaxation strictly tightens at fractional ``y``.
#
# Two cases, both derived for a normalised ``≤`` row (``≥`` rows are reflected):
#
#   Umax = max activity of the rest (Σ_{j≠k} a_j x_j) over the current box.
#
#   * ``a_k > 0``  (Savelsbergh):  slack = b − Umax. If ``0 < slack < a_k``,
#     set ``a_k ← a_k − slack`` and ``b ← b − slack``. At y=0 the row becomes
#     ``rest ≤ Umax`` (implied — no point removed); at y=1 it is unchanged.
#   * ``a_k < 0``  (fixed-charge ``flow ≤ M·y`` ⇒ ``flow − M·y ≤ 0``, a_k=−M):
#     set ``a_k ← b − Umax`` and keep ``b``, applied only when this raises
#     ``a_k`` (reduces the big-M) and keeps it negative. At y=0 the row is
#     unchanged; at y=1 both old and new rows are implied by ``rest ≤ Umax``
#     (no point removed); at fractional y the RHS ``b + |a_k|·y`` shrinks.
#
# Feasible-set EQUIVALENCE (both directions — the #772 lesson):
#   * no integer-feasible point removed: at y ∈ {0,1} the new row is implied by
#     (old row + activity bounds), and the activity bounds hold at every point
#     that is feasible in the original problem with integral binaries (FBBT /
#     implied-bounds / probing inferences are valid exactly on that set);
#   * no point admitted: for y ∈ [0, 1] the new row is pointwise at least as
#     tight as the old one (a_k>0 case: the row shifts by ``slack·(1−y) ≥ 0``;
#     a_k<0 case: the y-coefficient only increases), so the rewritten model's
#     feasible set — even its continuous relaxation over the box — is a subset
#     of the original's.
#
# The activity bound ``Umax`` uses FBBT-tightened bounds (valid over the whole
# feasible region), so the strengthened row is globally valid and remains valid
# at every descendant B&B node (child boxes are subsets of the root box). This
# is exactly what SCIP's presolve does; on the #282 convex panel it is the
# load-bearing root-gap lever (syn40m root excess +2608% → +1145%).
#
# ── #772 POST-MORTEM (why the #770 version produced a FALSE PRIMAL) ──────────
# The #770 math above was valid, but the WRITE-BACK broke two model invariants:
#
#   1. ``Constraint.rhs`` is **always 0.0** in normalized form (documented on the
#      dataclass; every comparison operator constructs ``rhs=0.0``). Consumers —
#      ``NLPEvaluator``, the relaxation compilers, ``_infer_constraint_bounds``
#      (which derives cl/cu from the *sense alone*) — therefore compile the body
#      and test it against 0. #770 moved the Savelsbergh slack into ``con.rhs``
#      (0 → −slack), which every consumer silently dropped: the rewritten row was
#      read as ``body' ≤ 0`` instead of ``body' ≤ −slack`` — a RELAXATION by
#      ``slack`` — admitting integer points infeasible in the original problem
#      (rsyn0805m returned obj 1441.99 > opt 1296.12). The fix: fold the entire
#      tightened row into the body (``body' = sgn·(a·x − rhs)``) and keep
#      ``rhs = 0.0``.
#   2. The evaluator cache (``evaluator_fingerprint``) is keyed on constraint
#      *object identity*, so in-place mutation of ``con.body`` leaves previously
#      compiled evaluators serving the un-tightened rows. The fix: REPLACE each
#      rewritten ``Constraint`` with a new object, which invalidates the
#      fingerprint and forces a consistent re-compile. (The #779 final-incumbent
#      guard intentionally keeps its own pre-presolve snapshot reference.)
#
# NOTE ON LOCATION (why Python, not the Rust ``coefficient_strengthening`` pass):
# the in-tree relaxation LP is compiled from the *Python* model DAG plus
# separately-computed FBBT bound arrays. The Rust presolve orchestrator's
# rewritten constraint bodies are never propagated back to that DAG
# (``propagate_bounds_to_model`` copies bounds only), and the existing Rust
# pass additionally (a) reads *declared* bounds — so it bails on the ``[0,∞)``
# flows this family declares — and (b) skips negative (fixed-charge) binary
# coefficients. Rewriting the Python model at the root is the only place the
# tightened coefficients actually reach the relaxation.


def _strong_block_bounds(model: Model, time_limit_ms: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Block-aligned tightened bounds from the root presolve orchestrator.

    Runs FBBT + implied-bounds + probing + simplify (the full bound-tightening
    orchestrator, but with the *variable-eliminating* passes disabled) so the
    returned per-block arrays stay aligned 1:1 with ``model._variables`` — no
    elimination/aggregation remaps the index space. Probing tightens the
    binary-indicator flows far more than bare FBBT alone, and the extra activity
    slack it exposes is what the coefficient tightening spends: on syn40m probing
    takes the root excess to ~+1145% (vs bare FBBT's ~+2200%). Because probing
    is the expensive part, this is called **once** per solve; the compounding
    rounds reuse the cheap direct-FBBT binding.

    The tightened bounds are read from the ``stats['fbbt']`` channel because the
    returned repr's ``var_ub``/``var_lb`` are *not* updated by the orchestrator
    (a known gap — ``propagate_bounds_to_model`` reads the same stale repr, so
    the solver's own root FBBT tightening reaches only the tree, not this repr).
    """
    try:
        from discopt._relax.presolve_pipeline import run_root_presolve
        from discopt._rust import model_to_repr

        repr0 = model_to_repr(model, getattr(model, "_builder", None))
        _, stats = run_root_presolve(
            repr0,
            eliminate=False,
            aggregate=False,
            factorable_elim=False,
            redundancy=False,
            polynomial=False,
            fbbt=True,
            implied_bounds=True,
            probing=True,
            simplify=True,
            max_iterations=8,
            time_limit_ms=time_limit_ms,
        )
        fb = stats.get("fbbt")
        if not fb:
            return None
        lbs = np.asarray(fb["lb"], dtype=np.float64)
        ubs = np.asarray(fb["ub"], dtype=np.float64)
    except Exception as exc:
        logger.debug("coef-tighten: strong bound tightening unavailable: %s", exc)
        return None
    nblk = len(model._variables)
    if lbs.size < nblk or ubs.size < nblk:
        return None
    return lbs[:nblk], ubs[:nblk]


def _cheap_block_bounds(model: Model) -> tuple[np.ndarray, np.ndarray] | None:
    """Block-aligned bounds from the bare FBBT binding (fast, no probing).

    Used for the compounding rounds after the one-shot probing pass: once the
    big-M coefficients have been shrunk, plain FBBT re-derives tighter flow
    bounds from the strengthened rows, which unlocks a further round of
    coefficient tightening — at negligible cost (~0.01–0.05 s vs probing's ~2–5 s).
    """
    try:
        from discopt._rust import model_to_repr

        repr_ = model_to_repr(model, getattr(model, "_builder", None))
        lbs, ubs = repr_.fbbt(max_iter=20, tol=1e-9)
    except Exception as exc:
        logger.debug("coef-tighten: cheap FBBT unavailable: %s", exc)
        return None
    lbs = np.asarray(lbs, dtype=np.float64)
    ubs = np.asarray(ubs, dtype=np.float64)
    if lbs.size != len(model._variables) or ubs.size != len(model._variables):
        return None
    return lbs, ubs


class _FlatLayout:
    """Flat scalar-slot view of a model's variable blocks (#1610 C-14).

    Slot ``j`` is element ``j - offset[b]`` (row-major) of block ``b``, the same
    order ``_extract_linear_coefficients_sparse``, ``_extract_variable_info`` and
    the tree use. Before #1610 the pass declined on any model with a ``size > 1``
    block, so a model written with ``shape=(T,)`` variables got no tightening at
    all while the identical scalar model did.
    """

    def __init__(self, model: Model) -> None:
        from discopt.modeling.core import VarType

        self.blocks = list(model._variables)
        sizes = [int(b.size) for b in self.blocks]
        self.offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
        self.n = int(self.offsets[-1])
        self.block_of = np.repeat(np.arange(len(self.blocks), dtype=np.int64), sizes)
        if self.n:
            self.lb = np.concatenate(
                [np.asarray(b.lb, dtype=np.float64).ravel() for b in self.blocks]
            )
            self.ub = np.concatenate(
                [np.asarray(b.ub, dtype=np.float64).ravel() for b in self.blocks]
            )
        else:
            self.lb = np.zeros(0)
            self.ub = np.zeros(0)
        # Match the DISCRETE types explicitly. (#770 tested ``"IN" in vtype`` —
        # which also matches "contINuous", so a continuous variable whose bounds
        # happen to be [0, 1] was treated as binary.)
        disc = [b.var_type in (VarType.BINARY, VarType.INTEGER) for b in self.blocks]
        self.discrete = np.repeat(np.asarray(disc, dtype=bool), sizes)

    def slot_expr(self, j: int):
        """The modeling expression for flat slot ``j`` (a scalar variable or ``x[i]``)."""
        b = int(self.block_of[j])
        var = self.blocks[b]
        if var.size == 1 and var.shape in ((), (1,)):
            return var if var.shape == () else var[0]
        local = j - int(self.offsets[b])
        idx = np.unravel_index(local, var.shape)
        if len(idx) == 1:
            return var[int(idx[0])]
        return var[tuple(int(i) for i in idx)]

    def block_hull_to_flat(self, blk_lb: np.ndarray, blk_ub: np.ndarray):
        """Broadcast block-level bounds to every element of the block.

        Rust FBBT carries one interval per block, seeded from the element-wise
        union of the block's bounds (C-31), so it is a valid outer bound for
        every element; intersecting it with the element bounds is sound.
        """
        return blk_lb[self.block_of], blk_ub[self.block_of]


class _LinRow:
    """One scalar linear row ``Σ terms[j]·x_j + const ⋈ 0`` and where it came from."""

    __slots__ = ("ci", "elem", "n_elem", "terms", "const", "sense", "name", "changed")

    def __init__(self, ci, elem, n_elem, terms, const, sense, name):
        self.ci = ci
        self.elem = elem
        self.n_elem = n_elem
        self.terms = terms
        self.const = const
        self.sense = sense
        self.name = name
        self.changed = False


def _collect_linear_rows(model: Model, n: int) -> list[_LinRow]:
    """Every scalar linear row of ``model``, fanning vector-valued bodies out per element.

    A body the scalar extractor refuses because it is vector-valued
    (``x <= 1000 * y`` over ``shape=(T,)`` blocks) is expanded with
    :func:`discopt.export._arrays.scalarize_body` — the row-major expansion the
    ``.nl`` writer and the AD tape use — and each element is extracted on its
    own. Non-linear elements are skipped, never guessed.
    """
    from discopt._relax.problem_classifier import (  # local import: heavy _relax dep
        _extract_linear_coefficients_sparse,
        _NotLinearError,
    )
    from discopt.export._arrays import needs_scalarize, scalarize_body

    rows: list[_LinRow] = []
    for ci, con in enumerate(model._constraints):
        sense = getattr(con, "sense", None)
        if sense not in ("<=", ">=", "=="):
            continue
        name = getattr(con, "name", None)
        try:
            terms, const = _extract_linear_coefficients_sparse(con.body, model, n)
        except _NotLinearError:
            terms = None
        if terms is not None:
            rows.append(_LinRow(ci, 0, 1, dict(terms), float(const), sense, name))
            continue
        if not needs_scalarize(con.body):
            continue  # genuinely non-linear scalar body
        elems = scalarize_body(con.body)
        for e_idx, e_body in enumerate(elems):
            try:
                terms, const = _extract_linear_coefficients_sparse(e_body, model, n)
            except _NotLinearError:
                continue
            rows.append(_LinRow(ci, e_idx, len(elems), dict(terms), float(const), sense, name))
    return rows


def _propagate_linear_rows(
    rows: list[_LinRow],
    lb: np.ndarray,
    ub: np.ndarray,
    discrete: np.ndarray,
    *,
    max_passes: int = 10,
    int_tol: float = 1e-6,
) -> bool:
    """Per-element activity bound propagation over ``rows``, in place; True if infeasible.

    Rust FBBT holds one interval per *block*, so on a ``shape=(T,)`` block it can
    only report the hull of the elements (``x[3] <= 20`` is invisible when the
    other elements are capped at 40). This is the textbook single-row
    propagation (Savelsbergh 1994; Achterberg 2007 §7.1) on flat slots, with
    integer rounding on discrete slots — the rounding is what turns
    ``x1 >= 2, x1 <= 40·y1`` into the implied fixing ``y1 = 1`` (#1610 C-14).

    Every derived bound is relaxed outward by ``1e-9·max(1, |v|)`` before it is
    used, and discrete bounds round with ``int_tol`` slack, so floating-point
    error can only loosen, never cut.
    """
    for _ in range(max_passes):
        any_change = False
        for row in rows:
            items = [(j, a) for j, a in row.terms.items() if abs(a) > 1e-12]
            if not items:
                continue
            # Normalised forms ``a·x <= rhs``; an equality contributes both.
            forms: list[tuple[float, float]] = []
            if row.sense in ("<=", "=="):
                forms.append((1.0, -row.const))
            if row.sense in (">=", "=="):
                forms.append((-1.0, row.const))
            for sgn, rhs in forms:
                # Minimum activity, counting -inf contributions separately so a
                # single unbounded term still lets the other terms be bounded.
                mins: list[float | None] = []
                n_inf = 0
                fin_sum = 0.0
                for j, a in items:
                    a = sgn * a
                    v = a * lb[j] if a > 0 else a * ub[j]
                    if not np.isfinite(v):
                        n_inf += 1
                        mins.append(None)
                    else:
                        fin_sum += v
                        mins.append(v)
                if n_inf > 1:
                    continue
                for (j, a0), mj in zip(items, mins):
                    if n_inf == 1 and mj is not None:
                        continue  # the residual still contains the unbounded term
                    resid = fin_sum - (mj if mj is not None else 0.0)
                    a = sgn * a0
                    bound = (rhs - resid) / a
                    if not np.isfinite(bound):
                        continue
                    pad = 1e-9 * max(1.0, abs(bound))
                    if a > 0:
                        new_ub = bound + pad
                        if discrete[j]:
                            new_ub = float(np.floor(new_ub + int_tol))
                        if new_ub < ub[j] - 1e-9 * max(1.0, abs(ub[j])):
                            ub[j] = new_ub
                            any_change = True
                    else:
                        new_lb = bound - pad
                        if discrete[j]:
                            new_lb = float(np.ceil(new_lb - int_tol))
                        if new_lb > lb[j] + 1e-9 * max(1.0, abs(lb[j])):
                            lb[j] = new_lb
                            any_change = True
                    if lb[j] > ub[j] + int_tol:
                        return True
        if not any_change:
            break
    return False


def _rebuild_linear_body(layout: _FlatLayout, terms: dict[int, float], const: float):
    """Rebuild ``Σ terms[j]·x_j (+const)`` as a modeling expression over flat slots."""
    body = None
    for j in sorted(terms):
        c = float(terms[j])
        if abs(c) <= 1e-15:
            continue
        term = c * layout.slot_expr(j)
        body = term if body is None else body + term
    if abs(const) > 1e-15:
        body = float(const) if body is None else body + float(const)
    if body is None:
        body = 0.0
    return body


def tighten_bigm_coefficients(
    model: Model,
    *,
    max_rounds: int = 6,
    tol: float = 1e-7,
    probe_time_ms: int = 5000,
) -> int:
    """Activity-based big-M coefficient tightening on ``model`` (issue #282).

    Rewrites linear constraint rows that contain a binary variable, shrinking
    the binary's coefficient toward the activity slack of the rest of the row
    (both positive-coefficient Savelsbergh and negative-coefficient fixed-charge
    cases). Iterates with bound tightening to a fixed point: each round's
    tighter coefficients can tighten bounds, which enables further coefficient
    tightening.

    The transformation preserves the integer-feasible set EXACTLY — no feasible
    point removed AND no infeasible point admitted (see the section comment for
    the two-directional argument and the #772 post-mortem) — while the LP
    relaxation only shrinks. Rewritten rows are REPLACED as new ``Constraint``
    objects with the normalized ``rhs = 0.0`` invariant preserved. Returns the
    total number of rows whose coefficients were tightened across all rounds
    (0 when the flag is off or nothing tightens).

    #1610 C-14 widened the scope:

      * **Array blocks.** Rows are read over flat scalar slots, so a model built
        from ``shape=(T,)`` variables is tightened exactly like its scalar twin.
        A scalar row over ``x[t]`` is replaced in place; a vector-valued body
        (``x <= 1000 * y``) is fanned out per element, and each tightened element
        is APPENDED as its own scalar row — the original vector row stays (it is
        implied by the appended one), so no constraint index shifts.
      * **Per-element bounds.** Rust FBBT is block-level, so its bounds are
        intersected with a per-element linear propagation
        (:func:`_propagate_linear_rows`).
      * **Implied fixings.** Discrete bounds the propagation (or root probing)
        proves — e.g. ``y1 = 1`` from ``x1 >= 2`` and ``x1 <= 40·y1`` — are
        written to the variable bounds, so they reach the tree instead of being
        computed and discarded. The write lands on the solve's model, which
        ``_solve_owns_model`` restores on exit (#1610 C2), so the caller's
        declared bounds are untouched.

    Scope (conservative, sound):
      * only linear rows with an inequality sense and ≥1 binary are rewritten,
      * only rows whose worst-case activity is finite under the current bounds
        (unbounded activity ⇒ skipped, not guessed),
      * only discrete bounds are written back; continuous bounds stay internal.
    """
    if not coef_tighten_enabled():
        return 0

    layout = _FlatLayout(model)
    n = layout.n
    if n == 0 or not np.any(layout.discrete):
        return 0
    rows = _collect_linear_rows(model, n)
    if not any(any(layout.discrete[j] for j in r.terms) for r in rows):
        return 0

    flat_lb = layout.lb.copy()
    flat_ub = layout.ub.copy()

    total_changed = 0
    for _round in range(max_rounds):
        # One expensive probing pass for the first round (it exposes the most
        # activity slack); cheap bare-FBBT for the compounding rounds.
        fb = (
            _strong_block_bounds(model, probe_time_ms)
            if _round == 0
            else _cheap_block_bounds(model)
        )
        if fb is None:
            break
        h_lb, h_ub = layout.block_hull_to_flat(*fb)
        new_lb = np.maximum(flat_lb, h_lb)
        new_ub = np.minimum(flat_ub, h_ub)
        new_lb[layout.discrete] = np.ceil(new_lb[layout.discrete] - 1e-6)
        new_ub[layout.discrete] = np.floor(new_ub[layout.discrete] + 1e-6)
        if np.any(new_lb > new_ub + 1e-6):
            break  # infeasible box: leave it to the solver to report
        if _propagate_linear_rows(rows, new_lb, new_ub, layout.discrete):
            break
        flat_lb, flat_ub = new_lb, new_ub
        lb, ub = flat_lb, flat_ub

        round_changed = 0
        for row in rows:
            if row.sense not in ("<=", ">="):
                continue
            # Normalise to ``a·x ≤ rhs`` (fold const into rhs; reflect ≥).
            sgn = -1.0 if row.sense == ">=" else 1.0
            a = {j: sgn * c for j, c in row.terms.items() if abs(c) > 1e-12}
            rhs = -sgn * row.const
            # An integer variable with bounds [0, 1] IS binary-equivalent.
            bins = [j for j in a if layout.discrete[j] and lb[j] == 0.0 and ub[j] == 1.0]
            if not bins:
                continue
            # Worst-case (max) activity of each term over the box.
            term_max = {j: max(a[j] * lb[j], a[j] * ub[j]) for j in a}
            if not all(np.isfinite(v) for v in term_max.values()):
                continue  # unbounded rest activity — cannot tighten yet
            total_max = float(sum(term_max.values()))
            row_changed = False
            for k in bins:
                ak = a[k]
                u_rest = total_max - term_max[k]  # max activity of the rest (y_k excluded)
                if ak > tol:
                    slack = rhs - u_rest
                    if slack > tol and slack + tol < ak:
                        a[k] = ak - slack
                        rhs -= slack
                        row_changed = True
                elif ak < -tol:
                    new_ak = rhs - u_rest
                    if new_ak > ak + tol and new_ak < -tol:
                        a[k] = new_ak
                        row_changed = True
                else:
                    continue
                term_max[k] = max(a[k] * lb[k], a[k] * ub[k])
                total_max = u_rest + term_max[k]
            if not row_changed:
                continue
            # Back to the original orientation: ``sgn·(a·x − rhs) ⋈ 0``.
            row.terms = {j: sgn * c for j, c in a.items()}
            row.const = -sgn * rhs
            row.changed = True
            round_changed += 1
            if row.n_elem == 1:
                # Two hard requirements from the #772 post-mortem (see the
                # section comment above): fold the ENTIRE tightened row into the
                # body with ``rhs = 0.0``, and build a NEW Constraint object so
                # the identity-keyed evaluator-cache fingerprint changes.
                old = model._constraints[row.ci]
                model._constraints[row.ci] = Constraint(
                    _rebuild_linear_body(layout, row.terms, row.const),
                    old.sense,
                    0.0,
                    row.name,
                )
        total_changed += round_changed
        if round_changed == 0:
            break

    # Vector-valued rows: append each tightened element as its own scalar row.
    # The original vector row stays in place (implied by these), so constraint
    # indices are unchanged; ``_solve_owns_model`` removes the additions on exit.
    for row in rows:
        if row.changed and row.n_elem > 1:
            model._constraints.append(
                Constraint(
                    _rebuild_linear_body(layout, row.terms, row.const),
                    row.sense,
                    0.0,
                    None if row.name is None else f"{row.name}[{row.elem}]",
                )
            )

    # Implied fixings: write the discrete bounds proven above to the variables.
    n_fixed = 0
    disc_idx = np.nonzero(layout.discrete)[0]
    tighter = disc_idx[
        (flat_lb[disc_idx] > layout.lb[disc_idx]) | (flat_ub[disc_idx] < layout.ub[disc_idx])
    ]
    if tighter.size:
        for b in sorted({int(layout.block_of[j]) for j in tighter}):
            var = layout.blocks[b]
            sl = slice(int(layout.offsets[b]), int(layout.offsets[b + 1]))
            new_lb = np.maximum(layout.lb[sl], flat_lb[sl])
            new_ub = np.minimum(layout.ub[sl], flat_ub[sl])
            var.lb = new_lb.reshape(var.shape)
            var.ub = new_ub.reshape(var.shape)
        n_fixed = int(tighter.size)

    if total_changed or n_fixed:
        logger.info(
            "Big-M coefficient tightening (DISCOPT_COEF_TIGHTEN): strengthened %d "
            "constraint rows, tightened %d discrete bounds",
            total_changed,
            n_fixed,
        )
    return total_changed
