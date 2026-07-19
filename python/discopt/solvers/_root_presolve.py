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
    only smaller than the original, never larger, so it is sound, but it stays
    default-OFF until a corpus-wide differential panel graduates it.
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
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Run root FBBT and integer-bound rounding for solver tree bounds.

    Returns ``(tightened_lb, tightened_ub, infeasible, changed)``. If the Rust
    FBBT binding is unavailable, this still performs sound integer rounding.
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

    if model_repr is not None:
        try:
            fbbt_lbs, fbbt_ubs = model_repr.fbbt(max_iter=max_iter, tol=tol)
            tightened_lb, tightened_ub = _flat_fbbt_bounds(
                model,
                np.asarray(fbbt_lbs, dtype=np.float64),
                np.asarray(fbbt_ubs, dtype=np.float64),
                tightened_lb,
                tightened_ub,
            )
        except Exception as exc:
            logger.debug("Root FBBT bound tightening skipped: %s", exc)

    _round_integral_bounds(tightened_lb, tightened_ub, int_offsets, int_sizes)

    infeasible = bool(np.any(tightened_lb > tightened_ub + tol))
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


def _extract_row(model: Model, con, n: int):
    """Return ``(coeffs, const)`` for a linear constraint body, or ``None``.

    ``coeffs`` is a dense length-``n`` float vector over flat scalar-variable
    slots; ``const`` is the free term. ``None`` when the body is non-linear or
    otherwise not extractable — such rows are left untouched.
    """
    from discopt._jax.problem_classifier import (  # local import: heavy _jax dep
        _extract_linear_coefficients,
        _NotLinearError,
    )

    try:
        coeffs, const = _extract_linear_coefficients(con.body, model, n)
    except _NotLinearError:
        return None
    except Exception as exc:  # pragma: no cover - defensive; skip unparsable rows
        logger.debug("coef-tighten: row extraction failed: %s", exc)
        return None
    return np.asarray(coeffs, dtype=np.float64), float(const)


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
        from discopt._jax.presolve_pipeline import run_root_presolve
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


def _rebuild_linear_body(blocks, coeffs: np.ndarray, const: float):
    """Rebuild ``Σ coeffs[j]·var_j (+const)`` as a modeling expression."""
    body = None
    nz = np.nonzero(np.abs(coeffs) > 1e-15)[0]
    for j in nz:
        term = float(coeffs[j]) * blocks[j]
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
    cases). Iterates with FBBT bound tightening to a fixed point: each round's
    tighter coefficients can tighten bounds, which enables further coefficient
    tightening.

    The transformation preserves the integer-feasible set EXACTLY — no feasible
    point removed AND no infeasible point admitted (see the section comment for
    the two-directional argument and the #772 post-mortem) — while the LP
    relaxation only shrinks. Rewritten rows are REPLACED as new ``Constraint``
    objects with the normalized ``rhs = 0.0`` invariant preserved. Returns the
    total number of rows whose coefficients were tightened across all rounds
    (0 when the flag is off or nothing tightens).

    Scope (conservative, sound):
      * only linear rows with an inequality sense and ≥1 binary,
      * only rows whose non-binary worst-case activity is finite under the
        current FBBT bounds (unbounded activity ⇒ skipped, not guessed),
      * scalar (size-1) variable blocks only.
    """
    if not coef_tighten_enabled():
        return 0

    blocks = model._variables
    # Only operate when every block is a scalar variable — the flat slot j then
    # equals block j, so bounds/coeffs line up without per-element offsets.
    if any(getattr(b, "size", 1) != 1 for b in blocks):
        return 0
    n = len(blocks)

    def is_binary(i: int, lb: float, ub: float) -> bool:
        # Match the DISCRETE types explicitly. (#770 tested ``"IN" in vtype`` —
        # which also matches "contINuous", so a continuous variable whose FBBT
        # bounds happen to be [0, 1] was treated as binary; the y∈{0,1}
        # equivalence argument then removes its genuinely feasible fractional
        # values. An integer variable with bounds [0, 1] IS binary-equivalent.)
        vt = str(getattr(blocks[i], "vtype", getattr(blocks[i], "var_type", ""))).upper()
        discrete = vt.endswith("BINARY") or vt.endswith("INTEGER")
        return discrete and lb == 0.0 and ub == 1.0

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
        lb, ub = fb
        round_changed = 0
        for ci, con in enumerate(model._constraints):
            sense = getattr(con, "sense", None)
            if sense not in ("<=", ">="):
                continue
            row = _extract_row(model, con, n)
            if row is None:
                continue
            coeffs, const = row
            # Normalise to ``a·x ≤ rhs`` (fold const into rhs; reflect ≥).
            a = coeffs.copy()
            rhs = float(con.rhs) - const
            sgn = 1.0
            if sense == ">=":
                a = -a
                rhs = -rhs
                sgn = -1.0
            nz = [int(j) for j in np.nonzero(np.abs(a) > 1e-12)[0]]
            bins = [j for j in nz if is_binary(j, lb[j], ub[j])]
            if not bins:
                continue
            # Worst-case (max) activity of each term over the FBBT box.
            term_max = {j: max(a[j] * lb[j], a[j] * ub[j]) for j in nz}
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
            # Reflect back to the original sense and REPLACE the constraint.
            # Two hard requirements from the #772 post-mortem (see the section
            # comment above):
            #   1. fold the ENTIRE tightened row into the body and keep
            #      ``rhs = 0.0`` — the normalized-form invariant every consumer
            #      assumes (writing the slack into ``con.rhs`` is silently
            #      dropped and RELAXES the row → false primal);
            #   2. build a NEW Constraint object so the identity-keyed
            #      evaluator-cache fingerprint changes and no stale compiled
            #      evaluator keeps serving the old row.
            # Normalized row is ``a·x ≤ rhs``; in original orientation the body
            # is ``sgn·(a·x − rhs)`` compared against 0 with the original sense.
            a_orig = a * sgn
            new_const = -rhs * sgn
            new_body = _rebuild_linear_body(blocks, a_orig, new_const)
            model._constraints[ci] = Constraint(
                new_body, con.sense, 0.0, getattr(con, "name", None)
            )
            round_changed += 1
        total_changed += round_changed
        if round_changed == 0:
            break

    if total_changed:
        logger.info(
            "Big-M coefficient tightening (DISCOPT_COEF_TIGHTEN): strengthened %d constraint rows",
            total_changed,
        )
    return total_changed
