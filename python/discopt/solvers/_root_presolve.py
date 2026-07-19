"""Root-node presolve helpers shared by global solvers."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from discopt.modeling.core import Model

logger = logging.getLogger(__name__)


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
# Activity-based big-M coefficient tightening (issue #282 / #774)
# ─────────────────────────────────────────────────────────────────────
#
# Standard MIP presolve (Savelsbergh 1994; Achterberg 2007): for a linear
# row ``Σ_{j≠k} a_j x_j + a_k y ⋈ b`` with ``y ∈ {0,1}`` binary and the rest
# of the activity bounded, the binary coefficient ``a_k`` can be shrunk toward
# the activity slack of the rest of the row **without changing the set of
# integer-feasible points** — the LP relaxation strictly tightens at fractional
# ``y`` while every integer corner is preserved exactly.
#
# Two cases, both derived for a normalised ``≤`` row (``≥`` rows are reflected):
#
#   u_rest = max activity of the rest (Σ_{j≠k} a_j x_j) over the current box.
#
#   * ``a_k > 0`` (Savelsbergh): slack = rhs − u_rest. If ``0 < slack < a_k``,
#     set ``a_k ← a_k − slack`` and ``rhs ← rhs − slack``. At y=0 the row becomes
#     ``rest ≤ u_rest`` (an implied, box-tight bound — equivalent, no point
#     changed); at y=1 it is unchanged.
#   * ``a_k < 0`` (fixed-charge ``flow − M·y ≤ b``, a_k = −M): set
#     ``a_k ← rhs − u_rest`` and keep ``rhs``, applied only when this raises
#     ``a_k`` (reduces the big-M) and keeps it negative. At y=0 the row is
#     unchanged; at y=1 the new row is the implied ``rest ≤ u_rest`` and the old
#     row ``rest ≤ rhs − a_k`` is *also* implied (the applied guard
#     ``rhs − u_rest > a_k`` ⇔ ``u_rest < rhs − a_k``), so the two are equivalent.
#
# ``u_rest`` uses FBBT+probing bounds that are valid over the whole feasible
# region (verified on rsyn0805m: the row whose declared box allowed rest≤7 has a
# true feasible max of 5.755, exactly the probed u_rest). The strengthened row is
# therefore valid over the root box and every descendant B&B node. Rows whose
# rest activity is not finite under those bounds are SKIPPED (never guessed).
#
# ── #770/#774 ROOT CAUSE + THE FIX ────────────────────────────────────
# The reverted #770 pass had SOUND math but an unsound **write-back**. It stored
# ``con.body = rebuild(a', const)`` (constant left embedded in the body) AND
# ``con.rhs = rhs_norm' + const`` (a now-nonzero rhs), splitting the constant
# across body and rhs simultaneously. Every constraint ``from_nl`` produces keeps
# the whole constant in the body with ``con.rhs == 0`` (verified: 214/214 rsyn0805m
# inequality rows), so that split is a normal form the relaxation/feasibility
# pipeline never sees natively — a downstream component reads only one of the two,
# shifting the enforced row by ``const`` and admitting a false primal (rsyn0805m
# reported 1441.99 > =opt= 1296.12, feasible in the mutated model but not the
# original). This pass instead writes back in the native normal form: the entire
# tightened constant goes into the body and ``con.rhs`` is set to 0, so the
# rewritten row is byte-for-byte the shape ``from_nl`` emits. (The mislabel in the
# original diagnosis — "fixed-charge sign error" — was wrong: fixed-charge only
# changes ``a_k`` and never touches rhs; it was the Savelsbergh ``rhs -= slack``
# path whose write-back split the constant.)


def coef_tighten_enabled() -> bool:
    """Whether ``DISCOPT_COEF_TIGHTEN`` opt-in flag is set (default OFF).

    Bound-changing presolve (CLAUDE.md §5): the strengthened LP relaxation is
    only smaller than the original and preserves the integer-feasible set
    exactly, so it is sound, but it stays default-OFF until a corpus-wide
    differential panel graduates it. The solver call site gates on the
    :attr:`~discopt.solver_tuning.SolverTuning.coef_tighten` field; this env
    helper is the standalone/legacy default for direct callers.
    """
    val = os.environ.get("DISCOPT_COEF_TIGHTEN", "0").strip().lower()
    return val not in ("", "0", "false", "off", "no")


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
    binary-indicator flows far more than bare FBBT alone; the extra activity
    slack it exposes is what the coefficient tightening spends. Because probing
    is the expensive part, this is called **once** per solve; the compounding
    rounds reuse the cheap direct-FBBT binding.

    The tightened bounds are read from the ``stats['fbbt']`` channel because the
    returned repr's ``var_ub``/``var_lb`` are *not* updated by the orchestrator
    (``propagate_bounds_to_model`` reads the same stale repr). All bounds
    returned are valid over the feasible region.
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
    bounds from the strengthened rows, unlocking a further round of tightening
    at negligible cost. FBBT bounds are valid over the feasible region.
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


def _is_binary_var(var, lb: float, ub: float) -> bool:
    """Whether ``var`` takes only ``{0,1}`` under the box ``[lb, ub]``.

    Binary-like ⟺ an *integer-typed* (``BINARY`` or ``INTEGER``) variable with a
    ``[0, 1]`` box. A **continuous** ``[0, 1]`` variable must NOT qualify: the
    coefficient tightening changes the row at fractional values, which preserves
    the feasible set only when the variable is genuinely integral.

    NB: a naive ``"IN" in str(vtype).upper()`` test is WRONG — ``"CONTINUOUS"``
    contains ``"IN"``, so it would wrongly classify a continuous ``[0, 1]``
    variable as binary (the latent unsoundness this guards, #774). Match the
    type *name* against ``BINARY``/``INTEGER`` instead.
    """
    vt = getattr(var, "vtype", getattr(var, "var_type", ""))
    name = str(getattr(vt, "name", vt)).upper()
    is_integral = ("BINARY" in name) or ("INTEGER" in name)
    return is_integral and lb == 0.0 and ub == 1.0


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
    """Activity-based big-M coefficient tightening on ``model`` (issue #282/#774).

    Rewrites linear constraint rows that contain a binary variable, shrinking the
    binary's coefficient toward the activity slack of the rest of the row (both
    the positive-coefficient Savelsbergh case and the negative-coefficient
    fixed-charge case). Iterates with FBBT bound tightening to a fixed point.

    The transformation is **feasible-set-equivalent**: it changes no
    integer-feasible point (it only shrinks the LP relaxation at fractional
    binaries), so it is sound. It mutates ``model._constraints`` in place and
    returns the number of rows rewritten across all rounds.

    The rewritten rows are emitted in ``from_nl``'s native normal form (whole
    constant in the body, ``con.rhs == 0``); see the module header for the
    #770/#774 write-back root cause this avoids.

    Scope (conservative, sound):
      * only linear inequality rows with ≥1 binary,
      * only rows whose non-binary worst-case activity is finite under the
        current (valid) FBBT/probing bounds — unbounded activity ⇒ SKIPPED,
      * scalar (size-1) variable blocks only.

    Gating is the caller's responsibility (``SolverTuning.coef_tighten`` /
    ``DISCOPT_COEF_TIGHTEN``); calling this directly always runs it.
    """
    blocks = model._variables
    # Only operate when every block is a scalar variable — the flat slot j then
    # equals block j, so bounds/coeffs line up without per-element offsets.
    if any(getattr(b, "size", 1) != 1 for b in blocks):
        return 0
    n = len(blocks)

    def is_binary(i: int, lb: float, ub: float) -> bool:
        return _is_binary_var(blocks[i], lb, ub)

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
        for con in model._constraints:
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
                continue  # unbounded rest activity — cannot tighten soundly
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
            # Write back in the NATIVE normal form (the #770/#774 fix): fold the
            # entire constant into the body and set ``con.rhs`` to 0, so the row
            # matches exactly the shape ``from_nl`` emits — never a body-constant
            # coexisting with a nonzero rhs (the malformed split that admitted the
            # #770 false primal). ``new_coeffs·x + new_const ⋈ 0`` reproduces the
            # tightened normalised ``a·x ≤ rhs`` exactly under either sense.
            new_coeffs = a * sgn
            new_const = -sgn * rhs
            con.body = _rebuild_linear_body(blocks, new_coeffs, new_const)
            con.rhs = 0.0
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
