"""The single, scale-aware incumbent feasibility verifier.

Why this module exists
----------------------
Every certificate this solver issues rests on an *independently verified* primal
point: the native kernel refuses to seed its cutoff from an unverified point, the
convex kernel refuses to adopt one, and the Regime-C differential panels re-verify
every returned incumbent against a freshly parsed model. Before this module those
three consumers each carried their own hand-rolled loop over ``model._constraints``,
and the loops disagreed with the evaluator about which row is which.

Four defects were measured on the tree of 2026-07-29 (probe transcripts in the
consolidation plan's §6 entry "the incumbent verifier's tolerance is scale-blind"):

1. **Scale-blind tolerance (the named defect).** The tolerance was written
   ``abs_tol + rel_tol * abs(residual)`` — keyed on the *residual*, i.e. on the very
   quantity being judged. Solve it: an equality row passes iff
   ``|r| <= abs + rel*|r|`` iff ``|r| <= abs/(1 - rel)``. With ``abs=1e-6,
   rel=1e-4`` that is ``1.0001e-6`` — a pure **absolute** 1e-6 no matter how large
   the row is, and the ``rel_tol`` term is arithmetically dead. Consequence measured
   on ``nvs22``: a certified optimum (objective matching ``=opt= 6.05822`` to 5.7e-8)
   is REJECTED on two defined-variable equality rows whose residuals are 1.71e-5 and
   2.64e-4 against row scales 2.1e3 and 1.7e4 — relative residuals 8.1e-9 and 1.5e-8.
2. **Row misalignment.** Both verifiers advanced their row index once per
   *constraint object* while ``NLPEvaluator.evaluate_constraints`` emits one row per
   *flat element*. A model with a single vector constraint of size 3 had rows 1 and 2
   never examined: a point violating row 2 by 5.0 was **accepted as feasible**.
3. **Builder-resident rows were invisible.** Iterating ``model._constraints`` misses
   the rows that ``add_linear_constraints`` / the linear fast path put only in the
   Rust builder. ``NLPEvaluator._source_constraints`` already unions them in (the
   X-1 fix); iterating the evaluator's own row map inherits that coverage.
4. **``Constraint.rhs`` was ignored** (the body was compared against 0), and
   non-``Constraint`` entries (SOS / logical / indicator / disjunctive) were either
   silently skipped or raised ``AttributeError`` mid-loop.

This module is the fix for all four, and it is **strictly stricter** everywhere
except the one place the row scale genuinely belongs — see "Is this looser?" below.

The tolerance
-------------
For row ``i`` with body ``g_i(x)``, right-hand side ``b_i`` and sense ``s``::

    violation_i = max(g_i - b_i, 0)      s is "<="
                  max(b_i - g_i, 0)      s is ">="
                  |g_i - b_i|            s is "=="

    scale_i     = max(1, |b_i|, max_j |J_ij| * max(1, |x_j|))

    accept_i   <=>  violation_i <= abs_tol * scale_i

``scale_i`` is the examiner's row scale (``validation/examiner.py``, "Examiner's
scaled mode"), reused rather than reinvented per §0.8 of the consolidation plan.
Its meaning is a *displacement*: ``violation_i / scale_i`` is, to first order, the
relative move in the variables needed to satisfy the row. So the acceptance rule
reads "a genuinely feasible point exists within relative distance ``abs_tol`` of
this one", which is what a primal certificate is claiming.

Three properties of ``scale_i`` are load-bearing and each kills a *naive* widening
that would have accepted a bad point (regression tests in
``python/tests/test_incumbent_verifier_scale.py``):

* it is **row-local** — a large variable elsewhere in the model cannot widen this
  row's tolerance (a model-global ``max_j |x_j|`` scale accepts a unit row violated
  by 1e-2 in a model containing a 1e9 variable);
* it is the **infinity norm** over linearised terms, not the sum — a row with a
  million cancelling unit terms keeps a unit tolerance (a 1-norm scale accepts a
  0.5 violation on it);
* it is **floored at 1**, so on an ordinary unit-scale row the tolerance is exactly
  ``abs_tol`` — bit-for-bit the pre-existing behaviour.

Is this looser?
---------------
Only on rows whose own magnitude exceeds 1, and there only in proportion to that
magnitude. The relative coefficient is ``abs_tol`` itself (1e-6), **not** the repo's
``rel_tol`` (1e-4): reusing ``rel_tol`` here would have made the floor 1.01e-4 and
loosened every unit-scale row by 100x. The same function's *variable-bound* check
has used ``abs_tol + rel_tol * |bound|`` — a scale term with the 100x looser
coefficient — since #764; this makes the row check consistent with it and tighter.

Against that bounded widening the module closes four wrongly-**accept** holes
(defects 2, 3, 4 and the missing bounds/integrality checks on the convex-kernel
path). It refuses rather than guesses whenever it cannot vouch: an evaluator
failure, a non-finite value, an unknown sense, a constraint class the evaluator does
not cover, or a Jacobian that cannot be formed all return "not verified". In the
Jacobian-unavailable case the scale falls back to ``max(1, |b_i|)``, which is the
*stricter* direction — a verifier that cannot measure a row's scale must not widen
for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

#: Repo-conventional tolerances (CLAUDE.md "Numerical tolerances").
DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-4
DEFAULT_INT_TOL = 1e-5

__all__ = [
    "DEFAULT_ABS_TOL",
    "DEFAULT_INT_TOL",
    "DEFAULT_REL_TOL",
    "PointVerification",
    "RowViolation",
    "row_scales",
    "verify_point",
]


@dataclass(frozen=True)
class RowViolation:
    """One row that failed, with everything needed to argue about it."""

    label: str
    sense: str
    body: float
    rhs: float
    violation: float
    scale: float
    tol: float

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"row {self.label}: body={self.body:.12g} {self.sense} {self.rhs:.12g} "
            f"violation={self.violation:.6e} scale={self.scale:.6e} "
            f"tol={self.tol:.6e} relative={self.violation / self.scale:.6e}"
        )


@dataclass(frozen=True)
class PointVerification:
    """Outcome of :func:`verify_point`.

    ``ok`` is True only when the verifier positively checked every row, bound and
    integrality requirement and found none violated. ``refusal`` is non-None exactly
    when the verifier could not vouch (as opposed to having found a violation); both
    cases give ``ok is False`` — a caller must never read "no violations listed" as
    "verified".

    The ``n_*_checked`` counters exist so a caller can assert the probe fired
    (CLAUDE.md §6): a verification that examined zero rows is not a pass.
    """

    ok: bool
    refusal: str | None = None
    n_rows_checked: int = 0
    n_bounds_checked: int = 0
    n_integrality_checked: int = 0
    violations: tuple[RowViolation, ...] = field(default_factory=tuple)
    worst_relative: float = 0.0

    def __bool__(self) -> bool:
        return self.ok

    def describe(self) -> str:
        if self.refusal is not None:
            return f"NOT VERIFIED (refused): {self.refusal}"
        if self.ok:
            return (
                f"verified ({self.n_rows_checked} rows, {self.n_bounds_checked} bounds, "
                f"{self.n_integrality_checked} integrality; worst relative violation "
                f"{self.worst_relative:.3e})"
            )
        return "NOT VERIFIED: " + "; ".join(str(v) for v in self.violations[:5])


def row_scales(
    jac: np.ndarray | None,
    rhs: np.ndarray,
    x_flat: np.ndarray,
) -> np.ndarray:
    """Per-row scale ``max(1, |b_i|, max_j |J_ij| * max(1, |x_j|))``.

    ``jac`` may be None (no Jacobian available) — the scale then degenerates to
    ``max(1, |b_i|)``, the strict direction. A non-finite Jacobian entry is treated
    the same way *for that row*: a scale we cannot trust must not widen a tolerance.
    """
    rhs = np.asarray(rhs, dtype=np.float64)
    # Explicitly typed: numpy's stubs return Any from np.maximum/np.abs, and this
    # function's contract is an ndarray of per-row scales.
    scale: np.ndarray = np.maximum(1.0, np.abs(rhs))
    if jac is None:
        return scale
    jac = np.asarray(jac, dtype=np.float64)
    if jac.ndim != 2 or jac.shape[0] != rhs.shape[0] or jac.shape[1] != x_flat.shape[0]:
        # Shape disagreement means we are not looking at the rows we think we are.
        # Refuse to scale rather than scale by the wrong row.
        return scale
    weights = np.maximum(1.0, np.abs(np.asarray(x_flat, dtype=np.float64)))
    terms = np.abs(jac) * weights[None, :]
    finite_rows = np.all(np.isfinite(terms), axis=1)
    jac_scale = np.zeros(rhs.shape[0], dtype=np.float64)
    if terms.size:
        jac_scale[finite_rows] = np.max(terms[finite_rows], axis=1)
    combined: np.ndarray = np.maximum(scale, jac_scale)
    return combined


def _row_map(evaluator):
    """``(senses, rhs, labels)`` aligned one-to-one with ``evaluate_constraints``.

    Built from the evaluator's OWN row map (``_source_constraints`` x
    ``_constraint_flat_sizes``) rather than from ``model._constraints``, which is
    what defects 2 and 3 above were. Returns None when the evaluator does not expose
    the map — the caller must then refuse, not guess.
    """
    src = getattr(evaluator, "_source_constraints", None)
    sizes = getattr(evaluator, "_constraint_flat_sizes", None)
    if src is None or sizes is None:
        return None
    senses: list[str] = []
    rhss: list[float] = []
    labels: list[str] = []
    for c, sz in zip(src, np.asarray(sizes).tolist()):
        sz = int(sz)
        sense = c.sense if isinstance(c.sense, str) else getattr(c.sense, "value", c.sense)
        name = getattr(c, "name", None) or repr(getattr(c, "body", c))[:60]
        senses.extend([sense] * sz)
        rhss.extend([float(c.rhs)] * sz)
        labels.extend([f"{name}[{i}]" for i in range(sz)] if sz > 1 else [str(name)])
    return senses, np.asarray(rhss, dtype=np.float64), labels


def _unevaluable_kinds(model) -> list[str]:
    """Type names in ``model._constraints`` the NLP evaluator does not cover.

    SOS / logical / indicator / disjunctive constraints are not in
    ``NLPEvaluator._source_constraints``, so a row-based verifier is blind to them.
    Being blind to a constraint class while returning "feasible" is exactly the
    failure this module exists to stop, so the caller refuses instead.
    """
    from discopt.modeling.core import Constraint

    kinds: list[str] = []
    for c in getattr(model, "_constraints", ()):
        if not isinstance(c, Constraint):
            name = type(c).__name__
            if name not in kinds:
                kinds.append(name)
    return kinds


def verify_point(
    model,
    x_flat,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    int_tol: float = DEFAULT_INT_TOL,
    check_bounds: bool = True,
    check_integrality: bool = True,
    evaluator=None,
) -> PointVerification:
    """Verify that ``x_flat`` is feasible for ``model``. Never optimistic.

    Parameters
    ----------
    abs_tol
        Both the absolute row tolerance on a unit-scale row **and** the relative
        coefficient on the row scale: ``tol_i = abs_tol * scale_i`` with
        ``scale_i >= 1``. On a unit-scale row this is identical to the pre-existing
        absolute check.
    rel_tol
        Used only for the variable-bound check (``abs_tol + rel_tol * |bound|``),
        preserving the #764 calibration exactly.
    int_tol
        Integrality tolerance for INTEGER / BINARY variables.
    evaluator
        Optional pre-built ``NLPEvaluator``-like object. When None a cached
        evaluator is built for ``model``; the cache is keyed on the model's
        structural fingerprint and reads bounds live, so residuals are identical.

    Returns
    -------
    PointVerification
        ``ok`` is True only on a positively-checked, violation-free point.
    """
    from discopt.modeling.core import VarType

    x_flat = np.asarray(x_flat, dtype=np.float64).ravel()
    if not np.all(np.isfinite(x_flat)):
        return PointVerification(ok=False, refusal="point has non-finite coordinates")

    unevaluable = _unevaluable_kinds(model)
    if unevaluable:
        return PointVerification(
            ok=False,
            refusal=(
                "model carries constraint classes the NLP evaluator does not cover "
                f"({', '.join(unevaluable)}); cannot vouch for the point"
            ),
        )

    # ---- variable bounds + integrality against the DECLARED model ------------
    n_bounds = 0
    n_int = 0
    off = 0
    for v in model._variables:
        size = int(getattr(v, "size", 1))
        vals = x_flat[off : off + size]
        if vals.shape[0] != size:
            return PointVerification(
                ok=False,
                refusal=(
                    f"point length {x_flat.shape[0]} does not cover variable {v.name!r} "
                    f"(needed {size} at offset {off})"
                ),
            )
        if check_bounds:
            lb_flat = np.asarray(v.lb, dtype=np.float64).flatten()
            ub_flat = np.asarray(v.ub, dtype=np.float64).flatten()
            # Scale term keyed on the BOUND (#764: a bound-active variable comes back
            # a few ULPs off a large bound). Unchanged from the graduated behaviour.
            lb_tol = abs_tol + rel_tol * np.abs(lb_flat)
            ub_tol = abs_tol + rel_tol * np.abs(ub_flat)
            n_bounds += size
            if np.any(vals < lb_flat - lb_tol) or np.any(vals > ub_flat + ub_tol):
                return PointVerification(
                    ok=False,
                    n_bounds_checked=n_bounds,
                    n_integrality_checked=n_int,
                    violations=(
                        RowViolation(
                            label=f"bound:{v.name}",
                            sense="in",
                            body=float(vals[int(np.argmax(np.abs(vals)))]),
                            rhs=0.0,
                            violation=float(
                                np.max(np.maximum(lb_flat - lb_tol - vals, vals - ub_flat - ub_tol))
                            ),
                            scale=1.0,
                            tol=float(np.max(np.maximum(lb_tol, ub_tol))),
                        ),
                    ),
                )
        if check_integrality and v.var_type in (VarType.INTEGER, VarType.BINARY):
            n_int += size
            if np.any(np.abs(vals - np.round(vals)) > int_tol):
                return PointVerification(
                    ok=False,
                    n_bounds_checked=n_bounds,
                    n_integrality_checked=n_int,
                    violations=(
                        RowViolation(
                            label=f"integrality:{v.name}",
                            sense="int",
                            body=float(vals[int(np.argmax(np.abs(vals - np.round(vals))))]),
                            rhs=0.0,
                            violation=float(np.max(np.abs(vals - np.round(vals)))),
                            scale=1.0,
                            tol=int_tol,
                        ),
                    ),
                )
        off += size

    # ---- constraint rows -----------------------------------------------------
    if evaluator is None:
        try:
            from discopt._jax.nlp_evaluator import cached_evaluator

            evaluator = cached_evaluator(model)
        except Exception as exc:  # evaluator could not be built -> cannot vouch
            logger.debug("verify_point: evaluator construction failed: %s", exc)
            return PointVerification(
                ok=False,
                n_bounds_checked=n_bounds,
                n_integrality_checked=n_int,
                refusal=f"evaluator construction failed: {type(exc).__name__}: {exc}",
            )

    n_rows = int(getattr(evaluator, "n_constraints", 0) or 0)
    if n_rows == 0:
        return PointVerification(
            ok=True,
            n_bounds_checked=n_bounds,
            n_integrality_checked=n_int,
            n_rows_checked=0,
        )

    row_map = _row_map(evaluator)
    if row_map is None:
        return PointVerification(
            ok=False,
            n_bounds_checked=n_bounds,
            n_integrality_checked=n_int,
            refusal="evaluator does not expose its row map; cannot align senses to rows",
        )
    senses, rhs, labels = row_map
    if len(senses) != n_rows:
        return PointVerification(
            ok=False,
            n_bounds_checked=n_bounds,
            n_integrality_checked=n_int,
            refusal=(
                f"row map has {len(senses)} rows but the evaluator reports {n_rows}; "
                "refusing to compare misaligned rows"
            ),
        )
    unknown = sorted({s for s in senses if s not in ("<=", ">=", "==")})
    if unknown:
        return PointVerification(
            ok=False,
            n_bounds_checked=n_bounds,
            n_integrality_checked=n_int,
            refusal=f"unknown constraint sense(s) {unknown}; cannot vouch for the point",
        )

    try:
        body = np.asarray(evaluator.evaluate_constraints(x_flat), dtype=np.float64).ravel()
    except Exception as exc:
        logger.debug("verify_point: constraint evaluation failed: %s", exc)
        return PointVerification(
            ok=False,
            n_bounds_checked=n_bounds,
            n_integrality_checked=n_int,
            refusal=f"constraint evaluation failed: {type(exc).__name__}: {exc}",
        )
    if body.shape[0] != n_rows or not np.all(np.isfinite(body)):
        return PointVerification(
            ok=False,
            n_bounds_checked=n_bounds,
            n_integrality_checked=n_int,
            refusal=(
                f"constraint evaluation returned {body.shape[0]} rows (expected {n_rows})"
                if body.shape[0] != n_rows
                else "constraint evaluation returned a non-finite row value"
            ),
        )

    sense_arr = np.asarray(senses, dtype=object)
    signed = body - rhs
    viol = np.zeros(n_rows, dtype=np.float64)
    le = sense_arr == "<="
    ge = sense_arr == ">="
    eq = sense_arr == "=="
    viol[le] = np.maximum(signed[le], 0.0)
    viol[ge] = np.maximum(-signed[ge], 0.0)
    viol[eq] = np.abs(signed[eq])

    # Cheap gate first: scale >= 1 always, so a row within the ABSOLUTE tolerance is
    # within the scaled one too. The Jacobian (which can be the expensive part of a
    # verification, and this runs once per seed candidate) is formed only when some
    # row needs the scale term to decide it.
    suspect = viol > abs_tol
    n_suspect = int(np.count_nonzero(suspect))
    if n_suspect == 0:
        return PointVerification(
            ok=True,
            n_bounds_checked=n_bounds,
            n_integrality_checked=n_int,
            n_rows_checked=n_rows,
            worst_relative=float(np.max(viol)) if n_rows else 0.0,
        )

    jac = None
    try:
        jac = np.asarray(evaluator.evaluate_jacobian(x_flat), dtype=np.float64)
    except Exception as exc:
        # Not a refusal: falling back to |rhs|-only scale is the STRICT direction.
        logger.debug("verify_point: Jacobian unavailable, using rhs-only scale: %s", exc)
        jac = None
    scale = row_scales(jac, rhs, x_flat)
    tol = abs_tol * scale
    bad = viol > tol
    rel = viol / scale

    if not np.any(bad):
        return PointVerification(
            ok=True,
            n_bounds_checked=n_bounds,
            n_integrality_checked=n_int,
            n_rows_checked=n_rows,
            worst_relative=float(np.max(rel)),
        )

    order = np.argsort(-rel[bad])
    bad_idx = np.flatnonzero(bad)[order]
    violations = tuple(
        RowViolation(
            label=labels[i],
            sense=str(sense_arr[i]),
            body=float(body[i]),
            rhs=float(rhs[i]),
            violation=float(viol[i]),
            scale=float(scale[i]),
            tol=float(tol[i]),
        )
        for i in bad_idx[:10]
    )
    return PointVerification(
        ok=False,
        n_bounds_checked=n_bounds,
        n_integrality_checked=n_int,
        n_rows_checked=n_rows,
        violations=violations,
        worst_relative=float(np.max(rel)),
    )
