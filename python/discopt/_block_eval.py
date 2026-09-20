"""Compressed derivative evaluation over identical blocks (#1370 Part B).

The contingency blocks of an N-1 SCOPF, the scenario blocks of an extensive
form, the per-trajectory collocation blocks of ``dae.fit.fit_trajectories`` are
the same sub-model repeated. They share one sparsity pattern, therefore one
coloring, therefore **one vectorised pass**: the whole Jacobian is C
directional derivatives and the whole Lagrangian Hessian C' Hessian-vector
products, where C and C' are chromatic numbers that do not grow with the number
of blocks. The default evaluator — POUNCE's Rust AD tape — instead walks every
block's rows, so its cost grows linearly in K while this one does not.

Measured on the entry experiment's model class (docs/dev/1370-block-eval-entry-2026-09-20.md),
one compressed pass against the tape: **8.31x on the Jacobian and 30.19x on the
Lagrangian Hessian at K=32** with 1200-column blocks. Evaluation is 20-24% of
those solves, so the end-to-end ceiling is ~1.3x on its own and ~1.45x once
#1370 Part A's block factorization lands — recorded there in full, including
the correction to the issue's "evaluation is 48%" premise.

**This is assembled from parts that were already here, not rebuilt.**
``_relax/sparsity.compute_coloring``, ``_relax/sparse_jacobian`` and
``_relax/sparse_hessian`` (with its dense-row separation for exactly the
arrowhead case a shared border produces) already implement colored compression;
they take a pattern and a coloring as arguments, and that is what this module
supplies. What it does NOT reuse is the JAX evaluator's *detected* sparsity:
measured on a K=8 block model, its Hessian pattern is **96.7% dense — 44,750,500
entries against the tape's 19,060** — because a vectorised array body defeats the
detection. A coloring of that pattern would be worthless. The patterns here come
from the tape, which knows them exactly.

**Correctness.** The values this returns must equal the tape's, and that is
checked rather than argued: :func:`build_compressed_evaluator` recovers both
derivative sets at points in the model's own box and compares them entrywise
against the base evaluator before returning anything. A mismatch is a refusal,
not a fallback — silently serving a solver derivatives from an engine that
disagrees with the one it was validated against is how a false certificate gets
made. The check reports how many entries it compared, and a comparison count of
zero is itself a refusal (CLAUDE.md measurement discipline §6).

Default OFF behind ``DISCOPT_BLOCK_VECTOR_EVAL``; see
``docs/dev/flag-retirement-audit.md`` for its state under CLAUDE.md §5.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Model

_logger = logging.getLogger(__name__)

#: Entry gate. ``1`` routes a declared block-structured model's Jacobian and
#: Lagrangian-Hessian VALUES through the compressed path; every other quantity,
#: and every structure, still comes from the default evaluator.
BLOCK_VECTOR_EVAL_ENV = "DISCOPT_BLOCK_VECTOR_EVAL"

#: Agreement bars for the admission check, matching the tape evaluator's own
#: graduation bars (``_tape_nlp_evaluator``'s entry experiment): 1e-10 on the
#: Jacobian, 1e-8 on the Hessian, relative to the base evaluator's magnitude.
_JAC_TOL = 1e-10
_HESS_TOL = 1e-8


class CompressionRefused(RuntimeError):
    """The compressed path could not be built, or disagreed with the base evaluator."""


def block_vector_eval_requested() -> bool:
    """Is the compressed-derivative path switched on?

    Default OFF. It is a *derivative engine* change, which CLAUDE.md §5 counts
    as bound-changing — the B&B is path-dependent, so even last-digit
    differences can move the cut sequence — so it stays behind the flag until a
    differential panel passes, exactly as the tape evaluator itself did.
    """
    return os.environ.get(BLOCK_VECTOR_EVAL_ENV, "0") not in ("0", "", "false", "False")


def _pattern_from_structures(
    n: int,
    m: int,
    jac_rows: np.ndarray,
    jac_cols: np.ndarray,
    hess_rows: np.ndarray,
    hess_cols: np.ndarray,
):
    """A :class:`SparsityPattern` built from the BASE evaluator's exact structures.

    The Hessian half is symmetrized: ``hessian_structure`` reports the lower
    triangle, while the coloring needs the full symmetric pattern (a column's
    neighbours live on both sides of the diagonal).
    """
    import scipy.sparse as sp

    from discopt._relax.sparsity import SparsityPattern

    jac = sp.csr_matrix(
        (np.ones(jac_rows.size, dtype=bool), (jac_rows, jac_cols)), shape=(m, n), dtype=bool
    )
    lower = sp.csr_matrix(
        (np.ones(hess_rows.size, dtype=bool), (hess_rows, hess_cols)), shape=(n, n), dtype=bool
    )
    hess = (lower + lower.T).tocsr()
    hess.data = np.ones_like(hess.data, dtype=bool)
    return SparsityPattern(
        jacobian_sparsity=jac,
        hessian_sparsity=hess,
        n_vars=int(n),
        n_cons=int(m),
        jacobian_nnz=int(jac_rows.size),
        hessian_nnz=int(hess.nnz),
    )


def _make_jac_values_fn(cons_fn_xp, colors: np.ndarray, seed_matrix: np.ndarray, rows, cols):
    """``fn(x, params) -> Jacobian values aligned to (rows, cols)``.

    Deliberately not ``sparse_jacobian.make_sparse_jac_values_fn``: that one
    aligns its output to ``pattern.jacobian_sparsity.tocoo()``, i.e. to
    row-major CSR order, whereas these values have to align to the *base
    evaluator's* ``jacobian_structure()`` order, whatever that is. Recovering
    against the caller's own index arrays removes an ordering assumption that
    would be silent when wrong — the whole failure mode #1370 is about.
    """
    import jax
    import jax.numpy as jnp

    seeds = jnp.asarray(seed_matrix.T, dtype=jnp.float64)  # (n_colors, n)
    entry_color = np.asarray(colors, dtype=np.intp)[np.asarray(cols, dtype=np.intp)]
    entry_row = np.asarray(rows, dtype=np.intp)

    @jax.jit
    def batch_jvp(x, params):
        def one(seed):
            return jax.jvp(lambda xp: cons_fn_xp(xp, params), (x,), (seed,))[1]

        return jax.vmap(one)(seeds)  # (n_colors, m)

    def jac_values(x, params) -> np.ndarray:
        w = np.asarray(batch_jvp(x, params))
        return w[entry_color, entry_row].astype(np.float64, copy=False)

    return jac_values


class CompressedBlockEvaluator:
    """An evaluator that serves J/H values from one vectorised compressed pass.

    Every other quantity — the objective, the gradient, the constraint values,
    the bounds, and **both derivative structures** — is delegated to the base
    evaluator unchanged. The structures in particular are deliberately NOT
    recomputed here: they are what the solver indexes the values with, and two
    sources for them is the one way this could produce a wrong matrix while
    every individual number looks right.
    """

    # NO class-level ``timing_bucket``: it is delegated through ``__getattr__``
    # to the base evaluator, deliberately. ``_IpoptCallbacks`` charges every
    # callback to that bucket, and most callbacks here (objective, gradient,
    # constraints) really do run the base evaluator's tape — declaring "jax"
    # would charge Rust time to JAX, the exact two-sided misattribution
    # ``nlp_ipopt._charge_evaluator`` was written to end. The two callbacks that
    # DO run in XLA open their own ``charge("jax")`` inside that frame; since
    # ``charge`` records self time, the outer frame then keeps only the call
    # overhead and the inner one carries the work.

    def __init__(self, base: Any, jac_values_fn, hess_values_fn, params_fn, n_colors: tuple):
        self._base = base
        self._jac_values_fn = jac_values_fn
        self._hess_values_fn = hess_values_fn
        self._params_fn = params_fn
        self.n_jacobian_colors, self.n_hessian_seeds = n_colors

    def __getattr__(self, name):
        # Objective, gradient, constraints, bounds, row map, structures, the
        # Gauss-Newton flag — all the base evaluator's.
        return getattr(self._base, name)

    # The delegated attributes above cover everything except the two hot paths.

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        from discopt import _timing

        with _timing.charge("jax"):
            values = self._jac_values_fn(np.asarray(x, dtype=np.float64), self._params_fn())
        return np.asarray(values, dtype=np.float64)

    def evaluate_hessian_values(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        from discopt import _timing

        with _timing.charge("jax"):
            values = self._hess_values_fn(
                np.asarray(x, dtype=np.float64),
                float(obj_factor),
                np.asarray(lambda_, dtype=np.float64),
                self._params_fn(),
            )
        return np.asarray(values, dtype=np.float64)

    def summary(self) -> str:
        return (
            f"CompressedBlockEvaluator: {self.n_jacobian_colors} Jacobian colors, "
            f"{self.n_hessian_seeds} Hessian seeds"
        )


def _probe_points(base: Any, count: int) -> list[np.ndarray]:
    """Points inside the model's own box to check agreement at.

    Inside the box because that is where the solver will ask: an agreement
    check at a point the model forbids can pass on an expression that is
    undefined where it matters (a ``log`` of a negative, say, agreeing as NaN
    on both sides).
    """
    lb, ub = base.variable_bounds
    lb = np.clip(np.asarray(lb, dtype=np.float64), -1e3, 1e3)
    ub = np.clip(np.asarray(ub, dtype=np.float64), -1e3, 1e3)
    mid = 0.5 * (lb + ub)
    span = np.maximum(ub - lb, 1e-6)
    rng = np.random.default_rng(1370)
    out = [mid]
    for _ in range(count - 1):
        out.append(np.clip(mid + 0.25 * span * rng.uniform(-1.0, 1.0, size=mid.size), lb, ub))
    return out


def _agrees(a: np.ndarray, b: np.ndarray, tol: float) -> tuple[bool, float]:
    scale = np.maximum(1.0, np.abs(b))
    err = float(np.max(np.abs(a - b) / scale)) if b.size else 0.0
    return err <= tol, err


def build_compressed_evaluator(
    model: "Model",
    base: Any,
    *,
    probe_points: int = 2,
) -> CompressedBlockEvaluator:
    """Build the compressed evaluator for *model*, or refuse with a reason.

    Raises:
        CompressionRefused: the path could not be built, or its values disagree
            with *base*. Never returns an evaluator that was not checked.
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt._relax.sparse_hessian import build_hessian_coloring, make_sparse_hess_values_fn
    from discopt._relax.sparsity import compute_coloring, make_seed_matrix

    n = int(base.n_variables)
    m = int(base.n_constraints)
    if m == 0:
        raise CompressionRefused("the model has no constraints; there is nothing to compress")

    jr, jc = base.jacobian_structure()
    hr, hc = base.hessian_structure()
    jr = np.asarray(jr, dtype=np.intp)
    jc = np.asarray(jc, dtype=np.intp)
    hr = np.asarray(hr, dtype=np.intp)
    hc = np.asarray(hc, dtype=np.intp)
    if jr.size == 0 and hr.size == 0:
        raise CompressionRefused("the base evaluator reports no derivative structure")

    pattern = _pattern_from_structures(n, m, jr, jc, hr, hc)
    colors, n_colors = compute_coloring(pattern)
    if n_colors >= n:
        raise CompressionRefused(
            f"the Jacobian needs {n_colors} colors for {n} columns; compression would cost "
            "at least as much as the dense pass"
        )
    seed = make_seed_matrix(colors, n_colors, n)

    jax_ev = NLPEvaluator(model)
    cons_fn = getattr(jax_ev, "_cons_fn_jit", None)
    hvp_fn = getattr(jax_ev, "_lagrangian_hvp_fn_jit", None)
    if cons_fn is None or hvp_fn is None:
        raise CompressionRefused(
            "the JAX evaluator exposes no constraint function or Lagrangian HVP for this model "
            "(a Gauss-Newton objective or an unsupported body)"
        )
    # Both are the ``(x, params, ...)`` forms, NOT the ``_bind_x_only`` wrappers
    # beside them: the parameter values have to be threaded through explicitly
    # so a rebind does not force an XLA retrace.

    jac_values_fn = _make_jac_values_fn(cons_fn, colors, seed, jr, jc)
    h_seed, coo_seed_idx, coo_lookup_row = build_hessian_coloring(pattern, hr, hc)
    hess_values_fn = make_sparse_hess_values_fn(hvp_fn, h_seed, coo_seed_idx, coo_lookup_row)
    n_hess_seeds = int(h_seed.shape[1])

    params_fn = jax_ev._current_params

    # --- admission: the values must equal the base evaluator's, at points in the
    # box, before this is handed to a solver.
    compared = 0
    rng = np.random.default_rng(1370)
    for x in _probe_points(base, max(1, probe_points)):
        lam = rng.normal(size=m)
        base_j = np.asarray(base.evaluate_jacobian_values(x), dtype=np.float64)
        ours_j = np.asarray(jac_values_fn(x, params_fn()), dtype=np.float64)
        if base_j.shape != ours_j.shape:
            raise CompressionRefused(
                f"Jacobian value arrays differ in shape: {base_j.shape} vs {ours_j.shape}"
            )
        ok, err = _agrees(ours_j, base_j, _JAC_TOL)
        compared += int(base_j.size)
        if not ok:
            raise CompressionRefused(
                f"compressed Jacobian disagrees with the base evaluator by {err:.3e} "
                f"(bar {_JAC_TOL:.0e})"
            )

        base_h = np.asarray(base.evaluate_hessian_values(x, 1.0, lam), dtype=np.float64)
        ours_h = np.asarray(hess_values_fn(x, 1.0, lam, params_fn()), dtype=np.float64)
        if base_h.shape != ours_h.shape:
            raise CompressionRefused(
                f"Hessian value arrays differ in shape: {base_h.shape} vs {ours_h.shape}"
            )
        ok, err = _agrees(ours_h, base_h, _HESS_TOL)
        compared += int(base_h.size)
        if not ok:
            raise CompressionRefused(
                f"compressed Lagrangian Hessian disagrees with the base evaluator by {err:.3e} "
                f"(bar {_HESS_TOL:.0e})"
            )

    if compared == 0:
        # A check that compared nothing passed vacuously, which is indistinguishable
        # from a check that found nothing wrong (CLAUDE.md §6).
        raise CompressionRefused("the admission check compared no entries")

    _logger.info(
        "compressed derivative path admitted: %d Jacobian colors, %d Hessian seeds, "
        "%d entries checked against the base evaluator [block-vector-eval]",
        n_colors,
        n_hess_seeds,
        compared,
    )
    return CompressedBlockEvaluator(
        base, jac_values_fn, hess_values_fn, params_fn, (n_colors, n_hess_seeds)
    )


def maybe_wrap_evaluator(model: "Model", base: Any) -> Any:
    """Wrap *base* in the compressed path when it is switched on and usable.

    Returns *base* unchanged when the flag is off, when the model declares no
    block structure, or when the compressed path refuses — the refusal is
    logged with its reason rather than raised, because this sits on the solve
    path and the base evaluator is always a correct answer. What is never
    silent is a *disagreement*: that is the one refusal reason worth a warning,
    since it means two engines in this tree compute different derivatives for
    the same model.
    """
    if not block_vector_eval_requested():
        return base
    from discopt.block_structure import has_declaration

    if not has_declaration(model):
        _logger.debug(
            "%s is set but the model declares no block structure; using the default evaluator",
            BLOCK_VECTOR_EVAL_ENV,
        )
        return base
    try:
        return build_compressed_evaluator(model, base)
    except CompressionRefused as exc:
        if "disagrees" in str(exc):
            _logger.warning(
                "the compressed derivative path was REFUSED because it disagrees with the "
                "default evaluator: %s. Solving with the default evaluator "
                "[block-vector-eval-disagreement].",
                exc,
            )
        else:
            _logger.info("compressed derivative path not used: %s", exc)
        return base
    except Exception as exc:  # noqa: BLE001 - never fail a solve over an optimization
        _logger.info(
            "compressed derivative path unavailable (%s: %s); using the default evaluator",
            type(exc).__name__,
            exc,
        )
        return base
