"""
NLP solver wrapper using pounce (pure-Rust Ipopt port).

Mirrors :mod:`discopt.solvers.nlp_ipopt` exactly. ``pounce.Problem`` is
shape-compatible with ``cyipopt.Problem`` (same constructor signature,
same ``add_option`` method, same ``(x, info)`` return from ``solve``,
same Ipopt status codes), so the callback adapter and bound inference
from the cyipopt backend are reused unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model
from discopt.solvers import NLPResult, SolveStatus
from discopt.solvers.nlp_ipopt import (
    _IPOPT_STATUS_MAP,
    _infer_constraint_bounds,
    _IpoptCallbacks,
)

try:
    import pounce as _pounce  # noqa: F401

    POUNCE_AVAILABLE = True
except ImportError:
    POUNCE_AVAILABLE = False

_logger = logging.getLogger(__name__)


def solve_nlp(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]] = None,
    options: Optional[dict] = None,
    kkt_schur_block: Optional[Sequence[int]] = None,
    ordering: Optional[Sequence[int]] = None,
) -> NLPResult:
    """Solve an NLP using pounce with the NLPEvaluator callbacks.

    Same signature and semantics as :func:`discopt.solvers.nlp_ipopt.solve_nlp`,
    plus two optional structure-aware passthroughs to the underlying
    ``pounce.Problem`` (see pounce#180).

    Args:
        kkt_schur_block: Optional sequence of **KKT-space indices** (``0..dim``,
            block order ``x, slack, eq-dual, ineq-dual``) identifying a
            block-triangular / Schur partition of the KKT system. Handed to
            ``pounce.Problem.set_kkt_schur_block`` before solving; pounce falls
            back to the full-space path transparently when the partition is
            unsuitable, so the solution is unchanged and only factorization time
            differs (Parker, Garcia & Bent, arXiv:2602.17968). Honored only on
            the default FERAL + exact-Hessian path.
        ordering: Optional sequence of KKT-space indices giving a custom
            factorization ordering, handed to ``pounce.Problem.set_ordering``.
            Correctness-safe for the same reason as ``kkt_schur_block``.
    """
    if not POUNCE_AVAILABLE:
        raise ImportError(
            "pounce is required for this backend. Install it with:\n"
            "  pip install -e /path/to/pounce/python\n"
            "Or pick a different NLP backend (e.g. 'cyipopt')."
        )

    import pounce

    opts = dict(options) if options else {}
    opts.setdefault("print_level", 0)

    n = evaluator.n_variables
    m = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds
    # Snap tiny floating-point bound inversions (lb just above ub) so POUNCE's
    # IPM does not reject the problem as Invalid_Problem_Definition. These arise
    # from relaxation / bound-tightening rounding (e.g. an AMP integer-fixed
    # subproblem whose continuous bounds were tightened to lb=ub+1e-11); the
    # mirror of the same guard on the LP path (lp_pounce._snap_inverted_bounds).
    from discopt.solvers.lp_pounce import _snap_inverted_bounds

    lb, ub = _snap_inverted_bounds(
        np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)
    )

    if constraint_bounds is not None:
        cl = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        cu = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    elif m > 0:
        cl, cu = _infer_constraint_bounds(evaluator)
    else:
        cl = np.empty(0, dtype=np.float64)
        cu = np.empty(0, dtype=np.float64)

    callbacks = _IpoptCallbacks(evaluator)

    problem = pounce.Problem(
        n=n,
        m=m,
        problem_obj=callbacks,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    for key, value in opts.items():
        try:
            if isinstance(value, (np.floating, float)):
                problem.add_option(key, float(value))
            elif isinstance(value, (np.integer, int)):
                problem.add_option(key, int(value))
            else:
                problem.add_option(key, value)
        except (TypeError, ValueError, RuntimeError):
            _logger.debug("pounce option '%s' not accepted, skipping", key)

    # Structure-aware KKT passthroughs (pounce#180). Both are correctness-safe:
    # pounce transparently falls back to the full-space path when the partition
    # or ordering is unsuitable, so only factorization time changes. Guarded so
    # an older pounce without these methods degrades gracefully to the
    # full-space solve rather than raising.
    if kkt_schur_block is not None:
        block = [int(i) for i in kkt_schur_block]
        if hasattr(problem, "set_kkt_schur_block"):
            try:
                problem.set_kkt_schur_block(block)
            except (TypeError, ValueError, RuntimeError):
                _logger.debug("pounce rejected kkt_schur_block, using full space")
        else:
            _logger.debug("pounce has no set_kkt_schur_block; ignoring passthrough")
    if ordering is not None:
        order = [int(i) for i in ordering]
        if hasattr(problem, "set_ordering"):
            try:
                problem.set_ordering(order)
            except (TypeError, ValueError, RuntimeError):
                _logger.debug("pounce rejected ordering, using default")
        else:
            _logger.debug("pounce has no set_ordering; ignoring passthrough")

    t0 = time.perf_counter()
    x, info = problem.solve(x0.astype(np.float64))
    wall_time = time.perf_counter() - t0

    status_code = info.get("status", -100)
    status = _IPOPT_STATUS_MAP.get(status_code, SolveStatus.ERROR)

    multipliers = info.get("mult_g", None)
    if multipliers is not None and len(multipliers) == 0:
        multipliers = None
    mult_x_L = info.get("mult_x_L", None)
    if mult_x_L is not None and len(mult_x_L) == 0:
        mult_x_L = None
    mult_x_U = info.get("mult_x_U", None)
    if mult_x_U is not None and len(mult_x_U) == 0:
        mult_x_U = None

    return NLPResult(
        status=status,
        x=np.asarray(x),
        objective=float(info.get("obj_val", np.nan)),
        multipliers=np.asarray(multipliers) if multipliers is not None else None,
        bound_multipliers_lower=np.asarray(mult_x_L) if mult_x_L is not None else None,
        bound_multipliers_upper=np.asarray(mult_x_U) if mult_x_U is not None else None,
        iterations=int(info.get("iter_count", 0)),
        wall_time=wall_time,
    )


def solve_nlp_from_model(
    model: Model,
    x0: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
    kkt_schur_block: Optional[Sequence[int]] = None,
    ordering: Optional[Sequence[int]] = None,
) -> NLPResult:
    """Convenience: create an NLPEvaluator from a model and solve with POUNCE.

    Same signature and semantics as
    :func:`discopt.solvers.nlp_ipopt.solve_nlp_from_model`.

    ``kkt_schur_block`` and ``ordering`` are optional structure-aware
    passthroughs to :func:`solve_nlp` (see its docstring for the full contract
    and how to construct the KKT-space indices by hand). Both are
    correctness-safe: pounce falls back to the full-space path transparently, so
    the solution is unchanged and only factorization time differs. They require a
    pounce build exposing ``Problem.set_kkt_schur_block`` / ``set_ordering``
    (absent in pounce-solver 0.7.0, the current pin); until then they are silently
    no-ops.

    Args:
        model: A Model with objective and constraints set.
        x0: Initial point (n,). If None, uses midpoint of bounds clipped to [-100, 100].
        options: POUNCE/Ipopt options dict.
        kkt_schur_block: Optional Schur/block-triangular KKT partition (see above).
        ordering: Optional custom KKT-space factorization ordering (see above).

    Returns:
        NLPResult with solution.
    """
    evaluator = NLPEvaluator(model)

    if x0 is None:
        lb, ub = evaluator.variable_bounds
        lb_clipped = np.clip(lb, -100.0, 100.0)
        ub_clipped = np.clip(ub, -100.0, 100.0)
        x0 = 0.5 * (lb_clipped + ub_clipped)

    return solve_nlp(
        evaluator,
        x0,
        options=options,
        kkt_schur_block=kkt_schur_block,
        ordering=ordering,
    )
