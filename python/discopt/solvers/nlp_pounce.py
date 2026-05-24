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
from typing import Optional

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator
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
) -> NLPResult:
    """Solve an NLP using pounce with the NLPEvaluator callbacks.

    Same signature and semantics as :func:`discopt.solvers.nlp_ipopt.solve_nlp`.
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
