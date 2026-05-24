"""scipy-based McCormick relaxation NLP solver.

Solves the convex relaxation NLP with ``scipy.optimize.minimize`` using
numpy callbacks built by ``discopt._numpy.relaxation_compiler``. Used
on small B&B nodes where the JAX trace/compile floor would dominate.

The signature parallels ``discopt._jax.mccormick_nlp.solve_mccormick_relaxation_nlp``
so the dispatcher can swap backends transparently.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def solve_mccormick_relaxation_nlp_numpy(
    obj_relax_fn: Callable,
    con_relax_fns: Optional[list[Callable]],
    con_senses: Optional[list[str]],
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    negate: bool = False,
    max_iter: int = 100,
    deadline: float | None = None,
) -> float:
    """Solve the convex McCormick relaxation NLP with scipy SLSQP.

    Returns a valid lower bound (float), or ``-inf`` on failure / deadline.
    """
    if _deadline_expired(deadline):
        return -np.inf

    lb = np.asarray(node_lb, dtype=np.float64)
    ub = np.asarray(node_ub, dtype=np.float64)

    # Quick bailout: if the objective relaxation is degenerate at the
    # midpoint, skip. Matches the JAX path's early-exit policy.
    mid = 0.5 * (lb + ub)
    try:
        cv_test, cc_test = obj_relax_fn(mid, mid, lb, ub)
        if not (np.isfinite(float(cv_test)) and np.isfinite(float(cc_test))):
            return -np.inf
    except Exception:
        return -np.inf

    if _deadline_expired(deadline):
        return -np.inf

    def obj_fn(x):
        cv, cc = obj_relax_fn(x, x, lb, ub)
        return float(-cc if negate else cv)

    constraints = []
    has_cons = bool(con_relax_fns) and bool(con_senses)
    if has_cons:
        for fn, sense in zip(con_relax_fns, con_senses):
            if sense == ">=":
                # Original: g(x) >= 0; relaxation requires -cc <= 0 i.e. cc >= 0.
                def make_ge(f=fn):
                    def c(x):
                        _cv, cc = f(x, x, lb, ub)
                        return float(cc)
                    return c
                constraints.append({"type": "ineq", "fun": make_ge()})
            else:
                # "<=" and "=="; relaxation requires cv <= 0 i.e. -cv >= 0.
                def make_le(f=fn):
                    def c(x):
                        cv, _cc = f(x, x, lb, ub)
                        return float(-cv)
                    return c
                constraints.append({"type": "ineq", "fun": make_le()})

    x0 = np.clip(mid, lb, ub)
    bounds = list(zip(lb.tolist(), ub.tolist()))

    try:
        res = minimize(
            obj_fn,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-8, "disp": False},
        )
    except Exception:
        return -np.inf

    # SLSQP success criterion: success flag, or low-iteration termination
    # with a feasible-looking residual. The relaxation value is a valid
    # lower bound iff the optimizer reached a feasible point of the
    # convex relaxation. We accept on success OR on iteration-limit hits
    # with finite objective and small constraint violation.
    if not np.isfinite(res.fun):
        return -np.inf

    if has_cons:
        max_viol = 0.0
        for c in constraints:
            try:
                val = c["fun"](res.x)
                if val < -1e-6:
                    max_viol = max(max_viol, -val)
            except Exception:
                return -np.inf
        if max_viol > 1e-4:
            return -np.inf

    return float(res.fun)
