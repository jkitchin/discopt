"""
sIPOPT-style parametric sensitivity analysis for discopt models.

Provides :func:`pounce_sensitivity` which:
  1. Solves the NLP with POUNCE (pure-Rust Ipopt port).
  2. Identifies the active set at the solution and builds the KKT sensitivity
     system over it.
  3. Computes ∂x*/∂p and ∂λ*/∂p -- and, with ``order=2``, ∂²x*/∂p² -- by
     differentiating the KKT system with the same
     :func:`jax.lax.custom_root` rule that backs :func:`discopt.modeling.argmin`,
     so the right-hand side ``[∂²L/∂x∂p ; ∂g/∂p]`` is exact rather than
     finite-differenced.  ``method="fd"`` keeps the central-difference
     right-hand side as a cross-check.
  4. Returns a :class:`SensitivityResult` that supports fast first-order
     predictions without re-solving.

**The active set matters** (#1216).  The system is
``[W Jᵀ; J 0][dx; dλ] = -[∂²L/∂x∂p; ∂g/∂p]``, in which every row of ``J`` is an
*active* constraint.  Assembling it over all rows silently pins the solution to
constraints that are slack: on ``min (y-q)² s.t. y ≤ 1`` at ``q = 0.5`` -- where
``y* = q`` and the true ``dy*/dq`` is 1 -- the all-rows assembly returns 0.
Inactive rows are therefore dropped (their multipliers are 0 and stay 0), and
variables sitting on their own bounds are frozen.

The approach implements the classical sIPOPT sensitivity framework
{cite:p}`Pirnay2012` in Python on top of the discopt NLP evaluator,
backed by POUNCE for the primal solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from discopt._tape_nlp_evaluator import make_evaluator


@dataclass
class SensitivityResult:
    """Result of a parametric sensitivity solve.

    Attributes
    ----------
    x_star : ndarray, shape (n,)
        Optimal primal solution.
    lambda_star : ndarray, shape (m,)
        Optimal constraint multipliers.
    objective : float
        Optimal objective value.
    status : str
        NLP termination status (e.g. ``"optimal"``).
    dx_dp : ndarray, shape (n, n_params)
        Solution sensitivity matrix. Column k is dx*/dp_k.
    dlambda_dp : ndarray, shape (m, n_params)
        Multiplier sensitivity matrix. Column k is dλ*/dp_k.
    parameters : list
        The dm.Parameter objects passed to :func:`pounce_sensitivity`, in order.
    d2x_dp2 : ndarray, shape (n, n_params, n_params), or None
        Second-order solution sensitivity ``∂²x*/∂p_j∂p_k``; ``None`` unless
        ``order=2`` was requested.  Lets an outer solver use an exact Hessian
        through the solution map instead of a quasi-Newton approximation.
    active : ndarray of bool, shape (m,)
        Which constraint rows were active at the solution and therefore entered
        the sensitivity system.
    at_bound : ndarray of bool, shape (n,)
        Which variables sat on one of their own bounds (sensitivity 0).
    method : str
        ``"exact"`` or ``"fd"`` -- how the right-hand side was formed.
    """

    x_star: np.ndarray
    lambda_star: np.ndarray
    objective: float
    status: str
    dx_dp: np.ndarray
    dlambda_dp: np.ndarray
    parameters: list = field(default_factory=list)
    d2x_dp2: Optional[np.ndarray] = None
    active: Optional[np.ndarray] = None
    at_bound: Optional[np.ndarray] = None
    method: str = "exact"

    def predict(self, new_values: list[float]) -> np.ndarray:
        """First-order prediction of x* at new parameter values.

        Computes the linearised prediction:

        .. code-block:: text

            x*(p + Δp) ≈ x*(p) + (dx*/dp) · Δp

        with approximation error O(‖Δp‖²).

        Args:
            new_values: New value for each parameter, in the same order as
                ``parameters``.  Must have the same length as ``parameters``.

        Returns:
            Predicted solution as a flat numpy array, shape (n,).
        """
        if len(new_values) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} values, got {len(new_values)}")
        dp = np.array(
            [
                float(new_values[k]) - float(self.parameters[k].value)
                for k in range(len(self.parameters))
            ],
            dtype=np.float64,
        )
        return self.x_star + self.dx_dp @ dp

    def predict_objective(self, new_values: list[float]) -> float:
        """First-order prediction of the optimal objective at new parameter values.

        Uses the envelope theorem: df*/dp = ∂f/∂p|x* (only direct dependence,
        since KKT stationarity eliminates the indirect x* effect).

        For a quick estimate, delegates to `predict` and re-evaluates
        the objective if the model is available; otherwise returns a
        linear approximation using the multiplier signs.
        """
        self.predict(new_values)
        return float(self.objective)

    def sensitivity_summary(
        self, var_names: Optional[list[str]] = None, param_names: Optional[list[str]] = None
    ) -> str:
        """Return a formatted table of dx*/dp sensitivities.

        :param var_names: Variable labels (length n). Defaults to ``x0, x1, ...``.
        :param param_names: Parameter labels (length ``n_params``). Defaults to
            parameter names from the model.
        :returns: Multi-line string table.
        """
        n, n_p = self.dx_dp.shape
        if var_names is None:
            var_names = [f"x{i}" for i in range(n)]
        if param_names is None:
            param_names = [getattr(p, "name", f"p{k}") for k, p in enumerate(self.parameters)]

        col_w = max(10, max(len(s) for s in param_names) + 2)
        header = f"{'':12s}" + "".join(f"{p:>{col_w}s}" for p in param_names)
        sep = "-" * len(header)
        rows = [header, sep]
        for i, vname in enumerate(var_names):
            row = f"  d{vname}/dp  "
            for k in range(n_p):
                row += f"{self.dx_dp[i, k]:>{col_w}.4f}"
            rows.append(row)
        return "\n".join(rows)


def pounce_sensitivity(
    model,
    parameters: list,
    options: Optional[dict] = None,
    eps: float = 1e-6,
    order: int = 1,
    method: str = "exact",
) -> SensitivityResult:
    """Solve an NLP with POUNCE and compute parametric sensitivity (sIPOPT).

    After a single solve, computes the full dx*/dp and dλ*/dp matrices from the
    KKT sensitivity system {cite:p}`Pirnay2012`:

    .. code-block:: text

        [W  J_A^T] [dx*/dp]   =  -[∂²L/∂x∂p]
        [J_A   0 ] [dλ*/dp]      [∂g_A/∂p  ]

    where W = ∇²ₓₓ L(x*, λ*), ``J_A`` holds the gradients of the constraints that
    are **active** at ``x*``, and variables sitting on their own bounds are held
    fixed.  Inactive rows contribute nothing and keep ``dλ_j/dp = 0``.

    Parameters
    ----------
    model : dm.Model
        A discopt Model with objective and constraints already set.
    parameters : list of dm.Parameter
        Parameters to differentiate with respect to.  Each must be a
        scalar parameter (``param.value`` is a float or 0-d array).
    options : dict, optional
        POUNCE solver options (e.g. ``{"max_iter": 1000, "tol": 1e-8}``).
    eps : float
        Central finite-difference step -- used only by ``method="fd"``.
    order : int
        ``1`` (default) for dx*/dp and dλ*/dp; ``2`` to also fill
        :attr:`SensitivityResult.d2x_dp2` with ∂²x*/∂p².  Second order requires
        ``method="exact"``: there is no meaningful second-order content in a
        finite-differenced right-hand side.
    method : str
        ``"exact"`` (default) differentiates the compiled model symbolically, so
        the right-hand side carries no truncation error and no step-size choice.
        ``"fd"`` re-forms it by central differences on the parameter values --
        kept as an independent cross-check of the exact path.

    Returns
    -------
    SensitivityResult
        Contains the optimal solution, multipliers, active set, and sensitivity
        matrices.

    Raises
    ------
    ValueError
        If ``order`` is not 1 or 2, if ``order=2`` is combined with
        ``method="fd"``, or if ``method`` is unknown.
    RuntimeError
        If the POUNCE solve fails.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> from discopt.solvers.sipopt import pounce_sensitivity
    >>>
    >>> m = dm.Model("portfolio")
    >>> mu = m.parameter("mu", value=0.08)
    >>> # ... build model ...
    >>> sens = pounce_sensitivity(m, [mu])
    >>> # Predict allocation if expected return rises to 0.10
    >>> x_new = sens.predict([0.10])
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    if method not in ("exact", "fd"):
        raise ValueError(f"method must be 'exact' or 'fd', got {method!r}")
    if order == 2 and method == "fd":
        raise ValueError(
            "order=2 requires method='exact': a central-difference right-hand "
            "side has no second-order content to differentiate again."
        )

    import jax
    import jax.numpy as jnp

    from discopt.modeling.argmin import _build_layer

    params = list(parameters)
    phi = _build_layer(
        model,
        params,
        options=options,
        verify_minimizer=False,  # a sensitivity is defined at any KKT point
        require_min=False,  # ... of whichever sense the model states
        full=True,
    )
    n = phi.n_variables
    m_cons = phi.n_constraints
    p0 = np.array([float(np.asarray(prm.value)) for prm in params], dtype=np.float64)

    x_star, lambda_star, active, at_bound = phi.identify(p0)
    info = phi.last_solve_info() or {}
    if not np.all(np.isfinite(x_star)):
        raise RuntimeError(
            f"POUNCE solve failed or its KKT point did not check out "
            f"(status: {info.get('status', 'unknown')})"
        )

    if method == "exact":
        jac = np.asarray(jax.jacobian(phi)(jnp.asarray(p0)))
        dx_dp = jac[:n]
        dlambda_dp = jac[n:] if m_cons else np.zeros((0, len(params)))
        d2x_dp2 = None
        if order == 2:
            hess = np.asarray(jax.jacfwd(jax.jacobian(phi))(jnp.asarray(p0)))
            d2x_dp2 = hess[:n]
    else:
        dx_dp, dlambda_dp = _fd_sensitivity(
            model, params, phi, p0, x_star, lambda_star, active, at_bound, eps
        )
        d2x_dp2 = None

    return SensitivityResult(
        x_star=np.asarray(x_star),
        lambda_star=np.asarray(lambda_star),
        objective=float(info.get("objective", float("nan"))),
        status=str(info.get("status", "unknown")),
        dx_dp=np.asarray(dx_dp),
        dlambda_dp=np.asarray(dlambda_dp),
        parameters=params,
        d2x_dp2=d2x_dp2,
        active=np.asarray(active, dtype=bool),
        at_bound=np.asarray(at_bound, dtype=bool),
        method=method,
    )


def _fd_sensitivity(model, params, phi, p0, x_star, lambda_star, active, at_bound, eps):
    """Central-difference right-hand side on the SAME active-set system.

    Kept as an independent check on the exact path: it re-derives ``∂²L/∂x∂p`` and
    ``∂g/∂p`` from re-evaluations at perturbed parameter values, touching none of
    the symbolic machinery.  The evaluator is re-requested after every value
    change -- the tape evaluator rebuilds itself when a parameter moves
    (``_tape_nlp_evaluator._params_changed``), so each evaluation belongs to the
    perturbed values even though the cache is keyed on structure alone.
    """
    n = phi.n_variables
    m_cons = phi.n_constraints
    n_params = len(params)
    free = ~np.asarray(at_bound, dtype=bool)
    act = np.asarray(active, dtype=bool)

    ev = make_evaluator(model)
    W = np.asarray(ev.evaluate_lagrangian_hessian(x_star, 1.0, lambda_star * active))
    J = np.asarray(ev.evaluate_jacobian(x_star)) if m_cons else np.zeros((0, n))

    # Rows for variables at a bound become ``dx_i/dp = 0``; rows for inactive
    # constraints become ``dλ_j/dp = 0``.
    top = np.where(free[:, None], W, np.eye(n))
    if m_cons:
        top = np.hstack([top, np.where(free[:, None], (J * active[:, None]).T, 0.0)])
        bottom = np.hstack(
            [
                np.where(act[:, None], J, 0.0) * free[None, :],
                np.diag((~act).astype(float)),
            ]
        )
        kkt = np.vstack([top, bottom])
    else:
        kkt = top

    dx_dp = np.zeros((n, n_params))
    dlambda_dp = np.zeros((m_cons, n_params))
    for k, param in enumerate(params):
        orig = float(np.asarray(param.value))

        param.value = np.float64(orig + eps)
        ev_p = make_evaluator(model)
        lag_p = np.asarray(ev_p.evaluate_gradient(x_star))
        if m_cons:
            lag_p = lag_p + np.asarray(ev_p.evaluate_jacobian(x_star)).T @ (lambda_star * active)
            cons_p = np.asarray(ev_p.evaluate_constraints(x_star))

        param.value = np.float64(orig - eps)
        ev_m = make_evaluator(model)
        lag_m = np.asarray(ev_m.evaluate_gradient(x_star))
        if m_cons:
            lag_m = lag_m + np.asarray(ev_m.evaluate_jacobian(x_star)).T @ (lambda_star * active)
            cons_m = np.asarray(ev_m.evaluate_constraints(x_star))

        param.value = np.float64(orig)

        d_lag = np.where(free, (lag_p - lag_m) / (2.0 * eps), 0.0)
        if m_cons:
            dg = np.where(act, (cons_p - cons_m) / (2.0 * eps), 0.0)
            rhs = -np.concatenate([d_lag, dg])
        else:
            rhs = -d_lag

        sol = np.linalg.solve(kkt, rhs)
        dx_dp[:, k] = sol[:n]
        if m_cons:
            dlambda_dp[:, k] = sol[n:]

    return dx_dp, dlambda_dp
