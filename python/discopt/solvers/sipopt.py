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


class DegenerateSensitivityError(np.linalg.LinAlgError):
    """The solution map has no (unique) derivative at this KKT point (#1749).

    Raised when the active-set KKT matrix is singular *and* the singularity reaches
    the primal block: either no ``dx*/dp`` satisfies the linearised system (the
    active set changes under any perturbation of ``p``, so the one-sided derivatives
    differ) or many do.  A singular matrix whose null space touches only the
    multipliers -- the ordinary degenerate LP vertex -- is not an error: ``dx*/dp``
    is unique there and is returned (see :func:`_solve_kkt`).

    Subclasses :class:`numpy.linalg.LinAlgError`, what this case raised before.
    """


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
        Optimal objective value in the solver's internal minimization form
        (``-f*`` for a maximize model; :func:`discopt.sensitivity.sensitivity`
        converts to the model's sense).
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
    dobjective_dp : ndarray, shape (n_params,), or None
        Total derivative of :attr:`objective` along the solution path,
        ``∇ₓf(x*) · dx*/dp + ∂f/∂p`` -- what :meth:`predict_objective` uses. Same
        convention as ``objective``: the internal minimization form, so for a
        maximize model both are those of ``-f``.
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
    dobjective_dp: Optional[np.ndarray] = None

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

        .. code-block:: text

            f*(p + Δp) ≈ f*(p) + (df*/dp) · Δp

        with ``df*/dp`` = :attr:`dobjective_dp`, the total derivative of the
        objective along the solution path, ``∇ₓf · dx*/dp + ∂f/∂p`` (at a KKT point
        with a locally constant active set this equals the envelope-theorem value
        ``∂L/∂p``). Approximation error O(‖Δp‖²).

        Before #1618 (B-07b) this returned the base objective for any argument.

        Raises:
            ValueError: if ``new_values`` has the wrong length, or the result was
                built without ``dobjective_dp`` (a hand-constructed instance).
        """
        if len(new_values) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} values, got {len(new_values)}")
        if self.dobjective_dp is None:
            raise ValueError(
                "predict_objective needs dobjective_dp, which pounce_sensitivity fills; "
                "this SensitivityResult was constructed without it."
            )
        dp = np.array(
            [
                float(new_values[k]) - float(self.parameters[k].value)
                for k in range(len(self.parameters))
            ],
            dtype=np.float64,
        )
        return float(self.objective + np.asarray(self.dobjective_dp, dtype=np.float64) @ dp)

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
    x0: Optional[np.ndarray] = None,
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
    x0 : ndarray, optional
        Starting point for the forward POUNCE solve, flat and in model variable
        order.  On a nonconvex model this decides *which* KKT point the whole
        sensitivity describes; without it the solve starts from the midpoint of
        the clipped variable box (issue #1313).

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

    from discopt._optional_jax import require_jax

    require_jax("pounce_sensitivity (sIPOPT-style parametric sensitivity)")
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
        x0=x0,
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
        d2x_dp2 = None
        if order == 1:
            # #1615 B-01a: the KKT matrix comes from the tape evaluator's exact
            # Hessian/Jacobian and only the right-hand side -- one forward pass per
            # parameter -- goes through JAX.  ``jax.jacobian(phi)`` assembles the
            # same matrix by an (n+m)-row JAX Jacobian of the residual inside
            # ``custom_root``, which was ~94% of a 100-step MPC's sensitivity time.
            kkt = _kkt_matrix(model, n, m_cons, x_star, lambda_star, active, at_bound)
            z0 = jnp.asarray(np.concatenate([x_star, lambda_star]))
            act_j, bnd_j = jnp.asarray(active), jnp.asarray(at_bound)
            x_j0 = jnp.asarray(x_star)
            d_res = np.asarray(
                jax.jacfwd(lambda pp: phi.kkt_residual(z0, pp, x_j0, act_j, bnd_j))(jnp.asarray(p0))
            ).reshape(n + m_cons, len(params))
            sol = _solve_kkt(kkt, -d_res, n)
            dx_dp = sol[:n]
            dlambda_dp = sol[n:] if m_cons else np.zeros((0, len(params)))
        else:
            # ``phi``'s ``custom_root`` solves the same matrix with
            # ``jnp.linalg.solve``, which does not raise on a singular one: it
            # returns inf/nan or garbage. Refuse a degenerate point up front (#1749).
            kkt = _kkt_matrix(model, n, m_cons, x_star, lambda_star, active, at_bound)
            if np.linalg.matrix_rank(kkt) < kkt.shape[0]:
                raise DegenerateSensitivityError(
                    "order=2 sensitivity is not available at a degenerate KKT point "
                    "(the active-set KKT matrix is singular: more constraints are "
                    "active than the point needs, as at a degenerate LP vertex). "
                    "First-order derivatives (order=1) handle this case."
                )
            jac = np.asarray(jax.jacobian(phi)(jnp.asarray(p0)))
            dx_dp = jac[:n]
            dlambda_dp = jac[n:] if m_cons else np.zeros((0, len(params)))
            hess = np.asarray(jax.jacfwd(jax.jacobian(phi))(jnp.asarray(p0)))
            d2x_dp2 = hess[:n]
    else:
        dx_dp, dlambda_dp = _fd_sensitivity(
            model, params, phi, p0, x_star, lambda_star, active, at_bound, eps
        )
        d2x_dp2 = None

    # #1618 B-07b: the objective's total derivative along the solution path, in the
    # same convention as ``objective`` -- POUNCE's internal minimization form (``-f``
    # for a maximize model), which ``discopt.sensitivity`` relies on and converts.
    x_j = jnp.asarray(x_star, dtype=jnp.float64)
    p_j = jnp.asarray(p0)
    grad_x = np.asarray(jax.grad(lambda xx: phi.objective_fn(xx, phi.params_tuple(p_j)))(x_j))
    grad_p = np.asarray(jax.grad(lambda pp: phi.objective_fn(x_j, phi.params_tuple(pp)))(p_j))
    dobjective_dp = grad_x @ np.asarray(dx_dp) + grad_p

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
        dobjective_dp=np.asarray(dobjective_dp, dtype=np.float64).reshape(len(params)),
    )


def _kkt_matrix(model, n, m_cons, x_star, lambda_star, active, at_bound):
    """The active-set KKT matrix at ``(x*, λ*)``, from the tape evaluator.

    The same square system ``phi``'s ``custom_root`` linearizes: rows for variables
    at a bound become ``dx_i/dp = 0`` and rows for inactive constraints become
    ``dλ_j/dp = 0``, so only the right-hand side depends on how ``p`` enters.
    """
    free = ~np.asarray(at_bound, dtype=bool)
    act = np.asarray(active, dtype=bool)
    ev = make_evaluator(model)
    W = np.asarray(ev.evaluate_lagrangian_hessian(x_star, 1.0, lambda_star * active))
    J = np.asarray(ev.evaluate_jacobian(x_star)) if m_cons else np.zeros((0, n))
    top = np.where(free[:, None], W, np.eye(n))
    if not m_cons:
        return top
    top = np.hstack([top, np.where(free[:, None], (J * active[:, None]).T, 0.0)])
    bottom = np.hstack(
        [
            np.where(act[:, None], J, 0.0) * free[None, :],
            np.diag((~act).astype(float)),
        ]
    )
    return np.vstack([top, bottom])


#: Relative singular-value cutoff for the rank of a singular KKT matrix, and the
#: relative residual / null-space size below which a component counts as zero.
_KKT_RANK_RTOL = 1e-10
_KKT_CONSISTENT_RTOL = 1e-8


def _solve_kkt(kkt: np.ndarray, rhs: np.ndarray, n: int) -> np.ndarray:
    """Solve the active-set KKT system, handling a degenerate (singular) one.

    A nonsingular matrix goes through :func:`numpy.linalg.solve` unchanged.  A
    singular one is the normal case at a degenerate point -- at an LP vertex with
    more active constraints than variables (``x0 + x1 = 1`` with ``x = (1, 0)`` on
    ``[0, 1]^2``) both variables are frozen at bounds, so the equality's row and
    column vanish (#1749).  There the system is solved by SVD and judged:

    * inconsistent (no solution), or a null space with a primal component (``dx``
      not unique): :class:`DegenerateSensitivityError` -- there is no derivative to
      report, and returning a least-squares one would be a silent approximation;
    * otherwise ``dx*/dp`` is unique and exact.  The multiplier derivatives along
      the null space are not determined; those rows of ``dλ*/dp`` are ``nan``
      rather than one arbitrary choice among infinitely many.
    """
    try:
        direct: np.ndarray = np.linalg.solve(kkt, rhs)
        return direct
    except np.linalg.LinAlgError:
        pass
    U, s, Vt = np.linalg.svd(kkt)
    smax = float(s[0]) if s.size else 0.0
    rank = int(np.sum(s > _KKT_RANK_RTOL * smax)) if smax > 0.0 else 0
    rhs2 = rhs.reshape(kkt.shape[0], -1)
    sol = Vt[:rank].T @ ((U[:, :rank].T @ rhs2) / s[:rank, None])
    resid = np.linalg.norm(kkt @ sol - rhs2, axis=0)
    scale = smax * np.linalg.norm(sol, axis=0) + np.linalg.norm(rhs2, axis=0)
    if np.any(resid > _KKT_CONSISTENT_RTOL * np.maximum(scale, 1.0)):
        raise DegenerateSensitivityError(
            "the solution is not differentiable in the parameters at this point: "
            "the active-set KKT matrix is singular (a degenerate point, e.g. an LP "
            "vertex with more active constraints than variables) and the linearised "
            "system has no solution, so the active set changes under any parameter "
            "perturbation and the one-sided derivatives differ"
        )
    null = Vt[rank:]
    if null.size and np.max(np.abs(null[:, :n])) > _KKT_CONSISTENT_RTOL:
        raise DegenerateSensitivityError(
            "dx*/dp is not unique at this point: the active-set KKT matrix is "
            "singular and its null space moves the primal solution"
        )
    undetermined = np.max(np.abs(null), axis=0) > _KKT_CONSISTENT_RTOL if null.size else None
    if undetermined is not None:
        sol[undetermined] = np.nan
    out: np.ndarray = sol.reshape(rhs.shape)
    return out


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

    kkt = _kkt_matrix(model, n, m_cons, x_star, lambda_star, active, at_bound)

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

        sol = _solve_kkt(kkt, rhs, n)
        dx_dp[:, k] = sol[:n]
        if m_cons:
            dlambda_dp[:, k] = sol[n:]

    return dx_dp, dlambda_dp
