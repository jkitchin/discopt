"""Affine decision rules for adjustable robust optimization.

Adjustable robust optimization (ARO) {cite:p}`BenTalGoryashko2004` extends
static robust optimization by partitioning decision variables into two stages:

* **Here-and-now** variables ``x``: chosen *before* the uncertainty ``ξ`` is
  revealed.  These must be feasible for *every* realisation of ``ξ``.
* **Wait-and-see** (recourse) variables ``y``: chosen *after* observing ``ξ``,
  so they can adapt to the particular realization.

The general two-stage problem is:

.. math::

    \\min_{x,\\, y(\\cdot)}\\; f(x) + h(y(\\xi))
    \\quad \\text{s.t.}\\quad
    g(x,\\, y(\\xi),\\, \\xi) \\le 0 \\;\\forall\\, \\xi \\in \\mathcal{U}

Because optimising over arbitrary functions :math:`y(\\cdot)` is
computationally intractable, we restrict to **affine decision rules**
(also called *linear decision rules*) {cite:p}`BenTalGoryashko2004`:

.. math::

    y(\\xi) = y_0 + \\sum_{j=1}^{n_\\xi} Y_j \\cdot \\xi_j

where :math:`y_0` is the *intercept* (a new decision variable with the same
shape as ``y``), each :math:`Y_j` is a *policy column* (a new decision
variable with the same shape as ``y``), and
:math:`\\xi_j = p_j - \\bar{p}_j` is the signed deviation of the
:math:`j`-th scalar uncertain parameter from its nominal value.

After calling :meth:`AffineDecisionRule.apply`:

1. The recourse variable ``y`` is **substituted** throughout the model with
   the affine expression :math:`y_0 + \\sum_j Y_j \\xi_j`.
2. ``y`` is retired from the model's variable list; ``y_0`` and all
   :math:`Y_j` become ordinary decision variables.
3. The model still contains uncertain parameters (via the :math:`\\xi_j`
   terms).  A subsequent call to :class:`~discopt.ro.counterpart.RobustCounterpart`
   handles the remaining uncertainty via the static robust counterpart.

Limitations
-----------
Affine decision rules are *optimal* for problems where the feasible set is
convex and the uncertainty enters linearly {cite:p}`BenTalGoryashko2004`.
For general nonlinear recourse, they are an approximation (inner
approximation: the affine-rule solution is feasible but potentially
sub-optimal relative to the full adjustable problem).

References
----------
.. [BenTalGoryashko2004] Ben-Tal, A., Goryashko, A., Guslitzer, E., &
   Nemirovski, A. (2004). Adjustable robust solutions of uncertain linear
   programs. *Mathematical Programming*, 99(2), 351–376.
   https://doi.org/10.1007/s10107-003-0454-y

.. [Chen2009] Chen, X., Sim, M., Sun, P., & Zhang, J. (2009). A linear
   decision-based approximation approach to stochastic programming.
   *Operations Research*, 56(2), 344–357.

.. [Bertsimas2010] Bertsimas, D., & Goyal, V. (2010). On the power and
   limitations of affine policies in two-stage adaptive optimization.
   *Mathematical Programming*, 134(2), 491–531.
   https://doi.org/10.1007/s10107-011-0444-4
"""

from __future__ import annotations

import numpy as np


class AffineDecisionRule:
    """Parameterize a recourse variable as an affine function of uncertain parameters.

    Replaces a wait-and-see decision variable ``y`` with the affine rule

    .. math::

        y(\\xi) = y_0 + \\sum_{j=1}^{n_\\xi} Y_j \\cdot \\xi_j,
        \\quad \\xi_j = p_j - \\bar{p}_j

    where the intercept ``y_0`` and policy columns ``Y_j`` become ordinary
    (here-and-now) decision variables in the reformulated model.

    Parameters
    ----------
    recourse_var : Variable
        The wait-and-see decision variable to be parameterized.  Its shape
        determines the shape of ``y_0`` and each ``Y_j``.
    uncertain_params : Parameter or list[Parameter]
        Parameters whose values are uncertain.  The affine rule responds to
        the *deviation* ``ξ_j = p_j - p̄_j`` of each scalar component from
        its nominal value.  The total number of policy columns created equals
        the sum of all component sizes of all uncertain parameters.
    prefix : str
        Name prefix for the introduced variables (``{prefix}_intercept``,
        ``{prefix}_Y0``, ``{prefix}_Y1``, …).

    Attributes
    ----------
    intercept : Variable
        The intercept variable ``y_0`` (available after :meth:`apply`).
    policy_columns : list[Variable]
        Policy column variables ``[Y_0, Y_1, …]`` in parameter-component order
        (available after :meth:`apply`).
    affine_expression : Expression
        The full affine expression ``y_0 + ΣY_j·ξ_j`` (available after
        :meth:`apply`).
    n_policy_columns : int
        Total number of scalar uncertain parameter components.

    Examples
    --------
    Two-stage inventory problem with uncertain demand:

    >>> import discopt.modeling as dm
    >>> from discopt.ro import BoxUncertaintySet, RobustCounterpart
    >>> from discopt.ro import AffineDecisionRule
    >>>
    >>> m = dm.Model("inventory")
    >>> x = m.continuous("order", lb=0, ub=200)      # here-and-now order
    >>> y = m.continuous("recourse", lb=0, ub=100)   # wait-and-see extra order
    >>> d = m.parameter("d", value=80.0)             # uncertain demand
    >>>
    >>> m.minimize(2 * x + 5 * y)                    # ordering cost
    >>> m.subject_to(x + y >= d, name="demand")      # meet demand
    >>> m.subject_to(x + y <= 200, name="capacity")
    >>>
    >>> # Stage 1: substitute y with affine rule in ξ = d - 80
    >>> adr = AffineDecisionRule(y, uncertain_params=d)
    >>> adr.apply()
    >>>
    >>> # Stage 2: handle remaining uncertainty robustly
    >>> rc = RobustCounterpart(m, BoxUncertaintySet(d, delta=20.0))
    >>> rc.formulate()
    >>>
    >>> result = m.solve()
    """

    def __init__(
        self,
        recourse_var,                        # type: Variable
        uncertain_params,                    # type: Parameter | list[Parameter]
        prefix: str = "adr",
    ) -> None:
        from discopt.modeling.core import Parameter, Variable

        if not isinstance(recourse_var, Variable):
            raise TypeError(
                f"recourse_var must be a discopt Variable, got {type(recourse_var).__name__}"
            )
        if len(recourse_var.shape) > 1:
            raise ValueError(
                "AffineDecisionRule currently supports scalar and 1-D recourse variables. "
                f"Got shape {recourse_var.shape}."
            )

        if isinstance(uncertain_params, Parameter):
            uncertain_params = [uncertain_params]
        if not uncertain_params:
            raise ValueError("uncertain_params must not be empty")
        for p in uncertain_params:
            if not isinstance(p, Parameter):
                raise TypeError(
                    f"All uncertain_params must be discopt Parameters, got {type(p).__name__}"
                )

        self._recourse_var = recourse_var
        self._uncertain_params: list = list(uncertain_params)
        self._prefix = prefix
        self._model = recourse_var.model

        self._y0 = None
        self._Y_cols: list = []          # list of Variable
        self._perturbations: list = []   # list of Expression (ξⱼ = pⱼ - p̄ⱼ)
        self._affine_expr = None
        self._applied = False

    # ── Properties (available after apply()) ─────────────────────────────────

    @property
    def intercept(self):
        """Intercept variable ``y_0`` (same shape as recourse variable)."""
        if self._y0 is None:
            raise RuntimeError("Call apply() before accessing intercept")
        return self._y0

    @property
    def policy_columns(self) -> list:
        """Policy column variables ``[Y_0, Y_1, …]``."""
        if not self._Y_cols:
            raise RuntimeError("Call apply() before accessing policy_columns")
        return list(self._Y_cols)

    @property
    def perturbations(self) -> list:
        """Perturbation expressions ``[ξ_0, ξ_1, …] = [p_0-p̄_0, p_1-p̄_1, …]``."""
        if not self._perturbations:
            raise RuntimeError("Call apply() before accessing perturbations")
        return list(self._perturbations)

    @property
    def affine_expression(self):
        """Full affine expression ``y_0 + ΣY_j·ξ_j``."""
        if self._affine_expr is None:
            raise RuntimeError("Call apply() before accessing affine_expression")
        return self._affine_expr

    @property
    def n_policy_columns(self) -> int:
        """Total number of scalar uncertain parameter components."""
        return sum(
            max(1, int(np.prod(p.value.shape))) for p in self._uncertain_params
        )

    # ── Main method ──────────────────────────────────────────────────────────

    def apply(self) -> None:
        """Substitute the recourse variable with its affine rule.

        This method:

        1. Creates the intercept variable ``y_0`` and policy columns
           ``Y_0, …, Y_{n-1}`` via :meth:`~discopt.modeling.core.Model.continuous`.
        2. Builds the affine expression
           ``y_0 + Y_0·(p_0 - p̄_0) + Y_1·(p_1 - p̄_1) + …``.
        3. Replaces every occurrence of the recourse variable in the model's
           constraints and objective with this expression.
        4. Retires the recourse variable from the model's variable list and
           renumbers remaining variables.

        Can only be called once per instance.
        """
        if self._applied:
            raise RuntimeError("apply() has already been called")

        from discopt.modeling.core import (
            BinaryOp,
            Constant,
            IndexExpression,
            Objective,
            Constraint,
        )

        m = self._model
        y = self._recourse_var
        pfx = self._prefix

        # ── 1. Create intercept y₀ ────────────────────────────────────────────
        y0 = m.continuous(f"{pfx}_intercept", shape=y.shape, lb=-1e20, ub=1e20)
        self._y0 = y0

        # ── 2. Build affine expression ────────────────────────────────────────
        affine_expr = y0  # start with the intercept

        col_idx = 0
        for p in self._uncertain_params:
            nominal = p.value
            nominal_flat = nominal.ravel()
            n_p = len(nominal_flat) if nominal.ndim > 0 else 1

            for j in range(n_p):
                # Policy column Y_j (same shape as y)
                Y_col = m.continuous(
                    f"{pfx}_Y{col_idx}", shape=y.shape, lb=-1e20, ub=1e20
                )
                self._Y_cols.append(Y_col)

                # Perturbation ξⱼ = pⱼ - p̄ⱼ
                if nominal.ndim == 0 or n_p == 1:
                    # Scalar parameter
                    pert = BinaryOp("-", p, Constant(nominal))
                else:
                    # Vector/matrix parameter: index into j-th component
                    pert = BinaryOp(
                        "-",
                        IndexExpression(p, j),
                        Constant(np.array(nominal_flat[j])),
                    )
                self._perturbations.append(pert)

                # Accumulate: affine_expr += Y_j * ξⱼ
                affine_expr = BinaryOp("+", affine_expr, BinaryOp("*", Y_col, pert))
                col_idx += 1

        self._affine_expr = affine_expr

        # ── 3. Substitute y → affine_expr in all constraints and objective ───
        new_constraints = []
        for con in m._constraints:
            new_body = _substitute_var(con.body, y, affine_expr)
            new_constraints.append(
                Constraint(body=new_body, sense=con.sense, rhs=con.rhs, name=con.name)
            )
        m._constraints = new_constraints

        obj = m._objective
        if obj is not None:
            new_obj_expr = _substitute_var(obj.expression, y, affine_expr)
            m._objective = Objective(expression=new_obj_expr, sense=obj.sense)

        # ── 4. Retire y from the model variable list and renumber ────────────
        m._variables = [v for v in m._variables if v is not y]
        for i, v in enumerate(m._variables):
            v._index = i

        self._applied = True

    def summary(self) -> str:
        """Human-readable description of the affine rule."""
        if not self._applied:
            return (
                f"AffineDecisionRule({self._recourse_var.name!r}, "
                f"n_uncertain_scalars={self.n_policy_columns}, not yet applied)"
            )
        lines = [
            f"AffineDecisionRule for '{self._recourse_var.name}'",
            f"  Intercept  : {self._y0.name}  shape={self._y0.shape}",
            f"  Policy cols: {[c.name for c in self._Y_cols]}",
            f"  Perturbations: {[str(p) for p in self._perturbations]}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# DAG variable substitution
# ─────────────────────────────────────────────────────────────────────────────


def _substitute_var(expr, old_var, new_expr):
    """Recursively replace every occurrence of *old_var* in the DAG with *new_expr*.

    Handles all standard node types.  The substitution is exact for Variable
    nodes (matched by identity) and for IndexExpression nodes whose base is
    the variable.

    Parameters
    ----------
    expr : Expression
        Node to rewrite.
    old_var : Variable
        The variable to replace.
    new_expr : Expression
        Replacement expression.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    # ── Variable match ────────────────────────────────────────────────────────
    if isinstance(expr, Variable):
        return new_expr if expr is old_var else expr

    if isinstance(expr, Constant):
        return expr

    # ── IndexExpression: y[i] → new_expr[i] ─────────────────────────────────
    if isinstance(expr, IndexExpression):
        new_base = _substitute_var(expr.base, old_var, new_expr)
        return IndexExpression(new_base, expr.index)

    # ── Arithmetic nodes ─────────────────────────────────────────────────────
    if isinstance(expr, BinaryOp):
        return BinaryOp(
            expr.op,
            _substitute_var(expr.left, old_var, new_expr),
            _substitute_var(expr.right, old_var, new_expr),
        )

    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _substitute_var(expr.operand, old_var, new_expr))

    if isinstance(expr, MatMulExpression):
        return MatMulExpression(
            _substitute_var(expr.left, old_var, new_expr),
            _substitute_var(expr.right, old_var, new_expr),
        )

    if isinstance(expr, FunctionCall):
        new_args = [_substitute_var(a, old_var, new_expr) for a in expr.args]
        return FunctionCall(expr.func_name, *new_args)

    if isinstance(expr, SumExpression):
        return SumExpression(
            _substitute_var(expr.operand, old_var, new_expr), expr.axis
        )

    if isinstance(expr, SumOverExpression):
        return SumOverExpression(
            [_substitute_var(t, old_var, new_expr) for t in expr.terms]
        )

    # Parameter and unknown nodes pass through unchanged.
    return expr
