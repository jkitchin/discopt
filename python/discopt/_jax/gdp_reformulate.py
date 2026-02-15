"""GDP (Generalized Disjunctive Programming) reformulation pass.

Converts indicator constraints, disjunctive constraints, and SOS constraints
into standard MINLP constraints via big-M reformulation.

The reformulation is applied as a preprocessing step before the model is
passed to the NLP evaluator and solver. If no GDP constraints exist, the
original model is returned unchanged (zero overhead).
"""

from __future__ import annotations

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    UnaryOp,
    Variable,
    VarType,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    _SOSConstraint,
    _wrap,
)

_DEFAULT_BIG_M = 1e4


def reformulate_gdp(model: Model, method: str = "big-m") -> Model:
    """Replace GDP constraints with standard MINLP constraints.

    Parameters
    ----------
    model : Model
        Input model potentially containing indicator, disjunctive, or SOS
        constraints.
    method : str, default "big-m"
        Reformulation method for disjunctive constraints:
        ``"big-m"`` (default) or ``"hull"`` (convex hull).
        Indicator and SOS constraints always use big-M regardless.

    Returns
    -------
    Model
        A new model with GDP constraints replaced by equivalent standard
        constraints. If no GDP constraints exist, returns the original
        model unchanged.
    """
    has_gdp = any(
        isinstance(c, (_IndicatorConstraint, _DisjunctiveConstraint, _SOSConstraint))
        for c in model._constraints
    )
    if not has_gdp:
        return model

    new_model = Model(model.name)
    # Copy variables (share the same Variable objects so expressions still work)
    new_model._variables = list(model._variables)
    new_model._parameters = list(model._parameters)
    new_model._objective = model._objective

    # Track auxiliary binaries added by the reformulation
    _aux_counter = [0]

    def _add_aux_binary(prefix: str, shape=()) -> Variable:
        name = f"_gdp_aux_{prefix}_{_aux_counter[0]}"
        _aux_counter[0] += 1
        var = Variable(
            name,
            VarType.BINARY,
            shape if isinstance(shape, tuple) else (shape,),
            0.0,
            1.0,
            new_model,
        )
        new_model._variables.append(var)
        return var

    for c in model._constraints:
        if isinstance(c, _IndicatorConstraint):
            new_cons = _reformulate_indicator(c, new_model)
            new_model._constraints.extend(new_cons)
        elif isinstance(c, _DisjunctiveConstraint):
            if method == "hull":
                new_vars, new_cons = _reformulate_disjunction_hull(c, new_model, _add_aux_binary)
            else:
                new_vars, new_cons = _reformulate_disjunction(c, new_model, _add_aux_binary)
            new_model._constraints.extend(new_cons)
        elif isinstance(c, _SOSConstraint):
            new_cons = _reformulate_sos(c, new_model, _add_aux_binary)
            new_model._constraints.extend(new_cons)
        else:
            # Regular Constraint -- keep as-is
            new_model._constraints.append(c)

    return new_model


def _compute_big_m(
    constraint: Constraint,
    model: Model,
    default: float = _DEFAULT_BIG_M,
) -> float:
    """Compute a big-M bound for a constraint from variable bounds.

    For a constraint ``body sense 0``, we need an upper bound on ``body``
    (for ``<=``), a lower bound (for ``>=``), or both (for ``==``).

    Uses interval arithmetic over the expression tree to get tight M.
    Falls back to *default* if any bound is infinite.

    Parameters
    ----------
    constraint : Constraint
        The constraint whose body expression needs a big-M.
    model : Model
        Model containing variable bound information.
    default : float
        Fallback M value when bounds are infinite.

    Returns
    -------
    float
        A finite big-M value.
    """
    lo, hi = _bound_expression(constraint.body, model)

    # Treat bounds >= 1e15 as effectively infinite (default variable bounds are 1e20)
    _INF_THRESH = 1e15

    if constraint.sense == "<=":
        # body <= 0 is the active constraint; when deactivated we need body <= M
        # so M = upper bound of body
        M = hi if np.isfinite(hi) and abs(hi) < _INF_THRESH else default
    elif constraint.sense == ">=":
        # body >= 0 is active; when deactivated body >= -M
        # so M = -lower_bound of body = abs(lo)
        M = -lo if np.isfinite(lo) and abs(lo) < _INF_THRESH else default
    elif constraint.sense == "==":
        # Need both directions
        M_hi = hi if np.isfinite(hi) and abs(hi) < _INF_THRESH else default
        M_lo = -lo if np.isfinite(lo) and abs(lo) < _INF_THRESH else default
        M = max(M_hi, M_lo)
    else:
        M = default

    # Ensure M is positive and add small safety margin
    return max(abs(M), 1e-8) * 1.01


def _bound_expression(
    expr: Expression,
    model: Model,
) -> tuple[float, float]:
    """Compute interval bounds [lo, hi] for an expression via interval arithmetic.

    Traverses the expression DAG and propagates bounds from variable
    bounds through operations.

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound) of the expression.
    """
    if isinstance(expr, Variable):
        lo = float(np.min(expr.lb))
        hi = float(np.max(expr.ub))
        return lo, hi

    if isinstance(expr, Constant):
        val = float(np.min(expr.value))
        val_max = float(np.max(expr.value))
        return val, val_max

    if isinstance(expr, IndexExpression):
        base_lo, base_hi = _bound_expression(expr.base, model)
        # For indexed expressions on variables, get tighter bounds
        if isinstance(expr.base, Variable):
            v = expr.base
            idx = expr.index
            lb_slice = v.lb[idx] if v.shape != () else v.lb
            ub_slice = v.ub[idx] if v.shape != () else v.ub
            lo = float(np.min(lb_slice))
            hi = float(np.max(ub_slice))
            return lo, hi
        return base_lo, base_hi

    if isinstance(expr, BinaryOp):
        left_lo, left_hi = _bound_expression(expr.left, model)
        right_lo, right_hi = _bound_expression(expr.right, model)

        if expr.op == "+":
            return left_lo + right_lo, left_hi + right_hi
        elif expr.op == "-":
            return left_lo - right_hi, left_hi - right_lo
        elif expr.op == "*":
            products = [
                left_lo * right_lo,
                left_lo * right_hi,
                left_hi * right_lo,
                left_hi * right_hi,
            ]
            return min(products), max(products)
        elif expr.op == "/":
            if right_lo > 0 or right_hi < 0:
                # Divisor doesn't cross zero
                quotients = [
                    left_lo / right_lo,
                    left_lo / right_hi,
                    left_hi / right_lo,
                    left_hi / right_hi,
                ]
                return min(quotients), max(quotients)
            return -np.inf, np.inf
        elif expr.op == "**":
            # Conservative: could be tightened for integer exponents
            if isinstance(expr.right, Constant):
                p = float(expr.right.value)
                if p == 2.0:
                    # x^2 is always >= 0
                    vals = [left_lo**2, left_hi**2]
                    if left_lo <= 0 <= left_hi:
                        return 0.0, max(vals)
                    return min(vals), max(vals)
                if p == int(p) and p > 0:
                    vals = [left_lo**p, left_hi**p]
                    return min(vals), max(vals)
            return -np.inf, np.inf

    if isinstance(expr, UnaryOp):
        arg_lo, arg_hi = _bound_expression(expr.operand, model)
        if expr.op == "neg":
            return -arg_hi, -arg_lo
        elif expr.op == "abs":
            vals = [abs(arg_lo), abs(arg_hi)]
            if arg_lo <= 0 <= arg_hi:
                return 0.0, max(vals)
            return min(vals), max(vals)
        return -np.inf, np.inf

    if isinstance(expr, FunctionCall):
        arg_lo, arg_hi = _bound_expression(expr.args[0], model)
        if expr.func_name == "exp":
            lo = np.exp(arg_lo) if np.isfinite(arg_lo) else 0.0
            hi = np.exp(arg_hi) if np.isfinite(arg_hi) else np.inf
            return lo, hi
        elif expr.func_name == "log":
            lo = np.log(max(arg_lo, 1e-300)) if arg_lo > 0 else -np.inf
            hi = np.log(max(arg_hi, 1e-300)) if arg_hi > 0 else -np.inf
            return lo, hi
        elif expr.func_name == "abs":
            vals = [abs(arg_lo), abs(arg_hi)]
            if arg_lo <= 0 <= arg_hi:
                return 0.0, max(vals)
            return min(vals), max(vals)
        elif expr.func_name == "sqrt":
            lo = np.sqrt(max(arg_lo, 0.0))
            hi = np.sqrt(max(arg_hi, 0.0)) if np.isfinite(arg_hi) else np.inf
            return lo, hi
        elif expr.func_name in ("sin", "cos"):
            # Conservative for trig
            return -1.0, 1.0
        elif expr.func_name == "neg":
            return -arg_hi, -arg_lo

    # Fallback: unknown expression type
    return -np.inf, np.inf


# ── Indicator constraint reformulation ──


def _reformulate_indicator(
    ic: _IndicatorConstraint,
    model: Model,
) -> list[Constraint]:
    """Reformulate an indicator constraint to big-M constraints.

    ``if indicator == active_value then constraint`` becomes:
    - For ``body <= 0``: ``body <= M * (1 - indicator)``
      i.e. ``body - M*(1-indicator) <= 0``
    - For ``body >= 0``: ``body >= -M * (1 - indicator)``
      i.e. ``body + M*(1-indicator) >= 0``
    - For ``body == 0``: both ``<=`` and ``>=`` reformulations

    When ``active_value == 0``, the logic flips: we use ``M * indicator``
    instead of ``M * (1 - indicator)``.
    """
    con = ic.constraint
    y = ic.indicator
    M = _compute_big_m(con, model)

    if ic.active_value == 1:
        # When y=1 constraint is active; when y=0, relaxed by M
        deactivation_expr = _wrap(M) * (_wrap(1.0) - y)
    else:
        # When y=0 constraint is active; when y=1, relaxed by M
        deactivation_expr = _wrap(M) * y

    result = []

    if con.sense == "<=":
        # body <= M*(1-y) => body - M*(1-y) <= 0
        new_body = con.body - deactivation_expr
        result.append(Constraint(body=new_body, sense="<=", rhs=0.0, name=con.name))

    elif con.sense == ">=":
        # body >= -M*(1-y) => body + M*(1-y) >= 0
        new_body = con.body + deactivation_expr
        result.append(Constraint(body=new_body, sense=">=", rhs=0.0, name=con.name))

    elif con.sense == "==":
        # body == 0 when active => -M*(1-y) <= body <= M*(1-y)
        name_le = f"{con.name}_le" if con.name else None
        name_ge = f"{con.name}_ge" if con.name else None
        result.append(
            Constraint(
                body=con.body - deactivation_expr,
                sense="<=",
                rhs=0.0,
                name=name_le,
            )
        )
        result.append(
            Constraint(
                body=con.body + deactivation_expr,
                sense=">=",
                rhs=0.0,
                name=name_ge,
            )
        )

    return result


# ── Disjunctive constraint reformulation ──


def _reformulate_disjunction(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_aux_binary,
) -> tuple[list[Variable], list[Constraint]]:
    """Reformulate a disjunction via big-M.

    For ``either_or([[g1<=0, g2<=0], [h1<=0, h2<=0]])``:
    1. Introduce binary selectors y_0, y_1 with y_0 + y_1 == 1
    2. For each disjunct k and constraint j in disjunct k:
       g_j(x) <= M_j * (1 - y_k)

    Returns new variables and constraints.
    """
    n_disjuncts = len(dc.disjuncts)
    new_vars = []
    new_cons = []

    # Create selector binaries
    selectors = []
    for k in range(n_disjuncts):
        y_k = add_aux_binary(f"disj_{dc.name or 'anon'}_{k}")
        selectors.append(y_k)
        new_vars.append(y_k)

    # Sum of selectors == 1 (exactly one disjunct active)
    if n_disjuncts == 2:
        sum_expr = selectors[0] + selectors[1]
    else:
        sum_expr = selectors[0]
        for k in range(1, n_disjuncts):
            sum_expr = sum_expr + selectors[k]

    new_cons.append(
        Constraint(
            body=sum_expr - _wrap(1.0),
            sense="==",
            rhs=0.0,
            name=f"_gdp_select_{dc.name}" if dc.name else None,
        )
    )

    # Big-M reformulation for each constraint in each disjunct
    for k, disjunct in enumerate(dc.disjuncts):
        y_k = selectors[k]
        for j, con in enumerate(disjunct):
            M = _compute_big_m(con, model)
            deactivation = _wrap(M) * (_wrap(1.0) - y_k)

            if con.sense == "<=":
                new_body = con.body - deactivation
                new_cons.append(
                    Constraint(
                        body=new_body,
                        sense="<=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}" if dc.name else None,
                    )
                )
            elif con.sense == ">=":
                new_body = con.body + deactivation
                new_cons.append(
                    Constraint(
                        body=new_body,
                        sense=">=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}" if dc.name else None,
                    )
                )
            elif con.sense == "==":
                new_cons.append(
                    Constraint(
                        body=con.body - deactivation,
                        sense="<=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}_le" if dc.name else None,
                    )
                )
                new_cons.append(
                    Constraint(
                        body=con.body + deactivation,
                        sense=">=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}_ge" if dc.name else None,
                    )
                )

    return new_vars, new_cons


# ── Hull reformulation ──


def _reformulate_disjunction_hull(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_aux_binary,
    eps: float = 1e-8,
) -> tuple[list[Variable], list[Constraint]]:
    """Reformulate a disjunction via convex hull relaxation.

    For each original variable x_j appearing in the disjunction, creates
    disaggregated copies v_{j,k} for each disjunct k, with:

    - Selector binaries y_k with sum(y_k) == 1
    - Aggregation: x_j == sum_k(v_{j,k})
    - Bound linking: dlb_k * y_k <= v_{j,k} <= dub_k * y_k
    - Linear constraints: substitute x -> v_k, multiply RHS by y_k
    - Nonlinear constraints: perspective form with clamped y_k

    Parameters
    ----------
    dc : _DisjunctiveConstraint
        The disjunction to reformulate.
    model : Model
        Model with variable bound information.
    add_aux_binary : callable
        Factory for creating auxiliary binary variables.
    eps : float
        Small positive constant for clamping y_k in perspective functions.

    Returns
    -------
    tuple of (list[Variable], list[Constraint])
        New variables and constraints.
    """
    n_disjuncts = len(dc.disjuncts)
    new_vars: list[Variable] = []
    new_cons: list[Constraint] = []
    prefix = dc.name or "anon"

    # --- Selector binaries y_k with sum == 1 ---
    selectors: list[Variable] = []
    for k in range(n_disjuncts):
        y_k = add_aux_binary(f"hull_{prefix}_{k}")
        selectors.append(y_k)
        new_vars.append(y_k)

    if n_disjuncts == 1:
        sum_sel = selectors[0]
    else:
        sum_sel = selectors[0]
        for k in range(1, n_disjuncts):
            sum_sel = sum_sel + selectors[k]

    new_cons.append(
        Constraint(
            body=sum_sel - _wrap(1.0),
            sense="==",
            rhs=0.0,
            name=f"_hull_select_{prefix}",
        )
    )

    # --- Collect all variables across all disjuncts ---
    all_vars: dict[str, Variable] = {}
    for disjunct in dc.disjuncts:
        for con in disjunct:
            all_vars.update(_collect_variables(con.body))

    # --- Per-disjunct bounds and disaggregated variables ---
    # disjunct_bounds[k][var_name] = (dlb, dub)
    disjunct_bounds: list[dict[str, tuple[float, float]]] = []
    for k, disjunct in enumerate(dc.disjuncts):
        db = _extract_disjunct_bounds(disjunct, model)
        # Fill in global bounds for variables not constrained in this disjunct
        for vname, var in all_vars.items():
            if vname not in db:
                db[vname] = (float(np.min(var.lb)), float(np.max(var.ub)))
        disjunct_bounds.append(db)

    # disagg[k][var_name] = disaggregated Variable v_{j,k}
    disagg: list[dict[str, Variable]] = []
    for k in range(n_disjuncts):
        disagg_k: dict[str, Variable] = {}
        for vname, var in all_vars.items():
            dlb, dub = disjunct_bounds[k][vname]
            # Disaggregated bounds: [min(dlb, 0), max(dub, 0)]
            v_lb = min(dlb, 0.0)
            v_ub = max(dub, 0.0)
            v_jk = Variable(
                f"_hull_{prefix}_v_{vname}_{k}",
                VarType.CONTINUOUS,
                var.shape,
                v_lb,
                v_ub,
                model,
            )
            disagg_k[vname] = v_jk
            new_vars.append(v_jk)
            model._variables.append(v_jk)
        disagg.append(disagg_k)

    # --- Aggregation: x_j == sum_k(v_{j,k}) ---
    for vname, var in all_vars.items():
        agg_expr: Expression = disagg[0][vname]
        for k in range(1, n_disjuncts):
            agg_expr = agg_expr + disagg[k][vname]
        new_cons.append(
            Constraint(
                body=var - agg_expr,
                sense="==",
                rhs=0.0,
                name=f"_hull_agg_{prefix}_{vname}",
            )
        )

    # --- Bound linking: dlb * y_k <= v_{j,k} <= dub * y_k ---
    for k in range(n_disjuncts):
        y_k = selectors[k]
        for vname in all_vars:
            dlb, dub = disjunct_bounds[k][vname]
            v_jk = disagg[k][vname]
            # v_{j,k} <= dub * y_k  =>  v_{j,k} - dub * y_k <= 0
            new_cons.append(
                Constraint(
                    body=v_jk - _wrap(dub) * y_k,
                    sense="<=",
                    rhs=0.0,
                    name=f"_hull_ub_{prefix}_{vname}_{k}",
                )
            )
            # v_{j,k} >= dlb * y_k  =>  v_{j,k} - dlb * y_k >= 0
            new_cons.append(
                Constraint(
                    body=v_jk - _wrap(dlb) * y_k,
                    sense=">=",
                    rhs=0.0,
                    name=f"_hull_lb_{prefix}_{vname}_{k}",
                )
            )

    # --- Constraint reformulation per disjunct ---
    for k, disjunct in enumerate(dc.disjuncts):
        y_k = selectors[k]

        for j, con in enumerate(disjunct):
            cname = f"_hull_{prefix}_d{k}_c{j}"

            if _is_linear(con.body):
                # Linear f(x) = a^T x + b: hull gives a^T v_k + b * y_k
                # Substitute vars -> disagg vars and scale constants by y_k
                hull_body = _hull_linear_substitute(
                    con.body, {vname: disagg[k][vname] for vname in all_vars}, y_k
                )
                rhs_expr = _wrap(con.rhs) * y_k
            else:
                # Nonlinear: perspective form with clamped y_k
                # f(v_k / y_clamp) * y_clamp where y_clamp = y_k + eps
                y_clamp = y_k + _wrap(eps)
                persp_map = {vname: disagg[k][vname] / y_clamp for vname in all_vars}
                subst_body = _substitute_vars(con.body, persp_map)
                hull_body = subst_body * y_clamp
                rhs_expr = _wrap(con.rhs) * y_k

            if con.sense == "<=":
                new_cons.append(
                    Constraint(
                        body=hull_body - rhs_expr,
                        sense="<=",
                        rhs=0.0,
                        name=cname,
                    )
                )
            elif con.sense == ">=":
                new_cons.append(
                    Constraint(
                        body=hull_body - rhs_expr,
                        sense=">=",
                        rhs=0.0,
                        name=cname,
                    )
                )
            elif con.sense == "==":
                new_cons.append(
                    Constraint(
                        body=hull_body - rhs_expr,
                        sense="<=",
                        rhs=0.0,
                        name=f"{cname}_le",
                    )
                )
                new_cons.append(
                    Constraint(
                        body=hull_body - rhs_expr,
                        sense=">=",
                        rhs=0.0,
                        name=f"{cname}_ge",
                    )
                )

    return new_vars, new_cons


def _hull_linear_substitute(
    expr: Expression,
    var_map: dict[str, Expression],
    y_k: Expression,
) -> Expression:
    """Substitute variables and scale constants by y_k for hull linear reform.

    For a linear expression ``a*x + b``, returns ``a*v_k + b*y_k`` where
    ``v_k`` is the disaggregated variable from *var_map*.
    """
    if isinstance(expr, Variable):
        return var_map.get(expr.name, expr)
    if isinstance(expr, Constant):
        result: Expression = expr * y_k
        return result
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable) and expr.base.name in var_map:
            new_base = var_map[expr.base.name]
            return IndexExpression(new_base, expr.index)
        return expr
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            new_left = _hull_linear_substitute(expr.left, var_map, y_k)
            new_right = _hull_linear_substitute(expr.right, var_map, y_k)
            return BinaryOp(expr.op, new_left, new_right)
        if expr.op == "*":
            # For linear expressions, one side must be a constant
            if isinstance(expr.left, Constant):
                # const * linear_expr → const * hull_substitute(linear_expr)
                new_right = _hull_linear_substitute(expr.right, var_map, y_k)
                return BinaryOp("*", expr.left, new_right)
            if isinstance(expr.right, Constant):
                new_left = _hull_linear_substitute(expr.left, var_map, y_k)
                return BinaryOp("*", new_left, expr.right)
        if expr.op == "/":
            if isinstance(expr.right, Constant):
                new_left = _hull_linear_substitute(expr.left, var_map, y_k)
                return BinaryOp("/", new_left, expr.right)
    if isinstance(expr, UnaryOp):
        if expr.op in ("neg", "-"):
            new_operand = _hull_linear_substitute(expr.operand, var_map, y_k)
            return UnaryOp(expr.op, new_operand)
    # Fallback
    return _substitute_vars(expr, var_map)


# ── Hull reformulation helpers ──


def _collect_variables(expr: Expression) -> dict[str, Variable]:
    """Walk expression DAG and return Variable nodes keyed by name.

    Variables are deduplicated by name (Variable is not hashable due to
    ``__eq__`` returning a Constraint).
    """
    found: dict[str, Variable] = {}

    def _walk(e: Expression) -> None:
        if isinstance(e, Variable):
            found[e.name] = e
        elif isinstance(e, IndexExpression):
            if isinstance(e.base, Variable):
                found[e.base.name] = e.base
            else:
                _walk(e.base)
        elif isinstance(e, BinaryOp):
            _walk(e.left)
            _walk(e.right)
        elif isinstance(e, UnaryOp):
            _walk(e.operand)
        elif isinstance(e, FunctionCall):
            for arg in e.args:
                _walk(arg)

    _walk(expr)
    return found


def _is_linear(expr: Expression) -> bool:
    """Check if expression is linear in its variables (conservative).

    Returns True only for expressions involving +, -, and * where at least
    one operand of every multiplication is a constant. Returns False for
    any nonlinear operation (**, FunctionCall, bilinear terms).
    False negatives are safe (they cause fallback to perspective functions).
    """
    if isinstance(expr, (Variable, Constant)):
        return True
    if isinstance(expr, IndexExpression):
        return True
    if isinstance(expr, UnaryOp):
        if expr.op in ("neg", "-"):
            return _is_linear(expr.operand)
        return False
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            return _is_linear(expr.left) and _is_linear(expr.right)
        if expr.op == "*":
            # Linear only if at least one side is a constant
            if isinstance(expr.left, Constant) or isinstance(expr.right, Constant):
                return _is_linear(expr.left) and _is_linear(expr.right)
            return False
        if expr.op == "/":
            # x / constant is linear
            if isinstance(expr.right, Constant):
                return _is_linear(expr.left)
            return False
        # ** is always nonlinear
        return False
    if isinstance(expr, FunctionCall):
        return False
    return False


def _substitute_vars(
    expr: Expression,
    var_map: dict[str, Expression],
) -> Expression:
    """Return a copy of *expr* with variables renamed per *var_map*.

    *var_map* maps variable **names** to replacement expressions.
    Nodes not in the map are returned unchanged.
    """
    if isinstance(expr, Variable):
        return var_map.get(expr.name, expr)
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable) and expr.base.name in var_map:
            new_base = var_map[expr.base.name]
            return IndexExpression(new_base, expr.index)
        new_base = _substitute_vars(expr.base, var_map)
        if new_base is expr.base:
            return expr
        return IndexExpression(new_base, expr.index)
    if isinstance(expr, BinaryOp):
        new_left = _substitute_vars(expr.left, var_map)
        new_right = _substitute_vars(expr.right, var_map)
        if new_left is expr.left and new_right is expr.right:
            return expr
        return BinaryOp(expr.op, new_left, new_right)
    if isinstance(expr, UnaryOp):
        new_operand = _substitute_vars(expr.operand, var_map)
        if new_operand is expr.operand:
            return expr
        return UnaryOp(expr.op, new_operand)
    if isinstance(expr, FunctionCall):
        new_args = tuple(_substitute_vars(a, var_map) for a in expr.args)
        if all(n is o for n, o in zip(new_args, expr.args)):
            return expr
        return FunctionCall(expr.func_name, *new_args)
    # Constant or unknown — pass through
    return expr


def _extract_disjunct_bounds(
    disjunct: list[Constraint],
    model: Model,
) -> dict[str, tuple[float, float]]:
    """Extract per-variable bounds implied by simple constraints in a disjunct.

    Scans for single-variable bound patterns (``x <= c``, ``x >= c``,
    ``x == c``) and returns the tightest implied bounds, starting from the
    variable's global bounds.

    Parameters
    ----------
    disjunct : list[Constraint]
        Constraints in one arm of a disjunction.
    model : Model
        Model with global variable bound information.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from variable name to (lb, ub).
    """
    bounds: dict[str, list[float]] = {}

    def _init_var(v: Variable) -> None:
        if v.name not in bounds:
            bounds[v.name] = [float(np.min(v.lb)), float(np.max(v.ub))]

    for con in disjunct:
        # Identify single-variable body, possibly with constant offset
        body = con.body
        var: Variable | None = None
        offset = 0.0  # effective constraint: var (sense) rhs - offset

        if isinstance(body, Variable):
            var = body
        elif isinstance(body, IndexExpression) and isinstance(body.base, Variable):
            var = body.base
        elif isinstance(body, BinaryOp) and body.op == "-":
            # Pattern: var - const <= 0  =>  var <= const
            if isinstance(body.left, Variable) and isinstance(body.right, Constant):
                var = body.left
                offset = -float(body.right.value)
            elif isinstance(body.left, (Variable, IndexExpression)):
                lvar = body.left if isinstance(body.left, Variable) else None
                if lvar is None and isinstance(body.left, IndexExpression):
                    lvar = body.left.base if isinstance(body.left.base, Variable) else None
                if lvar is not None and isinstance(body.right, Constant):
                    var = lvar
                    offset = -float(body.right.value)
            # Pattern: const - var <= 0  =>  var >= const
            elif isinstance(body.left, Constant) and isinstance(body.right, Variable):
                var = body.right
                # const - var (sense) 0  =>  -var (sense) -const  =>  var (flip) const
                # We handle this by negating and flipping sense below
                offset = float(body.left.value)
                # For "const - var <= 0" => var >= const
                # For "const - var >= 0" => var <= const
                _init_var(var)
                effective_rhs = offset  # the constant value
                if con.sense == "<=":
                    # const - var <= 0 => var >= const
                    bounds[var.name][0] = max(bounds[var.name][0], effective_rhs)
                elif con.sense == ">=":
                    # const - var >= 0 => var <= const
                    bounds[var.name][1] = min(bounds[var.name][1], effective_rhs)
                elif con.sense == "==":
                    bounds[var.name][0] = max(bounds[var.name][0], effective_rhs)
                    bounds[var.name][1] = min(bounds[var.name][1], effective_rhs)
                continue
        elif isinstance(body, BinaryOp) and body.op == "+":
            # Pattern: var + const <= 0  =>  var <= -const
            if isinstance(body.left, Variable) and isinstance(body.right, Constant):
                var = body.left
                offset = float(body.right.value)

        if var is None:
            continue

        # effective: var (sense) rhs - offset
        rhs = con.rhs - offset
        _init_var(var)

        if con.sense == "<=":
            bounds[var.name][1] = min(bounds[var.name][1], rhs)
        elif con.sense == ">=":
            bounds[var.name][0] = max(bounds[var.name][0], rhs)
        elif con.sense == "==":
            bounds[var.name][0] = max(bounds[var.name][0], rhs)
            bounds[var.name][1] = min(bounds[var.name][1], rhs)

    return {name: (lb_ub[0], lb_ub[1]) for name, lb_ub in bounds.items()}


# ── SOS constraint reformulation ──


def _reformulate_sos(
    sc: _SOSConstraint,
    model: Model,
    add_aux_binary,
) -> list[Constraint]:
    """Reformulate SOS Type 1 or Type 2 constraint via binary indicators.

    SOS1: At most one variable x_i can be nonzero.
      - Introduce binary z_i for each variable
      - x_i <= ub_i * z_i (linking upper bound)
      - x_i >= lb_i * z_i (linking lower bound, handles negative vars)
      - sum(z_i) <= 1

    SOS2: At most two *adjacent* variables can be nonzero.
      - Introduce binary z_i for each variable
      - x_i <= ub_i * z_i
      - x_i >= lb_i * z_i
      - sum(z_i) <= 2
      - z_i + z_j <= 1 for all non-adjacent pairs |i-j| > 1
    """
    new_cons = []
    n = len(sc.variables)

    # Create indicator binaries
    indicators = []
    for i in range(n):
        z_i = add_aux_binary(f"sos{sc.sos_type}_{sc.name or 'anon'}_{i}")
        indicators.append(z_i)

    # Linking constraints: x_i <= ub_i * z_i and x_i >= lb_i * z_i
    for i in range(n):
        v = sc.variables[i]
        # Extract bounds — v may be a Variable or IndexExpression
        lo_v, hi_v = _bound_expression(v, model)
        ub_val = hi_v
        lb_val = lo_v

        # Clamp to finite values for big-M
        if not np.isfinite(ub_val):
            ub_val = _DEFAULT_BIG_M
        if not np.isfinite(lb_val):
            lb_val = -_DEFAULT_BIG_M

        # x_i - ub_i * z_i <= 0
        new_cons.append(
            Constraint(
                body=v - _wrap(ub_val) * indicators[i],
                sense="<=",
                rhs=0.0,
                name=f"_sos{sc.sos_type}_{sc.name}_ub_{i}" if sc.name else None,
            )
        )

        # x_i - lb_i * z_i >= 0  =>  x_i >= lb_i * z_i
        if lb_val < 0:
            new_cons.append(
                Constraint(
                    body=v - _wrap(lb_val) * indicators[i],
                    sense=">=",
                    rhs=0.0,
                    name=f"_sos{sc.sos_type}_{sc.name}_lb_{i}" if sc.name else None,
                )
            )

    if sc.sos_type == 1:
        # sum(z_i) <= 1
        if n == 1:
            sum_z = indicators[0]
        else:
            sum_z = indicators[0]
            for i in range(1, n):
                sum_z = sum_z + indicators[i]
        new_cons.append(
            Constraint(
                body=sum_z - _wrap(1.0),
                sense="<=",
                rhs=0.0,
                name=f"_sos1_{sc.name}_sum" if sc.name else None,
            )
        )

    elif sc.sos_type == 2:
        # sum(z_i) <= 2
        if n >= 2:
            sum_z = indicators[0]
            for i in range(1, n):
                sum_z = sum_z + indicators[i]
            new_cons.append(
                Constraint(
                    body=sum_z - _wrap(2.0),
                    sense="<=",
                    rhs=0.0,
                    name=f"_sos2_{sc.name}_sum" if sc.name else None,
                )
            )

            # Non-adjacency: z_i + z_j <= 1 for |i - j| > 1
            for i in range(n):
                for j in range(i + 2, n):
                    new_cons.append(
                        Constraint(
                            body=indicators[i] + indicators[j] - _wrap(1.0),
                            sense="<=",
                            rhs=0.0,
                            name=(f"_sos2_{sc.name}_nonadj_{i}_{j}" if sc.name else None),
                        )
                    )

    return new_cons
