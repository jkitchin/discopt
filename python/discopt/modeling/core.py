"""
discopt Modeling API

A clean, expressive Python API for formulating Mixed-Integer Nonlinear Programs.
Designed for:

- Readability: models look like the math
- JAX compatibility: expressions are traceable and JIT-compilable
- Rust interop: expression graphs map to the Rust DAG for structure detection
- LLM integration: the API doubles as the tool-calling schema for the formulation agent

Example::

    import discopt.modeling as dm

    m = dm.Model("blending")
    x = m.continuous("flow", shape=(3,), lb=0, ub=100)
    y = m.binary("active", shape=(2,))

    m.minimize(cost @ x + fixed_cost @ y)
    m.subject_to(A @ x <= b, name="mass_balance")
    m.subject_to(x[0] * x[1] <= 50 * y[0], name="bilinear_coupling")

    result = m.solve()
"""

from __future__ import annotations

import builtins as _builtins
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Iterator,
    Optional,
    Sequence,
    Union,
)

import numpy as np

builtins_sum = _builtins.sum

# ─────────────────────────────────────────────────────────────
# Variable Types
# ─────────────────────────────────────────────────────────────


class VarType(Enum):
    """
    Variable domain type.

    Attributes
    ----------
    CONTINUOUS : str
        Real-valued variable (default bounds: ``[-9.999e19, 9.999e19]``).
    BINARY : str
        Binary variable restricted to ``{0, 1}``.
    INTEGER : str
        General integer variable (default bounds: ``[0, 1e6]``).
    """

    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"


# ─────────────────────────────────────────────────────────────
# Expression System
#
# All operations on Variables produce Expression objects that
# build a DAG. This DAG is later compiled to:
#   (1) A Rust-side expression graph for structure detection
#   (2) A JAX-traceable function for evaluation and autodiff
# ─────────────────────────────────────────────────────────────


class Expression:
    """
    Base class for all mathematical expressions in a discopt model.

    Supports standard arithmetic (``+``, ``-``, ``*``, ``/``, ``**``),
    comparison operators (``<=``, ``>=``, ``==``) that produce
    :class:`Constraint` objects, and mathematical functions via the
    ``discopt.modeling`` namespace (``dm.exp``, ``dm.log``, ``dm.sin``, etc.).

    Expressions are lazy -- they build a directed acyclic graph (DAG) that
    is later compiled to a JAX-traceable function for evaluation and autodiff,
    and to a Rust-side expression graph for structure detection.

    Notes
    -----
    Do not instantiate ``Expression`` directly. Expressions are created by
    declaring variables with :meth:`Model.continuous`, :meth:`Model.binary`,
    or :meth:`Model.integer`, and then combining them with arithmetic
    operators and math functions.
    """

    def __add__(self, other):
        return BinaryOp("+", self, _wrap(other))

    def __radd__(self, other):
        return BinaryOp("+", _wrap(other), self)

    def __sub__(self, other):
        return BinaryOp("-", self, _wrap(other))

    def __rsub__(self, other):
        return BinaryOp("-", _wrap(other), self)

    def __mul__(self, other):
        return BinaryOp("*", self, _wrap(other))

    def __rmul__(self, other):
        return BinaryOp("*", _wrap(other), self)

    def __truediv__(self, other):
        return BinaryOp("/", self, _wrap(other))

    def __rtruediv__(self, other):
        return BinaryOp("/", _wrap(other), self)

    def __pow__(self, other):
        return BinaryOp("**", self, _wrap(other))

    def __rpow__(self, other):
        return BinaryOp("**", _wrap(other), self)

    def __neg__(self):
        return UnaryOp("neg", self)

    def __abs__(self):
        return UnaryOp("abs", self)

    # ── Comparison operators produce Constraints, not booleans ──

    def __le__(self, other):
        return Constraint(self - _wrap(other), sense="<=", rhs=0.0)

    def __ge__(self, other):
        return Constraint(_wrap(other) - self, sense="<=", rhs=0.0)

    def __eq__(self, other):
        return Constraint(self - _wrap(other), sense="==", rhs=0.0)

    # ── Indexing for array variables ──

    def __getitem__(self, idx):
        return IndexExpression(self, idx)

    # ── Matrix operations ──

    def __matmul__(self, other):
        return MatMulExpression(self, _wrap(other))

    def __rmatmul__(self, other):
        return MatMulExpression(_wrap(other), self)

    def _repr_latex_(self):
        """Jupyter/IPython LaTeX rendering."""
        return f"${self}$"


class Constant(Expression):
    """A numeric constant in the expression DAG."""

    def __init__(self, value: Union[float, int, np.ndarray]):
        if isinstance(value, np.ndarray):
            self.value = value.astype(np.float64)
        else:
            self.value = np.asarray(value, dtype=np.float64)

    def __repr__(self):
        if self.value.ndim == 0:
            return f"{float(self.value):.6g}"
        return f"Constant({self.value.shape})"


class Variable(Expression):
    """
    A decision variable in the optimization problem.

    Variables are created through :meth:`Model.continuous`,
    :meth:`Model.binary`, or :meth:`Model.integer` -- not directly.

    Attributes
    ----------
    name : str
        Unique variable name within the model.
    var_type : VarType
        One of CONTINUOUS, BINARY, or INTEGER.
    shape : tuple of int
        Shape of the variable (``()`` for scalar).
    lb : numpy.ndarray
        Element-wise lower bounds.
    ub : numpy.ndarray
        Element-wise upper bounds.
    """

    def __init__(
        self,
        name: str,
        var_type: VarType,
        shape: tuple[int, ...],
        lb: Union[float, np.ndarray],
        ub: Union[float, np.ndarray],
        model: "Model",
    ):
        self.name = name
        self.var_type = var_type
        self.shape = shape
        self.lb = np.broadcast_to(np.asarray(lb, dtype=np.float64), shape)
        self.ub = np.broadcast_to(np.asarray(ub, dtype=np.float64), shape)
        self.model = model
        self._index = len(model._variables)  # Position in flat variable vector

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def __hash__(self):
        return id(self)

    def __repr__(self):
        if self.shape == () or self.shape == (1,):
            return self.name
        return f"{self.name}{list(self.shape)}"


class IndexExpression(Expression):
    """Result of indexing into an array variable: x[i] or x[0, 1]."""

    def __init__(self, base: Expression, index):
        self.base = base
        self.index = index

    def __repr__(self):
        return f"{self.base}[{self.index}]"


class BinaryOp(Expression):
    """Binary operation: a op b."""

    def __init__(self, op: str, left: Expression, right: Expression):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class UnaryOp(Expression):
    """Unary operation: op(a)."""

    def __init__(self, op: str, operand: Expression):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"{self.op}({self.operand})"


class FunctionCall(Expression):
    """Named function call: exp(x), log(x), sin(x), etc."""

    def __init__(self, func_name: str, *args: Expression):
        self.func_name = func_name
        self.args = args

    def __repr__(self):
        arg_str = ", ".join(str(a) for a in self.args)
        return f"{self.func_name}({arg_str})"


class MatMulExpression(Expression):
    """Matrix multiplication: A @ x."""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} @ {self.right})"


class SumExpression(Expression):
    """Summation over expressions."""

    def __init__(self, operand: Expression, axis: Optional[int] = None):
        self.operand = operand
        self.axis = axis

    def __repr__(self):
        if self.axis is not None:
            return f"sum({self.operand}, axis={self.axis})"
        return f"sum({self.operand})"


class SumOverExpression(Expression):
    """Sum of expr(i) for i in index_set — the indexed summation pattern."""

    def __init__(self, terms: list[Expression]):
        self.terms = terms

    def __repr__(self):
        return f"Σ[{len(self.terms)} terms]"


def _wrap(x) -> Expression:
    """Convert a Python scalar or numpy array to a Constant expression."""
    if isinstance(x, Expression):
        return x
    return Constant(x)


# ─────────────────────────────────────────────────────────────
# Mathematical Functions (dm.exp, dm.log, dm.sin, etc.)
# ─────────────────────────────────────────────────────────────


def exp(x: Union[Expression, float]) -> Expression:
    """
    Exponential function.

    Parameters
    ----------
    x : Expression or float
        Input expression.

    Returns
    -------
    Expression
        Expression representing ``e**x``.
    """
    return FunctionCall("exp", _wrap(x))


def log(x: Union[Expression, float]) -> Expression:
    """
    Natural logarithm.

    Parameters
    ----------
    x : Expression or float
        Input expression (must be positive at evaluation).

    Returns
    -------
    Expression
        Expression representing ``ln(x)``.
    """
    return FunctionCall("log", _wrap(x))


def log2(x: Union[Expression, float]) -> Expression:
    """
    Base-2 logarithm.

    Parameters
    ----------
    x : Expression or float
        Input expression (must be positive at evaluation).

    Returns
    -------
    Expression
    """
    return FunctionCall("log2", _wrap(x))


def log10(x: Union[Expression, float]) -> Expression:
    """
    Base-10 logarithm.

    Parameters
    ----------
    x : Expression or float
        Input expression (must be positive at evaluation).

    Returns
    -------
    Expression
    """
    return FunctionCall("log10", _wrap(x))


def sqrt(x: Union[Expression, float]) -> Expression:
    """
    Square root.

    Parameters
    ----------
    x : Expression or float
        Input expression (must be non-negative at evaluation).

    Returns
    -------
    Expression
    """
    return FunctionCall("sqrt", _wrap(x))


def sin(x: Union[Expression, float]) -> Expression:
    """
    Sine.

    Parameters
    ----------
    x : Expression or float
        Input expression (radians).

    Returns
    -------
    Expression
    """
    return FunctionCall("sin", _wrap(x))


def cos(x: Union[Expression, float]) -> Expression:
    """
    Cosine.

    Parameters
    ----------
    x : Expression or float
        Input expression (radians).

    Returns
    -------
    Expression
    """
    return FunctionCall("cos", _wrap(x))


def tan(x: Union[Expression, float]) -> Expression:
    """
    Tangent.

    Parameters
    ----------
    x : Expression or float
        Input expression (radians).

    Returns
    -------
    Expression
    """
    return FunctionCall("tan", _wrap(x))


def asinh(x: Union[Expression, float]) -> Expression:
    """Inverse hyperbolic sine."""
    return FunctionCall("asinh", _wrap(x))


def acosh(x: Union[Expression, float]) -> Expression:
    """Inverse hyperbolic cosine (x >= 1)."""
    return FunctionCall("acosh", _wrap(x))


def atanh(x: Union[Expression, float]) -> Expression:
    """Inverse hyperbolic tangent (-1 < x < 1)."""
    return FunctionCall("atanh", _wrap(x))


def erf(x: Union[Expression, float]) -> Expression:
    """Gauss error function."""
    return FunctionCall("erf", _wrap(x))


def log1p(x: Union[Expression, float]) -> Expression:
    """Numerically stable log(1 + x) (x > -1)."""
    return FunctionCall("log1p", _wrap(x))


def tanh(x: Union[Expression, float]) -> Expression:
    """
    Hyperbolic tangent.

    Parameters
    ----------
    x : Expression or float
        Input expression.

    Returns
    -------
    Expression
        Expression representing ``tanh(x)``.
    """
    return FunctionCall("tanh", _wrap(x))


def sigmoid(x: Union[Expression, float]) -> Expression:
    """
    Logistic sigmoid: ``1 / (1 + exp(-x))``.

    Parameters
    ----------
    x : Expression or float
        Input expression.

    Returns
    -------
    Expression
        Expression representing ``sigmoid(x)``, valued in ``(0, 1)``.
    """
    return FunctionCall("sigmoid", _wrap(x))


def softplus(x: Union[Expression, float]) -> Expression:
    """
    Softplus: ``log(1 + exp(x))``.

    A smooth approximation of ReLU.

    Parameters
    ----------
    x : Expression or float
        Input expression.

    Returns
    -------
    Expression
        Expression representing ``softplus(x)``, always positive.
    """
    return FunctionCall("softplus", _wrap(x))


def abs_(x: Union[Expression, float]) -> Expression:
    """
    Absolute value.

    Exported as ``dm.abs`` in the ``discopt.modeling`` namespace.

    Parameters
    ----------
    x : Expression or float
        Input expression.

    Returns
    -------
    Expression
    """
    return FunctionCall("abs", _wrap(x))


def sign(x: Union[Expression, float]) -> Expression:
    """
    Sign function: returns -1, 0, or +1.

    Parameters
    ----------
    x : Expression or float
        Input expression.

    Returns
    -------
    Expression
    """
    return FunctionCall("sign", _wrap(x))


def minimum(x: Union[Expression, float], y: Union[Expression, float]) -> Expression:
    """
    Element-wise minimum of two expressions.

    Parameters
    ----------
    x : Expression or float
        First operand.
    y : Expression or float
        Second operand.

    Returns
    -------
    Expression
    """
    return FunctionCall("min", _wrap(x), _wrap(y))


def maximum(x: Union[Expression, float], y: Union[Expression, float]) -> Expression:
    """
    Element-wise maximum of two expressions.

    Parameters
    ----------
    x : Expression or float
        First operand.
    y : Expression or float
        Second operand.

    Returns
    -------
    Expression
    """
    return FunctionCall("max", _wrap(x), _wrap(y))


# ─────────────────────────────────────────────────────────────
# Aggregation Functions
# ─────────────────────────────────────────────────────────────


def sum(
    x: Union[Expression, list, Callable],
    *,
    over: Optional[Sequence] = None,
    axis: Optional[int] = None,
) -> Expression:
    """
    Summation over expressions.

    Supports three calling patterns:

    Parameters
    ----------
    x : Expression, list of Expression, or callable
        Expression to sum, list of terms, or a callable ``f(i)`` returning
        an expression for each index ``i`` in *over*.
    over : sequence, optional
        Index set for indexed summation (requires *x* to be callable).
    axis : int, optional
        Axis along which to sum (for array expressions).

    Returns
    -------
    Expression

    Examples
    --------
    >>> dm.sum(x)                                  # sum all elements
    >>> dm.sum(x, axis=0)                          # sum along axis 0
    >>> dm.sum(lambda i: cost[i] * x[i], over=range(n))  # indexed sum
    """
    if over is not None and callable(x):
        # Indexed summation: dm.sum(lambda i: expr(i), over=index_set)
        terms = [_wrap(x(i)) for i in over]
        return SumOverExpression(terms)
    if isinstance(x, list):
        terms = [_wrap(t) for t in x]
        return SumOverExpression(terms)
    return SumExpression(_wrap(x), axis=axis)


def prod(x: Union[Expression, list, Callable], *, over: Optional[Sequence] = None) -> Expression:
    """
    Product over expressions, analogous to :func:`sum`.

    Parameters
    ----------
    x : Expression, list of Expression, or callable
        Expression to multiply, list of terms, or a callable ``f(i)``
        returning an expression for each index ``i`` in *over*.
    over : sequence, optional
        Index set for indexed product (requires *x* to be callable).

    Returns
    -------
    Expression
    """
    if over is not None and callable(x):
        terms = [_wrap(x(i)) for i in over]
        result = terms[0]
        for t in terms[1:]:
            result = result * t
        return result
    return FunctionCall("prod", _wrap(x))


def norm(x: Expression, ord: int = 2) -> Expression:
    """
    Vector norm.

    Parameters
    ----------
    x : Expression
        Input vector expression.
    ord : int, default 2
        Norm order (e.g. 1 for L1-norm, 2 for L2-norm).

    Returns
    -------
    Expression
    """
    return FunctionCall(f"norm{ord}", _wrap(x))


# ─────────────────────────────────────────────────────────────
# Constraints
# ─────────────────────────────────────────────────────────────


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "=="


@dataclass
class Constraint:
    """
    A single constraint in the model.

    Internally stored in normalized form: ``body sense rhs`` where *body* is
    an :class:`Expression` and *rhs* is ``0.0``.

    Constraints are created via comparison operators on expressions, not
    directly:

    Examples
    --------
    >>> x[0] + x[1] <= 10
    >>> dm.exp(x[2]) == 1.0
    >>> A @ x >= b

    Attributes
    ----------
    body : Expression
        Left-hand side expression (normalized so that rhs == 0).
    sense : str
        One of ``"<="``, ``">="``, ``"=="``.
    rhs : float
        Right-hand side value (always 0.0 in normalized form).
    name : str or None
        Optional name for debugging and explanation.
    """

    body: Expression
    sense: str
    rhs: float = 0.0
    name: Optional[str] = None

    def __repr__(self):
        return f"{self.body} {self.sense} {self.rhs}"


@dataclass
class ConstraintList:
    """A collection of constraints created from vectorized expressions."""

    constraints: list[Constraint]
    name: Optional[str] = None

    def __len__(self):
        return len(self.constraints)


# ─────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────


class ObjectiveSense(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Objective:
    """Objective function with sense (minimize/maximize)."""

    expression: Expression
    sense: ObjectiveSense


# ─────────────────────────────────────────────────────────────
# Parameter (for parametric optimization / sensitivity)
# ─────────────────────────────────────────────────────────────


class Parameter(Expression):
    """
    A parameter -- a value fixed during a single solve but changeable between solves.

    Unlike constants, parameters are tracked in the expression DAG so that
    JAX can differentiate the optimal objective with respect to them via
    implicit differentiation through KKT conditions.

    Parameters are created through :meth:`Model.parameter`, not directly.

    Attributes
    ----------
    name : str
        Parameter name.
    value : numpy.ndarray
        Current parameter value.
    shape : tuple of int
        Shape of the parameter.

    Examples
    --------
    >>> price = m.parameter("price", value=50.0)
    >>> m.minimize(price * x[0] + cost * x[1])
    >>> result = m.solve()
    """

    def __init__(self, name: str, value: Union[float, np.ndarray], model: "Model"):
        self.name = name
        self.value = np.asarray(value, dtype=np.float64)
        self.shape = self.value.shape
        self.model = model

    def __repr__(self):
        return f"param({self.name})"


# ─────────────────────────────────────────────────────────────
# Solve Result
# ─────────────────────────────────────────────────────────────


@dataclass
class SolveResult:
    """
    Result returned by :meth:`Model.solve`.

    Attributes
    ----------
    status : str
        Termination status. Typical values are ``"optimal"``, ``"feasible"``,
        ``"infeasible"``, ``"time_limit"``, ``"node_limit"``,
        ``"iteration_limit"``, and ``"error"``.
    objective : float or None
        Best objective value found (None if infeasible).
    bound : float or None
        Best dual (lower) bound.
    gap : float or None
        Relative optimality gap ``(objective - bound) / |objective|``.
    x : dict of str to numpy.ndarray, or None
        Variable values keyed by name. None if no feasible solution found.
    wall_time : float
        Total wall-clock solve time in seconds.
    node_count : int
        Number of Branch & Bound nodes explored.
    rust_time : float
        Time spent in the Rust backend (B&B tree management).
    jax_time : float
        Time spent in JAX (NLP evaluations, autodiff).
    python_time : float
        Time spent in Python orchestration.
    convex_fast_path : bool
        True if the problem was detected as convex and solved with a
        single NLP call (no Branch & Bound), guaranteeing global optimality.
    nlp_bb : bool
        True if the problem was solved using nonlinear Branch & Bound
        (NLP-BB), where continuous NLP subproblems are solved at each
        node with discrete variables fixed via bound tightening.
    gap_certified : bool
        True if the reported optimality gap is mathematically certified.
        False when NLP-BB is used on a nonconvex problem (heuristic mode),
        where the NLP objective is not a valid lower bound.
    """

    status: str
    objective: Optional[float] = None
    bound: Optional[float] = None
    gap: Optional[float] = None
    x: Optional[dict[str, np.ndarray]] = None
    wall_time: float = 0.0
    node_count: int = 0

    # Layer profiling
    rust_time: float = 0.0
    jax_time: float = 0.0
    python_time: float = 0.0

    # KKT duals at the returned point, when the underlying solver exposes them.
    # ``constraint_duals`` is keyed by Constraint.name; entries with a vector
    # body have one multiplier per row. ``bound_duals_lower`` /
    # ``bound_duals_upper`` are keyed by Variable.name. All values are in the
    # internal-minimization sign convention (``>= 0`` at active bounds /
    # binding-from-below inequalities). For maximize problems, the multipliers
    # correspond to the negated objective the solver actually saw.
    constraint_duals: Optional[dict[str, np.ndarray]] = None
    bound_duals_lower: Optional[dict[str, np.ndarray]] = None
    bound_duals_upper: Optional[dict[str, np.ndarray]] = None

    # Convex fast path indicator
    convex_fast_path: bool = False

    # NLP-BB indicator and gap certification
    nlp_bb: bool = False
    gap_certified: bool = True

    # Examiner-style validation report (populated if validate=True).
    validation_report: Optional[object] = None

    # LLM explanation (populated if llm=True)
    _explanation: Optional[str] = None
    _model: Optional["Model"] = None

    # Sensitivity cache (populated lazily by .gradient())
    _sensitivity: Optional[np.ndarray] = None

    def value(self, var: Variable) -> np.ndarray:
        """
        Get the optimal value of a variable.

        Parameters
        ----------
        var : Variable
            A variable from the solved model.

        Returns
        -------
        numpy.ndarray
            Optimal value, with the same shape as the variable.

        Raises
        ------
        ValueError
            If no feasible solution is available.
        """
        if self.x is None:
            raise ValueError("No solution available")
        return self.x[var.name]

    def explain(self, llm: bool = False, model: str | None = None) -> str:
        """Get a human-readable explanation of the solve result.

        Parameters
        ----------
        llm : bool, default False
            Use LLM for a rich, context-aware explanation. Falls back
            to a template string if litellm is unavailable.
        model : str, optional
            LLM model string (e.g. ``"anthropic/claude-sonnet-4-20250514"``).
        """
        if llm:
            try:
                return self._explain_with_llm(model)
            except Exception:
                pass
        if self._explanation:
            return self._explanation
        return (
            f"Solved to {self.status} in {self.wall_time:.1f}s. "
            f"Objective: {self.objective}, Gap: {self.gap}, "
            f"Nodes: {self.node_count}"
        )

    def _explain_with_llm(self, llm_model: str | None = None) -> str:
        """Generate LLM-powered explanation (internal)."""
        from discopt.llm.prompts import EXPLAIN_SYSTEM, get_explain_prompt
        from discopt.llm.provider import complete
        from discopt.llm.safety import validate_explanation
        from discopt.llm.serializer import serialize_model, serialize_solve_result

        model_text = ""
        if hasattr(self, "_model") and self._model is not None:
            model_text = serialize_model(self._model) + "\n\n"

        result_text = serialize_solve_result(self, getattr(self, "_model", None))
        status_prompt = get_explain_prompt(self.status)

        text = complete(
            messages=[
                {"role": "system", "content": EXPLAIN_SYSTEM},
                {
                    "role": "user",
                    "content": (f"{model_text}{result_text}\n\n{status_prompt}"),
                },
            ],
            model=llm_model,
            max_tokens=1024,
            timeout=5.0,
        )
        return validate_explanation(text)

    def gradient(self, param: Parameter) -> Union[float, np.ndarray]:
        """
        Sensitivity of optimal objective w.r.t. a parameter.

        Uses the envelope theorem: for ``min_x f(x; p) s.t. g(x; p) <= 0``,
        the sensitivity is ``d(obj*)/dp = dL/dp |_{x*, λ*}`` where L is the
        Lagrangian and λ* are the optimal dual variables.

        Computed lazily on first call and cached for subsequent calls.

        Parameters
        ----------
        param : Parameter
            A parameter from the solved model.

        Returns
        -------
        float or numpy.ndarray
            Gradient ``d(obj*)/d(param)``, scalar for scalar parameters.

        Raises
        ------
        ValueError
            If the model has integer/binary variables, no model reference
            is attached, or no parameters exist.
        """
        if self._model is None:
            raise ValueError(
                "No model attached to this SolveResult. "
                "gradient() requires the model reference (set by Model.solve())."
            )
        if not self._model._parameters:
            raise ValueError("Model has no parameters. Nothing to differentiate.")

        # Check all variables are continuous
        for v in self._model._variables:
            if v.var_type != VarType.CONTINUOUS:
                raise ValueError(
                    "gradient() only supports continuous models. "
                    f"Variable '{v.name}' is {v.var_type.value}."
                )

        # Lazy computation: compute sensitivity from existing solution
        if self._sensitivity is None:
            from discopt._jax.differentiable import _compute_sensitivity_at_solution

            self._sensitivity = _compute_sensitivity_at_solution(self._model, self.x)

        # Extract the slice for this parameter
        from discopt._jax.differentiable import _get_param_slice

        start, end = _get_param_slice(param, self._model)
        grad_flat = self._sensitivity[start:end]
        if param.shape == () or (end - start) == 1:
            return float(grad_flat[0])
        return grad_flat.reshape(param.shape)

    def __repr__(self):
        return (
            f"SolveResult(status={self.status!r}, obj={self.objective}, "
            f"gap={self.gap}, time={self.wall_time:.1f}s, nodes={self.node_count})"
        )


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────


class Model:
    """
    A Mixed-Integer Nonlinear Program.

    The central object for formulating and solving optimization problems.
    Build a model by declaring variables, setting an objective, adding
    constraints, and calling :meth:`solve`.

    Parameters
    ----------
    name : str, default "model"
        Descriptive name for the model.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> m = dm.Model("my_problem")
    >>> x = m.continuous("x", shape=(3,), lb=0, ub=10)
    >>> y = m.binary("y", shape=(2,))
    >>> m.minimize(cost @ x + fixed_cost @ y)
    >>> m.subject_to(A @ x <= b, name="capacity")
    >>> result = m.solve()
    >>> result.value(x)
    """

    def __init__(self, name: str = "model"):
        self.name = name
        self._variables: list[Variable] = []
        self._parameters: list[Parameter] = []
        self._constraints: list[Constraint] = []
        self._objective: Optional[Objective] = None
        self._builder = None  # Optional PyModelBuilder, lazy-initialized

    # ── Variable constructors ──

    def continuous(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        lb: Union[float, np.ndarray] = -9.999e19,
        ub: Union[float, np.ndarray] = 9.999e19,
    ) -> Variable:
        """
        Create continuous decision variable(s).

        Parameters
        ----------
        name : str
            Variable name (must be unique in the model).
        shape : int or tuple of int, default ()
            Scalar ``()`` or tuple for array variables.
        lb : float or numpy.ndarray, default -9.999e19
            Lower bound (scalar broadcast to *shape*, or array matching *shape*).
        ub : float or numpy.ndarray, default 9.999e19
            Upper bound (scalar broadcast to *shape*, or array matching *shape*).

        Returns
        -------
        Variable
            Expression that can be used in objectives and constraints.

        .. warning::

            NLP solvers (ipm, ipopt, ripopt) use interior-point barrier methods
            that require finite, reasonably-sized bounds.  The defaults
            (±9.999×10¹⁹) exceed the safe threshold (~10¹⁵) and will cause
            NaN objectives or ``iteration_limit`` status.  Always supply
            explicit ``lb``/``ub`` when the problem has a known feasible range::

                x = m.continuous("x", lb=-100, ub=100)   # good
                x = m.continuous("x")                     # risky for NLP solvers

            A ``UserWarning`` is raised at solve time when bounds exceed 10¹⁵.

        Examples
        --------
        >>> x = m.continuous("x")                           # scalar, unbounded
        >>> flow = m.continuous("flow", shape=(5,), lb=0)   # 5-vector, non-negative
        >>> X = m.continuous("X", shape=(3, 4), lb=0, ub=1) # 3x4 matrix
        """
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.CONTINUOUS, shape, lb, ub, self)
        self._variables.append(var)
        if self._builder is not None:
            var._builder_idx = self._builder.add_variable(
                var.name,
                var.var_type.value,
                list(var.shape),
                var.lb.flatten().astype(np.float64),
                var.ub.flatten().astype(np.float64),
            )
        return var

    def binary(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
    ) -> Variable:
        """
        Create binary (0/1) decision variable(s).

        Parameters
        ----------
        name : str
            Variable name (must be unique in the model).
        shape : int or tuple of int, default ()
            Scalar ``()`` or tuple for array variables.

        Returns
        -------
        Variable
            Binary variable with bounds ``[0, 1]``.

        Examples
        --------
        >>> use = m.binary("use")                    # single binary
        >>> active = m.binary("active", shape=(5,))  # 5 binary indicators
        """
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.BINARY, shape, 0.0, 1.0, self)
        self._variables.append(var)
        if self._builder is not None:
            var._builder_idx = self._builder.add_variable(
                var.name,
                var.var_type.value,
                list(var.shape),
                var.lb.flatten().astype(np.float64),
                var.ub.flatten().astype(np.float64),
            )
        return var

    def integer(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        lb: Union[float, np.ndarray] = 0,
        ub: Union[float, np.ndarray] = 1e6,
    ) -> Variable:
        """
        Create general integer decision variable(s).

        Parameters
        ----------
        name : str
            Variable name (must be unique in the model).
        shape : int or tuple of int, default ()
            Scalar ``()`` or tuple for array variables.
        lb : float or numpy.ndarray, default 0
            Lower bound.
        ub : float or numpy.ndarray, default 1e6
            Upper bound.

        Returns
        -------
        Variable
            Integer-valued variable.

        Examples
        --------
        >>> n = m.integer("n_units", lb=0, ub=10)
        >>> batch = m.integer("batch", shape=(3,), lb=1, ub=100)
        """
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.INTEGER, shape, lb, ub, self)
        self._variables.append(var)
        if self._builder is not None:
            var._builder_idx = self._builder.add_variable(
                var.name,
                var.var_type.value,
                list(var.shape),
                var.lb.flatten().astype(np.float64),
                var.ub.flatten().astype(np.float64),
            )
        return var

    def parameter(
        self,
        name: str,
        value: Union[float, np.ndarray],
    ) -> Parameter:
        """
        Create a parameter for parametric optimization / sensitivity.

        Parameters are fixed during a solve but tracked in the expression
        DAG for differentiation via implicit diff through KKT conditions.

        Parameters
        ----------
        name : str
            Parameter name (must be unique in the model).
        value : float or numpy.ndarray
            Current parameter value.

        Returns
        -------
        Parameter
            Parameter expression usable in objectives and constraints.

        Examples
        --------
        >>> price = m.parameter("price", value=50.0)
        >>> demand = m.parameter("demand", value=np.array([100, 200, 150]))
        """
        self._check_name(name)
        param = Parameter(name, value, self)
        self._parameters.append(param)
        return param

    # ── Objective ──

    def minimize(self, expr: Expression):
        """
        Set the objective to minimize.

        Parameters
        ----------
        expr : Expression
            Expression to minimize.

        Examples
        --------
        >>> m.minimize(cost @ x)
        >>> m.minimize(dm.sum(lambda i: c[i] * x[i], over=range(n)))
        """
        self._objective = Objective(_wrap(expr), ObjectiveSense.MINIMIZE)

    def maximize(self, expr: Expression):
        """
        Set the objective to maximize.

        Parameters
        ----------
        expr : Expression
            Expression to maximize.

        Examples
        --------
        >>> m.maximize(profit @ x - dm.sum(penalty * y))
        """
        self._objective = Objective(_wrap(expr), ObjectiveSense.MAXIMIZE)

    # ── Constraints ──

    def subject_to(
        self,
        constraint: Union[Constraint, list[Constraint], bool],
        name: Optional[str] = None,
    ):
        """
        Add constraint(s) to the model.

        Parameters
        ----------
        constraint : Constraint or list of Constraint
            Constraint(s) created by comparison operators (``<=``, ``>=``,
            ``==``) on expressions.
        name : str, optional
            Name for the constraint(s). Named constraints enable better
            debugging and LLM-generated explanations.

        Examples
        --------
        >>> m.subject_to(x[0] + x[1] <= 10)
        >>> m.subject_to(dm.exp(x[0]) == 1.0)
        >>> m.subject_to(A @ x <= b, name="capacity")
        >>> m.subject_to([x[i] + x[i+1] <= limits[i] for i in range(n-1)],
        ...              name="adjacent_limits")
        """
        if isinstance(constraint, list):
            for k, c in enumerate(constraint):
                if isinstance(c, Constraint):
                    c.name = f"{name}_{k}" if name else None
                    self._constraints.append(c)
        elif isinstance(constraint, Constraint):
            constraint.name = name
            self._constraints.append(constraint)
        else:
            raise TypeError(
                f"Expected Constraint (from <=, >=, == on expressions), "
                f"got {type(constraint)}. Did you mean to compare expressions?"
            )

    # ── Fast construction API (direct arena building) ──

    def _get_builder(self):
        """Lazily initialize the Rust model builder, registering all existing variables."""
        if self._builder is None:
            from discopt._rust import PyModelBuilder

            self._builder = PyModelBuilder()
            for var in self._variables:
                var._builder_idx = self._builder.add_variable(
                    var.name,
                    var.var_type.value,
                    list(var.shape),
                    var.lb.flatten().astype(np.float64),
                    var.ub.flatten().astype(np.float64),
                )
        return self._builder

    def add_linear_constraints(
        self,
        A,
        x: Variable,
        sense: str,
        b,
        name: Optional[str] = None,
    ):
        """
        Add linear constraints in bulk: each row of A defines one constraint.

        Bypasses Python expression objects — builds directly into the Rust
        expression arena via a single PyO3 call. For large models (1000+
        constraints), this is orders of magnitude faster than operator
        overloading.

        Parameters
        ----------
        A : scipy.sparse matrix or numpy.ndarray
            Constraint coefficient matrix, shape ``(m, n)`` where
            ``n == x.size``. Any scipy sparse format (CSR, CSC, COO) or
            dense array. Automatically converted to CSR internally.
        x : Variable
            Array variable whose size matches ``A.shape[1]``.
        sense : str
            ``"<="``, ``"=="``, or ``">="``. Applied to all rows.
        b : numpy.ndarray or float
            Right-hand side, shape ``(m,)`` or scalar (broadcast).
        name : str, optional
            Prefix for constraint names (``"{name}_0"``, ``"{name}_1"``, ...).

        Raises
        ------
        ValueError
            If dimensions don't match or sense is invalid.
        """
        import scipy.sparse as sp

        if sense not in ("<=", "==", ">="):
            raise ValueError(f"Invalid sense '{sense}'. Expected '<=', '==', or '>='.")

        # Convert to CSR
        if not sp.issparse(A):
            A = sp.csr_matrix(np.asarray(A, dtype=np.float64))
        elif not sp.isspmatrix_csr(A):
            A = A.tocsr()
        A = A.astype(np.float64)

        m_rows, n_cols = A.shape
        if n_cols != x.size:
            raise ValueError(f"A has {n_cols} columns but variable '{x.name}' has size {x.size}.")

        b = np.broadcast_to(np.asarray(b, dtype=np.float64), (m_rows,)).copy()

        builder = self._get_builder()
        if not hasattr(x, "_builder_idx"):
            raise ValueError(
                f"Variable '{x.name}' is not registered in the builder. "
                "Ensure the variable was created via m.continuous/binary/integer."
            )

        builder.add_linear_constraints(
            A.indptr.astype(np.int64),
            A.indices.astype(np.int64),
            A.data,
            x._builder_idx,
            sense,
            b,
            name,
        )

    def add_linear_objective(
        self,
        c,
        x: Variable,
        constant: float = 0.0,
        sense: str = "minimize",
    ):
        """
        Set a linear objective: ``c'x + constant``.

        Parameters
        ----------
        c : numpy.ndarray
            Cost vector, shape ``(n,)`` matching ``x.size``.
        x : Variable
            Variable reference.
        constant : float, default 0.0
            Scalar offset.
        sense : str, default "minimize"
            ``"minimize"`` or ``"maximize"``.
        """
        c = np.asarray(c, dtype=np.float64).flatten()
        if c.shape[0] != x.size:
            raise ValueError(
                f"c has {c.shape[0]} elements but variable '{x.name}' has size {x.size}."
            )
        builder = self._get_builder()
        if not hasattr(x, "_builder_idx"):
            raise ValueError(f"Variable '{x.name}' is not registered in the builder.")
        builder.set_linear_objective(c, x._builder_idx, constant, sense)
        # Set a placeholder objective so validate() passes
        self._objective = Objective(
            Constant(np.float64(0.0)),
            ObjectiveSense.MINIMIZE if sense == "minimize" else ObjectiveSense.MAXIMIZE,
        )
        self._objective._is_placeholder = True

    def add_quadratic_objective(
        self,
        Q,
        c,
        x: Variable,
        constant: float = 0.0,
        sense: str = "minimize",
    ):
        """
        Set a quadratic objective: ``0.5 x'Qx + c'x + constant``.

        Parameters
        ----------
        Q : scipy.sparse matrix or numpy.ndarray
            Symmetric quadratic coefficient matrix, shape ``(n, n)``.
        c : numpy.ndarray
            Linear coefficient vector, shape ``(n,)``.
        x : Variable
            Variable reference.
        constant : float, default 0.0
            Scalar offset.
        sense : str, default "minimize"
            ``"minimize"`` or ``"maximize"``.
        """
        import scipy.sparse as sp

        n = x.size
        c = np.asarray(c, dtype=np.float64).flatten()
        if c.shape[0] != n:
            raise ValueError(f"c has {c.shape[0]} elements but variable '{x.name}' has size {n}.")

        if not sp.issparse(Q):
            Q = sp.csr_matrix(np.asarray(Q, dtype=np.float64))
        elif not sp.isspmatrix_csr(Q):
            Q = Q.tocsr()
        Q = Q.astype(np.float64)

        if Q.shape != (n, n):
            raise ValueError(f"Q has shape {Q.shape} but expected ({n}, {n}).")

        builder = self._get_builder()
        if not hasattr(x, "_builder_idx"):
            raise ValueError(f"Variable '{x.name}' is not registered in the builder.")
        builder.set_quadratic_objective(
            Q.indptr.astype(np.int64),
            Q.indices.astype(np.int64),
            Q.data,
            c,
            x._builder_idx,
            constant,
            sense,
        )
        # Set a placeholder objective so validate() passes
        self._objective = Objective(
            Constant(np.float64(0.0)),
            ObjectiveSense.MINIMIZE if sense == "minimize" else ObjectiveSense.MAXIMIZE,
        )
        self._objective._is_placeholder = True

    # ── Logical constraints (GDP) ──

    def if_then(
        self,
        indicator: Variable,
        then_constraints: list[Constraint],
        name: Optional[str] = None,
    ):
        """
        Add indicator (if-then) constraint.

        If ``indicator == 1``, all *then_constraints* must hold.
        If ``indicator == 0``, the constraints are relaxed.
        Avoids manual big-M formulation.

        Parameters
        ----------
        indicator : Variable
            A binary variable.
        then_constraints : list of Constraint
            Constraints that must hold when the indicator is active.
        name : str, optional
            Base name for the constraint group.

        Examples
        --------
        >>> m.if_then(y[0], [x[0] >= 10, x[1] <= 50], name="unit0_active")
        """
        for k, c in enumerate(then_constraints):
            c.name = f"{name}_then_{k}" if name else None
            # Store as indicator constraint; Rust presolve will handle
            # reformulation to big-M or GDP branching
            self._constraints.append(
                _IndicatorConstraint(
                    indicator=indicator,
                    constraint=c,
                    active_value=1,
                )
            )

    def either_or(
        self,
        disjuncts: list[list[Constraint]],
        name: Optional[str] = None,
    ):
        """
        Add disjunctive constraint (Generalized Disjunctive Programming).

        Exactly one group of constraints must hold.

        Parameters
        ----------
        disjuncts : list of list of Constraint
            Each inner list is a disjunct -- a group of constraints that
            must all hold together.
        name : str, optional
            Name for the disjunction.

        Examples
        --------
        >>> m.either_or([
        ...     [x[0] <= 5, x[1] >= 10],   # mode A
        ...     [x[0] >= 15, x[1] <= 3],   # mode B
        ... ], name="operating_mode")
        """
        self._constraints.append(
            _DisjunctiveConstraint(
                disjuncts=disjuncts,
                name=name,
            )
        )

    # ── Special ordered sets ──

    def sos1(self, variables: list[Variable], name: Optional[str] = None):
        """
        Add SOS Type 1 constraint: at most one variable can be nonzero.

        Parameters
        ----------
        variables : list of Variable
            Variables in the special ordered set.
        name : str, optional
            Constraint name.
        """
        self._constraints.append(_SOSConstraint(1, variables, name))

    def sos2(self, variables: list[Variable], name: Optional[str] = None):
        """
        Add SOS Type 2 constraint: at most two adjacent variables can be nonzero.

        Parameters
        ----------
        variables : list of Variable
            Variables in the special ordered set (order matters).
        name : str, optional
            Constraint name.
        """
        self._constraints.append(_SOSConstraint(2, variables, name))

    # ── Logical propositions ──

    @staticmethod
    def _validate_binaries(variables, method_name: str):
        """Check that all entries are binary variables (Variable or IndexExpression)."""
        for v in variables:
            if isinstance(v, IndexExpression):
                base = v.base
                if not isinstance(base, Variable):
                    raise TypeError(
                        f"{method_name}() requires binary variables, "
                        f"got IndexExpression with non-Variable base"
                    )
                if base.var_type != VarType.BINARY:
                    raise ValueError(
                        f"{method_name}() requires binary variables, "
                        f"but '{base.name}' has type {base.var_type.name}"
                    )
            elif isinstance(v, Variable):
                if v.var_type != VarType.BINARY:
                    raise ValueError(
                        f"{method_name}() requires binary variables, "
                        f"but '{v.name}' has type {v.var_type.name}"
                    )
            else:
                raise TypeError(
                    f"{method_name}() requires Variable or IndexExpression, got {type(v).__name__}"
                )

    def at_least(self, k: int, binaries: list, name: Optional[str] = None):
        """Add constraint: at least *k* of the binary variables must be 1."""
        self._validate_binaries(binaries, "at_least")
        self.subject_to(sum(binaries) >= k, name=name)

    def at_most(self, k: int, binaries: list, name: Optional[str] = None):
        """Add constraint: at most *k* of the binary variables can be 1."""
        self._validate_binaries(binaries, "at_most")
        self.subject_to(sum(binaries) <= k, name=name)

    def exactly(self, k: int, binaries: list, name: Optional[str] = None):
        """Add constraint: exactly *k* of the binary variables must be 1."""
        self._validate_binaries(binaries, "exactly")
        self.subject_to(sum(binaries) == k, name=name)

    def implies(self, y1, y2, name: Optional[str] = None):
        """Add implication constraint: y1 = 1 implies y2 = 1."""
        self._validate_binaries([y1, y2], "implies")
        self.subject_to(y1 <= y2, name=name)

    def iff(self, y1, y2, name: Optional[str] = None):
        """Add equivalence constraint: y1 = 1 if and only if y2 = 1."""
        self._validate_binaries([y1, y2], "iff")
        self.subject_to(y1 == y2, name=name)

    def disjunction(
        self,
        disjuncts: list[list],
        name: Optional[str] = None,
    ) -> "_DisjunctiveConstraint":
        """Create a disjunction object for nesting inside either_or().

        Unlike :meth:`either_or`, this does **not** add the disjunction to
        the model. Use it to build nested disjunctions.

        Parameters
        ----------
        disjuncts : list of list
            Each inner list is a group of constraints (a disjunct).
        name : str, optional
            Name for the disjunction.

        Returns
        -------
        _DisjunctiveConstraint
        """
        return _DisjunctiveConstraint(disjuncts=disjuncts, name=name)

    def make_disjunct(self, name: str) -> "Disjunct":
        """Create a named disjunct block with an auto-generated indicator.

        Parameters
        ----------
        name : str
            Block name. A boolean indicator ``{name}_active`` is created.

        Returns
        -------
        Disjunct

        Example
        -------
        >>> d1 = m.make_disjunct("mode_a")
        >>> d1.subject_to(x <= 3)
        """
        return Disjunct(name, self)

    def add_disjunction(
        self,
        disjuncts: list["Disjunct"],
        name: Optional[str] = None,
    ) -> None:
        """Add a disjunction over Disjunct blocks.

        Exactly one disjunct must be active. This maps to indicator
        constraints (``if_then``) and an ``exactly(1, ...)`` selector.

        Parameters
        ----------
        disjuncts : list of Disjunct
            The disjunct blocks to form the disjunction.
        name : str, optional
            Name for the disjunction.

        Example
        -------
        >>> d1 = m.make_disjunct("mode_a")
        >>> d1.subject_to(x <= 3)
        >>> d2 = m.make_disjunct("mode_b")
        >>> d2.subject_to(x >= 7)
        >>> m.add_disjunction([d1, d2], name="mode_select")
        """
        for d in disjuncts:
            self.if_then(d.indicator.variable, d._constraints, name=d.name)
        indicators = [d.indicator.variable for d in disjuncts]
        self.exactly(1, indicators, name=f"_disj_{name}_xor" if name else None)

    # ── Boolean logic (GDP) ──

    def boolean(
        self,
        name: str,
        shape: Union[tuple, int] = (),
    ) -> Union["BooleanVar", "BooleanVarArray"]:
        """Create boolean decision variable(s) backed by binary variables.

        Parameters
        ----------
        name : str
            Variable name.
        shape : tuple or int
            Shape of the boolean variable array. Scalar by default.

        Returns
        -------
        BooleanVar or BooleanVarArray
        """
        if isinstance(shape, int):
            shape = (shape,)
        var = self.binary(name, shape=shape)
        if shape == () or shape == (1,):
            return BooleanVar(var)
        return BooleanVarArray(var)

    def logical(
        self,
        expr: "LogicalExpression",
        name: Optional[str] = None,
    ) -> None:
        """Add a propositional logic constraint.

        Parameters
        ----------
        expr : LogicalExpression
            A boolean expression built from BooleanVars using ``&``, ``|``,
            ``~``, ``.implies()``, ``.equivalent_to()``.
        name : str, optional
            Constraint name.

        Examples
        --------
        >>> Y = m.boolean("choice", shape=(3,))
        >>> m.logical(Y[0].implies(Y[1] & ~Y[2]))
        """
        if not isinstance(expr, LogicalExpression):
            raise TypeError(f"Expected LogicalExpression, got {type(expr).__name__}")
        self._constraints.append(_LogicalConstraint(expr, name))

    # ── Solve ──

    def solve(
        self,
        time_limit: float = 3600,
        gap_tolerance: float = 1e-4,
        threads: int = 1,
        llm: bool = False,
        sensitivity: bool = False,
        stream: bool = False,
        deterministic: bool = True,
        partitions: int = 0,
        branching_policy: str = "fractional",
        initial_solution: Optional[dict] = None,
        skip_convex_check: bool = False,
        nlp_bb: Optional[bool] = None,
        lazy_constraints: Optional[Callable] = None,
        incumbent_callback: Optional[Callable] = None,
        node_callback: Optional[Callable] = None,
        solver: Optional[str] = None,
        validate: bool = False,
        **kwargs,
    ) -> Union[SolveResult, Iterator["SolveUpdate"]]:
        r"""
        Solve the model.

        For pure-continuous models, solves the NLP directly. For models with
        integer/binary variables, uses NLP-based spatial Branch & Bound.

        Parameters
        ----------
        time_limit : float, default 3600
            Wall-clock time limit in seconds.
        gap_tolerance : float, default 1e-4
            Relative optimality gap tolerance for termination.
        threads : int, default 1
            Number of CPU threads for Rust components.
        llm : bool, default False
            Enable LLM explanation of results.
        sensitivity : bool, default False
            Compute sensitivities w.r.t. Parameters.
        stream : bool, default False
            If True, return an iterator of :class:`SolveUpdate` instead of
            the final result.
        deterministic : bool, default True
            Ensure reproducible results across runs.
        partitions : int, default 0
            Number of piecewise McCormick partitions (0 = standard convex
            relaxation, k > 0 = k partitions for tighter relaxations).
        branching_policy : str, default "fractional"
            Variable selection policy: ``"fractional"`` (most-fractional)
            or ``"gnn"`` (GNN scoring, future hook).
        initial_solution : dict, optional
            Initial feasible solution mapping Variable objects to values
            (scalars, lists, or numpy arrays).  Used as a warm-start point
            for NLP solves and as the initial incumbent in Branch & Bound.
            Values are validated against variable bounds and integrality
            requirements; violations produce warnings and are corrected
            automatically (clamped / rounded).
        skip_convex_check : bool, default False
            If True, skip automatic convexity detection for continuous
            problems. When False (default), convex NLPs are solved with
            a single NLP call (no B&B), guaranteeing global optimality.
        nlp_bb : bool or None, default None
            Nonlinear Branch & Bound mode. When ``None`` (default),
            auto-selects NLP-BB for convex MINLPs and spatial B&B
            otherwise. When ``True``, forces NLP-BB (heuristic mode if
            nonconvex). When ``False``, forces spatial B&B.
        lazy_constraints : callable, optional
            Lazy constraint callback. Called at integer-feasible nodes.
            Should accept ``(ctx, model)`` and return a list of
            :class:`~discopt.callbacks.CutResult`. If cuts are returned,
            the solution is not accepted as incumbent until it satisfies
            all lazy constraints.
        incumbent_callback : callable, optional
            Incumbent callback. Called when a new incumbent is about to
            be accepted. Should accept ``(ctx, model, solution)`` and
            return ``True`` to accept or ``False`` to reject.
        node_callback : callable, optional
            Node callback. Called after each batch of nodes is processed.
            Should accept ``(ctx, model)`` and return ``None``.
        validate : bool, default False
            If True, run Examiner-style KKT validation on the returned
            point and attach the :class:`~discopt.validation.ExaminerReport`
            to ``result.validation_report``. Errors during validation are
            swallowed and leave ``validation_report`` as ``None``.
        \*\*kwargs
            Additional keyword arguments passed to the solver backend.

        Returns
        -------
        SolveResult or Iterator[SolveUpdate]
            Solve result, or a streaming iterator if ``stream=True``.

        Raises
        ------
        ValueError
            If the model fails validation (no objective, duplicate names, etc.).
        TypeError
            If *initial_solution* contains non-Variable keys.
        """
        self.validate()

        # Validate initial solution if provided
        _x0_flat = None
        if initial_solution is not None:
            from discopt.warm_start import validate_initial_solution

            _x0_flat = validate_initial_solution(self, initial_solution)

        # Pre-solve LLM analysis (advisory only, never blocks solving)
        if llm:
            try:
                from discopt.llm.advisor import presolve_analysis

                warnings = presolve_analysis(self)
                for w in warnings:
                    import logging

                    logging.getLogger("discopt.llm").info("Pre-solve: %s", w)
            except Exception:
                pass

        if stream:
            return self._solve_streaming(
                time_limit=time_limit, gap_tolerance=gap_tolerance, **kwargs
            )

        from discopt._jax.deadline import deadline_scope
        from discopt.solver import solve_model

        if solver is not None:
            kwargs["solver"] = solver

        # Install a process-global wall-clock deadline that JAX-compiled
        # while_loops (LP/QP/NLP IPM) can poll via host callback so they
        # self-terminate within ``time_limit + ε`` instead of running to
        # XLA convergence after Python's budget is gone (issue #80).
        with deadline_scope(time_limit):
            result = solve_model(
                self,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                threads=threads,
                deterministic=deterministic,
                partitions=partitions,
                branching_policy=branching_policy,
                initial_point=_x0_flat,
                skip_convex_check=skip_convex_check,
                nlp_bb=nlp_bb,
                lazy_constraints=lazy_constraints,
                incumbent_callback=incumbent_callback,
                node_callback=node_callback,
                **kwargs,
            )

        # Attach model reference and auto-generate LLM explanation
        result._model = self
        if llm:
            try:
                result._explanation = result._explain_with_llm()
            except Exception:
                pass

        if validate and result.x is not None:
            try:
                from discopt.validation.examiner import examine

                result.validation_report = examine(result, self)
            except Exception:
                result.validation_report = None

        return result

    def _solve_streaming(self, **kwargs) -> Iterator["SolveUpdate"]:
        """Streaming solve that yields updates during B&B."""
        raise NotImplementedError("Streaming solve requires solver backend")

    # ── Validation ──

    def validate(self):
        """
        Validate model consistency.

        Raises
        ------
        ValueError
            If the objective is not set, variable names are not unique,
            or variable bounds are inconsistent (lb > ub).
        """
        if self._objective is None:
            raise ValueError("No objective set. Call m.minimize() or m.maximize().")

        names = set()
        for var in self._variables:
            if var.name in names:
                raise ValueError(f"Duplicate variable name: '{var.name}'")
            names.add(var.name)
            if np.any(var.lb > var.ub):
                raise ValueError(f"Variable '{var.name}' has lb > ub at some index")

    # ── Model statistics ──

    @property
    def num_variables(self) -> int:
        return builtins_sum(v.size for v in self._variables)

    @property
    def num_continuous(self) -> int:
        return builtins_sum(v.size for v in self._variables if v.var_type == VarType.CONTINUOUS)

    @property
    def num_integer(self) -> int:
        return builtins_sum(
            v.size for v in self._variables if v.var_type in (VarType.INTEGER, VarType.BINARY)
        )

    @property
    def num_constraints(self) -> int:
        return len(self._constraints)

    def summary(self) -> str:
        """
        Return a human-readable model summary.

        Returns
        -------
        str
            Multi-line string with variable counts, constraint count,
            objective sense, and parameter count.
        """
        lines = [
            f"Model: {self.name}",
            f"  Variables: {self.num_variables} "
            f"({self.num_continuous} continuous, {self.num_integer} integer/binary)",
            f"  Constraints: {self.num_constraints}",
            f"  Objective: {self._objective.sense.value} {self._objective.expression}",
            f"  Parameters: {len(self._parameters)}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    # ── Export ──

    def to_mps(self, path: Union[str, None] = None) -> Union[str, None]:
        """Export the model to MPS format.

        Only linear and quadratic models are supported. Nonlinear
        expressions raise ``ValueError``.

        Parameters
        ----------
        path : str, optional
            File path to write. If ``None``, return the MPS string.

        Returns
        -------
        str or None
            MPS string if *path* is ``None``, otherwise ``None``.
        """
        from discopt.export.mps import to_mps

        return to_mps(self, path)

    def to_lp(self, path: Union[str, None] = None) -> Union[str, None]:
        """Export the model to CPLEX LP format.

        Only linear and quadratic models are supported. Nonlinear
        expressions raise ``ValueError``.

        Parameters
        ----------
        path : str, optional
            File path to write. If ``None``, return the LP string.

        Returns
        -------
        str or None
            LP string if *path* is ``None``, otherwise ``None``.
        """
        from discopt.export.lp import to_lp

        return to_lp(self, path)

    def to_gams(
        self,
        path: Union[str, None] = None,
        model_type: Union[str, None] = None,
    ) -> Union[str, None]:
        """Export the model to GAMS (.gms) format.

        Supports all model types including MINLP with nonlinear expressions.

        Parameters
        ----------
        path : str, optional
            File path to write. If ``None``, return the GAMS string.
        model_type : str, optional
            GAMS model type (LP, MIP, NLP, MINLP, etc.).
            Auto-detected from variable types and expression structure if not given.

        Returns
        -------
        str or None
            GAMS string if *path* is ``None``, otherwise ``None``.
        """
        from discopt.export.gams import to_gams

        return to_gams(self, path, model_type)

    def to_nl(self, path: Union[str, None] = None) -> Union[str, None]:
        """Export the model to AMPL .nl text format.

        Supports all model types including MINLP with nonlinear expressions.
        Produces text-mode .nl files compatible with AMPL-compatible solvers
        (Ipopt, BARON, Couenne, SCIP) and the discopt Rust .nl parser.

        Parameters
        ----------
        path : str, optional
            File path to write. If ``None``, return the .nl string.

        Returns
        -------
        str or None
            .nl string if *path* is ``None``, otherwise ``None``.
        """
        from discopt.export.nl import to_nl

        return to_nl(self, path)

    def _check_name(self, name: str):
        """Ensure variable/parameter name is unique."""
        existing = {v.name for v in self._variables} | {p.name for p in self._parameters}
        if name in existing:
            raise ValueError(f"Name '{name}' already used in model")


# Internal constraint types (not part of public API)


@dataclass
class _IndicatorConstraint:
    indicator: Variable
    constraint: Constraint
    active_value: int = 1
    name: Optional[str] = None


@dataclass
class _DisjunctiveConstraint:
    disjuncts: list[list[Constraint]]
    name: Optional[str] = None


@dataclass
class _SOSConstraint:
    sos_type: int
    variables: list[Variable]
    name: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Propositional logic for GDP
# ─────────────────────────────────────────────────────────────


class LogicalExpression:
    """Base class for propositional logic expressions over BooleanVars."""

    def __and__(self, other: "LogicalExpression") -> "LogicalAnd":
        return LogicalAnd(self, _wrap_logical(other))

    def __rand__(self, other: "LogicalExpression") -> "LogicalAnd":
        return LogicalAnd(_wrap_logical(other), self)

    def __or__(self, other: "LogicalExpression") -> "LogicalOr":
        return LogicalOr(self, _wrap_logical(other))

    def __ror__(self, other: "LogicalExpression") -> "LogicalOr":
        return LogicalOr(_wrap_logical(other), self)

    def __invert__(self) -> "LogicalNot":
        return LogicalNot(self)

    def implies(self, other: "LogicalExpression") -> "LogicalImplies":
        """Logical implication: self → other."""
        return LogicalImplies(self, _wrap_logical(other))

    def equivalent_to(self, other: "LogicalExpression") -> "LogicalEquivalent":
        """Logical equivalence: self ↔ other."""
        return LogicalEquivalent(self, _wrap_logical(other))


def _wrap_logical(x):
    """Wrap a BooleanVar or LogicalExpression, raise otherwise."""
    if isinstance(x, LogicalExpression):
        return x
    raise TypeError(f"Expected LogicalExpression, got {type(x).__name__}")


class BooleanVar(LogicalExpression):
    """A boolean decision variable backed by a binary Variable.

    Created via :meth:`Model.boolean`, not directly.
    """

    def __init__(self, variable):
        self.variable = variable

    def __repr__(self) -> str:
        return f"BooleanVar({self.variable.name})"


class BooleanVarArray:
    """Array of BooleanVars backed by a single array-shaped binary Variable."""

    def __init__(self, variable):
        self.variable = variable
        self._size = variable.size

    def __getitem__(self, idx) -> BooleanVar:
        return BooleanVar(self.variable[idx])

    def __len__(self) -> int:
        return int(self._size)

    def __iter__(self):
        for i in range(self._size):
            yield self[i]


@dataclass
class LogicalAnd(LogicalExpression):
    left: LogicalExpression
    right: LogicalExpression


@dataclass
class LogicalOr(LogicalExpression):
    left: LogicalExpression
    right: LogicalExpression


@dataclass
class LogicalNot(LogicalExpression):
    operand: LogicalExpression


@dataclass
class LogicalImplies(LogicalExpression):
    antecedent: LogicalExpression
    consequent: LogicalExpression


@dataclass
class LogicalEquivalent(LogicalExpression):
    left: LogicalExpression
    right: LogicalExpression


@dataclass
class LogicalAtLeast(LogicalExpression):
    k: int
    operands: list


@dataclass
class LogicalAtMost(LogicalExpression):
    k: int
    operands: list


@dataclass
class LogicalExactly(LogicalExpression):
    k: int
    operands: list


@dataclass
class _LogicalConstraint:
    expression: LogicalExpression
    name: Optional[str] = None


# Functional-style constructors for logical expressions


def land(*args: LogicalExpression) -> LogicalExpression:
    """Logical AND of multiple BooleanVars/expressions."""
    result = args[0]
    for a in args[1:]:
        result = LogicalAnd(result, a)
    return result


def lor(*args: LogicalExpression) -> LogicalExpression:
    """Logical OR of multiple BooleanVars/expressions."""
    result = args[0]
    for a in args[1:]:
        result = LogicalOr(result, a)
    return result


def lnot(x: LogicalExpression) -> LogicalNot:
    """Logical NOT."""
    return LogicalNot(x)


def atleast(k: int, *args: LogicalExpression) -> LogicalAtLeast:
    """At least k of the given boolean expressions must be true."""
    operands = list(args[0]) if len(args) == 1 and hasattr(args[0], "__iter__") else list(args)
    return LogicalAtLeast(k, operands)


def atmost(k: int, *args: LogicalExpression) -> LogicalAtMost:
    """At most k of the given boolean expressions may be true."""
    operands = list(args[0]) if len(args) == 1 and hasattr(args[0], "__iter__") else list(args)
    return LogicalAtMost(k, operands)


def exactly(k: int, *args: LogicalExpression) -> LogicalExactly:
    """Exactly k of the given boolean expressions must be true."""
    operands = list(args[0]) if len(args) == 1 and hasattr(args[0], "__iter__") else list(args)
    return LogicalExactly(k, operands)


# ─────────────────────────────────────────────────────────────
# Disjunct block abstraction
# ─────────────────────────────────────────────────────────────


class Disjunct:
    """A named block of constraints activated by a boolean indicator.

    Created via :meth:`Model.disjunct`, not directly.

    Parameters
    ----------
    name : str
        Block name. An indicator boolean ``{name}_active`` is created.
    model : Model
        The parent optimization model.

    Example
    -------
    >>> d1 = m.disjunct("mode_a")
    >>> d1.subject_to(x <= 3)
    >>> d2 = m.disjunct("mode_b")
    >>> d2.subject_to(x >= 7)
    >>> m.add_disjunction([d1, d2])
    """

    def __init__(self, name: str, model: "Model"):
        self.name = name
        self._model = model
        bv = model.boolean(f"{name}_active")
        assert isinstance(bv, BooleanVar)
        self.indicator: "BooleanVar" = bv
        self._constraints: list[Constraint] = []

    def subject_to(
        self,
        constraint: Union[Constraint, list[Constraint]],
        name: Optional[str] = None,
    ) -> None:
        """Add constraint(s) to this disjunct."""
        if isinstance(constraint, list):
            self._constraints.extend(constraint)
        else:
            self._constraints.append(constraint)

    @property
    def active(self) -> "BooleanVar":
        """The boolean indicator for this disjunct."""
        return self.indicator

    @property
    def constraints(self) -> list[Constraint]:
        """Constraints in this disjunct."""
        return list(self._constraints)

    def __repr__(self) -> str:
        return f"Disjunct({self.name!r}, {len(self._constraints)} constraints)"


# ─────────────────────────────────────────────────────────────
# Streaming updates
# ─────────────────────────────────────────────────────────────


@dataclass
class SolveUpdate:
    """
    Intermediate update yielded during a streaming solve.

    Attributes
    ----------
    elapsed : float
        Wall-clock time since solve start (seconds).
    incumbent : float or None
        Best feasible objective found so far.
    lower_bound : float
        Current global lower bound.
    gap : float or None
        Current relative optimality gap.
    node_count : int
        Total B&B nodes explored so far.
    open_nodes : int
        Number of open (unexplored) nodes.
    message : str or None
        LLM commentary if ``llm=True`` was passed to :meth:`Model.solve`.
    """

    elapsed: float
    incumbent: Optional[float]
    lower_bound: float
    gap: Optional[float]
    node_count: int
    open_nodes: int
    message: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Import functions
# ─────────────────────────────────────────────────────────────


def from_pyomo(pyomo_model) -> Model:
    """
    Import a Pyomo ConcreteModel as a discopt Model.

    Supports Var, Constraint, Objective, Param, Set.
    GDP (Disjunct/Disjunction) is mapped to :meth:`Model.either_or`.

    Parameters
    ----------
    pyomo_model : pyomo.environ.ConcreteModel
        A fully constructed Pyomo model.

    Returns
    -------
    Model

    Raises
    ------
    NotImplementedError
        Pyomo import is a Phase 4 feature.
    """
    raise NotImplementedError("Pyomo import requires pyomo bridge (Phase 4)")


def from_nl(path: str) -> Model:
    """
    Import a model from AMPL .nl format.

    Uses the Rust .nl parser for speed. Variables, bounds, constraints,
    and objective are extracted from the binary .nl representation.

    Parameters
    ----------
    path : str
        Path to the ``.nl`` file.

    Returns
    -------
    Model
        A Model ready to solve. The NLP evaluation is delegated to the
        the standard ``NLPEvaluator`` with JAX autodiff.

    Examples
    --------
    >>> model = dm.from_nl("problem.nl")
    >>> result = model.solve()
    """
    from discopt._jax.nl_reconstruction import reconstruct_dag
    from discopt._rust import parse_nl_file

    nl_repr = parse_nl_file(path)

    # Build a Python Model from the parsed representation
    import os

    model_name = os.path.splitext(os.path.basename(path))[0]
    m = Model(model_name)

    # Create variables matching the .nl file
    var_types = nl_repr.var_types()
    var_names = nl_repr.var_names()
    var_shapes = nl_repr.var_shapes()
    for i in range(len(var_names)):
        vt = var_types[i]
        name = var_names[i]
        lb_vals = nl_repr.var_lb(i)
        ub_vals = nl_repr.var_ub(i)
        shape_list = var_shapes[i]
        shape = tuple(shape_list) if shape_list else ()
        lb = np.array(lb_vals).reshape(shape) if shape else float(lb_vals[0])
        ub = np.array(ub_vals).reshape(shape) if shape else float(ub_vals[0])

        if vt == "continuous":
            m.continuous(name, shape=shape, lb=lb, ub=ub)
        elif vt == "binary":
            m.binary(name, shape=shape)
        elif vt == "integer":
            m.integer(name, shape=shape, lb=lb, ub=ub)

    # Reconstruct the expression DAG from the Rust arena
    objective_expr, constraint_tuples = reconstruct_dag(nl_repr, m._variables)

    # Set the objective with the reconstructed expression
    if nl_repr.objective_sense == "minimize":
        m.minimize(objective_expr)
    else:
        m.maximize(objective_expr)

    # Add constraints from the reconstructed DAG
    for body, sense, rhs in constraint_tuples:
        if sense == "<=":
            m.subject_to(body <= rhs)
        elif sense == ">=":
            m.subject_to(body >= rhs)
        elif sense == "==":
            m.subject_to(body == rhs)

    # Keep nl_repr for backward compatibility (Rust evaluator for validation)
    m._nl_repr = nl_repr

    return m


def from_gams(path: str) -> Model:
    """
    Import a model from GAMS .gms format.

    Parses GAMS source text and builds a discopt Model.  Supports the
    MINLP subset: Sets, Scalars, Parameters, Tables, Variables
    (positive/binary/integer/free), Equations with ``=e=``/``=l=``/``=g=``,
    bounds (``.lo``/``.up``/``.fx``), ``sum``/``prod`` over indexed domains,
    and nonlinear functions (``exp``, ``log``, ``sin``, ``cos``, ``sqrt``,
    ``power``, ``sqr``, ...).

    Parameters
    ----------
    path : str
        Path to the ``.gms`` file.

    Returns
    -------
    Model

    Examples
    --------
    >>> model = dm.from_gams("process_synthesis.gms")
    >>> result = model.solve()
    """
    from discopt.modeling.gams_parser import parse_gams_file

    result: Model = parse_gams_file(path)
    return result


def from_description(
    description: str,
    data: Optional[dict] = None,
    llm_model: str = "claude-sonnet-4-20250514",
    validate: bool = True,
    explain: bool = True,
) -> Model:
    """
    Create a model from a natural language description using an LLM.

    The LLM generates a discopt Model via function calling (not free-form
    code generation), ensuring type safety.

    Parameters
    ----------
    description : str
        Natural language problem description.
    data : dict, optional
        Named data arrays (DataFrames, numpy arrays, dicts) available
        to the formulation agent.
    llm_model : str, default "claude-sonnet-4-20250514"
        LLM model to use for formulation.
    validate : bool, default True
        Validate the generated model before returning.
    explain : bool, default True
        Print the LLM's explanation of the formulation.

    Returns
    -------
    Model

    Raises
    ------
    NotImplementedError
        LLM formulation is a Phase 2 feature.

    Examples
    --------
    >>> model = dm.from_description(
    ...     "Minimize total shipping cost from 3 warehouses to 5 customers.",
    ...     data={"supply": [100, 150, 200], "demand": [80, 60, 70, 40, 50]},
    ... )
    """
    from discopt.llm import is_available

    if not is_available():
        raise ImportError(
            "LLM formulation requires litellm. Install with: pip install discopt[llm]"
        )

    from discopt.llm.prompts import FORMULATE_SYSTEM, FORMULATE_USER
    from discopt.llm.provider import complete_with_tools
    from discopt.llm.serializer import serialize_data_schema
    from discopt.llm.tools import (
        TOOL_DEFINITIONS,
        ModelBuilder,
        execute_tool_calls,
    )

    data_text = serialize_data_schema(data) if data else ""
    user_msg = FORMULATE_USER.format(description=description, data_schema=data_text)
    messages: list[dict] = [
        {"role": "system", "content": FORMULATE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    builder = ModelBuilder()
    if data:
        builder._namespace.update(data)

    max_turns = 10
    for _ in range(max_turns):
        response = complete_with_tools(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            model=llm_model,
            max_tokens=4096,
            timeout=30.0,
        )

        # complete_with_tools returns the message directly
        msg = response
        msg_dict = msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else msg
        messages.append(msg_dict)

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            break

        tool_results = execute_tool_calls(tool_calls, builder)
        messages.extend(tool_results)

    if builder.model is None:
        raise ValueError("LLM did not create a model")

    if validate:
        builder.model.validate()

    if explain and messages:
        last = messages[-1]
        content = last.get("content", "") if isinstance(last, dict) else ""
        if content:
            print(f"LLM explanation: {content}")

    return builder.model
