"""
discopt Modeling API

A clean, expressive Python API for formulating Mixed-Integer Nonlinear Programs.
Designed for:
  - Readability: models look like the math
  - JAX compatibility: expressions are traceable and JIT-compilable
  - Rust interop: expression graphs map to the Rust DAG for structure detection
  - LLM integration: the API doubles as the tool-calling schema for the formulation agent

Example:
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
        Real-valued variable (default bounds: ``[-1e20, 1e20]``).
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
        Termination status. One of ``"optimal"``, ``"feasible"``,
        ``"infeasible"``, ``"time_limit"``, ``"node_limit"``.
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

    # LLM explanation (populated if llm=True)
    _explanation: Optional[str] = None
    _model: Optional["Model"] = None

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

    def gradient(self, param: Parameter) -> np.ndarray:
        """
        Sensitivity of optimal objective w.r.t. a parameter.

        Uses implicit differentiation through KKT conditions.

        Parameters
        ----------
        param : Parameter
            A parameter from the solved model.

        Returns
        -------
        numpy.ndarray
            Gradient ``d(obj*)/d(param)``.

        Raises
        ------
        NotImplementedError
            Sensitivity analysis is a Phase 3 feature.
        """
        raise NotImplementedError("Sensitivity analysis requires JAX backend (Phase 3 feature)")

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

    # ── Variable constructors ──

    def continuous(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        lb: Union[float, np.ndarray] = -1e20,
        ub: Union[float, np.ndarray] = 1e20,
    ) -> Variable:
        """
        Create continuous decision variable(s).

        Parameters
        ----------
        name : str
            Variable name (must be unique in the model).
        shape : int or tuple of int, default ()
            Scalar ``()`` or tuple for array variables.
        lb : float or numpy.ndarray, default -1e20
            Lower bound (scalar broadcast to *shape*, or array matching *shape*).
        ub : float or numpy.ndarray, default 1e20
            Upper bound (scalar broadcast to *shape*, or array matching *shape*).

        Returns
        -------
        Variable
            Expression that can be used in objectives and constraints.

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
        **kwargs,
    ) -> Union[SolveResult, Iterator["SolveUpdate"]]:
        """
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
        **kwargs
            Additional keyword arguments passed to the solver backend.

        Returns
        -------
        SolveResult or Iterator[SolveUpdate]
            Solve result, or a streaming iterator if ``stream=True``.

        Raises
        ------
        ValueError
            If the model fails validation (no objective, duplicate names, etc.).
        """
        self.validate()

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

        from discopt.solver import solve_model

        result = solve_model(
            self,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            threads=threads,
            deterministic=deterministic,
            partitions=partitions,
            branching_policy=branching_policy,
            **kwargs,
        )

        # Attach model reference and auto-generate LLM explanation
        result._model = self
        if llm:
            try:
                result._explanation = result._explain_with_llm()
            except Exception:
                pass

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

    Parameters
    ----------
    path : str
        Path to the ``.gms`` file.

    Returns
    -------
    Model

    Raises
    ------
    NotImplementedError
        GAMS import is a Phase 1 feature.
    """
    raise NotImplementedError("GAMS import requires Rust parser (Phase 1)")


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
