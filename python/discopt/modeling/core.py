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
    TYPE_CHECKING,
    Callable,
    Iterator,
    Optional,
    Sequence,
    Union,
    overload,
)

import numpy as np

if TYPE_CHECKING:
    from discopt.modeling.indexed import IndexedParam, IndexedVar
    from discopt.modeling.sets import _SetBase
    from discopt.solver_tuning import SolverTuning

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

    # Tell NumPy to defer to our __matmul__/__rmatmul__ (and other dunder
    # operators) when any Expression appears alongside an ndarray. Without
    # this, ``A @ x`` and ``c @ (xp - xm)`` route through numpy's matmul
    # gufunc, which sees the Expression as a 0-d object and raises
    # ``ValueError: Input operand 1 does not have enough dimensions``.
    __array_ufunc__ = None

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

    # __array_ufunc__ = None is inherited from Expression.

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
        # ``size`` is a hot property on the convexity / AD walkers (called
        # once per leaf, per node visited). ``shape`` is immutable after
        # construction, so cache the product once instead of recomputing
        # ``int(np.prod(shape))`` on every access.
        self._size = int(np.prod(shape)) if shape else 1

    @property
    def size(self) -> int:
        return self._size

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


class CustomCall(Expression):
    """Opaque, AD-only user function wrapping a JAX-traceable callable.

    Unlike :class:`FunctionCall` -- whose ``func_name`` is dispatched to a known
    value / relaxation / interval / ``.nl`` rule -- a ``CustomCall`` carries an
    arbitrary Python callable that discopt evaluates by *tracing it through JAX*.
    discopt can therefore autodifferentiate it for the local NLP path, but it
    CANNOT build the rigorous convex/concave relaxations and interval rules that
    global spatial branch-and-bound, the Rust presolve, and ``.nl`` export
    require.

    Consequences (enforced by the solver and the export/relaxation layers):

    - A model containing a ``CustomCall`` is solved on the **local NLP path
      only** -- the result carries **no global optimality certificate**
      (``gap_certified`` is ``False``).
    - The solver **raises** if integer/binary variables are present, because
      global B&B has no valid node relaxation for an opaque callable.
    - Relaxation compilation and ``.nl`` export **raise** a clear error.

    Built via :func:`custom`; do not instantiate directly in user code.
    """

    def __init__(self, fn: Callable, *args: Expression, name: Optional[str] = None):
        self.fn = fn
        self.args = tuple(args)
        self.name = name or getattr(fn, "__name__", "custom")

    def __repr__(self):
        arg_str = ", ".join(str(a) for a in self.args)
        return f"custom:{self.name}({arg_str})"


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


def _is_term_iterable(x) -> bool:
    """True for a generator/iterable of terms (not a scalar Expression/array/str).

    Indexed containers (:class:`IndexedVar` / :class:`IndexedParam`) are excluded
    even though they are iterable: their iterator yields index *keys*, not term
    expressions, so treating them as a term-iterable would silently fold the keys
    in as constants. ``sum``/``prod`` expand them explicitly instead.
    """
    if isinstance(x, (Expression, np.ndarray, str, bytes)):
        return False
    if getattr(x, "_is_indexed_container", False):
        return False
    return hasattr(x, "__iter__")


def _expand_indexed_container(x):
    """If *x* is an IndexedVar/IndexedParam, return its element expressions."""
    if getattr(x, "_is_indexed_container", False):
        return [x[k] for k in x.index_set]
    return x


def _call_over(fn: Callable, member, over):
    """Call ``fn`` on a set member, unpacking tuple members for ``dimen > 1`` sets."""
    if getattr(over, "dimen", 1) > 1 and isinstance(member, tuple):
        return fn(*member)
    return fn(member)


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


def atan(x: Union[Expression, float]) -> Expression:
    """Inverse tangent (arctan); image in (-π/2, π/2)."""
    return FunctionCall("atan", _wrap(x))


def asin(x: Union[Expression, float]) -> Expression:
    """Inverse sine (arcsin); domain [-1, 1]."""
    return FunctionCall("asin", _wrap(x))


def acos(x: Union[Expression, float]) -> Expression:
    """Inverse cosine (arccos); domain [-1, 1]."""
    return FunctionCall("acos", _wrap(x))


def sinh(x: Union[Expression, float]) -> Expression:
    """Hyperbolic sine."""
    return FunctionCall("sinh", _wrap(x))


def cosh(x: Union[Expression, float]) -> Expression:
    """Hyperbolic cosine."""
    return FunctionCall("cosh", _wrap(x))


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


def _find_owning_model(*exprs: Expression) -> Optional["Model"]:
    """Walk expression DAGs and return the first owning Model found."""
    stack = list(exprs)
    while stack:
        node = stack.pop()
        if isinstance(node, Variable):
            return node.model
        if isinstance(node, IndexExpression):
            stack.append(node.base)
        elif isinstance(node, BinaryOp):
            stack.extend((node.left, node.right))
        elif isinstance(node, UnaryOp):
            stack.append(node.operand)
        elif isinstance(node, FunctionCall):
            stack.extend(node.args)
        elif isinstance(node, MatMulExpression):
            stack.extend((node.left, node.right))
        elif isinstance(node, SumExpression):
            stack.append(node.operand)
        elif isinstance(node, SumOverExpression):
            stack.extend(node.terms)
    return None


def if_else(
    condition: "Constraint",
    then_value: Union[Expression, float],
    else_value: Union[Expression, float],
    *,
    name: Optional[str] = None,
) -> Variable:
    """Piecewise conditional expression (free-function form of :meth:`Model.if_else`).

    Discovers the owning :class:`Model` from the variables appearing in
    *condition* / *then_value* / *else_value* and delegates to
    :meth:`Model.if_else`. See that method for the full contract.

    Parameters
    ----------
    condition : Constraint
        An inequality constraint (e.g. ``x >= 0``).
    then_value, else_value : Expression or float
        Branch values.
    name : str, optional
        Base name for the auxiliary variable.

    Returns
    -------
    Variable
        Auxiliary variable standing for the conditional value.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> w = dm.if_else(x >= 0, dm.maximum(0.25, dm.exp(x) - 1), dm.log(-x + 3))
    """
    if not isinstance(condition, Constraint):
        raise TypeError(
            f"if_else() condition must be a Constraint (an inequality such as "
            f"'x >= 0'), got {type(condition).__name__}"
        )
    model = _find_owning_model(condition.body, _wrap(then_value), _wrap(else_value))
    if model is None:
        raise ValueError(
            "if_else() could not determine the owning Model (no variables found "
            "in condition/then/else); call model.if_else(...) directly instead."
        )
    return model.if_else(condition, then_value, else_value, name=name)


def udf(fn: Callable) -> Callable:
    """Compose a Python-callable user-defined function into the expression DAG.

    ``udf`` is a thin pass-through that documents intent: *fn* must build its
    result entirely from ``discopt.modeling`` primitives (``dm.exp``,
    ``dm.maximum``, ``dm.if_else``, arithmetic on variables, ...), so the
    returned expression is a normal DAG node the solver can relax and bound.
    Unlike an opaque numeric callback (e.g. JuMP's ``register``), the body must
    be symbolic — that is what lets discopt build rigorous relaxations.

    Parameters
    ----------
    fn : callable
        A function whose body uses only ``dm.*`` primitives and operator
        overloading on expressions.

    Returns
    -------
    callable
        *fn* unchanged; call it with expression arguments to build the DAG.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> user_fn = dm.udf(
    ...     lambda x: dm.if_else(x >= 0, dm.maximum(0.25, dm.exp(x) - 1), dm.log(-x + 3))
    ... )
    >>> expr = user_fn(x)   # ordinary discopt expression
    """
    if not callable(fn):
        raise TypeError(f"udf() expects a callable, got {type(fn).__name__}")
    return fn


def custom(fn: Callable, *, name: Optional[str] = None) -> Callable:
    """Wrap an opaque, JAX-traceable callable as an AD-only user function.

    Use this **only** when a function genuinely cannot be expressed with
    ``dm.*`` primitives. If you can write the body symbolically (with
    ``dm.exp``, ``dm.maximum``, ``dm.if_else``, arithmetic on variables, ...),
    use :func:`udf` instead -- that keeps full global-solver support, whereas a
    ``custom`` function is restricted to the local NLP path.

    *fn* must be differentiable by JAX: write it with ``jax.numpy`` operations
    on its array arguments so discopt can autodiff it for gradients/Hessians.
    The returned wrapper builds a :class:`CustomCall` node when called with
    expression arguments::

        import jax.numpy as jnp
        import discopt.modeling as dm

        weird = dm.custom(lambda x: jnp.sum(jnp.sinc(x) ** 2))
        m.minimize(weird(x) + dm.sum(x))

    Because the body is opaque to the relaxation machinery, a model that uses a
    ``dm.custom`` function is solved on the **local NLP path only** -- there is
    no global optimality certificate -- and the solver raises if integer/binary
    variables are present (global branch-and-bound cannot bound an opaque
    callable). See :class:`CustomCall`.

    Parameters
    ----------
    fn : callable
        A JAX-traceable function of one or more array arguments returning a
        scalar (objective term) or array (constraint body).
    name : str, optional
        Display name used in reprs and error messages. Defaults to
        ``fn.__name__``.

    Returns
    -------
    callable
        A builder; call it with expression arguments to produce a
        :class:`CustomCall` DAG node.
    """
    if not callable(fn):
        raise TypeError(f"custom() expects a callable, got {type(fn).__name__}")

    def _build(*args) -> Expression:
        return CustomCall(fn, *[_wrap(a) for a in args], name=name)

    _build.__name__ = name or str(getattr(fn, "__name__", "custom"))
    return _build


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
    x = _expand_indexed_container(x)  # dm.sum(indexed_var) -> sum of its elements
    if over is not None and callable(x):
        # Indexed summation: dm.sum(lambda i: expr(i), over=index_set). Tuple
        # members of dimen>1 named sets are unpacked: ``lambda i, j: ...``.
        terms = [_wrap(_call_over(x, i, over)) for i in over]
        return SumOverExpression(terms)
    if isinstance(x, list) or _is_term_iterable(x):
        # list/tuple/generator of terms: dm.sum(x[i] for i in S)
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
    x = _expand_indexed_container(x)  # dm.prod(indexed_var) -> product of elements
    if over is not None and callable(x):
        terms = [_wrap(_call_over(x, i, over)) for i in over]
        return _multiply_terms(terms)
    if isinstance(x, list) or _is_term_iterable(x):
        terms = [_wrap(t) for t in x]
        return _multiply_terms(terms)
    return FunctionCall("prod", _wrap(x))


def _multiply_terms(terms: list["Expression"]) -> Expression:
    """Fold a list of terms into a product; the empty product is the identity 1."""
    if not terms:
        return Constant(1.0)
    result: Expression = terms[0]
    for t in terms[1:]:
        result = result * t
    return result


def norm(x: Expression, ord: Union[int, float, str] = 2) -> Expression:
    """
    Vector norm.

    Parameters
    ----------
    x : Expression
        Input vector expression.
    ord : int, float, or str, default 2
        Norm order. ``1`` (L1), ``2`` (L2/Euclidean), and ``inf`` (Chebyshev,
        passed as ``float("inf")`` or ``"inf"``) are supported on the global
        certification path (Rust ``MathFunc``). Other integer orders are
        evaluated and relaxed on the JAX path but are not in the core IR.

    Returns
    -------
    Expression
    """
    if ord in (float("inf"), "inf", "Inf"):
        suffix = "inf"
    else:
        suffix = str(ord)
    return FunctionCall(f"norm{suffix}", _wrap(x))


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
    mip_count : int
        Number of MIP/MILP solves performed by the algorithm, when tracked.
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
    mip_count: int = 0

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

    # Witness for an infeasible result, when the backend computed one. An
    # ``InfeasibilityCertificate`` (per-row minimal constraint violations, in
    # LP-row order) for LPs solved via the POUNCE engine; None otherwise.
    infeasibility_certificate: Optional[object] = None

    # Convex fast path indicator
    convex_fast_path: bool = False

    # NLP-BB indicator and gap certification
    nlp_bb: bool = False
    gap_certified: bool = True

    # SubNLP primal-heuristic counters (zero unless the heuristic ran).
    subnlp_calls: int = 0
    subnlp_feasible: int = 0
    subnlp_incumbent_updates: int = 0

    # Examiner-style validation report (populated if validate=True).
    validation_report: Optional[object] = None

    # LLM explanation (populated if llm=True)
    _explanation: Optional[str] = None
    _model: Optional["Model"] = None

    # Sensitivity cache (populated lazily by .gradient())
    _sensitivity: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Soundness guard: a *certified* optimality gap requires a finite dual
        # bound. A non-finite bound (``±inf``) or an absent bound (``None``)
        # certifies nothing about optimality — e.g. a ``time_limit`` termination
        # where no node ever produced a finite relaxation bound leaves the
        # global lower bound at ``-inf``. Reporting ``gap_certified=True`` there
        # is a false certification (the benchmark gate would miscount it as a
        # solved/certified instance), so we downgrade it and clear the
        # meaningless bound/gap. Infeasibility certificates are exempt:
        # ``status="infeasible"`` with ``gap_certified=True`` certifies
        # infeasibility, not a gap, and legitimately carries ``bound=None``.
        if self.gap_certified and self.status != "infeasible":
            if self.bound is None or not np.isfinite(self.bound):
                self.gap_certified = False
                self.bound = None
                self.gap = None

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


def _require_no_shape(shape, ctor: str) -> None:
    """Reject ``shape=`` when ``over=`` is given (they are mutually exclusive)."""
    if shape not in ((), 0):
        raise ValueError(
            f"{ctor}(): 'over=' (named-set index) and 'shape=' are mutually "
            "exclusive; an indexed variable's size is determined by its set."
        )


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
        self._aux_counter = 0  # monotonic suffix for if_else auxiliary names
        # Complementarity conditions added via ``complementarity()``; recorded
        # for introspection and bound tightening (see ``discopt.mpec``).
        self._complementarities: list = []
        # Decomposition annotations (Benders / Lagrangian). Populated by
        # ``set_stage``/``first_stage``/``second_stage``/``set_block``/
        # ``mark_coupling``; consumed by ``discopt.decomposition``. Empty by
        # default, in which case structure is auto-detected.
        self._decomp_stages: dict[str, int] = {}
        self._decomp_blocks: dict[str, int] = {}
        self._coupling_keys: set = set()
        # Named index sets registered via ``set()`` (see ``discopt.modeling.sets``).
        self._sets: list = []
        # Linear constraint blocks emitted directly into the Rust builder
        # (fast-API ``add_linear_constraints`` and the indexed fast path). Each
        # entry is ``(A_csr, x, sense, b, name)``. Kept so .nl export and
        # introspection can recover constraints that bypass ``_constraints``.
        self._builder_linear_blocks: list = []
        # Linear objective emitted directly into the Rust builder via
        # ``add_linear_objective``: ``(c, x, constant, sense)`` or ``None``. Kept
        # so .nl export can recover an objective that bypasses ``_objective``
        # (which only holds a zero placeholder in that case). Linear is
        # ``(c, x, constant, sense)``; quadratic is ``(Q_csr, c, x, constant,
        # sense)`` for ``0.5 x'Qx + c'x + constant``. At most one is set.
        self._builder_linear_objective: Optional[tuple] = None
        self._builder_quadratic_objective: Optional[tuple] = None

    # ── Index sets ──

    def set(self, name: str, members, dimen: Optional[int] = None):
        """Declare a named index set.

        A :class:`~discopt.modeling.sets.Set` is the authoritative index for
        indexed variables, parameters, and constraints. Members may be scalars
        (``dimen == 1``) or fixed-arity tuples; duplicates are removed and
        first-occurrence order is preserved.

        Parameters
        ----------
        name : str
            Identifier for the set, unique among the model's sets.
        members : iterable
            The set members.
        dimen : int, optional
            Declared dimensionality; inferred from *members* when omitted.

        Returns
        -------
        Set
            The registered set, supporting algebra (``|``, ``&``, ``-``, ``*``)
            and filtering (``where``).

        Examples
        --------
        >>> m = Model()
        >>> plants = m.set("plants", ["pitt", "sf", "nyc"])
        >>> markets = m.set("markets", ["a", "b"])
        >>> len(plants * markets)
        6
        """
        from discopt.modeling.sets import Set

        if any(s.name == name for s in self._sets):
            raise ValueError(f"Set name '{name}' already used in model")
        s = Set(name, members, dimen=dimen)
        self._sets.append(s)
        return s

    # ── Variable constructors ──

    def _register_variable(self, var: "Variable") -> "Variable":
        """Append a variable and register it with the Rust builder if active."""
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

    def _make_indexed_var(
        self, name, var_type, index_set, lb, ub, default_lb, default_ub
    ) -> "IndexedVar":
        """Build an :class:`IndexedVar` backed by one flat variable over *index_set*."""
        from discopt.modeling.indexed import IndexedVar, resolve_indexed_values

        lb_arr = resolve_indexed_values(index_set, lb, default_lb, np.float64)
        ub_arr = resolve_indexed_values(index_set, ub, default_ub, np.float64)
        self._check_name(name)
        flat = Variable(name, var_type, (len(index_set),), lb_arr, ub_arr, self)
        self._register_variable(flat)
        return IndexedVar(flat, index_set)

    @overload
    def continuous(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = ...,
        lb: Union[float, np.ndarray] = ...,
        ub: Union[float, np.ndarray] = ...,
        over: None = ...,
    ) -> Variable: ...

    @overload
    def continuous(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = ...,
        lb: Union[float, np.ndarray] = ...,
        ub: Union[float, np.ndarray] = ...,
        *,
        over: "_SetBase",
    ) -> "IndexedVar": ...

    def continuous(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        lb: Union[float, np.ndarray] = -9.999e19,
        ub: Union[float, np.ndarray] = 9.999e19,
        over=None,
    ) -> Union[Variable, "IndexedVar"]:
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

            NLP solvers (ipm, ipopt, pounce) use interior-point barrier methods
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
        >>> ship = m.continuous("ship", over=links, lb=0)   # indexed over a named set

        When *over* is given (a :class:`~discopt.modeling.sets.Set`), an
        :class:`~discopt.modeling.indexed.IndexedVar` is returned: ``ship[key]``
        indexes by set member, and ``lb``/``ub`` may be a scalar, a ``dict``
        keyed by member, or a callable ``fn(member)``.
        """
        if over is not None:
            _require_no_shape(shape, "continuous")
            return self._make_indexed_var(
                name, VarType.CONTINUOUS, over, lb, ub, -9.999e19, 9.999e19
            )
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.CONTINUOUS, shape, lb, ub, self)
        return self._register_variable(var)

    @overload
    def binary(
        self, name: str, shape: Union[int, tuple[int, ...]] = ..., over: None = ...
    ) -> Variable: ...

    @overload
    def binary(
        self, name: str, shape: Union[int, tuple[int, ...]] = ..., *, over: "_SetBase"
    ) -> "IndexedVar": ...

    def binary(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        over=None,
    ) -> Union[Variable, "IndexedVar"]:
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
        >>> assign = m.binary("assign", over=workers * tasks)  # indexed binary
        """
        if over is not None:
            _require_no_shape(shape, "binary")
            return self._make_indexed_var(name, VarType.BINARY, over, 0.0, 1.0, 0.0, 1.0)
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.BINARY, shape, 0.0, 1.0, self)
        return self._register_variable(var)

    @overload
    def integer(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = ...,
        lb: Union[float, np.ndarray] = ...,
        ub: Union[float, np.ndarray] = ...,
        over: None = ...,
    ) -> Variable: ...

    @overload
    def integer(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = ...,
        lb: Union[float, np.ndarray] = ...,
        ub: Union[float, np.ndarray] = ...,
        *,
        over: "_SetBase",
    ) -> "IndexedVar": ...

    def integer(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        lb: Union[float, np.ndarray] = 0,
        ub: Union[float, np.ndarray] = 1e6,
        over=None,
    ) -> Union[Variable, "IndexedVar"]:
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
        >>> n = m.integer("n", over=plants, lb=0, ub=10)  # indexed integer
        """
        if over is not None:
            _require_no_shape(shape, "integer")
            return self._make_indexed_var(name, VarType.INTEGER, over, lb, ub, 0.0, 1e6)
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.INTEGER, shape, lb, ub, self)
        return self._register_variable(var)

    def _make_indexed_param(self, name, index_set, value) -> "IndexedParam":
        """Build an :class:`IndexedParam` backed by one flat parameter over *index_set*."""
        from discopt.modeling.indexed import IndexedParam, resolve_indexed_values

        arr = resolve_indexed_values(index_set, value, 0.0, np.float64)
        self._check_name(name)
        flat = Parameter(name, arr, self)
        self._parameters.append(flat)
        return IndexedParam(flat, index_set)

    @overload
    def parameter(
        self, name: str, value: Union[float, np.ndarray], over: None = ...
    ) -> Parameter: ...

    @overload
    def parameter(self, name: str, value, *, over: "_SetBase") -> "IndexedParam": ...

    def parameter(
        self,
        name: str,
        value: Union[float, np.ndarray],
        over=None,
    ) -> Union[Parameter, "IndexedParam"]:
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
        >>> cap = m.parameter("cap", over=plants, value={"pitt": 10, "sf": 20})

        When *over* is given (a :class:`~discopt.modeling.sets.Set`), an
        :class:`~discopt.modeling.indexed.IndexedParam` is returned and *value*
        may be a scalar, a ``dict`` keyed by member, or a callable ``fn(member)``.
        """
        if over is not None:
            return self._make_indexed_param(name, over, value)
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
        if isinstance(constraint, Constraint):
            constraint.name = name
            self._constraints.append(constraint)
            return
        if isinstance(constraint, bool):
            raise TypeError(
                f"Expected Constraint (from <=, >=, == on expressions), "
                f"got {type(constraint)}. Did you mean to compare expressions?"
            )
        # list / tuple / generator / any iterable of constraints (named
        # by ordinal: ``name_0``, ``name_1``, ...).
        try:
            items = list(constraint)
        except TypeError:
            raise TypeError(
                f"Expected Constraint or an iterable of Constraints, got {type(constraint)}."
            ) from None
        for k, c in enumerate(items):
            if not isinstance(c, Constraint):
                raise TypeError(
                    f"Expected Constraint (from <=, >=, == on expressions), "
                    f"got {type(c)} at position {k}."
                )
            c.name = f"{name}_{k}" if name else None
            self._constraints.append(c)

    def constraint(self, index_set, rule, name: Optional[str] = None, fast: bool = True):
        """Add a family of constraints indexed by a named set.

        For each member of *index_set* the *rule* is evaluated to produce a
        constraint; tuple members are unpacked into positional arguments
        (``rule(i, j)`` for a ``dimen == 2`` set, ``rule(i)`` otherwise). A rule
        may return :data:`~discopt.modeling.indexed.Skip` to omit that member.
        Generated constraints are named ``name[key]`` (e.g. ``capacity[pitt]``).

        When ``fast`` is ``True`` (default) and every generated constraint is
        affine in a single backing variable with a uniform sense, the whole
        family is emitted as one sparse-matrix call into the Rust builder instead
        of thousands of Python expression objects. This is purely a performance
        path: the resulting model is identical, and any family that is not
        single-variable-affine transparently falls back to the general path.
        Pass ``fast=False`` to force the general path (e.g. for debugging).

        Parameters
        ----------
        index_set : Set
            The set to iterate.
        rule : callable
            ``rule(member)`` -> Constraint (or ``Skip``).
        name : str, optional
            Prefix for the per-key constraint names.
        fast : bool, default True
            Allow linear fast-path emission into the Rust builder.

        Returns
        -------
        IndexedConstraint
            A keyed view of the generated constraints.

        Examples
        --------
        >>> m.constraint(plants, lambda p: ship_out(p) <= cap[p], name="capacity")
        """
        from discopt.modeling.indexed import IndexedConstraint, Skip, key_label
        from discopt.modeling.sets import call_member

        generated: list[tuple] = []
        for member in index_set:
            c = call_member(rule, member, index_set.dimen)
            if c is Skip:
                continue
            if not isinstance(c, Constraint):
                raise TypeError(
                    f"constraint rule for key {member!r} returned {type(c)}, "
                    "expected a Constraint (from <=, >=, == on expressions) or Skip."
                )
            c.name = f"{name}[{key_label(member)}]" if name else None
            generated.append((member, c))

        members = {m: c for m, c in generated}
        if fast and self._try_fast_linear_family([c for _, c in generated], name):
            # Rows were emitted into the Rust builder; keep the Constraint
            # objects only for introspection (not in self._constraints).
            return IndexedConstraint(name, index_set, members, fast=True)
        for _, c in generated:
            self._constraints.append(c)
        return IndexedConstraint(name, index_set, members, fast=False)

    def _try_fast_linear_family(self, constraints: list, name: Optional[str]) -> bool:
        """Emit a single-variable-affine, uniform-sense family into the builder.

        Returns ``True`` if the whole family was emitted as one
        ``add_linear_constraints`` call, ``False`` if it is ineligible (caller
        then uses the general expression path).
        """
        from discopt.modeling.indexed import affine_form

        if not constraints:
            return False
        sense = constraints[0].sense
        if any(c.sense != sense for c in constraints):
            return False

        rows = []
        var = None
        for c in constraints:
            aff = affine_form(c.body)
            if aff is None or aff.var is None:
                return False
            if var is None:
                var = aff.var
            elif aff.var is not var:
                return False
            rows.append(aff)

        import scipy.sparse as sp

        m_rows = len(rows)
        n_cols = var.size
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        b = np.empty(m_rows, dtype=np.float64)
        for r, aff in enumerate(rows):
            for pos in sorted(aff.coeffs):
                coeff = aff.coeffs[pos]
                if coeff != 0.0:
                    data.append(coeff)
                    indices.append(pos)
            indptr.append(len(data))
            # body sense 0  ==>  A x sense (-const)
            b[r] = -aff.const
        A = sp.csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(m_rows, n_cols),
        )
        self.add_linear_constraints(A, var, sense, b, name)
        return True

    # ── Decomposition annotations (Benders / Lagrangian) ──

    def _decomp_var_name(self, var) -> str:
        """Resolve a decomposition annotation target to a variable name.

        Accepts a :class:`Variable`, a name string, or an indexed reference
        (``y[i]``) / single-variable expression — resolving the latter to its
        *base variable name*, since decomposition staging is whole-variable.
        Resolving ``y[i]`` to ``"y"`` (rather than the stray ``str(y[i])``, e.g.
        ``"y[3][0]"``, which silently never matches) prevents an annotated
        variable from being misclassified into the recourse subproblem.
        """
        if isinstance(var, Variable):
            return var.name
        if isinstance(var, str):
            return var
        base = getattr(var, "base", None)
        if isinstance(base, Variable):
            return base.name
        try:
            from discopt._jax.gdp_reformulate import _collect_variables

            names = list(_collect_variables(var).keys())
        except Exception:
            names = []
        if len(names) == 1:
            return names[0]
        raise TypeError(
            f"Cannot resolve a decomposition variable from {var!r}; pass a Variable "
            "(e.g. model.first_stage(y)). Staging is per whole variable, so an "
            "expression spanning zero or multiple variables is ambiguous."
        )

    def set_stage(self, var: "Variable", stage: int) -> "Model":
        """Tag a variable with a decomposition stage.

        Stage ``1`` denotes a *complicating* / first-stage variable (held in
        the Benders master); higher stages denote recourse/subproblem
        variables. Consumed by :func:`discopt.decomposition.detect_decomposition`.
        Accepts a :class:`Variable`, a name string, or an indexed reference
        (``y[i]``, resolved to the whole variable). Returns ``self`` for chaining.
        """
        self._decomp_stages[self._decomp_var_name(var)] = int(stage)
        return self

    def first_stage(self, *vars: "Variable") -> "Model":
        """Mark variables as first-stage (complicating) for Benders."""
        for v in vars:
            self.set_stage(v, 1)
        return self

    def second_stage(self, *vars: "Variable") -> "Model":
        """Mark variables as second-stage (recourse/subproblem) for Benders."""
        for v in vars:
            self.set_stage(v, 2)
        return self

    def set_block(self, var: "Variable", block_id: int) -> "Model":
        """Assign a variable to an explicit decomposition block.

        Accepts a :class:`Variable`, a name string, or an indexed reference
        (``y[i]``, resolved to the whole variable).
        """
        self._decomp_blocks[self._decomp_var_name(var)] = int(block_id)
        return self

    def mark_coupling(self, constraint: Union[Constraint, str]) -> "Model":
        """Mark a constraint as *coupling* (to dualize in Lagrangian relaxation).

        Accepts the :class:`Constraint` object added via :meth:`subject_to`, or
        its name string. Coupling constraints are the linking rows whose removal
        separates the model into independent blocks.
        """
        if isinstance(constraint, str):
            self._coupling_keys.add(constraint)
        else:
            self._coupling_keys.add(id(constraint))
            cname = getattr(constraint, "name", None)
            if cname:
                self._coupling_keys.add(cname)
        return self

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
        # Record the block so exporters (.nl) and introspection can recover the
        # constraints that live only in the Rust builder. CSR ``A`` and ``b`` are
        # already in their final float form here.
        self._builder_linear_blocks.append((A, x, sense, b, name))

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
        # Record the block so .nl export can recover the objective that lives
        # only in the Rust builder (``_objective`` holds a zero placeholder).
        self._builder_linear_objective = (c, x, float(constant), sense)
        self._builder_quadratic_objective = None

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
        # Record the block so .nl export can recover the quadratic objective
        # that lives only in the Rust builder.
        self._builder_quadratic_objective = (Q, c, x, float(constant), sense)
        self._builder_linear_objective = None

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

    def complementarity(
        self,
        x: "Expression",
        y: "Expression",
        *,
        method: str = "gdp",
        name: Optional[str] = None,
    ):
        r"""Add a complementarity constraint :math:`0 \le x \perp y \ge 0`.

        Enforces ``x >= 0``, ``y >= 0`` and ``x * y == 0`` — at least one of
        ``x``, ``y`` is zero. This is the defining structure of MPCCs/MPECs,
        KKT-reformulated bilevel programs, and equilibrium models. The smooth
        bilinear equality ``x * y == 0`` is avoided: its relaxation cannot
        capture the either/or structure and it is degenerate at the solution
        (standard constraint qualifications fail).

        This is the fluent front-end for the reformulations in
        :mod:`discopt.mpec`; the condition is recorded on the model (see
        ``Model._complementarities``) and immediately reformulated into ordinary
        constraints solved by :meth:`solve`.

        Parameters
        ----------
        x, y : Expression
            The complementary pair. Both are constrained non-negative.
        method : {"gdp", "sos1"}, default "gdp"
            Reformulation. ``"gdp"`` lowers to the exact disjunction
            ``(x == 0) ∨ (y == 0)`` (big-M with a selector binary), so
            branch-and-bound branches on the finite either/or choice and the
            integrality-aware FBBT infers the partner is zero when one side is
            bounded away from zero — markedly fewer nodes than the bilinear
            encoding. ``"sos1"`` encodes the pair as a Special Ordered Set of
            type 1. For the Scholtes regularization homotopy (a *solve-time*
            algorithm, not a static reformulation), use
            :func:`discopt.mpec.solve_mpec` with ``method="scholtes"``.
        name : str, optional
            Base name for the generated constraints.

        Examples
        --------
        >>> # min (x-1)^2 + (y-1)^2  s.t.  0 <= x ⊥ y >= 0
        >>> m.complementarity(x, y)
        """
        from discopt import mpec

        pair = mpec.Complementarity(_wrap(x), _wrap(y), name)
        self._complementarities.append(pair)
        if method == "gdp":
            mpec.reformulate_gdp(self, [pair])
        elif method == "sos1":
            mpec.reformulate_sos1(self, [pair])
        elif method == "scholtes":
            raise ValueError(
                "method='scholtes' is a solve-time regularization homotopy, not "
                "a static reformulation. Build the pair with "
                "discopt.mpec.complementarity(...) and call "
                "discopt.mpec.solve_mpec(model, pairs, method='scholtes')."
            )
        else:
            raise ValueError(
                f"unknown complementarity method {method!r}; use 'gdp' or 'sos1' "
                "(or discopt.mpec.solve_mpec for 'scholtes')."
            )

    def _branch_bounds(
        self, then_expr: "Expression", else_expr: "Expression"
    ) -> tuple[float, float]:
        """Sound bounds for an if_else auxiliary variable.

        The auxiliary ``w`` must enclose the image of *both* branches, since
        either may be selected. Returns ``(min lo, max hi)`` over the static
        interval enclosures of the two branch expressions. Falls back to the
        default (huge) bounds when either enclosure is non-finite or cannot be
        computed; presolve/FBBT tighten from there. Always sound.
        """
        from discopt._jax.convexity.interval_eval import evaluate_interval

        los: list[float] = []
        his: list[float] = []
        for e in (then_expr, else_expr):
            try:
                iv = evaluate_interval(e, self)
                lo = float(np.asarray(iv.lo).reshape(()))
                hi = float(np.asarray(iv.hi).reshape(()))
            except Exception:
                return -9.999e19, 9.999e19
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return -9.999e19, 9.999e19
            los.append(lo)
            his.append(hi)
        return min(los), max(his)

    def if_else(
        self,
        condition: "Constraint",
        then_value: Union["Expression", float],
        else_value: Union["Expression", float],
        *,
        name: Optional[str] = None,
    ) -> Variable:
        """Piecewise conditional expression ``then if condition else otherwise``.

        Introduces an auxiliary continuous variable ``w`` and a two-way
        disjunction so that, when *condition* holds, ``w == then_value``, and
        when it fails, ``w == else_value``. The disjunction is lowered to a
        big-M / hull reformulation by the GDP pass at solve time, so ``w`` may
        be used anywhere a continuous expression is allowed (objective or
        constraint bodies), composes with the relaxation pipeline, and keeps a
        valid global lower bound under branch-and-bound.

        Parameters
        ----------
        condition : Constraint
            An *inequality* constraint (e.g. ``x >= 0``). Equality conditions
            are rejected: a zero-measure condition is not a meaningful split.
        then_value : Expression or float
            Value of the result when *condition* holds.
        else_value : Expression or float
            Value of the result when *condition* fails.
        name : str, optional
            Base name for the auxiliary variable and disjunction.

        Returns
        -------
        Variable
            The auxiliary variable ``w`` standing for the conditional value.

        Notes
        -----
        On the boundary where the condition holds with equality, *both*
        disjuncts are feasible, so the relaxed graph includes both branch
        values at that single point. This is a sound over-approximation for a
        global solver — it never excludes a true optimum.

        Examples
        --------
        >>> # user_function_1d(x) = (x >= 0) ? max(0.25, exp(x) - 1) : log(-x + 3)
        >>> w = m.if_else(x >= 0, maximum(0.25, exp(x) - 1), log(-x + 3))
        >>> m.subject_to(y >= w)
        """
        if not isinstance(condition, Constraint):
            raise TypeError(
                f"if_else() condition must be a Constraint (an inequality such "
                f"as 'x >= 0'), got {type(condition).__name__}"
            )
        if condition.sense == "==":
            raise ValueError(
                "if_else() condition must be an inequality, not an equality; "
                "an equality condition holds on a zero-measure set."
            )
        then_expr = _wrap(then_value)
        else_expr = _wrap(else_value)
        lb, ub = self._branch_bounds(then_expr, else_expr)
        self._aux_counter += 1
        base = name or "ifelse"
        w = self.continuous(f"_{base}_{self._aux_counter}", lb=lb, ub=ub)
        # condition is normalized to ``body <= 0``; its complement is ``-body <= 0``.
        complement = Constraint(-condition.body, sense="<=", rhs=0.0)
        # Force the hull (perspective) reformulation: it disaggregates the
        # disjunct variables, so each branch's nonlinear body is relaxed only
        # over its own active region. Big-M keeps every branch's equation in
        # the global model and yields an unreliable relaxation bound for
        # nonlinear equality disjuncts (it can cut off the true optimum).
        self._constraints.append(
            _DisjunctiveConstraint(
                disjuncts=[
                    [condition, w == then_expr],
                    [complement, w == else_expr],
                ],
                name=f"_{base}_{self._aux_counter}_disj",
                method="hull",
            )
        )
        return w

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
        gauss_newton: bool = False,
        tuning: Optional["SolverTuning"] = None,
        **kwargs,
    ) -> Union[SolveResult, Iterator["SolveUpdate"]]:
        r"""
        Solve the model.

        For convex pure-continuous models, solves the NLP directly. For
        nonconvex continuous models and models with integer/binary variables,
        uses spatial Branch & Bound unless another solver backend is selected.

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
        rlt : bool or str, default "auto"
            Reformulation-Linearization Technique control for the McCormick LP
            relaxation. ``"auto"`` defers per-node RLT cuts to the structure-gated
            cut policy; ``True`` engages RLT in full (build-time level-1 root-bound
            tightening plus per-node cuts); ``False`` forces it off. Replaces the
            legacy ``DISCOPT_RLT=1`` environment variable. Sound regardless of
            setting (a constraint×bound product never removes a feasible point).
            Passed through to :func:`discopt.solver.solve_model`.
        branching_policy : str, default "fractional"
            Variable selection policy: ``"fractional"`` (most-fractional)
            or ``"gnn"`` (GNN scoring, future hook).
        initial_solution : dict, optional
            Initial feasible solution mapping Variable objects to values
            (scalars, lists, or numpy arrays).  Used as a warm-start point
            for NLP solves, AMP local incumbent improvement, and as the initial
            incumbent in Branch & Bound.
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
        solver : str, optional
            Optional backend selector. Use ``solver="amp"`` to select
            Adaptive Multivariate Partitioning. AMP-specific keyword
            arguments include ``rel_gap``, ``abs_tol``, ``max_iter``,
            ``n_init_partitions``, ``partition_method``, ``milp_time_limit``,
            ``milp_gap_tolerance``, ``presolve_bt``, ``presolve_bt_algo``,
            ``presolve_bt_time_limit``, ``presolve_bt_mip_time_limit``,
            ``apply_partitioning``, ``disc_var_pick``,
            ``partition_scaling_factor``, ``partition_scaling_factor_update``,
            ``disc_add_partition_method``, ``disc_abs_width_tol``,
            ``convhull_formulation``, ``convhull_ebd``,
            ``convhull_ebd_encoding``, ``use_start_as_incumbent``,
            ``obbt_at_root``, ``obbt_with_cutoff``, ``alphabb_cutoff_obbt``,
            and ``obbt_time_limit``.
            Use ``solver="mip-nlp"`` to select the MIP-NLP decomposition
            family. Current implemented ``mip_nlp_method`` values are ``"oa"``,
            ``"ecp"``, and ``"fp"``; ``"goa"``, ``"roa"``, and
            ``"lp_nlp_bb"`` are reserved until their dedicated implementations
            land. Top-level OA/ECP options such as ``equality_relaxation``,
            ``ecp_mode``, ``feasibility_cuts``, ``heuristic_nonconvex``,
            ``add_slack``, ``max_slack``, ``oa_penalty_factor``,
            ``add_no_good_cuts``, ``feasibility_norm``, ``add_regularization``,
            ``level_coef``, ``stalling_limit``, ``cycling_check``, and
            ``init_strategy`` take precedence over duplicate keys in
            ``mip_nlp_options``. Supported ``add_regularization`` values are
            ``"level_L1"``, ``"level_L2"``, ``"level_L_infinity"``,
            ``"grad_lag"``, ``"hess_lag"``, ``"hess_only_lag"``, and
            ``"sqp_lag"``. Supported initialization strategies are ``"rNLP"``,
            ``"initial_binary"``, ``"max_binary"``, and ``"fp"``.
            The ``mip_nlp_method`` selector determines the
            effective ``ecp_mode`` and cannot be overridden by
            ``mip_nlp_options``.
        validate : bool, default False
            If True, run Examiner-style KKT validation on the returned
            point and attach the :class:`~discopt.validation.ExaminerReport`
            to ``result.validation_report``. Errors during validation are
            swallowed and leave ``validation_report`` as ``None``.
        tuning : SolverTuning, optional
            Advanced relaxation / branch-and-bound tuning (RLT families, McCormick
            separation toggles, node-bound mode, …) as a single typed, validated,
            per-call object — the supported replacement for the legacy
            ``DISCOPT_*`` environment variables (which remain as deprecated
            defaults). ``None`` resolves each field from its env default, exactly
            reproducing the prior behavior. Example:
            ``model.solve(tuning=SolverTuning(rlt_quad=False, node_bound_mode="milp"))``.
        gauss_newton : bool, default False
            If True and the objective is a non-negative-weighted sum of squares
            (e.g. ``dm.sum((C @ S - D) ** 2)`` or an explicit
            ``Σ (resid_i) ** 2``), use the Gauss-Newton objective Hessian
            ``2 Jᵀ J`` of the residuals instead of the dense ``jax.hessian``.
            This sidesteps the super-linear second-derivative XLA compile that
            can dominate least-squares solves (issue #98). The Gauss-Newton
            Hessian is always PSD and exact at a zero-residual solution, but
            drops the ``Σ rᵢ ∇²rᵢ`` curvature term, so it changes the Newton
            step (iteration path) without changing the KKT point converged to.
            Silently falls back to the exact dense Hessian when the objective
            is not a recognized sum of squares (or the model maximizes).
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

        # Opt-in Gauss-Newton objective Hessian (issue #98). Read by
        # ``solver._make_evaluator`` when (re)building the NLPEvaluator; toggling
        # it participates in the evaluator-cache fingerprint.
        self._gauss_newton_hessian = bool(gauss_newton)

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
                solver=solver,
                tuning=tuning,
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

    # ── Infeasibility analysis ──

    def compute_iis(self, *, include_bounds: bool = True, time_limit: float = 30.0):
        """Compute an Irreducible Infeasible Subsystem explaining infeasibility.

        Returns an :class:`~discopt.infeasibility.IISResult` — a minimal set of
        constraints (and, by default, variable bounds) that is infeasible but
        becomes feasible if any single member is removed. Use it to debug *why*
        ``solve()`` returned ``"infeasible"``. Raises ``ValueError`` if the model
        is not provably infeasible.

        >>> res = m.solve()
        >>> if res.status == "infeasible":
        ...     print(m.compute_iis().summary())
        """
        from discopt.infeasibility import compute_iis

        return compute_iis(self, include_bounds=include_bounds, time_limit=time_limit)

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
    # Optional per-disjunction reformulation override ("hull" | "big-m" | ...).
    # When set, it takes precedence over the solver-wide ``gdp_method``. Used by
    # ``Model.if_else`` to force the exact hull (perspective) reformulation,
    # which is robust for nonlinear equality disjuncts where big-M's relaxation
    # bound is unreliable.
    method: Optional[str] = None


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
    # Record the source path so the solver can hand POUNCE the original .nl for
    # native-AD node NLP solves (discopt.solvers.nlp_native), bypassing the JAX
    # callback bridge. The .nl column order is the model's variable order here,
    # so the native problem aligns with the evaluator's flat x (identity map).
    m._source_nl_path = os.path.abspath(path)

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
