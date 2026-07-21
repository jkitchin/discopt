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
import warnings
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
    Sequence,
    Union,
    overload,
)

import numpy as np

from discopt.constants import SENTINEL_THRESHOLD as _SENTINEL_THRESHOLD

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
        General integer variable. A user-supplied ``lb``/``ub`` is honored
        exactly; an *unspecified* bound falls back to the finite defaults
        ``[0, 1e6]`` and emits a ``UserWarning`` (see :meth:`Model.integer`),
        because a silent default box can cut a true optimum that lies below 0
        or above 1e6 (correctness issue C-6).
    """

    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"


# Finite fallback bounds for a general-integer variable whose lb/ub is left
# unspecified. B&B requires a bounded integer domain, so we substitute these —
# but LOUDLY (a UserWarning names the variable and the imposed range) rather
# than silently truncating the user's problem (correctness issue C-6). A
# user-provided bound is always honored exactly and never replaced by these.
_INTEGER_DEFAULT_LB = 0.0
_INTEGER_DEFAULT_UB = 1e6


# ─────────────────────────────────────────────────────────────
# Expression System
#
# All operations on Variables produce Expression objects that
# build a DAG. This DAG is later compiled to:
#   (1) A Rust-side expression graph for structure detection
#   (2) A JAX-traceable function for evaluation and autodiff
# ─────────────────────────────────────────────────────────────


# Sentinel for "static shape not yet computed" on a composite node, so a cached
# ``None`` ("computed, but unknowable") is distinguishable from "never computed"
# (M8 shape inference). A unique object, never a valid shape.
_UNSET_SHAPE: Any = object()


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

    # Cached static shape for composite nodes (populated at construction by
    # ``BinaryOp``/``UnaryOp`` where inferable; see M8). Class default of the
    # ``_UNSET_SHAPE`` sentinel means "not computed" so a cached ``None``
    # ("computed, unknown") is distinguishable. Leaf nodes (Variable/Constant/
    # Parameter) carry their own ``.shape`` and are handled in ``_known_shape``.
    _shape: Any = _UNSET_SHAPE

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

    def __ne__(self, other):
        # ``!=`` is not a valid optimization-constraint operator. Because
        # ``__eq__`` is overridden to build a Constraint, Python's default
        # ``__ne__`` would evaluate ``not <truthy Constraint>`` → ``False``
        # *silently*, mis-encoding the user's intent (correctness issue C-11).
        # Refuse loudly instead of silently transforming the model (CLAUDE.md §3).
        raise TypeError(
            "'!=' is not a valid constraint operator on modeling expressions. "
            "Use '==', '<=', or '>=' to build a constraint."
        )

    # ── Hashing / boolean context (M4) ──
    #
    # ``__eq__`` is overloaded to *build a Constraint* (the modeling DSL —
    # ``x == 5`` yields ``x - 5 == 0``), which is intentional and matches
    # Pyomo/cvxpy. But that overload has two silent, dangerous consequences we
    # must neutralize:
    #
    #   1. Defining ``__eq__`` makes every ``Expression`` subclass unhashable by
    #      default (Python sets ``__hash__`` to ``None``), so ``{u + v: 1}`` and
    #      ``expr in some_set`` blow up or misbehave. We restore **identity
    #      hashing** (``id(self)``) so expressions behave sanely as set/dict keys
    #      — two syntactically identical expressions are distinct objects, which
    #      is the only sound choice given value-equality is unavailable.
    #   2. ``expr in [other]`` / ``if expr:`` route through ``__eq__``/``__bool__``
    #      and would evaluate the truthiness of the *Constraint* returned by
    #      ``==``, which is always truthy — so ``u in [v]`` is silently ``True``
    #      for any expressions. Making membership sound requires the container to
    #      short-circuit on identity, which Python's ``in`` does *before* calling
    #      ``__eq__`` only when the objects are identical; for distinct objects it
    #      still calls ``__eq__``. We therefore also raise in ``Constraint.__bool__``
    #      (below) so ``if x == 5:`` / ``u in [v]`` fail loudly instead of lying.
    #
    # NOTE: identity hashing does NOT change the ``==``-builds-a-Constraint
    # overload — that remains the modeling API. It only fixes set/dict/`in`
    # semantics for expressions used as keys.
    __hash__ = object.__hash__

    # ── Indexing for array variables ──

    def __getitem__(self, idx):
        # Out-of-bounds guard (issue #816) lives on the *operator*, not on the
        # ``IndexExpression`` constructor: the expression-graph machinery (GAMS
        # import, NL export resolver, signomial/sign-domain pattern matching)
        # deliberately builds ``IndexExpression(base, idx)`` with out-of-range,
        # non-integer, or too-many-index values as lazy nodes that it resolves
        # conservatively later, and must never raise on construction. User-facing
        # ``x[i]`` typos, by contrast, should fail loudly — but only for the
        # unambiguous integer cases (a plain int, or a full-arity all-int tuple);
        # slices, ellipsis, wrong-arity, and non-integer indices stay lenient.
        base_shape = _known_shape(self)
        if base_shape is not None and len(base_shape) >= 1:
            err = _integer_index_out_of_range(base_shape, idx)
            if err is not None:
                raise err
        return IndexExpression(self, idx)

    # ── Shape / length / iteration for array-valued expressions (issue #816) ──

    @property
    def shape(self) -> tuple[int, ...]:
        """Static numpy-style shape of this expression.

        Inferred from the operands for the nodes whose shape follows
        unambiguously from numpy broadcasting: leaf ``Variable``/``Parameter``/
        ``Constant``, their element-wise ``+ - * / **`` / ``neg`` / ``abs``
        compositions, indexing/slicing, and element-wise function calls
        (``exp``, ``sqrt``, ``sin``, …). Raises :class:`AttributeError` when the
        shape is not statically known (reductions, matmul, custom calls, …) so
        that ``getattr(expr, "shape", default)`` and ``hasattr(expr, "shape")``
        keep their pre-existing behaviour for shape-unknown nodes.
        """
        s = _known_shape(self)
        if s is None:
            raise AttributeError(f"{type(self).__name__} has no statically known shape")
        return s

    @shape.setter
    def shape(self, value: tuple[int, ...]) -> None:
        # Leaf nodes (Variable/Parameter) store their declared shape here; the
        # getter then surfaces it through the same ``_known_shape`` path used for
        # composite nodes, keeping a single source of truth.
        self._shape = value

    def __len__(self) -> int:
        """Length along the leading axis (``shape[0]``) for array expressions.

        Raises ``TypeError`` for scalar or shape-unknown expressions, matching
        numpy's behaviour for 0-d arrays. Defining ``__len__`` alongside
        ``__iter__`` stops Python from falling back to the legacy
        sequence-iteration protocol that — because out-of-range indexing now
        raises ``IndexError`` — used to loop forever (issue #816).
        """
        s = _known_shape(self)
        if s is None:
            raise TypeError(
                f"object of type '{type(self).__name__}' has no len(): its "
                "shape is not statically known"
            )
        if len(s) == 0:
            raise TypeError(f"len() of unsized scalar {type(self).__name__}")
        return int(s[0])

    def __iter__(self) -> "Iterator[Expression]":
        """Yield the leading-axis elements (``expr[0]``, ``expr[1]``, …).

        Raises ``TypeError`` for scalar or shape-unknown expressions instead of
        hanging, so ``list(x)``, ``sum(x)``, ``np.array(x)`` and unpacking
        behave like numpy for shaped variables (issue #816).
        """
        s = _known_shape(self)
        if s is None:
            raise TypeError(
                f"cannot iterate over {type(self).__name__} with unknown shape; "
                "index it explicitly instead"
            )
        if len(s) == 0:
            raise TypeError(f"cannot iterate over scalar {type(self).__name__}")
        return (self[i] for i in range(int(s[0])))

    # ── Matrix operations ──

    def __matmul__(self, other):
        return MatMulExpression(self, _wrap(other))

    def __rmatmul__(self, other):
        return MatMulExpression(_wrap(other), self)

    def _repr_latex_(self):
        """Jupyter/IPython LaTeX rendering."""
        # Render the expression DAG as real math rather than wrapping the plain
        # repr in $...$ (which shows broken math in Jupyter) (L8).
        try:
            from discopt.modeling.latex import expr_to_latex

            return f"${expr_to_latex(self)}$"
        except Exception:
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


def _index_result_shape(base_shape: tuple[int, ...], index) -> Optional[tuple[int, ...]]:
    """Numpy result shape of ``base_shape[index]``, or ``None`` when static
    inference is not possible (issue #816).

    Builds a zero-storage broadcast *view* of the base shape (no allocation, even
    for large variables) and applies the *same* index the JAX compiler applies at
    evaluation time (``base[index]`` — see ``dag_compiler._compile_node``). This
    is a pure *best-effort shape query*: it NEVER raises. Any index numpy cannot
    resolve against the shape (out of range, too many indices, a symbolic/exotic
    index object) yields ``None`` = "unknown". The out-of-bounds *guard* that
    surfaces user typos lives in :meth:`Expression.__getitem__`, so that internal
    lazy ``IndexExpression`` construction stays conservative.
    """
    try:
        probe = np.broadcast_to(np.zeros((), dtype=np.int8), base_shape)
        return tuple(int(d) for d in np.shape(probe[index]))
    except Exception:
        return None


def _integer_index_out_of_range(base_shape: tuple[int, ...], idx):
    """Return an ``IndexError`` to raise if *idx* is pure-integer indexing that is
    out of range for *base_shape*, else ``None`` (issue #816).

    Only a plain integer or a full-arity all-integer tuple is checked — exactly
    the ``x[99]`` / ``X[i, j]`` typo the issue calls out. Slices, ellipsis,
    ``newaxis``, non-integer keys, and wrong-arity (too-many / too-few) tuples
    are left to the lenient path: those are the lazy forms the expression-graph
    machinery constructs and resolves downstream, so tightening them here would
    be a false rejection.
    """
    if isinstance(idx, (int, np.integer)):
        idx_tuple: tuple = (int(idx),)
    elif (
        isinstance(idx, tuple)
        and len(idx) == len(base_shape)
        and all(isinstance(i, (int, np.integer)) for i in idx)
    ):
        idx_tuple = tuple(int(i) for i in idx)
    else:
        return None
    for axis, (i, n) in enumerate(zip(idx_tuple, base_shape)):
        if i < -n or i >= n:
            return IndexError(f"index {i} is out of bounds for axis {axis} with size {n}")
    return None


class IndexExpression(Expression):
    """Result of indexing into an array variable: x[i] or x[0, 1]."""

    def __init__(self, base: Expression, index):
        self.base = base
        self.index = index
        # Best-effort static shape inference only (issue #816). Construction is
        # intentionally non-raising: the out-of-bounds guard lives on the ``[]``
        # operator (:meth:`Expression.__getitem__`) so that direct
        # ``IndexExpression(base, idx)`` construction by the expression-graph
        # machinery — which uses out-of-range / non-integer / too-many-index
        # forms as lazy nodes — stays conservative. A shaped (ndim >= 1) base
        # with a statically resolvable index gets a concrete shape; everything
        # else (scalar base, unknown base, or an index numpy cannot resolve)
        # stays shape-unknown.
        base_shape = _known_shape(base)
        if base_shape is not None and len(base_shape) >= 1:
            self._shape = _index_result_shape(base_shape, index)
        else:
            self._shape = None

    def __repr__(self):
        return f"{self.base}[{self.index}]"


def _known_shape(node: "Expression") -> Optional[tuple[int, ...]]:
    """Best-effort static shape of an expression, or ``None`` when unknown (M8).

    Deliberately *conservative*: it returns a concrete shape only for the nodes
    whose shape follows unambiguously from numpy rules — leaves
    (``Variable``/``Constant``/``Parameter``), element-wise ``BinaryOp`` /
    ``UnaryOp`` compositions, indexing/slicing (``IndexExpression``), and
    element-wise ``FunctionCall`` (all of which cache their shape at
    construction). For every other node (matmul, reductions, custom calls,
    indexed containers, …) it returns ``None`` = "don't know", so the caller must
    treat ``None`` as "cannot check" and never raise. This guarantees the M8
    shape guard only ever fires on a *provably* incompatible pair and never
    rejects a valid model (cert-baseline neutral).
    """
    cached = getattr(node, "_shape", _UNSET_SHAPE)
    if cached is not _UNSET_SHAPE:
        # ``cached`` is either a concrete shape tuple or ``None`` (computed-unknown).
        # Variable/Parameter store their declared shape here via the ``shape``
        # setter, so they are covered by this branch too.
        return None if cached is None else tuple(cached)
    if isinstance(node, Constant):
        return tuple(node.value.shape)
    return None


def _broadcast_shapes(
    op: str, s_left: tuple[int, ...], s_right: tuple[int, ...]
) -> tuple[int, ...]:
    """Numpy broadcast of two *known* shapes, raising a clear error on mismatch.

    Only called when both operand shapes are known (M8). A broadcast failure is a
    genuine build-time error — the same operation would blow up deep in the JAX
    NLPEvaluator with an opaque traceback, or (worse) be silently mis-extracted —
    so raise here with both operand shapes named.
    """
    # Hot path: building a large McCormick relaxation constructs millions of
    # element-wise ``BinaryOp`` nodes (~1.5M for a single qap node), the vast
    # majority on identical shapes — scalars ``() op ()`` or matched arrays. Equal
    # shapes broadcast to themselves, and a scalar broadcasts to the other operand;
    # both are exact and skip the ~10x-slower ``numpy.broadcast_shapes`` machinery
    # (profiled ~2.5s of a 12.6s 4-node build). Result is identical to the numpy
    # path, so this is bound-neutral. Differing non-scalar shapes still defer to
    # numpy, which additionally validates broadcast-compatibility.
    if s_left == s_right or s_right == ():
        return s_left
    if s_left == ():
        return s_right
    try:
        return tuple(int(d) for d in np.broadcast_shapes(s_left, s_right))
    except ValueError:
        raise ValueError(
            f"Incompatible operand shapes for '{op}': {s_left} and {s_right} do "
            f"not broadcast (numpy broadcasting rules). Check the shapes of the "
            f"two operands — a shaped variable/parameter must broadcast against "
            f"the other side of the operation."
        ) from None


class BinaryOp(Expression):
    """Binary operation: a op b."""

    def __init__(self, op: str, left: Expression, right: Expression):
        self.op = op
        self.left = left
        self.right = right
        # Static shape inference + build-time shape check (M8). When BOTH operand
        # shapes are statically known, verify they broadcast and cache the result
        # so compositions stay O(1). If either is unknown, the shape is unknown
        # (``None``) and we do not check — conservative, never a false rejection.
        s_left = _known_shape(left)
        s_right = _known_shape(right)
        if s_left is not None and s_right is not None:
            self._shape = _broadcast_shapes(op, s_left, s_right)
        else:
            self._shape = None

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class UnaryOp(Expression):
    """Unary operation: op(a)."""

    def __init__(self, op: str, operand: Expression):
        self.op = op
        self.operand = operand
        # ``neg``/``abs`` are element-wise and shape-preserving: propagate the
        # operand's known shape so it flows into a subsequent BinaryOp check (M8).
        self._shape = _known_shape(operand)

    def __repr__(self):
        return f"{self.op}({self.operand})"


# Named functions that are element-wise: their output shape is the numpy
# broadcast of their argument shapes. Reductions (``prod``) and any other
# non-element-wise name are deliberately excluded so their shape is reported as
# unknown (``None``) rather than mis-inferred (issue #816). Kept in sync with the
# ``FunctionCall``-producing helpers below and the JAX/Rust evaluators, which
# apply these functions element-wise.
_ELEMENTWISE_FUNCS: frozenset = frozenset(
    {
        "exp",
        "log",
        "log2",
        "log10",
        "log1p",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "atan",
        "asin",
        "acos",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "erf",
        "tanh",
        "sigmoid",
        "softplus",
        "abs",
        "sign",
        "min",
        "max",
    }
)


class FunctionCall(Expression):
    """Named function call: exp(x), log(x), sin(x), etc."""

    def __init__(self, func_name: str, *args: Expression):
        self.func_name = func_name
        self.args = args
        # Element-wise functions preserve / broadcast their argument shapes, so
        # cache the result shape to let it flow through subsequent element-wise
        # ops (issue #816). Non-element-wise names (e.g. the reduction ``prod``)
        # keep an unknown shape.
        if func_name in _ELEMENTWISE_FUNCS:
            shp: Optional[tuple[int, ...]] = ()
            for a in args:
                s = _known_shape(a)
                if s is None:
                    shp = None
                    break
                shp = _broadcast_shapes(func_name, shp, s)
            self._shape = shp
        else:
            self._shape = None

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

    - If the opaque body traces soundly through discopt's reduced-space McCormick
      type (``MCBox`` -- arithmetic ``+ - * / **`` and the ``discopt._jax.mcbox``
      intrinsic namespace), a **continuous** model is now solved **globally with a
      certificate** via the reduced-space engine, branching only on the original
      degrees of freedom while the callable's internal intermediates stay hidden
      (MAiNGO-parity plan P3.1, #713). A body that uses raw ``jnp`` intrinsics on
      its arguments (e.g. ``jnp.exp(x)`` rather than an ``MCBox``-aware op), a
      non-affine hidden division, a non-scalar leaf, or an unbounded box is **not**
      reduced-relaxable and falls back to the **local NLP path only** (no global
      certificate, ``gap_certified`` is ``False``) -- sound-or-refuse.
    - Integer/binary variables are supported **when the body is MCBox-relaxable**
      (plan P3.2): the model is branched over the integer + continuous DOF with
      reduced-space node bounds and certified globally. A **non**-MCBox-relaxable
      body together with integers has no valid node relaxation, so the solver
      **raises** (sound-or-refuse).
    - Rust presolve/FBBT does not run on a hidden model (the Rust IR has no opaque
      node); interval bounds on the ``CustomCall`` output come from the ``MCBox``
      ``lo/hi`` propagated during the reduced-space trace (Python interval only).
    - Relaxation compilation (the *lifted* McCormick compiler, which has no
      auxiliary column for an opaque node) and ``.nl`` export **raise** a clear
      error; the reduced-space path is the sound relaxation route.

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


def _iter_model_leaves(*exprs: Expression):
    """Yield every ``Variable`` / ``Parameter`` leaf reachable from *exprs*.

    Walks the expression DAG(s) and surfaces the model-owning leaf nodes
    (``Variable`` and ``Parameter``, both of which carry a ``.model`` back-
    reference). ``Constant`` leaves — scalars, numpy arrays, broadcasts — carry
    no model and are simply not yielded, so an ownership check built on this
    walker never trips on legitimate constants. Used by the cross-model
    ownership guard (finding M3) to detect a variable/parameter that belongs to
    a different :class:`Model` than the one it is being solved in.
    """
    stack = list(exprs)
    while stack:
        node = stack.pop()
        if isinstance(node, (Variable, Parameter)):
            yield node
        elif isinstance(node, IndexExpression):
            stack.append(node.base)
        elif isinstance(node, BinaryOp):
            stack.extend((node.left, node.right))
        elif isinstance(node, UnaryOp):
            stack.append(node.operand)
        elif isinstance(node, FunctionCall):
            stack.extend(node.args)
        elif isinstance(node, CustomCall):
            stack.extend(node.args)
        elif isinstance(node, MatMulExpression):
            stack.extend((node.left, node.right))
        elif isinstance(node, SumExpression):
            stack.append(node.operand)
        elif isinstance(node, SumOverExpression):
            stack.extend(node.terms)
        # Constant (and anything without children) contributes no owning leaf.


def _logical_backing_vars(expr) -> list:
    """Collect the backing ``Variable`` objects under a ``LogicalExpression`` tree.

    ``BooleanVar`` wraps a binary ``Variable`` (``.variable``); the composite
    logical nodes (``LogicalAnd``/``LogicalOr``/…) hold child logical
    expressions. Returns the flat list of backing variables so the ownership
    guard can check them too.
    """
    out: list = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, BooleanVar):
            if isinstance(node.variable, Variable):
                out.append(node.variable)
            else:
                # Indexed BooleanVar wraps an IndexExpression over a Variable.
                out.extend(_iter_model_leaves(node.variable))
        elif isinstance(node, (LogicalAnd, LogicalOr, LogicalEquivalent)):
            stack.extend((node.left, node.right))
        elif isinstance(node, LogicalNot):
            stack.append(node.operand)
        elif isinstance(node, LogicalImplies):
            stack.extend((node.antecedent, node.consequent))
        elif isinstance(node, (LogicalAtLeast, LogicalAtMost, LogicalExactly)):
            stack.extend(node.operands)
    return out


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


def _to_expr_object_array(a) -> np.ndarray:
    """Coerce *a* into a numpy object ndarray of scalar expressions (issue #816).

    Accepts a shaped :class:`Expression` (its scalar elements are pulled out by
    indexing along every axis), an existing ndarray, or a (possibly nested)
    list/tuple of expressions/numbers. Used by :func:`concatenate` / :func:`stack`
    to assemble a vector from pieces without adding a new DAG node type — each
    element remains an ordinary scalar expression the solver already handles.
    """
    if isinstance(a, np.ndarray):
        return a if a.dtype == object else a.astype(object)
    if isinstance(a, Expression):
        s = _known_shape(a)
        if s is None:
            raise TypeError(
                "concatenate()/stack() need expressions with a statically known "
                f"shape; got a {type(a).__name__} whose shape is not known. Index "
                "the pieces explicitly, or assemble the array from scalars."
            )
        out = np.empty(s, dtype=object)
        if s == ():
            out[()] = a
            return out
        for idx in np.ndindex(s):
            out[idx] = a[idx]
        return out
    if isinstance(a, (list, tuple)):
        subs = [_to_expr_object_array(x) for x in a]
        if subs and all(sub.ndim == 0 for sub in subs):
            out = np.empty(len(subs), dtype=object)
            for i, sub in enumerate(subs):
                out[i] = sub[()]
            return out
        return np.stack(subs, axis=0) if subs else np.empty(0, dtype=object)
    # Plain scalar / number.
    out = np.empty((), dtype=object)
    out[()] = a
    return out


def concatenate(arrays: Sequence, axis: int = 0) -> np.ndarray:
    """Join a sequence of array-valued expressions along an existing axis.

    Mirrors :func:`numpy.concatenate`. Each entry of *arrays* may be a shaped
    expression (e.g. ``x[:-1]``), an object ndarray of scalar expressions, or a
    (nested) list/tuple of scalar expressions/numbers. The result is a numpy
    object ndarray whose entries are scalar expressions, so it supports indexing,
    iteration, ``.shape`` and element-wise arithmetic — exactly what the
    method-of-lines "boundary value, interior values, boundary value" assembly
    pattern needs (issue #816).

    Note: to concatenate a scalar boundary value, wrap it in a length-1 sequence
    (``[p_left]``) so it has a leading axis to join along, as numpy requires.
    """
    parts = [_to_expr_object_array(a) for a in arrays]
    return np.concatenate(parts, axis=axis)


def stack(arrays: Sequence, axis: int = 0) -> np.ndarray:
    """Stack a sequence of array-valued expressions along a new axis.

    Mirrors :func:`numpy.stack`; see :func:`concatenate` for the accepted element
    forms and the object-ndarray result contract (issue #816).
    """
    parts = [_to_expr_object_array(a) for a in arrays]
    return np.stack(parts, axis=axis)


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
    """Relational sense of a constraint (``<=``, ``>=``, or ``==``)."""

    LE = "<="
    GE = ">="
    EQ = "=="


@dataclass(eq=False)
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

    def __bool__(self):
        # A Constraint has no boolean value. Because ``Expression.__eq__`` builds
        # a Constraint (the modeling DSL), ``if x == 5:``, ``u in [v]`` (which
        # truth-tests ``u == v``), and ``assert expr == other`` would otherwise
        # silently evaluate this object as truthy — a footgun that masks logic
        # errors (M4). Refuse loudly instead (the cvxpy approach). Use
        # ``m.subject_to(...)`` to add the constraint to the model, or compare the
        # constraint's ``.body``/``.sense``/``.rhs`` fields to test structure.
        raise TypeError(
            "A Constraint has no truth value. '==', '<=', '>=' on modeling "
            "expressions build a Constraint (they do not compare values), so "
            "using one in a boolean context (if/while/assert, 'in', bool()) is "
            "almost always a mistake. Add it with m.subject_to(...); to test an "
            "expression's identity use 'is' or a set/dict keyed by the object."
        )


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
    """Optimization direction of the objective (minimize or maximize)."""

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
    root_bound : float or None
        Strongest rigorous dual bound proved at the root box, in the reported
        objective sense. None if no root relaxation was built.
    root_gap : float or None
        Relative gap of ``root_bound`` against the final incumbent
        (``|objective - root_bound| / max(1, |objective|)``). None when either
        the root bound or the incumbent is unavailable.
    root_time : float or None
        Wall-clock seconds elapsed when the root node was fathomed/branched.
    solver_stats : dict of str to float, or None
        Instrumentation counters, or None. Two key families: per-family
        reduction/separation timers (``reduce/<fam>`` / ``separate/<fam>``,
        cumulative seconds) and per-source cut counts (``cuts/<source>``, added by
        the P3.1b cut-measurement work and read by the ``p3_1*`` scripts) and
        cut-pool gating telemetry (``pool/gate_*``, categorical decision codes). Only
        the timer families are seconds and bounded by ``wall_time``; ``cuts/`` are
        counts and ``pool/`` are decision codes.
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

    # Root-node certification instrumentation (Phase 0 / cert:T0.1). These
    # describe the state of the search at the moment the root node is fathomed
    # or branched — the point that governs whether the solver certifies at the
    # root (like BARON) or opens a tree. ``root_bound`` is the strongest
    # rigorous dual bound proved at the root box, in the *reported* objective
    # sense (already negated for MAXIMIZE, mirroring ``bound``). ``root_gap`` is
    # the relative gap ``|objective - root_bound| / max(1, |objective|)`` of
    # that bound against the final incumbent, using the same floored abs/rel
    # convention as ``gap``. ``root_time`` is the wall-clock seconds elapsed
    # from the whole-solve clock to that moment. All three are ``None`` when
    # the path never builds a root relaxation (e.g. an early infeasibility exit).
    root_bound: Optional[float] = None
    root_gap: Optional[float] = None
    root_time: Optional[float] = None

    # Per-family reduction/separation timers (cert:T0.3). A flat dict of
    # phase-name -> cumulative seconds across the solve, e.g.
    # ``{"reduce/fbbt": .., "reduce/obbt": .., "separate/psd": .., ...}``. Pure
    # instrumentation (never affects solver math); None when nothing was timed.
    solver_stats: Optional[dict[str, float]] = None

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

    # Structured MIP-NLP decomposition trace. Populated by solver="mip-nlp"
    # paths when iteration/provenance data is available.
    mip_nlp_trace: Optional[dict[str, object]] = None

    # Examiner-style validation report (populated if validate=True).
    validation_report: Optional[object] = None

    # Set True when the final incumbent-verification guard (Model.solve, default
    # on) found the returned incumbent INFEASIBLE in the ORIGINAL problem — a
    # false primal, which can only arise from an unsound presolve mutation or a
    # heuristic bug. When True, the incumbent (``x``/``objective``) has been
    # withheld and ``gap_certified`` forced False (a false primal is never
    # reported as a valid or certified solution). This should never be True on
    # correct solver code; it exists so such a bug cannot silently escape as a
    # false result (regression guard for the #770/#772 class).
    incumbent_verification_failed: bool = False

    # LLM explanation (populated if llm=True)
    _explanation: Optional[str] = None
    _model: Optional["Model"] = None

    # Sensitivity cache (populated lazily by .gradient())
    _sensitivity: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # A2 (correctness/API): the failure/no-relaxation sentinel must never
        # escape through the public bound/gap surface. Internally the solver
        # stores ``INFEASIBILITY_SENTINEL`` (``1e30``) as the "lower bound" of a
        # node whose relaxation failed or was declared infeasible, and on the
        # no-relaxation class (a model whose relaxation omits rows, so no dual
        # bound is ever produced) the tree's ``global_lower_bound`` stays at that
        # sentinel. It is *finite* (``np.isfinite(1e30)`` is True), so the
        # existing non-finite guard below does not catch it, and a raw
        # result-assembly path would surface ``bound=1e30`` — and a ``gap``
        # computed from it — as if it were a real dual bound. Map any
        # sentinel-magnitude bound (either sense, so ``-1e30`` from a MAXIMIZE
        # negation too) to ``None`` at this single chokepoint that every
        # ``SolveResult`` construction passes through, so no public consumer
        # (``.bound``/``.gap``, the callback, commentary, the dashboard, the
        # benchmark JSON) ever does arithmetic on it. This is a
        # representation/reporting normalisation only — the internal sentinel is
        # unchanged. ``root_bound``/``root_gap`` get the same treatment.
        if self.bound is not None and abs(self.bound) >= _SENTINEL_THRESHOLD:
            self.bound = None
            self.gap = None
        if self.root_bound is not None and abs(self.root_bound) >= _SENTINEL_THRESHOLD:
            self.root_bound = None
            self.root_gap = None

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


# --- Polynomial-degree classifier for the incumbent-verification guard (#772) ---
# Pure-Python and JAX-free: distinguishes the LP/MILP/QP/MIQP "fast family" (linear
# constraints + linear-or-quadratic objective, solved via the JAX-free Rust/POUNCE
# paths) from genuinely nonlinear models. The guard's snapshot builds a JAX evaluator,
# so it must be skipped for the fast family to preserve the JAX-free cold start of
# LP/MILP/QP/MIQP solves (test_lazy_jax_linear_path). Errs toward "nonlinear" on any
# unknown node — the safe direction: the guard runs (extra coverage), never a wrong skip.
_INF_DEGREE = float("inf")


def _degree_child_nodes(e) -> tuple:
    """The sub-expressions whose polynomial degree ``e``'s degree depends on.

    Deliberately narrow: a ``**`` reads its RHS as a *constant exponent* (not a
    degree), and non-``neg`` unary / opaque nodes short-circuit to ``inf`` without
    inspecting children, so those children are not returned.
    """
    if isinstance(e, IndexExpression):
        return (e.base,)
    if isinstance(e, SumExpression):
        return (e.operand,)
    if isinstance(e, SumOverExpression):
        return tuple(e.terms)
    if isinstance(e, UnaryOp):
        return (e.operand,) if e.op == "neg" else ()
    if isinstance(e, MatMulExpression):
        return (e.left, e.right)
    if isinstance(e, BinaryOp):
        return (e.left,) if e.op == "**" else (e.left, e.right)
    return ()  # leaves (int/float/Constant/Variable) and opaque (FunctionCall/...)


def _degree_combine(e, child_degs: list[float]) -> float:
    """Degree of ``e`` from its children's degrees (aligned with
    :func:`_degree_child_nodes`). Mirrors the original per-node rules exactly."""
    if isinstance(e, (int, float, Constant)):
        return 0
    if isinstance(e, Variable):
        return 1
    if isinstance(e, IndexExpression):
        return child_degs[0]
    if isinstance(e, SumExpression):
        return child_degs[0]
    if isinstance(e, SumOverExpression):
        return max(child_degs, default=0)
    if isinstance(e, UnaryOp):
        return child_degs[0] if e.op == "neg" else _INF_DEGREE
    if isinstance(e, (FunctionCall, CustomCall)):
        return _INF_DEGREE
    if isinstance(e, MatMulExpression):
        return child_degs[0] + child_degs[1]
    if isinstance(e, BinaryOp):
        if e.op in ("+", "-"):
            return max(child_degs[0], child_degs[1])
        if e.op == "*":
            return child_degs[0] + child_degs[1]
        if e.op == "/":
            return child_degs[0] if child_degs[1] == 0 else _INF_DEGREE
        if e.op == "**":
            left = child_degs[0]
            k = (
                e.right.value
                if isinstance(e.right, Constant)
                else (e.right if isinstance(e.right, (int, float)) else None)
            )
            try:
                return (
                    left * int(k)
                    if (k is not None and float(k).is_integer() and k >= 0)
                    else _INF_DEGREE
                )
            except (TypeError, ValueError):
                return _INF_DEGREE
        return _INF_DEGREE
    return _INF_DEGREE  # unknown node -> treat as nonlinear (guard runs; safe direction)


def _expression_degree(root) -> float:
    """Polynomial degree of ``root`` in the decision variables; ``inf`` for anything
    that is not a polynomial (transcendental, variable power, variable denominator,
    opaque).

    Iterative post-order over the expression DAG with memoization by node identity,
    so a deeply nested expression (e.g. a long left-associated sum) never overflows
    the Python recursion stack, and shared sub-expressions are walked once (#810).
    """
    memo: dict[int, float] = {}
    stack = [root]
    while stack:
        e = stack[-1]
        if id(e) in memo:
            stack.pop()
            continue
        kids = _degree_child_nodes(e)
        pending = [c for c in kids if id(c) not in memo]
        if pending:
            stack.extend(pending)  # compute children first (no recursion)
            continue
        memo[id(e)] = _degree_combine(e, [memo[id(c)] for c in kids])
        stack.pop()
    return memo[id(root)]


def _is_fast_linear_quadratic_family(model) -> bool:
    """True for LP/MILP/QP/MIQP: linear constraints and a linear-or-quadratic objective
    (the JAX-free solve families). Used to skip the JAX-importing incumbent-verification
    snapshot on those paths so their cold start stays JAX-free."""
    obj = model._objective.expression if model._objective is not None else None
    if obj is not None and _expression_degree(obj) > 2:
        return False
    for c in model._constraints:
        body = getattr(c, "body", None)
        if body is not None and _expression_degree(body) > 1:
            return False
    return True


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
        # Cached exclusive prefix-sum of variable sizes (flat start offsets); see
        # ``_flat_var_offset``. ``None`` until first requested / after growth.
        self._flat_var_offsets_cache: Optional[list[int]] = None
        self._parameters: list[Parameter] = []
        # Persistent set of declared variable/parameter names for O(1)
        # uniqueness checks (M7). Rebuilding ``{v.name ...} | {p.name ...}`` on
        # every declaration made model construction O(n²); this set is updated
        # incrementally at each registration site instead. Must stay in sync
        # with ``_variables``/``_parameters`` — every append to those lists that
        # introduces a name also adds it here.
        self._names: set[str] = set()
        self._constraints: list[Constraint] = []
        # #840: single-variable-affine constraint families emitted through the fast
        # path (``constraint(fast=True)`` -> ``_try_fast_linear_family``) are NOT
        # appended to ``_constraints`` because they are already emitted as sparse rows
        # into the Rust builder (adding them there too would double-count them in the
        # native solve). But the NLPEvaluator, ``_check_constraint_feasibility``, and
        # the #772 incumbent-verification guard build their view of the model from the
        # Constraint objects, so without this store they are BLIND to fast constraints
        # (evaluate_constraints -> [], the guard accepts any point — the #840
        # false-primal hole). We keep the generated Constraint objects here so those
        # consumers see the COMPLETE constraint set, while the native solve keeps
        # reading only the builder rows. Never fed to the builder/native path.
        self._fast_constraints: list[Constraint] = []
        self._objective: Optional[Objective] = None
        # R4: names of lifted product-factor auxes whose interval spans 0, set by
        # the factorable reform under ``DISCOPT_LIFT_ZERO_SPANNING_FACTORS`` so the
        # solver keeps them as spatial-branching candidates (see
        # ``factorable_reform._lift_zero_spanning_factors_enabled``). Empty by
        # default, so it never changes behaviour with the flag off.
        self._zero_spanning_factor_auxes: set[str] = set()
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

    # ── Rich representation (LaTeX / HTML in standard PSE form) ──

    def to_latex(self, max_rows: Optional[int] = None) -> str:
        """Render the model as a LaTeX ``aligned`` block in standard problem form
        (``minimize f(x)`` / ``subject to g(x) <= 0`` / variable domains).

        Parameters
        ----------
        max_rows : int, optional
            Cap on the number of constraint/variable rows shown; excess is
            summarised with a ``\\vdots`` row. ``None`` (default) renders everything.
        """
        from discopt.modeling import latex

        return latex.model_to_latex(self, max_rows=max_rows)

    def to_html(self, max_rows: Optional[int] = None) -> str:
        """Render the model as standalone HTML (the PSE LaTeX typeset via MathJax,
        with a header naming the model and its size). See :meth:`to_latex`."""
        from discopt.modeling import latex

        return latex.model_to_html(self, max_rows=max_rows)

    def _repr_latex_(self) -> str:
        from discopt.modeling import latex

        return f"$$\n{latex.model_to_latex(self, max_rows=latex._DEFAULT_MAX_ROWS)}\n$$"

    def _repr_markdown_(self) -> str:
        # VS Code's notebook renderer typesets ``text/markdown`` outputs with
        # KaTeX but does not reliably render ``text/latex`` outputs; emit the
        # same PSE block as fenced display math so it renders there too.
        # Jupyter/MathJax frontends prefer ``_repr_latex_``/``_repr_html_``.
        from discopt.modeling import latex

        return f"$$\n{latex.model_to_latex(self, max_rows=latex._DEFAULT_MAX_ROWS)}\n$$"

    def _repr_html_(self) -> str:
        from discopt.modeling import latex

        return latex.model_to_html(self, max_rows=latex._DEFAULT_MAX_ROWS)

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

    def _rebuild_name_index(self) -> None:
        """Recompute the persistent ``_names`` cache from the current lists (M7).

        Call after any bulk reassignment of ``_variables``/``_parameters`` that
        bypasses the registration sites (e.g. the reformulation passes that build
        a new ``Model`` with ``new_model._variables = list(model._variables)``),
        so a later ``_check_name`` on that model stays correct.
        """
        self._names = {v.name for v in self._variables} | {p.name for p in self._parameters}

    def _flat_var_offset(self, var: "Variable") -> int:
        """Return the flat start index of ``var`` in the stacked x vector.

        The offset is the sum of the sizes of every variable preceding ``var``
        in ``self._variables``. Summing that slice from scratch is O(n) per call,
        and the relaxation / AD / term-classifier builds resolve a flat index
        once per variable leaf per term — O(n·terms) overall. That quadratic was
        the dominant *uninterruptible* pre-B&B root-setup cost that made
        ``solve(time_limit=T)`` overrun its budget on large factorable models
        (issue #654; sub-sites #507 term-classifier build, #187 DAG compile).

        An exclusive prefix-sum table is memoized and rebuilt only when the
        (append-only) variable list grows, so each lookup is O(1). This is a pure
        speedup: ``_variables`` only ever grows and a Variable's ``_index`` /
        ``size`` are immutable after construction, so a cached offset can never
        go stale without the length changing. Indices at/past the end collapse to
        the full total, exactly as the ``[: var._index]`` slice did.
        """
        n = len(self._variables)
        # ``getattr`` fallback guards the rare path that builds a Model without
        # ``__init__`` (e.g. a shallow ``copy.copy`` of a pre-attribute object).
        offsets: Optional[list[int]] = getattr(self, "_flat_var_offsets_cache", None)
        if offsets is None or len(offsets) != n + 1:
            offsets = [0] * (n + 1)
            acc = 0
            for k, v in enumerate(self._variables):
                acc += v.size
                offsets[k + 1] = acc
            self._flat_var_offsets_cache = offsets
        idx = var._index
        return offsets[idx] if idx < n else offsets[n]

    def _register_variable(self, var: "Variable") -> "Variable":
        """Append a variable and register it with the Rust builder if active."""
        self._variables.append(var)
        self._names.add(var.name)  # keep the O(1) name set in sync (M7)
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
        lb: Optional[Union[float, np.ndarray]] = ...,
        ub: Optional[Union[float, np.ndarray]] = ...,
        over: None = ...,
    ) -> Variable: ...

    @overload
    def integer(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = ...,
        lb: Optional[Union[float, np.ndarray]] = ...,
        ub: Optional[Union[float, np.ndarray]] = ...,
        *,
        over: "_SetBase",
    ) -> "IndexedVar": ...

    def integer(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
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
        lb : float or numpy.ndarray, optional
            Lower bound. Honored **exactly** when supplied. When omitted, the
            finite fallback ``0`` is used and a ``UserWarning`` is emitted (see
            below).
        ub : float or numpy.ndarray, optional
            Upper bound. Honored **exactly** when supplied. When omitted, the
            finite fallback ``1e6`` is used and a ``UserWarning`` is emitted.

        Returns
        -------
        Variable
            Integer-valued variable.

        Notes
        -----
        Branch-and-bound needs a bounded integer domain, so an *unspecified*
        ``lb``/``ub`` falls back to ``[0, 1e6]``. Previously this substitution
        was silent, so a model whose true optimum needed a negative integer or
        one above ``1e6`` was quietly truncated and the wrong optimum reported
        as certified (correctness issue C-6). The fallback is now applied
        **loudly** — a ``UserWarning`` names the variable and the imposed
        range — and it **never** overrides a bound you provide. Pass explicit
        ``lb``/``ub`` (e.g. ``lb=-5, ub=10``) to silence the warning and to
        solve the problem you actually mean.

        Examples
        --------
        >>> n = m.integer("n_units", lb=0, ub=10)
        >>> batch = m.integer("batch", shape=(3,), lb=1, ub=100)
        >>> n = m.integer("n", over=plants, lb=0, ub=10)  # indexed integer
        """
        lb, ub = self._resolve_integer_defaults(name, lb, ub)
        if over is not None:
            _require_no_shape(shape, "integer")
            return self._make_indexed_var(
                name, VarType.INTEGER, over, lb, ub, _INTEGER_DEFAULT_LB, _INTEGER_DEFAULT_UB
            )
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.INTEGER, shape, lb, ub, self)
        return self._register_variable(var)

    @staticmethod
    def _resolve_integer_defaults(name, lb, ub):
        """Substitute finite fallback integer bounds for unspecified sides, loudly.

        Returns the (possibly-substituted) ``(lb, ub)``. A ``None`` side is
        replaced with the finite fallback (``0`` / ``1e6``) and recorded so a
        single ``UserWarning`` can name exactly which side(s) discopt had to
        default. A user-provided bound is passed through unchanged and never
        warns (correctness issue C-6).
        """
        defaulted: list[str] = []
        if lb is None:
            lb = _INTEGER_DEFAULT_LB
            defaulted.append(f"lb={_INTEGER_DEFAULT_LB:g}")
        if ub is None:
            ub = _INTEGER_DEFAULT_UB
            defaulted.append(f"ub={_INTEGER_DEFAULT_UB:g}")
        if defaulted:
            warnings.warn(
                f"integer variable '{name}': no explicit "
                f"{' and '.join(defaulted)} supplied; discopt is imposing the "
                f"finite default integer bound(s) so branch-and-bound has a "
                f"bounded domain. This default box silently cuts any optimum "
                f"below {_INTEGER_DEFAULT_LB:g} or above {_INTEGER_DEFAULT_UB:g}. "
                f"Pass explicit lb/ub to solve the intended problem and silence "
                f"this warning.",
                UserWarning,
                stacklevel=3,
            )
        return lb, ub

    def _make_indexed_param(self, name, index_set, value) -> "IndexedParam":
        """Build an :class:`IndexedParam` backed by one flat parameter over *index_set*."""
        from discopt.modeling.indexed import IndexedParam, resolve_indexed_values

        arr = resolve_indexed_values(index_set, value, 0.0, np.float64)
        self._check_name(name)
        flat = Parameter(name, arr, self)
        self._parameters.append(flat)
        self._names.add(name)  # keep the O(1) name set in sync (M7)
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
        self._names.add(name)  # keep the O(1) name set in sync (M7)
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

    def implicit(
        self,
        residual: Callable,
        u_inputs: Sequence,
        n_unknowns: int,
        x0=None,
        *,
        tol: float = 1e-10,
        max_iter: int = 50,
        name: str = "implicit",
    ) -> Expression:
        """Define a vector ``v`` implicitly by ``residual(u, v) = 0`` (issue #379).

        ``v`` (length ``n_unknowns``) is compiled to a differentiable JAX inner
        solve (Newton forward, implicit-function-theorem derivatives), returned
        as an expression node you index (``v[i]``) and use like any other. It
        rides on :func:`custom`, so a model containing it is solved on the
        **local NLP path only** (no global certificate) and integers are
        rejected. ``u_inputs`` are the model expressions the block depends on
        (the DAG edges into the solve); they are required, not inferred.

        See :func:`discopt.modeling.implicit` for parameter details.

        Examples
        --------
        >>> u = m.continuous("u", lb=0.1, ub=100.0)
        >>> v = m.implicit(lambda U, V: [V[0] ** 2 - U[0]], [u], n_unknowns=1)
        >>> m.minimize((v[0] - 3.0) ** 2)   # v = sqrt(u); optimum at u = 9
        """
        from discopt.modeling.implicit import implicit as _implicit

        return _implicit(residual, u_inputs, n_unknowns, x0, tol=tol, max_iter=max_iter, name=name)

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
            # M5: never let ``subject_to`` corrupt an earlier row or clobber a
            # name the caller set. The corruption case is *re-adding the same
            # object* (or one already carrying a different name): stamping
            # ``constraint.name`` in place would rename the earlier model row and
            # overwrite the caller's own name. In those cases store an
            # independent copy so each row is self-contained. For the common case
            # — a fresh, unnamed constraint added once — stamp the name in place
            # so the object the caller holds *is* the object in the model
            # (identity that ``mark_coupling`` and other object-keyed APIs rely
            # on). ``_dc_replace`` is only paid when it is actually needed.
            already_added = getattr(constraint, "_added_to", None) is not None
            has_own_name = constraint.name is not None and constraint.name != name
            if already_added or has_own_name:
                # Re-add of an already-placed object, or one the caller already
                # named: copy so we never rename the earlier row nor clobber the
                # caller's name. O(1) — no scan of ``_constraints``.
                self._constraints.append(_dc_replace(constraint, name=name))
            else:
                # First placement of a fresh, unnamed constraint: stamp in place
                # (preserving object identity for object-keyed APIs) and record
                # that it now belongs to a model so a later re-add copies instead.
                constraint.name = name
                constraint._added_to = id(self)
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
            # Copy-on-name here too (M5): never mutate the caller's list elements.
            self._constraints.append(_dc_replace(c, name=f"{name}_{k}" if name else None))

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
            # Rows were emitted into the Rust builder; keep the Constraint objects in
            # ``_fast_constraints`` (NOT ``_constraints`` — that would double-count them
            # in the native solve) so the NLPEvaluator / feasibility check / #772 guard
            # see the complete constraint set (#840). Introspection view still returned.
            self._fast_constraints.extend(c for _, c in generated)
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

    def analyze_decomposition(self):
        """Analyze this model's decomposition structure (does not solve or modify).

        Returns a
        :class:`~discopt.decomposition.advisor.DecompositionAdvisor` exposing the
        structure report, discovered decomposition candidates, block partition,
        and graph views/exports. See ``docs/design/decomposition-advisor.md``.
        """
        from discopt.decomposition import analyze_decomposition

        return analyze_decomposition(self)

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

        Vector/array operands are complementarity **elementwise**: for every
        index ``i`` independently, ``x_i · y_i == 0`` with ``x_i, y_i >= 0`` (a
        separate disjunction per index — *not* a single "all ``x`` zero or all
        ``y`` zero" choice). A scalar side broadcasts against a vector side;
        incompatible shapes are rejected.

        Parameters
        ----------
        x, y : Expression
            The complementary pair. Both are constrained non-negative. May be
            scalars or equally-shaped (or scalar-broadcastable) arrays.
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
        initial_solution: Optional[dict] = None,
        skip_convex_check: bool = False,
        nlp_bb: Optional[bool] = None,
        lazy_constraints: Optional[Callable] = None,
        incumbent_callback: Optional[Callable] = None,
        node_callback: Optional[Callable] = None,
        solver: Optional[str] = None,
        validate: bool = False,
        verify_incumbent: bool = True,
        gauss_newton: bool = False,
        tuning: Optional["SolverTuning"] = None,
        debug: Any = None,
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
            Use ``solver="gurobi"`` to select the optional Gurobi backend for
            LP, MILP, QP, MIQP, QCP, and MIQCP models. Quadratic objectives
            and quadratic constraints use the ``0.5 * x.T @ Q @ x + c.T @ x``
            convention; nonconvex quadratic objectives or constraints require
            opting into Gurobi's nonconvex mode explicitly,
            for example ``gurobi_options={"NonConvex": 2}``. Unsupported model
            classes raise ``NotImplementedError`` instead of silently falling
            back to another backend. Pass Gurobi parameters through
            ``gurobi_options={...}``.
            Use ``solver="mip-nlp"`` to select the MIP-NLP decomposition
            family. Current implemented ``mip_nlp_method`` values are ``"oa"``,
            ``"ecp"``, ``"fp"``, ``"goa"``, and ``"lp_nlp_bb"``; ``"roa"``
            is reserved until its dedicated implementation lands. The
            LP/NLP-BB method uses single-tree lazy OA cuts and currently
            requires ``milp_solver="gurobi"``. Top-level OA/ECP options such
            as ``equality_relaxation``,
            ``ecp_mode``, ``feasibility_cuts``, ``heuristic_nonconvex``,
            ``add_slack``, ``max_slack``, ``oa_penalty_factor``,
            ``add_no_good_cuts``, ``integer_to_binary``, ``feasibility_norm``,
            ``add_regularization``, ``level_coef``, ``stalling_limit``,
            ``cycling_check``, ``solution_pool``, ``num_solution_iteration``,
            and ``init_strategy``
            take precedence over duplicate keys in ``mip_nlp_options``.
            ``solution_pool`` currently requires ``milp_solver="gurobi"``.
            Experimental SHOT-parity controls are accepted only with
            ``mip_nlp_profile="shot"`` and include ``tree_strategy``,
            ``cut_strategy``, ``objective_epigraph``, ``anti_epigraph``,
            ``nonlinear_partitioning``, ``quadratic_partitioning``,
            ``absolute_value_auxiliaries``, ``monomial_extraction``,
            ``signomial_extraction``, ``integer_bilinear_strategy``,
            ``integer_bilinear_max_bits``, ``quadratic_extraction``,
            ``direct_quadratic_routing``, ``rootsearch_strategy``,
            ``fixed_nlp_strategy``, ``solution_pool_capacity``,
            ``hyperplane_max_per_iter``, ``hyperplane_selection_factor``,
            ``relaxation_phase``, ``mip_solution_limit_strategy``,
            ``convex_bounding``, ``master_repair``, and ``reduction_cuts``.
            MIP-NLP runs attach a structured ``result.mip_nlp_trace`` payload.
            For ``mip_nlp_method="goa"``,
            convexity-certified MINLPs use OA's valid master bounds and other
            models use AMP/global relaxations. AMP options such as ``rel_gap``,
            ``abs_tol``, ``max_iter``, ``n_init_partitions``,
            ``partition_method``, ``milp_time_limit``, ``milp_gap_tolerance``,
            ``presolve_bt``, and ``convhull_formulation`` may also be passed as
            top-level aliases. AMP-only options apply only on the nonconvex AMP
            path and are ignored with a warning when GOA automatically hands a
            convexity-certified model to OA.
            Supported ``add_regularization`` values are
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
        debug : bool, str, or DebugSession, optional
            Attach the interactive branch-and-bound debugger (a "pdb for B&B").
            ``True`` / ``"repl"`` drops into a human REPL at the first
            checkpoint; ``"on-error"`` enters only at termination when the
            outcome is not a certified optimum (limit hit, interrupted,
            "unknown", or certified infeasible) — not supported on the
            pure-Rust MILP fast-path, whose hook carries no final status; a
            ready-made :class:`discopt.debug.DebugSession` uses a custom
            frontend. ``None`` (default) leaves the solve untouched —
            fire-sites short-circuit on a module-global check, so a detached
            solve is bound-neutral. Currently instruments the spatial-McCormick
            B&B path (nonconvex MINLP / continuous). See :mod:`discopt.debug`.
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

        # Reject misspelled / unknown solve() keyword arguments loudly (M6).
        # ``**kwargs`` is forwarded verbatim to ``solve_model`` and its backends,
        # which historically only *warned* (or silently swallowed) unknown keys —
        # so ``m.solve(gap_tolerence=1e-3)`` ran at the DEFAULT gap while the user
        # believed they had tightened it (a results-integrity hazard). Validate
        # against the union of solve_model's parameters and the curated
        # backend-passthrough set; an unknown key raises TypeError with a
        # near-match suggestion (CLAUDE.md §3: loud refusal over silent swallow).
        if kwargs:
            from discopt.solver import solve_model_accepted_kwargs

            allowed = solve_model_accepted_kwargs()
            unknown = [k for k in kwargs if k not in allowed]
            if unknown:
                import difflib

                bad = unknown[0]
                close = difflib.get_close_matches(bad, sorted(allowed), n=1, cutoff=0.6)
                hint = f" Did you mean '{close[0]}'?" if close else ""
                raise TypeError(
                    f"solve() got an unexpected keyword argument '{bad}'.{hint} "
                    f"Unknown solver options are rejected rather than silently "
                    f"ignored (a swallowed option would leave the solver at its "
                    f"default while you believe it was set)."
                )

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

        # Convex LP-OA branch-and-cut kernel fast path (#798, gated by
        # DISCOPT_CONVEX_KERNEL, default-OFF). ``try_convex_solve`` routes ONLY
        # provably-convex composite-of-affine MINLPs and returns a fully-certified
        # SolveResult whose incumbent is verified feasible against this pristine
        # model (#779); it returns None — falling through to the default path
        # below, untouched — for the flag-off case, any non-convex model, or an
        # unverifiable incumbent. Wrapped so a kernel error can never break solve.
        if not skip_convex_check:
            try:
                from discopt.solvers._convex_kernel import try_convex_solve

                _ck_res = try_convex_solve(self, time_limit=time_limit, gap_tolerance=gap_tolerance)
            except Exception:
                _ck_res = None
            if _ck_res is not None:
                return _ck_res

        from discopt._jax.deadline import deadline_scope
        from discopt.solver import solve_model

        # Attach the interactive B&B debugger if requested. The fire-sites in
        # the solve loop short-circuit on a module-global ``None`` check, so a
        # detached solve (``debug=None``) is unaffected. ``debug`` may be True,
        # "repl", "on-error", or a ready-made ``DebugSession``.
        _debug_guard = None
        if debug is not None and debug is not False:
            from discopt import debug as _dbg

            _debug_guard = _dbg.attach(_dbg.make_session(debug))

        # --- Final-incumbent verification snapshot (#772 guard, default on) ---
        # ``solve_model``'s presolve mutates this model's constraint DAG IN PLACE,
        # so any feasibility check taken AFTER the solve reflects the (possibly
        # unsoundly) mutated model — it cannot catch a false primal produced by an
        # unsound presolve (the #770 class, where the incumbent is feasible in the
        # mutated model but not the original). Capture a COMPILED feasibility
        # evaluator of the ORIGINAL model here, before any mutation: a compiled JAX
        # evaluator has its structure baked in, so it remains a faithful, immutable
        # representation of the original problem across the solve (verified: it
        # agrees with a freshly parsed model after a real presolve run). Built once
        # and cache-warm for the solve itself; fully wrapped so it can never break a
        # solve.
        import logging as _logging

        # Skip the (JAX-importing) snapshot for the LP/MILP/QP/MIQP fast family: those
        # paths are JAX-free by design and do not run the nonlinear presolve mutation
        # this guard defends against, so building a JAX evaluator there would regress
        # their JAX-free cold start (``_is_fast_linear_quadratic_family`` is pure-Python).
        #
        # #822: the snapshot is NOT gated on ``verify_incumbent``. Returning an
        # incumbent that is infeasible in the original model is the cardinal error
        # (CLAUDE.md §1); a user flag must not be able to disable that soundness
        # withhold. On a nonlinear solve the evaluator is already built during the
        # solve, so ``cached_evaluator(self)`` here is a cache hit (~0 ms) — the
        # verification is effectively free and now always runs. ``verify_incumbent``
        # is retained for API compatibility but no longer suppresses the withhold.
        _verify_snap = None
        if self._constraints and not _is_fast_linear_quadratic_family(self):
            try:
                from discopt._jax.nlp_evaluator import cached_evaluator

                _verify_snap = (
                    cached_evaluator(self),
                    [v.name for v in self._variables],
                )
            except Exception as _snap_exc:  # pragma: no cover - defensive
                _logging.getLogger("discopt.solver").debug(
                    "incumbent-verification snapshot skipped: %s", _snap_exc
                )

        # Install a process-global wall-clock deadline that JAX-compiled
        # while_loops (LP/QP/NLP IPM) can poll via host callback so they
        # self-terminate within ``time_limit + ε`` instead of running to
        # XLA convergence after Python's budget is gone (issue #80).
        try:
            with deadline_scope(time_limit):
                result = solve_model(
                    self,
                    time_limit=time_limit,
                    gap_tolerance=gap_tolerance,
                    threads=threads,
                    deterministic=deterministic,
                    partitions=partitions,
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
        finally:
            if _debug_guard is not None:
                _debug_guard.__exit__(None, None, None)

        # Attach model reference and auto-generate LLM explanation
        result._model = self

        # --- Verify the final incumbent against the ORIGINAL problem (#772) ---
        # A reported incumbent that is INFEASIBLE in the original model is a false
        # primal — the worst error class (CLAUDE.md §1). It cannot happen on correct
        # solver code, but if an unsound presolve mutation or a heuristic bug ever
        # produces one, this guard refuses to return it: withhold the incumbent and
        # decertify, loudly. The check is deliberately LOOSE (abs tol 1e-3) so it can
        # never flag an incumbent that is feasible within the solver's own tolerance
        # — only a gross violation (the #770 violations were 0.4–17.6) trips it.
        if _verify_snap is not None and result.x is not None:
            try:
                import numpy as _np

                from discopt._jax.primal_heuristics import _check_constraint_feasibility

                _snap_ev, _snap_names = _verify_snap
                _flat = _np.concatenate(
                    [
                        _np.atleast_1d(_np.asarray(result.x[_n], dtype=_np.float64)).ravel()
                        for _n in _snap_names
                    ]
                )
                if not _check_constraint_feasibility(_snap_ev, _flat, tol=1e-3):
                    _logging.getLogger("discopt.solver").error(
                        "FALSE PRIMAL DETECTED: the reported incumbent (objective=%s) is "
                        "INFEASIBLE in the original problem. This indicates an unsound presolve "
                        "mutation or a heuristic bug. Withholding the incumbent and decertifying "
                        "— the result is NOT a valid solution.",
                        result.objective,
                    )
                    result.incumbent_verification_failed = True
                    result.gap_certified = False
                    result.x = None
                    result.objective = None
                    result.gap = None
            except Exception as _ver_exc:  # pragma: no cover - defensive
                _logging.getLogger("discopt.solver").debug(
                    "incumbent verification skipped: %s", _ver_exc
                )

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

    def _iter_owned_leaves(self):
        """Yield every ``Variable``/``Parameter`` leaf referenced by this model.

        Walks the objective expression and every constraint body (including the
        indicator/disjunctive/SOS/logical constraint families) so the ownership
        guard (M3) sees the same leaves the solver will eventually lower to the
        flat Rust variable vector.
        """
        if self._objective is not None:
            yield from _iter_model_leaves(self._objective.expression)

        def _from_constraint(c):
            if isinstance(c, Constraint):
                yield from _iter_model_leaves(c.body)
            elif isinstance(c, _IndicatorConstraint):
                yield from _iter_model_leaves(c.indicator)
                yield from _from_constraint(c.constraint)
            elif isinstance(c, _DisjunctiveConstraint):
                for disjunct in c.disjuncts:
                    for sub in disjunct:
                        yield from _from_constraint(sub)
            elif isinstance(c, _SOSConstraint):
                yield from _iter_model_leaves(*c.variables)
            elif isinstance(c, _LogicalConstraint):
                yield from _iter_model_leaves(*_logical_backing_vars(c.expression))
            elif hasattr(c, "body"):
                # Unknown arithmetic-bodied constraint type: still walk its body.
                yield from _iter_model_leaves(c.body)

        for c in self._constraints:
            yield from _from_constraint(c)

    def _check_ownership(self):
        """Reject expressions that reference a variable/parameter of another model.

        A :class:`Variable` carries a flat ``_index`` into *its own* model's
        variable vector. When the solver lowers this model it addresses each
        leaf by that flat index, so the precise soundness invariant is:

            ``self._variables[leaf._index] is leaf``

        i.e. the leaf's flat index must address *that very object* in the model
        being solved. A leaf from a foreign model has an index that addresses a
        **different** object (or is out of range) → it would silently **alias by
        flat index** onto this model's same-index variable, a wrong answer rather
        than an error (review finding M3). Raise loudly instead (CLAUDE.md §3).

        This index-slot identity is deliberately *not* an ``owner is self`` test:
        legitimate model rebuilds (e.g. ``reformulate_gdp``) construct a new
        ``Model`` that **shares** the original ``Variable`` objects in the same
        order — index-compatible, so those must pass — while adding new aux
        variables of their own. The identity check accepts both and rejects only
        a genuinely foreign, index-incompatible leaf. One O(DAG) walk per solve.
        """
        # Object-identity membership sets for O(1) lookup. For variables, index
        # equality is the real invariant; the identity set is the fast primary
        # check, with the index-slot check as the precise diagnostic.
        var_ids = {id(v) for v in self._variables}
        param_ids = {id(p) for p in self._parameters}
        n_vars = len(self._variables)
        for leaf in self._iter_owned_leaves():
            if isinstance(leaf, Parameter):
                if id(leaf) in param_ids:
                    continue
            else:  # Variable
                if id(leaf) in var_ids:
                    continue
                idx = getattr(leaf, "_index", None)
                # An index that happens to address this very object is also fine
                # (shared-object rebuilds), but that is already covered by the id
                # set above; reaching here means the object is not in this model.
                if idx is not None and 0 <= idx < n_vars and self._variables[idx] is leaf:
                    continue
            # Foreign leaf: index-incompatible with the model being solved.
            owner = getattr(leaf, "model", None)
            kind = "parameter" if isinstance(leaf, Parameter) else "variable"
            own_name = getattr(owner, "name", None) or "<unknown>"
            self_name = getattr(self, "name", None) or "<unnamed>"
            raise ValueError(
                f"Cross-model reference: {kind} '{leaf.name}' belongs to a "
                f"different model ('{own_name}'), not to model '{self_name}' being "
                f"solved. Variables/parameters carry a flat index local to their "
                f"own model, so mixing them across models would silently alias by "
                f"index and produce a wrong answer. Rebuild the offending "
                f"expression using only variables/parameters declared on model "
                f"'{self_name}'."
            )

    def validate(self):
        """
        Validate model consistency.

        Raises
        ------
        ValueError
            If the objective is not set, variable names are not unique,
            variable bounds are inconsistent (lb > ub), or any objective/
            constraint references a variable/parameter owned by a *different*
            model (which would silently alias by flat index — finding M3).
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

        # Duplicate constraint names are a results-integrity hazard (M5):
        # ``SolveResult.constraint_duals`` is keyed by constraint name, so two
        # rows sharing a name collide there. WARN (do not reject): indexed/array
        # constraint families legitimately reuse a base name (e.g. a GAMS
        # ``supply(i)`` equation loads as several ``supply`` rows), so a hard
        # error would break valid models. The real fix — disambiguating
        # ``constraint_duals`` for same-named indexed rows — is tracked as a
        # follow-up (#413, M5). Unnamed constraints (``name is None``) are exempt.
        con_names: set = set()
        for c in self._constraints:
            cname = getattr(c, "name", None)
            if cname is None:
                continue
            if cname in con_names:
                warnings.warn(
                    f"Duplicate constraint name: '{cname}'. result.constraint_duals "
                    "is keyed by name, so rows sharing a name collide there "
                    "(expected for indexed/array constraint families). Give distinct "
                    "names if you need per-row duals. (M5, #413)",
                    stacklevel=2,
                )
            con_names.add(cname)

        self._check_ownership()

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

    def _has_builder_only_rows(self) -> bool:
        """True if the model carries linear rows/objective living only in the builder.

        The fast-construction API (``add_linear_constraints`` / the
        ``Model.constraint`` linear fast path) and ``add_linear_objective`` /
        ``add_quadratic_objective`` emit rows/objective directly into the Rust
        builder; they are **not** mirrored in ``self._constraints`` /
        ``self._objective``. Consumers that read only those Python-side collections
        (classifiers, extractors, exporters, the validation examiner) must consult
        this predicate — a ``True`` return means they would otherwise see a strict
        subset of the model. See ``docs/dev/review-execution-plan.md`` §1 (X-1).
        """
        from discopt.export._common import has_builder_only_rows

        return has_builder_only_rows(self)

    def _num_builder_constraint_rows(self) -> int:
        """Count the scalar constraint rows that live only in the Rust builder."""
        blocks = getattr(self, "_builder_linear_blocks", None)
        if not blocks:
            return 0
        return int(builtins_sum(int(A.shape[0]) for A, *_ in blocks))

    def _materialize_builder_linear_rows(self) -> int:
        """Rewrite builder-resident linear constraint rows as expression constraints.

        The fast-construction API (``add_linear_constraints`` / the
        ``Model.constraint`` linear fast path) records constraint rows only in
        ``self._builder_linear_blocks`` / the Rust builder — they are **not**
        mirrored into ``self._constraints``. The JAX spatial-B&B consumers (the
        ``NLPEvaluator`` feasibility gate and the McCormick relaxer) read only
        ``self._constraints``, so on the nonlinear solve path those rows are
        silently dropped, and the solver can certify a **false optimum** on an
        infeasible incumbent (issue #681).

        This materialises each builder linear row into an equivalent expression
        :class:`Constraint` in ``self._constraints`` and resets the Rust builder so
        the rows are not *also* carried there (which would double-count them in
        ``model_to_repr``). The variable registration and any builder-resident
        objective (``add_linear_objective`` / ``add_quadratic_objective``) are
        preserved by rebuilding the builder and re-applying the objective from its
        retained Python-side block. The result is a single, consistent
        representation every consumer sees — the fast path's documented "the
        resulting model is identical" invariant, now honoured on the nonlinear
        path too.

        Idempotent (a no-op once the blocks are cleared). Returns the number of
        rows materialised. Mathematically model-preserving: it relocates rows
        between two internal representations without changing the feasible set,
        the objective, or ``num_constraints``.
        """
        from discopt.export._common import iter_builder_linear_rows

        blocks = getattr(self, "_builder_linear_blocks", None)
        if not blocks:
            return 0

        rows = iter_builder_linear_rows(self)
        new_constraints: list[Constraint] = []
        for row in rows:
            body: Optional[Expression] = None
            for v, local, coeff in row.terms:
                if v.shape == ():
                    comp: Expression = v
                elif len(v.shape) <= 1:
                    comp = v[local]
                else:
                    comp = v[tuple(int(i) for i in np.unravel_index(local, v.shape))]
                term = coeff * comp
                body = term if body is None else body + term
            if body is None:
                body = Constant(np.float64(0.0))
            if row.sense == "<=":
                c = body <= row.rhs
            elif row.sense == ">=":
                c = body >= row.rhs
            else:
                c = body == row.rhs
            c.name = row.name
            new_constraints.append(c)

        # Reset the Rust builder so it no longer carries the constraint rows
        # (avoids a double-count in ``model_to_repr``), preserving the variable
        # registration and re-applying any builder-resident objective from its
        # retained Python block.
        lin_obj = getattr(self, "_builder_linear_objective", None)
        quad_obj = getattr(self, "_builder_quadratic_objective", None)
        self._builder = None
        self._builder_linear_blocks = []
        # Re-register variables in a fresh builder so ``_builder_idx`` stays valid.
        self._get_builder()
        if lin_obj is not None:
            c_vec, x, constant, sense = lin_obj
            self.add_linear_objective(c_vec, x, constant=constant, sense=sense)
        elif quad_obj is not None:
            Q, c_vec, x, constant, sense = quad_obj
            self.add_quadratic_objective(Q, c_vec, x, constant=constant, sense=sense)

        self._constraints.extend(new_constraints)
        return len(new_constraints)

    @property
    def num_constraints(self) -> int:
        """Total scalar constraint rows, including fast-API / builder-resident rows.

        Counts both expression-path rows in ``self._constraints`` and the
        builder-resident linear rows emitted by ``add_linear_constraints`` / the
        ``Model.constraint`` linear fast path (X-1). Before this fix the fast-path
        rows were invisible, so a model built entirely through the fast API reported
        ``0`` constraints.
        """
        return len(self._constraints) + self._num_builder_constraint_rows()

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
        """Ensure variable/parameter name is unique.

        Consults the persistent ``self._names`` set (kept in sync at every
        registration site) for an O(1) check instead of rebuilding the full name
        set from ``_variables``/``_parameters`` on every declaration (M7 — that
        rebuild was O(n) per call, O(n²) over a model build).
        """
        if name in self._names:
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

    Created via :meth:`Model.make_disjunct`, not directly.

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
    Import a Pyomo ``ConcreteModel`` as a discopt :class:`Model`.

    The bridge round-trips the Pyomo model through a temporary AMPL ``.nl`` file
    (the same translation the ``SolverFactory('discopt')`` plugin uses) and reads
    it back with :func:`from_nl`. Continuous, integer and binary variables,
    linear/nonlinear constraints, and the objective are supported.

    Parameters
    ----------
    pyomo_model : pyomo.environ.ConcreteModel
        A fully constructed Pyomo model.

    Returns
    -------
    Model
        A discopt ``Model`` ready to ``.solve()``. Its variables/constraints are
        in the ``.nl`` column/row order, which may differ from the Pyomo model's
        declaration order.

    Raises
    ------
    ImportError
        If Pyomo is not installed (``pip install discopt[pyomo]``).
    ValueError
        If the Pyomo model has no variables to import.

    Examples
    --------
    >>> import pyomo.environ as pyo
    >>> m = pyo.ConcreteModel()
    >>> m.x = pyo.Var(bounds=(0, 10))
    >>> m.obj = pyo.Objective(expr=(m.x - 3) ** 2)
    >>> dmodel = dm.from_pyomo(m)  # doctest: +SKIP
    >>> dmodel.solve().objective  # doctest: +SKIP
    0.0
    """
    try:
        import pyomo.environ  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without pyomo
        raise ImportError(
            "from_pyomo requires Pyomo. Install it with 'pip install discopt[pyomo]'."
        ) from exc

    import os
    import tempfile

    from discopt.pyomo import _writer

    # Round-trip through a temporary .nl file. write_nl disables presolve/scaling so
    # the emitted columns are exactly the model's variables in the original space,
    # which is what from_nl reconstructs into a discopt Model.
    with tempfile.TemporaryDirectory(prefix="discopt_from_pyomo_") as workdir:
        nl_path = os.path.join(workdir, "model.nl")
        cols, _rows, _eliminated = _writer.write_nl(pyomo_model, nl_path)
        if not cols:
            raise ValueError(
                "from_pyomo: the Pyomo model has no variables to import (nothing to solve)."
            )
        return from_nl(nl_path)


# Depth at/above which a left-nested ``+`` chain is rebalanced into a shallow
# tree (#654). A flat N-term sum parsed from a ``.nl`` file is built as a depth-N
# left chain ``(((...(t1+t2)+t3)...)+tN)``; on a dense-quadratic model (qap: ~50k
# products) that is ~43000 deep, and every recursive tree-walk in the pipeline
# (expr->arena conversion, term classification, relaxation build, AD compile)
# recurses N deep and overflows the C stack — which manifests as a *hang*, not a
# clean error. The threshold is set well above any sum a working model carries but
# far below the ~15k-frame stack limit, so only pathologically deep chains (which
# otherwise hang) are touched; shallower sums are returned unchanged and every
# existing model stays byte-identical.
_SUM_REBALANCE_DEPTH = 2048


def _rebalance_deep_sum(expr: "Expression") -> "Expression":
    """Rebalance a deeply left-nested ``+`` chain into a shallow balanced tree.

    Only ``+`` chains are touched: addition is associative *and* commutative, so
    the rebalanced tree is exactly value-equal to the original (a balanced fold of
    the same terms). Chains shorter than :data:`_SUM_REBALANCE_DEPTH` are returned
    unchanged (identity), so the transform is a no-op — and byte-neutral — on every
    model that is not pathologically deep. The left-spine is walked *iteratively*
    (the chain is exactly the structure that would overflow a recursive walk)."""
    if not (isinstance(expr, BinaryOp) and expr.op == "+"):
        return expr
    # Measure the left-spine depth without recursing.
    depth = 0
    cur: Expression = expr
    while isinstance(cur, BinaryOp) and cur.op == "+":
        cur = cur.left
        depth += 1
    if depth < _SUM_REBALANCE_DEPTH:
        return expr
    # Deep chain: collect its terms (left-to-right) iteratively, then balanced-fold.
    terms: list[Expression] = []
    cur = expr
    while isinstance(cur, BinaryOp) and cur.op == "+":
        terms.append(cur.right)
        cur = cur.left
    terms.append(cur)  # deepest-left operand
    terms.reverse()
    while len(terms) > 1:
        terms = [
            terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return terms[0]


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
    from discopt._jax.nl_reconstruction import (
        reconstruct_complementarities,
        reconstruct_dag,
    )
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
            var = m.binary(name, shape=shape)
            # Preserve the parsed bounds (X-3 / M2 / INT-2): `Model.binary` hardcodes
            # ``[0, 1]``, so a bound-narrowed or fixed binary (e.g. ``lb == ub == 1``,
            # routine Pyomo/presolve output) would silently un-fix, giving the wrong
            # optimum on the round-trip. Clamp the parsed bounds into ``[0, 1]`` (a
            # binary column is 0/1 by definition) and stamp them onto the variable.
            var.lb = np.broadcast_to(np.clip(np.asarray(lb, dtype=np.float64), 0.0, 1.0), shape)
            var.ub = np.broadcast_to(np.clip(np.asarray(ub, dtype=np.float64), 0.0, 1.0), shape)
        elif vt == "integer":
            m.integer(name, shape=shape, lb=lb, ub=ub)

    # Reconstruct the expression DAG from the Rust arena
    objective_expr, constraint_tuples = reconstruct_dag(nl_repr, m._variables)

    # #654: rebalance pathologically deep ``+`` chains (flat sums parsed as depth-N
    # left chains) so downstream recursive tree-walks don't overflow the stack.
    # A no-op below _SUM_REBALANCE_DEPTH, so ordinary models are byte-identical.
    obj_expr: Expression = _rebalance_deep_sum(objective_expr)

    # Set the objective with the reconstructed expression
    if nl_repr.objective_sense == "minimize":
        m.minimize(obj_expr)
    else:
        m.maximize(obj_expr)

    # Add constraints from the reconstructed DAG
    for body, sense, rhs in constraint_tuples:
        body_expr: Expression = _rebalance_deep_sum(body)
        if sense == "<=":
            m.subject_to(body_expr <= rhs)
        elif sense == ">=":
            m.subject_to(body_expr >= rhs)
        elif sense == "==":
            m.subject_to(body_expr == rhs)

    # Complementarity (type-5) rows: lower each ``body ⊥ x`` relation through the
    # exact GDP disjunction machinery instead of adding it as an ordinary
    # constraint (#658). The complementarity rows are absent from
    # ``constraint_tuples`` above (consumed by the relation), so there is no
    # double-add.
    compl_pairs = reconstruct_complementarities(nl_repr, m._variables)
    for body, var_index, flag in compl_pairs:
        _add_nl_complementarity(m, body, var_index, flag)

    # Keep nl_repr for backward compatibility (Rust evaluator for validation)
    m._nl_repr = nl_repr
    # Record the source path so the solver can hand POUNCE the original .nl for
    # native-AD node NLP solves (discopt.solvers.nlp_native), bypassing the JAX
    # callback bridge. The .nl column order is the model's variable order here,
    # so the native problem aligns with the evaluator's flat x (identity map).
    #
    # Suppressed when the model carried complementarities: the GDP lowering above
    # added selector binaries / auxiliaries and disjunction constraints that the
    # in-memory model has but the original .nl does not, so POUNCE reading the raw
    # .nl would solve a structurally different (under-constrained) problem. The
    # ``to_nl`` fallback emits the reformulated model instead, staying consistent
    # with the JAX evaluator.
    if not compl_pairs:
        m._source_nl_path = os.path.abspath(path)

    return m


def _add_nl_complementarity(m: "Model", body: "Expression", var_index: int, flag: int) -> None:
    r"""Lower one ``.nl`` complementarity row into ``Model.complementarity``.

    The row asserts ``body ⊥ x`` where ``x = m._variables[var_index]`` and
    ``body`` carries the bounds encoded by ``flag`` (AMPL MP ``ComplInfo``:
    bit 0 ⇒ lower ``-inf``, bit 1 ⇒ upper ``+inf``; unset ⇒ 0). discopt's
    primitive is the symmetric nonnegative complementarity
    :math:`0 \le a \perp b \ge 0`, so we orient ``body`` and ``x`` into two
    nonnegative quantities before handing them over.

    Only the single-bounded MPEC forms map soundly onto that primitive: the body
    bounded on exactly one side (``body >= 0`` or ``body <= 0``) against a
    single-signed variable (``x >= 0`` or ``x <= 0``). Doubly-bounded / free /
    equality-pinned complementarity rows require a richer disjunction than the
    primitive expresses; rather than silently mis-model them we refuse loudly
    (correctness-first — the native complementarity node is the #231 follow-up).
    """
    inf_lb = bool(flag & 1)  # ComplInfo INF_LB: body lower bound is -inf
    inf_ub = bool(flag & 2)  # ComplInfo INF_UB: body upper bound is +inf
    body_ge0 = (not inf_lb) and inf_ub  # body in [0, +inf)
    body_le0 = inf_lb and (not inf_ub)  # body in (-inf, 0]

    x = m._variables[var_index]
    xlb = float(np.min(np.asarray(x.lb)))
    xub = float(np.max(np.asarray(x.ub)))
    x_ge0 = xlb >= 0.0  # x >= 0
    x_le0 = xub <= 0.0  # x <= 0

    def _unsupported(reason: str) -> ValueError:
        return ValueError(
            f"from_nl: complementarity on variable {x.name!r} has an unsupported "
            f"bound configuration ({reason}; flag={flag}, x in [{xlb:g}, {xub:g}]). "
            "The .nl → GDP slice (#658) handles single-bounded MPEC forms "
            "(body >= 0 or body <= 0, complementary to a single-signed variable); "
            "richer complementarity requires the native node tracked in #231."
        )

    if not (body_ge0 or body_le0):
        _kind = "body fixed to 0" if (not inf_lb and not inf_ub) else "body free / double-bounded"
        raise _unsupported(_kind)
    if not (x_ge0 or x_le0):
        raise _unsupported("complementary variable is free / straddles 0")

    # Orient both sides nonnegative. Negating a side flips the sign of the zero
    # product but not whether it is zero, so ``a * b == 0`` is preserved exactly.
    a = body if body_ge0 else -body
    b = x if x_ge0 else -x
    m.complementarity(a, b, method="gdp")


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
