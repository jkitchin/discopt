"""Public SymPy bridge — component **E** of #1248, via #1278.

``to_sympy`` turns a discopt :class:`~discopt.modeling.core.Expression` into a
SymPy expression; ``from_sympy`` turns one back. SymPy stays an **optional**
dependency (``pip install discopt[sympy]``) — importing this module without it
raises a message naming the extra, and nothing on the solve path imports it.

Why a bridge, and why a public one
----------------------------------
There is already a private, one-way translator in
``_relax/symbolic/cut_recognizer.model_to_sympy``, built for pattern matching
inside the cut recognizer. It is deliberately lossy in ways a public bridge must
not be:

* an indexed node it cannot key becomes ``sp.Dummy("opaque")`` — translation
  continues and the recognizer simply matches nothing, which is right for a
  recognizer and silently wrong for a round trip;
* it maps a call by ``getattr(sp, name)``, so discopt's own intrinsics
  (``entropy``/``sigmoid``/``softplus``/``log2``/``log10``/``log1p``) raise
  ``AttributeError`` — and ``entropy`` is exactly what #1248's acceptance names;
* every constant becomes ``sp.Float``, so ``x**2`` and ``x/3`` lose exactness.

This module refuses rather than guesses. Anything it cannot translate raises
:class:`SymbolicTranslationError` naming the node — CLAUDE.md §3: a silent
approximation in a translation layer is a wrong model that does not raise.

The six intrinsics SymPy has no function for
--------------------------------------------
``entropy`` (``dm.xlogx``), ``sigmoid``, ``softplus``, ``log2``, ``log10`` and
``log1p`` are represented by :class:`sympy.Function` subclasses defined here,
each carrying its own ``fdiff`` and numeric evaluation. Writing ``log2(x)`` as
``log(x)/log(2)`` would be mathematically equal and would not round-trip, and
``sigmoid``/``softplus``/``entropy`` have no closed sympy spelling at all. With
the subclasses, ``from_sympy(to_sympy(e)) == e`` structurally, ``sp.diff`` gives
the right derivative, and ``sp.lambdify`` evaluates numerically.

Registered atoms (#1248 A)
--------------------------
``register_function(name, lower)`` names a composite the relaxer envelopes whole.
:func:`sympy_function_for` returns a ``sympy.Function`` subclass bound to such a
registration, so a model built through the bridge keeps the atom rather than
flattening into the lowering. The binding is by NAME: translating that function
back produces a call to the registration that is live at that moment, and a name
with no registration is refused rather than silently lowered.

Examples
--------
>>> import discopt.modeling as dm                       # doctest: +SKIP
>>> from discopt.symbolic import to_sympy, from_sympy   # doctest: +SKIP
>>> m = dm.Model()                                      # doctest: +SKIP
>>> x = m.continuous("x", lb=0.1, ub=2.0)               # doctest: +SKIP
>>> s, syms = to_sympy(dm.exp(x) + dm.xlogx(x))         # doctest: +SKIP
>>> s                                                   # doctest: +SKIP
exp(x) + entropy(x)
>>> back = from_sympy(s, syms)          # a discopt Expression again
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Expression

__all__ = [
    "to_sympy",
    "numeric_modules",
    "from_sympy",
    "sympy_function_for",
    "SymbolicTranslationError",
    "DISCOPT_FUNCTIONS",
]


class SymbolicTranslationError(TypeError):
    """A node or SymPy object this bridge will not translate.

    Raised rather than approximated: a translation that quietly drops or
    reinterprets a term produces a *wrong model that does not raise*, which is
    the failure mode this module exists to avoid.
    """


def _require_sympy():
    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - exercised only without sympy
        raise ImportError(
            "discopt.symbolic needs SymPy, which is an optional dependency. "
            "Install it with `pip install discopt[sympy]`."
        ) from exc
    return sp


# --------------------------------------------------------------------------- #
# The intrinsics SymPy has no function for
# --------------------------------------------------------------------------- #
def _build_discopt_functions() -> dict:
    """``{discopt name: sympy.Function subclass}`` for the six SymPy lacks.

    Built lazily so importing this module's *name* costs nothing; each class
    carries ``fdiff`` (so ``sp.diff`` is correct) and ``eval``/``_eval_evalf``
    (so ``sp.lambdify`` and ``.evalf()`` work), because #1248's acceptance is a
    round trip matching **values and gradients**.
    """
    sp = _require_sympy()

    class entropy(sp.Function):  # noqa: N801 - the canonical node name
        """``t*log(t)`` — discopt's ``entropy`` intrinsic (``dm.xlogx``)."""

        @classmethod
        def eval(cls, t):
            if t.is_Number and t.is_positive:
                return t * sp.log(t)
            if t is sp.S.One:
                return sp.S.Zero

        def fdiff(self, argindex=1):
            if argindex != 1:
                raise sp.ArgumentIndexError(self, argindex)
            return sp.log(self.args[0]) + 1

        def _eval_evalf(self, prec):
            t = self.args[0]._eval_evalf(prec)
            return None if t is None else (t * sp.log(t))._eval_evalf(prec)

    class sigmoid(sp.Function):  # noqa: N801
        """``1/(1+exp(-t))``."""

        def fdiff(self, argindex=1):
            if argindex != 1:
                raise sp.ArgumentIndexError(self, argindex)
            s = sigmoid(self.args[0])
            return s * (1 - s)

        def _eval_evalf(self, prec):
            t = self.args[0]._eval_evalf(prec)
            return None if t is None else (1 / (1 + sp.exp(-t)))._eval_evalf(prec)

    class softplus(sp.Function):  # noqa: N801
        """``log(1+exp(t))``."""

        def fdiff(self, argindex=1):
            if argindex != 1:
                raise sp.ArgumentIndexError(self, argindex)
            return sigmoid(self.args[0])

        def _eval_evalf(self, prec):
            t = self.args[0]._eval_evalf(prec)
            return None if t is None else sp.log(1 + sp.exp(t))._eval_evalf(prec)

    def _log_base(cls_name, base_expr, doc):
        class _LogBase(sp.Function):
            def fdiff(self, argindex=1):
                if argindex != 1:
                    raise sp.ArgumentIndexError(self, argindex)
                return 1 / (self.args[0] * base_expr())

            def _eval_evalf(self, prec):
                t = self.args[0]._eval_evalf(prec)
                if t is None:
                    return None
                return (sp.log(t) / base_expr())._eval_evalf(prec)

        _LogBase.__name__ = cls_name
        _LogBase.__qualname__ = cls_name
        _LogBase.__doc__ = doc
        return _LogBase

    log2 = _log_base("log2", lambda: sp.log(2), "``log(t)/log(2)``.")
    log10 = _log_base("log10", lambda: sp.log(10), "``log(t)/log(10)``.")

    class log1p(sp.Function):  # noqa: N801
        """``log(1+t)``, kept as its own node so it round-trips."""

        def fdiff(self, argindex=1):
            if argindex != 1:
                raise sp.ArgumentIndexError(self, argindex)
            return 1 / (1 + self.args[0])

        def _eval_evalf(self, prec):
            t = self.args[0]._eval_evalf(prec)
            return None if t is None else sp.log(1 + t)._eval_evalf(prec)

    return {
        "entropy": entropy,
        "sigmoid": sigmoid,
        "softplus": softplus,
        "log2": log2,
        "log10": log10,
        "log1p": log1p,
    }


#: Numeric implementations of the six added functions, for :func:`sympy.lambdify`.
#: A ``sympy.Function`` subclass has no numeric meaning to ``lambdify`` on its own
#: -- the generated code calls ``entropy(x)`` and raises ``NameError`` -- so a
#: caller who lambdifies a bridged expression needs this as a module dict:
#:
#:     f = sp.lambdify(syms, expr, modules=[numeric_modules(), "numpy"])
#:
#: Each entry is the same function the solver evaluates, on the same branch-free
#: overflow-safe form (``logaddexp`` for softplus, the sign-split sigmoid).
def numeric_modules() -> dict:
    """``{name: callable}`` for :func:`sympy.lambdify`, covering the six added functions."""

    def _sigmoid(t):
        t = np.asarray(t, dtype=float)
        return np.where(
            t >= 0,
            1.0 / (1.0 + np.exp(-np.abs(t))),
            np.exp(-np.abs(t)) / (1.0 + np.exp(-np.abs(t))),
        )

    return {
        "entropy": lambda t: np.asarray(t, dtype=float) * np.log(np.asarray(t, dtype=float)),
        "sigmoid": _sigmoid,
        "softplus": lambda t: np.logaddexp(0.0, np.asarray(t, dtype=float)),
        "log2": lambda t: np.log2(np.asarray(t, dtype=float)),
        "log10": lambda t: np.log10(np.asarray(t, dtype=float)),
        "log1p": lambda t: np.log1p(np.asarray(t, dtype=float)),
    }


_DISCOPT_FUNCTIONS: Optional[dict] = None


def DISCOPT_FUNCTIONS() -> dict:  # noqa: N802 - a named table, built lazily
    """``{discopt intrinsic name: sympy.Function subclass}`` for the six SymPy lacks."""
    global _DISCOPT_FUNCTIONS
    if _DISCOPT_FUNCTIONS is None:
        _DISCOPT_FUNCTIONS = _build_discopt_functions()
    return _DISCOPT_FUNCTIONS


#: Intrinsics whose discopt name IS the SymPy name.
_SAME_NAME = frozenset(
    {
        "exp", "log", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "erf",
    }
)  # fmt: skip

#: Intrinsics whose SymPy spelling differs in capitalisation only.
_RENAMED = {"abs": "Abs", "sign": "sign"}

#: Registered-atom classes handed out by :func:`sympy_function_for`, by name.
_ATOM_FUNCTIONS: dict = {}


def sympy_function_for(name: str):
    """A ``sympy.Function`` subclass standing for the registered atom ``name``.

    ``name`` must already be registered through
    :func:`discopt.operators.register_function`; an unregistered name is refused
    rather than silently lowered, because the whole point of the atom is that the
    relaxer sees it instead of its lowering.

    The binding is by NAME and resolved at :func:`from_sympy` time, so
    re-registering the name (``replace=True``) changes what the same SymPy
    function translates to — the same contract the model layer has.
    """
    sp = _require_sympy()
    from discopt.operators import get_registered

    if get_registered(name) is None:
        raise SymbolicTranslationError(
            f"sympy_function_for({name!r}): no registered atom by that name. "
            "Call discopt.modeling.register_function(name, lower) first — a SymPy "
            "function with no registration behind it cannot be translated back."
        )
    if name in _ATOM_FUNCTIONS:
        return _ATOM_FUNCTIONS[name]

    cls = type(name, (sp.Function,), {"__doc__": f"discopt registered atom {name!r}."})
    _ATOM_FUNCTIONS[name] = cls
    return cls


# --------------------------------------------------------------------------- #
# discopt -> SymPy
# --------------------------------------------------------------------------- #
def _symbol_name(node, model) -> str:
    """``name`` for a scalar variable, ``name_i`` for element ``i`` of a vector."""
    from discopt.modeling.core import IndexExpression, Variable

    if isinstance(node, Variable):
        return str(node.name)
    assert isinstance(node, IndexExpression)
    base = node.base
    if not isinstance(base, Variable):
        raise SymbolicTranslationError(
            "to_sympy: an index into a compound expression "
            f"({type(base).__name__}) has no stable symbol; index the variable itself."
        )
    shape = tuple(base.shape or ())
    if len(shape) != 1:
        raise SymbolicTranslationError(
            f"to_sympy: {base.name!r} has shape {shape}; only scalar and 1-D "
            "variables have a symbol spelling this bridge can invert."
        )
    from discopt._flat_index import flat_index_in_shape

    flat = flat_index_in_shape(node.index, shape)
    if flat is None:
        raise SymbolicTranslationError(
            f"to_sympy: cannot resolve the index {node.index!r} of {base.name!r} "
            "to a single element."
        )
    return f"{base.name}_{int(flat)}"


def to_sympy(expr: "Expression", symbols: Optional[dict] = None, *, model=None):
    """Translate a discopt expression to SymPy.

    Parameters
    ----------
    expr : Expression
        Any discopt expression: variables, arithmetic, intrinsics, and atoms
        registered through ``register_function``.
    symbols : dict, optional
        ``{sympy.Symbol: discopt Expression}`` to extend, so several expressions
        over the same model share one symbol set. A fresh dict is made otherwise.
    model : Model, optional
        Unused for the translation itself; accepted so callers can pass the
        owning model without a second code path.

    Returns
    -------
    (sympy.Expr, dict)
        The expression and the symbol map, the latter being exactly what
        :func:`from_sympy` needs to invert it.

    Raises
    ------
    SymbolicTranslationError
        On any node with no exact SymPy spelling. Nothing is approximated.
    """
    sp = _require_sympy()
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        Expression,
        FunctionCall,
        IndexExpression,
        Parameter,
        UnaryOp,
        Variable,
    )
    from discopt.operators import atom_of

    if not isinstance(expr, Expression):
        raise SymbolicTranslationError(
            f"to_sympy: expected a discopt Expression, got {type(expr).__name__}."
        )

    symbols = {} if symbols is None else symbols
    by_name = {str(s): s for s in symbols}

    def _sym(node):
        name = _symbol_name(node, model)
        if name in by_name:
            return by_name[name]
        s = sp.Symbol(name, real=True)
        by_name[name] = s
        symbols[s] = node
        return s

    def _number(value) -> Any:
        arr = np.asarray(value)
        if arr.ndim != 0:
            raise SymbolicTranslationError(
                f"to_sympy: array-valued constant/parameter of shape {arr.shape}; "
                "this bridge translates scalar expressions."
            )
        f = float(arr)
        if not math.isfinite(f):
            raise SymbolicTranslationError(f"to_sympy: non-finite constant {f!r}.")
        # Keep an exactly-representable integer exact so `x**2` stays polynomial
        # and `x/3` does not become a float division.
        return sp.Integer(int(f)) if f == int(f) and abs(f) < 2**53 else sp.Float(f)

    def _walk(node):
        atom = atom_of(node)
        if atom is not None:
            name, arg = atom
            return sympy_function_for(name)(_walk(arg))
        if isinstance(node, Constant):
            return _number(node.value)
        if isinstance(node, Parameter):
            return _number(node.value)
        if isinstance(node, (Variable, IndexExpression)):
            return _sym(node)
        if isinstance(node, UnaryOp):
            inner = _walk(node.operand)
            if node.op == "neg":
                return -inner
            if node.op == "abs":
                return sp.Abs(inner)
            raise SymbolicTranslationError(f"to_sympy: unary op {node.op!r}.")
        if isinstance(node, BinaryOp):
            left, right = _walk(node.left), _walk(node.right)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            if node.op == "/":
                return left / right
            if node.op == "**":
                return left**right
            raise SymbolicTranslationError(f"to_sympy: binary op {node.op!r}.")
        if isinstance(node, FunctionCall):
            args = [_walk(a) for a in node.args]
            name = node.func_name
            table = DISCOPT_FUNCTIONS()
            if name in table:
                return table[name](*args)
            if name in _SAME_NAME:
                return getattr(sp, name)(*args)
            if name in _RENAMED:
                return getattr(sp, _RENAMED[name])(*args)
            if name in ("min", "max"):
                return (sp.Min if name == "min" else sp.Max)(*args)
            raise SymbolicTranslationError(
                f"to_sympy: intrinsic {name!r} has no SymPy spelling in this bridge. "
                "Add it to the table rather than approximating it."
            )
        raise SymbolicTranslationError(f"to_sympy: node type {type(node).__name__}.")

    return _walk(expr), symbols


# --------------------------------------------------------------------------- #
# SymPy -> discopt
# --------------------------------------------------------------------------- #
def from_sympy(expr, symbol_map: dict) -> "Expression":
    """Translate a SymPy expression back into a discopt expression.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to translate.
    symbol_map : dict
        ``{sympy.Symbol: discopt Expression}`` — the second return value of
        :func:`to_sympy`, or any mapping the caller builds. A free symbol with no
        entry is refused: inventing a variable would produce a model over
        something the caller never declared.

    Returns
    -------
    Expression

    Raises
    ------
    SymbolicTranslationError
        On a SymPy object with no discopt spelling (``Piecewise``, ``floor``,
        ``ceiling``, an unbound symbol, a complex number, an unregistered atom).
    """
    sp = _require_sympy()
    import discopt.modeling as dm
    from discopt.modeling.core import Expression

    table = DISCOPT_FUNCTIONS()
    by_name = {str(s): e for s, e in symbol_map.items()}
    inverse_atom = {cls.__name__: name for name, cls in _ATOM_FUNCTIONS.items()}
    inverse_table = {cls.__name__: name for name, cls in table.items()}
    same = {n: getattr(dm, n) for n in _SAME_NAME}
    renamed = {"Abs": dm.abs, "sign": dm.sign}

    def _walk(node):
        if isinstance(node, sp.Symbol):
            got = symbol_map.get(node, by_name.get(str(node)))
            if got is None:
                raise SymbolicTranslationError(
                    f"from_sympy: the symbol {node!r} is not in symbol_map. Pass the "
                    "map to_sympy returned, or add an entry for it — this bridge will "
                    "not invent a variable."
                )
            return got
        if node.is_Number:
            if not node.is_real:
                raise SymbolicTranslationError(f"from_sympy: non-real number {node!r}.")
            return float(node)
        if node in (sp.pi, sp.E) or node.is_NumberSymbol:
            return float(node.evalf())
        if isinstance(node, sp.Add):
            acc = _walk(node.args[0])
            for a in node.args[1:]:
                acc = acc + _walk(a)
            return acc
        if isinstance(node, sp.Mul):
            acc = _walk(node.args[0])
            for a in node.args[1:]:
                acc = acc * _walk(a)
            return acc
        if isinstance(node, sp.Pow):
            base, exponent = node.args
            b = _walk(base)
            if exponent.is_Number:
                e = float(exponent)
                # `x**-1` is how SymPy spells division; keep it as a division so the
                # model layer sees the same shape the caller wrote.
                if e == -1.0:
                    return 1.0 / b
                return b**e
            return b ** _walk(exponent)
        if isinstance(node, sp.Function):
            cls_name = type(node).__name__
            args = [_walk(a) for a in node.args]
            if cls_name in inverse_atom:
                from discopt.operators import get_registered

                fn = get_registered(inverse_atom[cls_name])
                if fn is None:
                    raise SymbolicTranslationError(
                        f"from_sympy: {cls_name!r} names a registered atom that is no "
                        "longer registered. Re-register it, or translate its lowering."
                    )
                return fn(*args)
            if cls_name in inverse_table:
                name = inverse_table[cls_name]
                builder = dm.xlogx if name == "entropy" else getattr(dm, name)
                return builder(*args)
            if cls_name in same:
                return same[cls_name](*args)
            if cls_name in renamed:
                return renamed[cls_name](*args)
            if cls_name in ("Min", "Max"):
                return (dm.min if cls_name == "Min" else dm.max)(*args)
            raise SymbolicTranslationError(
                f"from_sympy: SymPy function {cls_name!r} has no discopt spelling. "
                "Register it with dm.register_function and sympy_function_for, or "
                "rewrite it in the elementary set."
            )
        raise SymbolicTranslationError(
            f"from_sympy: SymPy object {type(node).__name__} ({node!r}) has no discopt "
            "spelling. Nothing is approximated here."
        )

    out = _walk(sp.sympify(expr))
    if not isinstance(out, Expression):
        # A constant-only expression translates to a float; wrap it so callers
        # always get an Expression back.
        from discopt.modeling.core import Constant

        return Constant(float(out))
    return out


def _unused(*_a: Callable) -> None:  # pragma: no cover - keeps linters honest
    pass
