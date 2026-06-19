"""Indexed containers backed by flat variables/parameters over named sets.

This module sits between the named-set layer (:mod:`discopt.modeling.sets`) and
the flat model representation in :mod:`discopt.modeling.core`. An indexed
variable is backed by exactly one flat :class:`~discopt.modeling.core.Variable`
of shape ``(len(index_set),)`` plus the set's ``key -> position`` map, so
indexing desugars to the existing ``IndexExpression`` that the JAX compiler and
``.nl`` exporter already flatten. Nothing downstream changes.

Containers are created through ``Model.continuous(..., over=SET)`` and friends,
not directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Hashable, cast

from discopt.modeling.sets import Set

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from discopt.modeling.core import (
        Constraint,
        IndexExpression,
        Parameter,
        SolveResult,
        Variable,
    )


class _SkipType:
    """Singleton sentinel returned from a constraint rule to omit that key."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Skip"


#: Sentinel a ``Model.constraint`` rule may return to skip a member
#: (Pyomo ``Constraint.Skip`` parity).
Skip = _SkipType()


def key_label(member) -> str:
    """Render a set member as a constraint-name label, e.g. ``pitt`` or ``pitt,a``."""
    if isinstance(member, tuple):
        return ",".join(str(p) for p in member)
    return str(member)


class IndexedVar:
    """A variable indexed by a named :class:`~discopt.modeling.sets.Set`.

    Backed by a single flat variable; ``iv[key]`` returns the scalar
    :class:`~discopt.modeling.core.IndexExpression` at the key's position.

    Attributes
    ----------
    flat : Variable
        The backing flat variable, shape ``(len(index_set),)``.
    index_set : Set
        The authoritative index.
    name : str
        Same as ``flat.name``.
    """

    def __init__(self, flat: "Variable", index_set: Set):
        self.flat = flat
        self.index_set = index_set
        self.name = flat.name

    def __getitem__(self, key: Hashable) -> "IndexExpression":
        pos = self.index_set.ordinal(key)
        return cast("IndexExpression", self.flat[pos])

    def __iter__(self):
        return iter(self.index_set)

    def __len__(self) -> int:
        return len(self.index_set)

    def __contains__(self, key: object) -> bool:
        return key in self.index_set

    def keys(self):
        """The index-set members, in order."""
        return self.index_set.members

    def value(self, result: "SolveResult") -> dict:
        """Return ``{key: optimal_value}`` from a solved result."""
        arr = result.value(self.flat)
        flat = arr.reshape(-1)
        return {k: float(flat[self.index_set.ordinal(k)]) for k in self.index_set}

    def __repr__(self) -> str:
        return f"IndexedVar({self.name!r}, over={self.index_set.name}, len={len(self)})"


class IndexedParam:
    """A parameter indexed by a named :class:`~discopt.modeling.sets.Set`.

    Backed by a single flat :class:`~discopt.modeling.core.Parameter`; ``ip[key]``
    returns the scalar element at the key's position.
    """

    def __init__(self, flat: "Parameter", index_set: Set):
        self.flat = flat
        self.index_set = index_set
        self.name = flat.name

    def __getitem__(self, key: Hashable) -> "IndexExpression":
        pos = self.index_set.ordinal(key)
        return cast("IndexExpression", self.flat[pos])

    def __iter__(self):
        return iter(self.index_set)

    def __len__(self) -> int:
        return len(self.index_set)

    def __contains__(self, key: object) -> bool:
        return key in self.index_set

    def keys(self):
        """The index-set members, in order."""
        return self.index_set.members

    def at(self, key: Hashable) -> float:
        """The current numeric value at ``key`` (not an expression)."""
        pos = self.index_set.ordinal(key)
        return float(self.flat.value.reshape(-1)[pos])

    def __repr__(self) -> str:
        return f"IndexedParam({self.name!r}, over={self.index_set.name}, len={len(self)})"


class IndexedConstraint:
    """A family of constraints generated over a named set, keyed by member.

    Returned by :meth:`Model.constraint`. Each present member maps to the flat
    :class:`~discopt.modeling.core.Constraint` that was added to the model
    (members whose rule returned :data:`Skip` are absent).
    """

    def __init__(self, name, index_set: Set, members: dict, fast: bool = False):
        self.name = name
        self.index_set = index_set
        self._members: dict = members
        #: True when the family was emitted via the linear fast path (rows live
        #: in the Rust builder; the held Constraint objects are for introspection
        #: only and are not in ``model._constraints``).
        self.fast = fast

    def __getitem__(self, key) -> "Constraint":
        from discopt.modeling.sets import _normalize_member

        norm = _normalize_member(key)
        if norm not in self._members:
            raise KeyError(f"no constraint for key {key!r} in '{self.name}'")
        return cast("Constraint", self._members[norm])

    def __iter__(self):
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def keys(self):
        """Members for which a constraint was generated (Skip omitted)."""
        return tuple(self._members.keys())

    def __repr__(self) -> str:
        return f"IndexedConstraint({self.name!r}, len={len(self)})"


class _Affine:
    """Affine form ``sum(coeffs[pos] * var[pos]) + const`` over a single variable."""

    __slots__ = ("const", "var", "coeffs")

    def __init__(self):
        self.const: float = 0.0
        self.var = None  # a single core.Variable, or None for a pure constant
        self.coeffs: dict = {}  # position -> coefficient


def _aff_scale(a, s: float):
    if a is None:
        return None
    b = _Affine()
    b.const = a.const * s
    b.var = a.var
    b.coeffs = {pos: c * s for pos, c in a.coeffs.items()}
    return b


def _aff_add(left, right):
    if left is None or right is None:
        return None
    if left.var is not None and right.var is not None and left.var is not right.var:
        return None  # two distinct variables -> not single-variable affine
    out = _Affine()
    out.const = left.const + right.const
    out.var = left.var if left.var is not None else right.var
    out.coeffs = dict(left.coeffs)
    for pos, c in right.coeffs.items():
        out.coeffs[pos] = out.coeffs.get(pos, 0.0) + c
    return out


def affine_form(expr):
    """Reduce a scalar expression to a single-variable :class:`_Affine`, or ``None``.

    Returns ``None`` (caller should fall back to the general path) for anything
    that is not affine in a single backing variable: nonlinear nodes, bilinear
    products, array nodes, or any reference to a :class:`Parameter` (whose value
    must stay symbolic for sensitivity) or to a second distinct variable.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        IndexExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    if isinstance(expr, Constant):
        if expr.value.ndim != 0:
            return None
        a = _Affine()
        a.const = float(expr.value)
        return a
    if isinstance(expr, IndexExpression):
        base = expr.base
        if not isinstance(base, Variable) or not isinstance(expr.index, int):
            return None
        a = _Affine()
        a.var = base
        a.coeffs = {expr.index: 1.0}
        return a
    if isinstance(expr, Variable):
        if expr.size != 1:
            return None
        a = _Affine()
        a.var = expr
        a.coeffs = {0: 1.0}
        return a
    if isinstance(expr, UnaryOp):
        if expr.op != "neg":
            return None
        return _aff_scale(affine_form(expr.operand), -1.0)
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            left = affine_form(expr.left)
            right = affine_form(expr.right)
            if expr.op == "-":
                right = _aff_scale(right, -1.0)
            return _aff_add(left, right)
        if expr.op == "*":
            left = affine_form(expr.left)
            right = affine_form(expr.right)
            if left is None or right is None:
                return None
            if left.var is None:
                return _aff_scale(right, left.const)
            if right.var is None:
                return _aff_scale(left, right.const)
            return None  # bilinear
        if expr.op == "/":
            left = affine_form(expr.left)
            right = affine_form(expr.right)
            if left is None or right is None or right.var is not None or right.const == 0:
                return None
            return _aff_scale(left, 1.0 / right.const)
        return None
    if isinstance(expr, SumOverExpression):
        acc = _Affine()
        for term in expr.terms:
            acc = _aff_add(acc, affine_form(term))
            if acc is None:
                return None
        return acc
    return None


def resolve_indexed_values(index_set: Set, spec, default, dtype) -> "np.ndarray":
    """Resolve a per-key bound/value spec into a flat array in set order.

    ``spec`` may be:

    * ``None`` -- use ``default`` for every member;
    * a scalar -- broadcast to every member;
    * a ``dict`` -- looked up by member (every member must be present);
    * a callable -- ``fn(member)`` (or ``fn(*member)`` for ``dimen > 1``).
    """
    import numpy as np

    from discopt.modeling.sets import call_member

    n = len(index_set)
    if spec is None:
        spec = default
    if callable(spec):
        vals = [call_member(spec, m, index_set.dimen) for m in index_set]
        return np.asarray(vals, dtype=dtype)
    if isinstance(spec, dict):
        try:
            vals = [spec[m] for m in index_set.members]
        except KeyError as exc:
            raise KeyError(
                f"no entry for member {exc.args[0]!r} of set '{index_set.name}'"
            ) from None
        return np.asarray(vals, dtype=dtype)
    return np.broadcast_to(np.asarray(spec, dtype=dtype), (n,)).copy()
