"""Named index sets and set algebra for sparse modeling.

This module adds a Pyomo/JuMP-style *named set* layer on top of the modeling
API. Sets are the authoritative index for indexed variables, parameters, and
constraints (added in later milestones). A :class:`Set` holds an order-stable,
de-duplicated collection of hashable members and supports set algebra:

* union ``A | B``
* intersection ``A & B``
* difference ``A - B``
* cross product ``A * B`` (lazy, via :class:`ProductSet`)
* filtering ``A.where(pred)``

Members may be scalars (``dimen == 1``) or fixed-arity tuples (``dimen == N``).
All members of a set share the same dimensionality (``dimen``), inferred from
the members and overridable through the ``dimen`` keyword (Pyomo parity).

The set is the single source of truth for which index tuples exist, so summing
over a set always iterates the full declared membership -- avoiding the sparse
aggregation footgun where only previously-accessed indices participate.

Examples
--------
>>> import discopt.modeling as dm
>>> m = dm.Model()
>>> plants = m.set("plants", ["pitt", "sf", "nyc"])
>>> markets = m.set("markets", ["a", "b"])
>>> links = (plants * markets).where(lambda p, k: (p, k) != ("nyc", "a"))
>>> len(links)
5
>>> ("nyc", "a") in links
False
"""

from __future__ import annotations

import itertools
from typing import Callable, Hashable, Iterable, Iterator, Union, cast

Member = Union[Hashable, tuple]


def _normalize_member(raw: object) -> Member:
    """Normalize a raw member to a scalar or tuple.

    Lists are converted to tuples so members are hashable. A one-element tuple
    is unwrapped to its scalar so that ``("pitt",)`` and ``"pitt"`` are treated
    as the same ``dimen == 1`` member.
    """
    if isinstance(raw, list):
        raw = tuple(raw)
    if isinstance(raw, tuple):
        if len(raw) == 1:
            return cast(Member, raw[0])
        return raw
    return cast(Member, raw)


def _arity(member: Member) -> int:
    """Dimensionality of a single member (tuple length, or 1 for a scalar)."""
    return len(member) if isinstance(member, tuple) else 1


def _apply_pred(pred: Callable, member: Member, dimen: int) -> bool:
    """Call a predicate on a member, unpacking tuple members for ``dimen > 1``."""
    if dimen == 1:
        return bool(pred(member))
    assert isinstance(member, tuple)
    return bool(pred(*member))


class _SetBase:
    """Shared algebra/filter/slice behavior for :class:`Set` and :class:`ProductSet`.

    Subclasses provide ``name``, ``dimen``, ``__iter__``, ``__len__``, and
    ``__contains__``; everything else is expressed in terms of those.
    """

    name: str
    dimen: int

    def __iter__(self) -> Iterator[Member]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def __contains__(self, member: object) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def members(self) -> tuple[Member, ...]:
        """Materialized, order-stable tuple of members."""
        return tuple(self)

    # ── set algebra ──

    def _check_same_dimen(self, other: "_SetBase", op: str) -> None:
        if self.dimen != other.dimen:
            raise ValueError(
                f"cannot apply '{op}' to sets of different dimensionality "
                f"({self.name}: dimen={self.dimen}, {other.name}: dimen={other.dimen})"
            )

    def __or__(self, other: "_SetBase") -> "Set":
        self._check_same_dimen(other, "|")
        seen: set = set()
        merged: list[Member] = []
        for member in itertools.chain(self, other):
            if member not in seen:
                seen.add(member)
                merged.append(member)
        return Set(f"({self.name}|{other.name})", merged, dimen=self.dimen)

    def __and__(self, other: "_SetBase") -> "Set":
        self._check_same_dimen(other, "&")
        right = set(other)
        kept = [m for m in self if m in right]
        return Set(f"({self.name}&{other.name})", kept, dimen=self.dimen)

    def __sub__(self, other: "_SetBase") -> "Set":
        self._check_same_dimen(other, "-")
        right = set(other)
        kept = [m for m in self if m not in right]
        return Set(f"({self.name}-{other.name})", kept, dimen=self.dimen)

    def __mul__(self, other: "_SetBase") -> "ProductSet":
        return ProductSet(self, other)

    # ── filtering & slicing ──

    def where(self, pred: Callable[..., bool]) -> "Set":
        """Return the subset whose members satisfy ``pred``.

        For ``dimen == 1`` the predicate is called as ``pred(member)``; for
        ``dimen > 1`` it is called with the unpacked components, ``pred(*member)``.
        """
        kept = [m for m in self if _apply_pred(pred, m, self.dimen)]
        return Set(f"{self.name}|where", kept, dimen=self.dimen)

    def with_first(self, key: Hashable) -> "Set":
        """Members whose first component equals ``key`` (requires ``dimen >= 2``)."""
        if self.dimen < 2:
            raise ValueError("with_first requires a set with dimen >= 2")
        return self.where(lambda *m: m[0] == key)

    def with_last(self, key: Hashable) -> "Set":
        """Members whose last component equals ``key`` (requires ``dimen >= 2``)."""
        if self.dimen < 2:
            raise ValueError("with_last requires a set with dimen >= 2")
        return self.where(lambda *m: m[-1] == key)

    def __repr__(self) -> str:
        return f"Set({self.name!r}, dimen={self.dimen}, len={len(self)})"


class Set(_SetBase):
    """A named index set of hashable members.

    Parameters
    ----------
    name : str
        Identifier for the set (used to synthesize derived-set names and
        constraint/variable index labels).
    members : iterable
        Members of the set. Scalars give ``dimen == 1``; tuples (or lists, which
        are coerced to tuples) of length ``N`` give ``dimen == N``. Duplicates
        are removed, preserving first-occurrence order.
    dimen : int, optional
        Declared dimensionality. Inferred from ``members`` when omitted; required
        (defaults to 1) when ``members`` is empty. When given, every member is
        validated against it.
    """

    def __init__(
        self,
        name: str,
        members: Iterable[object],
        dimen: int | None = None,
    ):
        self.name = name
        normalized: list[Member] = []
        seen: set = set()
        for raw in members:
            member = _normalize_member(raw)
            if member not in seen:
                seen.add(member)
                normalized.append(member)

        if dimen is None:
            dimen = _arity(normalized[0]) if normalized else 1
        if dimen < 1:
            raise ValueError(f"dimen must be >= 1, got {dimen}")
        for member in normalized:
            if _arity(member) != dimen:
                raise ValueError(
                    f"set '{name}' has member {member!r} of arity {_arity(member)}, "
                    f"expected dimen={dimen}"
                )

        self.dimen = dimen
        self._members: tuple[Member, ...] = tuple(normalized)
        self._index: dict[Member, int] = {m: i for i, m in enumerate(self._members)}

    @property
    def members(self) -> tuple[Member, ...]:
        return self._members

    def __iter__(self) -> Iterator[Member]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __contains__(self, member: object) -> bool:
        return _normalize_member(member) in self._index

    def ordinal(self, member: object) -> int:
        """Zero-based position of ``member`` (raises ``KeyError`` if absent)."""
        key = _normalize_member(member)
        if key not in self._index:
            raise KeyError(f"{member!r} is not a member of set '{self.name}'")
        return self._index[key]


class RangeSet(Set):
    """An integer index set ``RangeSet(n) == 1..n`` or ``RangeSet(a, b) == a..b``.

    Provided for parity with Pyomo's ``RangeSet``; the upper bound is inclusive.

    Examples
    --------
    >>> list(RangeSet(3))
    [1, 2, 3]
    >>> list(RangeSet(2, 5))
    [2, 3, 4, 5]
    """

    def __init__(self, start: int, stop: int | None = None, name: str | None = None):
        if stop is None:
            lo, hi = 1, start
        else:
            lo, hi = start, stop
        if name is None:
            name = f"RangeSet({lo},{hi})"
        super().__init__(name, range(lo, hi + 1), dimen=1)


class ProductSet(_SetBase):
    """Lazy Cartesian product of two or more sets.

    Members are flattened tuples of the factor members; iteration is lazy, so a
    product is never fully materialized until iterated (or until ``where`` filters
    it into a concrete :class:`Set`). ``A * B * C`` chains without nesting.
    """

    def __init__(self, *factors: _SetBase):
        flat: list[_SetBase] = []
        for f in factors:
            if isinstance(f, ProductSet):
                flat.extend(f.factors)
            else:
                flat.append(f)
        self.factors: tuple[_SetBase, ...] = tuple(flat)
        self.name = "(" + "*".join(f.name for f in self.factors) + ")"
        self.dimen = sum(f.dimen for f in self.factors)

    def _as_components(self, member: Member) -> tuple:
        return member if isinstance(member, tuple) else (member,)

    def __iter__(self) -> Iterator[Member]:
        for combo in itertools.product(*self.factors):
            flat: tuple = ()
            for f, part in zip(self.factors, combo):
                flat += self._as_components(part)
            yield flat[0] if len(flat) == 1 else flat

    def __len__(self) -> int:
        total = 1
        for f in self.factors:
            total *= len(f)
        return total

    def __contains__(self, member: object) -> bool:
        key = _normalize_member(member)
        comps = self._as_components(key)
        if len(comps) != self.dimen:
            return False
        pos = 0
        for f in self.factors:
            part = comps[pos : pos + f.dimen]
            probe = part[0] if len(part) == 1 else part
            if probe not in f:
                return False
            pos += f.dimen
        return True

    def __mul__(self, other: "_SetBase") -> "ProductSet":
        return ProductSet(self, other)
