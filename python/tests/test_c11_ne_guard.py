"""Regression test for correctness issue C-11.

The modeling API overrides ``Expression.__eq__`` to build an equality
``Constraint`` (``x == y`` → a constraint, not a bool). It did **not** define
``__ne__``. Python's default ``__ne__`` is ``not __eq__(...)``, so ``x != y``
evaluated to ``not <truthy Constraint>`` → ``False`` *silently* — with no error
and no ``Constraint`` object. User-side control flow such as ``if x != 0:`` then
misbehaved with no diagnostic (CLAUDE.md §3: the modeling API must not silently
mis-encode the user's intent; refuse loudly instead).

``!=`` is not a valid optimization constraint operator, so the sound fix is a
loud ``TypeError`` — not a silent ``False`` and not a fabricated constraint.

These tests call the modeling API directly (sub-second, no ``Model.solve()``).
"""

import pytest
from discopt import Model


@pytest.mark.smoke
def test_ne_against_constant_raises():
    """``x != 0`` must raise, not silently evaluate to False (fail-before/pass-after)."""
    m = Model()
    x = m.continuous("x")
    with pytest.raises(TypeError):
        x != 0


@pytest.mark.smoke
def test_ne_between_expressions_raises():
    """``x != y`` on two expressions must raise loudly."""
    m = Model()
    x = m.continuous("x")
    y = m.continuous("y")
    with pytest.raises(TypeError):
        x != y


@pytest.mark.smoke
def test_ne_on_derived_expression_raises():
    """The guard lives on the Expression base, so derived nodes are covered too."""
    m = Model()
    x = m.continuous("x")
    with pytest.raises(TypeError):
        (x + 1) != 2  # BinaryOp
    with pytest.raises(TypeError):
        (-x) != 0  # UnaryOp


@pytest.mark.smoke
def test_ne_on_indexed_expression_raises():
    """Indexed array-variable expressions are guarded as well."""
    m = Model()
    v = m.continuous("v", shape=(3,))
    with pytest.raises(TypeError):
        v[0] != 0


@pytest.mark.smoke
def test_eq_still_builds_a_constraint():
    """The fix must not disturb ``==``: it still yields an equality Constraint."""
    m = Model()
    x = m.continuous("x")
    c = x == 0
    assert type(c).__name__ == "Constraint"


@pytest.mark.smoke
def test_variable_remains_hashable():
    """Defining ``__ne__`` must not break hashability (used in dicts/sets)."""
    m = Model()
    x = m.continuous("x")
    # Should not raise; Variable defines __hash__ explicitly.
    hash(x)
    assert x in {x}
