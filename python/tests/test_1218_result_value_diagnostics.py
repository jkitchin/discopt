"""``SolveResult.value`` says why a lookup failed, instead of ``KeyError`` (#1218).

``result.x`` is keyed by variable name, so anything that is not a column of the
solved model has no entry in it. Looking the argument's ``.name`` up anyway gave
the user a bare ``KeyError`` on the node's *display* name -- the reported case
was ``KeyError: 'argmin'`` from ``result.value(v)`` on a ``dm.argmin`` block,
where the honest answer is structural: the follower is solved inside an opaque
node, so its ``y*`` is not a variable of the outer model at all.

The diagnosis must not cost the lookups that already worked: ``result.value``
resolves by name first, so every handle naming a real column (a ``Variable``, an
``IndexedVar``, anything else carrying the name) keeps working untouched, and
only a lookup that WOULD have raised reaches the new message.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult

pytestmark = pytest.mark.unit


def _result(**x) -> SolveResult:
    return SolveResult(status="optimal", x=dict(x), objective=0.0)


def test_a_plain_variable_still_resolves():
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=1.0)
    assert _result(x=np.array([0.25])).value(x) == pytest.approx(np.array([0.25]))


def test_an_indexed_container_still_resolves():
    """``IndexedVar`` is not a ``Variable``; it names one, and that is enough."""
    m = dm.Model("m")
    s = m.set("S", [10, 20, 30])
    y = m.continuous("y", lb=1.0, ub=5.0, over=s)
    assert not isinstance(y, dm.Variable)

    got = np.asarray(_result(y=np.ones(3)).value(y), dtype=float).ravel()
    assert got.size == 3


def test_an_opaque_custom_node_is_refused_with_the_reason():
    import jax.numpy as jnp

    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=1.0)
    node = dm.custom(lambda t: jnp.sin(t), name="argmin")(x)

    with pytest.raises(TypeError) as exc:
        _result(x=np.array([0.25])).value(node)
    message = str(exc.value)
    assert "argmin" in message  # which node
    assert "CustomCall" in message  # what it is
    assert "argmin_layer" in message and "argmin_kkt" in message  # what to do


def test_an_ordinary_expression_is_refused_with_the_reason():
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=1.0)

    with pytest.raises(TypeError, match="expression"):
        _result(x=np.array([0.25])).value(x + 1.0)


def test_a_variable_of_another_model_says_which_variable():
    other = dm.Model("other")
    stranger = other.continuous("stranger", lb=0.0, ub=1.0)

    with pytest.raises(KeyError) as exc:
        _result(x=np.array([0.25])).value(stranger)
    message = str(exc.value)
    assert "stranger" in message
    assert "not a variable of the model" in message
    assert "Solution variables: x." in message  # and what the solution does carry


def test_no_solution_still_reports_no_solution_first():
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=1.0)
    empty = SolveResult(status="infeasible", x=None)

    with pytest.raises(ValueError, match="No solution available"):
        empty.value(x)
