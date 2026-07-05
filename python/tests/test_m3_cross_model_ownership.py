"""M3 (#413) — cross-model expressions must be rejected, never aliased by index.

Finding M3 (``docs/dev/modeling-module-review.md``): a :class:`Variable` carries
a flat ``_index`` into *its own* model's variable vector. Before this fix, mixing
a variable from model B into model A's objective/constraint was silently
accepted; the solver addressed the foreign variable by that flat index and it
**aliased** whatever variable of the solved model had the same index → a
plausible-looking **wrong answer** rather than an error.

The guard is a single O(DAG) walk in ``Model.validate()`` (called by
``solve()``): every ``Variable``/``Parameter`` reached from the objective and
constraints must be *index-compatible* with the model being solved — i.e.
``self._variables[leaf._index] is leaf`` (or the leaf is one of this model's own
objects). This is deliberately an **index-slot identity** test, not an
``owner is self`` test, so that legitimate model rebuilds that *share* the same
Variable objects (e.g. ``reformulate_gdp``) are accepted while a genuinely
foreign, index-incompatible leaf is rejected.

These tests pin the *class* of the fix:

* the guard fires at **solve time** for any cross-model reference reaching the
  objective or a constraint, so the aliasing can never reach the Rust repr; and
* legitimate single-model expressions — constants, Python scalars, numpy-array
  constants, broadcasts, parameters, matmuls, and GDP models whose reformulation
  shares variable objects across an internal model rebuild — do **not** raise
  (no false positives).

Fails-before / passes-after: on pre-fix code the cross-model ``solve()`` cases
returned a silently wrong ``optimal`` result instead of raising.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model

pytestmark = pytest.mark.smoke


# ─────────────────────────── positive: must RAISE ───────────────────────────


def test_cross_model_aliasing_would_be_wrong_answer():
    """The headline M3 repro: min xa + xb with xb from another model.

    If honored, xa∈[0,10], xb∈[5,10] gives optimum 5 (xa=0, xb=5). Pre-fix, xb
    (index 0 in B) aliased xa (index 0 in A), collapsing the objective to
    ``2*xa`` → optimum ~0. The guard now refuses at solve time instead.
    """
    ma = Model("A")
    xa = ma.continuous("xa", lb=0, ub=10)
    mb = Model("B")
    xb = mb.continuous("xb", lb=5, ub=10)

    ma.minimize(xa + xb)
    with pytest.raises(ValueError, match="[Cc]ross-model"):
        ma.solve()


def test_cross_model_objective_various_operators_raise():
    """The whole class of cross-model objective combinations is rejected."""
    for build in (
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / (b + 1.0),
        lambda a, b: dm.exp(a) + dm.log(b + 1.0),
    ):
        ma = Model("A")
        xa = ma.continuous("xa", lb=1, ub=10)
        mb = Model("B")
        xb = mb.continuous("xb", lb=1, ub=10)
        ma.minimize(build(xa, xb))
        with pytest.raises(ValueError, match="[Cc]ross-model"):
            ma.solve()


def test_cross_model_foreign_objective_raises_at_solve():
    """A foreign var used as the *whole* objective (no combine) still raises."""
    ma = Model("A")
    ma.continuous("xa", lb=0, ub=10)  # index 0 in A
    mb = Model("B")
    xb = mb.continuous("xb", lb=5, ub=10)  # index 0 in B, aliases xa pre-fix

    ma.minimize(xb)
    with pytest.raises(ValueError, match="[Cc]ross-model"):
        ma.solve()


def test_cross_model_foreign_constraint_raises_at_solve():
    """A foreign var reaching a constraint body is caught too."""
    ma = Model("A")
    xa = ma.continuous("xa", lb=0, ub=10)
    mb = Model("B")
    xb = mb.continuous("xb", lb=0, ub=10)

    ma.minimize(xa)
    ma.subject_to(xa + xb <= 3.0)  # xb foreign
    with pytest.raises(ValueError, match="[Cc]ross-model"):
        ma.solve()


def test_cross_model_parameter_raises_at_solve():
    """A foreign Parameter mixed into the objective is rejected."""
    ma = Model("A")
    xa = ma.continuous("xa", lb=1, ub=10)
    mb = Model("B")
    pb = mb.parameter("pb", value=2.0)

    ma.minimize(pb * xa)
    with pytest.raises(ValueError, match="[Cc]ross-model"):
        ma.solve()


def test_same_index_different_object_is_rejected():
    """Two models with an identically-indexed but *different* var must not alias.

    Both models declare a first variable (index 0). Referencing model B's var in
    model A must raise, not silently address A's index-0 slot.
    """
    ma = Model("A")
    ma.continuous("xa", lb=0, ub=10)
    mb = Model("B")
    xb = mb.continuous("xb", lb=0, ub=10)
    assert xb._index == 0  # same flat index as A's xa

    ma.minimize(xb)
    with pytest.raises(ValueError, match="[Cc]ross-model"):
        ma.solve()


# ─────────────────────── negative: must NOT raise ───────────────────────────


def test_single_model_with_constants_and_scalars_ok():
    """Scalars and numeric constants must never trip the ownership guard."""
    m = Model("legit")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(3.0 * x + 2 - 1.5)  # python scalars / constants
    m.subject_to(x >= 1.0)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(3.0 * 1.0 + 0.5, abs=1e-5)


def test_single_model_with_numpy_broadcast_ok():
    """numpy-array constants, broadcasts, and matmul on ONE model are fine."""
    m = Model("legit2")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    coef = np.array([1.0, 2.0])
    m.minimize(dm.sum(x) + coef @ x + 3.0)
    m.subject_to(x >= 1.0)
    r = m.solve()
    assert r.status == "optimal"
    # x = [1, 1]: sum=2, coef@x=3, const=3 -> 8
    assert r.objective == pytest.approx(8.0, abs=1e-5)


def test_single_model_with_parameter_ok():
    """A parameter and variable from the SAME model must not raise."""
    m = Model("legit3")
    x = m.continuous("x", lb=0, ub=10)
    p = m.parameter("p", value=2.0)
    m.minimize(p * x)
    m.subject_to(x >= 1.0)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0, abs=1e-5)


def test_same_model_two_variables_ok():
    """Two variables from the same model combine freely."""
    m = Model("legit4")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.minimize(a + b)
    m.subject_to(a + b >= 3.0)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(3.0, abs=1e-5)


def test_gdp_reformulation_shared_variables_not_false_positive():
    """A GDP model whose reformulation builds a *new* Model sharing the original
    Variable objects (and adding aux binaries) must pass the ownership check.

    ``reformulate_gdp`` creates ``Model(model.name)`` and sets
    ``new_model._variables = list(model._variables)`` — different Model object,
    same shared, index-compatible Variable objects — then combines them with new
    aux vars. An ``owner is self`` check would false-positive here; the index-
    slot identity check must accept it.
    """
    m = dm.Model("gdp_share")
    Y = m.boolean("y", shape=(3,))
    m.logical(Y[0].implies(Y[1] & ~Y[2]))
    m.subject_to(Y[0].variable == 1)
    m.minimize(Y[1].variable + Y[2].variable)
    r = m.solve(time_limit=30)
    # Must reach a real result, not a cross-model ValueError.
    assert r.status in ("optimal", "feasible")


def test_validate_directly_accepts_index_shared_rebuild():
    """Directly exercise the index-slot identity path: a second model sharing the
    first's variable list (same objects, same order) validates clean."""
    m = Model("orig")
    x = m.continuous("x", lb=0, ub=5)
    m.minimize(x)
    m.subject_to(x >= 2.0)

    rebuilt = Model("orig")
    rebuilt._variables = list(m._variables)  # shared objects, index-compatible
    rebuilt._parameters = list(m._parameters)
    rebuilt._objective = m._objective
    rebuilt._constraints = list(m._constraints)
    # x.model is still the original `m`, but x._index addresses x in `rebuilt`.
    rebuilt.validate()  # must not raise
