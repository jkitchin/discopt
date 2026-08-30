"""#1066: static constraint bounds were re-derived on every evaluation.

``_infer_constraint_bounds`` walks the evaluator's source constraints and
allocates ``2 * n_constraints`` ``np.full`` arrays plus two concatenates.
Nothing about that result changes over the life of a compiled evaluator, yet
``oa._constraint_violation_data`` calls it once per objective *and* once per
gradient evaluation of ``_FeasibilityEvaluator`` -- i.e. at every ipopt
iteration of every feasibility subproblem.

Measured on ``portfol_classical050_1`` (103 constraints, 150 variables) before
the cache: 45,488 calls costing 8.80 s cumulative of a 32.765 s solve (27%),
with 9,370,529 ``np.full`` calls inside them (5.06 s, 15%). Isolated cost of
the rebuild on that evaluator: 111.4 us/call vs 0.3 us cached (389x).

This is a bound-neutral change (CLAUDE.md §5): it must not alter any returned
value, so the tests below pin equality against a freshly rebuilt result as well
as the call reduction.
"""

import numpy as np
import pytest
from discopt.modeling.core import Constraint, Model
from discopt.solvers import nlp_ipopt
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds


def _model():
    m = Model("bounds-cache")
    x = m.continuous("x", shape=(3,), lb=-10.0, ub=10.0)
    m.subject_to(x[0] + x[1] <= 4.0)
    m.subject_to(x[1] - x[2] == 1.0)
    # NOTE: the modeling layer normalises ``>=`` to ``<=`` at construction, so
    # every constraint an NLPEvaluator compiles from a Model is ``<=`` or
    # ``==``. A genuine ``>=`` row only reaches this function from a
    # hand-built Constraint, which the Fake-evaluator tests below cover.
    m.subject_to(x[0] * x[0] + x[2] >= 2.0)
    m.minimize(x[0] + x[1] + x[2])
    return m


def _evaluator(model):
    from discopt._relax.nlp_evaluator import NLPEvaluator

    return NLPEvaluator(model)


def test_repeated_calls_return_the_same_values_as_the_first():
    """The cache must be a pure optimisation: identical values, every time."""
    ev = _evaluator(_model())
    cl0, cu0 = _infer_constraint_bounds(ev)
    checks = 0
    for _ in range(5):
        cl, cu = _infer_constraint_bounds(ev)
        np.testing.assert_array_equal(cl, cl0)
        np.testing.assert_array_equal(cu, cu0)
        checks += 2
    assert checks == 10, "vacuous: no comparison executed"
    # and the values are the documented sense mapping, not just self-consistent
    assert [c.sense for c in ev._source_constraints] == ["<=", "==", "<="]
    assert cl.tolist() == [-1e20, 0.0, -1e20]
    assert cu.tolist() == [0.0, 0.0, 0.0]


def test_the_rebuild_runs_once_per_evaluator_not_once_per_call(monkeypatch):
    """The defect this fixes: N calls did N rebuilds. Counting the per-row
    allocation is what makes the test fail before the change and pass after --
    a timing assertion would be load-dependent (CLAUDE.md §9)."""
    ev = _evaluator(_model())
    calls = [0]
    real_full = np.full

    def counting_full(*args, **kwargs):
        calls[0] += 1
        return real_full(*args, **kwargs)

    monkeypatch.setattr(nlp_ipopt.np, "full", counting_full)

    _infer_constraint_bounds(ev)
    after_first = calls[0]
    assert after_first == 6, f"expected 2 rows x 3 constraints, got {after_first}"

    for _ in range(50):
        _infer_constraint_bounds(ev)
    assert calls[0] == after_first, (
        f"the bounds were rebuilt {calls[0] - after_first} times across 50 "
        "repeat calls; they are static for a compiled evaluator"
    )


def test_callers_get_their_own_arrays_so_a_write_cannot_poison_the_cache():
    """Several call sites do ``np.asarray(...)`` on the result, which does not
    copy. Handing out the cached array itself would let any of them corrupt
    every later caller."""
    ev = _evaluator(_model())
    cl_a, cu_a = _infer_constraint_bounds(ev)
    cl_a[0] = 12345.0
    cu_a[2] = -12345.0
    cl_b, cu_b = _infer_constraint_bounds(ev)
    assert cl_b[0] == -1e20
    assert cu_b[2] == 0.0
    assert cl_b is not cl_a and cu_b is not cu_a


def test_a_model_is_never_cached_because_it_can_gain_constraints():
    """A Model is mutable: caching on it would return stale bounds after a new
    ``subject_to``. Only the compiled-evaluator branch is cached."""
    m = _model()
    cl, cu = _infer_constraint_bounds(m)
    assert cl.size == 3
    x = m._variables[0]
    m.subject_to(x[0] >= 1.0)
    cl2, cu2 = _infer_constraint_bounds(m)
    assert cl2.size == 4, "the Model branch must see the new constraint"
    assert cl2[3] == -1e20


def test_a_rebuilt_constraint_list_misses_the_cache():
    """The fingerprint guards on the identity of the list the bounds came from,
    so an evaluator whose constraints are swapped out re-derives them rather
    than serving the previous evaluator's answer."""
    ev = _evaluator(_model())
    cl0, _ = _infer_constraint_bounds(ev)
    assert cl0.tolist() == [-1e20, 0.0, -1e20]

    flipped = []
    for c in ev._source_constraints:
        flipped.append(Constraint(c.body, "==", c.rhs))
    ev._source_constraints = flipped
    cl1, cu1 = _infer_constraint_bounds(ev)
    assert cl1.tolist() == [0.0, 0.0, 0.0], "stale cache served after a swap"
    assert cu1.tolist() == [0.0, 0.0, 0.0]


def test_an_evaluator_without_a_dict_still_works():
    """``__slots__`` evaluators go uncached rather than raising."""

    class Slotted:
        __slots__ = ("_source_constraints", "_constraint_flat_sizes")

    s = Slotted()
    s._source_constraints = [Constraint(None, "<=", 0.0), Constraint(None, ">=", 0.0)]
    s._constraint_flat_sizes = np.asarray([1, 2], dtype=np.intp)
    cl, cu = _infer_constraint_bounds(s)
    assert cl.tolist() == [-1e20, 0.0, 0.0]
    assert cu.tolist() == [0.0, 1e20, 1e20]
    # calling again must still work (and still not raise)
    cl2, _ = _infer_constraint_bounds(s)
    assert cl2.tolist() == cl.tolist()


def test_vector_valued_bodies_keep_their_row_expansion():
    """The sizes array drives the repeat; a cache keyed only on the constraint
    list would lose it."""

    class Fake:
        pass

    f = Fake()
    f._source_constraints = [Constraint(None, "<=", 0.0), Constraint(None, "==", 0.0)]
    f._constraint_flat_sizes = np.asarray([4, 2], dtype=np.intp)
    cl, cu = _infer_constraint_bounds(f)
    assert cl.size == 6 and cu.size == 6
    assert cl.tolist() == [-1e20] * 4 + [0.0, 0.0]
    assert cu.tolist() == [0.0] * 6


def test_an_unknown_sense_still_raises_loudly_on_a_cached_evaluator():
    """The cache must not turn a later bad sense into a silent hit."""

    class Fake:
        pass

    f = Fake()
    f._source_constraints = [Constraint(None, "<=", 0.0)]
    f._constraint_flat_sizes = np.asarray([1], dtype=np.intp)
    _infer_constraint_bounds(f)
    f._source_constraints = [Constraint(None, "!=", 0.0)]
    with pytest.raises(ValueError, match="Unknown constraint sense"):
        _infer_constraint_bounds(f)
