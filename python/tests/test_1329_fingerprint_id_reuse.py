"""#1329: an evaluator fingerprint must not be keyed on a recyclable ``id()``.

``evaluator_fingerprint`` stored ``id(model._objective)`` and the ids of every
constraint, variable and parameter, and kept no reference to any of them.
``id()`` is unique only among *live* objects, so once an ``Objective`` was
freed a replacement could land on its address and inherit its fingerprint. The
cache then served the **old** objective's evaluator for the new one, and the
solve returned the old objective's optimum, certified, with a matching bound.

The fix is :class:`discopt._evaluator_cache.Fingerprint`, which pins every
object whose id it holds: while a fingerprint is alive none of its objects can
be freed, so equal keys provably mean the same objects.

These tests are written so that they fail on the pre-fix code. Address reuse is
not deterministic, so they do not rely on provoking it once -- they either check
the invariant directly (a fingerprint of dead objects must not match a
fingerprint of live ones) or repeat the churn enough times that CPython's
free-list reuse is a near-certainty.
"""

import gc

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._evaluator_cache import evaluator_fingerprint, solution_state_fingerprint
from discopt._tape_nlp_evaluator import make_evaluator


def _double_well(name="i1329_dw"):
    """Two minima, so serving the wrong objective is visible in the objective."""
    m = dm.Model(name)
    x = m.continuous("x", lb=-5.0, ub=5.0)
    f = (x**2 - 4) ** 2 / 4.0 - x
    return m, x, f


# ── the headline: a recycled id must not produce a false certified optimum ──


@pytest.mark.correctness
def test_replaced_objective_is_not_served_the_old_evaluator():
    """The issue's repro: two ``minimize`` calls in a row on the same expression.

    The second ``Objective`` is created while the first is already unreferenced,
    so it frequently lands on the freed address. Pre-fix this returned the
    *first* objective's optimum (-2.059) as ``optimal``; ``g = f + 20`` cannot
    be less than 17.9 anywhere in the box.
    """
    m, x, f = _double_well()
    m.minimize(f)
    first = m.solve()
    assert first.status == "optimal"

    g = f + 20.0
    m.minimize(g)
    m.minimize(g)  # the second Objective may reuse the first one's freed id
    r = m.solve()

    assert r.objective is not None
    true_opt = first.objective + 20.0
    assert r.objective == pytest.approx(true_opt, abs=1e-4), (
        f"solve reported {r.objective} for f+20 whose optimum is {true_opt}: "
        "a stale evaluator was served for a recycled id"
    )
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6


@pytest.mark.correctness
def test_evaluator_matches_the_current_objective_after_churn():
    """``make_evaluator`` must evaluate the objective the model has *now*.

    Direct form of the same defect, and the one that reproduced 6/6 before the
    fix: the cached evaluator returned ``f(3)`` for a model whose objective is
    ``f + 20``.
    """
    m, x, f = _double_well("i1329_ev")
    m.minimize(f)
    make_evaluator(m)  # populate the cache under the first objective's id

    for shift in (20.0, 7.5, 100.0):
        g = f + shift
        m.minimize(g)
        m.minimize(g)
        got = float(make_evaluator(m).evaluate_objective([3.0]))
        want = (3.0**2 - 4) ** 2 / 4.0 - 3.0 + shift
        assert got == pytest.approx(want, abs=1e-9), (
            f"evaluator gave {got} for an objective whose value at x=3 is {want}"
        )


@pytest.mark.correctness
def test_weighted_sum_loop_never_serves_a_stale_evaluator():
    """The ordinary-use form: a weighted-sum sweep rebinding the objective.

    Pre-fix this hit a stale evaluator on 6-10 of 400 iterations. The loop is
    the churn; the assertion is on every iteration, so a single stale hit fails.
    """
    m = dm.Model("i1329_ws")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    f1 = (x - 1.0) ** 2
    f2 = (x + 1.0) ** 2

    checked = 0
    for i in range(400):
        w = i / 399.0
        m.minimize(w * f1 + (1.0 - w) * f2)
        got = float(make_evaluator(m).evaluate_objective([0.5]))
        want = w * (0.5 - 1.0) ** 2 + (1.0 - w) * (0.5 + 1.0) ** 2
        assert got == pytest.approx(want, abs=1e-9), f"stale evaluator at iteration {i}"
        checked += 1
    assert checked == 400, "the probe must actually have compared 400 evaluations"


# ── the invariant itself, independent of whether an address happens to recur ──


@pytest.mark.unit
def test_fingerprint_pins_the_objects_it_identifies():
    """A fingerprint must keep its objective/constraint/variable objects alive.

    This is what makes "equal ids" mean "same objects": a pinned object cannot
    be freed, so nothing else can take its address.
    """
    m, x, f = _double_well("i1329_pin")
    m.subject_to(x >= -4.0)
    c = m._constraints[-1]
    m.minimize(f)
    obj = m._objective

    from discopt._evaluator_cache import Fingerprint

    fp = evaluator_fingerprint(m)
    assert isinstance(fp, Fingerprint)

    import weakref

    refs = [weakref.ref(obj), weakref.ref(c), weakref.ref(x)]
    del obj, c, x
    m.minimize(f + 1.0)  # drop the model's own reference to the old objective
    m._constraints.clear()
    m._variables.clear()
    gc.collect()

    alive = [r for r in refs if r() is not None]
    assert len(alive) == 3, "the fingerprint must pin every object whose id it holds"
    assert fp.key[0] == id(alive[0]())


@pytest.mark.unit
def test_fingerprint_equality_is_by_key_and_hashable():
    """It has to drop in where the bare tuple went: dict key and ``==``."""
    m, x, f = _double_well("i1329_eq")
    m.minimize(f)
    a = evaluator_fingerprint(m)
    b = evaluator_fingerprint(m)
    assert a == b and hash(a) == hash(b)
    assert {a: 1}[b] == 1

    m.minimize(f + 1.0)
    assert evaluator_fingerprint(m) != a


@pytest.mark.unit
def test_cache_entries_keep_their_key_objects_alive():
    """The LRU's own keys are what pin the ids it later compares against."""
    m, x, f = _double_well("i1329_lru")
    m.minimize(f)
    make_evaluator(m)
    from discopt._evaluator_cache import Fingerprint

    cache = m._tape_evaluator_cache
    assert len(cache) >= 1
    for key in cache:
        assert isinstance(key, Fingerprint), "a bare-tuple key pins nothing"


# ── the cosmetic half: signed zero must not read as a different problem ──


@pytest.mark.unit
def test_signed_zero_is_not_a_different_problem():
    """``-0.0`` and ``0.0`` are the same bound, so the fingerprint must agree."""
    m = dm.Model("i1329_sz")
    x = m.continuous("x", lb=0.0, ub=3.0)
    p = m.parameter("p", value=0.0)
    m.minimize((x - 1.0) ** 2 + p * x)

    base = solution_state_fingerprint(m)
    x.lb = -0.0
    assert solution_state_fingerprint(m) == base, "-0.0 lower bound read as a new problem"
    p.value = -0.0
    assert solution_state_fingerprint(m) == base, "-0.0 parameter read as a new problem"

    x.lb = 0.5
    assert solution_state_fingerprint(m) != base, "a real bound change must still register"


@pytest.mark.unit
def test_signed_zero_does_not_invalidate_a_sensitivity_reference():
    """The user-visible half of the same defect (#1322's guard, #1329's report)."""
    m = dm.Model("i1329_sens")
    p = m.parameter("p", value=1.0)
    x = m.continuous("x", lb=0.0, ub=3.0)
    m.minimize((x - p) ** 2)
    m.solve()

    x.lb = -0.0
    s = m.sensitivity()
    assert getattr(s, "matches_reference", True) is not False, (
        "flipping a 0.0 bound to -0.0 must not report a DIFFERENT problem"
    )


@pytest.mark.unit
def test_value_bytes_leaves_everything_but_signed_zero_alone():
    """The normalisation must be exactly the ``-0.0`` fold, nothing else."""
    from discopt._evaluator_cache import _value_bytes

    assert _value_bytes(-0.0) == _value_bytes(0.0)
    assert _value_bytes(np.nan) == _value_bytes(np.nan)
    assert _value_bytes(np.inf) != _value_bytes(-np.inf)
    assert _value_bytes(1.0) != _value_bytes(-1.0)
    assert _value_bytes([1.0, -0.0]) == _value_bytes([1.0, 0.0])
    assert _value_bytes([1.0, 2.0]) != _value_bytes([2.0, 1.0])
