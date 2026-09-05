"""#1180 — the evaluator callback path's marshaling contract.

The change these tests pin is bound-neutral by construction: it hands pounce a
contiguous ``float64`` array where the old code rebuilt a Python list per
callback, and it replaces a generator-based ``_timing.charge`` with a
``__slots__`` context-manager class. Both compute identical numbers, and that is
exactly what has to stay true — a "cheaper" marshaling that quietly narrows the
iterate (``float32``, a copy that drops a dimension, a ravel that reorders) would
change every derivative in the solver with no visible symptom.
"""

from __future__ import annotations

import numpy as np
import pytest

from discopt import _timing

pytestmark = pytest.mark.smoke


def _tape_evaluator(model):
    from discopt._tape_nlp_evaluator import try_build

    ev = try_build(model)
    if ev is None:
        pytest.skip("model is not tape-representable")
    return ev


def _small_model():
    import discopt.modeling as dm

    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.minimize(x * y + dm.log(x) + y**3)
    m.constraint(x + y >= 1.5)
    m.constraint(x * y <= 4.0)
    return m


def test_x_marshals_to_a_contiguous_float64_array():
    ev = _tape_evaluator(_small_model())
    out = ev._x(np.array([1.25, 2.5]))
    assert isinstance(out, np.ndarray), "the iterate must reach pounce as an array"
    assert out.dtype == np.float64, "a narrowed dtype silently changes every derivative"
    assert out.flags["C_CONTIGUOUS"]
    assert out.shape == (2,)


def test_x_does_not_copy_an_already_contiguous_float64_vector():
    """The whole point of the change: no per-callback O(n) work on the hot path."""
    ev = _tape_evaluator(_small_model())
    x = np.array([1.25, 2.5], dtype=np.float64)
    assert ev._x(x) is x or np.shares_memory(ev._x(x), x)


@pytest.mark.parametrize("raw", [[1.25, 2.5], (1.25, 2.5), np.array([[1.25], [2.5]])])
def test_x_accepts_the_shapes_callers_actually_pass(raw):
    ev = _tape_evaluator(_small_model())
    out = ev._x(raw)
    assert out.dtype == np.float64
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, np.array([1.25, 2.5]))


def test_array_marshaling_is_bit_identical_to_the_old_list_path():
    """The regression this file exists for: same numbers, every entry point."""
    ev = _tape_evaluator(_small_model())
    problem = ev._problem
    rng = np.random.RandomState(0)
    lb, ub = ev.variable_bounds
    n_checked = 0
    for _ in range(5):
        x = lb + (ub - lb) * rng.uniform(size=lb.shape[0])
        as_list = [float(v) for v in np.asarray(x, dtype=float).ravel()]
        as_array = ev._x(x)
        lam = rng.uniform(size=ev.n_constraints)
        pairs = [
            (problem.objective(as_list), problem.objective(as_array)),
            (problem.gradient(as_list), problem.gradient(as_array)),
            (problem.constraints(as_list), problem.constraints(as_array)),
            (problem.jacobian(as_list), problem.jacobian(as_array)),
            (
                problem.hessian(as_list, lam=[float(v) for v in lam], obj_factor=1.0),
                problem.hessian(as_array, lam=np.asarray(lam, dtype=np.float64), obj_factor=1.0),
            ),
        ]
        for old, new in pairs:
            np.testing.assert_array_equal(
                np.asarray(old, dtype=np.float64).ravel(),
                np.asarray(new, dtype=np.float64).ravel(),
            )
            n_checked += 1
    assert n_checked == 25, "the comparison loop did not run — this test would pass vacuously"


def test_charge_is_not_a_generator_context_manager():
    """Pins the mechanism, because the cost is the whole reason it changed.

    Entered once per derivative callback (1.95 M times over the in-repo corpus at
    20 s each), against a tape evaluation of 0.2-0.6 us underneath it.
    """
    cm = _timing.charge("rust")
    assert type(cm).__name__ == "_Charge"
    assert not hasattr(cm, "gi_frame"), "charge is a generator again; see #1180"
    assert _timing.charge("rust") is not cm, "each call must return a fresh region"


def test_charge_still_records_self_time_after_the_rewrite():
    before = _timing.snapshot()
    with _timing.charge("pounce"):
        _spin(0.02)
        with _timing.charge("rust"):
            _spin(0.02)
    delta = _timing.since(before)
    assert delta["pounce"] >= 0.01
    assert delta["rust"] >= 0.01
    # The parent must not also count the child's wall.
    assert delta["pounce"] < 0.035, f"parent absorbed the child's time: {delta}"


def test_charge_charges_a_region_that_raises():
    before = _timing.snapshot()
    with pytest.raises(RuntimeError):
        with _timing.charge("rust"):
            _spin(0.02)
            raise RuntimeError("boom")
    assert _timing.since(before)["rust"] >= 0.01


def test_charge_rejects_an_unknown_bucket():
    with pytest.raises(ValueError):
        _timing.charge("not-a-bucket")


def _spin(seconds: float) -> None:
    import time

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass
