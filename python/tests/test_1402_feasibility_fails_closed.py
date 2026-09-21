"""``check_feasibility`` must never report a point it did not check (#1402).

``warm_start.check_feasibility`` is the *independent* feasibility verifier behind
a reported incumbent — it is what ``discopt_benchmarks/tests/test_correctness.py``
calls to "catch a solver that believes an infeasible point". It returned
``(True, [])`` for points it never evaluated, by two independent mechanisms:

* **A — the swallowed exception.** The constraint arm was wrapped in
  ``except Exception: logger.debug(...)``, and the function then returned
  ``len(violations) == 0``. An evaluator error left ``violations`` holding only
  what the bounds loop found, so it read as a clean bill of health.
* **B — NaN fails open.** Every violation test is a *strict* comparison, and every
  strict comparison against NaN is ``False``. An all-NaN point passed the bounds
  loop (``vals < lb - tol``, ``vals > ub + tol``) and all three sense branches
  (``val > tol``, ``abs(val) > tol``, ``val < -tol``) with no exception raised.

The class pinned here is *fail-closed*: every way of not reaching a verdict must
yield ``False`` plus a violation saying why. The control arm is not optional —
without it these tests would all pass against a function hardwired to ``False``,
which is CLAUDE.md §6 (an instrument that grades nothing).
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import modeling as dm
from discopt.warm_start import check_feasibility


def _infeasible_box_model():
    """``x + y >= 10`` with ``x, y in [0, 1]`` — infeasible at every point in the box.

    Chosen so that *any* honest verdict on an in-box point is "infeasible": a
    ``True`` from this model is unambiguous evidence the row was not checked,
    with no tolerance argument available.
    """
    m = dm.Model("f1402_infeasible")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x + y >= 10.0)
    m.minimize(x + y)
    return m


def _feasible_model():
    """``x + y <= 3`` with ``x, y in [0, 2]`` — the control: a real ``True`` exists."""
    m = dm.Model("f1402_feasible")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x + y <= 3.0)
    m.minimize(x + y)
    return m


def test_control_a_genuinely_feasible_point_is_still_accepted():
    """Guard the other direction: fail-closed must not mean always-closed."""
    ok, viols = check_feasibility(_feasible_model(), np.array([1.0, 1.0]))
    assert ok, f"a feasible point was rejected, so the fix over-closed: {viols}"
    assert viols == []


def test_control_a_violated_row_is_still_reported():
    ok, viols = check_feasibility(_infeasible_box_model(), np.array([0.0, 0.0]))
    assert not ok
    assert any("constraint" in v.lower() for v in viols), viols


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(np.array([np.nan, np.nan]), id="all-nan"),
        pytest.param(np.array([np.nan, 0.5]), id="one-nan"),
        pytest.param(np.array([np.inf, 0.5]), id="plus-inf"),
        pytest.param(np.array([-np.inf, 0.5]), id="minus-inf"),
    ],
)
def test_b_a_non_finite_point_is_never_reported_feasible(bad):
    """Mechanism B: a point carrying no usable value cannot be 'feasible'."""
    ok, viols = check_feasibility(_infeasible_box_model(), bad)
    assert not ok, (
        f"the non-finite point {bad!r} was reported FEASIBLE for a model that is "
        "infeasible everywhere in its box — every strict comparison against NaN "
        "is False, so the verdict was reached without checking anything"
    )
    assert viols, "reported infeasible but named no reason"


@pytest.mark.parametrize("n", [1, 3])
def test_a_a_wrong_length_vector_is_never_reported_feasible(n):
    """Mechanism A's original trigger: the evaluator raises on a length mismatch."""
    ok, viols = check_feasibility(_infeasible_box_model(), np.zeros(n))
    assert not ok, (
        f"a length-{n} vector for a 2-variable model was reported FEASIBLE; the "
        "evaluator raises ValueError here and the exception was swallowed"
    )
    assert any("entr" in v for v in viols), viols


def test_a_an_evaluator_that_raises_yields_a_refusal_not_a_pass(monkeypatch):
    """Mechanism A in general: ANY evaluator error must fail closed.

    Patched at the module ``check_feasibility`` imports from, so this exercises
    the ``except`` arm itself rather than the length gate that now precedes it —
    otherwise the arm would be untested and could silently regress.
    """
    import discopt._tape_nlp_evaluator as tne

    def boom(*_a, **_k):
        raise RuntimeError("injected evaluator failure (#1402)")

    monkeypatch.setattr(tne, "make_evaluator", boom)
    # A point that satisfies its BOUNDS, so the bounds loop contributes nothing
    # and only the constraint arm can produce a verdict.
    ok, viols = check_feasibility(_infeasible_box_model(), np.array([0.5, 0.5]))
    assert not ok, (
        "an evaluator that raised was reported as FEASIBLE: the bounds loop found "
        "nothing and the swallowed exception left violations empty"
    )
    assert any("could not be evaluated" in v for v in viols), viols
    assert any("injected evaluator failure" in v for v in viols), (
        f"the refusal must name the underlying error so a gate failure is diagnosable: {viols}"
    )


def test_b_a_nan_constraint_body_is_reported(monkeypatch):
    """Mechanism B inside the constraint arm, with a finite point.

    The finiteness gate on ``x_flat`` cannot catch this: the point is finite and
    the *body* evaluates to NaN (e.g. ``0/0`` or ``log`` of a negative at an
    in-bounds point). Patched to isolate that branch.
    """
    import discopt._tape_nlp_evaluator as tne

    real = tne.make_evaluator

    class _NanEval:
        def __init__(self, inner):
            self._inner = inner
            self.n_constraints = inner.n_constraints

        def evaluate_constraints(self, x):
            out = np.asarray(self._inner.evaluate_constraints(x), dtype=float).copy()
            out[0] = np.nan
            return out

        def __getattr__(self, k):
            return getattr(self._inner, k)

    monkeypatch.setattr(tne, "make_evaluator", lambda m, *a, **k: _NanEval(real(m, *a, **k)))
    ok, viols = check_feasibility(_infeasible_box_model(), np.array([0.5, 0.5]))
    assert not ok, (
        "a constraint whose body evaluated to NaN was treated as satisfied: "
        "`val > tol`, `abs(val) > tol` and `val < -tol` are all False for NaN"
    )
    assert any("cannot be checked" in v for v in viols), viols


def test_the_verifier_grades_every_arm():
    """CLAUDE.md §6: prove the arms above are not all skipping.

    Counts verdicts directly rather than trusting collection: each entry is a
    (point, expected-ok) pair on the infeasible-box model, plus the control.
    """
    m = _infeasible_box_model()
    cases = [
        (np.array([0.0, 0.0]), False),
        (np.array([np.nan, np.nan]), False),
        (np.array([np.inf, 0.5]), False),
        (np.zeros(1), False),
        (np.zeros(3), False),
    ]
    graded = 0
    for point, expected in cases:
        ok, _ = check_feasibility(m, point)
        assert ok is expected, f"{point!r}: expected ok={expected}, got {ok}"
        graded += 1
    ok, _ = check_feasibility(_feasible_model(), np.array([1.0, 1.0]))
    assert ok is True
    graded += 1
    assert graded == 6, f"graded {graded} verdicts, expected 6"
