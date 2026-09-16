"""Re-solving one model across ``Parameter`` value changes (#1245).

The plugin pattern behind the issue (discopt-calphad, #1249) is a *loop*: one
small global model solved once per phase, per (T, x) point of a phase-diagram
trace, or per data condition of a fitting round, with only the chemical
potentials and the temperature — both ``Parameter``s — moving between solves.
The ask was to stop rebuilding compiled relaxations on every such solve.

What the measurement found (recorded in
``docs/dev/performance-plan.md`` §65, with the probes it came from):

* the relaxation *compiler* the issue names (``_relax/relaxation_compiler.py``)
  is not on the default solve path at all — 0 calls across 3 solves;
* the box-independent relaxation analysis that IS rebuilt per solve costs
  4-208 ms depending on model size, i.e. 0.4-3.6% of these solves, and only
  ~35% of it is soundly reusable across a value change (the rest is either the
  canonicalization that embeds the new values or verdicts that depend on the
  declared bounds);
* a solve with a parameter change was not measurably slower than the same solve
  without one (1.716 +- 0.051 s vs 1.820 +- 0.023 s, 8 interleaved reps);
* what a parameter *did* cost on every solve was the tape evaluator's staleness
  check, which ran in front of every objective/gradient/constraint/Jacobian/
  Hessian evaluation and copied every parameter array on each call: 11.5 us per
  call, 4.5% of the wall of a parameterised solve.

So the reuse the issue asked for already exists where it is worth having (the
analysis is built once per solve and shared by every B&B node; the tape is
rebuilt once per parameter change, not once per evaluation) and the fix was to
make the *check* that guards that reuse cheap. These tests pin both halves —
the reuse behaviour and the soundness that makes rebuilding necessary at all.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._tape_nlp_evaluator import TapeNLPEvaluator, make_evaluator

pytestmark = [pytest.mark.smoke]


def _quadratic_param_model():
    """``min a*x^2`` on ``[0, 5]`` — one scalar parameter, analytic derivatives."""
    m = dm.Model("p1245_quadratic")
    x = m.continuous("x", lb=0, ub=5)
    a = m.parameter("a", 2.0)
    m.minimize(a * x * x)
    return m, x, a


# --------------------------------------------------------------------------- #
# 1. The staleness check still sees every way a value can move
# --------------------------------------------------------------------------- #
def test_scalar_rebind_is_seen():
    m, _x, a = _quadratic_param_model()
    ev = make_evaluator(m)
    assert isinstance(ev, TapeNLPEvaluator)
    xs = np.array([1.5])
    for v in (2.0, 3.0, -1.0, 0.0, 2.0):
        a.value = v
        assert float(ev.evaluate_objective(xs)) == pytest.approx(v * 2.25, abs=1e-12)
        assert float(np.asarray(ev.evaluate_gradient(xs))[0]) == pytest.approx(
            2 * v * 1.5, abs=1e-12
        )


def test_scalar_inplace_mutation_is_seen():
    """``p.value[()] = v`` bypasses the setter; the check must still catch it."""
    m, _x, a = _quadratic_param_model()
    ev = make_evaluator(m)
    xs = np.array([1.5])
    assert float(ev.evaluate_objective(xs)) == pytest.approx(2.0 * 2.25, abs=1e-12)
    a.value[()] = 7.0
    assert float(ev.evaluate_objective(xs)) == pytest.approx(7.0 * 2.25, abs=1e-12)


def test_array_parameter_rebind_and_inplace_are_seen():
    m = dm.Model("p1245_array_param")
    y = m.continuous("y", lb=0, ub=5)
    b = m.parameter("b", np.array([1.0, 2.0]))
    m.minimize(b[0] * y + b[1] * y * y)
    ev = make_evaluator(m)
    ys = np.array([2.0])
    for vals in ([1.0, 2.0], [3.0, 0.5], [0.0, 0.0]):
        b.value = np.array(vals)
        want = vals[0] * 2.0 + vals[1] * 4.0
        assert float(ev.evaluate_objective(ys)) == pytest.approx(want, abs=1e-12)
    b.value[0] = 7.0
    assert float(ev.evaluate_objective(ys)) == pytest.approx(14.0, abs=1e-12)


def test_nan_parameter_still_reads_as_changed():
    """Unchanged semantics: NaN never compares equal to itself, so a NaN-valued
    parameter reports *changed* on every check — as it did before #1245. Pinned so
    the rewrite's exactness is visible rather than assumed."""
    m, _x, a = _quadratic_param_model()
    ev = make_evaluator(m)
    a.value = float("nan")
    ev.evaluate_objective(np.array([1.0]))  # absorbs the rebuild
    assert ev._params_changed() is True


# --------------------------------------------------------------------------- #
# 2. The reuse itself: one rebuild per change, none per evaluation
# --------------------------------------------------------------------------- #
def test_tape_rebuilds_once_per_change_not_per_evaluation(monkeypatch):
    m, _x, a = _quadratic_param_model()
    ev = make_evaluator(m)
    xs = np.array([1.5])
    ev.evaluate_objective(xs)

    builds = {"n": 0}
    orig = TapeNLPEvaluator._build

    def counting_build(self):
        builds["n"] += 1
        return orig(self)

    monkeypatch.setattr(TapeNLPEvaluator, "_build", counting_build)

    for _ in range(50):  # values unchanged -> the compiled tape is reused
        ev.evaluate_objective(xs)
        ev.evaluate_gradient(xs)
    assert builds["n"] == 0, "an unchanged parameter must not rebuild the tape"

    for i in range(5):  # one rebuild per change, not one per evaluation
        a.value = 3.0 + i  # 3.0 .. 7.0; the tape was built at a = 2.0
        for _ in range(10):
            ev.evaluate_objective(xs)
            ev.evaluate_gradient(xs)
    assert builds["n"] == 5, builds["n"]


def test_staleness_check_does_not_snapshot(monkeypatch):
    """The check compares against the retained snapshot in place.

    This is the hot-path property the #1245 fix bought (11.5 us -> 0.80 us per
    call at 5 scalar parameters): a check that rebuilt the snapshot tuple on
    every call cost 4.5% of a parameterised solve's wall. Pinned as a mechanism
    rather than a wall-clock threshold so it cannot pass on a fast machine while
    the copies are back.
    """
    m, _x, a = _quadratic_param_model()
    ev = make_evaluator(m)
    ev.evaluate_objective(np.array([1.5]))

    calls = {"n": 0}
    orig = TapeNLPEvaluator._snapshot_params

    def counting_snapshot(self):
        calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(TapeNLPEvaluator, "_snapshot_params", counting_snapshot)

    for _ in range(25):
        assert ev._params_changed() is False
    assert calls["n"] == 0, "the staleness check must not copy the parameter values"

    a.value = 9.0  # a real change still re-snapshots, exactly once
    ev.evaluate_objective(np.array([1.5]))
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# 3. Acceptance: a re-solve equals a freshly built model, and a parameter that
#    flips curvature still gets a sound relaxation
# --------------------------------------------------------------------------- #
def _bilinear_model(p_value: float):
    m = dm.Model("p1245_bilinear")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    p = m.parameter("p", p_value)
    m.minimize(p * x * y - x - y)
    m.subject_to(x + y <= 1.5)
    return m, p


def test_resolve_after_parameter_change_matches_a_fresh_model():
    """Acceptance criterion 2 of #1245: bounds and solutions of a re-solved model
    match freshly built ones. The tolerance is 1e-12; the paths agree exactly."""
    m, p = _bilinear_model(1.0)
    for value in (1.0, 2.5, 0.25, -1.5, 4.0):
        p.value = value
        reused = m.solve(time_limit=30)
        fresh_model, _fresh_p = _bilinear_model(value)
        fresh = fresh_model.solve(time_limit=30)

        assert reused.status == fresh.status, value
        assert reused.objective == pytest.approx(fresh.objective, abs=1e-12), value
        assert (reused.bound is None) == (fresh.bound is None), value
        if reused.bound is not None:
            assert reused.bound == pytest.approx(fresh.bound, abs=1e-12), value
        for name, val in reused.x.items():
            np.testing.assert_allclose(val, fresh.x[name], atol=1e-12, err_msg=str(value))


def test_curvature_flipping_parameter_stays_sound():
    """Acceptance criterion 3: a parameter multiplying a nonconvex term, whose sign
    flips, still gets a sound relaxation on the SAME model object.

    ``min p*x^2 + x`` on ``[-1, 1]`` is convex for ``p > 0`` (optimum -0.25 at
    x = -0.5) and concave for ``p < 0`` (optimum -2 at the endpoint x = -1). A
    relaxation carried over from the previous value would answer the wrong one of
    those two — the #742 failure class — so the sequence is alternated.
    """
    m = dm.Model("p1245_curvature_flip")
    x = m.continuous("x", lb=-1, ub=1)
    p = m.parameter("p", 1.0)
    m.minimize(p * x * x + x)

    expected = {1.0: (-0.25, -0.5), -1.0: (-2.0, -1.0)}
    checked = 0
    for value in (1.0, -1.0, 1.0, -1.0):
        p.value = value
        r = m.solve(time_limit=30)
        want_obj, want_x = expected[value]
        assert r.status == "optimal", (value, r.status)
        assert r.objective == pytest.approx(want_obj, abs=1e-5), (value, r.objective)
        assert float(np.ravel(r.x["x"])[0]) == pytest.approx(want_x, abs=1e-4), value
        assert r.bound is not None and r.bound <= r.objective + 1e-6, (value, r.bound)
        checked += 1
    assert checked == 4
