"""``Model.sensitivity()`` -- the unified derivative entry point.

What is pinned here, and why each pin exists:

**The chain rule counts both halves.** ``d(expr)`` is a *total* derivative through
the solution map, ``∂e/∂p + (∂e/∂x)·dx*/dp``. Dropping the direct ``∂e/∂p`` term
is the classic wrong answer and is invisible whenever the expression happens not to
mention a parameter, so the expressions below deliberately mention one. Every
derivative assertion compares against a closed form or a central difference of a
**re-solved** problem, never against another code path alone.

**The active set is part of the answer.** A binding constraint puts the solution on
a different smooth branch with a different derivative; both branches are checked on
the same model.

**The entry point must not move the model.** The forward solve writes trial values
into ``Parameter.value``; a sensitivity that left them there would silently change
what every later solve in the session means.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.modeling import Model
from discopt.sensitivity import Sensitivity, sensitivity

pytestmark = pytest.mark.requires_pounce


def _separable(a_value: float = 2.0, b_value: float = 3.0):
    """``min (x-a)² + (y0-b)² + (y1-ab)²  s.t.  x + y0 >= 1``.

    Unconstrained at ``a=2, b=3`` (the row is slack), so the closed form is
    ``x*=a, y0*=b, y1*=ab`` and every derivative is known exactly.
    """
    m = Model("separable")
    a = m.parameter("a", value=a_value)
    b = m.parameter("b", value=b_value)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", shape=(2,), lb=-10.0, ub=10.0)
    m.minimize((x - a) ** 2 + (y[0] - b) ** 2 + (y[1] - a * b) ** 2)
    m.subject_to(x + y[0] >= 1.0)
    return m, a, b, x, y


def _binding(c_value: float):
    """``min x² + y²  s.t.  x + y >= c``: for c>0 the row binds, x*=y*=c/2."""
    m = Model("binding")
    c = m.parameter("c", value=c_value)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.minimize(x * x + y * y)
    m.subject_to(x + y >= c)
    return m, c, x, y


# ─────────────────────────────────────────────────────────────
# dx*/dp against closed forms
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_dx_dp_matches_closed_form_on_the_slack_branch():
    m, a, b, x, y = _separable()
    s = m.sensitivity()

    assert isinstance(s, Sensitivity)
    assert s.param_names == ["a", "b"]
    assert s.dx_dp.shape == (3, 2)
    # x*=a, y0*=b, y1*=a·b  ->  rows [1,0], [0,1], [b,a] = [3,2]
    assert np.allclose(s.dx_dp, [[1.0, 0.0], [0.0, 1.0], [3.0, 2.0]], atol=1e-5)
    assert not s.active.any(), "the inequality is slack at a=2, b=3"


@pytest.mark.slow
def test_dx_dp_follows_the_active_branch_when_the_constraint_binds():
    """Same model, both branches: the derivative is a property of the active set."""
    m, c, x, y = _binding(2.0)
    s = m.sensitivity()
    # binding: x* = y* = c/2  ->  dx/dc = 0.5
    assert bool(s.active[0]) is True
    assert float(s.d(x, c)) == pytest.approx(0.5, abs=1e-5)

    m2, c2, x2, y2 = _binding(-2.0)
    s2 = m2.sensitivity()
    # slack: x* = y* = 0 regardless of c  ->  dx/dc = 0
    assert bool(s2.active[0]) is False
    assert float(s2.d(x2, c2)) == pytest.approx(0.0, abs=1e-5)


# ─────────────────────────────────────────────────────────────
# d(): variables, expressions, selection
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_d_on_a_variable_keeps_the_variable_shape():
    m, a, b, x, y = _separable()
    s = m.sensitivity()

    assert s.d(y).shape == (2, 2)  # (*y.shape, n_params)
    assert np.allclose(s.d(y), [[0.0, 1.0], [3.0, 2.0]], atol=1e-5)
    assert np.allclose(s.d("y"), s.d(y)), "name and object must agree"
    assert np.allclose(s["y"], s.d(y)), "__getitem__ is d()"

    # naming one parameter drops the trailing axis
    assert s.d(y, b).shape == (2,)
    assert np.allclose(s.d(y, b), [1.0, 2.0], atol=1e-5)
    assert float(s.d(x, a)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_d_on_an_expression_is_the_total_derivative_not_the_partial():
    """The direct ``∂e/∂p`` term is counted, checked against a re-solved FD.

    ``e = a·x + y1`` has both halves: ``∂e/∂a = x*`` directly, and ``x*`` and
    ``y1*`` both move with ``a``. Using only the chain term, or only the partial,
    gives a different number -- both wrong answers are asserted against.
    """
    m, a, b, x, y = _separable()
    s = m.sensitivity()
    expr = a * x + y[1]

    got = s.d(expr)

    # closed form: e(a,b) = a·a + a·b -> de/da = 2a + b = 7, de/db = a = 2
    assert np.allclose(got, [7.0, 2.0], rtol=1e-5)

    # the two ways of being wrong
    partial_only = np.array([2.0, 0.0])  # ∂e/∂p at frozen x = x* = 2
    chain_only = np.array([5.0, 2.0])  # (∂e/∂x)·dx/dp, dropping ∂e/∂p
    assert not np.allclose(got, partial_only, atol=1e-3)
    assert not np.allclose(got, chain_only, atol=1e-3)

    # and an independent central difference of a genuine re-solve
    def optimal_value(a_val, b_val):
        mm, aa, bb, xx, yy = _separable(a_val, b_val)
        r = mm.solve()
        return a_val * float(r.x["x"]) + float(np.asarray(r.x["y"])[1])

    h = 1e-5
    fd = np.array(
        [
            (optimal_value(2.0 + h, 3.0) - optimal_value(2.0 - h, 3.0)) / (2 * h),
            (optimal_value(2.0, 3.0 + h) - optimal_value(2.0, 3.0 - h)) / (2 * h),
        ]
    )
    assert np.allclose(got, fd, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_d_on_an_expression_across_a_binding_constraint():
    """The chain rule must use the *active-branch* dx/dp, not the slack one."""
    m, c, x, y = _binding(2.0)
    s = m.sensitivity()
    # e = c·x, with x* = c/2 -> e = c²/2 -> de/dc = c = 2
    assert float(s.d(c * x, c)) == pytest.approx(2.0, abs=1e-4)
    assert s.d(c * x).shape == (1,), "without `wrt` the parameter axis is kept"


@pytest.mark.slow
def test_dobj_dp_agrees_with_the_envelope_theorem_path():
    """The new total-derivative route and the old ``SolveResult.gradient`` must agree.

    They are different derivations -- chain rule through ``dx*/dp`` versus
    ``∂L/∂p`` at the duals -- and coincide only because stationarity kills the
    indirect term. If they ever diverge, one of them is wrong.
    """
    m, c, x, y = _binding(2.0)
    s = m.sensitivity()
    # obj* = c²/2 -> dobj/dc = c = 2
    assert float(s.dobj_dp[0]) == pytest.approx(2.0, abs=1e-4)

    result = m.solve(sensitivity=True)
    assert float(result.gradient(c)) == pytest.approx(float(s.dobj_dp[0]), abs=1e-4)


# ─────────────────────────────────────────────────────────────
# prediction, second order, composition
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_predict_is_the_linearisation_and_accepts_a_partial_mapping():
    m, a, b, x, y = _separable()
    s = m.sensitivity()

    pred = s.predict({a: 2.1})  # b left alone
    assert float(pred["x"]) == pytest.approx(2.1, abs=1e-5)
    assert np.allclose(np.asarray(pred["y"]), [3.0, 6.0 + 0.1 * 3.0], atol=1e-5)
    assert set(pred) == {"x", "y"}
    assert np.allclose(s.predict_flat([2.1, 3.0]), s.predict_flat({a: 2.1}))


@pytest.mark.slow
def test_second_order_prediction_beats_first_order_on_a_curved_solution_map():
    """``y1* = a·b`` is exactly quadratic in (a,b), so order 2 must be exact."""
    m, a, b, x, y = _separable()
    s = m.sensitivity(order=2)
    assert s.d2x_dp2 is not None and s.d2x_dp2.shape == (3, 2, 2)
    assert np.allclose(s.d2x_dp2[2], [[0.0, 1.0], [1.0, 0.0]], atol=1e-4)

    first = s.predict({a: 2.5, b: 3.5}, order=1)
    second = s.predict({a: 2.5, b: 3.5}, order=2)
    exact = 2.5 * 3.5
    assert abs(float(np.asarray(second["y"])[1]) - exact) < 1e-6
    assert abs(float(np.asarray(first["y"])[1]) - exact) > 1e-3


@pytest.mark.slow
def test_order_2_prediction_refused_without_order_2_data():
    m, a, b, x, y = _separable()
    s = m.sensitivity()
    with pytest.raises(ValueError, match="order=2"):
        s.predict({a: 2.1}, order=2)


@pytest.mark.slow
def test_as_layer_reproduces_dx_dp():
    import jax
    import jax.numpy as jnp

    m, a, b, x, y = _separable()
    s = m.sensitivity()
    phi = s.as_layer()
    jac = np.asarray(jax.jacobian(phi)(jnp.array([2.0, 3.0])))
    assert np.allclose(jac, s.dx_dp, atol=1e-6)


@pytest.mark.slow
def test_fd_method_cross_checks_the_exact_method():
    m, c, x, y = _binding(2.0)
    exact = m.sensitivity(method="exact")
    fd = m.sensitivity(method="fd")
    assert np.allclose(exact.dx_dp, fd.dx_dp, atol=1e-5)


# ─────────────────────────────────────────────────────────────
# contract: the call must not move the model
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_parameter_values_are_restored():
    m, a, b, x, y = _separable()
    before = (float(a.value), float(b.value))
    m.sensitivity()
    assert (float(a.value), float(b.value)) == before


@pytest.mark.slow
def test_parameter_values_are_restored_even_when_the_solve_raises(monkeypatch):
    """A failure partway through must not leave the model describing another problem."""
    m, a, b, x, y = _separable()
    before = (float(a.value), float(b.value))

    import discopt.solvers.sipopt as sipopt_mod

    def boom(model, parameters, **kwargs):
        for p in parameters:
            p.value = np.asarray(99.0)
        raise RuntimeError("solve exploded")

    monkeypatch.setattr(sipopt_mod, "pounce_sensitivity", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        sensitivity(m)
    assert (float(a.value), float(b.value)) == before


# ─────────────────────────────────────────────────────────────
# refusals
# ─────────────────────────────────────────────────────────────


def test_model_without_parameters_is_refused():
    m = Model("noparams")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x * x)
    with pytest.raises(ValueError, match="no Parameters"):
        m.sensitivity()


def test_non_scalar_parameter_is_refused_by_name():
    m = Model("vec")
    m.parameter("p", value=np.array([1.0, 2.0]))
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x * x)
    with pytest.raises(ValueError, match="scalar Parameters only"):
        m.sensitivity()


def test_unknown_parameter_name_is_refused():
    m, a, b, x, y = _separable()
    with pytest.raises(KeyError, match="nope"):
        m.sensitivity(wrt="nope")


def test_parameter_from_another_model_is_refused():
    m, a, b, x, y = _separable()
    other = Model("other")
    q = other.parameter("q", value=1.0)
    with pytest.raises(ValueError, match="does not belong"):
        m.sensitivity(wrt=[q])


def test_duplicate_parameter_is_refused():
    m, a, b, x, y = _separable()
    with pytest.raises(ValueError, match="more than once"):
        m.sensitivity(wrt=[a, a])


def test_wrt_rejects_a_variable():
    m, a, b, x, y = _separable()
    with pytest.raises(TypeError, match="dm.Parameter"):
        m.sensitivity(wrt=[x])


@pytest.mark.slow
def test_integer_model_is_refused():
    m = Model("intmodel")
    p = m.parameter("p", value=1.0)
    x = m.continuous("x", lb=0.0, ub=5.0)
    m.integer("k", lb=0, ub=3)
    m.minimize((x - p) ** 2)
    with pytest.raises(ValueError, match="integer/binary"):
        m.sensitivity()


@pytest.mark.slow
def test_d_rejects_an_unknown_variable_name_and_a_non_expression():
    m, a, b, x, y = _separable()
    s = m.sensitivity()
    with pytest.raises(KeyError, match="zzz"):
        s.d("zzz")
    with pytest.raises(TypeError, match="Variable"):
        s.d(3.0)


@pytest.mark.slow
def test_d_rejects_a_parameter_the_sensitivity_was_not_computed_for():
    m, a, b, x, y = _separable()
    s = m.sensitivity(wrt=[a])
    with pytest.raises(ValueError, match="not included"):
        s.d(x, b)


@pytest.mark.slow
def test_summary_names_every_scalar_entry():
    m, a, b, x, y = _separable()
    text = m.sensitivity().summary()
    assert "y[0]" in text and "y[1]" in text and "x" in text
    assert "a" in text and "b" in text


# ─────────────────────────────────────────────────────────────
# solve(sensitivity=True) is no longer a dead flag
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_solve_sensitivity_flag_populates_the_result():
    m, c, x, y = _binding(2.0)
    lazy = m.solve()
    assert lazy._sensitivity is None, "plain solve() must not pay for derivatives"

    eager = m.solve(sensitivity=True)
    assert eager._sensitivity is not None, "solve(sensitivity=True) must do something"
    assert float(eager.gradient(c)) == pytest.approx(2.0, abs=1e-4)


@pytest.mark.slow
def test_solve_sensitivity_flag_refuses_an_unsupported_model_at_solve_time():
    """The flag surfaces the refusal where it was asked for, not at a later call."""
    m = Model("intflag")
    m.parameter("p", value=1.0)
    x = m.continuous("x", lb=0.0, ub=5.0)
    m.integer("k", lb=0, ub=3)
    m.minimize(x * x + 1.0 * 1)
    with pytest.raises(ValueError, match="only supports continuous"):
        m.solve(sensitivity=True)


@pytest.mark.slow
def test_predict_measures_from_the_expansion_point_not_the_live_value():
    """``p0`` is pinned, so editing the model afterwards cannot reinterpret a prediction."""
    m, a, b, x, y = _separable()
    s = m.sensitivity()
    expected = s.predict({a: 2.1})

    a.value = np.asarray(7.0)  # someone else reuses the model
    assert np.allclose(s.p0, [2.0, 3.0])
    got = s.predict({a: 2.1})
    assert np.allclose(np.asarray(got["x"]), np.asarray(expected["x"]))
    assert float(got["x"]) == pytest.approx(2.1, abs=1e-5)


@pytest.mark.slow
def test_d_evaluates_partials_at_the_solve_point_not_at_live_values():
    """``d()`` on an expression must not mix a stale ``dx_dp`` with fresh partials."""
    m, a, b, x, y = _separable()
    s = m.sensitivity()
    expr = a * x + y[1]
    before = s.d(expr).copy()

    a.value = np.asarray(99.0)  # the model moves on; this Sensitivity does not
    assert np.allclose(s.p_all, [2.0, 3.0])
    assert np.allclose(s.d(expr), before), "partials were re-evaluated at the new value"
    assert np.allclose(before, [7.0, 2.0], rtol=1e-5)


@pytest.mark.slow
def test_sensitivity_objects_use_identity_equality():
    """The generated dataclass __eq__/__hash__ would raise on the numpy fields."""
    m, a, b, x, y = _separable()
    s = m.sensitivity()
    assert s == s
    assert s != m.sensitivity()  # a distinct object, even with identical numbers
    assert isinstance(hash(s), int)
