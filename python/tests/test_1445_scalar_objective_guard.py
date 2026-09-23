"""#1445 -- a non-scalar objective must be refused where it is written.

``Model.minimize`` / ``maximize`` accepted an expression of any shape and
``validate()`` said nothing. The model then failed at solve time with
``jax.grad``'s "Gradient only defined for scalar-output functions", or with a
wall of ``pounce::py ERROR`` lines -- six frames below the caller's own code,
naming a library they never imported.

No route ever produced a false bound (every one failed loudly, and the MINLP
route's ``status="unknown"`` refusal to certify is exactly right), so this is a
legibility fix, not a soundness one. These tests pin both halves: the refusal
happens at the boundary, and every legitimate spelling of a scalar objective
still works.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt.modeling.core import Constant


class TestNonScalarObjectivesAreRefused:
    """The refusal names the call, the shape, and what to do instead."""

    @pytest.mark.parametrize("shape", [(3,), (2, 2), (1, 3), (5,)])
    @pytest.mark.parametrize("sense", ["minimize", "maximize"])
    def test_a_multi_element_objective_raises(self, shape, sense):
        m = Model("t")
        x = m.continuous("x", shape=shape, lb=0.0, ub=1.0)
        with pytest.raises(ValueError) as exc:
            getattr(m, sense)(x)
        msg = str(exc.value)
        assert sense in msg, msg
        assert str(tuple(shape)) in msg, msg
        assert "scalar" in msg, msg

    def test_the_message_points_at_working_reductions(self):
        """Advice that does not run is worse than none -- so run all of it."""
        m = Model("t")
        x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
        with pytest.raises(ValueError) as exc:
            m.minimize(x)
        msg = str(exc.value)
        assert "dm.sum(expr)" in msg and "expr[i]" in msg and "w @ expr" in msg, msg
        assert "discopt.mo" in msg, msg
        # Each suggested reduction must actually produce a scalar objective.
        for reduced in (dm.sum(x), x[1], np.array([1.0, 2.0, 3.0]) @ x):
            m2 = Model("r")
            m2.continuous("x", shape=(3,), lb=0.0, ub=1.0)
            Model.minimize(m2, reduced)  # no raise

    def test_an_expression_not_just_a_bare_variable(self):
        m = Model("t")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        y = m.continuous("y", shape=(2,), lb=-1.0, ub=1.0)
        with pytest.raises(ValueError, match=r"shape \(2,\)"):
            m.minimize(x + y)
        with pytest.raises(ValueError, match=r"shape \(2,\)"):
            m.minimize(x * x)

    def test_it_raises_before_the_objective_is_stored(self):
        """A refused objective must not leave the model half-configured."""
        m = Model("t")
        x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
        m.minimize(dm.sum(x))
        good = m._objective
        with pytest.raises(ValueError):
            m.minimize(x)
        assert m._objective is good, "the refused objective replaced the good one"


class TestEveryScalarSpellingStillWorks:
    """size == 1 is scalar for this purpose; all of these worked before."""

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda m, x: dm.sum(x), id="dm.sum"),
            pytest.param(lambda m, x: x[0], id="element"),
            pytest.param(lambda m, x: np.ones(3) @ x, id="matmul"),
            pytest.param(lambda m, x: Constant(0.0), id="Constant"),
            pytest.param(lambda m, x: 0.0, id="python_float"),
            pytest.param(lambda m, x: x[0] + 2.0 * x[1], id="affine"),
            pytest.param(lambda m, x: x[0] ** 2 - x[0], id="nonlinear"),
        ],
    )
    def test_accepted_and_solvable(self, make):
        m = Model("t")
        x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
        m.minimize(make(m, x))
        r = m.solve(time_limit=60.0)
        assert r.status in ("optimal", "feasible"), r.status
        assert r.objective is not None

    @pytest.mark.parametrize("shape", [(1,), (1, 1)])
    def test_a_size_one_shape_is_scalar(self, shape):
        """shape (1,) and (1, 1) denote one number and already worked."""
        m = Model("t")
        x = m.continuous("x", shape=shape, lb=0.0, ub=1.0)
        m.minimize(x + 0.0)  # no raise
        r = m.solve(time_limit=60.0)
        assert r.status == "optimal"
        assert abs(r.objective - 0.0) < 1e-6

    def test_a_scalar_variable_is_unaffected(self):
        m = Model("t")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.minimize(x * x - x)
        r = m.solve(time_limit=60.0)
        assert r.status == "optimal"
        assert abs(r.objective - (-0.25)) < 1e-5


class TestTheCaseThisWasFoundOn:
    """A DAE collocation state is 2-D; indexing one axis leaves a row."""

    def test_a_collocation_row_objective_is_refused(self):
        from discopt.dae import ContinuousSet, DAEBuilder

        m = Model("dae")
        cs = ContinuousSet("t", bounds=(0.0, 2.0), nfe=4, ncp=3)
        b = DAEBuilder(m, cs)
        b.add_state("x", initial=1.0, bounds=(-5.0, 5.0))
        b.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        b.discretize()
        state = b.get_state("x")
        assert state.shape == (4, 4), state.shape

        with pytest.raises(ValueError, match=r"shape \(4,\)"):
            m.minimize(0 * state[0])  # one index short -- a whole element's row

        m.minimize(0 * state[0, 0])  # the scalar the caller meant
        assert m.solve(time_limit=60.0).status == "optimal"


class TestTheSiblingSpellingIsUnchanged:
    """An object-dtype numpy array was already refused, with its own message."""

    def test_object_array_of_expressions_still_refused(self):
        m = Model("t")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        arr = np.array([x[0], x[1]], dtype=object)
        with pytest.raises(ValueError, match="object-dtype array"):
            m.minimize(arr)
