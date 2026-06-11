"""Tests for ``dm.custom`` / :class:`CustomCall` — opaque AD-only user functions.

A ``dm.custom`` function wraps an arbitrary JAX-traceable callable. discopt can
autodifferentiate it (so the local NLP path works), but it cannot build the
rigorous relaxations / interval rules / ``.nl`` export that global spatial
branch-and-bound needs. These tests pin down both halves of that contract:

* the value/autodiff path works through the DAG compiler and the local solver,
  and the result carries no global certificate (``gap_certified is False``);
* every machinery that would need a relaxation refuses loudly — integer/binary
  variables (solver ``ValueError``), ``.nl`` export (``ValueError``), and
  relaxation compilation (``NotImplementedError``); and ``custom`` itself
  rejects a non-callable argument (``TypeError``).
"""

from __future__ import annotations

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt.modeling import Model
from discopt.modeling.core import CustomCall

# ─────────────────────────────────────────────────────────────
# Node construction / repr
# ─────────────────────────────────────────────────────────────


class TestCustomCallNode:
    def test_custom_returns_builder(self):
        builder = dm.custom(lambda x: jnp.sum(x**2))
        assert callable(builder)

    def test_builder_produces_customcall(self):
        m = Model("c")
        x = m.continuous("x", lb=-5, ub=5)
        node = dm.custom(lambda x: jnp.sum(x**2))(x)
        assert isinstance(node, CustomCall)
        assert node.fn is not None
        assert len(node.args) == 1

    def test_name_defaults_to_fn_name(self):
        def weird(x):
            return jnp.sum(x)

        node = dm.custom(weird)(Model("c").continuous("x"))
        assert node.name == "weird"

    def test_explicit_name_overrides(self):
        m = Model("c")
        x = m.continuous("x")
        node = dm.custom(lambda x: x, name="my_fn")(x)
        assert node.name == "my_fn"
        assert "my_fn" in repr(node)
        assert repr(node).startswith("custom:")

    def test_lambda_default_name(self):
        # An unnamed lambda has __name__ == "<lambda>"; just ensure no crash.
        node = dm.custom(lambda x: x)(Model("c").continuous("x"))
        assert isinstance(node.name, str)

    def test_custom_rejects_non_callable(self):
        with pytest.raises(TypeError):
            dm.custom(42)
        with pytest.raises(TypeError):
            dm.custom("not a function")


# ─────────────────────────────────────────────────────────────
# Local NLP path: value + autodiff correctness, no certificate
# ─────────────────────────────────────────────────────────────


class TestLocalSolve:
    def test_scalar_custom_solves_and_uncertified(self):
        # minimize (x - 3)^2 written through an opaque callable.
        m = Model("scalar")
        x = m.continuous("x", lb=-10, ub=10)
        sq = dm.custom(lambda x: (x - 3.0) ** 2)
        m.minimize(sq(x))

        result = m.solve()
        assert result.status == "optimal"
        assert result.x["x"] == pytest.approx(3.0, abs=1e-4)
        # Opaque callable ⇒ no global optimality certificate.
        assert result.gap_certified is False

    def test_vector_custom_solves(self):
        # minimize sum((x - target)^2) over a vector variable.
        target = jnp.array([1.0, -2.0, 0.5])
        m = Model("vector")
        x = m.continuous("x", shape=(3,), lb=-10, ub=10)
        f = dm.custom(lambda x: jnp.sum((x - target) ** 2))
        m.minimize(f(x))

        result = m.solve()
        assert result.status == "optimal"
        np.testing.assert_allclose(result.x["x"], np.asarray(target), atol=1e-4)
        assert result.gap_certified is False

    def test_custom_in_constraint(self):
        # minimize x s.t. custom(x) <= 4  where custom(x) = x^2  ⇒  x >= -2.
        m = Model("constr")
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x)
        sq = dm.custom(lambda x: x**2)
        m.subject_to(sq(x) <= 4.0)

        result = m.solve()
        assert result.status == "optimal"
        assert result.x["x"] == pytest.approx(-2.0, abs=1e-4)
        assert result.gap_certified is False

    def test_custom_mixed_with_primitives(self):
        # Objective mixes an opaque term with ordinary dm.* primitives.
        m = Model("mixed")
        x = m.continuous("x", lb=-5, ub=5)
        opaque = dm.custom(lambda x: jnp.sin(x) ** 2)
        m.minimize(opaque(x) + (x - 1.0) ** 2)

        result = m.solve()
        assert result.status == "optimal"
        # Verify the reported point is a stationary point of the true objective:
        # d/dx [sin^2(x) + (x-1)^2] = sin(2x) + 2(x-1) = 0.
        xv = float(result.x["x"])
        assert abs(np.sin(2 * xv) + 2 * (xv - 1.0)) < 1e-4
        assert result.gap_certified is False


# ─────────────────────────────────────────────────────────────
# DAG compiler: value + gradient match the underlying callable
# ─────────────────────────────────────────────────────────────


class TestDagCompile:
    def test_compiled_value_and_gradient(self):
        from discopt._jax.dag_compiler import compile_expression

        m = Model("grad")
        x = m.continuous("x", shape=(2,), lb=-5, ub=5)

        def raw(v):
            return jnp.sum(v**3) + jnp.prod(v)

        node = dm.custom(raw)(x)
        fn = compile_expression(node, m)

        xv = jnp.array([1.5, -0.7])
        assert float(fn(xv)) == pytest.approx(float(raw(xv)))

        g_compiled = jax.grad(lambda v: fn(v))(xv)
        g_raw = jax.grad(raw)(xv)
        np.testing.assert_allclose(np.asarray(g_compiled), np.asarray(g_raw), atol=1e-6)


# ─────────────────────────────────────────────────────────────
# Refusal guards: integer vars, .nl export, relaxation compilation
# ─────────────────────────────────────────────────────────────


class TestRefusals:
    def test_integer_variable_raises(self):
        m = Model("int")
        x = m.continuous("x", lb=-10, ub=10)
        m.integer("k", lb=0, ub=5)
        f = dm.custom(lambda x: (x - 3.0) ** 2)
        m.minimize(f(x))

        with pytest.raises(ValueError, match="custom"):
            m.solve()

    def test_binary_variable_raises(self):
        m = Model("bin")
        x = m.continuous("x", lb=-10, ub=10)
        m.binary("b")
        f = dm.custom(lambda x: (x - 3.0) ** 2)
        m.minimize(f(x))

        with pytest.raises(ValueError, match="custom"):
            m.solve()

    def test_nl_export_raises(self):
        from discopt.export.nl import to_nl

        m = Model("nl")
        x = m.continuous("x", lb=-10, ub=10)
        f = dm.custom(lambda x: (x - 3.0) ** 2)
        m.minimize(f(x))

        with pytest.raises(ValueError, match="custom|opaque|AD-only"):
            to_nl(m)

    def test_relaxation_compile_raises(self):
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        m = Model("relax")
        x = m.continuous("x", lb=-10, ub=10)
        f = dm.custom(lambda x: (x - 3.0) ** 2)
        m.minimize(f(x))

        with pytest.raises(NotImplementedError, match="custom|opaque|relaxation"):
            compile_objective_relaxation(m)
