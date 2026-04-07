"""Tests for .nl DAG reconstruction: Rust ExprArena -> Python Expression trees."""

import math

import numpy as np
import pytest
from discopt._jax.nl_reconstruction import reconstruct_dag
from discopt._rust import parse_nl_string
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    Model,
    UnaryOp,
    Variable,
)

# ---------------------------------------------------------------------------
# Helper: build a Model with variables matching an nl_repr
# ---------------------------------------------------------------------------


def _make_model_from_nl(nl_repr):
    """Create a Model with variables matching a parsed .nl representation."""
    m = Model("test")
    var_types = nl_repr.var_types()
    var_names = nl_repr.var_names()
    for i in range(len(var_names)):
        vt = var_types[i]
        name = var_names[i]
        lb = float(nl_repr.var_lb(i)[0])
        ub = float(nl_repr.var_ub(i)[0])
        if vt == "continuous":
            m.continuous(name, lb=lb, ub=ub)
        elif vt == "binary":
            m.binary(name)
        elif vt == "integer":
            m.integer(name, lb=lb, ub=ub)
    return m


# ---------------------------------------------------------------------------
# .nl test strings
# ---------------------------------------------------------------------------


def _linear_nl():
    """min 2*x + 3*y  s.t. x + y <= 10"""
    return (
        "g3 1 1 0\n 2 1 1 0 0\n 0 0\n 0 0 0\n 0 0 0\n 0 0 0 1\n 0 0\n"
        " 2 2\n 0 0\n 0 0 0 0 0\nO0 0\nn0\nC0\nn0\nx2\n0 0\n1 0\nr\n"
        "1 10\nb\n0 0 100\n0 0 100\nk1\n1\nJ0 2\n0 1\n1 1\nG0 2\n0 2\n1 3\n"
    )


def _quadratic_nl():
    """min x^2 + y^2  s.t. x + y >= 1"""
    return (
        "g3 1 1 0\n 2 1 1 0 0\n 1 1\n 0 0\n 2 2 2\n 0 0 0 1\n 0 0\n"
        " 0 0\n 0 0\n 0 0 0 0 0\nO0 0\no0\no5\nv0\nn2\no5\nv1\nn2\n"
        "C0\nn0\nx2\n0 1\n1 1\nr\n2 1\nb\n0 -10 10\n0 -10 10\nk1\n1\n"
        "J0 2\n0 1\n1 1\nG0 2\n0 0\n1 0\n"
    )


def _nonlinear_nl():
    """min exp(x) + log(y)  s.t. x + y <= 5"""
    return (
        "g3 1 1 0\n 2 1 1 0 0\n 0 1\n 0 0\n 0 2 0\n 0 0 0 1\n 0 0\n"
        " 2 0\n 0 0\n 0 0 0 0 0\nO0 0\no0\no44\nv0\no43\nv1\n"
        "C0\nn0\nx2\n0 1\n1 1\nr\n1 5\nb\n0 0.1 10\n0 0.1 10\nk1\n1\n"
        "J0 2\n0 1\n1 1\n"
    )


def _trig_nl():
    """min sin(x) + cos(y)"""
    return (
        "g3 1 1 0\n 2 0 1 0 0\n 0 1\n 0 0\n 0 2 0\n 0 0 0 1\n 0 0\n"
        " 0 0\n 0 0\n 0 0 0 0 0\nO0 0\no0\no41\nv0\no46\nv1\n"
        "b\n3\n3\n"
    )


def _atan_nl():
    """min atan(x)"""
    return (
        "g3 1 1 0\n 1 0 1 0 0\n 0 1\n 0 0\n 0 1 0\n 0 0 0 1\n 0 0\n"
        " 0 0\n 0 0\n 0 0 0 0 0\nO0 0\no49\nv0\nb\n3\n"
    )


def _sinh_nl():
    """min sinh(x)"""
    return (
        "g3 1 1 0\n 1 0 1 0 0\n 0 1\n 0 0\n 0 1 0\n 0 0 0 1\n 0 0\n"
        " 0 0\n 0 0\n 0 0 0 0 0\nO0 0\no40\nv0\nb\n3\n"
    )


def _tanh_nl():
    """min tanh(x)"""
    return (
        "g3 1 1 0\n 1 0 1 0 0\n 0 1\n 0 0\n 0 1 0\n 0 0 0 1\n 0 0\n"
        " 0 0\n 0 0\n 0 0 0 0 0\nO0 0\no37\nv0\nb\n3\n"
    )


def _asin_nl():
    """min asin(x)"""
    return (
        "g3 1 1 0\n 1 0 1 0 0\n 0 1\n 0 0\n 0 1 0\n 0 0 0 1\n 0 0\n"
        " 0 0\n 0 0\n 0 0 0 0 0\nO0 0\no51\nv0\nb\n0 -1 1\n"
    )


def _sumlist_nl():
    """min sum(x_i^2) for i=0..3"""
    return (
        "g3 1 1 0\n 4 0 1 0 0\n 0 1\n 0 0\n 0 4 0\n 0 0 0 1\n 0 0\n"
        " 0 0\n 0 0\n 0 0 0 0 0\nO0 0\no54\n4\no5\nv0\nn2\no5\nv1\nn2\n"
        "o5\nv2\nn2\no5\nv3\nn2\nb\n3\n3\n3\n3\n"
    )


def _negation_nl():
    """min -x"""
    return (
        "g3 1 1 0\n 1 0 1 0 0\n 0 1\n 0 0\n 0 1 0\n 0 0 0 1\n 0 0\n"
        " 0 0\n 0 0\n 0 0 0 0 0\nO0 0\no16\nv0\nb\n3\n"
    )


# ---------------------------------------------------------------------------
# PyO3 binding tests
# ---------------------------------------------------------------------------


class TestPyO3Bindings:
    """Test the new PyO3 methods on PyModelRepr."""

    def test_arena_len(self):
        nl_repr = parse_nl_string(_linear_nl())
        assert nl_repr.arena_len() > 0
        assert nl_repr.arena_len() == nl_repr.n_nodes

    def test_get_node_variable(self):
        nl_repr = parse_nl_string(_linear_nl())
        node = nl_repr.get_node(0)
        assert node["type"] == "variable"
        assert node["index"] == 0
        assert node["name"] == "x0"

    def test_get_node_constant(self):
        nl_repr = parse_nl_string(_linear_nl())
        # Find a constant node
        found_const = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "constant":
                found_const = True
                assert isinstance(node["value"], float)
                break
        assert found_const

    def test_get_node_binary_op(self):
        nl_repr = parse_nl_string(_quadratic_nl())
        found_binop = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "binary_op":
                found_binop = True
                assert node["op"] in ("+", "-", "*", "/", "**")
                assert isinstance(node["left"], int)
                assert isinstance(node["right"], int)
                break
        assert found_binop

    def test_get_node_function_call(self):
        nl_repr = parse_nl_string(_nonlinear_nl())
        found_func = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "function_call":
                found_func = True
                assert node["func"] in ("exp", "log", "sin", "cos", "atan", "sinh", "tanh")
                assert isinstance(node["args"], list)
                break
        assert found_func

    def test_get_node_unary_op(self):
        nl_repr = parse_nl_string(_negation_nl())
        found_unary = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "unary_op":
                found_unary = True
                assert node["op"] in ("neg", "abs")
                assert isinstance(node["arg"], int)
                break
        assert found_unary

    def test_get_node_sum_over(self):
        nl_repr = parse_nl_string(_sumlist_nl())
        found_sum = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "sum_over":
                found_sum = True
                assert isinstance(node["terms"], list)
                assert len(node["terms"]) == 4
                break
        assert found_sum

    def test_objective_id(self):
        nl_repr = parse_nl_string(_linear_nl())
        obj_id = nl_repr.objective_id()
        assert isinstance(obj_id, int)
        assert 0 <= obj_id < nl_repr.arena_len()

    def test_constraint_ids(self):
        nl_repr = parse_nl_string(_linear_nl())
        cids = nl_repr.constraint_ids()
        assert isinstance(cids, list)
        assert len(cids) == nl_repr.n_constraints
        for cid in cids:
            assert 0 <= cid < nl_repr.arena_len()

    def test_constraint_info(self):
        nl_repr = parse_nl_string(_linear_nl())
        expr_id, sense, rhs = nl_repr.constraint_info(0)
        assert isinstance(expr_id, int)
        assert sense in ("<=", "==", ">=")
        assert isinstance(rhs, float)

    def test_get_node_out_of_range(self):
        nl_repr = parse_nl_string(_linear_nl())
        with pytest.raises(IndexError):
            nl_repr.get_node(nl_repr.arena_len() + 100)

    def test_new_math_func_atan(self):
        nl_repr = parse_nl_string(_atan_nl())
        found = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "function_call" and node["func"] == "atan":
                found = True
        assert found

    def test_new_math_func_sinh(self):
        nl_repr = parse_nl_string(_sinh_nl())
        found = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "function_call" and node["func"] == "sinh":
                found = True
        assert found

    def test_new_math_func_tanh(self):
        nl_repr = parse_nl_string(_tanh_nl())
        found = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "function_call" and node["func"] == "tanh":
                found = True
        assert found

    def test_new_math_func_asin(self):
        nl_repr = parse_nl_string(_asin_nl())
        found = False
        for i in range(nl_repr.arena_len()):
            node = nl_repr.get_node(i)
            if node["type"] == "function_call" and node["func"] == "asin":
                found = True
        assert found


# ---------------------------------------------------------------------------
# DAG reconstruction tests
# ---------------------------------------------------------------------------


class TestReconstruction:
    """Test reconstructing Python Expression DAGs from Rust arenas."""

    def test_linear_objective(self):
        nl_repr = parse_nl_string(_linear_nl())
        m = _make_model_from_nl(nl_repr)
        obj, cons = reconstruct_dag(nl_repr, m._variables)
        assert isinstance(obj, (BinaryOp, Variable, Constant, FunctionCall, UnaryOp))

    def test_linear_constraint(self):
        nl_repr = parse_nl_string(_linear_nl())
        m = _make_model_from_nl(nl_repr)
        _, cons = reconstruct_dag(nl_repr, m._variables)
        assert len(cons) == 1
        body, sense, rhs = cons[0]
        assert sense == "<="
        assert abs(rhs - 10.0) < 1e-12

    def test_quadratic_reconstruction(self):
        nl_repr = parse_nl_string(_quadratic_nl())
        m = _make_model_from_nl(nl_repr)
        obj, cons = reconstruct_dag(nl_repr, m._variables)
        assert obj is not None
        assert len(cons) == 1

    def test_nonlinear_reconstruction(self):
        nl_repr = parse_nl_string(_nonlinear_nl())
        m = _make_model_from_nl(nl_repr)
        obj, cons = reconstruct_dag(nl_repr, m._variables)
        assert obj is not None

    def test_negation_reconstruction(self):
        nl_repr = parse_nl_string(_negation_nl())
        m = _make_model_from_nl(nl_repr)
        obj, _ = reconstruct_dag(nl_repr, m._variables)
        assert isinstance(obj, UnaryOp)
        assert obj.op == "neg"

    def test_sumlist_reconstruction(self):
        nl_repr = parse_nl_string(_sumlist_nl())
        m = _make_model_from_nl(nl_repr)
        obj, _ = reconstruct_dag(nl_repr, m._variables)
        # SumOver becomes chain of BinaryOp("+", ...)
        assert isinstance(obj, BinaryOp)


# ---------------------------------------------------------------------------
# Round-trip evaluation tests: Rust eval == JAX eval
# ---------------------------------------------------------------------------


class TestRoundTripEval:
    """Verify reconstructed DAG evaluates identically to Rust evaluator."""

    def _eval_jax(self, nl_str, x_vals):
        """Compile and evaluate the reconstructed objective via JAX."""
        import jax.numpy as jnp
        from discopt._jax.dag_compiler import compile_objective

        nl_repr = parse_nl_string(nl_str)
        m = _make_model_from_nl(nl_repr)
        obj, _ = reconstruct_dag(nl_repr, m._variables)
        m.minimize(obj)
        f = compile_objective(m)
        return float(f(jnp.array(x_vals)))

    def _eval_rust(self, nl_str, x_vals):
        """Evaluate objective via Rust evaluator."""
        nl_repr = parse_nl_string(nl_str)
        return nl_repr.evaluate_objective(np.array(x_vals, dtype=np.float64))

    def test_linear_roundtrip(self):
        x = [3.0, 4.0]
        jax_val = self._eval_jax(_linear_nl(), x)
        rust_val = self._eval_rust(_linear_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        # 2*3 + 3*4 = 18
        np.testing.assert_allclose(jax_val, 18.0, atol=1e-10)

    def test_quadratic_roundtrip(self):
        x = [3.0, 4.0]
        jax_val = self._eval_jax(_quadratic_nl(), x)
        rust_val = self._eval_rust(_quadratic_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        np.testing.assert_allclose(jax_val, 25.0, atol=1e-10)

    def test_nonlinear_roundtrip(self):
        x = [1.0, math.e]
        jax_val = self._eval_jax(_nonlinear_nl(), x)
        rust_val = self._eval_rust(_nonlinear_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        expected = math.exp(1.0) + math.log(math.e)
        np.testing.assert_allclose(jax_val, expected, atol=1e-10)

    def test_trig_roundtrip(self):
        x = [math.pi / 2, 0.0]
        jax_val = self._eval_jax(_trig_nl(), x)
        rust_val = self._eval_rust(_trig_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        # sin(pi/2) + cos(0) = 1 + 1 = 2
        np.testing.assert_allclose(jax_val, 2.0, atol=1e-10)

    def test_atan_roundtrip(self):
        x = [1.0]
        jax_val = self._eval_jax(_atan_nl(), x)
        rust_val = self._eval_rust(_atan_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        np.testing.assert_allclose(jax_val, math.atan(1.0), atol=1e-10)

    def test_sinh_roundtrip(self):
        x = [1.0]
        jax_val = self._eval_jax(_sinh_nl(), x)
        rust_val = self._eval_rust(_sinh_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        np.testing.assert_allclose(jax_val, math.sinh(1.0), atol=1e-10)

    def test_tanh_roundtrip(self):
        x = [0.5]
        jax_val = self._eval_jax(_tanh_nl(), x)
        rust_val = self._eval_rust(_tanh_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        np.testing.assert_allclose(jax_val, math.tanh(0.5), atol=1e-10)

    def test_asin_roundtrip(self):
        x = [0.5]
        jax_val = self._eval_jax(_asin_nl(), x)
        rust_val = self._eval_rust(_asin_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        np.testing.assert_allclose(jax_val, math.asin(0.5), atol=1e-10)

    def test_sumlist_roundtrip(self):
        x = [1.0, 2.0, 3.0, 4.0]
        jax_val = self._eval_jax(_sumlist_nl(), x)
        rust_val = self._eval_rust(_sumlist_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        np.testing.assert_allclose(jax_val, 30.0, atol=1e-10)

    def test_negation_roundtrip(self):
        x = [5.0]
        jax_val = self._eval_jax(_negation_nl(), x)
        rust_val = self._eval_rust(_negation_nl(), x)
        np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)
        np.testing.assert_allclose(jax_val, -5.0, atol=1e-10)

    def test_random_points_linear(self):
        """Test at multiple random points."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            x = rng.uniform(0, 100, size=2).tolist()
            jax_val = self._eval_jax(_linear_nl(), x)
            rust_val = self._eval_rust(_linear_nl(), x)
            np.testing.assert_allclose(jax_val, rust_val, atol=1e-10)

    def test_random_points_nonlinear(self):
        """Test nonlinear at random points in valid domain."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            x = rng.uniform(0.1, 10, size=2).tolist()
            jax_val = self._eval_jax(_nonlinear_nl(), x)
            rust_val = self._eval_rust(_nonlinear_nl(), x)
            np.testing.assert_allclose(jax_val, rust_val, atol=1e-8)


# ---------------------------------------------------------------------------
# Integration: from_nl() produces a solvable model
# ---------------------------------------------------------------------------


class TestFromNlIntegration:
    """Test that from_nl() produces models with proper Expression DAGs."""

    def test_from_nl_has_objective(self):
        """from_nl() should produce a real objective, not Constant(0)."""
        import os
        import tempfile

        nl = _linear_nl()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nl", delete=False) as f:
            f.write(nl)
            path = f.name

        try:
            import discopt.modeling as dm

            model = dm.from_nl(path)
            # The objective should not be a bare Constant(0.0) anymore
            assert model._objective is not None
            expr = model._objective.expression
            assert not (isinstance(expr, Constant) and expr.value == 0.0)
        finally:
            os.unlink(path)

    def test_from_nl_has_constraints(self):
        """from_nl() should add proper Constraint objects."""
        import os
        import tempfile

        nl = _linear_nl()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nl", delete=False) as f:
            f.write(nl)
            path = f.name

        try:
            import discopt.modeling as dm

            model = dm.from_nl(path)
            assert len(model._constraints) >= 1
        finally:
            os.unlink(path)

    def test_from_nl_compiles(self):
        """from_nl() model can be compiled to JAX function."""
        import os
        import tempfile

        import jax.numpy as jnp

        nl = _quadratic_nl()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nl", delete=False) as f:
            f.write(nl)
            path = f.name

        try:
            import discopt.modeling as dm
            from discopt._jax.dag_compiler import compile_objective

            model = dm.from_nl(path)
            f = compile_objective(model)
            val = float(f(jnp.array([3.0, 4.0])))
            np.testing.assert_allclose(val, 25.0, atol=1e-10)
        finally:
            os.unlink(path)

    def test_from_nl_evaluator(self):
        """from_nl() model works with NLPEvaluator (not NLPEvaluatorFromNl)."""
        import os
        import tempfile

        nl = _nonlinear_nl()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nl", delete=False) as f:
            f.write(nl)
            path = f.name

        try:
            import discopt.modeling as dm
            from discopt._jax.nlp_evaluator import NLPEvaluator

            model = dm.from_nl(path)
            evaluator = NLPEvaluator(model)
            val = evaluator.evaluate_objective(np.array([1.0, math.e]))
            expected = math.exp(1.0) + math.log(math.e)
            np.testing.assert_allclose(val, expected, atol=1e-10)
        finally:
            os.unlink(path)

    def test_from_nl_gradient(self):
        """from_nl() model produces correct JAX gradients."""
        import os
        import tempfile

        import jax
        import jax.numpy as jnp

        nl = _quadratic_nl()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nl", delete=False) as f:
            f.write(nl)
            path = f.name

        try:
            import discopt.modeling as dm
            from discopt._jax.dag_compiler import compile_objective

            model = dm.from_nl(path)
            f = compile_objective(model)
            grad_f = jax.grad(f)
            x = jnp.array([3.0, 4.0])
            grad = grad_f(x)
            # d/dx(x^2 + y^2) = [2x, 2y] = [6, 8]
            np.testing.assert_allclose(np.array(grad), [6.0, 8.0], atol=1e-10)
        finally:
            os.unlink(path)
