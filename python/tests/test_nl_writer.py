"""
Tests for the .nl writer (discopt Model -> AMPL .nl format).

Validates structural correctness of emitted .nl files and round-trip
compatibility with the Rust .nl parser where available.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

pytestmark = pytest.mark.unit


class TestNLWriterLinear:
    """Test .nl export for linear models."""

    def test_simple_lp(self):
        m = dm.Model("simple")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(3 * x + 2 * y)
        m.subject_to(x + y <= 8)

        nl = m.to_nl()
        assert nl is not None
        assert nl.startswith("g3")
        # Should have 2 vars, 1 constraint, 1 objective
        lines = nl.split("\n")
        assert "2 1 1 0 0" in lines[1]  # n_vars n_cons n_objs

    def test_bounds_encoding(self):
        m = dm.Model("bounds")
        m.continuous("x", lb=0, ub=10)
        m.continuous("y", lb=-5, ub=5)
        m.continuous("z")  # free
        x, y, z = m._variables
        m.minimize(x + y + z)

        nl = m.to_nl()
        # Check b section
        assert "b\n" in nl
        assert "0 0.0 10.0" in nl  # x: range
        assert "0 -5.0 5.0" in nl  # y: range
        assert "\n3\n" in nl  # z: free

    def test_constraint_senses(self):
        m = dm.Model("senses")
        x = m.continuous("x", lb=0)
        m.minimize(x)
        m.subject_to(x <= 10)
        m.subject_to(x >= 1)
        m.subject_to(x == 5)

        nl = m.to_nl()
        assert "r\n" in nl
        # <= 10 -> type 1, >= 1 -> type 2, == 5 -> type 4
        r_section = nl[nl.index("r\n") :]
        assert "1 0.0" in r_section  # <= 0 (normalized: x - 10 <= 0)
        assert "4 0.0" in r_section  # == 0 (normalized: x - 5 == 0)


class TestNLWriterNonlinear:
    """Test .nl export for nonlinear models."""

    def test_exp_function(self):
        m = dm.Model("exp_test")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(dm.exp(x))

        nl = m.to_nl()
        assert "o44" in nl  # exp opcode

    def test_log_function(self):
        m = dm.Model("log_test")
        x = m.continuous("x", lb=0.1, ub=10)
        m.minimize(dm.log(x))

        nl = m.to_nl()
        assert "o43" in nl  # log opcode

    def test_trig_functions(self):
        m = dm.Model("trig")
        x = m.continuous("x", lb=0, ub=3.14)
        m.minimize(dm.sin(x) + dm.cos(x))

        nl = m.to_nl()
        assert "o41" in nl  # sin opcode
        assert "o46" in nl  # cos opcode

    def test_sqrt(self):
        m = dm.Model("sqrt_test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(dm.sqrt(x))

        nl = m.to_nl()
        assert "o39" in nl  # sqrt opcode

    def test_power(self):
        m = dm.Model("power_test")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(x**2)

        nl = m.to_nl()
        assert "o5" in nl  # power opcode

    def test_quadratic(self):
        m = dm.Model("quad")
        x = m.continuous("x")
        y = m.continuous("y")
        m.minimize(x * y + x**2)
        m.subject_to(x + y <= 10)

        nl = m.to_nl()
        assert nl is not None
        assert "o2" in nl  # multiply
        assert "o5" in nl  # power


class TestNLWriterMINLP:
    """Test .nl export for MINLP models."""

    def test_binary_vars(self):
        m = dm.Model("binary")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x + 5 * y)
        m.subject_to(x <= 10 * y)

        nl = m.to_nl()
        # binary var count should be 1
        lines = nl.split("\n")
        # Line 6 has discrete var counts
        assert "1 0 0 0 0" in lines[6]  # 1 binary, 0 integer

    def test_integer_vars(self):
        m = dm.Model("integer")
        x = m.continuous("x", lb=0)
        n = m.integer("n", lb=0, ub=10)
        m.minimize(x + n)

        nl = m.to_nl()
        lines = nl.split("\n")
        assert "0 1 0 0 0" in lines[6]  # 0 binary, 1 integer

    def test_minlp_with_nonlinear(self):
        m = dm.Model("minlp")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(dm.exp(x) + 3 * y)
        m.subject_to(x <= 5 * y)

        nl = m.to_nl()
        assert nl is not None
        assert "o44" in nl  # exp opcode
        lines = nl.split("\n")
        assert "1 0 0 0 0" in lines[6]  # 1 binary


class TestNLWriterArrayVars:
    """Test .nl export for models with array variables."""

    def test_vector_variable(self):
        m = dm.Model("vec")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0] + x[1] + x[2])
        m.subject_to(x[0] + x[1] <= 5)

        nl = m.to_nl()
        assert nl is not None
        lines = nl.split("\n")
        assert "3 1 1 0 0" in lines[1]  # 3 vars


class TestNLWriterFile:
    """Test writing to file."""

    def test_write_to_file(self, tmp_path):
        m = dm.Model("filetest")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)

        outpath = tmp_path / "test.nl"
        result = m.to_nl(str(outpath))
        assert result is None
        assert outpath.exists()
        content = outpath.read_text()
        assert content.startswith("g3")


class TestNLWriterRoundTrip:
    """Round-trip: discopt Model -> .nl -> Rust parser -> Model."""

    @pytest.fixture
    def has_rust_parser(self):
        try:
            from discopt._rust import parse_nl_string  # noqa: F401

            return True
        except ImportError:
            return False

    def test_roundtrip_linear(self, has_rust_parser, tmp_path):
        if not has_rust_parser:
            pytest.skip("Rust .nl parser not available")

        from discopt._rust import parse_nl_file

        m = dm.Model("rt_linear")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(3 * x + 2 * y)
        m.subject_to(x + y <= 8)

        nl_path = tmp_path / "test.nl"
        m.to_nl(str(nl_path))

        # Parse back with Rust parser and verify semantics
        nl_repr = parse_nl_file(str(nl_path))
        assert nl_repr.n_vars == 2
        assert nl_repr.n_constraints == 1
        assert nl_repr.objective_sense == "minimize"
        # Verify variable bounds
        assert nl_repr.var_lb(0) == [0.0]
        assert nl_repr.var_ub(0) == [10.0]
        assert nl_repr.var_lb(1) == [0.0]
        assert nl_repr.var_ub(1) == [5.0]

    def test_roundtrip_nonlinear(self, has_rust_parser, tmp_path):
        if not has_rust_parser:
            pytest.skip("Rust .nl parser not available")

        from discopt._rust import parse_nl_file

        m = dm.Model("rt_nlp")
        x = m.continuous("x", lb=0.1, ub=5)
        m.minimize(dm.exp(x) + dm.log(x))
        m.subject_to(x <= 4)

        nl_path = tmp_path / "test.nl"
        m.to_nl(str(nl_path))

        nl_repr = parse_nl_file(str(nl_path))
        assert nl_repr.n_vars == 1

    def test_roundtrip_minlp(self, has_rust_parser, tmp_path):
        if not has_rust_parser:
            pytest.skip("Rust .nl parser not available")

        from discopt._rust import parse_nl_file

        m = dm.Model("rt_minlp")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(dm.exp(x) + 3 * y)
        m.subject_to(x <= 5 * y)

        nl_path = tmp_path / "test.nl"
        m.to_nl(str(nl_path))

        nl_repr = parse_nl_file(str(nl_path))
        assert nl_repr.n_vars == 2
        assert nl_repr.n_constraints == 1
        assert nl_repr.objective_sense == "minimize"


class TestNLWriterMaximize:
    def test_maximize_sense(self):
        m = dm.Model("max_test")
        x = m.continuous("x", lb=0, ub=10)
        m.maximize(x)

        nl = m.to_nl()
        assert "O0 1" in nl  # sense=1 means maximize


class TestNLWriterNestedFunctions:
    def test_nested_exp_log(self):
        m = dm.Model("nested")
        x = m.continuous("x", lb=0.1, ub=5)
        m.minimize(dm.exp(dm.log(x)))

        nl = m.to_nl()
        assert "o44" in nl  # exp
        assert "o43" in nl  # log


class TestNLWriterCollocation:
    """Export of DAEBuilder / orthogonal-collocation models (issue #96).

    Collocation ``discretize()`` emits vectorized constraints whose bodies are
    array-valued (built from differentiation-matrix products and broadcasting).
    The scalar-oriented writer must expand each into one scalar row per output
    element rather than crashing in ``float()`` on an array constant.
    """

    @staticmethod
    def _build_dae_model():
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("dae")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=3, scheme="radau")
        dae = DAEBuilder(m, cs)
        dae.add_state("a", bounds=(0, 1), initial=0.0)
        dae.add_control("u", bounds=(0, 1))
        dae.set_ode(lambda t, x, z, u: {"a": -x["a"] + u["u"]})
        v = dae.discretize()
        m.minimize(sum((v["a"][i, j] - 0.5) ** 2 for i in range(4) for j in range(1, 4)))
        return m

    def test_collocation_to_nl_succeeds(self):
        """The repro from issue #96 must export without raising."""
        m = self._build_dae_model()
        nl = m.to_nl()
        assert nl  # non-empty
        # Header line carries the expanded scalar-constraint count.
        header = nl.split("\n")[1].split()
        n_vars, n_cons = int(header[0]), int(header[1])
        assert n_vars > 0
        assert n_cons > 0

    def test_scalarized_bodies_match_array_body(self):
        """Each scalar row must equal the corresponding element of the array body.

        Compile both the original (array-valued) constraint body and every
        scalar body produced by the writer's expansion, evaluate them at a
        random point, and require elementwise agreement.
        """
        import numpy as np
        from discopt._jax.dag_compiler import compile_expression
        from discopt.export.nl import _NLWriter

        m = self._build_dae_model()
        writer = _NLWriter(m)
        writer._build_var_map()

        nvar = sum(max(1, int(np.prod(var.shape))) for var in m._variables)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(nvar)

        assert m._constraints  # the collocation constraints exist
        for con in m._constraints:
            expected = np.asarray(compile_expression(con.body, m)(x)).ravel()
            bodies = writer._scalarize_body(con.body)
            got = np.array([float(compile_expression(b, m)(x)) for b in bodies])
            assert got.shape == expected.shape
            np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_collocation_roundtrip_parse(self, tmp_path):
        """The emitted .nl must parse back with the Rust parser, var counts intact."""
        try:
            from discopt._rust import parse_nl_file
        except ImportError:
            pytest.skip("Rust .nl parser not available")

        import numpy as np

        m = self._build_dae_model()
        nvar = sum(max(1, int(np.prod(var.shape))) for var in m._variables)

        nl_path = tmp_path / "dae.nl"
        m.to_nl(str(nl_path))

        rep = parse_nl_file(str(nl_path))
        assert rep.n_vars == nvar
        assert rep.n_constraints > 0
        assert rep.objective_sense == "minimize"
