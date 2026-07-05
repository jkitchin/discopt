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
        # The body constant is carried in the r-section rhs (ASL convention),
        # NOT baked into the C body with rhs 0.  Discopt normalizes:
        #   x <= 10  -> (x - 10) <= 0        -> type 1, rhs 10
        #   x >= 1   -> (1 - x) <= 0         -> type 1, rhs -1  (i.e. -x <= -1)
        #   x == 5   -> (x - 5) == 0         -> type 4, rhs 5
        r_section = nl[nl.index("r\n") :]
        assert "1 10.0" in r_section  # x <= 10
        assert "1 -1.0" in r_section  # -x <= -1  (x >= 1)
        assert "4 5.0" in r_section  # x == 5


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


class TestNLWriterDiscreteInNonlinear:
    """Discrete variables that appear in nonlinear expressions (issue #210).

    AMPL requires nonlinear variables to occupy the lowest indices and declares
    the discrete ones among them via the header's ``nbv niv nlvbi nlvci nlvoi``
    line (linear-binary, linear-integer, then integer/binary nonlinear in
    both / cons-only / objs-only). The previous writer wrote *every* discrete
    var as linear-discrete (``nbv``/``niv``) with ``nlvbi=nlvci=nlvoi=0``, so an
    AMPL-compatible solver read a discrete-in-nonlinear var as continuous and
    silently solved a relaxation.
    """

    @staticmethod
    def _disc_line(nl: str) -> str:
        return nl.split("\n")[6].split("#")[0].strip()

    @staticmethod
    def _nlvar_line(nl: str) -> str:
        return nl.split("\n")[4].split("#")[0].strip()

    def test_integer_in_nonlinear_objective(self):
        """An integer var inside the objective's nonlinear part -> nlvoi."""
        m = dm.Model("int_in_nl_obj")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n**2)
        nl = m.to_nl()
        # 1 nonlinear var in objectives (total), 0 in cons, 0 in both
        assert self._nlvar_line(nl) == "0 1 0"
        # nbv niv nlvbi nlvci nlvoi -> the integer is nonlinear-in-obj-only
        assert self._disc_line(nl) == "0 0 0 0 1"

    def test_binary_in_nonlinear_objective(self):
        """A binary var in a nonlinear objective counts as nonlinear-integer."""
        m = dm.Model("bin_in_nl_obj")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        # y appears nonlinearly (product with x) -> nonlinear discrete in obj
        m.minimize(x * y + dm.exp(x))
        nl = m.to_nl()
        # x and y both nonlinear in obj only
        assert self._nlvar_line(nl) == "0 2 0"
        # one of the two nonlinear-obj vars is discrete (binary y) -> nlvoi=1
        assert self._disc_line(nl) == "0 0 0 0 1"

    def test_integer_in_nonlinear_constraint(self):
        """An integer var nonlinear in a constraint only -> nlvci."""
        m = dm.Model("int_in_nl_con")
        x = m.continuous("x", lb=0, ub=10)
        n = m.integer("n", lb=0, ub=5)
        m.minimize(x + n)  # n linear in obj
        m.subject_to(n**2 + x <= 20)  # n nonlinear in constraint
        nl = m.to_nl()
        assert self._nlvar_line(nl) == "1 0 0"  # 1 nl var in cons (n)
        assert self._disc_line(nl) == "0 0 0 1 0"  # nlvci = 1

    def test_integer_nonlinear_in_both(self):
        """An integer var nonlinear in both obj and a constraint -> nlvbi."""
        m = dm.Model("int_in_both")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n**2)
        m.subject_to(n**2 <= 16)
        nl = m.to_nl()
        # n is nonlinear in both -> nlvc=1, nlvo=1, nlvb=1
        assert self._nlvar_line(nl) == "1 1 1"
        assert self._disc_line(nl) == "0 0 1 0 0"  # nlvbi = 1

    def test_mixed_linear_and_nonlinear_discrete(self):
        """Discrete vars split correctly between linear (nbv/niv) and nonlinear."""
        m = dm.Model("mixed")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")  # linear in obj
        n = m.integer("n", lb=0, ub=4)  # nonlinear in obj
        m.minimize(n**2 + 3 * y + x)
        m.subject_to(x <= 10 * y)
        nl = m.to_nl()
        # nbv=1 (linear binary y), niv=0, nlvoi=1 (integer n nonlinear in obj)
        assert self._disc_line(nl) == "1 0 0 0 1"

    def test_nonlinear_var_gets_lowest_index(self):
        """Variables nonlinear in the objective must occupy the lowest indices."""
        m = dm.Model("ordering")
        a = m.binary("a")  # linear
        n = m.integer("n", lb=0, ub=4)  # nonlinear in obj
        x = m.continuous("x", lb=0, ub=5)  # nonlinear in obj
        m.minimize(n**2 + dm.exp(x) + 2 * a)
        nl = m.to_nl()
        # The objective's nonlinear section must reference v0 and v1 (the two
        # nonlinear vars x, n), never v2 (the linear binary a).
        o_start = nl.index("O0")
        # objective body runs until the next top-level section (r or b)
        end = min(i for i in (nl.find("\nr\n", o_start), nl.find("\nb\n", o_start)) if i != -1)
        o_section = nl[o_start:end]
        assert "v2" not in o_section  # the linear binary is not in the nl body
        assert "v0" in o_section and "v1" in o_section

    def test_rust_parser_reads_discrete_in_nonlinear(self, tmp_path):
        """Round-trip: the Rust parser must read the nonlinear-discrete var as integer."""
        try:
            from discopt._rust import parse_nl_file
        except ImportError:
            pytest.skip("Rust .nl parser not available")
        m = dm.Model("rt_disc_nl")
        x = m.continuous("x", lb=0, ub=5)
        n = m.integer("n", lb=0, ub=4)
        m.minimize(n**2 + dm.exp(x))
        nl_path = tmp_path / "disc_nl.nl"
        m.to_nl(str(nl_path))
        rep = parse_nl_file(str(nl_path))
        # exactly one integer var survives the round-trip
        vtypes = list(rep.var_types())
        assert vtypes.count("integer") == 1
        assert vtypes.count("continuous") == 1

    def test_minlplib_integer_nonlinear_roundtrip_header(self):
        """to_nl(from_nl(f)) reproduces the original discrete-count header line.

        nvs15's objective is nonlinear in 3 integer variables; the original
        MINLPLib header declares them as ``0 0 0 0 3`` (nlvoi=3). The export
        must reproduce that, not bury them in ``niv`` (the #210 regression).
        """
        import os

        data = os.path.join(os.path.dirname(__file__), "data", "minlplib", "nvs15.nl")
        if not os.path.exists(data):
            pytest.skip("nvs15.nl fixture not available")
        orig = open(data).read().splitlines()
        m = dm.from_nl(data)
        out = m.to_nl().splitlines()

        def discrete(lines):
            toks = lines[6].split("#")[0].split()
            return [int(t) for t in toks] + [0] * (5 - len(toks))

        def nlvars(lines):
            toks = lines[4].split("#")[0].split()
            return [int(t) for t in toks] + [0] * (3 - len(toks))

        assert discrete(out) == discrete(orig)  # 0 0 0 0 3
        assert nlvars(out) == nlvars(orig)  # 0 3 0


class TestNLWriterMILPRegression:
    """MILP/LP export must be byte-for-byte unaffected by the #210 reorder.

    When every discrete variable is linear, the canonical reorder must leave
    the variable order and header exactly as before: discrete vars stay in
    ``nbv``/``niv`` with ``nlvbi=nlvci=nlvoi=0``.
    """

    def test_pure_milp_discrete_line_unchanged(self):
        m = dm.Model("milp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        z = m.integer("z", lb=0, ub=5)
        m.minimize(2 * x + 3 * y + z)
        m.subject_to(x + y + z <= 12)
        nl = m.to_nl()
        line6 = nl.split("\n")[6].split("#")[0].strip()
        # 1 linear binary, 1 linear integer, no nonlinear discrete
        assert line6 == "1 1 0 0 0"
        # no nonlinear vars at all
        assert nl.split("\n")[4].split("#")[0].strip() == "0 0 0"


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


def _nl_structure(nl: str) -> dict:
    """Extract the header/k/J/G structural counts from a .nl text.

    Returns a dict with:
      - ``header_jac_nz``  : header line-7 Jacobian nonzero count
      - ``header_nl_cons`` : header line-2 nonlinear-constraint count
      - ``j_lengths``      : {con_idx: declared J-block length}
      - ``j_entries``      : {con_idx: [(var_idx, coeff), ...]}  actual rows
      - ``k_cumulative``   : the k-section cumulative per-column counts
      - ``g_length``       : declared G0 length (or 0 if absent)
    """
    lines = [ln.split("#")[0].split("\t")[0].strip() for ln in nl.split("\n")]
    out: dict = {"j_lengths": {}, "j_entries": {}, "k_cumulative": [], "g_length": 0}
    out["header_nl_cons"] = int(lines[2].split()[0])
    out["header_jac_nz"] = int(lines[7].split()[0])
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("k") and ln[1:].isdigit():
            n = int(ln[1:])
            for j in range(n):
                out["k_cumulative"].append(int(lines[i + 1 + j]))
            i += n + 1
            continue
        if ln[:1] == "J" and len(ln) > 1 and ln[1].isdigit():
            ci, cnt = ln[1:].split()
            ci, cnt = int(ci), int(cnt)
            out["j_lengths"][ci] = cnt
            rows = []
            for j in range(cnt):
                vi, co = lines[i + 1 + j].split()
                rows.append((int(vi), float(co)))
            out["j_entries"][ci] = rows
            i += cnt + 1
            continue
        if ln[:1] == "G" and len(ln) > 1 and ln[1].isdigit():
            out["g_length"] = int(ln.split()[1])
        i += 1
    return out


class TestNLWriterJacobianConformance:
    """EX-1 (#413): ASL-conformant Jacobian structure.

    A variable appearing ONLY nonlinearly in a constraint must still get a
    (zero-coefficient) J entry, and the header nonzero count, the k
    cumulative-column-count section, and the J blocks must all be computed
    from that same union sparsity so they agree.
    """

    def _ex1_model(self):
        # exp(x) + y <= 5  (x appears ONLY nonlinearly);  x + y <= 3
        m = dm.Model("ex1")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x + y)
        m.subject_to(dm.exp(x) + y <= 5)
        m.subject_to(x + y <= 3)
        return m

    def test_nonlinear_only_var_gets_zero_j_entry(self):
        nl = self._ex1_model().to_nl()
        s = _nl_structure(nl)
        # C0 = exp(x)+y<=5: J block must list BOTH x (var 0, coeff 0) and y.
        assert 0 in s["j_entries"]
        c0 = dict(s["j_entries"][0])
        assert c0.get(0) == 0.0, f"x (var 0) missing its 0-coeff J entry: {s['j_entries'][0]}"
        assert 1 in c0  # y appears linearly
        assert s["j_lengths"][0] == 2

    def test_header_k_j_counts_agree(self):
        nl = self._ex1_model().to_nl()
        s = _nl_structure(nl)
        sum_j = sum(s["j_lengths"].values())
        # header nnz == total J entries emitted
        assert s["header_jac_nz"] == sum_j, (s["header_jac_nz"], sum_j)
        assert s["header_jac_nz"] == 4  # 2 (C0: x,y) + 2 (C1: x,y)
        # k-section: cumulative col-0 count == number of constraints touching x.
        # Both C0 (nonlinear x) and C1 (linear x) touch col 0 => 2.
        assert s["k_cumulative"][0] == 2, s["k_cumulative"]
        # Final implied column total (header) equals sum(J) — full agreement.
        assert s["k_cumulative"][-1] <= s["header_jac_nz"]

    def test_pure_linear_constraint_not_counted_nonlinear(self):
        # C1 (x+y<=3) is linear; only C0 is nonlinear.
        nl = self._ex1_model().to_nl()
        s = _nl_structure(nl)
        assert s["header_nl_cons"] == 1

    def test_constant_moves_to_rhs_body_is_n0(self):
        # The pure-constant body of x+y<=3 becomes an n0 body with rhs 3.
        nl = self._ex1_model().to_nl()
        assert "C1\nn0\n" in nl
        r_section = nl[nl.index("r\n") :]
        assert "1 5.0" in r_section  # exp(x)+y <= 5
        assert "1 3.0" in r_section  # x + y   <= 3

    def test_no_double_count_var_linear_and_nonlinear(self):
        # x appears both linearly and nonlinearly in the same constraint;
        # it must be counted ONCE in the header/J/k.
        m = dm.Model("mixed")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x + y)
        m.subject_to(x * x + x + y <= 5)  # x nonlinear (x*x) AND linear (+x)
        s = _nl_structure(m.to_nl())
        assert s["j_lengths"][0] == 2  # {x, y}, not 3
        assert s["header_jac_nz"] == 2

    def test_roundtrip_solve_matches(self, tmp_path):
        m = self._ex1_model()
        nl_path = tmp_path / "ex1.nl"
        m.to_nl(str(nl_path))
        m2 = dm.from_nl(str(nl_path))
        r1 = m.solve()
        r2 = m2.solve()
        assert r1.status == r2.status == "optimal"
        assert abs(r1.objective - r2.objective) < 1e-6

    @pytest.mark.parametrize(
        "kind",
        ["linear-only", "nl-only-var", "mixed-lin-nl", "obj-nonlinear"],
    )
    def test_structural_match_pyomo(self, kind):
        pyo = pytest.importorskip("pyomo.environ")
        import tempfile

        def build_discopt():
            m = dm.Model("t")
            x = m.continuous("x", lb=0.1, ub=5)
            y = m.continuous("y", lb=0, ub=5)
            if kind == "linear-only":
                m.minimize(x + 2 * y)
                m.subject_to(x + y <= 3)
            elif kind == "nl-only-var":
                m.minimize(x + y)
                m.subject_to(dm.exp(x) + y <= 5)
                m.subject_to(x + y <= 3)
            elif kind == "mixed-lin-nl":
                m.minimize(x + y)
                m.subject_to(x * x + x + y <= 5)
            else:  # obj-nonlinear
                m.minimize(dm.log(x) + y)
                m.subject_to(x + y <= 4)
            return m.to_nl()

        def build_pyomo():
            m = pyo.ConcreteModel()
            m.x = pyo.Var(bounds=(0.1, 5))
            m.y = pyo.Var(bounds=(0, 5))
            if kind == "linear-only":
                m.o = pyo.Objective(expr=m.x + 2 * m.y)
                m.c = pyo.Constraint(expr=m.x + m.y <= 3)
            elif kind == "nl-only-var":
                m.o = pyo.Objective(expr=m.x + m.y)
                m.c0 = pyo.Constraint(expr=pyo.exp(m.x) + m.y <= 5)
                m.c1 = pyo.Constraint(expr=m.x + m.y <= 3)
            elif kind == "mixed-lin-nl":
                m.o = pyo.Objective(expr=m.x + m.y)
                m.c = pyo.Constraint(expr=m.x * m.x + m.x + m.y <= 5)
            else:
                m.o = pyo.Objective(expr=pyo.log(m.x) + m.y)
                m.c = pyo.Constraint(expr=m.x + m.y <= 4)
            f = tempfile.mktemp(suffix=".nl")
            m.write(f, format="nl", io_options={"symbolic_solver_labels": False})
            return open(f).read()

        ds = _nl_structure(build_discopt())
        ps = _nl_structure(build_pyomo())
        assert ds["header_jac_nz"] == ps["header_jac_nz"]
        assert ds["header_nl_cons"] == ps["header_nl_cons"]
        assert ds["j_lengths"] == ps["j_lengths"]
        assert ds["k_cumulative"] == ps["k_cumulative"]
        assert ds["g_length"] == ps["g_length"]
        # J entries: same (var, coeff) sets per constraint.
        for ci in ds["j_entries"]:
            assert set(ds["j_entries"][ci]) == set(ps["j_entries"][ci]), ci


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


class TestNLWriterDeepNesting:
    """Deep, left-nested expression DAGs must export without recursion overflow.

    A model built with ``sum()`` over many terms produces a long chain of
    ``BinaryOp`` '+' nodes. The writer traverses the DAG iteratively (explicit
    work stacks), so export must succeed well past Python's recursion limit
    (~1000) and stay numerically faithful.
    """

    def test_deep_sum_objective_and_constraint_export(self):
        import sys

        n = 3 * sys.getrecursionlimit()  # comfortably past the recursion limit
        m = dm.Model("deep")
        xs = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(n)]
        m.minimize(sum((xs[i] - 0.5) ** 2 for i in range(n)))
        m.subject_to(sum(xs[i] for i in range(n)) >= 1)

        nl = m.to_nl()  # must not raise RecursionError
        assert nl is not None and nl.startswith("g3")

    def test_deep_sum_roundtrip_parse(self, tmp_path):
        try:
            from discopt._rust import parse_nl_file
        except ImportError:
            pytest.skip("Rust .nl parser not available")

        n = 2000
        m = dm.Model("deep")
        xs = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(n)]
        m.minimize(sum(xs))
        m.subject_to(sum(xs) <= 5)

        nl_path = tmp_path / "deep.nl"
        m.to_nl(str(nl_path))
        rep = parse_nl_file(str(nl_path))
        assert rep.n_vars == n
        assert rep.n_constraints == 1

    def test_deep_sum_values_match_compiled(self):
        """The exported nonlinear body must evaluate to the same value the
        compiled expression does (the iterative linear+nonlinear split stays
        faithful). A modest chain length keeps the JAX compiler used as the
        oracle below within its own recursion limit."""
        import numpy as np
        from discopt._jax.dag_compiler import compile_expression
        from discopt.export.nl import _NLWriter

        n = 150
        m = dm.Model("deep")
        xs = [m.continuous(f"x{i}", lb=-1, ub=1) for i in range(n)]
        m.minimize(sum((xs[i] - 0.5) ** 2 for i in range(n)))

        writer = _NLWriter(m)
        writer._build_var_map()
        writer._decompose_expressions()

        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        nl_part = (
            float(compile_expression(writer._obj_nonlinear, m)(x))
            if writer._obj_nonlinear is not None
            else 0.0
        )
        lin_part = sum(coeff * x[idx] for idx, coeff in writer._obj_linear.items())
        expected = float(compile_expression(m._objective.expression, m)(x))
        np.testing.assert_allclose(lin_part + nl_part, expected, atol=1e-9)
