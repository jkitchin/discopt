"""Tests for the discopt-as-a-GAMS-solver link (discopt.gams).

Covers the GAMS-library-free core: the nonlinear instruction translator, the
GMO-view -> Model translation, the solve wrapper, the status mapping, and the
solver registration files.  None of this requires a GAMS installation.
"""

from __future__ import annotations

import json
import sys

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.gams import (
    data_dir,
    gamsconfig_snippet,
    is_available,
    model_from_gmo,
    run_script,
    solve_view,
    status_to_gams,
    write_registration,
)
from discopt.gams.instructions import (
    FUNC_CODE,
    FUNC_NAME,
    GamsTranslationError,
    translate_instructions,
)
from discopt.gams.instructions import (
    GamsOpCode as Op,
)
from discopt.gams.register import render_gamsconfig
from discopt.modeling.core import Constant


# ── instruction translator ──────────────────────────────────────────────────
@pytest.fixture
def vc():
    m = dm.Model("t")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    return [x, y], [3.0, 2.0]


def test_func_code_table_is_consistent():
    # Spot-check known positions from the GAMS GamsFuncCode enum.
    assert FUNC_CODE["mapval"] == 0
    assert FUNC_CODE["sqr"] == 9
    assert FUNC_CODE["exp"] == 10
    assert FUNC_CODE["log"] == 11
    assert FUNC_CODE["log2"] == 50
    assert FUNC_CODE["rpower"] == 64
    assert FUNC_NAME[FUNC_CODE["vcpower"]] == "vcpower"


def test_translate_push_variable(vc):
    v, c = vc
    expr = translate_instructions([Op.nlPushV], [1], v, c)
    assert expr is v[0]


def test_translate_bilinear(vc):
    v, c = vc
    expr = translate_instructions([Op.nlPushV, Op.nlPushV, Op.nlMul], [1, 2, 0], v, c)
    assert repr(expr) == "(x * y)"


def test_translate_sum_of_squares(vc):
    v, c = vc
    ops = [Op.nlPushV, Op.nlCallArg1, Op.nlPushV, Op.nlCallArg1, Op.nlAdd]
    flds = [1, FUNC_CODE["sqr"], 2, FUNC_CODE["sqr"], 0]
    assert repr(translate_instructions(ops, flds, v, c)) == "((x ** 2) + (y ** 2))"


def test_translate_immediate_ops(vc):
    v, c = vc  # constants pool [3, 2]; 3*x + 2
    expr = translate_instructions([Op.nlPushV, Op.nlMulI, Op.nlAddI], [1, 1, 2], v, c)
    assert repr(expr) == "((x * 3) + 2)"


def test_translate_subtraction_and_division(vc):
    v, c = vc
    # x - y
    assert repr(translate_instructions([Op.nlPushV, Op.nlSubV], [1, 2], v, c)) == "(x - y)"
    # x / y
    assert repr(translate_instructions([Op.nlPushV, Op.nlDivV], [1, 2], v, c)) == "(x / y)"


def test_translate_unary_minus(vc):
    v, c = vc
    expr = translate_instructions([Op.nlPushV, Op.nlUMin], [1, 0], v, c)
    assert repr(expr) == "neg(x)"


def test_translate_muliadd(vc):
    v, c = vc  # y + x * const(1)=3  ->  (y + (x * 3))
    ops = [Op.nlPushV, Op.nlPushV, Op.nlMulIAdd]
    expr = translate_instructions(ops, [2, 1, 1], v, c)
    assert repr(expr) == "(y + (x * 3))"


def test_translate_binary_function_power(vc):
    v, c = vc  # rpower(x, y) -> x ** y
    ops = [Op.nlPushV, Op.nlPushV, Op.nlCallArg2]
    expr = translate_instructions(ops, [1, 2, FUNC_CODE["rpower"]], v, c)
    assert repr(expr) == "(x ** y)"


def test_translate_nary_min(vc):
    v, c = vc
    ops = [Op.nlPushV, Op.nlPushV, Op.nlFuncArgN, Op.nlCallArgN]
    expr = translate_instructions(ops, [1, 2, 2, FUNC_CODE["min"]], v, c)
    assert repr(expr) == "min(x, y)"


def test_translate_unary_functions(vc):
    v, c = vc
    for name in ("exp", "log", "sqrt", "sin", "cos", "tan"):
        expr = translate_instructions([Op.nlPushV, Op.nlCallArg1], [1, FUNC_CODE[name]], v, c)
        assert repr(expr).startswith(name)


def _value(expr, model, **point):
    """Evaluate ``expr`` at ``point`` (POUNCE tape; no JAX on this path)."""
    from discopt._point_eval import evaluate_at

    x = np.array([point[v.name] for v in model._variables], dtype=float)
    return float(np.asarray(evaluate_at(expr, model, x)))


# Values GAMS 53 computes for each intrinsic (``scalar v; v = f(...); display v;``).
# These are the definitions the link must reproduce; the sl*/sq* forms are NOT
# their base function past the threshold S, and entropy is -x*ln(x).
_GAMS_VALUES = [
    ("entropy", 0.5, (), 0.34657359),
    ("entropy", 2.0, (), -1.38629436),
    ("errf", 0.0, (), 0.5),
    ("errf", 1.0, (), 0.84134475),
    ("slrec", 0.5, (1.0,), 1.5),
    ("slrec", 2.0, (1.0,), 0.5),
    ("slrec", 1e-11, (), 1.9e10),
    ("sqrec", 0.5, (1.0,), 1.75),
    ("sqrec", 1e-11, (), 2.71e10),
    ("slexp", 3.0, (1.0,), 8.15484549),
    ("sqexp", 3.0, (1.0,), 13.59140914),
    ("sllog10", 0.5, (1.0,), -0.21714724),
    ("sqlog10", 0.5, (1.0,), -0.27143405),
    ("sllog10", 1e-151, (), -150.39087),
    ("centropy", 2.0, (1.0,), 1.38629436),
    ("centropy", 2.0, (1.0, 1.0), 0.81093022),
]


@pytest.mark.parametrize("name, x0, extra, want", _GAMS_VALUES)
def test_intrinsics_match_gams_values(name, x0, extra, want):
    """#1889 audit: entropy's sign, errorf, centropy's 3rd arg, the sl*/sq* forms."""
    from discopt.gams.instructions import _apply_func

    m = dm.Model("f")
    x = m.continuous("x", lb=-1e3, ub=1e3)
    expr = _apply_func(FUNC_CODE[name], [x, *(Constant(e) for e in extra)])
    assert _value(expr, m, x=x0) == pytest.approx(want, rel=1e-7)


@pytest.mark.parametrize("name", ["slrec", "sqrec", "sllog10", "sqlog10", "slexp", "sqexp"])
def test_smoothed_forms_equal_base_function_inside_threshold(name):
    from discopt.gams.instructions import _apply_func

    base = {"slrec": lambda v: 1 / v, "sqrec": lambda v: 1 / v}
    base.update({"sllog10": np.log10, "sqlog10": np.log10, "slexp": np.exp, "sqexp": np.exp})
    m = dm.Model("f")
    x = m.continuous("x", lb=-1e3, ub=1e3)
    expr = _apply_func(FUNC_CODE[name], [x, Constant(1.0)])
    inside = 0.5 if name.endswith("exp") else 2.0
    assert _value(expr, m, x=inside) == pytest.approx(base[name](inside), rel=1e-12)


def test_centropy_three_args_is_not_folded(vc):
    v, c = vc
    ops = [Op.nlPushV, Op.nlPushV, Op.nlPushV, Op.nlFuncArgN, Op.nlCallArgN]
    expr = translate_instructions(ops, [1, 2, 1, 3, FUNC_CODE["centropy"]], v, c)
    assert "centropy(centropy" not in repr(expr)


def test_only_min_max_fold_beyond_two_args(vc):
    v, c = vc
    ops = [Op.nlPushV, Op.nlPushV, Op.nlPushV, Op.nlFuncArgN, Op.nlCallArgN]
    with pytest.raises(GamsTranslationError):
        translate_instructions(ops, [1, 2, 1, 3, FUNC_CODE["power"]], v, c)


def test_slrec_solves_to_the_gams_optimum():
    # max slrec(x, 1) on [0.5, 2] is 1.5 at x = 0.5 (GAMS 53); the old 1/x
    # mapping gave 2.0.
    m = dm.Model("sg")
    x = m.continuous("x", lb=0.5, ub=2.0)
    m.maximize(
        translate_instructions(
            [Op.nlPushV, Op.nlPushI, Op.nlCallArg2], [1, 1, FUNC_CODE["slrec"]], [x], [1.0]
        )
    )
    result = m.solve(time_limit=30)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(1.5, abs=1e-6)


def test_translate_errors(vc):
    v, c = vc
    with pytest.raises(GamsTranslationError):
        translate_instructions([Op.nlPushV], [1, 2], v, c)  # length mismatch
    with pytest.raises(GamsTranslationError):
        translate_instructions([Op.nlAdd], [0], v, c)  # stack underflow
    with pytest.raises(GamsTranslationError):
        translate_instructions([Op.nlPushV], [99], v, c)  # var out of range
    with pytest.raises(GamsTranslationError):
        # unsupported function code (jdate)
        translate_instructions([Op.nlPushV, Op.nlCallArg1], [1, FUNC_CODE["jdate"]], v, c)


# ── GMO view -> Model ────────────────────────────────────────────────────────
class _FakeGmo:
    """In-memory GMO view: min x0^2 + x1^2 s.t. x0 + x1 >= 1, x in [0, 5]."""

    def __init__(self, discrete=False):
        self._discrete = discrete

    def name(self):
        return "fake"

    def num_vars(self):
        return 2

    def num_rows(self):
        return 1

    def minimize(self):
        return True

    def constants(self):
        return []

    def var_lower(self, j):
        return 0.0

    def var_upper(self, j):
        return 5.0

    def var_type(self, j):
        return 2 if self._discrete else 0

    def var_name(self, j):
        return f"x{j}"

    def var_level(self, j):
        return 0.0

    def obj_constant(self):
        return 0.0

    def obj_linear(self):
        return {}

    def obj_nl(self):
        ops = [Op.nlPushV, Op.nlCallArg1, Op.nlPushV, Op.nlCallArg1, Op.nlAdd]
        return ops, [1, FUNC_CODE["sqr"], 2, FUNC_CODE["sqr"], 0]

    def row_name(self, i):
        return "c1"

    def row_sense(self, i):
        return ">="

    def row_rhs(self, i):
        return 1.0

    def row_constant(self, i):
        return 0.0

    def row_linear(self, i):
        return {0: 1.0, 1: 1.0}

    def row_nl(self, i):
        return [], []


class _FakeGmoRawObj(_FakeGmo):
    """Mirror GMO's raw objective: the equation residual ``-f`` plus a -1 sign.

    GMO returns the objective *equation* residual (``objvar - f``), so its
    nonlinear instructions decode to ``-(x0^2 + x1^2)`` and the link recovers
    ``f`` via ``obj_nl_sign() == -1``.  The resulting model must match the
    direct-``f`` fake exactly.
    """

    def obj_nl(self):
        ops, flds = super().obj_nl()
        return [*ops, Op.nlUMin], [*flds, 0]  # append unary minus -> -(x0^2 + x1^2)

    def obj_nl_sign(self):
        return -1.0


def test_model_from_gmo_structure():
    m = model_from_gmo(_FakeGmo())
    assert [v.name for v in m._variables] == ["x0", "x1"]
    assert len(m._constraints) == 1
    assert m._objective is not None


def test_obj_nl_sign_recovers_objective():
    # The raw-residual fake (-f, sign -1) must solve to the same optimum as the
    # direct-f fake: min x0^2 + x1^2 s.t. x0 + x1 >= 1  ->  obj = 0.5.
    _, result = solve_view(_FakeGmoRawObj())
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.5, abs=1e-4)


def test_obj_nl_sign_defaults_to_one_when_absent():
    from discopt.gams.gmo_translate import _obj_nl_sign

    assert _obj_nl_sign(_FakeGmo()) == 1.0  # no obj_nl_sign attribute -> 1.0
    assert _obj_nl_sign(_FakeGmoRawObj()) == -1.0


def test_model_from_gmo_discrete_types():
    m = model_from_gmo(_FakeGmo(discrete=True))
    assert all(v.var_type == dm.VarType.INTEGER for v in m._variables)


class _FakeGmoCols(_FakeGmo):
    """One-row fake whose columns, bounds and row sense are set per test.

    ``min sum_j (x_j - target_j)^2`` subject to one row over all columns.
    """

    def __init__(self, cols, targets, sense=">=", rhs=-1e9):
        super().__init__()
        self._cols, self._targets, self._sense, self._rhs = cols, targets, sense, rhs

    def num_vars(self):
        return len(self._cols)

    def var_type(self, j):
        return self._cols[j][0]

    def var_lower(self, j):
        return self._cols[j][1]

    def var_upper(self, j):
        return self._cols[j][2]

    def constants(self):
        return list(self._targets)

    def obj_nl(self):
        ops, flds = [], []
        for j in range(len(self._cols)):
            ops += [Op.nlPushV, Op.nlSubI, Op.nlCallArg1]
            flds += [j + 1, j + 1, FUNC_CODE["sqr"]]
            if j:
                ops.append(Op.nlAdd)
                flds.append(0)
        return ops, flds

    def row_sense(self, i):
        return self._sense

    def row_rhs(self, i):
        return self._rhs

    def row_linear(self, i):
        return {j: 1.0 for j in range(len(self._cols))}


def test_fixed_binary_keeps_its_bounds():
    # x.fx = 1 on a binary; min (x - 0)^2 is 1, not 0 (#1889 audit).
    _, r = solve_view(_FakeGmoCols([(1, 1.0, 1.0)], [0.0]))
    assert r.objective == pytest.approx(1.0, abs=1e-6)


def test_integer_infinite_upper_bound_is_not_clamped():
    m = model_from_gmo(_FakeGmoCols([(2, 0.0, 1e300)], [0.0]))
    assert float(m._variables[0].ub) >= 1e19  # was clamped to 1e9


@pytest.mark.parametrize("vtype", [5, 6], ids=["semicont", "semiint"])
def test_semicontinuous_is_lowered_exactly(vtype):
    # x = 0 or 2 <= x <= 5; min (x - 0.5)^2 -> 0.25 at x = 0 (GAMS 53), where
    # the old continuous relaxation returned 2.25 at x = 2.
    model, r = solve_view(_FakeGmoCols([(vtype, 2.0, 5.0)], [0.5]))
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.25, abs=1e-6)
    assert r.x["x0"] == pytest.approx(0.0, abs=1e-6)
    assert [v.name for v in model._gams_columns] == ["x0"]


def test_semicontinuous_columns_keep_their_order():
    # Two semicontinuous columns with a continuous one between: the on/off
    # binaries must come after every GMO column, never between them.
    model = model_from_gmo(
        _FakeGmoCols([(5, 2.0, 5.0), (0, 0.0, 10.0), (5, 3.0, 4.0)], [0.5, 7.0, 3.5])
    )
    names = [v.name for v in model._variables]
    assert names[:3] == ["x0", "x1", "x2"]
    assert [v.name for v in model._gams_columns] == ["x0", "x1", "x2"]


def test_semicontinuous_with_infinite_bound_is_refused():
    with pytest.raises(GamsTranslationError, match="infinite bound"):
        model_from_gmo(_FakeGmoCols([(5, 2.0, 1e300)], [0.0]))


@pytest.mark.parametrize("vtype", [3, 4], ids=["sos1", "sos2"])
def test_sos_members_are_refused(vtype):
    with pytest.raises(GamsTranslationError, match="SOS"):
        model_from_gmo(_FakeGmoCols([(vtype, 0.0, 1.0)], [0.0]))


def test_free_row_imposes_nothing():
    # An =N= row (sense "free") is dropped, so min (x - 0)^2 stays 0 even though
    # the row would read x >= 3 under any other sense.
    m = model_from_gmo(_FakeGmoCols([(0, 0.0, 5.0)], [0.0], sense="free", rhs=3.0))
    assert len(m._constraints) == 0


def test_failing_obj_nl_sign_is_not_swallowed():
    class _Broken(_FakeGmoRawObj):
        def obj_nl_sign(self):
            raise RuntimeError("gmoObjJacVal failed")

    with pytest.raises(RuntimeError, match="gmoObjJacVal"):
        model_from_gmo(_Broken())


def test_solve_view_end_to_end():
    model, result = solve_view(_FakeGmo())
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.5, abs=1e-4)
    assert result.x["x0"] == pytest.approx(0.5, abs=1e-3)
    assert result.x["x1"] == pytest.approx(0.5, abs=1e-3)


# ── status mapping ───────────────────────────────────────────────────────────
def test_status_to_gams_optimal_continuous():
    # A certified global optimum requires a finite dual bound (SolveResult's
    # __post_init__ downgrades gap_certified when bound is None/non-finite).
    res = dm.SolveResult(status="optimal", objective=1.0, bound=1.0, x={"x": 0.0})
    assert status_to_gams(res, has_discrete=False) == (1, 1)  # Optimal, Normal


def test_status_to_gams_optimal_discrete():
    # A certified optimum is GAMS's 1 (proven optimal) for any problem class;
    # 8 (Integer Solution) is kept for a discrete point that is not certified.
    res = dm.SolveResult(status="optimal", objective=1.0, bound=1.0, x={"x": 0.0})
    assert status_to_gams(res, has_discrete=True) == (1, 1)


def test_status_to_gams_certified_global_default():
    # A finite bound keeps gap_certified True -> certified global optimum.
    res = dm.SolveResult(status="optimal", objective=1.0, bound=1.0, x={"x": 0.0})
    assert res.gap_certified is True
    assert status_to_gams(res, has_discrete=False) == (1, 1)  # Optimal


def test_status_to_gams_uncertified_optimal_is_local():
    # Heuristic NLP-BB on a nonconvex model: locally optimal only, not global.
    res = dm.SolveResult(status="optimal", objective=1.0, x={"x": 0.0}, gap_certified=False)
    # LocallyOptimal (2) is GAMS's NLP code; a discrete model's uncertified
    # point is an Integer Solution (8). Neither is Optimal (1).
    assert status_to_gams(res, has_discrete=False) == (2, 1)
    assert status_to_gams(res, has_discrete=True) == (8, 1)


def test_status_to_gams_infeasible():
    res = dm.SolveResult(status="infeasible")
    assert status_to_gams(res, has_discrete=False) == (4, 1)
    assert status_to_gams(res, has_discrete=True) == (10, 1)  # Integer Infeasible


@pytest.mark.parametrize("has_discrete", [False, True])
def test_status_to_gams_local_statuses(has_discrete):
    """The #1148 local outcomes used to fall through to 'error, no solution'."""
    local = dm.SolveResult(status="local_optimal", objective=1.0, x={"x": 0.0})
    assert status_to_gams(local, has_discrete) == ((8 if has_discrete else 2), 1)
    limit = dm.SolveResult(status="local_limit", objective=1.0, x={"x": 0.0})
    assert status_to_gams(limit, has_discrete) == (6, 2)  # intermediate infeasible
    assert status_to_gams(dm.SolveResult(status="local_infeasible"), has_discrete) == (5, 1)


def test_status_to_gams_time_limit_with_and_without_solution():
    with_sol = dm.SolveResult(status="time_limit", x={"x": 0.0})
    assert status_to_gams(with_sol, has_discrete=False) == (7, 3)  # Feasible, Resource
    no_sol = dm.SolveResult(status="time_limit")
    assert status_to_gams(no_sol, has_discrete=False) == (14, 3)  # NoSolution, Resource


def test_status_to_gams_error():
    res = dm.SolveResult(status="error")
    assert status_to_gams(res, has_discrete=False) == (13, 10)


# ── registration ─────────────────────────────────────────────────────────────
def test_gamsconfig_snippet_contains_solver_and_types():
    snip = gamsconfig_snippet("discopt-gams")
    assert "- discopt:" in snip  # solver name is the mapping key (per schema)
    assert "scriptName: discopt-gams" in snip
    assert "MINLP" in snip and "NLP" in snip


def test_gamsconfig_snippet_is_schema_valid():
    import yaml  # pyyaml is a transitive dep via the GAMS toolchain/tests

    snip = gamsconfig_snippet("/abs/discopt-gams")
    doc = yaml.safe_load(snip)
    entry = doc["solverConfig"][0]["discopt"]
    assert entry["scriptName"] == "/abs/discopt-gams"
    assert "MINLP" in entry["modelTypes"]
    # No empty/invalid fields that the GAMS schema rejects (licCodes minLength 2,
    # library requires non-empty libName) -- those keys must be absent entirely.
    assert "licCodes" not in entry and "library" not in entry


def test_run_script_invokes_link():
    script = run_script("/usr/bin/python3")
    assert script.startswith("#!/bin/sh")
    assert "discopt.gams.link" in script
    assert '"$@"' in script


def test_gamsconfig_requests_gmo_dictionary():
    """#1972: without ``dictType`` GAMS writes no ``gamsdict.dat`` and every
    name lookup prints "gmoDict failed"; both registration paths must set it."""
    import yaml

    entry = yaml.safe_load(gamsconfig_snippet("/abs/discopt-gams"))["solverConfig"][0]
    assert entry["discopt"]["dictType"] == 5
    text, _ = render_gamsconfig(None, "/abs/discopt-gams")
    assert yaml.safe_load(text)["solverConfig"][0]["discopt"]["dictType"] == 5


def test_run_script_launches_main_without_runpy_warning(tmp_path):
    """#1972: ``-m discopt.gams.link`` triggers runpy's "found in sys.modules"
    RuntimeWarning (the package imports ``.link``); the launcher must not."""
    import shlex
    import subprocess

    script = run_script(sys.executable)
    assert " -m discopt.gams.link" not in script
    exec_line = next(ln for ln in script.splitlines() if ln.startswith("exec "))
    argv = shlex.split(exec_line)[1:-1] + ["--help"]
    # -W error turns the runpy RuntimeWarning into a failure.
    proc = subprocess.run(
        [argv[0], "-W", "error::RuntimeWarning", *argv[1:]],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: discopt-gams" in proc.stderr
    assert "RuntimeWarning" not in proc.stderr


class _FakeGmoModule:
    """Just enough of the gamsapi GMO module for ``_GmoAdapter``'s name lookups."""

    def __init__(self, dict_path):
        self.dict_path = dict_path
        self.name_calls = 0

    def gmoN(self, h):
        return 2

    def gmoNLConst(self, h):
        return 0

    def gmoNLCodeSize(self, h):
        return 0

    def gmoNameDict(self, h):
        return self.dict_path

    def gmoGetVarNameOne(self, h, j):
        self.name_calls += 1
        return f"x_{j}"

    def gmoGetEquNameOne(self, h, i):
        self.name_calls += 1
        return f"e_{i}"


@pytest.mark.parametrize("has_file", [True, False])
@pytest.mark.parametrize("set_path", [True, False])
def test_gmo_adapter_skips_name_lookups_without_dictionary(tmp_path, has_file, set_path):
    """#1972: with no dictionary file (``option dictfile=0;``) the adapter must
    not call ``gmoGet*NameOne`` at all -- libgmo prints "gmoDict failed" per call."""
    from discopt.gams.link import _GmoAdapter

    dict_file = tmp_path / "gamsdict.dat"
    if has_file:
        dict_file.write_bytes(b"")
    gmo = _FakeGmoModule(str(dict_file) if set_path else "")
    ad = _GmoAdapter(object(), gmo)
    if has_file and set_path:
        assert (ad.var_name(1), ad.row_name(0)) == ("x_1", "e_0")
        assert gmo.name_calls == 2
    else:
        assert (ad.var_name(1), ad.row_name(0)) == ("", "")
        assert gmo.name_calls == 0


def test_parse_gams_args_prefers_control_file_over_param_file(tmp_path):
    """GAMS calls the script as ``<scrdir> <workdir> <prm> <cntr> <sysdir> <name>``.

    The control file must be picked specifically (``gamscntr*``); a regression
    guard against selecting the parameter file ``gmsprmun.dat`` -- which has the
    same ``.dat`` suffix and appears earlier in the argument list -- and pointing
    ``gevInitEnvironmentLegacy`` at the wrong scratch directory (loads an empty
    model and segfaults on the first bound query).
    """
    from discopt.gams.link import _parse_gams_args

    scr = tmp_path / "225a"
    scr.mkdir()
    prm = scr / "gmsprmun.dat"
    cntr = scr / "gamscntr.dat"
    sysdir = tmp_path / "sys"
    sysdir.mkdir()
    for p in (prm, cntr):
        p.write_text("x")
    (sysdir / "gmscmpun.txt").write_text("x")

    args = [str(scr) + "/", str(tmp_path) + "/", str(prm), str(cntr), str(sysdir), "DISCOPT"]
    control_file, found_sys = _parse_gams_args(args)
    assert control_file == str(cntr)
    assert found_sys == str(sysdir)


def test_parse_gams_args_single_control_file(tmp_path):
    from discopt.gams.link import _parse_gams_args

    cntr = tmp_path / "gamscntr.dat"
    cntr.write_text("x")
    control_file, found_sys = _parse_gams_args([str(cntr)])
    assert control_file == str(cntr)
    assert found_sys is None


def test_write_registration(tmp_path):
    written = write_registration(tmp_path)
    assert written["config"].exists()
    assert written["script"].exists()
    assert written["action"] == "created"
    # script should be executable
    import os

    assert os.access(written["script"], os.X_OK)
    assert "solverConfig" in written["config"].read_text()


def test_register_merges_into_existing_gamsconfig(tmp_path):
    import yaml

    # Pre-existing config: another solver + an unrelated top-level key.
    cfg = tmp_path / "gamsconfig.yaml"
    cfg.write_text(
        "defaultSolvers:\n  - NLP: conopt\n"
        "solverConfig:\n  - mysolver:\n      scriptName: /x/mysolver\n"
        "      modelTypes:\n        - NLP\n"
    )
    written = write_registration(tmp_path)
    assert written["action"] == "merged"
    doc = yaml.safe_load(cfg.read_text())
    names = [next(iter(item)) for item in doc["solverConfig"]]
    assert names == ["mysolver", "discopt"]  # existing preserved, discopt added
    assert "defaultSolvers" in doc  # other top-level keys preserved


def test_register_is_idempotent(tmp_path):
    import yaml

    write_registration(tmp_path)
    written = write_registration(tmp_path)  # second run
    assert written["action"] == "replaced"
    doc = yaml.safe_load((tmp_path / "gamsconfig.yaml").read_text())
    names = [next(iter(item)) for item in doc["solverConfig"]]
    assert names.count("discopt") == 1  # no duplicate entry


def test_check_gamsapi_returns_bool_and_message():
    from discopt.gams import check_gamsapi

    ok, message = check_gamsapi()
    assert isinstance(ok, bool)
    assert isinstance(message, str) and "gamsapi" in message


def test_is_available_returns_bool():
    assert isinstance(is_available(), bool)


# ── packaged .gms smoke corpus ───────────────────────────────────────────────
# Small GAMS models with known optima, shipped inside the package
# (discopt/gams/data/) so `discopt gams-verify` works for pip users. Here we
# exercise the same corpus through discopt's own from_gams() reader.
_GAMS_DATA = data_dir()
_MANIFEST = json.loads((_GAMS_DATA / "manifest.json").read_text())["models"]


@pytest.mark.parametrize("entry", _MANIFEST, ids=[m["file"] for m in _MANIFEST])
def test_smoke_gms_file_parses(entry):
    model = dm.from_gams(str(_GAMS_DATA / entry["file"]))
    assert len(model._variables) >= 1
    assert model._objective is not None


@pytest.mark.parametrize(
    "entry",
    [m for m in _MANIFEST if m.get("via_from_gams")],
    ids=[m["file"] for m in _MANIFEST if m.get("via_from_gams")],
)
def test_smoke_gms_optimum_via_from_gams(entry):
    model = dm.from_gams(str(_GAMS_DATA / entry["file"]))
    # Keep the solver budget well under pytest's --timeout=120: nonconvex
    # instances (e.g. Rosenbrock) never close their global gap, so they run to
    # the full time_limit. The correct incumbent is found in the first seconds;
    # the remaining time is pure gap-closing effort, so a short budget is enough
    # to validate the optimum without tripping the harness timeout.
    #
    # "A short budget" was 30s, which for nlp_rosenbrock meant 30s of exactly
    # the gap-closing effort this comment calls pointless — it was the single
    # most expensive case in this file. Every other entry certifies in well
    # under a second, so 5s only ever binds on the instances the comment is
    # about, and on those the incumbent is already final.
    result = model.solve(time_limit=5)
    # Nonconvex instances may return "feasible" (the solver declines to certify a
    # global gap); either way the reported optimum must match the known value.
    assert result.status in ("optimal", "feasible")
    assert result.objective == pytest.approx(entry["objective"], abs=max(entry["tol"], 1e-4))


def test_translate_entropy_signpower_centropy(vc):
    v, c = vc
    e = translate_instructions([Op.nlPushV, Op.nlCallArg1], [1, FUNC_CODE["entropy"]], v, c)
    assert repr(e) == "neg(entropy(x))"  # GAMS entropy is -x*ln(x)
    sp = translate_instructions(
        [Op.nlPushV, Op.nlPushV, Op.nlCallArg2], [1, 2, FUNC_CODE["signpower"]], v, c
    )
    assert repr(sp) == "signpower(x, y)"
    ce = translate_instructions(
        [Op.nlPushV, Op.nlPushV, Op.nlCallArg2], [1, 2, FUNC_CODE["centropy"]], v, c
    )
    assert repr(ce) == "(x * log(((x + 1e-20) / (y + 1e-20))))"


def test_translate_rejects_discontinuous(vc):
    v, c = vc
    for name in ("ceil", "floor", "round", "trunc", "frac", "mod"):
        with pytest.raises(GamsTranslationError, match="discontinuous"):
            translate_instructions([Op.nlPushV, Op.nlCallArg1], [1, FUNC_CODE[name]], v, c)


def test_entropy_solves_globally():
    # GAMS entropy(x) = -x*ln(x); max on [0.1, 3] -> 1/e at x = 1/e. The
    # translation negates discopt's convex x*log(x) node, so the solve is still
    # a *certified* global optimum.
    import math

    m = dm.Model("ent")
    x = m.continuous("x", lb=0.1, ub=3.0)
    expr = translate_instructions([Op.nlPushV, Op.nlCallArg1], [1, FUNC_CODE["entropy"]], [x], [])
    m.maximize(expr)
    result = m.solve(time_limit=30)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(1.0 / math.e, abs=1e-4)
    assert result.gap_certified


def test_no_cross_problem_contamination():
    """Solving other problems in the same process must not perturb a result.

    The daemon reuses one warm interpreter across solves; this guards against
    shared mutable state / order-dependence (each solve builds a fresh Model).
    """

    def solve(name: str) -> float:
        m = dm.from_gams(str(_GAMS_DATA / name))
        return m.solve(time_limit=120).objective

    first = solve("minlp_exp.gms")
    solve("nlp_circle.gms")  # a different problem in between
    solve("mip_knapsack.gms")  # ... and another
    again = solve("minlp_exp.gms")
    # Bit-for-bit identical: the solve is a pure function of the model, with no
    # leakage from the intervening solves.
    assert again == first
