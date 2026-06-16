"""Tests for the discopt-as-a-GAMS-solver link (discopt.gams).

Covers the GAMS-library-free core: the nonlinear instruction translator, the
GMO-view -> Model translation, the solve wrapper, the status mapping, and the
solver registration files.  None of this requires a GAMS installation.
"""

from __future__ import annotations

import json
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt.gams import (
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


def test_solve_view_end_to_end():
    model, result = solve_view(_FakeGmo())
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.5, abs=1e-4)
    assert result.x["x0"] == pytest.approx(0.5, abs=1e-3)
    assert result.x["x1"] == pytest.approx(0.5, abs=1e-3)


# ── status mapping ───────────────────────────────────────────────────────────
def test_status_to_gams_optimal_continuous():
    res = dm.SolveResult(status="optimal", objective=1.0, x={"x": 0.0})
    assert status_to_gams(res, has_discrete=False) == (1, 1)  # Optimal, Normal


def test_status_to_gams_optimal_discrete():
    res = dm.SolveResult(status="optimal", objective=1.0, x={"x": 0.0})
    assert status_to_gams(res, has_discrete=True) == (8, 1)  # Integer, Normal


def test_status_to_gams_infeasible():
    res = dm.SolveResult(status="infeasible")
    assert status_to_gams(res, has_discrete=False) == (4, 1)


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
    # script should be executable
    import os

    assert os.access(written["script"], os.X_OK)
    assert "solverConfig" in written["config"].read_text()


def test_is_available_returns_bool():
    assert isinstance(is_available(), bool)


# ── on-disk .gms smoke corpus ────────────────────────────────────────────────
# Small GAMS models with known optima (python/tests/data/gams/). The GAMS-side
# solver link is verified end-to-end by scripts/verify_gams_link.py; here we
# exercise the same corpus through discopt's own from_gams() reader.
_GAMS_DATA = Path(__file__).parent / "data" / "gams"
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
    result = model.solve(time_limit=120)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(entry["objective"], abs=max(entry["tol"], 1e-4))


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
