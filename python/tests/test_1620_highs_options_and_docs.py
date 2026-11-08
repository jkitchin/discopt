"""Regression tests for #1620: the HiGHS route's options/log, and API-shape fixes.

* ``Model.solve(highs_options=...)`` reaches HiGHS on the LP/MILP route: the log
  can be switched on, search options pass through, options the certificate
  depends on are refused loudly, and a solve that never reaches HiGHS warns that
  the options were not used.
* ``TreeFormulation`` / ``add_predictor`` default ``split_eps`` to the 1e-5 that
  #1447 established (was 1e-6, equal to the feasibility tolerance).
* ``DAEBuilder.integral()`` before ``discretize()`` raises a clear error instead
  of a bare ``KeyError``.
* OA's "generating OA cuts for k of n rows" is INFO, not WARNING.
* The documented constraint-dual sign convention is the one the HiGHS route
  returns (``>=`` rows carry ``mu >= 0``).
"""

from __future__ import annotations

import inspect
import logging
import warnings

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _small_milp():
    m = dm.Model("milp1620")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.maximize(3 * x + 2 * y)
    m.subject_to(2 * x + y <= 7.5, name="c1")
    m.subject_to(x + 3 * y <= 9.5, name="c2")
    return m


# --------------------------------------------------------------- highs_options
def test_highs_log_is_off_by_default(capfd):
    r = _small_milp().solve()
    assert r.status == "optimal"
    out = capfd.readouterr().out
    assert "HiGHS" not in out, out


@pytest.mark.requires_highs
def test_highs_options_output_flag_prints_the_highs_log(capfd):
    r = _small_milp().solve(highs_options={"output_flag": True})
    assert r.status == "optimal"
    out = capfd.readouterr().out
    assert "HiGHS" in out, "highs_options={'output_flag': True} printed no HiGHS log"


@pytest.mark.requires_highs
def test_highs_options_search_option_passes_through_and_keeps_the_answer():
    base = _small_milp().solve()
    opt = _small_milp().solve(highs_options={"mip_detect_symmetry": False, "presolve": "off"})
    assert base.status == opt.status == "optimal"
    assert opt.objective == pytest.approx(base.objective, abs=1e-9)
    assert opt.gap_certified


@pytest.mark.requires_highs
def test_highs_options_applies_to_the_lp_route(capfd, monkeypatch):
    # highs-parity D1: an LP defaults to the Rust route; this pins the HiGHS route.
    monkeypatch.setenv("DISCOPT_LP_BACKEND", "highs")
    m = dm.Model("lp1620")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.minimize(2 * a + 3 * b)
    m.subject_to(a + b >= 4, name="ge")
    r = m.solve(highs_options={"output_flag": True})
    assert r.status == "optimal"
    assert "HiGHS" in capfd.readouterr().out


@pytest.mark.parametrize(
    "key",
    ["mip_rel_gap", "time_limit", "threads", "mip_feasibility_tolerance", "presolve_rule_off"],
)
def test_highs_options_reserved_key_is_refused(key):
    with pytest.raises(ValueError, match="cannot be set"):
        _small_milp().solve(highs_options={key: 1})


@pytest.mark.parametrize(
    "key",
    [
        "no_such_highs_option",
        # Real HiGHS options a denylist let through (review of #1648): they change
        # the problem solved, the algorithm's exactness, or write files.
        "solver",
        "solve_relaxation",
        "user_objective_scale",
        "user_bound_scale",
        "allow_unbounded_or_infeasible",
        "parallel",
        "mip_root_presolve_only",
        "write_solution_to_file",
    ],
)
def test_highs_options_outside_the_allowlist_is_refused(key):
    with pytest.raises(ValueError, match="only logging and search-strategy options"):
        _small_milp().solve(highs_options={key: 1})


@pytest.mark.requires_highs
def test_highs_options_bad_value_for_an_allowed_option_is_refused_by_highs():
    with pytest.raises(RuntimeError, match="HiGHS rejected option"):
        _small_milp().solve(highs_options={"presolve": "no_such_value"})


@pytest.mark.requires_highs
def test_highs_options_allowlist_names_are_real_highs_options():
    import highspy
    from discopt.solvers.lp_milp_highs import ALLOWED_USER_OPTIONS

    h = highspy.Highs()
    for name in sorted(ALLOWED_USER_OPTIONS):
        status, _ = h.getOptionValue(name)
        assert status == highspy.HighsStatus.kOk, name


@pytest.mark.requires_highs
def test_highs_options_on_a_non_highs_route_warns():
    m = dm.Model("nlp1620")
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize((x - 1) ** 2)
    with pytest.warns(UserWarning, match="highs_options was not used"):
        r = m.solve(highs_options={"output_flag": True})
    assert r.status == "optimal"


@pytest.mark.requires_highs
def test_highs_options_used_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        r = _small_milp().solve(highs_options={"mip_detect_symmetry": False})
    assert r.status == "optimal"


# ------------------------------------------------------------- tree split_eps
def test_tree_formulation_split_eps_default_matches_1447():
    from discopt.ml.formulations.base import TreeFormulation
    from discopt.ml.formulations.tree_ensemble import TreeEnsembleFormulation

    outer = inspect.signature(TreeFormulation.__init__).parameters["split_eps"].default
    inner = inspect.signature(TreeEnsembleFormulation.__init__).parameters["split_eps"].default
    assert inner == 1e-5
    assert outer == inner


def test_add_predictor_uses_the_safe_split_eps():
    from discopt.ml.predictor import add_predictor
    from discopt.ml.tree import DecisionTree, TreeEnsembleDefinition

    tree = DecisionTree(
        n_features=1,
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, 1.0, 2.0]),
    )
    ens = TreeEnsembleDefinition(trees=[tree], n_features=1, input_bounds=(np.zeros(1), np.ones(1)))
    m = dm.Model("tree1620")
    x = m.continuous("x", shape=(1,), lb=0.0, ub=1.0)
    _out, form = add_predictor(m, x, ens)
    assert form._split_eps == 1e-5


# ---------------------------------------------------------- DAE integral order
def test_dae_integral_before_discretize_is_a_clear_error():
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model("dae1620")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=2, ncp=2)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=0.0, bounds=(-10, 10))
    dae.set_ode(lambda t, s, a, c: {"x": 1.0})
    with pytest.raises(RuntimeError, match="discretize"):
        dae.integral(lambda t, s, a, c: s["x"] ** 2)


# ------------------------------------------------------------ OA log severity
def test_oa_partial_cut_mask_logs_at_info_not_warning(caplog):
    m = dm.Model("oa1620")
    x = m.continuous("x", lb=-3, ub=3)
    z = m.binary("z")
    m.minimize(x**2 + z)
    m.subject_to(x >= 1 - 2 * z, name="lin")  # affine: already exact in the master
    m.subject_to(x**2 <= 4, name="conv")  # convex: gets OA cuts
    with caplog.at_level(logging.INFO, logger="discopt"):
        r = m.solve(solver="mip-nlp", mip_nlp_method="oa")
    assert r.status == "optimal"
    hits = [rec for rec in caplog.records if "generating OA cuts" in rec.getMessage()]
    assert hits, "the OA cut-mask message was not logged at all (probe did not fire)"
    assert all(rec.levelno == logging.INFO for rec in hits), [rec.levelname for rec in hits]


# ------------------------------------------------------------ .nl column order
def test_nl_column_order_maps_nl_columns_to_flat_variables():
    from discopt.export import nl_column_order

    m = dm.Model("nlcol1620")
    x = m.continuous("x", lb=0, ub=10)
    z = m.integer("z", lb=0, ub=5)
    y = m.continuous("y", shape=(2,), lb=-5, ub=5)
    m.minimize(x + z + y[0] ** 2 + y[1] ** 2)
    m.subject_to(x + y[0] + z >= 1)
    order = nl_column_order(m)
    # .nl puts nonlinear columns (y) first, then linear continuous, then integer.
    assert order.tolist() == [2, 3, 0, 1]
    # Round trip: the initial-point segment of to_nl lists values in this order.
    flat = np.array([3.0, 2.0, -4.0, 4.0])  # discopt flat order: x, z, y0, y1
    txt = m.to_nl(initial_point={x: 3.0, z: 2.0, y: [-4.0, 4.0]})
    seg = txt.split("\nx4\n", 1)[1].splitlines()[:4]
    by_col = {int(s.split()[0]): float(s.split()[1]) for s in seg}
    assert [by_col[j] for j in range(4)] == flat[order].tolist()


# ------------------------------------------------------- dual sign convention
def test_documented_dual_convention_ge_row_is_nonnegative():
    m = dm.Model("dual1620")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.minimize(2 * a + 3 * b)
    m.subject_to(a + b >= 4, name="ge")
    r = m.solve()
    assert float(r.constraint_duals["ge"]) == pytest.approx(2.0, abs=1e-7)
    doc = dm.SolveResult.__doc__ or ""
    assert "constraint_duals" in doc and "mu >= 0" in doc
