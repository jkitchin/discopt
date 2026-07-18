"""Unit + functional coverage for ``discopt.solvers.oa`` internals (#87).

Two layers:

* ``unit``: pure helpers (hook-payload validators, option normalizers, the
  feasibility-merit evaluator, MIP-start extension, integer-binary expansion,
  reduction cuts) checked against hand-computed values.
* ``smoke``: tiny convex MINLPs with closed-form optima driven through the
  option surface of ``solve_oa`` / ``solve_goa`` / ``solve_feasibility_pump``
  (external hooks, ECP cut senses, regularized masters, FP variants, SHOT
  profile toggles, limit paths). Options may change the path, never the
  certificate: every solve asserts the known optimum or a sound non-optimal
  status (no false certificate).

Reference instance: min (x-0.7)^2 + 1.3*i  s.t. x + i >= 2.5, i in {0..4},
x in [0,4]. For each i, optimal x = max(0.7, 2.5-i); objectives by i:
i=0 -> 1.8^2 = 3.24; i=1 -> 0.8^2+1.3 = 1.94; i=2 -> 0+2.6 = 2.6;
i>=3 -> 1.3*i >= 3.9. Optimum 1.94 at (i,x) = (1, 1.5).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import discopt.solvers.oa as oa
import numpy as np
import pytest

_OPT = 1.94  # (1.5-0.7)^2 + 1.3*1 at i=1, x=1.5 (see module docstring)
_TOL = 1e-4


def _cvx_minlp():
    m = dm.Model("oa_units")
    i = m.integer("i", lb=0, ub=4)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x + i >= 2.5)
    m.minimize((x - 0.7) ** 2 + 1.3 * i)
    return m


def _assert_no_false_certificate(res):
    """Certificate invariants for the reference instance (min sense).

    Any reported dual bound must not exceed the true optimum 1.94; any
    feasible incumbent objective must be >= 1.94 - tol; 'optimal' may only
    be claimed at the true optimum.
    """
    if res.bound is not None:
        assert res.bound <= _OPT + _TOL
    if res.objective is not None:
        assert res.objective >= _OPT - _TOL
    if res.status == "optimal":
        assert res.objective == pytest.approx(_OPT, abs=_TOL)


# ── unit: hook-payload validators ─────────────────────────────


@pytest.mark.unit
def test_normalize_optional_hook_accepts_callable_rejects_other():
    fn = lambda ctx: None  # noqa: E731
    assert oa._normalize_optional_hook("h", fn) is fn
    assert oa._normalize_optional_hook("h", None) is None
    with pytest.raises(ValueError, match="must be callable"):
        oa._normalize_optional_hook("h", 42)


@pytest.mark.unit
def test_finite_hook_float_rejects_non_numeric_and_non_finite():
    # 2.5 passes through exactly.
    assert oa._finite_hook_float("v", 2.5) == 2.5
    with pytest.raises(ValueError, match="finite number"):
        oa._finite_hook_float("v", "not-a-number")
    with pytest.raises(ValueError, match="finite"):
        oa._finite_hook_float("v", float("inf"))


@pytest.mark.unit
def test_external_hook_items_variants():
    # None -> empty list; a Mapping is a single payload; iterables listed.
    assert oa._external_hook_items(None, hook_name="h", item_name="cut") == []
    payload = {"rhs": 1.0}
    assert oa._external_hook_items(payload, hook_name="h", item_name="cut") == [payload]
    assert oa._external_hook_items([payload], hook_name="h", item_name="cut") == [payload]
    with pytest.raises(ValueError, match="iterable"):
        oa._external_hook_items(3.14, hook_name="h", item_name="cut")


@pytest.mark.unit
def test_validate_external_primal_candidates_accept_shapes():
    # None and empty 1-D array mean "no candidates".
    assert oa._validate_external_primal_candidates(None, n_vars=2) == []
    assert oa._validate_external_primal_candidates(np.array([]), n_vars=2) == []
    # 1-D point becomes one candidate; 2-D matrix one per row.
    out = oa._validate_external_primal_candidates(np.array([1.0, 1.5]), n_vars=2)
    assert len(out) == 1
    np.testing.assert_allclose(out[0]["point"], [1.0, 1.5])
    assert out[0]["source"] == "external"
    out = oa._validate_external_primal_candidates(np.array([[1.0, 1.5], [2.0, 0.5]]), n_vars=2)
    assert len(out) == 2
    np.testing.assert_allclose(out[1]["point"], [2.0, 0.5])
    # Dict payload carries objective (validated finite) and provider.
    out = oa._validate_external_primal_candidates(
        {"point": [1.0, 1.5], "objective": 1.94, "provider": "ext"},
        n_vars=2,
    )
    assert out[0]["objective"] == 1.94
    assert out[0]["provider"] == "ext"


@pytest.mark.unit
def test_validate_external_primal_candidates_reject_invalid():
    with pytest.raises(ValueError, match="dimensions"):
        oa._validate_external_primal_candidates(np.zeros((1, 2, 2)), n_vars=2)
    with pytest.raises(ValueError, match="'point'"):
        oa._validate_external_primal_candidates({"objective": 1.0}, n_vars=2)
    with pytest.raises(ValueError, match="length"):
        oa._validate_external_primal_candidates(np.array([1.0, 2.0, 3.0]), n_vars=2)
    with pytest.raises(ValueError, match="finite"):
        oa._validate_external_primal_candidates(np.array([1.0, np.nan]), n_vars=2)
    with pytest.raises(ValueError, match="objective"):
        oa._validate_external_primal_candidates(
            {"point": [1.0, 1.5], "objective": float("nan")}, n_vars=2
        )


@pytest.mark.unit
def test_validate_external_hyperplanes_accepts_and_defaults():
    # -x - i <= -2.5 is x + i >= 2.5: a valid cut for the reference model.
    out = oa._validate_external_hyperplanes(
        [{"coeffs": [-1.0, -1.0], "rhs": -2.5}],
        n_vars=2,
    )
    assert len(out) == 1
    np.testing.assert_allclose(out[0]["coefficients"], [-1.0, -1.0])
    assert out[0]["rhs"] == -2.5
    # Documented defaults: relaxable/global_valid/local_valid all True.
    assert out[0]["relaxable"] is True
    assert out[0]["global_valid"] is True
    assert out[0]["local_valid"] is True
    assert out[0]["supporting_point"] is None
    assert out[0]["constraint_id"] is None
    # Supporting point and violation are validated and passed through.
    out = oa._validate_external_hyperplanes(
        [
            {
                "coefficients": [1.0, 0.0],
                "rhs": 4.0,
                "supporting_point": [4.0, 0.0],
                "violation": 0.0,
                "constraint_id": 0,
                "global_valid": False,
            }
        ],
        n_vars=2,
    )
    np.testing.assert_allclose(out[0]["supporting_point"], [4.0, 0.0])
    assert out[0]["violation"] == 0.0
    assert out[0]["constraint_id"] == 0
    assert out[0]["global_valid"] is False


@pytest.mark.unit
def test_validate_external_hyperplanes_reject_invalid():
    with pytest.raises(ValueError, match="must be a dict"):
        oa._validate_external_hyperplanes([np.array([1.0, 1.0])], n_vars=2)
    with pytest.raises(ValueError, match="'coefficients' or 'coeffs'"):
        oa._validate_external_hyperplanes([{"rhs": 1.0}], n_vars=2)
    with pytest.raises(ValueError, match="'rhs'"):
        oa._validate_external_hyperplanes([{"coeffs": [1.0, 1.0]}], n_vars=2)
    with pytest.raises(ValueError, match="expected 2"):
        oa._validate_external_hyperplanes([{"coeffs": [1.0], "rhs": 0.0}], n_vars=2)
    with pytest.raises(ValueError, match="finite"):
        oa._validate_external_hyperplanes([{"coeffs": [np.inf, 1.0], "rhs": 0.0}], n_vars=2)
    with pytest.raises(ValueError, match="nonzero"):
        oa._validate_external_hyperplanes([{"coeffs": [0.0, 0.0], "rhs": 0.0}], n_vars=2)
    with pytest.raises(ValueError, match="constraint_id"):
        oa._validate_external_hyperplanes(
            [{"coeffs": [1.0, 1.0], "rhs": 0.0, "constraint_id": -1}], n_vars=2
        )
    with pytest.raises(ValueError, match="supporting_point"):
        oa._validate_external_hyperplanes(
            [{"coeffs": [1.0, 1.0], "rhs": 0.0, "supporting_point": [1.0]}], n_vars=2
        )
    with pytest.raises(ValueError, match="relaxable"):
        oa._validate_external_hyperplanes(
            [{"coeffs": [1.0, 1.0], "rhs": 0.0, "relaxable": "yes"}], n_vars=2
        )


@pytest.mark.unit
def test_validate_external_dual_bound_variants():
    assert oa._validate_external_dual_bound(None) is None
    # Bare scalar: bound with global_valid defaulting to True.
    payload = oa._validate_external_dual_bound(1.5)
    assert payload == {"bound": 1.5, "global_valid": True}
    payload = oa._validate_external_dual_bound(
        {"bound": 1.5, "global_valid": False, "provider": "ext"}
    )
    assert payload["bound"] == 1.5
    assert payload["global_valid"] is False
    assert payload["provider"] == "ext"
    with pytest.raises(ValueError, match="'bound'"):
        oa._validate_external_dual_bound({"global_valid": True})
    with pytest.raises(ValueError, match="boolean"):
        oa._validate_external_dual_bound({"bound": 1.0, "global_valid": "yes"})


@pytest.mark.unit
def test_validate_external_termination_requires_bool():
    assert oa._validate_external_termination(True) is True
    assert oa._validate_external_termination(False) is False
    with pytest.raises(ValueError, match="boolean"):
        oa._validate_external_termination(1)


# ── unit: option normalizers ──────────────────────────────────


@pytest.mark.unit
def test_option_normalizer_error_paths():
    with pytest.raises(ValueError, match="init_strategy must be a string"):
        oa._normalize_init_strategy(3)
    with pytest.raises(ValueError, match="feasibility_norm must be a string"):
        oa._normalize_feasibility_norm(3)
    with pytest.raises(ValueError, match="add_regularization must be a string"):
        oa._normalize_regularization(3)
    with pytest.raises(ValueError, match="Unknown add_regularization"):
        oa._normalize_regularization("bogus")
    with pytest.raises(ValueError, match="positive finite"):
        oa._normalize_positive_float("p", 0.0)
    with pytest.raises(ValueError, match="open interval"):
        oa._normalize_open_unit_float("p", 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        oa._normalize_nonnegative_float("p", -0.5)
    with pytest.raises(ValueError, match="positive integer or None"):
        oa._normalize_optional_positive_int("p", 0)
    with pytest.raises(ValueError, match="positive integer"):
        oa._normalize_positive_int("p", 0)


@pytest.mark.unit
def test_option_normalizer_accept_paths():
    # Case/hyphen-insensitive aliases documented in the docstrings.
    assert oa._normalize_init_strategy("rnlp") == "rNLP"
    assert oa._normalize_init_strategy("max-binary") == "max_binary"
    assert oa._normalize_feasibility_norm("l_infinity") == "L_infinity"
    assert oa._normalize_regularization(None) is None
    assert oa._normalize_regularization("level-l1") == "level_L1"
    assert oa._normalize_nonnegative_float("p", 0.0) == 0.0
    assert oa._normalize_optional_positive_int("p", None) is None
    assert oa._normalize_optional_positive_int("p", 3) == 3


@pytest.mark.unit
def test_fp_iteration_count_resolution():
    # Explicit limit wins over max_iterations and the legacy cap.
    assert oa._fp_iteration_count(100, 7) == 7
    # Legacy: min(max_iterations, default_cap), floored at 1.
    assert oa._fp_iteration_count(100, None, default_cap=10) == 10
    assert oa._fp_iteration_count(3, None, default_cap=10) == 3
    assert oa._fp_iteration_count(0, None) == 1
    with pytest.raises(ValueError, match="fp_iteration_limit"):
        oa._fp_iteration_count(100, 0)


@pytest.mark.unit
def test_require_solution_pool_backend_gate():
    # Only gurobi exposes a MIP solution pool; anything else refuses loudly.
    oa._require_solution_pool_backend("gurobi")
    with pytest.raises(RuntimeError, match="gurobi"):
        oa._require_solution_pool_backend("simplex")
    with pytest.raises(RuntimeError, match="gurobi"):
        oa._require_solution_pool_backend(None)


@pytest.mark.unit
def test_round_integral_and_max_integral_seed():
    # Half-up rounding then clamp to integer-compatible bounds.
    assert oa._round_integral_to_bounds(1.5, 0.0, 4.0) == 2.0
    assert oa._round_integral_to_bounds(7.2, 0.0, 4.0) == 4.0
    # No integer inside (0.6, 0.4 crossed after ceil/floor): clamp to raw bounds.
    assert oa._round_integral_to_bounds(0.9, 0.6, 0.7) == pytest.approx(0.7)
    # Practical finite upper bound -> floor(ub).
    assert oa._max_integral_seed(0.0, 4.0, fallback=1.0) == 4.0
    # Effectively unbounded upper -> rounded fallback (2.4 -> 2).
    assert oa._max_integral_seed(0.0, np.inf, fallback=2.4) == 2.0


# ── unit: feasibility evaluator merit/gradient math ───────────


@pytest.mark.unit
def test_feasibility_evaluator_merit_and_gradients():
    # Reference model has one constraint x + i >= 2.5. At (i,x)=(0,0) the
    # violation is 2.5 with active-lower sign -1 and Jacobian row [1, 1]:
    #   L1 merit = 2.5, grad = -1 * [1,1] = [-1,-1]
    #   L2 merit = 2.5^2 = 6.25, grad = 2*2.5*(-1)*[1,1] = [-5,-5]
    #   L_inf merit = 2.5, grad = [-1,-1] (single max row)
    decomp = oa._decompose_model(_cvx_minlp())
    pt = np.array([0.0, 0.0])
    feasible = np.array([2.0, 1.0])  # 1 + 2 = 3 >= 2.5: no violation

    fe = oa._FeasibilityEvaluator(decomp.evaluator, decomp.lb, decomp.ub, "L1")
    assert fe.evaluate_objective(pt) == pytest.approx(2.5)
    np.testing.assert_allclose(fe.evaluate_gradient(pt), [-1.0, -1.0])

    fe2 = oa._FeasibilityEvaluator(decomp.evaluator, decomp.lb, decomp.ub, "L2")
    assert fe2.evaluate_objective(pt) == pytest.approx(6.25)
    np.testing.assert_allclose(fe2.evaluate_gradient(pt), [-5.0, -5.0])

    fe3 = oa._FeasibilityEvaluator(decomp.evaluator, decomp.lb, decomp.ub, "L_infinity")
    assert fe3.evaluate_objective(pt) == pytest.approx(2.5)
    np.testing.assert_allclose(fe3.evaluate_gradient(pt), [-1.0, -1.0])

    # Feasible point: zero merit and zero gradient.
    assert fe.evaluate_objective(feasible) == 0.0
    np.testing.assert_allclose(fe.evaluate_gradient(feasible), [0.0, 0.0])

    # Bounds-only NLP facade: no constraints, zero Hessians.
    assert fe.n_constraints == 0
    assert fe.evaluate_constraints(pt).shape == (0,)
    assert fe.evaluate_jacobian(pt).shape == (0, 2)
    np.testing.assert_allclose(fe.evaluate_hessian(pt), np.zeros((2, 2)))
    np.testing.assert_allclose(fe.evaluate_lagrangian_hessian(pt, 1.0, None), np.zeros((2, 2)))
    lb, ub = fe.variable_bounds
    np.testing.assert_allclose(lb, decomp.lb)
    np.testing.assert_allclose(ub, decomp.ub)


@pytest.mark.unit
def test_feasibility_evaluator_jacobian_failure_gives_zero_gradient():
    # A Jacobian failure must degrade to a zero gradient, not crash.
    decomp = oa._decompose_model(_cvx_minlp())

    class _BrokenJac:
        n_variables = decomp.evaluator.n_variables
        n_constraints = decomp.evaluator.n_constraints
        variable_bounds = decomp.evaluator.variable_bounds

        def evaluate_constraints(self, x):
            return decomp.evaluator.evaluate_constraints(x)

        def evaluate_jacobian(self, x):
            raise RuntimeError("jacobian unavailable")

        def __getattr__(self, name):
            return getattr(decomp.evaluator, name)

    fe = oa._FeasibilityEvaluator(_BrokenJac(), decomp.lb, decomp.ub, "L1")
    np.testing.assert_allclose(fe.evaluate_gradient(np.array([0.0, 0.0])), [0.0, 0.0])


# ── unit: master MIP-start extension ──────────────────────────


@pytest.mark.unit
def test_extend_master_mip_start_paths():
    master = oa._MasterMILPData(
        c=np.zeros(3),
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=[(0.0, 4.0), (0.0, 4.0), (-10.0, 10.0)],
        integrality=np.array([1, 0, 0], dtype=np.int32),
        use_objective_epigraph=True,
        slack_index=None,
    )
    # No start point -> no MIP start.
    assert (
        oa._extend_master_mip_start(master, n_vars=2, mip_start=None, mip_start_objective=1.0)
        is None
    )
    # Too-short start -> no MIP start.
    assert (
        oa._extend_master_mip_start(master, n_vars=2, mip_start=[1.0], mip_start_objective=1.0)
        is None
    )
    # Epigraph master without a finite objective seed -> no MIP start.
    assert (
        oa._extend_master_mip_start(
            master, n_vars=2, mip_start=[1.0, 1.5], mip_start_objective=None
        )
        is None
    )
    assert (
        oa._extend_master_mip_start(
            master, n_vars=2, mip_start=[1.0, 1.5], mip_start_objective=np.inf
        )
        is None
    )
    # Values clamp into bounds: 5.0 -> 4.0 on x1; objective 20 -> 10.
    full = oa._extend_master_mip_start(
        master, n_vars=2, mip_start=[5.0, 1.5], mip_start_objective=20.0
    )
    np.testing.assert_allclose(full, [4.0, 1.5, 10.0])
    # Slack column is seeded at max(0, lo).
    master_slack = oa._MasterMILPData(
        c=np.zeros(4),
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=[(0.0, 4.0), (0.0, 4.0), (-10.0, 10.0), (0.0, 1000.0)],
        integrality=np.array([1, 0, 0, 0], dtype=np.int32),
        use_objective_epigraph=True,
        slack_index=3,
    )
    full = oa._extend_master_mip_start(
        master_slack, n_vars=2, mip_start=[1.0, 1.5], mip_start_objective=1.94
    )
    np.testing.assert_allclose(full, [1.0, 1.5, 1.94, 0.0])


# ── unit: integer-binary expansion bit math ───────────────────


@pytest.mark.unit
def test_integer_binary_expansion_bit_math():
    decomp = oa._decompose_model(_cvx_minlp())
    assert oa._build_integer_binary_expansion(decomp, enabled=False) is None
    exp = oa._build_integer_binary_expansion(decomp, enabled=True)
    # i in [0,4]: width 4 -> 3 bits. Logical layout [i, x, eta, b0, b1, b2].
    assert exp.bit_count == 3
    assert exp.logical_width == 2 + 1 + 3
    assert exp.logical_binary_indices == [3, 4, 5]
    spec = exp.variables[0]
    assert (spec.index, spec.lower, spec.upper, spec.bit_count) == (0, 0, 4, 3)
    # i=3 -> offset 3 -> bits little-endian (1,1,0).
    np.testing.assert_allclose(exp.bit_values_for_point([3.0, 1.0]), [1.0, 1.0, 0.0])
    np.testing.assert_allclose(exp.logical_point([3.0, 1.0]), [3.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    # A zero-bit expansion projects back to the original variables.
    empty = oa._IntegerBinaryExpansion(n_vars=2, variables=(), bit_count=0)
    np.testing.assert_allclose(empty.logical_point([3.0, 1.0]), [3.0, 1.0])


@pytest.mark.unit
def test_append_integer_binary_link_rows():
    decomp = oa._decompose_model(_cvx_minlp())
    exp = oa._build_integer_binary_expansion(decomp, enabled=True)
    a_eq, b_eq = [], []
    # Master layout [i, x, b0, b1, b2]: link row i - b0 - 2 b1 - 4 b2 == lower(=0).
    oa._append_integer_binary_link_rows(
        a_eq, b_eq, n_master=5, integer_binary_expansion=exp, integer_binary_start=2
    )
    assert len(a_eq) == 1
    np.testing.assert_allclose(a_eq[0], [1.0, 0.0, -1.0, -2.0, -4.0])
    assert b_eq == [0.0]
    # Disabled expansion appends nothing.
    a_eq2, b_eq2 = [], []
    oa._append_integer_binary_link_rows(
        a_eq2, b_eq2, n_master=5, integer_binary_expansion=None, integer_binary_start=None
    )
    assert a_eq2 == [] and b_eq2 == []


# ── unit: primal reduction cut ────────────────────────────────


@pytest.mark.unit
def test_primal_reduction_cut_skip_and_add():
    # Nonlinear objective: reduction cut refuses (no certified linear row).
    decomp_nl = oa._decompose_model(_cvx_minlp())
    trace = oa._add_primal_reduction_cut(decomp_nl, np.array([1.0, 1.5]), 1.94, [], [])
    assert trace["status"] == "skipped"
    assert trace["reason"] == "nonlinear_objective_without_certified_epigraph"
    # No incumbent: skipped with reason no_incumbent.
    trace = oa._add_primal_reduction_cut(decomp_nl, None, None, [], [])
    assert trace["status"] == "skipped"
    assert trace["reason"] == "no_incumbent"

    # Linear objective x + 2 i: cut is c^T x <= incumbent_obj - 1e-6*(1+|obj|).
    m = dm.Model("lin_obj")
    i = m.integer("i", lb=0, ub=4)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x + i >= 2.5)
    m.subject_to(x**2 <= 2.25)
    m.minimize(x + 2 * i)
    decomp = oa._decompose_model(m)
    rows, rhs = [], []
    incumbent = np.array([1.0, 1.5])  # obj = 1.5 + 2 = 3.5
    trace = oa._add_primal_reduction_cut(decomp, incumbent, 3.5, rows, rhs)
    assert trace["status"] == "added"
    expected_cutoff = 3.5 - 1e-6 * (1.0 + 3.5)
    assert trace["cutoff"] == pytest.approx(expected_cutoff, rel=1e-12)
    assert len(rows) == 1
    # Row is the linear objective coefficient vector [2 (i), 1 (x)].
    np.testing.assert_allclose(np.asarray(rows[0])[:2], [2.0, 1.0])
    assert rhs[0] == pytest.approx(expected_cutoff, rel=1e-12)


# ── smoke: external hooks never change the certificate ────────


@pytest.mark.smoke
def test_external_hooks_do_not_change_certificate():
    calls = {"hyper": 0}

    def primal_hook(ctx):
        # Feed the known optimum (i=1, x=1.5) as an external candidate.
        return [{"point": np.array([1.0, 1.5]), "objective": 1.94, "provider": "test"}]

    def hyper_hook(ctx):
        calls["hyper"] += 1
        if calls["hyper"] == 1:
            # -x - i <= -2.5 is the model constraint itself: globally valid.
            return [{"coefficients": np.array([-1.0, -1.0]), "rhs": -2.5}]
        return None  # later calls: no_output reject path

    def dual_hook(ctx):
        # Heuristic (non-certified) bound 0.0 <= true optimum: accepted once,
        # then rejected as not improving.
        return {"bound": 0.0, "global_valid": False}

    def term_hook(ctx):
        return False  # never terminate: rejected every iteration

    res = oa.solve_oa(
        _cvx_minlp(),
        time_limit=60.0,
        external_primal_candidate_hook=primal_hook,
        external_hyperplane_hook=hyper_hook,
        external_dual_bound_hook=dual_hook,
        termination_hook=term_hook,
    )
    assert res.status == "optimal"
    assert res.objective == pytest.approx(_OPT, abs=_TOL)
    _assert_no_false_certificate(res)
    hooks = res.mip_nlp_trace["summary"]["external_hooks"]
    assert hooks["call_counts"]["external_primal_candidate"] >= 1
    assert hooks["accepted_counts"].get("external_hyperplane", 0) >= 1
    assert hooks["accepted_counts"].get("external_dual_bound", 0) >= 1
    assert hooks["rejected_counts"].get("termination", 0) >= 1
    assert hooks["error_counts"] == {}


@pytest.mark.smoke
def test_external_hook_exception_raises_runtime_error():
    def bad_hook(ctx):
        raise KeyError("boom")

    with pytest.raises(RuntimeError, match="termination failed during MIP-NLP solve"):
        oa.solve_oa(_cvx_minlp(), termination_hook=bad_hook, time_limit=30.0)


@pytest.mark.smoke
def test_termination_hook_stop_never_claims_false_optimal():
    res = oa.solve_oa(_cvx_minlp(), termination_hook=lambda ctx: True, time_limit=30.0)
    # User termination on iteration 1: no optimality certificate may appear
    # unless the gap actually converged first.
    _assert_no_false_certificate(res)


# ── smoke: continuous-only and infeasible paths ───────────────


@pytest.mark.smoke
def test_continuous_only_model_solves_via_direct_nlp():
    # No integer variables: OA reduces to one NLP solve.
    # min (x-0.7)^2 s.t. x >= 1.5 -> x* = 1.5, obj = 0.8^2 = 0.64.
    m = dm.Model("cont")
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x >= 1.5)
    m.minimize((x - 0.7) ** 2)
    res = oa.solve_oa(m, time_limit=30.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(0.64, abs=_TOL)
    assert res.gap == pytest.approx(0.0, abs=1e-8)


@pytest.mark.smoke
def test_continuous_only_infeasible_model_reports_infeasible():
    # x in [0,1] with x >= 2 has no feasible point.
    m = dm.Model("cont_inf")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.subject_to(x >= 2.0)
    m.minimize((x - 0.7) ** 2)
    res = oa.solve_oa(m, time_limit=30.0)
    assert res.status == "infeasible"
    assert res.objective is None


@pytest.mark.smoke
def test_infeasible_convex_minlp_reports_infeasible():
    # x in [0,1], i in [0,4]: x + i >= 0, so x + i <= -1 is infeasible.
    m = dm.Model("minlp_inf")
    i = m.integer("i", lb=0, ub=4)
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.subject_to(x + i <= -1.0)
    m.minimize((x - 0.7) ** 2 + i)
    res = oa.solve_oa(m, time_limit=30.0)
    assert res.status == "infeasible"
    assert res.objective is None
    assert res.bound is None


# ── smoke: limit paths never yield a false certificate ────────


@pytest.mark.smoke
def test_iteration_limit_returns_feasible_not_false_optimal():
    res = oa.solve_oa(_cvx_minlp(), max_iterations=1, time_limit=30.0)
    # One iteration cannot certify: status must be a non-optimal claim with a
    # true feasible incumbent (>= 1.94) and a sound bound (<= 1.94).
    assert res.status in ("feasible", "iteration_limit")
    _assert_no_false_certificate(res)
    assert res.objective is not None


@pytest.mark.smoke
def test_tiny_time_limit_never_false_certificate():
    res = oa.solve_oa(_cvx_minlp(), time_limit=1e-9)
    assert res.status != "optimal"
    _assert_no_false_certificate(res)


@pytest.mark.smoke
def test_stalling_and_cycling_controls_are_certificate_safe():
    res = oa.solve_oa(_cvx_minlp(), stalling_limit=1, cycling_check=True, time_limit=30.0)
    # Early stop heuristics may downgrade the claim but never falsify it.
    _assert_no_false_certificate(res)
    assert res.objective is not None


# ── smoke: ECP cut senses ─────────────────────────────────────


@pytest.mark.smoke
def test_ecp_handles_ge_sense_nonlinear_constraint():
    # Add -x^2 >= -4 (x <= 2): does not cut off (i,x)=(1,1.5), optimum 1.94.
    m = dm.Model("ecp_ge")
    i = m.integer("i", lb=0, ub=4)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x + i >= 2.5)
    m.subject_to(-(x**2) >= -4.0)
    m.minimize((x - 0.7) ** 2 + 1.3 * i)
    res = oa.solve_oa(m, ecp_mode=True, time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(_OPT, abs=_TOL)
    _assert_no_false_certificate(res)


@pytest.mark.smoke
def test_ecp_handles_nonlinear_equality():
    # x^2 == 2.25 with x in [0,4] forces x = 1.5; then i >= 1 and the
    # objective (x-0.7)^2 + 1.3 i is minimized at i=1: 0.64 + 1.3 = 1.94.
    m = dm.Model("ecp_eq")
    i = m.integer("i", lb=0, ub=4)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x + i >= 2.5)
    m.subject_to(x**2 == 2.25)
    m.minimize((x - 0.7) ** 2 + 1.3 * i)
    res = oa.solve_oa(m, ecp_mode=True, time_limit=60.0, max_iterations=50)
    _assert_no_false_certificate(res)
    assert res.objective == pytest.approx(_OPT, abs=_TOL)


# ── smoke: regularized OA masters ─────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["level_L1", "level_L_infinity", "grad_lag"])
def test_regularization_modes_preserve_certificate(mode):
    res = oa.solve_oa(_cvx_minlp(), add_regularization=mode, time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(_OPT, abs=_TOL)
    _assert_no_false_certificate(res)


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["level_L1", "grad_lag"])
def test_regularization_with_slack_preserves_certificate(mode):
    res = oa.solve_oa(_cvx_minlp(), add_regularization=mode, add_slack=True, time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(_OPT, abs=_TOL)


@pytest.mark.smoke
def test_level_l2_regularization_solves_or_refuses_loudly():
    # level_L2 needs a MIQP-capable backend; without one it must refuse
    # loudly, never silently approximate. With one it must keep the
    # certificate.
    try:
        res = oa.solve_oa(_cvx_minlp(), add_regularization="level_L2", time_limit=60.0)
    except RuntimeError as exc:
        assert "QP/MIQP" in str(exc)
    else:
        assert res.status == "optimal"
        assert res.objective == pytest.approx(_OPT, abs=_TOL)


@pytest.mark.smoke
def test_regularization_rejected_in_ecp_mode():
    with pytest.raises(ValueError, match="only supported for OA"):
        oa.solve_oa(_cvx_minlp(), add_regularization="level_L1", ecp_mode=True)


# ── smoke: feasibility pump ───────────────────────────────────


@pytest.mark.smoke
def test_feasibility_pump_standalone_and_bad_kwarg():
    res = oa.solve_feasibility_pump(_cvx_minlp(), time_limit=30.0)
    # FP is a primal heuristic: feasible incumbent, never a certified gap.
    assert res.status == "feasible"
    assert res.objective >= _OPT - _TOL
    assert res.gap_certified is False
    with pytest.raises(ValueError, match="Unsupported feasibility-pump option"):
        oa.solve_feasibility_pump(_cvx_minlp(), bogus=1)


@pytest.mark.smoke
def test_feasibility_pump_continuous_paths():
    # Continuous feasible model: FP reduces to one NLP, reports 'feasible'
    # with the NLP objective 0.64 (min (x-0.7)^2, x >= 1.5).
    m = dm.Model("fp_cont")
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x >= 1.5)
    m.minimize((x - 0.7) ** 2)
    res = oa.solve_feasibility_pump(m, time_limit=30.0)
    assert res.status == "feasible"
    assert res.objective == pytest.approx(0.64, abs=_TOL)
    # Continuous infeasible model: no feasible point, no objective.
    m2 = dm.Model("fp_cont_inf")
    x2 = m2.continuous("x", lb=0.0, ub=1.0)
    m2.subject_to(x2 >= 2.0)
    m2.minimize(x2)
    res2 = oa.solve_feasibility_pump(m2, time_limit=30.0)
    assert res2.status == "no_feasible_point"
    assert res2.objective is None


@pytest.mark.smoke
def test_feasibility_pump_infeasible_minlp_finds_nothing():
    m = dm.Model("fp_inf")
    i = m.integer("i", lb=0, ub=4)
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.subject_to(x + i <= -1.0)
    m.minimize(x + i)
    res = oa.solve_feasibility_pump(m, time_limit=10.0)
    assert res.status == "no_feasible_point"
    assert res.objective is None


@pytest.mark.smoke
@pytest.mark.parametrize(
    "fp_kwargs",
    [
        {"fp_projcuts": False},  # direct-rounding fallback path
        {"fp_main_norm": "L1"},
        {"fp_main_norm": "L2"},  # L1 projection surrogate + L2 merit
        {"fp_discrete_only": False},  # continuous deviations penalized too
        {"fp_projzerotol": 0.5},  # zero-snapping of small targets
    ],
)
def test_fp_init_option_variants_preserve_certificate(fp_kwargs):
    res = oa.solve_oa(_cvx_minlp(), init_strategy="fp", time_limit=60.0, **fp_kwargs)
    assert res.status == "optimal", fp_kwargs
    assert res.objective == pytest.approx(_OPT, abs=_TOL), fp_kwargs


@pytest.mark.smoke
def test_unsupported_fp_options_raise_loudly():
    # MindtPy FP controls that discopt does not implement must refuse, not
    # silently ignore.
    with pytest.raises(ValueError, match="fp_norm_constraint"):
        oa.solve_oa(_cvx_minlp(), init_strategy="fp", fp_norm_constraint=True)
    with pytest.raises(ValueError, match="fp_cutoffdecr"):
        oa.solve_oa(_cvx_minlp(), init_strategy="fp", fp_cutoffdecr=0.1)


# ── smoke: GOA option surface ─────────────────────────────────


@pytest.mark.smoke
def test_goa_unsupported_option_raises():
    with pytest.raises(ValueError, match="Unsupported GOA option"):
        oa.solve_goa(_cvx_minlp(), bogus_opt=1)


@pytest.mark.smoke
def test_goa_tiny_time_limit_with_feasible_start_reports_feasible():
    # initial_binary start [i=1, x=1.6] is feasible (1 + 1.6 >= 2.5); with an
    # exhausted budget GOA may only claim 'feasible', never a certificate.
    res = oa.solve_goa(
        _cvx_minlp(),
        time_limit=1e-9,
        init_strategy="initial_binary",
        initial_point=np.array([1.0, 1.6]),
    )
    assert res.status == "feasible"
    _assert_no_false_certificate(res)
    assert res.gap_certified is False


@pytest.mark.smoke
def test_goa_tiny_time_limit_without_start_claims_nothing():
    res = oa.solve_goa(_cvx_minlp(), time_limit=1e-9)
    assert res.status != "optimal"
    _assert_no_false_certificate(res)


# ── smoke: SHOT profile toggles ───────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize(
    "shot_kwargs",
    [
        {"cut_strategy": "esh"},
        {"reduction_cuts": True},
        {"relaxation_phase": "initial"},
        {"mip_solution_limit_strategy": "force_optimal"},
    ],
)
def test_shot_profile_toggles_preserve_certificate(shot_kwargs):
    from discopt.solvers.mip_nlp_options import MIPNLPShotConfig

    cfg = MIPNLPShotConfig(**shot_kwargs)
    res = oa.solve_oa(
        _cvx_minlp(),
        mip_nlp_profile="shot",
        mip_nlp_shot_config=cfg,
        time_limit=60.0,
    )
    assert res.status == "optimal", shot_kwargs
    assert res.objective == pytest.approx(_OPT, abs=_TOL), shot_kwargs
    _assert_no_false_certificate(res)


@pytest.mark.smoke
def test_shot_reduction_cuts_on_linear_objective():
    # min x + 2i s.t. x + i >= 2.5, x^2 <= 2.25 (x <= 1.5): i >= 1 forced,
    # optimum x=1.5, i=1 -> 3.5. Reduction cuts apply to linear objectives.
    from discopt.solvers.mip_nlp_options import MIPNLPShotConfig

    m = dm.Model("shot_red")
    i = m.integer("i", lb=0, ub=4)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x + i >= 2.5)
    m.subject_to(x**2 <= 2.25)
    m.minimize(x + 2 * i)
    cfg = MIPNLPShotConfig(reduction_cuts=True)
    res = oa.solve_oa(m, mip_nlp_profile="shot", mip_nlp_shot_config=cfg, time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(3.5, abs=_TOL)
    assert res.bound is not None and res.bound <= 3.5 + _TOL


# ── smoke: misc option gates ──────────────────────────────────


@pytest.mark.smoke
def test_integer_to_binary_no_good_cuts_preserve_certificate():
    res = oa.solve_oa(
        _cvx_minlp(),
        integer_to_binary=True,
        add_no_good_cuts=True,
        time_limit=60.0,
    )
    assert res.status == "optimal"
    assert res.objective == pytest.approx(_OPT, abs=_TOL)


@pytest.mark.smoke
def test_heuristic_nonconvex_mode_suppresses_certification():
    res = oa.solve_oa(_cvx_minlp(), heuristic_nonconvex=True, time_limit=60.0)
    # Heuristic mode must not certify: feasible incumbent, no reported bound.
    assert res.status == "feasible"
    assert res.objective == pytest.approx(_OPT, abs=_TOL)
    assert res.bound is None
    assert res.gap_certified is False


@pytest.mark.smoke
def test_solution_pool_requires_gurobi_backend():
    with pytest.raises(RuntimeError, match="gurobi"):
        oa.solve_oa(_cvx_minlp(), solution_pool=True, milp_solver="simplex")
