"""Tests for the P1 presolve orchestrator (item A1 of issue #53).

Covers the new ``PyModelRepr.presolve()`` PyO3 binding and the
backward-compatible ``run_root_presolve`` wrapper. Behavioural parity
with the pre-refactor pipeline lives in
``test_presolve_pipeline.py`` and the regression suite under
``discopt_benchmarks/tests/``; this file targets the orchestrator
contract itself: termination reasons, fixed-point convergence, the
delta log shape, and the legacy-key compatibility shim.
"""

from __future__ import annotations

import discopt as do
from discopt._rust import model_to_repr


def _toy_quartic_model():
    m = do.Model("toy")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-3.0, ub=3.0)
    m.subject_to(x**4 + y <= 5.0)
    m.minimize(x)
    return m


def _singleton_eq_model():
    m = do.Model("eq")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(2.0 * x == 4.0)
    m.minimize(x)
    return m


def test_orchestrator_returns_delta_log():
    repr_ = model_to_repr(_toy_quartic_model())
    new_repr, stats = repr_.presolve(passes=["eliminate", "simplify", "fbbt", "probing"])
    assert "deltas" in stats
    assert "iterations" in stats
    assert "terminated_by" in stats
    assert "bounds_lo" in stats
    assert "bounds_hi" in stats
    # Each delta dict carries the documented keys.
    for d in stats["deltas"]:
        for k in (
            "pass_name",
            "pass_iter",
            "bounds_tightened",
            "aux_vars_introduced",
            "aux_constraints_introduced",
            "constraints_removed",
            "constraints_rewritten",
            "vars_fixed",
            "work_units",
            "wall_time_ms",
        ):
            assert k in d, f"delta missing key {k}"


def test_orchestrator_terminates_at_fixed_point():
    """A pass list with no progress on the toy box terminates at NoProgress."""
    repr_ = model_to_repr(_toy_quartic_model())
    _, stats = repr_.presolve(passes=["eliminate", "simplify", "fbbt", "probing"])
    assert stats["terminated_by"] == "NoProgress"
    # One sweep was enough — no pass made progress, so loop exits.
    assert stats["iterations"] == 1


def test_orchestrator_honors_iteration_cap():
    """Polynomial reformulation needs >1 sweep to converge on the toy."""
    repr_ = model_to_repr(_toy_quartic_model())
    _, stats = repr_.presolve(
        passes=["polynomial_reform", "fbbt"],
        max_iterations=1,
    )
    assert stats["terminated_by"] == "IterationCap"
    assert stats["iterations"] == 1


def test_orchestrator_fixes_singleton_eq_variable():
    repr_ = model_to_repr(_singleton_eq_model())
    new_repr, stats = repr_.presolve(passes=["eliminate", "fbbt"])
    # x should be pinned to 2.0 after eliminate.
    assert abs(stats["bounds_lo"][0] - 2.0) < 1e-9
    assert abs(stats["bounds_hi"][0] - 2.0) < 1e-9
    # And eliminate should have reported progress in the first delta.
    elim_deltas = [d for d in stats["deltas"] if d["pass_name"] == "eliminate"]
    assert any(d["bounds_tightened"] > 0 for d in elim_deltas)


def test_polynomial_reform_introduces_aux_vars():
    repr_ = model_to_repr(_toy_quartic_model())
    n_blocks_before = repr_.n_var_blocks
    new_repr, stats = repr_.presolve(passes=["polynomial_reform"])
    # Quartic monomial decomposes via two bilinear aux variables.
    assert new_repr.n_var_blocks > n_blocks_before
    poly_deltas = [d for d in stats["deltas"] if d["pass_name"] == "polynomial_reform"]
    assert any(d["aux_vars_introduced"] > 0 for d in poly_deltas)


def test_unknown_pass_raises():
    repr_ = model_to_repr(_toy_quartic_model())
    try:
        repr_.presolve(passes=["nonexistent_pass"])
    except ValueError as exc:
        assert "nonexistent_pass" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown pass")


# ─────────────────────────────────────────────────────────────────
# Legacy run_root_presolve wrapper — backward-compatibility shim.
# ─────────────────────────────────────────────────────────────────


def test_run_root_presolve_preserves_legacy_keys():
    """Callers that grep for `elimination` / `polynomial` / `fbbt` keep working."""
    from discopt._jax.presolve_pipeline import run_root_presolve

    repr_ = model_to_repr(_singleton_eq_model())
    new_repr, stats = run_root_presolve(repr_, polynomial=False)
    # Legacy keys present.
    assert "elimination" in stats
    assert "fbbt" in stats
    # New orchestrator metadata is also there.
    assert "iterations" in stats
    assert "terminated_by" in stats
    assert "deltas" in stats


def test_run_root_presolve_reports_elimination_count():
    from discopt._jax.presolve_pipeline import run_root_presolve

    repr_ = model_to_repr(_singleton_eq_model())
    _, stats = run_root_presolve(repr_, polynomial=False)
    assert stats["elimination"]["variables_fixed"] >= 1


def test_aggregate_pass_drops_unused_variable():
    """C1: x = 2y + 1, x not in objective → aggregate drops x."""
    m = do.Model("agg")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.subject_to(x - 2.0 * y == 1.0)
    m.minimize(y)
    repr_ = model_to_repr(m)
    n_blocks_before = repr_.n_var_blocks
    new_repr, stats = repr_.presolve(passes=["aggregate"])
    assert new_repr.n_var_blocks == n_blocks_before - 1
    agg_deltas = [d for d in stats["deltas"] if d["pass_name"] == "aggregate"]
    assert agg_deltas
    assert any(len(d.get("vars_aggregated", [])) > 0 for d in agg_deltas)


def test_run_root_presolve_aggregate_reports_count():
    from discopt._jax.presolve_pipeline import run_root_presolve

    m = do.Model("agg")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.subject_to(x - 2.0 * y == 1.0)
    m.minimize(y)
    repr_ = model_to_repr(m)
    _, stats = run_root_presolve(repr_)
    assert "aggregation" in stats
    assert stats["aggregation"]["variables_aggregated"] >= 1


def test_redundancy_drops_dominated_constraint():
    """C3: 2x + y <= 5 dominates 2x + y <= 10."""
    m = do.Model("redu")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(2.0 * x + y <= 5.0)
    m.subject_to(2.0 * x + y <= 10.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    n_before = repr_.n_constraints
    new_repr, stats = repr_.presolve(passes=["redundancy"])
    assert new_repr.n_constraints == n_before - 1
    redu = [d for d in stats["deltas"] if d["pass_name"] == "redundancy"]
    assert any(d["constraints_removed"] for d in redu)


def test_implied_bounds_pass_tightens_linear_row():
    """B1: 2x + y <= 5 with x,y in [0,10] tightens uppers to x<=2.5, y<=5."""
    m = do.Model("ib")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(2.0 * x + y <= 5.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["implied_bounds"])
    assert abs(stats["bounds_hi"][0] - 2.5) < 1e-9
    assert abs(stats["bounds_hi"][1] - 5.0) < 1e-9
    ib = [d for d in stats["deltas"] if d["pass_name"] == "implied_bounds"]
    assert any(d["bounds_tightened"] >= 2 for d in ib)


def test_run_root_presolve_implied_bounds_reports_count():
    from discopt._jax.presolve_pipeline import run_root_presolve

    m = do.Model("ib2")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(2.0 * x + y <= 5.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = run_root_presolve(repr_, polynomial=False)
    assert "implied_bounds" in stats
    assert stats["implied_bounds"]["bounds_tightened"] >= 1


def test_fbbt_fixed_point_pass_tightens_chained_rows():
    """B4: chained rows converge to the FBBT fixed point in one orchestrator iter."""
    m = do.Model("fp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y <= 5.0)
    m.subject_to(x - y <= 1.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["fbbt_fixed_point"])
    # The first row alone forces both x ≤ 5 and y ≤ 5.
    assert stats["bounds_hi"][0] <= 5.0 + 1e-6
    assert stats["bounds_hi"][1] <= 5.0 + 1e-6
    fp = [d for d in stats["deltas"] if d["pass_name"] == "fbbt_fixed_point"]
    assert fp


def test_scaling_pass_emits_scale_factors():
    """E1: scaling pass populates row/col scales without modifying bounds."""
    m = do.Model("scl")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(100.0 * x + 0.01 * y <= 1.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["scaling"])
    sc = [d for d in stats["deltas"] if d["pass_name"] == "scaling"]
    assert sc
    d0 = sc[0]
    assert "row_scales" in d0
    assert "col_scales" in d0
    assert len(d0["row_scales"]) == 1
    assert len(d0["col_scales"]) == 2
    # 100 / 0.01 row balanced ⇒ row_scale ≈ 1.
    assert abs(d0["row_scales"][0] - 1.0) < 1e-6


def test_run_root_presolve_scaling_surfaces_factors():
    from discopt._jax.presolve_pipeline import run_root_presolve

    m = do.Model("scl2")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.subject_to(4.0 * x <= 1.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = run_root_presolve(repr_, scaling=True, polynomial=False)
    assert "scaling" in stats
    assert stats["scaling"]["row_scales"] is not None
    assert stats["scaling"]["col_scales"] is not None
    assert abs(stats["scaling"]["col_scales"][0] - 0.25) < 1e-9


def test_clique_pass_extracts_set_packing():
    """F2: pairwise binary conflicts surface as edges on the delta."""
    m = do.Model("clq")
    b0 = m.binary("b0")
    b1 = m.binary("b1")
    b2 = m.binary("b2")
    m.subject_to(b0 + b1 + b2 <= 1.0)
    m.minimize(b0 + b1 + b2)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["cliques"])
    deltas = [d for d in stats["deltas"] if d["pass_name"] == "cliques"]
    assert deltas
    edges = deltas[0].get("cliques") or []
    assert sorted(tuple(e) for e in edges) == [(0, 1), (0, 2), (1, 2)]


def test_run_root_presolve_cliques_reports_edges():
    """F2: cliques flow through run_root_presolve into stats['cliques']."""
    from discopt._jax.presolve_pipeline import run_root_presolve

    m = do.Model("clq2")
    b0 = m.binary("b0")
    b1 = m.binary("b1")
    m.subject_to(2.0 * b0 + 2.0 * b1 <= 3.0)
    m.minimize(b0 + b1)
    repr_ = model_to_repr(m)
    _, stats = run_root_presolve(repr_, cliques=True)
    assert "cliques" in stats
    edges = [tuple(e) for e in stats["cliques"]["edges"]]
    assert (0, 1) in edges


def test_reduced_cost_fixing_pass_tightens_upper_bound():
    """E2: positive reduced cost shrinks the upper bound."""
    m = do.Model("rcf")
    x = m.continuous("x", lb=0.0, ub=100.0)
    m.subject_to(x >= 0.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    info = {"lp_value": 0.0, "cutoff": 10.0, "reduced_costs": [2.0]}
    _, stats = repr_.presolve(passes=["reduced_cost_fixing"], reduced_cost_info=info)
    # gap = 10, cbar = 2, lb = 0 ⇒ new_ub = 5.
    assert stats["bounds_hi"][0] <= 5.0 + 1e-9
    deltas = [d for d in stats["deltas"] if d["pass_name"] == "reduced_cost_fixing"]
    assert deltas
    assert deltas[0]["bounds_tightened"] >= 1


def test_reduced_cost_fixing_no_info_is_noop():
    """E2: without reduced_cost_info, the pass is a no-op."""
    m = do.Model("rcf_noop")
    x = m.continuous("x", lb=0.0, ub=100.0)
    m.subject_to(x >= 0.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["reduced_cost_fixing"])
    assert stats["bounds_hi"][0] == 100.0
    deltas = [d for d in stats["deltas"] if d["pass_name"] == "reduced_cost_fixing"]
    assert deltas[0]["bounds_tightened"] == 0


def test_run_root_presolve_reduced_cost_reports_count():
    """E2: reduced cost flows through run_root_presolve."""
    from discopt._jax.presolve_pipeline import run_root_presolve

    m = do.Model("rcf2")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(x >= 0.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    info = {"lp_value": 0.0, "cutoff": 6.0, "reduced_costs": [-3.0]}
    _, stats = run_root_presolve(repr_, reduced_cost=True, reduced_cost_info=info)
    assert "reduced_cost_fixing" in stats
    # cbar = -3, ub = 10, gap = 6 ⇒ new_lb = 10 + 6 / (-3) = 8.
    assert stats["reduced_cost_fixing"]["bounds_tightened"] >= 1


def test_run_root_presolve_polynomial_reports_aux_vars():
    from discopt._jax.presolve_pipeline import run_root_presolve

    repr_ = model_to_repr(_toy_quartic_model())
    _, stats = run_root_presolve(repr_, polynomial=True)
    assert "polynomial" in stats
    assert stats["polynomial"]["aux_variables_introduced"] >= 1


def test_run_root_presolve_no_passes_returns_input():
    """Disabling every pass returns the input unchanged with empty stats."""
    from discopt._jax.presolve_pipeline import run_root_presolve

    repr_ = model_to_repr(_toy_quartic_model())
    new_repr, stats = run_root_presolve(
        repr_,
        eliminate=False,
        aggregate=False,
        redundancy=False,
        polynomial=False,
        implied_bounds=False,
        scaling=False,
        cliques=False,
        reduced_cost=False,
        coefficient_strengthening=False,
        factorable_elim=False,
        fbbt=False,
        fbbt_fixed_point=False,
        simplify=False,
        probing=False,
    )
    assert stats == {}
    assert new_repr is repr_


# ─────────────────────────────────────────────────────────────────
# Determinism — Python-side check that two calls produce the same
# delta sequence. The Rust crate has its own determinism harness in
# tests/presolve_determinism.rs; this is a smoke check from Python.
# ─────────────────────────────────────────────────────────────────


def _delta_canon(stats):
    """Hashable representation of a presolve outcome (excluding wall_time)."""
    out = []
    for d in stats["deltas"]:
        out.append(
            (
                d["pass_name"],
                d["pass_iter"],
                d["bounds_tightened"],
                tuple(d["vars_fixed"]),
                d["aux_vars_introduced"],
                d["aux_constraints_introduced"],
                tuple(d["constraints_removed"]),
                tuple(d["constraints_rewritten"]),
                d["work_units"],
            )
        )
    return (
        stats["iterations"],
        stats["terminated_by"],
        tuple(out),
        tuple(stats["bounds_lo"].tolist()),
        tuple(stats["bounds_hi"].tolist()),
    )


def test_orchestrator_python_determinism():
    repr_ = model_to_repr(_toy_quartic_model())
    _, s1 = repr_.presolve(passes=["eliminate", "polynomial_reform", "simplify", "fbbt"])
    _, s2 = repr_.presolve(passes=["eliminate", "polynomial_reform", "simplify", "fbbt"])
    assert _delta_canon(s1) == _delta_canon(s2)


# ─────────────────────────────────────────────────────────────────
# A3 — Rust↔Python presolve handshake
# ─────────────────────────────────────────────────────────────────


class _RecordingPass:
    """Inert Python pass that records each invocation; never mutates."""

    name = "recording"

    def __init__(self):
        self.calls = 0

    def run(self, model_repr):
        from discopt._jax.presolve.protocol import make_python_delta

        self.calls += 1
        return make_python_delta(self.name, pass_iter=self.calls - 1)


class _OneShotTighten:
    """Python pass that tightens one variable's upper bound on first call only."""

    name = "tighten_x"

    def __init__(self, block_idx: int, new_ub: float):
        self.block_idx = block_idx
        self.new_ub = new_ub
        self.calls = 0

    def run(self, model_repr):
        from discopt._jax.presolve.protocol import make_python_delta

        self.calls += 1
        d = make_python_delta(self.name, pass_iter=self.calls - 1)
        if self.calls == 1:
            cur_lb = list(model_repr.var_lb(self.block_idx))
            cur_ub = list(model_repr.var_ub(self.block_idx))
            new_ub_vec = [min(self.new_ub, ub) for ub in cur_ub]
            n = model_repr.tighten_var_bounds(self.block_idx, cur_lb, new_ub_vec)
            d["bounds_tightened"] = n
        return d


class _RaisingPass:
    name = "boom"

    def run(self, model_repr):
        raise RuntimeError("intentional")


def test_a3_python_pass_runs_in_orchestrator_loop():
    repr_ = model_to_repr(_toy_quartic_model())
    rec = _RecordingPass()
    from discopt._jax.presolve.orchestrator import run_orchestrated_presolve

    _, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=["fbbt"],
        python_passes=[rec],
    )
    # Recording pass ran at least once and emitted a delta with its name.
    assert rec.calls >= 1
    py_deltas = [d for d in stats["deltas"] if d["pass_name"] == "recording"]
    assert py_deltas, "expected at least one recording delta"


def test_a3_python_tighten_persists_through_var_ub():
    repr_ = model_to_repr(_toy_quartic_model())
    # block 0 is x with ub=2.0; tighten to 1.0
    tighten = _OneShotTighten(block_idx=0, new_ub=1.0)
    from discopt._jax.presolve.orchestrator import run_orchestrated_presolve

    new_repr, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=["fbbt"],
        python_passes=[tighten],
    )
    assert new_repr.var_ub(0)[0] <= 1.0 + 1e-12
    # Tightening counted in delta, and progress flag drove >=1 extra sweep.
    py = [d for d in stats["deltas"] if d["pass_name"] == "tighten_x"]
    assert any(int(d["bounds_tightened"]) > 0 for d in py)


def test_a3_fixed_point_terminates_when_no_progress():
    repr_ = model_to_repr(_toy_quartic_model())
    rec = _RecordingPass()
    from discopt._jax.presolve.orchestrator import run_orchestrated_presolve

    _, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=["fbbt"],
        python_passes=[rec],
        max_iterations=8,
    )
    assert stats["terminated_by"] == "NoProgress"
    assert stats["iterations"] >= 1


def test_a3_python_pass_exception_recorded_not_raised():
    repr_ = model_to_repr(_toy_quartic_model())
    from discopt._jax.presolve.orchestrator import run_orchestrated_presolve

    _, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=["fbbt"],
        python_passes=[_RaisingPass()],
    )
    err_deltas = [d for d in stats["deltas"] if "error" in d]
    assert err_deltas, "expected an error-stamped delta from raising pass"
    assert "intentional" in err_deltas[0]["error"]


def test_a3_run_root_presolve_dispatches_python_passes():
    """run_root_presolve(python_passes=...) routes through the orchestrator wrapper."""
    from discopt._jax.presolve_pipeline import run_root_presolve

    repr_ = model_to_repr(_toy_quartic_model())
    rec = _RecordingPass()
    new_repr, stats = run_root_presolve(
        repr_,
        eliminate=False,
        aggregate=False,
        redundancy=False,
        implied_bounds=False,
        simplify=False,
        probing=False,
        fbbt=True,
        python_passes=[rec],
    )
    assert rec.calls >= 1
    assert any(d["pass_name"] == "recording" for d in stats["deltas"])


def test_a3_tighten_var_bounds_index_error():
    repr_ = model_to_repr(_toy_quartic_model())
    import pytest

    with pytest.raises(IndexError):
        repr_.tighten_var_bounds(99, [0.0], [1.0])


def test_a3_tighten_var_bounds_length_mismatch():
    repr_ = model_to_repr(_toy_quartic_model())
    import pytest

    with pytest.raises(ValueError):
        repr_.tighten_var_bounds(0, [0.0, 0.0], [1.0])


def test_d3_reduction_constraints_pass_fixes_vars_to_zero():
    """A sum-of-squares constraint ≤ 0 must fix every variable to zero."""
    m = do.Model("sos")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-3.0, ub=3.0)
    m.subject_to(x**2 + y**2 <= 0.0)
    m.minimize(x + y)
    repr_ = model_to_repr(m)
    new_repr, stats = repr_.presolve(passes=["reduction_constraints"])
    rc_deltas = [d for d in stats["deltas"] if d["pass_name"] == "reduction_constraints"]
    assert rc_deltas
    assert int(rc_deltas[0]["bounds_tightened"]) >= 4
    # Tightened bounds surface through the orchestrator's bounds arrays,
    # matching the convention every other bounds-only pass uses (FBBT,
    # implied bounds, reduced-cost fixing).
    assert stats["bounds_lo"][0] == 0.0
    assert stats["bounds_hi"][0] == 0.0
    assert stats["bounds_lo"][1] == 0.0
    assert stats["bounds_hi"][1] == 0.0
    # And the constraint was marked redundant.
    assert list(rc_deltas[0]["constraints_removed"]) == [0]


def test_d3_reduction_constraints_skips_non_sos_polynomials():
    """A non-sum-of-squares polynomial must not be touched."""
    m = do.Model("nonsos")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-3.0, ub=3.0)
    m.subject_to(x**2 + y <= 0.0)  # y has odd exponent
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["reduction_constraints"])
    rc_deltas = [d for d in stats["deltas"] if d["pass_name"] == "reduction_constraints"]
    assert rc_deltas
    assert int(rc_deltas[0]["bounds_tightened"]) == 0


def test_d3_run_root_presolve_reports_reduction():
    from discopt._jax.presolve_pipeline import run_root_presolve

    m = do.Model("sos")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-3.0, ub=3.0)
    m.subject_to(x**2 + y**2 <= 0.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = run_root_presolve(
        repr_,
        eliminate=False,
        aggregate=False,
        redundancy=False,
        implied_bounds=False,
        simplify=False,
        probing=False,
        fbbt=False,
        reduction_constraints=True,
    )
    assert "reduction_constraints" in stats
    assert stats["reduction_constraints"]["bounds_tightened"] >= 4
    assert sorted(stats["reduction_constraints"]["vars_fixed_to_zero"]) == [0, 1]


def test_d3_reduction_constraints_detects_infeasibility():
    """x^2 ≤ -1 should surface as infeasibility via empty interval."""
    m = do.Model("infeas")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.subject_to(x**2 <= -1.0)
    m.minimize(x)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["reduction_constraints"])
    assert stats["terminated_by"] == "Infeasible"


def test_d1_convex_reform_pass_marks_convex_inequality():
    """A convex `x^2 + y^2 ≤ 1` constraint should be marked convex."""
    from discopt._jax.presolve import ConvexReformPass, run_orchestrated_presolve

    m = do.Model("ball")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x**2 + y**2 <= 1.0)
    m.minimize(x + y)
    repr_ = model_to_repr(m)

    _, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=[],
        python_passes=[ConvexReformPass(m)],
    )
    cr_deltas = [d for d in stats["deltas"] if d["pass_name"] == "convex_reform"]
    assert cr_deltas
    assert list(cr_deltas[0].get("convex_constraints", [])) == [0]


def test_d1_convex_reform_pass_skips_nonconvex():
    """`x^2 - y^2 ≤ 1` is indefinite; the certificate must abstain."""
    from discopt._jax.presolve import ConvexReformPass, run_orchestrated_presolve

    m = do.Model("indef")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x**2 - y**2 <= 1.0)
    m.minimize(x)
    repr_ = model_to_repr(m)

    _, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=[],
        python_passes=[ConvexReformPass(m)],
    )
    cr_deltas = [d for d in stats["deltas"] if d["pass_name"] == "convex_reform"]
    assert cr_deltas
    assert not list(cr_deltas[0].get("convex_constraints", []))


def test_d1_convex_reform_pass_terminates_at_fixed_point():
    """A diagnostic pass must NOT cause the orchestrator to loop forever."""
    from discopt._jax.presolve import ConvexReformPass, run_orchestrated_presolve

    m = do.Model("ball")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x**2 + y**2 <= 1.0)
    m.minimize(x + y)
    repr_ = model_to_repr(m)

    _, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=["fbbt"],
        python_passes=[ConvexReformPass(m)],
        max_iterations=10,
    )
    assert stats["terminated_by"] == "NoProgress"


def test_b2_reverse_ad_pass_tightens_bounds():
    """Reverse-AD on `x + y == 5` with `x ∈ [0, 10]`, `y ∈ [3, 4]` should
    tighten `x` to ``[1, 2]`` (since x = 5 - y)."""
    from discopt._jax.presolve import ReverseADPass, run_orchestrated_presolve

    m = do.Model("revad")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=3.0, ub=4.0)
    m.subject_to(x + y == 5.0)
    m.minimize(x)
    repr_ = model_to_repr(m)

    new_repr, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=[],
        python_passes=[ReverseADPass(m)],
    )
    rad_deltas = [d for d in stats["deltas"] if d["pass_name"] == "reverse_ad"]
    assert rad_deltas
    assert int(rad_deltas[0]["bounds_tightened"]) >= 1
    # x must end up tightened to [1, 2].
    assert new_repr.var_lb(0)[0] >= 1.0 - 1e-9
    assert new_repr.var_ub(0)[0] <= 2.0 + 1e-9


def test_b2_reverse_ad_pass_no_constraints_is_noop():
    """A model with only an objective and no constraints: pass returns 0."""
    from discopt._jax.presolve import ReverseADPass, run_orchestrated_presolve

    m = do.Model("noconstr")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    repr_ = model_to_repr(m)

    _, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=[],
        python_passes=[ReverseADPass(m)],
    )
    rad = [d for d in stats["deltas"] if d["pass_name"] == "reverse_ad"]
    assert rad
    assert int(rad[0]["bounds_tightened"]) == 0


def test_b2_reverse_ad_interleaved_with_fbbt():
    """B2 + FBBT should reach a fixed point with at least as much
    tightening as either alone."""
    from discopt._jax.presolve import ReverseADPass, run_orchestrated_presolve

    m = do.Model("interleave")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=3.0, ub=4.0)
    m.subject_to(x + y == 5.0)
    m.minimize(x)
    repr_ = model_to_repr(m)

    new_repr, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=["fbbt"],
        python_passes=[ReverseADPass(m)],
        max_iterations=8,
    )
    assert stats["terminated_by"] in {"NoProgress", "IterationCap"}
    # x ∈ [1, 2] must hold after either pass picks up the implication.
    assert new_repr.var_lb(0)[0] >= 1.0 - 1e-9
    assert new_repr.var_ub(0)[0] <= 2.0 + 1e-9


def test_a3_python_only_works_with_no_rust_passes():
    repr_ = model_to_repr(_toy_quartic_model())
    tighten = _OneShotTighten(block_idx=0, new_ub=1.0)
    from discopt._jax.presolve.orchestrator import run_orchestrated_presolve

    new_repr, stats = run_orchestrated_presolve(
        repr_,
        rust_passes=[],
        python_passes=[tighten],
    )
    assert new_repr.var_ub(0)[0] <= 1.0 + 1e-12
    # Bounds arrays were rebuilt from the model since Rust didn't run.
    assert stats["bounds_lo"] is not None
    assert stats["bounds_hi"] is not None


# ----------------------------------------------------------------------
# D6 — Neural-network-embedded MINLP presolve
# ----------------------------------------------------------------------


def _force_dead_relu_network():
    """Network with a ReLU layer engineered so neuron 0 is dead-zero
    (pre-activation always negative on the input box) and neuron 1 is
    dead-active (pre-activation always positive)."""
    import numpy as np
    from discopt.nn.network import Activation, DenseLayer, NetworkDefinition

    # input shape (2,), output of layer is (3,):
    #   neuron 0: x0 - x1 - 10 ; over [0,1]^2 ⇒ pre ∈ [-12, -9]   ⇒ dead-zero
    #   neuron 1: x0 + x1 + 10 ; over [0,1]^2 ⇒ pre ∈ [10, 12]    ⇒ dead-active
    #   neuron 2: x0 - x1      ; over [0,1]^2 ⇒ pre ∈ [-1, 1]     ⇒ live
    W = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    b = np.array([-10.0, 10.0, 0.0], dtype=np.float64)
    layer = DenseLayer(weights=W, biases=b, activation=Activation.RELU)
    return NetworkDefinition(
        layers=[layer],
        input_bounds=(np.zeros(2, dtype=np.float64), np.ones(2, dtype=np.float64)),
    )


def test_d6_detect_dead_relus_classifies_neurons():
    from discopt.nn.presolve import detect_dead_relus, tighten_network

    net = _force_dead_relu_network()
    res = tighten_network(net)
    dead = detect_dead_relus(net, res.layer_bounds)
    assert len(dead) == 1
    layer = dead[0]
    assert layer.layer_index == 0
    assert bool(layer.dead_zero[0]) is True
    assert bool(layer.dead_active[1]) is True
    # Live neuron must NOT be in either mask.
    assert bool(layer.dead_zero[2]) is False
    assert bool(layer.dead_active[2]) is False


def test_d6_tighten_network_uses_input_box_override():
    """If the orchestrator hands in a tighter input box, the resulting
    activation envelopes must match that tighter box, not the network's
    declared one."""
    import numpy as np
    from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
    from discopt.nn.presolve import tighten_network

    W = np.array([[1.0]], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    layer = DenseLayer(weights=W, biases=b, activation=Activation.LINEAR)
    net = NetworkDefinition(
        layers=[layer],
        input_bounds=(np.array([-5.0]), np.array([5.0])),
    )

    # Wide declared box ⇒ pre ∈ [-5, 5]
    wide = tighten_network(net)
    assert wide.layer_bounds[0].pre_lb[0] == -5.0
    assert wide.layer_bounds[0].pre_ub[0] == 5.0

    # Tighter override ⇒ pre ∈ [1, 2]
    tight = tighten_network(
        net,
        input_lb=np.array([1.0]),
        input_ub=np.array([2.0]),
    )
    assert tight.layer_bounds[0].pre_lb[0] == 1.0
    assert tight.layer_bounds[0].pre_ub[0] == 2.0


def test_d6_nn_presolve_pass_emits_dead_relu_implications():
    """Run NNPresolvePass on a model whose input variables match the
    network's input layer; check the delta carries dead-relu
    implications for every dead neuron."""

    from discopt.nn.presolve import NNPresolvePass

    net = _force_dead_relu_network()

    m = do.Model("nn")
    m.continuous("x", lb=0.0, ub=1.0, shape=(2,))
    m.minimize(0.0)
    repr_ = model_to_repr(m)

    p = NNPresolvePass(net, input_block_index=0)
    delta = p.run(repr_)

    assert delta["pass_name"] == "nn_presolve"
    assert delta["nn_neurons_dead"] == 2
    impls = delta["nn_implications"]
    states = sorted((i["state"], i["neuron"]) for i in impls)
    assert ("dead_active", 1) in states
    assert ("dead_zero", 0) in states


def test_d6_nn_presolve_pass_no_block_index_falls_back():
    """Without an input_block_index, the pass uses the network's
    declared input_bounds and still runs to completion."""
    from discopt.nn.presolve import NNPresolvePass

    net = _force_dead_relu_network()

    m = do.Model("nn-noblock")
    m.continuous("x", lb=-3.0, ub=3.0, shape=(2,))  # wider than net's box
    m.minimize(0.0)
    repr_ = model_to_repr(m)

    p = NNPresolvePass(net)  # input_block_index=None
    delta = p.run(repr_)

    # Falls back to the network's declared [0, 1]^2 bounds, so the
    # forced-dead neurons are still detected.
    assert delta["nn_neurons_dead"] == 2


def test_d6_nn_presolve_pass_block_size_mismatch_falls_back():
    """If the named block's element count doesn't match the network's
    input size, the pass quietly falls back to declared bounds."""
    from discopt.nn.presolve import NNPresolvePass

    net = _force_dead_relu_network()  # input_size = 2

    m = do.Model("nn-mismatch")
    m.continuous("x", lb=0.0, ub=1.0, shape=(5,))  # not 2
    m.minimize(0.0)
    repr_ = model_to_repr(m)

    p = NNPresolvePass(net, input_block_index=0)
    delta = p.run(repr_)

    # Falls back to network's [0,1]^2 — same forced-dead result.
    assert delta["nn_neurons_dead"] == 2


def test_d6_nn_presolve_pass_no_dead_relus_is_clean():
    """A network with all-live ReLUs over its input box yields a delta
    with zero implications and zero dead count."""
    import numpy as np
    from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
    from discopt.nn.presolve import NNPresolvePass

    # pre = x0 + x1, over [-1, 1]^2 ⇒ pre ∈ [-2, 2] ⇒ live (straddles 0)
    W = np.array([[1.0], [1.0]], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    layer = DenseLayer(weights=W, biases=b, activation=Activation.RELU)
    net = NetworkDefinition(
        layers=[layer],
        input_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
    )

    m = do.Model("nn-live")
    m.continuous("x", lb=-1.0, ub=1.0, shape=(2,))
    m.minimize(0.0)
    repr_ = model_to_repr(m)

    p = NNPresolvePass(net, input_block_index=0)
    delta = p.run(repr_)

    assert delta["nn_neurons_dead"] == 0
    assert delta["nn_implications"] == []


# ----------------------------------------------------------------------
# D5 — Separability detection
# ----------------------------------------------------------------------


def test_d5_detects_two_disjoint_blocks():
    """Two independent constraints over disjoint variable sets ⇒ two
    blocks, ``separable`` flag set."""
    from discopt._jax.presolve import detect_separability

    m = do.Model("two-block")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.subject_to(x + y <= 1.0)
    m.subject_to(z * z <= 0.5)
    m.minimize(x + z)  # objective bridges the two blocks ⇒ NOT separable

    rep = detect_separability(m)
    # The objective links {x,y} with {z}, so all three end up in one block.
    assert len(rep.blocks) == 1
    assert sorted(rep.blocks[0]) == ["x", "y", "z"]
    assert rep.separable is False


def test_d5_detects_separable_with_constant_objective():
    """No objective coupling ⇒ blocks stay disjoint, ``separable`` true."""
    from discopt._jax.presolve import detect_separability

    m = do.Model("sep")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.subject_to(x + y <= 1.0)
    m.subject_to(z * z <= 0.5)
    m.minimize(x)  # objective only touches {x,y}

    rep = detect_separability(m)
    # Two blocks: {x, y} (linked by constraint 1 + objective) and {z}.
    assert len(rep.blocks) == 2
    block_with_x = next(b for b in rep.blocks if "x" in b)
    block_with_z = next(b for b in rep.blocks if "z" in b)
    assert sorted(block_with_x) == ["x", "y"]
    assert block_with_z == ["z"]
    assert rep.separable is True
    # Objective touches only the {x,y} block.
    obj_idx = rep.blocks.index(block_with_x)
    assert rep.objective_block == obj_idx


def test_d5_constraint_block_assignment():
    """Each constraint's block index points to the block containing its
    variables."""
    from discopt._jax.presolve import detect_separability

    m = do.Model("idx")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x <= 0.8)
    m.subject_to(y <= 0.8)
    m.minimize(x)

    rep = detect_separability(m)
    assert len(rep.blocks) == 2
    bx = next(i for i, b in enumerate(rep.blocks) if "x" in b)
    by = next(i for i, b in enumerate(rep.blocks) if "y" in b)
    assert rep.constraint_block == [bx, by]


def test_d5_detects_single_block_for_chained_constraints():
    """Constraints sharing a variable transitively form one block."""
    from discopt._jax.presolve import detect_separability

    m = do.Model("chain")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.subject_to(x + y <= 1.0)  # links x and y
    m.subject_to(y + z <= 1.0)  # links y and z ⇒ all three connected
    m.minimize(0.0)

    rep = detect_separability(m)
    assert len(rep.blocks) == 1
    assert sorted(rep.blocks[0]) == ["x", "y", "z"]
    assert rep.separable is False


def test_d5_pass_emits_blocks_in_delta():
    """The pass surfaces the partition through the delta dict."""
    from discopt._jax.presolve import SeparabilityPass

    m = do.Model("p")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x <= 0.5)
    m.subject_to(y <= 0.5)
    m.minimize(0.0)
    repr_ = model_to_repr(m)

    p = SeparabilityPass(m)
    delta = p.run(repr_)

    assert delta["pass_name"] == "separability"
    assert delta["separable"] is True
    blocks = sorted(sorted(b) for b in delta["separable_blocks"])
    assert blocks == [["x"], ["y"]]
    # Constraint i is assigned to whichever block holds its variable.
    assert delta["constraint_block"][0] != delta["constraint_block"][1]


def test_d5_pass_handles_constant_objective_and_no_constraints():
    """A degenerate model with no constraints and a constant objective
    yields one singleton block per declared variable."""
    from discopt._jax.presolve import SeparabilityPass

    m = do.Model("empty")
    m.continuous("a", lb=0.0, ub=1.0)
    m.continuous("b", lb=0.0, ub=1.0)
    m.minimize(0.0)
    repr_ = model_to_repr(m)

    p = SeparabilityPass(m)
    delta = p.run(repr_)

    assert delta["objective_block"] == -1
    assert sorted(sorted(b) for b in delta["separable_blocks"]) == [["a"], ["b"]]
    assert delta["separable"] is True


def test_c4_coefficient_strengthening_tightens_binary_row():
    """Savelsbergh row 5*b + y <= 5 with b binary, y in [0,2] tightens to
    2*b + y <= 2 — same integer feasible set, strictly tighter LP."""
    m = do.Model("c4")
    b = m.binary("b")
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(5.0 * b + y <= 5.0)
    m.minimize(b + y)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["coefficient_strengthening"])
    cs_deltas = [d for d in stats["deltas"] if d["pass_name"] == "coefficient_strengthening"]
    assert cs_deltas
    assert list(cs_deltas[0]["constraints_rewritten"]) == [0]


def test_d4_detects_two_interchangeable_binaries():
    """A packing constraint x1+x2<=1 with identical binary x1, x2
    yields one orbit {x1, x2}."""
    m = do.Model("d4")
    x1 = m.binary("x1")
    x2 = m.binary("x2")
    m.subject_to(x1 + x2 <= 1.0)
    m.minimize(x1 + x2)
    repr_ = model_to_repr(m)
    result = repr_.detect_symmetries()
    assert int(result["orbits_found"]) == 1
    assert sorted(result["orbits"][0]["vars"]) == [0, 1]


def test_d4_no_orbit_when_coefficients_differ():
    m = do.Model("d4-no")
    x1 = m.continuous("x1", lb=0.0, ub=5.0)
    x2 = m.continuous("x2", lb=0.0, ub=5.0)
    m.subject_to(x1 + 2.0 * x2 <= 10.0)
    m.minimize(x1)
    repr_ = model_to_repr(m)
    result = repr_.detect_symmetries()
    assert int(result["orbits_found"]) == 0
    assert list(result["orbits"]) == []


def test_d4_detects_orbit_of_three():
    m = do.Model("d4-3")
    x1 = m.binary("x1")
    x2 = m.binary("x2")
    x3 = m.binary("x3")
    m.subject_to(x1 + x2 + x3 <= 2.0)
    m.minimize(x1 + x2 + x3)
    repr_ = model_to_repr(m)
    result = repr_.detect_symmetries()
    assert int(result["orbits_found"]) == 1
    assert sorted(result["orbits"][0]["vars"]) == [0, 1, 2]


def test_b3_in_tree_presolve_tightens_at_branched_node():
    """A node-local lb on x propagates through x+y<=5 to tighten ub on y."""
    import numpy as np

    m = do.Model("b3")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y <= 5.0)
    m.minimize(x + y)
    repr_ = model_to_repr(m)
    delta = repr_.in_tree_presolve(
        np.array([3.0, 0.0]),
        np.array([10.0, 10.0]),
        node_depth=0,
        depth_stride=1,
        max_iter=16,
        tol=1e-9,
    )
    assert delta["ran"] is True
    assert delta["infeasible"] is False
    assert int(delta["bounds_tightened"]) >= 1
    assert delta["ub"][1] <= 2.0 + 1e-6
    assert delta["lb"][0] == 3.0


def test_b3_in_tree_presolve_skips_off_schedule():
    """Stride 4 + depth 1 must skip the pass and echo bounds unchanged."""
    import numpy as np

    m = do.Model("b3-skip")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y <= 5.0)
    m.minimize(x + y)
    repr_ = model_to_repr(m)
    delta = repr_.in_tree_presolve(
        np.array([3.0, 0.0]),
        np.array([10.0, 10.0]),
        node_depth=1,
        depth_stride=4,
    )
    assert delta["ran"] is False
    assert int(delta["bounds_tightened"]) == 0
    assert delta["ub"][1] == 10.0


def test_b3_in_tree_presolve_detects_infeasibility():
    """A node where x and y are both pinned to 10 must detect infeasibility."""
    import numpy as np

    m = do.Model("b3-infeas")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y <= 5.0)
    m.minimize(x + y)
    repr_ = model_to_repr(m)
    delta = repr_.in_tree_presolve(
        np.array([10.0, 10.0]),
        np.array([10.0, 10.0]),
        node_depth=0,
        depth_stride=1,
    )
    assert delta["ran"] is True
    assert delta["infeasible"] is True


def test_c2_factorable_elim_drops_definition_equation():
    """v == x*y with v in [-100, 100], x in [0,2], y in [0,3] (so derived
    v in [0, 6] ⊂ box) — the determining equation is dropped, v is freed."""
    m = do.Model("c2")
    v = m.continuous("v", lb=-100.0, ub=100.0)
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=3.0)
    m.subject_to(v - x * y == 0.0)
    m.minimize(x + y)
    repr_ = model_to_repr(m)
    new_repr, stats = repr_.presolve(passes=["factorable_elim"])
    fe_deltas = [d for d in stats["deltas"] if d["pass_name"] == "factorable_elim"]
    assert fe_deltas
    assert list(fe_deltas[0]["constraints_removed"]) == [0]
    assert new_repr.n_constraints == 0


def test_c2_factorable_elim_preserves_when_v_in_objective():
    """If v appears in the objective body, the pass abstains."""
    m = do.Model("c2-obj")
    v = m.continuous("v", lb=-100.0, ub=100.0)
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.subject_to(v - x == 0.0)
    m.minimize(v)  # v is in the objective
    repr_ = model_to_repr(m)
    new_repr, stats = repr_.presolve(passes=["factorable_elim"])
    fe_deltas = [d for d in stats["deltas"] if d["pass_name"] == "factorable_elim"]
    assert fe_deltas
    assert list(fe_deltas[0]["constraints_removed"]) == []
    assert new_repr.n_constraints == 1


def test_c4_coefficient_strengthening_skips_when_no_slack():
    """If b - U_minus_k <= 0 (no slack), the coefficient must not change."""
    m = do.Model("c4-noslack")
    b = m.binary("b")
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.subject_to(2.0 * b + y <= 5.0)  # slack = 5 - 5 = 0
    m.minimize(b + y)
    repr_ = model_to_repr(m)
    _, stats = repr_.presolve(passes=["coefficient_strengthening"])
    cs_deltas = [d for d in stats["deltas"] if d["pass_name"] == "coefficient_strengthening"]
    assert cs_deltas
    assert list(cs_deltas[0]["constraints_rewritten"]) == []
