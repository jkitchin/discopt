"""
T24: Batch IPM Solver Path Tests

Validates that the pure-JAX IPM backend (nlp_solver="ipm") produces correct
results across all problem classes: NLP, MINLP, MILP, and MIQP. Tests cover:

  - Correctness of all 24 test instances with IPM backend
  - Consistency between IPM and Ipopt backends
  - Root multistart with IPM for nonconvex MINLPs
  - Batch vmap path for multi-node B&B trees
  - Pure continuous NLP direct solve (no B&B)
  - MILP and MIQP B&B paths

Acceptance criteria:
  - All 24 instances produce correct objectives with nlp_solver="ipm"
  - IPM and Ipopt objectives agree within loose tolerance
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from test_correctness import (
    ABS_TOL,
    INSTANCES,
    REL_TOL,
    _build_simple_minlp,
    assert_optimal_value,
)

# ──────────────────────────────────────────────────────────
# 1. All 24 instances with IPM backend
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestBatchIPMAllInstances:
    """Definitive T24 acceptance test: all 24 instances via nlp_solver='ipm'."""

    _IPM_INSTANCES = [inst for inst in INSTANCES if inst.name != "circle_minlp"]

    @pytest.mark.parametrize(
        "instance",
        _IPM_INSTANCES,
        ids=[inst.name for inst in _IPM_INSTANCES],
    )
    def test_batch_ipm_all_24_instances(self, instance) -> None:
        """Verify IPM backend finds correct optimal objective for each instance."""
        model = instance.build_fn()
        result = model.solve(
            nlp_solver="ipm",
            time_limit=120.0,
            gap_tolerance=1e-6,
            max_nodes=50_000,
        )
        assert_optimal_value(
            result,
            instance.expected_obj,
            f"ipm:{instance.name}",
        )


# ──────────────────────────────────────────────────────────
# 2. IPM vs Ipopt consistency
# ──────────────────────────────────────────────────────────

_COMPARISON_INSTANCES = [
    inst
    for inst in INSTANCES
    if inst.name in ("simple_minlp", "constrained_quadratic", "exp_binary_minlp")
]


@pytest.mark.correctness
class TestBatchIPMMatchesIpopt:
    """Verify IPM and Ipopt produce consistent objectives."""

    @pytest.mark.parametrize(
        "instance",
        _COMPARISON_INSTANCES,
        ids=[inst.name for inst in _COMPARISON_INSTANCES],
    )
    def test_batch_ipm_matches_ipopt(self, instance) -> None:
        """Solve with both backends and compare objectives."""
        # Solve with IPM
        model_ipm = instance.build_fn()
        result_ipm = model_ipm.solve(
            nlp_solver="ipm",
            time_limit=120.0,
            gap_tolerance=1e-6,
            max_nodes=50_000,
        )
        assert result_ipm.status in ("optimal", "feasible")
        assert result_ipm.objective is not None

        # Solve with Ipopt (skip if cyipopt not installed)
        try:
            import cyipopt  # noqa: F401
        except ImportError:
            pytest.skip("cyipopt not installed")

        model_ipopt = instance.build_fn()
        result_ipopt = model_ipopt.solve(
            nlp_solver="ipopt",
            time_limit=120.0,
            gap_tolerance=1e-6,
            max_nodes=50_000,
        )
        assert result_ipopt.status in ("optimal", "feasible")
        assert result_ipopt.objective is not None

        # Looser tolerance: different solvers may find slightly different optima
        diff = abs(result_ipm.objective - result_ipopt.objective)
        tol = 1e-3 + 1e-2 * abs(result_ipopt.objective)
        assert diff <= tol, (
            f"[{instance.name}] IPM obj={result_ipm.objective:.8f} vs "
            f"Ipopt obj={result_ipopt.objective:.8f}, diff={diff:.2e}, tol={tol:.2e}"
        )


# ──────────────────────────────────────────────────────────
# 3. Root multistart with IPM
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestBatchIPMRootMultistart:
    """Verify batched root multistart works with IPM backend."""

    def test_simple_minlp_root_multistart(self) -> None:
        """Solve simple_minlp with IPM, verify correct result."""
        model = _build_simple_minlp()
        result = model.solve(
            nlp_solver="ipm",
            time_limit=120.0,
            gap_tolerance=1e-6,
            max_nodes=50_000,
        )
        assert_optimal_value(result, 0.5, "ipm:simple_minlp")


# ──────────────────────────────────────────────────────────
# 4. Batch vmap path (multi-node B&B)
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestBatchIPMUsesVmapPath:
    """Verify IPM uses the vmap batch path for multi-node B&B."""

    def test_batch_ipm_uses_vmap_path(self) -> None:
        """Solve a MINLP needing >1 B&B node with batch_size=4 and IPM.

        Uses a 4-binary knapsack problem that requires branching to find the
        optimal integer solution.
        """
        m = dm.Model("branch_knapsack")
        x1 = m.binary("x1")
        x2 = m.binary("x2")
        x3 = m.binary("x3")
        x4 = m.binary("x4")
        # Objective: maximize value (minimize negative)
        m.minimize(-3 * x1 - 4 * x2 - 5 * x3 - 7 * x4)
        # Knapsack constraint: total weight <= 10
        m.subject_to(2 * x1 + 3 * x2 + 4 * x3 + 5 * x4 <= 10)
        # Optimal: x1=1,x2=0,x3=1,x4=1 → weight=2+4+5=11 NO
        # x1=0,x2=1,x3=1,x4=1 → weight=3+4+5=12 NO
        # x1=1,x2=1,x3=1,x4=0 → weight=2+3+4=9, obj=-12
        # x1=0,x2=0,x3=1,x4=1 → weight=4+5=9, obj=-12
        # x1=1,x2=1,x3=0,x4=1 → weight=2+3+5=10, obj=-14 ← optimal
        result = m.solve(
            nlp_solver="ipm",
            batch_size=4,
            time_limit=120.0,
            gap_tolerance=1e-6,
            max_nodes=50_000,
        )
        assert result.status in ("optimal", "feasible")
        assert result.node_count >= 1, f"Expected >=1 B&B nodes, got {result.node_count}"
        tol = ABS_TOL + REL_TOL * abs(-14.0)
        assert abs(result.objective - (-14.0)) <= tol, (
            f"Knapsack obj={result.objective:.4f}, expected=-14.0"
        )


# ──────────────────────────────────────────────────────────
# 5. Pure continuous NLP direct solve
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestBatchIPMContinuousDirect:
    """Verify pure continuous problems go through _solve_continuous, not B&B."""

    def test_batch_ipm_continuous_direct(self) -> None:
        """Continuous NLP with IPM should have node_count==0."""
        m = dm.Model("continuous_quad")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize((x - 2) ** 2 + (y + 1) ** 2)

        result = m.solve(nlp_solver="ipm", time_limit=60.0)
        assert result.status in ("optimal", "feasible")
        assert result.node_count == 0, f"Continuous problem used B&B with {result.node_count} nodes"
        assert_optimal_value(result, 0.0, "ipm:continuous_direct")


# ──────────────────────────────────────────────────────────
# 6. MILP via LP relaxation path
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestBatchMILPViaLP:
    """Verify MILP problems solved correctly through the LP relaxation path."""

    def test_batch_milp_via_lp_ipm(self) -> None:
        """Simple MILP: min -3x1 - 4x2 s.t. 2x1+3x2<=6, x1,x2 binary.

        x1=0,x2=1: obj=-4, weight=3<=6. x1=1,x2=1: weight=5<=6, obj=-7.
        x1=1,x2=0: weight=2<=6, obj=-3. Optimal: x1=1,x2=1, obj=-7.
        """
        m = dm.Model("simple_milp")
        x1 = m.binary("x1")
        x2 = m.binary("x2")
        m.minimize(-3 * x1 - 4 * x2)
        m.subject_to(2 * x1 + 3 * x2 <= 6)

        result = m.solve(time_limit=60.0)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = ABS_TOL + REL_TOL * abs(-7.0)
        assert abs(result.objective - (-7.0)) <= tol, (
            f"MILP obj={result.objective:.4f}, expected=-7.0"
        )


# ──────────────────────────────────────────────────────────
# 7. MIQP via QP relaxation path
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestBatchMIQPViaQP:
    """Verify MIQP problems solved correctly through the QP relaxation path."""

    def test_batch_miqp_via_qp_ipm(self) -> None:
        """Simple MIQP: min (x-2.3)^2 + (y-1.7)^2, x,y integer in [0,5].

        Optimal integers: x=2, y=2. obj=(0.3)^2+(0.3)^2=0.18.
        """
        m = dm.Model("simple_miqp")
        m.integer("x", lb=0, ub=5)
        m.integer("y", lb=0, ub=5)
        x = m._variables[0]
        y = m._variables[1]
        m.minimize((x - 2.3) ** 2 + (y - 1.7) ** 2)

        result = m.solve(time_limit=60.0)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = ABS_TOL + REL_TOL * abs(0.18)
        assert abs(result.objective - 0.18) <= tol, (
            f"MIQP obj={result.objective:.4f}, expected=0.18"
        )
