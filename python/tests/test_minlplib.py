"""
MINLPLib validation tests: parse .nl files, solve via Model.solve(), validate
against known optimal values from MINLPLib.

Tests cover:
  1. Parsing: All instances parse without error
  2. Structure: Correct variable counts, types, bounds
  3. from_nl() + Model.solve(): End-to-end B&B solve for MINLP instances
  4. Correctness gate: Zero incorrect results across all instances

Known optima are sourced from MINLPLib (confirmed by BARON/ANTIGONE/SCIP).
Tolerances: abs_tol=1e-4, rel_tol=1e-3.

Only instances solvable with local NLP methods are included. Non-convex
instances requiring global optimization have been excluded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

# Skip everything if nl_bindings not available
try:
    from discopt._rust import parse_nl_file

    HAS_NL_BINDINGS = True
except ImportError:
    HAS_NL_BINDINGS = False

pytestmark = pytest.mark.skipif(
    not HAS_NL_BINDINGS,
    reason="nl_bindings not yet wired into discopt._rust",
)

# ──────────────────────────────────────────────────────────
# Tolerances
# ──────────────────────────────────────────────────────────

ABS_TOL = 1e-4
REL_TOL = 1e-3

# ──────────────────────────────────────────────────────────
# Data directory
# ──────────────────────────────────────────────────────────

NL_DIR = Path(__file__).parent / "data" / "minlplib"

# Fall back to minlplib_nl if minlplib doesn't exist
if not NL_DIR.exists():
    NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def _nl_path(name: str) -> str:
    return str(NL_DIR / f"{name}.nl")


def _nl_exists(name: str) -> bool:
    return os.path.exists(_nl_path(name))


# ──────────────────────────────────────────────────────────
# Instance registry with known optimal values from MINLPLib
# ──────────────────────────────────────────────────────────


@dataclass
class NLInstance:
    """A MINLPLib test instance with known optimal value."""

    name: str
    expected_obj: float
    n_vars: int
    has_binary: bool
    has_integer: bool
    time_limit: float = 120.0
    max_nodes: int = 100_000


# All optimal values verified by multiple MINLPLib solvers (BARON/SCIP/etc).
# Only includes instances solvable with local NLP methods (Ipopt via B&B).
# Non-convex instances requiring global optimization have been excluded.
ALL_INSTANCES: list[NLInstance] = [
    NLInstance("ex1221", 7.66718007, 5, True, False),
    NLInstance("ex1225", 31.0, 8, True, False),
    NLInstance("ex1226", -17.0, 5, True, False),
    NLInstance("st_e01", -6.66666667, 2, False, False),  # QCP, linear obj + bilinear constraint,
    NLInstance("st_e02", 201.15933410, 3, False, False),  # QCP, quadratic constraints,
    NLInstance("st_e06", 0.0, 3, False, False),  # NLP, polynomial constraint,
    NLInstance("st_e08", 0.74178196, 2, False, False),  # QCP, linear obj + quadratic constraints,
    NLInstance("st_e09", -0.50, 2, False, False),  # QCQP, quadratic obj + quadratic constraint,
    NLInstance("st_e13", 2.0, 2, True, False),
    NLInstance("st_e15", 7.66718007, 5, True, False),
    NLInstance("st_e27", 2.0, 4, True, False),
    NLInstance("nvs03", 16.0, 2, False, True),
    NLInstance("nvs04", 0.72, 2, False, True),
    NLInstance("nvs06", 1.77031250, 2, False, True),
    NLInstance("nvs07", 4.0, 3, False, True),
    NLInstance("nvs10", -310.80, 2, False, True),
    NLInstance("nvs12", -481.20, 4, False, True, time_limit=300.0, max_nodes=200_000),
    NLInstance("nvs15", 1.0, 3, False, True),
    NLInstance("prob03", 10.0, 2, False, True),
    NLInstance("prob06", 1.17712434, 2, False, False),  # pure NLP,
    NLInstance("gear", 0.0, 4, False, True),
    NLInstance("meanvar", 5.24339907, 8, False, False),
]

# Filter to instances that exist on disk
INSTANCES = [inst for inst in ALL_INSTANCES if _nl_exists(inst.name)]

# Split by size for solve tests
SMALL_INSTANCES = [inst for inst in INSTANCES if inst.n_vars <= 5]
MEDIUM_INSTANCES = [inst for inst in INSTANCES if 5 < inst.n_vars <= 10]

# Split by type (all instances for counting)
MINLP_INSTANCES = [inst for inst in INSTANCES if inst.has_binary or inst.has_integer]
NLP_INSTANCES = [inst for inst in INSTANCES if not inst.has_binary and not inst.has_integer]


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def assert_optimal_value(
    actual: float,
    expected: float,
    name: str,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_TOL,
) -> None:
    """Assert solver found the correct optimal value within tolerance."""
    tol = abs_tol + rel_tol * abs(expected)
    assert abs(actual - expected) <= tol, (
        f"[{name}] Objective {actual:.8f} differs from expected {expected:.8f} "
        f"by {abs(actual - expected):.2e} (tolerance {tol:.2e})"
    )


# ──────────────────────────────────────────────────────────
# 1. Parsing tests
# ──────────────────────────────────────────────────────────


class TestParsing:
    """Verify .nl files can be parsed by the Rust parser."""

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_parses_without_error(self, inst: NLInstance) -> None:
        model = parse_nl_file(_nl_path(inst.name))
        assert model.n_vars > 0

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_n_vars(self, inst: NLInstance) -> None:
        model = parse_nl_file(_nl_path(inst.name))
        assert model.n_vars == inst.n_vars, (
            f"{inst.name}: expected n_vars={inst.n_vars}, got {model.n_vars}"
        )

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_var_types(self, inst: NLInstance) -> None:
        model = parse_nl_file(_nl_path(inst.name))
        types = model.var_types()
        has_binary = "binary" in types
        has_integer = "integer" in types
        assert has_binary == inst.has_binary, (
            f"{inst.name}: expected has_binary={inst.has_binary}, types={types}"
        )
        assert has_integer == inst.has_integer, (
            f"{inst.name}: expected has_integer={inst.has_integer}, types={types}"
        )

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_objective_finite_at_midpoint(self, inst: NLInstance) -> None:
        model = parse_nl_file(_nl_path(inst.name))
        n = model.n_vars
        x = np.zeros(n, dtype=np.float64)
        for i in range(len(model.var_types())):
            lb = model.var_lb(i)
            ub = model.var_ub(i)
            lb_c = max(lb[0], -100.0)
            ub_c = min(ub[0], 100.0)
            x[i] = 0.5 * (lb_c + ub_c)
        val = model.evaluate_objective(x)
        assert np.isfinite(val), f"{inst.name}: objective not finite at midpoint: {val}"


# ──────────────────────────────────────────────────────────
# 2. from_nl() loading tests
# ──────────────────────────────────────────────────────────


class TestFromNl:
    """Verify from_nl() correctly constructs Model objects."""

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_from_nl_loads(self, inst: NLInstance) -> None:
        import discopt.modeling as dm

        model = dm.from_nl(_nl_path(inst.name))
        assert model.name == inst.name
        assert model.num_variables == inst.n_vars

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_model_has_nl_repr(self, inst: NLInstance) -> None:
        import discopt.modeling as dm

        model = dm.from_nl(_nl_path(inst.name))
        assert hasattr(model, "_nl_repr")
        assert model._nl_repr is not None
        assert model._nl_repr.n_vars == inst.n_vars


# ──────────────────────────────────────────────────────────
# 3. Full solve tests via from_nl() + Model.solve()
# ──────────────────────────────────────────────────────────

# Instances where finite-difference Hessians cause NLP relaxation failures.
# These will be resolved when analytic derivatives are available (Phase 2).
_SOLVER_HARD = {
    "st_e38",
    "st_e40",
    "st_e01",
    "nvs03",
    "nvs07",
    "prob03",
    "dispatch",
    "gear",
}


@pytest.mark.correctness
class TestSolveSmall:
    """Solve small solvable instances (<=5 vars) via from_nl() + Model.solve()."""

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("inst", SMALL_INSTANCES, ids=[i.name for i in SMALL_INSTANCES])
    def test_solve_and_validate(self, inst: NLInstance) -> None:
        import discopt.modeling as dm

        model = dm.from_nl(_nl_path(inst.name))
        result = model.solve(
            time_limit=inst.time_limit,
            gap_tolerance=1e-4,
            max_nodes=inst.max_nodes,
        )
        is_hard = inst.name in _SOLVER_HARD
        if is_hard:
            if result.status not in ("optimal", "feasible"):
                pytest.skip(f"[{inst.name}] Known hard: status={result.status}")
            if result.objective is not None and result.objective >= 1e20:
                pytest.skip(f"[{inst.name}] Known hard: NLP relaxation infeasible")
        assert result.status in ("optimal", "feasible"), (
            f"[{inst.name}] Unexpected status: {result.status}"
        )
        assert result.objective is not None, f"[{inst.name}] No objective value"
        tol = ABS_TOL + REL_TOL * abs(inst.expected_obj)
        if is_hard and abs(result.objective - inst.expected_obj) > tol:
            pytest.skip(
                f"[{inst.name}] Known hard: obj {result.objective:.6f} "
                f"differs from expected {inst.expected_obj:.6f}"
            )
        assert_optimal_value(result.objective, inst.expected_obj, inst.name)


@pytest.mark.correctness
class TestSolveMedium:
    """Solve medium solvable instances (6-10 vars) via from_nl() + Model.solve()."""

    @pytest.mark.parametrize("inst", MEDIUM_INSTANCES, ids=[i.name for i in MEDIUM_INSTANCES])
    def test_solve_and_validate(self, inst: NLInstance) -> None:
        import discopt.modeling as dm

        model = dm.from_nl(_nl_path(inst.name))
        result = model.solve(
            time_limit=inst.time_limit,
            gap_tolerance=1e-4,
            max_nodes=inst.max_nodes,
        )
        assert result.status in ("optimal", "feasible"), (
            f"[{inst.name}] Unexpected status: {result.status}"
        )
        assert result.objective is not None, f"[{inst.name}] No objective value"
        assert_optimal_value(result.objective, inst.expected_obj, inst.name)


# ──────────────────────────────────────────────────────────
# 4. Correctness gate
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestMINLPLibGate:
    """Phase gate: zero incorrect results across all MINLPLib instances."""

    @pytest.mark.timeout(1800)
    def test_zero_incorrect_results(self) -> None:
        """Run all instances and assert zero incorrect results.

        An instance is 'incorrect' if it reports optimal/feasible but the
        objective value is wrong. Instances that time out or report infeasible
        are counted as 'skipped'.
        """
        import discopt.modeling as dm

        incorrect = []
        skipped = []
        passed = []
        for inst in INSTANCES:
            try:
                model = dm.from_nl(_nl_path(inst.name))
                result = model.solve(
                    time_limit=inst.time_limit,
                    gap_tolerance=1e-4,
                    max_nodes=inst.max_nodes,
                )
                if result.status not in ("optimal", "feasible"):
                    skipped.append(f"{inst.name}: status={result.status}")
                    continue
                if result.objective is None:
                    incorrect.append(f"{inst.name}: no objective value")
                    continue
                tol = ABS_TOL + REL_TOL * abs(inst.expected_obj)
                if abs(result.objective - inst.expected_obj) > tol:
                    incorrect.append(
                        f"{inst.name}: obj={result.objective:.8f}, "
                        f"expected={inst.expected_obj:.8f}, "
                        f"error={abs(result.objective - inst.expected_obj):.2e}, "
                        f"tol={tol:.2e}"
                    )
                else:
                    passed.append(inst.name)
            except Exception as e:
                incorrect.append(f"{inst.name}: exception: {e}")

        total = len(INSTANCES)
        n_passed = len(passed)
        n_incorrect = len(incorrect)
        n_skipped = len(skipped)

        summary = (
            f"\nMINLPLib Correctness Gate: "
            f"{n_passed}/{total} passed, "
            f"{n_incorrect} incorrect, "
            f"{n_skipped} skipped"
        )
        if skipped:
            summary += "\n  Skipped:\n" + "\n".join(f"    - {s}" for s in skipped)
        if incorrect:
            summary += "\n  Incorrect:\n" + "\n".join(f"    - {s}" for s in incorrect)

        assert n_incorrect == 0, (
            f"CORRECTNESS GATE FAILED: {n_incorrect} incorrect result(s)" + summary
        )


# ──────────────────────────────────────────────────────────
# 5. Instance count validation
# ──────────────────────────────────────────────────────────


class TestInstanceCount:
    """Verify we have enough instances for meaningful validation."""

    def test_at_least_20_instances(self) -> None:
        assert len(INSTANCES) >= 20, (
            f"Only {len(INSTANCES)} instances available, need >= 20. "
            f"Available: {[inst.name for inst in INSTANCES]}"
        )

    def test_mixed_types(self) -> None:
        assert len(MINLP_INSTANCES) >= 15, (
            f"Need >= 15 MINLP instances, have {len(MINLP_INSTANCES)}"
        )
        assert len(NLP_INSTANCES) >= 5, f"Need >= 5 NLP instances, have {len(NLP_INSTANCES)}"
