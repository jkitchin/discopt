"""Base dataclass and registry for benchmark test problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Global registry: category -> list[TestProblem]
_REGISTRY: dict[str, list[TestProblem]] = {}

# Solver applicability per category
_SOLVER_MAP: dict[str, list[str]] = {
    "lp": ["ipm", "ripopt", "ipopt", "highs"],
    "qp": ["ipm", "ripopt", "ipopt"],
    "milp": ["ipm", "ripopt", "ipopt"],
    "miqp": ["ipm", "ripopt", "ipopt"],
    "minlp": ["ipm", "ripopt", "ipopt"],
    "global_opt": ["ipm", "ripopt", "ipopt"],
}


@dataclass
class TestProblem:
    """A benchmark test problem with known solution."""

    name: str
    category: str  # "lp"|"qp"|"milp"|"miqp"|"minlp"|"global_opt"
    level: str  # "smoke"|"full"
    build_fn: Callable  # () -> Model
    known_optimum: float
    applicable_solvers: list[str]
    n_vars: int = 0
    n_constraints: int = 0
    source: str = "programmatic"  # or "nl_file"
    tags: list[str] = field(default_factory=list)
    expected_status: str = "optimal"  # "optimal"|"infeasible"|"unbounded"


def register(problem: TestProblem) -> TestProblem:
    """Register a test problem in the global registry."""
    if problem.category not in _REGISTRY:
        _REGISTRY[problem.category] = []
    _REGISTRY[problem.category].append(problem)
    return problem


def get_problems(
    category: str,
    level: str = "smoke",
) -> list[TestProblem]:
    """Get problems for a category at the given level.

    level="smoke" returns only smoke problems.
    level="full" returns smoke + full problems.
    """
    # Ensure modules are imported so problems are registered
    _ensure_loaded()

    all_probs = _REGISTRY.get(category, [])
    if level == "smoke":
        return [p for p in all_probs if p.level == "smoke"]
    # "full" includes both smoke and full
    return list(all_probs)


def get_applicable_solvers(category: str) -> list[str]:
    """Return the list of applicable solvers for a category."""
    return list(_SOLVER_MAP.get(category, []))


def get_all_categories() -> list[str]:
    """Return all available categories."""
    return ["lp", "qp", "milp", "miqp", "minlp", "global_opt"]


_LOADED = False


def _ensure_loaded():
    """Import all problem modules to trigger registration."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import benchmarks.problems.global_opt  # noqa: F401
    import benchmarks.problems.lp_problems  # noqa: F401
    import benchmarks.problems.milp_problems  # noqa: F401
    import benchmarks.problems.minlp_problems  # noqa: F401
    import benchmarks.problems.miqp_problems  # noqa: F401
    import benchmarks.problems.qp_problems  # noqa: F401
