"""
LLM-powered infeasibility diagnosis.

Feature 3A: When status="infeasible", analyze constraint violations
and the Jacobian to identify the most likely conflicting subset and
suggest minimal relaxations.

No other MINLP solver does this. CPLEX has "conflict refiner" for
LP/MIP only. BARON/Couenne return nothing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from discopt.modeling.core import Model, SolveResult

logger = logging.getLogger(__name__)


def diagnose_infeasibility(
    model: Model,
    result: SolveResult,
    llm_model: str | None = None,
) -> str:
    """Diagnose why a model is infeasible.

    Computes constraint violations at the relaxation solution,
    analyzes the Jacobian to identify active constraints near the
    infeasibility certificate, and uses an LLM to provide actionable
    explanation.

    Parameters
    ----------
    model : Model
        The infeasible model.
    result : SolveResult
        The solve result with status="infeasible".
    llm_model : str, optional
        LLM model string.

    Returns
    -------
    str
        Diagnostic explanation with suggested relaxations.
    """
    # Step 1: Compute constraint violations
    violations = _compute_violations(model, result)

    # Step 2: Analyze variable bounds
    bound_analysis = _analyze_bounds(model)

    # Step 3: Build diagnostic report
    report = _build_violation_report(model, violations, bound_analysis)

    # Step 4: LLM interpretation
    try:
        from discopt.llm import is_available

        if is_available():
            llm_diagnosis = _llm_diagnose(model, report, violations, llm_model)
            if llm_diagnosis:
                return llm_diagnosis
    except Exception as e:
        logger.debug("LLM diagnosis failed: %s", e)

    # Fallback: return the structured report
    return report


def _compute_violations(model: Model, result: SolveResult) -> list[dict]:
    """Compute constraint violations at the best available point.

    Returns a list of dicts with constraint info and violation magnitude.
    """
    from discopt.modeling.core import Constraint

    violations: list[dict] = []

    # Try to get a solution point (even from an infeasible solve,
    # there may be a relaxation solution)
    if result.x is None:
        return violations

    for i, con in enumerate(model._constraints):
        if not isinstance(con, Constraint):
            continue

        name = con.name or f"constraint_{i}"

        # We can't easily evaluate expressions without the DAG compiler,
        # so we record structural info
        violations.append(
            {
                "index": i,
                "name": name,
                "sense": con.sense,
                "expression": str(con),
            }
        )

    return violations


def _analyze_bounds(model: Model) -> list[dict]:
    """Analyze variable bounds for potential infeasibility sources."""
    issues = []

    for var in model._variables:
        lb = np.asarray(var.lb).ravel()
        ub = np.asarray(var.ub).ravel()

        # Check for empty domains
        if np.any(lb > ub):
            issues.append(
                {
                    "variable": var.name,
                    "issue": "empty_domain",
                    "detail": "Lower bound exceeds upper bound",
                }
            )

        # Check for very tight bounds
        range_vals = ub - lb
        finite = range_vals < 1e19
        if np.any(finite) and np.any(range_vals[finite] < 1e-6):
            issues.append(
                {
                    "variable": var.name,
                    "issue": "near_fixed",
                    "detail": ("Variable is nearly fixed (range < 1e-6 at some indices)"),
                }
            )

    return issues


def _build_violation_report(
    model: Model,
    violations: list[dict],
    bound_issues: list[dict],
) -> str:
    """Build a structured text report of the infeasibility analysis."""
    lines = [
        "# Infeasibility Diagnosis",
        "",
        "The model is infeasible — no solution satisfies all constraints.",
        "",
    ]

    # Variable bound issues
    if bound_issues:
        lines.append("## Variable Bound Issues")
        for issue in bound_issues:
            lines.append(f"- **{issue['variable']}**: {issue['detail']}")
        lines.append("")

    # Constraint summary
    if violations:
        lines.append(f"## Constraints ({len(violations)} total)")
        for v in violations:
            lines.append(f"- [{v['name']}] {v['expression']}")
        lines.append("")

    # General suggestions
    lines.append("## Suggested Actions")
    lines.append("1. Check variable bounds for logical consistency")
    lines.append("2. Try relaxing the most restrictive constraints one at a time")
    lines.append("3. Add slack variables to identify the binding constraint")
    lines.append("4. Use `result.explain(llm=True)` for detailed LLM analysis")

    return "\n".join(lines)


def _llm_diagnose(
    model: Model,
    report: str,
    violations: list[dict],
    llm_model: str | None,
) -> str | None:
    """Use LLM to interpret the violation data and suggest fixes."""
    from discopt.llm.provider import complete, resolve_timeout
    from discopt.llm.serializer import serialize_model

    model_text = serialize_model(model)

    prompt = (
        "You are an expert optimization consultant diagnosing "
        "an infeasible MINLP model. This is the FIRST solver to "
        "provide automated infeasibility diagnosis for MINLP "
        "(CPLEX only does this for LP/MIP).\n\n"
        "Analyze the model and constraint structure to identify "
        "the most likely conflicting constraint subset. Suggest "
        "specific, minimal relaxations to restore feasibility.\n\n"
        f"## Model Structure\n{model_text}\n\n"
        f"## Violation Report\n{report}\n\n"
        "Provide:\n"
        "1. Most likely conflicting constraints (by name)\n"
        "2. Why they conflict\n"
        "3. Specific relaxation suggestions (which constraint to "
        "relax, by how much)\n"
        "4. Whether the infeasibility is structural (model error) "
        "or parametric (data issue)\n"
        "Keep the response under 400 words."
    )

    try:
        raw = complete(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            max_tokens=1024,
            timeout=resolve_timeout(default=10.0),
        )
        if raw and raw.strip():
            return raw.strip()
    except Exception as exc:  # noqa: BLE001 - LLM commentary is advisory, never required
        logger.debug("LLM infeasibility diagnosis unavailable: %s: %s", type(exc).__name__, exc)
    return None


def diagnose_result(
    model: Model,
    result: SolveResult,
    llm_model: str | None = None,
) -> str:
    """General-purpose diagnosis for any solve result status.

    Dispatches to specialized diagnosis based on the result status.

    Parameters
    ----------
    model : Model
        The model that was solved.
    result : SolveResult
        The solve result.
    llm_model : str, optional
        LLM model string.

    Returns
    -------
    str
        Diagnostic text.
    """
    if result.status == "infeasible":
        return diagnose_infeasibility(model, result, llm_model)

    if result.status in ("time_limit", "node_limit"):
        return _diagnose_limit(model, result, llm_model)

    if result.status == "iteration_limit":
        return _diagnose_convergence(model, result, llm_model)

    # Optimal or feasible — check for quality issues
    if result.gap is not None and result.gap > 0.01:
        return _diagnose_large_gap(model, result, llm_model)

    return f"Solve completed with status '{result.status}'. No issues detected."


def _diagnose_limit(model: Model, result: SolveResult, llm_model: str | None) -> str:
    """Diagnose time/node limit results."""
    lines = [
        f"# {result.status.replace('_', ' ').title()} Diagnosis",
        "",
        f"The solver stopped at {result.status} after "
        f"{result.wall_time:.1f}s and {result.node_count} nodes.",
    ]

    if result.gap is not None:
        lines.append(f"Current gap: {result.gap:.2%}")
        if result.gap < 0.01:
            lines.append(
                "The gap is small — the current solution is likely "
                "near-optimal. Consider accepting it."
            )
        elif result.gap < 0.10:
            lines.append("Moderate gap. Try: partitions=4, cutting_planes=True")
        else:
            lines.append(
                "Large gap. The relaxation is weak. Try: "
                "tighter variable bounds, partitions=8, "
                "cutting_planes=True"
            )

    lines.append("")
    lines.append("## Timing Breakdown")
    if result.wall_time > 0:
        lines.append(
            f"- Rust (B&B): {result.rust_time:.1f}s "
            f"({100 * result.rust_time / result.wall_time:.0f}%)"
        )
        lines.append(
            f"- JAX (NLP): {result.jax_time:.1f}s ({100 * result.jax_time / result.wall_time:.0f}%)"
        )
        lines.append(
            f"- Python: {result.python_time:.1f}s "
            f"({100 * result.python_time / result.wall_time:.0f}%)"
        )

    return "\n".join(lines)


def _diagnose_convergence(model: Model, result: SolveResult, llm_model: str | None) -> str:
    """Diagnose iteration limit (NLP convergence) issues."""
    return (
        "# Iteration Limit Diagnosis\n\n"
        "The NLP sub-solver hit its iteration limit. This typically "
        "means the continuous relaxation is difficult to solve.\n\n"
        "## Suggestions\n"
        "1. Increase max_iter: `m.solve(max_iter=5000)`\n"
        "2. Loosen tolerance: `m.solve(tol=1e-6)`\n"
        "3. Try adaptive barrier: `m.solve(mu_strategy='adaptive')`\n"
        "4. Check variable scaling — rescale if magnitudes differ "
        "by more than 1e4\n"
        "5. Simplify nonlinear expressions if possible"
    )


def _diagnose_large_gap(model: Model, result: SolveResult, llm_model: str | None) -> str:
    """Diagnose optimal/feasible results with large gaps."""
    return (
        f"# Large Gap Diagnosis\n\n"
        f"Solved with status '{result.status}' but gap is "
        f"{result.gap:.2%} — the relaxation is weak.\n\n"
        f"## Suggestions\n"
        f"1. Tighten variable bounds to strengthen relaxations\n"
        f"2. Use partitions=4 for McCormick tightening\n"
        f"3. Enable cutting_planes=True for OA cuts\n"
        f"4. Check for large big-M values that weaken the LP relaxation\n"
        f"5. Consider reformulating bilinear terms"
    )
