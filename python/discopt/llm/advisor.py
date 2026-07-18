"""
Solver strategy advisor and pre-solve model analysis.

Feature 2D: Pre-solve model analysis (advisory warnings when llm=True)
Feature 2E: suggest_solver_params() — rule-based + optional LLM augmentation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from discopt.modeling.core import Model

logger = logging.getLogger(__name__)


def suggest_solver_params(
    model: Model,
    llm: bool = False,
    llm_model: str | None = None,
) -> dict:
    """Analyze model structure and recommend solver parameters.

    Uses rule-based heuristics derived from problem structure, with
    optional LLM augmentation for edge cases.

    Parameters
    ----------
    model : Model
        The optimization model to analyze.
    llm : bool, default False
        If True, augment rule-based suggestions with LLM analysis.
    llm_model : str, optional
        LLM model string for augmented analysis.

    Returns
    -------
    dict
        Recommended solver parameters with keys:
        ``nlp_solver``, ``partitions``, ``cutting_planes``,
        ``batch_size``, ``gap_tolerance``,
        ``time_limit``, and ``reasoning`` (explanation string).
    """
    analysis = _analyze_structure(model)
    params = _rule_based_params(analysis)

    if llm:
        try:
            llm_suggestion = _llm_augment(model, analysis, params, llm_model)
            if llm_suggestion:
                params.update(llm_suggestion)
        except Exception as e:
            logger.debug("LLM advisor augmentation failed: %s", e)

    return params


def presolve_analysis(
    model: Model,
    llm_model: str | None = None,
) -> list[str]:
    """Run pre-solve model analysis and return advisory warnings.

    Called when ``llm=True`` is passed to ``Model.solve()``.
    Advisory only — never blocks solving.

    Parameters
    ----------
    model : Model
        The model to analyze.
    llm_model : str, optional
        LLM model string.

    Returns
    -------
    list of str
        Advisory warning messages.
    """
    warnings = []
    analysis = _analyze_structure(model)

    # Check for unbounded variables
    for var in model._variables:
        if np.any(var.lb <= -1e19) and np.any(var.ub >= 1e19):
            warnings.append(
                f"Variable '{var.name}' is unbounded — consider adding domain-appropriate bounds"
            )

    # Check for potential big-M weakness
    if analysis["has_big_m"]:
        warnings.append(
            "Model appears to use big-M constraints — "
            "consider using m.if_then() for tighter relaxations"
        )

    # Check for bilinear terms without partitioning
    if analysis["has_bilinear"] and analysis["n_integer"] > 0:
        warnings.append(
            "Model has bilinear terms with integer variables — "
            "consider using partitions=4 for tighter McCormick relaxations"
        )

    # Check variable scaling
    if analysis["bound_range_ratio"] > 1e6:
        warnings.append(
            "Variables have very different magnitude ranges — "
            "poor scaling may cause numerical issues"
        )

    # Large model warning
    if analysis["n_variables"] > 1000:
        warnings.append(
            f"Large model ({analysis['n_variables']} variables) — "
            f"consider batch_size=32 or batch_size=64 for throughput"
        )

    # LLM semantic analysis
    try:
        from discopt.llm import is_available

        if is_available():
            llm_warnings = _llm_presolve(model, analysis, llm_model)
            warnings.extend(llm_warnings)
    except Exception as e:
        logger.debug("LLM presolve analysis failed: %s", e)

    return warnings


# ─────────────────────────────────────────────────────────────
# Internal analysis helpers
# ─────────────────────────────────────────────────────────────


def _analyze_structure(model: Model) -> dict:
    """Analyze model structure and return a feature dictionary."""
    from discopt.modeling.core import Constraint

    n_vars = model.num_variables
    n_cont = model.num_continuous
    n_int = model.num_integer
    n_cons = model.num_constraints

    # Detect bilinear terms (heuristic: look for product expressions)
    has_bilinear = False
    has_big_m = False

    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        c_str = str(c)
        # Heuristic: multiplication of two non-constant terms
        if " * " in c_str:
            has_bilinear = True
        # Heuristic: large constants (> 1000) with binary vars
        for token in c_str.split():
            try:
                val = float(token)
                if abs(val) >= 1000 and n_int > 0:
                    has_big_m = True
            except ValueError:
                pass

    # Variable bound range analysis
    bound_ranges = []
    for var in model._variables:
        lb_arr = np.asarray(var.lb).ravel()
        ub_arr = np.asarray(var.ub).ravel()
        ranges = ub_arr - lb_arr
        finite_mask = ranges < 1e19
        if np.any(finite_mask):
            bound_ranges.append(float(np.max(ranges[finite_mask])))
            bound_ranges.append(max(float(np.min(ranges[finite_mask])), 1e-10))

    if len(bound_ranges) >= 2:
        bound_range_ratio = max(bound_ranges) / min(bound_ranges)
    else:
        bound_range_ratio = 1.0

    # Problem classification attempt
    problem_class = None
    try:
        from discopt._jax.problem_classifier import classify_problem

        problem_class = classify_problem(model)
    except Exception:
        pass

    return {
        "n_variables": n_vars,
        "n_continuous": n_cont,
        "n_integer": n_int,
        "n_constraints": n_cons,
        "has_bilinear": has_bilinear,
        "has_big_m": has_big_m,
        "bound_range_ratio": bound_range_ratio,
        "problem_class": str(problem_class) if problem_class else "NLP",
        "is_pure_continuous": n_int == 0,
    }


def _rule_based_params(analysis: dict) -> dict:
    """Generate solver parameters from rule-based heuristics."""
    params: dict = {"reasoning": []}

    # NLP solver selection
    if analysis["n_variables"] > 100:
        params["nlp_solver"] = "ipm"
        params["reasoning"].append("Using pure-JAX IPM for batch solving (>100 variables)")
    else:
        params["nlp_solver"] = "ipm"
        params["reasoning"].append("Using pure-JAX IPM (default)")

    # Partitioning for bilinear terms
    if analysis["has_bilinear"] and not analysis["is_pure_continuous"]:
        params["partitions"] = 4
        params["reasoning"].append("Enabling 4-partition McCormick (bilinear terms detected)")
    else:
        params["partitions"] = 0

    # Cutting planes for large MINLPs
    if analysis["n_integer"] > 10 and analysis["n_constraints"] > 20:
        params["cutting_planes"] = True
        params["reasoning"].append("Enabling cutting planes (large MINLP)")
    else:
        params["cutting_planes"] = False

    # Batch size based on problem size
    if analysis["n_variables"] > 500:
        params["batch_size"] = 64
    elif analysis["n_variables"] > 100:
        params["batch_size"] = 32
    else:
        params["batch_size"] = 16

    # Gap tolerance
    params["gap_tolerance"] = 1e-4

    # Time limit heuristic
    if analysis["n_variables"] * analysis["n_constraints"] > 50000:
        params["time_limit"] = 3600
        params["reasoning"].append("Large problem — setting 1-hour time limit")
    else:
        params["time_limit"] = 300

    # Join reasoning into a string
    params["reasoning"] = "; ".join(params["reasoning"])

    return params


def _llm_augment(
    model: Model,
    analysis: dict,
    base_params: dict,
    llm_model: str | None,
) -> dict | None:
    """Use LLM to refine solver parameter suggestions."""
    from discopt.llm.provider import complete
    from discopt.llm.serializer import serialize_model

    model_text = serialize_model(model)
    prompt = (
        "You are an optimization solver tuning expert. "
        "Given the model structure and the rule-based parameter "
        "suggestions below, recommend any adjustments.\n\n"
        f"## Model\n{model_text}\n\n"
        f"## Structure Analysis\n{analysis}\n\n"
        f"## Current Suggestions\n{base_params}\n\n"
        "Respond with ONLY a JSON object of parameter overrides "
        "(empty {} if no changes needed). Valid keys: "
        "nlp_solver, partitions, cutting_planes, "
        "batch_size, gap_tolerance, time_limit. "
        "Include a 'reasoning' key explaining changes."
    )

    try:
        raw = complete(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            max_tokens=512,
            timeout=5.0,
        )
        import json

        # Extract JSON from response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


def _llm_presolve(
    model: Model,
    analysis: dict,
    llm_model: str | None,
) -> list[str]:
    """Use LLM for semantic pre-solve analysis."""
    from discopt.llm.provider import complete
    from discopt.llm.serializer import serialize_model

    model_text = serialize_model(model)
    prompt = (
        "You are an optimization model reviewer. Analyze this model "
        "for potential issues. Focus on:\n"
        "1. Constraint conflicts that could cause infeasibility\n"
        "2. Bounds that seem too loose or too tight for the domain\n"
        "3. Potential numerical conditioning issues\n"
        "4. Missing constraints that a domain expert would expect\n\n"
        f"## Model\n{model_text}\n\n"
        "Respond with a bullet list of warnings (one per line, "
        "starting with '- '). If no issues found, respond with "
        "'No issues detected.'"
    )

    try:
        raw = complete(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            max_tokens=512,
            timeout=5.0,
        )
        warnings = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                warnings.append(f"[LLM] {line[2:]}")
        return warnings
    except Exception:
        return []
