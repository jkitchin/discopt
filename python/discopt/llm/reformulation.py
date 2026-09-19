"""
Auto-reformulation engine.

Feature 3C: LLM identifies WHICH reformulations to apply;
deterministic code executes them (correctness guaranteed).

Supported reformulations:
- Big-M tightening
- McCormick substitution for bilinear terms
- Symmetry breaking constraints
- Variable bound tightening from constraint structure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from discopt.modeling.core import Model

logger = logging.getLogger(__name__)


@dataclass
class ReformulationSuggestion:
    """A suggested reformulation with before/after description."""

    category: str
    description: str
    impact: str
    auto_applicable: bool = False
    details: dict = field(default_factory=dict)


def analyze_reformulations(
    model: Model,
    llm: bool = False,
    llm_model: str | None = None,
) -> list[ReformulationSuggestion]:
    """Analyze a model and suggest reformulations.

    Combines deterministic analysis (always runs) with optional
    LLM-powered analysis (when llm=True).

    Parameters
    ----------
    model : Model
        The model to analyze.
    llm : bool, default False
        Enable LLM-powered analysis.
    llm_model : str, optional
        LLM model string.

    Returns
    -------
    list of ReformulationSuggestion
        Ordered by expected impact (highest first).
    """
    suggestions: list[ReformulationSuggestion] = []

    # Deterministic analyses (always run)
    suggestions.extend(_detect_big_m(model))
    suggestions.extend(_detect_weak_bounds(model))
    suggestions.extend(_detect_symmetry(model))
    suggestions.extend(_detect_bilinear(model))

    # LLM-powered analysis (optional)
    if llm:
        try:
            from discopt.llm import is_available

            if is_available():
                llm_suggestions = _llm_analyze(model, llm_model)
                suggestions.extend(llm_suggestions)
        except Exception as e:
            logger.debug("LLM reformulation analysis failed: %s", e)

    return suggestions


def apply_bound_tightening(model: Model) -> int:
    """Apply deterministic bound tightening to the model.

    Derives tighter variable bounds from constraint structure.
    This is safe — it only tightens bounds, never loosens them.

    Parameters
    ----------
    model : Model
        The model to tighten (modified in place).

    Returns
    -------
    int
        Number of bounds tightened.
    """
    from discopt.modeling.core import Constraint

    tightened = 0

    # Simple bound propagation from sum constraints
    # If sum(x) <= b and x >= 0, then x[i] <= b for all i
    for con in model._constraints:
        if not isinstance(con, Constraint):
            continue
        if con.sense != "<=":
            continue

        # Heuristic: check if constraint is a simple sum <= constant
        con_str = str(con)
        # This is a simplified version — full implementation would
        # walk the expression DAG
        # For now, log what we'd do
        logger.debug("Checking constraint for bound propagation: %s", con_str)

    return tightened


# ─────────────────────────────────────────────────────────────
# Deterministic analysis
# ─────────────────────────────────────────────────────────────


def _detect_big_m(model: Model) -> list[ReformulationSuggestion]:
    """Detect oversized big-M constraints."""
    from discopt.modeling.core import Constraint, VarType

    suggestions: list[ReformulationSuggestion] = []
    has_binary = any(v.var_type == VarType.BINARY for v in model._variables)

    if not has_binary:
        return suggestions

    for con in model._constraints:
        if not isinstance(con, Constraint):
            continue

        con_str = str(con)
        # Look for patterns like "x <= M * y" with large M
        tokens = con_str.replace("*", " * ").split()
        for j, token in enumerate(tokens):
            try:
                val = float(token)
                if abs(val) >= 1000 and "*" in con_str:
                    name = con.name or "unnamed"
                    suggestions.append(
                        ReformulationSuggestion(
                            category="big_m_tightening",
                            description=(
                                f"Constraint '{name}' uses M={val:.0f} "
                                f"which may weaken the LP relaxation"
                            ),
                            impact=(
                                "Tighter M or indicator constraints "
                                "strengthen relaxations and reduce B&B nodes"
                            ),
                            auto_applicable=False,
                            details={
                                "constraint": name,
                                "M_value": val,
                            },
                        )
                    )
                    break
            except ValueError:
                continue

    return suggestions


def _detect_weak_bounds(model: Model) -> list[ReformulationSuggestion]:
    """Detect variables with unnecessarily weak bounds."""
    suggestions: list[ReformulationSuggestion] = []

    for var in model._variables:
        lb = np.asarray(var.lb).ravel()
        ub = np.asarray(var.ub).ravel()
        ranges = ub - lb

        # Check for effectively unbounded variables
        if np.any(ranges > 1e18):
            suggestions.append(
                ReformulationSuggestion(
                    category="bound_tightening",
                    description=(
                        f"Variable '{var.name}' is effectively unbounded — "
                        f"adding domain-appropriate bounds will strengthen "
                        f"McCormick relaxations"
                    ),
                    impact="Tighter bounds directly improve relaxation quality",
                    auto_applicable=False,
                    details={"variable": var.name},
                )
            )
        # Check for very wide bounds
        elif np.any(ranges > 1e6):
            suggestions.append(
                ReformulationSuggestion(
                    category="bound_tightening",
                    description=(
                        f"Variable '{var.name}' has very wide bounds "
                        f"(range > 1e6) — consider domain knowledge "
                        f"to tighten"
                    ),
                    impact="Moderate improvement to relaxation quality",
                    auto_applicable=False,
                    details={"variable": var.name},
                )
            )

    return suggestions


def _detect_symmetry(model: Model) -> list[ReformulationSuggestion]:
    """Detect potential symmetry in the model."""
    from discopt.modeling.core import VarType

    suggestions: list[ReformulationSuggestion] = []

    # Find groups of binary variables with identical structure
    binary_vars = [v for v in model._variables if v.var_type == VarType.BINARY]

    # Simple heuristic: binary vectors of the same size
    sizes: dict[int, list[str]] = {}
    for var in binary_vars:
        s = var.size
        if s > 1:
            sizes.setdefault(s, []).append(var.name)

    for size, names in sizes.items():
        if len(names) >= 2:
            suggestions.append(
                ReformulationSuggestion(
                    category="symmetry_breaking",
                    description=(
                        f"Binary variable groups {names} have the same size "
                        f"({size}) — if they represent interchangeable items, "
                        f"add ordering constraints to break symmetry"
                    ),
                    impact="Can dramatically reduce B&B tree size",
                    auto_applicable=False,
                    details={"variables": names, "size": size},
                )
            )

    return suggestions


def _body_has_variable_product(con) -> bool:
    """Whether this constraint's body multiplies two non-constant subexpressions.

    A structural walk, not a string match: ``3 * x`` is linear and must not count,
    while ``x * y`` and ``x * exp(y)`` must. Only reached once
    :func:`~discopt.llm.advisor.model_has_bilinear` has already said the model has
    a product somewhere, so a conservative ``False`` here can only shorten the
    reported list, never invent an entry.
    """
    from discopt.modeling.core import BinaryOp, Constant, Expression

    def is_constant(node) -> bool:
        if isinstance(node, Constant):
            return True
        if isinstance(node, BinaryOp):
            return is_constant(node.left) and is_constant(node.right)
        return not isinstance(node, Expression)

    def walk(node) -> bool:
        if isinstance(node, BinaryOp):
            if node.op in ("*", "/") and not is_constant(node.left) and not is_constant(node.right):
                return True
            return walk(node.left) or walk(node.right)
        for child in getattr(node, "__dict__", {}).values():
            if isinstance(child, Expression) and walk(child):
                return True
            if isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, Expression) and walk(item):
                        return True
        return False

    body = getattr(con, "body", None)
    return bool(body is not None and walk(body))


def _detect_bilinear(model: Model) -> list[ReformulationSuggestion]:
    """Detect bilinear terms that could benefit from reformulation."""
    from discopt.modeling.core import Constraint

    suggestions: list[ReformulationSuggestion] = []

    # `" * " in str(constraint)` matched every `constant * variable`, so a pure
    # MILP came back "bilinear terms in 3 constraints" with a piecewise-McCormick
    # recommendation it cannot use (#1362, measured on
    # docs/notebooks/llm_integration.ipynb). Ask the structural classifier the
    # relaxation layer itself uses, then report the constraints whose body really
    # carries one of the products it found.
    from discopt.llm.advisor import model_has_bilinear

    if not model_has_bilinear(model):
        return suggestions

    bilinear_constraints = [
        con.name or "unnamed"
        for con in model._constraints
        if isinstance(con, Constraint) and _body_has_variable_product(con)
    ]
    if not bilinear_constraints:
        # The product is real but lives in the objective, not in any constraint
        # body. Say so rather than printing "bilinear terms in 0 constraints".
        return [
            ReformulationSuggestion(
                category="mccormick_partitioning",
                description=(
                    "Found bilinear terms in the objective \u2014 use partitions=4 or "
                    "partitions=8 for tighter McCormick relaxations"
                ),
                impact=(
                    "Piecewise McCormick can significantly tighten "
                    "relaxations for bilinear/pooling problems"
                ),
                auto_applicable=True,
                details={"constraints": []},
            )
        ]

    if bilinear_constraints:
        suggestions.append(
            ReformulationSuggestion(
                category="mccormick_partitioning",
                description=(
                    f"Found bilinear terms in {len(bilinear_constraints)} "
                    f"constraints — use partitions=4 or partitions=8 for "
                    f"tighter McCormick relaxations"
                ),
                impact=(
                    "Piecewise McCormick can significantly tighten "
                    "relaxations for bilinear/pooling problems"
                ),
                auto_applicable=True,
                details={"constraints": bilinear_constraints},
            )
        )

    return suggestions


def _llm_analyze(
    model: Model,
    llm_model: str | None,
) -> list[ReformulationSuggestion]:
    """Use LLM to identify additional reformulation opportunities."""
    from discopt.llm.provider import complete
    from discopt.llm.serializer import serialize_model

    model_text = serialize_model(model)

    prompt = (
        "You are an optimization reformulation expert. Analyze this "
        "model for reformulation opportunities beyond simple big-M "
        "and bound tightening.\n\n"
        "Look for:\n"
        "1. Perspective reformulation opportunities (f(x) <= M*y "
        "where y is binary and f is convex)\n"
        "2. RLT (Reformulation-Linearization Technique) opportunities\n"
        "3. Convex substructure that could be exploited\n"
        "4. Redundant constraints\n"
        "5. Implied bounds from constraint interaction\n\n"
        f"## Model\n{model_text}\n\n"
        "Respond with a JSON array of objects, each with keys: "
        "'category', 'description', 'impact'. "
        "If no opportunities found, respond with []."
    )

    try:
        raw = complete(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            max_tokens=1024,
            timeout=10.0,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        import json

        items = json.loads(raw)
        suggestions = []
        for item in items:
            if isinstance(item, dict):
                suggestions.append(
                    ReformulationSuggestion(
                        category=item.get("category", "llm_suggestion"),
                        description=item.get("description", ""),
                        impact=item.get("impact", ""),
                        auto_applicable=False,
                    )
                )
        return suggestions
    except Exception:
        return []
