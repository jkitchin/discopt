"""
Centralized prompt templates for all LLM features.

All prompts are versioned strings that can be tested and improved
independently of the calling code.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# explain() prompts — status-specific
# ─────────────────────────────────────────────────────────────

EXPLAIN_SYSTEM = """\
You are an expert optimization consultant analyzing results from discopt, \
a hybrid MINLP solver using Rust (Branch & Bound), JAX (automatic differentiation), \
and Python orchestration.

Provide a clear, actionable explanation of the solve result. \
Focus on what the user should know and what they should do next. \
Be specific — reference variable names, constraint names, and numeric values. \
Keep the explanation under 300 words."""

EXPLAIN_OPTIMAL = """\
The model solved to optimality. Explain:
1. The optimal objective value and what it means
2. Which binary/integer variables are active (value = 1 or non-zero)
3. Which constraints are likely binding (at their bounds)
4. A plain-language interpretation of the solution
5. Any notable aspects of the solve (e.g., fast/slow, many nodes, large gap at root)

Model and result details:
{model_text}

{result_text}"""

EXPLAIN_INFEASIBLE = """\
The model is infeasible — no solution satisfies all constraints simultaneously. Explain:
1. What infeasibility means for this specific problem
2. Which constraints are most likely in conflict (based on the constraint structure)
3. Suggested relaxations: which constraint(s) to loosen, by how much
4. Whether the infeasibility might be due to overly tight bounds
5. Concrete next steps for the user

Model and result details:
{model_text}

{result_text}"""

EXPLAIN_ITERATION_LIMIT = """\
The solver hit its iteration limit before converging. Explain:
1. What this means (NLP solver couldn't converge within max iterations)
2. Likely causes: ill-conditioning, poor scaling, difficult nonconvexity
3. Suggested Ipopt parameter tuning: max_iter, tol, acceptable_tol, mu_strategy
4. Whether the current solution might still be useful (check constraint violations)
5. Model reformulation suggestions that could help convergence

Model and result details:
{model_text}

{result_text}"""

EXPLAIN_TIME_LIMIT = """\
The solver hit the time limit before proving optimality. Explain:
1. The current solution quality (gap, feasibility)
2. B&B progress: nodes explored, gap closure rate
3. Whether the incumbent solution is likely close to optimal
4. Suggestions: increase time_limit, use partitions for tighter relaxations, \
adjust branching_policy
5. Time breakdown analysis (Rust/JAX/Python fractions)

Model and result details:
{model_text}

{result_text}"""

EXPLAIN_NODE_LIMIT = """\
The solver hit the node limit. Explain:
1. Current solution quality and gap
2. Whether the B&B tree is growing too fast (weak relaxations)
3. Suggestions: increase max_nodes, use cutting_planes=True, increase partitions
4. Whether a different branching_policy might help

Model and result details:
{model_text}

{result_text}"""

EXPLAIN_GENERIC = """\
Explain this solve result to the user:

Model and result details:
{model_text}

{result_text}"""


def get_explain_prompt(status: str) -> str:
    """Get the appropriate explain prompt template for a given status.

    Parameters
    ----------
    status : str
        The solve status string.

    Returns
    -------
    str
        Prompt template with ``{model_text}`` and ``{result_text}`` placeholders.
    """
    prompts = {
        "optimal": EXPLAIN_OPTIMAL,
        "feasible": EXPLAIN_OPTIMAL,
        "infeasible": EXPLAIN_INFEASIBLE,
        "iteration_limit": EXPLAIN_ITERATION_LIMIT,
        "time_limit": EXPLAIN_TIME_LIMIT,
        "node_limit": EXPLAIN_NODE_LIMIT,
    }
    return prompts.get(status, EXPLAIN_GENERIC)


# ─────────────────────────────────────────────────────────────
# from_description() prompts
# ─────────────────────────────────────────────────────────────

FORMULATE_SYSTEM = """\
You are an expert optimization modeler building a discopt Model from a natural \
language description. You MUST use the provided tools to build the model — \
do NOT generate free-form Python code.

The discopt modeling API:
- create_model(name): Create a new Model
- add_continuous(name, shape, lb, ub): Add continuous variable(s)
- add_binary(name, shape): Add binary (0/1) variable(s)
- add_integer(name, shape, lb, ub): Add integer variable(s)
- add_parameter(name, value): Add a parameter with a fixed value
- set_objective(sense, expression): Set objective ("minimize" or "maximize")
- add_constraint(expression, sense, rhs, name): Add a constraint
- add_if_then(indicator, constraints, name): Add indicator constraint
- add_either_or(disjuncts, name): Add disjunctive constraint (exactly one group holds)
- add_implies(y1, y2, name): Add logical implication y1 => y2 for binary variables
- add_at_least(k, binaries, name): At least k of the listed binaries must be 1

Guidelines:
- Use descriptive variable names that match the problem domain
- Always name constraints for debuggability
- Set tight, meaningful bounds on all variables
- Use binary variables for discrete choices (build/don't build, select/don't select)
- Use indicator constraints (add_if_then) instead of big-M when possible
- Use add_either_or for mode selection or disjunctive operating regions
- Use add_implies for precedence or prerequisite relationships between decisions
- Prefer vectorized operations for indexed sets"""

FORMULATE_USER = """\
Build a discopt optimization model for this problem:

{description}

{data_schema}

Use the tools to construct the model step by step. Start with create_model, \
then add variables, set the objective, and add constraints."""


# ─────────────────────────────────────────────────────────────
# Teaching assistant prompts (Feature 3E)
# ─────────────────────────────────────────────────────────────

TEACHING_SYSTEM = """\
You are an optimization teaching assistant explaining how discopt solved \
a problem. Your audience is a graduate student learning mathematical \
optimization. Be pedagogical: explain concepts, use analogies, and \
reference the specific problem structure."""

TEACHING_WALKTHROUGH = """\
Generate an educational walkthrough of this solved optimization problem.

Cover:
1. **Mathematical formulation** with domain context
2. **Problem class**: why this problem is computationally hard \
(nonconvexity, integrality, etc.)
3. **How spatial B&B solved it**: branching decisions, relaxation quality, \
node exploration
4. **Solution interpretation**: what the optimal values mean in context
5. **What-if analysis**: what happens if key parameters change

Model details:
{model_text}

Solve result:
{result_text}"""

# ─────────────────────────────────────────────────────────────
# Model debugging assistant prompts (Feature 3F)
# ─────────────────────────────────────────────────────────────

DEBUG_SYSTEM = """\
You are a model debugging assistant for discopt. Help the user \
understand why their model behaves a certain way. Use the model \
structure and solution values to trace variable interactions."""

DEBUG_QUESTION = """\
The user asks about their solved optimization model:

Question: {question}

Model:
{model_text}

Solution:
{result_text}

Provide a specific, data-driven answer referencing variable names \
and constraint names. If the question involves sensitivity, explain \
the relationship between constraints and variable values."""
