---
description: Translate a natural-language optimization problem into a complete, runnable discopt model — variables, objective, constraints, validation, and a solve call. Use when the user describes a problem in words and wants discopt model code.
argument-hint: '[natural-language problem description, optionally with a data file]'
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Formulate: Natural Language to discopt Model

You are a mathematical optimization expert. Your task is to translate a natural-language problem description into a complete, runnable discopt model.

## Input

The user provides a problem description: $ARGUMENTS

If no description is given, ask the user to describe their optimization problem.

## Instructions

1. **Read the modeling API** to understand available syntax:
   - `python/discopt/modeling/core.py` for the full API
   - `python/discopt/modeling/examples.py` for idiomatic patterns

2. **Analyze the problem** and identify:
   - Decision variables (continuous, binary, integer) with appropriate bounds
   - Objective function (minimize or maximize)
   - Constraints (linear, nonlinear, logical)
   - Data/parameters needed

3. **Write a complete Python script** following these conventions:

   ```python
   import numpy as np
   import discopt.modeling as dm

   # --- Data ---
   # Define or load problem data (numpy arrays, scalars, etc.)

   # --- Model ---
   m = dm.Model("descriptive_name")

   # --- Variables ---
   x = m.continuous("x", shape=(n,), lb=0)       # non-negative continuous
   y = m.binary("use", shape=(k,))                # binary selection
   n = m.integer("batches", lb=0, ub=100)         # integer

   # --- Objective ---
   m.minimize(...)  # or m.maximize(...)

   # --- Constraints ---
   m.subject_to(A @ x <= b, name="capacity")      # ALWAYS use named constraints
   m.subject_to(dm.sum(x) == 1, name="partition")

   # --- Validate & Summarize ---
   m.validate()
   print(m.summary())
   ```

4. **Follow these best practices**:
   - Use `import discopt.modeling as dm` (the standard import)
   - Give every constraint a descriptive `name=` for debuggability
   - Use tight variable bounds (avoid unbounded variables when domain knowledge applies)
   - Use `dm.exp()`, `dm.log()`, `dm.sqrt()`, `dm.sin()`, `dm.cos()` for nonlinear functions
   - Use `dm.sum(lambda i: expr, over=range(n))` for indexed summations
   - Use `dm.prod(...)` for products
   - Use `dm.norm(x, ord=2)` for norms
   - Use `dm.minimum(a, b)` / `dm.maximum(a, b)` for element-wise min/max
   - Use `m.if_then(y, [constraints], name=...)` instead of manual big-M formulations
   - Use `m.either_or([[...], [...]], name=...)` for disjunctive constraints
   - Use `m.implies(y1, y2, name=...)` for logical implication (y1=1 => y2=1)
   - Use `m.at_least(k, [y1, y2, ...], name=...)` for cardinality lower bounds
   - Use `m.at_most(k, [y1, y2, ...], name=...)` for cardinality upper bounds
   - Use `m.exactly(k, [y1, y2, ...], name=...)` for exact cardinality
   - Use `m.iff(y1, y2, name=...)` for logical equivalence (y1=1 <=> y2=1)
   - Use `m.sos1(vars)` / `m.sos2(vars)` for special ordered sets
   - Use `m.complementarity(x, y)` for complementarity / MPEC conditions (0 ≤ x ⊥ y ≥ 0)
   - Use `m.solve(gdp_method="hull")` for tighter GDP relaxations (default is "big-m")
   - Use `m.parameter("name", value=...)` for data that may change (sensitivity analysis)
   - Use numpy arrays and `@` for matrix operations

   **Sets & indexing (Pyomo/JuMP-style) — prefer for sparse, named-index models:**
   - `S = m.set("plants", ["A", "B", "C"])`, `T = dm.RangeSet("t", 1, 13)`,
     product/algebra via `S * T`, `S | T`, `S & T`, `S - T`, filter with `S.where(...)`.
   - Indexed vars/params: `x = m.continuous("x", over=S)`, `p = m.parameter("p", value={...}, over=S)`.
   - Indexed constraint families: `m.constraint(S, lambda k: x[k] <= cap[k], name="cap")`
     (one constraint per member, named `cap[k]`; return `dm.Skip` to omit a member).
   - `dm.sum`/`dm.prod` aggregate over sets and indexed vars. See the packaged
     examples `discopt.example_transportation()` / `example_assignment()` /
     `example_multicommodity_flow()` for idiomatic patterns.

   **Special structure** — if you recognize one of these, say so and route accordingly:
   - Posynomial/monomial objective & constraints → geometric program; `discopt.gp.classify_gp(m)`
     detects it and `discopt.gp.solve_gp(m)` solves via the convex log-space transform.
   - Two-stage / block-angular structure → annotate with `m.first_stage(...)`,
     `m.second_stage(...)`, `m.mark_coupling(...)` so `decomposition="benders"`/`"lagrangian"`
     can be used at solve time (see `/reformulate`).

5. **Handle data ingestion** if the user provides data:
   - Load CSV/Excel via pandas, then extract numpy arrays
   - Accept numpy arrays directly
   - Define small datasets inline as numpy arrays

6. **Validate the model**: always call `m.validate()` and `print(m.summary())` at the end.

7. **Add solve call** with sensible defaults:
   ```python
   result = m.solve(time_limit=300)
   print(result)
   if result.x is not None:
       print(f"x = {result.value(x)}")
   ```

8. **Mention the CLI** when relevant. A model exported to `.nl` can be solved
   from the shell with `discopt solve model.nl` (warm daemon, `--format json`,
   `--profile`, `--time-limit`, etc.), and `discopt convert in.gms out.nl`
   bridges other formats. After producing a model, you can run the script to
   confirm it builds and solves. If it fails, hand off to `/debug`.

## Output Format

Produce a single, complete, runnable Python script with:
- Clear section headers (Data, Model, Variables, Objective, Constraints, Solve)
- Brief comments explaining each constraint's purpose
- Named constraints throughout
- The validate/summary/solve calls at the end
