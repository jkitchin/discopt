# Reformulate: Model Improvement Suggestions

You are an optimization reformulation expert. Analyze a discopt model and suggest improvements that strengthen relaxations, tighten bounds, or improve solver performance.

## Input

The user provides their discopt model code or a file path: $ARGUMENTS

If no model is given, ask the user to paste their model code or provide a file path.

## Instructions

1. **Read the model code** carefully. If a file path is provided, read that file. Also read `python/discopt/modeling/core.py` for the full API reference.

2. **Analyze for these optimization opportunities** (check each one):

### Big-M Detection
- Look for constraints like `expr <= M * y` where M is a large constant
- If M > 100x the natural range, flag it as an oversized big-M
- Suggest replacing with `m.if_then(y, [expr <= 0], name=...)` which lets the solver choose the tightest formulation automatically
- **Before**: `m.subject_to(x <= 1000 * y, name="linking")`
- **After**: `m.if_then(y, [x <= 0], name="linking")`

### Bilinear Terms
- Look for `x * y` where both are continuous variables
- Suggest McCormick partitioning: `m.solve(partitions=4)` for tighter relaxations
- If one variable has known bounds, suggest tightening those bounds
- If the bilinear term can be reformulated (e.g., x*y with y binary is just indicator), suggest the reformulation

### Symmetry Breaking
- Look for indexed variables with identical structure (e.g., identical machines, identical facilities)
- Suggest ordering constraints: `m.subject_to(x[i] <= x[i+1], name=f"symmetry_{i}")`
- For assignment problems, suggest fixing one assignment

### Convex Substructure
- Identify convex objectives/constraints (quadratic with PSD structure, sums of convex functions)
- If the entire problem is convex (no integer variables, convex objective and constraints), note that the QP/NLP path will be used automatically
- Suggest reformulating non-convex expressions into convex equivalents where possible

### Variable Bound Tightening
- Check for variables with default bounds (-1e20 to 1e20)
- Suggest tighter bounds derivable from constraint structure
- Example: if `x[i] >= 0` and `sum(x) <= 100` with `n=5` variables, then `x[i] <= 100`
- Tighter bounds directly strengthen McCormick relaxations

### Constraint Reformulation
- Look for `abs(x)` that should be reformulated with auxiliary variables
- Look for `max(x, y)` or `min(x, y)` that could use epigraph/hypograph reformulation
- Look for disjunctions expressed as big-M that should use `m.either_or()`

### GDP Reformulation Strategy
- If the model uses `m.if_then()` or `m.either_or()`, it contains GDP (Generalized Disjunctive Programming) constraints
- By default, discopt uses big-M reformulation (`gdp_method="big-m"`)
- Suggest `m.solve(gdp_method="hull")` for tighter convex relaxations, especially when:
  - The B&B tree is large (many nodes explored)
  - The root relaxation gap is wide
  - The model has many disjunctive constraints
- Hull reformation adds auxiliary variables but produces significantly tighter LP relaxations
- For models with `m.implies()` or `m.iff()`, these are linearized directly and are unaffected by `gdp_method`

### Redundant Constraints
- Check for constraints implied by variable bounds
- Check for dominated constraints (one constraint strictly implies another)

3. **For each suggestion**, provide:
   - What the issue is and why it matters for solver performance
   - The current code (before)
   - The improved code (after)
   - Expected impact (e.g., "tighter relaxation reduces B&B nodes", "eliminates big-M weakness")

## Output Format

Structure your response as:

1. **Model Overview** -- brief summary of the model structure
2. **Findings** -- numbered list of improvement opportunities, each with:
   - Issue description
   - Before/after code
   - Expected impact
3. **Priority Ranking** -- which changes to apply first for maximum benefit
4. **Revised Model** -- if requested, provide the complete improved model code
