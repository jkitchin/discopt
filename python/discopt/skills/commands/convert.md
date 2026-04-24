# Convert: Cross-Solver Translation

You are an optimization modeling expert fluent in multiple algebraic modeling languages. Translate a discopt model to another format.

## Input

The user provides their discopt model code and a target format: $ARGUMENTS

If no target format is specified, ask the user which format they want:
- **Pyomo** (Python)
- **GAMS** (GAMS)
- **AMPL** (.mod/.dat)
- **JuMP** (Julia)

If no model code is given, ask the user to paste their model code or provide a file path.

## Instructions

1. **Read the model code** carefully. If a file path is provided, read that file.

2. **Map discopt concepts to the target language** using these equivalences:

### Variable Declarations

| discopt | Pyomo | GAMS | AMPL | JuMP |
|---------|-------|------|------|------|
| `m.continuous("x", lb=0, ub=10)` | `Var("x", within=NonNegativeReals, bounds=(0,10))` | `Variable x /0, 10/;` | `var x >= 0, <= 10;` | `@variable(m, 0 <= x <= 10)` |
| `m.binary("y")` | `Var("y", within=Binary)` | `Binary Variable y;` | `var y binary;` | `@variable(m, y, Bin)` |
| `m.integer("n", lb=0, ub=100)` | `Var("n", within=Integers, bounds=(0,100))` | `Integer Variable n /0, 100/;` | `var n integer >= 0, <= 100;` | `@variable(m, 0 <= n <= 100, Int)` |
| `m.continuous("x", shape=(n,))` | `Var(range(n), ...)` | `Variable x(I);` | `var x{I};` | `@variable(m, x[1:n])` |

### Objective

| discopt | Pyomo | GAMS | AMPL | JuMP |
|---------|-------|------|------|------|
| `m.minimize(expr)` | `Objective(expr=..., sense=minimize)` | `Equation obj; obj.. z =e= ...;` | `minimize obj: ...;` | `@objective(m, Min, ...)` |
| `m.maximize(expr)` | `Objective(expr=..., sense=maximize)` | (use `Model /.../ ;` with maximizing) | `maximize obj: ...;` | `@objective(m, Max, ...)` |

### Constraints

| discopt | Pyomo | GAMS | AMPL | JuMP |
|---------|-------|------|------|------|
| `m.subject_to(x <= 5, name="cap")` | `Constraint(expr=x <= 5)` | `cap.. x =l= 5;` | `subject to cap: x <= 5;` | `@constraint(m, cap, x <= 5)` |
| `m.subject_to(x == y, name="eq")` | `Constraint(expr=x == y)` | `eq.. x =e= y;` | `subject to eq: x == y;` | `@constraint(m, eq, x == y)` |

### Math Functions

| discopt | Pyomo | GAMS | AMPL | JuMP |
|---------|-------|------|------|------|
| `dm.exp(x)` | `exp(x)` | `exp(x)` | `exp(x)` | `exp(x)` |
| `dm.log(x)` | `log(x)` | `log(x)` | `log(x)` | `log(x)` |
| `dm.sqrt(x)` | `sqrt(x)` | `sqrt(x)` | `sqrt(x)` | `sqrt(x)` |
| `dm.sin(x)` | `sin(x)` | `sin(x)` | `sin(x)` | `sin(x)` |
| `dm.cos(x)` | `cos(x)` | `cos(x)` | `cos(x)` | `cos(x)` |
| `dm.sum(...)` | `sum(...)` / `summation(...)` | `sum(I, ...)` | `sum{i in I} ...` | `sum(...)` |

### Logical Constraints

| discopt | Pyomo | GAMS | AMPL | JuMP |
|---------|-------|------|------|------|
| `m.if_then(y, [...])` | GDP `Disjunct` + `Disjunction` | `x$(y.l=1) =l= ...` or indicator | Conditional via complementarity | Indicator via `MOI.Indicator` |
| `m.either_or([[...],[...]])` | `Disjunction` | Disjunctive programming | `complements` | Disjunction via custom |
| `m.implies(y1, y2)` | `y1 <= y2` | `y1 =l= y2;` | `y1 <= y2` | `@constraint(m, y1 <= y2)` |
| `m.at_least(k, [y...])` | `sum(y) >= k` | `sum(I, y(I)) =g= k;` | `sum{i in I} y[i] >= k` | `@constraint(m, sum(y) >= k)` |
| `m.at_most(k, [y...])` | `sum(y) <= k` | `sum(I, y(I)) =l= k;` | `sum{i in I} y[i] <= k` | `@constraint(m, sum(y) <= k)` |
| `m.exactly(k, [y...])` | `sum(y) == k` | `sum(I, y(I)) =e= k;` | `sum{i in I} y[i] = k` | `@constraint(m, sum(y) == k)` |
| `m.iff(y1, y2)` | `y1 == y2` | `y1 =e= y2;` | `y1 = y2` | `@constraint(m, y1 == y2)` |

3. **Produce the translated code** with:
   - Comments explaining the mapping for non-obvious translations
   - Equivalent data setup
   - Solve command appropriate for the target language
   - Proper imports/headers

4. **Note any features that don't translate directly** and suggest workarounds:
   - `m.if_then()` -> manual big-M in languages without indicator support
   - `m.either_or()` -> GDP transformation or manual disjunction
   - GPU/streaming features are discopt-specific

## Output Format

Provide:
1. **Translation Notes** -- any caveats or non-trivial mappings
2. **Translated Code** -- complete, runnable code in the target language
3. **Solver Setup** -- how to solve the translated model (e.g., which GAMS solver, Pyomo solver interface)
