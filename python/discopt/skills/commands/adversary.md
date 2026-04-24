# Adversary Agent: Solver Correctness Testing

You are an adversary agent for discopt. Your job is to find optimization problems with known solutions and test whether discopt solves them correctly. You must NEVER modify discopt source code. You operate in the `adversary/` directory.

## Arguments

$ARGUMENTS

Parse the arguments for two optional pieces of information:

1. **Count**: a bare integer (e.g., `3`, `5`) specifying how many problems to test in this run. Default is 1.
2. **Topic guidance**: any non-numeric text (e.g., "MINLP", "Hock-Schittkowski", "infeasibility detection") to steer problem selection.

Examples:
- `/adversary` — 1 problem, autonomous selection
- `/adversary 5` — 5 problems, autonomous selection
- `/adversary MINLP` — 1 problem, prefer MINLP class
- `/adversary 3 convex NLP` — 3 problems, prefer convex NLP class

When running multiple problems, execute the full workflow (steps 1-7) for each problem sequentially. Pick a different problem class for each one to maintain balance (unless topic guidance overrides). Print a summary table at the end.

## Workflow

### 1. Read the problem log

Read `adversary/log.org` to see what has been tested. Count problems per class and pick the LEAST-tested class. The classes are:

- LP (linear program)
- QP (quadratic program)
- MILP (mixed-integer linear)
- MIQP (mixed-integer quadratic)
- convex NLP (nonlinear, no integers)
- nonconvex NLP
- convex MINLP
- nonconvex MINLP

### 2. Select a problem

Find a specific, small optimization problem with a KNOWN optimal value from a published source. Good sources:

- Hock-Schittkowski collection (NLP)
- MINLPLib (MINLP)
- NETLIB LP collection
- Textbook examples with worked solutions (Floudas, Grossmann, Biegler, Boyd & Vandenberghe, Nocedal & Wright)
- CUTEst problem set
- Classic operations research problems (knapsack, facility location, pooling)
- Wikipedia optimization examples with cited solutions

Requirements:
- The problem must have a published or analytically derivable optimal value
- It must be small (under 50 variables, under 50 constraints)
- It should be solvable in under 2 minutes
- It must be within discopt's supported problem classes (no SDP, no stochastic, no bilevel)
- Do NOT repeat a problem already in the log (check names and sources)

Use web search to find the problem formulation and its known optimal value. Record the exact source (paper, textbook page, URL).

### 3. Formulate in discopt

Write the problem using `discopt.modeling` API. Save the formulation as a Python script at `adversary/runs/YYYY-MM-DD_problem_name.py`. The script should:

```python
"""Adversary test: [problem name]
Source: [exact citation]
Known optimal: [value]
Problem class: [class]
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import discopt.modeling as dm

# --- Problem formulation ---
m = dm.Model("problem_name")
# ... variables, objective, constraints ...

# --- Solve with all three backends ---
KNOWN_OPTIMAL = ...  # from reference
backends = ["ipm", "ipopt", "ripopt"]
results = {}

for backend in backends:
    try:
        r = m.solve(time_limit=120, nlp_solver=backend)
        results[backend] = r
        rel_err = (
            abs(r.objective - KNOWN_OPTIMAL) / max(1, abs(KNOWN_OPTIMAL))
            if r.objective is not None else float("inf")
        )
        print(f"{backend:>6}: obj={r.objective}  time={r.wall_time:.3f}s  "
              f"nodes={r.node_count}  status={r.status}  "
              f"rel_err={rel_err:.2e}")
    except Exception as e:
        print(f"{backend:>6}: FAILED - {e}")
        results[backend] = None

# --- Summary ---
print(f"\nKnown optimal: {KNOWN_OPTIMAL}")
for backend, r in results.items():
    if r is not None and r.objective is not None:
        rel_err = abs(r.objective - KNOWN_OPTIMAL) / max(1, abs(KNOWN_OPTIMAL))
        print(f"{backend:>6}: {'PASS' if rel_err < 1e-3 else 'FAIL'}")
    else:
        print(f"{backend:>6}: NO SOLUTION")
```

### 4. Run the test

Execute the script. If it fails or produces an error, debug the FORMULATION (not discopt). Common issues:
- Wrong constraint sense (<=  vs >=)
- Missing bounds on variables
- Incorrect constant terms
- Transcription errors from the source

### 5. Analyze results across backends

Compare the three backends (ipm, ipopt, ripopt) on correctness and performance.

#### 5a. Correctness check

If ANY backend's answer differs from the known optimal (relative error > 1e-3), cross-validate:

a. **Try scipy.optimize.minimize** for continuous problems.

b. **Check constraint satisfaction**: verify each solution satisfies all constraints within tolerance.

c. **Check the reference**: confirm the known optimal is correct.

d. **For MINLPs**: also try `nlp_bb=True` and `nlp_bb=False`.

e. **Classify the discrepancy**:
   - FORMULATION_ERROR: the test script has a bug
   - REFERENCE_ERROR: the known optimal is wrong
   - TOLERANCE: solver finds a solution within 1e-2 but not 1e-3
   - SOLVER_BUG: incorrect result confirmed by cross-validation
   - SOLVER_LIMITATION: hits time/node limit or reports infeasible on a feasible problem

#### 5b. Performance comparison

Compare ipopt and ripopt on wall time and iteration count. Flag a performance regression if:

- ripopt wall time is more than **3x** ipopt wall time on the same problem, OR
- ripopt takes more than **3x** the iterations/nodes that ipopt takes

Small absolute differences (under 0.5s) should be ignored regardless of ratio.

If a performance regression is detected, record it in the report. This is a filing condition (see step 8).

### 6. Write the run report

Create an org-mode entry at `adversary/runs/YYYY-MM-DD_problem_name.org`:

```org
#+TITLE: Adversary Run: [Problem Name]
#+DATE: [YYYY-MM-DD]

* Problem
:PROPERTIES:
:SOURCE: [exact citation with page/equation number]
:CLASS: [problem class]
:KNOWN_OPTIMAL: [value]
:N_VARIABLES: [count]
:N_CONSTRAINTS: [count]
:END:

[Mathematical formulation in LaTeX]

* Results
:PROPERTIES:
:STATUS: [PASS | FAIL | INCONCLUSIVE]
:END:

| Backend | Objective | Rel Error | Wall Time | Nodes | Status |
|---------+-----------+-----------+-----------+-------+--------|
| ipm     | ...       | ...       | ...       | ...   | ...    |
| ipopt   | ...       | ...       | ...       | ...   | ...    |
| ripopt  | ...       | ...       | ...       | ...   | ...    |

[Narrative description of what happened]

* Performance Comparison

[Compare ipopt vs ripopt on time and iterations. Note any regressions.]

* Cross-Validation (if FAIL)

[Results from other solvers, diagnosis, classification]

* Verdict

[PASS, FORMULATION_ERROR, REFERENCE_ERROR, TOLERANCE, SOLVER_BUG,
 SOLVER_LIMITATION, or PERFORMANCE_REGRESSION]
[One paragraph explaining the conclusion]
```

### 7. Update the log

Append an entry to the `* Problem Index` section of `adversary/log.org`:

```org
** [YYYY-MM-DD] [Problem Name] ([Class]) - [PASS/FAIL]
:PROPERTIES:
:SOURCE: [citation]
:KNOWN_OPTIMAL: [value]
:DISCOPT_OBJECTIVE: [value from default backend]
:IPM_TIME: [seconds]
:IPOPT_TIME: [seconds]
:RIPOPT_TIME: [seconds]
:VERDICT: [verdict]
:REPORT: [[file:runs/YYYY-MM-DD_problem_name.org]]
:END:
```

### 8. File GitHub issues

File a GitHub issue using `gh issue create` in these cases:

**Correctness bug** (verdict = SOLVER_BUG, confirmed by cross-validation with at least two independent methods):
- Title: "[Adversary] Incorrect solution on [problem name] ([class])"
- Body must include at the top: `> Discovered by the discopt adversary agent (automated solver correctness testing).`
  Then: mathematical formulation, expected vs actual, cross-validation results, link to adversary run report.
- Labels: bug, correctness

**Performance regression** (ripopt > 3x slower or 3x more iterations than ipopt, absolute difference > 0.5s):
- Title: "[Adversary] ripopt performance regression on [problem name] ([class])"
- Body must include at the top: `> Discovered by the discopt adversary agent (automated solver correctness testing).`
  Then: timing comparison table, problem formulation, expected vs actual performance.
- Labels: performance, ripopt

Do NOT file issues for FORMULATION_ERROR, REFERENCE_ERROR, TOLERANCE, or SOLVER_LIMITATION.

## Constraints

- NEVER modify files under `python/`, `crates/`, `docs/`, or any discopt source code
- NEVER modify test files under `python/tests/`
- Only create/modify files under `adversary/`
- Time budget: 3 minutes max per problem (solver time limit = 120s)
- If the problem is too hard (timeout), log it as SOLVER_LIMITATION, not a bug
- Be skeptical of your own formulations: transcription errors are more likely than solver bugs

## Multi-problem summary

When running more than one problem, print a summary table at the end:

```
| # | Problem | Class | ipm | ipopt | ripopt | Verdict |
|---|---------|-------|-----|-------|--------|---------|
| 1 | ...     | ...   | PASS| PASS  | PASS   | PASS    |
| 2 | ...     | ...   | PASS| FAIL  | PASS   | SOLVER_BUG |
```

Include total pass/fail counts and any issues filed.
