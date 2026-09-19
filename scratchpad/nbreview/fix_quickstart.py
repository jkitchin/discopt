#!/usr/bin/env python3
"""Correctness fixes for docs/notebooks/quickstart.ipynb (issue #1362)."""
import sys

sys.path.insert(0, "scratchpad/nbreview")
from nbedit import NB  # noqa: E402

nb = NB("quickstart")

# --- 1. Installation prose: the default NLP backend is not a "pure-JAX IPM". ---
nb.sub(
    0,
    "discopt ships with a pure-JAX interior-point method (IPM) as the default NLP backend. For the optional [Ipopt](https://coin-or.github.io/Ipopt/) backend:",
    "discopt's default NLP engine is **POUNCE** (`pounce-solver`), a pure-Rust port of the\n"
    "Ipopt interior-point method {cite:p}`Wachter2006`. It is a core dependency, so nothing\n"
    "extra is needed. For the optional [Ipopt](https://coin-or.github.io/Ipopt/) backend,\n"
    "used here to cross-check POUNCE:",
)

# --- 2. §3 intro: a MINLP does not necessarily open a B&B node any more. ---
nb.sub(
    11,
    "When a model contains binary or integer variables, discopt uses spatial Branch & Bound {cite:p}`Land1960` to find the global optimum.",
    "When a model contains binary or integer variables, discopt solves it to **global**\n"
    "optimality. Spatial Branch & Bound {cite:p}`Land1960` is the general engine, but it is\n"
    "not the only route: at solve entry discopt classifies the model, and a *convex* MINLP\n"
    "(such as the convex MIQP below) is handed to the MIP--NLP family --- outer\n"
    "approximation {cite:p}`Duran1986` --- which certifies the optimum without ever opening a\n"
    "B&B node. That is why `result.node_count` reads `0` in the next two cells; the cells\n"
    "print `result.algorithm_route` so the route is visible rather than inferred, and solve a\n"
    "second time with `solver=\"bb\"` to show the spatial B&B arm on the same model.",
)

# --- 3. Binary example: show the route, and the B&B arm, instead of a bare 0. ---
nb.set_source(
    12,
    '''m = dm.Model("binary_example")
x = m.continuous("x", lb=0, ub=5)
y1 = m.binary("y1")
y2 = m.binary("y2")

m.minimize((x - 3) ** 2 + 2 * y1 + 3 * y2)
m.subject_to(x <= 5 * y1)
m.subject_to(x <= 4 * y2)

result = m.solve()
print(f"Status:    {result.status}")
print(f"Objective: {result.objective:.6f}  (expected: 5.0)")
print(f"x  = {result.x['x']:.6f}  (expected: 3.0)")
print(f"y1 = {result.x['y1']:.6f}  (expected: 1.0)")
print(f"y2 = {result.x['y2']:.6f}  (expected: 1.0)")
print(f"Nodes explored: {result.node_count}")
print(f"Route:          {result.algorithm_route}")

# The same model down the spatial-B&B arm, so "0 nodes" above is read as a route
# choice and not as "B&B did nothing". Both must agree on the optimum.
bb = m.solve(solver="bb")
print(f"\\nsolver='bb':    obj={bb.objective:.6f}, nodes={bb.node_count}")
assert result.node_count == 0, "the convex MIQP route should not open a B&B node"
assert bb.node_count > 0, "the solver='bb' arm must actually branch"
assert abs(result.objective - bb.objective) < 1e-6, "the two routes must agree"''',
    expect="binary_example",
)

# --- 4. Integer example: same treatment. ---
nb.sub(
    14,
    'print(f"Nodes explored: {result.node_count}")',
    'print(f"Nodes explored: {result.node_count}")\n'
    'print(f"Route:          {result.algorithm_route}")\n'
    "assert abs(result.objective) < 1e-6, \"this model's optimum is exactly 0\"",
)

# --- 5. §4: the AD engine is not JAX. ---
nb.sub(
    15,
    "All of these are automatically differentiated by JAX.",
    "All of these are differentiated by the **POUNCE AD tape** in the Rust core. JAX is not\n"
    "on the default solve path --- it is an optional dependency of the differentiable-solve\n"
    "and learned-relaxation subsystems only, and a plain `solve()` imports no `jax` module.",
)

# --- 6. §5 result table: jax_time is a stale name and rust/jax/python is not a partition. ---
nb.sub(
    19,
    "| `result.node_count` | Number of Branch & Bound nodes explored (0 for pure NLP) |\n"
    "| `result.rust_time` | Time spent in Rust backend |\n"
    "| `result.jax_time` | Time spent in JAX (AD, relaxation evaluation) |\n"
    "| `result.python_time` | Time spent in Python orchestration |",
    "| `result.node_count` | Number of Branch & Bound nodes explored (0 for a pure NLP, and 0 on the MIP--NLP routes) |\n"
    "| `result.algorithm_route` | Which route solved it, or `None` for plain spatial B&B |\n"
    "| `result.rust_time` | Time spent in the Rust backend |\n"
    "| `result.python_time` | Time spent in Python orchestration |\n"
    "| `result.pounce_time` | Time inside POUNCE --- a **subset** of `rust_time`, not a peer of it |\n"
    "| `result.jax_time` | Time inside JAX --- `0.0` on any default solve, since JAX is not on that path |\n"
    "\n"
    "```{warning}\n"
    "`rust_time` and `python_time` partition the wall clock; `pounce_time` and `jax_time`\n"
    "are **subsets** of them. Subtracting a subset from the wall clock double-counts, so\n"
    "`wall_time - jax_time - rust_time` is not \"the rest\" --- it is meaningless.\n"
    "```",
)

# --- 7. Layer profiling cell: print it as nesting, not as a three-way split. ---
nb.sub(
    20,
    '''print("=== Layer Profiling ===")
print(f"Rust time:    {result.rust_time:.4f}s")
print(f"JAX time:     {result.jax_time:.4f}s")
print(f"Python time:  {result.python_time:.4f}s")''',
    '''print("=== Layer Profiling ===")
# rust + python partition the wall clock; pounce/jax are subsets of that total.
print(f"Rust time:    {result.rust_time:.4f}s")
print(f"Python time:  {result.python_time:.4f}s")
print(f"  of which POUNCE: {result.pounce_time:.4f}s")
print(f"  of which JAX:    {result.jax_time:.4f}s  (0.0 -- JAX is not on the solve path)")
partition = result.rust_time + result.python_time
assert abs(partition - result.wall_time) < 0.05 * max(result.wall_time, 1e-3) + 1e-3, (
    f"rust+python ({partition:.4f}s) should account for the wall clock "
    f"({result.wall_time:.4f}s)"
)''',
)

# --- 8. §6: the .nl demo resolved to docs/python/... and silently skipped both instances. ---
nb.sub(
    21,
    "Here we load two instances from the included MINLPLib test data.",
    "Here we load two instances from the included MINLPLib test data and check each against\n"
    "its reference optimum. A missing corpus raises rather than printing `skipping`: a demo\n"
    "that quietly does nothing reads as a pass.",
)
nb.set_source(
    22,
    '''from pathlib import Path

# The notebook is executed from its own directory (jupyter-book) but may also be run
# from the repo root, so resolve against both rather than guessing one. The previous
# path, "../python/tests/data/minlplib", resolved to docs/python/... from here --- it
# never existed, and the exists() guard turned that into a silent "skipping".
NL_DIRS = [
    Path("../../python/tests/data/minlplib"),  # from docs/notebooks/
    Path("python/tests/data/minlplib"),  # from the repo root
]
nl_dir = next((d for d in NL_DIRS if d.is_dir()), None)
if nl_dir is None:
    raise FileNotFoundError(
        "MINLPLib test data not found in any of: " + ", ".join(str(d) for d in NL_DIRS)
    )

# Reference optima: nvs03 from python/tests/data/known_optima.toml, st_e13 from the
# in-repo reference in python/tests/test_monomial_lp_bound.py.
instances = {
    "st_e13": {"opt": 2.0, "type": "NLP"},
    "nvs03": {"opt": 16.0, "type": "MINLP"},
}

checked = 0
for name, info in instances.items():
    nl_path = nl_dir / f"{name}.nl"
    if not nl_path.exists():
        raise FileNotFoundError(nl_path)

    model = dm.from_nl(str(nl_path))
    # .nl models work with any backend; here we use Ipopt for demonstration.
    result = model.solve(time_limit=30, nlp_solver="ipopt")

    error = abs(result.objective - info["opt"])
    print(
        f"{name} ({info['type']}): status={result.status}, "
        f"obj={result.objective:.6f}, expected={info['opt']:.6f}, "
        f"error={error:.2e}, time={result.wall_time:.2f}s"
    )
    assert result.status == "optimal", f"{name}: {result.status}"
    assert error < 1e-6, f"{name}: {result.objective} != {info['opt']}"
    checked += 1

assert checked == len(instances), f"only {checked} instance(s) were checked"
print(f"\\n{checked} instance(s) solved and checked against their reference optima.")''',
    expect="nl_dir",
)

# --- 9. §7 options table: nlp_solver's default is "pounce", and "ipm" is not a JAX IPM. ---
nb.sub(
    23,
    '| `nlp_solver` | `"ipm"` | NLP backend: `"ipm"` (JAX IPM {cite:p}`Nocedal2006`), `"ipopt"` (Ipopt {cite:p}`Wachter2006`), or `"pounce"` (Rust) |',
    '| `nlp_solver` | `"pounce"` | NLP backend: `"pounce"` (the Rust Ipopt port {cite:p}`Wachter2006`), `"ipopt"` (cyipopt), or `"simplex"`. `"ipm"`/`"sparse_ipm"` are back-compat aliases that resolve to POUNCE {cite:p}`Nocedal2006` --- the pure-JAX IPM they once named is retired |',
)

nb.save()
