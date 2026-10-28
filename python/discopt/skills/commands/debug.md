---
description: Debug a broken discopt setup, model, or solve. Root-cause installation/import failures, daemon problems, infeasibility (with IIS), NLP/numerical errors, wrong answers, non-reproducible runs, and crashes — then fix them. Use when something is broken or behaving unexpectedly, as opposed to /diagnose which interprets a successful SolveResult.
argument-hint: '[error message | traceback | model.py | "infeasible" | "daemon hangs" | ...]'
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Debug: discopt Troubleshooting

You are a discopt debugging specialist. Your job is to take a broken or
surprising situation, isolate the root cause with concrete commands, and
apply or recommend a fix. Be systematic: reproduce, narrow, fix, verify.

## Input

The user provides an error message, traceback, model file, or a description
of the misbehavior: $ARGUMENTS

If nothing is given, ask what's wrong and request the exact command, the full
traceback (not a paraphrase), and the model file path if there is one.

## Step 0 — Always establish the environment first

Before debugging anything model-specific, confirm the installation is sane.
These are cheap and rule out a huge class of problems:

```bash
python -c "import discopt; print(discopt.__version__, discopt.__file__)"
discopt about          # versions: discopt, Rust ext, Python, optional deps
discopt test           # smoke test: import, Rust extension, build+solve
```

First decide which kind of install this is, because the fixes differ. If
`discopt.__file__` is under `site-packages`, it is a **PyPI install** (most
users). If it points into a git checkout (`.../python/discopt/`), it is a
**source/developer install**.

**PyPI install:**
- **`ModuleNotFoundError: No module named 'discopt'`** → not installed in *this*
  interpreter. `pip install discopt` (or `uv pip install discopt`). If the user
  says it is installed, compare `which python` / `sys.executable` with where pip
  installed it; a notebook kernel or IDE using a different environment is the
  usual cause.
- **pip says "no matching distribution"** → Python is older than 3.12. Use a
  newer interpreter (`uv venv --python 3.12`), do not try a source build.
- **pip starts compiling Rust / fails in `maturin`** → no prebuilt wheel for this
  platform (wheels exist for macOS x86_64/arm64, Linux x86_64/aarch64, Windows
  x86_64). Building from source needs a Rust toolchain (`rustup`); say so rather
  than silently installing one.
- **`Rust ext: not available`** on a PyPI install → a broken or partial install.
  `pip install --force-reinstall discopt`.

**Source/developer install:**
- **`Rust ext: not available`**, or `AttributeError: ... has no attribute ...` on
  a Rust object → the compiled extension is missing or stale. From the **repo
  root**, run `maturin develop --release` (re-run after any change under
  `crates/`). Running maturin inside `crates/discopt-python/` installs a separate
  package and leaves the in-tree `_rust` extension stale.

**Either:**
- **Optional dep `not installed`** (jax, cyipopt, onnx, litellm, pycutest) →
  only matters if the user needs that feature; a missing extra raises an
  `ImportError` naming it. Install the matching extra: `pip install
  "discopt[jax]"`, `[nn]`, `[ipopt]`, `[llm]`, `[pyomo]`, etc. (Design of
  experiments is a separate package: `pip install discopt-doe`.)
- **JAX backend unexpected** (GPU vs CPU) → set `JAX_PLATFORMS=cpu` (or `cuda`)
  before importing. JAX is optional and not on the default solve path.

## Step 1 — Reproduce minimally

Get a deterministic, smallest-possible reproduction:
- Re-run the exact failing command. For `discopt solve`, add `--no-daemon` to
  rule the daemon out and get the traceback in your own process.
- In Python, set `m.solve(deterministic=True)` (NOT the default) so the failure
  is stable across runs. It makes the solver's role-2 wall budgets inert — those
  are what otherwise move the dual bound and the node count run to run (#1116).
  A default solve is not reproducible on a large model.
- If it's a big model, try to shrink it (fewer variables/constraints) until the
  symptom flips — that localizes the offending structure.

## Step 2 — Classify the failure and follow the matching playbook

### A. Import / install / segfault
- Run `discopt about` / `discopt test` (Step 0).
- `ImportError: discopt._rust` → extension missing: reinstall the wheel (PyPI
  install) or rebuild it (source install); see Step 0.
- Segfault on solve → almost always a numpy dtype/shape passed into the Rust
  zero-copy boundary. Check arrays are contiguous `float64`/`int64`; print
  shapes. Re-run with `--no-daemon` so the crash is in-process and traceable.

### B. Daemon problems (`discopt solve` hangs, stale results, won't start)
The `discopt solve` path warm-routes through a background daemon. When it
misbehaves:
```bash
discopt daemon status          # is it alive? which version?
discopt daemon kill            # force-kill a wedged daemon (then re-solve)
discopt solve model.nl --no-daemon   # bypass entirely to confirm it's the daemon
discopt solve model.nl --hard-timeout 60   # daemon SIGKILLs itself on overrun
```
A daemon pinned to an old install serves stale code after an upgrade → `kill`
it so the next solve respawns fresh.

### C. Infeasible model (`status == "infeasible"`)
Don't guess — compute the Irreducible Infeasible Subsystem:
```python
import discopt

iis = discopt.compute_iis(m)  # deletion-filtering; exact for LP/MILP/convex
print(iis.summary())  # the minimal conflicting constraints + bounds
# iis.constraints, iis.variable_bounds, iis.proven_irreducible
```
The IIS is the smallest set of constraints/bounds that cannot all hold at once —
fix one of them. Common culprits: a typo'd RHS, a sign error, mutually exclusive
bounds (`lb > ub`), or an equality that over-constrains. If `compute_iis` is slow
on a large model, pass `time_limit=...` (best-effort on nonconvex).
For bound-only contradictions, FBBT will surface them directly:
```python
from discopt import tightening

bt = tightening.fbbt_box(m)
if bt.infeasible:
    print("bounds are contradictory before any solve")
```

### D. NLP sub-solver stalls / `iteration_limit` / numerical errors
The continuous relaxations are solved by an NLP backend (POUNCE by default, or
cyipopt). Stalls and `Restoration failed` usually mean bad scaling or bad bounds:
- **Scaling**: variables/constraints spanning many orders of magnitude. Rescale
  so values are O(1).
- **Bounds**: discopt warns when bounds approach the ±9.999e19 default — those
  exceed the NLP safe threshold (~1e15). Always set finite, reasonable bounds.
- **Bad starting point**: pass `initial_solution={var: value, ...}`.
- Switch backend to compare: `m.solve(nlp_solver="ipopt")` (needs cyipopt) and
  raise its limits via `m.solve(max_iter=5000, tol=1e-6, acceptable_tol=1e-4)`.
- For least-squares/estimation models, `m.solve(gauss_newton=True)`.

### E. Wrong / suspicious answer
If you doubt the result is correct:
- `m.validate()` on the model (catches malformed constraints, aliased names).
- `result = m.solve(validate=True)` runs a post-solve KKT check; inspect
  `result.validation_report`.
- Re-solve with a tighter relaxation to see if the bound moves: `rlt=True`,
  `psd_cuts=True` (QCQP), or `solver="amp"`. If the objective changes, the earlier run terminated on a weak
  bound, not a true optimum.
- For nonconvex problems, confirm you got a *global* solve (spatial B&B), not a
  local NLP: check `result.status`/`bound`. Force spatial B&B with `nlp_bb=False`.
- Cross-check against another backend: `m.solve(solver="amp")`, or convert and
  solve elsewhere (`discopt convert model.py-derived.nl out.mps` → external solver).

### F. Slow solve / huge B&B tree
- Read layer profiling on the result (`rust_time`, `jax_time`, `python_time`).
  High rust = tree too big (tighten bounds/relaxation); high jax = NLP evals
  expensive (simplify nonlinearity); high python = orchestration (normal for
  tiny solves).
- Tighten the relaxation: `rlt=True`, `psd_cuts=True` (QCQP), `solver="amp"` (bilinear),
  `gdp_method="hull"` for disjunctive models, `cutting_planes=True`.
- Tighten bounds (manually or `tightening.fbbt_box`) — directly strengthens
  McCormick.
- Set an acceptable gap: `gap_tolerance=1e-2` if 1% is good enough.
- Hand off detailed performance analysis to `/diagnose`.

### G. Crash / exception inside solve
- Capture the **full** traceback with `--no-daemon` (or in Python, no try/except).
- Read the deepest discopt frame: `python/discopt/solver.py` (orchestration),
  `python/discopt/_relax/` (relaxation/NLP eval), `crates/discopt-core/` (Rust IR,
  B&B, .nl parser). Match the error to the layer.
- Reproduce against a packaged example to isolate model-specific vs solver bugs:
  `discopt.example_simple_minlp()`, `example_transportation()`.

## Step 3 — Fix and verify

- Apply the smallest fix that addresses the root cause (not the symptom).
- Re-run the minimal reproduction from Step 1 and confirm the symptom is gone.
- If you changed solver options, state which option fixed it and why.
- If the root cause is a discopt bug (not user error), say so plainly, point to
  the file/line, and suggest a minimal reproduction the user can file as an issue.

## Output Format

1. **Diagnosis** — one-line root cause (or "needs more info: <what>").
2. **Evidence** — the commands you ran and the key output that proves it.
3. **Fix** — concrete change (code/option/command), applied or recommended.
4. **Verification** — how you confirmed (or how the user can confirm) it's fixed.
