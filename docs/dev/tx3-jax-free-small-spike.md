# TX3 — easy-class floor: JAX-free small-model path: entry experiment → KILL

Status: **KILL** (2026-07-14). Item: `docs/dev/tenx-plan.md` §3 TX3.
Companion: `docs/dev/performance-plan.md` Appendix B (the measured floor this item
targets); F2′ (`python/discopt/_jax/uniform_relax.py:_compiled_analytic`, gated
`DISCOPT_ANALYTIC_SEPGRAD`) — the jax-free `interval_ad` mechanism TX3 hoped to
generalize.

## TL;DR

The entry experiment ran step 5 (the "is it even reachable?" question) first, as
the item instructed. It is a decisive **KILL**: `import jax` is **unavoidable on
the nonlinear solve path even with the relaxation swapped**. `import jax` is the
*first* operation of both nonlinear entry points — `_make_evaluator(model)` at
`solver.py:9179` (`_solve_nlp_bb`) and `:8959` (`_solve_continuous`) — and it
lands there before any relaxation is chosen. `_make_evaluator` →
`_jax/nlp_evaluator.py`, whose module top does `import jax` (`:19`). The NLP
evaluator is **not** the relaxation TX3 proposed to swap; it is the foundational
provider of *point-mode* objective/constraint/gradient/Jacobian callables, woven
through the entire B&B: constraint-bound inference (`:9183`), warm-start eval
(`:9193`), the per-node POUNCE NLP bound source (the dominant cost per TX0), and
every primal heuristic (multistart, fpump, diving, RINS, local branching) +
candidate validation. `interval_ad` produces *interval enclosures*
(value/grad/Hessian as intervals for a convexity certificate), **not** the
point-mode f/grad/jac POUNCE needs to run NLP iterations — so it cannot replace
the evaluator. Additionally the McCormick relaxation itself is jax-based at module
load (`_jax/relaxation_compiler.py:14`, `_jax/dag_compiler.py:33`, `mccormick_lp.py`
all `import jax.numpy`). Swapping the root relaxation to `interval_ad` removes
none of these imports. Per the item's own kill criterion — "`import jax` is
unavoidable on the nonlinear solve path even with the relaxation swapped (the
300 ms floor stays)" — **land nothing**.

## Method / measurements

Scripts: `scratchpad/trace_jax.py` (a `builtins.__import__` hook that prints the
traceback of the *first* `import jax` during a solve and checks `sys.modules`).
Env `.venv/bin/python`, `JAX_PLATFORMS=cpu`.

### Step 1 — easy-class split (nonlinear vs MILP/MIQP)

Family A on the 62-panel (`docs/dev/data/tenx-attribution.json`) = **22
instances: 16 nonlinear, 6 MILP/MIQP**. The 6 MILP/MIQP (gbd, st_miqp1–4,
st_test1) already never import jax (§2, measured in Appendix B). **TX3's
addressable set = the 16 nonlinear family-A instances** (alan, chance, dispatch,
ex1221, ex1222, ex1226, flay02m, gkocis, nvs01, nvs03, nvs04, nvs06, nvs08, oaer,
st_e13, st_e38).

### Step 2 — atom coverage over the nonlinear subset

`interval_ad._impl` covers `+ − neg * / **`(int & positive-base frac)`, exp, log,
sqrt, sin, cos, tan`; it abstains (unbounded Hessian) on `abs/max/min` and any
other `FunctionCall`. The 16 nonlinear family-A instances are monomial /
bilinear / exp / log / div / power shapes — the great majority fall inside that
table (raw estimate ≥14/16, well over the <15 kill gate). **Moot**: coverage is
irrelevant once step 5 kills the item, because the relaxation is not what pins
the jax import.

### Step 3 — feasibility of a jax-free enclosure

Confirmed the mechanism is genuinely jax-free: `interval_hessian` on an nvs03
constraint body returns a sound value enclosure `[-200, 4000]` in **~0.9 ms**
with `sys.modules` containing **no `jax`** afterward. So `interval_ad` *can*
produce sound enclosures off-jax — but that only covers the convexity/bound
envelope, not the point-mode NLP callables the node bound source needs (step 5).

### Step 4 — cost

Cold `import jax` = **0.40–0.43 s** in this container (Appendix B records
240–300 ms on the maintainer's box). `interval_hessian` build = ~0.9 ms. The
forgone saving is real (~40 % of the ~0.6 s easy-class median) — but unreachable
by TX3's mechanism.

### Step 5 — is `import jax` avoidable on the nonlinear path? (DECISIVE)

`scratchpad/trace_jax.py` on **chance** (pure-continuous → `_solve_continuous`)
and **nvs03** (integer B&B → `_solve_nlp_bb`): in both, `jax` is absent from
`sys.modules` after `import discopt` and after `from_nl`, and the **first**
`import jax` fires at `_make_evaluator` — the opening line of the solve, *before*
any relaxation. `_jax/nlp_evaluator.py:19`, `relaxation_compiler.py:14`,
`dag_compiler.py:33` all `import jax` at module scope. The evaluator is threaded
through the whole loop (`:9183/:9193/…`, POUNCE node NLP, all heuristics), so it
cannot be elided for the easy-class instances.

## Verdict: KILL

Deciding fact: **`import jax` is unavoidable on the nonlinear solve path even with
the relaxation swapped** — it enters via the point-mode NLP evaluator
(`solver.py:9179`/`:8959` → `nlp_evaluator.py:19`), which is the node bound source
and the substrate of every NLP heuristic, not the relaxation. The floor stays.

## What would have to change (recorded finding for the roadmap)

To make the nonlinear easy-class path jax-free, three subsystems must be
re-implemented off jax — this is an engine rewrite, not a floor tweak:

1. **A jax-free point-mode NLP evaluator** over the model IR (numpy forward/reverse
   AD giving `f`, `grad`, `jac`, and Hessian at points) to replace
   `_jax/nlp_evaluator.py` + `_jax/dag_compiler.py`. `interval_ad` is *interval*
   forward-mode (enclosures), not the scalar point-mode POUNCE requires — related
   but a distinct, larger walker (must match jax's numerics for the byte-level
   gate).
2. **A jax-free relaxation builder** to replace `_jax/relaxation_compiler.py` /
   `_jax/mccormick_lp.py` (both `import jax.numpy` at module load).
3. **Routing POUNCE + every primal heuristic** (multistart, fpump, diving, RINS,
   local branching, candidate validation) through the numpy evaluator, then
   proving **byte-level bound/node equivalence** to the jax path on the 16
   qualifying instances.

Rough size: multi-file, multi-hundred-line new numeric core plus an equivalence
harness — comparable to the existing `_jax` engine, not a per-solve floor patch.
Recommendation: **demote TX3**; if the floor is to be pursued, it becomes its own
large "numpy point-evaluator engine" item, gated on byte-level equivalence, not a
relaxation swap. Note Appendix B already flags the same import as "out of repo"
(upstream `importlib.metadata` plugin discovery is a large slice) — even a full
jax-free port only removes the ~300 ms if the numpy engine reaches byte-level
parity, which is the hard part.
