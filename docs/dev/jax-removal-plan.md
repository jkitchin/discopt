# Remove JAX from the discopt solve path

**Status: in progress.** Tracking issues #74 (attribution) and #75 (tape
translator). Work lands on `refactor/remove-jax-from-solve-path` via draft
PR #922; `main` is untouched until the whole thing verifies end to end.

| stage | state |
|---|---|
| 0 — layer-time attribution | **done** — `c3a3d648`, `7fc69f7f` |
| 1 — DAG → `NlExpr` tape translator | **done, not wired in** — `1917a17b` (+ probe hardening `360a1e69`) |
| 2 — separation tangents → translator | not started; §5 panel required |
| 3 — NLP derivatives → translator | not started; §5 panel required |
| 4 — enforcement (`sys.modules` assert) | not started |

**Nothing blocks Stage 2/3.**

> ### ⚑ RETRACTION 2026-08-04 — the deep-sharing blocker was a stale build
>
> An earlier revision of this document, the `1917a17b` commit message, and a
> strict non-running `xfail` all recorded a blocker: a deeply shared chain
> (`node = node*node + node`) failing to terminate at depth 10, where the DAG
> holds ~20 distinct nodes. **That measurement was against a stale locally-built
> `pounce` extension** — the installed `_pounce.abi3.so` predated pounce #470 and
> did not export `NlExpr` at all, which also meant
> `python/tests/test_75_nl_expr_compiler.py` was skipping in its entirety behind
> `pytest.importorskip("pounce")` locally. (CI installs pounce from git `main`, so
> it *was* running there.)
>
> Re-measured after `make python-ext` against pounce `main` (`fa81f9d8`, PR #474,
> "stop copying operands" — operators now reference operands through a `Cse` node):
>
> | depth | distinct DAG nodes | tree nodes | lower | eval+grad |
> |---|---|---|---|---|
> | 25 | 50 | 3.4e7 | <0.001 s | <0.001 s |
> | 30 | 60 | **1.07e9** | 0.126 s | <0.001 s |
>
> At depth 30 the value matches the scalar recurrence `n←n²+n` **exactly** (rel
> 0.0) and the gradient to **2.60e-16**. The `xfail` is replaced by an executing
> test, `test_deeply_shared_chain_stays_linear_in_distinct_nodes`. Verified
> discriminating: with the translator's `id(expr)` memo disabled, the same chain
> does not return in 120 s.
>
> **Lesson, and it is the third naming/staleness miss in this work:** rebuild the
> native extension and assert the marker symbol before measuring against it
> (CLAUDE.md §8). `importorskip` on a stale optional dependency turns a whole test
> file into a silent no-op that reads as a pass (§6).

This document is the measured record as well as the plan: every number in it was
produced in-repo, and the superseded sections are kept deliberately so a
falsified assumption is visible rather than quietly rewritten.

> ## ⚑ RESCOPE 2026-08-04 — one backend, not two
>
> **What changed:** pounce shipped [#469](https://github.com/jkitchin/pounce/issues/469).
> `pounce.NlExpr` now exposes a Rust AD tape to Python with `eval(x)` and
> `gradient(x)`, plus `parse_nl_text` and `build_nl_problem`.
>
> **Measured, this session:**
> * `NlExpr.eval`/`gradient` agree with analytic truth **exactly** (value reldiff
>   `0.000e+00`, gradient maxabsdiff `0.000e+00`) on `exp(x)*y + tanh(x) + erf(y)` —
>   an expression `interval_ad` cannot evaluate at all.
> * **All 30** discopt DAG operators are expressible in `NlExpr`: 20 native
>   (incl. `erf`, `atan2`, `min`, `max`, `acosh`), 10 by decomposition
>   (`abs`→`max(a,-a)`, `sign`→`select`, `log1p`, `log2`, `sigmoid`, `softplus`,
>   `entropy`, `centropy`, `prod`, `signpower`).
>
> **Why this rescopes the work:** Jobs 1 and 2 were being sent to *different*
> replacement backends. They no longer need to be.
>
> | | old plan | rescoped |
> |---|---|---|
> | Job 1 — NLP derivatives | pounce tape | pounce tape |
> | Job 2 — separation tangents | `interval_ad` (**6 of 30** operators) | **same pounce tape** (30 of 30) |
> | shared prerequisite | none | **DAG → `NlExpr` translator** |
>
> **Consequences:**
> * Extending `interval_ad` by ~14 atoms (old "option B") is **dead work** — it would
>   reimplement in Python numerics that already exist in Rust.
> * Graduating `DISCOPT_ANALYTIC_SEPGRAD` is **dropped**. Its panel passed
>   cert-clean (0 unsound bounds, 0 cert regressions, 0 objective drift) but was
>   **neutral** on benefit — 48 of 49 instances identical, 1 better, 0 worse — and
>   §5 keeps a neutral flag OFF. It also bought nothing in isolation:
>   `nlp_evaluator.py:22` imports JAX on every nonlinear solve regardless, so
>   flipping it would have changed a default on a bound-changing path for zero
>   measurable gain — the `DISCOPT_CUT_INHERIT` mistake exactly.
> * The already-committed probe hardening (`360a1e69`) stays: it is default-inert
>   and improves `interval_ad` as a secondary fallback either way.
>
> **Nothing missing from pounce.** All three asks in pounce#469 shipped in PR #470:
> `parse_nl_text` + `NlExpr`/`Tape::build`, an `Erf` TapeOp, **and** the HVP —
> exposed as `NlProblem.hessian_vector_product(x, v, lam=None, obj_factor=1.0)`
> computing `(obj_factor·∇²f + Σ lam_i·∇²g_i)·v`, verified on heatexch_gen3
> (n=580, finite) and accepting **multiple directions** in one call (column-major
> `n × k`).
>
> *Correction:* an earlier revision of this document said the HVP was unexposed.
> That was wrong — it came from grepping `pounce-py/src/` for the Rust-internal
> name `hessian_directional` and reading zero hits as "no binding". The binding is
> named `hessian_vector_product`. Check the live Python surface, not source by
> name; this is the second name-based miss in this work (POUNCE also profiles as
> `<function Problem.solve>`, with "pounce" in neither filename nor function name).
>
> **Stage 3 is therefore unblocked** — no pounce-side prerequisite remains.
>
> The staged structure below is superseded by **§ Rescoped stages** at the end of
> this document. Everything above it is retained as the measured record.

## Context

JAX is a core dependency of discopt's solve path, but measurement shows it is
doing far less than the architecture implies, in a mode that gets none of its
benefit.

Established by measurement in this investigation:

- Of `_jax/`'s 68,408 LOC, **66% never import jax**; only 14,050 LOC (24 files)
  use a real JAX transform. The package name is a misnomer.
- JAX is imported **iff the model has ≥1 nonlinear constraint**. A quadratic
  *objective* does not trigger it (LP/MILP/QP/MIQP are JAX-free); integrality is
  irrelevant. 10 of 66 corpus instances never import it.
- On the solve path JAX does **exactly two jobs**:
  - **Job 1 — NLP subsolve derivatives** (`_jax/nlp_evaluator.py`): f, ∇f, g, J,
    ∇²L, HVPs, sparsity. Every solver family funnels through it — spatial B&B,
    AMP (`solvers/amp.py`), and OA (`solvers/oa.py:1562-1564`).
  - **Job 2 — separation tangents** (`_jax/uniform_relax.py:813-820`): `g(x₀)` and
    `∇g(x₀)` for the Kelley cutting-plane loop on composite convex lifts.
- **jit is unavailable on Job 2 by design.** `uniform_relax.py:441`: XLA fusion
  reorders floats (drift ~7e-15), and the Kelley loop is path-dependent, so that
  changes the cut sequence and the bound. The workaround (`_TracedEvalFn`) walks
  the jaxpr in Python via `eval_jaxpr` — JAX reduced to an interpreted IR.
- **Job 1's cost is first-call tracing, not evaluation.** `evaluate_hessian_values`
  on heatexch_gen3: 398 calls, 4.32 s total, **4.133 s of it call #1**; the rest
  average 0.463 ms. A Rust tape has no trace step at all.

Both replacements already exist in-tree. The goal is to reach **zero `import jax`
reachable from `Model.solve()` and no `jax.jit` on that path**, while keeping the
post-solve differentiable-optimization features on JAX.

**Out of scope — these stay on JAX:** `_jax/differentiable*.py`,
`_jax/pounce_layer.py`, `_jax/parametric.py`, `parametric.py`, `_jax/icnn*.py`,
`modeling/implicit.py`.

---

## Stage 0 — Fix the attribution MODEL (prerequisite, do not skip)

**Why first:** every before/after check below is measured with these counters, and
they are structurally wrong — not just miscalibrated. Two defects:

**(a) `jax_time` over-reports.** Measured: 9 corpus instances report `jax_time`
0.15–0.28 s on solves where `jax` never entered `sys.modules` (`alan`: 0.175 s of a
0.204 s solve). On heatexch_gen3 at a 30 s budget the counter says **20.72 s** while
cProfile measures **13.51 s** of real jax/jaxlib frames — ~50% over. Likely culprit:
the `_native_jax_s` phase timers (`solver.py:1088-1215`) wrap phases that call into
Rust/pounce, not JAX.

**(b) The bucket model cannot express the system.** There are **four** execution
domains — discopt-Rust, JAX/XLA, pounce-Rust, real Python — and three buckets, of
which one is a residual: `solver.py:1271` computes
`python_time = wall_time - rust_total - jax_total` (also `:7173`, `:7198`, `:7315`).
Consequences, both already observed:
  - **pounce is invisible.** Measured ~15.5 s of a 63 s heatexch_gen3 solve in
    pounce `Problem.solve` + `solve_problem_batch` — silently charged to "Python."
  - **Synchronous Rust calls are charged to Python.** Already documented at
    `docs/dev/performance-plan.md:1218` — *"the `python` bucket is dominated by
    synchronous Rust-extension calls (counted as Python)."*

A residual bucket makes every attribution error invisible: you cannot distinguish
"Python was slow" from "our instrumentation missed something." Fixing only (a) moves
the error into the residual rather than removing it.

**Change:**
1. Fix the `jax_time` phase timers so they charge only JAX frames.
2. Add a **`pounce_time`** bucket, instrumented at the FFI boundary
   (`solvers/nlp_pounce.py`, `solvers/lp_pounce.py`, `solvers/milp_pounce.py`,
   `solvers/qp_pounce.py`), not by phase.
3. Measure `python_time` **directly** instead of by subtraction.
4. Add an explicit **`unattributed_time = wall - Σ(measured buckets)`** and report
   it. This is the key change: it converts a silent error into a visible number.

**(c) The deepest defect — `jax_time` and `python_time` are NOT disjoint.** Measured
by splitting profiled jax frames into interpreted-Python vs native-XLA:

| instance | JAX interpreted Python | JAX native XLA | verdict |
|---|---|---|---|
| heatexch_gen3 | **13.34 s** | 0.56 s | **96% Python** |
| tspn08 | **0.79 s** | 0.04 s | **95% Python** |

"JAX time" is overwhelmingly **Python-interpreter time** (tracing, `core.bind`,
primitive dispatch), not accelerated computation. Modelling `jax_time` as a *peer*
of `python_time` was a category error: they overlap almost entirely. This is the
root cause of the confusion — "was it JAX or Python?" was never well-formed.

### Revised bucket model (owner steer: split on native vs interpreted)

Two **primary, disjoint** buckets, both measured at FFI boundaries:

- **`rust_time`** — all native Rust: `discopt._rust.*` **and** pounce. pounce is
  Rust; it belongs here, not as a separate top-level domain.
- **`python_time`** — all interpreted Python.

Two **diagnostic sub-labels**, documented as subsets, not peers:

- **`jax_python_time`** ⊂ `python_time` — Python time inside `jax/*`.
- **`pounce_time`** ⊂ `rust_time` — the NLP/LP subsolver's share.

Plus **`unattributed_time` = wall − rust − python**, reported, so gaps are visible.

**Naming hazard (measured):** pounce's PyO3 entry point profiles as
`<function Problem.solve at 0x...>` — "pounce" appears in **neither** the filename
(`~`) nor the function name. A predicate matching on "pounce" silently reports
**0.00** (this bit my own probe). Instrument by wrapping the call sites in
`solvers/*_pounce.py`, never by name-matching profile frames.

**Ground truth to reconcile against (tspn08, 20 s budget):** pounce **11.69 s**
(78%), JAX-Python 0.79 s (5%), discopt-Rust 0.24 s — currently reported as
`jax_time` **18.82 s**.

Instrument at FFI boundaries, not around phases — phase timers wrap mixed work,
which is exactly how defects (a) and (b) arose.

**Files:** `python/discopt/solver.py` (accumulators `:6523-6524`, roll-up
`:1241-1271`), `python/discopt/benchmarks/runner.py` (`:36-38` bucket fields),
`discopt_benchmarks/perf/measure.py`, the four `solvers/*_pounce.py` boundaries.

**Before/after check:**
- *Before:* `jax_time > 0.01` on ≥9 instances where `"jax" not in sys.modules`;
  pounce time unrepresentable; `python_time` = residual.
- *After:*
  - `jax_time == 0.0` on **every** instance where JAX is never imported;
  - `jax_time` within 15% of cProfile's jax-frame total on heatexch_gen3 / tspn08;
  - `pounce_time` ≈ 15 s on heatexch_gen3 (cProfile cross-check);
  - **`unattributed_time` < 10% of wall** on the perf panel — and if it is not, the
    instrument is incomplete and says so, which is the whole point.
- **Regression tests:** (i) `jax_time == 0.0` for a pure MILP solve (`st_test1`) —
  fails before, passes after; (ii) `unattributed_time` is reported and bounded on a
  nonlinear solve.

**Does this end the profiling confusion?** Largely, and it is the only stage that
addresses it. After Stage 0 the four domains are separately measured and the
leftover is named rather than hidden. It does **not** make cProfile unnecessary for
fine-grained questions ("*which* JAX call?") — the buckets answer "where did the
time go," not "why."

### Disposition of the #902 accounting tests

`python/tests/test_902_native_kernel_accounting.py` encodes the *phase-level*
attribution as intended behavior and must be rewritten — **preserving its intent,
replacing its mechanism**:

- `test_native_kernel_seed_time_is_charged_to_jax` (`:89`) asserts
  `jax_time >= 0.20 s` from a sleep injected into the seed phase, on the premise
  that "the seed phase is JAX work". It is not — the seed runs NLP solves whose
  optimization is pounce's Rust IPM, with JAX supplying only derivative callbacks.
  Rewrite as `test_native_kernel_seed_time_is_attributed`: assert the seed's time
  lands in a bucket (not dropped), under the new rust/python split.
- `test_native_kernel_time_split_is_consistent` (`:122`) asserts the identity
  `python_time == wall − rust − jax`. That identity changes by construction once
  `jax` is a subset of `python`; replace with `python_time + rust_time +
  unattributed_time == wall` and keep the no-negative-bucket assertions.

**#902's real guarantee must survive:** a ~7 s Rust-heavy solve must never report as
"pure Python". That was a genuine bug and the new model must still catch it — keep
the sleep-injection technique, which is load-independent (CLAUDE.md §9), and keep
the executed-call-count assertions (§6).

**Bound impact:** none (instrumentation only). Assert node_count and objective
unchanged on the cert panel.

---

## ⚠ CORRECTIONS (2026-08-03, measured — supersede text below)

Three findings from a 27-instance instrumented sweep change the plan's premises:

**1. Place 1 is NOT bound-neutral. The "bitwise" bar is unachievable.**
`_compiled_analytic` vs JAX at 128 lift-node compilations: **max |Δvalue| 2.38e-07,
max |Δgrad| 2.79e-09** (typical 1e-16..1e-12). The repo rejected `jax.jit` on this
very path over **7e-15** drift (`uniform_relax.py:441-444`). At 2.8e-09 the analytic
path is six orders *looser* than the thing already ruled bound-changing.
→ **Place 1 moves to the CLAUDE.md §5 differential panel, same bar as Place 2.**
My earlier "bounds bit-identical" result (5 instances) is not contradicted — it
means the drift did not perturb the cut sequence *on those instances*. It is not a
theorem, and 5 instances is not a panel.

**2. Coverage is better than feared: 96.2% measured, effectively 100%.**
133 lift-node compilations, **128 analytic, 5 fallbacks** (gkocis 2, oaer 2,
dispatch 1). All 5 contain only *covered* atom kinds — they are **probe/domain
failures, not atom gaps**: the one-shot midpoint probe (`uniform_relax.py:859-862`)
hits `log(≤0)` or a NaN midpoint from an infinite-bound variable. Fix is probe
hardening (finite-clipped interior point, or per-round eval with NaN-skip mirroring
`mccormick_lp.py:2033`), not new interval-AD rules.
*Caveat:* the atom table is only `{exp, log, sin, cos, tan, sqrt}` plus arithmetic
and integer powers (`interval_ad.py:924-956`; multi-arg calls bail at `:929`), and
the 27-instance sample was light on trig/hyperbolic lifts. One wider corpus sweep
before committing.

**3. There are SIX jobs, not two.** Jobs 3–6 are flag-gated and were **never loaded
on 12 default solves** [measured], but "zero `import jax` on the solve path" must
still account for them:
- **Job 3** McCormick/αBB node dual bounds — `mccormick_subgradient`,
  `mccormick_nlp`/`mccormick_evaluator`, `batch_evaluator`+`relaxation_compiler`
  tree, `alphabb`. Gated by `mccormick_bounds="nlp"` / rigorous-alpha.
  `alphabb`'s `jax.hessian` has a natural in-tree replacement: `interval_hessian`.
- **Job 4** JAX QP-IPM last resort (`solver.py:15510`) — fires only when pounce
  fails. Cannot be replaced *by pounce*; needs a scipy port or a loud refusal.
- **Job 5** learned/ICNN relaxations — opt-in `use_learned_relaxations=True`.
- **Job 6** symbolic structure cuts (`symbolic/constraint_cuts`, default-on but
  SymPy-gated) and the `symbolic/patterns` cluster.

**Good news:** on defaults the *first* jax import is **always
`nlp_evaluator.py:22`** (a top-level `import jax`), across all 8 nonlinear cases
measured — including pure-polynomial `nvs12`. Job 1 is the sole trigger. Removing
that one import is what unblocks the acceptance test for the default path.

### SCOPE DECISION (owner, 2026-08-03): the target is the EQUATION-ORIENTED path

discopt already has an equation-based modeling language, and `from_nl` is
equation-oriented. **Global optimization is built on algebraic structure** — you
cannot build a rigorous McCormick envelope, interval enclosure, or convexity
certificate for an opaque callable. The codebase already says this: `udf`
(`core.py:1479`) is a pass-through that requires the body be built from `dm.*`
primitives, *"Unlike an opaque numeric callback... the body must be symbolic — that
is what lets discopt build rigorous relaxations."*

**Therefore the goal is: zero `import jax` on the equation-oriented global solve
path** — models from `from_nl` or the modeling API (including `udf`). That is
achievable and is what this plan targets.

`dm.custom` / CustomCall is explicitly OUT OF SCOPE and **keeps JAX**:
- `core.py:1520-1522` — `custom` is *"restricted to the local NLP path"*; its
  contract requires the body be JAX-differentiable.
- `solver.py:6527-6591` — #713 later let a CustomCall that traces soundly through
  MCBox reach the global path via the *reduced-space* engine. Relaxing an opaque
  callable requires AD **through** it, so this path inherently needs JAX.
- Measured: this path is off by default (`DISCOPT_RELAX_SPACE="lifted"`;
  `_force_reduced_space` is set only for CustomCall models), and `mcbox` /
  `mccormick_subgradient` never loaded on 12 default solves.

**Consequence:** blocker (1) below is dissolved — it was never a blocker for
equation-oriented global optimization. CustomCall becomes a *documented,
JAX-requiring opt-in* (install `discopt[diff]`), not a violation. Job 3's MCBox
cluster and Job 5 (learned relaxations) fall out of scope with it.

The acceptance test scopes accordingly: assert `"jax" not in sys.modules` after
solving equation-oriented models across all 10 `ProblemClass` values; CustomCall
models are excluded by construction and get their own test asserting the capability
check fires loudly.

### Hard blockers — no in-tree replacement (decide before Stage 2)

1. ~~**`dm.custom` / CustomCall**~~ — **resolved by the scope decision above.**
   Out of scope; keeps JAX as a documented opt-in.
2. **`erf`** — no pounce TapeOp, no `.nl` opcode. Needs a new TapeOp
   (erf′ = 2/√π·e^{−x²}) or a loud refusal.
3. **Gauss-Newton residual Hessian** (`nlp_evaluator.py:655-670`, opt-in) — pounce
   has no residual concept; buildable as a second tape, nothing exists today.
4. **Job 4's QP rescue** — exists *because* pounce failed, so pounce cannot replace
   it.

### Stage 2 requires POUNCE-REPO work first (new S0)

The Python-API-model path **does not exist today**. pounce-py exposes only
file-path `read_nl`; `parse_nl_text` (`nl_reader.rs:322`) and `Tape::build`
(`nl_tape.rs:180-193`) are public Rust with **no pyo3 binding**, and
`hessian_directional` (HVP, `nl_tape.rs:696`) is unexposed.
And the `.nl` bridge is *not* the workaround: `export/nl.py:1211` raises on unknown
functions, and refuses `erf`/`sign`/`min`/`max`/`atan2` — several of which pounce's
tape *does* support. **The clean route is a direct tape-builder binding, not richer
.nl export.** This confirms the plan's flagged risk 2.1(b) as real, not hypothetical.

---

## Place 1 — Job 2: separation tangents (do this first)

**Why first:** the replacement is already written and coverage is ~100%. **NOT
bound-neutral** — see correction 1 above; it takes the §5 panel like Place 2.

`_jax/uniform_relax.py:827` `_compiled_analytic` already computes
`(value_fn, grad_fn)` with no JAX, via forward-mode interval AD over discopt's own
factorable IR (`_jax/convexity/interval_ad.py`, zero jax imports), pinning variables
to point intervals. Gated by `DISCOPT_ANALYTIC_SEPGRAD`, currently **unset ⇒ OFF**
(`uniform_relax.py:805`). It probes at the box midpoint and returns `None` →
JAX fallback on any uncovered atom.

**Measured so far (small sample):** on the 2 of 8 corpus instances that exercise the
composite-lift path, coverage was **100%** (tspn08 63/63, ex1224 3/3, zero
fallbacks) and bounds were **bit-identical** flag-on vs flag-off.

### Step 1.1 — Widen the coverage measurement (entry experiment, CLAUDE.md §4)

Before changing any default, establish the fallback rate across a real corpus.

- Instrument `_Builder._compiled_analytic` to count `ok` vs `None`, and record the
  `canonical_expr.atomize` kind of every atom that returns `None`.
- Run with `DISCOPT_ANALYTIC_SEPGRAD=1` over the full 66-instance
  `python/tests/data/minlplib_nl/` corpus, plus a draw of ≥100 instances from
  `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/` selected for composite
  multivariate structure (many corpus instances never exercise this path at all —
  6 of my 8 didn't, so an unfiltered sample measures nothing).
- **Kill criterion:** if the fallback rate is >0, the uncovered atom kinds are the
  work item; enumerate them and implement their interval-AD rules in
  `convexity/interval_ad.py` before proceeding. Do not graduate a flag that still
  needs a JAX fallback.

**Probe discipline (CLAUDE.md §6):** the probe must print the number of
`_compiled_analytic` invocations and exit non-zero if it is zero — a run where the
lift path never fires reports "0 fallbacks" and reads as a pass.

### Step 1.2 — Graduate the flag to default-ON

Flip the default at `uniform_relax.py:805` to ON, keeping `DISCOPT_ANALYTIC_SEPGRAD=0`
as the opt-out and the JAX path intact (CLAUDE.md §5 graduation rule).

**Before/after check — this is the effect verification:**

| | measure | before (OFF) | after (ON) — required |
|---|---|---|---|
| correctness | `bound`, `objective`, `node_count` on the cert panel | baseline | **bitwise identical** |
| correctness | `incorrect_count` | 0 | **0** |
| effect | `_compiled_analytic` fallbacks to JAX | n/a | **0** |
| effect | cProfile: `eval_jaxpr` + `_run_python_pjit` seconds on tspn08 | >0 | **≈0** |
| effect | `_TracedEvalFn` instantiations | >0 | **0** |
| perf | wall on tspn08 / ex1224, interleaved A/B, with sd | baseline | report, no regression |

Bitwise identity is the right bar here and is achievable: the analytic path is
deterministic point-interval AD, and this is exactly the property `_TracedEvalFn`
contorts to preserve. **If bounds are not bitwise identical, stop** — that means the
analytic gradient differs from the JAX gradient and the change is bound-changing,
not the flag graduation this stage assumes.

**Do not use node_count on time-limited runs as a neutrality signal.** Measured:
`clay0303hfsg` moved 159→191 nodes between two runs where the flag was a *no-op*
(0 analytic calls both sides) — pure timing nondeterminism. Run the neutrality
panel to optimality or to a node limit, not a time limit.

### Step 1.3 — Delete the JAX path in `_compiled`

Once 1.2 holds, remove the `import jax` / `jax.grad` / `_TracedEvalFn` branch at
`uniform_relax.py:813-820` and the `_TracedEvalFn` class (`:423-475`).

**After check:** `uniform_relax.py` contains zero `import jax`; repeat the 1.2
correctness panel; confirm bounds still bitwise identical.

### Tests after Place 1

```bash
pytest -m smoke
pytest -m slow python/tests/test_adversarial_recent_fixes.py
pytest python/tests/ -k "relax or mccormick or separat or convex or alphabb"
pytest python/tests/test_convex_claimer.py   # the composite-lift bound guard
```

---

## Place 2 — Job 1: NLP subsolve derivatives

Replace `NLPEvaluator`'s derivative supply with pounce's Rust AD tape.

`crates/pounce-nl/src/nl_tape.rs` (4,198 LOC) is a full operator tape with
value/tangent/adjoint: `eval` `:252`, `gradient_seed` `:261`, `hessian_directional`
`:696`, `hessian_accumulate` `:1130`, `hessian_sparsity` `:1443`. Exposed as
`pounce.read_nl(path)` → `NlProblem` with `.objective/.gradient/.constraints/
.jacobian/.hessian/.jacobian_structure/.hessian_structure/.nnz_jac/.nnz_hess`.

**Verified:** on `heatexch_gen3.nl` — reads in 8 ms, JAX never imported, and values
agree with the JAX path **exactly** (objective reldiff 0.00e+00, gradient
maxabsdiff 0.0).

**This stage is bound-CHANGING** and must go through the CLAUDE.md §5 differential
panel. Exact agreement on one instance does not generalize: `jacfwd∘jacfwd` and a
reverse-mode tape accumulate in different orders, so expect last-digit differences
elsewhere, and the B&B is path-dependent.

### Step 2.1 — Close the two known gaps (entry experiment)

Both must be answered before any implementation.

**(a) Operator coverage.** discopt's DAG supports ~25 unary functions plus
`atan2`, `signpower`, `centropy`, `prod` (`_jax/dag_compiler.py:243-288`). Enumerate
`nl_tape.rs`'s opcode set and diff. Any discopt operator the tape lacks is either a
tape addition (pounce repo) or a blocker.

**(b) Python-API models.** Models built through the modeling API never touch `.nl`,
but `read_nl` needs a file. Two candidate routes — decide by measurement, not
preference:
  - round-trip via `Model.to_nl` (`modeling/core.py:5125`, `export/nl.py`), or
  - a direct tape-builder binding that constructs the tape from the DAG in memory.

Measure the round-trip cost and, critically, whether `to_nl` is **lossless** for
every operator (this repo is currently on a branch fixing the .nl writer's inverse
hyperbolics — evidence that it has had gaps).

### Step 2.2 — Build a tape-backed evaluator behind a flag

New backend implementing the existing evaluator protocol —
`evaluate_objective / evaluate_gradient / evaluate_constraints / evaluate_jacobian /
evaluate_lagrangian_hessian / *_values / *_structure`. Keeping the protocol means
`solvers/nlp_pounce.py:19`, `solvers/oa.py:1562-1564`, and `solvers/amp.py:1026`
need no changes.

Gate behind `DISCOPT_NLP_EVAL=tape|jax`, default `jax`. Route sparse paths to
`hessian_sparsity` / `jacobian_structure` instead of
`_jax/sparse_jacobian.py` / `_jax/sparse_hessian.py`.

**Before/after check — derivative agreement, before any solve:**
- Sample ≥50 points per instance across ≥30 corpus instances spanning operator
  classes.
- Compare `f`, `∇f`, `g`, `J`, `∇²L` between JAX and tape at each point.
- **Required:** max relative difference ≤ 1e-10 on ∇f/J, ≤ 1e-8 on ∇²L. Report the
  distribution, not just the max.
- The probe must print its comparison count and fail if zero.
- This isolates *derivative correctness* from *solver path-dependence* — do it
  before interpreting any solve-level difference.

### Step 2.3 — Differential panel (the graduation gate)

Flag ON vs OFF over the in-repo corpus, requiring **both** bars from CLAUDE.md §5:

1. **Cert-clean:** `incorrect_count = 0`; no bound above its reference optimum
   (oracle: `~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu`, 1,585
   entries); no `gap_certified=True` instance regressing to uncertified; objective
   drift within tolerance; incumbents independently feasibility-verified.
2. **Net-positive:** measurably helpful broadly on node count / wall / bound — not
   merely sound. A cert-clean but neutral-or-harmful backend stays OFF, with the
   measurement recorded (the `DISCOPT_CUT_INHERIT` lesson).

**Expected effect, to be confirmed:** the 4.1 s first-call Hessian trace on
heatexch_gen3 disappears entirely — a tape has no trace step. Few-node instances
should gain most, since they currently pay the full trace with nothing to amortize
it against. State the predicted direction *before* running, so the run can falsify it.

### Step 2.4 — Graduate and delete

Default to `tape`, keep the `=jax` opt-out for one release, then remove
`_jax/nlp_evaluator.py`, `_jax/sparse_jacobian.py`, `_jax/sparse_hessian.py`.

### Tests after Place 2

```bash
pytest -m smoke
pytest -m slow python/tests/test_adversarial_recent_fixes.py
pytest python/tests/ -k "nlp or evaluator or oa or amp or hessian or jacobian"
cargo test -p discopt-core     # if Rust touched
```

---

## Stage 3 — Rename `_jax/` to match what it actually contains

**Why:** the directory name is a historical accident that actively caused errors in
this investigation. Created 2026-02-07 (`d326f654`, "T4 JAX DAG compiler") holding
only `__init__.py` + `dag_compiler.py` — at which point the name was accurate. It
then became the default home for solver numerics. Files added per month, and how
many of today's versions import jax:

| month | added | import jax today |
|---|---|---|
| 2026-02 | 39 | 18 (46%) |
| 2026-04 | 17 | 2 (12%) |
| 2026-05 | 22 | 7 |
| 2026-06 | 39 | 15 |
| 2026-07 | 25 | 3 (12%) |
| **total** | **143** | **45 (31%)** |

Today 66% of its LOC never imports jax. Its largest residents are
`milp_relaxation.py` (3,005), `nonlinear_bound_tightening.py` (2,987), `obbt.py`
(2,601), `primal_heuristics.py` (2,153), `gdp_reformulate.py` (1,824) and
`convexity/` (~10,600) — the guts of a global MINLP solver, not a JAX layer.

`_jax/__init__.py` already documents the strain: JAX is imported lazily via PEP 562
`__getattr__` so that *"importing a JAX-free submodule such as
`discopt._jax.deadline` no longer drags in `dag_compiler` and the heavy JAX/XLA
initialization"* — a workaround for the misnaming.

**Concrete cost incurred:** this naming is why `milp_relaxation.py:2037` delegating
to `uniform_relax.py` read as an unremarkable intra-package call, which produced a
wrong "the hot path is JAX-free" conclusion; and why AMP/OA look JAX-coupled when
only 1 of their 10 `_jax` dependencies touches JAX.

**Change (after Places 1 and 2 land):**
- `_jax/` → `_relax/` (or `_numeric/`) — the relaxation/presolve/cut/convexity core.
- Genuinely-JAX post-solve features → `_diff/`: `differentiable*.py`,
  `pounce_layer.py`, `parametric.py`, `icnn*.py`.
- Drop the PEP 562 lazy-`__getattr__` workaround in `__init__.py` — unnecessary once
  JAX is off the solve path.

**Size: 1,976 references** — `python/discopt` 549, `python/tests` 1,083,
`discopt_benchmarks` 199, docs/other 195.

**Hazard — dynamic imports evade a plain grep.** Measured:
- `dag_compiler.py:225,227` use `__import__("jax")` for `erf`/`sigmoid`. The string
  `__import__("jax")` does **not** contain the substring `import jax`, so a naive
  grep misses it. The Stage-1/2 acceptance test must catch these — assert on
  `sys.modules` after a solve, never on a source grep.
- `_jax/symbolic/__init__.py:48` resolves submodules via
  `importlib.import_module(f"{__name__}.{module}")` off a `_LAZY` dict — a
  string-built module path that a mechanical rename will not update. Audit it
  explicitly.

**Verification:** this is a pure rename with **no behavior change**, so the bar is
strict — `bound`, `objective`, `node_count` **bitwise identical** on the cert panel,
full test suite green, and `python -c "import discopt"` plus a solve of each
`ProblemClass`. Any drift means the rename was not mechanical.

**Sequencing:** last. It touches nearly 2,000 sites and would produce unreviewable
diffs and constant conflicts if interleaved with Places 1–2.

---

## Acceptance criterion (make it executable)

Add a test that enforces the goal rather than documenting it:

```python
def test_solve_path_is_jax_free():
    """Solving a nonlinear model must never import jax."""
    import sys, subprocess
    # subprocess so sys.modules is clean; assert jax absent AFTER a solve
    # that exercises nonlinear constraints, NLP subsolves, and the lift path.
```

Run it over a set spanning all 10 `ProblemClass` values. It fails today for every
class with a nonlinear constraint, and passing it *is* the definition of done.

---

## Branch strategy (owner decision, 2026-08-03)

**One long-lived branch, one DRAFT PR, updated per stage.** The owner wants CI
coverage but nothing merged to `main` until the whole thing is verified.

- Branch: `refactor/remove-jax-from-solve-path`, off `main` (`badf9f4c`).
- Draft PR against `main`; each stage lands as its own commit(s) and is pushed, so
  CI runs per stage. Not marked ready until Stages 0–2 are green.
- **`main` never moves** until the final merge.
- **Stage 3 (the rename) is EXCLUDED from this PR** — ~1,976 mechanical sites would
  bury the review of Stages 0–2. It becomes a follow-up PR after this one merges.

**Recovery anchors (local tags, never pushed):**
- `pre-jax-removal` → `badf9f4c` (state of `main` today)
- `pre-jax-removal-nlwriter` → `402ba8f5` (unpushed nl-writer branch work)

Revert to today: `git checkout main`. Abandon: delete the branch. Both tags make
the pre-change state recoverable regardless.

**Deviation from CLAUDE.md noted:** the house rule is one task per PR. This
consolidates three stages into one PR at the owner's request, for end-to-end
verification before anything merges. Stage boundaries are preserved as commits.

### Per-stage gate (no merge until all pass)

Because nothing merges until the end, each stage must be green *locally* as well as
in CI before the next begins:

```bash
pytest python/tests/ -q                                        # FULL suite, not smoke
pytest -m slow python/tests/test_adversarial_recent_fixes.py
cargo test -p discopt-core                                     # if Rust touched
ruff check python/ && ruff format --check python/
mypy python/discopt/
```
plus that stage's own before/after check and, where the stage claims
bound-neutrality, the cert-panel bitwise comparison.

### Original per-stage file sets (now commits on the single branch)

The file sets are **disjoint**, so the stages do not conflict:

| branch | files touched |
|---|---|
| `fix/jax-time-attribution` | `solver.py` (accumulators `:6523-6524`, roll-up `:1241-1271`), `discopt_benchmarks/perf/measure.py` |
| `refactor/analytic-separation-gradients` | `_jax/uniform_relax.py`, `_jax/convexity/interval_ad.py` |
| `refactor/tape-nlp-evaluator` | new evaluator backend, `_jax/nlp_evaluator.py`, `_jax/sparse_{jacobian,hessian}.py` |
| `refactor/rename-jax-package` | ~1,976 sites across the repo (Stage 3; strictly last) |

**But they are not independent for verification, in three ways:**

1. **Stage 0 must merge first.** Both later branches measure their effect with
   `jax_time`, and it is currently wrong by ~50%. Branching 1 and 2 off a `main`
   that still has the broken counter means their before/after numbers are
   unusable. Merge `fix/jax-time-attribution` to `main`, then branch 1 and 2 off it.

2. **The acceptance test cannot land on either branch alone.**
   `test_solve_path_is_jax_free` only passes once *both* places are done. Land it on
   the Place-1 branch marked `xfail(strict=True)` with a comment naming Place 2, and
   flip it to a passing assert on the Place-2 branch. `strict=True` matters — it
   fails loudly if it starts passing early, which would mean the reachability
   analysis was wrong.

3. **Merge them sequentially, not concurrently.** Place 1 is bound-neutral
   (bitwise) and Place 2 is bound-changing. If both land together and the panel
   shows drift, you cannot attribute it. Merge Place 1, re-baseline the cert panel
   on `main`, then run Place 2's differential against that new baseline.

Developing 1 and 2 in parallel is fine — it is the *merge* order that must be
serial. If Place 2's gap analysis (Step 2.1) turns out to need a pounce-repo
change, that becomes a fourth branch in the pounce repo, blocking only Place 2.

## Sequencing and risk

| stage | scope | bound impact | verification bar |
|---|---|---|---|
| 0 — attribution model | instrumentation, +2 buckets | none | node/objective unchanged; `unattributed` < 10% |
| 1 — Job 2 | probe hardening + flag | **bound-changing** (Δgrad 2.8e-09) | §5 differential panel |
| 2 — Job 1 | pounce bindings + new backend | **bound-changing** | §5 differential panel, both bars |
| 3 — rename | ~1,976 sites, no logic | none | **bitwise** identical; full suite green |

Do 0 → 1 → 2 → 3 strictly in order. **Stage 2 is now gated on pounce-repo work**
(pyo3 bindings for `parse_nl_text` / `Tape::build` / `hessian_directional`, plus
missing TapeOps) — that is a separate repo and a separate PR, and it blocks Stage 2
entirely. Stage 1 no longer enjoys a cheaper verification bar than Stage 2; its
advantage is now only that its replacement already exists in-tree. Stage 1 is low-risk and self-verifying; doing it
first also removes `_TracedEvalFn` and shrinks the surface Stage 2 must reason about.

**Biggest risk:** Step 2.1(b) — if `Model.to_nl` is lossy for any operator, the
Python-API path needs a direct tape-builder, which is a pounce-repo change and a
materially larger job. Resolve 2.1 before committing to Stage 2.

**Standing rule (CLAUDE.md §1):** if a stage can only pass by weakening a
validation, guard, or tolerance, the stage loses. Stop and surface it.

---

# § Rescoped stages (2026-08-04) — supersedes the staging above

## Stage 0 — attribution model — **DONE**

Commits `c3a3d648`, `7fc69f7f` on `refactor/remove-jax-from-solve-path` (PR #922).
Attribution defects 11 → 1; POUNCE given a bucket (tspn08 12.34 s, previously
absorbed); buckets partition the wall; bound-neutral on 12 instances.

## Stage 1 — DAG → `NlExpr` translator (the new shared prerequisite)

The single new component. Mirrors `_jax/dag_compiler.py`'s walk, emitting
`pounce.NlExpr` instead of a `jnp` callable.

**Files:** new `python/discopt/_nl_expr_compiler.py` (name TBD; it is not
JAX-related and must not live under `_jax/`).

**Operator table** — 20 native, 10 by decomposition, all verified reachable:

| decomposition | |
|---|---|
| `abs(a)` | `max(a, -a)` |
| `sign(a)` | `select(compare(a, '>', 0), 1, -1)` |
| `log1p(a)` | `log(1 + a)` |
| `log2(a)` | `log(a) / log(2)` |
| `sigmoid(a)` | `1 / (1 + exp(-a))` |
| `softplus(a)` | `log(1 + exp(a))` |
| `entropy(a)` | `-a * log(a)` |
| `centropy(a, b)` | `a * log(a / b)` |
| `prod(...)` | `*` chain |
| `signpower(a, p)` | `sign(a) * abs(a)**p` |

**Verification:** differential against `dag_compiler`'s JAX output — value and
gradient at ≥50 sampled points across ≥30 corpus instances spanning the operator
classes, with a printed comparison count. Required: exact or ≤1e-12 relative.
Reuse the harness in `scratchpad/stage1_entry.py`.

**Bound impact:** none — nothing is wired in yet.

## Stage 2 — route separation tangents through the translator (was Stage 1)

Replace the `jax.grad` arm of `_Builder._compiled` (`_jax/uniform_relax.py:813-820`)
with the translator, delete `_TracedEvalFn` (`:423-475`), and swap the `jnp`
marshaling at `_jax/mccormick_lp.py:2022` to numpy.

**Bound impact: CHANGING.** §5 differential panel. Reuse
`scratchpad/stage1_panel.py`, which already implements both bars.
**Kill criterion:** any cert regression, or loss of the composite-lift tightening
guarded by `test_convex_claimer` (`mccormick_lp.py:1181-1187`, −204 → −350).

## Stage 3 — route NLP derivatives through the translator (was Stage 2)

**Status: evaluator built, default OFF** (`ce212d67`). `_tape_nlp_evaluator.py`
implements the protocol from the tape; `DISCOPT_NLP_EVAL=tape` opts in.
Remaining for this stage: wire `build_evaluator` into the `cached_evaluator`
call sites, then the §5 panel.

### Measured record

**Entry experiment (§4), run before implementation.** Tape vs the JAX evaluator,
66 in-repo corpus instances, 5 points each, 1573 comparisons:

| quantity | comparisons | max rel diff | Step 2.2 bar |
|---|---|---|---|
| `f` | 330 | 5.48e-16 | — |
| `∇f` | 330 | 3.77e-16 | ≤1e-10 ✅ |
| `g` | 297 | 4.55e-13 | — |
| `J` | 298 | 3.06e-15 | ≤1e-10 ✅ |
| `∇²L` | 318 | 7.82e-13 | ≤1e-8 ✅ |

Zero coverage gaps, zero unexpected errors.

> #### ⚑ FALSIFIED — "verified on 40 corpus instances" was not evidence
>
> Stage 1 shipped claiming all 30 operators on the strength of a corpus
> differential. **A `.nl` corpus exercises six operators.** Census over 316
> MINLPLib instances (66 in-repo + 250 from the benchmark snapshot):
> `log, sqrt, exp, abs, sin, cos` — nothing else, ever. `.nl` has no opcode for
> `sigmoid`/`softplus`/`entropy`/`centropy`/`signpower`; they reach the DAG only
> through the modeling API and `factorable_reform`. **Nine of the ten rewrites
> had zero coverage**, and three were wrong (`08e1e0a1`):
>
> * `entropy` returned `-x·log(x)`; discopt's semantics is `x·log(x)`. A clean
>   factor of −1 — raises nothing, passes every structural check.
> * `_sign` called `compare(a, ">", b)`; the operator comes **first**. Raises
>   `TypeError`, which escapes `try_compile`'s fallback and crashes the caller.
> * `prod` was lowered as a variadic `*` chain. It is `jnp.prod(arr)` — one
>   **array** argument. Agreed with nothing (reldiff 0.90/1.70) while reporting
>   success. Now refused, with `norm1`/`norm2`/`norminf`.
>
> **Lesson: coverage claims must be counted in operators, not instances.** Corpus
> breadth is the wrong axis for an operator table; a 5-second per-operator
> differential found all three, and 316 instances over ~2.5 h found none.
> `test_every_rewrite_is_covered` now reads the compiler's source so the list
> cannot drift.

**Found in passing, filed as #923 (not caused by this work):** the *dense*
`evaluate_lagrangian_hessian` silently returns an **all-zero** matrix on
`emfl100_3_3` (n=2961). Finite differences (−2.00007), the JAX *sparse* path
(−2.0) and the tape (−2.0) all agree against it. Not a size threshold — a
synthetic sweep is correct to n=3000. `nlp_evaluator.py:268` compounds it by
using the dense Hessian as ground truth to validate the sparse values, which is
backwards here. Also observed: `dag_compiler` hits Python's recursion limit on a
plain `sum(xs)` objective at n≈300 (it skipped three `edgecross10-*` instances).

### Design notes that are load-bearing

* **The parameter hazard.** `evaluator_fingerprint` deliberately excludes
  `Parameter.value` because JAX reads it live; the tape bakes it in. A tape
  cached under that fingerprint serves stale derivatives with no error. The
  evaluator snapshots and rebuilds on change.
* **Two Hessian conventions.** `evaluate_lagrangian_hessian` is FULL dense (n,n);
  `hessian_structure`/`evaluate_hessian_values` are LOWER-TRIANGLE COO.
* **Consumed surface is wider than the protocol.** Call sites also read `_model`,
  `_obj_fn`, `_cons_fn`, `_source_constraints`, `_constraint_flat_sizes` and
  `_structural_linear_mask_cache`, so "same protocol, no call-site changes" is
  optimistic.

### Original plan text follows

Replace `_jax/nlp_evaluator.py` — the single trigger that imports JAX on *every*
nonlinear solve (`:22`, measured across 27 instances). Same protocol, so
`solvers/nlp_pounce.py:19`, `solvers/oa.py:1562-1564` and `solvers/amp.py:1026`
need no changes.

**Not blocked.** `NlProblem.hessian_vector_product` (multi-direction, column-major
`n × k`) covers the Hessian-free mode; `hessian` + `hessian_structure` cover the
dense/sparse Lagrangian Hessian; `jacobian_structure` replaces discopt's JAX
sparsity probe and CPR-coloring helpers outright.

**Bound impact: CHANGING.** §5 panel, both bars.

## Stage 4 — enforcement

Flip `test_solve_path_is_jax_free` from `xfail(strict=True)` to a passing assert.
Assert on `sys.modules` after a real solve, **never** a source grep — measured:
`dag_compiler.py:225` uses `__import__("jax")`, which no `import jax` grep matches.

## Out of scope (unchanged)

`dm.custom`/`CustomCall` keeps JAX by design (`core.py:1522` — its contract
requires a JAX-differentiable body; relaxing an opaque callable needs AD *through*
it). Post-solve differentiable-optimization features keep JAX.

## Rescoped sequencing

| stage | bound impact | gate |
|---|---|---|
| 0 attribution | none | done |
| 1 translator | none (not wired) | derivative differential ≤1e-12 |
| 2 separation | **changing** | §5 panel |
| 3 NLP derivatives | **changing** | §5 panel |
| 4 enforcement | none | `sys.modules` assert green |

Stage 1 unblocks both 2 and 3; 2 and 3 are then independent and can land in
either order.
