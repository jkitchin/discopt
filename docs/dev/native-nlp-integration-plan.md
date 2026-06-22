# Plan: route node NLP solves through POUNCE's native AD (drop the JAX callback bridge)

Status: proposed (no code changes yet). Target: discopt#281 convex NLP-BB node throughput
(and the #268 tier). Owner: TBD.

## 1. The seam — corrected

The pre-investigation worry was that we'd need to translate discopt's `ExprArena`
(`crates/discopt-core/src/expr.rs`) into POUNCE's `FbbtTape` / `ExpressionProvider`
(`pounce-nlp/src/expression_provider.rs`). **We do not.** Both sides already speak `.nl`
fully, in both directions:

| direction | discopt | POUNCE |
|---|---|---|
| read `.nl`  | `from_nl` / `parse_nl_file` (`modeling/core.py:3407`) | `pounce.read_nl(path)` → `NlProblem` (`pounce-py/src/nl_problem.rs:420`) |
| write `.nl` | `model.to_nl(path)` (`modeling/core.py:3059` → `export/nl.py`, Pyomo-NLv2-style writer) | — |

`pounce.read_nl` returns a native `NlProblem` carrying **POUNCE's own reverse-mode AD**
(`gradient`, `jacobian`/`jacobian_structure`, `hessian`/`hessian_structure`, `objective`,
`constraints`) plus `x0/x_l/x_u/g_l/g_u`. Confirmed present on the **installed** wheel
(`pounce-solver==0.5.0`, the current pin), not just repo HEAD.

Crucially `NlProblem.variant(x0=, x_l=, x_u=, g_l=, g_u=)` (`nl_problem.rs:321`) clones the
problem with **per-instance bound/start overrides while sharing the parsed DAG / AD tapes**
— this is exactly the B&B-node case ("nodes that only tighten variable bounds", per its
docstring). And `solve_nlp_batch(problems, x0s=, options=, share_structure=)` already accepts
a `Vec<NlProblem>` — the installed signature matches discopt's existing call site verbatim.

So the integration is **not** a translator. It is:

```
base = pounce.read_nl(nl_path)            # ONCE per model — POUNCE parses + builds AD tapes
node_prob = base.variant(x_l=node_lb, x_u=node_ub, x0=warm)   # per node — shares tapes, cheap
pounce.solve_nlp_batch([node_prob, ...], x0s=..., options=...) # native IPM + native AD, zero JAX
```

This deletes, from the per-node hot loop, the entire
`_BoundOverrideEvaluator` → `_IpoptCallbacks` → JAX `NLPEvaluator` chain
(`solver.py:6810-6825`).

## 2. The one real correctness risk: variable / constraint ordering

`node_lb`/`node_ub` and the warm start are indexed in the **JAX evaluator's flat variable
order** (`_jax/nlp_evaluator.py`, flattened model vars). The native `NlProblem` is indexed in
the **`.nl` column order**. For Stage 1/2 (solving an instance that *originated* from an `.nl`)
these are the same parse, but this must be **proven at runtime, never assumed**: a silent
permutation would corrupt every node bound and still "look like" a solve.

Guard (used in every stage): build a permutation `P` from `base.var_names` ↔ the evaluator's
flat variable names; assert it is the identity for the originated-from-`.nl` case, and apply
it (and its inverse on the returned `x`) for the emitted-`.nl` case. Same for constraint order
vs `g_l/g_u`. If names are unavailable, fall back to the Stage-0 numeric equivalence check
below as the gate.

## 3. Staged plan

### Stage 0 — offline equivalence harness (no solve-path change)
Prove the native AD agrees with JAX *before* wiring it into B&B.

- New test `python/tests/test_native_nlp_equivalence.py`. For a basket of `.nl` instances
  (smallinvDAXr2b050–055 for #281; a few #268 ex126x; one nonconvex):
  - `ev = NLPEvaluator(from_nl(path))`; `base = pounce.read_nl(path)`.
  - Assert `base.var_names`/order matches `ev` flat order (the §2 guard).
  - At ~20 random points `x` in `[x_l,x_u]`: compare
    `objective`, `gradient`, `constraints`, `jacobian` (densified via structure),
    `hessian` (Lagrangian, densified) between `base.*` and `ev.evaluate_*`.
  - **Pass:** max abs/rel diff ≤ 1e-9 (these are the *same* math, so agreement is tight,
    not just within solver tolerance). Sign/sense check: confirm `minimize` and objective
    sense line up (the evaluator negates `maximize`).
- **Why first:** catches ordering, sign, sparsity-pattern, and constant-term bugs with a
  cheap deterministic test instead of through a noisy end-to-end solve.

### Stage 1 — single-node native solve behind a flag
- New `_solve_node_nlp_pounce_native(evaluator, base_nl, x0, node_lb, node_ub, options)` in
  `solver.py`, selected when a `nlp_native=True` option is set (default off).
  - `node_prob = base_nl.variant(x_l=node_lb, x_u=node_ub, x0=warm)`; solve via the existing
    single-problem POUNCE entry; map result → `NLPResult` (status/x/objective/multipliers),
    reusing the status mapping already in `_solve_node_nlp_pounce`.
  - `base_nl` is built once and cached on the evaluator/model (e.g. `evaluator._pounce_base`),
    keyed by the originating `.nl` path.
- **Correctness verification:** a differential test that runs the *same* nodes through both
  paths (flag off vs on) on the Stage-0 basket and asserts, per node: same status class, and
  `|obj_native − obj_jax| ≤ 1e-6·(1+|obj_jax|)`, `‖x_native − x_jax‖∞ ≤ 1e-6`. Plus the
  global gate: full `pytest -m "correctness"` with `nlp_native=True`, **`incorrect_count == 0`**.
- **Perf measurement:** per-node wall time and the `r.jax_time / rust_time / python_time`
  split (the undistorted fields used in the earlier profiling) on smallinvDAXr2b050.
  Expectation: `jax_time → ~0`, per-node Python overhead (the 26µs/call dispatch × callbacks
  × iters) gone; per-node cost becomes ≈ Rust-only.

### Stage 2 — batch native (the actual #281 lever)
- At the batch construction site (`solver.py:6809-6825`), replace
  `pounce.Problem(problem_obj=_IpoptCallbacks(_BoundOverrideEvaluator(...)))` with
  `base_nl.variant(x_l=node_lb, x_u=node_ub, x0=warm)`. `solve_nlp_batch` already takes these
  natively; the `x0s=`, `options=`, `share_structure=True` call (`6840`) is unchanged.
  `share_structure=True` becomes *real* structure sharing (one parsed DAG, N bound variants)
  instead of N Python callback proxies.
- Keep the existing whole-batch `except` fallback to serial (`6846`), but point it at the
  native single-node path; keep the JAX path reachable when `nlp_native=False`.
- **Correctness verification:** same differential basket but at batch granularity (compare the
  reduced best-per-node result arrays `result_lbs/result_sols/trusted` between paths); full
  `-m "correctness and regression"`, **`incorrect_count == 0`**; specifically re-confirm the
  `trusted`/decertification logic (`6831-6837`) still fires identically for convex
  ITERATION_LIMIT nodes.
- **Perf measurement (the headline number):** deterministic node count at the 5 s budget on
  smallinvDAXr2b050–055 (today: 251 nodes @5 s, converges ≈7.4 s just past budget). Target:
  enough extra node throughput to converge inside 5 s. Report nodes/sec, time-to-optimal, and
  the #281 near-optimal-miss set before/after.

### Stage 3 — generalize to Python-API / reformulated models
Stages 1–2 require the solve to have *originated* from an `.nl`. For models built via the
Python API, or models discopt has reformulated (factorable lift, aux vars) so the in-memory
model ≠ the on-disk `.nl`, emit the **exact NLP-relaxation model** to a temp `.nl` via
`model.to_nl(...)` and `read_nl` that. No tape translator; we round-trip the representation
both sides already implement.
- **Correctness verification:** Stage-0 equivalence check, but the `base` comes from
  `to_nl`-then-`read_nl` of the *reformulated* model the evaluator was built from — this is the
  load-bearing check that the writer and the evaluator agree (ordering + aux vars + sense). Run
  it on a reformulated instance (a factorable/AMP case) and a pure Python-API model. Only enable
  the native path for a model once this check passes for it; otherwise fall back to JAX.

### Stage 4 — make native the default
- Flip `nlp_native` default on once Stages 1–3 are green across the full correctness +
  regression suite with `incorrect_count == 0`. Keep JAX as an explicit fallback
  (`nlp_native=False`) and as the automatic fallback when the §2 ordering guard or Stage-3
  equivalence check fails for a given model. JAX remains the engine for relaxation
  construction (McCormick/αBB) — this change touches only the **node NLP solve**.

## 3a. Measured performance (post-implementation)

Honest findings from direct timing (no cProfile):

* **Correctness:** native ON vs OFF gives identical optima across the `.nl`
  basket and synthetic convex MINLPs; 15 equivalence tests + 800+ correctness/
  regression tests pass with native default-on (`incorrect_count == 0`).
* **The node NLP solve is *not* the #281 bottleneck.** `python_time` in
  `SolveResult` is a *residual* (`wall − rust_time − jax_time`); the
  `jax_time` timer actually brackets the node-*solve* call site (the POUNCE
  solve), so it is mislabeled — POUNCE's solve counts as `jax_time`, not
  `python_time`. Measured, `jax_time` is ~1–15% of wall; the rest is B&B
  orchestration (tree marshalling, per-node FBBT tightening, batch assembly,
  branching) and, for pure (MI)QP, the QP-relaxation B&B driver
  (`_pounce_qp_relaxation_nodes`) — none of which native touches.
* **Where native *does* win: JAX JIT-compilation latency.** The JAX path
  compiles grad/Hessian on first use (≈0.1–0.7 s for transcendental objectives);
  native walks the AD tape with no compile step. So native wins on cold/shallow
  solves (1.3–2.5× wall measured on root-dominated convex MINLPs) and is roughly
  neutral on deep trees / warm repeated solves where the JAX compile amortizes
  over many nodes (and the evaluator cache reuses the XLA kernel — a warm second
  JAX solve drops from 1.93 s to 0.13 s).
* **Per-node mechanics are cheap and already multi-RHS.** Parsing + AD-tape
  build is paid once (cached on the model); each node is a `variant()` (~2–13 µs,
  shares tapes) and `solve_nlp_batch(share_structure=True)` reuses symbolic KKT
  structure across siblings. POUNCE is in-process (PyO3) — there is no subprocess
  startup to amortize.

Net: the native path is correct, removes JAX from the node-solve path (the
original motivation), and eliminates JIT-compile latency for one-shot/shallow
solves; it is not a throughput fix for deep-tree #281 cases, whose cost is the
Python B&B orchestration residual.

## 4. What this is *not*
- Not removing JAX from discopt — relaxations, McCormick, and AD for the global machinery stay.
- Not a new IR or a tape translator — explicitly avoided by using `.nl` as the shared format.
- Not a change to bound tightening, branching, or the LP/B&B core.

## 5. Open items to confirm during Stage 0
- `pounce-solver` pin: installed 0.5.0 has `read_nl`/`variant`/`solve_nlp_batch(x0s=)`. Repo
  HEAD has a newer `solve_nlp_batch(..., warms=)` arg — decide whether to bump the pin or stay
  on 0.5.0's `x0s=` interface (0.5.0 is sufficient for this plan).
- Multiplier/bound-multiplier mapping parity (for KKT polish + dual-based tightening) between
  native result and the current `NLPResult` fields.
- Hessian: confirm native Lagrangian-Hessian sign/`obj_factor` convention matches
  `_IpoptCallbacks.hessian(x, lagrange, obj_factor)` (it should — both target Ipopt's TNLP).
