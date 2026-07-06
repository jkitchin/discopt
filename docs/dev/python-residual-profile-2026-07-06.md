# Python-residual bottleneck profile — 2026-07-06 (PYPROF-1)

**Date:** 2026-07-06
**Status:** measured (measurement-only; no production `python/discopt` or Rust
code changed). Deliverable is this findings doc + two reusable scripts.
**Scope:** the "correct-but-slower-than-BARON-by-seconds" residual-overhead
panel (**m3, nvs13, nvs08, ex1224, fac2, nvs06**) where BARON is sub-second,
plus a trivial fixed-tax probe (**alan**). The task: confirm or refute that the
residual gap is **per-node / per-solve Python orchestration overhead** (vs
genuine Rust/JAX/POUNCE compute), rank the removable-Python sinks, and add a
**coverage map** of which solve path each instance takes and whether the recent
branch-and-reduce/PSD/node-LP capabilities even fire.
**Builds on:** `docs/dev/bottleneck-profile-2026-07-05.md` (§1.7 fixed-tax,
§6 "hda Python churn 32.2M size/21M abs", §2 per-instance tables) and
`docs/dev/bottleneck-profile-2026-07-02.md` (B3 certificate re-derivation per
LP, B6 structure re-derivation per call, item 6 "scipy csr→csc→hstack per LP
call ≈12%"). Note fac2 here is **6.1 s** (not the 07-05 report's 23.5 s) —
F1 LNS-enumeration budget has since merged to `main`, so the LNS binge is gone
and what remains is the residual profiled here.

> **Method.** Branch `pyprof1-python-residual-profile` at `origin/main`
> (7199ab8b). **Apple M4 Pro (14-core, arm64)**, Python 3.12.11, JAX 0.10.2
> (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`), pounce **0.7.0** (release wheel,
> `_pounce.abi3.so` = **4.5 MB** — verified NOT a debug build, which would
> dominate and mislead the profile), numpy 2.5.1, scipy 1.18.0,
> `maturin develop --release` (discopt `_rust` = 2.7 MB). Instances from
> `python/tests/data/minlplib_nl/`, `from_nl(path).solve(time_limit=60,
> gap_tolerance=1e-4)`, run one at a time.
> **Machine-noise caveat (honesty):** this box carried a steady background
> load average ~3.8–4.0 (out of 14 cores) throughout; runs were serialized
> (one solve on-CPU at a time) so wall numbers are stable to a few percent, but
> treat sub-0.1 s differences as noise. Absolute walls may read ~5–10 % high vs
> a fully idle machine; the *shares* (the load-bearing result) are noise-robust
> and agree across three measurement modes.
> Three modes via `discopt_benchmarks/scripts/python_residual_profile.py` (new):
> **clean** (uninstrumented wall + result fields), **cprofile** (per-function
> tottime + a call-count census; C-extension entries included so POUNCE/Rust
> builtins are timed directly — used for *shares and call counts*, never
> totals), and **sampled** (in-process 200 Hz stack sampler — py-spy needs root
> on macOS and was unusable; native Rust/C/JAX time is attributed to the
> enclosing Python frame, and the **leaf-frame** classification below is the
> honest wall attribution). Coverage map via
> `discopt_benchmarks/scripts/coverage_map.py` (new; monkeypatches the dispatch
> + capability entry points with counters, no production edit).

---

## 0. Executive summary — the verdict

**The Python-overhead hypothesis is CONFIRMED, but with a sharp
re-framing: the removable overhead is a per-solve/root-phase cost, not a
per-B&B-node one, and the single biggest sink is the *Python NLP-callback
bridge* that POUNCE re-enters on every IPM iteration.**

1. **Removable-Python fraction is 19–53 % of wall** on the panel (leaf-honest
   sampler attribution) — material, not <20 %, so the **kill criterion did NOT
   fire**: the panel gap is Python-side, not purely genuine compute. Genuine
   native compute (POUNCE IPM + Rust) is the *majority* (52–70 %) on 5 of 6
   instances, but the removable Python slice is large enough to be the lever.
2. **The single biggest removable sink is `P_nlp_bridge`** — the Python glue in
   `_jax/nlp_evaluator.py` + `solvers/nlp_ipopt.py` (the `_IpoptCallbacks`
   adapter) + `_jax/sparse_jacobian.py`/`sparse_hessian.py` that POUNCE calls
   **once per IPM iteration** to evaluate objective/gradient/constraints/
   Jacobian/Hessian. It is **24–31 % of wall on m3/fac2** and 14–15 % on
   nvs08/nvs06/ex1224. Call-count evidence: **fac2 issues 22,676
   `evaluate_constraints` + 21,131 `evaluate_objective` + 21,414 `array._value`
   calls** in a 6 s solve; each does a JAX-kernel dispatch plus np.asarray/
   `_value`/`__float__` marshaling around it.
3. **The #2 sink is scipy sparse format churn** (`_csc.py:tocsr`,
   `_coo_to_compressed`, `tocsc`) — the relaxation/LP structure rebuilt and
   re-converted per call. **40 % of wall on nvs13**, 16 % on ex1224. This is
   exactly the 07-02 profile's item-6 ("scipy csr→csc→hstack per LP call,
   ≈12 %"), still live and *larger* on the small instances.
4. **Per-node vs fixed: it is ROOT-PHASE dominated, driven by NLP-solve
   volume — not the B&B tree.** `root_time` is **60–97 % of wall** on every
   instance; nvs06 is 97 % root / 5 nodes / 911 POUNCE solves (root
   multistart). The removable overhead scales with **NLP-solve count** (root
   multistart + diving + separation), not `node_count`. First-solve fixed tax
   (lazy-import + JIT warmup) is a flat **~0.5 s**, dominant only on the
   sub-2 s instances (nvs13: 0.49 s = 33 %).
5. **Coverage map — the recent capability push is INERT on this panel by
   default.** All six instances take the **spatial `_solve_nlp_bb` McCormick
   LP-relaxer path** with POUNCE node NLPs. With default config,
   `run_root_fixpoint` (R2 root reduce) and `reduce_node` (R2 node reduce)
   fire **0 times on every instance** (they are gated behind
   `DISCOPT_ROOT_FIXPOINT`/`DISCOPT_NODE_REDUCE`, default-OFF). This is a
   root-cause for the "42→42 proved-optimal, no net change" BARON re-measure:
   the merged capabilities never execute on the standard panel. (When forced
   ON, they help ex1224/nvs08/nvs06 — nodes collapse to 5 — but *hurt* nvs13,
   19→49 nodes; §5.)

---

## 1. Per-instance wall bucket (the removable-Python-fraction table)

Leaf-frame sampler attribution (200 Hz; wall-proportional, does not over-weight
many-small-call Python functions the way cProfile does). "GEN" = genuine native
compute (POUNCE IPM native frames + Rust `_rust`/`solve_lp` + numpy-ufunc-math
C leaves you cannot remove in Python). "REMOV-PY" = removable Python
orchestration (the NLP-callback bridge, scipy sparse rebuild/convert,
Python FBBT/OBBT walk, McCormick/relaxation build, term classification, core
DAG props, np marshaling). "FIX" = import/JIT warmup fixed tax.

| instance | wall | nodes | root% | **GEN%** | **REMOV-PY%** | FIX% | dominant removable sinks |
|---|---:|---:|---:|---:|---:|---:|---|
| m3     | 3.31 s | 61 | 60 % | 54 % | **32 %** | 9 %  | nlp_bridge 31 % |
| fac2   | 6.07 s | 69 | 75 % | 67 % | **25 %** | 4 %  | nlp_bridge 24 % |
| nvs06  | 3.57 s | 5  | 97 % | 52 % | **36 %** | 8 %  | milp_simplex_py 20 %, nlp_bridge 15 % |
| ex1224 | 2.29 s | 53 | 76 % | 53 % | **32 %** | 13 % | scipy_sparse 16 %, nlp_bridge 14 % |
| nvs08  | 3.03 s | 57 | 91 % | 70 % | **19 %** | 8 %  | nlp_bridge 15 % |
| nvs13  | 1.55 s | 19 | 60 % | 14 % | **53 %** | 22 % | scipy_sparse 40 %, nlp_bridge 5 %, milp_simplex_py 4 % |

Cross-check honesty: cProfile *tottime* self-time under-reads the bridge
(the JAX kernel executes natively *under* the Python `evaluate_*` frame, so
self-time is tiny — 0.2–0.4 s) while a whole-stack "bridge present anywhere"
attribution over-reads it (99 % — it swallows the native JAX exec too). The
leaf-frame numbers above sit between and are the honest split: they charge a
sample to REMOV-PY only when a *Python* frame is innermost. `alan` (trivial):
0.16 s first / 0.03 s second solve — essentially all fixed tax; no residual to
mine.

---

## 2. The #1 removable sink: the per-IPM-iteration Python NLP-callback bridge

**What it is.** discopt's node/root NLPs are solved by POUNCE through the shared
`solvers/nlp_ipopt.py::_IpoptCallbacks` adapter, which on every IPM iteration
calls into `_jax/nlp_evaluator.py` (`evaluate_objective`,
`evaluate_constraints`, `evaluate_gradient`, `_evaluate_dense_jacobian` /
`sparse_jac_values`, `evaluate_lagrangian_hessian`). Each of those wraps a
JAX-compiled kernel but pays Python glue around it: `_current_params` dict
assembly, `np.asarray`/`array._value`/`__float__` marshaling of the JAX result
back to numpy, and (Ipopt-adapter) tuple/array packing.

**The call-count explosion (census, cProfile ncalls):**

| call | m3 | fac2 | nvs08 | nvs06 | ex1224 | nvs13 |
|---|---:|---:|---:|---:|---:|---:|
| POUNCE `solve` invocations | 258 | 240 | 828 | 911 | 260 | 44 |
| `evaluate_objective` | 10,331 | 21,131 | 17,534 | 13,102 | 7,415 | 634 |
| `evaluate_constraints` | 11,177 | 22,676 | 17,740 | 13,201 | 8,880 | 3,368 |
| `array._value` (JAX→np) | 10,563 | 21,414 | 17,576 | 13,128 | 7,468 | 785 |
| `np.asarray` (total) | 197,548 | 419,645 | 91,263 | 70,443 | 86,581 | 63,135 |

So fac2's 6 s solve does **~21 k JAX-kernel round-trips just for the objective**
and another ~23 k for constraints, each with a numpy marshaling tail. This is
the analog of the 07-05 hda "32.2 M size calls" churn note, but here it is on
the *hot* path (per IPM iteration) and it is where 24–31 % of the small-instance
wall actually goes.

**Why it is per-solve, not per-node:** `evaluate_*` count ≈ (IPM iterations) ×
(NLP solves), and NLP solves are dominated by the **root multistart + diving**,
not the tree — nvs06 has 911 POUNCE solves over 5 nodes (§4).

---

## 3. The #2 removable sink: scipy sparse structure rebuild/convert per call

`nvs13` spends **40 % of its wall** in scipy sparse *format conversions* —
`_csc.py:tocsr` (28 %), `_coo.py:_coo_to_compressed` (8 %), `_csr.py:tocsc`
(3 %) — and ex1224 spends ~14 % there. This is the relaxation matrix
(McCormick LP / MILP relaxation) being assembled as COO/CSR and converted to
the format the LP call wants, **rebuilt every node** even though the sparsity
pattern is identical across the tree (per the 07-02 profile's item 6 and B6:
"structure re-derivation per call"). The second-solve probe confirms it is
per-node work, not fixed tax (nvs13 second solve still 1.0 s). On the sub-2 s
instances this is the single largest removable slice.

---

## 4. Per-node vs fixed verdict (root-phase dominated)

`root_time` share and POUNCE-solve volume:

| instance | wall | nodes | root_time | root% | POUNCE solves | /node |
|---|---:|---:|---:|---:|---:|---:|
| m3     | 3.31 s | 61 | 1.98 s | 60 % | 258 | 4.2 |
| fac2   | 6.07 s | 69 | 4.53 s | 75 % | 240 | 3.5 |
| nvs06  | 3.57 s | 5  | 3.48 s | 97 % | 911 | 182.2 |
| ex1224 | 2.29 s | 53 | 1.74 s | 76 % | 260 | 4.9 |
| nvs08  | 3.03 s | 57 | 2.77 s | 91 % | 828 | 14.5 |
| nvs13  | 1.55 s | 19 | 0.93 s | 60 % | 44  | 2.3 |

**Verdict: the residual is ROOT-PHASE and NLP-solve-volume dominated, not
per-B&B-node.** The tree itself is cheap (fac2 tree = 1.5 s / 69 nodes ≈
22 ms/node, already at the cert-gate target, matching the 07-05 finding). The
removable Python overhead rides on the **root multistart / diving NLP binge**
(nvs06: 911 solves at the root) and the per-call relaxation rebuild. Fixed
import/JIT tax is a flat ~0.5 s (measured: nvs13 first 1.50 s → second 1.01 s;
m3 3.31 s → 2.78 s; alan 0.16 s → 0.03 s), so it only matters for the sub-2 s
class.

---

## 5. Coverage map — which path + do the new capabilities fire?

Per-instance dispatch path (traced by wrapping every `solver.py::_solve_*`) and
capability firing (monkeypatched counters). **All six take the spatial
`_solve_nlp_bb` McCormick LP-relaxer path**, with `_solve_root_node_multistart`
→ `_solve_node_nlp_pounce` as the per-node bound engine.

**Default config (caps OFF — this is the standard panel / BARON re-measure
config):**

| instance | path | run_root_fixpoint (calls/tightened) | reduce_node (calls/tightened) | PSD sep calls | wall | nodes |
|---|---|---:|---:|---:|---:|---:|
| m3     | spatial `_solve_nlp_bb` | 0 / 0 | 0 / 0 | 0  | 3.21 s | 61 |
| fac2   | spatial `_solve_nlp_bb` | 0 / 0 | 0 / 0 | 0  | 5.63 s | 69 |
| nvs13  | spatial `_solve_nlp_bb` | 0 / 0 | 0 / 0 | 18 | 1.50 s | 19 |
| nvs08  | spatial `_solve_nlp_bb` | 0 / 0 | 0 / 0 | 56 | 2.89 s | 57 |
| ex1224 | spatial `_solve_nlp_bb` | 0 / 0 | 0 / 0 | 52 | 2.11 s | 53 |
| nvs06  | spatial `_solve_nlp_bb` | 0 / 0 | 0 / 0 | 4  | 3.52 s | 5  |

**Forced ON (`DISCOPT_ROOT_FIXPOINT=1 DISCOPT_NODE_REDUCE=1
DISCOPT_PSD_COST_GATE=1`):**

| instance | run_root_fixpoint (calls/tightened) | reduce_node (calls/tightened) | wall | nodes |
|---|---:|---:|---:|---:|
| m3     | 0 / 0 | 0 / 0  | 3.55 s | 61 (unchanged — never reaches reduce block) |
| fac2   | 0 / 0 | 0 / 0  | 5.77 s | 69 (unchanged — never reaches reduce block) |
| nvs13  | 1 / 0 | 47 / 14 | 2.00 s | 49 (**worse**: 19→49) |
| nvs08  | 1 / 1 | 3 / 0  | 2.83 s | 5 (**better**: 57→5) |
| ex1224 | 1 / 1 | 3 / 1  | 1.73 s | 5 (**better**: 53→5) |
| nvs06  | 1 / 1 | 3 / 0  | 3.63 s | 5 (already 5) |

**Findings:**
- **By default the R2 branch-and-reduce machinery never executes** on any panel
  instance — a direct explanation of the "no net change" BARON re-measure: the
  merged capability is inert under the default flags the panel runs with.
- **Even forced ON, m3 and fac2 never reach the reduce block** (0 calls). Their
  spatial `_solve_nlp_bb` path bypasses the root-fixpoint/node-reduce block
  entirely — the block is only wired on the sub-path that nvs08/ex1224/nvs06
  reach (the reduce loop is gated at `solver.py:7396-7405` behind a "root
  already tight" skip *and* the block only sits on one spatial sub-path). So the
  two slowest small instances (fac2 6 s, m3 3.3 s) get **nothing** from R2 even
  when it is enabled.
- When it does fire it is a mixed bag: node-collapse win on ex1224/nvs08
  (53→5, 57→5) but a **regression on nvs13** (19→49 nodes) and much of
  `reduce_node` is inert (nvs13: 33 of 47 calls tighten nothing). This is a
  capability-targeting problem, not a Python-overhead one.

---

## 6. Ranked removable-Python bottlenecks (measured % of wall + fix + regime)

Ordered by (measured wall share on the panel) × (breadth across instances).
Regimes per CLAUDE.md §5: *bound-neutral* changes must keep `node_count` and
certified `objective` exactly unchanged; anything touching the relaxation math
is *bound-changing* (flagged, differential test, default-off).

1. **R1 — Collapse the per-IPM-iteration Python NLP-callback bridge**
   (`P_nlp_bridge`). **Measured: 24–31 % of wall on m3/fac2, 14–15 % on
   nvs06/nvs08/ex1224.** The cost is Python glue (`_current_params` dict build,
   `array._value`/`__float__`/`np.asarray` marshaling) wrapped around each JAX
   kernel call, executed ~21 k× per solve on fac2. **Fix hypothesis:** batch/
   fuse the objective+constraint+gradient+Jacobian evaluation into one JAX
   kernel returning a single packed array per IPM iteration (halving the round-
   trips and the marshaling), and hoist the `_current_params` assembly out of
   the per-call path; cache the JAX→numpy conversion buffer. **Regime:
   bound-neutral** (pure marshaling/eval-plumbing; the numbers POUNCE sees are
   identical, so `node_count`/`objective` must be byte-unchanged — verify on the
   cert panel). Expected win: if the glue is ~⅓ of the bridge share, ~8–10 % of
   m3/fac2 wall. Biggest, broadest lever.

2. **R2 — Cache the relaxation sparse structure across nodes; stop re-converting
   formats** (`P_scipy_sparse`). **Measured: 40 % of wall on nvs13, 16 % on
   ex1224** (smaller on the others). The McCormick/MILP relaxation matrix is
   rebuilt as COO/CSR and converted (`tocsr`/`tocsc`/`_coo_to_compressed`) every
   node though the sparsity pattern is invariant across the tree. **Fix
   hypothesis:** build the CSC/CSR structure once at the root, keep it, and
   patch only the changed coefficient/bound values per node (the 07-02 profile's
   own recommended fix, and the persistent bound-patchable-LP direction of
   cert-gap-plan Phase 1). **Regime: bound-neutral** (same LP, same optimum;
   only the assembly path changes). Highest per-instance ceiling on the sub-2 s
   class.

3. **R3 — Fixed import/JIT tax trim** (`F_import`). **Measured: flat ~0.5 s;
   33 % of nvs13, ~16 % of m3, negligible on the ≥3 s instances.** Lazy-import
   hoisting + JIT warmup reuse. **Regime: bound-neutral.** Low ceiling, low
   risk; only moves the sub-2 s instances and is dwarfed by R1/R2 on the rest.
   (This is the 07-05 F7 item, unchanged.)

Non-candidates (measured dead here): the Python FBBT/OBBT DAG walk
(`nonlinear_bound_tightening.py`) is a *huge* call-count sink (fac2: 470 k
`walk`, 6.25 M `isinstance`, 3.3 M `abs`, 812 k `core.size`) but only
~0.2 s of tottime and **~0 % of sampled wall** — the 07-05 "hda churn" pattern,
confirmed secondary here (each call is cheap; total wall is tiny). Do **not**
spend the CSE/hash-cons budget on it for this panel. Term/structure
classification and McCormick build are ≤10 % and only on ex1224/nvs13.

**Separately (not a Python-overhead item, but the biggest "why slow vs BARON"
finding):** the recent R2 branch-and-reduce capabilities are **default-OFF and
inert on the whole panel, and structurally unreachable on the two slowest
instances (m3, fac2) even when enabled** (§5). Wiring the reduce block onto the
m3/fac2 spatial sub-path, and re-targeting it so it does not regress nvs13,
is a bound-changing/relaxation task for the cert-gap-plan roadmap — tracked
there, not here.

---

## 7. Kill criterion

**The kill criterion (wall dominated by genuine compute, Python <20 %) did NOT
fire.** Removable Python overhead is **19–53 % of wall** across the panel and
**≥24 % on the two slowest instances (m3, fac2)**. Genuine POUNCE-IPM + Rust
compute is the majority (52–70 %) on 5 of 6, so the panel is *not* purely
Python-bound — but the removable slice is large, concentrated in two sinks
(the per-IPM Python NLP bridge and the per-node scipy sparse rebuild), and
therefore actionable. The residual gap vs BARON is **partly** removable
Python (R1/R2) and **partly** genuine per-solve NLP compute driven by the root
multistart volume (a policy/engine question, cf. 07-05 §5 F6, already resolved
as not-a-Python-lever).

---

## 8. Artifacts (reusable)

- `discopt_benchmarks/scripts/python_residual_profile.py` — Python-overhead
  profiler: clean/cprofile/sampled modes, top-level GENUINE-vs-REMOVABLE-Python
  bucketing, a call-count census, `--second-solve` fixed-tax probe, and a
  200 Hz in-process sampler with collapsed-stack (`--sample-out`) output for the
  leaf-frame attribution used in §1.
- `discopt_benchmarks/scripts/coverage_map.py` — coverage-map probe:
  monkeypatch-wraps the `solver.py` dispatch entry points (records the chosen
  path) and the capability entry points (`run_root_fixpoint`, `reduce_node`,
  `_separate_psd`) with call/tighten counters. Run with the `DISCOPT_*` flags to
  test firing under both default and forced-ON config.
- Raw per-instance JSON / pstats / collapsed stacks were produced in the session
  scratchpad (not committed; regenerate with the two scripts above — every table
  states its instance + mode).
