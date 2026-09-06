# #1180 — the per-node layer split, re-measured after the POUNCE-tape swap

**Status:** measurement complete; the build it names is scoped in §6.
Successor to #1026 (closed as superseded). This document replaces
`baron-gap-plan.md` §1.1 and §1.3 as the current account of where per-node time
goes; those tables measured the JAX evaluator, which no longer runs on any
default solve.

**Probes** (all measurement-only, all in `discopt_benchmarks/scripts/`):
`issue1180_layer_split.py`, `issue1180_floor_decomposition.py`,
`issue1180_node_nlp_candidates.py`, `issue1180_probe_lp_cost.py`.
Raw JSON records are cited per section.

---

## §0. What was measured on, and why the old absolute numbers are not comparable

Every number here was taken in one quiet Linux container (4 cores, load average
< 0.6 at every run start, arms interleaved where a comparison is claimed, spread
reported). That box is **much slower than the one the 2026-07-15 §1.1/§1.3
session used**, and the calibration is measured rather than guessed:

| yardstick | 2026-07-15 box (`baron-gap-plan.md`) | this box | ratio |
|---|---:|---:|---:|
| `alan` solve, fresh process | 80 ms | 1211 ms | **15.1×** |
| `import discopt` | 66 ms | 248 ms | 3.8× |
| `import pounce` | 148 ms | 210 ms | 1.4× |

So **wall-clock and nodes/s figures here must not be compared against BARON's
June numbers, and are not.** What travels across boxes is the *composition* — the
share of a solve each layer takes, the number of primitive calls per node, and
the cost of one call relative to another — and that is what this document
reports. Where an absolute time appears it is labelled with the box.

---

## §1. Deliverable 2 — the fresh-subprocess floor (supersedes §1.1)

`issue1180_floor_decomposition.py`, `alan.nl`, median of 5 fresh subprocesses,
`scratchpad`-recorded JSON. The parent times the whole process, so interpreter
startup is inside the total; the child reports its own phases.

| phase | 2026-07-15 (old box, JAX path) | **2026-09-05 (this box, tape path)** |
|---|---:|---:|
| `import jax` | 299 ms | **— (does not happen)** |
| interpreter startup + teardown | not separated | 94 ms |
| `import discopt` | 66 ms | 248 ms |
| `import pounce` | 148 ms | 210 ms |
| parse `.nl` | 2 ms | 2 ms |
| solve (13 nodes here / 21 there) | 80 ms | 1211 ms |
| **total process wall** | **595 ms** | **1752 ms (sd 38 ms)** |

Read the two columns as different machines, per §0 — **not** as a regression.
The transportable results are:

1. **The 299 ms `import jax` row is gone, and its absence is a measurement, not
   an assumption.** The probe carries a control arm: the identical decomposition
   under `DISCOPT_NLP_EVAL=jax` *does* import jax (217 jax modules in
   `sys.modules`) and *does* pay for it. Without that arm a probe that simply
   cannot see imports would report the same "no jax" and read as a pass; the
   probe asserts the control arm imports jax and exits non-zero if it does not.
2. **The floor was re-measured, not back-subtracted**, as the issue requires. On
   this box the whole pre-solve floor (startup + both imports + parse) is
   **554 ms of a 1752 ms process, i.e. 32 %** — against 86 % in the old table.
   Most of that shift is the 15× slower solve on this box inflating the
   denominator; the import rows themselves did not shrink.
3. **The tape default is worth 1.6× on a fresh trivial process, more than the
   import alone.** Same instance, same box, tape vs `DISCOPT_NLP_EVAL=jax`:
   **1752 ms vs 2806 ms** (sd 38 / 109), identical 13 nodes and identical
   objective. Standalone `import jax` on this box is 430 ms, so ~620 ms of the
   1054 ms difference is the JAX evaluator being slower *per solve*, not the
   import.

---

## §2. Deliverable 1 — the layer split at current `main` (supersedes §1.3)

### §2.1 Method

Two arms per instance, never a single-layer profiler label (`baron-gap-plan.md`
§8 item 5):

* **Clean arm** — no profiler. Wall, nodes, and the FFI-boundary split
  `discopt._timing` accumulates (`rust` and `python` partition the wall;
  `pounce ⊆ rust`, `jax ⊆ python`).
* **cProfile arm** — the same solve, every profile entry classified by the layer
  its *code* lives in and aggregated by **self** time, so a native frame that
  calls back into Python is not credited with the callback.

One classification rule had to be established rather than assumed, and it is the
§8-item-5 trap in its 2026 form: pounce ≥ 0.11 rebinds `Problem.solve` to a
Python warm-start shim (`pounce/_warm_start.py`) that calls the saved native
method, and cProfile records that inner call under a bare function repr with no
module path. Left unclassified it lands in "python" and invents a **27 %
Python share that is really the native IPM**. The rule was added only after
confirming the caller edge (`_solve_with_warm_start`) in the raw `pstats`.

### §2.2 `nvs05`, 20 s budget — the instance §1.3 was measured on

Clean arm: 27 nodes in 20.50 s. cProfile arm: 27 nodes in 20.31 s — **the same
node count and 0.99× the wall**, so the profiler is not distorting this
attribution.

| layer | share of self time (2026-09-05) | §1.3 (2026-07-15, JAX era) |
|---|---:|---:|
| **`discopt._rust` (the LP)** | **50.8 %** | 3.4 % |
| **POUNCE native (IPM + tape)** | **26.1 %** | 0.1 % |
| jax / XLA | **0.0 %** | 12.3 % |
| evaluator callback glue (Python) | 4.7 % | — (inside the 82.5 %) |
| `discopt` Python | 8.3 % | — |
| numpy / scipy Python | 4.8 % | — |
| other Python | 5.4 % | — |
| *(old table's aggregate "python")* | *23.2 %* | *82.5 %* |

**The attribution has inverted.** The row §1.3 dismissed — "`solve_lp_warm_csc_py`:
0.67 s — the node LP is nothing" — is now the largest single item in the solve,
and the Python marshaling that was 82.5 % is 23 %.

Component seams from the same profile (`cum` is an upper bound per seam and must
not be summed; `self` is additive):

| seam | calls | self | cum | note |
|---|---:|---:|---:|---|
| `discopt._rust.solve_lp_warm_csc_py` | 1175 | 10.19 s | — | 8.68 ms per call |
| `obbt.py:_solve_probe` | 1082 | — | 8.87 s | **92 % of all LP calls are OBBT probes** |
| `obbt_tighten_root` (root *and* per-node) | 14 | — | 10.49 s | 52 % of wall |
| `pounce` native IPM (`Problem.solve`) | 59 | 5.05 s | 6.55 s | 86 ms per NLP solve |
| `nlp_pounce.solve_nlp` | 59 | — | 6.57 s | |
| tape evaluation (all 5 `NlProblem` entries) | 59 516 | **0.24 s** | — | **4.1 µs per callback** |
| evaluator callback glue (Python frames) | 59 516 | 0.95 s | — | **16 µs per callback** |
| `build_milp_relaxation` | 47 | — | 1.61 s | |

Two readings follow directly:

* **The derivative callbacks are no longer the problem.** 59 516 callbacks cost
  0.24 s of actual tape arithmetic. The old table's `np.asarray` 1.71 s +
  `array._value` 1.53 s + `__float__` 2.17 s are gone with the engine that
  produced them.
* **The Python glue around those callbacks now costs ~4× the arithmetic it
  wraps** (16 µs vs 4.1 µs per callback) — but at 4.7 % of wall, removing *all*
  of it is worth at most 1.05×. It is a real inefficiency and a small lever.

### §2.3 Root vs tree on `nvs05` (`--root-arm`, `max_nodes=1`)

| | full solve (27 nodes) | root only (`max_nodes=1`) | difference = tree |
|---|---:|---:|---:|
| wall | 20.31 s | 12.10 s | 8.21 s |
| POUNCE NLP solves | 59 | **55** | **4** |
| POUNCE IPM self time | 5.05 s | 5.00 s | 0.05 s |
| OBBT probe LPs | 1082 | 293 | **789** |
| `solve_lp_warm_csc_py` self | 10.19 s | 4.29 s | 5.90 s |

**On this instance the node-NLP is not the per-node cost at all.** 55 of the 59
NLP solves are the *root* multistart; the tree runs 4. Per-node cost in the tree
is ~33 OBBT probe LPs per node, and it is the LP that is paid for.

### §2.4 The corpus panel — 66 in-repo instances, 20 s each

`nvs05` alone is a gate probe, not a class (CLAUDE.md §2), so the same two arms
were run over the whole in-repo corpus (`python/tests/data/minlplib_nl`, 66
instances, 20 s budget, 5185 nodes, 504.9 s of clean-arm wall, 333 executed
assertions). Layer shares are **wall-weighted over the corpus** (self time
summed across instances, then divided), with the per-instance median beside it
so one long instance cannot carry the table:

| layer | wall-weighted share | per-instance median (min–max) |
|---|---:|---|
| POUNCE native (IPM + tape) | **47.7 %** | 43.9 % (0.0–79.0) |
| `discopt` Python | 15.4 % | 19.2 % (2.5–55.0) |
| `discopt._rust` (LP/MILP) | 15.0 % | 4.8 % (0.2–54.5) |
| other Python (stdlib, contextlib, builtins) | 9.4 % | 10.6 % (2.2–32.9) |
| evaluator callback glue | 9.2 % | 7.5 % (0.0–21.5) |
| numpy / scipy Python | 3.4 % | 4.9 % (0.5–23.2) |
| **jax / XLA** | **0.00 %** | **0.0 % (0.0–0.0)** |

Corpus component totals (same rules: `cum` per seam is an upper bound, `self` is
additive):

| seam | calls | self | cum | share of self total |
|---|---:|---:|---:|---:|
| `nlp_pounce.solve_nlp` (the node/root NLP) | 3080 | — | 298.7 s | **57.4 % (cum)** |
| POUNCE native IPM | 3107 | 233.8 s | 302.3 s | **44.9 % (self)** |
| `discopt._rust.solve_lp_warm_csc_py` | 8542 | 48.2 s | — | 9.3 % |
| evaluator callback glue (Python frames) | 1 952 138 | 47.7 s | — | 9.2 % |
| `mccormick_lp._solve_at_node_impl` | 386 | — | 71.4 s | 13.7 % (cum) |
| `build_milp_relaxation` | 677 | — | 55.1 s | 10.6 % (cum) |
| `obbt.py:_solve_probe` | 5817 | — | 22.5 s | 4.3 % (cum) |
| `discopt._rust.solve_milp_csc_py` | 341 | 18.5 s | — | 3.6 % |
| **tape evaluation (all 5 native entries)** | **1 952 138** | **11.5 s** | — | **2.2 %** |

**The headline, corpus-wide: the per-node cost is the NLP solve, and the NLP
solve is native.** `solve_nlp` accounts for 57 % of profiled wall, and 45 % of
*all* self time is POUNCE's own IPM — code with no Python in it. The Rust LP is
the second consumer at 9.3 %, and `nvs05`'s OBBT-probe dominance (§2.2) is a
two-instance phenomenon (`nvs05` 41.6 %, `nvs09` 30.0 %, every other instance
≤ 10.6 %), not the class.

**The evaluator-callback picture repeats at corpus scale.** 1.95 million
callbacks cost **11.5 s of tape arithmetic (2.2 %)** wrapped in **47.7 s of
Python glue (9.2 %)** — 5.9 µs of math per callback under 24.5 µs of frames, the
same ~4× ratio as `nvs05`. Deleting *every* Python frame between POUNCE and the
tape is worth at most **1.10×** corpus-wide.

**Two honesty notes on this table.**

1. *cProfile distortion is not negligible here.* Profiled wall / clean wall is
   median **1.23×** (0.63×–2.18×), and 8 of 66 instances land on a different
   node count under the profiler. Python-frame-heavy instances inflate most, so
   every Python share above is an **upper bound** and every native share a lower
   bound. The direction of the headline is therefore safe: native dominance is
   understated, not overstated.
2. *The clean arm's FFI split is a cross-check on the headline only*, not a
   second estimate of the same partition — `discopt._timing` charges the tape
   callbacks (glue included) to `rust`, and anything native that is not inside a
   `charge` region falls into its `python` residual. Over the corpus it reports
   **rust 292.2 s / python 209.9 s / jax 0.00 s**, with `jax` absent from
   `sys.modules` on all 66. That agrees with cProfile on both things it can
   see: native-dominated, and no JAX.

A defect found while reading it: `_ElasticFeasibilityEvaluator` (solver.py)
declares no `timing_bucket`, so it trips the `[timing-bucket-unknown]` warning
and its derivative-callback time is left with the enclosing region. It is an
attribution gap in the FFI instrument, not a solver bug, and it is why the
cProfile arm is the primary instrument here.

---

## §3. Deliverable 3 — the three surviving candidates, separated

`issue1180_node_nlp_candidates.py`, 10 instances chosen from §2.4 as the
NLP-dominated ones, 20 s each. #1026's fourth candidate ("per-iterate JAX
dispatch latency") is not measured because it no longer exists.

**A measurement hazard found while building this probe, which invalidates any
profile taken with a `node_callback` attached.** Attaching one is *not*
observation-neutral: it is a documented routing signal — `_MIP_NLP_IGNORED_OPTIONS`
refuses to auto-route when it is set, and the GP probe and the
substitution-presolve gate do the same — so a probe that attaches one measures a
different engine. Measured on `alan` in fresh subprocesses, both orders, same 13
nodes and the same objective 2.925: **without a callback the solve runs 54 POUNCE
NLP solves and 11 130 tape evaluations; with one it runs 1 and 0.** Neither this
probe nor the layer-split probe attaches one; the root/tree split comes from a
companion `max_nodes=1` run instead. (`profile_instance.py` attaches a
`node_callback` by default — its `--no-trace` flag is not optional for an
auto-routable instance.)

### §3.1 Candidate 1 — POUNCE iterations per node-NLP

| instance | nodes | NLP solves | root-only solves | median iters | ms/solve | callbacks/solve | NLP wall / solve wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| alan | 13 | 54 | 4 | 36 | 16.3 | 211 | 69 % |
| gkocis | 7 | 61 | 60 | 38 | 17.6 | 221 | 81 % |
| nvs21 | 9 | 63 | 59 | 11 | 6.9 | 69 | 86 % |
| ex1225 | 5 | 133 | 133 | 18 | 63.0 | 542 | 96 % |
| st_e29 | 5 | 136 | 136 | 17 | 12.7 | 118 | 94 % |
| tspn08 | 15 | 139 | 139 | 31 | 87.0 | 774 | 80 % |
| tls2 | 245 | 320 | 17 | 15 | 15.0 | 89 | 90 % |
| nvs05 | 27 | 60 | 55 | 14 | 9.6 | 79 | 30 % |
| tanksize | 2607 | 33 | 72 | 53 | 233.5 | 1477 | 37 % |
| 4stufen | 3 | 9 | 9 | 137 | 791.0 | 4322 | 88 % |

Iteration counts are **ordinary** — a median of 11–53 on nine of ten instances,
137 on the one 157-variable model. There is no pathology here: an IPM taking 15–40
iterations on a node relaxation is what an IPM does. The lever is not "the solves
take too many iterations".

What the table does say is that **on most of these instances the NLP volume is
root work, not per-node work**: a `max_nodes=1` solve runs essentially the same
number of NLP solves as the full 20 s solve (ex1225 133/133, st_e29 136/136,
tspn08 139/139, gkocis 60/61, nvs21 59/63, nvs05 55/60, 4stufen 9/9). The
exceptions are `tls2` (17 of 320) and `tanksize`, whose root-only run does *more*
NLP solves than the full run (72 vs 33) — the root-only companion is a separate
solve with its own budget, not a prefix of the full one, so it is read as "a
root-only solve costs about this much", never as a subtraction.

### §3.2 Candidate 2 — Python frame overhead in the callback path

Per-callback cost by layer, median of 7 interleaved rounds × 2000 reps, sd in the
JSON record. The chain is `_IpoptCallbacks.<cb>` → `_charge_evaluator.wrapper` →
`_timing.charge` → `_BoundOverrideEvaluator.__getattr__` →
`TapeNLPEvaluator.evaluate_*` → `_x` → the native tape.

| instance | n | native only | + `_x` list build | + tape wrapper | + proxy | **full callback** |
|---|---:|---:|---:|---:|---:|---:|
| alan | 8 | 0.35 µs | 2.04 | 2.37 | 2.50 | **6.55 µs** |
| nvs05 | 15 | 0.38 | 2.63 | 2.89 | 3.11 | **6.92** |
| tspn08 | 44 | 1.35 | 5.23 | 5.49 | 5.72 | **9.92** |
| tanksize | 47 | 0.57 | 4.61 | 4.90 | 4.97 | **8.67** |
| 4stufen | 157 | 1.52 | 12.29 | 12.72 | 12.58 | **16.88** |

Two costs dominate, and both are pure plumbing:

1. **`_timing.charge` — the layer-profile instrument itself — is the single
   largest item**, ~4 µs of a 6.5 µs `alan` objective callback (the step from
   "proxy" to "full callback"). It is a `@contextlib.contextmanager` generator
   entered once per derivative callback.
2. **`_x`'s per-callback Python list build** is the second, and it *scales with
   n*: 1.7 µs at n=8, 10.8 µs at n=157, while the arithmetic underneath stays at
   0.35–1.5 µs.

So on `alan` the tape arithmetic is **5 %** of the callback that delivers it. The
honest sizing, though, is the corpus one from §2.4: the whole callback path is
9.2 % of wall, so removing all of it is worth at most 1.10×.

### §3.3 Candidate 3 — warm-start quality

Median iterations over up to 12 captured node subproblems per instance, each
re-solved three ways after the solve (results discarded; the probe changes no
bound):

| instance | production warm `x0` | cold box midpoint | + full `pounce.WarmStart` |
|---|---:|---:|---:|
| alan | 36.0 | 41.0 | 19 |
| gkocis | **11.5** | 39.5 | 18 |
| nvs21 | 28.5 | 44.0 | 25 |
| ex1225 | 15.5 | 19.0 | 16 |
| st_e29 | 17.0 | 16.5 | 16 |
| tspn08 | 47.0 | 47.5 | 37 |
| tls2 | 19.0 | 14.5 | 37 *(1 replayable case of 12)* |
| nvs05 | 90.5 | 104.0 | 75 |
| tanksize | **52.5** | 166.0 | 61 |
| 4stufen | 137 | 225 | 152 |

**"Is every node re-solving from scratch?" — No.** The parent's point, clipped
into the child box, already buys a real reduction against a cold start on 7 of 10
instances (`gkocis` 3.4×, `tanksize` 3.2×, `4stufen` 1.6×, `nvs21` 1.5×), is
neutral on `tspn08`, and is slightly *worse* on `st_e29` and `tls2`. The primal
warm start is working.

**The full dual warm start is falsified as a lever.** Handing POUNCE the previous
solve's complete state (multipliers, bound multipliers, barrier parameter) via
`pounce.WarmStart` — which POUNCE 0.11 supports and `discopt` never constructs —
changes the median iteration count by a ratio of **0.53× to 1.95×, median
0.99×**: better on 4 instances, worse on 4, neutral on 2. Kill criterion met; not
built.

There is also a **structural** reason it cannot be built as stated: on `tls2`,
**10 of 12 consecutive node NLPs had different `(n, m)`** (captured n=37 m=24
against target n=38 m=30) because the cut pool changes the row count between node
NLPs. A dual state cannot be replayed across nodes that do not have the same
dimensions, so any future attempt has to map the state through the cut pool, not
merely carry it.

---

## §4. Deliverable 4 — what the measurement names, and what was built

Reading §2 and §3 together, the levers sort as follows.

| candidate | share of corpus wall | verdict |
|---|---:|---|
| POUNCE's native IPM | 44.9 % of self time | **not reachable from this repo** — native code in the pounce crate |
| Rust LP / MILP bindings | 12.9 % | **not reachable by scheduling changes** — `performance-plan.md` §16 established the per-pivot gap is in `feral`, and six in-repo levers are falsified |
| OBBT probe orchestration | 4.3 % (and 41.6 % on `nvs05` alone) | **already native**: 1–3 % Python per probe (§5) |
| node count / relaxation strength | — | a bound problem, out of scope here (#196/#208) |
| **evaluator callback path** | **9.2 %** | **the one in-repo lever**, and it is bound-neutral |

So the build is the callback path, and its ceiling is stated up front: **at most
1.10× corpus-wide** if every Python frame between POUNCE and the tape vanished.

### §4.1 The two changes

1. **`_timing.charge` is a `__slots__` context-manager class** instead of a
   `@contextlib.contextmanager` generator. Measured in-process, 200 k reps:
   **1.93 µs → 1.05 µs** per `with`; fully inlining the bookkeeping into the
   callback wrapper would give 0.73 µs, and was rejected — duplicating the
   parent/child self-time accounting in a second place is exactly the kind of
   defect factory §3 of CLAUDE.md is about, for 0.3 µs.
2. **`TapeNLPEvaluator._x` hands pounce a contiguous `float64` array** instead of
   rebuilding a Python list per callback (same for the multiplier vector).
   pounce accepts both and returns bit-identical values; the list was an `O(n)`
   Python loop on the hottest path in the solver *and* slower inside pounce
   (0.308 µs vs 0.199 µs for the same evaluation). Build-and-evaluate:
   `nvs05` (n=15) **2.11 µs → 0.20 µs**, `4stufen` (n=157) **12.29 µs → 1.52 µs**.

Note what the second one means: the marshaling cost scaled with `n` while the
arithmetic under it did not, so the benefit grows with model size rather than
being concentrated on one family.

### §4.2 The gate (CLAUDE.md §5, bound-neutral regime)

*Bit-identity, over the whole in-repo corpus.* Old list path vs new array path on
every tape entry point (`objective` / `gradient` / `constraints` / `jacobian` /
`hessian`), 66 instances × 5 points, nan-aware exact equality: **1610
comparisons, 0 mismatches** (`scratchpad/issue1180/bitcheck.py`, which prints its
comparison count and exits non-zero at zero).

*Suites.* `pytest -m smoke`: **1133 passed, 21 skipped, 2 xpassed, 0 failures**.
`test_74_layer_time_attribution.py`: 9 passed. New regression test
`test_1180_callback_marshaling.py` pins the marshaling contract (float64,
contiguous, zero-copy on an already-contiguous vector, bit-identity across all
five entry points with a count assertion so the loop cannot pass vacuously) and
`charge`'s accounting after the rewrite.

*A/B.* Both arms interleaved **in one process** with alternating order and a
discarded warm-up, each arm asserting which variant is actually live before it
runs, on a `deterministic=True` budget so a faster arm cannot do more nodes and
be mistaken for a behavior change.

### §4.3 Results

**Callback-heavy panel** (14 instances chosen from §2.4 by callback-glue share;
`deterministic=True`, `max_nodes=50`, `time_limit=120`, 3 interleaved reps):

| instance | old | new | speedup | neutral? |
|---|---:|---:|---:|---|
| ex1226 | 5.77 s | 4.14 s | **1.394×** | yes |
| ex1221 | 2.31 | 1.72 | 1.348× | yes |
| gkocis | 1.41 | 1.05 | 1.345× | yes |
| oaer | 1.78 | 1.39 | 1.279× | yes |
| tspn08 | 65.62 | 51.96 | 1.263× | yes |
| st_e29 | 4.90 | 3.89 | 1.260× | yes |
| tls2 | 6.19 | 5.04 | 1.228× | yes |
| ex1225 | 6.56 | 5.40 | 1.214× | yes |
| tspn10 | 49.50 | 40.81 | 1.213× | yes |
| nvs21 | 2.95 | 2.44 | 1.208× | yes |
| tanksize | 51.16 | 47.70 | 1.072× | yes |
| nvs05 | 51.97 | 50.38 | 1.032× | yes |
| alan | 0.04 | 0.04 | 1.010× | yes |
| 4stufen | 120.26 | 120.58 | 0.997× | time-truncated † |
| **median** | | | **1.221×** | |

† `4stufen` hits the 120 s limit in *both* arms, so its wall is set by the limit
and not by speed; in that same wall the new arm reaches **7 nodes against 3**, at
an identical bound. A time-truncated row measures throughput, not behavior.

**Whole-corpus sweep** (66 instances, `deterministic=True`, `max_nodes=20`, 1 rep):
median **1.099×**, and **62 of 66 exactly neutral** on node count, objective and
bound. The four that are not:

* `4stufen`, `heatexch_gen1`, `heatexch_gen3` — time-truncated in at least one
  arm, so the faster arm reaches more nodes. Bound identical or better.
* `clay0303hfsg` — **pre-existing nondeterminism, not an effect of this change**.
  The *old* arm alone returns two different objectives across four identical
  repeats under `deterministic=True` (55092.52 three times, then 46785.55), and
  so does the new one, always at the same 27 nodes. The dual bound agrees to 12
  significant figures across every run; it is the *incumbent* that moves, so a
  primal heuristic is the nondeterministic component. Recorded in §5 as a
  separate defect — an instance that cannot reproduce itself cannot serve in a
  bound-neutrality gate.

**The one apparent regression, and why it is not one.** `beuster` measured
**0.516×** (233 s vs 120 s), reproducibly, over three interleaved reps at
identical node counts — which would have sunk the change if taken at face value.
Two measurements dissolve it:

1. *Where the time goes.* The new arm's `root_time` is **112.6 s against the old
   arm's 120.1 s** — the root gets *cheaper*, as intended — and the wall runs on
   to 231 s anyway. `faulthandler.dump_traceback_later` (a timing wrapper that
   prints on return never fires for a call that does not return) puts the process
   at t = 150 s, 30 s past the limit, inside **root OBBT**
   (`_solve_probe` ← `run_obbt_on_relaxation` ← `obbt_tighten_root` ←
   `root_reduce._stage_obbt` ← `run_root_fixpoint`). The extra wall is an
   existing phase running past its deadline, reached further because the root got
   cheaper — not new work the change introduced.
2. *The gate's own setting caused it.* `deterministic=True` renders the role-2
   wall budgets **inert by design**, so that OBBT stage has no deadline to stop
   at. Re-run on the **ordinary wall budget** — the configuration a user actually
   gets — three interleaved reps:

   | arm | wall | nodes | dual bound |
   |---|---:|---:|---:|
   | old | 120.0 / 120.1 / 120.1 s | 3 / 7 / 7 | 6395.11 |
   | new | 122.2 / 123.9 / 124.5 s | **15 / 15 / 15** | **8322.32** |

   Same limit honoured to within 2–4 % on both arms, **5× the nodes**, and a
   **30 % tighter dual bound**. On the user-facing budget `beuster` is one of the
   change's better results, not its worst.

   The lesson generalises and is the reason this section exists: **a deterministic
   budget makes node counts comparable but makes wall clock meaningless for any
   phase whose stopping rule *is* the wall budget.** Neutrality and speed have to
   be read from different arms.

`nvs12`, the only other sub-1.0× row in the corpus sweep, is **0.945×** over
three reps with fully overlapping spreads (new 37.17–45.42 s, old 36.04–47.46 s)
— noise, and reported as noise rather than as a result.

