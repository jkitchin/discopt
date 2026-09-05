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

