# PYFIX-1 — the per-node Python DAG walk is NOT a removable wall sink (kill criterion fired, twice)

**Date:** 2026-07-06
**Branch:** `pyfix1-nbt-walker` at `origin/main` (7199ab8b)
**Status:** measurement-only. **No production `python/discopt` or Rust code
changed.** PYFIX-1's premise was falsified by direct measurement; this note
records the falsification and the re-scope, per CLAUDE.md §4 ("the measurement
wins — record the falsification").
**Machine:** Apple M4 Pro (arm64), Python 3.12, JAX 0.10.2
(`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`), pounce 0.7.0, numpy, scipy,
`maturin develop --release`. Panel: fac2 (69 nodes, ~6 s) and m3 (61 nodes,
~3.3 s) — the "BARON sub-second / discopt seconds" residual-overhead class.

---

## 0. Verdict

**PYFIX-1 as scoped is rejected.** The task premised that
`python/discopt/_jax/nonlinear_bound_tightening.py` (the per-node FBBT DAG
walker) was ~40–60 % of fac2/m3 wall, and asked to cache/route its traversal.
That premise is **wrong**: the NBT walker is **~0 % of real wall**. Its large
cProfile *tottime* is a cProfile-per-call-overhead artifact on cheap,
high-call-count functions. PYPROF-1's 200 Hz leaf-frame sampler
(`docs/dev/python-residual-profile-2026-07-06.md`) and this session's own
measurements agree.

Re-targeting to PYPROF-1's stated #1 lever — the per-IPM-iteration Python
**NLP-callback bridge** (`_jax/nlp_evaluator.py` + `solvers/nlp_ipopt.py`) —
and measuring it directly showed that the bridge's large *sampled* share
(24–31 % of m3/fac2) is **81–93 % genuine JAX kernel dispatch + XLA execute**
(which runs natively *under* the Python `evaluate_*` frame, so the sampler
charges it to that Python frame), and only **~2 % of wall is removable Python
marshaling**. The kill criterion fired again for fac2/m3: there is **no large
removable-Python lever** on these two instances; their wall is genuine
POUNCE-IPM + JAX-Jacobian compute.

The one remaining genuine removable-Python per-node sink measured here is
**scipy sparse format churn** (PYPROF-1's #2), which is real but lives on a
*different* instance class (nvs13/ex1224, sub-2 s), not fac2/m3. It is the
correct next lever and is scoped below for a follow-on.

---

## 1. NBT walker is a cProfile artifact, not wall (falsifies the PYFIX-1 premise)

cProfile of m3 (release build, release pounce), sorted by tottime:

| function | ncalls | tottime | note |
|---|---:|---:|---|
| `pounce._pounce.Problem.solve` | 258 | 1.728 s | genuine IPM (the real work) |
| `numpy.asarray` | 197,665 | 0.563 s | marshaling |
| `isinstance` (builtin) | 2,420,503 | **0.115 s** | ~0 wall each; cProfile counts the call |
| `nlt.py:849 tighten` | 308 | 0.101 s | |
| `nlt.py:2381 walk` | 159,210 | **0.069 s** | the "469K walk" sink — 0.069 s tottime |
| `nlt.py:191 _constant_value` | 332,518 | **0.069 s** | the "925K" sink — 0.069 s tottime |
| `nlt.py:203 match` | 178,192 | 0.053 s | |
| `nlt.py:56 scalar_flat_index` | 283,666 | **0.053 s** | the "882K" sink — 0.053 s tottime |
| `abs` (builtin) | 797,867 | **0.033 s** | ~0 wall each |

The high-call-count NBT functions (`walk`, `_constant_value`,
`scalar_flat_index`) and the `isinstance`/`abs` churn they drive have tiny
*tottime* (0.03–0.07 s each) — a few % of a 3.3 s solve at most, and the
sampler attributes ~0 % of wall to them. cProfile inflates them only because it
instruments **every call**, and these are called millions of times but each
call is nanoseconds. Caching their (genuinely invariant) traversal structure
would remove call-count, not wall. **Kill criterion #1: the NBT walker's
per-node cost is not the lever.**

(For the record, the walker's structure *is* invariant across nodes and *could*
be cached — `PeriodicVariableBoundRule.walk`, `FunctionDomainBoundRule.walk`
and every rule re-derive the same constraint-DAG matches every node — but
doing so buys ~0 wall, so it is not worth the byte-identity risk.)

---

## 2. The NLP-callback bridge is ~93 % genuine JAX dispatch, ~2 % removable

PYPROF-1's #1 lever is the per-IPM-iteration bridge in `_jax/nlp_evaluator.py`.
Direct micro-measurement of each callback (20 k warm calls, m3/fac2):

**Objective** (`float(self._obj_fn_jit(x, params))`):

| step | m3 | fac2 |
|---|---:|---:|
| JAX jit dispatch only | 2.88 us | 2.89 us |
| + `float()` (current) | 5.71 us | 5.70 us |
| **removable `float()` tail** | **2.83 us** | **2.81 us** |

`float()` on a 0-d `jax.Array` takes the slow `__float__` device→host path;
`np.asarray(r).item()` is ~2.5 us cheaper and **bit-for-bit identical**
(verified: 500 random points × 4 instances = 2000 points, byte-equal via
`struct.pack('<d', …)`). Real, but small.

**Dense Jacobian** (`_evaluate_dense_jacobian`, the *largest* nlp_evaluator
leaf frame — 17.3 % m3 / 15.5 % fac2 in the sampler):

| step | m3 | fac2 |
|---|---:|---:|
| JAX jit dispatch only | 21.87 us | 14.32 us |
| + `np.asarray` + COO projection (current) | 23.45 us | 17.58 us |
| **removable marshaling+projection tail** | **1.58 us (7 %)** | **3.26 us (19 %)** |

So the biggest nlp_evaluator leaf frame is **93 % (m3) / 81 % (fac2) genuine
JAX Jacobian compute** — irreducible in Python.

**`_current_params()` is a no-op here:** fac2 and m3 have **0 parameters**, so
the per-call param re-marshaling PYPROF-1 flagged saves nothing on this panel
(measured 0.11 us/call, an empty tuple).

**Total removable marshaling across the whole bridge, per real solve**
(callback counts instrumented on an actual solve × measured removable tail):

| instance | obj | grad | cons | dense-jac | hess | total removable | **% of wall** |
|---|---:|---:|---:|---:|---:|---:|---:|
| m3   | 10,331 | 4,612 | 11,177 | 7,974 | 6,770 | **79 ms** | **2.4 %** |
| fac2 | 21,131 | 8,951 | 22,676 |   630 | 11,386 | **118 ms** | **2.2 %** |

**Kill criterion #2: removing 100 % of the bridge's removable Python marshaling
saves ~2 % of fac2/m3 wall, not the 8–10 % hypothesized.** The sampled 24–31 %
"bridge" share is dominated by genuine JAX kernel dispatch that lives under the
Python `evaluate_*` frame. fac2/m3 wall is genuine POUNCE-IPM + JAX-Jacobian
compute; there is no large removable-Python lever on them.

Cross-check (applied the byte-identical `float()`→`np.asarray().item()` fix and
re-solved): m3 3.30 s / 61 nodes / obj 37.79999966997884; fac2 5.51 s / 69
nodes / obj 331837498.18201375 — objective + node_count byte-identical, wall
change in the noise. Reverted (a ~0.4 %-of-wall change is not PYFIX-1).

---

## 3. The one real removable lever left: scipy sparse churn (nvs13/ex1224, NOT fac2/m3)

PYPROF-1's #2 sink, re-confirmed here on nvs13 (sampler leaf frames): **34.3 %
of wall in scipy sparse format conversion** — `_csc.py:tocsr` 22.9 %,
`_coo.py:_coo_to_compressed` 7.8 %, `_csr.py:tocsc` 3.0 %. fac2/m3 show ~0 %
here, so this is a *different instance class*.

Traced callers (folded-stack attribution):

- **`_csc.py:tocsr` (dominant, 22.9 %)** ← `milp_simplex.py:_fbbt_eq_bounds`
  (line 95, `sp.csr_matrix(a_std).tocoo()`) ← `_safe_lp_lower_bound_std` ←
  `_result_basis_cert` ← `solve_lp_warm_std`. The safe (Neumaier–Shcherbina)
  LP dual-bound certificate re-derives the CSR/COO of the equality matrix
  **every LP solve**, though the sparsity *pattern* is invariant across nodes.
- **`_coo_to_compressed` (7.8 %)** ← `milp_relaxation.py:build_milp_relaxation`
  ← `mccormick_lp.py:solve_at_node`. The McCormick relaxation matrix reassembled
  as COO→compressed every node.

**Follow-on (bound-neutral, the correct PYFIX-2):** build the relaxation/LP
CSC/CSR structure once at the root and patch only changed values per node;
memoize the `_fbbt_eq_bounds` COO decomposition keyed to the `a_std`
pattern. Same 07-02 profile item-6 recommendation. It helps the sub-2 s class
(nvs13, ex1224), and must be done on the certificate path with care, so it is
its own PR — not folded into PYFIX-1.

---

## 4. Method / reproducibility

- cProfile + 200 Hz in-process leaf-frame sampler via PYPROF-1's
  `discopt_benchmarks/scripts/python_residual_profile.py`
  (`--mode sampled --sample-out …`); leaf-frame aggregation is the honest
  wall attribution (does not over-weight many-small-call Python functions the
  way cProfile tottime does).
- Per-callback micro-benchmarks (20 k warm calls) split JAX jit dispatch from
  the numpy/float materialization tail.
- Byte-identity of the candidate `float()` replacement verified on 2000 random
  interior points across m3/fac2/nvs08/ex1224.
- Baseline objectives/nodes match `docs/dev/data/cert-baseline.jsonl`
  (fac2 331837498.18201375 / 69; m3 37.79999966997884 / 61).
