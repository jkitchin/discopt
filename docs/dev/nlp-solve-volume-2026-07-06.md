# NLP-solve-volume attribution + cut — 2026-07-06 (VOLUME-1)

**Date:** 2026-07-06
**Status:** measured + prototype cut shipped behind a **default-OFF** env flag
(`DISCOPT_ILS_SOLVE_CAP`). One production edit
(`python/discopt/_jax/primal_heuristics.py`), one reusable script
(`discopt_benchmarks/scripts/nlp_solve_volume.py`).
**Scope:** the confirmed lever behind "discopt is seconds where BARON is
sub-second" — the **volume** of POUNCE NLP subsolves. PYPROF-1 (#528,
`docs/dev/python-residual-profile-2026-07-06.md`) established the wall is
genuine NLP-solve + JAX-kernel VOLUME, not Python orchestration (~2 %
removable), with the smoking gun **nvs06 = 911 POUNCE NLP solves for a 5-node
problem**. VOLUME-1 is the *systematic accounting*: where do ALL the solves come
from, what is each source's incumbent hit rate, and is the total cuttable
without losing an incumbent or the certificate.
**Builds on:** `docs/dev/bottleneck-profile-2026-07-05.md` §B4 (120 POUNCE NLP
solves driven by diving/RINS on that panel), the F1 (#502, LNS-enumeration
budget) and F4 (#508, root-heuristic budget) volume-control history.

> **Method.** Branch `volume1-nlp-solve-volume` at `origin/main` (ed4b2265).
> Apple M4 Pro (arm64), Python 3.12.11, JAX 0.10.2 (`JAX_PLATFORMS=cpu`,
> `JAX_ENABLE_X64=1`), pounce **0.7.0** (release wheel `_pounce.abi3.so` =
> **4.7 MB** — verified NOT a debug build), numpy 2.5.1, scipy 1.18.0,
> `maturin develop --release`. Instances from `python/tests/data/minlplib_nl/`
> (plus nvs23/nvs24 drawn from the MINLPLib snapshot),
> `from_nl(path).solve(time_limit=60, gap_tolerance=1e-4)`, one at a time.
> Attribution via `discopt_benchmarks/scripts/nlp_solve_volume.py` (new): a
> pure monkeypatch that (1) wraps the single shared POUNCE entry point
> `discopt.solvers.nlp_pounce.solve_nlp`, walks the live stack, and buckets
> every solve by the nearest identifying caller frame; and (2) wraps each
> heuristic / multistart *entry* function to record the per-source
> incumbent-improvement HIT RATE (a call "hits" when it returns a feasible
> candidate strictly better than the running best-so-far). Node-count and
> objective for the cut A/B come from clean re-runs (no instrumentation).

---

## 0. Executive summary — the verdict

**The NLP-solve volume is NOT load-bearing on this panel — the kill criterion
did NOT fire — and it IS cuttable.** The single dominant source is the
objective-improvement coordinate descent inside `integer_local_search`
(`_objective_improve`, `primal_heuristics.py`), which on the integer-light
panel instances issues **hundreds** of continuous-repair sub-NLPs at a
**measured 0 % incumbent hit rate** — the incumbent is already found by the root
multistart's *first* start. A default-OFF solve cap on that descent cuts the
volume ~90 % with the certified objective, the node count, and the incumbent all
**byte-unchanged** on every instance tested.

1. **The 911 nvs06 solves decompose as: 888 `integer_local_search`, 15
   `integer_box_search`, 5 per-node engine NLPs, 2 feasibility_pump, 1 subnlp.**
   The 888 (97 % of the volume, 2.29 s of a 4.5 s wall) all come from
   `_objective_improve`'s inner `int_idx × {±1,±2}` coordinate search, each move
   a full continuous-repair `subnlp`. Its incumbent hit rate is **0/1 entry
   calls** — it never improves the incumbent on nvs06.
2. **The same source dominates nvs08 (779 solves, 0 % hit) and ex1224 (217, 0 %
   hit).** For **fac2 (179 solves) and m3 (131 solves)** the dominant source is
   instead **`rens`** (a nested B&B sub-solve) — a *different* structure, already
   partly budgeted by F1/F4, and NOT the VOLUME-1 lever (see §3).
3. **`integer_local_search` is dead weight even on its own cited class.** Its
   docstring justifies itself on nvs23 ("-287 → -1125"); on the current build
   nvs23's incumbent (-1124.8) also comes from the root multistart and ILS's hit
   rate there is **0 %** too. The value it once delivered is now delivered
   upstream by the root multistart.
4. **The cut is viable, general, and certificate-safe.** A default-OFF cap
   (`DISCOPT_ILS_SOLVE_CAP=k` ⇒ ≤ `k × n_int` sub-NLPs per descent) keyed on the
   integer dimension (not any instance name) cuts nvs06 888→31 and nvs08 779→27
   solves, wall **3.5 s→1.4 s (nvs06)** and **2.9 s→0.9 s (nvs08)**, with
   objective, bound, and node count identical. Across all 18 panel instances
   that fire ILS: **0 objective changes, 0 lost incumbents, 0 slowdowns > 10 %.**

---

## 1. Source-attribution table (where the solves come from)

Per-instance, `time_limit=60`. "solves" = POUNCE NLP solves attributed to that
source; "wall" = summed POUNCE solve wall for that source. Total wall is the
whole `solve()` wall.

### nvs06 — 911 solves, wall 4.5 s (5 nodes, optimal 1.7703125)

| source | solves | wall (s) | IPM iters |
|---|---:|---:|---:|
| **integer_local_search** (`_objective_improve`) | **888** | **2.29** | 8841 |
| integer_box_search | 15 | 0.05 | 158 |
| `_solve_node_nlp_pounce` (per-node engine) | 5 | 0.04 | 157 |
| feasibility_pump | 2 | 0.05 | 18 |
| subnlp (direct) | 1 | 0.00 | 8 |

### nvs08 — 828 solves, wall 4.3 s (57 nodes, optimal 23.4497)

| source | solves | wall (s) |
|---|---:|---:|
| **integer_local_search** | **779** | **2.16** |
| integer_box_search | 24 | 0.16 |
| `_solve_node_nlp_pounce` | 18 | 0.06 |
| feasibility_pump | 2 | 0.05 |
| others (subnlp, fractional_diving) | 5 | 0.02 |

### ex1224 — 254 solves, wall 2.2 s (53 nodes, optimal -0.94347)

| source | solves | wall (s) |
|---|---:|---:|
| **integer_local_search** | **217** | **1.13** |
| `_solve_node_nlp_pounce` | 23 | 0.09 |
| feasibility_pump | 6 | 0.08 |
| fractional_diving | 5 | 0.03 |
| others | 3 | 0.01 |

### fac2 — 198 solves, wall 5.8 s (69 nodes, optimal 3.3184e8)

| source | solves | wall (s) |
|---|---:|---:|
| **rens** (nested B&B sub-solve) | **179** | **1.87** |
| rins | 17 | 0.26 |
| `_solve_node_nlp_pounce` | 2 | 0.29 |

### m3 — 258 solves, wall 3.4 s (61 nodes, optimal 37.8)

| source | solves | wall (s) |
|---|---:|---:|
| **rens** | **131** | **1.26** |
| `_solve_node_nlp_pounce` | 102 | 1.12 |
| rins | 25 | 0.18 |

**The dominant 1–2 sources:** `integer_local_search` owns the 911 (nvs06), the
828 (nvs08), and the 254 (ex1224). `rens` owns fac2/m3. `_objective_improve`
(inside ILS) is the single largest NLP-solve source across the panel.

---

## 2. Per-source incumbent hit rate (the necessity question)

"entry calls" = number of times that source function was *entered*; "hits" =
number of those that returned a feasible candidate strictly better than the
best-so-far incumbent. Measured on the same runs.

| instance | source | entry calls | hits | hit rate |
|---|---|---:|---:|---:|
| nvs06 | `_solve_root_node_multistart` | 1 | **1** | **1.00** |
| nvs06 | **integer_local_search** | 1 | **0** | **0.00** |
| nvs06 | subnlp (all inner) | 903 | 0 | 0.00 |
| nvs06 | feasibility_pump | 2 | 0 | 0.00 |
| nvs08 | `_solve_root_node_multistart` | 1 | **1** | **1.00** |
| nvs08 | **integer_local_search** | 1 | **0** | **0.00** |
| ex1224 | `_solve_root_node_multistart` | 1 | **1** | **1.00** |
| ex1224 | **integer_local_search** | 1 | **0** | **0.00** |
| fac2 | `_solve_root_node_multistart` | 2 | 1 | 0.50 |
| fac2 | rens / rins / diving / local_branching | — | 0 | 0.00 |
| m3 | `_solve_root_node_multistart` | 2 | 1 | 0.50 |
| nvs23 | `_solve_root_node_multistart` | 1 | **1** | **1.00** |
| nvs23 | **integer_local_search** | 1 | **0** | **0.00** |

**The crux:** on every instance the incumbent comes from
`_solve_root_node_multistart` (start #1). `integer_local_search` — the source of
the hundreds of solves — has a **0 % hit rate on every panel instance AND on its
own docstring-cited nvs23**. The 888 nvs06 solves are load-*less*: they do the
work and find nothing. This is exactly the "500 solves at a 1 % hit rate"
lever the task asked for, here at **0 %**.

The root multistart, by contrast, has a 100 % (or 50 %, when it runs twice) hit
rate — it is the *load-bearing* source and is NOT touched.

---

## 3. Why `rens`/`_solve_node_nlp_pounce` are not the VOLUME-1 lever

fac2/m3's dominant source is `rens`, whose 179/131 solves are the **nested
`_solve_nlp_bb` B&B** over the RENS rounding neighbourhood — a single
`sub_solver(model)` call that itself branches. That is a structurally different
volume (real sub-problem search, and its incumbent can matter) and it is already
under F1's LNS-enumeration budget and F4's root-heuristic budget. Cutting it is
a re-scoped, higher-risk change (it can lose incumbents on the combinatorial
class F1/F4 were built for) and is out of scope here. `_solve_node_nlp_pounce`
is the per-node dual-bound engine — never cuttable without a certificate risk.
VOLUME-1's clean, safe, high-yield lever is `integer_local_search`.

---

## 4. The cut — `DISCOPT_ILS_SOLVE_CAP` (default OFF)

**Mechanism.** `integer_local_search._objective_improve` runs a first-improvement
coordinate descent over `int_idx × {±1, ±2}`, calling `subnlp` (a full
continuous-repair NLP) at each candidate, and re-sweeps until it hits a wall
deadline (~9 s at `time_limit=60`). The genuine value of a first-improvement
descent lands in the first sweep or two; the rest is the 0 %-hit plateau. The
cut caps the number of sub-NLP solves a single descent may issue to
`k × max(1, n_int)` where `k = DISCOPT_ILS_SOLVE_CAP` (default `0` ⇒ unlimited =
today's behavior). Keyed on the integer dimension, not any instance name
(CLAUDE.md §2). Sound: capping this descent only ever *weakens* the incumbent it
might find (all its points are sub-NLP-verified and re-verified by
`inject_incumbent`), and it never touches the dual bound or the certificate
(heuristic-policy regime, CLAUDE.md §5).

**The cut fires (verified via the source counter — the R2-didn't-fire lesson):**

| instance | `DISCOPT_ILS_SOLVE_CAP` | ILS solves | total NLP | wall (s) | objective | nodes |
|---|---:|---:|---:|---:|---|---:|
| nvs06 | 0 (default) | 888 | 911 | 3.54 | 1.7703125002 | 5 |
| nvs06 | **2** | **31** | **54** | **1.39** | 1.7703125002 | 5 |
| nvs08 | 0 (default) | 779 | 828 | 2.89 | 23.449727353 | 57 |
| nvs08 | **2** | **27** | **76** | **0.93** | 23.449727353 | 57 |

`DISCOPT_ILS_SOLVE_CAP=0` reproduces the exact 888 solves of the unset default
(byte-identical), confirming the flag is inert until armed.

**Incumbent quality + certificate unchanged on the full ILS-firing panel.**
A/B (cap 0 vs 2, `time_limit=60`) on all 18 panel instances that issue > 20 ILS
solves — every objective identical, no incumbent lost, no slowdown:

| instance | base obj | cut obj | base wall | cut wall | base/cut nodes |
|---|---|---|---:|---:|---|
| nvs08 | 23.449727353 | 23.449727353 | 2.89 | **0.93** | 57 / 57 |
| nvs06 | 1.7703125002 | 1.7703125002 | 3.54 | **1.39** | 5 / 5 |
| nvs01 | 12.469668822 | 12.469668822 | 5.87 | **2.82** | 17 / 17 |
| st_e38 | 7197.7271168 | 7197.7271168 | 1.25 | **0.56** | 3 / 3 |
| nvs09 | -43.13433692 | -43.13433692 | 60.03 | 60.11 | 221 / 221 |
| nvs05 | 5.4709341264 | 5.4709341264 | 60.11 | 60.17 | 815 / 845 |
| tspn05 | 191.25520768 | 191.25520768 | 18.92 | 19.30 | 39 / 39 |
| st_e29 | -0.943470491 | -0.943470491 | 2.11 | 2.05 | 53 / 53 |
| ex1224 | -0.943470491 | -0.943470491 | 2.06 | 2.10 | 53 / 53 |
| tspn12 | 262.64739525 | 262.64739525 | 13.37 | 13.49 | 1 / 1 |
| tspn08 | 290.56685375 | 290.56685375 | 16.64 | 16.71 | 1 / 1 |
| ex1225 | 30.999999952 | 30.999999952 | 1.16 | 1.16 | 7 / 7 |
| tspn10 | 225.12607140 | 225.12607140 | 15.01 | 14.43 | 1 / 1 |
| gkocis | -1.923098780 | -1.923098780 | 0.70 | 0.69 | 5 / 5 |
| ex1221 | 7.6671800711 | 7.6671800711 | 0.56 | 0.56 | 5 / 5 |
| oaer | -1.923098304 | -1.923098304 | 0.65 | 0.65 | 3 / 3 |
| ex1226 | -17.00000000 | -17.00000000 | 0.55 | 0.54 | 5 / 5 |
| tanksize | 1.2686437480 | 1.2686437480 | 60.11 | 60.10 | 557 / 557 |

Wall wins concentrate exactly where ILS was heavy (nvs06 2.5×, nvs08 3.1×,
nvs01 2.1×, st_e38 2.2×). The time-limited instances (nvs05/nvs09/tanksize/tspn)
are unchanged — ILS was not their bottleneck — and none regressed or lost its
incumbent. nvs05's node count rose 815→845 (freed heuristic time buys more
nodes) with the *same* objective: incumbent quality unchanged, as required.

`ex1224`'s ILS solves (217) are spread across ~24 short restarts (n_int=8), so
the per-descent cap of `2×8` does not bind there — the pathology (a *single*
long descent re-sweeping to its wall deadline) is what the cap targets, and it
binds precisely on the n_int-small, many-sweep instances (nvs06/nvs08) where the
volume actually blows up. This is the correct, general behavior.

---

## 5. Kill criterion — NOT fired

The kill criterion was: *if the volume is genuinely load-bearing (each source's
solves have a material hit rate, cutting them loses incumbents or slows
certification), the volume is irreducible and the gap is an algorithmic
difference vs BARON.* It did **not** fire: the dominant source
(`integer_local_search`) has a **0 % incumbent hit rate on every panel instance
and on its own cited class**, and capping it loses **no** incumbent and slows
**nothing**. The volume is reducible.

(The residual gap vs BARON on the time-limited instances — nvs05, nvs09,
tanksize, tspn — is genuinely a *bounding* problem, not a solve-volume one:
their wall is unchanged by the cut because their bottleneck is the weak root
relaxation / per-node bound work, i.e. the branch-and-reduce roadmap
(cert-gap-plan), fewer-nodes not fewer-solves-per-node. That half of the panel
points back to relaxation work, exactly as the kill-criterion's re-scope note
anticipated — but the *solve-volume* half (nvs06/nvs08/nvs01/st_e38) is real,
cuttable, and cut here.)

---

## 6. Verification run

- `pytest -m smoke` (python/tests): **617 passed, 14 skipped**.
- `pytest -m slow python/tests/test_adversarial_recent_fixes.py`: **10 passed**.
- `ruff check` + `ruff format --check` (v0.14.6): clean on the changed
  production file and the script.
- pre-commit `mypy` (whole `python/discopt/`): **Passed**.
- No Rust touched (`cargo test` not required).
- `DISCOPT_ILS_SOLVE_CAP=0` == unset default (888 solves, byte-identical) —
  flag is default-OFF and inert until armed.

---

## 7. Artifacts

- `discopt_benchmarks/scripts/nlp_solve_volume.py` — the source-attribution +
  hit-rate profiler (monkeypatch of `nlp_pounce.solve_nlp` with stack-walk
  bucketing and per-source incumbent-improvement hit rate). Reusable on any
  `.nl` instance.
- `python/discopt/_jax/primal_heuristics.py` — the `DISCOPT_ILS_SOLVE_CAP`
  (default-OFF) sub-NLP solve cap inside `integer_local_search._objective_improve`.
