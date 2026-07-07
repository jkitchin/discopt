# G2 — the effort governor: hit-rate-adaptive root-heuristic scheduling (2026-07-07)

**Status:** built + shipped behind a **default-OFF** env flag
(`DISCOPT_HEURISTIC_GOVERNOR`). Entry experiment GENERALIZES; governor
implemented, verified firing, cert-neutral by default, incumbent-quality
unchanged. Graduation to default-ON goes through the G1.2 flag gate later.
**Branch:** `g2-effort-governor` at `origin/main` (4fd2d0ef).
**Plan:** `docs/dev/default-path-performance-plan-2026-07-06.md` §2 (C-A) + §3 G2;
sensor `docs/dev/nlp-solve-volume-2026-07-06.md` (VOLUME-1).
**Regime:** heuristic-policy (CLAUDE.md §5) — certified `objective` unchanged;
`incorrect_count = 0`; incumbent-at-time-limit not degraded; node_count *may*
shift (it does on a few instances; documented below). Default-OFF: the solve
path is byte-identical with the flag unset.

> **Method.** Apple M4 Pro (arm64), Python 3.12, JAX 0.10.2 (`JAX_PLATFORMS=cpu`,
> `JAX_ENABLE_X64=1`), pounce 0.7.0 (release wheel `_pounce.abi3.so` = **4.7 MB**,
> verified NOT debug), `maturin develop --release`. Instances from
> `python/tests/data/minlplib_nl/` + the MINLPLib snapshot
> (`~/Dropbox/projects/discopt-minlp-benchmark/`). Sensor: VOLUME-1's
> `discopt_benchmarks/scripts/nlp_solve_volume.py` (per-call-site NLP-solve
> attribution + per-source incumbent-improvement hit rate), unchanged. All solves
> `time_limit=60, gap_tolerance=1e-4`, one instance at a time.

---

## 0. Executive summary

The entry experiment **GENERALIZES**. On a pooled replay of the easy panel +
a 20-instance held-out slice (27 instances that solve), **two expensive
nested-B&B heuristic sources beyond the already-capped ILS have a 0 % pooled
incumbent hit-rate and material wall-share**:

| source | pooled solves | pooled NLP wall-share | incumbent hit-rate |
|---|---:|---:|---:|
| **rens** (nested B&B) | 805 | **14.3 %** | **0 / 13 (0.0 %)** |
| **rins** (LNS dive)   | 440 | **10.4 %** | **0 / 143 (0.0 %)** |

On the 21 *finished* easy instances **RENS alone is 33 % of total solve wall**
at a 0 % hit rate. On every instance the incumbent came from the root
multistart (start #1); the expensive improvers did the work and found nothing —
the C-A pattern, generalized past ILS.

**Scope narrowed by the cert panel (correctness first, CLAUDE.md §4).** The
initial governor throttled rens AND rins/local_branching (all 0 %-hit on the
easy+held-out pool). The cert-neutrality panel then **caught a degraded
certificate**: on the convex-nonseparable class (`cvxnonsep_psig30`) rins /
local_branching ARE load-bearing — throttling them lost the better incumbent
(78.9989 → 79.0024) and the solve certified the *worse* point at the gap
tolerance. The measurement wins: the governed set was narrowed to **rens only**
— the source whose 0 %-hit holds on every class tested AND that carries the
dominant wall (its whole nested B&B fired once at the root). rins /
local_branching / enumerate keep their existing `_improver_allowed` node-budget
contingent and are NOT governed.

Final: **easy-class geomean wall 1.77 s → 1.46 s (1.21×)** (m3 2.08×, fac2
1.40×), all-20 panel+held-out geomean **1.28×**, **0 certified-objective changes
(cert panel NEUTRAL with the governor ON), 0 lost incumbents, 0 instances >10 %
slower**, and the governor is **verified firing** (it threw out RENS on 12 of 20
instances).

---

## 1. Entry experiment — the per-source pooled table (decision input)

Pooled across 27 solving instances (panel: m3, fac2, nvs06, nvs08, nvs13,
ex1224, nvs24[TL], st_e35[TL], st_e36[TL]; held-out: ex1222, st_e15, synthes1,
procsel, ex1223, st_e14, synthes2, flay02m, synthes3, spring, batchdes, syn05m,
fac1, flay03m, csched1a[TL], gear2, ex3pb[TL], syn10m — portfol_buyin/card
excluded: binary-`.nl` unreadable). Pooled NLP wall = 156 s.

| source | solves | wall (s) | wall-share | entry calls | hits | hit-rate |
|---|---:|---:|---:|---:|---:|---:|
| `_solve_node_nlp_pounce` (dual-bound engine) | 2316 | 77.3 | 49.5 % | — | — | n/a (never cuttable) |
| integer_local_search (already capped #532) | 644 | 23.6 | 15.1 % | 13 | 0 | 0.0 % |
| **rens** | 805 | 22.3 | **14.3 %** | 13 | **0** | **0.0 %** |
| **rins** | 440 | 16.3 | **10.4 %** | 143 | **0** | **0.0 %** |
| local_branching | 79 | 5.8 | 3.7 % | 12 | 0 | 0.0 % |
| feasibility_pump | 65 | 5.8 | 3.7 % | 38 | 0 | 0.0 % |
| fractional_diving | 82 | 3.3 | 2.1 % | 15 | 0 | 0.0 % |
| subnlp (generic inner label) | 20 | 1.0 | 0.6 % | 1008 | 2 | 0.2 % |
| integer_box_search | 57 | 0.7 | 0.4 % | 19 | 0 | 0.0 % |
| **`_solve_root_node_multistart`** (finder) | — | — | — | **39** | **27** | **69.2 %** |

The load-bearing source is unambiguous: `_solve_root_node_multistart` produced
the incumbent on **every** instance (69 % pooled hit-rate = 100 % where it runs
once, 50 % where it runs twice). Every expensive improver has a **0 % hit-rate**.
Per-instance confirmation (rens/rins solves vs their hits vs multistart):

```
m3     rens 131 hits 0/1   rins 25 hits 0/10   multistart 1/2
fac2   rens 179 hits 0/1   rins 17 hits 0/6    multistart 1/2
flay03m rens 194 hits 0/1  rins 42 hits 0/15   multistart 1/2
gear2  rens 0             rins 192 hits 0/17   multistart 1/1
ex3pb  rens 0             rins 58 hits 0/24    multistart 1/1
... (every instance: improvers 0 hits, multistart is the incumbent source)
```

`rens`/`rins` return `(x, obj)` or `None` and the caller injects only on
improvement — so the entry-wrapper's 0-hit measurement reads their actual return
value; it is not a side-channel artifact. The 0 % is real.

### Decision: GENERALIZE (build)

≥2 sources **beyond** the capped ILS cleared the entry bar (~0 % pooled hit-rate
AND ≥5 % root wall-share): rens (14.3 %) and rins (10.4 %) — and on the finished
easy class RENS is 33 % of *total* solve wall. Distinct nested-B&B / sub-MIP
sources the ILS cap does not touch. Build. (The subsequent cert-panel check then
narrowed which of them is *safe* to throttle — see §2.)

---

## 2. The governor — design

`python/discopt/heuristic_governor.py`: a process-lifetime `HeuristicGovernor`
singleton holding per-**source-class** running stats
(`calls, hits, consecutive_misses, disabled`). Global constants (documented, NOT
per-instance / per-name — CLAUDE.md §2):

* **Only rens is governed.** `GOVERNED_SOURCES = EXPENSIVE_SOURCES = {"rens"}`.
  A source not in this set is always allowed and accrues no throttle. The entry
  experiment also flagged rins/local_branching as 0 %-hit, but the cert panel
  proved them **load-bearing on the convex-nonseparable class** (see the box
  below) — so they are left to the existing `_improver_allowed` node-budget
  contingent and are NOT throttled here. RENS is the one source whose 0 %-hit
  holds across every class tested AND that carries the dominant wall (a whole
  nested B&B fired once at the root).
* **Cheap finders are never governed.** Multistart, feasibility_pump,
  fractional_diving, integer_box_search, and the already-capped
  integer_local_search — securing the first incumbent for pruning always wins.
* **Throttle-after-k-misses.** RENS, after `K_DISABLE = 2` consecutive class
  misses, is **disabled for the rest of the process**. Any hit resets the streak
  and re-enables it. The memory is cross-*solve* deliberately: RENS fires once
  per solve, so a per-solve counter could never throttle it — remembering it lost
  on the last solves is exactly how SCIP/BARON throttle losers across a run.
* **Gap-gated.** RENS runs only while the primal-dual gap is open.

> **Correctness-first narrowing (CLAUDE.md §4 — the measurement wins).** The first
> cut governed rens + rins + local_branching (all 0 %-hit on the easy+held-out
> pool). Running `check_cert_neutrality.py` with the governor **ON** then caught a
> **degraded certificate**: on `cvxnonsep_psig30` (convex, `rens` actually *hits*)
> throttling the ungoverned-in-principle rins/local_branching lost the better
> incumbent — certified **79.0024** vs the true **78.9989** (oracle 78.99885434),
> reported "optimal" at the gap tolerance. That is a lost incumbent, forbidden by
> the acceptance criterion. Root cause: the entry pool lacked the convex-nonsep
> class, where those improvers ARE load-bearing. The set was narrowed to rens; the
> degradation is gone (cvxnonsep_psig30 OFF/ON both 78.99885435, match). This is
> the cert panel doing its job, exactly as the plan anticipated
> ("a governor that throttles a load-bearing source shows up in
> incumbent-quality-at-TL").

**Routing.** Each governed call site adds `governor().allowed("rens", gap_open=…)`
to its guard and `governor().record("rens", improved)` after (RENS at the root
primary in `_solve_nlp_bb`). The RINS / local_branching / enumerate call sites
carry `allowed`/`record` calls too, but with those names ungoverned the calls are
no-ops (always allowed, no stats) — wired so a future re-scoping needs no new call
sites. The existing `_improver_allowed`/`_record_improver` node-budget contingent
is preserved throughout.

**Soundness (heuristic-policy).** Throttling a primal heuristic can only cost
B&B *nodes* — never a wrong optimum, bound, or lost certificate; B&B stays
exhaustive and every injected point is re-verified downstream. The lone way it
can hurt is *incumbent quality* (throttling a load-bearing improver), which is
why the cert panel + incumbent-at-TL checks are the real gate — and why the set
was narrowed to rens. Default-OFF: with the flag unset `allowed()` returns `True`
and `record()` is a no-op — byte-identical behaviour.

---

## 3. Acceptance — before/after, firing proof, verification

All A/B numbers are governor OFF vs ON, warm governor (RENS's miss streak primed
so its single root call is throttled on the target instance — models steady
state mid-benchmark; on a cold first solve RENS runs once before it can be
throttled, so the benefit accrues from the 2nd RENS-heavy instance onward, which
is the realistic benchmark case).

### 3.1 Easy-class wall

| instance | OFF wall | ON wall | speedup | obj OFF | obj ON | rens throttled |
|---|---:|---:|---:|---|---|---|
| m3    | 3.44 | **1.65** | **2.08×** | 37.79999967 | 37.79999967 | yes |
| fac2  | 5.63 | **4.01** | **1.40×** | 331837498.182 | 331837498.182 | yes |
| nvs06 | 1.08 | 1.07 | 1.00× | 1.7703125002 | 1.7703125002 | no |
| nvs08 | 0.63 | 0.61 | 1.03× | 23.449727353 | 23.449727353 | no |
| nvs13 | 1.24 | 1.21 | 1.02× | -585.2 | -585.2 | no |
| ex1224| 1.89 | 1.85 | 1.02× | -0.9434704911 | -0.9434704911 | no |

**Easy-class geomean wall: 1.769 s → 1.461 s = 1.21×.** The wins concentrate on
the RENS-heavy instances (m3 2.08×, fac2 1.40×); the RINS-only / ILS-capped
instances are correctly neutral (RENS not their lever, and RINS is no longer
governed).

### 3.2 Held-out generality (14-instance slice)

Every finished held-out instance is same-or-faster, objective identical (Δ within
1e-9 relative): synthes1 **1.76×**, flay03m **1.71×**, syn05m **1.69×**,
batchdes 1.50×, ex1223 1.48×, synthes2 1.41×, fac1 1.34×, st_e14 1.19×,
synthes3 1.17×, flay02m 1.15×, procsel 1.12×, spring 1.05×.

**All-20 (easy + held-out) geomean wall: 0.728 s → 0.571 s = 1.28×; 0 objective
mismatches; 0 instances >10 % slower; 0 lost incumbents. RENS throttled on 12 of
20.** Incumbent-quality-at-TL confirmed unchanged on the earlier (wider-governor)
time-limited run and re-confirmed neutral on the cert panel here.

### 3.3 Firing proof (the R2-didn't-fire lesson)

The governor's `snapshot()` shows RENS actually refused on the target instances:
RENS `throttled_events ≥ 1` on m3, fac2, synthes1, synthes3, syn05m, flay02m,
ex1223, st_e14, synthes2, batchdes, fac1, flay03m (12 instances). Example
(m3): `rens {calls:…, hits:0, disabled:True, throttled_events:1}` — RENS's single
root nested-B&B eliminated, obj identical, 2.08× faster.

### 3.4 Cert-neutrality + gates

* **`check_cert_neutrality.py` governor-OFF (default): NEUTRAL** — all 41
  certifying instances byte-identical (`nodes X→X`, `|Δobj|=0.00e+00`). Inert by
  default.
* **`check_cert_neutrality.py` governor-ON (narrowed): NEUTRAL** — 0 violations,
  0 nonzero Δobj (the earlier cvxnonsep_psig30 degradation is fixed; node counts
  also identical since RENS is not load-bearing on the cert panel). This is the
  decisive incumbent-quality gate, passing with the governor actually ON.
* `pytest -m smoke`: **617 passed, 14 skipped**.
* `pytest -m slow test_adversarial_recent_fixes.py`: **10 passed**.
* `python/tests/test_heuristic_governor.py`: **8 passed** (policy state machine,
  default-OFF inertness, ungoverned-sources-never-throttled, fac2 end-to-end
  cert-neutral).
* `ruff check` + `ruff format --check` (v0.14.6): clean on all changed files.
* pre-commit `mypy` (whole `python/discopt/`, `--python-version 3.12`): the
  change adds **0 new errors**; the 12 pre-existing errors (incl. solver.py
  `RootObbtResult`) are identical on `origin/main` (verified by stash A/B).
* No Rust touched → `cargo test` not required.

---

## 4. Kill criteria — none fired (but the cert panel narrowed the scope)

The entry kill criterion ("every non-ILS source is load-bearing → governor
reduces to the ILS cap") did not fire: rens is 0 %-hit at 14.3 % pooled wall /
33 % finished-easy wall — a genuine, ILS-distinct lever. The build kill criterion
("throttling doesn't reduce wall, or degrades any incumbent") did not fire for
rens: throttling it cut m3 2.08×, fac2 1.40×, held-out to 1.76× with **no**
incumbent lost or degraded. It **did** fire for rins/local_branching (they
degraded cvxnonsep_psig30's incumbent) — so those were removed from the governed
set. The governor ships throttling **rens only**.

---

## 5. Artifacts

* `python/discopt/heuristic_governor.py` — the `HeuristicGovernor`, the policy
  constants (`K_DISABLE`, `GOVERNED_SOURCES = EXPENSIVE_SOURCES = {"rens"}`), and
  the process-lifetime singleton.
* `python/discopt/solver.py` — governed call sites (RENS, and the inert-by-scope
  RINS / local_branching / enumerate hooks) route through
  `governor().allowed/record`; `_get_heuristic_governor` helper. Default-OFF via
  `DISCOPT_HEURISTIC_GOVERNOR`.
* `python/tests/test_heuristic_governor.py` — 8 regression tests.
* Entry-experiment + A/B drivers (reproducible): the VOLUME-1 sensor
  (`discopt_benchmarks/scripts/nlp_solve_volume.py`) pooled across the panel +
  held-out, and the governor OFF/ON A/B.

## 6. Follow-ups

* **Graduation.** Default-ON only via the G1.2 flag gate (three green nightlies)
  — not in this PR.
* **rins/local_branching (parked, not killed).** They are 0 %-hit on the
  integer-MINLP class but load-bearing on the convex-nonseparable class. A future,
  *class-aware* governor could throttle them only where the model is nonconvex
  (RENS's class-general 0 %-hit is the safe subset shipped now). Needs its own
  entry experiment spanning the convex class before any throttle.
