# G3 — Path-coverage map: is the 14× BARON gap a coverage problem? (2026-07-07)

**Branch:** `g3-path-coverage` (from `origin/main`)
**Task:** the G3 entry experiment from
`docs/dev/default-path-performance-plan-2026-07-06.md` §2 (cause C-C) + §3 G3, and
the diagnostic that decides UNIFY-vs-KILL. Thesis under test: discopt has
fragmented solve paths and capabilities are wired into some but not others, so the
14.4× wall gap on jointly-proved instances
(`docs/dev/v-baron-remeasure-2026-07-07.md`) is a *coverage* problem.
**Instrument:** `discopt_benchmarks/scripts/coverage_map.py` (rewritten this task
— see §1) over the 61 vendored panel `.nl` + a 30-instance held-out corpus slice.
**Build:** release `maturin develop --release`; pounce `.so` = 4.73 MB (release).
Run `PYTHONPATH=.../python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, `time_limit=30 s`,
`gap_tolerance=1e-4`.

---

## §1 What the instrument measures (and a correction to the stale one)

The `coverage_map.py` that existed pre-task (from #528) targeted capabilities that
**do not exist on `origin/main`**: `run_root_fixpoint`, `reduce_node`,
`DISCOPT_ROOT_FIXPOINT`, `DISCOPT_NODE_REDUCE`, `DISCOPT_PSD_COST_GATE` — there are
no `root_reduce`/`node_reduce` modules in `python/discopt/_jax/`, and none of those
symbols appear in `solver.py`. Those were parked R2/PSD-cost-gate capabilities from
an earlier campaign that were never merged in that form. Running the stale map
would report every capability as "0 calls / wrap error" — a false dark reading.

The rewritten map instruments the **actual default-path dispatch + capabilities on
`origin/main`**:

**Solve paths** (mutually exclusive; the one that ran), by wrapping the real
dispatch targets in `solver.py`:

| path | dispatch target | when |
|---|---|---|
| `amp` | `discopt.solvers.amp.solve_amp` | `solver="amp"` |
| `convex-fast` | `_solve_continuous` (+ `convex_fast_path`) | pure-continuous, proven convex |
| `lp` / `qp` | `_solve_lp` / `_solve_qp` | classified LP/QP (convex) |
| `miqp-bb` | `_solve_miqp_bb` | proven-convex MIQP |
| `milp-driver` | `_solve_milp_bb` | classified MILP (Rust-simplex B&B) |
| `nlp-bb` | `_solve_nlp_bb` | **proven-convex** MINLP (NLP objective = valid bound) |
| `spatial` | the fall-through McCormick spatial B&B loop inside `solve_model` | everything nonconvex / general MINLP |

**Capabilities** (real fire counters, not the stale set):
`governor_throttled_events` (the #541 `HeuristicGovernor` singleton `.snapshot()` —
a real per-source throttle counter), `rens/rins/dive/pump/ils` fire counts (the
governed heuristics), `alphabb_alpha_calls` (`rigorous_alpha`), `psd_gate_calls`
(`_apply_auto_cut_policy`) + `psd_separate_calls` (`_separate_psd`) +
`psd_enabled`/`rlt_enabled`, and `zerospan_lift_fired` (R4
`DISCOPT_LIFT_ZERO_SPANNING_FACTORS`, detected by wrapping `factorable_reformulate`
and reading `_zero_spanning_factor_auxes` on the reformulated model).

## §2 Static reachability (which capability *can* fire on which path)

Grep of each dispatch function's body for each capability symbol:

| capability | spatial | nlp-bb | milp-driver |
|---|---|---|---|
| McCormick LP relaxer (bound) | YES | — | — |
| PSD / RLT auto-cut gate | YES | — | — |
| alphaBB (`rigorous_alpha`) | YES (fallback-gated) | — | — |
| zero-span lift (R4) | YES | — | — |
| root-budget governor (`_root_heur_nlp_entry_ok`) | YES | — | — |
| G2 governor (rens, #541) | YES | YES | — |
| ILS (integer local search) | YES | — | — |
| fractional diving | YES | YES | — |
| feasibility pump | YES | YES | — |
| OBBT | YES | — | — |

**Reading of the dark cells — they are almost all *domain-appropriate*, not gaps:**

- **nlp-bb** is the *proven-convex* MINLP path: the NLP objective is already a
  valid dual bound, so McCormick/alphaBB/PSD/zero-span/OBBT would be redundant
  relaxation machinery, not a missing capability. ILS is explicitly guarded
  `if not _model_is_convex`, so it is N/A on a convex model by construction. The
  one capability nlp-bb *shares with spatial* is the G2 governor + diving + pump,
  and it **does** reach them.
- **milp-driver** (`_solve_milp_bb`) is a self-contained Rust-simplex MILP B&B with
  its own `_root_dive` primal heuristic and Rust-side incumbent injection. MILP has
  no nonlinear structure, so every McCormick-layer capability is N/A.
- **alphaBB on spatial** is gated behind `_mc_lp_relaxer is None` (only a fallback
  when the LP relaxer is not the bound source) — a *deliberate, A/B-measured*
  decision (`solver.py` ~L4632: "on the corpus that never changes the certified
  result (A/B: 0 regressions)"). Dark-by-design, not dark-by-omission.

## §3 The pooled picture — wall per path + which capabilities fire there

61-panel + 29 held-out (`du-opt` excluded: binary `.nl`, unsupported format — a
corpus artifact, not a solve). 90 instances, 1005 s total wall, `time_limit=30 s`.

| path | n | wall (s) | wall % | proved | notes |
|---|---|---|---|---|---|
| **spatial** | 63 | **806.8** | **80.3 %** | 40/63 | McCormick spatial B&B; the general MINLP path |
| **nlp-bb** | 15 | 164.1 | 16.3 % | 12/15 | proven-convex MINLP (NLP objective = bound) |
| milp-driver | 1 | 30.3 | 3.0 % | 0/1 | Rust-simplex MILP B&B |
| miqp-bb | 10 | 3.6 | 0.4 % | 10/10 | convex MIQP |
| convex-fast | 1 | 0.3 | 0.0 % | 1/1 | single convex NLP |

Panel-only: spatial **86.3 %**, nlp-bb 13.4 %. So **two paths carry ~97 % of all
wall**: spatial and nlp-bb.

**Capability FIRE rate** (instances where the counter > 0 / instances on the path):

| capability | spatial (n=63) | nlp-bb (n=15) | milp-driver | miqp-bb |
|---|---|---|---|---|
| G2 governor — rens *fired* | 0/63¹ | **15/15** | 0/1 | 0/10 |
| G2 governor — rens *throttled* | 0/63 | 0/15 | 0/1 | 0/10 |
| ILS (integer local search) | **47/63** | 0/15² | 0/1 | 0/10 |
| feasibility pump | **47/63** | **15/15** | 0/1 | 0/10 |
| PSD/RLT auto-cut gate (policy) | **52/63** | 0/15³ | 0/1 | 0/10 |
| PSD separator fired | **51/63** | 0/15³ | 0/1 | 0/10 |
| PSD enabled / RLT enabled | 15/63 · 18/63 | 0/15³ | — | — |
| fractional diving | 9/63 | 10/15 | 0/1 | 0/10 |
| rins | 22/63 | 11/15 | 0/1 | 0/10 |
| alphaBB (`rigorous_alpha`) | 3/63⁴ | 0/15³ | 0/1 | 0/10 |
| zero-span lift (R4) | 1/63⁵ | 0/15³ | 0/1 | 0/10 |

¹ spatial uses the McCormick RENS variant differently; its governed heuristics
here are ILS/pump/rins, all firing. ² ILS is guarded `if not _model_is_convex` —
N/A on the convex models nlp-bb serves. ³ McCormick/PSD/alphaBB/zero-span are
relaxation machinery for *nonconvex* bounds; nlp-bb's NLP objective is already a
valid bound, so they are domain-N/A, not omitted. ⁴ alphaBB is fallback-gated
(`_mc_lp_relaxer is None`) by design — fires only where the LP relaxer declines.
⁵ zero-span lift is structure-gated; only st_e36 in this corpus has a
zero-spanning product factor.

**Governor throttled_events across the entire 90-instance corpus = 0.** The #541
effort governor is present and reachable on both material-wall paths, but it never
actually *refuses* a heuristic on this corpus — every rens call it saw (15/15 on
nlp-bb) was allowed. So "the governor is unreachable on some path" is not a real
condition here; where it is reachable, it simply isn't binding.

## §4 The BARON-gap instances — path + what fires

The 4 panel instances BARON proves and discopt does not, plus the QCQP class the
PSD gate targets:

| instance | path | wall | status | capabilities that FIRED |
|---|---|---|---|---|
| nvs05 | spatial | 30.2 | feasible | ILS, PSD-gate, PSD-sep, RLT, diving, pump, rins |
| nvs09 | spatial | 30.2 | feasible | ILS, PSD-gate, PSD-sep, PSD-enabled, pump, rins |
| tanksize | spatial | 30.2 | feasible | ILS, alphaBB, PSD-gate, PSD-sep, diving, pump, rins |
| tls2 | nlp-bb | 31.9 | time_limit | governor(rens), diving, pump |
| nvs17 | spatial | 24.3 | **optimal** | ILS, PSD-gate, PSD-sep, PSD-enabled, pump, rins |
| nvs19 | spatial | 30.0 | feasible | ILS, PSD-gate, PSD-sep, PSD-enabled, pump, rins |
| nvs24 | spatial | 30.7 | feasible | ILS, PSD-gate, PSD-sep, PSD-enabled, pump |
| hda | spatial | 34.1 | time_limit | PSD-gate, PSD-sep |

Every nonconvex BARON-gap instance is on the **spatial** path — the path that
reaches *every* capability — and the relevant ones **already fire**. nvs05/09/19/24
run the full artillery (ILS + PSD + pump + rins + diving) and still return only a
feasible incumbent at the time limit: the **dual bound does not close**. tls2 is on
nlp-bb (convex) with the governor + pump + diving all firing and still times out on
the tree, not on a missing capability. hda (a no-finite-underestimator flowsheet,
per F5/TD-B) fires PSD but can't be bounded at all.

There is **no instance in this corpus where a BARON-gap or slow instance sits on a
path that fails to reach a capability that would demonstrably help its class.**

## §5 DECISION — KILL G3 path-unification; the 14× gap is bounding, not coverage

**The G3 thesis is falsified by the coverage map.** The two paths that carry the
material wall are:

- **spatial (80.3 %)**: reaches and *fires* every capability. Nothing to unify.
- **nlp-bb (16.3 %)**: the dark capabilities there are all either domain-N/A on
  convex models (McCormick/PSD/alphaBB/zero-span, ILS) or already reached (the G2
  governor, diving, pump all fire). No relevant capability is dark.

The remaining paths (milp-driver 3 %, miqp-bb 0.4 %, convex-fast 0 %) are
specialized engines whose McCormick-layer "dark" cells are structurally N/A (a MILP
has no nonlinear relaxation to tighten).

**No gap-path exists.** The G3 kill criterion (§3 of the plan) is met exactly:
*"every material-wall path already reaches the capabilities that help it → the gap
is per-node throughput or bounding strength."* The BARON-gap instances confirm it
directly — they are on the all-capabilities path with the artillery firing, and
still return feasible-not-optimal because the **dual bound doesn't close in the
budget**. That is a bounding-strength / hard-tail problem, which the plan itself
routes to **G4** (bounding strength for the hard tail: lift-before-distribute,
alphaBB-as-primary on no-McCormick-bound structures, bound-only fallback for the
flowsheet class) — explicitly flagged there as research-grade and the long pole.

**Recommendation:** do not build a path-unification seam. It would wire
already-covered paths and move nothing (the §1 lesson of the whole campaign:
capabilities that are reachable-but-inert don't help). Route the effort to **G4
(hard-tail bounding)** and, secondarily, spatial-path **per-node throughput** (the
14.4× is dominated by spatial instances that *do* prove — 40/63 — but slowly).

**One concrete, cheap follow-on surfaced by the map (NOT G3, noted for the
backlog):** the effort governor's `throttled_events = 0` corpus-wide means the
governed heuristics never actually pay their way *off* on this corpus — either the
corpus doesn't exercise the throttle, or the gate is too permissive. Worth a
targeted re-measure on an integer-heavy held-out slice before assuming the #541
governor is doing work on the default path; that is a G2-followup, not G3.

## §6 What this task changed (bound-neutral; measurement only)

- **Rewrote `discopt_benchmarks/scripts/coverage_map.py`** to instrument the actual
  `origin/main` dispatch + capabilities (§1). The prior version targeted
  `run_root_fixpoint`/`reduce_node`/`DISCOPT_PSD_COST_GATE` symbols that do not
  exist on `main` and would have reported a false all-dark map. **No `solver.py` or
  any solver-path change** — this is a pure measurement instrument, so there is no
  bound to be neutral about, and node counts/objectives are untouched by
  construction. The map's own path/fire detection was validated against known
  routes (alan→miqp-bb, chance→convex-fast, nvs06/st_e36→spatial, fac2→nlp-bb) and
  against the V doc's st_e36 zero-span-lift claim (fires: confirmed).

## Appendix — sweep artifact

`docs/dev/data/g3-coverage-30s-2026-07-07.jsonl` (90 rows, one per instance: path,
per-capability fire counts, governor snapshot, wall, node_count, status, objective,
bound). Regenerate one row:

```
PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python discopt_benchmarks/scripts/coverage_map.py <inst>.nl --time-limit 30
```

