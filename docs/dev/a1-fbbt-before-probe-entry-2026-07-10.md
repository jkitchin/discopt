# SOTA-P1 A1 (task #97): does FBBT-before-the-McCormick-LP-root-probe re-engage the strong relaxation on the unbounded-var stall class?

**Date:** 2026-07-10 · **Base:** `main` @ `bb701eea` (worktree `p1a1/fbbt-before-root-probe`) ·
**Measurement-first entry experiment** — verdict KILL, no solver change shipped.

## 0. TL;DR — VERDICT: **KILL** (the lever is INERT on the real solve path)

The A1 hypothesis (building on the F10 sub-finding): on nvs05/tanksize/casctanks the
McCormick-LP **root probe** runs on the raw ±inf box and returns `numerical`
(tanksize) / `optimal`-with-no-safe-bound (casctanks), which nulls `_mc_lp_relaxer`
(`solver.py:5205-5210`) and disables per-node OBBT + `node_reduce` for the whole
solve. The fix under test: run **sound FBBT** (`tighten_root_bounds_with_fbbt` → Rust
`fbbt`) to finitize the box the probe sees, so the probe succeeds, the LP relaxer stays
live, and the reduction machinery engages. Flag `DISCOPT_FBBT_BEFORE_ROOT_PROBE`,
default-OFF, bound-changing regime.

**The measurements KILL it — the flag is fully inert end-to-end on all three
instances, because the premise (F10's isolated raw-box probe failure) does not
reproduce *in-situ* in the real solve:**

| instance | probe box OFF (in-situ) | probe status OFF | probe status ON (FBBT) | `_mc_lp_relaxer` engaged? | certified bound OFF | certified bound ON | root_bound OFF/ON | nodes OFF→ON | status |
|---|---|---|---|---|---|---|---|---|---|
| nvs05 | finite (0 inf) | `optimal` / lb 0.674 → **useful** | `iteration_limit` | LIVE both | 1.3520892806701879 | 1.3520892806701879 (identical) | 0.6740 / 0.6740 | 493→497 (noise) | feasible |
| tanksize | 1 lb-inf, 26 ub-inf | `unbounded` → useless | `skipped_oversize` → **still useless** | (see note) | 0.8680315476476343 | 0.8680315476476343 (identical) | 0.8473 / 0.8473 | 175→177 (noise) | feasible |
| casctanks | finite (0 inf) | `optimal` / lb −90.18 → **useful** | `optimal` / lb −90.18 → useful (unchanged) | LIVE both | 1.3522533305715752 | 1.3522533305715752 (identical) | −90.18 / −90.18 | 25→25 | feasible |

**Certified bound and root_bound are byte-identical ON vs OFF on all three.** node_count
differs only by ±2/±4 timing noise (casctanks exactly 25==25). No engagement change, no
gap closed, no bound moved. Zero effect.

### Why the F10 isolated finding does not reproduce in-situ

The F10 sub-finding (`docs/dev/a-unbounded-entry-2026-07-10.md` §5) measured the probe on
the **pre-reform** model box in isolation (raw box: tanksize `numerical`, casctanks
`optimal`/None). In the *real* solve, two upstream steps run before the probe:

1. **casctanks** — the factorable-reform's pre-reform FBBT (`solver.py:3841-3871`) +
   periodic/domain reductions already **finitize the probe box** (in-situ: n=560, 0 inf
   on either flag). So the probe is already `optimal`/−90.18 (**useful**) OFF; FBBT-
   before-probe is a **no-op**. (The isolated raw-box `optimal`/None was on the un-
   reformed 500-var model; the solve reforms to 560 vars and finitizes first.)

2. **tanksize** — the box the probe sees is *not* fully finitized upstream (in-situ OFF:
   1 lb-inf, 26 ub-inf → probe `unbounded`, useless). FBBT-before-probe *does* finitize
   it (0 inf), **but** the factorable-lifted McCormick LP on that finite box then trips
   the dense-cell densification guard (`_MAX_RELAX_DENSE_CELLS = 1e8`,
   `mccormick_lp.py:737`, Issue #20) → `skipped_oversize` → **still no bound, still
   useless**. FBBT swaps one sound decline (`unbounded`) for another (`skipped_oversize`);
   the relaxer is nulled either way. (In isolation the FBBT-box probe returned
   `optimal`/0.840 because the un-reformed model's lift is small enough to stay under the
   dense cap — the reform is what pushes it over.)

3. **nvs05** — the probe box is finite OFF (n=8, 0 inf) and already `optimal`/0.674
   **useful**; the relaxer is live regardless. FBBT-before-probe only perturbs the probe
   status (`optimal`→`iteration_limit`) with no downstream effect.

So the "root probe on the raw ±inf box disables the relaxer" mechanism the task targets
is **not the binding disabler in the real solve** for this class: casctanks is already
finite (relaxer live, then OBBT size-gated at n=560>100 — unrelated to boundedness), and
tanksize is blocked by the dense-cell guard on the *finite* box, not by unboundedness.

## 1. Method

- Instances: nvs05, tanksize, casctanks from
  `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`; oracle `minlplib.solu`
  (nvs05 5.4709341, tanksize 1.2686438, casctanks 9.1634794).
- Rust ext built in this worktree (`maturin develop --release`, exit 0); import verified
  to resolve to the worktree native `_rust`.
- Flag `DISCOPT_FBBT_BEFORE_ROOT_PROBE` (prototype, since reverted) tightens only the
  *probe box* via `tighten_root_bounds_with_fbbt` (sound Rust FBBT) — it feeds the
  liveness decision only; the per-node relaxer still uses the live tree box, so no
  feasible point can be cut.
- `Model.solve(time_limit=60)` per instance, defaults, one process each, ON and OFF.
  Deterministic metrics (certified bound, root_bound, probe status, node count, status)
  drive the verdict; wall times indicative (shared machine — the P0-FLIP global50 agent
  was co-resident; that is why the verdict rests on bounds/status, not wall/nodes).
- In-situ probe box captured by tapping `MccormickLPRelaxer.solve_at_node`'s first call
  (the root probe) for box finiteness + status; isolated raw-vs-FBBT probe reproduced
  with `scripts/a_unbounded_probe.py`.

## 2. Verdict and falsification record

**KILL.** The A1 hypothesis ("FBBT-finitize the box before the McCormick-LP root probe so
the probe succeeds → `_mc_lp_relaxer` stays live → per-node OBBT + node_reduce engage and
the certified bound climbs") is falsified: the flag is **byte-identical** to OFF on
certified bound and root_bound for all three instances, because (a) casctanks' probe box
is already finitized upstream (no-op), (b) tanksize's FBBT-finitized box hits the dense-
cell oversize guard (`unbounded`→`skipped_oversize`, both useless), and (c) nvs05's probe
is already useful OFF. This is a **stronger** result than F10: FBBT-before-probe is not
merely "engine robustness with no gap closed" — it does not even change *engagement* on
the real path. Consistent with DECOMP-1 (this class is Lever-A / relaxation-strength) and
with F9/F10 (finitizing the vars is not the lever). No code shipped — a flag that changes
nothing is a dead flag (CLAUDE.md §3).

**Re-scope:** the tanksize disabler is the McCormick **dense-cell guard on the reformed
lifted LP**, not unboundedness — a distinct, separately-motivated axis (a *sparse* node-
LP path for large lifts, or a tighter lift), not an A1 wiring change. casctanks is
OBBT-size-gated (n=560 > 100) with a live relaxer, also not a probe-liveness problem. The
binding lever for the class remains **Lever A (per-node relaxation strength)** on the
alphaBB/interval-and-size-gated sub-classes (nvs05, tanksize), per DECOMP-1 §5.

## 3. Reproduction

```bash
# in-situ probe box + full-solve bound, ON vs OFF (prototype flag DISCOPT_FBBT_BEFORE_
# ROOT_PROBE; the flag was reverted on KILL, so ON==OFF now — the byte-identical bounds
# above were captured with the prototype flag live in this worktree before revert).
python scripts/a1_fbbt_before_probe_solve.py off nvs05 60      # and on/tanksize/casctanks
python scripts/a1_fbbt_before_probe_insitu.py tanksize 0       # in-situ probe box finiteness/status
python scripts/a_unbounded_probe.py                            # isolated raw-vs-FBBT probe (F10)
```

Numbers produced on macOS arm64, base `main` @ `bb701eea`, 2026-07-10, isolated worktree.
```
