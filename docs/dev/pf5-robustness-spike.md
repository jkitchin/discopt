# PF5 SPIKE — incumbents + LP robustness + the time-limit bug (2026-07-14)

Status: **DONE**. Candidate 1 (time-limit bug): **GO — partial fix prototyped**
(the dominant hole closed, one residual phase identified as a separate item).
Candidates 2–4: brief measured verdicts below. Companion:
`docs/dev/sota-proof-plan.md` §2 PF5; premise source: `docs/dev/pf2-pernode-spike.md`
(redirect) + the PF0 baseline (`docs/dev/data/pf-baseline.json`).

Isolation: prototype library edits live in an **isolated worktree** and are **NOT
pushed** (only this doc is committed). The Rust extension was rebuilt with
`maturin build` and the `_rust*.so` copied into the worktree `python/discopt/`;
everything run with `PYTHONPATH=<worktree>/python` so the shared `.venv` / main
tree were never modified (the PF2 isolation pattern). No conflict with PF1: the
prototype does **not** touch `solver.py` (see §5).

---

## 1. Candidate 1 — the time-limit-not-honored bug (PRIORITY): **GO**

### 1.1 Diagnosis — WHICH phase blows the budget (three phases, not one)

Instrumented `solve(time_limit=30)` on the 5 PF0 overrunning instances with a
`faulthandler` stack dump fired past the deadline (jobs=1, serial — contention
inflates walls ~2–2.5× and is non-deterministic per the PF3 caveat, so all
diagnosis walls are serial). The overrun is **not one phase** — it is three, and
only one was cheaply fixable:

| phase | where (stack) | which instance | polled? |
|---|---|---|---|
| **(1) warm-LP solve** | `solve_lp_warm_csc_py` / `solve_lp_warm_py` → simplex, **`deadline: None`** hardcoded (`crates/discopt-python/src/lp_bindings.rs:311/441/525/628`) | bchoco08, contvar (LP-spin) | **NO** |
| **(2) relaxation BUILD** | `build_uniform_relaxation` → convexity cert (`interval_hessian`/`_curvature_cert`, `uniform_relax.py:734/1711`), from the initial build `solver.py:5107` **and** the root fallback `MccormickLPRelaxer(model).solve_at_node` `solver.py:2631` | hda (pre-node, nodes=0), heatexch_gen3 (fallback) | **NO** |
| **(3) root presolve** | `PyModelRepr.presolve` (Rust), one-time 8–10 s (measured in PF2 §2) | all (fixed slice) | **NO** |

**Root cause of the dominant hole (phase 1).** The in-house simplex *already*
polls `SimplexOptions.deadline` every ~256 pivots (`primal.rs:1262`, `dual.rs:390`)
and the MILP driver injects it from `time_limit_s` (`milp_driver.rs:412`,
`lp_bindings.rs:806`). But the **warm-LP** bindings (`solve_lp_warm_py`,
`solve_lp_warm_csc_py`) and the LP batch binding pass `deadline: None` and cap
only on `max_iter=100_000`. Every solve routed through the warm fast path
(`milp_relaxation._solve_lp_warm` → `solve_lp_warm_std` → `solve_lp_warm_csc_py`,
which is the node-LP hot path and the root-fallback `solve_at_node`) runs the LP
**unbounded in wall time**.

Measured overrun of a 3 s budget on the root-relaxation fallback
(`_root_relaxation_lower_bound`, `solver.py:2631`), isolated:

| instance | `solve_at_node(budget=3s)` BEFORE | plain `relax.solve(3s)` BEFORE |
|---|---|---|
| contvar | **24.9 s** (8× over) | 5.7 s |
| bchoco08 | 2.6 s | 5.1 s |

### 1.2 Prototype — thread a wall-clock deadline into the warm LP

The fix mirrors what the MILP driver already does. Three files, backward-compatible
(the new Rust arg defaults to `0.0` = uncapped, so every existing caller is
unchanged):

- **`crates/discopt-python/src/lp_bindings.rs`** — `solve_lp_warm_py` and
  `solve_lp_warm_csc_py` gain `time_limit_s=0.0`; when `> 0`, set
  `deadline = Instant::now() + Duration::from_secs_f64(time_limit_s)`.
- **`python/discopt/solvers/milp_simplex.py`** — `solve_lp_warm_std` gains
  `time_limit`; forwards `time_limit_s`. Key subtlety: `None` ⇒ `0.0` (uncapped,
  preserves old behavior), but a numeric budget that has already elapsed (`<= 0`
  from `deadline - now`) is floored to `1e-6` so it **trips immediately** rather
  than reading as uncapped.
- **`python/discopt/_jax/milp_relaxation.py`** — `_solve_lp_warm` /
  `_solve_lp_warm_equilibrated` accept and forward `time_limit`; `solve()` passes
  its `time_limit` into the warm fast path.

`mccormick_lp.solve_at_node` already builds a per-node `_deadline` and hands each
internal solve `_remaining()` — so once the warm path honors it, the whole node
chain is bounded.

### 1.3 Before/after wall (serial, `solve(time_limit=30)`)

| instance | PF0 baseline (jobs=4) | BEFORE (serial, old `.so`) | AFTER (serial, new `.so`) | phase |
|---|---|---|---|---|
| contvar | killed @ 90 s | 58.9 s | **35.4 s** (tl+5) | LP-spin — **fixed** |
| bchoco08 | killed @ 90 s | 49.2 s | **34.7 s** (tl+5) | LP-spin — **fixed** |
| hda | killed @ 90 s | — | 39.6 s (tl+10) | build-bound (phase 2) |
| casctanks | killed @ 90 s | — | 40.1 s (tl+10) | build/LP mix |
| heatexch_gen3 | killed @ 90 s | 49.7 s | 49.7 s (tl+20) | build-bound (phase 2, fallback) |

Isolated root-fallback after the fix: contvar `solve_at_node` **24.9 → 10.6 s**,
plain `relax.solve` 5.7 → 3.5 s; bchoco08 `solve_at_node` 2.6 → 1.8 s. (contvar's
residual 10.6 s is the `solve_at_node` separation loop's per-solve floor
`_SOLVE_DEADLINE_FLOOR_S` × ~14 rounds — bounded, not the unbounded spin.)

**Reported `wall_time` was itself understated before** (contvar reported 43.2 s at
58.9 s actual); after the fix `reported ≈ actual` (30.9 s at 35.4 s).

### 1.4 Soundness — GREEN, zero tolerance met

- **`bound ≤ objective`**: 0 violations across the 14-instance verification panel
  (proved + feasible + the 5 overrunning).
- **PF0 diff gate** (`--vs pf-baseline.json`): **proofs gained 0 / lost 0, no
  LOOSER, no CROSSED bound**.
- **Bound-neutral where solves finish in time**: all 6 proved instances (alan,
  ex1226, nvs02, m3, st_e36, nvs11) return **byte-identical** status + bound.
- **Direct bound-neutrality unit check**: a warm LP with a *generous* deadline ==
  the *uncapped* solve (identical status/objective); an *expired* deadline returns
  `None`/IterLimit — never a bound. This is exactly the contract: an early
  deadline stops the simplex early and yields **same-or-weaker** (IterLimit ⇒
  caller treats as "no bound" and keeps the parent/cold bound), **never a
  false-tight** bound. The simplex only reports `optimal`+bound at true optimality.
- Bound shifts on nvs09 (−43.7→−43.2, *tighter*), tspn05 (feasible→optimal),
  casctanks (5.7→−90.2, looser but valid, non-crossing) are within the known
  wall-budget non-determinism band (PF3 caveat) and all satisfy `bound ≤ obj`.

### 1.5 Residual (phase 2) — NOT fixed here, and why not band-aided

hda (pre-node) and heatexch_gen3 (fallback) overrun in
`build_uniform_relaxation`'s convexity certification — a monolithic tree-walk
with per-term interval-Hessian eigen work, **not** an iteration loop, so the
"poll a deadline" pattern does not reach it cheaply. A guard was prototyped
(skip the tighter separated rebuild in the post-deadline fallback) and
**reverted**: it made heatexch_gen3 35.4 s **but dropped its only dual bound**
(43888 → None) — trading a valid rigorous certificate for wall time, which
`CLAUDE.md` (certificate before performance) forbids silently. Under the panel's
real 90 s grace, heatexch_gen3 at 49.7 s serial is **not** killed; the baseline
kills were a jobs=4 contention artifact. So phase 2 is left as a **separate
item** (§4).

---

## 2. Candidate 2 — root-presolve one-time cost: **KILL as a cut; fold into a build-deadline item**

PF2 measured `PyModelRepr.presolve` at 8–10 s, one-time. This spike adds that the
**relaxation build** (`build_uniform_relaxation` + convexity certs) is a *larger*
and more instance-variable pre-node cost (hda spends >15 s serial there;
heatexch's LMTD/log envelope build is the fallback's 20 s). Neither is reducible
without losing what it produces (the presolve's reductions; the build's whole
relaxation). **Verdict:** not reducible-and-sound as a *cut* (KILL that framing).
The real robustness lever is a **deadline threaded into `run_root_presolve`
(Rust `time_limit_ms`, already partially present) + `build_uniform_relaxation`**
so a heavy build/presolve yields a partial/loose relaxation at the budget instead
of overrunning — that is the natural PF5 phase-2 follow-up (§4), medium effort
(Rust presolve poll + a build-level deadline check), not a cheap cut.

## 3. Candidate 3 — incumbent latency (CC4): **KILL (for the stuck set)**

The overrunning instances are **build/bound-limited, not incumbent-limited**:
hda/heatexch never leave the root build (nodes 0–3); nvs05's bound is frozen
(PF4 root gap, already established). Time-to-first-incumbent is not on the
critical path for any stuck instance. An earlier feasibility-pump/diving incumbent
would prune nothing that a bound is not already failing to close. **KILL** — no
incumbent-limited class surfaced. (Revisit only if a future panel shows a
moving-bound, node-limited instance with a late first incumbent.)

## 4. Candidate 4 — contvar-class simplex robustness: **partially subsumed; deeper fix KILLed for this spike**

contvar's overrun *was* the un-budgeted root-relaxation LP (phase 1) — the
warm-LP deadline now bounds it (58.9 → 35.4 s). The separate "lost a finite bound
to an iteration budget on a large tightened LP" symptom (contvar's fallback bound
is non-deterministic: 171244.8 in one serial full solve, `None` under contention)
is a genuine **simplex conditioning** issue on the wide/ill-scaled tightened LP,
**not** fixable by a deadline (and a bumped iteration cap is forbidden by
`CLAUDE.md`). A scaling/restart fix is a real but **separate, deeper** item;
**KILL for this spike** (out of scope for the low-risk deadline fix; the deadline
at least stops the spin instead of hanging).

---

## 5. Verdict + sequencing for the full item

**Candidate 1 (warm-LP deadline) is a clean GO.** General (a whole class of warm
LP call sites, not one instance), sound (differential + bound-neutral + PF0 gate
all GREEN), and it closes the dominant unbounded time-limit hole — bchoco08 and
contvar go from 49–59 s to ~tl+5 s serially with no certificate regression.

**Full item = exactly the prototype in this worktree** (3 files, ~55 lines):
1. `crates/discopt-python/src/lp_bindings.rs` — `time_limit_s` arg + `deadline`
   on `solve_lp_warm_py`, `solve_lp_warm_csc_py` (and, for completeness,
   `solve_lp_batch_py`).
2. `python/discopt/solvers/milp_simplex.py::solve_lp_warm_std` — `time_limit`
   plumb, with the `None`-vs-expired distinction.
3. `python/discopt/_jax/milp_relaxation.py` — `_solve_lp_warm` /
   `_solve_lp_warm_equilibrated` / `solve` forward `time_limit`.
   Plus a regression test (a warm LP given an already-expired deadline returns
   non-optimal/None and never a bound; a generous deadline == uncapped).

**PF1 conflict: NONE.** The prototype does **not** modify `solver.py` — the fix
lives entirely in the LP bindings + relaxation warm path. It is disjoint from
PF1's node-loop landing (`solver.py:~6134` / `~6577`), so it can land in any order
relative to PF1. (Note: the `node_nlp_stride` gate PF2 flagged *does* share the
node-loop file with PF1 — but that is the node-NLP throttle, a *different* PF5
lever this spike did not build; the warm-LP deadline is not there.)

**Phase-2 follow-up item (recommended, separate PR):** thread a deadline into the
relaxation **build** (`build_uniform_relaxation` convexity cert loop) and the
Rust **root presolve** so hda/heatexch honor `time_limit` in the pre-node/build
phase too (they overrun tl+10..20 s serially, and > grace under jobs=4
contention). Larger and bound-affecting (a budget-truncated build ⇒ a looser but
valid relaxation) — needs its own differential gate. Do **not** ship the
bound-dropping fallback guard (§1.5).
