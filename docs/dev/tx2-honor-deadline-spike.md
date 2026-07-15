# TX2 — honor the time limit: entry experiment → STOP (land nothing)

Status: **STOP / NOT LANDED** (2026-07-14). Item: `docs/dev/tenx-plan.md` §3 TX2.
Companion: `docs/dev/sota-proof-plan.md` §3 PF5 (the falsified blunt fix this item
was meant to replace surgically) and its Open follow-up #1 (OBBT-probe budget) and
#4 (contvar simplex conditioning).

## TL;DR

The entry experiment (mandatory, run before any code) measured **which phase
ignores the deadline** on the four `binding_constraint=overrun` instances. The
answer falsifies TX2's premise: the overruns are **not** runaway heuristics or
extra node launches past a soft deadline (the node loop already gates those —
`solver.py:6471`, `:6592`). Every overrun is a **single in-progress native solve
(Rust MILP/LP simplex, or POUNCE NLP) ignoring the Python-supplied budget it was
handed**, and for the bound-producing subset that solve *is the dual bound*.
Truncating it changes the bound — the exact PF5 regression, which I reproduced on
casctanks. Per the item's own kill criterion ("the PF5 trap: any bound looser on
the panel") and CLAUDE.md §1, **land nothing**.

## Method

`scratchpad/tx2_phase_probe.py` / `tx2_callsite.py` wrap each solve phase
(`run_root_presolve`, `obbt_tighten_root`, `MccormickLPRelaxer.solve_at_node`,
`_solve_node_nlp*`, the primal heuristics) with wall timers relative to solve
start, and record the `solver.py` caller line + which phase's interval straddles
the 30 s deadline. `tx2_obbt_sens.py` / `tx2_probe_skip.py` test bound-neutrality
of truncating the two candidate levers. jobs=1, `time_limit=30`, fresh process
per instance. Baseline (flag-off) walls/bounds from `scratchpad/f2p_full_eval.json`.

## Phase → overrun attribution (the entry experiment)

All four instances process **nodes=1**: the overrun is inside root / first-node
processing, not the between-node loop (whose deadline check only fires between
iterations, of which there is one).

| instance | baseline wall | overrunning phase | call site | budget handed → actual | is it the bound source? |
|---|---|---|---|---|---|
| **contvar** | 61 s | discarded root **probe** `solve_at_node` | `solver.py:5265` | `time_limit=3.0s` → **21 s** | no (result only sets `_probe_useful`); the reported bound 171244.811 comes from the #138 root fallback `solve_at_node` @ `solver.py:2631` (+8 s, also overruns its ≤3 s budget) |
| **casctanks** | 49–78 s | root **OBBT** `obbt_tighten_root` | `solver.py:4721` | `deadline=+14.7s` → **48 s** | **yes** — one persistent probe LP (`solve_lp_warm_csc_py`, no wall deadline, 100k-iter cap) spins ~33 s; its tightening feeds the root bound −99.09 |
| **bchoco08** | 51 s | node-LP `solve_at_node` | node loop (`:6482`) | `_node_remaining` → **12 s over** | yes (node dual bound) |
| **heatexch_gen3** | 45 s | POUNCE **batch node-NLP** (`_solve_batch_pounce`) | node loop | `opts["max_wall_time"]` clamp → overruns by ~50–77 s | no (nonconvex node NLP is reset to −inf, `:6344` — a primal heuristic), but it is native-uninterruptible (first-time XLA Hessian compile, F4) |

Common root cause: the **Python phase scheduler is already correct** — it computes
and passes a per-phase budget everywhere (node loop `:6471`/`:6485`, root probe
`:5265`, root OBBT `:4726`, node-NLP clamp `:6605`; `solve_at_node` even converts
`time_limit` to a deadline and threads `_remaining()` into every internal
`milp.solve`, `mccormick_lp.py:1223`). The budgets are **ignored by the native
solver**: the Rust MILP/LP simplex (`milp.solve`, `_PersistentProbeLP.solve` →
`solve_lp_warm_csc_py`) takes only an iteration cap, no wall deadline; POUNCE
overruns its `max_wall_time` clamp (F4). This is the identical seam PF5 named
("warm-LP solve bindings hardcode `deadline: None`").

## Why the surgical fix is not cleanly separable → STOP

To honor these deadlines the **native** solve must truncate mid-solve. The same
`milp.solve` / Rust simplex is shared by (a) the discarded root probe [safe to
truncate], (b) the dual-bound node LP [unsafe], (c) the OBBT probe LP [unsafe],
and (d) the #138 fallback bound [unsafe]. There is no Python-layer seam that
enforces a deadline on only (a) without also enforcing it on (b)–(d); that would
require threading a per-call "truncatable" flag from each call site down into the
Rust simplex — deep native plumbing with a direct PF5 leak path.

**Reproduced the PF5 regression** (`tx2_obbt_sens.py`, casctanks, OBBT probe
iter-cap 5000 to force truncation): bound **−99.09 → +5.70**, wall **78 → 105 s**.
Truncating the OBBT probe changes the bound chaotically (and made wall *worse*).
This is exactly PF5's failure mode (m3 37.8→19.36, nvs11 −431→−2781) on a
different instance. The OBBT sub-budget *is* separate from the outer node loop
(it has its own deadline, `:4726`); the problem is not a leak of the outer
deadline into OBBT — it is that OBBT's own probe LP is native-uninterruptible, so
"honoring" its budget means truncating a bound-producing solve.

`tx2_probe_skip.py` confirmed contvar's probe @5265 is *bound-neutral* (final
bound is the #138 fallback's 171244.81128, not the probe's) — so that one phase
is separable. But (1) capping it still needs native truncation of `milp.solve`,
and (2) it is a single slow ill-conditioned root LP = PF5 open item #4 (contvar
simplex conditioning, "scaling/restart, not a bumped cap"), not a deadline miss.
Even a perfect cap fixes only contvar; the mechanism gate needs all four, and
casctanks/bchoco08/heatexch_gen3 remain native-truncation-bound. A flag that
improves 1/4 and cannot pass its own gate is a dead flag (CLAUDE.md §3).

## Verdict

**STOP — land nothing.** TX2's premise (unhonored *soft* deadline → runaway
heuristics/extra launches) does not match these instances: their overrun is slow
**bound-producing native math** that cannot be truncated without loosening the
bound (measured). The real fixes are the two PF5 follow-ups this item was carved
away from — **#1 OBBT-probe budgeting** (casctanks) and **#4 contvar simplex
conditioning** (contvar) — plus a native `time_limit`/`max_wall_time` deadline in
the Rust simplex and POUNCE (out of TX2's Python scope, and unsafe as a blanket).
Honoring the deadline the only available way (blanket native truncation)
regresses bounds; a loud stop beats a silent PF5 (CLAUDE.md §1).

Scripts: `scratchpad/tx2_phase_probe.py`, `tx2_callsite.py`, `tx2_obbt_sens.py`,
`tx2_probe_skip.py`.
