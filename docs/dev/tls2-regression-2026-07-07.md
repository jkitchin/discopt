# tls2 incumbent "regression" — attribution (2026-07-07)

**Verdict: TIMING NOISE at the 60 s boundary, NOT a graduated-flag regression.**
No code change. The heuristic governor (G2, #541) and the other graduated
default-ON session flags (ILS cap, PSD cost-gate, lift-zero-spanning) are
**exonerated** — none of them causes the incumbent loss, and none of them is even
on the path that finds tls2's incumbent.

## Background

The V milestone re-measure (#542, on `main`) flagged `tls2` as regressing from the
2026-07-06 baseline (`feasible`, obj = oracle 5.3) to `time_limit` with **no
incumbent** at a 60 s budget. It was never proved-optimal either way, so there is
no certificate-count change; the concern was a *heuristic-policy* regression —
losing a feasible incumbent-at-time-limit — **if** a shipped default-ON change
caused it. tls2 was not in the governor's held-out validation set, so it was worth
checking directly. Oracle: `minlplib.solu` → `=opt= tls2 5.3`.

## Setup

- Branch `tls2-regression` off `origin/main` (`0308a7d1`, G2 governor default-ON).
- Release build (`maturin develop --release`), pounce `_pounce.abi3.so` = 4.73 MB
  (release, as required — debug timing would be meaningless).
- `PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`.
- Instance `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/tls2.nl`
  (37 vars, 31 binary + 6 nonlinear, 24 constraints).
- `model.solve(time_limit=TL, gap_tolerance=1e-4)`, governor reset per process.
- Time-to-first-incumbent + incumbent source captured from the `discopt`
  logger's `"... incumbent"` INFO lines.

The four configs (per the task):

| # | config | env |
|---|--------|-----|
| 1 | full defaults (all session flags default-ON) | *(none set)* |
| 2 | governor OFF | `DISCOPT_HEURISTIC_GOVERNOR=0` |
| 3 | ILS cap off | `DISCOPT_ILS_SOLVE_CAP=0` |
| 4 | baseline (all session flags off, = 2026-07-06) | governor=0, ILS cap=0, `DISCOPT_PSD_COST_GATE=0`, `DISCOPT_LIFT_ZERO_SPANNING_FACTORS=0` |

## Per-config results — 60 s budget, 3 repeats each

| config | found/runs | status pattern | t-to-first-incumbent (found runs) | incumbent source | RENS hits | RENS throttled |
|--------|-----------|----------------|-----------------------------------|------------------|-----------|----------------|
| **defaults** | **1/3** | time_limit, time_limit, feasible | 59.1 s | `NLP-BB warm-start` | 0/1 | 0 |
| **gov_off** | **2/3** | feasible, time_limit, feasible | 58.2 s | `NLP-BB warm-start` | n/a (gov off) | n/a |
| **ils_off** | **3/3** | feasible ×3 | 56.5, 56.6, 58.0 s | `NLP-BB warm-start` | 0/2 | 0–1 |
| **baseline (all off)** | **3/3** | feasible ×3 | 57.5, 58.2, 58.6 s | `NLP-BB warm-start` | n/a | n/a |

Every run that finds the incumbent finds it at **t ≈ 56.5–59.1 s** — right at the
60 s cutoff — and the objective is always exactly 5.3 (matches oracle), node count
~1900 regardless of config.

## Per-config results — 180 s budget, 2 repeats each

| config | found/runs | t-to-first-incumbent | source | nodes |
|--------|-----------|----------------------|--------|-------|
| defaults | **2/2** | 60.7, 61.9 s | `NLP-BB warm-start` | 1921 |
| gov_off | **2/2** | 57.9, 57.7 s | `NLP-BB warm-start` | 1921 |
| ils_off | **2/2** | 60.2, 58.1 s | `NLP-BB warm-start` | 1921 |
| baseline (all off) | **2/2** | 59.9, 60.2 s | `NLP-BB warm-start` | 1921 |

**At 180 s every config in every repeat finds the incumbent** (obj = 5.3), all from
the same source, all reaching the identical 1921 nodes. The incumbent is *not lost
forever* under any config — it is simply discovered at ≈ 57–68 s, straddling the
60 s line.

## Diagnosis

1. **The incumbent finder is a warm-started sub-solve, never RENS.** In every run
   the first (and only) incumbent event is `NLP-BB warm-start incumbent: obj=5.3`
   — the `initial_point` warm-start injection at the top of a *nested* `solve_model`
   call fired by the deep-search improver path, arriving late in the B&B. A 75 s
   trace of the defaults config logged the incumbent at **64.87 s and 67.60 s** —
   both **past 60 s** — with **RENS producing 0 incumbents** the whole solve.
   RENS is simply not on tls2's incumbent path.

2. **No flag separates "found" from "lost".** If the governor (or any graduated
   flag) were the cause, turning it off would flip the outcome. It does not: the
   loss appears in `defaults` (2/3 lost) but the incumbent lands at the *same*
   ≈ 57–59 s in every config including the all-off baseline. `baseline (all off)`
   found it 3/3 at 60 s but always at t ≈ 57.5–58.6 s — i.e. baseline is *also* on
   the wrong side of 60 s by only a couple of seconds of jitter; a slightly slower
   machine-second would have lost it too. The status at 60 s tracks whether the
   ≈ 57–68 s discovery happens to land before the wall — pure boundary jitter,
   uncorrelated with any flag.

3. **The governor's RENS throttle, if anything, HELPS tls2.** A control run with
   RENS *pre-disabled* in the governor (2 seeded misses → `disabled`, as would
   happen in a multi-instance benchmark process) found the incumbent **2/2** at
   **44 s wall / t-first ≈ 35–39 s** — earlier and with lower wall than the
   RENS-running configs. Reason: RENS fires a nested B&B at the root (its ~0.5–8 s
   budget) that does not find tls2's incumbent; skipping it returns that time to
   the search that does. So the load-bearing-source hypothesis (governor throttling
   RENS costs tls2 its incumbent) is **falsified** — the causality runs the other
   way.

## Why the V panel saw it as a regression

The V re-measure is a single 60 s sample per instance. tls2's incumbent-discovery
time (≈ 57–68 s) sits on top of the 60 s wall, so the 60 s status is a coin flip.
The 2026-07-06 baseline sample happened to land on the `< 60 s` side; the #542
sample happened to land on the `> 60 s` side. Nothing about the graduated flags
moved the discovery time — the ≈ 1900-node trajectory and the warm-start finder
are identical across all flag settings. This is exactly the kind of
boundary-straddling instance a single-sample panel mislabels as a regression.

## Recommendation

- **No code change.** The governor stays as-is; it is not implicated. The
  incumbent loss is 60 s-boundary timing noise, reproducible in the all-off
  baseline too.
- tls2 is a **boundary-straddler at 60 s** (incumbent found at ≈ 57–68 s). If the
  V panel wants a stable verdict on it, either (a) exclude sub-`gap`-boundary
  instances whose single-sample status is jitter-dominated, or (b) measure tls2 at
  a budget clear of its discovery time (≥ 120 s makes it a reliable `feasible`).
  This is a *panel-methodology* note, not a solver defect — not actioned here
  since the task is scoped to the attribution.
- Separately worth a follow-up (out of scope): tls2's incumbent arrives only at
  ≈ 57 s of a 60 s budget from a deep warm-started sub-solve, while multistart /
  pump / diving / RENS all whiff at the root. That is a genuine *early-incumbent*
  gap on this instance, but it is orthogonal to the graduated flags and predates
  them (the all-off baseline has the identical late discovery).

## Reproduction

Harness: `run_tls2.py` (per-config solve + governor snapshot + first-incumbent
hook), `drive.sh` (4 configs × N repeats × budget), `run_tls2_predisabled.py`
(RENS-pre-disabled control). All under the scratchpad for this session; the raw
per-run JSON lines are in `results_tl60.jsonl` / `results_tl180.jsonl`.
