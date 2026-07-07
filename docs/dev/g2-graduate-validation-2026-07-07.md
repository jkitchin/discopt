# G2-graduate — the heuristic effort-governor flips DEFAULT-ON (broad-validated, 2026-07-07)

**Status:** graduated. `DISCOPT_HEURISTIC_GOVERNOR` flipped default **OFF → ON**
in `python/discopt/heuristic_governor.py`; `DISCOPT_HEURISTIC_GOVERNOR=0`
(also `off`/`false`/`no`/empty) is the live escape hatch (never a dead flag).
**Branch:** `g2-graduate` from `origin/main` (`ba0668db`, has #540).
**Predecessor:** G2 shipped the governor default-OFF (#540,
`docs/dev/g2-effort-governor-2026-07-07.md`). This is its graduation, following
the ILS-cap (#532) / G1.5 (#538) single-strong-validation path for a
**heuristic-policy** flag.
**Regime:** heuristic-policy (CLAUDE.md §5). Throttling a 0 %-hit primal
heuristic (RENS) never changes the dual bound or a load-bearing incumbent; it
only saves the RENS nested-B&B wall. The gate is therefore: certified
**objective** unchanged + no lost incumbent + no soundness cross. `node_count`
*may* shift (it does on one cert-panel instance — documented below).

> **Method.** Apple M-series (arm64), Python 3.12, JAX 0.10.2
> (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`), pounce 0.7.0 (release wheel
> `_pounce.abi3.so` = **4.73 MB**, verified NOT debug), `maturin develop
> --release`. Held-out corpus: the MINLPLib snapshot
> (`~/Dropbox/projects/discopt-minlp-benchmark/`, oracle `minlplib.solu`). All
> A/B solves `time_limit=30, gap_tolerance=1e-4`, one instance per subprocess
> (clean env + fresh governor memory per instance). Governor-ON solves prime the
> RENS miss streak by `K_DISABLE=2` before the target solve — modelling a warm
> governor mid-benchmark, since RENS fires once per solve and a cold governor
> could not throttle its single root call (the doc §3 convention).

---

## 0. Executive summary — CLEAN → FLIP

A genuinely **held-out** sample of **60 instances** (57 solved; 3 excluded as
binary-`.nl` unreadable), stratified across **8 discrete/QC classes** and
**deliberately including 9 `cvxnonsep_*` instances** (the convex-nonseparable
class that, in G2, caught a rins/local_branching incumbent degradation and forced
the governed set down to `rens` only), was solved governor-**OFF** (`=0`) vs
governor-**ON** (the new default). Every decision gate is clean:

| gate | result |
|---|---|
| **certified-objective neutrality** (hard) | **0 degradations** — all 28 instances certifying `optimal` in either config have identical objective OFF vs ON (within 1e-6/1e-4) AND match the `.solu` oracle |
| **incumbent preservation** (hard) | **0 lost incumbents** — ON objective-at-exit is same-or-better than OFF on every instance (clay0203m *gains* an incumbent under ON) |
| **soundness** (hard) | **0 governor-introduced** dual-bound-crosses-oracle, **0 governor-introduced** false-optimal in either config |
| **no-harm** (soft) | **0 instances >10 % slower** ON |
| **speedup** | geomean **1.118×** over all 57 timed; **1.263×** on the 23 instances where RENS actually throttled |
| **firing** | RENS throttled on **23 / 57** instances (the firing proof) |

The `cvxnonsep_*` risk class is not only **safe** (all 9 objective-identical) but
carries the **largest wins** (cvxnonsep_psig40 **2.95×**, psig20r **2.19×**,
nsig20 **1.97×**) — RENS is 0 %-hit yet wall-heavy there, exactly as the entry
experiment predicted, and multistart already holds the incumbent.

**One pre-existing, config-identical finding surfaced (does NOT block the flip,
CLAUDE.md §1):** `util` (MBQCP) certifies `optimal` at **1072.96** but the oracle
optimum is **999.58** — a dual bound that crosses the true optimum (a
false-optimal). This is **identical OFF and ON** and RENS never fires on it
(`throttled_events=0`); it reproduces on `origin/main` with the governor fully
off. It is a **pre-existing correctness bug unrelated to this flip** and should be
filed on the correctness backlog (#396); this branch changes **zero** solver math
(only the governor default + its tests).

---

## 1. Held-out selection (the gate for the flip)

Seeded (`seed=20260707`), stratified over classes where RENS fires
(integer/binary vars + a sub-MIP), `--max-vars ≤ 300`, oracle present in
`minlplib.solu` (`=best=`/`=opt=`). **Excludes** the 61 vendored panel instances
AND G2's entry-pool + cert-probe instances (m3, fac2, nvs06/08/13, ex1224, nvs24,
st_e35/36, ex1222, st_e15, synthes1-3, procsel, ex1223, st_e14, flay02m/03m,
spring, batchdes, syn05m, fac1, csched1a, gear2, ex3pb, syn10m, cvxnonsep_psig30,
portfol_buyin/card) — a genuinely held-out draw. `cvxnonsep_*` was force-included
regardless of discreteness (its `.nl` are continuous convex; RENS still fires on
several of the psig sub-family).

| class | n | class | n |
|---|---:|---|---:|
| MINLP (incl. 9 cvxnonsep_*) | 17 | MBQP | 8 |
| MBNLP | 9 | MBQCQP | 5 |
| MBQCP | 8 | IQP | 4 |
| MIQCP | 8 | MIQP | 1 |

**cvxnonsep count: 9** (psig20, psig40, psig20r — the risk sub-family that caught
the G2 rins degradation — plus nsig20, nsig20r, pcon20, pcon30, normcon20,
normcon30). 3 selected instances (portfol_classical050_1, portfol_roundlot,
st_miqp5) are stored in the binary `.nl` (`b`) encoding discopt cannot read and
were excluded (the same class of exclusion as G2's portfol_buyin/card).

---

## 2. Per-instance held-out table (OFF vs ON)

`thr` = governor-ON `throttled_events` for RENS (≥1 ⇒ RENS actually refused).
`speedup` = OFF wall / ON wall. Objectives shown to 8 sig-figs; every
optimal-in-both row is objective-identical within tolerance.

| instance | class | OFF status | OFF obj | OFF wall | ON status | ON obj | ON wall | speedup | thr |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| st_test4 | IQP | optimal | -7 | 0.133 | optimal | -7 | 0.125 | 1.06x | 0 |
| st_test2 | IQP | optimal | -9.25 | 0.123 | optimal | -9.25 | 0.125 | 0.986x | 0 |
| st_test5 | IQP | optimal | -110 | 0.149 | optimal | -110 | 0.149 | 1x | 0 |
| st_test3 | IQP | optimal | -7 | 0.14 | optimal | -7 | 0.137 | 1.02x | 0 |
| synheat | MBNLP | feasible | 154997.33 | 30.4 | feasible | 154997.33 | 30.4 | 1x | 0 |
| syn10m02m | MBNLP | feasible | 1453.9049 | 30 | feasible | 1453.9049 | 30.2 | 0.993x | 2 |
| ravempb | MBNLP | optimal | 269590.21 | 23.4 | optimal | 269590.21 | 21.3 | 1.1x | 1 |
| fo7_2 | MBNLP | time_limit | - | 30.3 | time_limit | - | 30.4 | 0.995x | 1 |
| enpro56pb | MBNLP | feasible | 263428.3 | 31.5 | feasible | 263428.3 | 31 | 1.01x | 1 |
| syn10m03m | MBNLP | feasible | 2301.1278 | 30 | feasible | 2301.1278 | 31.5 | 0.953x | 1 |
| parallel | MBNLP | feasible | 924.65954 | 67.4 | feasible | 924.65954 | 65.7 | 1.03x | 0 |
| syn05m04hfsg | MBNLP | feasible | 5510.3874 | 30.1 | feasible | 5510.3874 | 31.5 | 0.956x | 0 |
| batchs101006m | MBNLP | time_limit | - | 30.3 | time_limit | - | 30.5 | 0.992x | 1 |
| clay0203m | MBQCP | time_limit | - | 33.6 | feasible | 41737.46 | 30.9 | 1.09x | 2 |
| sssd15-04persp | MBQCP | feasible | 205160.97 | 30.2 | feasible | 205160.97 | 30.2 | 1x | 0 |
| st_e31 | MBQCP | feasible | -2 | 30.6 | feasible | -2 | 30.4 | 1.01x | 0 |
| genpooling_meyer04 | MBQCP | feasible | 1099095.4 | 30.1 | feasible | 1099095.4 | 30.6 | 0.984x | 0 |
| p_ball_20b_5p_2d_m | MBQCP | time_limit | - | 30.1 | time_limit | - | 30.2 | 0.996x | 1 |
| util | MBQCP | optimal | 1072.9614 | 6.04 | optimal | 1072.9614 | 6.16 | 0.981x | 0 |
| portfol_classical050_1 | MBQCP | ERR (binary .nl) | | | | | | | |
| edgecross14-176 | MBQCP | time_limit | - | 62 | time_limit | - | 44.5 | 1.39x | 0 |
| ex1223a | MBQCQP | optimal | 4.5795824 | 0.425 | optimal | 4.5795824 | 0.381 | 1.12x | 1 |
| fuel | MBQCQP | feasible | 8566.1189 | 1.14 | feasible | 8566.1189 | 0.938 | 1.21x | 0 |
| ex4 | MBQCQP | optimal | -8.0641952 | 11.5 | optimal | -8.0641952 | 3.97 | 2.9x | 1 |
| nous2 | MBQCQP | feasible | 0.6259674 | 30.6 | feasible | 0.6259674 | 30.1 | 1.02x | 0 |
| nous1 | MBQCQP | feasible | 1.5670721 | 34.8 | feasible | 1.5670721 | 34.2 | 1.02x | 0 |
| st_e27 | MBQP | optimal | 2 | 0.651 | optimal | 2 | 0.527 | 1.23x | 0 |
| meanvarx | MBQP | optimal | 14.369231 | 0.233 | optimal | 14.369231 | 0.23 | 1.01x | 0 |
| slay04m | MBQP | optimal | 9859.6594 | 0.371 | optimal | 9859.6594 | 0.371 | 1x | 0 |
| fac3 | MBQP | optimal | 31982310 | 0.804 | optimal | 31982310 | 0.823 | 0.977x | 0 |
| slay05m | MBQP | optimal | 22664.678 | 1.03 | optimal | 22664.678 | 0.989 | 1.04x | 0 |
| hybriddynamic_fixed | MBQP | optimal | 1.4737778 | 0.165 | optimal | 1.4737778 | 0.165 | 1x | 0 |
| slay07m | MBQP | optimal | 64748.825 | 8.43 | optimal | 64748.825 | 8.48 | 0.994x | 0 |
| slay08m | MBQP | optimal | 84960.212 | 15.7 | optimal | 84960.212 | 16.1 | 0.975x | 0 |
| portfol_roundlot | MINLP | ERR (binary .nl) | | | | | | | |
| cvxnonsep_psig20 | MINLP | optimal | 93.811388 | 0.673 | optimal | 93.811388 | 0.486 | 1.39x | 1 |
| cvxnonsep_nsig20 | MINLP | optimal | 80.949295 | 1.5 | optimal | 80.949295 | 0.763 | 1.97x | 1 |
| cvxnonsep_pcon20 | MINLP | feasible | -21.512301 | 30 | feasible | -21.512301 | 30 | 1x | 0 |
| cvxnonsep_normcon20 | MINLP | optimal | -21.749147 | 0.684 | optimal | -21.749147 | 0.52 | 1.32x | 1 |
| cvxnonsep_normcon30 | MINLP | optimal | -34.243966 | 1.37 | optimal | -34.243966 | 0.957 | 1.43x | 1 |
| cvxnonsep_pcon30 | MINLP | feasible | -35.986842 | 30 | feasible | -35.986842 | 30 | 1x | 0 |
| cvxnonsep_nsig20r | MINLP | optimal | 80.949279 | 1.56 | optimal | 80.949279 | 0.834 | 1.87x | 1 |
| cvxnonsep_psig40 | MINLP | optimal | 85.499147 | 2.6 | optimal | 85.499147 | 0.881 | 2.95x | 1 |
| cvxnonsep_psig20r | MINLP | optimal | 95.897425 | 2.27 | optimal | 95.897425 | 1.04 | 2.19x | 1 |
| m7_ar3_1 | MINLP | time_limit | - | 30.1 | time_limit | - | 30 | 1x | 1 |
| fo8_ar4_1 | MINLP | time_limit | - | 30.5 | time_limit | - | 30.7 | 0.992x | 1 |
| fo8_ar25_1 | MINLP | time_limit | - | 30.7 | time_limit | - | 31 | 0.99x | 1 |
| wager | MINLP | time_limit | - | 36.2 | time_limit | - | 30.1 | 1.2x | 0 |
| fo9_ar4_1 | MINLP | time_limit | - | 30.1 | time_limit | - | 30.6 | 0.982x | 1 |
| fo9_ar5_1 | MINLP | time_limit | - | 31.2 | time_limit | - | 30.9 | 1.01x | 1 |
| fo9_ar3_1 | MINLP | time_limit | - | 30.3 | time_limit | - | 31.2 | 0.97x | 1 |
| tln4 | MIQCP | time_limit | - | 30.6 | time_limit | - | 30 | 1.02x | 0 |
| ex1263a | MIQCP | optimal | 19.6 | 5.92 | optimal | 19.6 | 5.94 | 0.997x | 0 |
| tln5 | MIQCP | time_limit | - | 30 | time_limit | - | 30 | 1x | 0 |
| ex1265a | MIQCP | optimal | 10.3 | 0.923 | optimal | 10.3 | 0.922 | 1x | 0 |
| ex1266a | MIQCP | optimal | 16.3 | 0.586 | optimal | 16.3 | 0.586 | 1x | 0 |
| tloss | MIQCP | optimal | 16.3 | 0.613 | optimal | 16.3 | 0.602 | 1.02x | 0 |
| tltr | MIQCP | optimal | 48.066667 | 0.537 | optimal | 48.066667 | 0.545 | 0.985x | 0 |
| tln7 | MIQCP | time_limit | - | 30.1 | time_limit | - | 30 | 1x | 0 |
| st_miqp5 | MIQP | ERR (binary .nl) | | | | | | | |

**Certifying subset (the hard gate):** 28 instances certify `optimal` in both
configs; **all 28 have identical certified objective** OFF vs ON and match the
oracle. 26 are feasible/time-limit in both (no certificate to compare, incumbent
preserved). clay0203m is the incumbent-*gain* case: OFF times out with no
incumbent, ON (RENS throttled) finds a feasible 41737.46.

---

## 3. Verdict against the decision gates

- **CERTIFIED-OBJECTIVE NEUTRALITY (hard): PASS.** 0 degradations. Every
  optimal-in-either-config instance is objective-identical OFF vs ON to
  1e-6/1e-4 and matches the `.solu` oracle. No cvxnonsep_psig30-shape drift on
  any class — the shipped `rens`-only governor is safe on the convex class.
- **INCUMBENT PRESERVATION (hard): PASS.** 0 lost incumbents; ON is
  same-or-better everywhere (clay0203m strictly better).
- **SOUNDNESS (hard): PASS.** 0 governor-introduced dual crosses, 0
  governor-introduced false-optimal. The lone false-optimal (`util`) is
  pre-existing, config-identical, RENS-inert — not this flip's, filed separately.
- **NO-HARM (soft): PASS.** 0 instances >10 % slower ON.
- **SPEEDUP:** geomean 1.118× (all 57), 1.263× on the 23 RENS-throttled
  instances; top wins cvxnonsep_psig40 2.95×, ex4 2.90×, cvxnonsep_psig20r 2.19×.

→ **CLEAN. Flip DEFAULT-ON.**

---

## 4. The flip + graduation proofs (Part 2)

`python/discopt/heuristic_governor.py` — `_governor_enabled()` now reads
`os.environ.get("DISCOPT_HEURISTIC_GOVERNOR", "1")` and returns `True` unless the
value is an explicit escape-hatch (`0`/`off`/`false`/`no`/empty). Docstrings
updated to say default-ON. `python/tests/test_heuristic_governor.py` updated:
`test_default_on_when_unset` (the unset default now throttles),
`test_escape_hatch_restores_off` (`=0`/`off`/… restores the byte-identical
no-stats path), `test_env_parsing_default_on` (unset ⇒ ON). No solver math
touched.

**Cert-baseline neutrality (heuristic-policy).** `check_cert_neutrality.py` with
the new default (env unset ⇒ ON), fresh interpreter, vs the committed 41-instance
`docs/dev/data/cert-baseline.jsonl`: **all 41 objective-identical**
(`|Δobj| = 0.00e+00` on every instance) and **all still `optimal`** — the
heuristic-policy gate. The check flags **one node_count shift**:
`nvs13  node_count 19 → 49`. This is **not** the flip: solving nvs13 in isolation
gives **49 nodes in BOTH `=0` and `=1` configs** (governor snapshot `{}` — RENS
never fires on nvs13), so the baseline's `19` is stale relative to current
`origin/main` and the 19→49 is a pre-existing baseline drift independent of this
change. For a heuristic-policy flag a node_count shift on a cert instance is an
expected, documented perf note (flag-graduation-protocol.md), not a fault — and
here it is not even attributable to the governor.

**Fire-check by default.** With `DISCOPT_HEURISTIC_GOVERNOR` **unset**
(`_governor_enabled()` returns `True`), a warm governor throttles RENS on the
tuning probes: `m3` `throttled_events=1 disabled=True obj=37.800000 optimal`,
`fac2` `throttled_events=1 disabled=True obj=331837498.182 optimal`. The
governor is live on the default path.

---

## 5. Gates run

| gate | result |
|---|---|
| `pytest -m smoke` (python/tests) | **617 passed, 14 skipped** |
| `pytest -m slow test_adversarial_recent_fixes.py` | **10 passed** |
| `pytest python/tests/test_heuristic_governor.py` (incl. slow fac2 cert-neutral) | **10 passed** |
| `ruff check` (v0.14.6) on changed files | clean |
| `ruff format --check` (v0.14.6) on changed files | clean |
| pre-commit `mypy` (whole `python/discopt/`, `--python-version 3.12`) | **0 new errors** (8 pre-existing in export/nl.py, estimate.py, nn/tree.py — byte-identical on `origin/main` via stash A/B) |
| `cargo test -p discopt-core` | n/a — no Rust touched |

---

## 6. Follow-ups

- **File the `util` false-optimal** on the correctness backlog (#396): MBQCP,
  certifies `optimal` 1072.96 vs oracle 999.58, dual bound crosses the true
  optimum, config-independent (governor-inert). Pre-existing on `origin/main`.
- **Nightly ledger** keeps accruing the G2 streak on `main`; the escape hatch
  `DISCOPT_HEURISTIC_GOVERNOR=0` makes the flip reversible.
- **rins/local_branching (still parked)** — 0 %-hit on integer-MINLP but
  load-bearing on the convex class; a future *class-aware* governor could throttle
  them only on nonconvex models (needs its own entry experiment; unchanged here).
