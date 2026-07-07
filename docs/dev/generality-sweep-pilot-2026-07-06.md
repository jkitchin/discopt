# Generality-sweep pilot (GEN-1) — do the flagged capabilities help a *class*?

**Date:** 2026-07-06
**Status:** measured (pilot). A FIRST out-of-panel signal + proof the harness
works — **not** the definitive sweep (that is the follow-up at larger N, §6).
**Scope:** the instrument for CLAUDE.md §0.2 ("fix the *class*, not the
instance"). discopt ships three default-OFF, bound-changing capabilities:

- **branch-and-reduce** — the root branch-and-reduce fixpoint
  (`DISCOPT_ROOT_FIXPOINT`) + the per-node cheap reduction (`DISCOPT_NODE_REDUCE`);
- **PSD cost gate** — the cost-aware gate on the per-node PSD (moment) cut
  separation loop (`DISCOPT_PSD_COST_GATE`).

They were tuned on named probes (nvs17/19/23/24, st_e36, …). The V-remeasure
(`uncertified-tail-plan-results-2026-07-06.md` §2; `root-throughput-entry-
2026-07-06.md` §8) showed them **invisible on the 61-instance vendored panel**
(42→42 proved-optimal, 0 net change) — because that panel does not contain their
target structures. That is an *instrument* gap, not evidence of inertness. This
sweep draws a **held-out** sample that DOES carry the target structures and asks,
honestly, per capability: **class win, probe win, or inert on the held-out
sample?**

> **Method.** `discopt_benchmarks/scripts/generality_sweep.py --n 20 --seed 0
> --time-limit 15 --max-vars 300`. Release maturin build (`_pounce.abi3.so`
> 4.73 MB — the debug-build artifact does not apply), Python 3.12, JAX 0.10
> (`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`). Each instance solved **flags-OFF**
> (stock defaults) and **flags-ON** (all three flags set) in isolated
> subprocesses (the `global_opt_baron_vs_discopt.py` worker; flags via `env`).
> Corpus `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`, oracle
> `minlplib.solu` (`=best=` primal + `=bestdual=` dual). Report:
> `reports/generality_sweep_2026-07-06T13-59-41.{json,md}` (the JSON records the
> exact instance list + every per-instance row — reproducible at this seed).

---

## 1. Instance selection (N = 20, held-out, stratified, seeded)

**Held out:** the 61 vendored panel instances (`python/tests/data/minlplib_nl/`)
AND the 10 named tuning probes (nvs17, nvs19, nvs24, nvs23, nvs13, st_e36, nvs09,
nvs05, ex1224, wastewater04m2). **Kept:** corpus `.nl` with a `.solu` oracle and
`vars ≤ 300` (to keep each pilot solve cheap). Eligible pool after exclusions:
**246**. Stratified by MINLPLib probtype (round-robin across strata, cheapest-first
within a stratum), seed 0. The exact 20 (solve order):

```
kall_congruentcircles_c61, ex5_3_3, sonet22v5, graphpart_clique-70, spring,
wastepaper3, ex8_1_4, elf, sonet22v4, hybriddynamic_fixedcc, tln12,
kall_circles_c6b, powerflow0014r, sonet23v4, qap, eg_all_s, autocorr_bern30-15,
ex6_2_6, ringpack_10_1, sonet23v6
```

**Structural prevalence (the honest reach — R4-style corpus scan):**

- **PSD gate** target = a QCQP-family probtype (a quadratic form the moment cut
  can bind on): **14/20**.
- **branch-and-reduce** target = a non-trivial reducible box (≥1 var, ≥1
  constraint): **19/20**.

Structural prevalence is a *reach upper bound* (probtype/size proxy), not a claim
that PSD binds on every QCQP; a non-QCQP is a true negative for the PSD gate.

---

## 2. Soundness (the hard gate) — 0 violations

**0 dual-bound crossings, 0 false-optimal in EITHER config**, checked on all 20
instances against the **`[=bestdual=, =best=]` bracket** (the true optimum lies in
that bracket; a valid dual bound cannot cross the achievable `=best=`, and a
certified-optimal cannot land outside the bracket). **No P0.**

### 2.1 A loose-`=best=` finding (NOT a violation) — and why the bracket matters

`hybriddynamic_fixedcc` returns `optimal @ 1.47352` in **both** configs, which is
**below** MINLPLib's `=best=` primal `1.47378`. A naive check against `=best=`
alone would (wrongly) flag this as a false-optimal P0. It is not: `=bestdual=` is
`1.47350`, and `1.47352 ≥ 1.47350` — discopt's incumbent is **inside the
`[dual, best]` bracket**, i.e. it found a *feasible point better than MINLPLib's
recorded primal* while respecting the valid dual fence. `=best=` is a **loose
MINLPLib primal**, not a discopt error; the certificate invariant holds.

**This drove a harness hardening.** The soundness check now consults *both*
`=best=` and `=bestdual=` and judges against the bracket — a *more* correct check,
not a weakened one (per CLAUDE.md #1: the fence for "impossible" is the dual
bound; beating a loose primal is legitimate). Because this appeared identically in
OFF and ON, it is a pre-existing oracle-data fact, unrelated to the flags.

---

## 3. Per-capability verdict

Each capability's metrics are computed over the subset of scored instances that
**carry its target structure** (structure-carrying ∩ ran in both configs). A
"benefit" = a status upgrade (feasible→optimal), a material node ↓ (≥5%), or a
wall ↓ >5%; a "regression" = a status downgrade, or a wall/node ↑ >5% with no
compensating win.

| capability | structural prevalence (K/N) | scored | benefit-fraction | regression-rate | geomean node (on/off) | geomean wall | pilot verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **branch-and-reduce** (`ROOT_FIXPOINT`+`NODE_REDUCE`) | **19/20** | 13 | **31 %** (4/13) | **8 %** (1/13) | **0.92** | **0.93** | class-leaning |
| **PSD cost gate** (`PSD_COST_GATE`) | **14/20** | 10 | **30 %** (3/10) | **10 %** (1/10) | **0.96** | **0.93** | class-leaning |

**Overall (whole sample, flags bundle ON vs OFF):** scored 14 (6 errored),
benefit-fraction **29 %** (4/14), regression-rate **7 %** (1/14), geomean node
**0.92**, geomean wall **0.93**.

### 3.1 What drives the benefit (the honest breakdown)

Four instances improve with flags-ON; all are structure-carriers:

| instance | type | vars | benefit | attributable to |
|---|---|---:|---|---|
| **spring** | MINLP | 17 | nodes 49→25 (0.51×), wall 9.6→7.8 s | branch-and-reduce (reduce-responsive box; no QCQP) |
| **kall_congruentcircles_c61** | QCP | 16 | nodes 21→15 (0.71×), wall 27.3→15.4 s | PSD gate (QCQP) + reduce |
| **ex5_3_3** | QCQP | 62 | nodes 203→183 (0.90×) | PSD gate + reduce |
| **ringpack_10_1** | MBQCP | 70 | wall 20.3→15.2 s (0.75×) | PSD gate + reduce |

The one regression is `hybriddynamic_fixedcc` (wall 1.5→1.7 s, +9 % — a 5-node
already-optimal QP where the ON reductions add setup cost that does not pay off).
No status was ever lost; no bound ever loosened past the oracle.

### 3.2 Honest verdict per capability

- **branch-and-reduce — CLASS-LEANING (promising, not yet definitive).** Its
  target structure is nearly universal (19/20), and it helps a *spread* of
  distinct structures — a pure-integer MINLP (`spring`), a QCP, a QCQP, an MBQCP —
  not one probe. That breadth is the signal that it is a *class* effect, not the
  named-probe effect the panel missed. But at N=20 the win rests on **4
  instances** and the benefit/regression margin (31 % vs 8 %) is thin; the label
  is "class-leaning", to be confirmed at larger N (§6). It is **not inert** on the
  held-out sample — the panel's 0/0 was an instrument artifact.
- **PSD cost gate — CLASS-LEANING (promising, narrower reach).** It reaches 14/20
  (QCQP-family) and helps 3 of the 10 scored QCQP carriers (kall, ex5_3_3,
  ringpack) with one thin regression. Again a *spread* of QCQP subtypes rather
  than a single probe — encouraging, but 3 instances at N=20 is a first signal,
  not proof. Note the bundle: because all three flags run together, part of these
  QCQP wins is also branch-and-reduce; isolating PSD alone is a §6 follow-up
  (an ON-with-only-`PSD_COST_GATE` arm).

**Bottom line (§0.2):** neither capability is a *probe-only* win on the held-out
sample — both help multiple distinct held-out structures they were never tuned on,
which is the class signal the 61-panel could not show. Neither is yet a
*definitive* class win at N=20; the pilot's job is to establish the signal and
prove the instrument, which it does.

---

## 4. Caveats (read before scaling)

1. **Small scored N.** 14 of 20 scored (6 errored, below); per-capability scored
   counts are 13 and 10. Benefit-fractions are over single-digit numerators.
   Treat the verdicts as directional.
2. **Bundled flags.** flags-ON sets all three at once (the shipped bundle).
   Per-capability attribution is by *structure carried*, not isolation. A
   definitive per-capability read needs single-flag ON arms (§6).
3. **Metric jitter on non-terminating solves.** For instances that hit the time
   limit, `node_count`/`wall` vary run-to-run (the B&B is wall-bounded, not
   deterministic). Several instances flipped neutral↔benefit↔regression between
   two runs of the identical command (e.g. `powerflow0014r`, `wastepaper3`). The
   *terminating* instances (spring, kall, ex5_3_3, ringpack, hybriddynamic) are
   the stable signal; TL-bounded rows are noisy. Larger N + longer TL both help.
4. **The `time_limit` is not honored on 6 hard instances (a real finding).** Six
   solves (`graphpart_clique-70`, `qap`, `sonet23v6`, `eg_all_s`,
   `autocorr_bern30-15`, and `sonet22v5` OFF) **overshot the 15 s `time_limit`
   to the harness's 105 s outer-timeout** (`tl+90`) in one or both configs — the
   root loop runs past the budget before checking the clock. This is present in
   *both* OFF and ON on 5 of the 6 (so it is a pre-existing solver
   time-limit-adherence gap, not a flag regression), and it is what makes large
   QCQP dominate the pilot's wall. The harness records these as errored and
   **excludes them from ratios** (it does not silently count a hung solve as a
   result). This is worth a tracking issue independent of GEN-1.

---

## 5. Gates (this PR)

- **No solver change** — a benchmark script + this doc only. The harness sets env
  flags per subprocess; solver math is untouched.
- `pytest -m smoke` (python/tests): **617 passed, 14 skipped** (the release build
  is sound; the change touches no solver code).
- `ruff check` + `ruff format --check` + `mypy --ignore-missing-imports` on the
  new script: **clean**. (Pre-commit `ruff`/`mypy` hooks are scoped to
  `^python/` / `^python/discopt/` and deliberately exclude
  `discopt_benchmarks/` — verified in `.pre-commit-config.yaml` — so these were
  run manually.)
- Analysis logic (bracket soundness, benefit/regression compare, geomean,
  per-capability tally) unit-self-tested in the session.

---

## 6. How to scale to the full sweep (the user's follow-up)

1. **Larger N + longer TL:** `--n 100 --seed 0 --time-limit 60 --max-vars 1000`.
   More scored instances shrink the single-digit-numerator problem; a longer TL
   lets more QCQP terminate (turning noisy TL rows into stable node/wall signal).
   Budget: at 60 s/solve and the current 105 s-overshoot tail, expect a few hours
   — run it as an overnight job, not interactively.
2. **Isolate each capability** (the attribution fix): add ON arms that set only
   `DISCOPT_PSD_COST_GATE=1`, only `DISCOPT_ROOT_FIXPOINT=1 DISCOPT_NODE_REDUCE=1`,
   and the bundle — so a QCQP win is attributed to the flag that caused it, not
   the bundle. (Straightforward extension: a `--configs` list of env dicts.)
3. **Multiple seeds** for the noisy-metric caveat: `--seed 0,1,2`, report the
   median benefit-fraction per capability so TL-jitter averages out.
4. **Fix the time-limit overshoot first** (or raise `--max-vars` cautiously): the
   6 outer-timeouts are pure wasted wall and pollute the sample; a solver-side
   `time_limit`-adherence fix (tracked separately) would let the sweep include the
   large QCQP that carry the most PSD structure.
5. **Structure-targeted strata:** to stress one capability, filter the eligible
   pool to its structure (e.g. only QCQP-family for the PSD gate) so K/N ≈ 1 and
   every scored instance is a real test of reach.

The harness already writes the exact instance list + all env flags into the JSON,
so any scaled run is reproducible and diffable against this pilot.
