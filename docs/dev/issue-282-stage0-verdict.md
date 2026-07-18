# Issue #282 — Stage 0 ensemble-probe verdict (2026-07-18)

**Entry experiment gating the convex-half campaign** (SOTA-review staged plan, comment
"SOTA review of the convex-half frontier … VERDICT: closable"). Throwaway out-of-solver
probe — no shipped solver code. Harness:
`discopt_benchmarks/scripts/issue282_stage0_ensemble_probe.py`; raw data:
`discopt_benchmarks/results/issue282/stage0_ensemble_probe_20260718T204831.json`.

## What was measured

An out-of-solver root cutting loop on the two convex NLP-BB instances `rsyn0805m` and
`syn40m` (MAXIMIZE; opt from `minlplib.solu`):

- **Root LP** = the big-M linear rows from `reformulate_gdp(method="big-m")` + OA tangents
  of the convex nonlinear `<=` rows, integer columns marked. Built by numerically
  extracting the affine rows / objective / Jacobian from `NLPEvaluator` over the
  reformulated model. **Validation:** the warm-started root LP bound `B0` reproduces the
  known exact continuous-relaxation excess to 3 decimals (**rsyn0805m +62.873 %**,
  **syn40m +2608.353 %**) — the extraction is faithful.
- **30 rounds:** solve LP → separate at the LP vertex → add OA tangents for violated
  nonlinear rows → re-solve. LP objective tracked each round (an upper bound, MAXIMIZE).
- **Arm A:** discopt's existing trusted separators as-is — `separate_cmir` (single-row +
  **dual-weighted** + pairwise-binding aggregation c-MIR), `separate_cover_cuts` (knapsack
  cover), LP duals passed for the dual-weighted aggregation.
- **Arm B:** Arm A **plus** a prototype Marchand–Wolsey variable-upper-bound (VUB)
  substitution — detect `x ≤ U·y` (`y` binary) rows, substitute `x → U·y − s` (`s ≥ 0`)
  into the aggregated row before the MIR δ-scan, then map the cut back to `x`-space.
  Directly tests the documented "follow-on" at `crates/discopt-core/src/lp/aggregation.rs:46`.

## Result

| instance | spread (pts) | B0 excess | arm A → r30 | **A closed** | arm B → r30 | **B closed** | rounds with cuts (A/B) |
|---|---|---|---|---|---|---|---|
| `rsyn0805m` | 46.80 | +62.873 % | +60.448 % | 5.18 % | +59.656 % | **6.87 %** | 1 / 1 |
| `syn40m` | 2604.86 | +2608.353 % | +2219.857 % | 14.91 % | +2140.999 % | **17.94 %** | 3 / 3 |

Spread = discopt root (+62.873 / +2608.35 %) − SCIP root (+16.07 / +3.49 %); kill bar = 10 %
of spread. Every bound stayed **≥ opt** at every round on both arms (sound; no cut removed
the true optimum).

### Kill-criterion verdict — Arm B is a SURVIVOR

Arm B closes **17.94 % of the discopt→SCIP spread on `syn40m`** (≥ 10 % on ≥ 1 instance).
The ensemble hypothesis is **NOT falsified** for our separators → **GO** to Stages 1–3.
(Below bar on `rsyn0805m`: 6.87 %.)

### The mechanism, honestly — a one-shot, not a sustained ensemble

The 30-round loop is nearly moot: **all separation happens in rounds 1–3, then the loop
saturates completely** (rsyn0805m: cuts only in round 1, `added=[8/9, 0, 0, …]`; syn40m:
`added=[24,5,1,0,…]` / `[45,14,3,0,…]`). After round 3 discopt's separators — even with
duals and VUB substitution — find **zero** further violated cuts. This is the **opposite**
of the SCIP mechanism the SOTA review attributed (sustained 21–28 productive
aggregation rounds closing 74–99 % of the spread). discopt's separators capture a fraction,
then die; passing LP duals changed the trajectory by **exactly nothing** (byte-identical),
so the dual-weighted aggregation is not the missing piece either.

**VUB substitution (Arm B) is a real but secondary lever.** It roughly doubles the cuts
found in the live early rounds (rsyn round 1: 8→9; syn40m round 1: 24→45) and adds a genuine
increment to the bound: **+1.69 pts of spread on rsyn0805m** (5.18→6.87 %) and **+3.03 pts
on syn40m** (14.91→17.94 %). It validates the `aggregation.rs:46` follow-on directionally —
but it does not, by itself, carry either instance near SCIP, and even Arm A alone already
clears the bar on `syn40m` (14.91 %).

**82–95 % of the spread remains uncut** by everything discopt can currently separate.

## Recommendation: GO, re-scoped — VUB is necessary-but-not-sufficient

Proceed to the staged campaign, but the driving lever is **not** VUB substitution alone:

1. **Stage 1 (presolve big-M coefficient tightening) is the load-bearing lever, do it
   first.** SCIP evidence in the SOTA review: presolve coefficient tightening *alone*
   (separation fully off) closes +80→+51.5 % on `rsyn0805m` and +2935→+975 % on `syn40m` —
   i.e. it moves the root bound far more than every cut family this probe could fire (which
   together moved ≤ 18 % of spread). It is independent of the cut loop and benefits every
   big-M model. This is where the majority of the discopt→SCIP root gap lives.
2. **Stage 2 (VUB substitution in `aggregation.rs`/`mir.rs`) is confirmed worthwhile but
   modest** — implement it, but expect ~+2–3 pts of spread, not family closure. The probe's
   saturation shows discopt's *aggregation set itself* is too weak: it exhausts in ≤ 3
   rounds where SCIP sustains 21–28. Stage 2 must strengthen the aggregation loop (many-row
   Marchand–Wolsey aggregations + a cut pool + re-separation), not merely bolt VUB onto the
   current single/pairwise-binding aggregation, or the sustained-round mechanism will not
   materialize.
3. **Stage 3 (NLP-BB root cut stage) consumes the above** and is only worthwhile once
   Stages 1–2 make the root LP demonstrably tighter; gate it on the §1.5 node-reduction bar.

**Caveat on the `rsyn` vs `syn40m` split (real, do not force one mechanism onto both).**
`syn40m`'s +2608 % gross excess is far more cut-responsive in *relative-spread* terms
(17.94 % closed) than `rsyn0805m`'s +62.9 % (6.87 %). The families respond differently; the
probe confirms the diagnosis's open question that these are distinct failure modes.

**Frontier honesty.** This is *not* a ~0 frontier result — the cuts do move the bound
(15–18 % of spread on `syn40m`). But it is also *not* a reproduction of SCIP's iterated
ensemble on discopt's separators: the loop saturates in ≤ 3 rounds and leaves 82–95 % of the
spread on the table. The convex half is *reachable* with the SOTA machinery, but Stage 0
shows the dominant lever is **presolve coefficient tightening (Stage 1)**, with VUB
substitution a confirmed secondary contributor, and a materially stronger sustained
aggregation separator (beyond the current implementation) required to approach SCIP.
