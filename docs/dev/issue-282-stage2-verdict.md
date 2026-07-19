# Issue #282 — Stage 2 entry-experiment verdict (2026-07-18)

**Stage 2 = the cut-side lever:** Marchand–Wolsey variable-bound (VUB/VLB)
substitution in the c-MIR aggregation/MIR δ-scan, **plus a stronger *sustained*
aggregation loop** (deeper many-row MW aggregations + a cut pool) intended to fix
the round-3 saturation that Stage 0 recorded.

**VERDICT: FALSIFIED — do not ship Stage-2 cut code.** With Stage-1 coefficient
tightening (`DISCOPT_COEF_TIGHTEN=1`) as the base, VUB + sustained aggregation
closes **< 10 % of the remaining post-Stage-1 spread on all four instances**
(kill criterion: ≥ 10 % on ≥ 1 instance). Recommend going straight to **Stage 3**
(the full NLP-BB root LP-OA ensemble loop) rather than shipping a marginal cut
flag.

Harness: `discopt_benchmarks/scripts/issue282_stage2_probe.py`; raw data:
`discopt_benchmarks/results/issue282/stage2_probe_20260718T223408.json`.

## Why the base changed (and why that is the whole point)

Stage 0 measured cut gain against the *untightened* root (`B0 = +2608 %` on
`syn40m`) and reported arm B closing **17.94 % of the full discopt→SCIP spread**.
That number is not the Stage-2 increment. Stage 1 (coefficient tightening) shipped
between Stage 0 and Stage 2 and **already consumed the low-hanging root gap**:

| instance | Stage-0 B0 (no coef-tighten) | **post-Stage-1 B0** | SCIP root | remaining spread |
|---|---|---|---|---|
| `rsyn0805m` | +62.87 % | **+62.27 %** | +16.07 % | 46.20 pts |
| `rsyn0810m` | — | **+67.33 %** | +9.5 % | 57.82 pts |
| `rsyn0815m` | — | **+87.69 %** | +17.9 % | 69.78 pts |
| `syn40m` | +2608.35 % | **+1145.40 %** | +3.49 % | 1141.91 pts |

Coefficient tightening is the load-bearing lever exactly as Stage 0 predicted:
`syn40m` +2608 → +1145 %. On the `rsyn*` family it moves the root barely at all
(52–62 rows tightened, +62.87 → +62.27 % on `rsyn0805m`) — the `rsyn` gap is a
*different* failure mode from `syn40m`'s big-M charge, confirming the Stage-0/§R3
"do not force one mechanism onto both" caveat.

Stage 2 must therefore be measured as the **incremental** cut gain *on the tighter
root*, and that is where it collapses: the same MIR cuts move the bound far less on
the already-tightened rows.

## Result — incremental cut gain on the post-Stage-1 root

| instance | remaining spread | arm A (base seps) closed | **arm B (VUB + sustained agg) closed** | Δ from VUB/agg |
|---|---|---|---|---|
| `rsyn0805m` | 46.20 pts | 3.84 % | **3.84 %** | +0.00 pts |
| `rsyn0810m` | 57.82 pts | 2.31 % | **2.31 %** | +0.00 pts |
| `rsyn0815m` | 69.78 pts | 3.93 % | **4.24 %** | +0.22 pts |
| `syn40m`    | 1141.91 pts | 8.53 % | **8.68 %** | +1.71 pts (0.15 % of spread) |

- **Kill criterion NOT met.** Arm B closes < 10 % of the remaining spread on every
  instance (best: `syn40m` 8.68 %). Stage 2 is falsified.
- **Soundness clean.** Every LP bound stayed ≥ opt at every round on both arms
  (valid relaxation of a MAXIMIZE); no cut removed the true optimum.

### The two Stage-2 mechanisms, measured honestly

1. **VUB/VLB substitution is real but negligible on the tighter root.** It roughly
   doubles the cuts found in the live early rounds (`rsyn0815m` 24→34, `syn40m`
   24→48) but adds only **+0.22 pts** (`rsyn0815m`) and **+1.71 pts** (`syn40m`) to
   the bound — a rounding error against the 46–1142 pt spreads. On `rsyn0805m` /
   `rsyn0810m` arm B is **byte-identical** to arm A (the extra cuts are dominated
   and never make the final LP). Stage 0's "+1.7/+3.0 pts of spread" shrinks to
   "+0.2/+1.7 pts" once coefficient tightening has already claimed the easy gap.

2. **The sustained-aggregation loop does not sustain.** The deeper machinery —
   greedy many-row MW continuous-cancellation aggregations (up to 4-row chains),
   a persistent cut pool, and both plain and VUB c-MIR on every aggregate — still
   **dies after round 1** on `rsyn` (`added = [4, 0, 0, …]`, identical to base) and
   after round 3 on `syn40m` (`[48, 48, 17, 0, …]`). Adding more aggregation rows
   does not expose new violated faces; discopt's MIR machinery genuinely saturates.
   The missing piece is **not** more aggregation depth or VUB — it is whatever lets
   SCIP sustain 21–28 productive rounds (hybrid cut selection interleaved with OA
   refinement, and cut families — Gomory / strong-CG — outside the MIR/aggregation
   scope of Stage 2).

## The frontier, stated plainly

- **Coefficient tightening (Stage 1) is where the recoverable root gap lives** on
  the big-M family (`syn40m` −1463 pts); it barely touches `rsyn*`.
- **The residual is not reachable by more MIR/aggregation cutting.** VUB + a deeper
  sustained aggregation loop — the entire Stage-2 mandate — moves the tightened root
  by ≤ 1.7 pts (≤ 8.7 % of the remaining spread) and saturates in ≤ 3 rounds. This
  is a *recorded falsification*, not an unmeasured hunch: the cuts fire, are sound,
  and simply do not bite the tighter root.
- **The remaining ~90 % of the post-Stage-1 spread needs Stage 3**, or is
  structurally hard. The SOTA-review attribution located SCIP's residual gain in the
  *iterated ensemble over an OA-refined root LP with hybrid cut selection*, not in a
  stronger single separator. Stage 2's job was to test whether a stronger separator
  (VUB + sustained aggregation) alone suffices; it does not.

## Recommendation

**Do not ship a Stage-2 cut flag.** Skip to **Stage 3** — the NLP-BB root LP-OA
ensemble loop (`DISCOPT_NLPBB_ROOT_CUTS`, default-OFF): a Quesada–Grossmann-style
root master (linear rows + OA tangents), ≥ 20–30 ensemble separation rounds with a
cut pool, re-OA at each new vertex, adopting the final LP bound via `max`. Gate it
on the §1.5 node-reduction bar. Stage 3 is the only remaining mechanism the
attribution supports; Stage 2 has been measured to insufficiency and closed as a
frontier. Whether Stage 3 itself clears the bar on `rsyn*` (whose gap coefficient
tightening barely moved and cuts barely move) is itself an open question its own
entry experiment must answer — the `rsyn` vs `syn40m` split is real.

*A recorded falsification is a valid outcome (CLAUDE.md §4). Stage 2 does not ship.*
