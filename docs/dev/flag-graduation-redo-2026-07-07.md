# Flag-graduation gate — G1.3 REDO, post-C-38 (2026-07-07)

**Task:** roadmap G1.3-redo of
`docs/dev/default-path-performance-plan-2026-07-06.md` — re-run the graduation
gate (`discopt_benchmarks/scripts/graduation_gate.py`) on the two candidates that
were **clean on regression** in the first run (`flag-graduation-run-2026-07-07.md`)
but blocked by contaminants that are now fixed on `main`:

- **C-38** (#536) — `kall_circles_c8a` false-optimal on the **default path**
  contaminated the shared OFF control's soundness check in every arm.
- **CUTOFF-SOUND-1** (#535) — over-strict byte-tolerance in the cert-neutrality
  check; now **regime-aware** (a bound-changing flag is judged against the true
  optimum, not byte-identity).

The two candidates: **`psd_cost_gate`** (THRU-2a, cost-aware PSD-cut gate) and
**`lift_zero_spanning`** (R4, lift zero-spanning product factors). Both were
cert-neutral + sound-where-absent + fire-on-target in the first run; both were
INELIGIBLE only because the shared OFF control tripped C-38.

## Part 1 — C-38 is gone (default path re-certifies `kall_circles_c8a`)

`kall_circles_c8a`, all flags OFF, release build, tl 120 s, gap 1e-4:

| | status | objective | dual bound | nodes | verdict |
|---|---|---|---|---|---|
| **before (#536 not merged)** | `optimal` | 3.6142348 | **3.6142347** | 3 | **FALSE-OPTIMAL** — dual bound sits ~42 % ABOVE true min 2.5409 |
| **after (#536 on main)** | `feasible` | 3.6142348 | **≈ −1.0e-9** | 3 | SOUND — bound is a valid underestimator |
| MINLPLib oracle | | `=best= 2.5409191` | `=bestdual= 2.5409129` | | |

Post-fix the solver no longer certifies `optimal` on a wrong bound: it returns
`feasible` with a dual bound `≈ 0`, which satisfies both certificate invariants —
`bound (≈0) ≤ incumbent (3.614)` and `bound ≤ =best= (2.5409)`. The false
optimality certificate is gone. The merge took.

## Part 2 — re-verdict (before → after C-38)

Gate config (identical to the first run for comparability):
`graduation_gate.py --flags psd_cost_gate,lift_zero_spanning --n 40 --seed 0
--time-limit 25`, release build, `pounce-solver 0.7.0` (`_pounce.abi3.so` 4.5 MB),
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`. Seed-0 N=40 sample includes
`kall_circles_c8a` + `kall_circles_c6b/c6c` (the C-38 family) — the exact instances
that tripped the first run.

Report: `reports/graduation_gate_2026-07-07T06-32-13.{json,md}`; verdicts appended
to `docs/dev/data/graduation-ledger.jsonl` (2 rows, both `eligible:true` — green
#1 for each). The C-38 family solved clean this run (no `!!VIOL`):
`kall_circles_c8a` `feasible 3n`, `kall_circles_c6b` `feasible 63n`,
`kall_circles_c6c` `feasible 31n` — in BOTH the OFF control and each arm's ON.

| flag | metric | **first run** (pre-C-38) | **this run** (post-C-38) |
|---|---|---|---|
| psd_cost_gate | eligible | no | **YES** |
| | soundness_ok | N (C-38 in shared OFF control) | **Y** (0 violations) |
| | cert_neutral | Y | **Y** (bound-changing regime) |
| | regression_rate | 5 % (1/19) | **0 %** (0/20) |
| | benefit_fraction | 11 % | 5 % (elf) |
| | streak | 0/3 | **1/3** |
| lift_zero_spanning | eligible | no | **YES** |
| | soundness_ok | N (C-38 in shared OFF control) | **Y** (0 violations) |
| | cert_neutral | Y | **Y** (bound-changing regime) |
| | regression_rate | 8 % (2/26) | **3.8 %** (1/26, sonet22v5) |
| | benefit_fraction | 8 % | 8 % (tln12, ringpack_10_2) |
| | streak | 0/3 | **1/3** |

Both flags flip from INELIGIBLE → **ELIGIBLE**. The delta is exactly the two fixes:
soundness now passes (C-38 no longer false-optimals `kall_circles_c8a` on the
shared OFF control), and cert-neutrality passes under the regime-aware
bound-changing check (#535). `lift_zero_spanning`'s single regression (sonet22v5) is
a large-QCQP off-structure re-solve timing artifact (a 115 s-class outer-timeout
instance where the lift is byte-identical/inert), well under the 10 % ceiling.

## Part 2 — bound-changing verification (CLAUDE.md §5), per flag

Before any flip, the bound-changing regime requires (a) a differential dual-bound
test (ON bound a valid underestimator, never crossing `=opt=`) and (b) feasible-
point sampling (no valid point cut). Both flags are, by construction, sound:
`psd_cost_gate` only ever *drops* PSD cuts (loosens the relaxation — cannot cut a
feasible point); `lift_zero_spanning` adds an exactness-preserving McCormick
envelope on a lifted auxiliary (R4 #518). Measured confirmation below.

### Differential dual-bound test (fixed instances, tl 40-60 s, min sense)

| flag | instance | =opt= | bound OFF | bound ON | ON ≤ opt? | direction | verdict |
|---|---|---|---|---|:-:|---|---|
| psd_cost_gate | nvs17 | −1100.4 | −1100.4000022 | −1100.4100022 | **yes** | ON looser (drops cuts) | valid underestimator, no cross |
| psd_cost_gate | ex5_3_3 | 3.234018 | 1.631318 | 1.631318 | **yes** | equal | valid underestimator, no cross |
| lift_zero_spanning | st_e36 | −246.0 | −304.490 | −246.019 | **yes** | ON **tighter** (feasible→optimal) | valid underestimator, no cross |

`psd_cost_gate` is a *loosening* flag: its ON dual bound is ≤ the OFF bound (fewer
cuts) but still ≤ `=opt=` — a valid underestimator. `lift_zero_spanning` is a
*tightening* reform: ON bound ≥ OFF bound (−246.019 ≥ −304.490) and still ≤ `=opt=`
— exactly the R4 win (root un-pins from −304.5 to ≈ optimum). Neither crosses
`=opt=` on any instance.

### Feasible-point sampling (no valid point cut)

Incumbent point (original variables) recovered OFF vs ON:

| flag | instance | OFF incumbent point | ON incumbent point | obj OFF | obj ON | =opt= | verdict |
|---|---|---|---|---|---|---|---|
| lift_zero_spanning | st_e36 | `x0=5, x1=20` | `x0=5, x1=20` | −246.0 | −246.0 | −246.0 | identical point survives reform |
| psd_cost_gate | nvs17 | `2,6,3,2,8,6,6` | `2,6,3,2,8,6,6` | −1100.4 | −1100.4 | −1100.4 | identical point survives gate |

Both flags recover the **identical** optimal feasible point ON as OFF, and both
certify the true optimum. If either had cut a feasible point, the incumbent would
be wrong/worse or the dual bound would rise above `=opt=` — neither occurs.
`psd_cost_gate` can only *enlarge* the relaxation's feasible set (drops
constraints), so cutting a valid point is impossible by construction; the sample
confirms it.

### Fire-check (does the capability execute on its target?)

- **psd_cost_gate** — `coverage_map.py` with the flag ON: **nvs17** `psd_calls=170`
  (certifies −1100.4), **ex5_3_3** `psd_calls≥1`. The PSD separation loop the gate
  controls is on the active QCQP path and fires.
- **lift_zero_spanning** — st_e36, flag OFF: `feasible, bound −304.49` (root pinned,
  cannot certify). Flag ON: **`optimal`, bound −246.02** — the documented
  feasible→optimal status upgrade. Structure-gated: where the zero-spanning product
  structure is absent, the reform is byte-identical (inert), so this is a free
  on-structure win.

### Cert-neutrality proof (standalone, bound_changing regime, 41-instance panel)

Re-solve of the committed `cert-baseline.jsonl` with each flag ON, checked against
the true-optimum oracle (`cert-optima.json`) in the **bound_changing** regime
(certified objective must bracket the true optimum; node_count may shift):

| flag | hard violations (objective/status/missing) | node-drift instances | neutral? |
|---|---|---|---|
| psd_cost_gate | **0** | nvs13 (documented PSD perf note) | **YES** |
| lift_zero_spanning | **0** | none (panel has no zero-spanning structure) | **YES** |

Both: **0 certified-objective changes, 0 lost optimal statuses.** `psd_cost_gate`'s
only node-count shift is nvs13 (the documented, expected perf note where PSD
structure is present — the certified objective is unchanged). `lift_zero_spanning`
shows **zero** drift: the cert panel carries no zero-spanning product structure, so
the flag is fully inert there — confirming structure-narrow inertness. No
certified-objective cross → cert-neutral (a cross would be a STOP; none occurred).

### Pre-existing (flip-independent) test note

`test_r4_flag_on_unpins_pinned_product` (slow, `test_factorable_reform.py`) fails on
this machine/build — its **synthetic** `_st_e36_shaped` probe stays pinned at −304.5
in *both* arms within the 60 s budget (a wall-clock race the docstring itself flags
as machine-dependent), so the ≥25 % un-pin **performance** claim fails. Its
**soundness** assertions (dual bound never crosses the optimum) PASS. Confirmed the
**identical** failure on `origin/main` with the source reverted — both arms set
`DISCOPT_LIFT_ZERO_SPANNING_FACTORS` explicitly, so the default-flip cannot affect
it. The **real** `st_e36.nl` un-pins correctly (−304.5 → −246.02 optimal, measured
above). Left untouched (not weakened to pass); out of scope for this graduation.

## Flip decisions

**Both flags flipped default-ON** (one PR each, base `main`).

- **psd_cost_gate → default-ON** (`solver_tuning.py`, `_env_flag(...,default=True)`;
  `DISCOPT_PSD_COST_GATE=0` escape hatch preserved). Clean, wide-margin single
  validation: **0 % regression**, 0 soundness violations, cert-neutral, differential
  bound a valid underestimator with no `=opt=` cross, feasible-point sample
  identical ON/OFF, fires (nvs17 `psd_calls=170`). Sound by construction (only drops
  cuts). PR: `perf(graduate): G1.3 — psd_cost_gate default-ON (gate-validated
  post-C-38)`.
- **lift_zero_spanning → default-ON** (`factorable_reform.py`, env-resolved default
  ON with `DISCOPT_LIFT_ZERO_SPANNING_FACTORS=0` escape hatch). Eligible: **3.8 %
  regression** (1 off-structure timing artifact, ≪ 10 % ceiling), 0 soundness
  violations, cert-neutral, differential bound **tighter** (feasible→optimal on
  st_e36) and still ≤ `=opt=`, feasible-point sample identical ON/OFF, fires
  (st_e36 feasible→optimal). Structure-narrow but sound + cert-neutral + inert
  off-structure → a valid free default-ON. PR: `perf(graduate): G1.5 —
  lift_zero_spanning default-ON (gate-validated post-C-38)`.

**On the 3-green streak.** The protocol's nightly path graduates a flag after 3
consecutive green ledger verdicts; this run banks **green #1** for each. Both flips
are authorized here under the plan's single-validation rule (a *clean, wide-margin*
verdict — 0 %/3.8 % regression, 0 soundness violations, cert-neutral, plus the
bound-changing differential + feasible-point verification the regime requires),
mirroring the ILS-cap (#532) held-out-validation graduation. The escape hatch
(`DISCOPT_<FLAG>=0`) means neither flip is irreversible; the nightly gate continues
to accrue the streak on `main`, and a future non-eligible verdict would resurface
either flag for review.

Neither flip "barely clears" and neither differential/feasible-point test revealed
a bound cross, so the honest-scope kill (bank + leave OFF) does not apply.

## Honest scope

- **psd_cost_gate** helps the **QCQP class** (nvs17/19/24, ex5_3_x). Most of the
  61-instance cert panel is not QCQP, so the flag is inert there (cert-neutral by
  absence); its benefit is where PSD separation dominates the node wall. It fires +
  is sound on nvs17/ex5_3_3 (cited above). A cut-*dropping* flag is sound
  everywhere by construction.
- **lift_zero_spanning** is **structure-narrow**: it helps `st_e36` (feasible→
  optimal) and is provably inert where the zero-spanning product structure is
  absent (byte-identical reform). A structure-narrow but cert-neutral + sound flag
  is a valid default-ON — a free win on-structure, invisible elsewhere.
