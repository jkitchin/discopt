# #1204 — the cert panel's wall budgets, and the box they were chosen on

The cert panel's per-instance budgets are wall-clock seconds picked on the machine
that generated `cert-baseline.jsonl`. There the slowest rows certify at roughly
half of them:

| instance | budget | reference wall | fraction |
|---|---:|---:|---:|
| `nvs17` | 30 s | 16.5 s | 0.55 |
| `tanksize` | 60 s | 31.0 s | 0.52 |
| `tls2` | 60 s | 30.4 s | 0.51 |
| `nvs05` | 60 s | 28.3 s | 0.47 |
| `clay0303hfsg` | 60 s | 28.3 s | 0.47 |

Any runner about 2× slower tips all of them, and then *whether an instance
certifies* is a fact about the runner. #1204 is what that does downstream: a row
the flag-OFF control lost to the wall is disarmed for every arm, while the same row
armed hard-fails whichever arm tips first — so two graduation-gate runs of
identical PR code failed on **different** four-arm subsets, and a `main` run that
drifted *more* than either passed. Gate run 29 (2026-09-14) passed the same way:
control `nvs05` `feasible`, control `tanksize` `time_limit`, both then reported
UNMEASURED for all seven arms.

## Entry experiment (CLAUDE.md §4)

**Hypothesis.** Certification inside a wall budget is a property of the *runner*;
scaling every budget by the measured host ratio gives each row the reference's
*work* allowance and makes certification a property of the model again.

**Kill criterion.** If a cliff row still fails to certify at the calibrated budget,
or certifies to a different answer, the calibration does not fix the class and must
not ship as the mechanism.

**Box.** 12-core container, load 0.07 before the probe. `maturin develop --release`
at `3868cd9`.

### Step 1 — measure the box

18 probe rows (settled, unrouted, reference wall in (0.05, 3.0] s), solved
flag-OFF at nominal budgets; 16 reproduced their `node_count` exactly and were
kept.

```
RATIO median=3.172 min=0.836 max=3.495 sd=0.607   (n=16, probe wall 37.6 s)
```

Two rows were dropped by the equal-node filter (`nvs02` 421→297 nodes, `nvs14`
839→273) — the filter doing its job: their walls carry a tree change, not a speed.
`dispatch` came back at 0.836 on a 0.17 s reference wall, which is why the
statistic is a median and why the spread is reported with it (CLAUDE.md §9).

### Step 2 — the cliff rows, nominal vs calibrated (×3.172)

| instance | budget | status | wall | nodes (ref → run) | objective |
|---|---:|---|---:|---|---|
| `tanksize` | 60 s | `time_limit` | 60.5 s | 16649 → 13548 | 1.2686437535708868 |
| `tanksize` | **190 s** | **`optimal`** | 71.4 s | 16649 → 17139 | 1.2686437535708868 |
| `nvs05` | 60 s | `feasible` | 60.0 s | 149 → 91 | 6.935776493421913 |
| `nvs05` | **190 s** | **`optimal`** | 106.6 s | 149 → 231 | 5.4709341091342365 |
| `clay0303hfsg` | 60 s | `optimal` | 37.8 s | 297 → 235 | 26669.10957284308 |
| `clay0303hfsg` | **190 s** | `optimal` | 55.8 s | 297 → 235 | 26669.10957284308 |
| `tls2` | 60 s | `feasible` | 60.7 s | 153 → 323 | 5.299999999987323 |
| `tls2` | **190 s** | **`optimal`** | 76.5 s | 153 → 329 | 5.299999999988753 |

Reference optima: `tanksize` 1.2686437535708868, `nvs05` 5.470934075438167,
`clay0303hfsg` 26669.10957, `tls2` 5.2999999999869125.

**Result: the hypothesis holds.** Three of the four vendored cliff rows tipped at
the nominal budget on this box; **all three certify at the calibrated one, at the
reference's answer** — `tanksize` to the last bit, `tls2` to 2e-12, and `nvs05` to
1e-8 of the true optimum its nominal run missed by 27 %. The fourth,
`clay0303hfsg`, did not tip here and was unaffected.

Three things the measurement said that the plan did not:

1. **The slowdown is not uniform across instances.** The panel median is 3.17, but
   `clay0303hfsg` ran at 1.34× and certified inside the *nominal* budget, while
   `nvs05` needed 3.77×. A fixed "2× headroom" would have rescued `tanksize` and
   missed `nvs05`; scaling by the *measured* ratio is what covers both. This is
   also why the scale is a single number applied to every budget rather than a
   per-instance correction — a per-instance fit is the hand-gating CLAUDE.md §2
   rules out, and `_KNOWN_PERF_GATED`'s lone `nvs17` entry is what that looks like.
2. **A larger budget costs wall even on rows that already certified.**
   `clay0303hfsg` returned the same 235 nodes and the same objective in 37.8 s at
   60 s and 55.8 s at 190 s. That is expected rather than alarming: the solver's
   role-2 sub-budgets are fractions of `time_limit` (`_role2_slice` and friends), so
   a calibrated budget buys those stages the same *work* the reference gave them.
   It is the mechanism working below the top-level stop — and it is the cost line to
   watch on the gate's wall-clock, since it applies to every row, not only the
   cliff ones.
3. **`nvs05`'s nominal row is the #1195 shape.** Its `feasible` incumbent sits 27 %
   above the optimum — the expected form of an open gap, not a false certificate,
   which is why #1195 had to stop bracketing uncertified rows against the oracle.
   Calibrated, the row certifies and the bracket applies again, at full strength.

## What ships, and what it does not claim

* Budgets are calibrated (`utils/host_calibration.py`), clamped to `[1.0, 4.0]`:
  never tighten a faster box's budgets, never turn a pathologically slow box into
  an overnight job.
* When the calibration cannot be measured (fewer than 5 equal-node unrouted probe
  rows) the budgets stay nominal and the run says so — in the probe output *and*
  next to the verdict.
* The calibration is measured **once per gate run**, flag-OFF, before the control,
  and the same scale is given to the control and all seven arms. Re-measuring per
  arm would put a fresh noisy multiplier between an arm and its reference, which is
  a new version of the asymmetry being removed.
* It does **not** claim to remove wall-limited rows entirely. A box beyond the cap,
  a failed probe, or an instance whose per-instance ratio exceeds the panel median
  can still tip. That residue is handled — not hidden — by
  `cert_neutrality.wall_limited_arms`: a certification lost at the wall is reported
  as `wall_regression` (perf-class), never as a soundness fault, because a
  wall-limited row is never `optimal` and so carries no certificate to be false.

## Step 3 — the shipped pipeline, end to end

`check_cert_neutrality.main()` run for real (probe → host scale → panel-cost fit →
scaled budgets → `wall_limited_arms` → verdict), on a 20-row subset panel (the 18
probe candidates plus the three vendored cliff rows) so it costs ~10 min instead of
~25. Nothing stubbed.

```
host-speed calibration: this box is x3.23 the reference machine's wall
  (median over 15 equal-node unrouted rows, sd 0.64, load 0.42);
  budgets scaled x3.23 so each row gets the reference's work allowance
  [19/20] tanksize   optimal  nodes 16649->17139  |Δobj|=0.00e+00
  [20/20] tls2       optimal  nodes   153->475    |Δobj|=2.15e-12
3 VIOLATION(S):
  nvs05   [objective/soundness]      |Δobj|=3.370e-08 (5.470934075438167 -> 5.4709341091342365)
  nvs05   [node_regression/perf]     node_count 149 -> 231 (+55%)
  tls2    [node_regression/perf]     node_count 153 -> 475 (+210%)
```

**Zero rows wall-limited** — the `#1204` section that listed three of them in the
previous run is absent, because there were none to list. What remains are genuine
tree differences against a month-old reference from another machine: an objective
drift of 3.4e-8 (past this regime's 1e-8 byte-reproducibility bar, ~3000x inside
the 1e-4 correctness tolerance) and two node counts. That is the panel measuring
the tree instead of the runner, which is the whole point.

### The bug this run found

The first end-to-end attempt printed `budgets scaled x3.12` and then reported all
three cliff rows as having "ran out of its **60 s** budget" — the scale was
measured, announced, and never applied. `predicted_panel_wall` summed over
`budgets` (`_instance_budgets()`: global50 plus the perf panel) rather than over the
panel actually solved (the cert baseline), charging ~49 instances that never run a
full unscaled budget each. It predicted ~9000 s against the 900 s ceiling and
`fit_scale_to_panel` dutifully shrank x3.12 back to x1.0: a cost guard silently
cancelling the calibration it was there to bound.

Every unit test passed throughout, because they all built `budgets` as
`dict.fromkeys(baseline, 60.0)` — keys identical to the panel, the one case that
cannot expose the mismatch. It took running the real `main()`. The regression test
is `test_budgets_for_instances_outside_the_panel_are_not_charged`, and it was
checked to fail against the old loop (1340 s vs 20 s on the same inputs) rather
than assumed to.
