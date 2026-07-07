# Flag-graduation protocol (G1.2)

**Status:** active instrument. Roadmap task G1.2 of
`docs/dev/default-path-performance-plan-2026-07-06.md` (§3 G1, cause C-B).
**What it is:** the reusable, per-flag validation gate + ledger that lets
discopt's parked default-OFF capabilities flip to default-ON on *evidence*, not by
hand. It is benchmark-infra + docs only — it changes **no solver math**.

## Why this exists (the process gap)

discopt's bound-changing regime (CLAUDE.md §5) correctly parks each new capability
behind a default-OFF env flag "pending 3 green nightlies". But no nightly pipeline
existed, so **nothing ever graduated** — five sound, class-validated capabilities
sit inert on `main` (the plan's §1 table). Every future bound-changing win would
suffer the same fate. This gate is the missing pipeline: it produces, per flag,
the evidence a graduation PR needs, and records it durably so "3 consecutive
green" is a checkable fact rather than a manual claim.

The template is already on `main`: **ILS-cap (#532)** graduated to default-ON
after its held-out validation. Every other flag follows the same path through this
gate.

## The parked flags

| arm (this gate) | env flag(s) | regime | target structure |
|---|---|---|---|
| `root_fixpoint` | `DISCOPT_ROOT_FIXPOINT=1` | bound-changing | reduce-responsive box (≥1 var, ≥1 con) |
| `node_reduce` | `DISCOPT_NODE_REDUCE=1` | bound-changing | reduce-responsive box |
| `psd_cost_gate` | `DISCOPT_PSD_COST_GATE=1` | bound-changing | QCQP (any Q[C]P family) |
| `lift_zero_spanning` | `DISCOPT_LIFT_ZERO_SPANNING_FACTORS=1` | bound-changing | products of zero-spanning factors (R4) |
| `lift_loose_products` | `DISCOPT_LIFT_LOOSE_PRODUCTS=1` | bound-changing | loose composite products (TD-A) |

The `off` control and the `all` bundle are also defined as arms for cross-checks
(`all` reproduces the pre-G1.2 bundled sweep). The escape-hatch convention holds
for every graduated flag: **`DISCOPT_X=0` restores the old behavior** — a graduated
flag is never a dead flag, it is a default that a user can turn off.

## The arms — isolation the pilot lacked

The N=20 pilot ran **all** flags together (`FLAGS_ON`), so a benefit could not be
attributed to a single capability and each flag's regression-rate was contaminated
by its neighbours (the nvs13 19→49-node regression is PSD's, but the bundle hid it
inside branch-and-reduce's numbers). The gate runs each parked flag as an
**isolated arm**: it sets exactly that capability's env flags, with every other
capability at its default OFF, against a shared `off` control. One run yields, per
flag, its own benefit-fraction / regression-rate / soundness / structural
prevalence.

Arms live in `discopt_benchmarks/scripts/generality_sweep.py` (`ARMS`,
`GRADUATION_ARMS`, `run_arm`, `arm_stats`). Run the sweep alone in arms mode with:

```bash
PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python discopt_benchmarks/scripts/generality_sweep.py \
    --arms psd_cost_gate,root_fixpoint --n 100 --seed 0 --time-limit 30
```

## The gate — what it runs per flag

`discopt_benchmarks/scripts/graduation_gate.py` runs, per requested flag:

1. **Held-out per-flag arm** — a seeded, stratified held-out sample (excludes the
   61 vendored panel instances + the named tuning probes), solved OFF vs the flag
   ON in isolation. Yields `benefit_fraction`, `regression_rate`,
   `structural_prevalence`, and any **soundness** violation (a dual bound crossing
   the best-known primal, or a certified-optimal outside the
   `[=bestdual=, =best=]` oracle bracket, in *either* config). The OFF control
   solve is shared across arms, so an N-flag sweep pays for it once.

2. **Cert-panel neutrality** — re-runs the cert-neutrality check
   (`check_cert_neutrality.py`'s logic) **in a fresh subprocess with the flag ON**
   (so the fresh interpreter reads the env flag at import) against the committed
   41-instance `docs/dev/data/cert-baseline.jsonl`:
   - a **bound-neutral** flag must be **byte-identical** (node_count + objective);
   - a **bound-changing / heuristic-policy** flag must keep the certified
     **objective** unchanged and the **optimal status** (soundness), while a
     node_count change on a cert instance carrying the flag's structure is an
     *expected, documented perf note*, not a fault.

3. **`incorrect_count = 0` + no oracle cross** — folded into (1) and (2): the
   held-out arm's oracle bracket is the held-out incorrect-count guard; the cert
   panel's objective-to-tolerance is the panel incorrect-count guard. A changed
   certified objective or a lost optimal status is a hard, non-zero-exit fail.

### Eligibility thresholds (documented; global, not per-instance)

A flag's verdict is `eligible` **iff**:

- **0** soundness violations in the held-out arm (either config), **AND**
- cert-neutral (objective unchanged + still optimal; node byte-identical too for a
  bound-neutral flag), **AND**
- `regression_rate` ≤ **`MAX_REGRESSION_RATE = 0.10`** (the documented ceiling; the
  pilot measured branch-and-reduce at 8 % and PSD at 10 % *bundled* — the isolated
  arms must land at or below 10 %).

Changing a threshold is a reviewed decision, recorded here and in the script
constant. These constants live at the top of `graduation_gate.py`
(`MAX_REGRESSION_RATE`, `GREEN_STREAK_REQUIRED`).

## The ledger — 3 consecutive green

Every full-gate run appends one verdict per flag to
`docs/dev/data/graduation-ledger.jsonl` (append-only, one JSON object per line).
The verdict shape:

```json
{"timestamp": "...", "flag": "psd_cost_gate", "eligible": true,
 "benefit_fraction": 0.33, "regression_rate": 0.0, "soundness_ok": true,
 "cert_neutral": true, "regime": "bound_changing", "structural_prevalence": 12,
 "scored": 11, "n_soundness_violations": 0, "cert_kind": "objective_only",
 "cert_violations": [], "benefit_instances": ["..."], "notes": ["..."]}
```

A flag is **graduation-eligible** (ready for its flip PR) only after
**`GREEN_STREAK_REQUIRED = 3` consecutive `eligible` verdicts** for that flag in
the ledger — the "3 green nightlies" rule made checkable. Any non-eligible verdict
resets the streak (`green_streak()` in the script). The gate prints the current
streak and flags `→ GRADUATION-ELIGIBLE` when it reaches 3.

## The flip is a separate, reviewed PR (never automated)

The gate produces evidence; it does **not** flip a default. Graduating a flag
(G1.3 PSD, G1.4 R2, G1.5 R4/TD-A) is a separate, reviewed PR that:

1. changes the flag's default in `solver_tuning.py` / the relevant reader
   (default OFF → default ON), keeping `DISCOPT_X=0` as the escape hatch;
2. cites the ledger's 3 green verdicts + the held-out benefit table + the
   cert-panel proof in the PR body;
3. re-runs the measurement of record (`global_opt_baron_vs_discopt.py`, 60 s, 61
   instances) with the new default and reports defaults-only numbers.

**Kill criterion.** Any lost incumbent / soundness violation in an arm → that flag
stays OFF with the instance recorded. That is the gate doing its job, not a plan
failure (plan §3 G1 "Kill").

## Local-full vs CI-subset (the honest corpus constraint)

The held-out arm needs the full ~4,800-instance MINLPLib snapshot in
`~/Dropbox/projects/discopt-minlp-benchmark/`, which **GitHub CI does not have**.
So a corpus-wide nightly **cannot** run in GitHub Actions. The split:

- **Local / nightly (corpus machine)** — the *full* gate: held-out arm +
  cert-neutrality + ledger append. Run it with:

  ```bash
  make graduation-gate           # all parked flags, N=100, seed 0, 30 s/solve
  # or one flag:
  make graduation-gate GRAD_FLAGS=psd_cost_gate GRAD_N=100
  ```

  Schedule it via cron on a machine with the corpus (from the repo root):

  ```cron
  17 4 * * *  cd /path/to/discopt && PYTHONPATH=$PWD/python JAX_PLATFORMS=cpu \
    JAX_ENABLE_X64=1 make graduation-gate >> reports/graduation-nightly.log 2>&1
  ```

  Only the full gate appends to the ledger, so only the full gate accrues the
  3-green streak.

- **CI subset (GitHub Actions)** — `.github/workflows/graduation-gate.yml`
  (scheduled `cron:` + `workflow_dispatch`) runs **only the cert-neutrality +
  `incorrect_count`** portion over the **vendored** cert panel (the 41 instances
  that ARE in the repo), via `graduation_gate.py --ci-subset`. It:
  - proves that turning a parked flag ON does not change a certified objective or
    lose an optimal status on the panel (a regression guard);
  - **skips** the held-out arm (no corpus) and **does not** append a ledger
    verdict (a nightly-only fact).

  Reproduce the CI subset locally with `make graduation-gate-ci`.

## Determinism

Selection is seeded (`--seed`, default 0): the same seed draws the same held-out
sample and the same solve order, so a verdict is reproducible. The ledger records
the seed + N + time-limit with each verdict.

## Files

- `discopt_benchmarks/scripts/generality_sweep.py` — held-out selection + the
  per-flag `ARMS` + `run_arm`/`arm_stats` (arms mode via `--arms`).
- `discopt_benchmarks/scripts/graduation_gate.py` — the gate wrapper (verdict +
  ledger + CI subset).
- `discopt_benchmarks/scripts/check_cert_neutrality.py` +
  `discopt_benchmarks/utils/cert_neutrality.py` — the cert-baseline neutrality
  check the gate reuses.
- `docs/dev/data/cert-baseline.jsonl` — the 41-instance cert baseline (the
  neutrality reference).
- `docs/dev/data/graduation-ledger.jsonl` — the append-only verdict ledger.
- `.github/workflows/graduation-gate.yml` — the CI subset workflow.
- `Makefile` targets `graduation-gate` (full) and `graduation-gate-ci` (subset).
