# Flag-graduation gate run — G1.3–G1.5 (2026-07-07)

**Task:** roadmap G1.3/G1.4/G1.5 of
`docs/dev/default-path-performance-plan-2026-07-06.md` — run the graduation gate
(`discopt_benchmarks/scripts/graduation_gate.py`, built in G1.2 #533) on the five
parked default-OFF flags, then flip the cleanly-eligible ones to default-ON.

**Verdict: NO flag graduated this run.** All five are INELIGIBLE. The gate did its
job — it exposed a **pre-existing P0 false-optimal on the default path**
(`kall_circles_c8a`, now filed as **C-38** in `docs/dev/correctness-issues.md`)
that contaminates every arm's soundness check, and independently none of the five
flags clears the eligibility bar (regression ≤ 10 % AND cert-neutral AND 0
soundness violations) on this single run. This is the first ledger entry of the
protocol's 3-green streak; every flag's streak is **0/3**.

## Gate configuration

- **Instrument:** `graduation_gate.py --flags root_fixpoint,node_reduce,psd_cost_gate,lift_zero_spanning,lift_loose_products --n 40 --seed 0 --time-limit 25`
- **Held-out sample:** N=40, seed 0, max-vars 500, drawn from the ~4,800-instance
  MINLPLib corpus, excluding the 61 vendored panel instances + the 10 named tuning
  probes (so `st_e36`/`nvs09` — the R4/TD-A probes — are correctly held OUT of the
  arm and used only for the separate fire-check below). Structural prevalence in the
  sample: PSD-target 28/40, reduce-target 39/40, lift-target (no static proxy) 40/40.
- **Per solve:** 25 s time limit, gap 1e-4, isolated subprocess, OFF control shared
  across all five arms (paid once).
- **Cert panel:** the committed 41-instance `docs/dev/data/cert-baseline.jsonl`,
  re-solved in a fresh subprocess with each flag ON.
- **Build:** release (`maturin develop --release`), `pounce-solver` 0.7.0
  (`_pounce.abi3.so` 4.5 MB), `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`.
- **Report:** `reports/graduation_gate_2026-07-07T00-57-47.{json,md}`; verdicts
  appended to `docs/dev/data/graduation-ledger.jsonl` (5 rows, all `eligible:false`).

## Per-flag verdict table

| flag | regime | eligible | benefit_fraction | regression_rate | soundness_ok | cert_neutral | structural_prevalence | scored | notes |
|---|---|:-:|---|---|:-:|:-:|---|---|---|
| root_fixpoint | bound_changing | **no** | 20 % (5/25) | **20 %** (5/25) | N | **N** | 39/40 | regression > 10 %; cert objective drift (st_e38 \|Δobj\|=3.2e-5); flag-caused status downgrade optimal→feasible on kall_circles_c8a |
| node_reduce | bound_changing | **no** | 8 % (2/25) | **20 %** (5/25) | N | **N** | 39/40 | regression > 10 %; cert objective jitter (ex1225 \|Δobj\|=4.8e-8, 30.9999999518→31.0) |
| psd_cost_gate | bound_changing | **no** | 11 % (2/19) | 5 % (1/19) | N | Y | 28/40 | **cleanest**: regression 5 % ✓, cert-neutral ✓; blocked ONLY by the shared OFF-control C-38 soundness failure |
| lift_zero_spanning | bound_changing | **no** | 8 % (2/26) | 8 % (2/26) | N | Y | 40/40 | cert-neutral ✓; benefit/regression are off-structure re-solve noise (structure absent in-sample); fires + decisive on st_e36 (see fire-check); blocked by C-38 |
| lift_loose_products | bound_changing | **no** | 15 % (4/26) | **19 %** (5/26) | N | Y | 40/40 | regression > 10 % (off-structure noise); cert-neutral ✓; fires + tightens on nvs09; blocked by C-38 |

`soundness_ok = N` on **all five** arms because `kall_circles_c8a` (the shared OFF
control) is a false-optimal on the default path — see C-38 below. The flag-specific
disqualifiers (regression, cert drift) are listed above and hold independently of it.

## The blocker: C-38 — false-optimal on the DEFAULT path (`kall_circles_c8a`)

`kall_circles_c8a` was the **only** held-out instance to flag a soundness
violation, and it flagged in **every** arm. In the `node_reduce`, `psd_cost_gate`,
`lift_zero_spanning`, and `lift_loose_products` arms the flag-ON solve is
**byte-identical** to the OFF control (`optimal, obj 3.6142, 3 nodes`), which
proves the fault is on the OFF (default, all-flags-OFF) path, not the flag:

```
default (all flags OFF):  status=optimal  obj=3.6142348  bound=3.6142347  nodes=3
MINLPLib oracle:          =best= 2.5409191   =bestdual= 2.5409129
```

discopt certifies obj ≈ **3.614** as optimal with a dual bound of **3.614** when the
true optimum is **2.541** — the dual bound sits ~42 % **above** the true minimum, an
invalid underestimator, i.e. a genuine false-optimality certificate. Filed as
**C-38 (P0, open)** in `docs/dev/correctness-issues.md`. It is a bounding-strength /
node-bound-validity bug in the QCP (circle-packing) machinery, **out of scope for
this heuristic-flag graduation task**, but it is a hard **blocker**: because the
gate's held-out soundness bracket is checked on the shared OFF control too, C-38
forces `soundness_ok=False` on every arm until it is fixed. Independence from the
flags was verified directly: `psd_cost_gate`/`lift_zero_spanning`/`lift_loose_products`
each reproduce the identical false-optimal (obj 3.6142, 3 nodes); `root_fixpoint`
and `node_reduce` change only node_count/status and do not cure the wrong bound
(`root_fixpoint` in fact makes it worse — optimal→feasible).

## Fire-checks (does the capability execute on the default path?)

Per plan §0.2 (the R2-didn't-fire lesson), a flag that is structurally unreachable
is not a real graduation candidate. All five FIRE:

- **root_fixpoint / node_reduce / psd_cost_gate** — `coverage_map.py` on `ex5_3_3`
  (QCQP, a benefit instance) with all three ON:
  `run_root_fixpoint 1 call (tightened), reduce_node 137 calls (25 tightened),
  psd 1 call`. All three execute and tighten on the default spatial path.
- **lift_zero_spanning (R4) → st_e36** — flag OFF: `feasible, bound −304.49` (root
  pinned, cannot certify). Flag ON: **`optimal`, bound −246.02** — a
  feasible→optimal status upgrade, the documented R4 win. The enabled-check is
  consulted 5× during the solve. (Structure-gated: absent → the reform is
  byte-identical, so the flag is inert off-structure.)
- **lift_loose_products (TD-A) → nvs09** — flag OFF: `feasible, bound −60.38`. Flag
  ON: `feasible, bound −48.80` (same obj −43.13) — a ~19 % tighter root/dual bound;
  `_scan_for_liftable_call_power` hits 42×. The documented nvs09 win.

So the two lift flags are cert-neutral, sound where absent, and deliver a real
structure-gated win where their (rare) structure is present — they are the
"eligible-but-narrow" class the plan anticipated. They remain OFF this run ONLY
because the shared OFF control (C-38) fails the arm's soundness check; once C-38 is
fixed they should re-run and, on a clean streak, graduate (free win on-structure,
inert elsewhere).

## Cert-panel neutrality (per flag, 41-instance panel, flag ON in a fresh subprocess)

- **root_fixpoint** — NOT neutral: 1 objective violation, `st_e38`
  `|Δobj|=3.169e-05` (7197.727117 → 7197.727149). This is a legitimate
  bound-changing effect (the root fixpoint tightens the box and the certified value
  shifts at the 5th decimal), within the *correctness* tolerance (rel 1e-4) but over
  the *reproducibility* baseline tolerance (OBJ_TOL 1e-8). Because the protocol
  enforces an unchanged certified objective for a bound-changing flag, this is a
  cert-neutrality fail. **Not silently rebaselined.**
- **node_reduce** — NOT neutral: 1 objective violation, `ex1225`
  `|Δobj|=4.818e-08` (30.9999999518 → 31.0) — convergence jitter from the tighter
  box (the reduce lets it converge to exactly 31.0). Same tolerance story as above;
  a cert-neutrality fail by the strict baseline tolerance. Not rebaselined.
- **psd_cost_gate / lift_zero_spanning / lift_loose_products** — cert-neutral
  (0 objective violations; the gate reported node-count drift only where the
  structure is present, which is the documented, non-fatal perf note for a
  bound-changing flag).

## Flip decisions

**None.** No flip PR is opened.

- The single-run rule (plan §honest-scope): a flag flips on one validation only with
  a *clean, wide-margin* verdict (0 regression, byte-identical cert, fires). No flag
  meets that this run — every arm has `soundness_ok=False` from C-38, and
  root_fixpoint/node_reduce/lift_loose_products additionally exceed the 10 %
  regression ceiling or fail cert-neutrality.
- **psd_cost_gate** is the closest (regression 5 %, cert-neutral, fires) — but it is
  a single, contaminated run at N=40; per protocol it banks toward the 3-green streak
  and stays OFF pending (a) the C-38 fix and (b) more streak.
- The two lift flags are cert-neutral + sound + fire with a real structure-gated win,
  but likewise blocked by C-38 this run.

## Ledger state

`docs/dev/data/graduation-ledger.jsonl` now has 5 rows (this run), all
`eligible:false`; every flag's `green_streak = 0/3`. This is the first entry of the
protocol's streak. Next actions for a future session:

1. **Fix C-38** (the default-path false-optimal on `kall_circles_c8a`). This is the
   gating item — no bound-changing flag can graduate while the shared OFF control
   fails the arm soundness check.
2. Re-run the gate (ideally with a smaller max-vars or a curated-short sample; see
   the wall note below) to bank green #1 for the clean flags (psd_cost_gate + the two
   lifts) once C-38 is cured.
3. root_fixpoint/node_reduce need their regression rate brought under 10 % (plan G3
   widens their reach) AND a bound-changing-objective policy decision on the cert
   panel (their objective drifts are within correctness tolerance but over the strict
   reproducibility tolerance — a reviewed call, recorded here, not a silent
   rebaseline).

## Honest-scope notes

- **N and wall.** Ran at N=40 (the target), seed 0, tl 25 s. The seed-0 sample pulled
  in many large QCQPs (200–450 vars: qapw, qap, sonet*, autocorr, color_lab*) that
  hit the 25 s time limit in both configs and, worse, several hang past budget to the
  `tl+90 = 115 s` outer timeout — those contribute `neutral` verdicts with no signal
  and dominate wall (~3 h for the full 5-arm run). A future run should cap max-vars
  lower (e.g. 150) or draw from `problems_short.txt` to raise the fraction of
  instances that actually finish and carry signal.
- The off-structure "benefit"/"regression" verdicts for the two lift flags
  (e.g. lift_loose_products flagged crudeoil_pooling_ct1, ex8_1_4) are re-solve
  timing/node jitter, NOT the flag — the lift is provably inert where its structure
  is absent (byte-identical reform). The gate's structural_prevalence for the lift
  arms is 40/40 only because the lifts have no cheap static structural proxy, so the
  arm scores the whole sample; the real reach is the rare on-structure instances
  (st_e36, nvs09-class) exercised in the fire-check.
