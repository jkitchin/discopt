# Issue #956 follow-through: make undecided node LPs decidable

Status: **in progress** (started 2026-08-09)

## §0 Why this plan exists (binding context)

#956 is fixed — the McCormick envelopes no longer cut their own graph — but the fix
ships **default-OFF** because it measured harmful (`performance-plan.md` §15):
6/6 decisive instances regress in 3/3 interleaved replicates, and on `ex1252` under
the #707 reform flags the undecided-node fraction gets *worse*, 55.3 % → 81.7 %.

The measurements that produced that verdict also produced the explanation, and it
is not about envelopes at all:

* `nvs20`: 46 % of node LPs come back `Numerical`. Guard size is irrelevant to it
  (a 10⁶× larger guard changes nothing); propagation and OBBT were tested and
  excluded.
* `x*y >= c, x + y <= 1` over `[0,1]²` is **never** certified infeasible, for any
  `c` in {0.3, 0.45, 0.49, 0.55, 0.6, 0.75, 1.0, 1.5, 2.0} — ~12 000 nodes to the
  time limit, in both arms.

Reading the code closes the loop between those two symptoms: they are the **same
defect**. `lp_spatial_bb._node_bound` fathoms a node only on

```python
verdict = "fathom" if (_st == "infeasible" and _farkas) else "unresolved"
```

so an LP that is infeasible but whose Farkas ray does not *certify* becomes
`"unresolved"`, folds into `unresolved_lb`, and the final status collapses to
`time_limit` — the tree can never conclude `infeasible`. The Rust tree draws the
same distinction (`verdict_for`: only `LpStatus::Infeasible` is a proof; since #927
everything else branches). So both open items reduce to one question:

> **why do these node LPs fail to produce a certifiable verdict?**

That is the whole subject of this plan. It is also on the critical path for
finishing #956: the guard's measured harm is that it *increases* the undecided
fraction, so if undecided nodes become decided, the harm mechanism may disappear
and the guard can graduate default-ON.

**Verification regime.** Every task here is bound-changing or verdict-changing, so
CLAUDE.md §5 applies: cert-clean AND net-positive, measured with the *interleaved
replicate* method (`solver.py` / #902 — a `max_nodes` budget is unusable because
`node_limit` routes the kernel back to the Python path, so the arms would compare
OFF against OFF). A win must hold in EVERY replicate, a regression in a majority,
and an instance whose replicates disagree is quarantined as unresolved.

**Soundness is not negotiable here, and this plan is unusually exposed to it.**
Every task makes the solver *more willing to declare a region empty*. A wrong
`Infeasible` is a false certificate — the #927 failure mode exactly. No task may
relax a Farkas check by loosening a tolerance; the only admissible fixes are ones
that construct or verify a certificate more completely.

## §1 The three witnesses

Referenced by task ID below; all three must be measured for every change.

| id | witness | current behaviour |
|---|---|---|
| W1 | `x*y >= 0.6, x + y <= 1` on `[0,1]²` (Python path, 0 kernel calls) | ~12 000 nodes, no verdict |
| W2 | `nvs20` (native kernel) | 46 % of node LPs `Numerical` |
| W3 | `ex1252` + `DISCOPT_INTEGER_MULTILINEAR_REFORM=1` + `DISCOPT_MULTILINEAR_COUPLING_RLT=1` | 55 % undecided (82 % with the #956 guard on) |

## §2 Executable task list

Work top to bottom. Each task states its done-condition; do not start a task whose
predecessor's done-condition is unmet. Tick the box and record the measurement
inline when a task completes.

- [x] **T1 — Entry experiment: classify the undecided verdicts (NO implementation
  until this reports).** DONE 2026-08-09 — **kill criterion NOT met; T2's hypothesis
  is falsified.** See §4 log entry T1 for the full histogram. Headlines:
  * `FarkasRejectOpen = 0` on every witness. The ray is *never* rejected for want of
    a finite bound on an unbounded column, so T2 as written was wrong.
  * "infeasible-but-uncertified" is 12 % of undecided verdicts on W2 and 0 % on
    W1/W3 — under the 20 % threshold everywhere, so the plan's own kill criterion
    fires and T2 is re-scoped below.
  * The actual dominant bucket on W2 is `LpAuditBoundsFail` = 1300, **82 % of all
    undecided verdicts**: the simplex reaches "optimal" and the final audit finds a
    basic variable outside its box.
  * Those excursions are NOT near-misses. As a multiple of the audit's own relative
    tolerance: `<10×` 218, `<1e3×` 582, `<1e6×` 488, `≥1e6×` 12 — 83 % are at least
    10× beyond tolerance. **No tolerance change is admissible**; these are wrong
    bases, exactly as §0 requires us to assume.
  * W1 and W3 register **zero** counters on both instrumented paths. Their node LPs
    never reach the Rust simplex at all, which makes T3 (not T2) W1's whole story.

  *(Original scope, for the record: instrument the LP layer to count per node which
  outcome fired — `Optimal`, `Infeasible`+certified, `Infeasible`-but-uncertified,
  `Numerical` from phase-1 residual / from the feasibility audit / from a
  factorization error, `IterLimit` — via the `profile.rs` counter facility so it
  ships as permanent instrumentation, and evaluate the 20 % kill criterion in
  writing. All of that was done; the criterion fired.)*

- [ ] **T2′ — Re-scoped by T1: attack the bound-excursion class, not Farkas.**
  T1 killed the original T2 (`FarkasRejectOpen = 0`, and uncertified-infeasibility
  is only 12 % of the problem). What actually happens on W2 is that the simplex
  declares optimality on a basis that is **primal infeasible**: a basic variable
  sits 10×–10⁶× the audit tolerance outside its box, the audit catches it, and —
  because a bound excursion is the one failure iterative refinement provably cannot
  repair (`discopt#364`) — the whole solve is downgraded to `Numerical`.

  So the node LP is not undecidable; the simplex is *losing* a solution it had.
  Sub-question to settle first (cheap, one more instrumented run): is the offending
  column basic or nonbasic, and is the excursion present at the last
  refactorization or accumulated between them? That distinguishes three fixes:
  a Harris/EXPAND drift reset, a bound-flip cleanup of the offending nonbasics, or
  a forced refactorize-and-reprice before the audit runs.
  **Constraint (unchanged):** no tolerance may be loosened — T1 measured 83 % of
  excursions at ≥10× tolerance, so a tolerance fix would be laundering wrong bases
  into certified bounds.
  **Done when:** `LpAuditBoundsFail` is materially reduced on W2 and the terminal
  histogram shifts from `Numerical` to `Optimal`/`Infeasible`, with T5 green.

- [ ] **T3 — Give the Python cold path a verdict at all (now W1's whole story).**
  T1 raised this task's priority: W1 runs 10 847 nodes and registers **zero**
  verdicts on either Rust simplex path, so its node LPs never even attempt a
  certificate. `_node_bound`'s
  `cut_enabled=False` branch calls `_relax_bound`, which "collapses every failure
  mode into `None` and cannot prove infeasibility", so every node on that path is
  `"unresolved"` by construction. Plumb the same `(status, farkas)` pair the
  incremental path returns through the cold path so a certified-infeasible node can
  fathom there too.
  **Done when:** on W1 the cold path reports certified-infeasible nodes (counted,
  not inferred), and the fraction of `"unresolved"` nodes drops.

- [ ] **T4 — W1 terminates.** `x*y >= c, x + y <= 1` on `[0,1]²` returns
  `infeasible` for `c` in {0.5, 0.6, 1.0, 2.0}, in both guard arms.
  **Done when:** all eight combinations return `infeasible`, and the original
  `_infeasible_bilinear` fixture (`x*y >= 0.5`) is **restored** in
  `test_debug_adversarial.py` — the fixture swap made in the #956 branch exists only
  because the solver could not certify that model, so restoring it is the honest
  regression test for this work. Its docstring's original claim ("infeasible only
  after branching") must then actually hold: assert the solve branches (node count
  > 1) rather than closing at the root on a rounding.

- [ ] **T5 — Soundness gate (blocking, runs with T2/T3).** Every newly-certified
  `Infeasible` must be a true emptiness proof. For a sample of nodes newly reported
  infeasible on W1/W2/W3, verify no feasible point exists by an independent check
  (dense LP via scipy/HiGHS on the same assembled system, plus random/vertex
  sampling inside the node box).
  **Done when:** zero contradictions across the sample, and the check is left in the
  tree as a `#[cfg(debug_assertions)]` / test-only audit so a future regression
  trips it.

- [ ] **T6 — Panel: did it help?** Re-run the CLAUDE.md §5 panel with the guard
  OFF (i.e. measuring T2+T3 alone, against `origin/main`): corpus-wide cert-clean
  check plus interleaved 3-replicate A/B on the decisive instances
  (nvs17/19/20/23/24, tanksize) and W3.
  **Done when:** cert-clean holds and the verdict (win / neutral / regression) is
  recorded here and in `performance-plan.md`. A regression here means T2/T3 revert.

- [ ] **T7 — Re-decide the #956 guard default.** With undecided nodes now decidable,
  re-run the guard's own interleaved-replicate A/B (the table in
  `performance-plan.md` §15) and the W3 undecided-fraction measurement.
  **Done when:** either the guard is cert-clean AND net-positive → flip
  `DISCOPT_ENVELOPE_OUTWARD_ROUND` to default-ON, remove the test-only opt-in seams,
  and #956 is closable; or it is still harmful → it stays OFF and this plan records
  why, with #956 closable on the narrower ground that the defect is fixed and
  available. Either way the outcome is written down, not left implicit.

- [ ] **T8 — Close out.** `cargo test -p discopt-core`, `pytest -m smoke`, the
  adversarial suite, clippy/fmt/ruff. Update `performance-plan.md` §15 with the
  final state, update this file's status line, and post the summary to #956 stating
  explicitly whether it can be closed.

## §3 Observations parked here (not tasks)

* **Budget overrun.** W1 with `time_limit=8.0` returned after **21.1 s** wall. The
  spatial path overruns its own deadline by ~2.6×. Not this plan's subject; if it
  obstructs a measurement, note it and work around it rather than fixing it here.

## §4 Log

### T1 — entry experiment (2026-08-09)

Permanent instrumentation added (`profile.rs` counters + `_rust.profile_counters_py`
/ `profile_reset_py`, so a caller gets the numbers back instead of parsing the
stderr dump): terminal verdict histograms for BOTH simplex entry points, the
`LpInfeasUncertified` bucket, the two audit-failure causes, the two Farkas
rejection reasons, and excursion-magnitude buckets.

W2 `nvs20`, 20 s, 1651 nodes:

| counter | value | share of verdicts |
|---|---|---|
| `LpVerdictOptimal` | 143 | 8.3 % |
| `LpVerdictNumerical` | 1584 | 91.7 % |
| `LpVerdictInfeasible` | 0 | 0 % |
| `LpAuditBoundsFail` | **1300** | 82 % of undecided |
| `LpAuditRowsFail` | 62 | |
| `LpInfeasUncertified` | 190 | 12 % of undecided |
| `FarkasRejectOpen` | **0** | — |
| `FarkasRejectMargin` | 768 | |
| excursion `<10×` / `<1e3×` / `<1e6×` / `≥1e6×` | 218 / 582 / 488 / 12 | |

W1 and W3: every counter **0** — neither witness's node LPs reach the instrumented
Rust simplex paths.

**Falsified:** the T2 hypothesis (Farkas ray rejected for want of a finite bound on
an unbounded structural column). `FarkasRejectOpen` is exactly 0; the rejections
that do happen are margin failures, which is evidence those LPs are not infeasible
rather than that the certificate is incomplete. The plan's kill criterion
("uncertified ≥ 20 % of undecided on some witness") is not met anywhere — 12 % is
the maximum — so T2 is re-scoped to the bound-excursion class per §2 T2′.

**Also learned:** `DISCOPT_PROFILE` costs throughput badly (147k–475k stderr lines
per run, W3 fell from ~6 400 nodes to 31), so node counts measured under it must
never be compared against unprofiled runs. The verdict *shares* are still valid.
