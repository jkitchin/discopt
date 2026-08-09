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

- [ ] **T1 — Entry experiment: classify the undecided verdicts (NO implementation
  until this reports).** Instrument the LP layer to count, per node, which outcome
  fired: `Optimal`, `Infeasible`+certified, `Infeasible`-claimed-but-uncertified,
  `Numerical` from phase-1 residual, `Numerical` from the post-optimal feasibility
  audit, `Numerical` from a factorization/linsolve error, `IterLimit`. Use the
  existing `profile.rs` counter facility (`DISCOPT_PROFILE`) so this ships as
  permanent instrumentation rather than a throwaway patch. Report the histogram for
  W1, W2, W3.
  **Kill criterion:** if "infeasible-but-uncertified" is under 20 % of undecided
  nodes on all three witnesses, the Farkas hypothesis below is wrong — stop, record
  the falsification in `performance-plan.md`, and re-scope T2 to whatever the
  histogram actually shows (most likely conditioning of the assembled LP).
  **Done when:** the histogram is recorded here for W1/W2/W3 and the kill criterion
  has been evaluated in writing.

- [ ] **T2 — Fix the dominant cause the histogram names.** The leading hypothesis,
  from reading `farkas_ray_certifies_cols`: it bails (`open = true`) whenever the
  ray selects a column whose relevant side is at the `1e20` sentinel and it cannot
  recover a finite bound — it recovers such bounds for *slack* columns only
  (`slack_upper_bounds`), so a ray touching an unbounded **structural/auxiliary**
  column is rejected and the honest-but-useless `Numerical` is returned instead of a
  proof. The lifted relaxations here are full of aux columns that start at ±1e20.
  Candidate fix: recover a finite implied bound for structural columns the same way
  (from their defining rows / the node box, which the caller already holds), so a
  genuine infeasibility still certifies.
  **Constraint:** the certificate must remain rigorous — the recovered bound must be
  a proven superset bound, established by the same directed-rounding argument the
  slack recovery uses. Widening what counts as a certificate without proving it is
  the one thing this plan may not do.
  **Done when:** the T1 histogram re-run shows the dominant bucket materially
  reduced on at least two of the three witnesses, with zero new `Infeasible` claims
  that a brute-force feasible-point search can contradict (see T5).

- [ ] **T3 — Give the Python cold path a verdict at all.** `_node_bound`'s
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

(Append one entry per completed task: what was measured, what changed, what it cost.)
