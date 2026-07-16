# #517 hda-class no-bound — root cause is in-house simplex robustness, not relaxation coverage (2026-07-16)

**Issue:** #517 (re-scoped by TD-B,
`uncertified-tail-plan-results-2026-07-06.md` §5). Goal: give the hda-class
flowsheets a **first finite dual bound**.

This record supersedes the original #517 framing ("lift `log()`/`x**expr` for
relaxation coverage") *and* a rejected fix attempt ("subtol-fold + external-HiGHS
node-LP rescue"). Both are wrong levers. The measured blocker is the **in-house
Rust simplex failing to solve hda's root relaxation LP** — a core-engine numerics
problem, not a coverage or reformulation gap.

## 0. What went stale, and what was rejected

* **Original framing (coverage):** the #632 cutover to the uniform factorable
  engine (`uniform_relax.build_uniform_relaxation`) already covers every hda atom
  (`log(<expr>)`, `x**expr`, divisions, fractional powers) with sound
  interval-floor envelopes. hda's 23 formerly-omitted rows are gone. Coverage is
  **not** the blocker.
* **Rejected fix (external-HiGHS rescue):** a `DISCOPT_NODE_LP_RESCUE` flag that,
  when the in-house node-LP chain returned no certified verdict, re-solved the
  relaxation with SciPy-bundled HiGHS and kept only a Neumaier–Shcherbina safe
  bound from its duals. **Rejected and not shipped.** It violates the stated
  architecture (CLAUDE.md: "the default per-node LP engine is the in-house Rust
  simplex … not HiGHS; do not plan work against a HiGHS backend"; highspy is
  OA/GDP-only). It was also the *entire* cause of a measured 7× per-node
  regression on heatexch_gen3 (37 s → 258 s, still no bound), and it produced
  hda's bound *only* in combination with a subtol-fold, i.e. never through the
  in-house engine. A companion `DISCOPT_RELAX_SUBTOL_FOLD` flag is likewise not
  shipped: without the rescue it yields hda no bound, and its other measured
  panel effects were entangled with the rescue, so shipping it alone would be a
  bound-changing lever with no independently-demonstrated benefit (Dev-Philosophy
  #4).

## 1. Root cause (measured by in-engine instrumentation on the *real* solve)

hda's root relaxation LP is **2974 rows × ~1145 cols**, coefficient spread
~1e26 (Arrhenius pre-exponentials `~6.3e10` against `exp(-261.67/x)` envelope
atoms in `[2e-13, 2.4e-12]`, plus LinForm cancellation residue near machine
epsilon). The blocker was pinned by temporary `DISCOPT_LP_DEBUG` `eprintln`
tracing in `primal.rs` (scaling entry, phase-1 residual, phase-2 exit, and the
`assemble()` feasibility audit), driving a real `Model.solve()` — **not** a
captured/replayed LP (an earlier capture-and-replay pass was invalidated: the
captured arrays were a malformed/infeasible subset — HiGHS agreed they were
infeasible — so every replay conclusion from it, including "phase-1
false-certifies infeasible" and "presolve/redundancy is the lever", is
**withdrawn**).

On the *real* per-node LP the trace is unambiguous:

```
phase1 done: m=2974 artificial_infeas_sum=3.5e-15      <- phase 1 CONVERGES (feasible)
phase2 loop -> Numerical                                <- phase 2 breaks down
assemble: audit=Bounds -> final=Numerical obj=1.6e10    <- drifted point rejected
assemble: audit=Rows   -> final=Numerical obj=-2.1e4    <- Ax=b residual rejected
```

So:

1. **Phase 1 succeeds** — the LP is feasible and the engine finds a feasible
   basis (artificial infeasibility ≈ 3.5e-15). Coverage, the equilibration noise
   floor, and row redundancy are all **not** the blocker (all three earlier
   hypotheses falsified).
2. **Phase 2 fails** — the objective-optimizing simplex loop returns
   `LpStatus::Numerical` (a mid-solve `refactorize`/`basic_values` breakdown on
   the ill-conditioned basis), or returns a point that the `assemble()`
   feasibility audit rejects as `Bounds` (a Harris ratio-test excursion past a
   variable bound) or `Rows` (accumulated `Ax=b` drift). The assembled objectives
   are garbage-scale (1.6e10, 5.8e6, −3.6e7), confirming numerical drift.
3. The audit **soundly** downgrades every such solve to `Numerical` — it refuses
   to certify a drifted point (no unsound bound is ever produced; the failure
   mode is strictly *no bound*). Because *every* root-LP solve fails this way,
   the node contributes no dual bound and the tree never fathoms.

The blocker is therefore **phase-2 simplex + factorization robustness on hda's
ill-conditioned basis**, not relaxation coverage, conditioning-of-the-matrix, or
row redundancy.

## 2. Candidate fixes (scoped, not yet implemented — core-engine work)

Since phase-1 already reaches a feasible basis, two sound directions target the
phase-2/audit failure. Both are certificate-critical core-engine changes and must
ship under the bound-neutral regime (Dev-Philosophy #5): `node_count` and
certified `objective` **exactly unchanged** on the certifying panel + `cargo test
-p discopt-core`. Each is its own PR, not a rider on this investigation.

**(A) Safe dual bound from the in-house basis (most tractable, additive) —
IMPLEMENTED (flag `DISCOPT_NODE_NUMERICAL_DUAL_BOUND` /
`SolverTuning.node_numerical_dual_bound`, default OFF).** `primal.rs` `assemble()`
exports the Optimal-style dual candidate on a `Numerical` breakdown;
`solve_lp_warm_std` computes the NS safe bound from it; `MilpRelaxationModel.solve`
stashes it and attaches it as a last-resort floor when the whole in-house chain
produced no bound; `MccormickLPRelaxer._solve_at_node_impl` reports it as a
bound-only node. Measured: hda flag-ON → `bound = -1.80e10` (finite, `≤` optimum
`-5964.53` — its first), flag-OFF → `bound=None` (baseline unchanged); clean
certifying instances (alan, ex1221, nvs05) byte-identical. Tests:
`python/tests/test_issue_517_numerical_dual_bound.py`.

*Extension (#362, 2026-07-16):* the same stashed NS bound is also surfaced as
`safe_bound` on an `optimal` generic-path solve (which computes no certificate
of its own), closing the nvs05 taint-at-the-certification-edge class — see
`docs/dev/nvs05-decline-taint-2026-07-16.md`. With the extension nvs05
flag-ON is no longer byte-identical (it *certifies*: `optimal`, bound 5.47057).
**Graduated default-ON with #362** (same doc, §Graduation: 65-instance panel
0 lost / no loosening, differential GREEN + feasible-point 0 cuts,
graduation_gate eligible=YES); `=0` restores the legacy no-rescue behavior.

The mechanism: a valid *lower* bound needs only a dual point + directed rounding
(Neumaier–Shcherbina), **not** a clean primal. When phase-2 drifts (audit
`Bounds`/`Rows`) or breaks down, the engine still holds a basis whose dual
`y = B⁻ᵀc_B` through the in-repo NS certificate
(`milp_simplex._safe_lp_lower_bound_std`) gives a sound bound where today there is
none — architecture-respecting (own dual, no HiGHS). *Caveat:* the bound is
**loose** (the basis is drifted): it lifts `best_bound` off the `1e30` sentinel
and satisfies #517's literal acceptance but gives little fathoming power on its own
— a tight, useful bound still needs (B). Env-gated probe measurements before wiring
were `-1.447e6` / `-2.650e8` on the raw node LP; the end-to-end driver bound is
looser (`-1.80e10`) as the frontier aggregates post-FBBT/OBBT node boxes.

**(B) Phase-2 / factorization robustness (the deeper fix).** Keep phase-2 from
drifting/breaking down on the ill-conditioned basis — more frequent
refactorization, tighter Harris ratio-test tolerances, or a better-conditioned
basis factorization (the `#557` density-aware LU route, or basis scaling). Higher
blast radius; only if (A) proves insufficient.

Acceptance for #517 (unchanged): hda reports a **finite** dual bound (its first),
`bound ≤ −5964.534084` (the published optimum), with no panel regression.

## 2a. Verification of (A) — differential panel (flag ON vs OFF)

16-instance in-repo panel, tl=25 s, oracle-checked against published optima:

| class | instances | result |
|---|---|---|
| target | **hda** | `None → −1.80e10` (finite, ≤ opt −5964.53) — **first bound** |
| tighter (still sound) | **4stufen** `19713 → 20282` (≤ opt 96908), **casctanks** `−90.182 → −90.179` (≤ opt 9.163) | flag explores a tighter frontier; every bound ≤ its true optimum |
| byte-identical | nvs05, contvar, heatexch_gen1, beuster, tanksize, st_e36, nvs09, st_e11, gkocis, ex1221, alan, nvs21 | inert — the floor fires only on a numerically-failed node LP |

`panel gate: PASS`; **no bound exceeds its true optimum** on any instance with a
known oracle. *Perf caveat:* heatexch_gen3 flag-ON took 240 s vs 32 s OFF (the
extra bounded nodes drive more per-node NS work) — default-OFF, so no CI impact;
a reason the flag stays opt-in until (B) tightens the underlying solve.

## 2b. Full verification (flag ON) — PR #662

| suite | result |
|---|---|
| differential panel (16 inst, ON vs OFF) | `PASS` — all bounds ≤ oracle; 12 byte-identical |
| adversarial recent-fixes (`-m slow`, ON) | 10 passed, 0 failed (hda oracle-checked to −5964.53) |
| `test_issue_517_numerical_dual_bound.py` | 5 passed (hda first-bound, flag-off baseline, alan/ex1221 inert, default-off) |
| smoke (`-m smoke`, ON) | 651 passed, 13 skipped, 0 failed |
| `cargo test -p discopt-core` | 446 + 4 + 1 passed, 0 failed |

Graduation: flag stays **default OFF** (bound-changing regime) until it accrues
green runs and the perf caveat above is understood; a default-ON flip is a
separate change.

## 3. Disposition

* Candidate (A) is **implemented and shipped default-OFF** on this branch (PR
  #662). #517's literal acceptance (a finite first dual bound for hda) is met.
* #517 stays **open** for the *tight*-bound goal — the phase-2/factorization
  robustness blocker in §1 (candidate B) — since (A)'s bound is loose.
* The external-HiGHS rescue and the subtol-fold are **not shipped** (branch reset;
  no source change to the relaxation or solver was retained).
* The root cause was measured with temporary `DISCOPT_LP_DEBUG` instrumentation in
  `primal.rs`, since reverted (no debug code retained). Reproducible from the
  corpus instance `python/tests/data/minlplib_nl/hda.nl` via a real
  `Model.solve()` — do **not** capture-and-replay the node LP (§1).
