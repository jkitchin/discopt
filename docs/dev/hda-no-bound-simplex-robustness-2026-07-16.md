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
VIABILITY CONFIRMED.** A valid *lower* bound needs only a dual point + directed
rounding (Neumaier–Shcherbina), **not** a clean primal. When phase-2 drifts
(audit `Bounds`/`Rows`) or breaks down, the engine still holds a basis and can
export its dual `y = B⁻ᵀc_B`; feeding that through the in-repo NS certificate
(`milp_simplex._safe_lp_lower_bound_std`, already used elsewhere) attaches a sound
bound where today there is none — architecture-respecting (own dual, no HiGHS).

*Probe result (env-gated `DISCOPT_LP_NUMERICAL_DUAL`, since reverted):* on hda's
`Numerical` root-LP solves the exported dual yields a **finite** NS safe bound —
measured `-1.447e6` and `-2.650e8`, both valid (`≤` optimum `-5964.53`). So the
in-house dual survives the phase-2 breakdown well enough to certify hda's **first
finite dual bound**, with no HiGHS. *Caveat:* the bound is **loose** (the basis is
drifted), so it lifts `best_bound` off the `1e30` sentinel and satisfies #517's
literal acceptance but gives little fathoming power on its own — a tight, useful
bound still needs (B). To ship, wire the numerical-path dual → NS bound behind a
flag (bound-changing regime, default OFF) with the differential-bound +
feasible-point + no-fathom gates.

**(B) Phase-2 / factorization robustness (the deeper fix).** Keep phase-2 from
drifting/breaking down on the ill-conditioned basis — more frequent
refactorization, tighter Harris ratio-test tolerances, or a better-conditioned
basis factorization (the `#557` density-aware LU route, or basis scaling). Higher
blast radius; only if (A) proves insufficient.

Acceptance for #517 (unchanged): hda reports a **finite** dual bound (its first),
`bound ≤ −5964.534084` (the published optimum), with no panel regression.

## 3. Disposition

* #517 stays **open**, re-scoped to the phase-2/factorization robustness blocker
  in §1, with candidate fix (A) (safe in-house dual bound) as the next step.
* The external-HiGHS rescue and the subtol-fold are **not shipped** (branch reset;
  no source change to the relaxation or solver was retained).
* The root cause was measured with temporary `DISCOPT_LP_DEBUG` instrumentation in
  `primal.rs`, since reverted (no debug code retained). Reproducible from the
  corpus instance `python/tests/data/minlplib_nl/hda.nl` via a real
  `Model.solve()` — do **not** capture-and-replay the node LP (§1).
