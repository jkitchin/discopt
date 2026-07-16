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

## 1. Root cause (measured, reproducible)

hda's root relaxation LP is **3054 rows × 1145 cols** — grossly redundant (2.7×
more inequality rows than columns; the lifted McCormick / interval-floor
envelopes emit many dependent rows). Its coefficient spread is **4.13e17**
(Arrhenius pre-exponentials `~6.3e10` against `exp(-261.67/x)` envelope atoms in
`[2e-13, 2.4e-12]`).

On this LP the in-house simplex fails at **0 pivots** — it never gets past
phase-1 / initial factorization:

| path | verdict on hda's folded root LP |
|---|---|
| bare cold simplex, un-equilibrated (`solve_lp_py`) | **false `infeasible`**, 0 pivots, 0.2 s |
| bare cold simplex, geometric-mean equilibrated (spread → 1.9e4) | **`numerical`**, 0 pivots, ~14 s |
| per-node path `milp.solve(backend="simplex")` | **false `infeasible`**, ~0.5 s |
| `MilpRelaxationModel._solve_lp_warm_equilibrated()` (the repo's ill-conditioned handler, built for nvs21) | **None** (numerical/iter-limit) after ~55 s |
| SciPy-HiGHS (has presolve) | `optimal` (collapses the redundancy first) |

Equilibration *fixes the conditioning* (4.13e17 → 1.9e4) but the engine still
returns `numerical` at 0 pivots — so conditioning is **not** the whole story.
The distinguishing factor is HiGHS's **presolve**, which removes the redundant /
dependent rows before factorizing; the in-house engine has no such LP presolve
and its (already hardened) phase-1 cannot factor the raw redundant basis.

The in-house phase-1 is already sophisticated for this exact class — Farkas-ray
certification to distinguish genuine from numerical false-infeasible, plus the
C-39 `drive_out_basic_artificials` degenerate-cleanup pass
(`crates/discopt-core/src/lp/simplex/primal.rs:769–851`). hda defeats even that
and hits the honest, non-fathoming `LpStatus::Numerical` return
(`primal.rs:846`). No unsound fathom ever occurs — the failure mode is *no
bound*, never a wrong one (the C-38 no-Farkas-no-prune guard holds). TD-B's
"stock reform yields an infeasible root LP" observation is this same numerical
false-infeasible, not an invalid relaxation.

## 2. The real fix (scoped, not yet implemented — core-engine work)

Give the in-house LP layer a **presolve that removes redundant / linearly
dependent inequality rows** (and empty/singleton rows) before the simplex
factorizes — the step HiGHS uses to make this LP tractable. Candidate home:
`crates/discopt-core/src/lp/` (a new `presolve` alongside `basis.rs`), applied on
the relaxation-LP solve path.

This is a **certificate-critical core-engine change** and must ship under the
bound-neutral verification regime (Dev-Philosophy #5): assert `node_count` and
certified `objective` **exactly unchanged** on the certifying panel for every
instance that already solves, plus `cargo test -p discopt-core`, before it can be
trusted. It is therefore its own PR, not a rider on this investigation.

Acceptance for #517 (unchanged): hda reports a **finite** dual bound (its first),
`bound ≤ −5964.534084` (the published optimum), with no panel regression.

## 3. Disposition

* #517 stays **open**, re-scoped to the in-house-simplex robustness blocker above.
* The external-HiGHS rescue and the subtol-fold are **not shipped** (branch reset;
  no source change to the relaxation or solver was retained).
* Evidence for the rejection and the root cause is this document; the probes are
  reproducible from the corpus instance `python/tests/data/minlplib_nl/hda.nl`.
