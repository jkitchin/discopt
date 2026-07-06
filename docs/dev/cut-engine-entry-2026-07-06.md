# CUT-1 — aggregation / complemented-MIR entry experiment (GO/NO-GO)

**Date:** 2026-07-06
**Task:** CUT-1 (certification-gap-plan §7 Phase 3, part 2 — native Marchand–Wolsey
aggregation / complemented-MIR separator).
**Type:** measurement + tightly-scoped oracle-injection prototype. No solver math
changed; no separator built. Deliverable is a grounded GO/NO-GO verdict.
**Tooling:** pyscipopt 6.2.1 / SCIP 10.0 available and used as the oracle (no
fallback needed). discopt built via `maturin develop --release` on branch
`cut1-aggregation-cmir-entry` (from `origin/main`).
**Script:** `discopt_benchmarks/scripts/cut1_cmir_oracle_injection.py`
(reusable; raw JSON in `discopt_benchmarks/results/cut1_cmir_oracle_injection_*.json`).
**Panel:** nvs17, nvs19, nvs24 — the integer-product family
(`scip-gap-closing-plan.md` §1.3 names aggregation/c-MIR as its cut workhorse).
Oracle optima from `minlplib.solu`: nvs17 −1100.4, nvs19 −1098.4, nvs24 −1033.2.

---

## VERDICT: **NO-GO** (relaxation-mismatch — the capability is not the lever for
discopt's relaxation as-is)

The kill criterion fired on all three instances. Injecting SCIP's *own*
aggregation/complemented-MIR cuts into discopt's root LP closes **≤ 1.8%** of the
root gap (nvs17), **0%** (nvs19, nvs24) — far below the ~15% GO threshold on ≥2 of
3. More decisively, the injection is testing a relaxation that is *looser than the
one discopt actually uses*: discopt's **default** root relaxation already closes
**99.9%** of the root gap on nvs17/nvs19 — statistically identical to, and on nvs17
slightly **tighter than**, SCIP's fully-cut root bound. There is essentially no
root gap left on this family for an aggregation-c-MIR cut to close. Building the
native separator would be **inert on this class**, exactly as the parked zerohalf
build was on graphpart (§7 "Phase 3 zerohalf — build results").

This does not contradict the 0b GO; it supersedes it with the direct measurement 0b
could not make (0b used SCIP's root bound as a *proxy*; CUT-1 injected SCIP's actual
cut coefficients into discopt's LP and separately measured discopt's real relaxation
strength). Per §0.4, the measurement wins.

---

## 1. Baseline root gaps (the decisive comparison)

discopt's **actual default** root dual bound (`SolveResult.root_bound`, default
`model.solve(max_nodes=1)` path) vs SCIP's root-with-cuts bound (node limit 1) and
SCIP's separators/propagation/presolve-OFF LP floor (the shared "trivial" anchor).
Root gap closed by bound `B` (min sense): `(B − trivial)/(opt − trivial)`.

| instance | opt | trivial (SCIP LP floor) | **discopt root** | SCIP root (all cuts) | **discopt gap-closed** | SCIP gap-closed | discopt − SCIP |
|---|---:|---:|---:|---:|---:|---:|---:|
| nvs17 | −1100.4 | −6404.5 | **−1105.89** | −1105.10 | **0.9991** | 0.9991 | **−0.79** (tighter) |
| nvs19 | −1098.4 | −7174.5 | **−1104.24** | −1104.48 | **0.9990** | 0.9990 | **+0.24** (parity) |
| nvs24 | −1033.2 | −25156.4 | *does not converge at node 1 in 280 s* | −1037.34 | — | 0.9998 | — |

**Reading:** on nvs17 and nvs19 discopt's root relaxation is already at SCIP's
fully-cut root (nvs17 is 0.79 *below* SCIP's bound). discopt reaches this WITHOUT
any MIR/aggregation cut, via its default spatial machinery: piecewise-McCormick
bilinear/monomial envelopes + on-demand multilinear hull + RLT-level-1 + PSD/minor
cuts + iterated lifted-FBBT box tightening (the root reduce↔relax↔separate
fixpoint). discopt's relaxation on this MIQCQP family is *structurally stronger*
than SCIP's, which is why SCIP needs a heavy cut loop to catch up and discopt does
not.

**nvs24 is a different bottleneck — cost, not strength.** discopt's root relaxation
for nvs24 (10 vars → 45 bilinear + 10 square aux columns) does not solve to a tight
bound at node 1 within 280 s. A single separated node solve at SCIP's tightened box
returns −12645 (loose); the tight bound only emerges from the iterated root
fixpoint, which times out. So nvs24's residual is **relaxation-solve throughput /
root-fixpoint cost** (Phase 1/2/4 territory), not a missing cut family. A c-MIR
separator would not touch it.

### Why "discopt's own cuts are net-negative" (reproduced, §1.5)

With `DISCOPT_CMIR_AGGREGATION=1` the root bound on nvs17/nvs19 is **bit-identical**
to cuts-off (−1105.89 / −1104.24). The native aggregation-c-MIR separator (already
built, `crates/discopt-core/src/lp/aggregation.rs`, default-off) finds **nothing to
add**: at discopt's already-tight root LP optimum there is no violated aggregation
c-MIR cut, so it correctly self-disables. The §1.5 "bound worse with cuts on"
result was on the older LP-spatial cut path at a much weaker relaxation; on the
default MIQCQP relaxation the separator is simply inert (no cut, no regression).

## 2. The oracle test (the crux) — SCIP's c-MIR cuts injected into discopt's LP

Method: run SCIP root-only (node limit 1, presolve off so cuts stay in the
un-presolved integer box that matches discopt's), read `getLPRowsData()`, keep the
`cmir`/`agg` separator rows. Resolve each SCIP auxiliary variable
(`auxvar_prod_k`, `auxvar_pow_k`) to its monomial (product `x_i·x_j` or square
`x_i²`) from the McCormick envelope row names, the `minor_<pow_A>_<pow_B>_<prod_C>`
PSD-minor triples (which encode `prod_C = a·b`, `pow_A = a²`, `pow_B = b²`), and a
numeric anchor at the LP optimum. Map each cut into discopt's lifted column space
via the relaxation `varmap` (bilinear `(i,j)→col`, monomial `(i,2)→col`), append
the mapped rows to discopt's root lifted McCormick LP, and re-solve. All three
liftings share the same variable frame (SCIP's presolved integer box), so the cut
is generated and applied in one frame.

| instance | root_off (discopt lifted LP) | root_on (+ SCIP c-MIR cuts) | Δbound | **gap-closed** | SCIP c-MIR cuts | cuts fully mapped |
|---|---:|---:|---:|---:|---:|---:|
| nvs17 | −1811.10 | −1798.43 | +12.67 | **0.0178** | 5 | 2 |
| nvs19 | −4556.70 | −4556.70 | 0.00 | **0.000** | 1 | 0 |
| nvs24 | −12645.33 | −12645.33 | 0.00 | **0.000** | 1 | 0 |

Injecting the same mapped cuts through discopt's *separated* relaxation
(`MccormickLPRelaxer.solve_at_node(inherited_cuts=…)`, which runs the full
separation chain then appends the cut pool) gives the identical numbers
(sep_gap_closed 0.0178 / 0 / 0).

**Two independent reasons the number is ~0, both pointing to relaxation-mismatch:**

1. **The relaxation these cuts land on is already looser than discopt's real root.**
   The bare/one-shot lifted LP (root_off −1811 on nvs17) lacks discopt's iterated
   RLT/PSD/multilinear/lifted-FBBT rounds; discopt's *real* root (−1105.89) is
   already ~99.9% closed and beats SCIP's cut-root. There is <0.1% of gap for any
   cut to close on the relaxation discopt actually runs — so even a *perfectly*
   injected, perfectly-mapped SCIP c-MIR cut has almost nothing to bite on.
2. **SCIP's c-MIR cuts are expressed over SCIP's own auxiliary lifting, which is a
   different envelope than discopt's.** SCIP generates very few c-MIR rows on this
   family (5 / 1 / 1) and they combine SCIP's `auxvar_prod`/`auxvar_pow` moment
   variables under SCIP's bound-complementation frame; the fraction that maps
   cleanly into discopt's column space and is violated at discopt's LP point is
   small (2 / 0 / 0 fully mapped). This is the same relaxation-mismatch the zerohalf
   build hit: the cut is valid but *tight, not violated*, at discopt's LP vertex.

The 1c "separator DEPTH, not plumbing" conclusion is refined by CUT-1: for the
nvs17/19/24 MIQCQP family it is **neither depth nor plumbing — it is that discopt's
relaxation is already as strong as SCIP-with-cuts, so there is no gap to separate.**

## 3. SCIP's cut win is SCIP-vs-SCIP, not a discopt gap

Re-ran the `scip-gap-closing-plan.md` §1.2 ablation (SCIP node counts, cuts on vs
off, 120 s):

| instance | SCIP nodes (cuts on) | SCIP nodes (cuts off) | SCIP cut speedup |
|---|---:|---:|---:|
| nvs17 | 57 | 6,796 | 119× |
| nvs19 | 123 | 16,242 | 132× |
| nvs24 | 290 | 70,683 | 244× |

The 97–169× (here 119–244×) node reduction is real **but it is measured against
SCIP's own no-cut LP**, which is far weaker than discopt's default spatial
relaxation (SCIP's trivial floor is −6404 on nvs17 where discopt's root is −1105.9).
Reading that speedup as "discopt is missing 100× of cut strength" was the premise
error: discopt already *starts* where SCIP's cut loop *ends* on this family.

## 4. Feasibility of the native build (part 3) — already built, and inert

The ingredients the §7 build item 2 calls for already exist in the Rust `lp/` layer:

- `crates/discopt-core/src/lp/aggregation.rs` (`separate_aggregation_mir`):
  Marchand–Wolsey 2-row nonnegative-weight aggregation (continuous-cancel target,
  fractional fallback), feeding the existing complemented MIR. Sound (500/3000
  random-system validity tests; no integer-feasible point cut). PR #416.
- `crates/discopt-core/src/lp/mir.rs` (`separate_mir`): full MIR with **upper-bound
  complementation already implemented** (`comp[j]` at upper bound, bound
  substitution, integer-shift guard). The plan's "mir.rs:59 lacks upper-bound
  complementation" note is **stale** — it was completed (PR #415). The δ-scan and
  coefficient snapping are present.
- `crates/discopt-core/src/lp/{gomory.rs,cover.rs,cut_select.rs}`: GMI, cover,
  SCIP-style efficacy/orthogonality selection.

So there is **no 4–6 EW build to green-light** — the separator exists, is sound, and
is wired default-off (`DISCOPT_CMIR_AGGREGATION`, `aggregation_mir_cuts_py`). CUT-1
confirms (§1) it is **measurably inert on the integer-product family**: the
relaxation is already tight, so it emits no cut. The remaining work would be a build
against a lever that this experiment shows is not present.

## 5. Recommendation

1. **NO-GO on a native aggregation/c-MIR *build* for the integer-product family.**
   The capability exists and self-disables because discopt's relaxation already
   dominates SCIP-with-cuts here. Do not schedule the §7 part-2 build against this
   class. Keep the existing default-off separator parked (it is sound and may help a
   class whose LP optimum *is* c-MIR-cuttable — a distinct, unproven target).
2. **Re-aim the residual on this family to where the measurement points:**
   nvs24 shows the real bottleneck is **root-relaxation solve cost / the
   reduce↔relax↔separate fixpoint not converging at node 1 in time** — i.e. Phase 1
   (per-node engine throughput) / Phase 2 (native node loop) / Phase 4 (structure /
   smaller lifted DAG), not cuts. discopt closes 99.9% of the root gap on nvs17/19
   but pays for it in per-node LP cost.
3. **Do not read SCIP's cut ablations as a discopt cut gap.** The 119–244× is
   relative to SCIP's weak no-cut LP; discopt's spatial relaxation is the equivalent
   of "SCIP with its cut loop already run."

## 6. What was run

- `pytest`/`cargo test`: none needed — no solver code changed (measurement-only
  script + this doc). Bound-neutral by construction on the default path.
- Oracle: pyscipopt 6.2.1 / SCIP 10.0 (root LP-row extraction, cut coefficients,
  auxvar identity resolution, ablation node counts).
- discopt: default `model.solve(max_nodes=1)` root bounds; `MccormickLPRelaxer` /
  `build_milp_relaxation` lifted-LP injection; `DISCOPT_CMIR_AGGREGATION` on/off.
