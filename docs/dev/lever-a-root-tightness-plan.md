# Lever-A Root-Tightness Plan — log-space monomial relaxation campaign

**Status:** PROPOSED (evidence gathered 2026-07-11; entry experiment not yet run)
**Maintainer directive (2026-07-11, verbatim intent):** *"we should be able to fix
this Lever A"* and — the binding success criterion — *"more wall time is not the
issue, if it's not fast, it is not working."*
**Audience:** this document is written to be executed by Opus in stages, task by
task, without additional context. Every task names its entry experiment, kill
criterion, verification regime, and definition of done. Do not start a build step
before its entry experiment's verdict is recorded in §7.

This plan is the *execution layer* for the residual certification gap identified
by the V-2 re-measure (`docs/dev/gap-closing-execution-plan.md` §1a) and the
DECOMP-1 decomposition (`docs/dev/certification-effort-decomposition-2026-07-10.md`
§5(ii)). It inherits, verbatim, §0 of `docs/dev/gap-closing-execution-plan.md`
(binding constraints: correctness before performance, fix-the-class, entry
experiment before build, two verification regimes, PR workflow, falsifications
binding, concurrency hygiene). Read that §0 first; it is not repeated here.

---

## §1 The success criterion (what "fixed" means)

BARON certifies the target instances **at or near the root**:

| instance | BARON wall | discopt (2026-07-11 re-measure) | discopt root bound | optimum |
|---|---|---|---|---|
| nvs09    | **0.0 s** (root) | stalls 66 s, `feasible` | — | −43.134 |
| nvs05    | **0.5 s** | stalls 60 s, bound 1.352 (taint floor) | 0.674 | 5.4709 |
| tanksize | **2.6 s** | stalls 60 s, bound 0.868 | 0.847 | 1.2686 |
| tls2     | **0.0 s** (root) | stalls 60 s, dual 91 % | — | 5.3 |
| casctanks | 61.9 s, **NOT certified by BARON either** | stalls | — | 9.1635 |

Therefore:

1.1 **The bar is root-tightness, not tree throughput.** A change qualifies as a
    fix only if the instance certifies in **seconds** (single-digit node counts
    or a root-level certificate), matching the BARON column. "The bound improved
    but the tree still needs minutes" is a **KILL** under the maintainer's
    criterion — record it and stop.
1.2 **casctanks is out of scope.** BARON cannot certify it at TL=60 either
    ("8 Integer Solu"); it is not a discopt-vs-BARON gap. Do not spend budget on
    it in this campaign.
1.3 **tls2 is a different class** (MIP feasibility / integer-linear cut
    strength, §6 F14) and is front LR-T, sequenced last.

The measured target set for the core campaign is **nvs05, nvs09, tanksize**.

---

## §2 Measured evidence (2026-07-11 probes; all reproducible)

All probes below were run on main @ `f4e9f2c8` (worktree, TL as stated,
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`).

2.1 **Operator census (ground truth from `.nl` opcodes, not the modeling layer):**
    - `nvs05` (8 vars, 9 cons): mult ×29, **pow ×11**, div ×5, sqrt ×3 —
      signomial monomials (`x0²·x1`-type; welded-beam). All variable lower
      bounds strictly positive (x0,x1 ∈ [0.01,200], x2,x3 ∈ [1,200]).
    - `nvs09` (10 **integer** vars ∈ [3,9], **0 constraints**): log ×20, pow ×21
      — objective `Σᵢ[(ln(xᵢ−2))² + (ln(10−xᵢ))²] − (Πᵢxᵢ)^0.2`. The coupling
      term is a **10-way positive product**.
    - `tanksize` (47 vars, 74 cons): **mult ×84**, sqrt ×3 — bilinear-dominated.
    - `tls2` (37 vars, 24 cons): mult ×4 only; the difficulty is combinatorial
      (F14), not envelope tightness.
2.2 **The existing strengthening stack is inert or harmful here.** With ALL of
    `DISCOPT_RLT, RLT_QUAD, MULTILINEAR_SEPARATE, EDGE_CONCAVE, SQUARE_SEPARATE,
    TRILINEAR, NODE_REDUCE, LIFTED_FBBT, ROOT_FIXPOINT` ON simultaneously
    (TL=30): **nvs05 byte-identical to OFF** (root 0.674, final 1.352, 659
    nodes); **tanksize strictly worse** (1 node, no root bound, no incumbent —
    the stack path stalls in the dense lift). Graduating existing separators is
    **not** the path. (Consistent with F3 — reduced-space also no-win.)
2.3 **The dense-cell guard is not the binding issue on tanksize.** Raising
    `_MAX_RELAX_DENSE_CELLS` 1e8 → 1e12 in-process: root bound **byte-identical**
    (0.8473), final 0.868, 625 nodes @ TL=60. The lifted McCormick LP, even when
    allowed to build, is simply **loose** (33 % root gap the tree cannot close).
    A sparse-LP data path is therefore NOT the tanksize lever (kills the F12
    re-scope hypothesis for this instance).
2.4 **Taint repair is insufficient for nvs05 under §1.1.** The provable frontier
    reaches 4.875/5.4709 (F9), but that took 659 nodes / 30 s — fails "fast."
    The reported-bound taint floor (1.352, F8) is real but fixing it does not
    meet the bar; only a root bound ≈ 5.47 does.
5. **Why recursive McCormick is the common loose mechanism.** All three
    instances bound positive products/monomials by *recursive* pairwise
    McCormick (`w=x²` secant → `w·y`; 9 chained bilinears for the nvs09
    product). Looseness compounds with recursion depth and box width
    (nvs09: `[3,9]¹⁰` → product range spans 5 decades). BARON's standard
    treatment of positive products is the **exponential/functional transform**
    (log-space), which removes the recursion entirely (see refguide §BARON).

---

## §3 The central hypothesis (H-LOG)

> For a monomial term `t = ∏ᵢ xᵢ^{aᵢ}` in which **every** `xᵢ` has a strictly
> positive lower bound (possibly after root FBBT), the log-space relaxation —
> `s = Σᵢ aᵢ zᵢ` (linear), `zᵢ ↔ xᵢ` linked by the **exact** univariate `ln`
> envelope, `t ↔ s` linked by the **exact** univariate `exp` envelope — is
> dramatically tighter than recursive McCormick, and on nvs05/nvs09/tanksize it
> is tight enough that the root LP certifies within tolerance (BARON-shaped,
> §1.1).

**Evidence for:** §2 census (all three are positive-monomial-dominated); BARON's
root-level certification of exactly these instances with exactly this transform
in its standard kit (Maranas & Floudas 1997 signomial relaxation; BARON
functional transforms — see `docs/dev/reference-solver-guide.md` and
`docs/references.bib`); discopt's own relaxation catalog **scopes this as open
follow-up #114** ("mixed-sign signomial global solver — the log-space
counterpart to the x-space `relax_signomial_multi`", catalog §9), i.e. the
x-space machinery exists (`relax_signomial_multi`, `envelopes.py`) but the
log-space track was never built.

**Soundness sketch (why every piece is rigorous, no approximations):**
- `zᵢ = ln xᵢ` on `[lᵢ, uᵢ]`, `lᵢ > 0`: `ln` is concave ⇒ tangent lines are
  rigorous **over**estimators, the secant is a rigorous **under**estimator.
  Closed-form, exact envelope.
- `s = Σ aᵢ zᵢ`: exactly linear. No relaxation.
- `t = exp(s)` on `s ∈ [s_lo, s_hi]` (computed by interval arithmetic from the
  `zᵢ` ranges): `exp` is convex ⇒ tangents are rigorous **under**estimators,
  secant is a rigorous **over**estimator. Closed-form, exact envelope.
- Both directions (under/over) exist, so the term is usable wherever it appears
  (objective or constraints, either sign).
- Strict positivity is a **hard precondition**, checked on the FBBT-tightened
  root box: if any `lᵢ ≤ 0` the term is skipped (falls back to the existing
  recursive path). No epsilon-shifting, no tolerance games (§0.1/§0.3 of the
  parent plan).

**What H-LOG does NOT cover:** nvs09's per-variable `(ln(xᵢ−2))² + (ln(10−xᵢ))²`
pairs. Interval/lifted composition lets each square independently reach 0, which
costs ~37.9 of bound (analysis: the per-variable exact 1-D envelope min is
3.7866; term-by-term composition min is ~1.9 via the shared secant, 0 via pure
interval). That is hypothesis **H-UNI** (task LR-2): *subtrees that are
functions of a single variable should get exact univariate envelopes, computed
rigorously from verified curvature, instead of composed relaxations.*

---

## §4 Task list

Execute strictly in order. Every task: feature branch + PR, task ID in the
title (`cert:LR-<n> …`), gates per parent-plan §0.5.

### LR-0 — Entry experiment: prototype log-space root LP (GATES EVERYTHING)

**Hypothesis:** H-LOG (§3). **Do not write solver code for this task.** Build a
standalone prototype (scipy `linprog`/`highspy` is fine — it is a probe, not a
ship path) that constructs the log-space root LP for each target instance and
reports the bound.

- Parse the instance (use `from_nl` for structure or hand-transcribe — nvs09 is
  10 vars/0 cons and trivially hand-transcribable; do it first).
- For every positive-monomial term: `z` vars with ln-envelope rows (secant + 3
  tangents at l, mid, u), `s` linear row, `t` with exp-envelope rows (secant +
  3 tangents).
- For nvs09's univariate composites, ALSO build variant (b) with the exact 1-D
  envelope of `(ln(x−2))² + (ln(10−x))²` over [3,9] (computable to rigorous
  tolerance for the probe by fine secant-piecewise construction on verified
  convex pieces) — this measures H-UNI's marginal contribution.
- Constraints of nvs05/tanksize: linearize non-monomial parts with the same
  envelopes discopt already uses (McCormick for genuinely bilinear-of-
  non-positive terms, secants for sqrt) — the probe only needs to be *sound*,
  not beautiful.
- Report per instance: root LP bound vs optimum; % of root gap closed vs the
  measured discopt root bounds (§1 table).

**GO criterion (per §1.1):** on **≥ 2 of the 3** target instances the prototype
root bound is within `1e-4·(1+|opt|)` of the optimum (i.e. a root certificate),
OR within a gap that a trivial tree closes (≤ 10 nodes in a probe-level
branch-and-bound). **KILL criterion:** the log-space root bound still leaves
> 50 % of the root gap on all three — then the transform is not the lever;
record the falsification in §7 + parent-plan §6 and STOP this campaign (the
residual is then genuinely research-grade; report that honestly).

**Also record** (cheap, same probe): nvs09 variant (a) H-LOG-only vs (b)
H-LOG + H-UNI, to size LR-2's necessity.

**Budget:** 1 day. **DoD:** verdict + numbers table committed to §7 of this doc.

### LR-1 — Core build: log-space monomial relaxation in the compiler

**Blocked by:** LR-0 GO. **Flag:** `DISCOPT_LOG_MONOMIAL` (default OFF —
bound-changing, parent-plan §0.4 regime 2).

Build in `python/discopt/_jax/` alongside the existing envelope machinery:

1. **Detection** — in the factorable reform / term classifier, recognize
   monomial terms `∏ xᵢ^{aᵢ}` (reuse `_try_extract_signomial_factors`, which
   already exists for the x-space path) and check strict positivity of every
   factor's lower bound **on the FBBT-tightened root box** (sequencing: FBBT
   runs first; this is the F10-banked engine-robustness ordering).
2. **Lifting** — introduce `zᵢ` (one per distinct positive variable appearing
   in any accepted monomial; shared across terms), `s_k` per term (linear row),
   `t_k` per term with exp envelopes. Tangent counts: start with secant + 3
   tangents per link, refine at the incumbent point per round (the same
   Kelley-round pattern the reduced-space path uses).
3. **Node behavior** — envelopes are box-dependent; recompute tangent/secant
   coefficients from the node box exactly as existing McCormick rows do. The
   positivity precondition re-checks per node (a node box can only shrink, so
   root acceptance is monotone-safe — assert this, don't assume it).
4. **Fallback** — any term failing the precondition uses the existing recursive
   path unchanged. Flag OFF must be **byte-identical** to current main (test).

**Verification (regime 2, all mandatory):**
- Differential bound test on fixed boxes: new root bound ≥ old root bound AND
  ≤ oracle, on the 3 targets + the full 62-instance corpus.
- Feasible-point sampling: no valid point cut (sample ≥ 10³ feasible points per
  instance where obtainable; assert all satisfy the relaxation rows).
- `incorrect_count = 0` on the 62-panel, flag ON.
- Adversarial suite green; smoke green; flag-OFF byte-identity test.
- Unit tests: envelope rows vs closed-form `ln`/`exp` values at box corners and
  random interior points to 1e-12; monotone-shrink assertion.

**DoD:** flag-ON certifies nvs09 and nvs05 (or whichever LR-0 predicted) in
seconds on the dev machine; all gates above green; falsification-or-win recorded.

### LR-2 — Univariate-composite exact envelopes (H-UNI)

**Blocked by:** LR-0 (its variant (b) measurement decides necessity: if
H-LOG-only already certifies nvs09, park LR-2 with the measurement recorded; do
NOT build a dead flag).

Recognize maximal subtrees whose variable support is a single `x`, and replace
their composed relaxation with the exact 1-D convex/concave envelope. Rigorous
construction (no sampled envelopes — §0.1): symbolic second derivative via the
existing autodiff, curvature sign verified by **interval arithmetic** on
subintervals (bisect until sign-definite; refuse loudly past a subdivision
budget); envelope = function on convex pieces + verified connecting secants.
Flag `DISCOPT_UNIVARIATE_ENVELOPE`, default OFF, same regime-2 gates as LR-1.

### LR-3 — Graduation

**Blocked by:** LR-1 (and LR-2 if built). Standard T2.6 rule: **3 consecutive
green verdicts** on independent gate runs (different instance draws/TLs),
`incorrect_count = 0`, zero certificate losses, before any default-ON flip.
Gate panels must include the 62-corpus plus a draw from
`~/Dropbox/projects/discopt-minlp-benchmark/` biased to signomial/product
instances (grep `minlplib_types.csv` for signomial/posynomial classes) so the
win is demonstrated on the *class*, not the three probes (§0.2).

### LR-V — Validation re-measure

**Blocked by:** LR-3. Re-run
`discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py --time-limit 60`.
**Success:** certified-optimal ≥ 46/62 (43 + the 3 targets), violations 0, and
each target's wall in single-digit seconds. Bank the result in
gap-closing-execution-plan §1b and close the campaign task.

### LR-T — tls2 front (separate, last)

Per F14's scoping: MIP-feasibility machinery over a strengthened integer-linear
core — MIP presolve on the linear rows, cut-strengthened root LP (the CUTS-1
c-MIR machinery re-targeted at the integer-linear substructure), LP-based
diving/pump on top. **Entry experiment first:** measure the root LP integrality
gap of tls2's linear core with c-MIR cuts applied (machinery exists, flag-ON
probe); GO only if binaries move materially off ~0.37 toward integrality
(the F14 kill showed rounding never lands in-basin from the current relaxation).
This front is optional for campaign success; sequence it only after LR-V.

---

## §5 Sequencing and budget

```
LR-0 (1 day, gates all) ──GO──► LR-1 (2–4 days) ──► LR-3 (gate runs) ──► LR-V
                        │                 ▲
                        │(variant b)      │ (only if needed)
                        └──────► LR-2 ────┘
KILL at LR-0 ► record falsification, STOP (residual is research-grade; report).
LR-T strictly after LR-V, own entry experiment.
```

Parallelism: LR-0's three instance probes are independent (can be one agent
each); LR-1 is a single build. Do not run gate verdicts concurrently with other
heavy agents on the same machine (§0.7).

---

## §6 Binding falsifications inherited (do not relitigate)

From `docs/dev/gap-closing-execution-plan.md` §6: F3 (reduced-space no-win),
F8/F9 (taint accounting / node retention), F10 (finitize-unbounded no-op),
F12 (FBBT-before-probe inert; **and this plan's §2.3 kills its tanksize
re-scope** — the dense-cell guard is not binding), F13 (ex6_2 joint-OA unsound —
note ex6_2 is a *different* family from this campaign's targets), F14 (tls2
primal wiring). New from this plan's evidence (2026-07-11): **F15 — the existing
cut/strengthening stack, all flags ON simultaneously, is byte-identical-inert on
nvs05 and harmful on tanksize; graduating existing separators is not the Lever-A
path.**

---

## §7 Progress ledger (update per task)

| task | state | verdict/notes |
|---|---|---|
| LR-0 entry experiment | **not started** | — |
| LR-1 log-monomial build | blocked (LR-0) | — |
| LR-2 univariate envelopes | blocked (LR-0 variant b) | — |
| LR-3 graduation | blocked (LR-1) | — |
| LR-V validation | blocked (LR-3) | — |
| LR-T tls2 MIP front | blocked (LR-V) | — |
