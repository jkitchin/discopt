# AVM canonical normal form — replacing the claim federation (issue #632)

**Status:** committed direction (maintainer decision, 2026-07-12) · **Owner issue:**
#632 · **Prereq reading (every executor, every stage):** this file top-to-bottom,
then issue #632, PR #631's description (the collision post-mortem),
`docs/design/relaxation-catalog.md` §3–§4, and CLAUDE.md §Development Philosophy.

This document is written to be executed **stage by stage by a fresh Opus session**
with no other context. Each stage has: verified codebase facts, deliverables, a
test spec, and a gate. Execute stages in order. After each stage, update the
**State ledger** (§9) in the same PR.

---

## 0. Mandate (binding — read this before anything else)

1. **The architecture is decided; do not hedge it.** BARON, Couenne, and SCIP all
   relax through one canonical factorable decomposition with one envelope per atom
   — there is decades of evidence the architecture works. discopt's federated
   claim system (overlapping specialized lift paths arbitrated by a hand-grown
   defer-list) is slower to evolve and demonstrably fragile (PR #631). This plan
   **replaces** the federation with the canonical architecture on the default
   lifted path. There is no kill criterion for the architecture itself: an
   obstacle in a stage is fixed in that stage (or the stage's *design* adapts),
   never routed around by parking the work behind a flag.
2. **No new run-time flags; the flag count goes DOWN.** Maintainer decision: the
   prior flag regime (per-capability default-OFF env flags, byte-identity OFF
   proofs, graduation ledgers, multi-flag interaction matrices) was ineffective
   and confusing, and produced parked capabilities and overlapping claim
   configurations instead of shipped improvements. This plan introduces **zero**
   new `DISCOPT_*` flags and deletes at least two
   (`DISCOPT_UNIVARIATE_ENVELOPE`, `DISCOPT_LOG_MONOMIAL` — their machinery
   becomes always-on *rules* selected by dominance, §2.4). The rollback unit for
   every stage is **`git revert` of its PR**, not an environment variable. Do not
   "helpfully" re-add a flag or a graduation-gate arm; that is a contract
   violation under this plan even though CLAUDE.md §5 describes a flag regime —
   this is a deliberate, maintainer-authorized process deviation for this work,
   recorded here so an executor does not re-litigate it. The *verification
   substance* of CLAUDE.md §5 (differential bound evidence, feasible-point
   sampling, exact neutrality where neutrality is claimed) is kept in full — it
   moves from flag gymnastics into the test suite and per-PR evidence (§3).
3. **Correctness gates are absolute and unchanged.** `incorrect_count ≤ 0` with
   zero slack; certified objectives never change; a dual bound never crosses the
   oracle; never weaken a validation or fallback to pass a gate. A failed
   correctness gate means fix-and-retry within the stage, not descope.
4. **One boundary at all times.** At no point do two claim systems ship
   side-by-side as user-selectable configurations. During the cutover series the
   legacy predicates exist only until the PR that replaces their callers, and
   each cutover PR deletes what it obsoletes. Coexistence lives in *tests*
   (differential harness vs a committed baseline snapshot), not in the product.
5. **General mechanisms only.** Dispatch order is derived from dominance
   invariants (provably at-least-as-tight when applicable), never from instance
   names or shapes discovered through test failures. Named instances (nvs09,
   ex7_2_3, …) are probes.
6. **Workflow.** One stage = one PR series from `main`, task ID in titles
   (`refactor(claims): R2.1 — …`). Every PR: the stage's test spec green,
   `pytest -m smoke`, the adversarial suite
   (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`), and the
   claim-boundary set green **serially** (the R0 job). State what was run and the
   result in the PR body.
7. **Measurement beats plan.** A falsified assumption is recorded in §9 (dated,
   `performance-plan.md` §6 style) and the *design* re-scoped before further code
   — the destination does not change.

## 1. Verified codebase facts (verified 2026-07-12 on `main` ≈ `9937ff7`)

All line numbers are `python/discopt/_jax/milp_relaxation.py` unless another file
is named.

### 1.1 The federation and its arbitration

- Entry point `build_milp_relaxation` (**:5360**). Composite-claim collectors run
  in order: `_collect_univariate_relaxations` (:5999, def :4826) →
  `_univariate_claimed_ids` seed (:6013) →
  `_collect_composite_univariate_relaxations` (:6014, def :3992, H-UNI home) →
  `_collect_aliased_monomial_hull_relaxations` (:6033, def :4207) → H-LOG
  (:6042–6103) → `_multivar_claimed_ids` (:6110) →
  `_collect_composite_multivar_relaxations` (:6111, def :4643) → univariate
  squares (:6133) → finite-domain trig-square table (:6147–6182) → piecewise
  (:6184–6249) → fractional powers (:6256) → lifted products (:6310, :6326) →
  affine-square lift (:6365–6449) → affine-power lift (:6453–6493) → the
  issue-267 ratio/nested-division/univariate-product walk (`_walk_lift` :7095,
  called :7142–7144).
- **The defer-list:** `_should_claim_composite` (**:3586**) +
  `_has_genuine_composite_subterm` (:3658), `_defers_to_finite_domain_trig_table`
  (:3718), `_is_tabulatable_trig_square` (:3690), the multivar twin
  `_should_claim_composite_multivar` (:4437), and the
  `claimed_ids`/`seen`/`_pre_existing_claim` gates inside the collectors
  (:4036–:4137). Every clause is a negative special case discovered through test
  collisions (the comments at :3634–3638 and :3664–3667 say so).
- **Two keying regimes coexist.** The *product* side is structurally keyed and
  collision-free: `bilinear_var_map[(i,j)]`, `monomial_var_map[(i,p)]`,
  trilinear/multilinear/fractional-power/univariate-square maps (:5576–5583).
  The *composite* side is `id()`-keyed: `composite_var_map[id(node)] = col`
  (consulted by `_linearize_expr` :5073 at :5122–5123 and `_decompose_product`
  :1564/:1620). The fragility lives entirely on the id()-keyed side.
- **id()-keying is load-bearing and fragile.** `distribute_products`
  (`term_classifier.py:329`) rebuilds nodes; claims survive only via
  `protected_squares = frozenset(affine_square_protected_ids | composite_var_map)`
  (:7131–7132, applied :7134/:7139); a second distribution pass would orphan
  every key (comment :7114–7130). `_nested_div_keepalive` (:6540) exists solely
  to stop Python recycling claimed ids (the ex7_2_3 false-cache-hit bug class).
  Expressions are id-hashed, not content-hashed (`modeling/core.py:210–227`,
  `:318`); the closest structural key in the codebase is
  `factorable_reform.py:356` `_Lifter._expr_cache` keyed on
  `(repr(expr), lb_floor)` (comment :347–355 explains why `id()` was abandoned
  there).
- **`factorable_reform.py`** does targeted (not canonical) lifting at the solver
  level (`solver.py:3860→3909`): division clearing, repeated-factor monomial
  lifting (:737), blow-up-preventing product lifting (:531), TD-A call powers
  (:596), transcendental-arg lifting (:658), entropy canonicalization. It keeps
  running first; the canonical pass sees its output.

### 1.2 The incremental engine constraint

`IncrementalMcCormickLP` (`incremental_mccormick.py:103`, default-on via
`DISCOPT_INCREMENTAL_MC`, `mccormick_lp.py:574–598`) patches bilinear/monomial
rows per node and validates cold/patch agreement (`_validate` :305 — row-set +
bounds equality vs a fresh full build; mismatch → `ok=False` → sound cold
fallback). Column identities (`mccormick_lp.py:96`) tag structural columns
(`("bilinear",(i,j))`, `("monomial",(i,p))`, …); composite-claimed aux columns
are `("opaque", k)` — position-locked and excluded from root-pool cut
inheritance (`_remap_pool_rows` `mccormick_lp.py:146`). Consequences: (a) any
change to which columns exist flips the structural/opaque partition and can
silently disable the fast path (perf, not soundness); (b) today's composite
columns *already* never inherit pool cuts. `_validate` is a load-bearing safety
mechanism — never weakened; the plan instead **teaches the engine the canonical
layout** (R4) so agreement holds by construction.

### 1.3 The order-mask mechanism (why xdist green ≠ safe)

Claim flags are read fresh from `os.environ` on every call (:3544, :3572,
:4505); some tests write `os.environ` raw (`test_convex_claimer.py:29–39`);
`conftest.py` has no autouse `DISCOPT_*` reset. CI
(`.github/workflows/ci.yml`) runs both Python jobs with `-n 2 --dist loadgroup`
(:176–182, :243–250); there is **no serial Python job**, and
`-m "... not correctness ..."` excludes correctness-marked tests from the
standard path. A leaked flag flips claim behavior for every later test in the
same worker — the #631 experience (test_issue_267 green under `-n4`, red
serially).

### 1.4 H-UNI / H-LOG status

`DISCOPT_UNIVARIATE_ENVELOPE` default-OFF (:3544–3569 records the deferral).
The ON path is sound (PR #631: nvs09 certifies, tree 215→3,
`incorrect_count = 0`; measured gain global50 43→44). Hull builder:
`_collect_composite_univariate_relaxations` + `discopt._jax.univariate_hull`,
guarded against effectively-unbounded boxes (|bound| < 1e19). H-LOG
(`DISCOPT_LOG_MONOMIAL`, :3572) is the positive-product log-space chain, also
default-OFF. Flag-ON tests: `test_lr2_huni_unbounded_guard.py`,
`test_lr2_nvs09_cert.py` (subprocess-isolated), `test_lr2_alias_shape_guard.py`,
`test_lr2_offneutral_relaxation.py`.

### 1.5 Binding prior falsifications (do not re-litigate)

- **Reduced-space is NOT the vehicle.** `maingo-parity-plan.md` §7 P2.4 KILLED:
  reduced-space root bounds tie or lose vs lifted on every measured class. This
  plan canonicalizes the **lifted AVM path itself** — BARON's move. The reduced
  evaluator is reused only as an independent test oracle (§3.3).
- **id()-keyed expression caching across rebuilds is unsound** (ex7_2_3). Any
  new identity must be content-based.
- **H-UNI's tightness is real; its claim boundary was reverse-engineered from
  flaky tests** (PR #631). It graduates here by becoming a rule, not by more
  defer clauses.

## 2. Target architecture

### 2.1 The canonical pass

A **canonicalization pass** over the objective + constraint trees (run once per
model build, cached; per-node work is dispatch only — see §2.5) producing a
content-addressed, hash-consed canonical DAG, plus an **atomizer** that
partitions every nonlinear canonical node into exactly one *atom* with exactly
one *owner* (envelope family). Builders keep their envelope math; the *claim
decision* is centralized in one total dispatch.

Canonical node grammar (immutable, interned; a total ordering on keys makes
every normalization deterministic):

```
ckey := ("var", flat_index)
      | ("const", c)
      | ("sum", ((coef, ckey), …sorted), const)   # n-ary, flattened, folded
      | ("prod", ((ckey, exponent), …sorted))      # repeated factors merged
      | ("pow", ckey, p)
      | ("call", name, ckey)                       # univariate intrinsics
      | ("callN", name, (ckey, …))                 # centropy, min/max, norm, prod
      | ("opaque", token)                          # unsupported: relaxed by the
                                                   #   composed fallback, never rewritten
```

Normalization rules (each a pure rewrite with a property test): sum/product
flattening; constant folding; `neg`/`sub` → coefficients; repeated-factor
merging; `x**1 → x`, `x**0 → 1`; division → `("prod", …, (den, -1))` only when
the denominator is sign-definite on the root box (else `opaque`);
deterministic child ordering. Canonicalization is **box-independent**;
box-dependent decisions (curvature, finite-domain, sign-definiteness) belong to
the dispatcher, which takes the node box as input.

### 2.2 Atom taxonomy (exactly one owner per kind)

| Atom kind (canonical shape) | Owner (existing machinery — reused, not rebuilt) |
|---|---|
| affine | linear rows, no aux (unchanged) |
| `("prod")` of ≥2 distinct unit-exponent factors | bilinear/trilinear/multilinear + RLT hull (already structurally keyed; unchanged) |
| `("pow", var, p)` | monomial-secant / fractional-power lift |
| `("pow", affine, 2)` / `("pow", affine, p≥3)` | affine-square / affine-power lift |
| **univariate atom** = maximal single-variable nonlinear canonical subtree | the univariate envelope dispatcher (§2.4) |
| `("callN","centropy",…)` / certified-convex multivar subtree | composite-multivar gradient cuts |
| positive product `∏ xᵢ^{aᵢ}` (all lb > 0) | H-LOG chain (a rule, no flag) |
| `("opaque", …)` | composed fallback (sound, looser) on that node only |

The **univariate atom** is the load-bearing novelty: today five overlapping
claimers (univariate-of-affine, composite-univariate/H-UNI, univariate-square,
trig table, aliased-monomial-hull) grab fragments and defer to each other; the
atomizer identifies each maximal single-variable nonlinear subtree once, and one
dispatcher chooses its envelope.

### 2.3 Identity: canonical keys replace `id()`

`composite_var_map`, the protected set, and the claim seeds re-key from
`id(node)` to `ckey`. Content-based identity means: claims survive
`distribute_products` by construction (the `protected_squares` plumbing and
keep-alive pinning become deletable); the ex7_2_3 hazard class dies;
structurally identical subexpressions share one aux column (CSE — the
"hash-consed, one aux per elementary op" half of the issue); and column
identities extend to `("canon", ckey)`, converting today's `("opaque", k)`
position-locked columns into remappable identities so root-pool cut inheritance
covers composite columns (R4).

### 2.4 The univariate dispatcher: dominance order, not defer-list

For a univariate atom `u(x)` over the node box (post-FBBT), the first applicable
rule wins; each rule is provably at-least-as-tight as every rule below it
whenever it applies, so the order is an invariant, not an arbitration:

1. **Exact finite-domain table** — `x` integer, `|dom(x)| ≤ cap`: the convex
   hull of the finite graph is exact. Generalizes the trig-square table from
   `sin/cos(affine)**2` to any univariate atom over a small integer domain — the
   special case becomes a theorem instead of a defer clause.
2. **Certified convex/concave on the box** — exact envelope + secant (the
   univariate-of-affine machinery and the monomial/square/fractional kernels are
   instances; a bare `x**p` atom dispatches to the same monomial-secant kernel
   as today, so there is nothing to defer to).
3. **Exact 1-D hull** (the machinery currently behind H-UNI) — neither convex
   nor concave, box effectively finite.
4. **Composed fallback** — hull abstains (unbounded box, hull failure):
   decompose one level, relax the pieces with today's composed envelopes.

Ordering proof obligations (R1 test spec): 1 ⊐ 2 and 1 ⊐ 3 (exactness); 2 = 3
where both apply (a convex function's hull *is* its envelope — assert equality;
2 first because cheaper); 3 ⊐ 4 (the nvs09 measurement). No rule names an
operator except through a mathematical property (integrality, curvature
certificate, boundedness).

### 2.5 Cost model (design decision, not a kill criterion)

`build_milp_relaxation` runs per node. Canonicalization is box-independent, so:
**canonicalize + hash-cons once per model** (after `factorable_reformulate`),
cache the DAG + atom partition on the relaxer, and per node run only the
box-dependent dispatch (rules 1–4 predicate checks) on the cached atoms. Per-node
dispatch must be O(#atoms) with cheap predicates; the existing per-node curvature
/ interval machinery it calls is already per-node cost today.

### 2.6 What this plan does not touch

Envelope kernels (`mccormick.py`, `envelopes.py`, `univariate_hull`), the
product/RLT side's keying, FBBT/OBBT/DBBT, the simplex, branching,
`factorable_reform.py`'s solver-level rewrites, the reduced-space/MCBox line.

## 3. Verification doctrine (this replaces the flag regime)

Safety comes from evidence, not configuration. Three instruments, built in R0
and used by every later PR:

### 3.1 The committed baseline snapshot

Before any behavior change, capture on the in-repo corpus
(`python/tests/data/minlplib_nl/`) + the 41-instance cert panel
(`docs/dev/data/cert-baseline.jsonl`): relaxation fingerprint (the #630
mechanism), root LP bound, certified objective, node count, and solve status →
`docs/dev/data/claim-baseline.jsonl` (committed). This snapshot is what "old
behavior" means once the legacy code is deleted.

### 3.2 The differential gate (every behavior-changing PR)

Against the baseline, partition instances:
- **Unchanged dispatch** (no rule fired differently): fingerprint must be
  **byte-identical**. Any drift is a bug — find it or revert.
- **Changed dispatch**: root bound may move, but (i) certified objective
  identical, (ii) bound sound vs `minlplib.solu` (never crosses the oracle),
  (iii) feasible-point sampling clean (`utils/soundness.py::assert_bound_sound`
  — no valid point cut), (iv) the change is *attributed*: the PR lists which
  dispatcher rule changed the instance's relaxation. Unattributed changes block
  the PR.
- `incorrect_count = 0` over the whole set, both suites (parallel AND serial).

### 3.3 Independent oracles

`minlplib.solu` + the cert panel; the reduced-space evaluator
(`mccormick_subgradient.py`, sound post-#583) as a three-way consistency probe
where it applies (reduced bound ≤ lifted LP bound ≤ oracle optimum); the AVM
claim auditor (R0) asserting **exactly one owner per nonlinear node and zero
defer-clause firings** once the dispatcher owns a shape class.

Rollback for any landed regression: `git revert` the PR. No flag.

## 4. R0 — the correctness net (ships first; everything else depends on it)

**Deliverables**
1. `claim_boundary` pytest marker (registered in `pyproject.toml`) on the
   claim-sensitive files: `test_power_certification.py`,
   `test_centropy_relaxation.py`, `test_lr2_*.py`,
   `test_issue_267_univariate_product_lift.py`, `test_convex_claimer.py`,
   `test_factorable_reform.py`.
2. `ci.yml` job "Python claim-boundary (serial)": `pytest -m claim_boundary -n0`
   — collisions can no longer be order-masked. Keep it <10 min.
3. Autouse fixture in `python/tests/conftest.py`: snapshot `DISCOPT_*` before
   each test, **fail loudly** on unmonkeypatched leaks. Convert known offenders
   (`test_convex_claimer.py:29–39`) to `monkeypatch.setenv`.
4. **Full-suite serial status measurement** (time-boxed, one session): run the
   PR-fast suite `-n0` at defaults, record pass/fail inventory in §9. Failures
   found are *reported as issues*, not silently fixed in this PR (each is a
   pre-existing latent collision — triage severity first).
5. The claim auditor (`_jax/claim_audit.py`, test-only import): for one build,
   record per nonlinear node which claimers would claim it and who won. Changes
   no behavior; it is the exactly-one-owner assertion for the rest of the plan.
6. The baseline snapshot (§3.1) + the differential-gate harness as a pytest
   utility (`python/tests/support/claim_differential.py` or similar).

**Gate:** serial job green on the marker set; leak fixture green on the marker
set; baseline committed; auditor demonstrated on 3 corpus instances.
**PR series:** `ci(claims): R0.1 serial job + leak guard`,
`test(claims): R0.2 auditor + baseline + differential harness`.

## 5. R1 — canonical core + risk-first vertical slice

Two half-stages; R1.2's measurements come before any broad cutover.

### R1.1 The module (library + tests, nothing wired)

`python/discopt/_jax/canonical_expr.py`:

```python
@dataclass(frozen=True)
class CNode: ...                    # kind, children, payload; interned
class CanonicalDAG:
    roots: list[CNode]
    of: dict[int, CNode]            # id(orig node) -> CNode for this build's pinned trees
def canonicalize(model) -> CanonicalDAG            # pure, box-independent, cached per model
def atomize(dag, model) -> AtomPartition           # box-independent atom identification
def dispatch(part, box, flat_types) -> ClaimPlan   # per-node: atom -> (owner, column spec)
class UnsupportedCanonicalization(Exception): ...  # -> ("opaque",…), never escapes
```

Generalize `_Lifter._expr_cache`'s structural key into the interning table. The
univariate-atom extraction is a maximal-single-variable-subtree computation on
the canonical DAG (canonicalizing the `_composite_referenced_var` :3507 logic).

**Test spec** (`python/tests/test_canonical_expr.py`):
- *Semantic equivalence*: ≥200 generated expression trees (all supported ops) +
  every in-repo corpus instance; evaluate original vs canonical at 1k random box
  points via the DAG compiler; max error ≤ 1e-12; `opaque` nodes round-trip
  untouched.
- *Idempotence*, *interning/CSE* (`x*y + x*y` → one product node),
  *determinism* (same expression built two ways → equal keys), *refusal*
  (sign-spanning division, unsupported ops → `opaque`).
- *Atomizer* unit tests per §2.2 row (nvs09's `(ln(x-2))**2 + (ln(10-x))**2` →
  one univariate atom; `sin(x)**2` small-int-domain → rule 1; post-reform
  `x**2·y` → product atom) + the §2.4 ordering obligations as sampled-tightness
  property tests (≥20 random atoms per rule pair).
- *Cost*: canonicalize+atomize once ≤ a small fraction of one
  `build_milp_relaxation`; per-node `dispatch` on the largest in-repo instance
  ≤ 5% of its per-node build time.

### R1.2 Vertical slice — measure the two real integration risks NOW

Wire **one atom kind end-to-end** — the univariate atom, since it subsumes the
five colliding claimers — on a branch, and measure before the broad cutover:

1. **Re-keying:** `composite_var_map` keyed by `ckey` (id→CNode bridge for the
   linearizer) for univariate atoms only. Differential gate (§3.2) over the
   corpus: instances with unchanged dispatch must be byte-identical.
2. **Engine engagement:** fraction of panel instances where
   `IncrementalMcCormickLP` fast path stays engaged, before vs after. If
   engagement drops, the fix is R4's identity extension pulled forward — do it,
   don't record-and-proceed.
3. **Panel drift:** cert-panel objectives identical; node-count changes
   attributed per §3.2.

**Gate:** R1.1 spec green; R1.2 differential gate green with every change
attributed; engine engagement not reduced (after the identity fix if needed);
serial marker suite green. Findings recorded in §9.
**PR series:** `feat(claims): R1.1 canonical DAG + atomizer`,
`refactor(claims): R1.2 univariate atoms on canonical dispatch`.
Note R1.2 is a **real cutover of that shape class** — its PR deletes the
univariate fragments of the defer-list it obsoletes
(`_defers_to_finite_domain_trig_table`, `_is_tabulatable_trig_square`,
`_has_genuine_composite_subterm`, the `allow_general` additive clauses), per
§0.4. The hull rule (rule 3) activates here — reaching it no longer reads
`DISCOPT_UNIVARIATE_ENVELOPE` (see R3 for the flag's formal removal), which is
expected to certify nvs09 on the default path in this stage.

## 6. R2 — cutover of the remaining composite claims

Move the remaining id()-keyed claimers onto the ClaimPlan, in dependency order,
one PR each, each PR deleting the arbitration it replaces and passing the
differential gate:

- **R2.1** affine-square + affine-power lifts (`("pow", affine, p)` atoms) —
  removes their `composite_var_map[id]` writes and
  `affine_square_protected_ids`.
- **R2.2** composite-multivar (centropy / norm / convex-sum) —
  `_should_claim_composite_multivar` becomes the multivar atom classification;
  its predicate deletes.
- **R2.3** the issue-267 ratio/nested-division/univariate-product walk — its
  shapes become canonical atoms (`("prod")` with negative exponents where
  sign-definite, univariate atoms otherwise); `_nested_div_keepalive` and the
  `_walk_lift` re-collection passes delete.
- **R2.4** `protected_squares` plumbing removal: with all claims ckey-based,
  `distribute_products` no longer needs the protection set from this path;
  remove the dead wiring (`term_classifier.py` keeps the parameter if other
  callers use it).
- **R2.5** `_should_claim_composite` itself deletes; the auditor's
  zero-defer-firings assertion goes into the serial CI job permanently.

**Gate (per PR and at stage end):** differential gate; auditor exactly-one-owner
+ zero defer firings on the corpus; full PR-fast suite green **serially and in
parallel**; adversarial suite; cert-panel objectives identical;
`incorrect_count = 0`. Stage-end grep gate: no shape-named defer predicate
remains (`git grep` for the deleted symbols is empty).

## 7. R3 — flags become rules; the runway dividends

- **R3.1 Delete `DISCOPT_UNIVARIATE_ENVELOPE` and `DISCOPT_LOG_MONOMIAL`.**
  Their machinery is now rules 3 and the H-LOG row of §2.2, always on where the
  dominance order selects them. Update the flag-ON test files to default-path
  tests; update `relaxation-catalog.md` §3–§4 and the :3544 docstring (rule, not
  deferral). **Issue-#632 acceptance check, verified in this PR:** (a) no
  instance-shape defer-list exists (CI-enforced by the auditor); (b) full suite
  green serially; (c) ≥1 added certification vs the R0 baseline (nvs09 is the
  known candidate; report the global50 count) with `incorrect_count = 0` on the
  measurement of record (`global_opt_baron_vs_discopt.py`, 60 s, defaults).
- **R3.2 Generalize rule 1** (finite-domain table beyond trig-squares) — now a
  small dispatcher change; differential-gated like everything else.
- **R3.3 Corpus sweep of record:** the full-corpus sample sweep
  (`generality_sweep.py` machinery, defaults-only — no arms) proving
  `incorrect_count = 0` and reporting cert/bound deltas vs baseline. This is the
  evidence a release note cites.

## 8. R4 — fragility dividends (the "long runway" payoff)

With one canonical identity for every column:

- **R4.1** `col_identities` gains `("canon", ckey)` for all composite columns
  (if not already pulled into R1.2); root-pool cut inheritance
  (`_remap_pool_rows`) covers composite columns — measure inherited-row counts
  before/after on the panel.
- **R4.2** Teach `IncrementalMcCormickLP._build_structure` the canonical layout
  (structure map derived from the ClaimPlan) so cold/patch agreement holds by
  construction instead of by probe-box validation; `_validate` stays as the
  safety net (never weakened), but its decline rate on the panel should go to
  ~0 — measure and report.
- **R4.3** Delete the remaining dead plumbing the cutover obsoleted and file the
  follow-up issue list (e.g. piecewise/trig-piecewise migration onto atom
  dispatch, canonical keys for the product side's stage maps) — each a small,
  differential-gated change on the now-single boundary.

**Gate:** differential gate; engine decline rate and pool-inheritance coverage
reported with numbers; full suites green both orders.

## 9. State ledger (update in every stage PR)

| Stage | Status | Evidence / notes |
|---|---|---|
| R0.1 serial CI + leak guard | not started | |
| R0.2 auditor + baseline + differential harness | not started | |
| R1.1 canonical DAG module | blocked on R0 | |
| R1.2 univariate-atom cutover (risk slice) | blocked on R1.1 | engine engagement + panel drift measured HERE |
| R2.1–R2.5 remaining cutover | blocked on R1.2 | one PR each; each deletes what it replaces |
| R3.1 flags→rules + #632 acceptance | blocked on R2 | flag count −2 |
| R3.2 rule-1 generalization | blocked on R3.1 | |
| R3.3 corpus sweep of record | blocked on R3.1 | |
| R4.1–R4.3 identity dividends | blocked on R2 (R4.1 may pull into R1.2) | |
| Full-suite serial inventory (R0 item 4) | not started | pre-existing failures → issues, not scope creep |

Falsifications and design adaptations recorded here as they occur (dated,
`performance-plan.md` §6 style). Reminder (§0.1): an adaptation changes *how*,
never *whether*.
