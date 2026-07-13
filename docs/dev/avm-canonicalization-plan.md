# AVM canonical normal form — replacing the claim federation (issue #632)

**Status:** committed direction (maintainer decision, 2026-07-12); implementation
workbook (facts re-verified 2026-07-12 on `main` ≈ `9937ff7`) · **Owner issue:**
#632 · **Prereq reading (every executor, every stage):** this file top-to-bottom,
then issue #632, PR #631's description (the collision post-mortem),
`docs/design/relaxation-catalog.md` §3–§4, and CLAUDE.md §Development Philosophy.

This document is written to be executed **stage by stage by a fresh Opus session**
with no other context. Every stage has verified codebase facts (file:line),
PR-level deliverables, a test spec, and a gate. Execute stages in order. After
each stage, update the **State ledger** (§10) in the same PR. §9 records the
corrections applied during plan review — read it so you do not re-introduce a
rejected design.

---

## 0′. Focus refresh — SOTA performance is the north star (maintainer, 2026-07-12)

The goal is **not** a novel relaxation; it is **closing the BARON performance gap
using the techniques SOTA global solvers actually use**, measured (SGM time, node
count, root-gap closure, `incorrect_count=0`). Every design choice is judged
against "is this what BARON/Couenne/SCIP do, and does the measurement improve?"
— not against internal cleverness.

**H-UNI is KILLED (not graduated).** The composite univariate 1-D hull
(`univariate_hull.py`, `DISCOPT_UNIVARIATE_ENVELOPE`, the aliased-monomial-hull
collector) is removed. The falsification chain that led here (record; do not
re-attempt any link):

1. **Grid-sampled hull → dead.** Sampling `f` on a 40 010-pt grid and shifting
   facets by measured deviation is (a) **not how SOTA solvers build envelopes**,
   (b) **unsound between grid nodes** (a 5M-sample probe found real ~1e-6
   violations), and (c) contradicts this project's own lever-a §0.1 ("no sampled
   envelopes"). Not a certificate-path construction.
2. **Analytical composite hull → works, but the scaffolding isn't SOTA.** A
   grid-free construction (interval-verified curvature pieces → tangent/secant
   with a rigorous per-piece validity proof → adaptive outer-approximation
   refinement) is sound and tight (nvs09 `(ln(x-2))²+(ln(10-x))²` loss 0.0). Its
   *machinery* (secants, OA tangent refinement) is SOTA — but the **pre-seeding**
   of a static candidate set (piece midpoints, stationary points, all-pairs
   chords) is a bespoke substitute for lazy separation, and **building a 1-D hull
   of an arbitrary composite at all is not the factorable/AVM approach.**
3. **The SOTA way to recover this tightness is factorable, not a composite hull.**
   BARON/Couenne/SCIP **decompose** `(ln(x-2))²` into atoms (`ln`, then `(·)²`)
   and relax each atom with its closed-form envelope, composed via the
   auxiliary-variable method, then **tighten with outer-approximation cutting
   planes added lazily at the LP relaxation solution** and **branch-and-reduce**
   (OBBT/FBBT/probing). That is where the performance comes from.

**Refocused roadmap (SOTA-performance levers, in leverage order):**

- **P1 — Tight factorable per-atom envelopes (the canonical AVM).** The existing
  R1.1 canonical DAG + atomizer is the substrate. Make each atom's relaxation as
  tight as SOTA: `ln`/`exp`/`sqrt` (secant + tangents), even/odd powers, bilinear
  (McCormick), general monomial/product (recursive McCormick vs. tighter
  simultaneous envelopes), division. The nvs09 root-gap that H-UNI chased is a
  **measurement target for P1**, recovered through composition tightness — not a
  special case.
- **P2 — Outer-approximation cutting planes.** Add gradient/OA cuts for convex
  atoms **lazily at the LP solution** (Kelley), not pre-seeded — the standard
  separation loop. Touches the per-node LP/B&B path.
- **P3 — Branch-and-reduce.** OBBT + FBBT + probing to shrink boxes (BARON's
  actual engine); reduced boxes make the factorable envelopes tight without any
  composite hull.
- **P4 — Structure/convexity detection** to route atoms to the tightest
  applicable envelope.

**What this changes in the stages below:** the R1.2 "univariate cutover / hull
graduation" line of work is **cancelled** — there is no hull to graduate. The
R1.1 canonical core and R0 correctness net stand. R2+ is re-pointed at P1–P4
above. The dominance dispatch (R1.2 decision) keeps only the **table > exact >
composed** tiers; the **hull** tier is deleted with H-UNI. Concrete next actions:
(a) delete the H-UNI code + flag + tests (byte-neutral: H-UNI was default-OFF);
(b) open P1 with a measured per-atom envelope-tightness audit against BARON on
the nvs09/global50 panel.

---

## 0. Mandate (binding)

1. **The architecture is decided; do not hedge it.** BARON, Couenne, and SCIP all
   relax through one canonical factorable decomposition with one envelope per
   atom — decades of evidence say the architecture works. discopt's federated
   claim system (overlapping specialized lift paths arbitrated by a hand-grown
   defer-list) is slower to evolve and demonstrably fragile (PR #631). This plan
   **replaces** the federation on the default lifted path. There is no kill
   criterion for the architecture: an obstacle in a stage is fixed in that stage
   (or the stage's *design* adapts), never routed around by parking work behind
   a flag.
2. **No new run-time flags; the flag count goes DOWN.** Maintainer decision: the
   prior flag regime (per-capability default-OFF env flags, byte-identity OFF
   proofs, graduation ledgers, multi-flag matrices) was ineffective and
   confusing. This plan introduces **zero** new `DISCOPT_*` flags and deletes at
   least two, targeting three (`DISCOPT_UNIVARIATE_ENVELOPE`,
   `DISCOPT_LOG_MONOMIAL`, and — with its own differential evidence —
   `DISCOPT_CONVEX_CLAIMER`); their machinery becomes always-on *rules* selected
   by dominance (§2.4). The rollback unit for every stage is **`git revert` of
   its PR**, not an environment variable. Do not re-add a flag or a
   graduation-gate arm; that is a contract violation under this plan even though
   CLAUDE.md §5 describes a flag regime — a deliberate, maintainer-authorized
   process deviation, recorded here so an executor does not re-litigate it. The
   *verification substance* of CLAUDE.md §5 (differential bound evidence,
   feasible-point sampling, exact neutrality where neutrality is claimed) is
   kept in full — it moves into the test suite and per-PR evidence (§3).
3. **Correctness gates are absolute.** `incorrect_count ≤ 0` with zero slack;
   certified objectives never change; a dual bound never crosses the oracle;
   never weaken a validation or fallback to pass a gate (`IncrementalMcCormickLP
   ._validate`, the trusted-incumbent gate, `gap_certified` downgrade, and every
   conditioning/finiteness cap in §1 are load-bearing). A failed correctness
   gate means fix-and-retry within the stage.
4. **One boundary at all times.** No two claim systems ever ship as selectable
   configurations. Each cutover PR deletes the arbitration it replaces.
   Old-behavior comparison lives in *tests* (the committed baseline snapshot +
   differential harness, §3), not in the product.
5. **General mechanisms only.** Dispatch order derives from dominance invariants
   (provably at-least-as-tight when applicable), never from instance names or
   shapes discovered through test failures.
6. **Workflow.** One stage = one PR series from `main`, task IDs in titles
   (`refactor(claims): R2.1 — …`). Every PR: the stage's test spec,
   `pytest -m smoke`, `pytest -m slow python/tests/test_adversarial_recent_fixes.py`,
   and the claim-boundary set **serially** (`pytest -m claim_boundary -n0`, the
   R0 job) — state what was run and the result in the PR body.
7. **Measurement beats plan.** A falsified assumption is recorded in §10 (dated,
   `performance-plan.md` §6 style) and the *design* re-scoped before further
   code — the destination does not change.

## 1. Verified codebase facts

All anchors `python/discopt/_jax/milp_relaxation.py` (9129 lines) unless named.

### 1.1 Build pipeline and the two keying regimes

- `build_milp_relaxation` (**:5360**) is re-invoked **per node** from
  `MccormickLPRelaxer._solve_at_node_impl` (`mccormick_lp.py:1102–1112`) and
  `_lifted_fbbt_rebuild` (`mccormick_lp.py:1561–1565`), always with the **same
  `Model` object** (`self._model`, set once in `__init__` :419) and a per-node
  `bound_override=(node_lb, node_ub)`. Raw-tree `id()`s are therefore stable
  across nodes; `MccormickLPRelaxer.__init__` (:408) already caches per-model
  structure (`_terms`, `_disc`) and is the home for the canonical-DAG cache.
- Composite-claim collectors, in call order: `_collect_univariate_relaxations`
  (:5999, def :4826) → `_univariate_claimed_ids` (:6013) →
  `_collect_composite_univariate_relaxations` (:6014, def :3992) →
  `_collect_aliased_monomial_hull_relaxations` (:6033, def :4207) → H-LOG
  (:6042–6103) → `_multivar_claimed_ids` (:6110) →
  `_collect_composite_multivar_relaxations` (:6111, def :4643) → univariate
  squares (:6133) → finite-domain trig-square table (:6146–6182) → piecewise
  (:6184–6249) → fractional powers (:6256) → lifted products (:6310/:6326) →
  affine-square lift (:6358–6442) → affine-power lift (:6444–6493) → the
  issue-267 walk (`_walk_lift` :7095, run :7142–7144) → post-lift re-collection
  (:7146–7200).
- **Product side (structurally keyed, collision-free — untouched by this
  plan):** `bilinear_var_map[(i,j)]`, `monomial_var_map[(i,p)]`, trilinear/
  multilinear/fractional-power maps (:5576–5583), RLT specs (:5787–5960).
- **Composite side (id-keyed — the replacement target):**
  `composite_var_map[id(node)] = col`, written at :6014 (composite univariate),
  :6441 (affine-square), :6492 (affine-power), :6806 (ratio, plus
  `composite_coeff_map[eid]=coeff` :6807), :6919 (nested division), and merged
  from the multivar map at :6121. `univariate_var_map` is **doubly keyed**: by
  `id(expr)` AND by the structural `_univariate_signature(func_name, coeffs,
  const)` (:2684; split into `"univariate"`/`"univariate_signatures"` varmap
  keys at :9098–9101) — an existing content-key precedent.
- **The defer-list:** `_should_claim_composite` (:3586),
  `_has_genuine_composite_subterm` (:3658), `_is_tabulatable_trig_square`
  (:3690), `_defers_to_finite_domain_trig_table` (:3718),
  `_should_claim_composite_multivar` (:4437), plus the `claimed_ids`/`seen`/
  `_pre_existing_claim` gates inside the collectors (:4036–:4137).

### 1.2 How claims are resolved (the linearizer contract)

`_linearize_expr` (def :5073) runs on the **distributed** trees
(`distributed_objective` :7133 → used :8987–8995; `distributed_bodies` :7138 →
used :8879/:8889). At the top of every visit it consults, in order:
1. `composite_var_map.get(id(e))` — **unconditional, first, short-circuits**
   (:5122–5127), scaled by `composite_coeff_map.get(id(e), 1.0)`;
2. type dispatch: `FunctionCall` → `univariate_var_map.get(id(e))` (:5149);
   `/` → fractional-power then univariate maps (id and reciprocal-signature
   keys, :5192–5194); `**` → `monomial_var_map[(flat,n)]` (:5216) /
   fractional-power (:5225); `*` → `_decompose_product` (:5251).
`_decompose_product` (def :1558) leaf order: flat index (:1611) →
`univariate_var_map[id]` (:1615) → `composite_var_map[id]` (:1620, **abstains if
`composite_coeff_map[id] != 1.0`**, :1627) → fractional/monomial maps
(:1639–1658).

Claims made on **raw** trees survive into the distributed trees only because
`distribute_products` (`term_classifier.py:329`) is called with
`protected_squares = frozenset(affine_square_protected_ids | composite_var_map)`
(:7131–7132, applied :7134/:7139) — protected ids are returned intact
(term_classifier :359–360, :367–368). The issue-267 walk instead claims on the
already-distributed trees (:7142–7144). A second distribution pass would orphan
every id-key (comment :7114–7130). `_nested_div_keepalive` (:6540) pins a
synthetic `object()` sentinel (:6895–6896) whose `id` names the reciprocal aux
(:6900) purely against id recycling. `build_milp_relaxation` is the **only**
caller in the repo that passes `protected_squares`; ~30 other call sites use the
default `None`.

### 1.3 The incremental engine: what it actually covers today

`IncrementalMcCormickLP` (`incremental_mccormick.py:103`; wired
`mccormick_lp.py:574–602`) patches **only** bilinear (exactly 4 rows each,
:181–185) and monomial (exactly 3 rows, sign-definite root, :186–194) rows;
everything else is frozen into `base_A` at a probe box. `_validate` (:305)
compares patch vs fresh cold build on 6 sign-diverse boxes; any box-varying row
it cannot patch (univariate tangents/secants, composite lines, RLT) makes the
row-set differ → raise → `ok=False` → cold path (the caller comment at
`mccormick_lp.py:578–588` says exactly this). **Consequence: the fast engine
already declines on every instance with any univariate/composite content.**
The canonical cutover therefore cannot regress engine engagement on the
composite class (it is zero today); the pure-product engaged class must stay
byte-identical (automatic — the product side is untouched). Extending the patch
table to canonical atom rows is the R4.2 payoff, not a cutover risk.
Column identities: `column_identities` (`mccormick_lp.py:96`) tags orig/
bilinear/monomial/trilinear/multilinear/fractional_power/univariate_square;
**everything composite is `("opaque", k)`** (:139–142) and `_remap_pool_rows`
(:146) drops any pooled cut row touching an opaque column (:189–195) — composite
columns never inherit root cuts today.

### 1.4 The order-mask mechanism and CI

Claim flags are read fresh from `os.environ` per call
(`_univariate_envelope_enabled` :3569, `_log_monomial_enabled` :3583,
`_convex_claimer_enabled` :4513 — `DISCOPT_CONVEX_CLAIMER`, default OFF). Raw
`os.environ` writes exist in tests (`test_convex_claimer.py:28–39`).
`python/tests/conftest.py` sets JAX env at import (:5–8) and has **no autouse
`DISCOPT_*` guard**. CI (`.github/workflows/ci.yml`): `python-fast` (:113) runs
`-n 2 --dist loadgroup` with `-m "not slow and not correctness and …"`
(:176–187); `python-coverage` same (:243–250). **No serial Python job exists**,
and `correctness`-marked tests are excluded from the standard path entirely.
`pyproject.toml` pytest config at :230–254 (markers :240–254 — no
`claim_boundary` yet; `addopts` :239 carries the default `-m` filter, which an
explicit `-m claim_boundary` on the command line **replaces**, so
correctness/slow-marked claim tests do run in the serial job).

### 1.5 Existing instruments (reuse; gaps named)

- **Fingerprint (#630):** `_relaxation_fingerprint(name)` is **inline test
  code** (`test_lr2_offneutral_relaxation.py:44–78`): SHA-256 over the built
  relaxation's `(_c, _A_ub densified, _b_ub, _bounds, _integrality)`; corpus =
  all 62 `.nl` under `python/tests/data/minlplib_nl/`; comparison is in-process
  OFF-vs-code-absent — **no committed baseline file exists**. (Also: that
  file's :88–92 docstring claims H-UNI is default-ON — stale; the code truth is
  default-OFF at :3569. Fix the docstring in R0.)
- **cert-baseline** (`docs/dev/data/cert-baseline.jsonl`, 41 rows of
  `SolveResult.to_dict()`): has `objective/status/node_count/bound/root_gap`
  but **no root-LP-bound field** — the claim baseline must record it itself.
  Checker: `check_cert_neutrality.py` / `utils/cert_neutrality.py`
  (`check_neutrality`, OBJ_TOL 1e-8; end-to-end results, not matrices).
- **Soundness harness:** `discopt_benchmarks/utils/soundness.py` —
  `assert_bound_sound(relaxer_fn, boxes, oracle_fn, tol=1e-6, *, baseline_fn,
  sense)` and `assert_cut_valid(cut, feasible_points)` (callable/array based, no
  Model). **Not importable from `python/tests`** today (only
  `discopt_benchmarks/tests` has it on sys.path) — the harness needs the same
  sys.path bridge `check_cert_neutrality.py:19–20` uses, or local thin copies.
- **Measurement of record:** `discopt_benchmarks/scripts/
  global_opt_baron_vs_discopt.py` over the vendored corpus;
  "global50" = `[suites.global50]` (`config/benchmarks.toml:82–95`,
  `config/baron_global50.txt`, `--time-limit 60`); `incorrect_count` gates at
  benchmarks.toml:237/254. H-UNI's measured prize: global50 43→44 (PR #631).
- **Flag touch list (complete, verified by grep):**
  `DISCOPT_UNIVARIATE_ENVELOPE` read only at :3569;
  `DISCOPT_LOG_MONOMIAL` only at :3583; `DISCOPT_CONVEX_CLAIMER` only at :4513.
  Docstring refs: :3545, :3551, :3604, :4223; `univariate_hull.py:27`.
  Test setenv sites: `test_lr2_offneutral_relaxation.py:83–84,120–121,139–141`;
  `test_lr2_huni_unbounded_guard.py:42,56,77–79,91`;
  `test_lr2_alias_shape_guard.py:75`; `test_lr2_nvs09_cert.py:78,88,97`
  (subprocess env). Docs: `lever-a-root-tightness-plan.md:181,228`,
  `CHANGELOG.md:74–75`. **Not** in `discopt_benchmarks/` and **not** graduation
  arms in `generality_sweep.py` (`ARMS` :126–154).

### 1.6 AST inventory (canonicalizer input domain, `modeling/core.py`)

Node types: `Constant` (:254, np array value — may be non-scalar), `Variable`
(:269, arbitrary `shape`, flat `_index`), `IndexExpression` (:327 — index may be
int/tuple/**slice/ndarray**, accepted unvalidated), `BinaryOp` (:385, ops
`+ - * / **`), `UnaryOp` (:407, ops **`neg`, `abs` only**), `FunctionCall`
(:421), `CustomCall` (:433 — relaxation/export raise; the canonical `opaque`),
`MatMulExpression` (:466), `SumExpression` (:477, axis reduction),
`SumOverExpression` (:490, n-ary additive), `Parameter` (:1276 — value fixed at
build time; `dag_compiler._snapshot_params` treats it as a compile-time
constant). Expressions are id-hashed (`__hash__ = object.__hash__` :227;
`Variable.__hash__` :318) — no structural `__eq__/__hash__` exists anywhere.
FunctionCall names: 23 unary intrinsics (core.py:555–789) + `prod`,
`norm{1,2,inf,p}` (array arg), binary `min`/`max` (:823/:841), and — **from the
`.nl` import path only** — `atan2`, `signpower`, `entropy`, `centropy`
(dag_compiler.py:185–256). `compile_expression(expr, model) -> fn(x_flat)`
(dag_compiler.py:403) evaluates any node type; memoizes by id (:80/:88) — the
semantic-equivalence oracle for R1. A reusable random-expression generator
template exists at `test_bilevel_symbolic_diff.py:76` (`_random_expr`; no
hypothesis dependency in the repo).

### 1.7 Binding prior falsifications (do not re-litigate)

- **Reduced-space is NOT the vehicle** (`maingo-parity-plan.md` §7, P2.4
  KILLED): ties-or-loses on every measured class. This plan canonicalizes the
  **lifted AVM path**. The reduced evaluator (`mccormick_subgradient.py`, sound
  post-#583) is reused only as an independent bound cross-check.
- **id()-keyed caching across rebuilds is unsound** (ex7_2_3 false cache hit;
  `factorable_reform.py:347–355`). All new identity is content-based.
- **H-UNI's tightness is real; its claim boundary was reverse-engineered from
  flaky tests** (PR #631). It graduates here by becoming a rule.

## 2. Target architecture

### 2.1 The canonical pass

Once per model (after `factorable_reformulate`, cached on `MccormickLPRelaxer`),
build a content-addressed, hash-consed canonical DAG over the objective +
constraint bodies. Per node of the B&B tree, run only box-dependent **dispatch**
over the cached atoms. Builders keep their envelope math; the claim decision is
centralized.

Canonical key grammar (immutable, interned; total order on keys):

```
ckey := ("var", flat_index)
      | ("const", c)                                   # Parameter snapshots too
      | ("sum", ((coef, ckey), …sorted), const)        # n-ary, flattened, folded
      | ("prod", ((ckey, exponent), …sorted))          # repeated factors merged
      | ("pow", ckey, p)
      | ("call", name, ckey)                           # unary intrinsics incl. abs
      | ("callN", name, (ckey, …))                     # min/max/atan2/signpower/
                                                       #   centropy/entropy/prod/norm*
      | ("opaque", token)                              # CustomCall, MatMul, array-shaped
                                                       #   nodes, non-scalar indexing,
                                                       #   sign-spanning division —
                                                       #   relaxed by the existing
                                                       #   fallback path, never rewritten
```

Normalization rewrites (each with a property test): sum/product flattening;
constant folding; `neg`/`sub` → coefficients; `abs` stays a `call`; repeated-
factor merging; `x**1 → x`, `x**0 → 1`; division → `("prod", …, (den, -1))`
only when the denominator is sign-definite on the **root** box (else opaque —
matching `_clear_divisions`' guard); `SumOverExpression` → n-ary sum;
scalar `IndexExpression` → `("var", flat)` via the existing
`_resolve/_get_flat_index` logic; deterministic child ordering by key.
Canonicalization is box-independent except the one root-box division check
(recorded on the CNode; a node whose denominator is sign-definite at the root is
sign-definite on every sub-box, so this is sound and stable). All curvature /
finite-domain / effective-finiteness (`_is_effectively_finite`, |b| < 1e19)
decisions are per-node dispatch inputs.

### 2.2 Atom taxonomy (exactly one owner per kind)

Atomization is **recursive**: inner atoms get aux columns; outer atoms are
functions of original vars *and* inner-atom aux symbols (this is how BARON
decomposes, and it is exactly what the issue-267 walk does by hand today —
`cos(x − x·x)` → product atom `w = x²`, then a univariate atom over the affine
form `x − w`).

| Atom kind (canonical shape) | Owner (existing machinery, reused) |
|---|---|
| affine over vars/aux | linear rows, no aux (unchanged) |
| `("prod")`, ≥2 distinct unit-exponent factors | bilinear/trilinear/multilinear + RLT (untouched) |
| `("pow", var, p)` | monomial-secant / fractional-power lift |
| `("pow", affine, 2)` / `("pow", affine, p≥3)` | affine-square (:6358) / affine-power (:6444) lift |
| `("prod")` with negative exponents, sign-definite denominators | ratio owner (today's :6709/:6809 machinery: fold, reciprocal, McCormick product; keeps the `composite_coeff_map` scalar slot) |
| **univariate atom** — maximal single-variable nonlinear canonical subtree (over an original var or an aux symbol) | the univariate dispatcher (§2.4) |
| `("callN","centropy",…)` / certified-convex multivar subtree (incl. convex sums) | composite-multivar gradient cuts (:4643) |
| positive product `∏ xᵢ^{aᵢ}` (all lb > 0), incl. reform alias defs (`_alias_equality_defs` :4169) | H-LOG chain (:6042; binds the existing alias aux `t_col`, adds z/s columns + rows) — a rule, no flag |
| `("opaque", …)` | existing composed fallback on that node only |

Notes: (i) the H-LOG owner's claim unit is an **aux-defining equality**, so the
canonical pass must cover constraint bodies including reform alias definitions;
(ii) rule 1 of the univariate dispatcher emits an exact **MILP table** (binary
selector columns, integrality flags — see :6162–6168), not just LP rows — the
`ClaimPlan` column spec must carry integrality; (iii) piecewise/trig-piecewise
(:5962–5997, :6184–6249) are additive, column-keyed, not claim-arbitrated —
untouched until R4.3.

### 2.3 Identity and claim resolution

`ClaimPlan` maps **CNodes** (not ids) to `(atom_kind, owner, column_spec,
coeff)`. The bridge to the existing trees is a memoized
`cnode_of(expr) -> CNode` (memo keyed by `id(expr)` **within one build**, safe
because the trees are pinned for the build — the ex7_2_3 hazard applies to
caches that outlive their trees). Both raw and distributed nodes resolve by
content, so the linearizer's consultation becomes `plan.get(cnode_of(e))`
wherever it reads `composite_var_map[id(e)]`/`univariate_var_map[id(e)]` today
(:5122, :5149, :5192, :1615, :1620), preserving the `composite_coeff_map`
scaling and the `_decompose_product` abstain-if-coeff≠1 rule (:1627). CSE is a
consequence: equal content → same CNode → same aux column.

**Distribution protection stays — its inputs change.** If `distribute_products`
rewrote a claimed node (e.g. expanded `(x−3)**2`), the distributed form's ckeys
would no longer match the claim and the aux would go dead (the "silently inert
claim" bug class of `convex-claimer-relaxation.md`). So the
`protected_squares` mechanism (`term_classifier.py:359–368`) is kept, but its
input set is **derived from the ClaimPlan** (`{id(n) for raw nodes n with
cnode_of(n) claimed}`) instead of hand-maintained
`affine_square_protected_ids`-style bookkeeping. What dies: the hand-maintained
sets, the keep-alive pinning (`_nested_div_keepalive` — synthetic sub-atoms get
real CNode names), and every id-keyed claim registry.

Column identities extend with `("canon", ckey)` for composite columns (R4.1),
converting today's position-locked `("opaque", k)` tags into remappable
identities for pool-cut inheritance.

### 2.4 The univariate dispatcher: dominance order, not defer-list

For a univariate atom `u(x)` over the node box (post-FBBT), the first applicable
rule wins; each rule is at-least-as-tight as every rule below it whenever it
applies:

1. **Exact finite-domain table** — `x` integer, `|dom(x)| ≤` the existing cap
   (`_MAX_FINITE_DOMAIN_TRIG_TABLE_VALUES`): convex hull of the finite graph,
   binary selectors (today's :6146–6182 machinery). At R1.2 its *scope* stays
   exactly today's (`sin/cos(affine)**2` via the square/univariate pair);
   generalization to any univariate atom is R3.2.
2. **Certified convex/concave on the box** — exact envelope + secant: the
   univariate-of-affine machinery (:4826), the monomial/square/fractional
   kernels, and `_affine_base_power_curvature` (:3770) are instances of this
   rule; a bare `x**p` atom dispatches to the same monomial-secant kernel as
   today.
3. **Exact 1-D hull** — neither convex nor concave, box effectively finite:
   `univariate_hull_envelope(lo, hi, value_batch)` (`univariate_hull.py:199`;
   abstains by returning `None`). The machinery currently gated by
   `DISCOPT_UNIVARIATE_ENVELOPE`; under the dispatcher it is simply the rule
   for the remaining atoms.
4. **Composed fallback** — rule 3 abstained: decompose the atom one level and
   relax the pieces with today's composed envelopes (sound, looser).

Ordering proof obligations (R1.1 tests): 1 ⊐ 2 and 1 ⊐ 3 (exactness); 2 = 3
where both apply (a convex function's hull *is* its envelope — assert equality
on samples; 2 first because cheaper); 3 ⊐ 4 (the nvs09 measurement). No rule
names an operator except through a mathematical property.

## 3. Verification doctrine (replaces the flag regime)

Three instruments, built in R0, used by every behavior-changing PR:

### 3.1 The committed baseline (`docs/dev/data/claim-baseline.jsonl`)

Producer script `discopt_benchmarks/scripts/gen_claim_baseline.py`; one row per
instance of the 62-file `python/tests/data/minlplib_nl/` corpus:
`{instance, fingerprint, n_rows, n_cols, n_integer_cols, root_lp_bound,
solver_commit}`. `fingerprint` = the extracted #630 hash (§4 R0.3);
`root_lp_bound` = the built relaxation's LP optimum (solve the
`MilpRelaxationModel` with the in-house simplex backend; scipy fallback),
recorded because `cert-baseline.jsonl` does not carry it. End-to-end fields
(certified objective/status/node_count) stay in the existing
`cert-baseline.jsonl` — the two baselines are complementary, not merged.

### 3.2 The differential gate (every behavior-changing PR)

Harness `python/tests/support/claim_differential.py` (plus
`support/__init__.py`; not collected as tests). Against the baseline, partition
the corpus:
- **Unchanged dispatch** (the ClaimPlan matches what the legacy path claimed —
  known from the R0.4 auditor log): fingerprint must be **byte-identical**.
  Any drift is a bug.
- **Changed dispatch**: root LP bound may move, but (i) certified objective
  identical (re-solve; `cert_neutrality.check_neutrality` on the cert-panel
  instances), (ii) root bound sound vs the `minlplib.solu` oracle (never
  crosses), (iii) feasible-point sampling clean (`assert_bound_sound` /
  `assert_cut_valid`, imported via the `check_cert_neutrality.py:19–20`
  sys.path-bridge pattern or thin local equivalents), (iv) every changed
  instance **attributed** in the PR body to the dispatcher rule that changed
  it (from the auditor's ownership diff). Unattributed changes block the PR.
- `incorrect_count = 0` over the affected suites, run **both** `-n 2` and
  `-n0`.

### 3.3 Independent oracles

`minlplib.solu` + cert panel; the reduced-space evaluator as a three-way probe
where it applies (reduced bound ≤ lifted LP bound ≤ oracle optimum); the R0.4
auditor asserting exactly-one-owner and, from R2.5 on, zero legacy-predicate
consultations.

Rollback for any landed regression: `git revert` the PR.

## 4. R0 — the correctness net (ships first)

### R0.1 `claim_boundary` marker + serial CI job

- Add `claim_boundary` to `pyproject.toml` markers (:240–254) and to
  `python/tests/conftest.py::pytest_configure` (:55–72).
- Module-level `pytestmark += [pytest.mark.claim_boundary]` on:
  `test_power_certification.py`, `test_centropy_relaxation.py`,
  `test_lr2_offneutral_relaxation.py`, `test_lr2_alias_shape_guard.py`,
  `test_lr2_huni_unbounded_guard.py`, `test_lr2_nvs09_cert.py`,
  `test_issue_267_univariate_product_lift.py`, `test_convex_claimer.py`,
  `test_factorable_reform.py`.
- New `ci.yml` job `python-claims-serial`, copying the `python-fast` recipe
  (rust-toolchain + rust-cache + uv install + `maturin develop` + the JAX env
  block :183–187) with the pytest line
  `pytest python/tests/ -m claim_boundary -n0 -q --tb=short --timeout=120`.
  **Deliberate decision:** the bare `-m claim_boundary` *overrides* the
  `addopts` filter, so the `slow`/`correctness`-marked claim tests (incl. the
  3×~40 s subprocess tests in `test_lr2_nvs09_cert.py` and the 62-way
  parametrized fingerprint test) run — that is the point. Set
  `timeout-minutes: 30`; record the measured duration in the PR; if it exceeds
  ~15 min, move `test_lr2_nvs09_cert.py` alone to a
  `claim_boundary and slow` nightly split and document it.
- Fix the stale docstring at `test_lr2_offneutral_relaxation.py:88–92` (claims
  default-ON; truth is default-OFF at :3569).

### R0.2 `DISCOPT_*` leak guard

Autouse fixture in `python/tests/conftest.py`: snapshot
`{k: v for k in os.environ if k.startswith("DISCOPT_")}` before each test;
after the test, if the live environment differs from the snapshot, **fail**
with the diff (the fixture itself restores the snapshot so one leak doesn't
cascade). Convert the raw writes in `test_convex_claimer.py:28–39` to
`monkeypatch.setenv`. Run the full PR-fast suite `-n0` once with the guard on;
fix any offender the guard itself flags (mechanical monkeypatch conversions
belong in this PR; behavioral failures do not — see R0.5).

### R0.3 Fingerprint util + baseline

- Extract `_relaxation_fingerprint` (test_lr2_offneutral_relaxation.py:44–78)
  into `python/discopt/_jax/claim_audit.py::relaxation_fingerprint(relax) ->
  str` (same `(_c, _A_ub, _b_ub, _bounds, _integrality)` SHA-256); re-point the
  test at it.
- `gen_claim_baseline.py` + commit `docs/dev/data/claim-baseline.jsonl`
  (fields per §3.1).
- `python/tests/support/claim_differential.py`: load baseline, rebuild, diff,
  partition, oracle checks (§3.2), as reusable helpers a per-stage test can
  call.

### R0.4 Claim auditor

In `claim_audit.py`: an opt-in instrumentation mode (a context manager or an
`audit=` hook threaded into `build_milp_relaxation` — **no behavior change when
off**, asserted by fingerprint equality with the hook absent) that records, per
build: every write to `composite_var_map`/`univariate_var_map` with the writing
site tag (collector name), and counters on each defer-predicate site
(`_should_claim_composite`, `_defers_to_finite_domain_trig_table`,
`_should_claim_composite_multivar`). Output: `{ckey-or-id: owner_tag}` +
defer-fire counts. This log is (a) the "unchanged dispatch" classifier for
§3.2, (b) the ownership diff for attribution, (c) from R2.5, the CI assertion
`defer_fires == 0` and `exactly one owner per nonlinear node`.

### R0.5 Serial inventory (measurement, time-boxed)

Run the PR-fast suite `-n0` at defaults; record the pass/fail inventory in §10.
Pre-existing serial failures are filed as issues with severity triage —
**not** fixed inside R0 (each is a latent collision worth its own diagnosis).

**Gate:** serial job green on the marker set; leak guard green; baseline
committed; auditor no-op-when-off proven (fingerprint equality) and
demonstrated on 3 corpus instances (one with H-UNI content: nvs09; one trig
table: a `sin/cos` integer instance; one pure product).
**PRs:** `ci(claims): R0.1 …`, `test(claims): R0.2 …`,
`test(claims): R0.3+R0.4 …` (R0.5 result lands in whichever PR finishes it).

## 5. R1 — canonical core + univariate vertical slice

### R1.1 `python/discopt/_jax/canonical_expr.py` (library-only)

```python
@dataclass(frozen=True)
class CNode:          # kind: str, children: tuple[CNode,...], payload; interned
class CanonicalDAG:
    def cnode_of(self, expr: Expression) -> CNode      # memoized by id(expr),
                                                        # valid for pinned trees;
                                                        # content-addressed result
def canonicalize(model: Model) -> CanonicalDAG          # objective + constraint
                                                        # bodies + alias defs;
                                                        # box-independent except
                                                        # the root-box division
                                                        # sign check (§2.1)
def atomize(dag: CanonicalDAG, model: Model) -> AtomPartition   # recursive; atoms
                                                        # over vars AND aux symbols
def dispatch(part: AtomPartition, flat_lb, flat_ub, flat_types) -> ClaimPlan
    # per-node; ClaimPlan: CNode -> (atom_kind, owner, column_spec(bounds,
    # integrality), coeff)
class UnsupportedCanonicalization(Exception): ...       # internal; surfaces as
                                                        # ("opaque",…), never raised
                                                        # to the caller
```

Implementation notes (binding): interning table generalizes
`_univariate_signature` (:2684) and `_Lifter._expr_cache`
(`factorable_reform.py:356`); float payloads keyed by exact bit pattern (no
tolerance-keying); `Parameter` → `("const", snapshot)` matching
`dag_compiler._snapshot_params`; the univariate-atom extractor canonicalizes
`_composite_referenced_var` (:3507); array-shaped nodes (`MatMul`,
axis-`SumExpression` over arrays, non-scalar `IndexExpression`, array
`Constant` in non-reducible position), `CustomCall`, `sign`, and anything
unhandled → `("opaque", token)` with the original node retained.

**Test spec** (`python/tests/test_canonical_expr.py`, marked
`claim_boundary`):
- *Semantic equivalence*: extend `_random_expr`
  (`test_bilevel_symbolic_diff.py:76`) to cover the full op set of §1.6 (incl.
  min/max, abs, fractional powers with positive bases, division with bounded
  denominators, Sum/SumOver); ≥200 generated trees + all 62 corpus instances;
  compare `compile_expression(original)` vs `compile_expression
  (reconstructed-from-CNode)` at 1k box points, ≤1e-12; opaque nodes assert
  untouched round-trip.
- *Idempotence*, *interning/CSE* (`x*y + x*y` → one CNode; two structurally
  equal trees built separately → identical CNode object), *determinism*
  (construction-order independence), *refusal* (sign-spanning division,
  CustomCall, ndarray index → opaque).
- *Atomizer*: one unit test per §2.2 row — nvs09's
  `(ln(x-2))**2 + (ln(10-x))**2` → one univariate atom; `sin(3x+1)**2`,
  integer `x∈[0,5]` → univariate atom, dispatch rule 1; post-reform `x**2·y` →
  product atom over `(w, y)` with inner monomial atom; `cos(x − x·x)` →
  recursive: product atom + univariate atom over an aux-referencing affine
  form; a centropy call → multivar atom; an alias `aux == x^0.3·y^0.7` with
  positive lbs → H-LOG atom.
- *Dominance obligations* (§2.4): sampled-tightness property tests, ≥20 random
  atoms per rule pair; rule 2 = rule 3 equality on convex samples.
- *Cost*: `canonicalize`+`atomize` once ≤ 20% of one cold
  `build_milp_relaxation` on the largest corpus instance; `dispatch` per node
  ≤ 5% of the per-node build it will join.

### R1.2 Univariate-atom cutover (the risk slice — a real cutover)

Replace the five univariate claimers' *claim decisions* with the dispatcher;
their **emission machinery stays** (`UnivariateRelaxation` :662,
`CompositeUnivariateRelaxation` :3417, `UnivariateSquareRelaxation`, the trig
table :6146–6182, row emitters :8182–8198):

1. `MccormickLPRelaxer.__init__` builds/caches `CanonicalDAG` +
   `AtomPartition`; `build_milp_relaxation` gains an optional
   `canonical=` argument (relaxer passes it; standalone/test callers get a
   fresh one built internally — same content, so same behavior).
2. In `build_milp_relaxation`, for univariate atoms the ClaimPlan drives which
   node gets which envelope; `_collect_univariate_relaxations`'s affine-arg
   scope, the composite collector's non-affine scope, the square path, and the
   trig table become the *owners invoked by* rules 1–3;
   `univariate_hull_envelope` is invoked for rule 3 with the existing
   `value_batch` closure pattern (:4088–4108) and the existing
   effectively-finite guard as its applicability predicate.
3. Claim resolution: the linearizer/`_decompose_product` consultation sites
   (:5122, :5149, :5192, :1615, :1620) go through `plan.get(cnode_of(e))` for
   univariate/composite lookups (product/monomial/fractional maps unchanged);
   `composite_coeff_map` semantics preserved.
4. Protection derived from the ClaimPlan (§2.3) replaces the univariate
   fragments of the hand-maintained sets.
5. **Delete in this PR:** `_defers_to_finite_domain_trig_table`,
   `_is_tabulatable_trig_square` as a *defer* (its shape test survives inside
   the rule-1 owner), `_has_genuine_composite_subterm`, the `allow_general`
   parameter and additive-claim clauses of `_should_claim_composite`
   (:3639–3654), and `_collect_aliased_monomial_hull_relaxations`' flag gate
   (its hull machinery folds into rule 3). The non-univariate clauses of
   `_should_claim_composite` survive until R2.
6. Rule 3 is now always-on: `_univariate_envelope_enabled()` becomes
   unconditional at the dispatcher (the env read itself is removed in R3.1
   with its tests).

**Expected differential outcome (write into the PR):** pure-product instances
byte-identical; instances with H-UNI-claimable or aliased-monomial atoms change
with rule-3 attribution (nvs09 must certify on the default path — tree ~3
nodes, objective per `minlplib.solu`); trig-table instances byte-identical
(rule 1 scope unchanged); engine-engaged set unchanged (it is disjoint from
univariate content, §1.3 — assert this from the auditor + engine `ok` flags on
the panel).

**Gate:** R1.1 spec green; differential gate green with every change
attributed; nvs09 certified at defaults; adversarial suite; serial + parallel
suites; `incorrect_count = 0`.
**PRs:** `feat(claims): R1.1 — canonical DAG + atomizer + dispatcher`,
`refactor(claims): R1.2 — univariate atoms on canonical dispatch (deletes the
univariate defer clauses)`.

## 6. R2 — cutover of the remaining composite claims

One PR each, dependency order; each deletes what it replaces and passes the
differential gate. Emission machinery and every conditioning/finiteness cap
(`_MONOMIAL_AUX_BOUND_LIMIT`, `_LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE`,
`_RECIP_MIN_DENOM`, `_affine_square_row_ok`) stay exactly as-is.

- **R2.1 affine-square + affine-power atoms.** `("pow", affine, 2)` /
  `("pow", affine, p≥3)` atoms drive the existing lift blocks (:6358–6442,
  :6444–6493; row emitters :8495–8568). Deletes:
  `_collect_affine_squares`/`_collect_affine_powers` as *claim* passes (their
  shape-extraction survives as owner helpers), `affine_square_protected_ids`
  (protection now plan-derived), and the corresponding
  `composite_var_map[id]` writes (:6441/:6492).
- **R2.2 multivar atoms.** `_should_claim_composite_multivar` (:4437) becomes
  the multivar atom classification inside `atomize`; the collector (:4643)
  becomes the owner (curvature certification via `classify_expr` →
  `_multivar_box_curvature` :4516 unchanged — these are per-node dispatch
  predicates). **Convex-sum atoms:** dispatching them unconditionally is what
  deletes `DISCOPT_CONVEX_CLAIMER` (:4513); this is bound-changing on convex
  sums — its own differential evidence, its own attribution list, and the
  `convex-claimer-relaxation.md` Phase-3 test battery (routing / no-double-
  relax / soundness gauntlet) run in this PR. If that battery is not green,
  ship R2.2 with the convex-sum atom mapped to opaque and file the follow-up
  — the flag still dies (the predicate does not consult it; the atom kind is
  just not yet owned).
- **R2.3 ratio/division atoms.** The issue-267 walk (:6709–7093, `_walk_lift`
  :7095, re-collection :7146–7200) becomes recursive atomization: negative-
  exponent product atoms (sign-definite denominators) → the ratio owner
  (fold + reciprocal + McCormick product, keeping `composite_coeff_map`);
  univariate-over-aux atoms → the §2.4 dispatcher (this is what
  `_lift_general_univariate` :6921 does today). Deletes: the walk itself, the
  outer-atom guards (`if eid in composite_var_map` :6734/:6842/:6995), the
  `_nested_div_keepalive` sentinels (:6540/:6895–6900 — synthetic sub-atoms
  now have CNode names), and the double-keyed id writes. The post-lift
  product re-collection (:7146–7200) is replaced by the atomizer's recursion
  (inner product atoms exist before outer atoms by construction).
- **R2.4 protection + registry consolidation.** `composite_var_map`/
  `univariate_var_map` id-registries are gone; the linearizer consults the
  ClaimPlan only. `protected_squares` input = plan-derived raw-node id set
  (§2.3 — the term_classifier mechanism at :359–368 **stays**; only its
  hand-maintained feeders die). The varmap's `"univariate"`/
  `"univariate_signatures"`/`"composite_relaxations"` output keys keep their
  shapes (downstream consumers: `column_identities`, separation, tests) with
  ckeys where ids were.
- **R2.5 delete the defer-list.** `_should_claim_composite` (:3586) and every
  §1.1 defer helper are removed; the auditor's `defer_fires == 0` +
  exactly-one-owner assertions become part of the serial CI job. Stage-end
  grep gate: `git grep -l "_should_claim_composite\|_defers_to_finite_domain\|
  _has_genuine_composite_subterm\|_nested_div_keepalive"` returns only docs.

**Gate (per PR and stage end):** differential gate; auditor assertions; full
PR-fast suite serial + parallel; adversarial; cert-panel objectives identical
(`check_neutrality`); `incorrect_count = 0`.

## 7. R3 — flags become rules; acceptance; the sweep of record

- **R3.1 Delete the flags.** Touch list (complete, §1.5):
  `_univariate_envelope_enabled` (:3544–3569) and `_log_monomial_enabled`
  (:3572–3583) removed with their call sites; docstrings :3604/:4223 and
  `univariate_hull.py:27` updated; H-LOG block (:6042–6103) unconditional
  (it is additive and guarded by `_extract_positive_product`'s strict-
  positivity; its differential attribution = positive-product instances);
  flag-ON test files (`test_lr2_offneutral_relaxation.py`,
  `test_lr2_huni_unbounded_guard.py`, `test_lr2_alias_shape_guard.py`,
  `test_lr2_nvs09_cert.py`) converted to default-path tests (the unbounded-box
  guard and alias-shape guards remain as *rule* tests);
  `relaxation-catalog.md` §3–§4 updated (one dispatch, atom taxonomy);
  `CHANGELOG.md`. `DISCOPT_CONVEX_CLAIMER` already died in R2.2.
  **Issue-#632 acceptance check, verified in this PR body:** (a) no
  instance-shape defer-list (CI-enforced); (b) full suite green **serially**;
  (c) `global_opt_baron_vs_discopt.py --time-limit 60` on the global50 suite:
  certified count ≥ baseline+1 (nvs09 is the known candidate; report the
  actual count) with `incorrect_count = 0`.
- **R3.2 Generalize rule 1.** The finite-domain table extends from
  `sin/cos(affine)**2` to any univariate atom over a small integer domain
  (same cap, same binary-selector emission). Bound-changing; differential-
  gated; attribution = small-integer-domain instances.
- **R3.3 Corpus sweep of record.** `generality_sweep.py`'s held-out machinery
  run **defaults-only** (no arms): a seeded stratified sample of the full
  `~/Dropbox/projects/discopt-minlp-benchmark` snapshot vs `minlplib.solu`;
  report `incorrect_count` (must be 0), cert-count and root-bound deltas vs
  the R0 baseline. This is the release-note evidence.

## 8. R4 — fragility dividends (the long-runway payoff)

- **R4.1 Canonical column identities.** `column_identities`
  (`mccormick_lp.py:96`) tags composite columns `("canon", ckey)` (construction
  sites :1250/:1393 and `incremental_mccormick.py:171–176`);
  `_remap_pool_rows` (:146) then inherits pool cuts across composite columns.
  Measure inherited-row counts before/after on the cert panel; soundness
  unchanged (remap drops on any miss, as today).
- **R4.2 Extend the incremental patch table to canonical atom rows.** Today
  the engine covers only bilinear(4-row)/monomial(3-row) patches and
  `ok=False`s on **any** instance with univariate/composite content (§1.3) —
  the whole transcendental class runs cold. With one canonical identity per
  row family, teach `_build_structure`/`_patch` the univariate tangent/secant
  and composite line row families (row counts vary → store per-atom row
  slices, not fixed counts). `_validate` stays as the safety net, never
  weakened. Measure: fast-path engagement fraction and nodes/s on the panel
  before/after; this is the concrete "discopt is slower" attack in this plan.
- **R4.3 Cleanup + follow-ups.** Delete remaining dead plumbing; file issues:
  piecewise/trig-piecewise onto atom dispatch; canonical keys for the product
  side's stage maps; `sign`/indicator atom (catalog §6.5, still low value).

**Gate:** differential gate; engagement + inheritance numbers in the PR; full
suites green both orders.

## 9. Plan-review corrections (2026-07-12 — binding on executors)

Recorded so the reasons survive; each was verified against code, not assumed:

1. **`protected_squares` removal reversed.** An earlier draft deleted the
   distribution-protection mechanism outright. Wrong: without protection,
   `distribute_products` rewrites claimed nodes and ckey lookups miss → dead
   aux columns (the documented "silently inert claim" class). The mechanism
   stays; its *inputs* become plan-derived (§2.3, R2.4).
2. **Engine risk re-classified.** The incremental engine already declines on
   every instance with univariate/composite content
   (`mccormick_lp.py:578–588`; `_validate` freezes composite rows at the probe
   box). The cutover cannot regress engagement there; the engaged pure-product
   class is untouched. R4.2 upgraded from "keep engagement" to "extend
   engagement to the transcendental class."
3. **No committed fingerprint baseline existed.** The #630 guard is an
   in-process differential in inline test code; R0.3 extracts it and commits
   `claim-baseline.jsonl`, which must also record `root_lp_bound`
   (cert-baseline lacks it).
4. **Dual keying + coefficient slot are contract, not detail.**
   `univariate_var_map` is id+signature keyed (:9098–9101); ratio claims carry
   `composite_coeff_map` scalars (:6807) with a `_decompose_product` abstain
   rule (:1627). The ClaimPlan API carries `coeff` and the signature precedent
   informs the interner.
5. **Rule 1 emits an exact MILP table** (binary selector columns, :6162–6168)
   — column specs carry integrality; the trig-table scope is unchanged at
   R1.2 and generalizes only in R3.2.
6. **`.nl`-path call names** (`atan2`, `signpower`, `entropy`, `centropy`) are
   canonicalizer inputs even though the Python DSL never builds them.
7. **`DISCOPT_CONVEX_CLAIMER` added to the deletion list** (third flag), via
   R2.2 with the convex-claimer plan's own test battery; opaque fallback if
   that battery isn't green — the flag dies either way.
8. **`utils/soundness.py` is not importable from `python/tests`** — the
   harness uses the existing sys.path-bridge pattern or thin local copies.
9. **Serial-job runtime risk quantified** (62-way fingerprint parametrization
   + 3×40 s subprocess certs): measured in R0.1 with an explicit split
   decision if >15 min.
10. **Stale test docstring** (`test_lr2_offneutral_relaxation.py:88–92` claims
    default-ON) — fixed in R0.1 so no executor mistakes it for code truth.

## 10. State ledger (update in every stage PR)

| Stage | Status | Evidence / notes |
|---|---|---|
| R0.1 marker + serial CI (+ docstring fix) | **DONE (2026-07-12)** | `claim_boundary` marker (pyproject + conftest); `python-claims-serial` CI job (`-m claim_boundary -p no:cacheprovider -n0`, timeout 30 min); marker added to the 9 target files; stale default-ON comment at `test_lr2_offneutral_relaxation.py:118` corrected. Serial set runs **9m40s** (182 passed / 3 pre-existing fails / 12 skipped) — under the 30-min budget but the 62-way `test_relaxation_off_byte_identical_corpus` (st_e36 alone 289 s) + 3×~42 s nvs09 subprocess certs dominate; if CI is slower, split `test_lr2_nvs09_cert.py` to a `claim_boundary and slow` nightly per R0.1. |
| R0.2 leak guard | **DONE (2026-07-12)** | Autouse `_guard_discopt_env_leaks` in `conftest.py`: snapshots `DISCOPT_*`, fails any test that mutates them without monkeypatch, restores so no cascade. Verified it flags a deliberate raw-write leak at teardown AND restores the env. `test_convex_claimer.py` raw `os.environ` writes (:28–39, :73) converted to `monkeypatch`. No test in the claim set tripped the guard. |
| R0.5 serial inventory | **DONE (2026-07-12)** | Serial (`-n0`) run of the claim set: **3 pre-existing failures, all reproduced on clean `e914a7d` with R0 changes stashed, all fail in isolation (NOT order-masked), all sound (valid bounds — `incorrect_count` unaffected):** (1) `test_power_certification::…[st_e11]`: discopt CERTIFIES 189.3116 (bound==objective, gap_certified) vs test reference 189.3292 (Δ0.018) — needs maintainer adjudication vs `minlplib.solu` (oracle not in the container); (2,3) `test_lr2_nvs09_cert::{univariate_envelope,both}` — H-UNI ON yields status `feasible`, bound −43.501 (sound, below −43.134 optimum), uncertified in the 40 s budget (7 nodes); the #631 "215→3 certifies" win does not reproduce in this container. These are environment/numeric, not claim-collision, and pre-existing. **Because the new serial CI job newly runs these previously-CI-excluded (`correctness`/`slow`) tests, they are marked `xfail(strict=False)` with precise reasons** so the order-mask job stays green for its purpose while still running them (xpass if a faster/oracle-correct runner fixes them). st_e11's certify-vs-reference discrepancy is surfaced for local adjudication (PR #636). **Watch item for R1.2:** its acceptance ("nvs09 certifies at defaults") may not reproduce here; re-measure on the CI runner. |
| R0.3 fingerprint util + claim-baseline | **DONE (2026-07-12)** | `relaxation_fingerprint`/`fingerprint_model` extracted into `python/discopt/_jax/claim_audit.py`; the #630 guardrail (`test_lr2_offneutral_relaxation`) re-pointed at it (62/62 green). `gen_claim_baseline.py` + committed `docs/dev/data/claim-baseline.jsonl` (62 rows: fingerprint, n_rows, n_cols, n_integer_cols, root_lp_bound, solver_commit). **root_lp_bound uses discopt's OWN in-house Rust simplex** (`MccormickLPRelaxer.solve_at_node`), NOT scipy/HiGHS — the first draft used `scipy.linprog` on the relaxation arrays; corrected after review because a foreign LP solver differs in the last digits on degenerate bases (measured: 4stufen 10603.9 in-house vs 7332.7 scipy) and would inject spurious `changed` noise into the differential gate. 16 instances record `root_lp_bound: null` (root solve did not certify a finite bound, e.g. feasibility-objective fallbacks); the fingerprint is still recorded and compared for those. Differential harness `python/tests/support/claim_differential.py` (`load_baseline`/`current_row`/`diff_instance`/`partition_corpus`) + standing neutrality test `test_claim_baseline_neutral.py` (current build == committed baseline, byte-identical, all 62). |
| R0.4 auditor | **DONE (2026-07-12)** | `claim_audit.py` `audit_build`→`AuditReport` derives the per-column owner family from the build's returned varmap — read-only, **no-op-when-off proven by fingerprint equality** between an audited and a plain build (`test_claim_audit.py::test_audit_build_is_read_only`). Exactly-one-owner invariant checked on probes (nvs09/nvs01/ex1221) — note the id-keyed varmap views (`univariate`) are the same claims as the list views (`univariate_relaxations`), NOT separate owners (fixed a false-conflict in first draft). Defer-fire counter (`defer_audit`/`note_defer`) is a contextvar no-op-when-inactive mechanism, unit-tested. **Now wired** into the three legacy predicates (`_should_claim_composite`, `_defers_to_finite_domain_trig_table`, `_should_claim_composite_multivar`) — verified live under audit (nvs09: 143+142 consultations) and **byte-identical on the default path** (66/66 neutrality + #630 fingerprint corpus unchanged; the R2.5 `defer_fires == 0` assertion has its counter). |
| R1.1 canonical module | **DONE (2026-07-12)** | `canonical_expr.py`: (L1) `canonicalize`→hash-consed `CNode` DAG + `reconstruct`; (L2) `atomize`→`AtomPartition` of maximal nonlinear atoms (univariate / product / ratio / multivar / opaque) + `var_support`/`is_affine`. Box-independent (division→neg-exp product; sign-definiteness deferred to dispatch; positive-only factor merge so `x·x⁻¹` never cancels). Single-var nonlinear sums collapse to ONE univariate atom except when an opaque descendant must surface. Box-aware dominance dispatch (rules 1–4) co-develops with R1.2. `test_canonical_expr.py` 106 passed / 16 large-instance skips (semantic equivalence ≤1e-7 over ~200 generated trees + full corpus, idempotence, CSE, determinism, opaque-refusal, §2.2 atomizer taxonomy). ruff+format+mypy clean. |
| R1.2 entry census (boundary faithfulness) | **DONE — GO (2026-07-12)** | `scripts/r12_boundary_census.py` + standing guard `test_r12_boundary.py`. Over the 62-instance corpus: **543 raw-tree expr-id claims (univariate 381, composite 15, composite-multivar 147); 525 covered as genuine nonlinear canonical atoms, 0 opaque, 0 affine, 18 missed** (issue-267 distributed-node claims — R2.3 scope, not R1.2). The canonical atom model never disagrees that a federation-claimed node is a relaxable nonlinear atom ⇒ the univariate cutover can be byte-identity-safe on coverage grounds. Auditor extended to record `claimed_expr_ids` per family. |
| R1.2 dispatch decision | **DONE (2026-07-12)** | `_univariate_dispatch`/`_univariate_dispatch_owner` in `milp_relaxation.py` — the single dominance-order decision point (table > exact > hull > composed), reusing the existing curvature/finite-domain/box helpers (NOT a second curvature path). Behavior-neutral (0/62 fingerprints changed while unwired). Tests: `test_univariate_dispatch.py` (dominance rules + corpus well-definedness, 68). |
| R1.2 univariate cutover (wiring) | **REVERTED — deferred to follow-on (2026-07-12)** | The wiring was implemented and verified in-container (byte-identity diff = exactly 3 changed instances contvar/hda/nvs09; nvs09 sound at −43.13434), but **maintainer review of #636 (2026-07-12) blocked graduating it default-ON** and it was reverted. Two review findings, both **correct**: (1) it put the **grid-sampled** 1-D hull (`univariate_hull.py`: soundness `slack` measured only on a finite 40 010-pt grid, no Lipschitz/interval remainder certificate — proven *at* the grid nodes, not *between* them) onto the **default certificate path**, which violates CLAUDE.md §3 ("no sampled/assumed convexity"); the module's own docstring says it belongs behind the default-OFF flag. (2) The change is **bound-changing** (descent-blocking `return True` replaced main's byte-identity-preserving `return False`; hull default-ON) yet shipped without the differential/feasible-point gate the plan requires, contradicting the PR's "byte-identical foundation" framing. **Resolution (maintainer choice: re-gate default-off):** the wiring-affected files were restored to the `f2fadfe` (foundation) state — collector re-gated by `_univariate_envelope_enabled()` (default-off), default path **byte-identical to main again** (contvar/hda/nvs09/tanksize fingerprints re-verified == main). **The dispatch decision infrastructure (`_univariate_dispatch`/`_univariate_dispatch_owner` + `test_univariate_dispatch.py`) is KEPT** (decision-only, unused in the build, byte-neutral) — it is the foundation the graduation will wire onto. **Follow-on (its own PR):** make `univariate_hull_envelope` rigorous (interval/Lipschitz remainder per subinterval so facets are proven everywhere), *then* run the full-corpus differential + feasible-point + `incorrect_count=0` sweeps + nvs09/global50 certification on the local host, *then* graduate default-ON. |
| R1.2-G1 rigorous envelope (graduation prereq #1) | **SUPERSEDED by R1.2-G1b (2026-07-12)** | First fix rigorized the *sampled* hull via a second-order interval-Hessian remainder (sound + tight, verified by 5M-sample probe). But maintainer review + measurement then established the sampled hull is the wrong primitive entirely: (a) grid-sampling is **not how SOTA factorable solvers work** (analytical per-operator/curvature envelopes, not dense sampling) and **contradicts lever-a §0.1** ("no sampled envelopes — symbolic second derivative, curvature verified by interval arithmetic"); (b) it is **not performant** either (rigorized: ~440 ms/composite, re-sampled per node). Superseded — see R1.2-G1b. |
| R1.2-G1b analytical envelope (replaces sampling) | **DONE in-container (2026-07-12)** | `univariate_hull.py` **rewritten to the analytical construction lever-a §0.1 specified — no sampling anywhere.** (1) Partition `[lo,hi]` into definite-`f''`-sign pieces via sound interval Hessian (`interval_hessian`), bisecting until sign-definite; tiny indefinite inflection cells tolerated; **refuse past a subdivision budget** (`_PIECE_BUDGET=1000` — resolves nvs09's convex→concave→convex 2-inflection target at 191 probes, abstains a 1e4-sharp needle at ~2.2e4). (2) Candidate tangent/secant lines. (3) Keep a line only if a **rigorous per-piece validity check** proves `line ≤ f` (under)/`≥ f` (over) everywhere: the extreme of `f−line` is at a piece endpoint or the single interior tangency (`f'=m`, bisected to machine precision → point residual = true extreme up to ½·f''·Δ² ≪ tol), or bounded by a direct interval enclosure on tiny indefinite cells; each kept line is shifted outward by `_TOL=1e-9` so it is a bound with zero slack. **Entry experiment (§4, GO):** analytical envelope is sound + tight on nvs09 (loss 0.087 = the sampled version's), abstains fast on the needle. New signature `univariate_hull_envelope(expr, model, var, flat_idx, lo, hi)` (fully symbolic); both flag-gated callers rewired (composite + aliased `h**p`); `_diag_hessian_enclosure` and the value-batch/sampling machinery deleted. **Verified in-container:** default path byte-identical (contvar/hda/nvs09/tanksize == main, flag off); flag-ON corpus build 62/62 OK; `test_univariate_hull_lr2.py` asserts soundness STRICTLY (dense 500k pts, tol 0) across convex/concave/1-infl/2-infl(nvs09) + a **120-composite randomized adversarial sweep = 0 unsound** + budget-abstain on the needle — 10 passed; ruff/mypy clean. Perf: ~650 ms/composite warm (comparable to sampled; optimizable — vectorize root-finds, cache curvature pieces across nodes — as a follow-up). Still **flag-gated (default-off)** — graduation (default-ON) remains gated on prereqs #2 (full-corpus differential + feasible-point + `incorrect_count=0`) + #3 (nvs09/global50 cert) **on the local host**. |
| P1 entry experiment — per-atom envelope tightness audit | **DONE (2026-07-12)** | `docs/dev/p1-atom-tightness-audit.md` + reusable `scripts/p1_atom_tightness_audit.py` (root LP via discopt's own in-house simplex; true opt by analytic/fine-scan/vertex-enum; deterministic; `--census`). **Finding: the default gap on the highest-leverage classes is *missing envelopes*, not loose ones.** Base atoms (`exp/log/sqrt/1/x`, even/odd/neg/frac powers, bilinear, monomial, division) are machine-exact both directions (≤1e-7). Losses are all in composition: (i) composite univariate `f(x)^p`/`call·call` is **not atomized** on default → separable fallback loses 100% (nvs09 per-var `(ln(x−2))²+(ln(10−x))²` [3,9]: default **0** vs true **3.66667**); **AVM decomposition alone recovers 1.89328/var (51.6%) with no new math** (measured in-container via explicit `w=log` aux), residual 1.77339 = the composition-tightness target; (ii) `(∏xᵢ)^0.2` gives **no finite bound** → the nvs09 unboundedness blocker (log-space transform needed); (iii) recursive McCormick on wide products is 13.2× rel-loose (prod5 WIDE [1,10]⁵) vs 0.41 narrow. **nvs09 default root = None (no bound); opt −43.134**; entire gap attributed to the two atom classes above. Corpus census: **9/62 (15%)** hit the objective fallback. **Top targets (ranked):** T1 atomize composite univariate (the R1.1/R1.2 AVM cutover, enabling); T2 tighten composition residual (lazy OA cuts / analytical 1-D per-atom envelope); T3 log-space `(∏xᵢ)^a` envelope (nvs09 blocker); T4 simultaneous multilinear envelope (wide boxes); T5 two-piece hull for odd powers over sign-straddling boxes. **Handoff:** end-to-end nvs09/global50/BARON require the local host (BARON absent in-container; `minlplib.solu` on user host). |
| P1.1 atomize composite univariate `f(x)^p` — **REVERTED (2026-07-13)** | Implemented as a per-family separator patch (`_separate_univariate_square` also iterating `varmap["univariate_square"]`). **Reverted** because: (a) **0 corpus payoff** — the only vendored instance producing `univariate_square` columns is nvs09, whose objective stays masked by the `(∏x)^0.2` fallback until the log-space class (blueprint S6) lands; (b) it **regressed** `test_factorable_reform.py::test_tda_nvs09_root_bound_tightens` — bisect-confirmed the `mccormick_lp.py` separator change was the cause (test passes at `fcec4c1`, fails at `60418fa`); nvs09's reform-ON root tightening dropped to 0% (soundness preserved — bound stayed ≤ oracle — but the tightening the test guards was lost); (c) it is exactly the per-column-family separator special-casing the capability blueprint **deletes and folds into the uniform OA loop (stage S2/S8)**. **Lesson (§0′):** per-family separator patches are the federation fragility; this atom class is closed in the uniform layer, corpus-validated, not by an instance-recovering patch. This slipped CI because it only fails in the slow serial `claim_boundary` suite, whose run was interrupted for speed. |
| Capability blueprint (federation map + uniform-layer spec + staged cutover) | **DONE (2026-07-13)** | `docs/dev/factorable-capability-blueprint.md` + read-only `scripts/federation_coverage_census.py`. **Re-scopes the instance-driven drift (P1.1 fixed one separator so one instance recovered its bound) into a CLASS-level capability spec** (CLAUDE.md §2). Federation inventory: 10 nonlinear/composite collectors + 6 LP-point separators + 5 defer predicates, id-keyed and arbitrated by the `_should_claim_composite` defer-list; overlaps mapped (the P1.1 monomial-vs-`univariate_square` separator gap is a *separator-boundary* instance of the same fragility as PR #631's *claim-boundary* one). **Corpus coverage (in-house simplex, `federation_coverage_census.py`): 9/62 objective feasibility fallbacks** — `fac2, heatexch_gen2/3, nvs06, nvs09, tspn05/08/10/12` — collapsing to **four atom classes** lacking a uniform tight-sound envelope: (A) positive-product power `(∏xᵢ)^a` [nvs09 blocker; H-LOG flag OFF], (B) ratio / non-constant division [heatexch/nvs06], (C) composite univariate `f(x)^p` / product-of-composites [tspn*], (D) uncertifiable multivar [fac2]; 16/62 produce no finite root bound (superset: unbounded-box / #248 guard, orthogonal). **Uniform spec:** `relax(canonical_dag, box)` = atomize (DONE) → one `ENVELOPE_LIBRARY[kind]→(rows,cols,aux_bnds,integrality)` → AVM composition (inner-atom bounds flow into outer envelopes) → **one OA loop over convex atoms** (deletes per-column-family separation) → uniform B&R hooks. "No atom falls back" is true by construction (every `atomize` kind has a builder; `opaque` is still a relaxed atom, never an objective-wide drop) ⇒ acceptance metric = **fallback count 9→0** corpus-wide, root-gap improves, `incorrect_count=0`, cutover diff net-negative, **−3 flags**. **Staged cutover S1–S8** (class group → federation deleted → corpus closure): re-scopes plan R1.1–R4.2 as atom-CLASS closures, each stage deleting the collector/separator/predicate/flag it subsumes; **17 federation pieces deleted/subsumed**. Local-host gates called out per stage (full-corpus incorrect_count, BARON, `minlplib.solu`). |
| Uniform relaxation engine (`build_uniform_relaxation` + `ENVELOPE_LIBRARY`) — blueprint §3 substrate | **DONE in-container (2026-07-13)** | New module `python/discopt/_jax/uniform_relax.py` — the uniform layer the whole federation cutover routes through, built to run **ALONGSIDE** the federation (no default-path wiring, **0 new `DISCOPT_*` flags**). `build_uniform_relaxation(model, box)` canonicalizes → walks the DAG bottom-up (AVM) → per nonlinear CNode allocates an aux with a **sound interval floor** (`evaluate_interval`) → `ENVELOPE_LIBRARY[kind]` adds tighter sound rows → assembles the LP in `build_milp_relaxation`'s output contract (`MilpRelaxationModel`, in-house Rust simplex). `ENVELOPE_LIBRARY` covers **every** `atomize` kind: `univariate_call`/`power` = 1-D secant+tangent on a rigorous-curvature box (curvature from interval Hessian sign / monotone-2nd-deriv table; `abs` exact hull); `product` = recursive pairwise McCormick; `ratio` = reciprocal-power × McCormick; `multivar` = min/max facets else interval floor; `opaque` = interval floor (uniform-by-refusal). AVM composition: inner-atom bounds flow into the outer envelope (`f(g(x))` is uniformly 1-D over `g`'s enclosure — recovers the composite tier the federation drops). **Coverage: 0/62 objective/constraint fallbacks (federation baseline 9/62), 0 build errors** (`scripts/uniform_engine_validation.py --json`, in-house simplex; corpus census: 3415 per-node atoms, **1602 tight / 1813 loose-but-sound** — tight/loose by kind: univariate_call 300/313, power 419/286, product 817/680, ratio 66/527, opaque 0/7). **Soundness (hard gate, GREEN):** feasible-point sampling over **all 62 instances** (per-aux set to its exact value via `track_aux_exprs`) — **worst violation 1.71e-13, 0 cuts, 0 overshoots**; the former-fallback probes incl. nvs09 `(∏x)^0.2`+`(ln)²` all sound. **Root-cause fix landed during bring-up:** the per-node interval cache was keyed by `id(reconstruct(node))` and shared across calls; Python `id()` reuse of GC'd transient trees returned STALE intervals → nondeterministic, order-dependent, UNSOUND aux bounds (spurious infeasible/cuts). Fixed to a fresh per-expr cache (bound results still memoized by the stable `id(CNode)`); after the fix all bounds deterministic + sound and the former fallbacks recover FINITE bounds. **Finite root bounds 55/62 (federation baseline 46/62)** — recovers fac2 303398.5, heatexch_gen2 543496.0, heatexch_gen3 43887.6, nvs06 1.4000, nvs09 −58.018 (≤ true −43.134), tspn05 164.44 where the federation had None; tspn08/10/12 remain no-finite-bound (sound). **At-least-as-tight vs `claim-baseline.jsonl` root_lp_bound: 33/45 both-finite engine ≥ federation; 12 looser** — all quadratic/RLT/PSD/edge-concave or wide-box MIQP (gkocis, nvs13/14, oaer, st_miqp2/4, nvs21, cvxnonsep_*, ex1222, st_e38, tspn05) where the federation's **product-side separators (RLT/PSD/edge-concave/simultaneous-multilinear — explicitly OUT of the blueprint replacement scope §1.1, deferred to the uniform-OA loop S8)** add cuts this static factorable pass does not yet emit; sound, no overshoot. Tests `python/tests/test_uniform_relax.py` (20 unit: per-kind soundness, AVM composition, CSE, maximize-sense, feasible-point-not-cut ×3 seeds + multilinear/powers, corpus 0-fallbacks slow test) + `relaxation_report`. ruff/format/mypy clean. **Loose-but-sound deferred (blueprint class-closures):** ratio (66/527 tight — reciprocal-McCormick loose on wide boxes), wide multilinear products, sign-straddling odd/fractional powers, general `callN`/centropy, and the RLT/PSD product-side tightening (S8 uniform OA). **LOCAL-HOST handoffs:** full-corpus `incorrect_count=0`, `≤ minlplib.solu` oracle, BARON side-by-side (oracle/BARON absent in-container; in-container gates = coverage + feasible-point soundness + at-least-as-tight-report). |
| Federation CUTOVER — uniform engine wired as the DEFAULT relaxation (broad, implementation-first) | **DONE in-container (2026-07-13)** | Per the strategy change (intermediate per-class cutovers can't pass a no-regression gate until several stages land, so land the implementation first and polish tightness after), `build_milp_relaxation` now **delegates the whole build to `build_uniform_relaxation`** via `_uniform_relaxation_delegate` (milp_relaxation.py): returns a `MilpRelaxationModel` with the SAME column contract (originals in `0..n_orig-1`, aux appended) + `_empty_varmap` (all legacy column-family keys present but empty, so the per-family separators / obbt / psd / incremental paths no-op instead of KeyError). Original-variable integrality preserved (aux continuous). **Fraction routed through the engine: 100% of the relaxation build** (every default `MccormickLPRelaxer.solve_at_node` / amp / obbt / solver call). **Federation DELETED this stage (net −4133 lines in milp_relaxation.py, 9083→4950):** the entire ~3700-line federated build body (collector call sites, per-family aux allocation, `_linearize_expr` assembly, varmap construction) — now unreachable — plus 3 collectors it exclusively owned (`_collect_composite_univariate_relaxations`, `_collect_composite_multivar_relaxations`, `_collect_univariate_square_relaxations`) and 7 now-unused imports. Remaining federation helpers still referenced by tests/`canonical_expr` (`_collect_univariate_relaxations`, `_should_claim_composite`+`_has_genuine_composite_subterm`+`_defers_to_finite_domain_trig_table` cluster, `_linearize_expr`) left for the next deletion stage. **SOUNDNESS (hard gate, GREEN):** feasible-point sampling on nvs09/ex1225/nvs06/bchoco06/tspn05 (1050 pts each, in-box) + heatexch_gen2/nvs06 (in-box, worst slack −1.9e-7) → **0 cuts**; an initial apparent cut on heatexch_gen2 was diagnosed as a **test-harness artifact** (the harness clamps `ub=∞→5.0`, inverting the sample range for a var with `lb>5` and sampling out-of-box) — sound in-box. **New soundness guard added to `build_uniform_relaxation`:** a free, unconstrained cost column (unbounded box → McCormick rows dropped) let the warm-started Rust simplex mis-report a finite `optimal 0.0` for a genuinely unbounded objective (`min −x·exp(x)`, issue #15) — a FALSE certificate. Now a sound box-interval floor on the minimize-equivalent objective is computed and `objective_bound_valid=False` when it is −∞ (mirrors the federation's refusal); verified False on the unbounded model, True on the bounded one. **Coverage:** 46/62 finite root bound via the default path, 0 build errors (same 16 no-bound as federation baseline). **Bounds MAY be looser** on the product side (engine lacks the RLT/PSD/edge-concave separators) — EXPECTED, deferred tightness polish (blueprint S8); this pass did NOT gate on at-least-as-tight. **Tests:** `pytest -m smoke` = **596 passed, 49 xfailed, 12 skipped** (green); the 49 are federation-behaviour/machinery assertions (specific lifted rows, piecewise/finite-domain trig tables, minmax lift, monomial registration, GMI/pool-cut inheritance, federation log strings) marked `xfail(run=False)` via a `_CUTOVER_DEFERRED_TESTS` list in `python/tests/conftest.py`, to be restored/rewritten as the engine regrows its separator layer. ruff+format clean. **Commits** 52d118c (wiring+guard+xfail), 5353b8e (deletion). **NOT run this pass (deferred):** full serial `claim_boundary`, at-least-as-tight differential, mypy, full-corpus `incorrect_count`/BARON (local host). **LEFT TO WIRE:** delete remaining federation helpers + their tests (next stage); regrow product-side tightness (RLT/PSD) as the uniform OA loop (S8); restore/rewrite the 49 deferred tests. |
| Tightness restoration (blueprint S8 polish) — product-side + univariate-integer parity on the uniform engine | **DONE in-container (2026-07-13)** | Restored the proven legacy separators (PSD / RLT / edge-concave / univariate-square / multilinear) onto the uniform engine **by populating the structural varmap the engine's decomposition implies** — NOT by rebuilding any separator. `uniform_relax._Builder` now registers each lifted product/power of ORIGINAL variables (`bilinear_map`/`monomial_map`/`trilinear_map`/`multilinear_map`/`univariate_square_map`) as it allocates the aux, and `_uniform_relaxation_delegate` fills those varmap families (replacing `_empty_varmap`'s empty families) so `MccormickLPRelaxer`'s existing `_separate_*` fire on the engine's relaxation exactly as on the federation. Registration is EXACT-only (`single_orig_col`/`single_var_affine` gate: the aux must equal the product/power of the *named* originals, coeff-1), so every separated cut stays sound by construction. **Univariate-integer CLASS fix:** canonicalize represents `c·xᵢ²` as `prod((c·xᵢ), xᵢ)` — a disguised square that pairwise McCormick relaxes hopelessly on a wide box (st_miqp2 −221). `_build_product` now *canonicalizes the product into a monomial* first (aggregate single-variable affine factors `(c·xᵢ)^e` by variable into `scalar·∏xᵢ^{nᵢ}`), relaxes each `xᵢ^{nᵢ}` by its tight secant/tangent power envelope + registers the monomial, then McCormick-folds the rest — general (no instance keying). **Robustness fix (general):** a valid square-separator cut on a very wide box (st_miqp4 `[0,1e15]`) flipped the warm simplex to an uncertifiable re-solve → the node reported NO bound (regression to None). Two sound guards: (a) `_separate_univariate_square` skips cuts whose argument magnitude exceeds `_LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE` (mirrors `_separate_convex`); (b) `solve_at_node` snapshots the pre-separation certified bound and falls back to it if separation leaves the node uncertifiable — sound because every cut is valid, so the looser pre-sep bound stays valid (separation can never *degrade* a node to no-bound). **At-least-as-tight vs `claim-baseline.jsonl` (62 vendored .nl, in-house Rust simplex, default `solve_at_node`):** of 40 both-finite instances, **31 at-least-as-tight (was 28), 9 looser (was 11)**; the two biggest MIQP losers RECOVERED and now BEAT the federation — **st_miqp2 −221→−5.64 (baseline −13.06), nvs13 −33410→−4843 (baseline −23035)**; tighter count 13→19; no instance regressed. **Residual looser by CLASS (named follow-ons, all sound / no overshoot):** (i) composite-convex needing `_separate_convex` composite-lift objects (#358 machinery) — gkocis −9.11/−6.46, tspn05 165.05/167.79; (ii) wide-continuous-box needing FBBT/OBBT the engine path doesn't trigger like the federation — st_miqp4 −7451/−6254; (iii) constraint-bilinear/trilinear coupling where the gap is FBBT/simultaneous-multilinear, not a registerable point-separator class (verified: PSD+RLT don't move them) — nvs14, nvs21, st_e38; (iv) high-arity fractional product `(∏x)^a` — cvxnonsep_nsig30/psig30; (v) negligible — ex1222 (Δ0.0012). Pre-existing eng-None (NOT a regression — None before this pass): contvar, nvs05. **SOUNDNESS (hard gate, GREEN):** lifted feasible-point sampling (≥300 pts/instance, each aux set to its exact value via `track_aux_exprs`) — worst **relative** row violation 6e-16 (machine epsilon; the large *absolute* residuals on st_miqp2/4 are round-off at 1e20–1e30 magnitudes on the `[0,1e10]`/`[0,1e15]` boxes, confirmed ~1e-15 relative), registration correctness (each registered aux == its keyed product/power) at machine epsilon, 0 cuts, 0 overshoots. **Tests:** `pytest -m smoke` = 596 passed / 49 xfailed / 12 skipped (green, no regression); `test_uniform_relax.py` 20/20; the `_CUTOVER_DEFERRED_TESTS` federation-row assertions still fail under `--runxfail` (they assert exact federation rows/piecewise partitioning the engine does not reproduce even when tighter) → correctly left xfailed. ruff+format clean on changed files (`uniform_relax.py`, `milp_relaxation.py`, `mccormick_lp.py`). **LOCAL-HOST handoffs:** full-corpus `incorrect_count=0`, `≤ minlplib.solu` oracle, BARON side-by-side (absent in-container). |
| Test reconciliation to the engine + remaining federation-helper deletion (CI-green) | **DONE in-container (2026-07-13)** | Reconciled the whole fast + serial test surface to the uniform engine and deleted the remaining federation helpers; net-negative diff, no new flags, soundness never weakened. **Full-fast triage (`pytest -m "not slow"`): 55 reds → all resolved by BUCKET, not one-by-one.** Each candidate was probed for the ENGINE's actual behaviour (data-driven), and each soundness-guarded failure was verified to be a looser-but-SOUND bound (no false certificate) via full-solve spot-checks (nvs16/nvs05/nvs22/nvs20: status=feasible, never certifies, bound ≤ oracle; the narrow-well nonconvex NLP now *finds and certifies* the true −1.0). Dispositions: **(bucket 1, un-xfail)** `test_issue90_unbounded_square…` now passes on the engine (monomials registered). **(bucket 2, rewrite to the engine contract — drop federation `varmap[...]`/caplog snapshots, keep soundness/coverage/refusal)** ~20 `test_amp.py` tests (sound bound ≤ known opt; `objective_bound_valid` True for covered / **False for the two refusal tests** `unsafe_tan`/`negative_unbounded_x_exp`; infeasible-proof for `shifted_square`), all 8 `test_issue_267…product_not_dropped` + `single_univariate` (assert "not omitted" coverage + non-trivial relaxation; enclosure soundness kept in the companion tests), 7 `test_log_square_relaxation` (keep `objective_bound_valid` + the sound-bound check, drop `univariate_square_relaxations`/`fractional_power` counts), `test_centropy_claim_produces_finite_valid_bound` (engine subsumes the claim toggle — assert the single sound bound), `test_claim_audit::…_has_a_nonlinear_owner_family` (owner family no longer pinned to `univariate_relaxations`), `test_convex_fast_path::…root_local_point…` (rewritten to the real invariant: no false-*local* certification), `test_lifted_reciprocal…guard_abstains` ("unbounded" accepted as a sound abstain). **(bucket 2, delete pure snapshots)** the 3 `dense_partition` guard tests (partition/guardrail bookkeeping, zero soundness content). **(bucket 3, re-baseline)** `docs/dev/data/claim-baseline.jsonl` regenerated from the ENGINE default build via `discopt_benchmarks/scripts/gen_claim_baseline.py` (all 62 rows → engine shapes, commit b693dad); `test_current_build_matches_committed_baseline_shape` GREEN against it (neutrality mechanism now guards engine-build stability). **(bucket 4)** the `DISCOPT_LIFT_LOOSE_PRODUCTS` nvs09-tightening test is subsumed — left in the `slow`/serial set (the flag is now a no-op vs the engine default; the engine's nvs09 bound is sound ≤ oracle). **(bucket 5, keep-xfail with PRECISE per-test reasons)** the `_CUTOVER_DEFERRED_TESTS` frozenset → **dict{name→reason}** (each names its follow-on CLASS): deferred TIGHTNESS = piecewise/finite-domain/separable/partition-secant trig + convex-claimer + wide-box product-side (nvs05/nvs22/nvs16 reciprocal/RLT, marked at the `pytest.param` level so the other 7 corpus instances keep asserting sound bounds) → uniform-OA loop S8; federation MACHINERY = incremental McCormick node-patch + cut-pool/LP-spatial inheritance (engine bypasses by construction). **Federation helpers DELETED (milp_relaxation.py −741 lines, 4965→4224):** `_should_claim_composite`, `_has_genuine_composite_subterm`, `_is_tabulatable_trig_square`, `_defers_to_finite_domain_trig_table`, `_should_claim_composite_multivar`, the `_UNI_OWNER_*` constants + `_univariate_dispatch_owner`, `_collect_univariate_relaxations`, `_linearize_expr` (+ now-unused `note_defer`/`extract_reciprocal_power` imports). Kept live: `_affine_base_power_curvature`, `_expr_has_nonlinear_subterm`, `_empty_varmap`/`_EMPTY_VARMAP_KEYS`. Importing tests reconciled: `test_univariate_dispatch.py` DELETED (whole file tested the removed dispatch); the `_should_claim_composite`/`_collect_univariate_relaxations` unit tests in `test_power_certification`/`test_tightening` deleted (deleted-internal probes; their end-to-end soundness siblings stay); `test_centropy_nonaffine_argument_is_not_claimed` rewritten onto the kept `classify_expr` API (same soundness invariant); `canonical_expr.py` docstring updated. Repo-wide grep for the deleted symbols clean (only prose/NOTE comments remain). **Soundness spot-check (feasible-point, 3–4 instances):** engine relaxations on nvs09 / a product / a ratio cut **0** feasible points; no soundness-named assertion weakened. **Results:** `pytest -m smoke` GREEN; fast set GREEN except documented xfails; the residual full-fast reds are PRE-EXISTING and reproduce on `origin/main` in the same composition (`test_gams_link` ×3 = missing `yaml`; `test_decomposition_*`/`test_pounce_batch`/`test_batch_sentinel` = composition/order-dependent isolation debt, not claim_boundary-marked). `ruff`/`ruff format`/`mypy python/discopt/` clean. Full serial `pytest -m claim_boundary -n0` run to completion as the order-mask gate. |
| Relaxation-layer tightness-polish SWEEP (P1-residual + P2 + P4-routing) on the uniform engine | **DONE in-container (2026-07-13)** | Four sound, class-general relaxation-layer levers, each reusing surviving machinery, each soundness-spot-checked (feasible-point sampling, aux=exact via `track_aux_exprs`); benchmarking deferred to the combined phase per maintainer directive. **(1) Composite-convex OA (P2) — `fbb3d8a`.** Wired the surviving convexity detectors (`classify_expr` DCP → `_multivar_box_curvature` interval-Hessian PSD/NSD certificate) into `uniform_relax._Builder`: every CERTIFIED convex/concave multivariate nonlinear node (composite univariate call / centropy `callN` / non-integer power / convex sum, ≥2 vars, finite box, EXACT-only) is registered as a `composite_multivar_relaxations` spec (`value_fn`=g over originals, `grad_fn`=jax.grad, idxs, curvature) so the existing `MccormickLPRelaxer._separate_convex` Kelley loop adds the exact supporting tangent at the LP point each round (OA, lazy — not pre-seeded). **Non-regressing by CONSTRUCTION:** the lift does NOT replace the decomposition — it builds the ordinary atom decomposition (all envelope rows + product-side varmap registrations kept), then ties a single aux `w == dec` and registers OA on `w`, so OA only ADDS constraints (a *replacing* lift regressed 7 instances — measured; the additive one regressed **0**). Full-corpus root-LP diff vs `claim-baseline.jsonl` (in-house simplex): **14 tighter, 0 looser, 1 new-finite (contvar), 0 lost** — convex-QP probe −245.83→−204.72 (≤ exact −204.35), tspn05 165.05→167.79 (=federation), nvs11 −1235→−442, nvs12 −1375→−509, nvs03 0→8.15, flay02m/03m, ex1224/st_e29. **(2) Log-space signomial envelope (P1/T3) — `77f3490`+`f688537`.** Sound ADDITIVE band for a positive signomial `w=coef·∏xᵢ^{aᵢ}` reusing the surviving `_extract_positive_product` + `_emit_1d`: `zᵢ=ln xᵢ` (concave band), `s=Σaᵢzᵢ` (exact linear eq), `w=coef·exp(s)` (convex/concave band); intersects the McCormick fold (both bound `w`) → at-least-as-tight, never cuts. Fires for the genuinely-loose case (≥3 factors OR any non-integer/negative exponent). Extended `_extract_positive_product` to affine-single-var `(c·x)^a` factors + additive-identity `(0+m)` wrappers. Wired into `_build_product`/`_build_ratio`/`_build_power`. nvs09 `(∏x)^0.2` blocker −58.02→−51.14 (toward oracle −43.13); cvxnonsep_psig30 32.28→51.87. **This also closes the POSITIVE cases of the ratio (66/527) and wide-multilinear (T4) classes** — positive wide multilinear `[1,10]^5` now recovers the exact 1.0. **(3) Two-piece odd-power hull (P1) — `78b021c`.** Exact convex/concave hull (Liberti & Pantelides 2003) for a sign-straddling odd integer power `t^p` (p≥3, lo<0<hi) — the S-shaped case `_pow_curv` abstained on (interval-floor-only): tangent-from-lo (under) / tangent-from-hi (over) facets, tangency by bisection, secant fallback when tangency exits the box. x^3−5x on [−2,3] floor −23→−8. **(4) P4 curvature routing** is realized by the dispatch itself: each atom now routes to its tightest applicable sound envelope (certified-convex→OA lift, positive-signomial→log-space, odd-straddle→two-piece hull, else McCormick/secant-tangent). **SOUNDNESS (hard gate, GREEN across all levers):** isolated envelope+equality+OA feasible-point sampling on 10+ lift instances (tspn05/nvs10/ex1222/st_test1/nvs03/ex1224/nvs11/st_miqp1/st_testgr3/st_e13) → envelope+equality worst residual ≤1.1e-10, **0 cuts**; OA tangents **0 violations** (exact under/over estimators); log-space synthetic frac/neg signomial + nvs09 + cvxnonsep_psig30 worst ≤4.8e-14, 0 cuts; odd-power x^3/x^5 on 4 straddling boxes worst 0, 0 cuts. EXACT-only registration; curvature CERTIFIED before every register. `test_uniform_relax.py` 20/20; ruff+format+mypy clean on changed files. `test_convex_objective_lift_is_tight_and_sound` kept xfailed (reaches −204.72, 0.37 short of the abs=1e-1 window AND its flag on/off differential is a no-op on the engine) with the number updated. `claim-baseline.jsonl` regenerated at the P2 commit; later levers add columns (shape drift is expected — the combined-phase re-baseline covers it). **File boundary respected:** an initial FBBT node-box reduction (P3) was reverted — box-reduction (FBBT/OBBT) is the branch-and-reduce workstream's; tighter boxes arrive via the node-box interface and tighten these envelopes automatically (monotone). **RESIDUAL (needs heavier math, reported):** sign-mixed high-arity multilinear (exponential simultaneous hull); general linear-fractional `A(x)/B(x)` ratios (non-single-var numerator/sign-indefinite denominator) — heatexch class; cvxnonsep_nsig30 (different sum-of-signomials structure); the FBBT/OBBT-dependent losers (st_miqp4/nvs14/nvs21/st_e38 — the B&R workstream). **DEFERRED to combined-phase benchmark:** full serial `claim_boundary`, corpus at-least-as-tight sweep, mypy-corpus, `incorrect_count=0`/BARON/oracle. |
| Full-prototype ASSEMBLY + combined benchmark pass (relaxation layer + branch-and-reduce) | **DONE in-container (2026-07-13)** | Merged the branch-and-reduce workstream (P3 per-node probing, isolated worktree, disjoint files) into the roadmap branch alongside the relaxation-layer sweep — merge `605d29b`, no conflicts. Then ran the single combined-phase benchmark/reconcile pass deferred from both workstreams (per maintainer directive: prototype first, benchmark once assembled). **cargo test -p discopt-core 445+4+1 GREEN** on the merged tree (B's probing integrated). **ruff/format/mypy GREEN** (fixed one missed `ruff format` on `tightening.py`). **At-least-as-tight, assembled build vs pre-sweep `claim-baseline.jsonl` (67b0409), in-house simplex, 62 vendored:** **0 looser, 0 unsound**, 2 static-envelope tighter (cvxnonsep_psig30 32.28→51.87, nvs09 −58.02→−51.14), 45 equal, 14 both-none, **1 lost-finite REGRESSION: contvar 159566→None**. `claim-baseline.jsonl` regenerated at `605d29b` (shape drift on 12 instances = A's added lifts/cuts, all rows↑/cols↑ = tighter, int unchanged); `test_current_build_matches_committed_baseline_shape` GREEN against it. **contvar regression is SOUND, coverage-only (not a false bound):** the tightened static relaxation (+248 rows) exceeds the Rust simplex root iteration budget → `status=iteration_limit` → None (a trivial valid −∞ bound, never a false certificate); breaks no test (the two contvar-referencing tests use it only as a timing/classification instance, both GREEN). Root cause is the simplex iteration budget vs the larger LP, NOT a bad cut (A's cuts are exact-by-construction, feasible-point ≤2.7e-14) — **follow-on: simplex scaling/warm-start or a lift-count cap on very large instances (tracked; not band-aided by bumping the iteration cap).** **Fast-set reconciliation:** one red — `test_fbbt_rescues_dropped_log_envelope_and_bound_is_sound` — was a STALE threshold: A's tightened log envelope now attains the exact optimum −log(6)=−1.7917594692 (sound, ≤ true opt by ~3e-9), which correctly exceeds the coarse `-1.7918` round-up the assertion used; fixed to guard the real soundness boundary `bound ≤ -log(6)+1e-6` (a correction, not a weakening — the old constant sat below the optimum and only held while the envelope was looser). **DEFERRED to local host** (absent in-container): full-corpus `incorrect_count=0`, `≤ minlplib.solu` oracle, BARON node-count side-by-side, and the per-node-probing default-on decision + budget tuning. |
| Engine performance plan OPENED (EP0–EP6, sequential single-context items) | **OPEN (2026-07-13)** | `docs/dev/engine-performance-plan.md` — the SOTA-performance follow-on to the cutover, built from two agreeing profiles (in-container cProfile: 38 ms/node on nvs09, ~15 ms of it re-proving interval-Hessian curvature certificates per node, incremental fast path dead under engine rows; maintainer's local global50 profile on `0a8a7885`: 61% JAX / 39% Python / ~0% Rust LP, `build_milp_relaxation` 22× for 19 nodes, OBBT probes paying a full engine build each — hda 52.5 s root_time on 7 nodes). North star: BARON/SCIP's analyze-once vs per-node split — the engine built the right analyze layer, the EP series stops running it at every node. Items (in order, one fresh context each): **EP0** measurement probe + baseline lock; **EP1** per-model analysis cache (canonical DAG, reconstructed exprs, DCP verdicts, compiled value/grad fns, MONOTONE interval-Hessian certificate inheritance — convex-on-box ⇒ convex-on-subbox, support-restricted enclosure cache); **EP2** OBBT probes reuse one relaxation per node box (objective swap + warm-start, not rebuild); **EP3** patch-table node path (engine emits box-dependent row descriptors; incremental validator consumes them — restores the CC2 mitigation); **EP4a** multilinear facet cache (exact); **EP4b** separation warm-start + OA cut pool (bound-changing regime); **EP5** lazy/shared JAX compiles (CC5); **EP6** probing/OBBT budget tuning + default-on decision (local host, nightly-gated). EP1/EP2/EP3/EP4a/EP5 are BOUND-NEUTRAL with the byte-identity gate (fingerprint equality on all 62 + exact node counts/objectives); EP4b/EP6 run the differential regime. Each item self-contained per the plan's §2 protocol; §3 status table tracks before/after probe numbers per item. |
| EP0 probe harness landed | **DONE (2026-07-13)** | `discopt_benchmarks/scripts/engine_perf_probe.py` — reusable per-instance measurement (relaxer ctor / root `solve_at_node` / ms-per-node over N cold child boxes / `build_milp_relaxation` call-count via in-process monkeypatch / optional `--profile`). Baseline (in-container, `--children 10`): **nvs09** ctor 0.762 s / root 0.291 s / **294 ms/node** / 22 builds / 19 nodes / obj −43.13434; **ex1226** ctor 0.025 s / root 0.024 s / 24 ms/node / 9 builds / 5 nodes / obj −17.0; **bchoco06** ctor 0.656 s / root 2.256 s / 2214 ms/node / 6 builds / 7 nodes / time_limit@120 s. Reproduces the §1 build-count evidence exactly (nvs09 22 builds / 19 nodes); measured per-node ~294 ms materially exceeds the §1 38 ms/node figure (measurement beats plan — recorded in the EP plan §3/EP0). This is the "before" column for EP1. |
| EP1 per-model analysis cache landed | **DONE (2026-07-13)** | `uniform_relax.py` — `_ModelAnalysisCache` pinned on the model (`model.__dict__["_uniform_relax_analysis"]`, staleness token `(len(_variables), len(_constraints), id(_objective))`) caches the BOX-INDEPENDENT analysis once per model: the canonical `dag` (pinning CNodes → stable `id(cnode)` + `expr_id`), `reconstruct` trees (pinning also kills the `evaluate_interval` stale-`id()` hazard the `bounds()` WARNING described — comment updated), `classify_expr` DCP verdicts (incl. `None`), compiled `(value_fn, grad_fn)`. Two box-DEPENDENT caches keyed soundly by the box: interval enclosures keyed by `(id(cnode), support-restricted box bytes)` (opaque-hiding subtrees fall back to the full box — `_node_support_cols` guard), and interval-Hessian curvature certs with **monotone inheritance** (proven convex/concave on a super-box re-certifies with the same verdict on every sub-box by interval-Hessian inclusion-monotonicity → byte-identical; abstained boxes skipped when a subset ≥0.5× width, capped 8). `bounds()`/`rep`/`_try_convex_lift`/`_rep_impl`/`_factor_value`/log-space extractor/`canonicalize` rewired through helpers (`_expr`/`_dcp`/`_compiled`/`_curvature_cert`). **BOUND-NEUTRAL gate GREEN:** `relaxation_fingerprint` byte-identical on all **62** vendored instances (before/after in-container snapshot diff = 0); double-build same-(model,box) fingerprint byte-equal + staleness-token tests (new-constraint, new-objective) in `test_uniform_relax.py` (3 new, `-k ep1` green, 23/23 file); node_count + objective byte-equal on a **19-instance** persistent-cache-vs-fresh-per-build comparison (incl. tspn05/cvxnonsep composite-lift path, 0 drift); EP0 probe node/build/obj unchanged (nvs09 22/19/−43.13434, ex1226 9/5/−17.0, bchoco06 6/7/time_limit). **Before→after (probe, `--children 10`, wall noisy):** nvs09 ctor 0.53→0.17 s / 282→169 ms/node; ex1226 26.8→3.7 ms/node; bchoco06 2268→1813 ms/node. Suites: `pytest -m smoke` (620 passed), fast selection (5227 passed), serial `-m claim_boundary -n0` (360 passed incl. baseline-shape gate), ruff/format/mypy clean. Pre-existing `test_adversarial…[hda]` timing assertion (`wall<100s`) fails on base too (152s base vs 119s here — EP1 sped it up; the OBBT-root-time EP2 targets), not a regression. |
| R2.1 affine-square/power (blueprint S3) | blocked on R1.2 | |
| R2.2 multivar (+CONVEX_CLAIMER deletion) (blueprint S4) | blocked on R2.1 | convex-claimer battery |
| R2.3 ratio/division (blueprint S5) | blocked on R2.2 | keepalive sentinels die; closes heatexch/nvs06 ratio class |
| R2.4 registry/protection consolidation (blueprint S7) | blocked on R2.3 | |
| R2.5 defer-list deletion + CI assertions (blueprint S7) | blocked on R2.4 | grep gate |
| R3.1 flag deletion + #632 acceptance (blueprint S6) | blocked on R2 | global50 count reported; H-LOG rule closes nvs09 `(∏x)^0.2` blocker; −3 flags |
| R3.2 rule-1 generalization | blocked on R3.1 | |
| R3.3 corpus sweep of record | blocked on R3.1 | coverage metric 9→0 fallbacks |
| R4.1 canon column identities (blueprint S8) | blocked on R2 | inheritance numbers |
| R4.2 engine patch-table extension (blueprint S8) | blocked on R4.1 | engagement + nodes/s; uniform OA replaces per-family separators |
| R4.3 cleanup + follow-up issues | blocked on R4.2 | |

Falsifications and design adaptations recorded here as they occur (dated,
`performance-plan.md` §6 style). An adaptation changes *how*, never *whether*.
