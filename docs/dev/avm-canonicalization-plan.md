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
| R0.5 serial inventory | **DONE (2026-07-12)** | Serial (`-n0`) run of the claim set: **3 pre-existing failures, all reproduced on clean `e914a7d` with R0 changes stashed, all fail in isolation (NOT order-masked), all sound (valid bounds — `incorrect_count` unaffected):** (1) `test_power_certification::…[st_e11]` objective 189.3116 vs 189.3292 (Δ0.0176 > 1e-2 tol) — numerics/tolerance; (2,3) `test_lr2_nvs09_cert::{univariate_envelope,both}` — H-UNI ON yields status `feasible`, bound −43.501 (sound, below −43.134 optimum), uncertified in the 40 s budget (7 nodes); the #631 "215→3 certifies" win does not reproduce in this container. These are environment/numeric, not claim-collision, and are pre-existing — NOT fixed in R0 (plan §4 R0.5). **Watch item for R1.2:** its acceptance ("nvs09 certifies at defaults") may not reproduce here; re-measure on the CI runner before treating it as a gate. |
| R0.3 fingerprint util + claim-baseline | **DONE (2026-07-12)** | `relaxation_fingerprint`/`fingerprint_model` extracted into `python/discopt/_jax/claim_audit.py`; the #630 guardrail (`test_lr2_offneutral_relaxation`) re-pointed at it (62/62 green). `gen_claim_baseline.py` + committed `docs/dev/data/claim-baseline.jsonl` (62 rows: fingerprint, n_rows, n_cols, n_integer_cols, root_lp_bound, solver_commit). **root_lp_bound uses discopt's OWN in-house Rust simplex** (`MccormickLPRelaxer.solve_at_node`), NOT scipy/HiGHS — the first draft used `scipy.linprog` on the relaxation arrays; corrected after review because a foreign LP solver differs in the last digits on degenerate bases (measured: 4stufen 10603.9 in-house vs 7332.7 scipy) and would inject spurious `changed` noise into the differential gate. 16 instances record `root_lp_bound: null` (root solve did not certify a finite bound, e.g. feasibility-objective fallbacks); the fingerprint is still recorded and compared for those. Differential harness `python/tests/support/claim_differential.py` (`load_baseline`/`current_row`/`diff_instance`/`partition_corpus`) + standing neutrality test `test_claim_baseline_neutral.py` (current build == committed baseline, byte-identical, all 62). |
| R0.4 auditor | **DONE (2026-07-12)** | `claim_audit.py` `audit_build`→`AuditReport` derives the per-column owner family from the build's returned varmap — read-only, **no-op-when-off proven by fingerprint equality** between an audited and a plain build (`test_claim_audit.py::test_audit_build_is_read_only`). Exactly-one-owner invariant checked on probes (nvs09/nvs01/ex1221) — note the id-keyed varmap views (`univariate`) are the same claims as the list views (`univariate_relaxations`), NOT separate owners (fixed a false-conflict in first draft). Defer-fire counter (`defer_audit`/`note_defer`) is a contextvar no-op-when-inactive mechanism, unit-tested. **Now wired** into the three legacy predicates (`_should_claim_composite`, `_defers_to_finite_domain_trig_table`, `_should_claim_composite_multivar`) — verified live under audit (nvs09: 143+142 consultations) and **byte-identical on the default path** (66/66 neutrality + #630 fingerprint corpus unchanged; the R2.5 `defer_fires == 0` assertion has its counter). |
| R1.1 canonical module | **DONE (2026-07-12)** | `canonical_expr.py`: (L1) `canonicalize`→hash-consed `CNode` DAG + `reconstruct`; (L2) `atomize`→`AtomPartition` of maximal nonlinear atoms (univariate / product / ratio / multivar / opaque) + `var_support`/`is_affine`. Box-independent (division→neg-exp product; sign-definiteness deferred to dispatch; positive-only factor merge so `x·x⁻¹` never cancels). Single-var nonlinear sums collapse to ONE univariate atom except when an opaque descendant must surface. Box-aware dominance dispatch (rules 1–4) co-develops with R1.2. `test_canonical_expr.py` 106 passed / 16 large-instance skips (semantic equivalence ≤1e-7 over ~200 generated trees + full corpus, idempotence, CSE, determinism, opaque-refusal, §2.2 atomizer taxonomy). ruff+format+mypy clean. |
| R1.2 entry census (boundary faithfulness) | **DONE — GO (2026-07-12)** | `scripts/r12_boundary_census.py` + standing guard `test_r12_boundary.py`. Over the 62-instance corpus: **543 raw-tree expr-id claims (univariate 381, composite 15, composite-multivar 147); 525 covered as genuine nonlinear canonical atoms, 0 opaque, 0 affine, 18 missed** (issue-267 distributed-node claims — R2.3 scope, not R1.2). The canonical atom model never disagrees that a federation-claimed node is a relaxable nonlinear atom ⇒ the univariate cutover can be byte-identity-safe on coverage grounds. Auditor extended to record `claimed_expr_ids` per family. |
| R1.2 univariate cutover (wiring) | blocked on R1.2 entry (GO) | box-aware dominance dispatch lives HERE (next to the build's curvature/hull machinery — deliberately NOT reimplemented in canonical_expr, to avoid a second curvature path). nvs09 cert measured here (R0.5 watch item — may need CI runner). This step changes relaxation behaviour → differential-gate + soundness-sampling verified; benefits from local PR testing. |
| R2.1 affine-square/power | blocked on R1.2 | |
| R2.2 multivar (+CONVEX_CLAIMER deletion) | blocked on R2.1 | convex-claimer battery |
| R2.3 ratio/division | blocked on R2.2 | keepalive sentinels die |
| R2.4 registry/protection consolidation | blocked on R2.3 | |
| R2.5 defer-list deletion + CI assertions | blocked on R2.4 | grep gate |
| R3.1 flag deletion + #632 acceptance | blocked on R2 | global50 count reported |
| R3.2 rule-1 generalization | blocked on R3.1 | |
| R3.3 corpus sweep of record | blocked on R3.1 | |
| R4.1 canon column identities | blocked on R2 | inheritance numbers |
| R4.2 engine patch-table extension | blocked on R4.1 | engagement + nodes/s |
| R4.3 cleanup + follow-up issues | blocked on R4.2 | |

Falsifications and design adaptations recorded here as they occur (dated,
`performance-plan.md` §6 style). An adaptation changes *how*, never *whether*.
