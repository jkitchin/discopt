# AVM canonical normal form — retiring the claim-boundary defer-list (issue #632)

**Status:** proposed · **Created:** 2026-07-12 · **Owner issue:** #632 · **Prereq
reading (every executor, every stage):** this file top-to-bottom, then issue #632,
PR #631's description (the collision post-mortem), `docs/design/relaxation-catalog.md`
§3–§4, `docs/dev/flag-graduation-protocol.md`, CLAUDE.md (§Development Philosophy,
§5 verification regimes), and — for the "why not reduced-space" framing —
`docs/dev/maingo-parity-plan.md` §7 (P2.4 KILLED entry).

This document is written to be executed **stage by stage by a fresh Opus session**
with no other context. Each stage has: verified codebase facts, deliverables,
an implementation checklist, a test spec, a gate with a kill criterion, and a PR
boundary. Execute stages in order; do not start a stage whose predecessor's gate
has not passed. After each stage, update the **State ledger** (§8) in the same PR.

---

## 0. Binding contract

1. **Correctness before everything.** `incorrect_count ≤ 0` with zero slack on every
   gate; the certificate invariant (`bound ≤ incumbent`, dual bound never crossing
   the oracle) holds on every panel. Never weaken a validation, fallback, or guard
   to pass a gate — if a gate can only pass that way, the stage FAILS and the
   failure is recorded in §8.
2. **Flag regime.** All behavior-changing work ships behind
   `DISCOPT_CANONICAL_CLAIMS` (default **OFF**). OFF must be **byte-identical** to
   prior main, proven by the #630 relaxation-fingerprint pattern
   (`test_lr2_offneutral_relaxation.py::test_relaxation_off_byte_identical_corpus`
   is the template). ON is bound-changing: differential-bound + feasible-point
   sampling + cert-panel objective neutrality, then graduation through the G1.2
   gate (`docs/dev/flag-graduation-protocol.md`: 3 consecutive green ledger
   verdicts before the flip PR).
3. **General mechanisms only.** The dispatch order in §2.4 is derived from
   *dominance invariants* (provably-tighter-when-applicable), never from instance
   names or shapes discovered through test failures. Any arbitration that can only
   be justified by "test X wants it" is a contract violation — surface it instead.
4. **Sound-or-refuse.** A canonical atom with no owning envelope decomposes one more
   level or falls back to the existing loose path; it never gets a "probably fine"
   envelope. Refusal to canonicalize (unsupported node) falls back to the current
   federation unchanged.
5. **Measurement beats plan.** A falsified assumption is recorded in §8 (dated, in
   the `performance-plan.md` §6 house style) and the plan re-scoped before further
   code. Binding prior falsifications (§1.5) are adopted wholesale.
6. **Workflow.** One stage = one PR from `main`, task ID in the title
   (`refactor(claims): AVM-2 — …`). Every PR: the stage's test spec green,
   `pytest -m smoke`, `pytest -m slow python/tests/test_adversarial_recent_fixes.py`,
   and — because this work is exactly the class the issue calls order-masked — the
   claim-sensitive set green **serially** (`-p xdist -n0`; the AVM-0 job). State
   what was run and the result in the PR body.

## 1. Verified codebase facts (verified 2026-07-12 on `main` ≈ `9937ff7`; re-verify only if stale)

All line numbers are `python/discopt/_jax/milp_relaxation.py` unless another file
is named.

### 1.1 The federation and its arbitration

- Entry point `build_milp_relaxation` (**:5360**) runs a sequential column-allocation
  pipeline. The composite-claim collectors run in this order:
  `_collect_univariate_relaxations` (:5999, def :4826, univariate-of-affine) →
  `_univariate_claimed_ids` seed (:6013) →
  `_collect_composite_univariate_relaxations` (:6014, def :3992, H-UNI home) →
  `_collect_aliased_monomial_hull_relaxations` (:6033, def :4207, H-UNI aliased) →
  H-LOG log-monomial (:6042–6103) → `_multivar_claimed_ids` (:6110) →
  `_collect_composite_multivar_relaxations` (:6111, def :4643; centropy/norm/convex-sum)
  → univariate squares (:6133) → finite-domain trig-square table (:6147–6182) →
  piecewise (:6184–6249) → fractional powers (:6256) → lifted products (:6310, :6326)
  → affine-square lift (:6365–6449) → affine-power lift (:6453–6493) → the issue-267
  ratio/nested-division/univariate-product walk (`_walk_lift` :7095, called
  :7142–7144).
- **The defer-list is** `_should_claim_composite` (**:3586**) + its helpers
  `_has_genuine_composite_subterm` (:3658), `_defers_to_finite_domain_trig_table`
  (:3718), `_is_tabulatable_trig_square` (:3690), plus the multivar twin
  `_should_claim_composite_multivar` (:4437) and the `claimed_ids`/`seen`/
  `_pre_existing_claim` gates inside the collectors (:4036–:4137). Every clause is a
  negative special case ("defer to the square lift / monomial lift / trig table")
  discovered through test collisions — the comments at :3634–3638 and :3664–3667 say
  so explicitly.
- **Two keying regimes coexist.** The *product* side is already structurally keyed
  and collision-free: `bilinear_var_map[(i,j)]`, `monomial_var_map[(i,p)]`,
  `trilinear/multilinear/fractional_power_var_map`, `univariate_square_var_map`
  (declared :5576–5583). The *composite* side is `id()`-keyed:
  `composite_var_map[id(node)] = col` (consulted by `_linearize_expr` :5073 at
  :5122–5123 and `_decompose_product` :1564/:1620). The defer-list problem lives
  entirely on the id()-keyed side.
- **id()-keying is load-bearing and fragile.** `distribute_products`
  (`term_classifier.py:329`) rebuilds nodes, so claims survive only via
  `protected_squares = frozenset(affine_square_protected_ids | composite_var_map)`
  (:7131–7132, applied :7134/:7139); a second distribution pass would orphan every
  key (comment :7114–7130). `_nested_div_keepalive` (:6540) exists solely to stop
  Python from recycling claimed ids (the ex7_2_3 false-cache-hit bug class).
  Expressions are **id-hashed, not content-hashed** (`modeling/core.py:210–227`,
  `:318`); there is no structural `__eq__`/`__hash__` and no hash-consing utility
  anywhere in `python/discopt/` — the closest thing is
  `factorable_reform.py:356` `_Lifter._expr_cache` keyed on `(repr(expr), lb_floor)`
  (with the :347–355 comment explaining why `id()` was abandoned there).
- **`factorable_reform.py` already does targeted (not canonical) lifting** at the
  solver level (`solver.py:3860→3909`): division clearing, repeated-factor monomial
  lifting (`_lift_expr` :737), blow-up-preventing product lifting (:531), TD-A call
  powers (:596), transcendental-arg lifting (:658), entropy canonicalization. It is
  pattern-targeted with a local string-structural CSE — not "one aux per elementary
  op", and not a shared canonical identity the relaxation layer can key on.

### 1.2 The incremental engine constraint

`IncrementalMcCormickLP` (`incremental_mccormick.py:103`, on by default via
`DISCOPT_INCREMENTAL_MC`, `mccormick_lp.py:574–598`) patches bilinear/monomial rows
per node and **validates cold/patch agreement** (`_validate` :305: row-set +
bounds equality vs a fresh `build_milp_relaxation` on sign-diverse probe boxes;
mismatch → `ok=False` → sound cold fallback). Column identities
(`mccormick_lp.py:96`) tag structural columns (`("bilinear",(i,j))`,
`("monomial",(i,p))`, …); composite-claimed aux columns are `("opaque", k)` —
position-locked, excluded from root-pool cut inheritance (`_remap_pool_rows`
`mccormick_lp.py:146`). Consequence: **any claimer that diverts a term changes the
structural/opaque partition** — the engine soundly declines (perf loss, silent),
and pool rows over composite columns never inherit. This is a §0.3-class safety
mechanism (CLAUDE.md; certification-gap-plan §0.3): never weakened, and the
canonical work must *extend* the identity scheme, not bypass `_validate`.

### 1.3 The order-mask mechanism (why xdist green ≠ safe)

- Claim flags are read fresh from `os.environ` on every call
  (`_univariate_envelope_enabled` :3544, `_log_monomial_enabled` :3572,
  `_convex_claimer_enabled` :4505) — no caching. A test that leaks a `DISCOPT_*`
  env var (raw `os.environ` writes exist, e.g. `test_convex_claimer.py:29–39`)
  silently flips claim behavior for **every later test in the same worker
  process**. `conftest.py` has no autouse `DISCOPT_*` reset.
- CI (`.github/workflows/ci.yml`) runs both Python jobs with
  `-n 2 --dist loadgroup` (:176–182, :243–250). There is **no serial Python job**,
  and `-m "... not correctness ..."` excludes the correctness-marked tests from the
  standard path entirely. Sharding means a leak only manifests when leaker and
  victim share a worker in the wrong order — exactly the #631 experience
  (test_issue_267 green under `-n4`, red serially).

### 1.4 H-UNI status (the graduation blocked by all this)

`DISCOPT_UNIVARIATE_ENVELOPE` default-OFF (:3544–3569 docstring records the
deferral rationale). The ON path is sound (PR #631: nvs09 certifies, tree 215→3,
`incorrect_count = 0`; measured whole-suite gain: global50 43→44). Its exact 1-D
hull builder lives in `_collect_composite_univariate_relaxations` /
`discopt._jax.univariate_hull`, guarded against effectively-unbounded boxes
(solver-sense `is_effectively_finite`, |bound| < 1e19). Flag-ON regression tests:
`test_lr2_huni_unbounded_guard.py`, `test_lr2_nvs09_cert.py` (subprocess-isolated),
`test_lr2_alias_shape_guard.py`, `test_lr2_offneutral_relaxation.py`.

### 1.5 Binding prior falsifications (do not re-litigate)

- **Reduced-space is NOT the vehicle for the default path.** maingo-parity-plan §7
  P2.4 is KILLED: reduced-space root bounds tie or lose vs lifted on every measured
  class (pooling/bilinear/QP/transcendental), ~8× per-node Kelley tax; it stays
  opt-in (`DISCOPT_RELAX_SPACE=reduced`). Issue #632 names it as "the vehicle", but
  the measurement wins: **this plan canonicalizes the *lifted AVM* path itself** —
  BARON's move (one decomposition, one envelope per atom *in the lifted space*),
  not MAiNGO's. The MCBox/reduced machinery is reused only as a test oracle (§5).
- **id()-keyed expression caching across rebuilds is unsound** (ex7_2_3 freed-id
  false cache hit → false optimal). Any new identity must be content-based.
- **H-UNI's tightness is real but its claim boundary was reverse-engineered from
  flaky tests** (PR #631). Do not re-graduate it by adding more defer clauses.

## 2. Design (settle before AVM-2; changes recorded here)

### 2.1 What "canonical AVM normal form" means here

A per-build **canonicalization pass** over the objective + constraint trees that
produces a content-addressed (hash-consed) canonical DAG, plus an **atomizer** that
partitions every nonlinear canonical node into exactly one *atom* with exactly one
*owner* (envelope family). Builders keep their envelope math; only the *claim
decision* moves out of them into one total dispatch.

Canonical node grammar (immutable, interned; total ordering on keys makes every
normalization deterministic):

```
ckey := ("var", flat_index)
      | ("const", c)
      | ("sum", ((coef, ckey), …sorted), const)         # n-ary, flattened, folded
      | ("prod", ((ckey, exponent), …sorted))            # n-ary, repeated factors merged
      | ("pow", ckey, p)                                 # non-integer p, or integer p kept atomic
      | ("call", name, ckey)                             # univariate intrinsics
      | ("callN", name, (ckey, …))                       # centropy, min/max, norm, prod
      | ("opaque", token)                                # unsupported → federation fallback
```

Normalization rules (each a pure rewrite with a property test): sum/product
flattening; constant folding; `neg`/`sub` → coefficients; repeated-factor merging
(`x·x → x^2`); `x**1 → x`, `x**0 → 1`; division → `("prod", …, (den, -1))` **only
when the denominator is sign-definite on the box** (else `opaque` — the sound
refusal, matching `_clear_divisions`' guard); deterministic child ordering.
Canonicalization is **box-independent** (pure structure); box-dependent decisions
(curvature, finite-domain, sign-definiteness) belong to the atomizer/dispatcher,
which takes the box as input — this keeps one canonical DAG valid across all nodes
of the tree while per-node dispatch stays box-aware, matching how
`build_milp_relaxation` is re-invoked per node with `bound_override`.

### 2.2 Atom taxonomy (exactly one owner per kind)

| Atom kind (canonical shape) | Owner (existing machinery — reused, not rebuilt) |
|---|---|
| affine (`sum` of vars + const) | linear rows, no aux (unchanged) |
| `("prod")` of ≥2 distinct unit-exponent factors | bilinear/trilinear/multilinear + RLT hull (`_ensure_bilinear_aux`, RLT specs) |
| `("pow", var, p)` — bare monomial / fractional power | monomial-secant / fractional-power lift (`monomial_var_map`, `fractional_power_var_map`) |
| `("pow", affine, 2)` | affine-square lift (:6365) |
| `("pow", affine, p≥3)` | affine-power lift (:6453) |
| **univariate atom** = maximal single-variable nonlinear canonical subtree | the **univariate envelope dispatcher** (§2.4) — this is where H-UNI, the trig table, the univariate-of-affine path, and the univariate-square path merge into one owner |
| `("callN","centropy",…)` / certified-convex multivar subtree | composite-multivar gradient cuts (:4643) |
| positive product `∏ xᵢ^{aᵢ}` (all lb > 0, flag) | H-LOG chain (:6042) |
| `("opaque", …)` | canonicalization refusal → the node (and only it) falls back to today's federation path |

The **univariate atom** is the load-bearing novelty: instead of five overlapping
claimers (univariate-of-affine, composite-univariate/H-UNI, univariate-square,
trig table, aliased-monomial-hull) each grabbing fragments and deferring to each
other, the atomizer identifies each *maximal* single-variable nonlinear subtree
once, and a single dispatcher chooses its envelope.

### 2.3 Identity: canonical keys replace `id()`

`composite_var_map`, the protected set, and the claim seeds re-key from
`id(node)` to `ckey`. Because `ckey` is content-based:

- claims survive `distribute_products` rebuilds **by construction** (no
  `protected_squares` heroics, no keep-alive pinning — though both stay in place
  for the fallback path until AVM-5);
- the ex7_2_3 freed-id hazard class dies;
- structurally identical subexpressions share one aux column (CSE) — this is the
  "hash-consed, one aux per elementary op" half of the issue;
- the incremental engine's `col_identities` gains a `("canon", ckey)` tag for
  composite columns, converting today's `("opaque", k)` position-locked columns
  into remappable identities (root-pool cut inheritance extends to composite
  columns for free — measured, not assumed: see AVM-3 gate).

### 2.4 The univariate dispatcher: dominance order, not defer-list

For a univariate atom `u(x)` over box `[l, u]` (post-FBBT), choose the **first
applicable** rule; each rule is provably at-least-as-tight as every rule below it
*whenever it applies*, so the order is an invariant, not an arbitration:

1. **Exact finite-domain table** — `x` integer with `|dom(x)| ≤ cap`: the convex
   hull of the finite graph `{(v, u(v))}` is exact; nothing beats exact. This
   *generalizes* the trig-square table (:3002) from `sin/cos(affine)**2` to any
   univariate atom over a small integer domain — the special case becomes a
   theorem instead of a defer clause.
2. **Certified convex/concave on the box** — exact envelope on the tight side +
   secant on the other (the existing univariate-of-affine machinery and the
   monomial/square/fractional kernels are instances of this rule; a bare `x**p`
   atom dispatches here to the *same* monomial-secant kernel as today, so there is
   nothing to "defer" to).
3. **Exact 1-D hull (H-UNI)** — neither convex nor concave, box effectively
   finite: `univariate_hull` builds the convex/concave hull. Under canonical
   dispatch this is just "the rule for the remaining univariate atoms" — the
   issue's acceptance statement.
4. **Composed fallback** — hull construction abstains (unbounded box, hull
   failure): decompose the atom one level and relax the pieces with today's
   composed envelopes (sound, looser).

Rule ordering proof obligations (AVM-2 test spec): 1 ⊐ 2, 1 ⊐ 3 (exactness), 2 ⊐ 3
where both apply (a convex function's hull *is* its envelope — assert equality, so
either order is sound; 2 first because it is cheaper), 3 ⊐ 4 (hull ≤ composition,
the nvs09 measurement). No rule mentions an operator name except through its
mathematical property (integrality, curvature certificate, boundedness).

### 2.5 What this plan does NOT touch

The product/RLT side (already structurally keyed), the envelope kernels
(`mccormick.py`, `envelopes.py`, `univariate_hull`), FBBT/OBBT/DBBT, the simplex,
branching, `factorable_reform.py`'s solver-level reformulations (it keeps running
first; the canonical pass sees its output), and the reduced-space/MCBox line.
The federation code path remains intact and is the OFF behavior until AVM-5.

## 3. Stages

### AVM-0 — interim guard: serial CI + env-flag hygiene (ships first, independent)

The issue's "cheap, orthogonal" ask. No solver math.

**Deliverables**
1. A pytest marker `claim_boundary` (registered in `pyproject.toml`) applied to the
   claim-sensitive files: `test_power_certification.py`,
   `test_centropy_relaxation.py`, `test_lr2_offneutral_relaxation.py`,
   `test_lr2_alias_shape_guard.py`, `test_lr2_huni_unbounded_guard.py`,
   `test_lr2_nvs09_cert.py`, `test_issue_267_univariate_product_lift.py`,
   `test_convex_claimer.py`, `test_factorable_reform.py` (module-level
   `pytestmark`).
2. A `ci.yml` job "Python claim-boundary (serial)": `pytest -m claim_boundary -n0`
   (single process, one worker, file order) so collisions cannot be order-masked.
   Keep it fast (<10 min) — it is a boundary probe, not a second full suite.
3. An **autouse fixture** in `python/tests/conftest.py` that snapshots `DISCOPT_*`
   env vars before each test and **fails the test loudly** if it leaked a changed
   `DISCOPT_*` var (i.e. mutated without `monkeypatch`). This removes the leak
   *mechanism*, not just the mask. Fix any offender it finds (known: raw
   `os.environ` writes in `test_convex_claimer.py:29–39`) by converting to
   `monkeypatch.setenv`.

**Gate:** the serial job green on `main`; the leak fixture green across the full
suite serially (`-n0`) — any failure it surfaces is fixed in the same PR.
**Kill criterion:** none (infra).
**PR:** `ci(claims): AVM-0 — serial claim-boundary job + DISCOPT_* leak guard (#632)`.

### AVM-1 — entry experiment: claim census + dominance-order hypothesis test

**Run BEFORE building anything.** Hypothesis: every arbitration the defer-list
encodes is derivable from (atom kind, box facts) via the §2.4 dominance order —
i.e. a total, instance-blind dispatch reproduces today's owner on every node where
today's owner is the tighter choice.

**Deliverables (scratch + a committed report, no package code)**
1. **Claim auditor**: an opt-in instrument (`DISCOPT_CLAIM_AUDIT=1`, or a
   test-only helper importing the collectors) that, for one
   `build_milp_relaxation` invocation, records for every nonlinear node: which
   claimers *would* claim it (run each predicate independently), which one *won*,
   and the resulting column. This can live as a debug hook module
   (`_jax/claim_audit.py`) since it changes no behavior when off; keep it — it
   becomes AVM-4's exactly-one-owner assertion.
2. **Census**: run the auditor over the in-repo corpus
   (`python/tests/data/minlplib_nl/`, 61+ files) plus a ~100-instance stratified
   sample of `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/` (seeded,
   recorded), with H-UNI/H-LOG both OFF and ON. Output: the overlap matrix
   (claimer × claimer, count of doubly-claimable nodes), the defer-clause firing
   counts, and per-overlap the §2.4 rule that resolves it.
3. **Report**: `docs/dev/avm1-claim-census-<date>.md` — every observed overlap
   class mapped to a dominance rule, or flagged as *not derivable*.

**Gate:** every overlap class resolves under §2.4 (possibly after adding a
*property-based* rule to §2.4 — allowed; adding a *shape-named* rule is not).
**Kill criterion:** an overlap class exists where the tighter owner **cannot** be
decided from atom kind + box facts (e.g. two owners each tighter on different
instances of the same atom kind with no measurable discriminator). If it fires:
record it in §8, and re-scope the affected atom kind to keep its current federation
arbitration *encapsulated behind the dispatcher* (one documented exception with a
tracking issue), rather than abandoning the plan.
**PR:** `exp(claims): AVM-1 — claim census + dominance-order verification (#632)`
(report + auditor only).

### AVM-2 — the canonicalizer (library-only)

**Deliverable:** `python/discopt/_jax/canonical_expr.py` —

```python
@dataclass(frozen=True)
class CNode: ...                      # kind, children (CNodes), payload; interned

class CanonicalDAG:
    roots: list[CNode]                # objective + constraint bodies
    of: dict[int, CNode]              # id(orig_node) -> CNode, valid for this build's pinned trees
    # interning table => structural equality is identity equality on CNode

def canonicalize(model: Model) -> CanonicalDAG   # pure, box-independent
class UnsupportedCanonicalization(Exception): ...  # -> ("opaque",…) atom, never raised to caller

def atomize(dag: CanonicalDAG, model, box, flat_types) -> ClaimPlan
    # ClaimPlan: CNode -> (atom_kind, owner, column_spec); exactly-one-owner by construction
```

Generalize `_Lifter._expr_cache`'s structural key into the interning table (keep
its `(…, lb_floor)`-style guards where domain enters; note the :347–355 comment in
its docstring). The univariate-atom extraction is a maximal-single-variable-subtree
computation on the canonical DAG (the existing `_composite_referenced_var` :3507
logic, made canonical).

**Test spec:** `python/tests/test_canonical_expr.py`
- *Semantic equivalence* (the load-bearing property test): for ≥200 randomly
  generated expression trees (all supported ops, mixed depth) + every in-repo
  corpus instance, evaluate original vs canonical at 1k random box points via the
  existing DAG compiler — max abs/rel error ≤ 1e-12. Any `opaque` node is exempt
  from rewriting by definition (assert it round-trips untouched).
- *Idempotence*: `canonicalize(canonicalize(e)) == canonicalize(e)` (key equality).
- *Interning/CSE*: structurally equal subtrees map to the same `CNode` object;
  `x*y + x*y` yields one product node.
- *Determinism*: key output independent of construction order (build the same
  expression two ways, assert equal keys).
- *Refusal*: division by a sign-spanning denominator, unsupported ops → `opaque`,
  never a rewritten form.
- *Atomizer*: unit tests per §2.2 row (incl. nvs09's
  `(ln(x-2))**2 + (ln(10-x))**2` → one univariate atom; `sin(x)**2` with small
  integer `x` → univariate atom whose dispatch is rule 1; `x**2*y` (post-reform)
  → product atom) + the §2.4 ordering proof obligations as property tests
  (sampled tightness comparisons on ≥20 random atoms per pair).

**Gate:** test spec green; wall-clock for `canonicalize`+`atomize` on the largest
in-repo instance ≤ 10% of its current `build_milp_relaxation` time (it will run
per node — budget it now).
**Kill criterion:** if canonicalization cost cannot get under that budget on honest
effort, re-scope to canonicalize once at the root and re-atomize per node on the
cached DAG (box-dependent dispatch only) — record the numbers.
**PR:** `feat(claims): AVM-2 — hash-consed canonical expression DAG + atomizer (#632)`.
Library-only; nothing on any solve path.

### AVM-3 — canonical dispatch behind the flag

Wire `ClaimPlan` into `build_milp_relaxation` behind `DISCOPT_CANONICAL_CLAIMS`
(default OFF):

- ON: compute `canonicalize` + `atomize` up front; the composite collectors
  (§1.1's f–k list) consult the ClaimPlan instead of their own predicates
  (`_should_claim_composite(...)` becomes, in canonical mode, a lookup:
  "am I the owner of this node's atom?"). `composite_var_map` re-keys to `ckey`
  with an `id→CNode` bridge for the linearizer. Builders' envelope math unchanged.
- The univariate dispatcher (§2.4) is implemented as the single owner for
  univariate atoms; the finite-domain table generalization (rule 1 beyond trig) is
  **deferred to AVM-6** — in this stage rule 1 applies exactly where the trig
  table applies today (same tightness, one owner).
- `col_identities` gains `("canon", ckey)` for composite columns (in canonical
  mode only).
- OFF: not a single changed byte in the built relaxation.
- H-UNI under canonical mode: `DISCOPT_UNIVARIATE_ENVELOPE` still gates rule 3
  (so canonical-ON/H-UNI-OFF is a meaningful intermediate config), but the
  *claim boundary* no longer depends on it — an unclaimed-by-rule-3 univariate
  atom falls to rule 4's decomposition, which must reproduce today's coverage.

**Test spec**
- *OFF byte-identity*: extend the #630 fingerprint corpus test to assert
  OFF ≡ prior main across the corpus (same mechanism, new flag).
- *ON soundness battery*: full smoke + adversarial suite with the flag ON;
  feasible-point sampling harness (`utils/soundness.py::assert_bound_sound`) on
  the corpus — no valid point cut, no bound crossing the `minlplib.solu` oracle;
  `incorrect_count = 0`.
- *ON differential root-bound*: for every corpus instance, root bound ON vs OFF
  recorded; ON may differ per instance (CSE and hull dispatch change the LP), but
  **no instance's certified objective may change**, and the cert-panel
  (`docs/dev/data/cert-baseline.jsonl`) objective/optimal-status must hold ON
  (node_count drift ON is expected and documented — bound-changing regime).
- *Incremental-engine agreement*: with ON, `IncrementalMcCormickLP._validate`
  passes (or the engine declines soundly) on the panel; assert **no silent
  whole-panel decline** — measure the fraction of instances where the fast path
  stays engaged ON vs OFF and report it in the PR (a wholesale loss of the fast
  path is a perf regression to fix, not accept).
- *Exactly-one-owner*: the AVM-1 auditor runs in canonical mode over the corpus
  and asserts every nonlinear node has exactly one owner and zero defer-clause
  firings.
- *Serial*: the AVM-0 job + the full PR-fast suite green serially with ON.

**Gate:** all of the above.
**Kill criterion:** if `_validate`-engagement ON drops below half of OFF and the
identity-remap fix doesn't recover it within the stage, ship the stage with the
engine finding recorded and open a dedicated follow-up before AVM-5 (the
graduation gate's regression-rate ceiling will hold graduation hostage to it
anyway — that is the system working).
**PR:** `feat(claims): AVM-3 — canonical claim dispatch behind DISCOPT_CANONICAL_CLAIMS (#632)`.

### AVM-4 — H-UNI as a rule: graduate the pair through the G1.2 gate

- Add `canonical_claims` (and the `canonical_claims`+`univariate_envelope` combo)
  as arms to `generality_sweep.py` `ARMS` / `graduation_gate.py` per the protocol.
- Nightly runs accrue ledger verdicts; requirement: **3 consecutive `eligible`**
  for the combo arm (0 soundness violations, cert-neutral objectives,
  regression_rate ≤ 0.10).
- The flip PR (separate, reviewed, per protocol): `DISCOPT_CANONICAL_CLAIMS`
  default-ON and `DISCOPT_UNIVARIATE_ENVELOPE` default-ON **together** (under
  canonical dispatch H-UNI is just rule 3; flipping them separately re-creates an
  intermediate boundary nobody will run). `=0` escape hatches kept for both.
- **Issue-#632 acceptance check, verified in the flip PR body:** (a) in canonical
  mode `_should_claim_composite`'s successor contains no instance-shape defer
  clause (the auditor's zero-defer assertion is CI-enforced); (b) full suite green
  **serially**; (c) ≥1 added certification vs the pre-flip baseline (nvs09 is the
  known candidate; global50 43→44 was the #631 measurement) with
  `incorrect_count = 0` on the measurement of record.

**Gate:** the ledger streak + the acceptance check.
**Kill criterion:** any lost incumbent/soundness violation in an arm → flags stay
OFF, instance recorded (protocol §Kill). A regression_rate > 0.10 that traces to
the incremental-engine finding of AVM-3 blocks until that follow-up lands.
**PR:** `feat(claims): AVM-4 — graduate canonical claims + H-UNI default-ON (#632)`.

### AVM-5 — delete the defer-list (cleanup)

Only after AVM-4 has been default-ON through ≥2 further green nightlies:

- Remove `_should_claim_composite`'s defer clauses, `_has_genuine_composite_subterm`,
  `_defers_to_finite_domain_trig_table`, the `allow_general` parameter, and the
  id()-keyed arbitration seeds; the federation collectors either route through the
  ClaimPlan unconditionally or are absorbed into the dispatcher. The OFF escape
  hatch (`DISCOPT_CANONICAL_CLAIMS=0`) must still work — decide here whether OFF
  preserves a frozen copy of the legacy boundary or is retired with a deprecation
  note (maintainer call, surfaced in the PR).
- Remove `protected_squares`/keep-alive plumbing made redundant by ckey claims
  (only what the canonical path provably no longer needs; the fallback path keeps
  its guards).
- Docs: update `relaxation-catalog.md` §3/§4 (one dispatch, atom taxonomy) and the
  H-UNI docstring (:3544) to describe the rule, not the deferral.

**Gate:** full suite serial + parallel green; fingerprint test updated to the new
default; cert panel objective-stable; `git grep` finds no shape-named defer.
**PR:** `refactor(claims): AVM-5 — retire the claim-boundary defer-list (#632)`.

### AVM-6 — dividend: generalize the finite-domain table (optional, measured)

Rule 1 currently mirrors the trig-square table's scope. Generalizing it to *any*
univariate atom over a small integer domain is now a ~small change (the dispatcher
already owns the decision) and is exactly the kind of "every future envelope
graduates for free" dividend the issue promises. Bound-changing: own flag, own
differential test, G1.2 arm. File as a follow-up issue at AVM-5 time; do not fold
into this plan's gates.

## 4. Stage → verification matrix

| Stage | Regime | OFF proof | ON proof |
|---|---|---|---|
| AVM-0 | infra | n/a | serial job + leak fixture green |
| AVM-1 | measurement | auditor changes no behavior (audit-off no-op test) | census report |
| AVM-2 | library-only | not on any path | property tests (equivalence/idempotence/CSE/refusal) |
| AVM-3 | bound-changing, flag OFF | fingerprint byte-identity corpus test | smoke + adversarial + soundness harness + cert-panel objective + exactly-one-owner + serial suite |
| AVM-4 | graduation | escape hatch `=0` | 3-green ledger + #632 acceptance (serial suite, ≥1 cert, incorrect 0) |
| AVM-5 | cleanup | decided in-stage | full suites both orders + grep gate |

## 5. Test oracles available (reuse, don't rebuild)

- `minlplib.solu` (corpus oracle) + `docs/dev/data/cert-baseline.jsonl` (41-instance
  panel) + `utils/soundness.py` (T0.4 harness: `assert_bound_sound`,
  `assert_cut_valid`).
- The #630 relaxation-fingerprint pattern for byte-identity.
- The reduced-space evaluator (`mccormick_subgradient.py`, sound after #583) as an
  *independent* bound cross-check on canonical-mode instances it supports: reduced
  bound ≤ canonical LP bound ≤ oracle optimum must hold wherever both apply — a
  cheap three-way consistency probe for AVM-3's battery.
- `graduation_gate.py` + `generality_sweep.py` arms for AVM-4.

## 6. Risk register

| risk | severity | mitigation |
|---|---|---|
| Canonical rewrite changes semantics (a bad rewrite rule) | **critical** | AVM-2 random-evaluation equivalence gate (1e-12) on generated + corpus trees; opaque-refusal for anything unproven |
| Dominance order wrong for some atom class → looser or (if a rule misapplies) unsound envelope | **critical** | AVM-1 census before code; §2.4 ordering property tests; AVM-3 soundness battery + oracle cross-check; sound-or-refuse rule 4 |
| Incremental-engine fast path silently disabled in canonical mode | high (perf) | `("canon", ckey)` identities; engagement-fraction measured in AVM-3 gate; `_validate` never weakened |
| CSE column-sharing changes conditioning (shared aux with wide bounds) | medium | `_MONOMIAL_AUX_BOUND_LIMIT`-style caps apply unchanged; differential root-bound recording catches outliers |
| Canonicalization cost per node | medium | AVM-2 ≤10% budget + root-cache re-scope path |
| Plan stalls mid-way, two boundaries live forever | medium | stages are individually shippable; OFF byte-identity means a stall leaves `main` unchanged; AVM-0 guard is independent value |
| A defer clause encodes real knowledge the census misses (rare structure not in corpus) | medium | AVM-1 samples the full 4,800-instance snapshot, stratified; graduation gate's held-out arm is a second, seeded sample; any post-flip incident reverts via the escape hatch and records the structure as a §2.4 property rule |

## 7. Explicit non-goals

Reduced-space as default (falsified, §1.5); new envelope math; touching the
product/RLT side's keying; solver-level `factorable_reform` rewrites; performance
targets beyond "no unexplained regression" (this is a correctness-of-process and
graduation-unblocking plan — the wins are H-UNI's certification(s), the death of
the order-masked-collision class, and every future envelope graduating without a
whack-a-mole tax).

## 8. State ledger (update in every stage PR)

| Stage | Status | Evidence / notes |
|---|---|---|
| AVM-0 serial guard | not started | |
| AVM-1 claim census | not started | run before any AVM-2 code |
| AVM-2 canonicalizer | blocked on AVM-1 | |
| AVM-3 flag wiring | blocked on AVM-2 | |
| AVM-4 graduation | blocked on AVM-3 + 3-green ledger | |
| AVM-5 defer-list deletion | blocked on AVM-4 + 2 further green nightlies | |
| AVM-6 finite-domain generalization | optional follow-up | file its own issue |

Falsifications recorded here as they occur (dated, `performance-plan.md` §6 style).
