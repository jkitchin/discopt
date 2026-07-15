# Factorable relaxation capability blueprint — replacing the claim federation with one uniform engine (#632)

**Status:** design blueprint (2026-07-13) · **Owner issue:** #632 · **Kind:**
BLUEPRINT — no solver/relaxation code changes; deliverable is this doc plus one
read-only measurement helper (`discopt_benchmarks/scripts/federation_coverage_census.py`).
**Prereq reading:** `docs/dev/avm-canonicalization-plan.md` (all of it, esp. §0′,
§0, §1, §2, §10), `docs/dev/p1-atom-tightness-audit.md`,
`docs/design/relaxation-catalog.md`, `CLAUDE.md` §Development Philosophy.

## 0. The reframe (binding — read before anything else)

Prior stages (P1.1 and the R1.2 wiring attempts) drifted into **instance-driven
patching** — fixing whatever blocked `nvs09` one atom at a time (the P1.1 row in
the plan ledger fixed exactly one separator so that exactly one instance's outer
square recovered its bound). That is the **federation-maintenance trap** issue
#632 exists to escape, and it violates CLAUDE.md §2 ("fix the class, not the
instance; named instances are gate probes only").

This blueprint specifies a **flexible, robust, GENERAL capability**, not another
instance patch. The capability is:

> **One uniform factorable relaxation engine, driven by the R1.1 canonical DAG,
> that relaxes every atom CLASS soundly and tightly, composes via the
> auxiliary-variable method (AVM), separates uniformly (OA cuts for every convex
> atom at the LP point), and REPLACES the current federation of ~17 special-case
> collectors / separators / defer-predicates.**

Everything below is judged **corpus-wide and class-level**. Named instances
(`nvs09`, `contvar`, `hda`, `heatexch_gen*`, `tspn*`, `fac2`, …) appear ONLY as
illustrative probes of an atom class, never as targets. A change whose benefit is
confined to a named instance is rejected by construction; the acceptance metric is
a **corpus-wide coverage number** (0 objective fallbacks), not "does nvs09
certify".

All file:line anchors are verified on branch `claude/issue-632-opus-plan-ffxld4`
(2026-07-13). Corpus counts are computed in-container over the 62 vendored
instances in `python/tests/data/minlplib_nl/` with discopt's **own in-house Rust
simplex** for every bound (never scipy/HiGHS), reproduced by
`federation_coverage_census.py` (§1.4) and `p1_atom_tightness_audit.py --census`.

---

## 1. Part 1 — Federation inventory (what we are replacing)

The default lifted LP path relaxes nonlinearity through **two federations**: a
set of collectors in `milp_relaxation.py` that *claim* a node and allocate its aux
column(s), and a set of LP-point *separators* in `mccormick_lp.py` that add cuts
per column family. Both are arbitrated by a hand-grown defer-list. The product
side (bilinear/monomial/trilinear/multilinear/fractional-power maps + RLT) is
**structurally keyed and collision-free** — it is *not* part of the replacement
target and stays intact; the replacement target is the **composite / id-keyed**
side.

### 1.1 Nonlinear / composite collectors (`milp_relaxation.py`)

Call order inside `build_milp_relaxation` (call sites verified this branch):

| # | Collector (def) | Call site | Atom class claimed | Keying | Registry written |
|---|---|---|---|---|---|
| 1 | `_collect_univariate_relaxations` (:4706) | :5879 | univariate call of an **affine** arg (`log/exp/sqrt/sin/cos/tan/inverse-trig`) | `id(expr)` **and** structural `_univariate_signature` | `univariate_var_map` (dual-keyed) |
| 2 | `_collect_composite_univariate_relaxations` (:4032) | :5895 | univariate call of a **non-affine** subtree (`f(g(x))`), gated by `_should_claim_composite` (:4078) | `id(node)` | `composite_var_map` |
| 3 | H-LOG (aliased-monomial / positive-product) block | :5914 (`_log_monomial_enabled()`) | positive product `∏ xᵢ^{aᵢ}` (all lb>0), incl. reform alias defs | `id` + alias `t_col` | `composite_var_map`; **flag `DISCOPT_LOG_MONOMIAL`, default OFF** |
| 4 | `_collect_composite_multivar_relaxations` (:4523) | :5983 | certified-convex multivar subtree / `centropy` (gradient cuts), gated by `_should_claim_composite_multivar` (:4564) | `id` | multivar → `composite_var_map` merge; **flag `DISCOPT_CONVEX_CLAIMER` for convex sums** |
| 5 | `_collect_univariate_square_relaxations` (:4837) | :6005 | outer square `s = w**2` of a lifted call aux `w` (issue #369) | `(base_col, 2)` | `univariate_square` map |
| 6 | finite-domain trig-square table | (post :6005) | exact `trig(affine int)**2` over a small integer domain — **MILP table** (binary selectors, integrality) | value table | binary selector columns |
| 7 | affine-square lift `(E)**2` (:6363 write) | inline | `("pow", affine, 2)` | `id(node)` | `composite_var_map[id]=w_col` |
| 8 | affine-power lift `(E)**p, p≥3` | inline | `("pow", affine, p≥3)` | `id(node)` | `composite_var_map` |
| 9 | ratio / nested-division owner (:6677 `r_col`, :6790 `prod_col`, :6847/:6950 `univariate_var_map[eid]`) | inline | division with sign-definite denominator on the root box | `id(eid)` + synthetic sentinel | `composite_var_map`, `composite_coeff_map`, `univariate_var_map`; `_nested_div_keepalive` sentinels (:6767) |
| 10 | issue-267 `_walk_lift` (:6966) + post-lift re-collection | :7013–7018 | univariate-function *factors* inside a product (`cos(x − x·x)`) — claims on the **distributed** trees | `id` on distributed nodes | `univariate_var_map` |

Additive, column-keyed passes that are **not claim-arbitrated** (untouched by the
replacement until the far-out R4.3-equivalent): piecewise / trig-piecewise
(`_trig_piecewise_interval_specs` etc.), lifted-product collectors
(`_collect_lifted_bilinear_products` :1768, `_collect_lifted_higher_products`
:1834 — product side), fractional-power lift.

### 1.2 LP-point separators (`mccormick_lp.py`), dispatch order

Run per node inside `solve_at_node` (:1337–1377), each keyed by a **column
family** in `varmap`:

| Order | Separator (def) | Column family it reads | Atom class |
|---|---|---|---|
| 1 | `_separate_multilinear` (:1626) | `bilinear`/`trilinear`/`multilinear` | product hull (Rikun/Meyer-Floudas) |
| 2 | `_separate_edge_concave` (:2149) | `xᵢ²`/`xᵢxⱼ` auxes | coupled quadratic vertex-polyhedral (Tardella) |
| 3 | `_separate_univariate_square` (:1716) | **`monomial` and `univariate_square`** | `s = w²` supporting tangent at the LP point |
| 4 | `_separate_convex` (:1856) | convex composite lifts (#358) | supporting hyperplane of a convex atom |
| 5 | `_separate_psd` (:2056) | fully-lifted cliques | moment / SDP eigencut |
| 6 | `_separate_rlt` (:1969) | bound × constraint factors | RLT product cuts |

Separator 3 is the archetypal **per-column-family** separator and the site of the
claim-boundary bug the P1.1 row fixed (§1.3).

### 1.3 Arbitration, overlaps (claim-boundary problem), and holes

**Arbitration mechanisms.**
- The **defer-list**: `_should_claim_composite` (:3559, `_note_defer` :3589),
  `_has_genuine_composite_subterm` (:3635), `_is_tabulatable_trig_square`
  (:3667), `_defers_to_finite_domain_trig_table` (:3695),
  `_should_claim_composite_multivar` (:4316) — each a hand-written predicate that
  decides which collector may claim a node, plus the `claimed_ids`/`seen`/
  `_pre_existing_claim` gates inside the collectors.
- **Linearizer dominance** (`_linearize_expr` :5073 area): `composite_var_map.get
  (id(e))` is consulted **first, unconditionally, short-circuiting** (scaled by
  `composite_coeff_map`), then type dispatch to `univariate_var_map` /
  fractional / monomial maps; `_decompose_product` abstains if
  `composite_coeff_map[id] != 1.0`.
- **Pool-cut dominance**: `column_identities` (`mccormick_lp.py:96`) tags every
  aux; anything composite is `("opaque", k)` (:139–142) and `_remap_pool_rows`
  (:146) **drops any pooled cut row touching an opaque column** (:189–195) — so
  composite columns never inherit root cuts.

**Overlaps (two claimers can grab the same node — the PR #631 collision class).**
- `g(x)**2`: claimable by the **composite-univariate** collector (#2), by the
  **univariate-square** collector (#5, as `w²`), or pre-rewritten by
  `factorable_reform._prelift_call_powers` into a bare **monomial** `t**2`
  (product side). The three paths disagree on which separator later tightens the
  column. **The P1.1 bug (plan ledger, 2026-07-13):** `_separate_univariate_square`
  built its `(base, aux)` spec list **only from `varmap["monomial"]`**, skipping
  the `univariate_square` family, so the composite-atom column kept its loose
  static endpoint envelope (bound stuck at 0). This is a *separator-boundary*
  instance of the same federation fragility as the *claim-boundary* one — the fix
  extended one separator to one more column family, i.e. **another federation
  patch**, not a class closure.
- `(x−3)**2`: affine-square lift (#7) vs monomial (product side) vs
  composite-univariate (#2).
- `1/f(x)`: ratio owner (#9) vs univariate (#1) vs composite (#2).
- issue-267 walk (#10) claims on **distributed** trees while #1–#9 claim on
  **raw** trees; `distribute_products` is called with
  `protected_squares = affine_square_protected_ids | composite_var_map` to keep
  raw claims alive, and `_nested_div_keepalive` pins `object()` sentinels purely
  against `id` recycling — bookkeeping that exists *only* because identity is
  id-keyed.

**Holes (atom classes that fall through to the separable / feasibility fallback).**
Measured corpus-wide (§1.4): **9 of 62 instances drop the objective to the
fallback path** (no genuine per-atom envelope applied → bound 0 or none), and
**16 of 62 produce no finite root bound**.

### 1.4 Corpus quantification (reproducible)

`federation_coverage_census.py` builds each instance's root relaxation with the
in-house simplex, captures the "could not linearize the objective" warning, and
attributes each fallback to the canonical atom-kind histogram
(`canonical_expr.atomize`). Result (2026-07-13):

**Objective fallback — 9/62 (coverage HOLE):** `fac2, heatexch_gen2,
heatexch_gen3, nvs06, nvs09, tspn05, tspn08, tspn10, tspn12`.

**No finite root bound — 16/62:** `alan, bchoco07, bchoco08, casctanks,
cvxnonsep_psig40r, fac2, hda, heatexch_gen1, heatexch_gen2, heatexch_gen3,
nvs06, nvs09, tanksize, tspn08, tspn10, tspn12`.

**Fallback → atom-class attribution** (the canonical atom kinds present in each
fallen-back instance):

| Instance | atom kinds present | Class that caused the hole |
|---|---|---|
| `fac2` | `multivar:3` | uncertifiable **multivar** atom (curvature not proven → collector #4 declines) |
| `heatexch_gen2` | `univariate:27, ratio:26, product:40` | **ratio / non-constant division** (collector #9 declines on the objective path) |
| `heatexch_gen3` | `univariate:95, ratio:110, product:200` | **ratio / division** |
| `nvs06` | `univariate:1, ratio:2` | **ratio / division** |
| `nvs09` | `univariate:20, multivar:1` | **positive-product power `(∏xᵢ)^0.2`** (the multivar atom — no envelope; H-LOG flag OFF) + composite-univariate squares |
| `tspn05` | `univariate:10, product:10` | **composite univariate `f(x)^p` / product-of-composites** |
| `tspn08` | `univariate:16, product:28` | composite univariate / product-of-composites |
| `tspn10` | `univariate:20, product:45` | composite univariate / product-of-composites |
| `tspn12` | `univariate:24, product:66` | composite univariate / product-of-composites |

So the 9 holes collapse to **four atom classes**: (A) positive-product power
`(∏xᵢ)^a`; (B) ratio / non-constant division; (C) composite univariate `f(x)^p`
and products of composites; (D) uncertifiable multivar. The 16 "no finite bound"
instances are a **superset issue**: some (e.g. `alan`, `hda`, `tanksize`,
`casctanks`) linearize but the bound is invalidated by an unbounded box or the
#248 under-constrained guard (`_omitted_obj_linked`, :8922) — an orthogonal
bound-validity gate, not a coverage hole, and out of scope for the atom-envelope
capability.

### 1.5 The incremental engine, for completeness

`IncrementalMcCormickLP` patches **only** bilinear (4 rows) and monomial (3 rows);
every composite row is frozen at a probe box, so `_validate` forces the cold path
on **any** instance with univariate/composite content (plan §1.3). The composite
class therefore has **zero** fast-engine engagement today — the cutover cannot
regress it, and extending the patch table to canonical atom rows is a downstream
payoff (§4, Stage G), not a cutover risk.

---

## 2. Part 2 — Atom-class coverage map (the capability's requirement spec)

Taxonomy from `canonical_expr.atomize` (kinds: `univariate`, `product`, `ratio`,
`multivar`, `opaque`) refined to the concrete sub-classes. Tightness measured in
`p1_atom_tightness_audit.py` (root LP via in-house simplex vs the true 1-D/2-D
envelope). **Class-level — not organized by instance.**

| Atom class (canonical shape) | Covered? | Sound? | Tight vs exact envelope | Owner (collector / separator) | Uniform, or holes |
|---|---|---|---|---|---|
| affine over vars/aux | yes | yes | exact | linear rows | uniform |
| univariate-of-affine call `f(a·x+b)` (`exp/log/sqrt/trig/inverse-trig`) | yes | yes | **machine-exact both directions** (≤1e-8, p1 §2) | `_collect_univariate_relaxations` (#1) + secant/tangent rows | uniform |
| monomial `x**p` (even/odd/frac/neg) | yes | yes | machine-exact on a definite-curvature box (≤1e-7); **loose on a sign-straddling odd power** (`x³−3x` [−2,2]: −10 vs true −2, p1 §3) | monomial-secant / fractional lift + `_separate_univariate_square` | **hole:** two-piece hull for sign-straddling odd powers (p1 Target 5) |
| product `∏ xᵢ` (distinct unit factors) | yes | yes | exact hull ≤ cap (RLT); **13.2× rel-loose wide/high-arity** (prod5 WIDE, p1 §3.2) | bilinear/tri/multilinear maps + `_separate_multilinear`/`_separate_rlt` | **hole:** simultaneous multilinear envelope for wide boxes (p1 Target 4) |
| `("pow", affine, 2)` / `("pow", affine, p≥3)` | yes | yes | tight | affine-square (#7) / affine-power (#8) lift | uniform (id-keyed) |
| **composite univariate `f(x)**p` / `call·call`** | **partial** | yes | **atomized** (inner aux `w=f(x)` + outer square) but composition-loose: AVM recovers 51.6%/var, residual **48.4%/var** (nvs09 per-var: default 0 → AVM 1.893 → exact 3.667, p1 §3.1) | composite-univariate (#2) + univariate-square (#5) + separator #3 | **NOT uniform** — the P1.1 monomial-vs-`univariate_square` separator gap is the archetype hole; residual needs OA cuts (Target 2) |
| **positive product power `(∏xᵢ)^a`** (all lb>0) | **no on default** | — | **no finite bound either direction** ([3,9]¹⁰ → None, p1 §4) | H-LOG (#3), **flag `DISCOPT_LOG_MONOMIAL` OFF** | **HOLE** — the nvs09 blocker; log-space transform (Target 3) |
| **ratio / non-constant division** | **partial** | yes | sign-definite denom on the root box handled; **objective-path fallback** on `heatexch_gen2/3`, `nvs06` | ratio / nested-division owner (#9) + `_walk_lift` (#10) | **holes** — id-keyed, sentinel-pinned, declines on the objective path |
| multivar convex / `centropy` | partial | yes | exact gradient cuts **when curvature certified**; **uncertifiable → opaque/fallback** (`fac2`) | `_collect_composite_multivar_relaxations` (#4) + `_separate_convex`, **flag `DISCOPT_CONVEX_CLAIMER`** | **holes** — flag-gated convex sums; curvature-abstain → fallback |
| finite-domain `trig(affine int)**2` | yes | yes | exact (convex hull of the finite graph) | trig-square table (#6), MILP binary selectors | uniform within its scope (rule 1) |
| `("opaque")` (CustomCall/MatMul/array/`sign`/sign-spanning division) | by design | yes | loose (composed fallback) | existing composed fallback | uniform-by-refusal |

**Reading.** Base atoms are already SOTA-exact; **every loss is in composition**
(p1 headline). The capability requirement is therefore not "tighter envelopes" but
**(i) atomize the composites the federation drops** (classes A/C above),
**(ii) one envelope library so no class is flag-gated or curvature-abstained into a
hole**, and **(iii) uniform separation so a covered atom is never left with a
loose static envelope because a per-family separator forgot its column** (the
P1.1 class).

---

## 3. Part 3 — The uniform capability spec

### 3.1 The target pipeline

One entry point, per B&B node box:

```
relax(canonical_dag, box) =
    atoms          = atomize(canonical_dag)                 # R1.1, DONE — recursive AVM
    for atom in atoms (inner atoms before outer, by construction):
        kind, box_a = classify(atom), refine_box(atom, box) # inner-aux bounds flow up
        rows, cols, aux_bnds, integ = ENVELOPE_LIBRARY[kind](atom, box_a)
        emit(rows, cols, aux_bnds, integ)                   # AVM composition
    # uniform separation, per node LP solution:
    for atom in convex_atoms:
        add_OA_cut(atom, x_lp)                               # one loop, not per-family
    # uniform branch-and-reduce hooks:
    for atom: register_obbt_fbbt(atom, box)
```

The decompose step is **already done** (`atomize`, plan ledger R1.1). The four
things this blueprint specifies as *new capability contracts* are the envelope
library, the AVM composition contract, uniform separation, and the "no atom falls
back" invariant.

### 3.2 The envelope-library interface (atom class → columns + rows)

A single table `ENVELOPE_LIBRARY : atom_kind → EnvelopeBuilder`, where each
builder has the **uniform signature**

```
build(atom: Atom, box: AtomBox) -> Envelope
Envelope = (rows:   list[(coeff_vector, rhs)],      # A z <= b over (orig ∪ aux) cols
            cols:   list[AuxCol],                    # aux columns THIS atom introduces
            bounds: dict[AuxCol, (lo, hi)],          # interval bounds on each aux
            integrality: dict[AuxCol, bool])         # rule-1 tables carry binary selectors
```

This is the generalization of the existing per-collector emission, unified so the
caller never special-cases a class. The `integrality` slot is load-bearing:
finite-domain tables (owner #6) emit binary selector columns, so the column spec
must carry integrality (plan §9 correction 5). The builders **reuse the existing
envelope math verbatim** (`UnivariateRelaxation`, `CompositeUnivariateRelaxation`,
the affine-square/power emitters, the ratio fold, the H-LOG log-space chain, the
multilinear RLT) — only the *dispatch* and the *interface* are new. Float payloads
are keyed by exact bit pattern (content addressing), never id.

The atom kinds map to builders as:

| atom_kind (from `atomize`) | builder | source math reused |
|---|---|---|
| `univariate` | dominance dispatcher (§3.4) | `_collect_univariate_relaxations`, composite-univariate, square, trig table |
| `product` | product / RLT builder | multilinear maps + `_separate_multilinear` (+ log-space for `∏xᵢ^{aᵢ}`, lb>0) |
| `ratio` | ratio builder | nested-division fold + reciprocal + McCormick product |
| `multivar` | curvature-certified convex builder, else `opaque` | `_multivar_box_curvature` + gradient/OA cuts |
| `opaque` | composed fallback on that node only | existing fallback |

### 3.3 The AVM composition contract

Atomization is recursive: an inner atom gets its aux column(s) first; an outer
atom is then a function of original vars **and inner-atom aux symbols**. The
contract that makes composition tight and sound:

1. **Inner-atom bounds flow into outer-atom envelopes.** The interval bounds an
   inner builder returns (`Envelope.bounds`) become the **box** the outer builder
   sees for the aux symbol. (E.g. `w = ln(x−2)` on [3,9] returns `w ∈ [0, ln7]`;
   the outer `w²` builder is then given `[0, ln7]`, not `(−∞,∞)`.) This is exactly
   how BARON composes and is what recovers the AVM tier with **no new envelope
   math** (p1 §3.1: 0 → 1.893/var).
2. **Content-addressed identity ⇒ CSE for free.** Equal canonical content → same
   CNode → same aux column; a shared subexpression is relaxed once.
3. **Distribution protection is plan-derived.** The `protected_squares` mechanism
   (`term_classifier.py:359–368`) stays but its input becomes `{id(n) : cnode_of
   (n) is a claimed atom}` — the hand-maintained `affine_square_protected_ids` and
   `_nested_div_keepalive` sentinels die (their sub-atoms now have real CNode
   names). This kills the "silently inert claim" bug class.

### 3.4 The univariate dominance dispatcher (rules, not a defer-list)

For a univariate atom over the node box, first applicable rule wins; each is
at-least-as-tight as every rule below it *when it applies* (plan §2.4 tiers, with
H-UNI's grid-sampled hull tier **removed** per §0′):

1. **Exact finite-domain table** (integer var, small domain) — MILP binary
   selectors.
2. **Certified convex/concave on the box** — exact envelope + secant.
3. **Composed fallback** — decompose one level, relax the pieces (sound, looser).

No rule names an operator except through a mathematical property (curvature,
finite-domain), so the dispatcher is general by construction. The rigorous
analytical 1-D envelope (`univariate_hull.py`, rewritten grid-free in R1.2-G1b)
is available as the tightener for rule 2's remaining nonconvex-nonconcave atoms
**only once it passes the local-host differential + feasible-point + incorrect_count
gate** (plan ledger R1.2 revert) — it is *not* a default-path primitive in this
blueprint until then.

### 3.5 Why "no atom falls back" is true by construction

Every canonical node is exactly one of the five `atomize` kinds; each kind has a
builder; the two kinds that currently fall back (`multivar` when uncertifiable,
and the composite/ratio/positive-product classes the collectors decline) either
get a sound builder (ratio, positive-product via log-space, composite via AVM) or
route to `opaque` — and **`opaque` is still a relaxed atom** (composed fallback on
that node only), not an *objective-wide* feasibility drop. The objective-wide
fallback (§1.4, the "could not linearize the objective" path) exists **only**
because a single unclaimed node makes `_linearize_expr` raise and collapses the
whole objective; once every atom has a builder, that raise cannot fire, so the
objective fallback count goes to **0 by construction**.

### 3.6 Acceptance criteria (CAPABILITY metrics, corpus-wide)

The capability is accepted iff, measured over the corpus (and, on the local host,
the full MINLPLib snapshot):

1. **Coverage = 0 objective feasibility fallbacks** (from `federation_coverage_census.py`;
   baseline today 9/62). This is the headline capability number.
2. **Root-gap distribution improves corpus-wide** vs the federation baseline
   (`claim-baseline.jsonl` root_lp_bound; differential harness): no instance's
   sound root bound drops, and the aggregate closes gap.
3. **`incorrect_count = 0`** with zero slack (hard gate, CLAUDE.md §Key Constraints).
4. **The cutover DIFF is net-negative** in the collector/separator layer (deletes
   federation code) with **zero instance-named or shape-hardcoded branches**
   (CI-enforced grep gate), and the **`DISCOPT_*` flag count goes DOWN by 3**
   (`DISCOPT_UNIVARIATE_ENVELOPE`, `DISCOPT_LOG_MONOMIAL`, `DISCOPT_CONVEX_CLAIMER`).

Metrics 2–3 require the local-host oracle (`minlplib.solu`, BARON side-by-side);
metric 1 and the grep/flag part of 4 are in-container.

---

## 4. Part 4 — Staged cutover plan (federation deletion, capability-validated)

Stages are ordered by **enabling dependency** and each is defined by an atom-CLASS
group, never an instance. Each stage (a) routes a class group through the uniform
layer, (b) DELETES the federation collector(s)/separator(s)/predicate(s) it
subsumes, (c) is validated corpus-wide (differential bound: new ≥ old AND ≤ oracle;
feasible-point sampling clean; the class-coverage metric improves; no drop).
Emission math and every conditioning/finiteness cap survive as owner helpers.

This blueprint **re-scopes the plan's R1.2–R3.1 stages as atom-class closures** and
places the current corpus fallbacks (§1.4) as the **class closures** each stage
delivers — not as nvs09 patches.

| Stage | Class group routed uniform | Federation DELETED | Corpus closure (validation target) | Needs local host |
|---|---|---|---|---|
| **S1** (plan R1.1) — canonical core | — (library only; no cutover) | nothing | R1.1 semantic-equivalence + atomizer spec green (DONE) | no |
| **S2** (plan R1.2) — **composite univariate `f(x)^p` / `call·call`** | dispatcher owns collectors #1,#2,#5,#6; separator #3 folds into uniform per-atom OA (§4.1) | `_defers_to_finite_domain_trig_table`, `_has_genuine_composite_subterm`, the `allow_general`/additive clauses of `_should_claim_composite`, the flag gate of the aliased-monomial hull, **and the monomial-vs-`univariate_square` special-casing in `_separate_univariate_square`** (the P1.1 patch becomes a class rule) | closes the **tspn05/08/10/12** class + nvs09 squares (composite-univariate no longer falls back) | full-corpus differential + incorrect_count on host |
| **S3** (plan R2.1) — **affine-square / affine-power** | `("pow",affine,2/p)` atoms drive lifts #7/#8 | `_collect_affine_squares`/`_collect_affine_powers` as *claim* passes, `affine_square_protected_ids`, the `composite_var_map[id]` writes | affine-power class uniform; no drop | differential on host |
| **S4** (plan R2.2) — **multivar / convex** (+ `CONVEX_CLAIMER` deletion) | `multivar` atoms → curvature-certified builder, uncertifiable → `opaque` | `_should_claim_composite_multivar`, **flag `DISCOPT_CONVEX_CLAIMER`** (predicate stops consulting it) | closes the **fac2** class (curvature-certified) or routes it to a relaxed opaque atom (no objective-wide drop) | convex-claimer battery + differential on host |
| **S5** (plan R2.3) — **ratio / non-constant division** | negative-exponent product atoms → ratio builder; univariate-over-aux → dispatcher | the issue-267 `_walk_lift` (#10), the outer-atom `if eid in composite_var_map` guards, **`_nested_div_keepalive` sentinels**, the double-keyed id writes | closes the **heatexch_gen2/3, nvs06** ratio class (objective no longer falls back) | differential + feasible-point on host |
| **S6** (plan R3.1 part) — **positive-product power `(∏xᵢ)^a`** as an always-on rule | H-LOG log-space builder becomes unconditional (guarded by strict-positivity on the FBBT root box) | **flag `DISCOPT_LOG_MONOMIAL`** and its machinery | closes the **nvs09** blocker (finite bound on `−(∏x)^0.2`); wide positive products tighten (p1 Target 3/4) | nvs09/global50 cert + BARON on host |
| **S7** (plan R2.4/R2.5) — **registry + defer-list deletion** | linearizer consults the ClaimPlan only | `composite_var_map`/`composite_coeff_map`/`univariate_var_map` id-registries; `_should_claim_composite` (:3559) + every §1.1 defer helper; grep gate returns only docs | zero legacy-predicate consultations (auditor `defer_fires == 0`, exactly-one-owner) | serial CI (in-container) |
| **S8** (plan R4.1/R4.2) — **uniform separation + engine** | per-atom OA replaces per-column-family separators; `column_identities` tags `("canon", ckey)` so composite columns inherit pool cuts; incremental patch table extends to canonical atom rows | the per-family spec special-casing in the separators; the position-locked `("opaque", k)` tag for composite columns | separator loop is one per-atom OA pass; fast-engine engagement extends to the transcendental class | engagement + nodes/s on host |

**Flag count trajectory (goes DOWN, never up):** S4 deletes `DISCOPT_CONVEX_CLAIMER`,
S6 deletes `DISCOPT_LOG_MONOMIAL`, and the `DISCOPT_UNIVARIATE_ENVELOPE` read
(already default-off, re-gated per the R1.2 revert) is removed with S2/S7 — **−3
flags, 0 added** (plan §0 mandate 2).

**Federation pieces deleted/subsumed by the cutover (count):** **17** —
collectors #2,#3,#5,#7,#8,#9,#10 as claim passes (7); defer/arbitration predicates
`_should_claim_composite`, `_has_genuine_composite_subterm`,
`_is_tabulatable_trig_square` (as defer), `_defers_to_finite_domain_trig_table`,
`_should_claim_composite_multivar` (5); the `_nested_div_keepalive` sentinel
mechanism (1); the per-column-family special-casing in `_separate_univariate_square`
folded into uniform OA (1); the id-registries `composite_var_map`/
`composite_coeff_map` and the id-key half of `univariate_var_map` (3). Collectors
#1/#4/#6 survive as **owner helpers** invoked by the dispatcher (their emission
math is reused), not as independent claim passes.

### 4.1 Uniform separation (the P1.1 class closure)

The P1.1 fix (extend `_separate_univariate_square` to one more column family) is
the *symptom* of the separator federation: each separator hand-iterates a specific
`varmap` family, so a newly-covered atom class silently keeps a loose static
envelope until a human remembers to add its family to the right separator. The
uniform closure: **one OA loop that iterates the ClaimPlan's convex atoms and adds
the supporting hyperplane at the LP point**, keyed by canonical column identity
(`("canon", ckey)`), so any convex atom — monomial square, composite square,
convex composite — is separated identically. This deletes the per-family branching
in separator #3 and lets separators #4/#3 merge.

### 4.2 What each stage hands to the local host

Every stage's in-container evidence is the **coverage metric** (fallback count
from `federation_coverage_census.py`) and the **root-bound differential** (in-house
simplex vs `claim-baseline.jsonl`). The following are **out-of-container** and are
gate conditions, not claimed in-container: full-corpus **`incorrect_count = 0`**
(needs the MINLPLib snapshot + `minlplib.solu` on the user host), the **BARON
side-by-side** (`global_opt_baron_vs_discopt.py --time-limit 60`, BARON absent
in-container), and the end-to-end **certified** root gap on `nvs09`/global50 (plan
ledger R0.5 flags nvs09 in-container certification as environment-sensitive).

---

## 5. Reproduction

```bash
cd discopt && source .venv/bin/activate
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python discopt_benchmarks/scripts/federation_coverage_census.py --json /tmp/fed.json
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python discopt_benchmarks/scripts/p1_atom_tightness_audit.py --census
```

Both are deterministic, read-only, and use discopt's in-house Rust simplex for
every bound. `federation_coverage_census.py` is the standing coverage instrument:
the capability's acceptance metric (§3.6, criterion 1) is exactly its
"objective fallback" count going 9 → 0.
