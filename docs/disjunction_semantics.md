# Disjunction semantics — the source contract

**Status:** normative for the modeling layer and every GDP lowering.
**Issue:** [#1124](https://github.com/jkitchin/discopt/issues/1124) (PR 0 of the
MCP/GDP RFC, [#1123](https://github.com/jkitchin/discopt/issues/1123)).

This document fixes the meaning of a disjunction *at the source level*, so that
every later lowering — big-M, hull, MBigM, LOA, and the continuous/complementarity
targets proposed in #1123 — has a contract to preserve rather than a meaning to
infer from whichever activation it happens to emit.

## 1. The two axes

A disjunction over disjuncts `S_k = {x : g_kj(x) <= 0, j in J_k}` is lowered with
one selector binary `y_k` per disjunct. Two *independent* choices define what the
disjunction means:

**Activation** — how `y_k` relates to the predicate `x in S_k`:

| | meaning | consequence |
|---|---|---|
| `ONE_WAY` | `y_k = 1 => x in S_k` | nothing is asserted when `y_k = 0`; a *deselected* predicate may still be true |
| `REIFIED` | `y_k = 1 <=> x in S_k` | `y_k = 0` forces `x not in S_k` — a strict complement needing an explicit negation policy |

**Cardinality** — how many selectors may be active: `AT_LEAST_ONE` (`sum_k y_k >= 1`),
`EXACTLY_ONE` (`sum_k y_k == 1`), `AT_MOST_ONE` (`sum_k y_k <= 1`).

`DisjunctionSemantics` members are defined *as* the pair: a member's `.value`
**is** its `(activation, cardinality)` tuple, so the axes are carried by the
member itself rather than by a side table that could drift out of sync. The
lowercase spelling (`"select_one"`, available as `.label`) is a coercion alias
accepted by the `semantics=` argument, not the value.

Lowerings branch on the pair, never on the member name: the GDP pass declares
the pair its rows encode and serves a disjunction iff its pair matches. Adding a
combination later (an optional-mode disjunction is `(ONE_WAY, AT_MOST_ONE)`) is
then an added member, not a redefinition of the enum's meaning — and a future
member carrying an already-implemented pair is served without touching the
lowering.

| member | pair | status |
|---|---|---|
| `SELECT_ONE` | `(ONE_WAY, EXACTLY_ONE)` | **implemented** — the meaning of `either_or` |
| `OR` | `(ONE_WAY, AT_LEAST_ONE)` | declared, not implemented |
| `EXACTLY_ONE_TRUE` | `(REIFIED, EXACTLY_ONE)` | declared, not implemented — owes a boundary policy, §2 |

## 2. `either_or` is `SELECT_ONE`

Exactly one disjunct is **selected**, and the selected disjunct's constraints are
enforced. An unselected disjunct's constraints are not enforced but *may still
happen to hold*. The projection of the feasible set onto the model variables is
the **union** of the disjuncts, so a point in the overlap of two disjuncts remains
feasible and is assigned to one selected mode.

This is not "exactly one group of constraints holds". That stronger reading is
`EXACTLY_ONE_TRUE`.

The distinction is **semantic, not convexity-driven**. With the convex sets
`S_1 = {x <= 1}` and `S_2 = {x >= 0}`, the point `x = 1/2` is feasible under
`SELECT_ONE` and infeasible under `EXACTLY_ONE_TRUE` because both predicates are
true there. The same holds verbatim for nonconvex `S_k`; convexity governs
relaxation strength and whether a hull formulation is ideal, not the Boolean
meaning. Both cases are pinned in
`python/tests/test_1124_disjunction_semantics.py`.

### Truth semantics owes a boundary policy

`EXACTLY_ONE_TRUE` over **closed** predicates that overlap does not merely exclude
the interior of the overlap — it excludes the overlap's *boundary* too, and that
can leave a problem with no optimum at all.

Take the discriminator above, `min (x - 1/2)^2` over `[x <= 1] v [x >= 0]` with
`x in [-2, 3]`. Both predicates hold on all of `[0, 1]`, endpoints included, so
"exactly one true" admits only the **open** set `[-2, 0) u (1, 3]`. On it the
objective approaches `0.25` at either boundary and never reaches it: `0.25` is an
**infimum, not an attained optimum**, and writing "the optimum is 0.25 at
`x in {0, 1}`" locates it at two points the semantics has just excluded.

This is not a defect in `SELECT_ONE` — every lowering this repo ships is
select-one, and the select-one optimum `0` at `x = 1/2` is attained and asserted.
It is a constraint on work not yet written: **before `EXACTLY_ONE_TRUE` is
implemented it needs an explicit strict-complement/boundary policy**, deciding
what a lowering does when the complement of a closed predicate is open. Candidates
are refusing such a disjunction, requiring strict predicates, or introducing an
epsilon-margin — each with different soundness consequences, none of them free,
and the choice must be made deliberately rather than falling out of whichever
big-M happens to be emitted.

Until then the value is useful only as a bound to compare against, which is how
the test module uses it (`TRUTH_INF`, never `TRUTH_OPT`).

*Raised in the #1128 review, after merge.*

### The indicator trap

`add_disjunction` gives each `Disjunct` block a **named** indicator, which makes
the select-one reading directly observable — and is where users most often misread
it. Constraining an indicator to `0` means *"this mode is not selected"*, **not**
*"this disjunct's predicate is false"*. With overlapping disjuncts a deselected
block's constraints may still hold at the optimum.

## 3. Naming rules

- The selector-cardinality row is a **select** row, not an XOR row
  (`_disj_<name>_select`, `_gdp_select_<name>`, `_hull_select_<name>`).
- **`EXACTLY_ONE_TRUE`, never `XOR`.** XOR is associative, so over three or more
  operands it means *odd parity*, not "exactly one true" — and `either_or` is not
  limited to two disjuncts. The token `xor` is already spoken for in this codebase
  by the pairwise Boolean operator (the GAMS `boolxor` opcode).
- **No bare `xor=` keyword, ever.** Pyomo.GDP spells select-one as
  `Disjunction(xor=True)`, which is exactly the ambiguity this contract removes;
  the spelling is refused with a message naming both meanings.

## 4. Boolean identities vs. existential auxiliaries

Two kinds of Boolean live in a lowered model and must not be conflated:

- **User Boolean identities** — created by `Model.boolean(...)` or as a
  `Disjunct`'s indicator. They are named by the user, are meaningful in the source
  model, and survive lowering under their own name.
- **Existential compiler auxiliaries** — selector binaries and Tseitin variables
  invented by a lowering. They carry the reserved `_gdp_aux_` prefix, are chosen
  existentially, and have no source-level meaning. Only the forward implication is
  needed for such a variable when it occurs positively in exactly one clause (see
  `_cardinality_implication_constraints`).

The separation is **enforced**, not merely conventional:

- `Model._check_name` — the single choke point for every user-facing variable
  and parameter factory — refuses any name inside the reserved prefix, so a user
  Boolean cannot take an auxiliary's namespace.
- The GDP pass allocates auxiliary names against the names already present in
  the model, so a generated selector cannot reuse a reserved-prefix name that a
  previous pass (or an imported model) already introduced.

Without both, a user Boolean named `_gdp_aux_disj_d_0_0` and the selector the
lowering mints for disjunction `d` collide: two distinct variables share one
name, the identity/auxiliary distinction breaks, and name-keyed result lookup
becomes ambiguous.

## 5. Unimplemented semantics refuse loudly

A lowering that cannot honor the declared semantics **raises**. It never falls back
to one-way activation and never approximates. Serving an `EXACTLY_ONE_TRUE` model
with select-one rows would return a different feasible set than the model declares
and then certify it — a false certificate in the sense of CLAUDE.md §1.

The same rule covers structural gaps: a nested disjunction under `hull` raises
`NotImplementedError` naming big-M as the working path, rather than dying on an
`AttributeError` from inside the disaggregation.

## 6. Local versus certified results (terminology)

Recorded here for the result contract that later PRs in #1123 must use; no result
type changes in PR 0.

- A **certified** result carries a globally valid dual bound. `optimal` means the
  incumbent is proved optimal to tolerance against that bound; `gap_certified`
  distinguishes a real gap from an uncertified one.
- A **local** result is a stationary point of a continuous reformulation (a
  Scholtes homotopy, an NCP-function smoothing). It carries no global bound and
  proves nothing about the discrete optimum.

Two rules follow, and both are load-bearing:

1. **A local result must never occupy the `optimal` status.** The release gate
   keys on status alone — `incorrect_count` skips anything where
   `status != OPTIMAL`, and `proved_optimal_count` counts a result with no bound
   but `OPTIMAL` status as *proved optimal*. A local mode reusing that status
   would be counted as a certificate. A distinct terminal status makes every
   existing consumer fail closed.
2. **A local result may become an incumbent only after independent feasibility
   verification, and may never become a bound.** A feasible point is a valid upper
   bound on a minimum; a dual bound is the certificate and never comes from a local
   solve. This asymmetry is what lets a local mode warm-start the global solver
   without contaminating it.

## 7. The exact continuous (simplex/CNF) lowering — `gdp_method="simplex"`

Issue #1182 (deferred from RFC #1123) adds a third exact lowering beside big-M and
hull, implemented in `discopt._relax.simplex_lowering`. It is **opt-in**, and the
reasons it is opt-in are measurements, recorded in `docs/dev/performance-plan.md`
§26 with the probes in `scratchpad/issue1182/`.

### What it emits

Theorem 1 of Wehbeh & Kerrigan ([arXiv:2601.03906v1](https://arxiv.org/abs/2601.03906v1))
replaces a CNF clause $\bigvee_j [p_{ij}(z) \le 0]$ with

$$\sum_j \lambda_{ij}\,p_{ij}(z) \le 0,\qquad \lambda_i \ge 0,\qquad \sum_j \lambda_{ij} = 1 .$$

A discopt disjunction is $\bigvee_j \bigwedge_k c_{jk}$, so CNF conversion
distributes over the conjunctions and produces $\prod_j |P_j|$ clauses, where an
equality row counts as **two** predicates. That blowup is real and is refused above
`MAX_CNF_CLAUSES` rather than expanded silently; the size quantities — clauses,
literal occurrences, weight variables, rows — are recorded **separately** on
`Model._simplex_lowerings`, because reducing one says nothing about the others.
The fourth quantity, structural Jacobian nonzeros, is a whole-model number and is
read with `structural_jacobian_nonzeros(lowered_model)`; it refuses a model still
carrying a disjunction rather than counting that row as zero.

### What it guarantees, and what it does not

- **Exact in projection onto the model variables.** The lifted problem stays
  nonconvex; exactness is a statement about the projected feasible set. Because the
  objective is a function of the source variables, a certified global solve of the
  lowered model is a certificate for the source model — this path introduces no
  local-only result and needs no local/certified distinction of its own (§6).
- **`ONE_WAY` activation only.** `EXACTLY_ONE_TRUE` is refused, not approximated:
  §3.1 of the paper represents strict negation with an existential exponential lift,
  and substituting a closed inequality or a fixed margin would change the declared
  feasible set (§5's rule).

### The weights are witnesses, not selectors

$\lambda_{ij}$ is an **existential witness** for "some literal of clause $i$ holds".
By §4's taxonomy it is an existential compiler auxiliary — but unlike a selector
binary it is *continuous* in $[0, 1]$, so:

- a fractional $\lambda$ at a feasible point is **not** failed Boolean integrality,
  and must never be reported as one;
- it must not be turned into a recovered named Boolean assignment. Reading truth off
  the weighted row is wrong in a way the source rows expose: at $\lambda = (0.4, 0.6)$
  with $p_1 = 2$ and $p_2 = -4$ the weighted row reads $-1.6 \le 0$ while the first
  literal is violated by $2$.

The supported questions are answered on the **declared** predicates at the returned
point:

- `disjunction_residuals(model, point)` — per disjunction, $\min_j \max_k p_{jk}(z)$,
  each residual carrying the definition string that produced it. The point is an
  argument, never cached state, so a report taken elsewhere cannot stand in for
  source validation. A model declaring no disjunction raises rather than returning an
  empty report that would read as a pass.
- `selected_disjuncts(model, point)` — which disjuncts actually hold. The answer is a
  **set**: under `SELECT_ONE` a point may lie in several disjuncts (§2), and it is
  empty when the disjunction is violated, which callers must treat as a failed
  validation rather than as "the first disjunct".

### Why it is not the default

On every corpus benchmarked in #1182 it is slower than big-M and never certifies
more. On general disjunctions it certifies strictly less; on the MPEC models, where
the exact SOS1 and GDP encodings are also available, all four certify the same
optimum and the lowering is 100–250× slower in wall time with a mixed node-count
signal. It exists for the disjunct rows the other two lowerings **refuse**: big-M
raises when a row's interval enclosure is unbounded, and hull raises
`HullPerspectiveOriginError` when the row is not finite at the origin. Theorem 1
needs neither — its weights are bounded by construction and it forms no perspective.
Scanning 11,058 GDPlib disjunct rows found 18 (in `stranded_gas`, `log` of a capacity
sum whose box includes 0) that hit **both** refusals.
