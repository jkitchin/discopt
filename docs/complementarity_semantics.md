# Complementarity semantics — the source contract

**Status:** normative for the modeling layer, every complementarity lowering,
and every model-rebuilding pass.
**Issue:** [#1147](https://github.com/jkitchin/discopt/issues/1147) (slice 1 of
the MCP/MPEC RFC, [#1123](https://github.com/jkitchin/discopt/issues/1123)),
the companion of [disjunction semantics](disjunction_semantics.md) (#1124).

This document fixes what a complementarity condition *is* at the source level,
so a later consumer — a source residual, a KKT diagnostic, a native MCP backend —
has a contract to read rather than a lowering to reverse-engineer.

## 1. Why the source relation has to survive

`Model.complementarity(x, y)` lowers the condition on the spot: it becomes a
selector binary, big-M rows and generated auxiliaries. Before #1147 the *relation*
was dropped by every pass that rebuilds a `Model` — GDP lowering, integer-product
expansion, factorable reformulation, binary-multilinear linearization — so after
lowering the only surviving trace of a declared pair was its name, baked into
identifiers like `_gdp_aux_disj_pair0_0_0`.

That is not a cosmetic loss. The next slice ([#1148](https://github.com/jkitchin/discopt/issues/1148))
must report a **source** complementarity residual, `r = max_i min(f_i(x), g_i(x))`
over the *declared* operands, separately from the lowered NLP's own residual. With
the source operands gone, such a probe would have measured the relaxed row instead,
printed a small number, and been believed — the instrument-that-measures-nothing
failure the measurement discipline in `CLAUDE.md` §6–§7 exists to prevent.

So: **the relation is the durable object; the rows are a lowering of it.**

## 2. The two forms

### NCP pair — `0 <= f ⊥ g >= 0`

Both operands are declared nonnegative and at most one is nonzero (`f·g = 0`).
This is what `Model.complementarity` and the `.nl` type-5 importer produce, and
the only form any lowering in `discopt.mpec` encodes today (GDP disjunction,
SOS1, Scholtes regularization).

### Box MCP — `F(z) ⊥ z in [l, u]`

A *residual* paired with a *bounded* variable:

| state of `z` | asserted |
|---|---|
| `z = l` | `F(z) >= 0` |
| `l < z < u` | `F(z) = 0` |
| `z = u` | `F(z) <= 0` |

`l = 0, u = +inf` reduces exactly to the NCP pair, and `Model.mcp` records that
case *as* an NCP pair and lowers it, so the two forms cannot diverge on the case
they share. (Bounds at discopt's ±9.999e19 unbounded sentinel are normalized to
infinities first; comparing `ub == inf` literally would misclassify
`m.continuous("a", lb=0)` as a general box.)

The general box form is **represented and not lowered** in this slice. It is a
three-branch condition, so handing it to a two-branch lowering would encode a
different feasible set and certify it. Both entry points refuse loudly instead:
`mpec.reformulate_{gdp,sos1,scholtes}` raise `NotImplementedError`, and the
**solver boundary** refuses any model carrying a relation no lowering emitted
rows for — a declared condition solved as if it were absent is a false
certificate, not a missing feature.

That boundary is `solve_model`, not `Model.solve`. `Model.solve` checks too (so
an alternate backend is refused before it dispatches), but the check must sit
where every solve passes through: the differentiable-solve paths and the primal
heuristics call `solve_model` directly, and with the guard only on `Model.solve`
they certified `optimal` on a point the declared relation forbids. Use
`mpec.require_all_relations_lowered(model, context=...)` from any new entry
point.

## 3. What the relation carries

`mpec.Complementarity` is compared and hashed **by identity** (`eq=False`): a
relation is a node in the model's IR, not a value, so it stays a usable dict/set
key as passes carry it from one `Model` to the next. (Value equality was never
available: `Expression.__eq__` builds a `Constraint`.)

| field | meaning |
|---|---|
| `f`, `g` | the **source** operands, never rewritten by a lowering or a pass |
| `role` | `ComplementarityRole`: `NCP_PAIR`, `BOX_MCP`, `FROM_KKT`, `FROM_DISJUNCT` |
| `f_bounds`, `g_bounds` | the bounds the *relation* declares — its semantics, not the operand's interval enclosure, which presolve moves |
| `scale` | characteristic residual magnitude, for a meaningful tolerance when the operands carry unrelated physical magnitudes; `effective_scale` derives one from the declared bounds when it is not given |
| `parent` | identity of the construct that generated the relation (the lower-level row for a KKT pair, the disjunction for a disjunct pair) |
| `f_shape`, `g_shape`, `index`, `source` | shape/index information for vectorized relations: `elements(model)` yields one scalar relation per index, each with its multi-index and a back-pointer to the declared relation (a scalar relation yields `[self]`, so the identity key holds there too) |
| (no lowering field) | which models carry the generated rows, and by which method, is **model** state — `Model._lowered_complementarities`, read via `pair.is_lowered_into(model)` / `pair.lowering_in(model)` |

`role` is *provenance*, never a lowering switch. Every lowering and every bound
rule branches on `Complementarity.is_symmetric_nonnegative` — the **declared
bounds** — and never on `role`. This is not a stylistic preference: `Complementarity`
is public and constructible directly, so a relation with `g_bounds=(-1, 1)` and
the default `role=NCP_PAIR` reaches the lowerings. A `role`-gated refusal let
exactly that through and emitted `g >= 0` against a declared `lb = -1`, certifying
a point the relation forbids. A label can be wrong; the declared bounds *are* the
semantics.

`FROM_DISJUNCT` is reserved vocabulary with no in-tree producer yet — exactly as
`DisjunctionSemantics.OR` was defined by #1124 before any lowering emitted it.

The lowering marks — **and the method** — live on the model
(`Model._lowered_complementarities`, an identity map from relation to method),
not on the relation. The mark was weak references on the relation first, which
made a model carrying one unpicklable and made `copy.deepcopy` silently drop it,
so the clone refused to solve. The *method* then stayed behind as a plain field,
which was the same mistake one level down: lowering a relation by GDP into `m1`
and by SOS1 into `m2` left the field reading `"sos1"`, so `m1`'s provenance no
longer described its own disjunctive rows.

The rule this settles: **a relation is shared; what a given model did with it is
that model's fact.** Ask the model that carries the rows you are reasoning
about, never the relation.

## 4. Provenance is keyed to objects, never to indices or names

Presolve/FBBT eliminate and renumber variables, so an index-keyed map is stale
exactly when it becomes useful; and a name-keyed one matches a generated
`_gdp_aux_disj_pair0_0_0` that is not the source operand at all. Therefore:

* `resolve_source_variables(model, pair, context=...)` resolves by **object
  identity** and raises `ComplementarityProvenanceError` — naming the relation
  and the caller — when the model does not hold one of the source variables.
* `flat_source_indices(model, pair)` derives backend-facing flat columns **on
  demand at the solver boundary**; no index is ever persisted on the relation. It
  is element-accurate for a statically indexed operand (`x[2] ⊥ y` returns the two
  columns it reads, not all of `x`), and widens to the whole variable for an index
  it cannot resolve statically — conservative, never narrower than the truth, so a
  consumer can rely on the set covering everything the relation reads.
* Those columns are prefix sums over **the target model's own `_variables`**, not
  `Variable._index`. `_index` is the variable's position in the model it was
  *declared* on, and a relation's operands are shared objects: a model holding a
  two-element `x` and a scalar `y` as `[y, x]` has them at `[1, 2]` and `[0]`,
  while `_index` reports `[0, 1]` and `[1]`. `Model._flat_var_offset` is right for
  the rebuilding passes (they preserve declaration order) and wrong as a general
  provenance accessor. A provenance query must also never *write* shared state —
  no renumbering, no renaming — since other live models hold the same objects.
* An expression node the walker cannot descend into raises rather than
  contributing nothing, so resolution can never report success without having
  looked at the operand.

## 5. What every rebuilding pass owes the relation

A pass that constructs a fresh `Model` **must** call
`mpec.carry_complementarities(src, dst, pass_name=...)`. It forwards the same
relation objects (so identity-keyed provenance holds across any number of
passes), checks each against `dst` first, and propagates the lowering mark with
`src`'s own method — `dst` was rebuilt from `src`'s rows, so it already carries
the generated ones and must not be re-lowered.

The propagation test is `pair.is_lowered_into(src)` and nothing weaker. Keying it
on "the relation has been lowered *somewhere*" let an unlowered `src` sharing a
relation with another model hand `dst` a lowered mark, so `dst` claimed rows
neither model had and walked straight past the solve guard of §2.

Passes with a defensive `except Exception: return model` handler re-raise
`ComplementarityProvenanceError` ahead of it. A broken provenance chain is not an
out-of-scope model; degrading it to a silent abstain is the failure mode this
contract exists to remove.

One guard was *removed* under this contract rather than kept:
`binary_multilinear_reform` used to abstain whenever `_complementarities` was
non-empty — a guard keyed on state every earlier pass emptied, so its premise
evaporated at a pass boundary. Forwarding the relation set replaces it.


## 6. Naming

A relation's name is the base name of every row generated from it, so an unnamed
relation must not reuse one. `_ensure_relation_name` assigns `compl{k}` unique
against the model's existing relation and constraint names, at the single point
every lowering funnels through (`_scalarize_pairs`) — not at a call site. The
fallback previously counted *within the list handed to the lowering*, and
`Model.complementarity` hands over exactly one relation, so every unnamed
declaration on a model became `compl0` and emitted `compl0_f_nonneg` twice.


## 7. Source residuals — what a number means

**Status:** normative for every consumer that reports a complementarity residual.
**Issue:** [#1148](https://github.com/jkitchin/discopt/issues/1148) (slice 2),
implemented in `discopt.mpec_report`.

### The two residuals are different numbers

POUNCE's internal barrier/KKT complementarity is a property of the **generated
NLP**. The MPCC complementarity `0 <= f ⊥ g >= 0` is a property of the **user's
model**. They are not the same measurement and must never be reported as if they
were.

Measured on `max x + y  s.t.  x == y, 0 <= x ⊥ y >= 0`, after a Scholtes homotopy
stopped at `t = 1e-8`:

| quantity | value |
|---|---|
| max violation of the generated rows (`x >= 0`, `y >= 0`, `x·y <= t`) | `1.0e-8` |
| source complementarity `min(x, y)` on the declared operands | `1.4e-4` |

Four orders of magnitude, at the same point, because the *regularized* problem
admits `x = y ≈ sqrt(t)` and every row it holds is satisfied there. A result that
reports only the first number says the point is complementary. It is not: the
true MPCC forces `x = y = 0`, and the returned objective (`-2.8e-4`) is *below*
the true optimum (`0`). `SourceResidualReport` carries both, side by side.

### What is reported, and its definition

Every residual is a `Residual` carrying `value`, the `definition` string that
produced it, the `scale` it was measured against, and the algorithmic `floor`
below which it cannot be driven. A number without its formula cannot be compared
across solvers, which is why the definition is a field and not a docstring.

| residual | definition |
|---|---|
| `source_complementarity` | `max_i min(f_i(x), g_i(x))` for a nonnegative pair; the normal map `\|z - mid(l, u, z - F(x))\|` for a box MCP. `product` and Fischer–Burmeister are selectable. |
| `source_operand_bounds` | violation of the bounds the **relation** declares on each operand (nonnegativity for an NCP pair, the box for an MCP) |
| `source_primal_feasibility` | max violation of the **source** constraint rows and the **original** declared variable bounds, on the source expression tree |
| `selector_integrality` | `max_i y_i(1 - y_i)` over binary selectors, **after** checking `0 <= y_i <= 1` |
| `lowered_row_residual` | max violation of the rows the lowering generated — the comparison quantity, never the source claim |

Which definition is used is chosen from the relation's **declared bounds**, never
from its `role` — the same rule §3 states for the lowerings, and for the same
reason.

### Scale, and the accuracy floor

A uniform tolerance is meaningless when the two operands carry unrelated physical
magnitudes (a multiplier in 1e-6 against a flow in 1e3), so a residual is reported
both raw and divided by the relation's `effective_scale` (§3). A Scholtes
regularization at `t` cannot attain a complementarity better than `sqrt(t)` —
POUNCE Gate 0's documented floor
([jkitchin/pounce#794](https://github.com/jkitchin/pounce/issues/794)) — so
`Residual.floor` carries it and `at_floor` says whether the number is at it. A
report that omitted the floor would imply exact orthogonality.

### The instrument checks itself

Operands are evaluated with the interval evaluator over a **degenerate** box
(`[x, x]` per variable, keyed by object identity). A correct evaluation stays
degenerate; the evaluator widens to `[-inf, +inf]` for an atom it cannot walk, so
a widened result proves the graph was *not* evaluated and `evaluate_at_point`
raises rather than reporting a midpoint. Likewise `relation_residuals` refuses
when it is handed relations and measures none, and every constraint-list entry
kind (`Constraint`, `_DisjunctiveConstraint`, `_SOSConstraint`) has an explicit
residual or raises — a row silently skipped would make the maximum read as
complete. This is CLAUDE.md §6 applied to the instrument itself.


## 8. Local results versus certificates — the contract

**Status:** normative for every solver path that can return a local result.
**Issue:** [#1148](https://github.com/jkitchin/discopt/issues/1148) §B/§C.
Vocabulary in `discopt.status`; enforcement in `SolveResult.__post_init__`.

### A distinct terminal status, not a flag

`discopt.status` defines two statuses that make **no global claim**:

* `local_optimal` — a local stationary point of the problem actually solved;
* `local_infeasible` — a local solver found no point. This is **not** an
  infeasibility proof. A stalled MPEC continuation lands here and never on
  `infeasible`, which is a certificate in the other direction.

It is a status and not a `certified=False` flag beside `"optimal"` because every
consumer that pattern-matches on the *status* would inherit the bug. Measured on
the benchmark harness: `is_solved` is `status == OPTIMAL`, and
`proved_optimal_count` counts `gap is None and is_solved` as a proved optimum — a
local stationary point reported as `"optimal"` is scored by the release gate as a
**certificate**, which is CLAUDE.md §1's hard gate with no slack. An unmapped
status maps to `UNKNOWN` and both counters skip the row, so a consumer that has
never heard of the local mode fails closed. `DISCOPT_STATUS_MAP` names
`local_optimal`/`local_infeasible` explicitly so that skip is intentional and
tested rather than accidental.

The states the vocabulary keeps apart: certified optimal; certified bound +
incumbent with a gap; feasible with no bound; local stationary with no global
claim; and certified-infeasible versus local-search-failed.

### The asymmetry that makes a local mode safe

> A local result **may become an incumbent only after independent feasibility
> verification**, and **may never become a bound.**

A feasible point is a valid upper bound on a minimum; a dual bound is the
certificate and never comes from a local solve. That asymmetry is what lets a
local mode feed the global solver without contaminating it. It is enforced
structurally, not by convention: `SolveResult.__post_init__` **raises** if a
local status carries a `bound` or a `root_bound`, and forces `gap_certified` to
`False`. Refusing rather than silently clearing the field is deliberate — a
caller that set one has a bug in what it believes it proved.
`mpec_report.accept_local_incumbent` is the verification side: it runs the shared
`validation.feasibility.verify_point` and returns `None` rather than vouching for
a point it cannot verify.

Two further rules:

* **Validation is performed on the source model**, not only on the generated NLP.
  A smoothing tolerance can be small while a truth value near a strict predicate
  boundary has already flipped — §7's table is that case.
* **Absent certification is interpreted as not certified.**
  `status.is_certified_status` is a membership test against a closed set, never
  `not is_local_status(...)`.

### Stationarity is not claimed

C-/M-/S-stationarity classifications are reported **only** if discopt has actually
checked the required conditions. It has not, so `SourceResidualReport.stationarity`
is `None` and no code path sets it. Reporting a classification the solver did not
verify would be exactly the §1 error this document exists to prevent.

### One return type

`mpec.solve_mpec` returns a `modeling.core.SolveResult` for `scholtes`, `sos1`
**and** `gdp`. Before #1148 switching one keyword changed the returned type, the
type of `status` (enum versus str) and whether a certification field existed at
all, so no caller could write one branch that read a result. The Scholtes arm
carries its continuation trace — per-stage `t`, subsolver status, accept/reject
reason and the achieved source residual — rather than discarding it, so
"converged at 1e-8" and "stalled at 1e-2" are distinguishable from the result
alone.
