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
`mpec.reformulate_{gdp,sos1,scholtes}` raise `NotImplementedError`, and
`Model.solve` refuses any model carrying a relation no lowering emitted rows for
— a declared condition solved as if it were absent is a false certificate, not a
missing feature.

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
| `f_shape`, `g_shape`, `index`, `source` | shape/index information for vectorized relations: `elements(model)` yields one scalar relation per index, each with its multi-index and a back-pointer to the declared relation |
| `lowering` + per-model marks | which method lowered it, and which models already carry the generated rows |

`role` is *provenance*, never a lowering switch: a lowering branches on the
declared operand bounds, never on the role. `FROM_DISJUNCT` is reserved
vocabulary with no in-tree producer yet — exactly as `DisjunctionSemantics.OR`
was defined by #1124 before any lowering emitted it.

## 4. Provenance is keyed to objects, never to indices or names

Presolve/FBBT eliminate and renumber variables, so an index-keyed map is stale
exactly when it becomes useful; and a name-keyed one matches a generated
`_gdp_aux_disj_pair0_0_0` that is not the source operand at all. Therefore:

* `resolve_source_variables(model, pair, context=...)` resolves by **object
  identity** and raises `ComplementarityProvenanceError` — naming the relation
  and the caller — when the model does not hold one of the source variables.
* `flat_source_indices(model, pair)` derives backend-facing flat columns **on
  demand at the solver boundary**; no index is ever persisted on the relation.
* An expression node the walker cannot descend into raises rather than
  contributing nothing, so resolution can never report success without having
  looked at the operand.

## 5. What every rebuilding pass owes the relation

A pass that constructs a fresh `Model` **must** call
`mpec.carry_complementarities(src, dst, pass_name=...)`. It forwards the same
relation objects (so identity-keyed provenance holds across any number of
passes), checks each against `dst` first, and propagates the lowering mark —
`dst` was rebuilt from `src`'s rows, so it already carries the generated ones and
must not be re-lowered.

Passes with a defensive `except Exception: return model` handler re-raise
`ComplementarityProvenanceError` ahead of it. A broken provenance chain is not an
out-of-scope model; degrading it to a silent abstain is the failure mode this
contract exists to remove.

One guard was *removed* under this contract rather than kept:
`binary_multilinear_reform` used to abstain whenever `_complementarities` was
non-empty — a guard keyed on state every earlier pass emptied, so its premise
evaporated at a pass boundary. Forwarding the relation set replaces it.
