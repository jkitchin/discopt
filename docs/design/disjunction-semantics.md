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

`DisjunctionSemantics` members are defined *as* the pair, and lowerings branch on
the pair rather than on the member name. Adding a combination later (an
optional-mode disjunction is `(ONE_WAY, AT_MOST_ONE)`) is then an added member,
not a redefinition of the enum's meaning.

| member | pair | status |
|---|---|---|
| `SELECT_ONE` | `(ONE_WAY, EXACTLY_ONE)` | **implemented** — the meaning of `either_or` |
| `OR` | `(ONE_WAY, AT_LEAST_ONE)` | declared, not implemented |
| `EXACTLY_ONE_TRUE` | `(REIFIED, EXACTLY_ONE)` | declared, not implemented |

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

A user Boolean is never given the reserved prefix, and no auxiliary is ever
presented as a source identity.

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
