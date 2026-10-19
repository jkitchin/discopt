# The discopt model document format

The native format written by `Model.save()` / `discopt.modeling.dumps()` and read by
`discopt.load()` / `loads()` (`python/discopt/serialize.py`). One JSON object,
optionally gzipped. The user guide is `docs/notebooks/model_serialization.ipynb`;
the plan that completed the format is `docs/dev/serialization-completion-plan.md`.

## Compatibility

A document written at schema `discopt.model/1.N` loads in any later discopt whose
major is 1. A newer document loads in an older discopt only when it uses nothing
that discopt does not know; otherwise a reader from schema 1.5 on refuses it and
says to upgrade. Readers older than that (1.0–1.4) skip a key they do not know
inside a record, so a few newer documents load into them without that content: the
fixture manifest (`python/tests/data/serialize_fixtures/MANIFEST.json`, *forward*
cells) records every case, and the one known misread is a 1.5 disjunction's
`indicators` link. Every content added since is a new row kind or index kind, which
every shipped reader refuses. `pickle` is same-version transport only: a pickle of
another layout is refused, and one of another discopt version with the same layout
warns.

`discopt upgrade-model IN [OUT]` rewrites a document at the current schema (in place
without `OUT`; `--check` only reports).

## How a reader treats a document

| Document | Reader does |
|---|---|
| a different major | refuses |
| this minor or older, only keys a writer of that minor produced | loads |
| this minor or older, a key no writer of that minor produced | refuses: the document is malformed or hand-edited |
| a newer minor, only keys this reader knows | loads, with `SchemaVersionWarning` |
| a newer minor, a key, node op, row kind or index kind this reader does not know | refuses, naming it, and says to upgrade |
| a non-numeric minor | refuses |

The rule holds inside every record (variables, nodes, rows, builder blocks), not
only at the top level: the reader records which keys it read and refuses the rest.

## Schema history

| minor | added |
|---|---|
| 0 | the format: variables, parameters, a flat expression node table, objective, rows (algebraic, indicator, SOS, disjunction), builder blocks, complementarities, state, initial point, embedded solution |
| 1 | `provenance` |
| 2 | a `bool` index kind and a list sum axis; the solution subtree no longer tagged |
| 3 | the registered-atom tag |
| 4 | `state`: `nl_source`, `_gams_initial_values`, `_recentre_inner` |
| 5 | `logical` rows; the rows a complementarity lowering emitted; `indicators` on a disjunction row |
| 6 | `none` / `ellipsis` index kinds; a disjunction as a disjunct member |
| 7 | `block_indicator` and `nested_disjunction` rows; `state._gdp_blocks` (Disjunct-block members, local variables) |

## Top-level keys

<!-- generated: _TOP_LEVEL_KEYS (test_serialize_docs_match.py) -->
| key | since minor |
|---|---|
| `schema` | 0 |
| `discopt` | 0 |
| `name` | 0 |
| `variables` | 0 |
| `parameters` | 0 |
| `nodes` | 0 |
| `objective` | 0 |
| `builder_objective` | 0 |
| `rows` | 0 |
| `builder_blocks` | 0 |
| `complementarities` | 0 |
| `state` | 0 |
| `initial_point` | 0 |
| `solution` | 0 |
| `provenance` | 1 |
<!-- end generated -->

## `state` keys

<!-- generated: _STATE_KEYS (test_serialize_docs_match.py) -->
| key | since minor |
|---|---|
| `_aux_counter` | 0 |
| `_decomp_stages` | 0 |
| `_decomp_blocks` | 0 |
| `_zero_spanning_factor_auxes` | 0 |
| `_coupling_keys` | 0 |
| `_sets` | 0 |
| `_simplex_lowerings` | 0 |
| `_atan2_preconditions` | 2 |
| `_block_labels_var` | 2 |
| `_block_labels_con` | 2 |
| `_implied_integer_auxes` | 3 |
| `_free_by_construction` | 3 |
| `_piecewise_domains` | 3 |
| `_recentre_inner` | 4 |
| `_gams_initial_values` | 4 |
| `nl_source` | 4 |
| `_gdp_blocks` | 7 |
<!-- end generated -->

## Refused, permanently

These raise `SerializationError` on save. None is a section waiting to be written.

<!-- generated: refusals (test_serialize_docs_match.py) -->
| what | why | instead |
|---|---|---|
| `dm.custom` (a `CustomCall`) | wraps an arbitrary Python callable; its name would reload into a different model | algebraic expressions, or rebuild it in code |
| output of `binary.multilinear` | records point back at the source model; some feed a bound | save the source model, re-apply the pass after loading |
| output of `integer.bilinear / integer.multilinear` | records point back at the source model; some feed a bound | save the source model, re-apply the pass after loading |
| output of `nonsmooth.epigraph` | records point back at the source model; some feed a bound | save the source model, re-apply the pass after loading |
| output of `gdp.transform_current_disjunctive_state` | records point back at the source model; some feed a bound | save the source model, re-apply the pass after loading |
| a `Model.disjunction()` / `Disjunct` never attached | `validate` refuses the model anyway (#1430) | attach it, or drop it |
| an unclassified private attribute | dropping it would reload a different model | classify it in `serialize.py` |
<!-- end generated -->
