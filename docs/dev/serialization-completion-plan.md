# Model serialization — completion plan

> **Plan v1.11 — 2026-10-19 — author: Claude (planning session for jkitchin).**
> Status: proposed; no work package started. Every claim about the code below was
> read from the tree at `3191cc65` (schema `discopt.model/1.4`, `discopt 0.9.1.dev0`,
> after PRs #1748 and #1764). Agents executing a package **re-verify** the lines they
> rely on first (§0.4) and bump the revision header + §9 change log when they change
> this document.

This plan closes the model serialization subsystem — `python/discopt/serialize.py`
(the native JSON format), the pickle/deepcopy/clone path in
`python/discopt/modeling/core.py`, `python/discopt/result_io.py` (the embedded
`SolveResult`), and `python/discopt/provenance.py` — to a stated, testable
**Definition of Done** (§2). It is written to be executed one work package per
Opus agent session, one PR each (§4), in the order of §5.

---

## 0. Implementation contract (binding on the implementing agent)

A PR that violates a clause is wrong even if its tests are green. These are
CLAUDE.md's rules applied to this subsystem; CLAUDE.md wins on any conflict.

### 0.1 Correctness of the *document* is the product

1. **No silent drop, no silent approximation.** Every model element either
   round-trips byte-exactly (`dumps(loads(d), provenance=False) == d`) or is refused
   with `SerializationError` naming the element and the remedy
   (`_refuse_unsupported`, `_refuse_unclassified_attributes`). A reader that skips an
   unknown `op`, key-kind or relation is a defect, not a compatibility feature.
2. **Reads are strict** (`_StrictRefs`, `_dec_index_int`, `_KNOWN_SENSES`,
   `_KNOWN_DISJUNCTION_METHODS`, `_check_index_in_range`, `_check_axis_in_range`).
   A new section gets the same treatment on the first PR that writes it, with
   malformed-document cases added to `MALFORMED` in
   `python/tests/test_serialize_adversarial.py` in the same PR.
3. **Anything touching a solve path is bound-neutral** (CLAUDE.md §5): a reloaded,
   unpickled or cloned model must give **exactly** the same `node_count`, `status`
   and `objective` as the original (`test_solve_is_bound_neutral_after_round_trip`,
   `test_pickle_of_a_solved_model_is_bound_neutral` are the pattern). "Close" is a
   failure. This covers `_source_still_matches` in `solvers/nlp_native.py`, the
   `_DetachedNlRepr` guard, lowering marks, `_free_by_construction`,
   `_disjunctive_config_floor` (never carried) and the decomposition annotations.
4. **Never weaken a refusal to make a test pass.** If a builder (§4 WP-6) produces a
   model the format refuses, the outcomes are: an encoding (a schema minor bump,
   §3.1), or a *documented final refusal* with its remedy. Removing the attribute
   from `_refuse_unclassified_attributes`'s view is not an option.

### 0.2 Measurement discipline (CLAUDE.md §6–§11)

5. Every probe/harness/generator in this plan **prints an executed-check count and
   exits non-zero when it is zero** (`test_corpus_probe_actually_fired`,
   `test_the_malformed_probe_fired` are the in-tree pattern). A corpus harness that
   finds no files prints *why* and fails, it does not pass vacuously.
6. **No swallowed exceptions in an instrument.** A fuzzer classifies exceptions; it
   never `except Exception: pass`.
7. **Verify which code is loaded** before any measurement or fixture generation:
   assert `discopt.__file__` is under the intended tree **and** a marker unique to
   the version under test (for fixtures: `discopt.serialize.SCHEMA ==
   "discopt.model/1.N"`; for a baseline run: the marker *absent*).
8. **Timing claims** (WP-7) need `uptime`/load check, interleaved A/B, ≥ 5 repeats,
   and a reported sd. Sequential A then B is not a measurement.
9. A measurement that contradicts this plan wins; record the falsification in the
   affected §4 package *in this file* before continuing (dated, in the style of
   `performance-plan.md` §6).

### 0.3 Versioning discipline (defined in §3, binding from WP-2 on)

10. Any PR that changes the document shape bumps `SCHEMA`'s minor (or major per
    §3.1), adds the new version's fixtures to the golden corpus (§3.3) in the same
    PR, and extends the compatibility matrix test. A shape change without a bump is
    a defect (the 1.2 lesson in `serialize.py`'s `SCHEMA` comment).
11. The golden fixtures for versions **already shipped** are immutable; a test that
    needs them changed is wrong.

### 0.4 Workflow

12. One package per PR on a feature branch; title names the package
    (`serialize:WP-3 …`). Every PR runs and reports:
    `pytest -m smoke`, `pytest -m slow python/tests/test_adversarial_recent_fixes.py`,
    `pytest python/tests/test_serialize_roundtrip.py python/tests/test_serialize_adversarial.py python/tests/test_model_pickle.py python/tests/test_deep_expression_walks.py python/tests/test_1302_review_followups.py python/tests/test_1248_atom_survives_serialization.py python/tests/test_1290_serialize_bool_index_tuple_axis.py python/tests/test_transformations.py -m ""`,
    plus `ruff check python/ && ruff format --check python/ && mypy python/discopt/`.
    `cargo test -p discopt-core` when `crates/` is touched (none of the packages
    below should need to).
13. New behaviour ships with a regression test that **fails before / passes after**;
    the PR description names the test and quotes the failure.
14. Grep before naming an attribute (`model._variables`, `model._constraints`,
    `Constraint.body`, `Model._objective`, module-level `dm.from_nl`). Guessing
    attribute names is a documented cause of invisible no-op probes.
15. Update §8 (progress table) and, if the plan changed, §9 in the same PR.
    `Closes #N` only when the package finishes an issue; otherwise `Contributes to`.
16. **Stop and escalate** (do not improvise) when: a kill criterion fires; a
    fixture from a shipped version fails to load and the cause is not found in one
    session; meeting a goal would require relaxing a §0.1 refusal; the owner
    decisions in §7 are needed and not yet taken.

---

## 1. Current state (verified at `3191cc65`; the gaps this plan closes)

**Done.** `dumps`/`loads`/`save`/`load` round-trip variables (incl. the
`_bound_stack` of a `fix()` scope), parameters, objective, algebraic rows, fast-API
builder blocks and builder objective, indicator/SOS/disjunction, complementarity
relations with lowering *marks*, initial point, `_gams_initial_values`, atan2 and
piecewise guards, `_sets`, decomposition annotations (`_decomp_*`, `_coupling_keys`,
`_block_labels_*`), registered-atom tags (re-derived on load by `_retag_atoms` /
`lowering_reproduces`), `from_nl` provenance (`_DetachedNlRepr` marker +
`_source_nl_path` + `_source_nl_sha256`), provenance block with
`ProvenanceSkewWarning`. Attributes attached after `__init__` are classified in
`_MODEL_STATE` / `_LAZY_STATE`; unknown private ones are refused by name;
`_bml_*/_ipx_*/_nsl_*` reformulation records are refused with the remedy.
`pickle`/`deepcopy`/`clone` work for solved, `from_nl`, fast-API, deep (50k-node),
id-keyed-mark and registered-atom models (`Model.__getstate__`/`__setstate__`,
`__deepcopy__`, `_row_marks_by_position`, `_PICKLE_SCOPES`/`_ChildrenFirstEnd`,
`RegisteredFunction.__reduce__` → `_PickledAtomRef`). `solve_batch` ships models as
`dumps` text. 66 in-repo MINLPLib `.nl` files round-trip `.nl`-byte-identically
(`slow`); 4 of them are solve-bound-neutral.

**Gaps** (numbering reused throughout):

| # | Gap | Closed by |
|---|---|---|
| G1 | Refused: `CustomCall` (`dm.custom`), `_LogicalConstraint`/`BooleanVar` (`Model.logical`, `Model.boolean`), reformulation outputs with back-references | WP-3 (decisions in §3.6) |
| G2 | Lossy: `LoweringRecord.rows` reloads as `None`; per-solve state; `Variable._fix_scope_tokens`; parsed `.nl` only as marker; user public attributes | WP-3 (`rows`); rest become *documented* non-goals (§3.6) |
| G3 | Versioning: no forward-compat policy; no golden fixtures for 1.0–1.4; pickle carries no version; no major-bump/deprecation/migration policy; provenance skew only warns | WP-1, WP-2 |
| G4 | Testing: no property-based round trip, no fuzzing of `loads`, full MINLPLib/QPLIB never round-tripped, builders (DAE/ML/GDP factory/stochastic/RO/bilevel) never round-tripped | WP-4, WP-5, WP-6 |
| G5 | Performance: `dumps`/`loads` never benchmarked at scale; model pickle +19.6% (stated in the #1764 session; **not recorded in-tree** — must be re-measured) | WP-7 |
| G6 | Security: `loads` hardened but unfuzzed; pickle inherently unsafe and undocumented as such in the API reference | WP-4, WP-8 |

Environment facts an agent must plan around: this clone is **shallow** (405
commits; `git rev-parse --is-shallow-repository` → true; `git log -S'SCHEMA = "discopt.model/'`
finds only one commit), so historical writers need `git fetch --unshallow` or the
owner's full clone (§3.3). `hypothesis` is **not** installed and not in
`[project.optional-dependencies] dev` (`pyproject.toml:95`). Tag `v0.9.0` writes
`discopt.model/1.3`; `v0.8.0` has no `serialize.py`; 1.0–1.2 lie between the two tags.
The full MINLPLib/QPLIB corpora live at `~/Dropbox/projects/discopt-minlp-benchmark/`
only on the owner's machine (`python/tests/test_qplib_corpus.py` already uses
`DISCOPT_QPLIB_DIR` + `skipif` with a printed reason — copy that pattern).

---

## 2. Definition of Done

**Serialization is complete when every row below is green on `main`**, each proven
by the named test in the named CI lane (lanes: *fast* = `python-fast` in
`.github/workflows/ci.yml`; *slow* = `python-correctness-slow` / manual +
weekly cron added by WP-8; *local* = owner's machine, result recorded in §8).

| DoD | Criterion ("complete when …") | WP | Proof |
|---|---|---|---|
| D1 | Every document written by any shipped schema version 1.0–1.4 (and every later minor) loads with the current reader into a model whose `to_nl()` equals the text recorded at generation, and re-saves at the current schema without change on a second reload | WP-1 | `test_serialize_compat_matrix.py::test_every_golden_fixture_loads` (fast) |
| D2 | The compat matrix records, for the current writer vs every shipped reader, one of {loads identical, refuses loudly}; no cell is "loads a different model" | WP-1 | `test_serialize_compat_matrix.py::test_forward_matrix_has_no_silent_misread` (fast) |
| D3 | A reader's behaviour on an unknown minor/major and on an unknown top-level key is the one §3.2 specifies, tested per case | WP-2 | `test_serialize_versioning.py` (fast) |
| D4 | A pickled `Model` carries a format stamp; unpickling across a different stamp behaves as §3.4 specifies (owner decision O1) | WP-2 | `test_model_pickle.py::test_pickle_carries_a_format_stamp`, `::test_mismatched_pickle_format_is_refused` (fast) |
| D5 | `discopt upgrade-model` (load + re-save at current schema, `--check` dry run) exists and is tested | WP-2 | `test_cli_upgrade_model.py` (fast) |
| D6 | Propositional logic constraints round-trip (schema 1.5); `CustomCall` and reformulation records are *final* refusals with the remedy in the message and the docs | WP-3 | `test_serialize_roundtrip.py::test_logical_constraints_round_trip*`, existing refusal tests (fast) |
| D7 | `LoweringRecord.rows` round-trips by row position (1.5) | WP-3 | `test_serialize_roundtrip.py::test_lowering_record_rows_round_trip` (fast) |
| D8 | A hypothesis-generated model (variables of every type/shape, every op in `_KNOWN_OPS`/`_known_funcs()`, every relation kind, shared subexpressions, fixed scopes) satisfies: `dumps` stable under reload; `model_fingerprint` equal; `pickle` ≡ `dumps` on fingerprint — ≥ 50 examples in fast, ≥ 2000 in slow | WP-4 | `test_serialize_property.py` (fast + slow) |
| D9 | `loads` on mutated/random input raises only `SerializationError` (never another exception, never hangs past the deadline) and never returns a model whose re-dump equals the unmutated document when a semantic field was mutated | WP-4 | `test_serialize_fuzz.py` (fast + slow) |
| D10 | Every in-repo corpus file (`python/tests/data/minlplib_nl/*.nl` 66, `data/qplib/qplib/*.qplib` 9, `data/binary_nl/*`, `data/mpcc_complementarity.nl`) round-trips `.nl`-byte-identically and pickle/`dumps` agree on fingerprint; a 12-instance panel is solve-bound-neutral | WP-5 | `test_serialize_corpus.py` (slow) |
| D11 | ~~The full MINLPLib (1,610) and QPLIB (421 usable) corpora round-trip with 0 failures when present~~ **Dropped by the owner (2026-10-18, #1792): a full-corpus run is not needed.** What remains: absent corpora skip with a printed reason and the run still asserts its executed count ≥ the in-repo floor; the harness can still be pointed at the full corpora | WP-5 | `scripts/serialize_corpus_roundtrip.py` |
| D12 | Every model-producing builder in the package (§4 WP-6 table) is either round-trip-tested or listed with a final, tested refusal | WP-6 | `test_serialize_builders.py` (fast where the build is < 1 s, else slow) |
| D13 | `dumps`/`loads`/`pickle`/`deepcopy` have a recorded size/time ladder and a budget test; the pickle overhead vs the pre-#1764 baseline is re-measured with sd and either cut or recorded as accepted (owner decision O2) | WP-7 | `test_serialize_perf_budget.py` (slow); `docs/dev/data/serialize-perf-2026-*.jsonl` |
| D14 | Users can read, in one place, what is carried, what is refused, what the version policy guarantees, and why pickle is not a storage format; module docstrings match | WP-8 | notebook builds with 0 warnings; `test_serialize_docs_match.py` asserts the refused-list in the docs equals `_REFUSED_RELATIONS` ∪ the `CustomCall` check (fast) |
| D15 | CI runs D1–D9, D12, D14 on every PR with ≤ +25 s added wall to `python-fast`; D10, D13 in the slow lane; D11 on demand | WP-8 | `ci.yml` diff; `--durations=20` output quoted in the PR |

---

## 3. Versioning design

### 3.1 Schema version policy (`SCHEMA = "discopt.model/<major>.<minor>"`)

- **Minor bump — additive or disambiguating.** Any change a *current* reader with
  the new code must act on, including: a new top-level key or section
  (`provenance` in 1.1, `nl_source` in 1.4), a new node `op`/index kind/axis
  spelling (1.2), a new tag (1.3), a key whose *absence* changes meaning (1.4's
  "not read from a file" vs "writer did not record it"). Rule of thumb from the
  `SCHEMA` comment: if the minor is what lets the reader tell two encodings apart,
  it must bump.
- **Major bump — non-additive.** Renaming or removing a key the reader requires,
  changing a value's type or unit, changing the node-table id scheme, changing the
  meaning of an existing field. A major bump requires an upgrade path (§3.5) and a
  deprecation window of **one minor release** during which the writer still writes
  the old field *and* the new one, and the reader accepts both.
- **Writer always writes the current `SCHEMA`.** There is no "write as 1.3" option;
  a document for an older reader is produced by that older discopt (`pip install
  discopt==X`), never by the current one pretending. (Rationale: a down-writer
  would have to drop content, which §0.1 forbids.)
- **Deprecation**: a deprecated key is listed in a module-level
  `_DEPRECATED_KEYS = {key: (since_minor, replacement)}`; reading it logs at
  `DEBUG` and migrates; writing it stops at the next major. Nothing is currently
  deprecated.

### 3.2 Reader behaviour (implemented by WP-2; today's behaviour in brackets)

| Document | Reader does |
|---|---|
| major ≠ `SCHEMA_MAJOR` | refuse (`SerializationError`) — [same today] |
| minor ≤ reader's, only known top-level keys | load strictly — [same] |
| minor ≤ reader's, **unknown top-level key** | **refuse**: a writer at that minor never wrote that key, so it is malformed or hand-edited — [today: ignored] |
| minor > reader's, only known top-level keys | load strictly and emit `SchemaVersionWarning` ("written by a newer format 1.N; this reader is 1.M; content it understands was loaded; upgrade discopt to be sure") — [today: silent] |
| minor > reader's, unknown top-level key **or** unknown op/kind/tag | **refuse**, message names the key and says to upgrade — [today: unknown key ignored, unknown op refused] |
| non-numeric minor, trailing components | as today (`_loads_document`): non-numeric minor refused, `1.3.1` read as minor 3 |

Known top-level keys become an explicit `_TOP_LEVEL_KEYS: dict[str, int]` (key →
minor it appeared in), which is also what WP-1's matrix test and WP-8's docs are
generated from. **This changes one existing test**
(`test_a_future_minor_schema_is_readable_but_a_future_major_is_not`): a future
minor stays readable *when it carries only known keys*; a new assertion covers the
unknown-key refusal. The old behaviour ("ignore unknown keys so 1.0 reads 1.1") is
exactly the silent-drop §0.1 forbids once a key can carry model content; the
`provenance` precedent was metadata and is grandfathered by being in `_TOP_LEVEL_KEYS`.

**Nested records (#1780, closed after WP-2).** WP-2 covered the top level and
`state` only; a key inside a variable, node, row or builder block was still
ignored (probe: `"zzz"` added to `nodes[0]` or `variables[0]` of a 1.0 document
loaded cleanly). Now every nested record follows the same policy: node records and
their index objects against `_NODE_KEYS` / `_INDEX_KEYS`, everything else by
recording which keys the decoder reads (`serialize._Record`) and refusing the
rest; a key repeated within one JSON object is refused as well. See the v1.5
change log for the measurements.

`result_io.SCHEMA_VERSION` (currently 3, integer) stays its own counter: the
solution subtree is `deserialize_result`'s to version, and `_SCHEMA_MINOR_UNTAGGED_SOLUTION`
already ties the model minor to the one non-additive change there. WP-2 adds the
same unknown-key rule to `deserialize_result` (it currently passes unknown keys to
`SolveResult(**kwargs)`, which raises `TypeError` → wrapped as `SerializationError`;
make that an explicit, named refusal).

### 3.3 Golden fixture corpus (WP-1)

**Location**: `python/tests/data/serialize_fixtures/v1.<minor>/<case>.json` plus one
`python/tests/data/serialize_fixtures/MANIFEST.json`. Per version the manifest
records: generating commit SHA, `discopt.__version__`, `SCHEMA` string asserted at
generation (the §0.2.7 marker), date, and per case: the builder name, the
`to_nl()` text (or `null` with a reason when `to_nl` refuses, e.g. a model with an
indicator constraint), the `model_fingerprint` tuple canonical text where
`transformations.model_fingerprint` existed at that commit, and the
forward-compat cell (§3.3.3).

**Cases** (`python/tests/serialize_fixture_models.py`, a fixed registry of
zero-argument builders; **use only API that existed at 1.0** so the same code runs
at every historical commit; a builder that raises at an old commit is recorded as
`"unbuildable_at": "1.N: <error>"` in the manifest, never skipped silently):
`scalar_nlp` (continuous, exp/log/sqrt, `**`), `array_bounds_inf_nan` (shape (3,2),
±inf and sentinel bounds, integer var), `mip_indicator_sos_disjunction`
(`Model.if_then`, `sos1`, `either_or`), `shared_dag` (one subexpression reused
20×), `fixed_scope` (saved inside a `with x.fixed(1.0):` scope — `Variable.fix()` is not a context manager; use whichever scoped spelling exists at the generating commit and record `unbuildable_at` if neither does), `complementarity_lowered`
(`Model.complementarity` + one lowering mark), `with_result` (a 2-variable solve
embedded via `save(result=…)`), `from_nl_nvs01`, `from_nl_st_e13`,
`from_nl_mpcc` (`data/mpcc_complementarity.nl`), `fast_api_blocks`
(`add_linear_constraints` + `add_quadratic_objective`), `parameters_and_start`
(array/scalar parameters + `set_initial_point`), `decomposition_marks`
(`mark_coupling` by name + `set_stage`), `registered_atom` (`register_function`,
1.1+; the tag is written from 1.3). As built: see `python/tests/serialize_fixture_models.py`
(v1.3 change log for why `gams_initial_values` / `recentre_marker` were dropped).

**Generation procedure** (`scripts/serialize_fixture_gen.py --version 1.N --commit <sha> --out …`):
1. `git fetch --unshallow` (or run on the owner's full clone); list the bumps with
   `git log --format=%h -G'^SCHEMA = "discopt\.model/' -- python/discopt/serialize.py`
   (**`-G`, not `-S`**: every bump keeps the `SCHEMA = "discopt.model/` prefix, so
   `-S` sees only the commit that added the file — corrected in v1.3). The last tree
   that wrote 1.N is the parent of the commit that introduced 1.N+1.
2. `git worktree add /tmp/wt-1.N <sha>`; in it: `pip install -e . --no-deps`
   into a dedicated venv and `maturin develop --release` **from that worktree root**
   (CLAUDE.md install note; the Rust ABI must match that commit's Python side).
3. Run the generator **inside that venv**; it asserts
   `discopt.__file__.startswith("/tmp/wt-1.N")` and `discopt.serialize.SCHEMA == "discopt.model/1.N"`
   (**1.0 spells it `"discopt.model/1"`**, a bare major) and exits otherwise; it also
   drops from `sys.path` every entry providing another discopt checkout and, after
   the run, refuses if any loaded `discopt.*` module came from outside the tree
   (see the WP-1 finding in §8: a `--system-site-packages` venv otherwise imports
   modules missing from the old tree from the current one); writes the fixtures with `provenance=False` (byte-stable)
   plus one `*_with_provenance.json` per version so the provenance-skew path has a
   real historical block.
4. For the forward cell (§3.3.3) the generator also *loads the current HEAD's
   fixtures* with the old reader and records the outcome.
5. Prints `cases written: N, unbuildable: M, forward cells: K`; exit non-zero at
   `N == 0`.
6. The 1.4 fixtures are generated from **current HEAD before WP-3 bumps to 1.5**
   (hence WP-1 precedes WP-3 in §5). From WP-3 on, every bump PR generates its own
   version's fixtures from its own HEAD (trivial marker check) — §0.3.10.

**Fallback if a historical commit cannot be built** (Rust toolchain mismatch,
missing dependency): record the exact failure in the manifest and §8; the matrix
then has a documented hole, not a silent one. Do **not** hand-write a "1.1-style"
document from memory — `test_legacy_11_document_restores_a_tagged_float_in_the_trace`
already embeds one such hand-made 1.1 excerpt; keep it, but it does not substitute
for a writer-produced fixture.

#### 3.3.3 Compatibility matrix

Rows = writer version, columns = reader version. Backward cells (old doc → current
reader) are **live tests** (`test_every_golden_fixture_loads`, parametrized over
the manifest; executed-count assertion `>= 10 × number_of_versions_present`).
Forward cells (current doc → old reader) are **recorded at generation** in the
manifest with one of `loads_identical` (old reader's `to_nl` equals ours),
`refused: <message prefix>`, `loads_different: <diff>`; the test asserts the
column exists for every present version and contains **no** `loads_different`.
If generation finds one, that is a real defect in a shipped reader — record it in
§8 and in the user-facing compatibility statement (WP-8) as "documents written by
≥ 1.N must not be opened with ≤ 1.M".

### 3.4 Pickle format stamp (WP-2; owner decision **O1**)

`Model.__getstate__` currently returns `(attrs, had_builder, row_marks)` and the
module docstring says "pickle carries no format version". Add a module constant
`_MODEL_PICKLE_FORMAT = 1` (bumped on any change to what `__getstate__` writes or
`__setstate__` expects) and write `attrs["_pickle_format"] = (_MODEL_PICKLE_FORMAT, discopt.__version__, SCHEMA)`.
In `__setstate__`:

- missing stamp or `format != _MODEL_PICKLE_FORMAT` → **refuse** with
  `TypeError("this Model was pickled by a different discopt pickle layout (…); pickle is for same-version transport — use Model.save()/load() to keep a model, or re-pickle it with the writing version")`.
  Recommended: refuse. A layout mismatch means `__dict__.update` would install
  attributes the new code does not expect or miss ones it does — exactly the
  silent misbehaviour G3 names.
- same format, different `discopt.__version__` → `ProvenanceSkewWarning` (reuse
  the category; its docstring already describes this use) and proceed.
  *Alternative for the owner*: refuse here too (strictest; breaks `pickle` across a
  dev-version bump within one day). The plan recommends **warn**.

The same stamp is written by `Expression.__reduce_ex__`'s outer scope? **No** — a
lone expression pickle stays unstamped: it is an internal transport
(`test_pickled_expression_keeps_identity_with_its_model`), and stamping every node
would reintroduce per-node cost (#1764 measured a per-node weakref at ~+50%).
Documented as such.

### 3.5 Migration / upgrade path (WP-2)

`discopt upgrade-model <in> [<out>] [--check]` in `python/discopt/cli.py`
(`subparsers.add_parser("upgrade-model", …)`): `load` → `save` at the current
`SCHEMA`, preserving the embedded result and carrying the old provenance block as
`previous` into `provenance.capture(previous=…)` (the parameter exists). `--check`
loads, reports schema in → schema out and whether a rewrite would change bytes,
writes nothing. Exit codes: 0 ok, 1 refused (message to stderr), 2 usage. That is
the whole migration story for minors; a future major adds a `_migrate_1_to_2(doc)`
step in `_loads_document` *before* strict decoding, and the fixture corpus from
§3.3 is what proves it.

### 3.6 Decisions on the G1/G2 items

| Item | Decision | Why |
|---|---|---|
| `_LogicalConstraint` / `BooleanVar` | **Encode at 1.5** (WP-3) | Finite propositional tree over nine dataclasses (`LogicalAnd/Or/Not/Implies/Equivalent/AtLeast/AtMost/Exactly`, leaf `BooleanVar.variable` = a `Variable` or `IndexExpression` → already encodable in the node table). `_relax/gdp_reformulate.py::_logical_to_linear` lowers it deterministically, so a faithful encoding exists and the current refusal is only "this version has no section", not "nothing faithful to write". |
| `CustomCall` | **Final refusal** | Wraps an arbitrary Python callable (`CustomCall.fn`). Writing its name reloads a different model; writing its bytecode is code execution on `loads`, which `loads` on untrusted input must never be. Remedy in the message (already: `test_custom_call_refusal_names_a_working_spelling`): rewrite with `dm.udf`/`dm.*` primitives. Document as permanent in WP-8. |
| Reformulation outputs (`_bml_*`, `_ipx_*`, `_nsl_*`) | **Final refusal** | Hold the pre-pass source model; remedy is lossless (save source, re-apply via `transformations.apply_to`). WP-6 verifies the remedy round-trips for each pass. |
| `LoweringRecord.rows` | **Encode at 1.5** by row position (WP-3) | Rows are `Constraint` objects of this model; `_row_marks_by_position` already moves id-keyed marks by position for pickle. `rows=None` remains the encoding of "not tracked", distinct from `[]`. |
| Per-solve state (`_LAZY_NOT_CARRIED`) | Non-goal, documented | Describes a process, not a model. |
| `Variable._fix_scope_tokens` | Non-goal, documented | Live context-manager tokens; the *bound stack* they guard is carried. A reloaded model is outside every `with`. |
| Parsed `.nl` (`_nl_repr`) | Non-goal, documented | Rust `PyModelRepr`; marker + digest-checked path is the faithful equivalent. |
| User public attributes | Non-goal, documented | Not discopt state (`test_a_public_attribute_the_user_attached_is_not_discopt_state`). |

---

## 4. Work packages

Each is one Opus session / one PR. Fields: goal, files, entry experiment (§4 of
CLAUDE.md) with kill criterion, steps, tests (fail-before/pass-after), acceptance
commands, out of scope, dependencies, schema version.

### WP-1 — Golden fixture corpus + compatibility matrix

- **Goal**: D1, D2. Writer-produced documents for every shipped minor 1.0–1.4,
  the manifest, and the matrix test.
- **Files**: new `python/tests/serialize_fixture_models.py`,
  `scripts/serialize_fixture_gen.py`, `python/tests/data/serialize_fixtures/**`,
  `python/tests/test_serialize_compat_matrix.py`; this plan §8.
- **Entry experiment**: `git fetch --unshallow && git log --format='%h %ad %s' --date=short -G'^SCHEMA = "discopt\.model/' -- python/discopt/serialize.py`
  must list ≥ 5 commits (the file's introduction at 1.0 plus one per bump to 1.4). Then build the
  *oldest* (1.0) worktree per §3.3 step 2 and import it with the marker check.
  **Kill criterion**: the 1.0 tree cannot be built in ≤ 2 h of effort. If it
  fires: record the exact failure in §8, generate from the oldest buildable minor,
  and open the hole explicitly in the manifest + WP-8's compatibility statement
  ("documents older than 1.N are untested"). Do not fabricate fixtures.
- **Steps**: (1) write the builder registry (API ≤ 1.0 only); (2) generator with
  marker check, executed count, forward cells; (3) run for 1.0–1.4 (1.4 from
  current HEAD); (4) matrix test; (5) `MANIFEST.json` committed; (6) record the
  per-version SHAs in §8.
- **Tests**: `test_every_golden_fixture_loads` (parametrized; fails before because
  the fixtures do not exist — assert the directory is non-empty as the probe),
  `test_golden_fixture_reload_is_stable` (load → dumps → loads → dumps equal),
  `test_forward_matrix_has_no_silent_misread`, `test_fixture_probe_fired`
  (`>= 10 × versions`), `test_manifest_markers_match_versions` (each version's
  recorded `SCHEMA` string equals its directory name).
- **Acceptance**: `pytest python/tests/test_serialize_compat_matrix.py -v` (< 10 s);
  §0.4.12 commands.
- **Out of scope**: changing reader behaviour (WP-2); any 1.5 content.
- **Depends on**: nothing. **Schema**: no bump.

### WP-2 — Versioning policy: reader behaviour, pickle stamp, `upgrade-model`

- **Goal**: D3, D4, D5; §3.2, §3.4, §3.5 implemented.
- **Files**: `python/discopt/serialize.py` (`_TOP_LEVEL_KEYS`, `SchemaVersionWarning`,
  `_loads_document`), `python/discopt/result_io.py` (`deserialize_result` unknown
  keys), `python/discopt/modeling/core.py` (`_MODEL_PICKLE_FORMAT`,
  `__getstate__`/`__setstate__`), `python/discopt/cli.py`, `python/discopt/serialize.py`
  module docstring, `python/tests/test_serialize_versioning.py` (new),
  `test_model_pickle.py`, `test_cli_upgrade_model.py` (new),
  `test_serialize_roundtrip.py` (revise the future-minor test per §3.2).
- **Entry experiment**: over the WP-1 fixtures and the 66-file corpus, count
  documents carrying a top-level key not in the proposed `_TOP_LEVEL_KEYS`. Expected
  0. **Kill criterion**: any legitimately written document has such a key →
  the key list is incomplete; fix the list, not the rule. Second probe: unpickle a
  model pickled at the parent of #1764 with HEAD (`git worktree`); record what
  happens today (expected: loads, possibly wrong `_pickle_expressions` handling) —
  this is the fail-before for the stamp.
- **Steps**: (1) `_TOP_LEVEL_KEYS` + refusal paths + warning; (2) `result_io`
  unknown-key refusal; (3) pickle stamp per §3.4 with O1's choice (default: refuse
  on format mismatch, warn on version); (4) `upgrade-model`; (5) docstrings.
- **Tests**: `test_unknown_key_at_known_minor_is_refused`,
  `test_unknown_key_at_future_minor_is_refused_with_upgrade_hint`,
  `test_future_minor_with_known_keys_warns_and_loads`,
  `test_result_unknown_key_is_a_named_refusal`,
  `test_pickle_carries_a_format_stamp`, `test_mismatched_pickle_format_is_refused`,
  `test_pickle_version_skew_warns`, `test_upgrade_model_rewrites_1_0_fixture_to_current`,
  `test_upgrade_model_check_writes_nothing`, `test_upgrade_model_carries_previous_provenance`.
- **Acceptance**: §0.4.12; `discopt upgrade-model python/tests/data/serialize_fixtures/v1.0/scalar_nlp.json /tmp/x.json && python -c "import discopt.modeling as dm; dm.load('/tmp/x.json')"`.
- **Out of scope**: new sections. **Depends on**: WP-1 (fixtures are the test
  bed). **Schema**: no bump (reader-side only; the stamp is pickle-side).

### WP-3 — Schema 1.5: logical constraints + `LoweringRecord.rows`; final refusals

- **Goal**: D6, D7; §3.6 decisions landed.
- **Files**: `serialize.py` (`SCHEMA = "discopt.model/1.5"`, `_enc_rows`/`_dec_rows`
  gain a `"kind": "logical"` row with a `logic` sub-table; `_REFUSED_RELATIONS` loses
  `_LogicalConstraint`; `_enc_complementarities`/`_dec_complementarities` gain
  `"rows": [positions] | null`; `_TOP_LEVEL_KEYS` unchanged — the change is inside
  `rows`/`complementarities`, but it is non-additive for a 1.4 reader
  (`_dec_rows` would hit an unknown row kind → clean refusal; verify and record in
  the matrix), hence the minor bump); `modeling/core.py` `_expression_roots` must
  include logical leaves (`BooleanVar.variable`) so pickle's children-first list
  and `_expressions_children_first` cover them; `python/tests/data/serialize_fixtures/v1.5/**`
  from this PR's HEAD; `test_serialize_roundtrip.py`, `test_serialize_adversarial.py`
  (`MALFORMED` gains: logical op unknown, `k` out of range for AtLeast, leaf not a
  variable/index node, `rows` position out of range / duplicated / pointing at a
  non-`Constraint`), notebook paragraph "Propositional logic constraints are refused
  for a milder reason" (update in WP-8, note here).
- **Entry experiment**: build 6 logical models (each connective once, nested, over
  `BooleanVarArray` indices), solve each and the hand-lowered equivalent via
  `transformations.check`-style diff; confirm `_logical_to_linear` is deterministic
  (same CNF twice). **Kill criterion**: lowering is nondeterministic or depends on
  state outside the model → encoding the *tree* would not be bound-neutral; stop
  and record. (Expected: deterministic.)
- **Steps**: (1) logic encoding: node kinds `and|or|not|implies|iff|atleast|atmost|exactly|var`,
  `var` → node-table id of the leaf expression (must resolve to a `Variable` or an
  `IndexExpression` of a binary `Variable`; refuse otherwise), `k` an int in
  `[0, len(operands)]`; (2) `rows` positions; (3) bump + fixtures + matrix row;
  (4) make the `CustomCall` and reformulation refusals' docstrings say "permanent"
  and why (§3.6); (5) `_REFUSED_RELATIONS` may become empty — keep the mechanism.
- **Tests** (fail-before: today's refusal): `test_logical_constraints_round_trip`
  (each connective), `test_logical_constraint_over_boolean_array_index`,
  `test_logical_round_trip_is_bound_neutral` (solve before/after, exact),
  `test_lowering_record_rows_round_trip`, `test_lowering_rows_none_stays_none`,
  `test_malformed_logic_is_refused` (via `MALFORMED`), `test_1_4_reader_refuses_a_1_5_logical_row`
  (replay the 1.4 reader from the WP-1 worktree — recorded in the manifest's
  forward column, asserted by the matrix test), `test_schema_minor_is_1_5`.
- **Acceptance**: §0.4.12; `pytest python/tests/test_gdp*.py python/tests/test_logical*.py -q`
  (whatever exists; grep first) to show the lowering path is untouched.
- **Out of scope**: `CustomCall` encoding (never); builder sweep (WP-6).
- **Depends on**: WP-1 (1.4 fixtures must be frozen first), WP-2 (bump protocol).
  **Schema**: **1.5**.

### WP-4 — Property-based round trip + fuzzing of `loads` (hypothesis)

- **Goal**: D8, D9.
- **Files**: `pyproject.toml` (`dev` extra gains `"hypothesis>=6.100"`; CI jobs that
  `pip install pytest pytest-timeout pytest-xdist` by hand — `ci.yml:299,399,502,603,655,731`
  — add `hypothesis`), `python/tests/test_serialize_property.py`,
  `python/tests/test_serialize_fuzz.py`, `python/tests/strategies_models.py`
  (hypothesis strategies, importable by WP-6 too), `conftest.py` hypothesis
  profiles (`ci`: `max_examples=50, deadline=None`; `slow`: 2000).
- **Entry experiment**: 200 generated models through `loads(dumps(m))`; count
  distinct `_KNOWN_OPS`/funcs/relation kinds exercised (print a coverage table).
  **Kill criterion**: the strategy exercises < 80 % of `_KNOWN_OPS ∪ _known_funcs()`
  after one session of tuning → the generator is not a property test of the
  format; fix the strategy before asserting anything.
- **Steps**: (1) strategy: variables (every `VarType`, shapes `()`, `(n,)`, `(n,m)`,
  bounds incl. ±inf/sentinel), parameters, expression trees drawing every
  `BinaryOp`/`UnaryOp`/`FunctionCall` name in `_KNOWN_OPS`/`_EXTRA_FUNCS`/
  `_known_funcs()`, `IndexExpression` (int/slice/tuple/bool kinds per #1290),
  `SumExpression`/`SumOverExpression` with list axes, `MatMulExpression`, shared
  subtrees, relations (indicator/SOS/either_or/complementarity/logical after WP-3),
  optional `fix()` scope, optional fast-API block; (2) properties: dumps stable,
  `model_fingerprint(loads(dumps(m))) == model_fingerprint(m)`, `pickle` round trip
  gives the same fingerprint, `to_nl` equal when `to_nl` accepts the model
  (otherwise assert it *refuses* both before and after — same exception type);
  (3) fuzz: structure-aware mutators over a valid document (delete key, retype
  value, ±1/negative/`true` on every integer ref, swap `op`, truncate `nodes`,
  duplicate names, NaN tokens) plus `binary()` raw input; property: outcome ∈
  {`Model`, `SerializationError`}; `deadline` per example 2 s via hypothesis
  settings (a hang is a failure); if a mutation touched `nodes`/`rows`/`variables`
  and a `Model` came back, `dumps(model, provenance=False) != original` (no silent
  acceptance); (4) every example that found a bug becomes a named regression case
  in `MALFORMED`.
- **Tests**: `test_generated_model_round_trips`, `test_generated_model_pickle_matches_dumps`,
  `test_strategy_covers_the_op_table` (the executed-coverage assertion),
  `test_mutated_document_never_escapes_serialization_error`,
  `test_mutation_of_content_is_never_silently_accepted`, `test_random_bytes_are_refused`.
- **Acceptance**: `pytest python/tests/test_serialize_property.py python/tests/test_serialize_fuzz.py -q` ≤ 20 s in the `ci` profile; `-m slow` profile run once, output pasted in the PR.
- **Out of scope**: fuzzing `result_io` beyond the embedded `solution` subtree;
  fuzzing `.nl` parsing. **Depends on**: WP-3 (so logical rows are generated;
  can start on 1.4 and extend). **Schema**: none.

### WP-5 — Corpus-wide round-trip harness (in-repo + optional full MINLPLib/QPLIB)

- **Goal**: D10, D11.
- **Files**: `scripts/serialize_corpus_roundtrip.py`, `python/tests/test_serialize_corpus.py`
  (replaces the two corpus tests in `test_serialize_roundtrip.py`, which move here
  unchanged), `docs/dev/data/serialize-corpus-<date>.jsonl` (one row per
  instance: name, n_vars, n_rows, n_nodes, dumps_s, loads_s, bytes, gz_bytes,
  pickle_s, status).
- **Entry experiment**: run the harness on the in-repo 66 + 9 QPLIB + `binary_nl`
  + `mpcc_complementarity.nl`; expect 0 failures (the 66 already pass).
  **Kill criterion**: none — any failure is a defect to fix in this PR (or, if it
  is a missing encoding, a §0.4.16 escalation naming the minor it needs).
- **Steps**: (1) harness: `-u`, per-file progress line, never `/dev/null`,
  `faulthandler` *not* armed by default (CLAUDE.md §10 #1758), exit non-zero at
  0 executed; corpora resolved from `DISCOPT_MINLPLIB_NL_DIR`
  (default `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl`) and
  `DISCOPT_QPLIB_DIR` (as `test_qplib_corpus.py`), QPLIB filtered on
  `qplib_manifest.csv` `usable_oracle` (never by name); absent → print
  `"corpus X not present at <path>; skipping"` and continue with the in-repo
  floor; (2) checks per instance: `dumps` stable, `to_nl` byte-identical for
  `.nl`-sourced models, `from_qplib(...)` → `model_fingerprint` equal after
  `loads(dumps())` and after `pickle`, `_source_nl_sha256` preserved; (3) pytest
  wrapper (slow) over the in-repo set; a 12-instance solve-bound-neutral panel
  chosen by size from `python/tests/data/known_optima.toml` entries that solve
  < 20 s (list them in the test, with the selection rule in a comment);
  (4) run the full corpora on the owner's machine; record the jsonl and the summary
  in §8.
- **Tests**: `test_in_repo_corpus_round_trips` (parametrized), `test_qplib_in_repo_round_trips`,
  `test_corpus_probe_fired` (`>= 75`), `test_round_trip_panel_is_bound_neutral`
  (12 instances, exact `node_count`/`objective`/`status`).
- **Acceptance**: `python -u scripts/serialize_corpus_roundtrip.py --in-repo-only`
  prints `executed: N ≥ 75, failed: 0`; `pytest -m slow python/tests/test_serialize_corpus.py`.
- **Out of scope**: performance budgets (WP-7 consumes the jsonl). **Depends on**:
  WP-3 preferred (so `.nl` complementarities with lowering rows round-trip fully);
  can run on 1.4. **Schema**: none.

### WP-6 — Builder coverage sweep

- **Goal**: D12. Every model-producing builder round-trips or is a final, tested
  refusal. Enumerated from the tree:

  | Area | Entry point (grep-verified) | Expected |
  |---|---|---|
  | DAE | `dae.collocation.DAEBuilder` (`add_state/add_algebraic/add_control/set_ode/discretize`), `dae.finite_difference.FDBuilder`, `dae.mol.MOLBuilder`, `dae.fit.fit_trajectories` | round-trip (algebraic rows, shared subtrees) |
  | ML frozen | `ml.predictor.add_predictor(model, inputs, predictor, method=…)` for `full_space`, `reduced_space`, `relu_bigm`, tree ensemble (`formulations/tree_ensemble.py`), `linear_tree`; readers sklearn/torch/onnx only if the optional dep is installed (skip with reason) | round-trip; `relu_bigm`/trees add binaries + big-M rows |
  | ML trainable | `ml.trainable.TrainableNetwork` inside a model; `TrainableNetwork.freeze()` | round-trip (weights are `Variable`s) |
  | GDP factory | `Model.disjunction`, `make_disjunct`, `either_or`, `add_disjunction(gdp_method=…)`, `Model.boolean` + `Model.logical` | round-trip (logical after WP-3); an unattached block must still be refused (`_reject_unattached_gdp_blocks`) |
  | Stochastic | `stochastic.extensive_form.build_extensive_form`, `CVaR`/`MeanCVaR` (sets `_free_by_construction`), `chance_constraint`, `solve_lshaped` *inputs* | round-trip |
  | RO | `ro.counterpart.RobustCounterpart(...).formulate()` for Box/Ellipsoidal/Polyhedral/budget sets, `ro.affine_policy.AffineDecisionRule` | round-trip or refusal — the entry experiment decides |
  | Bilevel | `bilevel.problem.BilevelProblem.formulate(method="kkt"|"strong_duality", mpec_method=…)` | round-trip (complementarities + lowering marks) or refusal |
  | GP | `gp.as_geometric_program` (if it builds a `Model`) | round-trip |
  | Transformations | every name in `transformations` registry via `create_using` | `gdp.*`/`mpec.*` round-trip; `binary.multilinear`/`integer.*`/`nonsmooth.epigraph` refused **and** the remedy (save source, `apply_to` after load) verified fingerprint-equal |

- **Entry experiment** (`scripts/serialize_builder_sweep.py`): build one small
  instance per row, run `loads(dumps(m))` and `pickle`, classify
  {round-trips, refused: <attr/relation>, crashed: <exc>}, print the table and the
  executed count. **Kill criterion**: a *crash* (non-`SerializationError`) or a
  refusal for an attribute not in §3.6's final list means a missing classification
  or encoding; if it needs a new section, stop and add a WP-3b row to §4 with its
  minor (1.6) before coding it — do not fold it into the sweep PR.
- **Steps**: (1) sweep; (2) tests per row, each asserting fingerprint equality
  and, where the solve is < 5 s, exact bound neutrality; (3) refusal rows assert
  the message *and* the remedy path; (4) table in §8.
- **Tests**: `test_serialize_builders.py::test_<row>_round_trips` /
  `::test_<row>_is_refused_with_remedy`, `test_builder_sweep_probe_fired`.
- **Acceptance**: `pytest python/tests/test_serialize_builders.py -m "not slow" -q` ≤ 15 s; slow part once, pasted.
- **Out of scope**: fixing builder bugs unrelated to serialization (file them).
  **Depends on**: WP-3. **Schema**: none unless the kill criterion opens WP-3b.

### WP-3b — Schema 1.6 / 1.7: the structures the WP-6 sweep found refused

**Split at entry (2026-10-19, v1.10).** Items 1 and 2 are self-contained
encodings; item 3 grew while WP-6 was in flight (#1790 `Disjunct.variable`,
#1795 G1.4 OR blocks) and a faithful reload needs the `Disjunct` objects
themselves (their row identities feed `_reject_escaped_local_vars`), a different
design. So: **WP-3b-1 = schema 1.6** (items 1, 2), **WP-3b-2 = schema 1.7**
(item 3: G1.2 parent links, `_gdp_block_members`, G1.4 block semantics, local
variables).

Opened by WP-6's kill criterion (2026-10-19): builders produce three structures
the 1.5 format has no encoding for. Each is refused by name today with a
same-version remedy (pickle / `Model.clone()`), pinned by
`test_serialize_builders.py`.

- **Goal**: encode, at one minor gate (1.6):
  1. **`None` and `Ellipsis` index kinds.** `DAEBuilder.add_control`
     (`dae/collocation.py`, `var[:, None]` / `var[:, c, None]`) and every
     `TrainableNetwork` forward pass (`ml/trainable.py`, `expr[..., None]`)
     index with them; `_enc_index` knows int / slice / tuple / bool only. A new
     `k` value each; a 1.5 reader refuses the unknown kind, which is the
     intended forward behaviour.
  2. **A disjunction as a disjunct member** (`Model.disjunction()` passed to
     `either_or`). The disjunction row encodes members as constraints only;
     writing just the nested rows would turn an OR into an AND.
  3. **Nested Disjunct blocks** (#1775 G1.2: `Disjunct.add_disjunction` /
     `Disjunct.subject_to`): the rows' `parent` / `parent_indicator` links and
     `_gdp_block_members`.
- **Entry experiment**: encode each, reload the WP-6 cases, and require equal
  fingerprints and an identical `(status, node_count, objective)` for the ones
  that solve terminally. **Kill criterion**: a nested structure that does not
  reload to the same feasible set (compare against the pickle copy's solve).
- **Tests**: the four WP-6 `REFUSED` cases (`dae:collocation with a control`,
  `ml:trainable network`, `gdp:nested disjunction()`, `gdp:nested Disjunct
  blocks`) move to `ROUND_TRIPS`; v1.6 fixtures for each via
  `scripts/serialize_fixture_gen.py --cases`; the forward matrix shows each 1.5
  reader refusing exactly these documents.
- **Out of scope**: OR / EXACTLY_ONE_TRUE disjunction *solving* (#1124; they
  already round-trip). **Depends on**: WP-6. **Schema**: 1.6.

### WP-7 — Performance ladder, budgets, and the pickle overhead

- **Goal**: D13; owner decision **O2** gets its measurement.
- **Files**: `scripts/serialize_perf_ladder.py`, `python/tests/test_serialize_perf_budget.py`
  (slow), `docs/dev/data/serialize-perf-<date>.jsonl`, possibly
  `modeling/core.py` (`_expressions_children_first`, `__getstate__`) and
  `serialize.py` (`_NodeTable.add`), this file §4/WP-7 for the record.
- **Entry experiment** (the measurement that G5 lacks): with `uptime` load < 1.0
  and no stray `pytest`/probe processes (`pgrep -af python`), two worktrees — HEAD
  and the **parent of the #1764 merge** (`git log --format=%P -1 20d663c6`, first
  parent) — each built with `maturin develop --release` and marker-checked
  (`_PICKLE_SCOPES` present / absent in `core.py`); interleaved A/B, 7 repeats, on
  (a) the 66-file corpus pickled in one loop, (b) three synthetic deep models
  (5k/50k/200k nodes), (c) `copy.deepcopy`; report mean ± sd per arm. **Kill
  criterion for the optimisation half**: if HEAD's overhead vs baseline is < 5 %
  (i.e. the +19.6% does not reproduce), record that and do not optimise; if ≥ 5 %,
  attempt at most two variants (candidates: skip `_descendants_children_first`
  when the deepest root is < 500 nodes and fall back to plain reduction;
  build the children-first list once per `__getstate__` and reuse it for
  `_row_marks_by_position`), each must keep `test_deep_expression_walks.py` and
  `test_model_pickle.py` green and be bound-neutral; stop after two.
- **Steps**: (1) ladder script over the WP-5 jsonl + synthetic sizes
  (10³–10⁶ nodes) for `dumps`, `loads`, gz size, `pickle.dumps/loads`,
  `deepcopy`; (2) budgets as *ratios* so they are machine-independent:
  `loads(dumps(m))` wall ≤ 5 × `from_nl` wall of the same file; gz document ≤ 3 ×
  `.nl` bytes; `pickle` ≤ 1.25 × `dumps` wall (owner may tighten after O2);
  (3) the A/B; (4) optional optimisation under the kill criterion; (5) record
  the numbers here and in the jsonl; a budget test (slow) that recomputes the
  ratios on 5 fixed in-repo instances.
- **Tests**: `test_serialize_perf_budget.py::test_round_trip_within_ratio_budget`,
  `::test_document_size_within_ratio_budget`, `::test_budget_probe_fired`.
- **Acceptance**: PR quotes the table (mean ± sd, both arms, `uptime`).
- **Out of scope**: changing the document format for size (would be a major).
  **Depends on**: WP-5 (jsonl). **Schema**: none.

### WP-8 — Documentation, compatibility statement, CI wiring

- **Goal**: D14, D15.
- **Files**: `docs/notebooks/model_serialization.ipynb` (new sections:
  "Versions and compatibility" — the §3.2 table in user words, the matrix from
  the manifest, `upgrade-model`; update "What is refused, and why" for logical
  constraints now carried and `CustomCall` permanent; update "`pickle` is for
  moving a model" with the stamp behaviour), `docs/_toc.yml` unchanged,
  `serialize.py` and `core.py` pickling-section docstrings, `docs/dev/serialization-format.md`
  (**new, short**: top-level keys with the minor each appeared in, generated
  from `_TOP_LEVEL_KEYS` by a test so it cannot drift), `python/tests/test_serialize_docs_match.py`,
  `.github/workflows/ci.yml`.
- **Entry experiment**: `jupyter-book build docs/` warning count before (must be
  0 after); `pytest python/tests/ -m "not slow …" --durations=20 -n 2` wall before
  vs after the new fast tests. **Kill criterion**: fast lane grows > 25 s → move
  the heaviest new file to slow and record it.
- **Steps**: (1) docs; (2) `test_serialize_docs_match.py` asserts the refused list
  and key table in the docs equal the code's; (3) CI: fast lane picks up the new
  test files automatically (unmarked); `test_serialize_corpus.py` and the perf
  budget are `slow` (manual `python-correctness-slow`) and additionally in a new
  weekly `serialize-corpus` job (cron, in-repo corpus only; the full corpora
  cannot run in CI); the hypothesis `ci` profile is selected by env in `conftest.py`;
  (4) the user-facing compatibility statement (one paragraph, in the notebook and
  `docs/dev/serialization-format.md`): "A document written by discopt ≥ 0.9 at
  schema 1.N loads in any later discopt with major 1. A newer document loads in an
  older discopt only when it uses no newer section; otherwise it is refused and
  tells you to upgrade. Pickle is same-version transport only."
- **Tests**: `test_serialize_docs_match.py::test_docs_refused_list_matches_code`,
  `::test_format_doc_key_table_matches_code`; `jupyter-book build docs/` 0 warnings.
- **Acceptance**: build log + `--durations` quoted.
- **Out of scope**: new encodings. **Depends on**: all of WP-1…WP-7 (last).
  **Schema**: none.

---

## 5. Execution order and dependency graph

```
WP-1 (fixtures 1.0–1.4) ──► WP-2 (policy, stamp, upgrade-model) ──► WP-3 (1.5: logical, rows)
                                                                        │
                        ┌───────────────────────────┬───────────────────┼───────────────┐
                        ▼                           ▼                   ▼               │
                 WP-4 (property+fuzz)        WP-5 (corpus harness)   WP-6 (builders)    │
                                                    │                                   │
                                                    ▼                                   │
                                             WP-7 (perf, O2)                            │
                                                    │                                   │
                                                    └──────────► WP-8 (docs, CI) ◄──────┘
```

Strict order for the first three (WP-1 must freeze 1.4 fixtures before WP-3 bumps;
WP-2 defines the bump protocol WP-3 follows). WP-4/5/6 may run in parallel sessions
after WP-3 merges. WP-7 needs WP-5's jsonl. WP-8 is last and is the only package
allowed to touch `ci.yml` beyond adding `hypothesis` to the install lines (WP-4).
Rough sizing: WP-1 is the largest (historical builds); WP-2, WP-3, WP-4 one session
each; WP-5 one session + an owner-machine run; WP-6 one session (the sweep output
sets the test count); WP-7 one session; WP-8 one session.

---

## 6. CI wiring summary (implemented by WP-4 install lines and WP-8)

| Lane | Files | Marker | Budget |
|---|---|---|---|
| `python-fast` (every PR) | `test_serialize_compat_matrix.py`, `test_serialize_versioning.py`, `test_cli_upgrade_model.py`, `test_serialize_property.py` (ci profile), `test_serialize_fuzz.py` (ci profile), `test_serialize_builders.py` (`not slow`), `test_serialize_docs_match.py`, existing serialize/pickle files | unmarked / smoke | ≤ +25 s total |
| `python-correctness-slow` (manual) + new weekly `serialize-corpus` cron | `test_serialize_corpus.py`, `test_serialize_perf_budget.py`, property/fuzz `slow` profile, builders `slow` | `slow` | ≤ 15 min |
| owner machine (recorded in §8) | `scripts/serialize_corpus_roundtrip.py` on full MINLPLib/QPLIB; `scripts/serialize_fixture_gen.py` for historical minors | — | — |

---

## 7. Risks and owner decisions

**Decisions the owner must take (flag in the PR that needs them):**

- **O1 — pickle across versions** (§3.4; WP-2): recommended *refuse* on
  `_MODEL_PICKLE_FORMAT` mismatch or missing stamp, *warn* on `discopt.__version__`
  skew. Alternative: refuse on any skew. Trade-off: strictness vs. dev-loop
  friction; `solve_batch` is unaffected either way (it ships `dumps` text).
- **O2 — is the pickle overhead acceptable?** (WP-7): decide after the interleaved
  A/B is in §8. The plan's default: accept up to +20 % if the two optimisation
  variants fail, because the alternative (per-node bookkeeping) measured ≈ +50 %
  and the recursion fix is a correctness fix (CLAUDE.md §1).
- **O3 — unknown-key refusal at known minors** (§3.2; WP-2): the plan makes an
  unknown top-level key a refusal. This breaks any external tool that annotates
  documents with its own keys. Alternative: reserve a namespaced `"x-"` prefix
  that is ignored (and never read by discopt). Recommended: adopt the `x-` escape
  hatch *and* the refusal for everything else.
- **O4 — the 1.0–1.2 fixtures** (WP-1): if the historical trees cannot be built
  (kill criterion), accept a documented hole ("untested before 1.3") or fund the
  effort to make them build (pinning an old Rust toolchain). Documents from those
  minors exist only if someone saved with a pre-`v0.9.0` dev install.
- **O5 — logical constraints at 1.5** (§3.6; WP-3): confirm the encoding (vs. keeping
  the refusal and asking users to lower first). The plan recommends encoding: the
  refusal is a missing section, not an impossibility, and `solve_batch(workers>1)`
  cannot ship a model with `Model.logical` today.

**Risks:**

- Historical worktrees need the matching Rust ABI; a `_rust.abi3.so` from the
  wrong tree gives `AttributeError` on `PyTreeManager` (CLAUDE.md install note) —
  the marker check (§0.2.7) catches a wrong import, not a wrong `.so`; also assert
  `discopt._rust.__file__` is under the worktree.
- A forward-matrix cell of `loads_different` in a shipped reader is a real,
  unfixable-in-place defect; it becomes a documented warning, not a code change.
- Hypothesis shrinking can produce models that are legal for the modeling API but
  that `to_nl` or `solve` refuse; properties are stated in terms of the format
  (`dumps`/`fingerprint`), with `to_nl`/`solve` guarded by "refuses identically".
- Adding `hypothesis` to six hand-written CI install lines: miss one and that job
  errors at collection (the PyYAML precedent in `pyproject.toml`'s `dev` comment).
  `test_dev_extra_declares_test_deps.py` exists — extend it to `hypothesis`.
- Fast-lane wall: property/fuzz tests must use the `ci` profile and `deadline=None`
  (xdist + JAX warm-up make per-example deadlines flaky); the slow profile is where
  the deadline is a hang detector.

---

## 8. Progress (living table — update in the same PR as the work)

| WP | Status | PR | Date | Notes / measurements |
|---|---|---|---|---|
| WP-1 fixtures + matrix | **done** | #1769 | 2026-10-18 | writers: 1.0 `0a655f244`, 1.1 `f31547172`, 1.2 `a82f89b0f`, 1.3 `8a4880f7c`, 1.4 HEAD at generation (serializer = `ca63cb2e7`'s); 69 fixtures (14 cases × 5 versions − `registered_atom`@1.0, unbuildable: no `discopt.operators` before 1.1); backward: 69/69 load and equal today's build minus content not yet written (`_WRITTEN_FROM`: atom tags < 1.3, `state.nl_source` < 1.4); forward: 136 `loads_identical`, 4 `loaded` (1.0 reading `registered_atom`, nothing to compare), **0 `loads_different`, 0 `refused`**; test file 216 tests in 1.5 s (2.6 s under `-n 4`); two reader mutations caught (15 and 10 failures); the live test builds each case before loading its fixture, for the same registry reason as (d) below — found when `-n 4` ordering failed `v1.3-registered_atom` once. No hole: all five writers built (O4 not needed). |
| WP-2 policy, stamp, upgrade-model | **done** | #1776 | 2026-10-18 | entry probe 1: 135 documents (69 fixtures + 66 corpus round-trips) carry exactly the 15 listed top-level keys, 0 unknown; a scan of every revision of `serialize.py` finds no top-level or `state` key ever written that the tables lack. Entry probe 2 (fail-before for the stamp): a model pickled by the `8a4880f7c` tree (layout `(attrs, had_builder)`) raised a bare `ValueError` (unpack 3, got 2) at HEAD, **with the same `__version__` `0.9.1.dev0`** — only a layout stamp tells them apart; the pickle is committed as a fixture. Pre-WP-2 reader loaded all five probe documents (unknown top key at known/future minor, unknown `state` key, future minor, unknown result key) silently with 0 warnings. `upgrade-model`: 67/69 fixtures upgrade; the 2 `registered_atom` documents are refused without `--import` — measured, a bare process strips their atom tags (see v1.4). |
| WP-3 schema 1.5 | **done** | #1785 (+ #1782) | 2026-10-18 | entry experiment: 6 logical models (+1 xor/exactly), lowered twice and from a rebuild → identical rows; model vs rebuild solve: same status/objective/node count (24 checks) — kill criterion not fired. Logic written as a flat per-row table (`logic`), leaves = node ids of the binary variable behind each `BooleanVar`; lowering rows as positions in a third `lowered` element (absent when untracked, so 1.4 still reads those). Combined with #1782 (G1.1 `indicators` on disjunction rows) into one 1.5. Fixtures: v1.5 18 cases (written by `6420787e`); `logical_constraints`, `complementarity_vector_lowered`, `gdp_indicators`, `gdp_indicators_logic` recorded for 1.0–1.4 via `--cases` (logic and indicator cases unbuildable, each version's own error). Forward matrix: 240 cells — 205 `loads_identical`, 10 `loaded`, 25 `refused` (each older reader refuses exactly the 5 documents with a logical row or tracked lowering rows), **0 `loads_different`**; the 150 cells not involving 1.5 are unchanged. Logic over `either_or` indicators round-trips (same status/objective/node count; optima 10/45/50 hand-computed). |
| WP-4 property + fuzz | **done** | #1788 | 2026-10-18 | op-coverage: 33/41 (80 %, at the kill line) before tuning → 39/41 (95 %) after weighting toward function calls; every node op, index kind and row kind reached (200 derandomized examples). `slow` profile (2000 examples per test): 13 passed in 110 s, load 0.2. Fuzzing found five reader bug classes, all fixed with named `FUZZ_FOUND` cases (20; 16 fail before their fix, 4 already refused): CSR structure (out-of-range / negative column loaded — a negative one onto another column; broken `indptr` → Rust **panic**; int32 overflow escaped as `OverflowError`), array nesting (`[0.0]` for a scalar), malformed atom tags (skipped like an unregistered one), `{}`/`""` where a list belongs (~25 sites read it as empty), and a non-`0.0` placeholder objective beside a builder objective (loaded unread). Review of #1788 (B1): the first atom-tag check (`arg < i`) refused tags the writer itself emits — it appends a node before `add`ing the argument, so an identity lowering records `arg == i` and one that ignores its argument `arg > i`; the bound is the table length, and the strategy now generates atoms of all three tag shapes (the property test fails on the old bound). The deep run then found nested atoms (`sq(sq(x))`) losing the outer tag on reload: re-derivation compared node tables byte-for-byte, so the reload's sharing of an interned `1.0` (one `Constant` per node) against re-lowering's (`_wrap` interning) dropped a valid tag; it now compares hash-consed trees. Known, deliberate: the strict nesting refuses `lb: [0.5]` with shape `[]` for a scalar variable, which a writer from before #1332 (2026-09-15 → 09-18) could emit; no fixture has it. Known, sound: an atom whose lowering closes over the *writing* model's variables cannot be re-derived in the reloaded model, so its tag is dropped (DEBUG log) and `dumps(loads(s)) != s` for it. |
| WP-5 corpus harness | **done** (D11 dropped by the owner) | #1792 | 2026-10-18 | in-repo: **134/134** (floor raised from 75: `data/minlplib/` adds 56 instances not in `minlplib_nl/`; + 9 QPLIB, 2 `binary_nl`, the complementarity `.nl`), 0 failed — `dumps` stable, `.nl` byte-identical, fingerprint equal after reload and after pickle, source digest preserved on all 134 (present on 124; the complementarity `.nl` and the QPLIB models carry none). Entry run found one harness defect, not a reader one: it demanded a digest of every `.nl` model, but `from_nl` deliberately records none once it lowered complementarities; the check is now "preserved, absence included". Totals (`docs/dev/data/serialize-corpus-2026-10-18.jsonl`): `dumps` 0.36 s, `loads` 0.63 s, 4.48 MB (0.47 MB gzipped); largest QPLIB_3871 (1025 vars, 1040 rows, 9154 nodes, 575 kB / 52 kB gz). Panel (12 `known_optima` entries, < 20 s, largest by vars+rows): identical `(status, node_count, objective)` before/after and on a fresh re-parse; nvs17 excluded — its node count drifts by 2 across fresh parses at 50 s. Step (4)/D11, the full corpora: **dropped by the owner** — not needed. Before that call, a started run (corpora fetched once network access was granted: MINLPLib 1,609 `.nl`; QPLIB 453, of which 440 `usable_oracle` in the 2026-10-12 release, not the 421 recorded earlier) had executed 136 instances (the in-repo 134 + 2 MINLPLib) with 0 failures when it was stopped; no full-corpus result is claimed. The harness labels the external corpora `minlplib_full` / `qplib_full` (the in-repo `data/minlplib/` is `minlplib`). |
| WP-6 builder sweep | **done** | #1797 | 2026-10-19 | 56 cases over every builder area (DAE, frozen + trainable ML, GDP, stochastic, RO, bilevel, GP, all 16 registered transformations): **44 round-trip, 12 refused by name with a verified remedy, 0 crashed** (the 12th, `Disjunct.variable`, arrived with #1790 while this was in flight and is refused under G1.2) (`scripts/serialize_builder_sweep.py --solve`). Entry sweep found **2 crashes**, fixed here: (C1) a `Model.disjunction()` nested in `either_or` crashed `dumps`, **pickle, deepcopy, `clone` and every `create_using`** (`Model._expression_roots` and the row encoder read `.body` off the nested disjunction) — the walk now recurses and the copy owns its nested rows; `dumps` refuses it by name (WP-3b); (C2) the #1430 unattached-block guard raised a bare `ValueError` from `dumps` — now `SerializationError` (a `ValueError` subclass). And **3 refusals for an unclassified or unencodable structure**: `None`/`Ellipsis` indices (DAE controls, `TrainableNetwork`) and nested Disjunct blocks → WP-3b (1.6); `gdp.transform_current_disjunctive_state`'s `_gdp_state_tokens` (undo tokens = its pre-pass rows) → classified with the reformulation records (final, §3.6), so the message now names the remedy instead of telling the user to edit discopt; `_gdp_block_members` → the G1.2 refusal. Remedies verified per refusal: pickle and `clone` fingerprint-equal; save-source-and-re-apply fingerprint-equal for all 5 passes; an unattached block, attached, round-trips. Bound neutrality: 39 cases solving terminally in < 5 s give identical `(status, node_count, objective)` after reload (OR / EXACTLY_ONE_TRUE disjunctions raise the same `NotImplementedError` on both sides, #1124); excluded: fit_trajectories (time limit — its own node count is 337/345/361 across three runs of the *unsaved* model), ml full_space 16.6 s, reduced_space sigmoid 19.4 s, frozen network and bilevel strong_duality (end `feasible`). `-m "not slow"`: 67 tests in 1.6 s (budget 15 s); slow: 39 in 15.5 s. |
| WP-3b-1 schema 1.6 | **done** | #1801 | 2026-10-19 | `{"k": "none"}` / `{"k": "ellipsis"}` index kinds and a disjunction as a disjunct member (`{"disjunction": {...}}`), each read only from minor ≥ 6 and refused by name in an older document (7 tests; 6 fail on the 1.5 reader). The three WP-6 refusals now round-trip: nested `Model.disjunction()` solves to the same optimum (16; a nested OR whose outer arm is infeasible keeps its optimum 7 — flattened to an AND it would be infeasible), DAE collocation with a control identical, `TrainableNetwork` fingerprint-equal (its solve stops at the time limit, not compared). Fixtures: v1.6 20 cases (written by `eca6abfb`); the two new cases recorded unbuildable for 1.0–1.5 via `--cases` (1.5 writer = `6420787e`, the commit that wrote v1.5). Forward matrix vs v1.6: 1.5 reader 18 `loads_identical` + 2 `refused` (exactly the new cases); 1.0–1.4 refuse 7 each (the 1.5 and 1.6 additions), **0 `loads_different`**. |
| WP-3b-2 schema 1.7 | **done** | (this PR) | 2026-10-19 | Disjunct-block state, written only when set: a `block_indicator` row (an indicator row plus its nested block's `parent` and/or a non-default `semantics`), a `nested_disjunction` row (a disjunction row plus `parent_indicator`), and a `_gdp_blocks` state entry (`_gdp_block_members`; when a block declares `Disjunct.variable` columns, the attached disjuncts — rows by position, nesting, locals — rebuilt as fresh `Disjunct` objects on load, so `validate` still refuses an escaped local). Plain models write what 1.6 wrote: 19/20 v1.6 fixtures equal their v1.7 rewrite apart from the schema string (the 20th, `with_result`, differs only in solve timings). **Design correction found by the forward matrix**: written first as `parent`/`semantics` keys on an `indicator` row, the 1.0–1.4 readers (pre-#1780, which skip an unknown nested key) recorded `loaded` for both new cases — an OR block read as select-one, a nested block as top-level, no warning; as new row *kinds* every older reader refuses them (14/14 cells `refused`, pinned by `test_every_older_reader_refuses_block_state`). Only the local-variable record can still be lost by a pre-#1780 reader (an unknown `state` key is skipped), which drops a `validate` refusal, never a row. Round trips: the four GDP tests that pinned the refusal (#1775 G1.2 nesting, G1.4 OR, #1790 locals and an all-empty nested block) now pin the reload — same tree shape, same solve; new fixture cases `gdp_nested_blocks` (optimum 7, hand-checked: B with inner arm x ≤ 8 gives (7, 0)) and `gdp_or_block` (13) solve to the same status, objective and node count as their pickle. Builder sweep: 3 refusals now round-trip (nested blocks, OR blocks, `Disjunct.variable`); the 7 left are final (#1430 unattached blocks, reformulation records). Fixtures: v1.7 22 cases (writer = this PR's code commit); the two new cases recorded unbuildable for 1.0–1.6 (1.6 writer `eca6abfb`; 1.0–1.5 lack `Disjunct.variable` / OR blocks, 1.6 refuses by name). Forward matrix: 514 cells — 379 `loads_identical`, 113 `refused`, 22 `loaded`, **0 `loads_different`**. 13 gate tests in `test_serialize_schema_1_7.py`. |
| WP-7 perf | not started | — | — | pickle A/B: HEAD — ± —, base — ± — ; O2 decided (≤ +20 %) |
| WP-8 docs + CI | not started | — | — | fast-lane delta: — s |

Owner decisions (taken 2026-10-18 by jkitchin: "go with your recommendations"):
O1 — refuse on `_MODEL_PICKLE_FORMAT` mismatch or missing stamp, warn on
`discopt.__version__` skew. O2 — accept up to +20 % model-pickle overhead if WP-7's
two optimisation variants fail (re-measured A/B still required). O3 — unknown
top-level keys refused, with an ignored `x-`-prefixed namespace as the escape hatch.
O4 — if the 1.0–1.2 writers cannot be rebuilt within WP-1's kill criterion, accept
a documented hole ("untested before 1.N"). O5 — encode logical constraints at 1.5.

---

## 9. Change log

- **v1.11 (2026-10-19, WP-3b-2 done)** — Schema 1.7: Disjunct-block state as two
  new row kinds (`block_indicator`, `nested_disjunction`) and a `_gdp_blocks` state
  entry. The kinds replace the first design (new keys on the old kinds) after the
  forward matrix showed the pre-#1780 readers loading those documents into a
  different model without a word; a new kind is refused by every shipped reader.
  `_gdp_block_members` is classified as lazy state (it is set by the first
  `add_disjunction`, not by `__init__`). With 1.7, every WP-6 refusal that was not
  final is closed; WP-3b is complete.
- **v1.10 (2026-10-19, WP-3b-1 done)** — Schema 1.6: `none` / `ellipsis` index kinds
  and a disjunction as a disjunct member. WP-3b split into 1.6 and 1.7 (see its
  section). Four tests that pinned `SCHEMA == 1.5` now pin the minor their feature
  arrived at, so the next bump does not break them.
- **v1.9 (2026-10-19, WP-6 done; WP-3b opened)** — `scripts/serialize_builder_sweep.py` and `test_serialize_builders.py` over the shared case table `python/tests/serialize_builder_cases.py` (56 cases). The kill criterion fired as designed: two crashes (fixed: nested-disjunction copy/pickle, #1430 error class) and three structures without a classification or encoding. One was classifiable now (`_gdp_state_tokens`, a reformulation record, final); the other two need a new section, so per the WP-6 rule they open **WP-3b** (schema 1.6) instead of being folded into the sweep PR. Plan correction: `add_disjunction` takes no `gdp_method=` (the method is chosen at solve time or by a transformation), and the registry has 16 names, including `gdp.fix_disjuncts`, `gdp.transform_current_disjunctive_state`, `gdp.simplex` and `core.relax_integrality`, which §WP-6's table did not list.
- **v1.8 (2026-10-18, WP-5 in-repo done)** — `scripts/serialize_corpus_roundtrip.py` (the harness: in-repo floor always, full corpora from `DISCOPT_MINLPLIB_NL_DIR` / `DISCOPT_QPLIB_DIR` with `usable_oracle` filtering, a printed skip when absent, a per-instance progress line, exit 1 on any failure and 2 below the floor) and `test_serialize_corpus.py` (the two corpus tests moved unchanged from `test_serialize_roundtrip.py`; per-instance round trips over all 134; a probe that a planted lossy `loads` is caught by every check; the 12-instance bound-neutral panel). Two corrections: the in-repo floor is 134, not 75 (a second vendored MINLPLib directory), and the digest check is "preserved, absence included" (a complementarity `.nl` has none by design). Step (4), the full corpora (D11), was dropped by the owner.
- **v1.7 (2026-10-18, WP-4 done)** — Property tests (`test_serialize_property.py`:
  fixed point, fingerprint, pickle, `.nl` agreement; the coverage floor is a test)
  and fuzzing (`test_serialize_fuzz.py`: structure-aware mutations of every golden
  fixture, random bytes, arbitrary JSON; outcome is a `Model` or a
  `SerializationError` within 2 s, and a value mutation of model content that loads
  must load into a different model). Two corrections: (a) the coverage universe is
  the 41 node ops + function names; the strategy also reaches every index and row
  kind, asserted separately. (b) The silent-acceptance property applies to *value*
  mutations only: deleting an optional key legitimately means its default. The
  fuzzer's process registers the fixtures' atoms first (as a program does), else a
  moved atom `arg` is dropped in both original and mutant by design. A Rust
  `PanicException` is a `BaseException`, so hypothesis did not print its
  falsifying example; it was reproduced by hand from the traceback. After review: the
  `ci` profile is derandomized (`derandomize=True`, no database, `print_blob`) so an
  undiscovered bug cannot fail an unrelated PR at random -- `slow` keeps fresh
  examples; a mutant that loads must also be a `dumps` fixed point; and atoms are
  generated (the review's B1 above).

- **v1.6 (2026-10-18, WP-3 done; #1785 carrying #1782)** — Schema 1.5 is one
  format with three additions, each read only from a document of minor ≥ 5
  (`_SCHEMA_MINOR_LOGIC_LOWERING_INDICATORS`) and refused as malformed in an older
  one: the `logical` row kind (WP-3), tracked complementarity lowering rows (WP-3),
  and `indicators` on a disjunction row (#1775 G1.1, #1782). The two PRs had each
  claimed 1.5 for their own half; neither was released, so they were combined into
  one 1.5 rather than one taking 1.6.

  *WP-3.* Three corrections to WP-3 as planned. (a) The plan said to refuse a
  cardinality `k` outside `[0, len(operands)]`; the API builds such a constraint
  (an infeasible or vacuous one), and a reader must not refuse a model the writer
  can produce, so `k` is required to be an `int` (not `bool`) and nothing more.
  (b) "`rows` pointing at a non-`Constraint`": a lowering's rows are a slice of
  `_constraints` and may be any row kind, so positions are checked for range and
  uniqueness only; a row no longer in the model is refused **on write**. (c) The
  lowering-rows element is written only when the rows are tracked: an untracked
  record keeps the two-element 1.4 form, so a 1.4 reader still reads it (the
  three-element form is refused by every older reader — recorded in the forward
  cells). The logic is a flat table, not nested JSON: `land(...)` builds a
  left-deep chain (a 3000-link chain is tested). Leaves must be a binary variable
  of size 1 or an element of a binary array — what `Model.boolean` builds, and
  what `either_or(...).indicators` allocates (a scalar binary per disjunct) —
  refused both ways otherwise. `_REFUSED_RELATIONS` is empty but kept; the
  `CustomCall` and reformulation-output refusals are documented as permanent.

  *G1.1.* (d) `indicators` is written only once the indicators were read, so a
  model that never reads them writes the 1.4 content under the 1.5 identifier
  (`v1.5/mip_indicator_sos_disjunction.json` equals its v1.4 fixture but for the
  schema string — tested). (e) On read it must be one distinct scalar binary column
  per disjunct — the GDP lowering uses the columns as selectors, so a reference to
  anything else would lower to a different model. (f) Logic over the indicators —
  the case G1.1 exists for, refused on save while the format had no logical rows —
  now round-trips: `test_logic_over_indicators_round_trips` saves, reloads and
  solves to the same status, objective and node count, with the reloaded leaves
  resolving to the reloaded indicator columns, against hand-computed optima (10
  unconstrained, 45 under `mode_1 → pump_1`, 50 with `mode_0 ∧ ¬pump_1` forced).
  No decoder change was needed: an indicator column is a scalar binary, which the
  leaf check already accepts.

  *Fixtures.* (g) One mechanism for a case added after a version shipped: the
  generator's `--cases` mode records it for every already-shipped version
  (#1782's `INTRODUCED` table, which declared absence instead, was removed). Each
  record is what that version's writer did: `logical_constraints`,
  `gdp_indicators` and `gdp_indicators_logic` are unbuildable at 1.0–1.4, each with
  its own error; `complementarity_vector_lowered` is written by each. An existing
  record is refused, not replaced, and no shipped fixture file changed. An add-only
  run under a different interpreter records it per case in `cases_python`.
  `v1.5/` holds 18 cases, written by the combining merge `6420787e`. Forward
  matrix: 240 cells — 205 `loads_identical`, 10 `loaded`, 25 `refused`, **0
  `loads_different`**; the 150 cells not involving 1.5 are unchanged. Each older
  reader (1.0–1.4 at their manifest commits, rebuilt for this run under 3.12.11)
  refuses exactly the five 1.5 documents with a logical row or tracked lowering
  rows, and loads `gdp_indicators.json` without the link (see (i)).
  (h) The generator compares a `from_nl` source by file name, as the matrix test
  does (a 1.5 document written from a different checkout path read as
  `loads_different` before), and records a forward-only reader run's interpreter
  under `forward_python`. (i) Known hole in shipped readers, not fixable in place:
  the pre-#1780 readers (1.0–1.4 at their manifest commits) do not track nested
  keys and load `gdp_indicators.json` *without* the link — under the 1.4 reader
  (`0ff5cb53`) it solves to **10.0** instead of **45.0**, with no warning — recorded
  as `loaded` (nothing to compare: unbuildable there). The post-#1780 1.4 reader
  (`ee504c66`) refuses it with "newer than this reader … Upgrade discopt".

- **v1.5 (2026-10-18, #1780)** — Nested-record keys. A per-record table for every
  record kind would have had to mirror every decoder branch, so the reader instead
  records which keys it reads (`_Record`, built from the parsed document outside
  the node table) and refuses any key left unread. That is derived from the
  decoder, so it cannot drift; `_keys_lost` in `upgrade-model` is now a backstop.
  Tracking the node table too measured **+130 %** on `loads` of a 24k-node / 1.4 MB
  document (5 interleaved pairs, load < 2: 0.060 s → 0.141 s min), so nodes and
  index objects use explicit `_NODE_KEYS` / `_INDEX_KEYS` tables, and a test decodes
  every fixture with *tracked* nodes to prove the tables are exactly what the
  decoder reads. Final cost: 24k-node document 0.061 → 0.088 s (min of 7, 5
  interleaved pairs); `loads(dumps(from_nl(f))) / from_nl(f)` over the 8 largest
  in-repo corpus files 0.99× → 1.37× (3 interleaved runs, load < 1), inside WP-7's
  5× budget. Two real holes found on the way: (a) beside a builder objective the
  writer records the zero placeholder `objective`, which the reader never read — a
  document with a real objective there would have lost it; it is now checked
  (`placeholder` true, sense agreeing); (b) a result's own `provenance` block
  (daemon-stamped) was stored as the parsed object — now a plain copy. A JSON key
  repeated in one object (kept-last by `json`) is refused too.

- **v1.4 (2026-10-18, WP-2 done)** — Implemented §3.2/§3.4/§3.5 with three
  measured additions. (a) The unknown-key rule covers the `state` section too
  (`_STATE_KEYS`), not only the top level: a 1.4 key there (`nl_source`) is model
  content, and the same silent drop applies; the `x-` escape is top-level only.
  The tables' minors are derived from the first revision of `serialize.py` that
  wrote each key, not from the last writer of each minor (a key added between two
  bumps is written under the earlier one). A *known* key in a document older than
  the key is not refused — it decodes correctly, so refusing it buys nothing.
  (b) §3.4's "a lone expression pickle stays unstamped" is true per node but not
  per pickle: a variable references its model, so a lone expression pickle carries
  the model and its one stamp (tested: one stamp in a 2000-node expression).
  (c) `upgrade-model` refuses any rewrite that would drop a key the old document
  carried. Found by running it on the fixtures: atom tags are re-derived only for
  atoms the process has registered (v1.3 (d)), so a bare CLI process silently
  stripped them; `--import MODULE` registers them first. It also carries the old
  block's `author` forward (an upgrade does not change who made the model) and
  verifies the rewrite reloads to the same model before writing. `result_io`
  unknown keys were silently *skipped*, not passed to `SolveResult(**kwargs)` as
  §3.2 said; now refused either way.

- **v1.3 (2026-10-18, WP-1 done)** — §3.3/WP-1 corrected by measurement:
  (a) the bump search needs `git log -G`, not `-S` (`-S` found only the file's
  introduction); (b) schema 1.0 writes the identifier `"discopt.model/1"`, so the
  1.0 marker is that string; (c) a `--system-site-packages` venv imported
  `discopt.operators` — absent at 1.0 — from the *current* tree, because
  `discopt/__init__.py` uses `pkgutil.extend_path`; the first 1.0 `registered_atom`
  fixture was therefore a two-version mix and was discarded. The generator now
  isolates `sys.path` and refuses on any foreign `discopt.*` module. (d) A forward
  cell first read `loads_different` for 1.3←1.4 `registered_atom`: the reader's
  process had not registered the atom, so the (sound, designed) re-derivation left
  the tag off — process state, not a misread. Forward mode now builds every case
  before reading, as a program registers its atoms at import. (e) The case list
  swaps the plan's `gams_initial_values` / `recentre_marker` (internal attributes no
  public 1.0 API sets) for `parameters_and_start` and `decomposition_marks`; the
  internal attributes stay covered by `test_serialize_adversarial.py`. (f)
  `with_result` fixtures are not byte-reproducible across runs (the embedded result
  carries timings); fixtures are frozen once, so the immutability check is unaffected.
- **v1.2 (2026-10-18)** — owner decisions O1–O5 recorded in §8 (the §7
  recommendations, adopted as written).
- **v1.1 (2026-10-18, review)** — claims spot-checked against `3191cc65`: shallow
  clone; `v0.9.0` writes 1.3 and `v0.8.0` has no `serialize.py`; `hypothesis` absent
  from `dev`; all cited test names, data files, `result_io.SCHEMA_VERSION = 3`,
  `provenance.capture(previous=)`, the CLI subparsers, the logical classes,
  `_logical_to_linear`, `20d663c6` = #1764 merge, the six `ci.yml` install lines and
  the WP-6 builder entry points exist as stated. Corrected one error: the
  `fixed_scope` fixture said `with x.fix(1.0)`; `fix()` is not a context manager.
- **v1 (2026-10-18, Claude for jkitchin)** — initial plan from a read of
  `serialize.py`, `result_io.py`, `provenance.py`, `batch.py`, `modeling/core.py`
  (pickle/deepcopy section, `CustomCall`, logical classes, `from_nl`),
  `operators.py`, `solvers/nlp_native.py`, `transformations.py`, the eight
  serialization-related test files, the notebook, `ci.yml`, `pyproject.toml`, and
  `git log` (shallow). No code changed.
