# Export Module Review — Correctness, Thoroughness, Performance

**Date:** 2026-07-03
**Scope:** `python/discopt/export/` (`_extract.py`, `nl.py`, `mps.py`, `lp.py`,
`gams.py` — 2,445 lines) and `test_export.py`.
**Method:** Full read of all five files; every finding reproduced end-to-end. The
MPS/LP writers were **cross-validated against HiGHS** (round-trip solve comparison)
and the `.nl` writer against **Pyomo's reference NL writer** on the same model.
Baseline: **32 tests passed** (0.3 s).

Exports are certificate-relevant: a wrong export means external solvers (BARON,
Couenne, SCIP — the benchmarking framework's comparison baselines) solve a
*different* model than discopt did, silently corrupting every cross-solver
comparison this repository exists to make.

---

## 1. Summary of findings

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| EX-1 | **P0 interop** | `nl.py` | `.nl` Jacobian structure is **non-conformant with the ASL convention**: variables appearing only nonlinearly in a constraint get no zero-coefficient J entry, so the header nonzero count, the k section, and the J sections mutually disagree (4 vs 1 vs 3 on a 2-var model). Pyomo's writer on the same model emits `J0 2 / 0 0 / 1 1` and consistent counts [CONFIRMED by direct diff] |
| EX-2 ✅ RESOLVED 1dc3278 | **P0 silent-wrong** | `mps.py`, `lp.py`, `gams.py` | **Builder-resident models export as empty/zero models**: fast-API constraint rows and `add_linear_objective` are invisible — MPS emits a ROWS section with only OBJ, LP emits `obj: 0`, GAMS emits `obj_var =e= 0`. Only `nl.py` handles builder blocks [CONFIRMED]. **FIXED (X-1):** MPS/LP/GAMS now emit builder rows via `export._common.iter_builder_linear_rows` and recover a builder-resident linear/quadratic objective via `export._common.builder_objective` (was `obj: 0`). Exported MPS round-trips through HiGHS to the true optimum (3.0). Regression: `test_x1_builder_resident_rows.py::test_ex2_*`. |
| EX-3 | **P0 silent-wrong** | `mps.py:158`, `lp.py:127`, `gams.py:163` | **Fixed binaries lose their fixing** in all three formats (the binary branch ignores `lb`/`ub` entirely: `BV BND`, implicit LP binary, no GAMS bounds). Writer-side sibling of the `from_nl` import bug (modeling-review M2); `.nl`'s b-section is correct [CONFIRMED] |
| EX-4 | **P1 silent-wrong** | `gams.py:176-191` | **Heterogeneous per-element array bounds are silently dropped entirely** (written only when uniform) — a DAE-style model with `u[0]` pinned to 1 via bounds exports with **no bounds at all**; the `.nl` control correctly writes `4 1.0` [CONFIRMED] |
| EX-5 | P1 crash | `gams.py:168-169` | `to_gams` **crashes** (`TypeError: only 0-dimensional arrays…`) on any shape-`(1,)` variable — which every `over=` indexed variable with a 1-element set produces [CONFIRMED] |
| EX-6 | P1 | `gams.py` | No scalarization pass: vector/broadcast constraint bodies and `SumExpression`/`MatMul` render as scalar-syntax garbage (`(x)` for `sum(x)`, `(A * x)` for `A@x`), and unknown nodes are written **into the file** as `<unsupported:…>` instead of raising — export "succeeds", GAMS compile fails later (or worse) [BY INSPECTION] |
| EX-7 | P2 | `_extract.py:153,475` | Array-valued `Constant` in a scalar position silently collapses to its **first element** (`value.flat[0]`); `sum(constant_vector)` likewise returns the first element rather than the sum [BY INSPECTION] |
| EX-8 | P2 | `mps.py`/`lp.py`/`gams.py` | Constraint-name hygiene: duplicates never checked (duals/rows silently collide), `_sanitize_name` handles only space/dash — indexed-family names like `cap[pitt]` corrupt LP files (brackets delimit quadratic sections) and break GAMS |
| EX-9 | P3 | `nl.py:912-924` | k-section built by an O(n_vars × n_cons) nested scan — quadratic build time on large models (also wrong per EX-1; fix together) |

Checked and found **correct** (worth as much as the findings):

- **MPS and LP numeric conventions verified against HiGHS end-to-end**: LP
  minimize/maximize (single-line `OBJSENSE MAX` accepted), QP — both the MPS
  `QUADOBJ` convention (diagonal ×2, off-diagonal once, `0.5·xᵀQx` form) and the
  LP `[…]/2` convention (coefficients ×2) — and MILP `INTORG/INTEND` markers +
  integer bounds: HiGHS reproduces discopt's optimum exactly on all four probes.
  The long agonized comment in `mps.py` about the QUADOBJ factor reaches the
  right answer.
- **`nl.py` is the flagship writer**: it scalarizes vector/broadcast bodies
  (handling exactly the DAE-style constraints the LP classifier mishandles),
  recovers builder-resident rows *and* the placeholder objective, preserves fixed
  binaries, implements the issue-#210 canonical variable reordering (nonlinear
  groups with discrete members last, correct `nbv/niv/nlvbi/nlvci/nlvoi` header),
  uses iterative traversal everywhere against recursion-limit blowups, refuses
  `CustomCall` loudly with a helpful message, and decomposes `tan/log2/log1p/
  sigmoid/softplus` into real opcodes. EX-1 is the one structural defect in an
  otherwise carefully engineered writer.
- **GAMS scalar happy path verified**: nonlinear model → `to_gams` → `from_gams`
  → solve reproduces the optimum exactly (1.302671 both ways); the
  `power`/`rPower` integer-exponent distinction and quoted 1-based set labels are
  handled correctly.
- `_extract.py`'s quadratic expansion algebra ((Σcᵢxᵢ+k)² cross-terms, product
  bilinear terms) is correct, and its refusals on genuinely nonlinear content are
  loud with actionable messages.

---

## 2. The `.nl` conformance bug (EX-1) in detail

Reproduction (2 variables, `exp(x) + y <= 5` and `x + y <= 3`):

| Quantity | discopt | Pyomo reference |
|---|---|---|
| Header Jacobian nonzeros | 4 | 4 |
| J-section entries | **3** | 4 (`J0 2` incl. `0 0` for the nonlinear-only x) |
| k-section (col-0 cumulative) | **1** | 2 |
| Header nonlinear constraints | **2** | 1 |

Three intertwined defects:

1. **Missing zero-coefficient J entries.** ASL requires every variable appearing
   in a constraint — linearly *or* nonlinearly — to have a J entry carrying its
   linear coefficient (0 if none). `_write_J_sections` writes only
   `_con_linear`; the nonlinear appearances counted into header line 7 (via
   `_collect_var_indices`) never materialize. ASL builds its column-wise Jacobian
   structures from header+k+J; on mismatch it errors or mis-addresses gradients.
2. **Header double-counting**: a variable appearing both linearly and nonlinearly
   in the same constraint is counted twice in `n_jac_nz` (`len(lin)` +
   un-deduplicated nonlinear refs).
3. **Constants inflate the nonlinear-constraint count**: `_collect_linear` sends
   nonzero constants to the `nonlinear` list, so a purely linear `x + y <= 3`
   gets `C1 = n-3.0` and is counted in header line 2's nonlinear constraints.
   (Constant-in-C is *valid* encoding; the count is what's wrong. Pyomo instead
   carries the constant in the r-section rhs.)

Why tests never caught it: the only `.nl` consumer in CI is the in-house Rust
parser, which recomputes structure from the expressions — round-trip verified
exact here. The victims are exactly the external ASL solvers the exporter exists
for.

**Fix (one PR):** per constraint, take `union(vars(linear), vars(nonlinear))`;
emit J entries for the union with 0.0 for nonlinear-only vars; compute header
`n_jac_nz` as the sum of union sizes (same for the objective gradient with
`G`-entries); build the k section from the same union map (also fixes EX-9's
quadratic scan — accumulate per-column counts in one pass); keep constants in
the C body but count a constraint nonlinear only when its nonlinear part
references at least one variable. **Acceptance:** byte-level structural
equivalence with Pyomo's writer on a small conformance corpus (linear-only,
nonlinear-only-var, mixed, objective-nonlinear); in-house round-trip unchanged;
ideally one smoke test running an ASL solver binary if available in CI.

---

## 3. Cross-writer parity: the silent-empty-model class (EX-2/EX-3/EX-4)

`nl.py` learned three lessons that the other writers never received:

- builder-resident rows (`_decompose_builder_blocks`) and the placeholder
  objective (`_decompose_builder_objective`),
- per-element bounds (b-section is per flat element),
- fixed binaries (bounds written regardless of var type).

MPS/LP/GAMS have none of the three. The result is the worst failure mode:
`m.to_mps()` on a fast-API model **succeeds** and produces a syntactically valid
file describing "minimize 0 subject to nothing". GAMS additionally drops *all*
bounds on any array variable whose bounds aren't uniform — erasing DAE initial
conditions — and crashes outright on shape-(1,) variables.

**Fix direction:** extract the three behaviors from `nl.py` into shared helpers
(`export/_common.py`): `iter_all_rows(model)` yielding expression-path rows *and*
builder-block rows uniformly, `iter_scalar_bounds(model)` yielding per-element
`(name, type, lb, ub)` (fixed-binary aware), and `resolve_objective(model)`
returning the real objective (placeholder-aware). Port all three writers onto
them. Until the GAMS scalarization gap (EX-6) is closed, `to_gams` should
*refuse loudly* on array-structured bodies rather than emit garbage — same for
the `<unsupported:…>` fallback (raise, don't embed).

---

## 4. Test coverage gaps

`test_export.py` (32 tests) covers scalar happy paths per format. Missing, in
order of the damage each gap allowed:

1. **No conformance comparison for `.nl`** against a reference writer, and no
   test with a variable appearing only nonlinearly in a constraint (EX-1).
2. **No external-reader round-trip** for MPS/LP — the HiGHS harness used in this
   review (write → `highspy.readModel` → solve → compare optimum) is ~20 lines
   and would pin the format conventions permanently; parameterize over
   LP/QP/MILP/maximize.
3. **No fast-API model** through any writer (EX-2); no fixed-binary bound test
   (EX-3); no heterogeneous-bound array (EX-4); no shape-(1,) variable (EX-5);
   no vector-body GAMS export (EX-6); no bracketed constraint names (EX-8).
4. No `to_gams → from_gams` round-trip test (verified working here — pin it).

---

## 5. Implementation plan (for Opus)

House rules per CLAUDE.md. `highspy` and `pyomo` are already optional deps of the
ecosystem — the new tests should skip gracefully when absent.

### Phase 1 — `.nl` conformance (PR `fix(export): EX-1`)

Union-based J/G/k/header emission per §2; Pyomo-diff conformance corpus;
`from_nl` round-trip unchanged; k-section single-pass (EX-9). This is the
highest-stakes fix: it unblocks trustworthy BARON/Couenne/SCIP comparisons.

### Phase 2 — cross-writer parity (PR `fix(export): EX-2..EX-6`)

| ID | Task | Acceptance |
|----|------|-----------|
| EX-2 | Shared `iter_all_rows`/`resolve_objective`; port MPS/LP/GAMS | Fast-API repro exports the true model in all four formats; HiGHS round-trip reproduces the optimum |
| EX-3 | Fixed-binary bounds: MPS `BV` → `FX` when lb==ub (or `LI/UI`), LP explicit bounds line, GAMS `.fx` | Fixed-binary repro round-trips through HiGHS at optimum 11.0 (not 2.0) |
| EX-4 | GAMS per-element bounds (`u.lo('1') = …`) with uniform-case compaction | Pinned-element repro exports the pin; `from_gams` round-trip preserves the optimum |
| EX-5 | Scalar-vs-(1,) shape handling in `_write_variables` | shape-(1,) model exports without crash |
| EX-6 | GAMS: raise on array-structured bodies and unknown nodes (then, later, port `nl.py`'s `_scalarize` for full support) | Vector-body model raises `ValueError` naming the constraint (currently emits garbage) |

### Phase 3 — hygiene (PR `fix(export): EX-7..EX-8`)

Array-constant refusal in `_extract._get_constant_value`/`_extract_linear_recursive`
(raise instead of `.flat[0]`); name sanitization shared across writers
(alphanumeric+underscore, dedup with numeric suffixes, length caps for MPS);
HiGHS round-trip test harness landed as a permanent fixture.
