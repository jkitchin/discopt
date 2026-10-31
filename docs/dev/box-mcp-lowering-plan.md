# Exact lowering of the box-bounded MCP relation — implementation plan

**Status:** plan, ready to implement. No source file is changed by this document.
**Tree:** written against `main` at `677af90` (2026-10-31). Line numbers below are
from that commit; re-`grep` the function names before editing.
**Scope:** `Model.mcp` / `discopt.mpec.box_mcp` relations with a general box
`z ∈ [l, u]` become **lowerable and solvable** (GDP, SOS1, Scholtes), the `.nl`
type-5 importer stops refusing doubly-bounded / free-body / equation rows, and
the normative contract `docs/complementarity_semantics.md` §2 is updated.
**Predecessors:** #231 (closed), #1123 RFC (closed), #1147 representation
("do not lower in this slice", closed), #1148 residuals (closed). No open issue
tracks box lowering; §11 proposes the issue and PR titles.

Entry-experiment results in this document are **measured** on this tree
(Rust extension built from `677af90`, Linux, `pounce-solver` installed) unless a
paragraph says **proposed**. The scripts are reproduced in Appendix B.


> **Erratum (2026-10-31, #2056 — read this before §2.1 H2, §4 commit 5, §6.5 N1–N4,
> Appendix A).** This plan took the `.nl` type-5 flag bits from discopt's importer as it
> was at `677af90`, and that importer had them **swapped**. AMPL MP `ComplInfo`
> (`include/mp/nl-reader.h`) defines `INF_UB = 1` = *finite variable lower bound* and
> `INF_LB = 2` = *finite variable upper bound*; Pyomo's `mpec4` writer agrees (+1 for a
> lower bound, +2 for an upper bound); and AMPL itself writes `3 <= x <= Infinity
> complements y` as `5 1 1`, `-Infinity <= x <= -3 complements y` as `5 2 1`,
> `3 <= x <= 10 complements y` as `5 3 1`. So **flag 1 → box `[lb, +∞)` (body ≥ 0 at
> lb)** and **flag 2 → box `(−∞, ub]`**, the reverse of the commit-5 table, the
> `inf_lb, inf_ub = bool(flag & 1), bool(flag & 2)` line, probe H, N1–N4 and A.1–A.5.
> Consequences:
> * The "one committed fixture" `mpcc_complementarity.nl` (flag 2 on `y ∈ [0, 10]`) is
>   **malformed** under AMPL's semantics (AMPL would write flag 3), so "bit-for-bit on the
>   NCP path for that fixture" is not a valid regression target. Since #2056 `from_nl`
>   refuses it as malformed; the file is kept only because serialization golden
>   documents pin its digest. Use `python/tests/data/nl_complementarity/ampl_*.nl`
>   (genuine AMPL output, each with its `.mod`) instead.
> * The fix for #2056 already does the single-bounded half of commit 5: flag 1 is lowered
>   as `body ⊥ (x − lb)`, flag 2 as `(−body) ⊥ (ub − x)`, and a flag inconsistent with the
>   variable's declared bounds is refused as malformed. Commit 5 reduces to routing the
>   still-refused **flag 3** (doubly bounded, `ampl_box.nl`) and **flag 0** (free, body
>   = 0) through `Model.mcp`, using the corrected table above.
> * Swap flags 1 and 2 throughout the N- and A-tests before implementing them.

---

## 0. One-paragraph summary

A box-MCP relation `F(z) ⊥ z ∈ [l, u]` is exactly the conjunction of (at most)
two symmetric NCP pairs on shifted slacks:

```
F = w⁺ − w⁻,  w⁺, w⁻ ≥ 0,
0 ≤ w⁺ ⊥ (z − l) ≥ 0,        (absent when l = −∞)
0 ≤ w⁻ ⊥ (u − z) ≥ 0,        (absent when u = +∞)
```

with the one-sided cases needing **no** `w±` at all (`0 ≤ F ⊥ (z − l) ≥ 0`, or
`0 ≤ −F ⊥ (u − z) ≥ 0`), the free case reducing to the row `F = 0`, and `l = u`
constraining `F` not at all. Each half is handed to the **existing, unchanged**
NCP emitters of `reformulate_gdp` / `reformulate_sos1` / `reformulate_scholtes`,
so (i) the NCP path is bound-neutral by construction, (ii) all three methods get
the box form from one decomposition, and (iii) the Scholtes homotopy gets its
analog for free (`w⁺·(z − l) ≤ t`, `w⁻·(u − z) ≤ t`). Finite upper bounds for
`w±` come from the interval enclosure of `F`; a non-finite enclosure is a loud
refusal, never a fabricated big-M.

The **one correction to the proposed approach**: the `.nl` importer is not just
"stop refusing k=3". It currently encodes `x·F = 0` for a complementary variable
with `lb > 0` (or `ub < 0`), where AMPL's meaning of `x >= 3 complements F >= 0`
is `(x − 3)·F = 0`. Measured (§2.3): a hand-written row with `x ∈ [3, 10]`
solves to objective **1.0** today and to **0.0** under the box reading, with the
point `(3, 1)` that the current import cuts off. Routing every type-5 row through
`Model.mcp` with the box read off the variable's bounds fixes the class, and keeps
the one committed fixture (`x ≥ 0`, flag 2) *bit-for-bit* on the NCP path.

---

## 1. Problem statement and definition of done

### 1.1 Today (verified by reading and by running the tests at §9.1)

* `python/discopt/mpec.py` — `Complementarity` (l.158) carries `f_bounds`,
  `g_bounds`, `role`; `is_symmetric_nonnegative` / `is_box_mcp` (l.242/257) read
  the **declared bounds**, never `role`. `box_mcp` (l.444) builds the box relation
  and returns an `NCP_PAIR` for `l=0, u=+inf`. `_require_lowerable` (l.1089)
  raises `NotImplementedError("… box-MCP relation … has no 'gdp' lowering …
  (#1147)")` from all three of `reformulate_scholtes` (l.1155),
  `reformulate_sos1` (l.1177), `reformulate_gdp` (l.1219). `_lower_each` (l.1118)
  scalarizes one relation at a time and records its rows positionally via
  `mark_lowered`. `require_all_relations_lowered` (l.871) is the solver-boundary
  guard with a message that says the box form is "represented but not lowered
  (#1147)". `tighten_complementarity_bounds` (l.1288) skips box relations.
* `python/discopt/modeling/core.py` — `Model.complementarity` (l.8682) lowers via
  `transformations.get("mpec.gdp"|"mpec.sos1")`; `Model.mcp` (l.8779) lowers
  **only** the `l=0,u=inf` case and otherwise records an unlowered relation that
  `Model.solve` (l.10149) refuses. `_add_nl_complementarity` (l.14246) refuses
  flag `0`/`3` and a complementary variable that "straddles 0"; `from_nl` calls
  it at l.14101.
* `python/discopt/mpec_report.py` — the natural map `|z − mid(l, u, z − F)|`
  (`_complementarity_residual`, l.708; `_resolve_kind`, l.721) already measures a
  box relation; `source_residual_report` (l.848), `accept_local_incumbent`
  (l.1258) need no change.
* `python/discopt/transformations.py` l.767–782 registers `mpec.gdp`, `mpec.sos1`,
  `mpec.scholtes` (the last `exact=False`). **No new registry key is needed**; the
  box lowering lives inside the existing three.
* `python/discopt/serialize.py` `_enc_complementarities` (l.2093) already
  round-trips `role`, `f_bounds`, `g_bounds` and the per-model lowering marks.
* `docs/complementarity_semantics.md` §2 states the box form is "represented and
  not lowered".
* Tests that assert the **refusal** and must change deliberately (§6.1):
  `python/tests/test_complementarity_provenance.py::test_solve_refuses_a_model_carrying_an_unlowered_relation`
  (l.353), `::test_box_mcp_is_refused_by_every_lowering_rather_than_mis_encoded`
  (l.365), `::test_lowering_is_gated_on_declared_bounds_not_on_role` (l.553);
  `python/tests/test_nl_complementarity.py::test_unsupported_complementarity_config_refuses_loudly`
  (l.86). `::test_box_mcp_round_trips_through_a_rebuilding_pass` (l.276) keeps
  passing but its docstring ("an unlowered relation emits no GDP rows") becomes
  false and gains an assertion.
* The `.nl` **writer** refuses the lowered rows (`discopt/export/_common.py:
  refuse_non_algebraic_relations`, called from `export/nl.py` l.140/175/243/271),
  so `.nl` fixtures with type-5 rows are hand-written (Appendix A), as the
  existing `python/tests/data/mpcc_complementarity.nl` is.
* No discopt-side corpus instance carries a type-5 row other than that fixture
  (`grep -rlE '^5 [0-9]+ [0-9]+ *$' python/tests/data discopt_benchmarks
  --include=*.nl` → the one file). The MINLPLib snapshot at
  `~/Dropbox/projects/discopt-minlp-benchmark/` is **not present** on the machine
  this plan was written on; the implementer should re-run that grep there (§9.5)
  — MINLPLib has no MPEC instances, so expect zero, and record the count.

### 1.2 Done means

1. `m.mcp(F, z, lb=l, ub=u, method=...)` lowers every box configuration in the
   case table of §3.2 exactly, for `method ∈ {"gdp", "sos1"}`; `solve_mpec(m,
   [box_mcp(...)], method="scholtes")` runs the homotopy on it and returns a
   *local* result; `m.solve()` on the lowered model returns a **certified**
   result whose incumbent has natural-map residual `≤ 1e-6` on the source
   relation.
2. `reformulate_{gdp,sos1,scholtes}` no longer raise for a box relation;
   `_require_lowerable` is gone; the only refusals left are the ones in §3.5
   (declared form not expressible, non-finite enclosure for `w±`, empty box,
   array `lb`/`ub`), each a `ValueError` naming the relation and the fix.
3. `from_nl` lowers all four type-5 flag values through `Model.mcp`; the existing
   fixture produces **identical rows and node count** to today (bound-neutral);
   the `x ∈ [3, 10]` fixture of §2.3 solves to `0.0`.
4. Every test in §6 exists and passes; the pre-change failures listed in §6.6 are
   confirmed to fail on `main`.
5. `docs/complementarity_semantics.md` §2, the docstrings in §7, the notebook
   section and BibTeX entries of §8, and `CHANGELOG.md` are updated.
6. The bound-neutral NCP panel (§9.3) shows `node_count` and `objective`
   **exactly unchanged** before/after, and the PR records the numbers.

### 1.3 Explicit non-goals

* No PATH / native MCP backend; no GAMS `MCP` model-type matching; no variational
  inequalities; no M-/C-/S-stationarity claims (semantics §8 stays: `stationarity
  = None`).
* No per-element boxes on a vector relation (`g_bounds` stays a scalar pair;
  per-element boxes are declared per element — §3.6).
* No change to `tighten_complementarity_bounds` beyond keeping it a sound no-op
  on box relations (the generalization is specified as a follow-up in §10.3).
* No change to the GDP/SOS1 big-M machinery, to `either_or`, or to any
  relaxation. The NCP emitters are called, not edited.

---

## 2. Hypothesis, evidence, kill criterion, entry experiment (CLAUDE.md §4)

### 2.1 Hypothesis

**H1 (exactness).** For `l < u` both finite, the projection onto `(z, F)` of
`{F = w⁺ − w⁻, w± ≥ 0, 0 ≤ w⁺ ⊥ z − l ≥ 0, 0 ≤ w⁻ ⊥ u − z ≥ 0, l ≤ z ≤ u}` is
exactly the MCP set `{z = l, F ≥ 0} ∪ {l < z < u, F = 0} ∪ {z = u, F ≤ 0}`.
*Proof.* (⊆) If `l < z < u` both slacks are positive, so `w⁺ = w⁻ = 0`, `F = 0`.
If `z = l < u` then `u − z > 0` forces `w⁻ = 0`, so `F = w⁺ ≥ 0`. Symmetrically
`z = u > l` gives `F = −w⁻ ≤ 0`. (⊇) Take `w⁺ = max(F, 0)`, `w⁻ = max(−F, 0)`:
at `z = l`, `F ≥ 0` gives `w⁻ = 0` and `w⁺·0 = 0`; interior, `F = 0` gives both
zero; at `z = u`, symmetric. ∎ The one-sided cases are the same argument with one
half; `l = −∞, u = +∞` gives `F = 0` directly; `l = u` makes both slacks
identically zero, so the halves impose nothing on `F`, which is the semantics
table (`z` pinned, `F` free). Finite `w±` bounds `max(0, hi F)`, `max(0, −lo F)`
from a valid enclosure of `F` cut no point of the set, since `w⁺ ≤ max(F,0) ≤ hi`.

**H2 (importer semantics).** AMPL's `.nl` type-5 row `5 k i` means `F ⊥ x_i` with
the complementarity box being the **variable's own bounds** selected by `k`:
`k=3` → `[lb, ub]`, `k=2` → `[lb, +∞)`, `k=1` → `(−∞, ub]`, `k=0` → `(−∞, +∞)`
(i.e. `F = 0`). In particular `x >= 3 complements F >= 0` is `(x−3)·F = 0`, not
`x·F = 0`.

### 2.2 Evidence

* H1: the proof above, and the measured probes of §2.3 (A–G) where a hand-built
  split through today's `Model.complementarity` reproduces a brute-force
  enumeration of the MCP-feasible set on 7 models, 2 methods, 19 checks.
* H2: AMPL's documented meaning of `expr1 >= 0 complements expr2 >= 0` is
  "`expr1 ≥ 0, expr2 ≥ 0, expr1·expr2 = 0`" (Fourer–Gay–Kernighan, *AMPL*, ch. 19;
  Ferris–Fourer–Gay 1999, §2–§5). In the `.nl` encoding the body's finite bound
  is always 0 (`k` only says which bounds are infinite — see the repo's own
  reading in `_add_nl_complementarity` and `nl_reconstruction.reconstruct_complementarities`,
  l.157–161), so the constant `3` can only live in the variable's bound. PATH's
  AMPL link reads the MCP box from the variable's bounds for the same reason.
  **Not yet confirmed byte-for-byte** on this machine (no AMPL here). The owner's
  machine has AMPL (`/Applications/AMPL`, per CLAUDE.md) — §2.4 gives the
  one-minute check to run before merging the importer commit.

### 2.3 Entry experiment — **measured** on `677af90`

Script: Appendix B.1 (`entry_experiment.py`), B.2 (`entry_experiment2.py`). Each
hand-builds the §3 lowering with today's `Model.complementarity` (so it exercises
the exact rows the implementation will emit), compares against a dense-grid
enumeration using the natural map `|z − mid(l, u, z − F)| < 1e-9`, and ends with
an executed-check counter. `uptime` load average at start: 0.03.

| probe | model | brute force | gdp | sos1 |
|---|---|---|---|---|
| A (guard probe) | `z∈[−1,1]`, `max z`, `F = z+1` | feasible `{−1}` | `optimal −1.0`, 0 nodes | `optimal −1.0`, 0 nodes |
| B | `z∈[−1,1]`, `F = −z`, `min (z−0.8)²` | feasible `{−1, 0, 1}` → 0.04 | `0.04`, 3 nodes | `0.04`, 3 nodes |
| C (nonlinear F) | `z∈[−1,2]`, `F = z²−1` | feasible `{−1, 1}` | `max z = 1.0`, `min z = −1.0` | — |
| D (z an expression) | `z = x+y ∈ [0.5,1.5]`, `F = x−y`, `min x+3y`, `x,y∈[0,1]` | 0.5 at (0.5, 0) | `0.5` at (0.5, 0), 1 node | — |
| E (u = +∞) | `z ≥ 1`, `F = 3−z` | feasible `{1, 3}` | `max 3.0`, `min 1.0` | — |
| F (l = −∞) | `z ≤ 2`, `F = z−1` | feasible `{1}` | `1.0` both senses | — |
| G (Cournot, caps 2/10) | `F_i = c − (a − bQ) + b q_i`, `max q₁+q₂` | unique eq. (2, 3.5) | `5.5` at (2, 3.5), 0 nodes | — |
| S (Scholtes on B's split) | `solve_mpec(method="scholtes", t_min=1e-8)` | — | `local_optimal 0.0399999951`, `bound None`, `gap_certified False`, source natural-map residual `1.2e-08` | — |

**Executed checks: 19 (B.1) + 8 (B.2); exit 0 both.**

**Importer probe H (measured).** Hand-written `.nl` (Appendix A.2): `min
(x−3)² + (y−1)²`, body `y`, `5 2 1` (flag 2, complements `x`), `x ∈ [3, 10]`,
`y ∈ [0, 10]`. Parser: `n_constraints = 0`, `n_complementarities = 1`,
`var_index = 0`, `flag = 2`. **Today's `from_nl` → `optimal 1.000000002` at
`(3.00005, −6.6e-12)`** (it emitted `0 ≤ y ⊥ x ≥ 0`, which with `x ≥ 3` forces
`y = 0`). The box reading `0 ≤ y ⊥ (x − 3) ≥ 0` built by hand → **`optimal
2.9e-19` at `(3, 1)`**. A feasible point of the declared relation is cut by the
current import — a CLAUDE.md §1 defect on an accepted input, fixed by §4 commit 5.

**NCP bound-neutral baseline (measured, this machine; re-measure on yours).**
See §9.3 for the 9-solve panel and its numbers.

### 2.4 Kill criteria

* **H1** is killed if any probe in §6.3/§6.4 (case matrix, closed-form projection
  check, fix-and-solve differential) finds an MCP-feasible point the lowered rows
  cut, or an MCP-infeasible point they admit. None did in §2.3; the §6 tests
  keep the check alive.
* **H2** is killed if the AMPL probe below writes the row differently. Run on the
  owner's machine **before merging commit 5**:

  ```
  # probe.mod
  var x >= 3, <= 10;  var y >= 0, <= 10;
  minimize o: (x-3)^2 + (y-1)^2;
  subject to c: x >= 3 complements y >= 0;
  # ampl:  model probe.mod; option presolve 0; write gprobe;   -> probe.nl (text)
  ```

  Expected (H2 holds): the `r` segment holds `5 2 i` for the row, the `b`
  segment holds `0 3 10` for `x` (the 3 lives in the variable bound), and no
  extra variable is introduced. Then today's importer provably reads this file
  as `x·y = 0` and the fix is a reachable-bug fix. If AMPL instead introduces an
  auxiliary variable or shifts `x`, then `lb ≠ 0` never reaches the importer from
  AMPL and commit 5 is a hardening with the same code; record either outcome in
  the PR (CLAUDE.md §11) and in this file's §2.3.

---

## 3. Design

### 3.1 Where the box lowering lives

One private function in `python/discopt/mpec.py`, called from each of the three
`emit` closures **instead of** `_require_lowerable`:

```python
def _emit_relation(model: Model, p: Complementarity, emit_ncp, *, operand_lifts) -> None:
    """Lower one SCALAR relation: an NCP pair goes straight to ``emit_ncp``; a box
    relation is decomposed into (at most two) NCP halves plus linking rows, and each
    half goes to the same ``emit_ncp``."""
    if p.is_symmetric_nonnegative:
        emit_ncp(p)
        return
    for half in _box_decomposition(model, p, operand_lifts):
        emit_ncp(half)
```

`emit_ncp` is the existing per-method closure body, **unchanged**. The NCP path
therefore emits exactly the rows it emits today (bound-neutral by construction;
verified by §9.3).

### 3.2 The decomposition — case table (normative)

Let `tag = p.name` (always set by `_scalarize_pairs`), `z = p.g`, `F = p.f`,
`(l, u) = _box_form(p)` (§3.5). `fin(·)` = `np.isfinite`.

| case | box rows on `z` | linking rows / aux | NCP halves handed to `emit_ncp` |
|---|---|---|---|
| **B0** `l = u` (fin) | `z ≥ l`, `z ≤ u` (both; they pin `z`) | none | none — `F` is free |
| **B1** `fin(l)`, `fin(u)`, `l < u` | `z ≥ l`, `z ≤ u` | `w⁺ ∈ [0, max(0, hi F)]`, `w⁻ ∈ [0, max(0, −lo F)]`, row `F == w⁺ − w⁻` | `(f=w⁺, g=shift(z,l))`, `(f=w⁻, g=u − z)` |
| **B2** `fin(l)`, `u = +∞` | `z ≥ l` | none | `(f=F, g=shift(z,l))` |
| **B3** `l = −∞`, `fin(u)` | `z ≤ u` | none | `(f=−F, g=u − z)` |
| **B4** `l = −∞`, `u = +∞` | none | row `F == 0` | none |

`shift(z, l)` is `z` itself when `l == 0.0` and `z - l` otherwise — so B2 with
`l = 0` emits **exactly** today's NCP rows (`{tag}_f_nonneg: F ≥ 0`,
`{tag}_g_nonneg: z ≥ 0`, `either_or([[F==0],[z==0]])`), which is what makes the
`.nl` fixture and `Model.mcp(l=0,u=inf)` bit-for-bit unchanged. (`box_mcp`
already maps `l=0,u=inf` to an `NCP_PAIR` with `f_bounds=(0,inf)`, which takes
the `is_symmetric_nonnegative` branch directly; B2-with-`l=0` is reached only by
a hand-built `Complementarity(f_bounds=(-inf,inf), g_bounds=(0,inf))`, and
emits the same rows.)

Row names (all through `model.subject_to(..., name=...)`): `{tag}_z_lb`,
`{tag}_z_ub`, `{tag}_split`, `{tag}_eq`. Half names: `{tag}_lo`, `{tag}_hi`, so
their NCP rows come out as `{tag}_lo_f_nonneg`, `{tag}_lo_g_nonneg`,
`{tag}_lo` (the disjunction / SOS1 set / `_reg` row). Auxiliary variable names
follow `_gdp_operand`: `model._aux_counter += 1; model.continuous(f"_{tag}_wp_{model._aux_counter}", ...)`,
same for `wm`.

The half relations are plain `Complementarity` objects with default
`(0, inf)` bounds, `role=p.role`, `parent=p.parent`, `scale=p.scale`,
`index=p.index`, `source=p.source or p`, `f_shape=g_shape=()`. They are **not**
appended to `model._complementarities` and are **not** `mark_lowered` — they are
lowering internals, not declared relations; the residual report and the solve
guard see only the parent, whose `LoweringRecord.rows` (via `_lower_each`'s
positional slice) covers every row the halves emitted.

Why the box rows on `z` are emitted even when `z` is a `Variable` whose declared
bounds already imply them: `box_mcp` accepts an explicit `lb`/`ub` that may be
*narrower* than the variable's declared box, and the relation's semantics are
"`z ∈ [l, u]`" (semantics §2). Emitting rows rather than tightening the shared
`Variable`'s bounds also keeps the lowering from mutating state other models
hold (semantics §4: a provenance-bearing object is shared). When the declared
box is *narrower* than `[l, u]` the rows are redundant and FBBT drops them.

### 3.3 Why the split, and not a direct three-way disjunction

Considered: `either_or([[z == l, F >= 0], [F == 0], [z == u, F <= 0]])` for GDP
(one fewer aux pair; three selectors instead of two pairs). Rejected for this
slice because (a) it is a second encoding that exists only for GDP — SOS1 and
Scholtes still need the split, so the split is needed anyway and a GDP-only
variant doubles the surface to verify; (b) it does not reuse `emit_ncp`, so the
bound-neutral-by-construction argument for the NCP path would have to be made
twice; (c) on nonlinear `F` it still needs the `_gdp_operand` lift, so the aux
saving is one variable, not two. If a measurement later shows the three-way form
wins on node count for linear `F`, it can be added behind the §5 regime for
bound-changing work; this plan records it as a **proposed, unmeasured**
optimization, not a task.

The split's only degeneracy is B0 (`l = u`), where `w±` would be unconstrained;
B0 therefore emits no halves at all.

### 3.4 Scholtes analog

`reformulate_scholtes` gets the same `_emit_relation`; its `emit_ncp` emits
`f ≥ 0`, `g ≥ 0`, `f·g ≤ t` per half, i.e. `w⁺·(z−l) ≤ t`, `w⁻·(u−z) ≤ t`
(B1), `F·(z−l) ≤ t` (B2), `(−F)·(u−z) ≤ t` (B3), and B4's `F == 0` is already
exact. `solve_mpec(method="scholtes")` then works unchanged: `register_relations`
records the parent, `_report_for` measures the **natural map** on the source
operands (`_resolve_kind` picks it from the declared bounds), the result is
`local_*` with no bound. `RELAXING_METHODS` and `relaxed_relations` need no
change: the parent is marked `"scholtes"`, so a later `Model.solve` is still
refused as a relaxation. Measured in §2.3 probe S.

### 3.5 Refusals (all `ValueError`, all at **lowering** time, all naming `p.describe()`)

Implemented in `_box_form(p) -> tuple[float, float]` and in `_box_decomposition`:

1. **Declared residual bounds not expressible.** A box relation declares `F`
   free: `f_bounds == (−inf, inf)`. Any other `f_bounds` on a relation that is not
   the symmetric pair (`Complementarity` is public, so `(1.0, inf)` is
   constructible) is refused: *"declares residual bounds (lo, hi); a box-MCP
   relation declares F free (−inf, +inf) and the symmetric pair declares (0,
   +inf); no other form is lowered."*
2. **Empty box** `l > u` — already refused by `box_mcp`; re-checked here because
   a direct `Complementarity(g_bounds=(3, 1))` bypasses `box_mcp`.
3. **Non-finite enclosure of `F` in case B1.** `w±` need finite upper bounds;
   `evaluate_interval(F, model)` non-finite (or raising) ⇒ *"cannot bound the
   residual F of {describe} over the declared box (enclosure [lo, hi]); the split
   auxiliaries w⁺/w⁻ need a finite bound, and a fabricated big-M could cut the
   optimum. Declare finite bounds on the variables F reads."* Include the
   evaluator's exception text when it raised (no swallowed cause, CLAUDE.md §7).
   B2/B3 need no new check: a nonlinear `F` is lifted by `_gdp_operand` /
   `_aux_upper_bound` exactly as for an NCP pair today, and an unbounded lift is
   refused by the existing GDP-1 (`_unbounded_big_m_error`) / SOS1 (`"Add a
   finite upper bound"`) messages at solve time.
4. **Array `lb`/`ub`** in `box_mcp` (today a bare `float()` `TypeError`): check
   `np.ndim(lb) == 0 and np.ndim(ub) == 0` and say *"lb/ub are scalars; a vector
   relation uses one box for every element — declare per-element relations for
   per-element boxes."*

No new exception class. No `DISCOPT_*` flag (§5 justification).

### 3.6 Vector relations

`_scalarize_pairs` already expands a vector relation into scalar elements that
inherit `g_bounds`; `_emit_relation` runs per element, so `m.mcp(F_vec, q_vec,
lb=0.0, ub=2.0)` is `n` box relations with the same box. Per-element boxes are
declared per element (`m.mcp(F[i], q[i], lb=0, ub=cap[i])`); the notebook example
(§8.2) does exactly that.

### 3.7 Interactions (each has a test in §6)

| component | effect | test |
|---|---|---|
| `_lower_each` / `LoweringRecord.rows` | halves emit inside the parent's `start:end` window, so `rows_in(model)` = box rows + split row + half rows + disjunction/SOS | `test_box_rows_are_tracked_on_the_parent` |
| `generated_rows` → `lowered_row_residual` | complete, since halves are not separate relations | `test_box_residual_report_*` |
| `carry_complementarities` | unchanged; parent forwarded with method mark | updated `test_box_mcp_round_trips_through_a_rebuilding_pass` |
| `_operand_lifts` (warm start, `solve_mpec` l.1616–1654) | B1 appends `(w⁺, maximum(F, 0.0))`, `(w⁻, maximum(−F, 0.0))` — `evaluate_at_point` supports `max` (`interval_eval.py` l.698); `_gdp_operand` lifts of nonlinear `F` in B2/B3 are appended as today | `test_box_warm_start_lifts_the_split_auxiliaries` |
| `tighten_complementarity_bounds` | unchanged; still `continue`s on `not is_symmetric_nonnegative` (sound no-op). Update its comment to say "(generalization: §10.3 of the plan)". | existing `test_complementarity_bound_tightening_skips_box_relations` |
| `require_all_relations_lowered` | message text updated (no longer "represented but not lowered"); behaviour unchanged | updated `test_solve_refuses_a_model_carrying_an_unlowered_relation` |
| `Model.__deepcopy__` / `__getstate__` / `clone` / `serialize` | no change needed — relations and `LoweringRecord`s are already copied/encoded with `g_bounds` | `test_box_lowered_model_survives_clone_deepcopy_pickle_save_load` |
| `transformations.check("mpec.gdp", m, pairs=[rel])` | works on the copy as for NCP | `test_box_mcp_check_on_copy` |
| `from_nl` | every type-5 row through `Model.mcp` | §6.5 |

---

## 4. Implementation — commits in order

Branch: `git checkout -b box-mcp-lowering` from `main`. One PR (§11).

### Commit 1 — `mpec.py`: the decomposition and the three emitters

1. Delete `_require_lowerable` (l.1089–1112).
2. Add, right after `_gdp_operand` (l.1263):

```python
def _box_form(p: Complementarity) -> tuple[float, float]:
    """Validated (l, u) of a relation that is not the symmetric pair (§3.5 items 1–2)."""
    if p.f_bounds != (-_INF, _INF):
        raise ValueError(...)            # §3.5 item 1 text
    lo, hi = (float(p.g_bounds[0]), float(p.g_bounds[1]))
    if not lo <= hi:
        raise ValueError(f"{p.describe()}: empty box [{lo:g}, {hi:g}]")
    return lo, hi


def _enclosure(model: Model, expr: Expression, p: Complementarity) -> tuple[float, float]:
    """Finite interval enclosure of ``expr`` or a loud ValueError (§3.5 item 3)."""
    from discopt._relax.convexity.interval_eval import evaluate_interval
    try:
        iv = evaluate_interval(expr, model)
        lo = float(np.min(np.asarray(iv.lo))); hi = float(np.max(np.asarray(iv.hi)))
        why = None
    except Exception as exc:  # re-raised below with its text; never swallowed
        lo, hi, why = -_INF, _INF, f"{type(exc).__name__}: {exc}"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(... f"enclosure [{lo:g}, {hi:g}]" + (f" ({why})" if why else "") ...)
    return lo, hi


def _half(parent: Complementarity, f: Expression, g: Expression, name: str) -> Complementarity:
    return Complementarity(f, g, name, role=parent.role, scale=parent.scale,
                           parent=parent.parent, f_shape=(), g_shape=(),
                           index=parent.index,
                           source=parent.source if parent.source is not None else parent)


def _shift(z: Expression, l: float) -> Expression:
    return z if l == 0.0 else z - l


def _box_decomposition(model, p, operand_lifts) -> list[Complementarity]:
    """Emit the box rows / split of §3.2 and return the NCP halves to lower."""
    from discopt.modeling.core import maximum
    tag = p.name or "compl"
    l, u = _box_form(p)
    z, F = p.g, p.f
    if np.isfinite(l): model.subject_to(z >= l, name=f"{tag}_z_lb")
    if np.isfinite(u): model.subject_to(z <= u, name=f"{tag}_z_ub")
    if l == u:                                   # B0
        return []
    if np.isfinite(l) and np.isfinite(u):        # B1
        lo, hi = _enclosure(model, F, p)
        model._aux_counter += 1
        wp = model.continuous(f"_{tag}_wp_{model._aux_counter}", lb=0.0, ub=max(0.0, hi))
        model._aux_counter += 1
        wm = model.continuous(f"_{tag}_wm_{model._aux_counter}", lb=0.0, ub=max(0.0, -lo))
        model.subject_to(F == wp - wm, name=f"{tag}_split")
        if operand_lifts is not None:
            operand_lifts.append((wp, maximum(F, 0.0)))
            operand_lifts.append((wm, maximum(-F, 0.0)))
        return [_half(p, wp, _shift(z, l), f"{tag}_lo"), _half(p, wm, u - z, f"{tag}_hi")]
    if np.isfinite(l):                           # B2
        return [_half(p, F, _shift(z, l), f"{tag}_lo")]
    if np.isfinite(u):                           # B3
        return [_half(p, -F, u - z, f"{tag}_hi")]
    model.subject_to(F == 0, name=f"{tag}_eq")   # B4
    return []


def _emit_relation(model, p, emit_ncp, *, operand_lifts=None) -> None:
    if p.is_symmetric_nonnegative:
        emit_ncp(p)
    else:
        for half in _box_decomposition(model, p, operand_lifts):
            emit_ncp(half)
```

   Notes for the implementer: `z >= l` with a Python float goes through
   `Expression.__ge__`, which folds the constant into the body (normalized `rhs
   = 0`), so `_reject_unnormalized_rhs` (#1858) is satisfied; do **not** build
   `Constraint(z, ">=", l)` by hand. `maximum` is `discopt.modeling.core.maximum`
   (l.3242; `FunctionCall` name `"max"`). Scholtes passes `operand_lifts=None`.

3. Rewrite the three reformulations' bodies:

```python
def reformulate_scholtes(model, pairs, t):
    todo = _pending(model, pairs)
    def emit_ncp(p): ... (unchanged three subject_to lines)
    _lower_each(model, todo, "scholtes", lambda p: _emit_relation(model, p, emit_ncp))

def reformulate_sos1(model, pairs, *, _operand_lifts=None):
    todo = _pending(model, pairs)
    def emit_ncp(p): ... (unchanged)
    _lower_each(model, todo, "sos1",
                lambda p: _emit_relation(model, p, emit_ncp, operand_lifts=_operand_lifts))

def reformulate_gdp(model, pairs, *, _operand_lifts=None):
    todo = _pending(model, pairs)
    def emit_ncp(p): ... (unchanged)
    _lower_each(model, todo, "gdp",
                lambda p: _emit_relation(model, p, emit_ncp, operand_lifts=_operand_lifts))
```

4. Docstrings: module header (l.1–34) — add the box form and the split to the
   bullet list; `box_mcp` (l.454–475) — replace the "This slice **represents** the
   relation; it does not lower it" paragraph with the case table reference;
   `Complementarity` docstring (l.161–170) — unchanged semantics, add "lowered
   by decomposition into NCP halves, see `_box_decomposition`";
   `require_all_relations_lowered` message (l.888–895) → *"… with no lowering:
   …. Lower it with Model.complementarity / Model.mcp (which lower on
   declaration), or reformulate_gdp / reformulate_sos1 / solve_mpec on the pair;
   a relation handed to a model without lowering would be solved as if absent."*
   `tighten_complementarity_bounds` comment (l.1311–1316): keep the skip,
   reword "could cut the solution" → "the partner is pinned to a bound of
   `[l, u]`, not to 0; the box generalization is a separate change".

### Commit 2 — `core.py`: `Model.mcp` lowers; `box_mcp` scalar check

1. `Model.mcp` (l.8779): add `method: str = "gdp"` keyword (same position and
   validation as `Model.complementarity`: `"gdp"`/`"sos1"` accepted, `"scholtes"`
   → the same `ValueError` text pointing at `solve_mpec`, else unknown-method
   `ValueError`). Body becomes:

```python
pair = mpec.box_mcp(_wrap(residual), _wrap(variable), lb=lb, ub=ub, name=name, scale=scale)
_get_transformation(f"mpec.{method}").apply(self, pairs=[pair])
self._complementarities.append(pair)
return pair
```

   Docstring: delete the "This slice **represents** … refused by solve" paragraph;
   add the case table in prose, the `method` parameter, the per-element-box note,
   and the example of §8.1.
2. `mpec.box_mcp` (l.444): add the scalar check of §3.5 item 4 before
   `_normalize_bound`.

### Commit 3 — tests for commits 1–2 (§6.2–6.4, §6.7–6.9) + deliberate updates (§6.1)

### Commit 4 — residual / warm-start / copy / rebuild interaction tests (§6.7–6.9)

(Separate commit only so the diff is reviewable; no source change.)

### Commit 5 — `.nl` importer: every type-5 row through `Model.mcp`

Replace the body of `_add_nl_complementarity` (l.14246–14301):

```python
def _add_nl_complementarity(m, body, var_index, flag, *, index=0):
    """Lower one .nl complementarity row ``5 k i`` as ``body ⊥ x_i`` on the box the
    flag selects from x_i's declared bounds (H2 of docs/dev/box-mcp-lowering-plan.md):
    k=3 -> [lb, ub]; k=2 -> [lb, +inf); k=1 -> (-inf, ub]; k=0 -> (-inf, +inf) (body == 0).
    x_i's declared bounds stay ordinary bounds. A flag whose selected bound is not
    finite is an inconsistent encoding and is refused."""
    from discopt.mpec import _normalize_bound
    x = m._variables[var_index]
    xlb = _normalize_bound(float(np.min(np.asarray(x.lb))))
    xub = _normalize_bound(float(np.max(np.asarray(x.ub))))
    inf_lb, inf_ub = bool(flag & 1), bool(flag & 2)
    lo = -np.inf if inf_lb else xlb        # body may be <= 0 only when z can sit at u ... see table
    ...
```

   Concretely (write it as a 4-way table, not bit arithmetic, so the reader can
   check it against §2.1 H2):

   | flag | body bounds | box passed to `Model.mcp` | refuse when |
   |---|---|---|---|
   | 3 | free | `(xlb, xub)` | `xlb` or `xub` infinite |
   | 2 | `[0, +∞)` | `(xlb, +inf)` | `xlb` infinite |
   | 1 | `(−∞, 0]` | `(-inf, xub)` | `xub` infinite |
   | 0 | `[0, 0]` | `(-inf, +inf)` | never |

   then `m.mcp(body, x, lb=lo, ub=hi, name=f"nl_compl{index}")` (default
   `method="gdp"`, as today). Refusal message: *"from_nl: complementarity row on
   variable {x.name!r} has flag {flag} (body bounds …) but the variable bound that
   flag selects is infinite (x in [xlb, xub]); the encoding is inconsistent, so
   the row is refused rather than guessed."* Keep the `ValueError` type (the
   existing test matches on it).

   Bound-neutrality on the fixture: flag 2, `y ∈ [0, 10]` → `Model.mcp(x, y, lb=0,
   ub=inf)` → `box_mcp` returns `NCP_PAIR` → `emit_ncp` → rows `nl_compl0_f_nonneg`,
   `nl_compl0_g_nonneg`, disjunction `nl_compl0` — identical to today's
   `m.complementarity(a, b, method="gdp", name="nl_compl0")`. `from_nl`'s "suppress
   `_source_nl_path` when `compl_pairs`" logic (l.14128) is unchanged.

   Update the module comment at l.14094–14098 and the docstring's "#658 / #231"
   references.

### Commit 6 — docs: semantics §2, notebook, bib, CHANGELOG (§7–§8)

### Commit 7 — (optional, separate PR if it grows) `tighten_complementarity_bounds` generalization, §10.3. **Not** in this PR by default.

---

## 5. Flag? No. (CLAUDE.md §5 justification)

* On every model that **solves today**, the change is **bound-neutral**: NCP
  relations go through the unchanged `emit_ncp`; verified by the §9.3 panel
  (exact `node_count` / `objective`).
* On models that are **refused today** (any general box), there is no baseline
  to be neutral against; the new behaviour is an exact encoding (H1, proved and
  measured), not a relaxation or a cut, so the bound-changing regime does not
  apply.
* The one behaviour change on an **accepted** input — the `.nl` importer with a
  complementary variable whose bound is non-zero — is a correctness fix (the
  current rows cut a feasible point, §2.3 H). CLAUDE.md §1: correctness fixes do
  not ship behind opt-in flags; a flag would keep a false certificate reachable
  by default.

So no `DISCOPT_*` gate is introduced and no row is owed to
`docs/dev/flag-retirement-audit.md`.

---

## 6. Test plan

New file: `python/tests/test_box_mcp_lowering.py` (fast lane; module marker
`pytestmark = pytest.mark.smoke`; Scholtes tests additionally guarded by
`pytest.importorskip("pounce")` at module top, as `test_mpec_source_residuals.py`
l.57 does). Every solve uses `time_limit=30`. All probes of §2.3 solved in
well under a second each; the whole file should stay under 15 s per test and
~10 s total (if a test approaches 5 s, CI prints it — split it or mark `slow`).

Helper at the top of the file (used by every solve test — this is the
"executed-assertion" discipline of CLAUDE.md §6):

```python
def _natural_map(F, z, l, u): return abs(z - min(max(z - F, l), u))

def _assert_box_optimum(m, rel, *, objective, point, F_at, tol=1e-6):
    r = m.solve(time_limit=30)
    assert r.status == "optimal", r.status
    assert r.gap_certified
    assert r.objective == pytest.approx(objective, abs=tol)
    for name, v in point.items():
        assert float(r.x[name]) == pytest.approx(v, abs=1e-5), (name, r.x)
    # the SOURCE relation holds at the incumbent (natural map), not just the rows
    from discopt.mpec_report import point_from_flat, relation_residuals
    from discopt.solver import _flat_point  # or rebuild the flat vector from r.x in model order
    rows = relation_residuals(m, [rel], point_from_flat(m, _flat(m, r)))
    assert rows and max(rw.complementarity.value for rw in rows) < tol
    return r
```

(`_flat(m, r)`: concatenate `np.atleast_1d(r.x[v.name]).ravel()` over
`m._variables`, as `mpec._result_point` l.1461–1468 does — call that private
helper directly rather than re-implementing it: `from discopt.mpec import
_result_point`.)

### 6.1 Deliberate updates to existing tests

| test | change |
|---|---|
| `test_complementarity_provenance.py::test_solve_refuses_a_model_carrying_an_unlowered_relation` | Build the unlowered relation with `box_mcp(...)` + `register_relations(m, [rel])` (not `m.mcp`, which now lowers). Keep `pytest.raises(NotImplementedError)` and `"mcp0" in str(exc.value)`; add `assert "no lowering" in str(exc.value)` and `assert "#1147" not in str(exc.value)`. |
| `::test_box_mcp_is_refused_by_every_lowering_rather_than_mis_encoded` | Rename to `test_box_mcp_is_lowered_by_every_lowering`. Same model (`z ∈ [−2, 5]`, `min z`, `box_mcp(z, z, lb=-2, ub=5)` i.e. `F = z`). For gdp and sos1 on fresh models: `lower(m, [rel])`, assert `rel.is_lowered_into(m)`, `rel.lowering_in(m) == method`, rows named `boxed_z_lb`, `boxed_z_ub`, `boxed_split` present, two aux `_boxed_wp_*`, `_boxed_wm_*`; solve → feasible set: `z=−2 ⇒ F=−2 ≥ 0` ✗; interior `F=0 ⇒ z=0` ✓; `z=5 ⇒ F=5 ≤ 0` ✗ → **unique z = 0, objective 0.0**. For Scholtes: `reformulate_scholtes(m, [rel], 0.1)` emits rows and marks `"scholtes"`. |
| `::test_lowering_is_gated_on_declared_bounds_not_on_role` | Keep the mislabelled relation (`role=NCP_PAIR`, `g_bounds=(−1, 1)`, `max z + w`, `w ∈ [0,1]`, `F = z + w`). Now assert it is lowered **as a box** (bounds decide): gdp solve → `z=1 ⇒ F ≤ 0 ⇒ 1+w ≤ 0` ✗; interior `F=0 ⇒ w=−z`, obj `0`; `z=−1 ⇒ F ≥ 0 ⇒ w ≥ 1 ⇒ w=1`, obj `0` → **objective 0.0**. An NCP mis-encoding (`z ≥ 0` emitted) would give `z=0, w=1`, objective `1.0` — assert `objective == approx(0.0)` and `< 0.5`. Keep the role/`is_box_mcp` asserts. |
| `::test_box_mcp_round_trips_through_a_rebuilding_pass` | Add after carry: `assert rel.is_lowered_into(m) and rel.lowering_in(m) == "gdp"`, `assert carried.lowering_in(out) == "gdp"`, `assert carried.rows_in(out) is None` (untracked after a pass, per `carry_complementarities`). Fix the docstring. Note the parametrized `(1.5, 1.5)` case: `z` declared on `[max(lb,−50), min(ub,50)] = [1.5, 1.5]` — fine, B0 emits only the two box rows. |
| `test_nl_complementarity.py::test_unsupported_complementarity_config_refuses_loudly` | Replace by `test_free_body_row_lowers_as_a_box` (§6.5 N2) and `test_inconsistent_flag_is_refused` (§6.5 N4). |

Everything else in `test_complementarity_provenance.py`, `test_mpec*.py`,
`test_transformations.py`, `test_1814_*`, `test_1854_*` is NCP-only and must
pass unchanged (§9.1).

### 6.2 Case matrix — `test_box_mcp_lowering.py`

Parametrize over `method ∈ {"gdp", "sos1"}` unless stated. Expected values are
from §2.3 (measured) or from the hand enumeration given in each row.

| test | model | expected |
|---|---|---|
| `test_guard_probe_z_at_lower_bound[method]` | `z∈[−1,1]`, `max z`, `m.mcp(z+1, z, method=…)` | obj `−1.0`, `z=−1` (A) |
| `test_three_point_feasible_set[method]` | `z∈[−1,1]`, `min (z−0.8)²`, `m.mcp(-z, z)` | obj `0.04`, `z=1` (B); also with objective `min (z+0.9)²` → `z=−1`, obj `0.01`; `min (z−0.1)²` → `z=0`, obj `0.01` |
| `test_nonlinear_residual[sense]` | `z∈[−1,2]`, `m.mcp(z*z-1, z)`; `max z` / `min z` | `1.0` / `−1.0` (C) |
| `test_z_is_an_expression` | `x,y∈[0,1]`, `min x+3y`, `m.mcp(x-y, x+y, lb=0.5, ub=1.5)` | `0.5` at `(0.5, 0)` (D) |
| `test_one_sided_upper_infinite[sense]` | `z∈[1,50]`, `m.mcp(3-z, z, lb=1, ub=np.inf)`; `max`/`min` | `3.0` / `1.0` (E); assert `rel.role is BOX_MCP`, no `_wp` aux created (`len(m._variables) == 1`) |
| `test_one_sided_lower_infinite[sense]` | `z∈[−50,2]`, `m.mcp(z-1, z, lb=-np.inf, ub=2)` | `1.0` both (F); no aux |
| `test_both_infinite_is_an_equation` | `z∈[−50,50]`, `max z`, `m.mcp(z-4, z, lb=-np.inf, ub=np.inf)` | `4.0`; rows == `[boxed_eq]`; `rel.rows_in(m)` has exactly 1 row |
| `test_fixed_box_leaves_F_free` | `z∈[−5,5]`, `min z - 7` … use `x∈[0,10]`, `min x`, `m.mcp(x - 7, z, lb=2, ub=2)` | `x=0`, obj `0.0`; `rel.rows_in(m)` == the two box rows only; `z=2` |
| `test_declared_bounds_narrower_than_box` | `z∈[0,1]` declared, `m.mcp(z - 0.5, z, lb=-1, ub=2)`, `max z` | neither bound reachable ⇒ `F=0` ⇒ `z=0.5`, obj `0.5` |
| `test_constant_residual_pins_z_to_l` | `z∈[−1,1]`, `max z`, `m.mcp(2.0, z)` | `F=2>0 ⇒ z=l=−1`; obj `−1` |
| `test_negative_constant_residual_pins_z_to_u` | same with `m.mcp(-2.0, z)`, `min z` | `z=1`, obj `1` |
| `test_vector_relation_same_box[method]` | Cournot `a=10,b=1,c=1`, `q∈[0,2]²`, `m.mcp(F(q), q, lb=0.0, ub=2.0)`, `max sum q` | unique eq. `(2,2)` (hand check in Appendix C) → `4.0`; `len(rel.elements(m)) == 2`; rows named `c_0_lo…`, `c_1_hi…` |
| `test_per_element_boxes_cournot` | caps `(2, 10)`, two scalar `m.mcp` calls | `(2, 3.5)`, obj `5.5` (G) |
| `test_price_cap_market` | §8.1 example, `pmax=3` → `p=3`, obj `49`; `pmax=5` → `p=4`, obj `16` | as stated |
| `test_scholtes_on_a_box_is_local` | probe B via `solve_mpec(m, [box_mcp(-z, z, lb=-1, ub=1)], method="scholtes", t_min=1e-8)` | `status in {"local_optimal","local_limit"}`, `bound is None`, `gap_certified is False`, `objective ≈ 0.04 (abs 1e-4)`, `mpec_report.complementarity.definition == ComplementarityKind.formula(NATURAL_MAP)`, `.value < 1e-4`; then `m.solve()` raises `NotImplementedError` with "RELAXATION" (relaxed guard still bites) |
| `test_gdp_and_sos1_agree_on_random_linear_boxes` | 20 seeded random `(l,u,a,b)`: `z∈[l,u]`, `F = a z + b`, objective `min (z−c)²` | gdp obj == sos1 obj (abs 1e-6) and both == brute force over `{l, u, −b/a if interior}` |

### 6.3 Closed-form projection check (exactness of the decomposition, unit)

`test_box_decomposition_projection_matches_natural_map`: for `(l,u) ∈ {(−1,1),
(0,2), (−3,0), (0.5,0.5)}` and a 401-point grid of `z ∈ [l−0.5, u+0.5]` ×
`F ∈ {−2, −0.5, 0, 0.5, 2}`, compute (a) MCP membership by natural map with
`tol=1e-12`, (b) admission by the B0/B1 rows via the exhaustive case split
(`w⁺ = 0 ∨ z = l`) × (`w⁻ = 0 ∨ z = u`) in closed form:

* `(w⁺=0, w⁻=0)`: `F = 0`;
* `(w⁺=0, z=u)`: `z == u` and `w⁻ = −F ∈ [0, max(0,−lo)]`;
* `(z=l, w⁻=0)`: `z == l` and `w⁺ = F ∈ [0, max(0,hi)]`;
* `(z=l, z=u)`: `l == u == z`;

plus `l ≤ z ≤ u`, with `lo = −2, hi = 2`. Assert (a) ⇔ (b) on every grid
point; count comparisons; assert the count equals `4 × 401 × 5 = 8020`.
Pure Python, `unit` marker, < 0.1 s. This is the executable form of H1's proof.

### 6.4 Fix-and-solve differential (`slow`)

New file `python/tests/test_box_mcp_differential.py`, `pytestmark =
[pytest.mark.slow, pytest.mark.correctness]`. For each of 6 shapes (A, B, C, D,
E, G of §2.3 plus `test_declared_bounds_narrower_than_box`) and each method:

1. Build the declared model, lower with `m.mcp`.
2. Sample 40 points of the **source** variables: the 2–3 boundary points
   (`z=l`, `z=u`, interior roots of `F`) plus seeded uniform points in the
   declared box; decide MCP membership by natural map (`tol=1e-9`) on the source
   relation and ordinary source rows.
3. For each sample, `c = m.clone()`; fix every source variable to the sample
   (`v.lb = v.ub = value`) and `c.solve(time_limit=10)`.
   * MCP-feasible sample ⇒ `status == "optimal"` (**no feasible point cut** —
     the lowered rows, big-M values and presolve admit it);
   * MCP-infeasible sample (natural map `> 1e-3`) ⇒ `status == "infeasible"`
     (**no infeasible point admitted**).
4. Print and assert `n_feasible_checked ≥ 3` and `n_infeasible_checked ≥ 3` per
   shape (both arms must have fired, CLAUDE.md §6), total comparisons printed.

Budget: ~6 shapes × 2 methods × 40 solves × ≪0.1 s — well under the 300 s
timeout; marked `slow` regardless. (The GDP big-M/hull lowering of the *halves*
is additionally covered by the existing `gdp_differential.py` battery for
NCP-shaped `either_or`s; no change there.)

### 6.5 `.nl` round-trip (`test_nl_complementarity.py`)

Fixtures are hand-written strings in the test module (Appendix A), written with
`tempfile.mkstemp(suffix=".nl")` as the existing test does. Each parses with
`parse_nl_string` first and asserts `n_complementarities == 1`,
`n_constraints == 0` (no fabricated range row — the #658 regression stays
locked), then `dm.from_nl`.

| test | fixture | expected |
|---|---|---|
| N1 `test_existing_fixture_is_bound_neutral` | committed `mpcc_complementarity.nl` | `objective == 1.0`, **`node_count` equals the value recorded in §9.3 before the change** (`3` on the plan author's machine), rows named `nl_compl0_f_nonneg`, `nl_compl0_g_nonneg`; `m._complementarities[0].role is NCP_PAIR` |
| N2 `test_free_body_row_lowers_as_a_box` | A.1: `5 3 2`, `x ∈ [−10,10]`, `y ∈ [0,10]`, `min (x+1)² + (y−10)²` | box `[0,10]` on `y`, `F = x` free: `y=10 ⇒ x ≤ 0 ⇒ x=−1`, **obj `0.0`** at `(−1, 10)`; an NCP reading (`x ≥ 0`) would give `1.0` — assert `< 0.5` too. `role is BOX_MCP`, `g_bounds == (0.0, 10.0)` |
| N3 `test_nonzero_lower_bound_is_the_complementarity_bound` | A.2 (probe H): `5 2 1`, `x ∈ [3,10]`, body `y`, `min (x−3)² + (y−1)²` | **obj `0.0`** at `(3, 1)`; **fails on `main` with `1.0`** (measured) |
| N3b `test_nonzero_upper_bound_is_the_complementarity_bound` | A.3: `5 1 1`, `x ∈ [−10, −3]`, body `−y`… simplest: body `y` with flag 1 means `y ≤ 0`; use `y ∈ [−10, 0]`, `min (x+3)² + (y+1)²` | `x=u=−3 ⇒ F=y ≤ 0`: `y=−1` → obj `0.0`; `main` gives `1.0` |
| N4 `test_inconsistent_flag_is_refused` | A.1 with `b` line for `y` changed to `3` (free) and flag 2 | `ValueError` matching `"inconsistent"`; also flag 3 with `y` free, flag 1 with `y ≥ 0` (`2 0`) |
| N5 `test_equation_row_with_free_variable` | A.4: `5 0 2` with `y` free (`3`), body `x − 4` | `F == 0 ⇒ x = 4`; objective `min y²` → `(4, 0)`, obj `0` |
| N6 `test_nl_import_marks_relation_lowered_and_suppresses_native_path` | any of the above | `rel.lowering_in(m) == "gdp"`, `not hasattr(m, "_source_nl_path")` |

### 6.6 Tests that must FAIL on `main` (fail-before/pass-after)

`test_guard_probe_z_at_lower_bound` (today: `Model.mcp` records, `solve` raises
`NotImplementedError`), every row of §6.2 (same reason), N2 (today `ValueError
unsupported bound configuration`), **N3 / N3b (today solve to `1.0`, a wrong
answer, not an error — the regression that matters most)**, N5 (today refused).
Confirm by running the new file against `main` before commit 1 (`git stash` the
source, keep the tests) and record "N failed, 0 passed" in the PR.

### 6.7 Residual-report consistency

* `test_box_residual_report_from_solve_mpec_gdp`: probe B via
  `solve_mpec(m, [rel], method="gdp")`: `r.mpec_report is not None`,
  `r.mpec_report.complementarity.definition` is the natural-map formula,
  `.value < 1e-6`, `r.mpec_report.lowered_row_residual is not None and .value <
  1e-6`, `r.mpec_report.n_scalar_relations == 1`, `bound_violation.value == 0`.
* `test_box_rows_are_tracked_on_the_parent`: after `m.mcp(-z, z)` (B1),
  `rows = rel.rows_in(m)`; assert `rows is not None`, names ⊇ `{c_z_lb, c_z_ub,
  c_split, c_lo_f_nonneg, c_lo_g_nonneg, c_hi_f_nonneg, c_hi_g_nonneg}` and two
  `_DisjunctiveConstraint`s named `c_lo`, `c_hi`; `generated_rows(m, [rel]) ==
  rows`; `m._complementarities == [rel]` (halves not registered).
* `test_source_residuals_after_plain_solve`: `m.mcp(...)`, `r = m.solve()`,
  `rep = source_residuals(m, r)`: natural map `< 1e-6`.

### 6.8 Warm start

`test_box_warm_start_lifts_the_split_auxiliaries`: probe D; `solve_mpec(m,
[rel], method="gdp", initial_solution={x: 0.5, y: 0.0})` → no `RuntimeWarning`
about an un-liftable auxiliary (`warnings.catch_warnings(record=True)`, assert
none mention "could not lift"); result objective `0.5`. And a direct unit check:
after lowering with `_operand_lifts=lifts`, `lifts` has two entries whose
expressions evaluate at `(x,y)=(0.5,0)` to `w⁺ = 0.5`, `w⁻ = 0` via
`mpec_report.evaluate_at_point`.

### 6.9 Clone / deepcopy / pickle / save-load / rebuilding pass / check-on-copy

* `test_box_lowered_model_survives_clone_deepcopy_pickle_save_load[how]`:
  `how ∈ {clone, deepcopy, pickle, save_load}` (save/load via `Model.save` l.13074
  and `discopt.serialize.load` l.3978 to a `tmp_path` file). Copy must have the
  same `len(_constraints)`, `len(_variables)`, one relation with
  `g_bounds == (−1.0, 1.0)` marked `"gdp"`, and solve to the same objective
  (probe B, `0.04`); the original still solves afterwards (as
  `test_1854_clone_lowered_fast_api.py` does).
* `test_box_mcp_check_on_copy[name]` (`name ∈ {mpec.gdp, mpec.sos1}`), mirroring
  `test_transformations.py::test_mpec_create_using_maps_pairs_onto_the_copy`
  (l.248): `report = dt.check(name, m, pairs=[box_mcp(...)])`, original
  untouched, copy's relation lowered, copy solves to `0.04`.
* Rebuilding pass: the updated provenance test (§6.1) covers
  `expand_integer_products` (`discopt._relax.integer_product_reform`). **Measured**
  on `main`: a hand-built B1 split lowered by GDP passes through it with both half
  relations carried and marked `"gdp"`, so the parent relation will too.

### 6.10 Refusals

* `test_refuses_unbounded_residual_in_two_sided_box`: `z∈[−1,1]`, `y`
  unbounded (`m.continuous("y")`), `m.mcp(y, z)` → `ValueError` matching
  `"cannot bound the residual"`; the model has **no** `_wp` aux afterwards and
  `len(m._complementarities) == 0` (the raise happens before the append — check
  the order in `Model.mcp`; if rows were already appended, document that a
  failed lowering leaves the box rows behind and assert that instead — but the
  cleaner contract is: validate the enclosure **before** emitting any row, so
  move `_enclosure` to the top of the B1 branch).
* `test_refuses_inexpressible_residual_bounds`:
  `Complementarity(z, z, "q", f_bounds=(1.0, np.inf), g_bounds=(-1.0, 1.0))` →
  `ValueError` matching `"declares residual bounds"` from all three lowerings.
* `test_refuses_empty_box_built_directly`: `Complementarity(..., g_bounds=(3.0,
  1.0))` → `ValueError("empty box")`.
* `test_box_mcp_refuses_array_bounds`: `box_mcp(F, q, lb=np.array([0,0]),
  ub=np.array([2,2]))` → `ValueError` matching `"scalars"`.
* Existing `test_box_mcp_rejects_an_empty_box` and
  `test_box_mcp_requires_explicit_bounds_for_a_non_variable_side` unchanged.

---

## 7. Documentation and message updates

1. `docs/complementarity_semantics.md` §2 "Box MCP": replace the paragraph "The
   general box form is **represented and not lowered** in this slice …" and the
   following one with:
   * the decomposition table of §3.2 (cases B0–B4) and the one-line exactness
     argument;
   * "all three lowerings accept it; `Model.mcp(method=…)` lowers on
     declaration; `solve_mpec(method="scholtes")` regularizes each half";
   * keep the paragraph on the solver boundary (`solve_model`, not only
     `Model.solve`) — it is still true for a relation handed to a model without
     lowering;
   * add "The `.nl` type-5 row `5 k i` is read as `F ⊥ x_i` on the box `k`
     selects from `x_i`'s declared bounds (table of commit 5); the body's finite
     bound is always 0 in the format, so a non-zero complementarity bound lives in
     the variable."
   Update the **Status/Issue** header to name the new issue beside #1147/#1148.
2. `python/discopt/mpec.py`: docstrings listed in commit 1; the module's
   `__all__` is unchanged (no new public name).
3. `python/discopt/modeling/core.py`: `Model.mcp` docstring (commit 2),
   `_add_nl_complementarity` docstring (commit 5), the `from_nl` comment block at
   l.14094–14098.
4. `python/discopt/mpec_report.py` l.127–132: the comment "which is the box
   form's …" can stay; no code change.
5. `CHANGELOG.md` → `[Unreleased]` → **Added**: "Exact lowering of box-bounded
   MCP relations (`Model.mcp`, `mpec.box_mcp`) by decomposition into NCP halves
   for GDP, SOS1 and Scholtes (#NNNN)." → **Fixed**: "`from_nl` read a type-5
   complementarity row with a non-zero variable bound as `x·F = 0`; it is
   `(x − l)·F = 0`. Doubly-bounded, free-body and equation rows are now imported
   instead of refused."
6. Grep and update every remaining mention of "not lowered" + "#1147" in
   `python/discopt/` (`mpec.py` l.472–474, l.891–894, l.1096–1097, l.1109–1110;
   `core.py` l.8802–8807) and in `docs/complementarity_semantics.md`.

---

## 8. Examples

### 8.1 `Model.mcp` docstring example — a price-capped market (linear MCP)

```python
>>> # Excess supply F(p) = S(p) − D(p) = (2p − 2) − (10 − p) = 3p − 12 on p ∈ [0, p_max].
>>> # Uncapped equilibrium p = 4. With a price cap p_max = 3 the market clears at
>>> # the cap with excess demand: p = 3, F(3) = −3 ≤ 0 (the "z = u ⇒ F ≤ 0" branch).
>>> m = dm.Model("price_cap")
>>> p = m.continuous("p", lb=0, ub=3)
>>> m.minimize((p - 10) ** 2)          # any objective; the MCP has one solution
>>> rel = m.mcp(3 * p - 12, p)         # box defaults to p's declared [0, 3]
>>> round(m.solve().objective, 6)
49.0
```

(`p_max = 5` gives `p = 4`, objective `16.0` — both used in
`test_price_cap_market`.)

### 8.2 Notebook: new section in `docs/notebooks/complementarity_mpec.ipynb`

Insert after "## Vectorized (elementwise) complementarity" (before the POUNCE
flash section) a section **"## Box-bounded MCP: a capacity-constrained Cournot
oligopoly"**:

* Markdown: the MCP definition `F(z) ⊥ z ∈ [l, u]` with the three-branch table
  {cite:p}`FerrisPang1997`; the NCP as the `[0, ∞)` special case; why a
  capacity-constrained Nash–Cournot game is an MCP (each firm's first-order
  condition is complementary to its output on `[0, cap_i]`)
  {cite:p}`MurphySheraliSoyster1982`; a sentence on PATH as the standard local
  MCP solver {cite:p}`DirkseFerris1995` versus the *global, certified* solve
  here; a sentence on the LCP special case {cite:p}`CottlePangStone1992`.
* Code: `a, b, c = 10.0, 1.0, 1.0; caps = (2.0, 10.0)`; two `m.mcp(c − (a − b·Q)
  + b·q_i, q_i, lb=0, ub=caps[i], name=f"foc{i}")`; `m.maximize(q1 + q2)`;
  print the result and `source_residuals(m, r)` — expected `(2.0, 3.5)`, firm 1
  at its cap with `F₁ = −1.5 ≤ 0`, firm 2 interior with `F₂ = 0`.
* Code: the split, shown explicitly once (`w⁺, w⁻` and the two halves) so the
  reader sees what the lowering emits, and `rel.rows_in(m)` names.
* Code: the price-cap example of §8.1 as a one-liner LCP with `p_max ∈ {3, 5}`.
* Markdown: the decomposition's exactness in two sentences, and the Scholtes
  analog with `solve_mpec(..., method="scholtes")` on the Cournot model, reading
  the natural-map residual from `res.mpec_report` {cite:p}`Scholtes2001`.

`docs/_config.yml` has `execute_notebooks: "off"`, so **execute the notebook
locally and commit the outputs**; then `jupyter-book build docs/` must report
zero warnings (CLAUDE.md "When adding a new notebook" — this is a section, not a
new notebook, so `_toc.yml` is **unchanged**).

### 8.3 BibTeX to add to `docs/references.bib` (none of these keys exist; verified by grep)

```bibtex
@article{FerrisPang1997,
  author    = {Ferris, Michael C. and Pang, Jong-Shi},
  title     = {Engineering and Economic Applications of Complementarity Problems},
  journal   = {SIAM Review},
  volume    = {39},
  number    = {4},
  pages     = {669--713},
  year      = {1997},
  doi       = {10.1137/S0036144595285963},
}

@article{DirkseFerris1995,
  author    = {Dirkse, Steven P. and Ferris, Michael C.},
  title     = {The {PATH} Solver: A Non-Monotone Stabilization Scheme for Mixed Complementarity Problems},
  journal   = {Optimization Methods and Software},
  volume    = {5},
  number    = {2},
  pages     = {123--156},
  year      = {1995},
  doi       = {10.1080/10556789508805606},
}

@book{CottlePangStone1992,
  author    = {Cottle, Richard W. and Pang, Jong-Shi and Stone, Richard E.},
  title     = {The Linear Complementarity Problem},
  publisher = {Academic Press},
  address   = {Boston},
  year      = {1992},
  note      = {Reprinted as SIAM Classics in Applied Mathematics 60, 2009},
}

@article{FerrisFourerGay1999,
  author    = {Ferris, Michael C. and Fourer, Robert and Gay, David M.},
  title     = {Expressing Complementarity Problems in an Algebraic Modeling Language and Communicating Them to Solvers},
  journal   = {SIAM Journal on Optimization},
  volume    = {9},
  number    = {4},
  pages     = {991--1009},
  year      = {1999},
  doi       = {10.1137/S105262349833338X},
}

@article{MurphySheraliSoyster1982,
  author    = {Murphy, Frederic H. and Sherali, Hanif D. and Soyster, Allen L.},
  title     = {A Mathematical Programming Approach for Determining Oligopolistic Market Equilibrium},
  journal   = {Mathematical Programming},
  volume    = {24},
  pages     = {92--106},
  year      = {1982},
  doi       = {10.1007/BF01585096},
}
```

Already present and reused: `LuoPangRalph1996`, `Scholtes2001`,
`FletcherLeyffer2004`. Cite `FerrisFourerGay1999` from the
`_add_nl_complementarity` docstring (plain text) and from semantics §2.

---

## 9. Verification protocol

### 9.1 Environment and load-gate (CLAUDE.md §8–§9)

```bash
maturin develop --release                      # repo root; after any crates/ change (none planned)
python -c "import discopt, discopt._rust; print(discopt.__file__)"   # must be this worktree
python -c "import discopt.mpec as m; assert hasattr(m, '_box_decomposition')"   # marker present (after)
uptime                                          # load average < 1 before any node-count panel
```

For the **baseline** run (before commit 1) assert the marker **absent**:
`python -c "import discopt.mpec as m; assert not hasattr(m, '_box_decomposition')"`.

### 9.2 Commands

```bash
ruff check python/ && ruff format --check python/ && mypy python/discopt/mpec.py python/discopt/modeling/core.py
pytest -q python/tests/test_box_mcp_lowering.py python/tests/test_nl_complementarity.py \
          python/tests/test_complementarity_provenance.py python/tests/test_mpec.py \
          python/tests/test_mpec_source_residuals.py python/tests/test_mpec_mp1_mp2.py \
          python/tests/test_transformations.py python/tests/test_1814_deep_complementarity_operand.py \
          python/tests/test_1854_clone_lowered_fast_api.py python/tests/test_bilevel*.py
pytest -q -m slow python/tests/test_box_mcp_differential.py
pytest -m smoke                                                  # per PR template
pytest -m slow python/tests/test_adversarial_recent_fixes.py     # per PR template
pytest --time-budget 15 python/tests/test_box_mcp_lowering.py    # fast-lane budget
jupyter-book build docs/                                         # zero warnings
```

`cargo test -p discopt-core`: N/A (no Rust change).

### 9.3 Bound-neutral panel (CLAUDE.md §5, bound-neutral regime)

Run Appendix B.2's panel section **before commit 1 and after commit 5** on the
same machine, same build flags, load average < 1; `node_count` and `objective`
must be **identical** (not approximately — any drift means the NCP path was
touched). Measured on the plan author's machine (`677af90`, before):

| model | method | status | objective | node_count |
|---|---|---|---|---|
| toy `min (x−1)²+(y−1)²`, `x ⊥ y`, `[0,10]²` | gdp | optimal | `1.0` | 3 |
| toy | sos1 | optimal | `0.9999999998147175` | 5 |
| vec (3-vector, `x_i ⊥ y_i`) | gdp | optimal | `3.0` | 7 |
| vec | sos1 | optimal | `2.999999999771892` | 15 |
| nonlin `x²−1 ⊥ y` | gdp | optimal | `1.0` | 3 |
| nonlin | sos1 | optimal | `0.9999999999999996` | 3 |
| balanced `max x+y, x==y, x ⊥ y` | gdp | optimal | `-0.0` | 0 |
| balanced | sos1 | optimal | `-0.0` | 0 |
| `python/tests/data/mpcc_complementarity.nl` | gdp (from_nl) | optimal | `1.0` | 3 |

Also treat these existing files as the certifying panel and run them unchanged:
`test_mpec.py`, `test_mpec_source_residuals.py`, `test_mpec_mp1_mp2.py`,
`test_1814_deep_complementarity_operand.py`, `test_complementarity_provenance.py`
(NCP half), `test_transformations.py` (mpec tests), `test_bilevel_*.py` (KKT
pairs through `build_kkt`), `test_nl_complementarity.py::test_from_nl_mpcc_matches_python_build`.

The box models themselves are **bound-changing by definition relative to
"refused"** — there is no old bound; their gate is §6.3/§6.4 exactness plus the
natural-map check on every incumbent (`_assert_box_optimum`).

### 9.4 PR description must contain

* the §9.3 before/after table (numbers), and the statement that every row is
  identical;
* the fail-before list of §6.6 with the actual `main` output (`N failed`);
* the H2 AMPL probe outcome (§2.4), verbatim `r`/`b` lines;
* `pytest -m smoke`, adversarial-suite, `ruff`, `mypy` results as numbers;
* the correctness checklist from `.github/pull_request_template.md` ticked, with
  the sentence "no validation, fallback or guard weakened: `_require_lowerable`
  is replaced by an exact lowering plus the §3.5 refusals; the solver-boundary
  guard is unchanged";
* "Closes #NNNN" (the issue of §11) — and **no** negated closing keyword
  anywhere (CLAUDE.md Workflow).

### 9.5 Corpus check (record the count even when zero)

```bash
grep -rlE '^5 [0-9]+ [0-9]+ *$' ~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl | wc -l
```

Expected 0 (MINLPLib carries no MPECs); if non-zero, add each file to N1–N6's
style of test with its `.solu` oracle.

---

## 10. Risks, open questions, follow-ups

### 10.1 Risks

* **Enclosure quality for `w±`.** A loose but finite `hi(F)` only loosens the
  big-M of the `w⁺ == 0` disjunct; soundness is unaffected (the bound is valid).
  FBBT tightens it at the root. Watch node counts in §6.2's SOS1 rows; if a
  shape needs > 50 nodes, record it — it is a performance note, not a blocker.
* **`l == 0` shift special case.** `_shift` exists only for bit-for-bit
  neutrality of the NCP rows reached through a hand-built
  `(−inf,inf)/(0,inf)` relation; `box_mcp` never produces that object. Keep the
  special case tiny and tested (`test_hand_built_zero_lower_box_emits_ncp_rows`:
  `Complementarity(F, z, f_bounds=(-inf,inf), g_bounds=(0,inf))` lowers to rows
  named exactly as `Model.complementarity` would, by comparing
  `[c.name for c in rows_in]`).
* **`Model.mcp` ordering on refusal.** Emit nothing before `_enclosure` succeeds
  (§6.10) so a refused declaration leaves the model untouched.
* **Scholtes on B1 has two bilinear rows per relation**; convergence behaviour of
  the homotopy is the subsolver's business and the result is local anyway.

### 10.2 Open questions (answer in the PR, not by guessing)

1. H2's AMPL byte check (§2.4) — required before merging commit 5.
2. Does the owner want `Model.mcp` to **also** accept `method="scholtes"` by
   delegating to `solve_mpec`? Plan says no (mirror `Model.complementarity`, which
   refuses it with a pointer); keep the two front-ends symmetric.
3. Should `box_mcp` accept `lb`/`ub` arrays for vector relations (per-element
   boxes in one call)? Out of scope here (§1.3); the refusal message points at
   per-element declarations. If wanted later, `g_bounds` becomes
   `tuple[ndarray, ndarray]` and `_element_of` indexes it — a representation
   change that touches `serialize.py` and the residual report.

### 10.3 Follow-up (separate issue): box-aware `tighten_complementarity_bounds`

For a box relation with scalar `Variable`s `z` and `F` (sound, exact
implications; same `ValueError`-on-infeasible rule as MP-1):
`F.lb > 0 ⇒ z.ub := min(z.ub, l)` (requires `fin(l)`; `z.lb > l` ⇒ infeasible);
`F.ub < 0 ⇒ z.lb := max(z.lb, u)`; `z.lb > l and z.ub < u ⇒ F.lb := max(F.lb,
0), F.ub := min(F.ub, 0)` (`F.lb > 0 or F.ub < 0` ⇒ infeasible). Needs its own
differential test; not required for exactness, so not in this PR.

---

## 11. Issue and PR

**Issue title:** `mpec: lower the general box-bounded MCP relation exactly (Model.mcp, .nl type-5 rows with non-zero/double bounds)`

**Issue body (summary):** Since #1147 `Model.mcp` represents `F(z) ⊥ z ∈ [l, u]`
but every lowering refuses it and `Model.solve` refuses the model; the `.nl`
importer refuses flags 0/3 and, measured on `677af90`, reads a type-5 row whose
complementary variable has `lb = 3` as `x·F = 0` instead of `(x−3)·F = 0`
(objective 1.0 returned where 0.0 is feasible). Proposal: decompose the box
relation into at most two NCP halves on shifted slacks (`F = w⁺ − w⁻`, `0 ≤ w⁺ ⊥
z−l`, `0 ≤ w⁻ ⊥ u−z`; one-sided boxes need no `w±`; free box is `F = 0`; `l = u`
leaves `F` free) and feed the halves to the existing GDP/SOS1/Scholtes emitters —
exact (proof + measured on 7 probes), bound-neutral on NCP (panel), no flag.
Route every `.nl` type-5 row through `Model.mcp` with the box read from the
variable's bounds per the flag. Plan: `docs/dev/box-mcp-lowering-plan.md`.
Non-goals: PATH backend, GAMS MCP, VIs, per-element vector boxes.

**PR title:** `feat(#NNNN): exact box-MCP lowering by decomposition into NCP halves; .nl type-5 rows read on the variable's box`

**PR body:** per §9.4; ends with the attribution lines from the session's
system reminder.

---

## Appendix A — hand-written `.nl` fixtures

Format reminders (match `python/tests/data/mpcc_complementarity.nl`): header line
2 is `n_var n_con n_obj n_ranges n_eqns`; a type-5 row in `r` is `5 k i` with `i`
the **1-based** complementary variable; `b` lines are `0 lb ub`, `1 ub`, `2 lb`,
`3` (free), `4 v` (fixed); `k1` holds the cumulative Jacobian nonzero count of the
first `n_var − 1` columns; `J0 n` lists the row's linear coefficients. The
committed fixture puts the objective's nonlinear part in `O0` with no `G`
segment, which the parser accepts; keep that layout.

### A.1 — free body, doubly-bounded variable (N2)

**Measured:** `parse_nl_string` → `n_constraints = 0`, `n_complementarities = 1`,
`complementarity_info(0) = (body_id, var_index=1, flag=3)`; the hand-built split
(`wp, wm ∈ [0, 10]`, `x == wp − wm`, halves `(wp, y)`, `(wm, 10 − y)`) solves to
`optimal 1.2e-20` at `(−1.0, 10.0)`.

`min (x+1)² + (y−10)²`, body `x` (flag 3 = free) complements `y ∈ [0, 10]`,
`x ∈ [−10, 10]`. Expected box reading: `0.0` at `(−1, 10)`.

```
g3 1 1 0
 2 1 1 0 0
 0 1
 0 0
 0 2 0
 0 0 0 1
 0 0
 1 0
 0 0
 0 0 0 0 0
O0 0
o0
o5
o0
v0
n1
n2
o5
o1
v1
n10
n2
C0
n0
x2
0 0.5
1 0.5
r
5 3 2
b
0 -10 10
0 0 10
k1
1
J0 1
0 1
```

### A.2 — non-zero lower bound (N3; measured probe H)

`min (x−3)² + (y−1)²`, body `y` (flag 2 = `y ≥ 0`) complements `x ∈ [3, 10]`.
`main`: `1.0`; box reading: `0.0` at `(3, 1)`.

```
g3 1 1 0
 2 1 1 0 0
 0 1
 0 0
 0 2 0
 0 0 0 1
 0 0
 1 0
 0 0
 0 0 0 0 0
O0 0
o0
o5
o1
v0
n3
n2
o5
o1
v1
n1
n2
C0
n0
x2
0 3
1 0.5
r
5 2 1
b
0 3 10
0 0 10
k1
0
J0 1
1 1
```

### A.3 — non-zero upper bound (N3b)

Same as A.2 with: objective `(x+3)² + (y+1)²` (`o0 v0 n3` for `x+3`, `o0 v1 n1`
for `y+1`), `r` line `5 1 1` (body `y ≤ 0`), `b` lines `0 -10 -3` for `x` and
`0 -10 0` for `y`. Expected box reading `0.0` at `(−3, −1)`; `main` `1.0`.

### A.4 — equation row, free variable (N5)

**Measured:** the text below parses to `n_constraints = 0`, `n_complementarities = 1`,
`complementarity_info(0) = (body_id, var_index=1, flag=0)`.

Body `x − 4` (constant part `C0 / n-4`, linear part `J0 1 / 0 1`), `r` line
`5 0 2` (flag 0 = body fixed to 0) complements `y` free (`b` line `3`),
`x ∈ [−10, 10]`, objective `y²`. Expected `x = 4`, `y = 0`, objective `0`.

```
g3 1 1 0
 2 1 1 0 0
 0 1
 0 0
 0 2 0
 0 0 0 1
 0 0
 1 0
 0 0
 0 0 0 0 0
O0 0
o5
v1
n2
C0
n-4
x2
0 0.5
1 0.5
r
5 0 2
b
0 -10 10
3
k1
1
J0 1
0 1
```

### A.5 — inconsistent flag (N4)

A.1 with `y`'s `b` line replaced by `3` (free) while the flag stays 3 → refuse;
A.2 with `x`'s `b` line `3` and flag 2 → refuse; A.3 with `x` `2 0` (`x ≥ 0`,
`ub = +inf`) and flag 1 → refuse.

## Appendix B — entry-experiment scripts (measured; keep as the implementer's panel tools)

B.1 `entry_experiment.py` and B.2 `entry_experiment2.py` were run from the
scratchpad of the planning session; they are reproduced here verbatim so the
numbers in §2.3 and §9.3 can be regenerated. They hand-build the §3 lowering
through today's `Model.complementarity`, so they run on `main` **before** the
change (that is the point: the baseline is measured with the code under test
absent, CLAUDE.md §8).

### B.1

```python
"""Entry experiment: hand-built split lowering vs brute force (prints an executed-check count)."""
import sys, numpy as np, discopt, discopt.modeling.core as dm
from discopt._relax.convexity.interval_eval import evaluate_interval
assert "/python/discopt/" in discopt.__file__.replace("\\", "/")
CHECKS = 0
def check(c, msg):
    global CHECKS; CHECKS += 1
    if not c: raise AssertionError(msg)
    print("  ok:", msg)
def enclosure(m, e):
    iv = evaluate_interval(e, m); return float(np.asarray(iv.lo).reshape(())), float(np.asarray(iv.hi).reshape(()))
def lower_box_split(m, F, z, l, u, tag, method="gdp"):
    if np.isfinite(l): m.subject_to(z - l >= 0, name=f"{tag}_z_lb")
    if np.isfinite(u): m.subject_to(u - z >= 0, name=f"{tag}_z_ub")
    if l == u: return
    if np.isfinite(l) and np.isfinite(u):
        lo, hi = enclosure(m, F); assert np.isfinite(lo) and np.isfinite(hi)
        wp = m.continuous(f"{tag}_wp", lb=0.0, ub=max(0.0, hi)); wm = m.continuous(f"{tag}_wm", lb=0.0, ub=max(0.0, -lo))
        m.subject_to(F - wp + wm == 0, name=f"{tag}_split")
        m.complementarity(wp, z - l, method=method, name=f"{tag}_lo"); m.complementarity(wm, u - z, method=method, name=f"{tag}_hi")
    elif np.isfinite(l): m.complementarity(F, z - l, method=method, name=f"{tag}_lo")
    elif np.isfinite(u): m.complementarity(-F, u - z, method=method, name=f"{tag}_hi")
    else: m.subject_to(F == 0, name=f"{tag}_eq")
def nm(F, z, l, u): return abs(z - min(max(z - F, l), u))
# A
for method in ("gdp", "sos1"):
    m = dm.Model("A"); z = m.continuous("z", lb=-1, ub=1); m.maximize(z); lower_box_split(m, z + 1, z, -1.0, 1.0, "p", method)
    r = m.solve(time_limit=60); check(r.status == "optimal" and abs(r.objective + 1) < 1e-6, f"A/{method} {r.objective}")
    check(abs(float(r.x["z"]) + 1) < 1e-6, f"A/{method} z")
grid = np.linspace(-1, 1, 20001); feas = [v for v in grid if nm(v + 1, v, -1, 1) < 1e-9]; check(feas == [-1.0], "A brute")
# B
feas = sorted({round(v, 6) for v in grid if nm(-v, v, -1, 1) < 1e-9}); check(feas == [-1.0, 0.0, 1.0], "B brute")
for method in ("gdp", "sos1"):
    m = dm.Model("B"); z = m.continuous("z", lb=-1, ub=1); m.minimize((z - 0.8) ** 2); lower_box_split(m, -z, z, -1.0, 1.0, "p", method)
    r = m.solve(time_limit=60); check(r.status == "optimal" and abs(r.objective - 0.04) < 1e-6, f"B/{method} {r.objective} nodes={r.node_count}")
# C
g2 = np.linspace(-1, 2, 30001); feas = sorted({round(v, 6) for v in g2 if nm(v * v - 1, v, -1, 2) < 1e-9}); check(feas == [-1.0, 1.0], "C brute")
for sense, exp in (("max", 1.0), ("min", -1.0)):
    m = dm.Model("C"); z = m.continuous("z", lb=-1, ub=2); (m.maximize if sense == "max" else m.minimize)(z); lower_box_split(m, z * z - 1, z, -1.0, 2.0, "p")
    r = m.solve(time_limit=60); check(r.status == "optimal" and abs(r.objective - exp) < 1e-6, f"C/{sense} {r.objective}")
# D
best = min((xx + 3 * yy, xx, yy) for xx in np.linspace(0, 1, 401) for yy in np.linspace(0, 1, 401) if nm(xx - yy, xx + yy, 0.5, 1.5) < 1e-9)
check(abs(best[0] - 0.5) < 1e-9, f"D brute {best}")
m = dm.Model("D"); x = m.continuous("x", lb=0, ub=1); y = m.continuous("y", lb=0, ub=1); m.minimize(x + 3 * y); lower_box_split(m, x - y, x + y, 0.5, 1.5, "p")
r = m.solve(time_limit=60); check(r.status == "optimal" and abs(r.objective - 0.5) < 1e-6, f"D {r.objective}")
# E / F
for sense, exp in (("max", 3.0), ("min", 1.0)):
    m = dm.Model("E"); z = m.continuous("z", lb=1, ub=50); (m.maximize if sense == "max" else m.minimize)(z); lower_box_split(m, 3 - z, z, 1.0, np.inf, "p")
    r = m.solve(time_limit=60); check(r.status == "optimal" and abs(r.objective - exp) < 1e-6, f"E/{sense} {r.objective}")
for sense in ("max", "min"):
    m = dm.Model("F"); z = m.continuous("z", lb=-50, ub=2); (m.maximize if sense == "max" else m.minimize)(z); lower_box_split(m, z - 1, z, -np.inf, 2.0, "p")
    r = m.solve(time_limit=60); check(r.status == "optimal" and abs(r.objective - 1.0) < 1e-6, f"F/{sense} {r.objective}")
# G
a, b, c, caps = 10.0, 1.0, 1.0, (2.0, 10.0); eq = [(q1, q2) for q1 in np.linspace(0, caps[0], 201) for q2 in np.linspace(0, caps[1], 1001)
      if nm(c - (a - b * (q1 + q2)) + b * q1, q1, 0, caps[0]) < 1e-9 and nm(c - (a - b * (q1 + q2)) + b * q2, q2, 0, caps[1]) < 1e-9]
check(eq == [(2.0, 3.5)], f"G brute {eq}")
m = dm.Model("G"); q1 = m.continuous("q1", lb=0, ub=caps[0]); q2 = m.continuous("q2", lb=0, ub=caps[1]); m.maximize(q1 + q2); Q = q1 + q2
lower_box_split(m, c - (a - b * Q) + b * q1, q1, 0.0, caps[0], "p1"); lower_box_split(m, c - (a - b * Q) + b * q2, q2, 0.0, caps[1], "p2")
r = m.solve(time_limit=60); check(r.status == "optimal" and abs(r.objective - 5.5) < 1e-6, f"G {r.objective} {r.x}")
print("EXECUTED CHECKS:", CHECKS); sys.exit(0 if CHECKS else 1)
```

### B.2 (importer probe H, Scholtes probe S, NCP baseline panel)

```python
import os, sys, tempfile, warnings, numpy as np, discopt, discopt.modeling.core as dm
from discopt._rust import parse_nl_string
from discopt.mpec import complementarity, solve_mpec
warnings.simplefilter("ignore"); CHECKS = 0
def check(c, msg):
    global CHECKS; CHECKS += 1
    if not c: raise AssertionError(msg)
    print("  ok:", msg)
NL_H = open(<Appendix A.2 written to a file>).read()           # the A.2 text verbatim
rep = parse_nl_string(NL_H); check(rep.n_complementarities == 1 and rep.n_constraints == 0, "parsed")
fd, path = tempfile.mkstemp(suffix=".nl"); os.write(fd, NL_H.encode()); os.close(fd)
r = dm.from_nl(path).solve(time_limit=60); print("current importer:", r.status, r.objective, r.x)
m2 = dm.Model("H"); x = m2.continuous("x", lb=3, ub=10); y = m2.continuous("y", lb=0, ub=10)
m2.minimize((x - 3) ** 2 + (y - 1) ** 2); m2.complementarity(y, x - 3, name="nl_compl0")
r2 = m2.solve(time_limit=60); check(abs(r2.objective) < 1e-6, "box reading 0.0"); check(abs(r.objective - 1.0) < 1e-6, "main importer 1.0")
os.unlink(path)
# S
m = dm.Model("S"); z = m.continuous("z", lb=-1, ub=1); wp = m.continuous("wp", lb=0, ub=1); wm = m.continuous("wm", lb=0, ub=1)
m.minimize((z - 0.8) ** 2); m.subject_to(-z - wp + wm == 0, name="split")
rs = solve_mpec(m, [complementarity(wp, z + 1, name="lo"), complementarity(wm, 1 - z, name="hi")], method="scholtes", t_min=1e-8)
print(rs.status, rs.objective, rs.mpec_report.complementarity.value); check(str(rs.status).startswith("local") and rs.bound is None, "local")
# panel
def toy(me):
    m = dm.Model(f"toy_{me}"); x = m.continuous("x", lb=0, ub=10); y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2); m.complementarity(x, y, method=me, name="c"); return m
def vec(me):
    m = dm.Model(f"vec_{me}"); x = m.continuous("x", shape=3, lb=0, ub=10); y = m.continuous("y", shape=3, lb=0, ub=10)
    m.minimize(dm.sum((x - 1) ** 2) + dm.sum((y - 2) ** 2)); m.complementarity(x, y, method=me, name="v"); return m
def nonlin(me):
    m = dm.Model(f"nonlin_{me}"); x = m.continuous("x", lb=0, ub=4); y = m.continuous("y", lb=0, ub=4)
    m.minimize((x - 2) ** 2 + (y - 1) ** 2); m.complementarity(x * x - 1, y, method=me, name="n"); return m
def balanced(me):
    m = dm.Model(f"balanced_{me}"); x = m.continuous("x", lb=0, ub=1); y = m.continuous("y", lb=0, ub=1)
    m.maximize(x + y); m.subject_to(x == y); m.complementarity(x, y, method=me, name="b"); return m
def nl_fixture(_): return dm.from_nl("python/tests/data/mpcc_complementarity.nl")
n = 0
for build in (toy, vec, nonlin, balanced, nl_fixture):
    for me in ("gdp", "sos1"):
        if build is nl_fixture and me == "sos1": continue
        m = build(me); r = m.solve(time_limit=60); n += 1
        print(f"{m.name:22s} {r.status:8s} obj={r.objective!r} nodes={r.node_count}")
check(n == 9, "panel ran 9 solves"); print("EXECUTED CHECKS:", CHECKS); sys.exit(0 if CHECKS else 1)
```

## Appendix C — hand enumeration for the same-box Cournot test (`caps = (2, 2)`)

`F_i = c − (a − b(q₁+q₂)) + b q_i = q₁ + q₂ + q_i − 9`. Both at cap: `F =
(−3, −3) ≤ 0` ✓. One at cap, other interior: `2 q_j + 2 − 9 = 0 ⇒ q_j = 3.5 >
2` ✗. Both interior: `q = 3 > 2` ✗. Any firm at 0 needs `F_i ≥ 0 ⇒ q_j ≥ 9` ✗.
Unique equilibrium `(2, 2)`, `q₁ + q₂ = 4`.
