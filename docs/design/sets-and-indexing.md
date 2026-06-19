# Set & Index Abstractions — Design

**Status:** planned · **Last updated:** 2026-06-19 ·
**Scope:** named index sets, set algebra, indexed (dense + sparse) variables/parameters/
constraints, aggregation over sets, and the performance routing that maps these
abstractions onto the existing flat model representation.
**Bottom line:** this is a pure-Python **desugaring layer** on top of the current
modeling API. By solve time only flat `Variable`/`Constraint` objects exist, so the Rust
core, JAX compiler, `.nl` exporter, B&B, and every phase gate are untouched. Correctness
rests on *equivalence to the existing positional API*.

This realizes the Phase 7 roadmap item:

> **Set and index abstractions** — Named sets, indexed variables/constraints, set algebra
> for sparse models.

---

## 1. Motivation

The modeling API today exposes **dense, NumPy-style `shape=` arrays with positional
integer indexing** (`core.py`):

```python
x = m.continuous("flow", shape=(3, 4), lb=0)   # dense 3x4 tensor
m.subject_to(x[0, 0] + x[1, 2] <= 10)          # integer indices only
for i in range(3):
    for j in range(4):
        m.subject_to(x[i, j] >= 0, name=f"nn_{i}_{j}")   # manual loops
```

This is the equivalent of JuMP's `Array` tier / Pyomo's dense `Var`. It cannot express:

- **Labels** — `ship["pitt", "a"]` instead of `x[0, 0]`.
- **Sparsity** — a variable defined only on a subset of pairs (a sparse transportation
  network) without allocating the full dense tensor and leaving entries unused.
- **Set algebra** — unions/intersections/cross-products/filters that drive which index
  tuples exist.

## 2. SOTA survey (design lessons adopted)

**JuMP** picks the *tightest container that fits*: `Array` (rectangular, `1:n`) →
`DenseAxisArray` (rectangular, arbitrary **labels**, including integer labels like
`x[:h2, 10]`) → `SparseAxisArray` (subset of entries, triggered by a filtering condition
such as `i <= j`). `DenseAxisArray` exists specifically to avoid positional/label
ambiguity; `SparseVariables.jl` extends the sparse tier for very sparse models.

**Pyomo** makes a `Set` the first argument (index) of `Var`/`Constraint`/`Param`; provides
`RangeSet` (now on `NumericRange`, supports unbounded/continuous ranges); exposes **set
algebra** via `|` `&` `-` `*`; and supports **sparse construction** through conditional
rules (`Constraint.Skip`). It documents a footgun: summing over a *sparse* var only covers
indices already materialized.

**GAMS/AMPL** contribute the canonical mental model: declare sets, declare data/params/
vars over sets, write constraints with a "for-all" domain plus a summation domain.

**Lessons adopted:**
1. Keep dense `shape=` as the "Array" tier; add label-indexed and sparse tiers on top
   (JuMP's container ladder).
2. Make the **set the single source of truth** for which tuples exist — avoids Pyomo's
   sparse-sum footgun.
3. Set algebra as operators (`|` `&` `-` `*`) plus filtering (`Set.where(pred)`).
4. Arbitrary **hashable labels**, not just integers — the core expressiveness win.

## 3. Architecture

All new code in `python/discopt/modeling/`:

| File | Change |
|------|--------|
| `sets.py` | **NEW** — `Set`, `RangeSet`, `ProductSet`, `IndexedVar`, `IndexedParam`, `IndexedConstraint`, `Skip` |
| `core.py` | `Model.set()`, `over=` kwarg on `continuous`/`binary`/`integer`/`parameter`, `Model.constraint()`, `subject_to()` accepts generators with key-tuple naming |
| `modeling/__init__.py`, `discopt/__init__.py` | export `Set`, `RangeSet` |

### 3.1 Backing mechanism (the key idea)

An indexed variable is **backed by exactly one flat `Variable`** of shape `(len(index),)`
plus an order-stable `key_tuple -> position` map. `IndexedVar.__getitem__(key)` returns the
existing `IndexExpression(flat_var, position)` — which the JAX compiler and `.nl` exporter
already flatten. **Nothing downstream changes.** Per-key bounds map onto the flat
`Variable.lb/ub` arrays (scalar, dict `{key: bound}`, or callable `key -> bound`).

### 3.2 `Set`

```python
class Set:
    name: str
    members: tuple[Hashable, ...]    # order-stable, de-duplicated
    dimen: int                        # inferred from member arity; override via dimen=
    def __or__/__and__/__sub__(self, other) -> Set     # union / intersection / difference
    def __mul__(self, other) -> ProductSet             # lazy cross product
    def where(self, pred) -> Set                       # filter, e.g. .where(lambda i, j: i < j)
    def with_first(self, k) / with_last(self, k)       # slice helpers for sparse 2D sets
    def __iter__ / __len__ / __contains__
```

- `RangeSet(n)` / `RangeSet(a, b)` — integer convenience subtype (Pyomo parity).
- `ProductSet` is **lazy** (materializes on iteration); `.where` filters a product into a
  sparse subset (JuMP's `SparseAxisArray`-via-condition, Pyomo's filtered set).

### 3.3 Model API

```python
m.set(name, members, dimen=None) -> Set
m.continuous(name, over=SET, lb=, ub=) -> IndexedVar    # over= and shape= are exclusive
m.binary(name, over=SET) -> IndexedVar
m.integer(name, over=SET, lb=, ub=) -> IndexedVar
m.parameter(name, over=SET, value=) -> IndexedParam     # value: scalar | dict | callable
m.constraint(SET, rule=fn, name=) -> IndexedConstraint  # rule may return Skip
m.subject_to((expr(i) for i in SET), name=)             # generator -> key-tuple naming
```

New model state: `self._sets: list[Set]` (named, dedup'd via existing `_check_name`).

### 3.4 Aggregation

`sum(fn, over=...)`/`prod`/`norm` already accept any iterable, so a `Set` works
immediately. Python-builtin-style `sum(ship[p, k] for p, k in LINKS.with_first(p))` also
works via `Expression.__radd__`.

## 4. Orchestration (when used at solve time)

These abstractions are **compile-time sugar**. By `Model.solve()` only flat
`Variable`/`Constraint` objects exist → no solver-orchestration change, no new runtime
decision. That is the safety property.

One **performance** routing decision (JuMP's tightest-container spirit): when an indexed
constraint family is **affine** (DAG contains only `+`/`-`/`*const`/`IndexExpression`/
`Constant`), batch the whole family into `PyModelBuilder.add_linear_constraints(A, x,
sense, b)` as sparse rows instead of thousands of Python `Constraint` objects. Nonlinear/
mixed families use the current per-constraint path. Routing lives in
`_emit_indexed_constraints()`, defaults to `fast=True`, with `fast=False` escape hatch;
both paths are asserted to produce identical flat models.

## 5. Correctness

The phase-gate invariant `incorrect_count <= 0` must hold. Correctness rests on
**desugaring equivalence**.

1. **Equivalence oracle** — build each model twice (sets vs today's `shape=` + manual
   loops); assert identical flat models (var count/bounds/types, normalized constraint
   DAGs / `.nl` serialization, objective). Master check.
2. **Fast-path vs expressive-path** — every indexed-constraint test built with `fast=True`
   and `fast=False`; assert identical flat models and solutions.
3. **Known optima** — port transportation/assignment/multi-commodity-flow into
   `tests/data/known_optima.toml`; mark `@pytest.mark.correctness`.
4. **Round-trip** — `.nl` export/re-import, plus one `from_pyomo` cross-check.
5. **Property tests** — set-algebra laws (`(A|B)-B subset A`, `A&B = B&A`,
   `|A*B| = |A||B|`, `where` idempotent); `|vars| == |distinct keys|`; no phantom/dup
   constraints.

## 6. Tests

New `python/tests/test_sets.py` (conventions from `test_fast_construction.py`):
`TestSetAlgebra`, `TestIndexedVariables`, `TestIndexedConstraints`,
`TestDesugarEquivalence`, `TestFastPathEquivalence`, `TestCorrectness`
(`@pytest.mark.correctness`), `TestExportRoundTrip`. Total coverage ≥65%; `sets.py` ≥90%.
`ruff` (100-col, py310) + `mypy discopt.modeling` clean.

## 7. Documentation

`docs/notebooks/sets_and_indexing.ipynb` (motivation → named sets → algebra → indexed
vars/constraints → sparse vs dense → transportation end-to-end → migration from positional
`shape=`), with `{cite:p}` citations. `docs/references.bib`: Pyomo (Hart et al.), JuMP
(Dunning/Huchette/Lubin SIAM Review 2017; Lubin et al. 2023), AMPL (Fourer/Gay/Kernighan),
GAMS (Bisschop/Meeraus). Add to `docs/_toc.yml`; cross-link from `modeling_guide.ipynb`.
Docstrings render via sphinx-autoapi. `jupyter-book build docs/` with zero warnings.
`CHANGELOG.md` under `[Unreleased] -> Added`.

## 8. Examples

`python/discopt/modeling/examples.py`: `example_transportation()` (sparse network, the
canonical showcase), `example_assignment()` (binary product set with filter),
`example_multicommodity_flow()` (product `*` + `.where`, exercises linear fast-path). Each
doubles as a correctness fixture.

## 9. Milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| M1 | `Set`/`RangeSet`/`ProductSet` + algebra + `m.set()` | `sets.py`, `TestSetAlgebra`, exports |
| M2 | `IndexedVar` (dense+sparse) + `over=` on constructors | indexing, bounds, desugar-equivalence harness |
| M3 | Indexed constraints (generator + rule + `Skip`) + key-tuple naming | `m.constraint`, `TestIndexedConstraints` |
| M4 | `IndexedParam` + aggregation parity | params, agg tests |
| M5 | Linear fast-path routing into `PyModelBuilder` | `_emit_indexed_constraints`, `TestFastPathEquivalence` |
| M6 | Correctness suite + `.nl`/Pyomo round-trip | `TestCorrectness`, known-optima entries |
| M7 | Docs notebook + bib + TOC + examples + CHANGELOG | docs build clean |

## 10. Risks & mitigations

- **Sparse-sum footgun (Pyomo's)** — set is the authoritative index; `sum(... over=SET)`
  always iterates the full declared set.
- **Product-set blowup** — `ProductSet` lazy; `.where` filters before materialization.
- **Label hashability/order** — require hashable members; store order-stable de-duplicated
  tuples; deterministic flattening (matches `.nl` determinism).
- **Fast-path divergence** — M5 ships with `TestFastPathEquivalence`; routing only fires
  when bodies prove affine.
- **mypy noise** — keep `sets.py` strictly typed so it needs no `core.py`-style override.
