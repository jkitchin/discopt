# The Decomposition Advisor — architecture and design specification

**Date:** 2026-07-01
**Status:** design specification — Phases 1–7 implemented (see §20 for status & verification)
**Scope:** A new subsystem, the **Decomposition Advisor**, that analyzes an
optimization model, discovers exploitable structure, scores and recommends
decomposition strategies, explains its reasoning, and — when profitable —
automatically reformulates the model into a coordinator + master + subproblem
system. Designed as a *compiler optimization pass* over the model, on the same
footing as presolve, FBBT/OBBT, and convexity detection.

**Relationship to existing code:** discopt already ships a thin decomposition
layer — `discopt.decomposition.structure.DecompositionStructure` /
`detect_decomposition` (union-find blocks + a guarded bridge-constraint coupling
heuristic), and executable `solve_benders` / `solve_gbd` / `solve_lagrangian`
drivers. This document specifies the *advisor* that sits **above** those
drivers: it generalizes structure detection from one heuristic into a graph
analysis engine, adds scoring/ranking/selection/explanation, and turns the
drivers into pluggable back-ends of a `Reformulation` interface. Wherever
possible the design is additive and backward-compatible with the existing
`DecompositionStructure` contract.

---

## Table of contents

1. System overview and design philosophy
2. Module hierarchy and ownership
3. Core traits / interfaces
4. Graph representations (Task 2)
5. Structure-discovery algorithms (Task 3)
6. Scoring system (Task 4)
7. Method selection (Task 5)
8. Automatic reformulation & the decomposition IR (Task 6)
9. Explainability (Task 7)
10. Learning from experience (Task 8)
11. Interaction with presolve (Task 9)
12. Interaction with global optimization (Task 10)
13. Parallel execution (Task 11)
14. Public API (Task 12)
15. Visualization (Task 13)
16. Research opportunities (Task 14)
17. Implementation roadmap (Task 15)
18. Cross-cutting tradeoffs and challenged assumptions
19. Appendix: pseudocode

---

## 1. System overview and design philosophy

### 1.1 The one-line vision

```python
model.solve()          # advisor runs silently, picks no-decomp / Benders / DW / ...
model.solve(decomposition="auto")   # explicit opt-in, same machinery
```

The user never rewrites the model. The advisor is a transformation pass: given a
model it either returns the model unchanged (with a *reason*), or returns a
`DecomposedModel` that solves as a coordinated system and reports results in the
original variable space.

### 1.2 Central thesis: decomposition is a graph problem

Every decomposition method is, at bottom, a statement about a **graph cut**:

| Method | Graph object it exploits | What is "cut" |
|---|---|---|
| Connected-component / block-diagonal | variable–constraint bipartite graph | nothing (already disconnected) |
| Classical / Generalized Benders | bipartite graph after *removing complicating vars* | a small vertex set (complicating vars) |
| Dantzig–Wolfe / Column generation | block-angular constraint hypergraph | a few *linking rows* |
| Lagrangian / Augmented Lagrangian / ADMM | coupling-constraint graph | a set of *coupling edges* (dualized) |
| Schur complement | KKT sparsity graph | a *border* (bordered block-diagonal) |
| Progressive Hedging | scenario graph | *non-anticipativity* links across scenarios |
| Nested Benders / nested dissection | stage graph / elimination tree | a *separator hierarchy* |

So the advisor's job is: **build the right graphs, find the cheapest structural
cut, and map that cut to a decomposition method.** This reframing is the key
design decision — it makes the advisor extensible (new method = new mapping from
a graph cut to a `Reformulation`) and unifies the whole taxonomy under one
engine.

### 1.3 Pass pipeline placement

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                         Model.solve() pipeline                          │
   │                                                                          │
   │  parse ─► symbolic simplify ─► presolve(FBBT) ─► convexity detect ─►     │
   │                                                                          │
   │      ┌───────────────── Decomposition Advisor ─────────────────┐         │
   │      │  StructureAnalyzer → CandidateGenerator → Scorer →       │         │
   │      │  MethodSelector → Explainer → Reformulator               │         │
   │      └──────────────────────────┬───────────────────────────────┘         │
   │                                 │                                          │
   │      no-decomp ◄────────────────┤────────────────► DecomposedModel        │
   │           │                                              │                 │
   │           ▼                                              ▼                 │
   │   monolithic B&B / global                     Coordinator drives          │
   │                                               master+subproblem solves     │
   │                                                                          │
   │  ◄───────────────── SolveResult (original var space) ◄────────────────    │
   └──────────────────────────────────────────────────────────────────────┘
```

The advisor runs **after** convexity detection (it needs curvature labels to
score subproblems) and **after** a first FBBT sweep (tighter bounds change what
looks separable), but **before** spatial B&B / OA. See §11 for why the ordering
is actually *iterative*, not a straight line.

### 1.4 Design goals, in priority order

1. **Correctness first.** A decomposition that changes the optimal value is a
   bug, full stop. This mirrors the repo's non-negotiable
   `incorrect_count ≤ 0` gate. Every reformulation must carry a *soundness
   certificate* (see §8.3, §14 research) and default to **no-decomp** whenever
   soundness cannot be established (e.g. a nonconvex, non-separable coupling).
2. **Zero-cost when off.** Analysis is opt-in-cheap: connected components and a
   coupling scan are near-linear; anything superlinear (spectral, METIS, tree
   decomposition) is gated behind budgets exactly like today's
   `_BRIDGE_SCAN_BUDGET`.
3. **Explainability is a first-class output**, not an afterthought. Every
   recommendation ships a machine-readable *and* human-readable rationale.
4. **Extensibility.** Adding "Column Generation with stabilization" should be a
   new `DecompositionMethod` plugin + a scoring rule, touching no core graph
   code.
5. **Backward compatibility.** `detect_decomposition()` keeps working; it becomes
   one (cheap) `CandidateGenerator` among several.

---

## 2. Module hierarchy and ownership

The advisor spans **Python orchestration** (policy, explanation, reformulation
assembly) and **Rust core** (the hot graph kernels: partitioning, SCC, nested
dissection, min-cut). The split follows the repo's existing seam — heavy sparse
graph work belongs in `crates/discopt-core`, exposed via PyO3, while the modeling
and policy logic stays in Python where the `Model` DAG lives.

```
python/discopt/decomposition/
│
├── __init__.py                 # public re-exports (unchanged surface + advisor)
├── structure.py                # EXISTING: DecompositionStructure, detect_decomposition
│                               #   → refactored to delegate to graph/ + advisor/
│
├── advisor/
│   ├── __init__.py             # DecompositionAdvisor (façade), analyze_decomposition()
│   ├── analyzer.py             # StructureAnalyzer: model → StructureReport
│   ├── candidates.py           # CandidateGenerator plugins → Iterable[Candidate]
│   ├── scoring.py              # Scorer, ScoreVector, metric registry
│   ├── selection.py            # MethodSelector: decision tree / policy
│   ├── explain.py              # Explainer, Explanation, Rationale, Concern
│   └── policy.py               # pluggable Policy (rules | learned | portfolio)
│
├── graph/                      # graph *views* over a Model (Python side, thin)
│   ├── __init__.py
│   ├── base.py                 # ModelGraph trait, GraphView cache handle
│   ├── incidence.py            # VariableConstraintGraph (bipartite)
│   ├── hypergraph.py           # ConstraintHypergraph
│   ├── jacobian.py             # JacobianSparsityGraph
│   ├── hessian.py              # HessianSparsityGraph
│   ├── kkt.py                  # KKTGraph
│   ├── stage.py                # StageGraph / ScenarioGraph
│   ├── coupling.py             # CouplingGraph, BlockGraph
│   └── elimination.py          # EliminationTree / VariableEliminationGraph
│
├── ir/                         # decomposition intermediate representation
│   ├── __init__.py
│   ├── decomposed_model.py     # DecomposedModel (the reformulation product)
│   ├── master.py               # MasterModel
│   ├── subproblem.py           # SubproblemModel
│   ├── cutgen.py               # CutGenerator trait + Benders/GBD/LBBD/lift-project
│   ├── coordinator.py          # Coordinator trait + driver loop
│   └── comm.py                 # CommunicationLayer trait + backends
│
├── methods/                    # DecompositionMethod plugins (map cut→reformulation)
│   ├── benders.py              # wraps EXISTING solve_benders / solve_gbd
│   ├── lagrangian.py           # wraps EXISTING solve_lagrangian
│   ├── admm.py
│   ├── dantzig_wolfe.py
│   ├── column_generation.py
│   ├── schur.py
│   ├── progressive_hedging.py
│   └── nested.py
│
├── learning/                   # Task 8 — optional, feature-gated
│   ├── record.py               # SolveRecord schema, telemetry sink
│   ├── features.py             # instance feature extraction / embeddings
│   ├── store.py                # performance database (sqlite / parquet)
│   └── policies.py             # bandit / portfolio / GNN-backed Policy
│
└── viz/                        # Task 13
    ├── graph_plot.py           # matplotlib/graphviz static
    ├── interactive.py          # optional (pyvis/plotly) explorer
    └── report.py               # HTML/Jupyter recommendation card

crates/discopt-core/src/decomp/          # Rust hot kernels
├── mod.rs
├── csr.rs                # CSR/CSC sparse incidence, the shared substrate
├── components.rs         # union-find CC + Tarjan SCC (streaming)
├── partition.rs          # recursive bisection, Fiedler/spectral, METIS FFI bridge
├── nested_dissection.rs  # separator trees over KKT/Hessian graph
├── mincut.rs             # Stoer–Wagner, max-flow min s-t cut, articulation/bridges
├── tree_decomp.rs        # chordal completion, min-degree/min-fill, treewidth bound
└── ffi.rs                # PyO3 zero-copy numpy in/out (mirrors crates/discopt-python)
```

### 2.1 Ownership and lifetimes

The governing principle: **the advisor never owns the `Model`; it borrows it and
produces derived, owned artifacts.**

- **`Model`** — owned by the user. The advisor takes an immutable borrow for
  analysis. Rust kernels receive **zero-copy views** of the CSR incidence arrays
  (same pattern as `crates/discopt-python`'s numpy zero-copy), never the model
  itself.
- **`GraphView`** — a cache handle owned by the advisor, keyed by
  `(model_fingerprint, graph_kind)`. Holds Rust-side CSR buffers behind an
  `Arc`; Python holds a lightweight handle. Invalidated by a model-version
  counter (see §4.14 incremental updates). Graphs are **lazily built and
  memoized**: nobody pays for a Hessian graph on a pure-LP.
- **`StructureReport`** — an owned, immutable value (frozen dataclass, like the
  existing `DecompositionStructure`). Cheap to clone, serializable.
- **`Candidate` / `Explanation` / `ScoreVector`** — owned immutable values.
- **`DecomposedModel`** — an owned reformulation. It **holds references back to
  the source `Model`** (for reporting in original variable space) but owns its
  master/subproblem/coordinator objects. Dropping it must not invalidate the
  user's `Model`.
- **`Coordinator`** — owns the solve-time mutable state (incumbent, cut pool,
  multiplier estimates, worker handles). It is the only mutable actor at solve
  time; masters/subproblems are solved through it.

Rust side: CSR buffers are `Arc<CsrGraph>` so multiple graph views (Jacobian,
Hessian sharing the same variable order) can share vertex arrays. Partitioners
take `&CsrGraph` and return owned `Partition { block_of: Vec<u32>, separator:
Vec<u32> }`. No interior mutability in kernels — they are pure functions, which
is what let the existing Rayon MILP driver stay bit-reproducible (see
`design/rayon-parallelization.md`), and we keep that discipline.

### 2.2 Why this split (challenging the "all-Python" alternative)

An all-Python advisor is tempting (fast to write, easy to hack). Rejected for the
hot path because: (a) nested dissection and spectral partitioning on a
100k-constraint KKT graph are genuinely superlinear and must be fast to be usable
inside a solve loop; (b) the repo already commits to Rust for
"LP solving, B&B tree management, .nl parsing, FBBT/presolve" — graph kernels are
the same class of work; (c) zero-copy numpy ↔ Rust is already a solved,
load-bearing pattern here. Conversely, **policy, scoring weights, explanation
text, and reformulation assembly stay in Python** because they change often, are
not hot, and benefit from the modeling DAG being right there.

---

## 3. Core traits / interfaces

These are the seams. Everything else is a plugin behind one of them.

### 3.1 `ModelGraph` (graph view trait)

```python
class ModelGraph(Protocol):
    kind: GraphKind                      # VAR_CONSTRAINT, JACOBIAN, HESSIAN, KKT, ...
    def num_vertices(self) -> int: ...
    def num_edges(self) -> int: ...
    def csr(self) -> CsrHandle: ...       # zero-copy handle into Rust buffers
    def fingerprint(self) -> bytes: ...   # for cache keying / invalidation
    def project(self, remove: VertexSet) -> "ModelGraph": ...  # e.g. drop complicating vars
    def is_stale(self, model_version: int) -> bool: ...
```

Rust mirror: `trait Graph { fn n(&self)->usize; fn neighbors(&self, v:u32)->&[u32]; fn weight(&self,e:Edge)->f64; }`
implemented by `CsrGraph`, and by lightweight *view adaptors* (`ProjectedGraph`,
`WeightedGraph`) that wrap a base without copying.

### 3.2 `CandidateGenerator`

```python
class CandidateGenerator(Protocol):
    name: str
    def applicable(self, report: StructureReport) -> bool: ...
    def generate(self, model: Model, report: StructureReport) -> Iterable[Candidate]: ...
```

Each generator proposes zero or more `Candidate`s. Generators are cheap-first and
budget-gated. Examples: `ConnectedComponentGenerator` (free), `BridgeCoupling`
(the *existing* heuristic, now a generator), `SpectralBisection`,
`NestedDissection`, `StageDetector`, `ScenarioDetector`, `UserAnnotation`.

### 3.3 `Candidate`

```python
@dataclass(frozen=True)
class Candidate:
    method: MethodKind                   # BENDERS, GBD, DANTZIG_WOLFE, ADMM, ...
    partition: Partition                 # blocks + separator/coupling
    structure: DecompositionStructure    # the EXISTING contract, reused verbatim
    provenance: str                      # which generator, which algorithm+params
    est_soundness: Soundness             # PROVEN_EQUIVALENT | RELAXATION | HEURISTIC
```

Crucially, `Candidate.structure` **is** today's `DecompositionStructure`, so the
existing `solve_benders`/`solve_lagrangian` drivers consume candidates with no
adapter.

### 3.4 `Scorer`

```python
class Scorer(Protocol):
    def score(self, model: Model, cand: Candidate, report: StructureReport) -> ScoreVector: ...

@dataclass(frozen=True)
class ScoreVector:
    metrics: Mapping[str, float]         # named metrics (see §6)
    aggregate: float                     # scalarized rank key
    confidence: float                    # [0,1], how sure the estimate is
```

### 3.5 `DecompositionMethod` (the reformulation plugin)

```python
class DecompositionMethod(Protocol):
    kind: MethodKind
    def can_reformulate(self, cand: Candidate, model: Model) -> ReformulationVerdict: ...
    def reformulate(self, cand: Candidate, model: Model) -> DecomposedModel: ...
    def cut_generator(self) -> CutGenerator | None: ...
    def coordinator(self) -> Coordinator: ...
```

### 3.6 `Policy` (selection strategy, swappable)

```python
class Policy(Protocol):
    def rank(self, scored: list[tuple[Candidate, ScoreVector]],
             ctx: SelectionContext) -> list[Ranked]: ...
```

The default `RuleBasedPolicy` implements the §7 decision tree. `LearnedPolicy`,
`BanditPolicy`, `PortfolioPolicy` (Task 8) implement the same interface — the
advisor is agnostic to which is installed. This is the single most important
extensibility seam: *learning slots in without touching analysis or
reformulation.*

### 3.7 `Coordinator` / `CutGenerator` / `CommunicationLayer`

Solve-time traits, detailed in §8. The coordinator drives the master/subproblem
loop; the cut generator turns subproblem duals/rays into master constraints; the
communication layer abstracts thread-pool vs Ray vs MPI so the *algorithm* is
written once (see §13).

---

## 4. Graph representations (Task 2)

The advisor maintains a **graph registry** — a lazily populated, cache-keyed set
of views over the model. Below, each graph gets: construction, complexity,
memory, what it reveals, caching, and incremental update. Sizes: `n` variables,
`m` constraints, `nnz` Jacobian nonzeros, `d` average degree.

Two conventions used throughout:
- All graphs are built from one **canonical CSR incidence** derived from the
  expression DAG (§4.1). Build it once; every other graph is a transform of it.
- "Cache?" answers assume the model-version-counter invalidation of §4.14.

### 4.1 Variable–Constraint Graph (bipartite incidence) — the substrate

- **Build:** one pass over the expression DAG collecting, per constraint, the set
  of variables it references — exactly what the existing code does via
  `_collect_variables` / `_vars_in` in `structure.py`. Emit CSR: row pointers
  over constraints, column indices = variable ids. Objective is an extra row.
- **Complexity:** `O(nnz)` build. `O(size)` to collect variables per expression.
- **Memory:** `O(nnz)` (two `u32`/`f64` arrays + row pointers). This is the
  cheapest and most-reused structure.
- **Reveals:** connected components (block-diagonal structure directly), bridge
  constraints (today's coupling heuristic), degree distribution (hub constraints
  = coupling candidates), the fundamental "remove these vertices → it splits"
  analysis behind Benders.
- **Cache?** **Yes, always.** It is the substrate for everything.
- **Incremental:** adding/removing a constraint is a row insert/delete
  (append-friendly CSR or a small overlay list); adding a variable is a new
  column. Bound changes do *not* change it. Version counter bumps on structural
  edits only.

### 4.2 Constraint Hypergraph

- **Build:** each constraint = a hyperedge over its variable set; vertices =
  variables. Directly the transpose grouping of §4.1.
- **Complexity:** `O(nnz)`. **Memory:** `O(nnz)`.
- **Reveals:** the *right* object for hypergraph partitioning (METIS/KaHyPar).
  Minimizing hyperedge cut = minimizing the number of *coupling constraints*,
  which is precisely the Dantzig–Wolfe / Lagrangian objective. Bipartite
  min-edge-cut over-counts; the hypergraph cut counts each linking constraint
  once — this distinction matters and is why we keep both.
- **Cache?** Yes when a partitioner will run; derived cheaply from §4.1.
- **Incremental:** hyperedge add/remove; partition is *not* incrementally
  maintained (re-partition on demand, gated by budget).

### 4.3 Expression DAG

- **Build:** already exists — discopt's modeling API *is* an expression DAG
  (`python/discopt/modeling/`). The advisor consumes it read-only.
- **Complexity/Memory:** owned by the modeling layer; advisor pays `O(1)` to
  borrow.
- **Reveals:** shared subexpressions (common factors → column-generation
  substructure), nonlinearity/curvature per node (feeds §6 nonlinear
  localization), separability of the objective (Lagrangian assumes separable
  objective — this graph verifies it), and which variables enter which nonlinear
  atoms (feeds Hessian sparsity without a second symbolic pass).
- **Cache?** Owned upstream; advisor caches only its *annotations* (curvature,
  nonlinearity flags) keyed by DAG node id.
- **Incremental:** upstream concern; annotations invalidated per-node on rewrite.

### 4.4 Dependency Graph (variable → variable)

- **Build:** project the bipartite graph onto variables: two variables are
  adjacent iff they co-occur in a constraint (or objective term). This is the
  "column intersection graph."
- **Complexity:** `O(Σ_c deg(c)^2)` naive; use the hypergraph form to avoid
  materializing dense cliques for high-degree coupling rows (a single dense row
  over `k` vars creates `k^2` edges — materialize lazily / keep as hyperedges).
- **Memory:** can blow up on dense coupling rows → **do not materialize** the
  clique expansion for rows above a degree threshold; represent them implicitly.
- **Reveals:** community structure (Louvain/Leiden), the graph on which spectral
  partitioning and min-cut naturally operate.
- **Cache?** Only if a variable-graph algorithm is requested; guard the dense-row
  blowup.
- **Incremental:** local — a constraint edit touches only its variables'
  adjacency.

### 4.5 Jacobian Sparsity Graph

- **Build:** ∂c_i/∂x_j nonzero pattern. For **linear** constraints this equals
  §4.1. For nonlinear, the *structural* pattern comes from the DAG (a variable is
  in the pattern iff it appears in the constraint body, even if the value is
  currently zero — we want the *structural* Jacobian, not a point evaluation).
- **Complexity:** `O(nnz)` structural. **Memory:** `O(nnz)`.
- **Reveals:** essentially coincides with §4.1 for structure; kept as a distinct
  view because the *numeric* Jacobian (with magnitudes) feeds edge weights for
  weighted min-cut ("cut the weakest linkages") and OBBT prioritization.
- **Cache?** Structural: yes (≈ alias of §4.1). Numeric-weighted: recompute at a
  reference point, cache with a short TTL (values drift as bounds tighten).
- **Incremental:** structural like §4.1; numeric invalidated by bound/point
  changes.

### 4.6 Hessian Sparsity Graph

- **Build:** ∂²(L)/∂x_i∂x_j nonzero pattern of the Lagrangian (objective +
  constraints). Two variables are adjacent iff they appear **together in a
  nonlinear term**. Extract directly from DAG nonlinear atoms (a bilinear `x·y`
  → edge; a separable `x²` → no cross edge).
- **Complexity:** `O(Σ nonlinear-term arities²)`, small in practice (nonlinear
  atoms are low-arity). **Memory:** `O(#nonlinear coupling pairs)`.
- **Reveals:** the *nonlinear coupling* structure — the make-or-break for GBD and
  Schur. If the Hessian is block-diagonal after removing complicating variables,
  the recourse subproblems are *separable convex NLPs* (the ideal GBD case). The
  Hessian graph is also what a Schur-complement/interior-point decomposition
  partitions.
- **Cache?** Yes for QP/NLP/MINLP; skipped entirely for LP/MILP (no Hessian).
- **Incremental:** invalidated by structural edits to nonlinear terms only.

### 4.7 KKT Graph

- **Build:** the sparsity graph of the KKT/augmented system
  `[[H, Jᵀ],[J, 0]]` — union of Hessian (§4.6) among variables, Jacobian (§4.5)
  between variables and constraint-duals, plus dual–dual structure.
- **Complexity:** `O(nnz + hessian_nnz)`. **Memory:** same order.
- **Reveals:** the graph that **interior-point / Schur-complement decomposition**
  actually factorizes. Nested dissection of the KKT graph *is* the Schur
  decomposition (border = separator). Fill-in of its elimination ordering (§4.11)
  predicts the cost of a direct solve — a direct input to the "KKT fill-in"
  metric (§6).
- **Cache?** Yes when a Schur/IPM decomposition is a live candidate.
- **Incremental:** rebuild on active-set / structural change; not maintained
  across IPM iterations.

### 4.8 Variable Elimination Graph / Elimination Tree

- **Build:** run a fill-reducing ordering (min-degree / min-fill / nested
  dissection) on §4.6 or §4.7; the elimination tree records the dependency of the
  factorization.
- **Complexity:** ordering is `O(nnz·α)`-ish (near-linear with good heuristics);
  exact treewidth is NP-hard so we only ever compute *bounds*.
- **Memory:** `O(n)` tree + `O(fill)` predicted.
- **Reveals:** separators (→ nested Benders / Schur border), the natural *stage*
  ordering for nested decomposition, and fill-in cost. The elimination tree's
  branching factor tells you whether *parallel* multifrontal / nested
  decomposition will have exploitable independent subtrees.
- **Cache?** Yes; it's the bridge from "graph" to "decomposition hierarchy."
- **Incremental:** recompute on structural change; a bound change never touches
  it.

### 4.9 Block Graph

- **Build:** quotient of §4.4 by a candidate partition — one super-vertex per
  block, super-edges = coupling between blocks (via shared coupling constraints
  or dualized links). This is exactly the object today's
  `DecompositionStructure.blocks` + `coupling_constraints` imply.
- **Complexity:** `O(nnz)` given a partition. **Memory:** `O(#blocks + #coupling)`.
- **Reveals:** block-size variance, coupling density, number of blocks — the
  headline scoring inputs (§6). Its own connectivity says whether the coupling is
  "star" (one master, many independent subs → Benders/DW) vs "chain" (stages →
  nested) vs "mesh" (ADMM territory).
- **Cache?** Per-candidate (cheap to rebuild). **Incremental:** rebuild per
  candidate.

### 4.10 Coupling Graph

- **Build:** the subgraph induced by coupling constraints/variables only — the
  "border" of a bordered block-diagonal form. Vertices = blocks + coupling
  entities; edges = participation.
- **Reveals:** separator size, communication cost (each coupling entity = a
  message per iteration), and whether coupling is *linear* (Benders/DW cuts are
  exact) or *nonlinear* (GBD needed, cuts weaker). Distinguishes
  *variable-coupling* (→ Benders: fix vars) from *constraint-coupling*
  (→ Lagrangian/DW: dualize/price rows). This is the single most decision-
  relevant graph after the substrate.
- **Cache?** Per-candidate. **Incremental:** per-candidate.

### 4.11 Stage Graph

- **Build:** a DAG over temporal/logical stages, inferred from (a) user
  annotations (`first_stage`/`second_stage` already exist as `_decomp_stages`),
  (b) index-set structure (time-indexed variables → stages by time index), or (c)
  the elimination tree's level structure. Nodes = stages; edges = "stage t
  constrains stage t+1."
- **Complexity:** `O(n+m)` from annotations/indices; ordering-based inference
  reuses §4.8.
- **Reveals:** the recursion structure for **nested Benders / dual dynamic
  programming (SDDP)**. A linear chain → classic multistage; a tree → scenario
  tree.
- **Cache?** Yes. **Incremental:** cheap; annotation-driven.

### 4.12 Scenario Graph

- **Build:** for stochastic models, nodes = scenarios, connected by
  *non-anticipativity* (shared first-stage decisions). Detected from a scenario
  index set or an explicit `scenario` dimension on variables/constraints.
- **Reveals:** the structure **Progressive Hedging** and scenario-wise Lagrangian
  exploit — each scenario is an independent subproblem coupled only by
  non-anticipativity. Extremely regular (all subproblems isomorphic) → excellent
  parallel efficiency and warm-start reuse.
- **Cache?** Yes. **Incremental:** add/remove scenario = add/remove an isomorphic
  block.

### 4.13 Dual Dependency Graph & Master/Subproblem Graph

- **Dual dependency graph:** which subproblem duals feed which master
  constraints. Built *after* a candidate is chosen; it is the dataflow graph of
  the coordination loop. Reveals cut-sharing opportunities (a dual that appears in
  many masters → aggregate cut) and the critical path of one iteration.
- **Master/Subproblem graph:** the concrete bipartite "who talks to whom" graph
  used by the scheduler (§13) — masters, subproblems, and the messages between
  them. This is the *execution* graph, not an analysis graph; it is generated by
  the reformulator (§8) and consumed by the coordinator/scheduler.
- **Cache?** Lives with the `DecomposedModel`, not the analysis cache.

### 4.14 Additional graphs worth adding

- **Symmetry graph.** Vertices = variables/constraints, colored by
  type/coefficients; its automorphism group (via a graph-automorphism backend
  like nauty/saucy) exposes symmetric blocks → *aggregate* identical subproblems
  (huge for DW/PH where scenarios or blocks are isomorphic). Also feeds symmetry-
  breaking in presolve (§11). High value, medium cost — gate behind a flag.
- **Convexity-labeled overlay.** Not a new graph but a *labeling* of §4.1/§4.6
  edges/nodes with curvature from the convexity pass. Turns "is this block
  convex?" into an `O(block)` lookup instead of a re-analysis. Essential for §6's
  convexity and nonlinear-localization metrics.
- **Integer-incidence overlay.** Nodes flagged integer/binary. Lets the advisor
  answer "removing these `k` integer vars disconnects into how many components?"
  in one projected-CC pass — the core Benders test, and a direct generalization
  of today's default (complicating vars = the integer/binary vars).
- **Temporal/streaming delta graph.** For warm-restarted or streaming solves,
  the *diff* graph between successive models, enabling incremental
  re-decomposition (§14 research).

### 4.15 The incremental-update contract (applies to all)

A single **`model_version: u64`** counter, bumped on any structural edit
(add/drop variable or constraint, change a term's variable set). Numeric edits
(bounds, coefficients' *values*) bump a separate `numeric_version`. Each cached
graph stores the `(model_version, numeric_version)` it was built at.
`is_stale()` compares. Structural graphs invalidate on `model_version`; weighted
graphs on either. This gives the FBBT/OBBT loop (which changes *bounds*, not
structure) free reuse of all structural graphs across many bound-tightening
rounds — the common case in the solve pipeline.

---

## 5. Structure-discovery algorithms (Task 3)

Each algorithm is a `CandidateGenerator` (or a kernel one uses). Guidance is
strength / weakness / complexity / suitability-for-optimization.

| Algorithm | Complexity | Strength | Weakness | Best for |
|---|---|---|---|---|
| **Connected components** (union-find) | `O(nnz·α)` | trivially exact, already implemented in `structure._components` | only finds *already-disconnected* structure | block-diagonal; the free first check |
| **Strongly connected components** (Tarjan) | `O(n+e)` | directionality (dual dependency, stage precedence) | needs a *directed* graph (KKT dataflow, stage graph) | stage/nested ordering, cyclic-coupling detection |
| **Articulation points / bridges** | `O(n+e)` | finds the *cheapest* cut vertices/edges exactly and fast; generalizes today's bridge heuristic | only single-vertex/edge separators | small separators; the existing coupling detector |
| **Min s-t cut / max-flow** | `O(n·e)`–`O(e·√n)` | exact min *edge* cut; weightable ("cut weakest links") | needs terminals; balance not guaranteed | isolating a known subsystem; 2-block splits |
| **Global min cut** (Stoer–Wagner) | `O(n·e + n²log n)` | no terminals needed | unbalanced cuts possible | detecting a natural weak link |
| **Normalized / spectral cut** (Fiedler vector) | `O(e·√κ)` per eigenpair (Lanczos) | balanced cuts, principled, smooth quality | eigen-solve cost, needs post-rounding | 2-way balanced bisection; seeds recursive bisection |
| **Recursive spectral bisection** | `k`·(above) | k blocks, balanced | greedy, no global optimum | k-way balanced partitions |
| **Hypergraph partitioning** (METIS/KaHyPar) | near-linear (multilevel) | **directly minimizes coupling-constraint count**, balance-constrained, industrial-strength | external dep, nondeterministic unless seeded | Dantzig–Wolfe / Lagrangian block-angular detection — the workhorse |
| **Nested dissection** | `O(n^{3/2})` on planar-ish, worse dense | produces a *separator hierarchy* + good elimination order | cost on dense/expander graphs | Schur/IPM decomposition, nested Benders, fill reduction |
| **Community detection — Louvain** | `O(n log n)` typical | fast, finds natural modules without a preset k | resolution limit, nondeterministic | discovering *unknown* block count; exploratory |
| **Community detection — Leiden** | `O(n log n)` | fixes Louvain's disconnected-community defect; well-connected guarantee | slightly slower | same, when correctness of communities matters |
| **Tree decomposition / treewidth** | exact NP-hard; heuristic bounds near-linear | if treewidth small → provably efficient dynamic-programming decomposition | exact intractable; only bounds usable | certifying a model is "nearly decomposable" |
| **Chordal completion** (min-fill/min-degree) | `O(nnz·α)` heuristic | gives elimination tree + clique tree cheaply | heuristic fill | elimination trees (§4.8), Schur ordering |
| **Clique detection** | NP-hard exact; greedy fast | dense sub-blocks = tight coupling to keep together | exact intractable | keeping bilinear cliques intact for relaxation |
| **Minimum vertex cut / separator** | `O(n·e)` via flow | the *right* object for Benders (which vertices to fix) | balance not guaranteed | choosing complicating variables optimally |

### 5.1 Which to actually run, and in what order

The advisor runs a **cheap-to-expensive cascade**, stopping early when a
high-scoring candidate is found (mirroring the `_BRIDGE_SCAN_BUDGET` guard
already in `structure.py`):

```
1. Connected components on the integer-projected graph      (free)        → block-diagonal? Benders-obvious?
2. Articulation/bridge scan (existing heuristic)            (near-linear) → single coupling constraint?
3. Annotation ingestion (first_stage/mark_coupling)         (free)        → user knows best
4. Community detection (Leiden)                             (n log n)     → natural block count k
5. Hypergraph partition to k blocks (METIS/KaHyPar)         (near-linear) → refined block-angular form
6. Nested dissection on KKT/Hessian                        (superlinear) → Schur/nested, only if 1–5 promising
7. Min vertex cut to optimize complicating-var set          (flow)        → tighten a promising Benders split
```

Steps 5–7 are **budget-gated** and only run if a cheaper step already suggests
exploitable structure *and* the model is large enough that decomposition could
pay off. On a small or clearly-monolithic model, the cascade stops at step 2 with
a "no-decomp" verdict — near-zero overhead, which honors design goal #2.

### 5.2 Why not "just run METIS on everything"

METIS is the workhorse but it is the *wrong first tool*: it always returns a
k-way balanced partition even when the model is genuinely monolithic (it will
happily "cut" an expander into k pieces with a huge separator). The cascade uses
exact, cheap detectors (CC, bridges) first precisely so we only invoke balanced
partitioners when structure is already indicated, and so we can tell the
difference between "found real structure" and "forced a bad cut." The scorer
(§6) then has to *reject* forced cuts — a large-separator partition must score
worse than no-decomp.

---

## 6. Scoring system (Task 4)

### 6.1 The scoring problem

After candidate generation we may have several partitions × several methods. We
need a total order. But the metrics are heterogeneous (counts, densities,
probabilities) and the "right" weighting depends on the *method* and the
*hardware target*. So scoring is **two-stage**: per-metric estimation, then a
method-aware scalarization with a confidence.

### 6.2 The metric catalog

Grouped by what they predict. Each has a cheap estimator computable from the
graphs of §4.

**Structural quality (predicts: does a good decomposition even exist?)**
- `num_blocks` — more independent blocks = more parallelism, but diminishing
  returns and coordination overhead.
- `block_size_variance` (coefficient of variation) — high variance = stragglers
  = poor load balance. **Dominant for parallel efficiency.**
- `coupling_density` = coupling entities / total — the single best predictor of
  whether decomposition pays. Low (≤ ~5–10%) is the green light; the task's own
  example ("coupling density only 3%") is exactly this. **Top-tier metric.**
- `separator_size` — for Schur/nested; the Schur complement is dense `s×s` where
  `s` = separator, so cost scales `s³`. **Dominant for Schur.**

**Cost model (predicts: how fast per iteration?)**
- `estimated_communication_cost` — coupling entities × iterations × message size;
  from the coupling graph (§4.10) + a method-specific iteration estimate.
- `kkt_fill_in` — predicted factorization fill from the elimination tree (§4.8);
  decomposition wins when it *reduces* fill vs monolithic.
- `memory_locality` / `cache_friendliness` — whether blocks fit in cache; block
  size vs cache size. Feeds the GPU/CPU target choice.
- `gpu_suitability` — many small *dense, identical* blocks (scenarios!) = ideal
  for batched GPU (discopt already uses JAX/GPU) → PH/DW on GPU. Irregular sparse
  blocks = poor GPU fit.

**Convergence quality (predicts: how many iterations?)**
- `expected_cut_strength` — for Benders/DW: linear coupling → strong cuts;
  nonconvex coupling → weak cuts / no finite convergence guarantee. Estimated
  from coupling-graph curvature labels.
- `warm_start_potential` — isomorphic/repeated blocks (scenarios, DW columns)
  reuse factorizations and bases across iterations → big constant-factor win.
- `convexity` (of subproblems) — convex subproblems → duals are meaningful,
  GBD/Benders cuts valid. Nonconvex subproblem → need Lagrangian bounds or
  spatial B&B inside the sub. **Gatekeeper metric** (can veto a method).

**Localization (predicts: does decomposition isolate the hard part?)**
- `integer_localization` — fraction of integer vars confined to the
  master/few blocks. High = Benders/LBBD shine (solve one small MILP master,
  continuous subs). This is the payoff the existing "complicating vars = integer
  vars" default is chasing.
- `nonlinear_localization` — fraction of nonlinearity confined to subproblems vs
  the master. High = GBD/OA-friendly.

**Reliability**
- `confidence` — every estimate carries one; a partition from exact CC has
  confidence 1.0, a spectral-rounded cut ~0.6, a learned prediction whatever the
  model reports.

### 6.3 Which metrics dominate

From first principles, the decision decomposes into **feasibility**, then
**benefit**, then **cost**:

1. **Gatekeepers (veto, not weigh):** `convexity`/`expected_cut_strength` for the
   *chosen method's* soundness, and `est_soundness` on the candidate. If a method
   would be unsound on this structure, it is removed, not down-weighted.
   (Correctness-first, per goal #1.)
2. **Primary benefit signal:** `coupling_density` and `separator_size`. If
   coupling is dense / separator is large, *no* decomposition helps — return
   no-decomp regardless of other metrics. These two carry the most weight.
3. **Speedup magnitude:** `num_blocks` × `block_size_variance` (parallel
   efficiency) and `integer_localization` / `nonlinear_localization` (how much of
   the hard combinatorics/nonlinearity got isolated).
4. **Constant factors / tie-breakers:** communication cost, warm-start, GPU
   suitability, cache friendliness.

Concretely, the aggregate is a **lexicographic-then-weighted** scalarization:

```
if not sound(method, cand):            score = -inf            # veto
elif coupling_density > τ_couple:      score = no_decomp_baseline - penalty
else:
    benefit  = w1·parallel_efficiency(num_blocks, size_cv)
             + w2·localization(integer, nonlinear)
             + w3·cut_strength
    cost     = w4·comm_cost + w5·fill_in + w6·(1 - warm_start)
    score    = benefit − cost,  reported with confidence
```

Weights `w*` and threshold `τ_couple` are **method- and target-specific**
(defaults in `config/`, learnable in Task 8). Crucially the baseline competitor
is **always "no-decomp"**: a candidate must beat monolithic on *estimated total
wall-clock*, not on abstract elegance. This is the guard against the METIS
"forced cut" failure mode (§5.2).

### 6.4 Estimating speedup and parallel efficiency (the explainer's numbers)

The task's example explanation quotes "18× speedup, 92% parallel efficiency."
These come from an **analytic performance model**, not a guess:

- Amdahl with the coupling as the serial fraction:
  `speedup ≈ 1 / (f_coord + (1−f_coord)/p_eff·B)` where `B`=blocks, `p_eff` from
  block-size CV, `f_coord` from coupling density × iteration count.
- Iteration count estimated per-method (Benders: ~ separator dimension +
  constant; DW/CG: column-count model; ADMM: condition-number-driven).
- All estimates carry a confidence and are **logged then compared to reality**
  (Task 8) so the model self-calibrates. Early on, be honest: report ranges and
  low confidence rather than false precision.

---

## 7. Method selection (Task 5)

Selection is a decision procedure over the `StructureReport`. The default
`RuleBasedPolicy` encodes the tree below; a `LearnedPolicy` (Task 8) can override
or reweight it. The tree is intentionally *conservative* — it defaults to
no-decomp on ambiguity.

### 7.1 The decision tree

```
                         ┌─────────────────────────────────────┐
                         │  Is the model already block-diagonal │
                         │  (≥2 CC with no coupling)?           │
                         └───────────────┬──────────────────────┘
                                    yes  │  no
              ┌──────────────────────────┘  └───────────────────────┐
              ▼                                                       ▼
   Independent-block solve                          ┌────────────────────────────────┐
   (parallel, no coordination)                      │ What kind of coupling remains?  │
                                                     └──────┬───────────────┬─────────┘
                                         VARIABLE-coupling  │               │  CONSTRAINT-coupling
                                    (fixing few vars splits)│               │ (few linking rows)
                          ┌──────────────────────────────────┘               └───────────────────────┐
                          ▼                                                                            ▼
         ┌────────────────────────────────┐                              ┌──────────────────────────────────┐
         │ Are complicating vars integer? │                              │ Are linking rows + blocks LINEAR? │
         └───────┬──────────────┬─────────┘                              └────────┬───────────────┬─────────┘
             yes │              │ no (continuous complicating)                 yes │               │ no
                 ▼              ▼                                                  ▼               ▼
   ┌────────────────────┐  ┌─────────────────────────┐          ┌───────────────────────┐  ┌──────────────────────────┐
   │ Subproblem convex? │  │ Subproblem convex NLP?  │          │ Master reoptimized     │  │ Blocks convex, coupling   │
   ├─────────┬──────────┤  ├──────────┬──────────────┤          │ often, cols priced?    │  │ nonlinear/nonconvex?      │
   │ LINEAR  │ nonconvex│  │  yes     │  no           │          ├──────────┬─────────────┤  ├───────────┬──────────────┤
   ▼         ▼          ▼  ▼          ▼               ▼          ▼          ▼             ▼  ▼           ▼              ▼
Classical  Logic-Based  spatial-  Generalized   Lagrangian/   Dantzig-  Column      Lagrangian/  Augmented-Lag /  no-decomp
Benders    Benders(LBBD) B&B in   Benders(GBD)  spatial in    Wolfe     Generation  ADMM         ADMM             (or Schur if
(recourse  (CP/MILP     the sub   (convex       the sub                 (dynamic                 (splitting)      KKT-separable)
 LP)        subproblem)           recourse NLP)                          columns)
```

### 7.2 Rules in prose (the load-bearing ones)

- **Block-diagonal, no coupling →** just solve blocks independently in parallel.
  No cuts, no coordination. (The scorer still checks it's worth the thread
  overhead.)
- **Few complicating *variables*, fixing them separates the model →** Benders
  family. Then split by subproblem type: linear recourse → **classical Benders**;
  convex nonlinear recourse → **Generalized Benders** (discopt already dispatches
  `solve_gbd` here); subproblem itself has integers/logic → **Logic-Based
  Benders**; nonconvex recourse → Benders is unsound, fall to spatial-B&B-in-sub
  or no-decomp.
- **Few linking *constraints*, blocks priced repeatedly, master is a
  set-partitioning/covering-like →** **Dantzig–Wolfe / Column Generation**
  (identical mechanics; CG = DW with dynamic columns). Isomorphic blocks →
  aggregate + strong symmetry win.
- **Many coupling constraints, want a bound / distributed solve →**
  **Lagrangian relaxation** (dual bound) or **ADMM / Augmented Lagrangian**
  (splitting for consensus, robust when many moderate couplings and you want
  first-order scalability). discopt's `solve_lagrangian` is the entry point.
- **Stochastic scenario structure (non-anticipativity) →** **Progressive
  Hedging** (or scenario-wise Lagrangian) — one subproblem per scenario, GPU-
  batchable.
- **Multistage chain / tree →** **Nested Benders / SDDP**.
- **Sparse KKT with a small separator, continuous/IPM-heavy →** **Schur
  complement** decomposition inside the linear algebra (transparent to the user;
  a linear-algebra-level decomposition rather than an algorithmic one).
- **Ambiguous / dense coupling / unsound →** **no-decomp**, with an explanation.

### 7.3 Hybrid strategies

Real models are layered; the advisor supports **nested / composed**
decompositions via the same IR (§8):

- **Benders outer + Dantzig–Wolfe inner:** integer complicating vars in a Benders
  master, each Benders subproblem itself block-angular → price it with DW.
- **Lagrangian outer + block-parallel inner:** dualize the few couplings, solve
  the resulting independent blocks in parallel (this is what
  `solve_lagrangian` + block detection already gestures at).
- **Scenario (PH) outer + Benders inner:** per-scenario subproblem is itself a
  two-stage MILP → Benders inside each scenario.
- **Cross-decomposition:** alternate Benders and Lagrangian to get both primal
  cuts and dual bounds (classic Van Roy). The coordinator supports running two
  cut generators against one master.

Nesting is expressed as a `DecomposedModel` whose subproblems are themselves
`DecomposedModel`s — the recursion is uniform, and the advisor can be invoked
*recursively* on each subproblem (budget-limited depth).

---

## 8. Automatic reformulation & the decomposition IR (Task 6)

Once a `(Candidate, method)` wins, the `DecompositionMethod.reformulate()` builds
a **`DecomposedModel`** — the intermediate representation that the coordinator
executes. This IR is the contract between "what to solve" and "how to solve it in
parallel."

### 8.1 The IR object graph

```
DecomposedModel
├── source: &Model                      # borrow; for reporting in original space
├── method: MethodKind
├── master: MasterModel                 # owns a discopt Model + its var/con maps
├── subproblems: list[SubproblemModel]  # each owns a Model + fixing/pricing hooks
├── cut_generator: CutGenerator         # sub duals/rays → master cuts/columns
├── coordinator: Coordinator            # the solve loop + convergence test
├── comm: CommunicationLayer            # thread pool | Ray | MPI | GPU-batch
├── schedule: SchedulingGraph           # dataflow: who runs when, dependencies
├── var_map: VariableMapping            # original ↔ (master, sub) variable ids
└── certificate: SoundnessCertificate   # why this equals the original problem
```

### 8.2 The pieces and their responsibilities

- **`MasterModel`** — owns a real discopt `Model` (the coordinating problem:
  Benders master MILP, DW restricted master LP, Lagrangian dual, PH consensus).
  Exposes `add_cuts(...)`, `add_columns(...)`, `warm_start(...)`, `solve()`. It
  reuses the *entire existing solver stack* — a master is just a discopt model.
- **`SubproblemModel`** — owns a discopt `Model` for one block, plus a
  parameterization hook: `restrict(fixed_master_vars)` for Benders (built on the
  **existing** `restricted_bounds()` in `structure.py` — pin `lb==ub`), or
  `reprice(duals)` for DW/Lagrangian. Returns primal + **dual/ray** info.
- **`CutGenerator`** — the method-specific map from subproblem output to master
  update. Benders optimality/feasibility cuts, GBD cuts (via the convex dual),
  LBBD "no-good"/logic cuts, DW columns, Lagrangian subgradients. This is where
  `solve_benders`/`solve_gbd`/`solve_lagrangian`'s cut logic gets refactored to
  live behind one trait.
- **`Coordinator`** — owns the loop: solve master → dispatch subproblems (via
  `comm`) → collect → generate cuts/columns → convergence test → repeat. Owns the
  incumbent, bounds, cut pool, stall detection, and stopping criteria. It is the
  only stateful actor.
- **`CommunicationLayer`** — abstracts *where* subproblems run (§13). The
  coordinator calls `comm.map(subproblems, task)`; whether that's Rayon threads,
  a Ray actor pool, MPI ranks, or a JAX `vmap` batch is a backend detail.
- **`SchedulingGraph`** — the master/subproblem dataflow DAG (§4.13); the
  scheduler uses it to overlap master solve with subproblem prep, pipeline cut
  aggregation, and place work by data locality.
- **`VariableMapping` + `SoundnessCertificate`** — the correctness spine. The
  mapping lets results come back in the user's variable names (the existing
  drivers already return original-space results). The certificate records *why*
  the decomposition is equivalent (exact reformulation) or what it bounds
  (relaxation), and is checkable in tests — the mechanism that enforces goal #1.

### 8.3 Ownership at solve time

- The `DecomposedModel` owns master/subproblems/coordinator.
- The coordinator holds `&mut` to shared solve state; subproblems are solved
  through `comm`, which may **move** them to workers (Ray/MPI) — hence
  subproblems must be **serializable** (they own their `Model`, which discopt can
  already serialize for result IO). For shared-memory backends they stay in place
  behind `Arc`.
- Cuts/columns flow **into** the master (owned there); subproblem state
  (factorizations, warm bases) is owned by each subproblem for reuse across
  iterations — the warm-start win.
- On completion, the coordinator assembles a `SolveResult` in original variable
  space via `var_map` and drops the workers; the user's `Model` is untouched.

### 8.4 Why an IR rather than method-specific glue

The existing drivers each hard-code their own master/sub/loop. That does not
compose (you cannot nest Benders-in-PH) and duplicates the loop, warm-start, and
stopping logic three times. The IR factors out the **coordinator + comm + cutgen**
seams so: (a) a new method is a `CutGenerator` + a `reformulate()`, not a whole
solver; (b) nesting is free (subproblem = another `DecomposedModel`); (c) the
parallel backend is chosen once, centrally (§13); (d) correctness lives in one
place (the certificate + var_map), not scattered.

---

## 9. Explainability (Task 7)

Explainability is a required output of *every* recommendation, produced by the
`Explainer` from the same numbers the scorer used — never post-hoc
rationalization.

### 9.1 The explanation data model

```python
@dataclass(frozen=True)
class Explanation:
    recommendation: MethodKind | None          # None ⇒ no-decomp
    headline: str                               # "Recommended: Generalized Benders"
    reasons: list[Rationale]                    # positive, evidence-backed
    concerns: list[Concern]                     # risks, with severity
    metrics: ScoreVector                        # the raw numbers
    alternatives: list[RankedAlternative]       # runners-up + why they lost
    estimated: PerformanceEstimate              # speedup, p_eff, iters, confidence
    provenance: Provenance                      # which graphs/algos/params produced this
    confidence: float

@dataclass(frozen=True)
class Rationale:
    claim: str                                  # human sentence
    evidence: Mapping[str, float | str]         # the metrics that justify it
    graph_ref: GraphKind | None                 # clickable → visualization

@dataclass(frozen=True)
class Concern:
    issue: str                                  # "Constraint C19 is nonconvex"
    severity: Severity                          # INFO | WARN | BLOCKER
    affected: list[str]                         # constraint/var names
    mitigation: str | None                      # "consider OBBT on x7,x8 first"
```

### 9.2 Rendering — matching the task's target

`advisor.explain()` renders, e.g.:

```
Recommended: Generalized Benders                                  confidence 0.86
────────────────────────────────────────────────────────────────────────────────
Why:
  • Removing 8 integer variables disconnects the model into 46 components.
        evidence: integer_localization=1.0, components_after_projection=46   [graph: BLOCK]
  • Coupling density is only 3%.
        evidence: coupling_density=0.03, coupling_constraints=12             [graph: COUPLING]
  • Subproblems are convex NLPs.
        evidence: subproblem_convexity=convex (45/46), method=GBD            [graph: HESSIAN]
Estimated:
  • speedup ≈ 18× (Amdahl, f_coord=0.04), parallel efficiency ≈ 92%
        block_size_cv=0.11 across 46 blocks                        [confidence 0.7]
Potential issues:
  • Constraint C19 is nonconvex.               severity=WARN   affects: C19
        mitigation: OBBT on {x12,x13}; else block 17 needs spatial B&B in-sub.
  • Cut convergence may be weak (nonlinear coupling on 2 links). severity=INFO
Alternatives considered:
  • Classical Benders — rejected: recourse is nonlinear (would be unsound).
  • Lagrangian relaxation — feasible, ranked #2: gives a bound but slower
        convergence here (dual-only, est. 3.1× vs 18×).
  • no-decomp — baseline; est. 18× slower than GBD.
```

Every bullet is backed by a metric and, where relevant, a `graph_ref` the
`visualize()` API can render (§15). `advisor.explain(format="json")` returns the
machine-readable `Explanation` for tooling, and `format="markdown"` for the
LLM-features layer (the repo's `llm/` module can narrate it further).

### 9.3 Counterfactual and "why not" explanations

Beyond "why this," the API answers **"why not that"**: for any method the user
asks about, the explainer reports which gatekeeper or metric ruled it out, and
what would have to change (e.g. "Benders would apply if C19 were convex; it is
not"). This turns the advisor into a *teaching* tool and integrates with the
repo's existing `tutor.py` / `llm` explanation features.

---

## 10. Learning from experience (Task 8)

### 10.1 Can it improve over time? Yes — via the `Policy` seam

Because selection is isolated behind `Policy` (§3.6), learning never touches
analysis or reformulation. We record ground truth after every solve and use it to
improve ranking. The progression, cheapest to most speculative:

- **Instance-based recommendation (start here).** Store `(features →
  best-observed method + wall-clock)` and, for a new instance, retrieve the
  nearest neighbors' winners. Robust, interpretable, no training loop. This alone
  captures most of the value.
- **Solver performance database + portfolio.** Treat the methods as a portfolio;
  pick per-instance using empirical performance (algorithm selection, à la
  SATzilla). Log-based, offline.
- **Bandits (online).** For repeated/similar solves (a user re-solving a family),
  run a contextual bandit (LinUCB/Thompson) over methods with the score vector as
  context — balances exploiting the known-good method vs exploring a possibly-
  better one. Cheap, online, safe (the gatekeepers still veto unsound arms).
- **Offline supervised policy.** Learn `Policy.rank` weights / a gradient-boosted
  ranker on the accumulated database (features → runtime). Replaces hand-tuned
  `w*` in §6.
- **Reinforcement learning.** For *sequential* decisions (which block to branch,
  when to switch cut types, nested-decomposition depth) framed as an MDP. Higher
  variance, later.
- **Meta-learning.** Learn to adapt quickly to a *new* problem family from few
  solves — valuable because optimization users cluster into families
  (scheduling, OPF, network design).
- **Graph neural networks.** The models *are* graphs (§4). A GNN over the
  variable–constraint graph can predict good partitions / method suitability
  end-to-end — the most research-forward option (§16), and a natural fit since
  the advisor already materializes these graphs.

### 10.2 What to record after every solve (`SolveRecord`)

```python
@dataclass(frozen=True)
class SolveRecord:
    instance_fingerprint: bytes
    features: InstanceFeatures            # §10.3
    chosen: MethodKind
    considered: list[tuple[MethodKind, ScoreVector]]   # incl. runners-up
    predicted: PerformanceEstimate        # what the advisor forecast
    observed: ObservedPerformance         # wall-clock, iters, #cuts/#cols,
                                          #   parallel efficiency, peak mem,
                                          #   converged?, gap, sub/master time split
    decomposition: DecompositionStructure # the actual partition used
    solver_config: dict                   # tolerances, threads, backend
    outcome: Outcome                      # OPTIMAL | TIMEOUT | INFEASIBLE | ...
    timestamp: int
```

The key discipline: **record `predicted` alongside `observed`** so the advisor's
own error is measurable and its confidence estimates get calibrated. This is the
data that turns §6.4's analytic speedup model from a guess into a regression.

### 10.3 Feature engineering / embeddings

Cheap scalar features (size, density, block count, coupling density, integer
fraction, nonlinearity fraction, treewidth bound, block-size CV, symmetry group
size) for the retrieval/portfolio models; **graph embeddings** (spectral moments,
Weisfeiler–Lehman hashes, or learned GNN embeddings) for the neural options. The
feature extractor lives in `learning/features.py` and reuses the already-built
graphs — no extra passes.

### 10.4 Storage & privacy

`learning/store.py` writes to a local parquet/sqlite database by default (opt-in,
off unless enabled). Shareable, anonymized "decomposition benchmark" datasets are
a research artifact (§16). Nothing leaves the machine unless the user exports it.

---

## 11. Interaction with presolve (Task 9)

### 11.1 The ordering question, resolved: **iterative, not linear**

The naive answer ("decompose then solve") is wrong, and so is the opposite
("presolve fully, then decompose"). The right structure is a **fixed-point loop**
between presolve and structure detection, because each changes the other's input:

```
   ┌───────────────────────────────────────────────────────────┐
   │  repeat until neither changes (or budget):                 │
   │    1. symbolic simplify + constraint aggregation            │
   │    2. FBBT bound tightening                                 │
   │    3. convexity re-detection                                │
   │    4. (re)build structural graphs  ── only if step 1 edited │
   │    5. structure analysis (cheap detectors)                  │
   │    6. targeted OBBT on coupling/complicating variables ◄────┤
   └───────────────────────────────────┬───────────────────────┘
                                        ▼
                         final decomposition decision
```

### 11.2 Why each direction matters

**Presolve → decomposition (presolve first, mostly):**
- **Symbolic simplification & aggregation change the graph.** Substituting out a
  defining constraint `y = f(x)` removes `y`'s vertex and can *merge or split*
  blocks. Aggregation can collapse a coupling constraint into a bound (removing
  the coupling entirely). So structural detection must run on the *presolved*
  graph, or it decomposes a model that no longer exists.
- **FBBT can fix variables** (bound to a point), which can disconnect blocks —
  fixed complicating variables are "free" Benders splits. Presolve makes more
  structure visible.
- **Convexity detection is a prerequisite**, not a peer: method selection
  gatekeepers (§6.3) need curvature labels, and OA/GBD need to know subproblems
  are convex.

**Decomposition → presolve (decomposition informs presolve targeting):**
- **OBBT is expensive** (an LP/NLP per bound); you cannot afford it everywhere.
  The advisor tells OBBT *where it pays*: tighten the **coupling variables** and
  **complicating variables**, because those bounds control cut strength (tighter
  complicating-var bounds → stronger Benders cuts, faster convergence). This is
  the highest-leverage OBBT targeting signal available, and today it is unused.
- **Per-block presolve.** Once blocks are identified, presolve each block
  *independently and in parallel* — cheaper and more effective than presolving
  the coupled monolith, and it exposes block-local structure.

### 11.3 Recommended ordering (the default)

1. One full presolve pass (simplify → FBBT → convexity) on the monolith.
2. Cheap structure detection (CC, bridges, annotations).
3. If structure found: targeted OBBT on the separator/complicating set, per-block
   presolve, then **re-detect** (bounds may have changed connectivity).
4. Stop when the partition stabilizes or the budget is hit.

The loop is **guaranteed to terminate** because presolve is monotone (bounds only
tighten, constraints only removed) and re-detection is only triggered by a
*structural* edit — bound-only rounds reuse cached graphs (§4.15) and cannot loop
forever. In practice 1–2 iterations suffice; deep iteration is gated.

---

## 12. Interaction with global optimization (Task 10)

Decomposition and global optimization are **mutually reinforcing**, and this is
where discopt's McCormick/spatial-B&B stack (`python/discopt/_jax/`,
`crates/discopt-core`) makes the advisor especially valuable.

### 12.1 Decomposition → better relaxations

- **Block-wise McCormick / relaxations are tighter per block** and cheaper:
  relaxing each block over its *tightened, smaller* domain (post per-block OBBT)
  gives sharper envelopes than a global relaxation over loose boxes. Decomposition
  shrinks the boxes McCormick acts on — and the repo's own relaxation work
  (`design/relaxation-patterns.md`, issue #208 OBBT) confirms tighter boxes are
  the dominant lever.
- **Localized spatial branching.** If nonconvexity is confined to a few blocks
  (high `nonlinear_localization`), spatial B&B only branches inside those blocks;
  the convex blocks are solved once. This is a large win — spatial B&B cost is
  exponential in the number of branched variables, and decomposition slashes that
  count.
- **Decomposed bounds.** Lagrangian relaxation gives valid **dual bounds** that
  strengthen the global lower bound at each B&B node; Benders/GBD cuts tighten the
  master's bound. These plug into the existing B&B as node-level bound
  improvements.

### 12.2 Relaxations → better decomposition

- **McCormick/OBBT tightening changes what's separable** (§11): a variable pinned
  by OBBT can disconnect blocks; a relaxation that proves a nonlinear term
  inactive can drop a coupling edge.
- **Convexification enables more methods.** If a nonconvex subproblem can be
  *convexified* (McCormick relaxation, RLT, the repo's envelope patterns), then
  GBD/Benders become sound where they were vetoed — the relaxation *unlocks* a
  decomposition the gatekeeper would otherwise forbid. The advisor should offer
  "relax-then-decompose" as an explicit hybrid.
- **Interval arithmetic seeds the graphs.** Interval-inferred always-inactive
  constraints/edges prune the incidence graph before partitioning.

### 12.3 Concrete integration points

- **Outer approximation** is Benders' cousin: the advisor treats OA as a Benders
  variant where cuts are gradient linearizations of convex constraints (discopt
  has OA infrastructure). Selection between OA and GBD is a §7 leaf.
- **Cut generation** is shared plumbing: Benders cuts, McCormick cuts, and OA
  cuts all flow through the same master cut pool — the coordinator's
  `CutGenerator` seam unifies them, and the repo already has a
  `global-cut-pool.md` design to build on.
- **Domain reduction inside subproblems.** Each subproblem solve runs FBBT/OBBT
  locally; tightened sub-bounds feed back as tighter cuts to the master (a form of
  in-loop domain reduction).

**Net:** decomposition and global opt should share the B&B tree, the cut pool,
and the OBBT engine — the advisor is not a bolt-on but a *reorganizer* of the
existing global solve.

---

## 13. Parallel execution (Task 11)

The `CommunicationLayer` trait (§3.7, §8.2) lets one coordinator algorithm run on
many backends. This is deliberately modeled on the repo's existing, *bit-
reproducible* Rayon MILP parallelization (`design/rayon-parallelization.md`): a
pure per-task function + a deterministic reduce.

### 13.1 Backends

| Backend | Target | When the scorer picks it |
|---|---|---|
| **Thread pool / Rayon** | shared-memory multicore | default; blocks fit in RAM, low comm cost. Reuses the exact map→reduce pattern already proven in `milp_driver.rs` |
| **Tokio async** | many I/O-bound or heterogeneous subproblem solves (e.g. external solver calls) | subproblems are coarse and call out to HiGHS/IPOPT/BARON |
| **Ray** | distributed cluster, Python-native, elastic | many blocks, memory exceeds one node, dynamic worker pool (DW column pricing scales elastically) |
| **MPI** | tightly-coupled HPC, bulk-synchronous | regular, synchronous iterations (nested Benders/SDDP, PH) on a supercomputer |
| **GPU (JAX vmap/pmap)** | many small **identical** dense blocks | scenarios (PH) or DW subproblems that are isomorphic → batch on GPU; discopt already runs JAX/GPU |
| **Heterogeneous** | CPU master + GPU sub-batch + external MILP | scheduler places each task on its best device |

### 13.2 Scheduling

The `SchedulingGraph` (§4.13) drives a work-stealing scheduler that:
- **Overlaps** master solve with subproblem preparation (pipeline the loop).
- **Aggregates cuts** as subproblems finish rather than at a barrier — a Benders
  master can start incorporating early cuts (asynchronous/partial Benders) when
  the method permits, cutting straggler cost.
- **Load-balances by predicted block cost** (from `block_size` + history), not by
  block index — high block-size variance (§6) is the enemy of parallel
  efficiency, so the scheduler front-loads the biggest blocks.
- **Respects data locality:** place a subproblem where its data/factorization
  already lives across iterations (warm-start reuse dominates constant factors).

### 13.3 Synchronous vs asynchronous

- **Synchronous** (barrier per iteration) is simplest and what classic
  Benders/DW/PH assume; correctness is easy, but stragglers hurt. Good for regular
  MPI/GPU workloads.
- **Asynchronous** (master updates as subproblems trickle in) hides stragglers and
  scales better, but complicates convergence proofs — offer it only for methods
  with async-convergence guarantees (async ADMM, async column generation) and
  keep it behind a flag, defaulting to synchronous for correctness (goal #1).

### 13.4 Determinism

Following the repo's hard-won lesson (the Rayon MILP path is *bit-identical* to
serial), the default parallel decomposition must be **reproducible**: a
deterministic reduce order for cuts/columns so the master sees the same sequence
regardless of worker timing. Nondeterministic async modes are opt-in and clearly
labeled, because a decomposition that gives different answers on different runs
would violate the correctness gate.

---

## 14. Public API (Task 12)

Two layers: an **implicit** path (the vision — `model.solve()` just works) and an
**explicit/introspective** path (`analyze_decomposition()` for control and
teaching). Both use the same engine.

### 14.1 Implicit — decomposition as an automatic transformation

```python
model.solve()                          # advisor runs; may or may not decompose
model.solve(decomposition="auto")      # explicit opt-in to the same behavior
model.solve(decomposition="none")      # force monolithic
model.solve(decomposition="benders")   # force a method (advisor still builds the reformulation)
model.solve(decomposition={            # force method + parameters
    "method": "gbd",
    "complicating": ["y1", "y2"],      # override the split
    "backend": "ray",
    "max_iters": 200,
})
```

`decomposition="auto"` is the intended future default of bare `solve()`, guarded
by a config flag until proven safe on the benchmark suite (the phase gates).

### 14.2 Explicit — the advisor object

```python
advisor = model.analyze_decomposition()          # runs analysis, no solve

advisor.summary()          # StructureReport summary (extends today's .summary())
advisor.recommendation()   # -> Explanation (§9): method, reasons, concerns, estimate
advisor.explain()          # pretty-printed rationale; explain(format="json"|"markdown")
advisor.explain(method="benders")   # counterfactual: why (not) Benders  (§9.3)

advisor.candidates()       # -> list[Candidate], all considered
advisor.scores()           # -> list[(Candidate, ScoreVector)], the full ranking
advisor.blocks()           # -> DecompositionStructure (the EXISTING contract)
advisor.structure()        # -> StructureReport (graphs + metrics)

advisor.graph(kind="var_constraint")   # -> ModelGraph view (§4)
advisor.export_graph(kind, format="graphml"|"json"|"dot"|"metis")

advisor.visualize(kind="block")        # -> figure/interactive (§15)

# commit to a decomposition and get a solvable object:
decomposed = advisor.decompose()              # -> DecomposedModel (advisor's pick)
decomposed = advisor.decompose(method="dw")   # override
result = decomposed.solve(backend="ray", max_iters=100)

# or one-shot:
result = model.solve(decomposition=advisor.recommendation().recommendation)
```

### 14.3 Overrides & annotations (build on existing surface)

The existing model annotations remain the user-override mechanism and now feed the
advisor as the highest-confidence `CandidateGenerator`:

```python
model.first_stage("y")            # existing → complicating var hint
model.mark_coupling("link_row")   # existing → coupling constraint hint
model.set_block("x", 3)           # existing → block assignment hint
# new, optional:
model.decomposition_hint(method="benders", scenarios="s")   # nudge, not mandate
```

Precedence (already the rule in `detect_decomposition`): explicit call args >
model annotations > auto-detection. The advisor honors this and *reports* when it
disagrees with a hint (a `Concern`), rather than silently overriding.

### 14.4 Config

A `[decomposition]` section in `config/benchmarks.toml` / solver config:
budgets (`bridge_scan_budget` already exists), thresholds (`τ_couple`), scoring
weights, default backend, learning on/off, determinism mode. Single source of
truth, matching the repo's `benchmarks.toml` convention.

---

## 15. Visualization (Task 13)

Each visualization answers a specific decision question and links back to the
explanation (§9). Static (matplotlib/graphviz) by default; interactive
(pyvis/plotly) optional, consistent with the repo's Jupyter-Book docs workflow.

| Visualization | What it reveals | Decision it supports |
|---|---|---|
| **Variable–Constraint graph** | raw bipartite structure, hub constraints, degree | "is there structure at all?"; spot coupling hubs |
| **Block graph** | blocks as super-nodes, coupling as super-edges, sized by block cardinality | number/balance of blocks; star vs chain vs mesh coupling → method |
| **Coupling graph / border** | which entities link which blocks; separator size | Benders vs DW vs Lagrangian; how expensive coordination is |
| **Tree decomposition** | separator hierarchy, treewidth | is the model "nearly decomposable"? nested-decomposition depth |
| **Master/subproblem layout** | the reformulated system: master in the center, subproblems around, cut/column arrows | understand the chosen decomposition concretely |
| **Nested hierarchy** | recursive decomposition as a tree (Benders-in-PH etc.) | verify/tune a hybrid strategy |
| **Sparsity heat map** | reordered Jacobian/Hessian showing bordered-block-diagonal form emerging after permutation | *visual proof* the partition produces BBD structure |
| **Interactive explorer** | pan/zoom/filter the variable-constraint graph, click a node → its constraints/blocks/metrics | debugging, teaching, model understanding |
| **Coupling-density heat map** | block×block coupling intensity matrix | identify the few dense couplings to break or dualize |
| **Convergence dashboard** (post-solve) | master bound vs subproblem bound vs incumbent over iterations; per-block wall-clock (straggler view) | diagnose weak cuts / load imbalance; feeds Task 8 |

The reordered sparsity heat map is the single most convincing artifact: showing
the Jacobian permuted into bordered-block-diagonal form makes the decomposition
*self-evidently correct* to a human, and doubles as a regression-test fixture.

---

## 16. Research opportunities (Task 14)

At least fifteen publishable directions, roughly ordered by tractability:

1. **Automatic decomposition discovery as a benchmarked task.** A standardized
   corpus (MINLPLib/MIPLIB-derived) with *ground-truth best decompositions* and a
   leaderboard — none exists rigorously today.
2. **Learning decomposition policies from solver telemetry.** Portfolio/algorithm-
   selection over decomposition methods with the §10 database; measure regret vs
   an oracle.
3. **Graph neural networks for partition & method prediction.** GNN over the
   variable–constraint graph predicting (a) a good partition and (b) the winning
   method end-to-end; compare to METIS + rules.
4. **Compiler-inspired decomposition passes.** Formalize decomposition as a
   *sound rewrite system* over the model IR with a cost model and pass ordering —
   bringing PL/compiler theory (pass scheduling, fixed-point analysis) to MP.
5. **Symbolic decomposition.** Detect decomposable structure directly from the
   *symbolic* expression DAG (shared subexpressions, algebraic separability)
   before any numeric graph is built.
6. **Adaptive / dynamic decomposition.** Change the decomposition *during* B&B as
   bounds tighten and structure changes at nodes — decomposition as a node-level
   decision, not a one-time preprocessing choice.
7. **Incremental decomposition for streaming/re-solved models.** Maintain the
   partition under model edits (§4.15 delta graph) in sublinear time — critical
   for MPC, rolling-horizon, and interactive modeling.
8. **Nested & recursive decomposition auto-discovery.** Automatically finding the
   *depth* and *shape* of a hybrid (Benders-in-PH-in-DW) that a human would never
   hand-design.
9. **Decomposition certificates.** Machine-checkable proofs that a reformulation
   preserves optimality (or bounds it) — a formal-methods contribution that would
   make automatic decomposition *trustworthy* for safety-critical use.
10. **Human-in-the-loop decomposition.** Interactive refinement where the solver
    proposes and the expert edits partitions; measure how little expert input
    yields large speedups (active learning over decompositions).
11. **Relaxation-aware decomposition (the discopt-native angle).** Co-designing
    McCormick/RLT relaxations *and* the decomposition so each strengthens the
    other; quantify the "relax-then-decompose vs decompose-then-relax" tradeoff.
12. **Treewidth-guided guarantees for MINLP.** Provable complexity bounds for
    global optimization parameterized by the treewidth of the (Hessian/KKT) graph.
13. **Optimal complicating-variable selection.** The min-vertex-cut formulation of
    "which variables to fix for Benders" as an optimization problem itself —
    approximation algorithms and their effect on convergence.
14. **Asynchronous decomposition with reproducibility guarantees.** Async ADMM/CG
    that is *both* straggler-tolerant *and* deterministic — squaring the circle the
    repo cares about (bit-reproducibility).
15. **Instance embeddings & meta-learning for optimization families.** Learn a
    latent space of models where proximity predicts shared decomposition strategy;
    few-shot transfer across problem families.
16. **GPU-native decomposition.** Which methods (PH, batched DW) map to batched
    linear algebra on GPU, and a cost model for when GPU decomposition beats CPU —
    directly leveraging discopt's JAX backend.
17. **Decomposition + presolve fixed-point theory.** Prove convergence/quality of
    the iterative presolve↔decompose loop (§11); characterize when iteration helps.

---

## 17. Implementation roadmap (Task 15)

Effort in engineer-weeks (EW), rough. Each phase ships behind a feature flag,
gated by the repo's benchmark/phase-gate machinery, and preserves the
`incorrect_count ≤ 0` invariant.

### Phase 1 — Foundational graph infrastructure  (~6–8 EW, moderate)
- `crates/discopt-core/src/decomp/csr.rs` + zero-copy FFI; `components.rs`
  (union-find CC + Tarjan SCC), `mincut.rs` (articulation/bridges).
- Python `graph/` views over the `Model` DAG; the version-counter cache (§4.15).
- **Refactor** `structure.py` so `detect_decomposition` delegates to CC + bridge
  generators — *behavior-preserving*, guarded by the existing tests.
- Deliverable: `advisor.graph(...)`, `advisor.export_graph(...)`, all cheap
  detectors. **Risk:** low; mostly moving/known algorithms into a clean home.

### Phase 2 — Automatic block detection  (~5–7 EW, moderate)
- `analyzer.py` → `StructureReport`; `candidates.py` with CC/bridge/annotation/
  community(Leiden) generators; METIS/KaHyPar FFI (optional dep, budget-gated).
- Integer-projection + coupling-graph construction; the §5.1 cascade.
- Deliverable: `advisor.candidates()`, `advisor.blocks()` (superset of today).
  **Risk:** medium (external partitioner deps, determinism/seeding).

### Phase 3 — Scoring  (~4–6 EW, moderate)
- `scoring.py` metric registry (§6), the analytic performance model (§6.4),
  `ScoreVector`. Default weights in config.
- Deliverable: `advisor.scores()`; ranked candidates. **Risk:** medium — the
  estimates are only as good as the model; must ship with *honest confidence*.

### Phase 4 — Recommendation engine  (~4–5 EW, low-moderate)
- `selection.py` decision tree (§7) + `policy.py` `RuleBasedPolicy`;
  `explain.py` full `Explanation` (§9) + renderers (text/json/markdown).
- Deliverable: `advisor.recommendation()`, `advisor.explain()`. **Risk:** low;
  it's assembling numbers already computed. High user-visible value.

### Phase 5 — Automatic reformulation  (~10–14 EW, high — the hard core)
- `ir/` (`DecomposedModel`, master/sub/cutgen/coordinator), `methods/` plugins
  that **wrap the existing** `solve_benders`/`solve_gbd`/`solve_lagrangian`
  behind the new traits, then add DW/CG/ADMM/PH.
- `VariableMapping` + `SoundnessCertificate` + equivalence tests.
- Deliverable: `advisor.decompose()`, `model.solve(decomposition="auto")`.
  **Risk:** high — correctness of reformulations is the whole ballgame; needs the
  most test investment (equivalence vs monolith on the benchmark suite).

### Phase 6 — Parallel execution  (~8–12 EW, high)
- `comm.py` backends (Rayon reuse first, then Ray, then MPI, then JAX-batch);
  `SchedulingGraph` + work-stealing scheduler; deterministic reduce.
- Deliverable: `decomposed.solve(backend=...)` with reproducibility. **Risk:**
  high — distributed correctness + determinism; stage backends incrementally,
  Rayon/shared-memory first (reuse the proven MILP pattern).

### Phase 7 — Learning  (~6–10 EW, moderate but open-ended)
- `learning/record.py` telemetry (cheap, ship *early* even if unused — data is
  the asset), `features.py`, `store.py`, then `policies.py` (instance-based →
  bandit → learned ranker → GNN).
- Deliverable: `LearnedPolicy` behind the `Policy` seam; self-calibrating
  confidence. **Risk:** moderate for retrieval/bandit; research-grade for GNN/RL.

**Critical path & sequencing note:** Phases 1→2→3→4 are a clean, low-risk chain
that delivers the *advisor as an analysis/explanation tool* (arguably 80% of the
user value) before any reformulation risk. Phase 5 is the correctness-critical
core and should not start until the benchmark harness can assert
decomposition-vs-monolith equivalence automatically. **Start Phase 7's telemetry
recording in Phase 4** — collecting `SolveRecord`s costs almost nothing and the
learning phases are worthless without accumulated data.

---

## 18. Cross-cutting tradeoffs and challenged assumptions

- **"Decomposition always helps for structured problems" — false.** A model can
  be block-angular yet decompose *worse* than monolithic if coupling is dense,
  blocks are imbalanced, or cuts converge slowly. The scorer's baseline is always
  no-decomp, and the honest answer is frequently "don't." Building the advisor to
  *say no confidently* is as important as saying yes.
- **"More blocks = more parallelism = faster" — misleading.** Coordination and
  straggler cost grow with block count; there's an optimum. `block_size_variance`
  and coupling density cap the benefit well before "one block per core."
- **Automatic vs manual.** Experts sometimes know a decomposition the graphs
  can't see (domain semantics). Hence annotations are first-class and the advisor
  *explains disagreements* rather than overriding — automation augments, not
  replaces, the modeler.
- **Rust/Python split cost.** The FFI boundary adds complexity; justified only for
  the genuinely hot kernels. Policy/scoring/explanation stay in Python where churn
  is high. Resist the urge to push scoring into Rust prematurely.
- **External partitioner dependency (METIS/KaHyPar).** Powerful but non-Rust,
  non-deterministic without seeding, and a packaging burden. Keep optional; ship a
  pure-Rust recursive-spectral fallback so core functionality never depends on it.
- **Determinism vs async speed.** The repo prizes bit-reproducibility; async
  decomposition trades it away. Default synchronous+deterministic; async is opt-in
  and flagged. Do not let a speed win silently break reproducibility.
- **Estimation honesty.** Speedup/efficiency numbers are *estimates*; presenting
  them as certainties would erode trust the first time reality disagrees. Every
  number carries a confidence, and Task 8's predicted-vs-observed logging keeps the
  advisor accountable to its own claims.
- **Scope creep vs the existing drivers.** The temptation is to rewrite
  Benders/GBD/Lagrangian. Don't — wrap them behind the IR traits first, prove
  parity, *then* extend. The existing `DecompositionStructure` contract is the
  compatibility anchor for the whole subsystem.

---

## 19. Appendix: pseudocode

### 19.1 Top-level advisor flow

```python
def analyze_decomposition(model, *, budget=DEFAULT_BUDGET) -> DecompositionAdvisor:
    # 0. presolve/convexity assumed already run (see §11); reuse their results.
    report = StructureAnalyzer(budget).analyze(model)      # builds graphs lazily

    # 1. generate candidates cheap→expensive, stop early on a strong hit (§5.1)
    candidates = []
    for gen in CANDIDATE_GENERATORS:                        # CC, bridge, annot, ...
        if not gen.applicable(report):        continue
        if budget.exhausted():                break
        candidates.extend(gen.generate(model, report))
        if any_strong(candidates, report):    break         # skip expensive gens

    # 2. drop unsound candidates (gatekeepers), then score survivors (§6)
    scored = []
    for cand in candidates:
        if not sound(cand.method, cand, report):  continue  # veto, not weigh
        scored.append((cand, Scorer().score(model, cand, report)))

    # 3. rank via the installed policy (rules | learned | portfolio) (§7, §10)
    ranked = POLICY.rank(scored, SelectionContext(model, report))

    # 4. always include no-decomp as the baseline competitor (§6.3)
    ranked = insert_baseline_no_decomp(ranked, report)

    # 5. build explanation from the SAME numbers (§9)
    explanation = Explainer().explain(ranked, report)
    return DecompositionAdvisor(model, report, ranked, explanation)
```

### 19.2 Reformulate + solve (§8, §13)

```python
def solve_decomposed(model, choice, backend="auto", **cfg) -> SolveResult:
    method = METHODS[choice.method]                         # DecompositionMethod plugin
    verdict = method.can_reformulate(choice.candidate, model)
    if not verdict.ok:
        return solve_monolithic(model, **cfg)               # safe fallback

    dm = method.reformulate(choice.candidate, model)        # -> DecomposedModel (§8.1)
    assert dm.certificate.is_sound()                        # correctness gate (§8.3)

    comm = CommunicationLayer.select(backend, dm.schedule, model)   # §13
    coord = dm.coordinator
    coord.init(dm.master, dm.subproblems, dm.cut_generator, comm)

    while not coord.converged() and not coord.limit_hit():
        master_sol = dm.master.solve()                      # reuse discopt solver
        tasks      = [sp.parameterize(master_sol) for sp in dm.subproblems]
        sub_out    = comm.map(tasks, solve_subproblem)      # parallel (§13), deterministic reduce
        cuts       = dm.cut_generator.generate(sub_out, master_sol)
        dm.master.add(cuts)                                 # cuts and/or columns
        coord.update_bounds(master_sol, sub_out)

    return dm.assemble_result(coord)                        # original var space via var_map
```

### 19.3 The scoring scalarization (§6.3)

```python
def score(model, cand, report) -> ScoreVector:
    m = compute_metrics(model, cand, report)                # §6.2, from cached graphs
    if not sound(cand.method, cand, report):
        return ScoreVector(m, aggregate=-INF, confidence=1.0)
    if m["coupling_density"] > TAU_COUPLE:                  # dense coupling ⇒ don't
        return ScoreVector(m, aggregate=no_decomp_baseline(report) - PENALTY, confidence=0.9)
    w = weights_for(cand.method, target=report.hw_target)   # method/hardware-specific
    benefit = (w.parallel * parallel_efficiency(m)
             + w.localize  * localization(m)
             + w.cuts      * m["expected_cut_strength"])
    cost    = (w.comm * m["est_communication_cost"]
             + w.fill * m["kkt_fill_in"]
             + w.warm * (1 - m["warm_start_potential"]))
    agg = benefit - cost
    return ScoreVector(m, aggregate=agg, confidence=estimate_confidence(m, cand))
```

---

## 20. Implementation status & verification (living section)

Phases 1–7 of §17 are implemented on the initial branch. The subsystem lives in
`python/discopt/decomposition/` (`graph/`, `advisor/`, `ir/`, `parallel/`,
`learning/`) with Rust kernels in `crates/discopt-core/src/decomp/`. The analysis
→ recommendation → reformulation → parallel-plan → record flow is exercised in
`docs/notebooks/decomposition_advisor.ipynb`.

**What was verified where.** Once `feral` was repinned to a crates.io release
(main #375), `discopt-core` and the `_rust` extension build in the dev
environment, so most of what was previously CI-only is now verified here. The one
remaining gap is an end-to-end *solve*, which needs the POUNCE node-LP engine (a
separate GitHub package the egress policy blocks):

| Surface | Verified in the dev environment | Deferred to CI |
|---|---|---|
| Python advisor (graph/advisor/ir/parallel/learning) | ✅ full pytest + ruff + mypy (118 tests) | — |
| Rust graph kernels (`decomp/`) | ✅ in-crate `cargo test -p discopt-core` (15 tests), plus `clippy` + `rustfmt`, edition 2021 / `#![deny(missing_docs)]` | — |
| PyO3 FFI for the kernels (`decomp_bindings`) | ✅ built with maturin; callable from Python via `discopt._rust`, and the Python `articulation_and_bridges` fast path is asserted bit-for-bit equal to the pure-Python reference | — |
| `DecomposedModel.solve()` dispatch | ✅ verified by monkeypatching the drivers (right driver, right `structure`, kwargs threaded) | end-to-end solve equivalence vs the monolith (base MILP solve needs POUNCE, unavailable here) |
| Docs notebook | ✅ every code cell executed during authoring (pure analysis, no solve); citations resolve | `jupyter-book build` (executes all notebooks, needs the full stack) |

**Correctness gate for the remainder.** Per §17, the correctness-critical
equivalence work — proving each reformulation's `solve()` reproduces the
monolithic optimum — must be gated by the benchmark harness's
`incorrect_count ≤ 0` criterion in CI before `solve(decomposition="auto")` becomes
a default. The `SoundnessCertificate` (§8.2) is the in-code guard until then: it
refuses to run a merely-heuristic reformulation.

---

*End of specification. This document defines the architecture; implementation
proceeds per the phased roadmap (§17), each phase gated by the repo's benchmark
and correctness (`incorrect_count ≤ 0`) criteria.*
