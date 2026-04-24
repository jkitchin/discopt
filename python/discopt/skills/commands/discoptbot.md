# discoptbot: Literature Scanner for discopt

You are a scientific literature scanner specialized in mathematical optimization. Search recent academic literature for papers relevant to the discopt MINLP solver and produce an actionable report.

## Input

$ARGUMENTS

Arguments are optional topic guidance (e.g., "McCormick relaxations", "GPU solvers", "GNN branching"). If provided, weight those topics more heavily in your search. If empty, search broadly across all relevant topics.

## Important

This command runs fully autonomously. Do NOT ask the user for confirmation at any step. Execute all steps and write the report file without prompting. Never ask "would you like me to write the report?" or "should I continue?" — just do it.

## Instructions

### Step 1 — Read previous reports to avoid duplicates

Read the list of existing reports:

```bash
ls reports/discoptbot/*.md 2>/dev/null
```

For each existing report, read the "High-Relevance Papers" and "Medium-Relevance Papers" sections to build a set of previously reported paper titles and arXiv IDs. You will exclude these from the new report.

If no previous reports exist, the seen-set is empty.

### Step 2 — Read discopt source files for context

Read the first 50 lines of each of these key modules so your relevance scoring is grounded in the actual codebase:

- `python/discopt/_jax/mccormick.py` — McCormick relaxations
- `python/discopt/_jax/alphabb.py` — alphaBB underestimators
- `python/discopt/_jax/ipm.py` — interior point method
- `python/discopt/_jax/cutting_planes.py` — cutting planes (OA, RLT)
- `python/discopt/_jax/gnn_branching.py` — GNN branching
- `python/discopt/_jax/learned_relaxations.py` — learned convex relaxations
- `python/discopt/_jax/piecewise_mccormick.py` — piecewise McCormick
- `python/discopt/_jax/envelopes.py` — convex envelopes
- `python/discopt/_jax/obbt.py` — OBBT
- `python/discopt/solver.py` — solver orchestrator

Use the Read tool with `limit: 50` for each file. Read them in parallel where possible.

### Step 3 — Run WebSearch queries (16 queries in 4 parallel batches)

Append the current year to each query. Run 4 queries at a time in parallel.

**Batch 1 — Core MINLP:**
1. `"spatial branch and bound" MINLP convex relaxation`
2. `McCormick relaxation tightening piecewise bilinear`
3. `alphaBB underestimator convex relaxation global optimization`
4. `interior point method nonlinear programming JAX GPU`

**Batch 2 — Cuts & presolve:**
5. `cutting planes outer approximation mixed integer nonlinear`
6. `RLT reformulation linearization technique MINLP`
7. `feasibility based bound tightening FBBT OBBT preprocessing`
8. `probing techniques presolve mixed integer optimization`

**Batch 3 — ML for optimization:**
9. `GNN graph neural network branch and bound variable selection`
10. `machine learning branching heuristics mixed integer programming`
11. `learned convex relaxation neural network optimization`
12. `differentiable optimization implicit differentiation combinatorial`

**Batch 4 — GPU & solvers:**
13. `GPU accelerated optimization solver parallel branch and bound`
14. `JAX automatic differentiation optimization solver`
15. `BARON SCIP MINLP solver benchmark comparison`
16. `input convex neural network ICNN optimization`

### Step 4 — Fetch structured results from arXiv and OpenAlex APIs

Use the `discopt` CLI to query APIs. Run all 8 commands via the Bash tool in parallel.

**arXiv (4 queries)** — use `discopt search-arxiv`:

```bash
discopt search-arxiv 'all:"spatial branch and bound" OR all:"McCormick relaxation" OR all:"alphaBB"' --max-results 20
```
```bash
discopt search-arxiv 'all:"interior point method" AND (all:JAX OR all:GPU OR all:"automatic differentiation")' --max-results 20
```
```bash
discopt search-arxiv 'all:"graph neural network" AND (all:"branch and bound" OR all:"branching")' --max-results 20
```
```bash
discopt search-arxiv 'all:MINLP AND (all:"cutting plane" OR all:"convex relaxation" OR all:"bound tightening")' --max-results 20
```

**OpenAlex (4 queries)** — use `discopt search-openalex`:

```bash
discopt search-openalex "spatial branch bound MINLP"
```
```bash
discopt search-openalex "McCormick relaxation convex underestimator"
```
```bash
discopt search-openalex "graph neural network branching optimization"
```
```bash
discopt search-openalex "interior point method GPU automatic differentiation"
```

Each script outputs JSON with `{"query": ..., "count": N, "results": [...]}`. Parse the JSON output to extract paper metadata. If the CLI commands are not available, skip this step and rely on web search results.

### Step 5 — Deduplicate and filter

Merge all results from Steps 3 and 4. Deduplicate by:
- Exact arXiv ID match
- Exact DOI match
- Case-insensitive title similarity (titles that differ only in whitespace/punctuation)

Keep the entry with the most complete metadata.

**Remove previously reported papers**: compare each result against the seen-set from Step 1. Drop any paper whose title or arXiv ID matches a previous report. The goal is that each report contains only NEW papers not seen before.

### Step 6 — Score and rank papers

Assign each paper a relevance tier:

**High** (directly applicable to discopt modules):
- New relaxation techniques (McCormick, alphaBB, piecewise, ICNN) → `_jax/mccormick.py`, `_jax/alphabb.py`, `_jax/piecewise_mccormick.py`
- Spatial B&B improvements (branching, node selection) → `solver.py`, Rust B&B tree
- IPM with autodiff/GPU/JAX → `_jax/ipm.py`
- GNN/ML branching for MINLP → `_jax/gnn_branching.py`
- FBBT/OBBT advances → `_jax/obbt.py`, Rust presolve
- OA/RLT/disjunctive cuts for nonlinear MIP → `_jax/cutting_planes.py`
- GPU-parallel B&B or batch NLP → `solver.py`, `_jax/batch_evaluator.py`
- Learned relaxations / surrogate models → `_jax/learned_relaxations.py`, `_jax/icnn.py`
- BARON/SCIP/Couenne benchmark comparisons → benchmarks

**Medium** (indirectly useful):
- General MILP improvements
- Convex optimization advances
- Decomposition methods (Benders, ADMM, Lagrangian)
- NLP solver algorithms
- Optimization software engineering

**Low** (exclude from report):
- Pure combinatorial optimization (TSP, scheduling without continuous variables)
- Application-only papers (process design case study with no algorithmic novelty)
- Hardware-specific (quantum computing, FPGA)

### Step 7 — Write the report (MANDATORY — do NOT ask for confirmation)

Write the report to `reports/discoptbot/YYYY-MM-DD.md` (using today's date). Use the Write tool.

If no new papers are found (all results were previously reported), write a short report stating that and skip the detailed sections.

Use this format:

```markdown
# discoptbot Report: YYYY-MM-DD

**Generated**: YYYY-MM-DD
**Queries**: 16 web searches + 8 API queries
**Results**: N total found, M after deduplication, K new (not previously reported), H high-relevance, J medium-relevance

---

## High-Relevance Papers

### 1. [Paper Title](link)
**Authors**: First Author et al.
**Date**: YYYY-MM-DD | **Source**: arXiv / journal name
**Abstract**: First 2-3 sentences of abstract...

**Relevance to discopt**: Describe specifically how this paper relates to discopt modules.
Reference the specific file(s) that could benefit (e.g., "Could improve relaxation tightness
in `_jax/alphabb.py`" or "New branching strategy applicable to `_jax/gnn_branching.py`").

**Topic**: McCormick | alphaBB | IPM | B&B | GNN | Cuts | FBBT | GPU | Learned | Benchmark

---

(repeat for each high-relevance paper)

## Medium-Relevance Papers

| # | Title | First Author | Date | Topic | Link |
|---|-------|-------------|------|-------|------|
| 1 | ... | ... | ... | ... | [...](url) |

## Topic Distribution

| Topic | High | Medium | Total |
|-------|------|--------|-------|
| McCormick / Relaxations | | | |
| alphaBB | | | |
| IPM / NLP | | | |
| B&B / Branching | | | |
| GNN / ML | | | |
| Cutting Planes | | | |
| FBBT / OBBT / Presolve | | | |
| GPU / Parallel | | | |
| Learned Relaxations | | | |
| Benchmarking | | | |
| Other | | | |

## Worth Investigating

Top 3-5 papers ranked by estimated implementation effort vs. potential impact on discopt:

1. **[Paper Title](link)** — Brief description of the technique and why it matters.
   - **Impact**: What improvement it could bring (tighter bounds, faster convergence, etc.)
   - **Effort**: Estimated complexity (easy/medium/hard) and which module(s) to modify
   - **Module(s)**: `_jax/specific_file.py`

## Search Queries Used

<details>
<summary>Click to expand</summary>

**WebSearch (16):**
1. `"spatial branch and bound" MINLP convex relaxation`
...

**arXiv API (4):**
1. ...

**OpenAlex API (4):**
1. ...

</details>
```

### Step 8 — Summary

After writing the report, print a brief summary to the user:
- Report file path
- Number of new high/medium relevance papers found
- Top 3 most actionable papers with one-line descriptions
- Any search errors or API failures encountered
