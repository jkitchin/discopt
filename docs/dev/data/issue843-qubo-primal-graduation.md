# #843 — DISCOPT_QUBO_PRIMAL graduation record (default-ON)

**Flag:** `DISCOPT_QUBO_PRIMAL` — the QUBO/Ising greedy-1opt + tabu local-search
primal seed (#846, default-OFF at merge).
**Decision:** graduated **default-ON** (opt-out `DISCOPT_QUBO_PRIMAL=0` retained,
legacy no-seed path intact) per the CLAUDE.md §5 panel policy (2026-07-17 revision:
one passing graduation-gate run meeting both bars suffices).
**Regime:** heuristic-policy / purely primal — the seed only ever injects a
verified-feasible incumbent; it never touches a relaxation, bound, or certificate.

## What changed at graduation (beyond the flag flip)

The #846 implementation built the quadratic form through the JAX `NLPEvaluator`
(sparse Hessian structure + values). A default-ON seed runs its gate on **every**
solve, so the heuristic was rebuilt **JAX-free** in `discopt/qubo_primal.py`:

* structural gate `is_qubo` — pure Python/numpy (unconstrained, incl. #840
  builder rows; all-binary boxes);
* degree gate — the JAX-free `problem_classifier.classify_problem` must say
  **MIQP** (a linear objective classifies MILP and is left to the exact, JAX-free
  simplex B&B; degree > 2 classifies MINLP, where the constant-Hessian 1-flip
  delta would be wrong — the #846 code did not enforce this);
* quadratic form via the `extract_qp_data` ladder (builder-repr → algebraic DAG
  walk → Rust-repr probe; the autodiff **JAX** rung is a last resort reached only
  if both JAX-free extractors fail). The repr-probe rung also *widens* coverage
  vs #846-era algebraic-only extraction: affine-product objectives like
  `Σ J(2xᵢ−1)(2xⱼ−1)` (the raw Ising spin form) now extract and seed.
* deadline is polled every 256 tabu iterations (not just between starts), so a
  default-ON seed cannot blow a tight budget on a huge instance (#863 lesson).

Cold-start invariants preserved and newly pinned by
`python/tests/test_843_qubo_primal.py`:
`test_843_qubo_search_is_jax_free` (the search itself, subprocess) and
`test_843_unconstrained_binary_milp_solve_stays_jax_free` (the nearest-miss
model — unconstrained all-binary *linear* — solves end-to-end JAX-free with the
seed ON). `test_lazy_jax_linear_path` (LP/MILP/QP/MIQP) stays green.

## Panel design

Instrument: `discopt_benchmarks/scripts/issue843_qubo_primal_graduation_panel.py`
(executed-assertion counts printed; zero assertions = non-zero exit, §6).

The flag's structure (unconstrained ∧ all-binary ∧ quadratic objective) appears
in **zero** instances of the vendored corpus, so the cert arm is:

* **N1** — executed structural no-fire proof: `is_qubo` + `qubo_local_search`
  over every vendored `.nl` (must be False/None everywhere);
* **N2** — interleaved ON-vs-OFF differential solve over a fast corpus subset,
  requiring identical (status, objective, node_count).

The net-positive arm draws from the QUBO class itself (§2: the class, not the
named probe — `chimera_k64ising-*.nl` is not vendored and this container has no
corpus access):

* **S1** — small dense QUBOs vs a **brute-force oracle** (soundness against the
  true optimum);
* **S2** — interleaved OFF/ON full solves of generated **chimera-topology**
  Ising instances (C_s = s×s grid of K4,4 cells, ±1 couplers — the D-Wave
  topology of the `chimera_k64ising` family; C12 = 1152 binary vars ≈ the named
  instance's 1192) plus a dense n=60 QUBO. Metric: incumbent presence/quality
  at the time limit; soundness: MAXIMIZE incumbent ≤ the run's certified dual
  bound.

## Entry probe (gap reproduction on the class)

Generated chimera-topology Ising, seed 7, interleaved OFF→ON, one solve each:

| instance | arm | status | incumbent | dual bound | nodes | wall |
|---|---|---|---|---|---|---|
| C8 (512 vars, 1472 edges), tl=60s | OFF | time_limit | **None** | 1455.0 | 55 | 60.6s |
| C8 | ON | feasible | **846.0** | 1472.0 | 3 | 62.6s |
| C12 (1152 vars, 3360 edges), tl=120s | OFF | time_limit | **None** | 3350.0 | 13 | 121.9s |
| C12 | ON | feasible | **1770.0** | 8880.0 | 3 | 124.8s |

The exact `chimera_k64ising-01` failure mode (#843: "no incumbent") reproduces
on generated class members at ≥512 vars, and the seed closes it with a sound
incumbent (≤ dual bound in every run). Standalone seed cost at C12: **1.45 s**
(extraction + 12-start tabu search) — noise against the 120 s solve.
The named-instance measurement stands from #846's entry experiment (recorded in
PR #846): chimera-01 incumbent **7.2** (was none), sound vs optimum 24.3.

Note the node counts: with an incumbent injected the ON arm processes fewer,
more expensive nodes in the same wall budget. Node drift on structure-carrying
instances is expected for a heuristic-policy flag (the graduation_gate.py
convention); the neutrality requirement applies where the structure is absent
(N2), and there it is byte-identical.

## Panel result — PASS (2026-07-28, exit 0, 87 executed assertions)

JSON (vendored): `docs/dev/data/issue843-graduation-panel-2026-07-28.json`.

* **N1** — 67/67 vendored `.nl` instances checked; the QUBO gate fired on **0**
  (as required — no corpus instance is unconstrained+all-binary).
* **N2** — 8/8 fast-subset instances **byte-identical** ON vs OFF
  (status, objective, node_count): gbd, alan, ex1222, st_e13, st_miqp1,
  st_miqp2, st_test1, nvs04.
* **S1** — 5/5 small dense QUBOs (n=12) certified `optimal` at exactly the
  brute-force optimum with the seed ON (27, 20, 30, 23, 39).
* **S2** — interleaved OFF/ON differential, all four strictly better ON,
  0 soundness violations (every incumbent ≤ its run's dual bound):

| instance | OFF incumbent | ON incumbent | OFF nodes | ON nodes |
|---|---|---|---|---|
| chimera_c4 (128 vars, tl 30s) | 22.0 | **208.0** | 621 | 511 |
| dense_n60 (tl 30s) | **None** | **326.0** | 15 | 7 |
| chimera_c8 (512 vars, tl 60s) | 12.0 | **846.0** | 62 | 3 |
| chimera_c12 (1152 vars, tl 120s) | **None** | **1770.0** | 18 | 3 |

Verdict: **cert-clean AND net-positive** → graduated default-ON.

## Provenance

* Worktree: `claude/discopt-issue-843-3r0s14` (editable install; `discopt.__file__`
  under `/home/user/discopt/python`), JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1.
* 4-core container, load average 0.25 before the panel (no competing jobs; the
  suites were run after, not during, the panel — §9).
* Wall times are single-run and reported for context only; no timing *claims*
  are made beyond incumbent presence/quality, which is deterministic given the
  fixed seeds.
