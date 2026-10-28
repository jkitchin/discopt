# discopt Benchmark Report: global50

**Generated:** 2026-10-28 16:34:14
**Benchmark timestamp:** 2026-10-28T16:23:55.023462
**Instances:** 50
**Solvers:** discopt, scip

## Summary

| Metric | discopt | scip |
|--------|--------|--------|
| **Solved (global opt)** | 48 | 44 |
| Solved (≤10 vars) | 33 | 32 |
| Solved (≤30 vars) | 43 | 39 |
| Solved (≤50 vars) | 45 | 41 |
| Solved (≤100 vars) | 48 | 44 |
| **Geom. mean time (s)** | 0.57 | 0.35 |
| **Incorrect results** | ✅ 0 | ✅ 0 |

## Pairwise Comparison (discopt vs Others)

| Comparison | Geom. Mean Ratio | discopt Faster | Other Faster | Common Solved |
|------------|------------------|-----------------|--------------|---------------|
| vs scip | 0.91x | 17 | 27 | 44 |

## Global Optimality

| Metric | discopt | scip |
|--------|--------|--------|
| Proved optimal | 48 | 44 |
| Proved optimal rate | 97.9% | 91.7% |
| Mean final gap | 1.38% | 0.00% |
| Median final gap | 0.00% | 0.00% |
| Bound quality rate | 64.6% | 62.5% |

## Results by Problem Class

| Class | Instances | discopt solved | scip solved |
|-------|-----------|--------|--------|
| nonconvex_minlp | 21 | 21/21 | 21/21 |

## Root Gap Analysis

| Solver | Mean Root Gap | Median Root Gap | Max Root Gap |
|--------|-------------- |-----------------|-------------|
| discopt | 0.6137 | 0.0770 | 10.7940 |
| scip | nan | nan | nan |

## Layer Profiling (discopt)

Layers measured on 50 run(s). `rust` + `python` partition the 
wall clock; `JAX` is a subset of Python and `POUNCE` a subset of Rust, 
so the rows do not sum to 100%.

| Layer | Mean Time Fraction | Target |
|-------|-------------------|--------|
| Rust (tree, LP, sparse LA) | 39.8% | — |
| ↳ POUNCE (NLP/LP/QP subsolves) | 36.1% | — |
| Python orchestration | 60.2% | <5% |
| ↳ JAX (relaxations, autodiff) | 0.0% | — |
| Max Python overhead | 99.8% | <10% |

## Node Throughput

- **Mean:** 205 nodes/sec
- **Median:** 56 nodes/sec
- **Min:** 0 nodes/sec
- **Max:** 2236 nodes/sec

## Hardest Instances (by solve time)

| Instance | Time (s) | Nodes | Root Gap | Vars |
|----------|----------|-------|----------|------|
| cvxnonsep_nsig30 | 35.8 | 13711 | 0.0011 | 30 |
| cvxnonsep_psig30 | 34.8 | 14649 | 0.0032 | 30 |
| nvs05 | 17.1 | 89 | 0.8749 | 8 |
| st_e36 | 3.9 | 109 | 0.2378 | 2 |
| nvs09 | 3.2 | 31 | 0.2444 | 10 |
| tspn05 | 1.3 | 23 | 0.1227 | 20 |
| clay0303hfsg | 1.3 | 182 | — | 99 |
| ex1226 | 1.3 | 5 | 0.2353 | 5 |
| ex1225 | 0.9 | 5 | 0.0770 | 8 |
| tanksize | 0.9 | 519 | 0.3375 | 47 |
| nvs01 | 0.8 | 5 | 0.6005 | 3 |
| nvs13 | 0.4 | 23 | 0.0065 | 5 |
| nvs08 | 0.4 | 5 | 0.3065 | 3 |
| cvxnonsep_psig40r | 0.4 | 8 | — | 82 |
| tls2 | 0.3 | 19 | — | 37 |
| st_e29 | 0.3 | 5 | 0.0258 | 11 |
| ex1224 | 0.3 | 5 | 0.0258 | 11 |
| gkocis | 0.3 | 5 | 2.2759 | 11 |
| flay03m | 0.3 | 242 | — | 26 |
| oaer | 0.2 | 3 | 0.3685 | 9 |
