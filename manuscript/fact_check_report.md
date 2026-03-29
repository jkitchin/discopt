# Manuscript Fact-Check Report

Date: 2026-02-15 (fixes applied same day)
Reviewers: 3 automated scientific reviewers cross-referencing `manuscript/discopt.tex` against codebase

**Status: ALL ERRORS FIXED, ALL NOTEBOOKS CREATED**

---

## ERRORS (Must Fix)

### E1. "100–226x speedups" should be "17–226x" (Section 3.1, ~line 163)
- **Abstract correctly says "17–226x"** but Section 3.1 says "100–226x"
- LP speedup is 17x (not 100x), QP speedup is 226x
- **Fix**: Change "100–226x" → "17–226x" in Section 3.1

### E2. CUTEst benchmark numbers are outdated (Section 4.2, ~line 433)
- Manuscript: ripopt 552/727 (75.9%), Ipopt 557/727 (76.6%), both solve 519, agree 431
- Actual (from ripopt repo): ripopt **583–586/727 (80.6%)**, Ipopt **558–560/727 (77.0%)**, both solve **544–548**, agree **452–455**
- **Fix**: Update all CUTEst numbers from `RIPOPT_VS_IPOPT.md` / `CUTEST_REPORT.md`

### E3. "identical mean iteration counts (14.5)" is wrong (Section 4.2, ~line 433)
- Actual: ripopt mean = 14.5, Ipopt mean = **13.3** (not identical)
- **Fix**: Change to "comparable mean iteration counts (ripopt: 14.5, Ipopt: 13.3)"

### E4. McCormick function count: 19 → 25 (Section 5.1, ~line 461; also abstract)
- `mccormick.py` has **25** `relax_*` functions (6 added later: atan, asin, acos, sinh, cosh, tanh)
- **Fix**: Update count to 25 in Section 5.1 and abstract

### E5. Table 3 caption says "10 instances" but only 9 rows (~line 748)
- Count: chance, dispatch, ex1221, ex1226, gear, nvs01, nvs03, nvs04, nvs06 = 9
- **Fix**: Change caption to "9 instances" or add the missing 10th

### E6. Couenne lacks a citation (Introduction, ~line 42)
- Couenne is mentioned but never cited
- **Fix**: Add Couenne citation (Belotti et al., 2009, or dedicated reference)

---

## SUSPICIOUS / QUESTIONABLE (Should Address)

### S1. Table 3 "chance" objective 29.421 < known optimum 29.894
- discopt reports 29.421 as "optimal" but this is below the known optimum
- Likely a local minimum that violates constraints, or the known optimum is incorrect
- The test file marks this as xfail: "Non-convex NLP: Ipopt finds local minimum"
- **Fix**: Change status from "optimal" to "feasible" or add footnote about local minimum

### S2. Table 1: SCIP GDP = "Yes" is overstated
- SCIP has indicator constraints but not full GDP (no disjunctions, hull reformulation, logical propositions)
- **Fix**: Change to "Partial" or add footnote

### S3. Table 1: DiffOpt.jl GDP = "Yes" is questionable
- DiffOpt.jl differentiates JuMP models; GDP is not core JuMP functionality
- **Fix**: Change to "No" or "Via extensions"

### S4. Abstract: "9 NLP and 24 MINLP benchmark problems" is misleading
- The 24 problems from test_correctness.py are 12 NLP + 12 MINLP
- The 9 NLP are from ipm_vs_ipopt.ipynb, overlapping with the 12 NLP
- **Fix**: Clarify: "24 correctness problems (12 NLP + 12 MINLP) and 9 NLP backend comparison problems"

### S5. ICNN "approximately 10% gap reduction" — test threshold is only 5%
- `test_learned_relaxations.py` asserts `reduction >= 0.05` (5%), not 10%
- **Fix**: Say "5–10%" or match the test threshold

---

## UNVERIFIED (Need Reproducible Source)

### U1. Piecewise McCormick gap values in Table 4 (~lines 497–510)
- exp: 0.999→0.066 (93.4%), log: 0.254→0.021 (91.9%), x²: 2.664→0.167 (93.7%)
- These specific numbers do NOT appear in any notebook or test
- Tests only assert `reduction >= 0.60` (60%)
- **Action**: Create `manuscript/piecewise_gap_reduction.ipynb`

### U2. Predictor-corrector "30–50% iteration reduction" (~line 334)
- Standard literature claim but not benchmarked in this codebase
- **Action**: Either cite a reference or create benchmark notebook

### U3. Cholesky "approximately 3x faster" than eigenvalue decomposition (~line 338)
- Code comment says "~3x faster" but no benchmark
- **Action**: Create microbenchmark or soften to "substantially faster"

### U4. alphaBB "10–100x speedup" (~line 487)
- Stated in MEMORY.md but no reproducible source
- **Action**: Create benchmark or soften language

### U5. LP/QP speedup numbers (17x, 226x) (~line 772)
- No notebook or script produces these numbers
- **Action**: Create `manuscript/benchmark_lp_qp_speedup.ipynb`

### U6. 36% DFL regret reduction is seed-dependent (~line 665)
- Uses `np.random.seed(42)` with n_train=30, n_test=15 (small samples)
- **Action**: Add multi-seed analysis or confidence interval, or caveat as "single demonstration"

### U7. Table 2 timing numbers are hardware-dependent (~lines 670–696)
- Numbers come from `ipm_vs_ipopt.ipynb` but require execution on specific hardware
- **Action**: Add hardware specification footnote (already partially done with "M4 Pro" mention)

---

## MINOR ISSUES

### M1. Cite key mismatches
- `Grossmann2002` → published 2003
- `Lee2001` → published 2000
- `Nemhauser1999` → published 1988
- **Fix**: Rename cite keys to match publication year

### M2. `set_branch_hints()` method name (~line 216)
- Could not find this exact method in tree_manager.rs
- **Fix**: Verify actual API name and update manuscript

### M3. FBBT "runs at every B&B node" (~line 235)
- Claim not verified against solver.py B&B loop
- **Fix**: Verify and update if incorrect

### M4. OBBT "problems with ≤ 200 variables" threshold (~line 227)
- Threshold not verified in solver.py
- **Fix**: Verify actual threshold in code

---

## REPRODUCIBILITY PLAN

### Existing Notebooks (just need execution)
| Data | Notebook | Notes |
|------|----------|-------|
| Table 2 (NLP backend timing) | `docs/notebooks/ipm_vs_ipopt.ipynb` | Run and verify |
| 36% regret reduction | `docs/notebooks/decision_focused_learning.ipynb` | Add multi-seed |
| 24/24 correctness | `python/tests/test_correctness.py` | pytest run |
| MINLPLib 49/73 | `scripts/phase3_gate.py` | Script run |

### New Notebooks Needed (create in `manuscript/`)
| Data | Proposed Notebook | Description |
|------|-------------------|-------------|
| Table 4 (piecewise gap) | `manuscript/piecewise_gap_reduction.ipynb` | Compute gap reduction for exp, log, x² with N=8 breakpoints |
| LP/QP speedup | `manuscript/benchmark_lp_qp_speedup.ipynb` | Time IPM on LP(n=100) and QP(n=100), cold vs warm |
| Table 3 (MINLPLib smoke) | `manuscript/minlplib_smoke_test.ipynb` | Run 9-10 MINLPLib instances, record objective/status/time |
| Predictor-corrector benefit | `manuscript/mehrotra_benchmark.ipynb` (optional) | Compare iteration counts with/without PC on standard problems |

---

## VERIFIED CLAIMS (No Action Needed)

The following major claims were verified against source code:
- Pure-JAX IPM: augmented KKT, Mehrotra PC, carry-based while_loop, filter line search ✓
- ripopt: L-BFGS, augmented Lagrangian, slack reformulation, 3-stage restoration ✓
- ripopt: HS 119/120 (99.2%) vs 118/120 ✓
- McCormick: pure JAX, jit+vmap compatible ✓
- alphaBB: eigenvalue + Gershgorin methods ✓
- GDP: big-M, hull, SOS, logical propositions ✓
- Convexity detector: composition rules, caching ✓
- DFL: linear model (not neural network), correct QP formulation ✓
- Differentiable solve: L1 envelope, L3 implicit KKT, fallback chain ✓
- DiffSolveResultL3: all 6 methods present ✓
- B&B: reliability branching, pseudocost, strong branching LP ✓
- OBBT: incumbent cutoff, warm-start ✓
- FBBT: Rust implementation with cutoff ✓
- All specialized envelopes: power_int, exp_tight, log_tight, sin_tight, cos_tight, signomial_multi ✓
- ICNN: softplus weights, LearnedRelaxationRegistry ✓
