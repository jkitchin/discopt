# DOE Module Review — Correctness and SOTA

**Date:** 2026-07-03
**Scope:** `python/discopt/doe/` (~13,000 lines, 24 files) — Design of Experiments:
optimal design (A/D/E/ME criteria + FIM), model discrimination, model selection,
estimability/identifiability, profile likelihood, model-based active learning,
classical DOE (ANOVA, fractional factorials), plus GUI/workbook/templates/CLI
plumbing.
**Method:** Delegated verification pass — the load-bearing math read carefully and
checked against closed forms / analytic optima; the D-optimality result
independently re-confirmed here.

**Bottom line: this is the cleanest large module in the review series.** No
correctness bugs in the optimality-criterion, FIM, sensitivity, or statistical
math. The findings are four minor doc/robustness nits, none of which produces a
wrong design or wrong statistic in normal use. Unusually careful for its size.

---

## 1. Verified correct (with evidence)

- **FIM construction** (`fim.py:293-298`): `FIM = Jᵀ Σ⁻¹ J`, `Σ⁻¹ = diag(1/σ²)`.
  Verified: single-point FIM at `x=0.5` is the exact rank-1 outer product
  `[[1,.5],[.5,.25]]`; with `σ=2` a ±1 two-point design gives `diag(.5)` — the
  `1/σ²` weighting applied correctly.
- **D/A/E/ME directions** (`fim.py:63-86`, `design.py:577-586`): D = maximize
  `log det(FIM)`; A = minimize `trace(FIM⁻¹)`; E = maximize `min eigenvalue` (via
  `eigvalsh`, valid for symmetric FIM); ME = minimize condition number.
  `_is_maximization` is True only for D and E — correct. All three recover the
  analytic linear-regression optimum: design points at **x = [−1, +1]** on
  `y = a + b·x`, x∈[−1,1]. Independently re-confirmed here: `det(FIM)` for `[-1,1]`
  is 4.0 vs 1.0 for interior designs — D-optimality correctly places at the
  extremes.
- **Sensitivity Jacobian** (`fim.py:893-902`): JAX `jacobian` matches central FD
  and the closed-form derivative to ~15 digits on `y = exp(−k·t)`.
- **Statistical claims**: `parameter_covariance = FIM⁻¹` (Cramér–Rao) verified;
  Belsley/Gutenkunst identifiability diagnostics (VIF, condition indices,
  variance decomposition, QR null-space) textbook-correct.
- **Discrimination** (`discrimination.py`): Buzzi-Ferraris, Hunter-Reiner, the
  Jensen-Rényi α=2 closed form, and MI nested-MC all derived correctly; prediction
  covariance `V = J·FIM⁻¹·Jᵀ` is the right linearized form.
- **Model selection** (`selection.py`): AIC/BIC/AICc, LRT `G²` on χ², Vuong `z` —
  all correct under the stated deviance convention (the shared constant cancels in
  ranking/weights).
- **Profile likelihood** (`profile.py`): deviance threshold `D(θ̂)+χ²_{1,1−α}`
  (no ½ factor, matching the deviance objective) — correct.
- **Active learning** (`acquisition.py`/`optimize.py`/`model_based.py`): EI/UCB/LCB
  direction handling, incumbent tracking, linearized predictive variance
  `diag(J Σ_θ Jᵀ) + σ²` — all correct.
- **Classical DOE**: balanced Type-I ANOVA (inclusion-exclusion cell SS),
  screening main effects with pooled two-sample SE, and the fractional-factorial
  MILP orthogonality-per-resolution (III/IV/V) are statistically correct.

## 2. Minor findings (all SUSPECTED, non-blocking)

| # | Loc | Finding |
|---|-----|---------|
| DOE-1 | `fim.py:229-231` vs `905-921` | `fd_step` documented as "relative" but `_compute_jacobian_fd` uses an **absolute** perturbation (`x[idx]+step`) — inaccurate sensitivities for large-magnitude params on the non-default `method="finite_difference"` path. Doc/behavior mismatch |
| DOE-2 | `design.py:464-469` (also `discrimination.py:214`, `design.py:521`) | Broad `except Exception` turns *any* `compute_fim_batch` failure into a `None` sentinel, surfacing only as "No feasible design point found" — a real compile/JAX bug is masked. Per the repo's no-swallowed-exceptions rule; the sentinel is always correctly worse than a feasible point, so not a math error |
| DOE-3 | `anova.py:52-53` | Docstring says "two-sided"; the code correctly computes the one-sided upper-tail F-test (`f_dist.sf`). Doc-only |
| DOE-4 | `profile.py:157-160` | Initial-step curvature comment assumes `2·FIM_ii·h²` vs the true Gauss-Newton `FIM_ii·h²` — a factor of 2, but it only sizes the *first* profile step, which is then adaptively corrected, so CIs are unaffected. Benign |

## 3. SOTA

Competitive with and in places ahead of **Pyomo.DoE** (Wang & Dowling): both use
exact autodiff FIM and D/A/E/ME criteria, but discopt adds a broader statistical
toolkit Pyomo.DoE lacks — Belsley/Gutenkunst sloppy-model diagnostics, Yao/Brun
estimability ranking with rank-revealing QR, Chu-Hahn D-optimal subset selection,
five model-discrimination criteria, Raue profile-likelihood CIs, and
AIC/BIC/LRT/Vuong selection. Pyomo.DoE's edge is embedding the FIM directly in a
simultaneous NLP (the stochastic-program formulation); discopt uses multi-start +
scipy L-BFGS-B/SLSQP over the design with a per-point FIM loop — correct, but not a
single monolithic global optimization, so it can miss the global design optimum on
multimodal criteria (mitigated by multi-start and the in-house solver on the
batch-MILP fractional path). Versus JMP/Design-Expert, the classical side is sound
and the model-based/Bayesian side is more advanced than those tools offer.

## 4. Plan (for Opus)

A single small hygiene PR `fix(doe): DOE-1..DOE-4` — make `_compute_jacobian_fd`
relative (or fix the docstring), narrow the batch `except Exception` to log/re-raise
genuine errors while keeping the worse-than-feasible sentinel, correct the two
docstrings (ANOVA one-sided, profile curvature). No math changes required. A
larger, optional SOTA step would be a simultaneous-NLP FIM formulation (Pyomo.DoE
style) for the design optimization to avoid the multimodal-miss risk — design-doc
first.
