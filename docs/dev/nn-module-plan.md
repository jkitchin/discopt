# NN-module remediation plan (`python/discopt/ml/`)

**Date:** 2026-07-03
**Status:** proposed (from a full-module review, 2026-07-03: all 17 files ~2,100
lines + `test_nn_formulations.py` read; wiring claims verified against the
modeling API and presolve pipeline). Every finding below carries a file:line
anchor from that review.
**Scope:** the NN/tree embedding layer only (`python/discopt/ml/`). No solver-core
changes, so the certification-gap-plan baselines (`cert-baseline.jsonl`) are not
implicated; the ground truth here is *model equivalence* — the embedded
formulation must agree with the trained predictor it claims to encode.
**Relationship to existing docs:**
- `docs/dev/correctness-issues.md` — findings F1–F4 below are certificate-class
  for the modeling layer ("global optimum over the surrogate" can be wrong);
  T-N4.3 files them as C-cards so the ledger stays complete.
- `docs/dev/certification-gap-plan.md` — independent; may proceed in parallel.
- `docs/notebooks/nn_embedding.ipynb` — must be re-verified after T-N0.2 (its
  scaling examples exercise the fixed semantics).

---

## 0. Implementation contract (binding on the implementing agent)

1. **Execute tasks in the §3 order.** T-N0.1 (the equivalence harness) lands
   first — it is the failing-test generator for every fix that follows. Within
   a wave, tasks are independent and may be parallel PRs.
2. **Every fix PR contains its regression test, failing before / passing
   after.** For F1/F2 the failing test is an end-to-end solve-and-compare case
   built on the T-N0.1 harness — not a "builds without error" assertion.
3. **Refuse loudly, never approximate silently.** A reader that cannot faithfully
   represent a model raises `ValueError`/`TypeError` with a message naming what
   is unsupported. Do not drop attributes, default missing structure to zeros,
   or embed logits while claiming probabilities. (House rule; the review found
   four silent-substitution paths.)
4. **Per-PR gates:** `ruff check` + `ruff format --check`, `mypy python/discopt/`,
   `pytest python/tests/test_nn_formulations.py python/tests/test_nn_equivalence.py`,
   `pytest -m smoke`, and the adversarial suite
   (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`). State
   results in the PR description. One task per PR where practicable; name the
   task ID (e.g. `fix(nn): T-N0.2 …`).
5. **Wave N3 (completeness) is opt-in.** Do not start any T-N3.x without
   maintainer approval; T-N3.2 and T-N3.4 additionally require their entry
   experiment to be run and recorded here first.
6. If a measurement or reproduction contradicts a claim in this plan, the
   measurement wins — record a dated correction in the affected task and
   re-scope before coding on.

### 0.1 Status table (update in the same PR as the work)

Implemented 2026-07-03 on branch `nn-module-remediation` (Waves 0/1/2/4). Wave 3
is deferred per §0.5 (opt-in; needs maintainer approval + per-task entry
experiments). Verification: full nn suite 160 passed together; `-m smoke` 193
passed / 1 skipped; ruff + format clean; regression tests confirmed
fail-before/pass-after for F1 (8 cases) and F2 (2 cases).

| Task | Wave | Status | PR | Notes |
|---|---|---|---|---|
| T-N0.1 equivalence harness | 0 | **done** | — | `test_nn_equivalence.py`; 9 baseline cases green. Tree fixed-input tol 1e-5 (solver integrality floor), recorded per §0.6 |
| T-N0.2 scaling/bounds domain fix | 0 | **done** | — | F1. `propagate_bounds(input_bounds=…)` keyword; scaled-domain propagation in full/relu/reduced; `tighten_network` hack removed. 8 regression cases fail-before/pass-after |
| T-N0.3 tree big-M validity + tightening | 0 | **done** | — | F2. Per-constraint tight M in `tree_ensemble.py`. 2 regression cases fail-before/pass-after |
| T-N0.4 output bounds | 0 | **done** | — | Sign-safe scaled-output + tree-output bounds; harness proves no reachable prediction excluded |
| T-N1.1 ONNX reader hardening | 1 | **done** | — | F3. alpha/beta applied, transA raises, MatMul weight=input[1], dataflow-chain check → non-sequential raises. onnxruntime oracle tests |
| T-N1.2 sklearn semantics | 1 | **done** | — | F4. `out_activation_` handling; classifiers raise for trees/ensembles; single-leaf reshape crash fixed |
| T-N1.3 torch bias=False | 1 | **done** | — | `_linear_wb` helper; bias=None → zeros |
| T-N2.1 reduced_space honesty + bounds | 2 | **done** | — | Bounded intermediates (scaled domain), redundant final z removed, docstrings corrected |
| T-N2.2 NNPresolvePass: park or wire | 2 | **done (parked)** | — | Truthful v0 docstring; bare except narrowed to ValueError + logger.debug; v1 wiring left as noted future work. §6 E1 (2026-11-22): stable-neuron removal is low-value, LP tightening almost never fixes a neuron |
| T-N2.3 add_predictor ergonomics | 2 | **done** | — | Bounds harvested from inputs var; length validation; missing file → FileNotFoundError |
| T-N3.1 LeakyReLU/Clip activations | 3 | deferred (opt-in) | — | needs maintainer approval |
| T-N3.2 LP/OBBT big-M tightening | 3 | **done** (`DISCOPT_NN_LP_BIGM`) | — | `ml.bounds.lp_tighten_bounds`, Neumaier–Shcherbina-safe LP bounds. §5 panels (2026-11-23): default ON for `relu_bigm` (each certified net faster, every uncertified bound tighter); OFF for `relu_extended`, whose spread panel showed an incumbent loss, and for `relu_partition`, whose own panel showed a 2x slower certification and two looser bounds (opt-in `=1`). Numbers in `flag-retirement-audit.md` |
| T-N3.5 ICNN as an LP (`relu_icnn`) | 3 | **done** | — | §6 E2: no binaries, exactness checked per result (repair onto the graph + `verify_point`), refused otherwise. Median 8.6× over `relu_bigm` on E2's ICNNs |
| T-N3.3 XGBoost/LightGBM readers | 3 | deferred (opt-in) | — | |
| T-N3.4 Mišić split encoding + multi-output | 3 | deferred (opt-in + entry exp.) | — | |
| T-N4.1 coverage omit removal | 4 | **done** | — | onnx_reader.py un-omitted; onnx/onnxruntime added to `[dev]` extra |
| T-N4.2 CLAUDE.md refresh | 4 | **done** | — | nn paragraph lists full module surface |
| T-N4.3 correctness-issues C-cards | 4 | **done** | — | F1→C-25 (P1), F2→C-26 (P1), F3→C-27 (P2), F4→C-28 (P2) |
| T-N4.4 notebook re-verification | 4 | **done** | — | `nn_embedding.ipynb` uses no scaling; executes clean (exit 0), no change needed |

---

## 1. Findings being addressed (from the 2026-07-03 review)

Correctness (ranked):
- **F1 — Scaling + bound-propagation domain mismatch** (`formulations/full_space.py:72`,
  `formulations/relu_bigm.py:73`). `propagate_bounds(net)` runs on
  `net.input_bounds` (user/unscaled domain — those bounds are applied to the
  unscaled `inputs` var) while layer 1 consumes `scaled_in = (inputs −
  x_offset)/x_factor`. All `zhat`/`z` variable bounds and ReLU big-M constants
  are therefore computed from the wrong box. Any nontrivial `x_offset`/`x_factor`
  can make the model infeasible or silently cut the true optimum. Untested: every
  scaling test uses offset 0 / factor 1.
- **F2 — Tree big-M invalid for out-of-box thresholds**
  (`formulations/tree_ensemble.py:79,98-111`). `M_j = ub_j − lb_j` keeps the
  leaf constraints inert only when `lb_j ≤ thr ≤ ub_j − eps`. A threshold
  outside the declared feature box (optimization bounds tighter than training
  data — common) makes a non-selected leaf's constraint cut feasible points.
- **F3 — ONNX reader silent mis-reads** (`readers/onnx_reader.py`).
  (a) Gemm `alpha`/`beta`/`transA` ignored (:104-121); (b) `MatMul → Add` with a
  non-initializer Add (e.g. residual) consumes the Add and substitutes zero
  biases (:74-79); (c) no dataflow verification — a branched graph of supported
  ops parses into a wrong sequential net.
- **F4 — sklearn classifier semantics** (`readers/sklearn_reader.py`).
  `load_sklearn_mlp` ignores `out_activation_` (MLPClassifier embeds logits);
  `load_sklearn_ensemble` on GradientBoostingClassifier gets `base_score`
  silently wrong (classifier `init_` has no `constant_`). Docstrings overclaim
  classifier support.
- **F5 — crash edges:** `torch_reader.py:48` assumes `Linear.bias` is not None;
  a single-leaf sklearn tree makes `t.value.squeeze()` 0-d and
  `DecisionTree.__post_init__`'s `len()` raises (`sklearn_reader.py:66`,
  `tree.py:49-53`).

Design: reduced_space is full_space-minus-bounds, not "no intermediate
variables", and its intermediates are unbounded (`reduced_space.py:112-118`);
`NNPresolvePass` is unwired dead code with an overclaiming docstring and a
swallowed `except Exception` (`presolve.py:172,236`; verified: no non-test
constructor, `nn_implications` has zero consumers); missing free output bounds
(scaled outputs, tree ensemble output); `add_predictor` duplicates input vars
and cannot harvest bounds from the user's variables (`predictor.py:97-103`).

Tests/docs: **no end-to-end solve test exists** — nothing checks the embedded
optimum against `net.forward()`/`ensemble.predict()`; `onnx_reader.py` is
excluded from coverage (`pyproject.toml:263`); CLAUDE.md's nn section is stale.

---

## 2. Verification principle

The product of this module is the equivalence
`solve(embedded model) ≡ trained predictor`. Every task is verified against
that, two ways:

1. **Fixed-input equivalence:** constrain the formulation's inputs to sampled
   points (equality constraints or `lb=ub`), solve, and compare the output
   variables to the reference prediction
   `predict(x) = y_factor · net.forward((x − x_offset)/x_factor) + y_offset`
   (identity when no scaling), to `1e-6` for MILP-exact encodings (relu_bigm,
   trees) and `1e-4` for smooth NLP encodings.
2. **Optimization equivalence:** minimize/maximize an output over the input box
   and compare the certified optimum against dense enumeration (grid + random
   points, tiny nets only — 1–2 inputs, ≤ 5 hidden units) with the guarantee
   `certified ≤ min(enumerated) + tol` **and** the incumbent's inputs
   re-evaluate through `forward()` to the reported objective. A certified
   optimum *below* every enumerated point beyond tolerance is the
   cut-the-optimum failure mode F1/F2 produce — the harness must detect exactly
   that.

---

## 3. Task breakdown

### Wave 0 — correctness (do first)

**T-N0.1 — Equivalence test harness (prerequisite; ~1 day).**
New file `python/tests/test_nn_equivalence.py` with two reusable helpers (module
level, imported by later regression tests):
- `assert_embedding_matches(net_or_ensemble, *, strategy, scaling=None,
  n_points=8, tol)` — fixed-input equivalence (§2.1). For trees, reference is
  `ensemble.predict`. Sample points inside the input box with a fixed seed;
  avoid sampling within `split_eps` of any tree threshold (the documented
  eps-exclusion band is not a bug).
- `assert_optimum_matches(net_or_ensemble, *, strategy, sense, tol)` —
  optimization equivalence (§2.2).
Initial coverage (all should pass on current code — this PR is harness +
green baseline): relu_bigm / full_space / reduced_space on the existing helper
nets (`_make_relu_net`, `_make_sigmoid_net`, `_make_tanh_net`), tree ensemble
from `_make_tree_ensemble`, identity scaling. Mark the cheapest cases
`@pytest.mark.smoke`. Deliberately do **not** include non-identity scaling or
out-of-box thresholds here — those are the F1/F2 regression cases and belong in
the fix PRs (contract §0.2).

**T-N0.2 — Fix F1: propagate bounds in the scaled domain (~1 day).**
- Add an optional override to `propagate_bounds(network, input_bounds=None)`
  (`bounds.py:42`) — when given, use it instead of `network.input_bounds`.
  Refactor `tighten_network` (`presolve.py:131-148`) to use the new argument,
  deleting its mutate-and-restore hack.
- In `full_space.py` and `relu_bigm.py`: when `self._scaling is not None`,
  compute `s_lo`/`s_hi` (already done for the scaled-var bounds, lines ~84-89)
  **before** propagation and call `propagate_bounds(net, input_bounds=(s_lo,
  s_hi))`. No behavior change when scaling is None or identity.
- **Regression tests (fail before, pass after):**
  `assert_embedding_matches` and `assert_optimum_matches` with
  `OffsetScaling(x_offset=[100,…], x_factor=[0.5,…])` and with a negative
  `x_factor`, for both `full_space` (sigmoid net) and `relu_bigm` (relu net).
  Before the fix these are infeasible or certify a wrong optimum.

**T-N0.3 — Fix F2: valid + tight tree big-M (~0.5–1 day).**
- `tree_ensemble.py:98-111`: replace the per-feature `M_j = ub_j − lb_j` with
  per-constraint coefficients:
  left split → `x_j ≤ thr + max(ub_j − thr, 0)·(1 − z)`;
  right split → `x_j ≥ (thr + eps) − max(thr + eps − lb_j, 0)·(1 − z)`.
  These are inert for `z=0` on any threshold position (clamped-to-0 cases are
  inert because the variable bounds already dominate), correctly make
  box-unreachable leaves infeasible when selected, and are strictly tighter LP
  relaxations than the current constants. Drop the now-unused `feat_range`.
- **Regression tests:** an ensemble with one threshold `> ub_j − eps` and one
  `< lb_j`; `assert_embedding_matches` + `assert_optimum_matches` vs
  enumeration (fails before the fix: feasible points are cut). Plus a
  no-behavior-change check on `_make_tree_ensemble` (in-box thresholds).

**T-N0.4 — Free output bounds (~0.5 day).**
- Scaled outputs (`full_space.py:160`, `relu_bigm.py:143`,
  `reduced_space.py:124`): bound `outputs[j]` by
  `[min, max](y_factor_j · post_lb/ub) + y_offset_j` from the last layer's
  propagated bounds (sign-safe min/max; only when bounds were propagated).
- Tree output (`tree_ensemble.py:125`): bound by
  `Σ_t [min, max leaf value of t] + base_score`.
- **Test:** bounds contain all sampled predictions (reuse harness points);
  equivalence unchanged. This is pure strengthening — variable bounds must
  never exclude a reachable prediction, which the harness now proves.

### Wave 1 — readers refuse loudly

**T-N1.1 — ONNX reader hardening (~1–2 days).**
`readers/onnx_reader.py`:
- Gemm: read `alpha`/`beta` and apply (`W *= alpha`, `b *= beta`); raise
  `ValueError` on `transA=1` (:104-121).
- MatMul: verify the initializer is `input[1]` (data flows as `x @ W`); raise
  on weight-as-`input[0]` rather than silently mis-orienting (:123-129). Raise
  a clear `ValueError` (not `KeyError`) when the weight is not an initializer.
- Dataflow verification: track the current tensor name starting from
  `graph.input[0]`; every consumed node's data input must match it, and its
  output becomes the new current tensor. Any mismatch (residual/branch
  topology) → `ValueError("non-sequential graph")`. This eliminates the
  silent residual-Add-as-zero-bias path (:74-79) as a side effect.
- **Tests:** (a) Gemm with `alpha=2.0, beta=0.5` loads and
  `assert_embedding_matches` passes — *and* the loaded net's `forward` matches
  `onnxruntime` inference on random points (onnxruntime is already in the
  `[nn]` extra — use it as the reference oracle wherever ONNX tests run);
  (b) `transA=1` raises; (c) a two-branch graph with a joining Add raises;
  (d) `transB=1` round-trips correctly (currently untested).

**T-N1.2 — sklearn semantics + tree edge (~1 day).**
`readers/sklearn_reader.py`:
- `load_sklearn_mlp`: read `model.out_activation_`; `identity` → LINEAR,
  `logistic` → SIGMOID on the final layer, `softmax` → `ValueError`
  (multi-class not embeddable as a scalar map). Update the docstring.
- `load_sklearn_tree` / `load_sklearn_ensemble`: raise `TypeError` on
  classifiers (`sklearn.base.is_classifier`), with a message explaining
  regressor-only support; fixes the silent log-odds/`base_score` wrongness.
  Docstrings updated to regressors-only.
- `_sklearn_tree_to_decision_tree` (:66): replace `squeeze()` with
  `value = t.value.reshape(len(feature), -1)`; raise on `shape[1] != 1`
  ("multi-output trees are not supported"). Fixes the single-leaf 0-d crash.
- **Tests:** binary `MLPClassifier` loads with SIGMOID final layer and matches
  `predict_proba[:, 1]` through the harness; `GradientBoostingClassifier` and
  `DecisionTreeClassifier` raise `TypeError`; a constant-target
  `DecisionTreeRegressor` (single leaf) loads and predicts.

**T-N1.3 — torch `bias=False` (~0.5 day).**
`readers/torch_reader.py`: extract weights/bias via one helper
`_linear_wb(linear)` returning zeros when `linear.bias is None` (three
duplicated sites: :47-49, :55-57, :67-69).
- **Test:** `nn.Sequential(nn.Linear(2, 3, bias=False), nn.ReLU(),
  nn.Linear(3, 1))` loads and matches the torch forward pass on random points.

### Wave 2 — design cleanups

**T-N2.1 — reduced_space: honest docs + bounded intermediates (~0.5–1 day).**
Recommendation (pick this unless the maintainer prefers true nesting): keep the
one-variable-per-layer structure but (a) attach propagated post-activation
bounds to each layer `z` when `input_bounds` exist (in the **scaled** domain
when scaling is present — same rule as T-N0.2); (b) skip the redundant final
`z` var by emitting the last layer's expressions directly into the output
constraints; (c) rewrite the class/strategy docstrings to say what it actually
is (a lean full-space variant: one var per layer, affine+activation fused into
one constraint) — remove "no intermediate variables". True single-expression
nesting is rejected for now: deep nested DAGs hurt the JAX compile and the
McCormick machinery needs boxes anyway; note this in the docstring.
- **Test:** harness equivalence; all intermediate vars have finite bounds when
  the net has input_bounds.

**T-N2.2 — NNPresolvePass: park (recommended) or wire (decision needed).**
Verified state: no solve path constructs it; `nn_implications` consumed only by
tests; docstring claims `tighten_var_bounds` writeback that `run()` never does;
`except Exception: return delta` swallows everything (`presolve.py:236`).
- **(a) Park (recommended):** fix the docstrings to describe v0 truthfully
  (informational, manually-invoked); narrow the `except Exception` to the
  specific shape/`ValueError` failures with a `logger.debug` note; open a
  tracking issue for the v1 design (register the pass via
  `run_root_presolve(python_passes=…)` — the hook exists,
  `presolve_pipeline.py:64` — and have `ReluBigMFormulation` register its
  binaries so dead-ReLU implications can fix `q=0` after root tightening).
  Rationale: the build-time short-circuit in `relu_bigm.py:179-184` already
  exploits static bounds; the marginal value of post-FBBT re-tightening is
  unmeasured, and the house rules forbid shipping it on a hypothesis.
- **(b) Wire now:** requires the entry experiment first — measure, on ≥ 3
  realistic embedded-NN instances, whether root FBBT/OBBT tightens NN input
  boxes enough to kill additional neurons vs. build-time bounds. Kill if no
  instance gains a fixing.

**T-N2.3 — add_predictor ergonomics (~0.5 day).**
`predictor.py`:
- When `input_bounds is None` and the predictor carries none, harvest bounds
  from the user's `inputs` variable if finite; raise a clear error naming the
  three ways to supply bounds otherwise (currently the error surfaces from
  deep inside `ReluBigMFormulation`).
- Validate `inputs` length == `n_features` up front with a clear message.
- Missing file → `FileNotFoundError` (not `TypeError`) (:114-120).
- Keep the link-constraint design (presolve eliminates the duplicates); note it
  in the docstring.
- **Tests:** bounds harvested from variable bounds end-to-end; length mismatch
  message; missing-file behavior.

### Wave 3 — completeness (opt-in; do not start without maintainer approval)

Priority order by value-per-effort:
1. **T-N3.1 — LeakyReLU + Clip/ReLU6** activations end-to-end (enum, `forward`,
   `bounds.py` — both monotone; big-M analogue for LeakyReLU; readers for
   torch/ONNX). Small, unblocks common models.
2. **T-N3.2 — LP-based big-M tightening.** Entry experiment first (kill
   criterion): on ~5 random relu nets (2–10 inputs, 2–3 layers), compare
   interval big-Ms vs LP-tightened per-neuron bounds (solve min/max `zhat_j`
   over the partially-built relaxation — Grimstad & Andersson 2019, already
   cited in `presolve.py`) and measure solve-time change on the embedded MILP.
   Kill if median big-M shrinkage < 20% or solve time does not improve.
3. **T-N3.3 — XGBoost/LightGBM readers** (booster dump → `DecisionTree` arrays;
   `base_score` finally earns its name). Regressor-only, per T-N1.2's rule.
4. **T-N3.4 — Mišić split-based tree encoding + multi-output ensembles.** Entry
   experiment: node-count/LP-strength A/B of per-leaf big-M vs split encoding
   on ≥ 3 ensembles (deep trees). Adopt only if it wins; keep the per-leaf
   encoding as the fallback path.
Out of scope entirely (record here so it isn't relitigated): Conv/pooling
layers, BatchNorm folding, Keras/TF reader, classification argmax embedding —
each is real scope with no current user; file issues if demand appears.

### Wave 4 — hygiene (fast follows)

**T-N4.1 —** Remove `python/discopt/ml/readers/onnx_reader.py` from the
coverage omit list (`pyproject.toml:263`) once T-N1.1's tests exist. If CI does
not install the `[nn]` extra, add `onnx`/`onnxruntime` to the dev extra first
(pure-python wheels) — do not leave the omit in place to hide untested code.
**T-N4.2 —** Update CLAUDE.md's `python/discopt/ml/` paragraph: it omits
`predictor.py` (`add_predictor`), `tree.py`/`tree_ensemble.py`, `presolve.py`,
`scaling.py`, and the sklearn/torch readers.
**T-N4.3 —** File F1–F4 as cards in `docs/dev/correctness-issues.md` (its §0
protocol; F1/F2 are P1-class for the modeling layer, F3/F4 are P2
silent-substitution class like C-5), each linking to its T-N task. May be done
immediately, before the fixes land, so the ledger is complete.
**T-N4.4 —** Re-verify `docs/notebooks/nn_embedding.ipynb` after T-N0.2 (its
scaling examples must reflect the fixed semantics); rebuild the Jupyter Book
with zero warnings if it changes.

---

## 4. Sequencing

```
T-N0.1 (harness)
  ├─► T-N0.2 (F1)  ─┐
  ├─► T-N0.3 (F2)  ─┼─► T-N0.4 (output bounds)
  ├─► T-N1.1 / T-N1.2 / T-N1.3 (readers, parallel)
  └─► T-N2.1 / T-N2.3 (parallel)          T-N2.2 (decision → park or entry exp.)
Wave 3: opt-in, after Wave 0–1            Wave 4: T-N4.3 anytime; rest last
```

Estimated effort: Wave 0 ≈ 3–4 d; Wave 1 ≈ 2.5–3.5 d; Wave 2 ≈ 1.5–2 d (+
decision); Wave 4 ≈ 1 d. Wave 3 unbounded until scoped by its entry
experiments.

## 5. Risks

1. **F1's fix changes models that were "accidentally working"** (identity-like
   scaling with nonzero offsets that happened to stay feasible): bounds get
   *tighter and correct*; the harness plus the T-N0.1 green baseline guard
   against over-tightening.
2. **Reader hardening breaks users of previously-silent paths** (Gemm alpha≠1,
   classifiers): intended — a loud error replacing a wrong answer. Release
   notes must say so.
3. **Equivalence tests depend on solver behavior** (a MILP solve per case):
   keep nets tiny (≤ 5 binaries) so smoke stays fast; mark the enumeration
   cases `slow` where needed.
4. **`propagate_bounds` signature change** ripples to `tighten_network` and any
   external callers: additive keyword-only argument, default preserves current
   behavior.

## 6. Entry experiments from PWLNet and the "MIP surrogates" post (2026-11-22)

**Provenance.** The questions below are taken from two sources, and the credit is
theirs: S. Sanyal, C. Demir, M. Türkay, E. Aydın, *PWLNet: A Framework for
Linearized Neural Surrogates in Nonlinear Process Optimization*,
arXiv:2611.25969v1 (2026), code at https://github.com/SunoySanyal/PWLNet; and
J. Kitchin, *MIP surrogates* (2026-10-28),
https://kitchingroup.cheme.cmu.edu/posts/mip-surrogates/, whose 2-D test function
the panels use. The methods are older: LP-based neuron bounds (Tjeng, Xiao &
Tedrake 2019; Grimstad & Andersson 2019), input-convex networks (Amos, Xu &
Kolter 2017). No code was copied from either source; the claims were re-measured
on discopt's own formulations.

**Instruments.** `discopt_benchmarks/scripts/surrogate_entry/` (`e1_…`, `e2_…`,
`e3_…`), measured at commit `015535b` on a 4-CPU container with the arms
interleaved and nothing else running. Each script counts its executed checks and
exits non-zero if none ran. The networks are sklearn MLPs (or a numpy-trained
ICNN for E2) with fixed seeds, so the panels are reproducible from the scripts.
Raw rows were kept out of the repo; the numbers below are the record.

### E1 — LP big-M tightening (T-N3.2's entry experiment): **passes**

Arms: `relu_bigm` with interval bounds, vs the same formulation with per-neuron
bounds from a layer-by-layer LP over the triangle relaxation (scipy/HiGHS, pad
1e-6(1+|v|)). 10 ReLU nets, 2–10 inputs, 2–3 hidden layers; 2 reps; 120 s limit.
Soundness: 17,240,000 sampled pre-activations all fell inside the LP bounds, and
all 9 objective comparisons between certifying arms agreed.

* Median big-M shrinkage over layers ≥ 1 is **45–66% per net**, against the 20%
  kill threshold. (Layer 0 is exact under interval arithmetic and is excluded.)
* On every net both arms certified, LP tightening is faster, with its bound
  computation included: 1.6→1.2 s, 8.9→4.8 s, 5.3→3.7 s, 17.3→15.3 s.
* On 2; 64, 64, only the LP arm certifies in both reps (34 s, 4420 nodes). The
  interval arm hit the limit once and took 95 s (8108 nodes) the other time.
* On the 5 nets that time out either way, the LP arm always ends with the tighter
  bound (e.g. −28.1 vs −41.4, −111.8 vs −139.1). On 2 of them, interval bounds
  never find an incumbent and LP bounds do.
* LP tightening costs 0.1–0.94 s.
* **The unstable-neuron count barely moves.** The gain comes from smaller
  constants, not from fixing neurons, so the T-N2.2 presolve v1 (which only
  removes stable neurons) has little to offer.

Before shipping: replace the pad with safe dual bounds (Neumaier–Shcherbina),
and put it behind a flag with the §5 differential panel. It is bound-changing.

**Shipped (2026-11-23)** as `DISCOPT_NN_LP_BIGM`, with both done. The §5
class panel (`nn_lp_bigm_panel.py`) confirmed E1 on `relu_bigm`: certified 5 =
5, each faster (101 → 27.8 s on the slowest), every uncertified bound tighter,
2 incumbents gained. On `relu_extended` it was mixed, and since that panel is
deterministic, a spread panel (`nn_lp_bigm_perturb_panel.py`) solved both arms
under three sound 1e-9..1e-8 widenings. HiGHS's search alone moves a
`relu_extended` solve from 19 to 321 nodes under such a widening. Against that
spread, the bound stays better on 5 of 7 uncertified nets. But on the d5_32x32
pair the incumbents get worse (s0: OFF −3.35..−3.48, ON −1.21..−2.81), and the
wide nets stall at the root bound. So the default is ON for `relu_bigm` and
OFF for `relu_extended` (`=1` opts in). The primal loss is unexplained.

`relu_partition` has its own class panel (`nn_lp_bigm_panel.py --strategies relu_partition --time-limit 120`, at `3a6fa89` plus the #2323 review fixes, which do not touch this path; E1's 10 nets, 36 checks clean). It certified 5 = 5 with identical objectives. With the flag on it was faster on 3 (11.2->9.1, 107->50, 8.0->6.5 s), level on 1 (2.9->3.0 s) and 2x slower on d5_32x32 s0 (27.6->55.0 s, 10,709->19,292 nodes). At the 120 s limit the dual bound was tighter on 3 of 5 (-9.29->-7.60, -36.1->-29.1, -27.2->-26.4) but looser on 2 (d5_64x64 -51.8->-61.0, d10_50x50 -133.6->-167.1), and incumbents were better on 3 (-1.099->-1.196, -2.845->-2.986, none->+1.30). A looser bound on 2 nets and a 2x slower certification fail bar 2, so `relu_partition` keeps interval bounds by default (`=1` opts in). It had been graduated ON by inheritance from `relu_bigm` without this panel; the #2323 review caught it.

### E2 — ICNN minimised as an LP with no binaries: **passes**

Arms: default `relu_bigm` MILP, vs the epigraph LP `zhat = Wᵀz + b, z ≥ 0,
z ≥ zhat`. This LP is exact only for an input-convex network whose output is
minimised or bounded above. 8 sequential ICNNs (every weight after the first
layer ≥ 0), 2–10 inputs; 3 reps. All 24 objective comparisons agreed.

| net | MILP s | LP s | ratio |
|---|---|---|---|
| 2; 32, 16 | 0.42 | 0.03 | 14 |
| 2; 64, 64 (seed 0 / 1) | 1.83 / 2.13 | 0.08 / 0.10 | 23 / 21 |
| 2; 64, 64, 64 | 3.64 | 0.13 | 28 |
| 5; 64, 64 (seed 0 / 1) | 2.35 / 1.84 | 0.10 / 0.08 | 24 / 23 |
| 5; 128, 128 | 5.22 | 0.16 | 33 |
| 10; 64, 64 | 1.38 | 0.09 | 15 |

The spread is ≤ 0.09 s. The median ratio is ≈ 23 against a kill threshold of 2.
The MILP closes in 0–1 nodes, so its cost is building and presolving the
binaries, not branching. This reproduces the post's 0.47 s vs 0.06 s.

Candidate: an ICNN route in `add_predictor`, chosen by structure (weights after
the first layer ≥ 0, output used only as minimised or bounded above). It must
refuse in the other direction, where the relaxation is not exact.

**Shipped (2026-11-23) as `strategy="relu_icnn"`**
(`ml/formulations/relu_icnn.py`), opt-in rather than auto-selected. How the
output is used is a property of the rest of the model, which is not known when
the rows are built. So exactness is checked on each result instead of assumed.
`Model.solve` runs the check the formulation registers
(`Model._solve_result_checks`, top-level solves only). It puts every network
variable at its forward value at the reported input and re-verifies the whole
model with `verify_point`, requiring the objective to be unchanged. That is
sufficient because the rows are a relaxation of the graph: a point that is
optimal for the relaxation and also on the graph is optimal for the network.
Any failure raises `ICNNExactnessError`, as does an `unbounded` relaxation;
`infeasible` stands, since it proves the network model infeasible. The check
judges the answer, not the use. With `y >= c` (which rewards a larger
output) the solve is refused when the relaxation's point is off the graph, and
answered when it happens to be on it, as for 3.0 on the test net. A
non-input-convex network is refused at build. A model carrying the check
refuses `serialize.dumps`, which has no section for it. The rows are one array
row `z >= zhat` per layer, with `z` boxed in `[max(lb,0), max(ub,0)]`. That
single row is the epigraph for unstable and stable neurons alike.

Panel (`icnn_strategy_panel.py`, E2's 8 ICNNs; `min` the output, and `cap` =
minimise `sum(x)` subject to `y <= min + 0.5`; 3 interleaved reps; load ≤ 2.4 on
4 CPUs; 48 checks clean). Speedup over `relu_bigm`: median **8.6×** on `min`
and **9.1×** on `cap`, range 5.6–27×. Every objective lies inside the other
arm's [bound, objective] window. `relu_bigm` stops at the default 1e-4 gap: on
`d10_64x64 cap` it reported −5.128406 against the LP's exact −5.128788, and
its 1e-9 run reproduces the LP. The 8.6× is lower than E2's 23× against a
hand-built LP. Per-neuron scalar rows had it at about 5×, and the array row
recovered part of the difference. The result check costs nothing measurable
(0.156 vs 0.153 s and 0.475 vs 0.465 s, interleaved, sd 0.01–0.05).

### E3 — piecewise-linear tanh (PWLNet §4.4–4.5): certified route **killed**, chord route **heuristic only**

Arms: `full_space` (library build), the same model written one scalar constraint
per neuron with identical bounds, and `nonlinear_to_pwl` in `"outer"` (8
segments, certified) and `"approximate"` (4 or 8 chords, PWLNet's interpolant,
uncertified) modes. tanh nets, 2–5 inputs; 2 reps; 300 s limit. Every incumbent
is re-evaluated through `forward`, and no certified bound may exceed the best
incumbent. All 24 checks passed.

| net | full_space | per-neuron | outer | approx4 | approx8 |
|---|---|---|---|---|---|
| 2; 8 | 6.5 s ✓ (131 nodes) | 2.6 s ✓ (19 nodes) | 38 s ✓ | 0.27 s, opt | 0.98 s, opt |
| 2; 16 | TL, bnd −1.55 | 24 s ✓ | TL, bnd −1.494 | 0.79 s, opt | 1.3 s, opt |
| 2; 32 | TL, bnd −12.9 | 140 s ✓ | TL, bnd −1.67 | 4.2 s, opt | 10 s, opt |
| 2; 16, 16 | TL, bnd −11.0 | TL, bnd −7.95 | TL, bnd −2.34 | 5.1 s, **−1.109** | 15 s, opt |
| 5; 16 | TL, bnd −9.25 | TL, bnd −6.1 | TL, bnd −3.691 | 1.4 s, opt | 3.3 s, opt |
| 5; 16, 16 | TL, bnd −23.8 | TL, bnd −23.6 | TL, bnd −12.0, inc −2.43 | 12 s, **−3.144** | 108 s, opt |
| 2; 16 (seed 1) | 296 s ✓ (25,543 nodes) | 31 s ✓ (165 nodes) | TL, bnd −1.561 | 1.0 s, opt | 2.8 s, opt |

✓ = certified; TL = 300 s limit; "opt" = the approximate arm's point, checked
through the exact network, equals the best-known optimum. On 2; 16, 16 the best-known
value is −1.6147 (no arm certified it), so approx4 misses by 0.51, 17× the net's test RMSE of 0.029.
On 5; 16, 16 (best-known −4.0512, uncertified) approx4 misses by 0.91, 4× the
test RMSE of 0.226; `outer` never found a better incumbent than −2.43.

* `outer` certifies only the smallest net, 15× slower than the per-neuron exact
  solve. That is the kill. It does reach the tightest bound at the limit on every
  timed-out net, which may be worth knowing for bounding-only use.
* `approximate` is 5–70× faster and gives good points, but no bound, and 4 chords
  can be badly wrong. If it is used at all, use it as a primal heuristic whose
  point is verified through the exact network, never as the solve.

**Two defects found along the way:**

1. **`full_space` relaxes much more weakly than the per-neuron build of the same
   model.** The bounds are identical, yet 2; 16 goes from certified in 24 s to
   uncertified at 300 s, and 2; 32 from certified in 140 s to a bound of −12.9
   at the limit. This is the same class as the `MatMulExpression` note in
   `full_space.py` (#1215 §55): the bound depends on how the constraint is
   written.

   *Root-cause work (2026-11-22).* The root LP bound is identical (−7.1527 on
   2; 8) whether the rows are written as arrays or per neuron, so the node LP is
   not the weaker part. A search of the solve path found several stages that only
   act on size-1 variables. The leading suspect,
   `_relax/dependent_vars.find_functionally_dependent_names` skipping any
   variable with `size != 1`, was tested by patching it to report the tanh-output
   arrays. Kill criterion: patched array build within 1.5× of the scalar build's
   node count.

   | net | array build | + OBBT gate on | + OBBT + deprioritise | scalar build |
   |---|---|---|---|---|
   | 2; 8 | 131 | 33 | 51 | 19 |
   | 2; 16 s0 | TL (29,871) | ✓ 913 (139 s) | ✓ 997 | ✓ 117 (27 s) |
   | 2; 16 s1 | TL (24,771) | ✓ 1241 (194 s) | ✓ 1521 | ✓ 165 (32 s) |

   **Falsified as the whole cause.** Turning on the per-node OBBT gate alone
   turns both timeouts into certified solves, but it leaves a 3–9× node gap, and
   deprioritising the tanh outputs slightly hurts.

   *Localisation, continued.* Knocking out root FBBT, root cutoff-FBBT and the
   per-node linear-row FBBT one at a time in the scalar build changes nothing
   (node counts identical on 2; 8 and 2; 16). Variable order changes nothing
   (interleaved vs block scalar build: 19/19, 117/117). With detection emptied,
   the array build and the scalar build are **identical** (105 and 2,953 nodes,
   identical bounds), and the root bound of array-affine and per-element-affine
   rows is bit-identical on 12 random sub-boxes. So the relaxation, FBBT and
   OBBT treat both forms the same; the **whole** difference is which variables
   the detector marks.

   On the scalar build the detector marks both the tanh outputs `z_j` *and* the
   pre-activations `zh_j`. Marking exactly that set on the array build gives the
   scalar build's node counts exactly (19 / 117 / 165 on 2; 8, 2; 16 s0, 2; 16
   s1). Marking only the tanh outputs gives 51 / 997 / 1521. (The earlier "B"
   column above marked only the outputs, which is why deprioritising looked
   harmful.)

   **But `zh_j` is marked by accident.** The detector is meant to mark only
   variables pinned by a *nonlinear* equality; affine ones are left to presolve
   by design. The row `zh_j − (w·x + b) == 0` is affine, yet
   `_body_is_nonlinear` returns True because it has no `IndexExpression` case
   (`x[0]` is an indexed reference to the array input) and treats an
   unrecognised node as nonlinear. The same row over scalar inputs would be
   classed affine and `zh_j` left unmarked. So there are two defects, and they
   interact:

   * **D1** — `find_functionally_dependent_names` / `dependent_columns_for_model`
     skip every variable with `size != 1`. Any model with array intermediates
     gets no per-node OBBT and no deprioritisation.
   * **D2** — `_body_is_nonlinear` misclassifies affine rows that reference an
     indexed variable, so whether a variable is deprioritised depends on whether
     its neighbours are scalars or array elements.

   The measurement also says the design rule behind D2's gate ("affine-pinned
   outputs are presolve's job") does not hold here. Presolve keeps `zh_j`
   (bounded, used in a nonlinear row), and deprioritising it is worth 3–9× in
   nodes. A correct fix therefore needs a rule for *which side* of an affine
   defining equality is the output (in `zh = Wx + b` either side can be solved
   for, and marking the inputs too would deprioritise everything). That rule
   changes branch order on every model, not only on arrays, so it needs the
   corpus panel.

   *Fixed 2026-11-22: `DISCOPT_DEFINED_VARS`, graduated default ON.*
   `dependent_vars.find_defined_variables` keys array elements as
   `(name, flat, size)` (D1), treats indexed references as affine leaves (D2),
   and gives each affine `==` row one output: the unmarked candidate occurring in
   strictly fewer scalar constraint rows than every other candidate (greedy
   minimum degree; a tie marks nothing). `=0` restores the legacy detector. With
   it, `full_space` and the per-neuron build run identical searches (15 / 79 /
   107 nodes on 2; 8, 2; 16 s0, 2; 16 s1). Panels, deterministic and
   interleaved: the in-repo corpus (74 instances, 30 s) is cert-clean with 70
   instances identical and 3 improved (`haverly` 69 → 35 nodes, `ex8_5_3`
   143 → 137, `nvs05` bound 4.32 → 5.07), wall 748.6 → 747.1 s; the class panel
   (12 tanh / sigmoid / softplus `full_space` nets, 120 s) is clean over 54 checks
   with certified 6 → 11 and wall 948 → 380 s. Full rows in
   `docs/dev/flag-retirement-audit.md`.
2. **`nonlinear_to_pwl` finds 0 terms and reports 0 skipped on array-valued
   constraints** (`full_space`'s one-constraint-per-layer form). It is a silent
   no-op; it must replace those terms or list them as skipped.

   *Fixed 2026-11-23.* `_pwl_transform` now splits a non-affine array-valued
   constraint into one scalar row per element (`scalarize.scalar_elements`) and
   a full reduction `sum(f(v))` into `f(v[0]) + ...`, so each element is an
   ordinary univariate term; affine array rows stay whole. A vector-valued node
   that still cannot be expanded is reported in `skipped` with the reason
   "vector-valued", never dropped. On the E3 2; 8 net, `full_space` now gives 8
   terms in both modes (was 0), the same as the per-neuron build. On the 74
   in-repo corpus instances (all scalar), terms, skipped, `fully_linear` and
   the constraint count are identical before and after.

**Re-run after both fixes (2026-11-23, at `98916c8`): heuristic not built.**
The case for a PWL primal heuristic was measured against the slow
`full_space`, so E3 was re-run unchanged (2 reps, 300 s). It ran on the first 6
nets; 5; 16, 16's second rep was cut by the job limit (rep 0 complete), and
2; 16 (seed 1) was not re-run. The arms are deterministic: every repeated
arm returned the same objective, so one rep stands.

| net | full_space | outer | approx4 | approx8 |
|---|---|---|---|---|
| 2; 8 | 4.2 s ✓ (15 nodes) | 37 s ✓ | 0.3 s, opt | 0.7 s, opt |
| 2; 16 | 18 s ✓ (79) | TL, bnd −1.4945 | 0.6 s, opt | 2.7 s, opt |
| 2; 32 | 93 s ✓ (147) | TL, bnd −1.673 | 3.2 s, opt | 18 s, opt |
| 2; 16, 16 | TL, bnd −1.616, inc −1.6147 | TL, bnd −2.34 | 5.5 s, **−1.109** | 15 s, opt |
| 5; 16 | TL, bnd −4.515, inc −3.6846 | TL, bnd −3.691 | 1.2 s, opt | 2.8 s, opt |
| 5; 16, 16 | TL, bnd −20.7, inc −4.0512 | TL, bnd −12.3, inc −3.359 | 18 s, **−3.144** | 91 s, opt |

Every incumbent was re-evaluated through `forward` (the `forward_obj` field
matches each objective to 1e-9). `full_space` now certifies 3 of 6, where it had
certified 1 of 6 on the same nets, and on 2; 16, 16 its bound went from −11.0 to −1.616.

* **The primal heuristic has nothing to add.** On all 6 nets `full_space`'s own
  incumbent at the limit equals the best point any arm found. The approximate
  arm never found a better one, and 4 chords still miss on the same two nets.
  Where `full_space` times out, what is missing is the bound, which a heuristic
  cannot supply. One case would reopen it: `full_space` finding its incumbent
  late in the run. Time-to-incumbent was not recorded, so that is the
  measurement to take first.
* **`outer` as a bound source is open, not established.** On the two 5-input
  nets its bound at the limit is tighter than `full_space`'s (−3.691 vs
  −4.515; −12.3 vs −20.7). On the 2-input nets it is looser, and it never
  certifies faster. An entry experiment would compare root and limit bounds on
  more 5+-input nets before any flag.
* The `pwl_outer` arm printed "HiGHS MILP route: the … certificate is refuted
  by a presolve-free HiGHS solve; reporting the verified point uncertified
  (#1634)" on 2; 16 and 2; 32. That is the HiGHS route's own certificate check
  firing and failing closed. It is not a wrong answer: those arms are reported
  uncertified above.

Also seen once during E1: an overflow `RuntimeWarning` (inf × value) at
`solvers/lp_milp_highs.py:820`. Not yet investigated.
