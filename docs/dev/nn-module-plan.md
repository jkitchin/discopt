# NN-module remediation plan (`python/discopt/nn/`)

**Date:** 2026-07-03
**Status:** proposed (from a full-module review, 2026-07-03: all 17 files ~2,100
lines + `test_nn_formulations.py` read; wiring claims verified against the
modeling API and presolve pipeline). Every finding below carries a file:line
anchor from that review.
**Scope:** the NN/tree embedding layer only (`python/discopt/nn/`). No solver-core
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
| T-N2.2 NNPresolvePass: park or wire | 2 | **done (parked)** | — | Truthful v0 docstring; bare except narrowed to ValueError + logger.debug; v1 wiring left as noted future work |
| T-N2.3 add_predictor ergonomics | 2 | **done** | — | Bounds harvested from inputs var; length validation; missing file → FileNotFoundError |
| T-N3.1 LeakyReLU/Clip activations | 3 | deferred (opt-in) | — | needs maintainer approval |
| T-N3.2 LP/OBBT big-M tightening | 3 | deferred (opt-in + entry exp.) | — | |
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

**T-N4.1 —** Remove `python/discopt/nn/readers/onnx_reader.py` from the
coverage omit list (`pyproject.toml:263`) once T-N1.1's tests exist. If CI does
not install the `[nn]` extra, add `onnx`/`onnxruntime` to the dev extra first
(pure-python wheels) — do not leave the omit in place to hide untested code.
**T-N4.2 —** Update CLAUDE.md's `python/discopt/nn/` paragraph: it omits
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
