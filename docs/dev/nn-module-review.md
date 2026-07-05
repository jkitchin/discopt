# NN Module Review — Correctness, Robustness, SOTA

**Date:** 2026-07-03
**Scope:** `python/discopt/nn/` — `network.py`, `bounds.py`, `presolve.py`,
`formulations/{full_space,reduced_space,relu_bigm,tree_ensemble}.py`,
`readers/{onnx,sklearn,torch}_reader.py`, `scaling.py`, and tests.
**Method:** Delegated verification pass with numerical repros; the headline P0
independently re-confirmed in this review. discopt embeds trained feedforward NNs
(and tree ensembles) as algebraic constraints in MINLP models, OMLT-style.

For a **global** solver, an NN embedding that is not exactly equivalent to the
network's forward pass — or whose big-M is too small — silently cuts feasible
activations and yields a certified-optimal answer to the wrong problem.

---

## 1. Summary of findings

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| NN-1 | **P0 correctness → ✅ FIXED (C-25)** | `bounds.py:propagate_bounds` via `relu_bigm.py:73`, `full_space.py:72` | Bound propagation runs in the **unscaled** input space, but the first affine layer consumes the **scaled** input. With any non-identity `scaling`, the pre-activation bounds are wrong, so the big-M constants and `zhat` variable bounds are too tight → feasible ReLU activations cut → **wrong certified optimum** [CONFIRMED, independently re-run: relu net returns 1.0, true 10.0]. **FIXED** PR #411 `df96f1a` (scaled-domain propagation `propagate_bounds(net, input_bounds=(s_lo,s_hi))`); re-verified #413 — certified max now **10.0**, 0/30 feasible pts cut |
| NN-2 | **P1 silent-wrong → ✅ FIXED (C-26)** | `formulations/tree_ensemble.py:39-133` | `TreeEnsembleFormulation` accepts and stores `scaling` but **never applies it** — split thresholds compare against raw inputs, outputs omit `y_factor/y_offset`. Any non-trivial scaling → wrong predictions/optimum, no error [CONFIRMED]. The related tree big-M validity for **out-of-box thresholds** is carded as **C-26**. **FIXED** PR #411 `df96f1a` (per-constraint tight big-M `max(ub−thr,0)`/`max(thr+eps−lb,0)`); re-verified #413 — out-of-box ensemble feasible, 0/40 feasible pts cut |
| NN-3 (=C-27) | P1 silent-wrong | `readers/onnx_reader.py:104-121` | ONNX `Gemm` reads only `transB`; **ignores `alpha`, `beta`, `transA`**. `alpha≠1`/`beta≠1` (legal, emitted by some exporters/quantizers) → weights/biases wrong by a scalar, no error [**FIXED** PR #411: `alpha`/`beta` folded, `transA=1` raises; live-onnxruntime repro confirmed the divergence, max\|Δ\|=0.684 → 1.6e-07] |
| NN-4 (=C-27) | P2 | `readers/onnx_reader.py:66-129` | `MatMul` weight-orientation assumed (`x @ W`, weight = the initializer operand); node **chaining not verified** (list order assumed to be a chain, no output→input tensor check); `Reshape`/`Flatten` skipped blindly. Mis-loads a non-sequential or `W @ x` graph [**FIXED** PR #411: MatMul weight verified `input[1]`, single-tensor dataflow chain → non-sequential/residual/branch raises] |
| NN-5 | P3 float | `bounds.py:80-81` | Interval propagation uses round-to-nearest, no outward rounding — a bound-achieving vertex can land an epsilon outside the interval. Absorbed by solver tol in practice; note only [SUSPECTED] |
| NN-6 | P3 | `tree_ensemble.py:98-111` | Right-split big-M `M_j = ub−lb` omits `split_eps`, cutting the `[lb, lb+eps)` sliver. Harmless at `eps=1e-6` [VERIFIED by inspection] |

Checked and found **correct** (verified):

- **The ReLU big-M constraint set** (`relu_bigm.py:185-194`) is the textbook exact
  formulation and reproduces the forward pass to ~4e-16 against brute force **when
  scaling is identity** — the encoding itself is right; NN-1 is entirely in the
  *bounds* feeding it.
- **Interval propagation's positive/negative weight split** (`bounds.py:77-81`) is
  sound for the affine map + monotone activations (given a correct input box).
- **`reduced_space` is scaling-correct** and immune to NN-1: it bounds
  `scaled_in` via the transformed box and leaves intermediates unbounded — the
  repro returns 10.0 correctly, which is what isolates NN-1 to the
  `propagate_bounds`-based (`relu_bigm`, `full_space`) formulations.
- Truly-unknown ONNX ops **do raise** (`onnx_reader.py:89-92`) and `onnx.checker`
  runs — the reader fails loudly on the genuinely unsupported, just not on the
  silently-mishandled `Gemm` attributes (NN-3).
- **`presolve.py`'s dead-ReLU classification** logic is correct *given correct
  bounds* — but it reads `network.input_bounds`, so it inherits NN-1's scaling
  blindness wherever its dead-neuron implications drive big-M binary fixing.

---

## 2. NN-1 in detail (the P0)

`propagate_bounds(net)` seeds layer 0 with `net.input_bounds` — the unscaled
user-domain box. But the emitted model applies `scaled_in = (input − x_offset)/
x_factor` before the first affine layer (`relu_bigm.py:82-94`,
`full_space.py:82-100`). So the pre-activation interval `[pre_lb, pre_ub]` is
computed for the wrong range, and it is then used **twice**:

1. as the `zhat` pre-activation **variable bounds** (`relu_bigm.py:106-110`), and
2. as the **big-M constants** `M = ub_j`, `−lb_j` (`relu_bigm.py:176-192`).

Both too-tight. **Reproduced** (re-run in this review): 1-input → 1-ReLU → linear
net, `input_bounds=[0,1]`, `x_factor=0.1` so the true function is `relu(10·input)`
with max 10.0 at input=1. `propagate_bounds` reports `zhat ∈ [0,1]`; the solver
returns **`optimal`, objective = 1.0000**. The full-space sigmoid variant returns
sigmoid(1)=0.731 instead of sigmoid(10)=0.99995. The always-active/always-inactive
ReLU classification (`relu_bigm.py:179-184`) inherits the wrong bounds, so a
genuinely active neuron can be hard-wired to `z=0`, compounding the cut.

**Fix:** propagate through the *scaled* input box. The scaled corners `s_lo/s_hi`
are already computed (`relu_bigm.py:82-85` / `full_space.py:85-89`); feed those
into `propagate_bounds` (add an `input_box=` argument, or set
`net.input_bounds = (s_lo, s_hi)` as `tighten_network` already does). **Regression
test:** a non-identity-scaling model with a solved optimum — *every* current
scaling test uses `x_offset=0, x_factor=1`, which is exactly why CI is blind to
this. Assert the repro returns 10.0.

---

## 3. SOTA

A competent, clean OMLT-clone: the three canonical encodings (full-space,
reduced-space, ReLU big-M), interval bound propagation, tree-ensemble MILP (Mišić
2020), and ONNX/sklearn/torch readers — roughly feature-parity with OMLT's core and
comparable to gurobi-machinelearning's `add_predictor`. The ReLU big-M encoding
matches OMLT exactly. The distinctive `presolve.py` (NN-aware FBBT / dead-ReLU
detection feeding the global B&B) is genuinely **ahead** of OMLT, which delegates
bound tightening to the MIP solver. But OMLT and gurobi-ML propagate bounds in the
**scaled** space; discopt propagates in the **unscaled** space (NN-1), producing an
unsound big-M under non-identity scaling — disqualifying for a global solver until
fixed, and invisible to CI because every scaling test uses the identity transform.

---

## 4. Implementation plan (for Opus)

### Phase 1 — correctness (PR `fix(nn): NN-1..NN-2`) — ✅ DONE (PR #411 `df96f1a`; verified #413)

| ID | Task | Acceptance |
|----|------|-----------|
| NN-1 ✅ | Propagate bounds through the scaled input box (`input_box=` arg to `propagate_bounds`, or set `net.input_bounds` to the scaled corners before propagation); ensure `presolve.py`'s dead-ReLU path uses the same scaled bounds | **MET** — `propagate_bounds(net, input_bounds=(s_lo,s_hi))` in `relu_bigm.py`/`full_space.py`; non-identity-scaling repro returns **10.0** (was 1.0 on pre-fix main); identity-scaling tests unchanged; `test_nn_formulation_fixes.py::test_*scaling*` fail-before/pass-after |
| NN-2 ✅ (C-26) | Valid tree big-M for out-of-box thresholds: per-constraint `max(ub−thr,0)` / `max(thr+eps−lb,0)`, inert for `z=0` at any threshold | **MET** — out-of-box ensemble now feasible & optimum-preserving (was infeasible pre-fix); in-box case unchanged; `test_tree_out_of_box_*` + `test_tree_in_box_unchanged`. (Wiring `OffsetScaling` *into* the tree formulation remains deferred per the plan's opt-in contract; the formulation does not silently mis-apply it.) |

### Phase 2 — reader hardening (PR `fix(nn): NN-3..NN-4`) — ✅ DONE (PR #411)

NN-3 (=C-27): validate `alpha==1 and beta==1 and transA==0` in `Gemm`, else raise (or apply
them). NN-4 (=C-27): verify `MatMul` weight orientation and that consecutive nodes actually
chain (output tensor of *k* = input of *k+1*), raise on a non-sequential graph;
validate that skipped `Reshape`/`Flatten` are genuine no-ops. Add ONNX fixtures with
`alpha≠1` and a non-chain graph.

**Resolved (PR #411, 2026-07-03; verified 2026-07-05).** ONNX (C-27): Gemm
`alpha`/`beta` folded into weight/bias, `transA=1` raises, MatMul weight verified
as `input[1]`, single-data-tensor dataflow chain from `graph.input[0]` → any
residual/branch topology raises `non-sequential`. sklearn (C-28,
`sklearn_reader.py`, carded separately in `correctness-issues.md`): `load_sklearn_mlp`
honors `out_activation_` (identity→LINEAR, logistic→SIGMOID, softmax→`ValueError`);
tree/ensemble readers raise `TypeError` on classifiers (kills the silent
logit/log-odds/`base_score` mis-embed); single-leaf `reshape` fixes the 0-d crash.
Fidelity vs onnxruntime/sklearn oracles in `test_nn_reader_fixes.py`
(before/after: Gemm 0.684→1.6e-07; MLPClassifier logit-vs-proba 3.93→1.1e-16;
GBClassifier silent `base_score=0` → loud `TypeError`).

### Phase 3 — rigor (PR `fix(nn): NN-5..NN-6`)

Outward rounding in interval propagation (or document the reliance on solver
tolerance); add `split_eps` to the tree right-split big-M.
