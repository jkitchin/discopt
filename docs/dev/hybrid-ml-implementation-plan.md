# Hybrid physics+ML models — implementation plan (trainable surrogates + neural-DAE training)

Status: READY FOR IMPLEMENTATION (phases HM0–HM3 unlocked; HM4 locked behind entry experiments)
Evidence base: `docs/dev/hybrid-ml-exploration-2026-07-10.md` + `scripts/hybrid_ml/` (all
claims below are measured there unless marked *hypothesis*).
Prototypical target: Lueg, Alves, Schicksnus, Kitchin, Laird & Biegler,
*A Simultaneous Approach for Training Neural Differential-Algebraic Systems of
Equations*, arXiv:2504.04665 (add to `docs/references.bib` in HM3.1).

---

## 0. Implementation contract (binding on the implementing agent)

This section is a contract, not guidance. A PR that violates a clause is wrong even
if its benchmarks improve.

### 0.1 Execution order

1. Start at **§9** (the task-level breakdown). Execute tasks in ID order within a
   phase; a task may start only when its listed dependencies are merged. HM0 tasks
   are independent of each other and of HM1–HM3.
2. **Phase HM4 is locked** until its entry experiment (per work item, §8) has been
   run and its results + derived task list have been recorded in §9/§11 of this
   file. Coding an HM4 item before its entry experiment is a contract violation.
3. Before writing any HM1 code, run `scripts/hybrid_ml/neural_dae_prototype.py` and
   `scripts/hybrid_ml/exp_c_paper_scale.py` and confirm the recorded numbers
   (status OPTIMAL; iterations within 2× of the recorded 19 / 17). If they no
   longer reproduce on current `main`, stop and record the regression in §11 —
   something moved underneath this plan.

### 0.2 Correctness invariants (zero slack, every PR)

Every PR must pass, and its description must state what was run and the result:

1. `pytest -m smoke` — 0 failures.
2. `pytest -m slow python/tests/test_adversarial_recent_fixes.py` — 0 failures.
3. `ruff check python/ && ruff format --check python/` and
   `mypy python/discopt/` clean on touched files.
4. New behavior ⇒ a regression test that fails before the change and passes after
   (state this explicitly in the PR).
5. Coverage stays ≥ 65 %.
6. **HM0–HM3 are solver-math-neutral**: no file under `crates/`, no change to
   relaxation, FBBT, B&B, certification, or `Model.solve` dispatch semantics. If a
   task in those phases appears to need one, stop (§0.6). HM4 items are
   bound-changing and follow the CLAUDE.md bound-changing regime: feature flag,
   default-off, differential bound test (new bound ≥ old bound AND ≤ true box
   optimum on fixed boxes) + feasible-point sampling, green on consecutive
   nightlies before any default flip.

### 0.3 Design invariants (load-bearing; do not trade away)

1. **Weights are plain `Variable`s and surrogates emit ordinary symbolic
   expressions.** Never wrap a surrogate in `dm.custom` / `CustomCall` — that
   disables global certification, integer variables, and `.nl` export
   (`modeling/core.py:433`).
2. **Matrix emission only.** Trainable layers are built with array-shaped weight
   `Variable`s and `MatMulExpression` (`h @ W`), never per-neuron scalar loops.
   `Variable @ Variable` and the batched pattern
   `dm.softplus(c[:, :, None] @ W1 + b1) @ W2` are verified working
   (`scripts/hybrid_ml/exp_b_matmul.py`).
3. **Smooth activations only on the trainable path**: LINEAR, TANH, SIGMOID,
   SOFTPLUS. RELU is refused with a loud `ValueError` naming the reason
   (nonsmooth under a gradient NLP solver) and the alternative (softplus).
   No silent substitution.
4. **The frozen path (`discopt.nn` formulations) is not modified.** Trainable and
   frozen are two regimes of one story, bridged by `freeze()` /
   `from_definition()`; they do not share constraint-emission code.
5. LLM safety invariant untouched; nothing in this plan touches `discopt/llm/`.

### 0.4 Measurement beats plan

If a measurement contradicts this document, the measurement wins. Record the
falsification in §11 (dated, in the `performance-plan.md` §6 house style),
re-scope, then continue. Known falsification already on record: *the paper-scale
decomposition premise* — full-space training converged in 17 iterations / 37 s at
1021 weights × 3 trajectories, so no decomposition machinery is built in this plan
(revisit only if HM2.4's larger gate instance fails).

### 0.5 Scope discipline

- General mechanisms only; nothing keyed to a named instance or to the batch-
  reactor example beyond tests/docs.
- One task per PR where practicable; PR titles name the task ID (e.g.
  `feat(nn): HM1.1 TrainableDense/TrainableNetwork`).
- Do not rebuild what exists: collocation transcription (`discopt.dae`),
  least-squares detection (`_jax/least_squares.py`), warm-start validation
  (`discopt/warm_start.py`), readers (`discopt/nn/readers/`), frozen formulations
  (`discopt/nn/formulations/`). Check `docs/design/relaxation-catalog.md` before
  any HM4 envelope work.

### 0.6 Stop-and-escalate conditions

Stop and surface to the maintainer when: a kill criterion fires; an HM0–HM3 task
seems to require touching solver math (§0.2.6); `exp_c_paper_scale.py` stops
reproducing (§0.1.3); or an HM4 gate could only be met by weakening a validation
or certification guard (the gate loses — CLAUDE.md philosophy §1).

### 0.7 Status table (living — update in the same PR as the work)

| Task | Status | PR | Notes |
|---|---|---|---|
| HM0.1 | done | (this branch) | option (b): docstring corrected + pounce version TODO |
| HM0.2 | done | (this branch) | ValueError refusals in `_extract.py`, `callbacks.py`; 2 tests |
| HM1.1 | not started | | |
| HM1.2 | not started | | |
| HM1.3 | not started | | |
| HM1.4 | not started | | |
| HM2.1 | not started | | |
| HM2.2 | not started | | |
| HM2.3 | not started | | |
| HM2.4 | not started | | |
| HM3.1 | not started | | |
| HM3.2 | not started | | |
| HM4.* | LOCKED | | entry experiments first (§8) |

---

## 1. What this builds

A first-class hybrid-modeling capability: models that are part physics (algebraic
constraints, mass/energy balances, ODE/DAE dynamics via `discopt.dae`) and part
trainable ML (dense NNs, kernel/GP-mean expansions), trained *simultaneously* with
the discretized states as one sparse NLP — the arXiv:2504.04665 recipe — plus the
train-then-freeze bridge into the existing frozen-surrogate optimization path.

User-visible surface when done:

```python
import discopt.modeling as dm
from discopt.nn import TrainableNetwork
from discopt.dae import ContinuousSet, DAEBuilder, Trajectory, fit_trajectories

m = dm.Model()
net = TrainableNetwork(m, sizes=[1, 30, 30, 1], activation="softplus",
                       weight_bounds=(-8.0, 8.0), name="rate")

def rhs(t, s, a, u):
    r = net(s["cA"])                      # expression in, expression out
    return {"cA": -r, "cB": r}

fit = fit_trajectories(
    m,
    trajectories=[Trajectory(t_data=t1, y_data={"cA": yA1, "cB": yB1},
                             initial={"cA": 1.0, "cB": 0.0}), ...],
    states=[("cA", dict(bounds=(0.0, 1.5))), ("cB", dict(bounds=(0.0, 1.5)))],
    rhs=rhs, t_span=(0.0, 2.0), nfe=20, ncp=2,
)
m.minimize(fit.least_squares() + 1e-4 * net.l2_penalty())

from discopt.nn import train
result = train(m, initial_solution=fit.warm_start() | net.initial_values(seed=0))

frozen = net.freeze(result)               # NetworkDefinition for the frozen path
```

Non-goals (this plan): decomposition machinery (falsified need, §0.4), GP
hyperparameter learning inside the NLP, decision-tree *training*, ReLU training,
any `discopt.nn` frozen-formulation change, AMP vectorized-constraint support
(HM4 candidate only).

## 2. Evidence base (all measured, 2026-07-10)

| Fact | Where measured |
|---|---|
| Neural-DAE training composes from existing parts; 19 IPM iters / 1.3 s at 19 weights | `scripts/hybrid_ml/neural_dae_prototype.py` |
| 1021-weight, 3-trajectory full-space training: 17 iters / 37 s from Glorot init | `scripts/hybrid_ml/exp_c_paper_scale.py` |
| `Variable @ Variable` + batched layer emission works | `scripts/hybrid_ml/exp_b_matmul.py` |
| Gauss-Newton detector fires on `DAEBuilder.least_squares` sums; build 0.78→0.02 s | `scripts/hybrid_ml/exp_a_gauss_newton.py` |
| Kernel (linear-in-α) surrogate: 12 iters, zero init, no restarts | `scripts/hybrid_ml/gp_dae_prototype.py` |
| Certified global training of scalarized kernel hybrid: 1.1 % scaled gap, 127 nodes / 300 s | `scripts/hybrid_ml/exp_e2_global_scalar.py` |
| Default path on abstained-convexity continuous models = uncertified local NLP; no force-spatial knob | exploration doc §6.4 |
| AMP is scalar-constraint-only (clean error on vectorized models, post-fix) | exploration doc §6.4 |
| pounce-solver 0.7.0 lacks `set_kkt_schur_block`; `nlp_pounce.py:194` cites nonexistent `discopt.aggregation.schur` | exploration doc §6.2 |

## 3. Module layout

```
python/discopt/nn/trainable.py      # NEW: TrainableDense, TrainableNetwork,
                                    #      TrainableKernelExpansion, train()
python/discopt/dae/fit.py           # NEW: Trajectory, fit_trajectories, TrajectoryFit
python/discopt/warm_start.py        # ADD: unflatten_solution()
python/discopt/nn/__init__.py       # export the new names (lazy, as existing style)
python/discopt/dae/__init__.py      # export Trajectory, fit_trajectories
python/tests/test_nn_trainable.py   # NEW
python/tests/test_dae_fit.py        # NEW
docs/notebooks/neural_dae.ipynb     # NEW (HM3)
```

---

## 4. Phase HM0 — hygiene from the exploration (small, independent, ~0.5 EW total)

**HM0.1 — fix the stale Schur docstring.** `solvers/nlp_pounce.py:194` (and the
mirrored text in `solve_nlp_from_model`) references
`discopt.aggregation.schur.kkt_schur_indices(model)`, which exists nowhere in the
repo. Either (a) implement that helper (KKT-space indices for a chosen variable
block, block order `x, slack, eq-dual, ineq-dual` per the pounce docstring), or
(b) correct the docstring to describe how to construct the indices by hand and
note the pounce ≥ version requirement. Decision rule: implement (a) only if the
current pounce dependency actually exposes `set_kkt_schur_block` (0.7.0 does
not); otherwise do (b) plus a TODO naming the pounce version gate. Test: docstring
example (if (a)) round-trips on a 3-variable model.

**HM0.2 — loud-refusal exception types on the two remaining scalar-index sites.**
`export/_extract.py:98` and `callbacks.py:212` still feed possibly-sliced
subscripts to `np.ravel_multi_index`; unlike the analysis paths fixed on this
branch (which must abstain), these are refusal paths that should raise
`ValueError` with the variable name and subscript, not a raw `TypeError`. Extend
`python/tests/test_tightening_sliced_index.py` with one test per site (fails
before, passes after).

## 5. Phase HM1 — trainable surrogate API (`discopt/nn/trainable.py`, ~1.5–2 EW)

### HM1.1 — `TrainableDense` + `TrainableNetwork`

```python
class TrainableDense:
    """One dense layer with Variable weights: y = act(x @ W + b)."""
    def __init__(self, model, n_in: int, n_out: int, *,
                 activation: str | Activation = "tanh",
                 weight_bounds: tuple[float, float] = (-8.0, 8.0),
                 name: str): ...
    W: Variable   # shape (n_in, n_out)
    b: Variable   # shape (n_out,)
    def __call__(self, x): ...          # expression -> expression

class TrainableNetwork:
    def __init__(self, model, sizes: Sequence[int], *,
                 activation: str | Activation | Sequence[...] = "tanh",
                 output_activation: str | Activation = "linear",
                 weight_bounds: tuple[float, float] = (-8.0, 8.0),
                 name: str): ...
    layers: list[TrainableDense]
    def __call__(self, *inputs): ...    # see input contract below
    def parameters(self) -> list[Variable]: ...
    def n_parameters(self) -> int: ...
    def l2_penalty(self): ...           # sum of squares over all parameters
    def initial_values(self, seed: int | np.random.Generator = 0,
                       scale: float = 0.8) -> dict[Variable, np.ndarray]: ...
    def freeze(self, values) -> NetworkDefinition: ...
    @classmethod
    def from_definition(cls, model, definition: NetworkDefinition, *,
                        weight_bounds=(-8.0, 8.0), name: str) -> "TrainableNetwork": ...
```

Binding details:

- **Emission**: reuse the verified batched pattern. For an expression `x` of shape
  `S` (scalar or any array shape) and `n_in` inputs: scalar-per-point inputs are
  passed as separate arguments and stacked along a new trailing axis; a single
  argument with trailing axis `n_in` is used as-is. Layer forward is
  `act(x @ W + b)` via `MatMulExpression`; a 1-input network accepts a bare
  `(nfe, ncp)` expression and internally lifts to `(nfe, ncp, 1)` with `[:, :, None]`
  (verified working). Output squeezes the trailing axis when `sizes[-1] == 1`.
  Add shape validation with actionable error messages — this is the part users
  will get wrong.
- **Activations**: map through the existing `discopt.nn.network.Activation` enum;
  RELU raises per §0.3.3. Use `dm.tanh` / `dm.sigmoid` / `dm.softplus` intrinsics
  only (native, relaxable — no CustomCall).
- **`initial_values`**: Glorot-style, `scale/sqrt(n_in)` std for `W`, `0.3·scale`
  for `b` (the exact recipe measured in `exp_c_paper_scale.py`); deterministic
  given `seed`.
- **`freeze(values)`**: `values` is a `dict[Variable, np.ndarray]`, an `NLPResult`
  (use `unflatten_solution`, HM2.1), or a `SolveResult` (use `.value`). Returns a
  `NetworkDefinition` whose `forward()` matches the symbolic net; parity is a test.
- **`from_definition`**: variables created with `lb/ub = weight_bounds` but
  `initial_values()` returning the definition's weights — the fine-tuning /
  warm-from-pretrained path (this is how the existing `readers/` become
  *initializers* instead of freezers; no reader changes needed).

### HM1.2 — `TrainableKernelExpansion`

```python
class TrainableKernelExpansion:
    """r(x) = sum_j alpha_j * k(x, c_j); linear in trainable alpha.

    kernel="rbf": k(x, c) = exp(-(x - c)^2 / (2 l^2)) (1-D input, v1).
    """
    def __init__(self, model, centers: np.ndarray, *, lengthscale: float,
                 kernel: str = "rbf", alpha_bounds=(-50.0, 50.0), name: str): ...
    alpha: Variable
    def __call__(self, x): ...
    def parameters(self) -> list[Variable]: ...
    def l2_penalty(self): ...
    def initial_values(self) -> dict[Variable, np.ndarray]: ...   # zeros — measured sufficient
```

Fixed centers/lengthscale by design (hyperparameters stay outside the NLP —
exploration doc §4); the docstring must say so and say why. 1-D input only in v1;
reject multi-D with a clear error rather than emitting an untested formulation.

### HM1.3 — `train()` convenience

```python
def train(model, *, initial_solution: dict | None = None,
          gauss_newton: bool = True, options: dict | None = None,
          backend: str = "auto") -> NLPResult
```

Thin: `validate_initial_solution` → `NLPEvaluator(model, gauss_newton=...)` →
`solvers.nlp_pounce.solve_nlp` (or `nlp_ipopt` when `backend="cyipopt"`,
`"auto"` = pounce, cyipopt fallback — mirror `_default_nlp_solver` behavior). It
must NOT reimplement solve logic and must NOT route through `Model.solve()` (which
would try global dispatch; exploration doc §6.4). Docstring states plainly: this
is a **local** solve; the result carries no global certificate.

### HM1.4 — tests for HM1 (single PR with 1.1–1.3 or split; each test marked)

`python/tests/test_nn_trainable.py`:

1. Shape/broadcast unit tests for `TrainableDense`/`TrainableNetwork` over scalar,
   `(n,)`, `(nfe, ncp)` inputs; mismatch raises with the variable name in the
   message.
2. ReLU refusal on the trainable path.
3. **End-to-end regression (smoke-marked)**: the `neural_dae_prototype.py`
   experiment rebuilt on the new API (1-6-1 tanh, nfe=20, ncp=2, seed 0):
   status OPTIMAL, iterations ≤ 40, recovered-rate RMSE ≤ 0.03 over the visited
   range, resimulation RMSE ≤ 2× noise. Thresholds have ≥ 2× margin on the
   measured values (19 iters, 0.0107, 1× noise) — they gate regressions, not luck.
4. Kernel variant of (3): zero init, iterations ≤ 25 (measured 12).
5. `freeze`/`from_definition` round-trip: `net.freeze(sol).forward(x)` matches the
   trained symbolic net at 10 random points to 1e-8; `from_definition(freeze(...))`
   `initial_values` reproduce the weights exactly.
6. Gauss-Newton: `NLPEvaluator(m, gauss_newton=True).is_gauss_newton` is True for
   the (3) model's objective (guards the detector against regressions on this
   objective class).

## 6. Phase HM2 — multi-trajectory fitting glue (`discopt/dae/fit.py`, ~1–1.5 EW)

### HM2.1 — `unflatten_solution` (prerequisite, tiny)

`discopt/warm_start.py`:

```python
def unflatten_solution(model, x_flat: np.ndarray) -> dict[Variable, np.ndarray]
```

Inverse of `validate_initial_solution`'s flattening (same ordering contract:
`model._variables` declaration order). The exploration scripts all hand-rolled
this via the private `_compute_var_offset` — make it public API once. Round-trip
property test: `unflatten(validate_initial_solution(m, d))` == `d` (up to
clamping) on a model with scalar + multi-dim variables.

### HM2.2 — `Trajectory` / `fit_trajectories` / `TrajectoryFit`

```python
@dataclass
class Trajectory:
    t_data: np.ndarray
    y_data: dict[str, np.ndarray]        # state name -> observations
    initial: dict[str, float | np.ndarray]
    weights: dict[str, float | np.ndarray] | None = None   # per-state or per-point LS weights

def fit_trajectories(model, *, trajectories, states, rhs, t_span, nfe, ncp=2,
                     scheme="radau", algebraics=(), controls=(),
                     align_grid: bool = False, name: str = "traj") -> TrajectoryFit
```

Behavior (all composition, no new math): one `ContinuousSet` + `DAEBuilder` per
trajectory on the shared `model` (unique names `f"{name}{k}"`); same `rhs`
callable for every builder (weights shared through closure — the measured
`exp_c_paper_scale.py` pattern); `align_grid=True` routes through
`dae.align_time_grid` per trajectory.

`TrajectoryFit` surface:

```python
class TrajectoryFit:
    builders: list[DAEBuilder]
    def least_squares(self, interpolate: bool = True): ...   # weighted sum over all trajectories/states
    def warm_start(self) -> dict[Variable, np.ndarray]: ...  # data-interpolated states (measured init)
    def extract(self, result, k: int, state: str): ...       # delegate to builders[k].extract_solution
    def resimulate_rmse(...)  # NO — out of scope, keep the class thin; users own assessment
```

(The struck line is deliberate: no scipy-based assessment helpers in the library;
they live in the notebook.)

`warm_start()` must accept both an `NLPResult` path (via HM2.1) and merge cleanly
with `net.initial_values()` via plain dict union — no special types.

### HM2.3 — tests for HM2

`python/tests/test_dae_fit.py`: (1) HM2.1 round-trip; (2) two-trajectory shared-
weight fit with the kernel surrogate (fast: nfe=8) — OPTIMAL, both trajectories'
collocation-vs-data RMSE below 2× noise; (3) `align_grid=True` produces element
boundaries containing every interior measurement time (property of
`align_time_grid`); (4) constraint/variable counts scale as
`n_traj × (per-trajectory count)` + shared weights exactly (guards accidental
variable duplication).

### HM2.4 — paper-scale gate (regression-marked, `slow`)

Port `exp_c_paper_scale.py` onto the HM1+HM2 API as
`python/tests/test_neural_dae_scale.py::test_paper_scale_full_space` (marked
`slow`): 1-30-30-1 softplus, 3 trajectories, OPTIMAL, iterations ≤ 40 (measured
17), wall ≤ 5 min. **This is the canary for the falsified-decomposition premise**
(§0.4): if it starts failing, the premise needs re-examination — record in §11,
do not silently bump thresholds.

## 7. Phase HM3 — documentation (~0.5–1 EW)

**HM3.1 — `docs/notebooks/neural_dae.ipynb`.** Follows CLAUDE.md notebook rules
exactly: lives in `docs/notebooks/`, `{cite:p}`/`{cite:t}` citations (add BibTeX:
arXiv:2504.04665; Biegler 2010 nonlinear programming (key likely exists — check
`references.bib` first); Chen et al. 2018 neural ODEs; optionally
Schweidtmann & Mitsos for the frozen-surrogate/global connection), added to
`docs/_toc.yml`, `jupyter-book build docs/` with zero warnings. Content: the
batch-reactor story end-to-end on the new API — build, train, assess honestly
(resimulate the learned ODE with scipy — assessment code lives here, not in the
library), freeze, then *use* the frozen net in a small design optimization to
close the train-then-freeze loop. One section on the kernel surrogate and when
linear-in-parameters is the better choice (conditioning; measured 12-iter/zero-init
result).

**HM3.2 — CLAUDE.md + module docs.** Add `nn/trainable.py` and `dae/fit.py` to the
CLAUDE.md architecture section (one sentence each); extend `discopt/nn/__init__.py`
module docstring with the trainable-vs-frozen regime distinction and the
train-then-freeze bridge.

## 8. Phase HM4 — LOCKED: global-certification track (bound-changing, research)

Each item requires its entry experiment first (§0.1.2), its own flag, and the
§0.2.6 bound-changing regime. These are *not* required for HM1–HM3 to ship.

**HM4.1 — α×kernel product envelope.** Hypothesis: the dominant bound gap in
certified kernel-hybrid training is AMP/spatial soundly omitting
`alpha * exp(q(x))` products (`q` concave-quadratic) from the relaxation; with
`z = exp(q(x)) ∈ (0, e^{max q}]` bounded, McCormick over `alpha × z` plus the
existing exp/quadratic envelopes gives a valid tightening. Check
`docs/design/relaxation-catalog.md` first — do not rebuild an existing rule.
Entry experiment: root bound on `exp_e2_global_scalar.py`'s instance before/after
a hand-built reformulation that introduces `z_j` as explicit auxiliary variables
(`z_j == dm.exp(...)`; products become plain bilinear `alpha_j * z_j` that the
existing McCormick machinery already handles). If the *reformulated* model's root
bound is not materially tighter, kill: no envelope work justified. If it is
tighter, the shippable item may simply be **a documented reformulation recipe (or
an opt-in structural rewrite pass)**, not new envelope math.

**HM4.2 — force-spatial escape hatch.** Today a pure-continuous model whose
convexity classification abstains routes to an uncertified local NLP with no way
to request the sound spatial B&B (`solver.py` ~4436 region; `nlp_bb=False` does
not reach it). Design decision for the maintainer before coding (§0.6): a
`solver="spatial"` selector vs. honoring `skip_convex_check=False` + explicit
flag. Entry experiment: run the scalarized kernel instance through the forced
path and confirm it reproduces the AMP-certified result (`bound ≤ incumbent`,
sane nodes). Must not alter any default route.

**HM4.3 — global path for vectorized (DAEBuilder) models.** Two candidate routes:
(a) AMP/classifier support for sliced vectorized constraints (large; touches many
scalar-assuming extractors), or (b) an automatic scalarization pass applied when a
vectorized continuous model is sent to a global backend (semantics-preserving
rewrite; bounded blowup `nfe·ncp·n_states`). Entry experiment for (b): scalarize
the exp-E vectorized instance mechanically and diff solve results vs the
hand-scalarized `exp_e2_global_scalar.py`. Kill for (b): scalarization changes
certified objective/bound on a fixed instance (it must be exactly
representation-neutral) or blows model build time by > 10× at nfe=20.

## 9. Task list (executable)

| ID | Task | Deps | Est. | Test / gate |
|---|---|---|---|---|
| HM0.1 | Schur docstring fix (or helper, per decision rule §4) | — | 0.1 EW | docstring example test if helper |
| HM0.2 | ValueError refusal in `export/_extract.py`, `callbacks.py` | — | 0.2 EW | extend `test_tightening_sliced_index.py`, fails-before |
| HM1.1 | `TrainableDense`/`TrainableNetwork` | §0.1.3 rerun | 0.7 EW | HM1.4 tests 1,2,3,5 |
| HM1.2 | `TrainableKernelExpansion` | HM1.1 (file/layout) | 0.3 EW | HM1.4 test 4 |
| HM1.3 | `train()` | HM1.1 | 0.2 EW | HM1.4 tests 3,6 route through it |
| HM1.4 | `test_nn_trainable.py` complete | HM1.1–1.3 | 0.3 EW | all six tests green; smoke marks |
| HM2.1 | `unflatten_solution` | — | 0.1 EW | round-trip property test |
| HM2.2 | `Trajectory`/`fit_trajectories`/`TrajectoryFit` | HM1.1, HM2.1 | 0.6 EW | HM2.3 |
| HM2.3 | `test_dae_fit.py` | HM2.2 | 0.3 EW | 4 tests green |
| HM2.4 | paper-scale slow gate | HM2.2 | 0.2 EW | `test_neural_dae_scale.py`, `slow` mark |
| HM3.1 | `neural_dae.ipynb` + bibliography + toc | HM1, HM2 | 0.7 EW | `jupyter-book build docs/` zero warnings |
| HM3.2 | CLAUDE.md / module docstrings | HM3.1 | 0.1 EW | review |
| HM4.1 | α×kernel: entry experiment then recipe/pass | HM1.2; UNLOCK | — | differential bound + feasible-point, flagged |
| HM4.2 | force-spatial hatch | maintainer decision; UNLOCK | — | reproduces AMP-certified result |
| HM4.3 | vectorized→global (scalarization pass) | UNLOCK | — | representation-neutrality diff |

Suggested PR partitioning: HM0.1+HM0.2 (one PR), HM1.1–HM1.4 (one or two),
HM2.1–HM2.4 (one), HM3 (one). HM4 items: one PR each, after their entry
experiment is recorded here.

## 10. Out of scope (do not do under this plan)

Decomposition (Benders/Lagrangian/ADMM consensus) for training — premise
falsified at target scale; GP hyperparameters in the NLP; GP predictive-variance
constraints (frozen-GP feature, separate proposal); decision-tree training or
linear-leaf tree embedding in dynamics; ReLU training; multi-D kernel inputs;
any `crates/` change; performance work on expression-graph build time (measure
first if it hurts, then propose separately).

## 11. Falsification / deviation log (append-only, dated)

- 2026-07-10 (pre-plan): decomposition-needed premise falsified at 1021 weights ×
  3 trajectories (17 iters / 37 s full-space). Plan contains no decomposition
  work as a result.
- (append here)
