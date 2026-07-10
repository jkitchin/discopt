# Hybrid physics+ML models in discopt — exploration (2026-07-10)

**Question.** Can discopt support *hybrid* models — part physics, part ML (e.g. a
neural network for a rate constant or an ODE right-hand side) — trained the way
Lueg, Alves, Schicksnus, Kitchin, Laird & Biegler do in *A Simultaneous Approach
for Training Neural Differential-Algebraic Systems of Equations*
(arXiv:2504.04665)? And how should the `discopt.nn` module generalize to make
this first-class, including ML models beyond NNs (Gaussian processes,
linear-model decision trees)?

**Answer (measured, not hypothesized).** The simultaneous neural-DAE training
recipe is *already composable from existing discopt parts with zero framework
changes*. Three entry experiments below train a hybrid batch-reactor model
(physics: mass balances; ML: unknown rate law) end-to-end as a single
collocation NLP. What is missing is ergonomics (a trainable-surrogate API),
not infrastructure.

## 1. The paper's recipe, mapped onto discopt

The paper: embed a feedforward NN (softplus, e.g. 2×30) as algebraic
constraints inside a DAE; discretize everything with orthogonal collocation on
finite elements (Lagrange-Radau, e.g. 20 elements × 2 points, via Pyomo.DAE);
make NN weights *and* discretized states joint decision variables; minimize
least-squares data mismatch + regularization; solve the one large sparse NLP
with IPOPT. No ODE integrator in the loop, no adjoints, algebraic constraints
enforced exactly.

| Paper ingredient | discopt counterpart | Status |
|---|---|---|
| Pyomo.DAE collocation transcription | `discopt.dae.DAEBuilder` + `ContinuousSet` (Radau/Legendre, non-uniform elements) | shipped |
| NN as algebraic constraints | `dm.tanh`/`dm.sigmoid`/`dm.softplus`/`dm.exp` intrinsics; array `Variable`s; broadcasting | shipped |
| Weights as decision variables | plain `m.continuous(shape=...)` closed over by the ODE RHS callable | shipped (no helper API) |
| Least-squares mismatch at measurement times | `DAEBuilder.least_squares(state, t_data, y_data)` — exact polynomial interpolation via `state_at` | shipped |
| Large sparse local NLP solve | `solvers.nlp_pounce.solve_nlp_from_model` / `nlp_ipopt` (cyipopt); sparse Jac/Hess | shipped |
| Warm start | `warm_start.validate_initial_solution` → flat x0 | shipped |
| Gauss-Newton for sum-of-squares | `_jax/least_squares.py`, `Model.solve(gauss_newton=True)` | shipped, untested on this class |
| Trainable-NN convenience layer | — | **gap** |
| Multi-trajectory / multi-experiment training | — (loop by hand; `estimate.py` is single-shot per model) | **gap** |

Key structural facts found while mapping:

- `DAEBuilder.set_ode(rhs)` takes `rhs(t, states, algebraics, controls) ->
  dict[str, Expression]` and invokes it **once** with vector-shaped `(nfe, ncp)`
  expressions (`dae/collocation.py:492`). Any expression the modeling layer can
  build — including one that closes over weight `Variable`s — is a valid RHS.
  This is the hook that makes the whole exploration cheap.
- `discopt.nn` (OMLT-style) embeds **frozen** networks only: every formulation
  bakes weights to `np.float64` constants (`nn/formulations/full_space.py:130`).
  It is the right tool for *deploying* a trained surrogate inside a design/control
  MINLP, and the wrong tool for *training* — those are two regimes of the same
  hybrid-model story.
- `dm.custom` (opaque JAX callable) would also "work" but disables global
  certification, integer variables, and `.nl` export (`modeling/core.py:433`).
  Build surrogates symbolically; never via `CustomCall`.

## 2. Entry experiments (all in `scripts/hybrid_ml/`)

Setup common to all three: batch reactor A→B, truth `dcA/dt = -r(cA)`,
`r(c) = 1.5 c² / (0.3 + c)` (Michaelis–Menten-like, unknown to the model);
15 measurements of both states on [0, 2] with σ=0.01 Gaussian noise; physics
part is the stoichiometric mass-balance structure; ML part replaces `r`.
Collocation: 20 elements × 2 Radau points → 120 state variables. Trained by a
single POUNCE (in-house IPM) solve via `solve_nlp_from_model`; states
warm-started by interpolating the noisy data onto the collocation grid
(paper-style), weights at small random / zero.

| experiment | surrogate | params | iters | wall | rate-law RMSE (rel.) | resim RMSE vs truth |
|---|---|---|---|---|---|---|
| `neural_dae_prototype.py` | 1–6 tanh–1 NN | 19 | 19 | 1.3 s | 0.0107 (1.6 %) | 0.0025 (< noise) |
| `neural_dae_wide.py` | 1–24 softplus–1 NN | 73 | 18 | 9.6 s | 0.0054 (0.8 %) | 0.0024 (< noise) |
| `gp_dae_prototype.py` | GP mean / RBF expansion, 12 fixed centers, trainable α | 12 | 12 | 2.6 s | 0.0382 (5.7 %) | 0.0033 (< noise) |

Observations:

- All three converge to `OPTIMAL` in ≤ 19 IPM iterations from a trivial
  initialization. The simultaneous transcription does the heavy lifting, as in
  the paper.
- "Resim RMSE" is the honest generalization test: integrate the *learned* ODE
  with scipy from t=0 and compare to the noiseless truth. All models sit below
  the noise floor.
- The GP/RBF variant is **linear in its trainable parameters**, so the ML block
  contributes no nonconvexity of its own (only the dynamics couple it) — fewer
  iterations, no random initialization, no weight-symmetry local minima. Its
  slightly worse rate-law RMSE is an expressiveness/edge-extrapolation effect of
  12 fixed centers, not an optimization failure.
- Wall time grows with expression-graph size (Python-loop scalar graph per
  hidden unit), not iteration count → the v1 API should emit matrix-shaped
  NN layers (`MatMulExpression`) instead of per-neuron scalar loops.

## 3. The decomposition question

The paper needed a decomposition to solve its larger instances — the full-space
NLP becomes hard as networks/data grow (nonconvex weights × long horizons ×
many trajectories). discopt is unusually well positioned here; this is where it
could go *beyond* the paper rather than reproduce it:

1. **Structure-exploiting linear algebra (same NLP, faster factorization).**
   Collocation training NLPs are block-bordered: per-element (or
   per-trajectory) state blocks coupled only through continuity constraints and
   the *shared* surrogate weights. `solvers/nlp_pounce.py` already exposes
   `kkt_schur_block` / `ordering` passthroughs to POUNCE (Schur partition of
   the KKT system, correctness-safe fallback). Pinning the weight block as the
   Schur complement is the classic parameter-estimation trick, and needs no new
   solver code — just a helper that computes the KKT indices from the model
   structure. This should be the *first* decomposition experiment: same
   solution, measured factorization speedup.
2. **True decomposition (different algorithm).** `python/discopt/decomposition/`
   already has Benders, Lagrangian, a structure-detection graph layer, and an
   advisor. Multi-experiment training (one collocation block per trajectory,
   consensus on shared weights) is exactly the block-angular shape Lagrangian /
   ADMM-style consensus handles; each subproblem is a small single-trajectory
   fit. This mirrors how the neural-DAE community scales training and is a
   natural `discopt.decomposition` showcase.
3. **Warm-start homotopy (what the paper effectively did by hand).** Train on a
   short horizon / coarse grid, extend, re-solve warm-started. Cheap to script
   with `validate_initial_solution`; worth codifying in the eventual API.

## 4. Beyond NNs: GPs and linear-model decision trees

The generalization the `nn` module wants is not "more network types" but a
**common trainable-surrogate abstraction** with per-family formulations:

- **Gaussian processes.** Two regimes:
  - *Frozen GP* (trained elsewhere): posterior mean `m(x) = Σ αᵢ k(x, xᵢ)` is a
    smooth algebraic expression (`dm.exp` of quadratics for RBF) — embeddable
    today exactly like a frozen NN; the predictive *variance* is also algebraic,
    enabling **trust-region constraints** (`σ²(x) ≤ ε`) that confine the
    optimizer to where the surrogate is trustworthy — important for global
    optimization over hybrid models (cf. Schweidtmann/Misener-style GP
    embeddings).
  - *Simultaneous training* with fixed kernel centers/lengthscales and trainable
    α (experiment 3 above): linear-in-parameters, best conditioning of the
    three. Hyperparameter (lengthscale) learning does **not** fit the algebraic
    NLP cleanly (log-marginal likelihood needs matrix inverses/determinants);
    the honest split is: hyperparameters outside (marginal-likelihood or CV
    loop), α inside the NLP — or treat ℓ as a bounded NLP variable and accept
    the nonconvexity, which is worth a falsification experiment before building
    anything.
- **Linear-model decision trees** (constant- or linear-leaf trees):
  - *Frozen*: `nn/formulations/tree_ensemble.py` already does Mišić-style
    per-leaf MILP encoding for **constant** leaves; extending to linear leaves
    (OMLT's linear-tree formulations; big-M or GDP via the existing
    `gdpopt_loa` machinery) is a scoped, incremental change. But note the
    dynamics caveat: a piecewise rate law makes the ODE RHS discontinuous →
    collocation's smoothness assumption breaks at switching times, and each
    collocation point needs its own leaf binaries → an MINLP whose size scales
    with `nfe × ncp`. Embedding trees in *dynamics* is therefore a research
    problem (hybrid systems), not a formulation-only change; embedding them in
    *algebraic* (steady-state) parts of a hybrid model works today.
  - *Simultaneous training* of a tree inside the DAE = optimal-decision-tree
    MINLP (Bertsimas-style) coupled to collocation. Genuinely novel — discopt
    is one of few tools where this is even expressible — but should be treated
    as a stretch experiment, not roadmap.
- **discopt-specific differentiators** worth naming in any writeup/paper:
  - *Certified global training* of small bounded-weight surrogates via spatial
    B&B (the paper is local-only). Plausible only for very small nets/GP-α
    problems; entry experiment: global-solve experiment 3 (linear ML block,
    bilinear coupling only through dynamics) and measure.
  - *Train-then-freeze pipeline*: train weights in the NLP (this work), freeze
    into the existing `nn` formulations, then do certified global design/control
    over the trained hybrid model — the two halves of the module become one
    story.

## 5. Proposed shape of the generalization (for discussion, not commitment)

```
discopt.surrogates (or generalize discopt.nn)
├── TrainableDense(m, n_in, n_out, activation, weight_bounds, init=...)   # emits Variable weights
├── TrainableKernelExpansion(m, centers, kernel, ...)                     # linear-in-α; GP mean
├── FrozenPredictor(...)              # today's nn module, incl. GP mean/variance, linear trees
└── common protocol: callable on expressions → expression; .parameters() → [Variable];
    .initialize(reader_output) → warm start from sklearn/torch/onnx (readers/ reused for
    *initialization*, not just freezing)
```

plus a `dae`-level convenience for multi-trajectory fitting (shared parameters
across `DAEBuilder` instances) and a structure helper emitting
`kkt_schur_block` indices for the weight block.

Deliberately *not* proposed: ReLU on the training path (nonsmooth; the paper
uses softplus for the same reason), `CustomCall` wrapping, and any change to
solver math.

## 6. Round-2 experiments — results (2026-07-10, same session)

All five §6 questions were run to ground; experiment scripts are in
`scripts/hybrid_ml/` (`hybrid_common.py` + `exp_*.py`). Results are measured.

1. **Gauss-Newton detector: fires.** `NLPEvaluator(m, gauss_newton=True)` on the
   experiment-1 model: `is_gauss_newton=True`, identical iterations (19) and
   objective to all printed digits — expected, since the residuals are linear in
   the collocation states so GN is exact — with evaluator build 0.78 s → 0.02 s
   and solve wall 3.4 s → 1.9 s (compile savings). No kill.
2. **Schur weight-block: environment-blocked, with findings.** The PyPI
   `pounce-solver` 0.7.0 predates `set_kkt_schur_block`; the wrapper silently
   degrades to full-space (by design). Worse, the `nlp_pounce.py:194` docstring
   points to `discopt.aggregation.schur.kkt_schur_indices`, which exists nowhere
   in the repo — stale reference, needs fixing. Given result 3 below, the Schur
   experiment also matters less than §3 assumed at this scale.
3. **Paper-scale probe: the decomposition pain does NOT reproduce (falsifies
   §3's premise at this scale).** 1-30-30-1 softplus net (1021 weights,
   matrix-emitted via `Variable @ Variable` — see 5), three trajectories
   (cA0 ∈ {1.0, 0.8, 0.6}) sharing weights, 1381 variables total: full-space
   POUNCE converges from a random Glorot init in **17 iterations / 37 s**,
   rate-law RMSE 2 % relative, resimulation below the noise floor on all three
   trajectories. Caveat: a benign 1-D-input rate law; the paper's harder case
   studies may still need decomposition — but at this problem shape, full-space
   is simply fine.
4. **Certified global training: achieved, after fixing three crashes.**
   - The *default* solve path cannot get there: the convexity classifier
     abstains on exp-of-quadratic kernels, and for
     continuous models with unknown convexity `solve()` routes to a
     best-effort local NLP (`gap_certified=False`) regardless of
     `nlp_bb=False` — there is no "force spatial" knob on that path today.
   - `solver="amp"` crashed on DAE models: three call sites assume scalar
     subscripts on `IndexExpression`s while `DAEBuilder` emits sliced ones
     (`var[:, 1:]`). Fixed in this branch (see below). Post-fix, AMP *errors
     cleanly* on vectorized models (its MILP builder + point evaluator are
     scalar-constraint-only) — vectorized support in AMP is a real work item.
   - **Scalarized** collocation (same math, one scalar equation per node,
     cB eliminated by conservation; 4 RBF centers, nfe=6, ncp=2, 43 vars):
     the global path runs — 127 nodes / 300 s, valid dual bound 0.010477 vs
     incumbent 0.021576, `gap_certified=True` at scaled gap
     (obj−bound)/(1+|obj|) ≈ 1.1 %. The warm-started **local optimum was
     certified (near-)globally optimal** — the capability the paper's
     local-only approach lacks. Bound quality is limited by AMP soundly
     *omitting* `α·exp(−(x−c)²/2ℓ²)` products from its MILP relaxation
     ("cannot be linearized safely"): a product-with-bounded-nonlinear-factor
     envelope (kernel output ∈ (0,1], so McCormick over α×z applies) is the
     concrete relaxation-catalog item that would tighten this.
5. **Matrix-shaped NN emission: works today.** `Variable @ Variable` bilinear
   matmul and the batched layer pattern
   `dm.softplus(c[:, :, None] @ W1 + b1) @ W2` both compile and solve
   (probes in session scratchpad; pattern used by experiment 3). The v1 API
   should emit this form; the per-neuron scalar loop in the round-1 scripts is
   what made the 24-unit net's wall time grow.

### Bugs found and fixed in this branch (regression-tested)

All one defect class: *structure-analysis code assumes an `IndexExpression`
subscript is one integer per axis; DAE-transcribed models routinely carry
slices.* Analysis paths must **abstain** (return None / classify not-linear) on
non-scalar subscripts, never crash:

- `_jax/nonlinear_bound_tightening.py` `FlatVariableMetadata.scalar_flat_index`
  — crashed in AMP presolve (`TypeError: only int indices permitted`).
- `_jax/convexity/rules.py` `_hash_index` — tuple subscripts containing slices
  produced an unhashable surrogate via the object-array `tolist()` branch.
- `solvers/amp.py` `_flat_var_index` + three sites in
  `_jax/problem_classifier.py` — same pattern, same fix shape.

Regression tests: `python/tests/test_tightening_sliced_index.py` (fails before
the fixes, passes after). Suites run: tightening/FBBT files (25 passed), AMP
(150 passed), convexity/classifier (123 passed), `pytest -m smoke` (631 passed,
14 skipped). Not fixed (loud-refusal paths, wrong exception *type* only):
`export/_extract.py:98`, `callbacks.py:212` — follow-up candidates.

## 7. Remaining open questions

1. Reproduce the paper's actual case studies (their DAE has algebraic states;
   ours was ODE-only — `add_algebraic`/`set_algebraic` exist but are untested
   in this hybrid context).
2. AMP vectorized-constraint support (or an automatic scalarization pass for
   the global path) — prerequisite for certified training of DAEBuilder models
   without hand-scalarization.
3. The α×kernel product envelope (see 4 above) — entry experiment: measure root
   bound before/after on the scalarized instance.
4. A "force spatial B&B" escape hatch (or classifier support for
   exp-of-quadratic) so the default path can certify models like these.
5. Multi-experiment API + trainable-surrogate layer (§5) — now unblocked by
   result 5.

References: arXiv:2504.04665 (Lueg et al. 2025); Biegler, *Nonlinear
Programming* (2010) for the simultaneous/collocation background;
`docs/dev/dae-module-review.md`; `docs/design/relaxation-catalog.md`.
