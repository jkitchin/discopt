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

## 6. Open questions / next experiments (with kill criteria)

1. `gauss_newton=True` on experiment 1's objective — does the detector fire on
   a `DAEBuilder.least_squares` sum? Kill: detector misses or GN slows solve.
2. Schur weight-block experiment: same NLP, `kkt_schur_block` from model
   structure; measure factorization time on a 10× larger instance. Kill: no
   speedup on problems ≥ paper scale.
3. Scale probe at paper scale (2×30 softplus, ~1000 weights, multi-trajectory):
   does full-space POUNCE/IPOPT converge without decomposition? This measures
   whether the paper's decomposition pain even reproduces here.
4. Global solve of the GP-α variant on a small box. Kill: node count explodes
   with no bound progress.
5. Matrix-shaped NN emission (weights as `(n_in, n_out)` Variables through
   `MatMulExpression`) — needed before any wide-net work; verify the DAG
   supports Variable @ Variable matmul or add it.

References: arXiv:2504.04665 (Lueg et al. 2025); Biegler, *Nonlinear
Programming* (2010) for the simultaneous/collocation background;
`docs/dev/dae-module-review.md`; `docs/design/relaxation-catalog.md`.
