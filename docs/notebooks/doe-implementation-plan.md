# Implementation Plan: `discopt.doe` and `discopt.estimate`

Model-based parameter estimation and optimal design of experiments for discopt,
leveraging JAX autodiff for exact Fisher Information Matrix computation.

**Reference paper**: Wang & Dowling, "Pyomo.DOE: An open-source package for
model-based design of experiments in Python", *AIChE Journal*, 2022.
DOI: 10.1002/aic.17813

---

## Phase 0: Prerequisites and Infrastructure (1 week)

Ensure the existing `differentiable_solve` machinery is robust enough to
support the FIM computation pipeline. No new public API yet.

### 0.1 Expose `SolveResult.gradient()` for continuous models

The stub at `modeling/core.py:972` currently raises `NotImplementedError`.
Wire it to call `differentiable_solve` internally when the model is purely
continuous and has parameters.

**Files to modify:**
- `python/discopt/modeling/core.py` — `SolveResult.gradient()` implementation
- `python/discopt/_jax/differentiable.py` — ensure `DiffSolveResult` stores
  primal solution in a format compatible with `SolveResult`

**Tests (add to `python/tests/test_differentiable.py`):**

```python
class TestSolveResultGradient:
    def test_gradient_matches_envelope_theorem(self):
        """SolveResult.gradient(param) == DiffSolveResult.gradient(param)."""
        m = dm.Model()
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize((x - p) ** 2)
        result = m.solve()
        # d(obj*)/dp = 0 at optimum (x* = p), verified by envelope theorem
        assert result.gradient(p) == pytest.approx(0.0, abs=1e-6)

    def test_gradient_linear_objective(self):
        """min p*x s.t. x >= 1 => x*=1, obj*=p, d(obj*)/dp = 1."""
        m = dm.Model()
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=10)
        m.minimize(p * x)
        result = m.solve()
        assert result.gradient(p) == pytest.approx(1.0, abs=1e-6)

    def test_gradient_raises_for_integer_models(self):
        m = dm.Model()
        m.parameter("p", value=1.0)
        m.binary("y")
        m.minimize(Constant(0.0))
        result = m.solve()
        with pytest.raises(ValueError, match="continuous"):
            result.gradient(m._parameters[0])
```

### 0.2 Parametric response evaluation

Add a helper that compiles arbitrary model expressions (not just objective/
constraints) into JAX-differentiable functions of `(x_flat, p_flat)`.
This is needed to compute dy/dθ for responses that aren't the objective.

**File to create:** `python/discopt/_jax/parametric.py`

```python
def compile_response_function(
    expressions: dict[str, Expression],
    model: Model,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Compile named expressions into f(x_flat, p_flat) -> response_vector."""
    ...
```

This is a thin wrapper around `_compile_parametric_node` from
`differentiable.py`, but returns a stacked vector of responses instead of
a single scalar.

**Tests (`python/tests/test_parametric.py`):**

```python
class TestParametricResponse:
    def test_single_response(self):
        """Compiled response matches direct evaluation."""
        m = dm.Model()
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        expr = p * x + 1
        fn = compile_response_function({"y": expr}, m)
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([2.0])
        assert float(fn(x_flat, p_flat)[0]) == pytest.approx(7.0)

    def test_jacobian_wrt_params(self):
        """jax.jacobian of responses w.r.t. parameters is exact."""
        m = dm.Model()
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        y1 = p * x        # dy1/dp = x = 3
        y2 = p ** 2 + x   # dy2/dp = 2p = 4
        fn = compile_response_function({"y1": y1, "y2": y2}, m)
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([2.0])
        J = jax.jacobian(fn, argnums=1)(x_flat, p_flat)
        np.testing.assert_allclose(J[:, 0], [3.0, 4.0], atol=1e-12)

    def test_multiple_params_jacobian_shape(self):
        """Jacobian shape is (n_responses, n_params)."""
        m = dm.Model()
        a = m.parameter("a", value=1.0)
        b = m.parameter("b", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        fn = compile_response_function({"y1": a * x, "y2": b * x, "y3": a + b}, m)
        x_flat = jnp.array([5.0])
        p_flat = jnp.array([1.0, 2.0])
        J = jax.jacobian(fn, argnums=1)(x_flat, p_flat)
        assert J.shape == (3, 2)
```

---

## Phase 1: Parameter Estimation (`discopt.estimate`) (2 weeks)

### 1.1 Core estimation API

**File to create:** `python/discopt/estimate.py`

Public API:

```python
class Experiment:
    """Base class. Subclass and implement create_model()."""
    def create_model(self, **kwargs) -> ExperimentModel: ...

class ExperimentModel:
    """Model + metadata: parameters, design_inputs, responses, measurement_error."""
    model: dm.Model
    parameters: dict[str, dm.Parameter]
    design_inputs: dict[str, dm.Variable]
    responses: dict[str, dm.Expression]
    measurement_error: dict[str, float]

def estimate_parameters(
    experiment: Experiment,
    data: dict[str, np.ndarray],
    *,
    initial_guess: dict[str, float] | None = None,
    method: str = "least_squares",
    solver: str = "ipopt",
    solver_options: dict | None = None,
) -> EstimationResult: ...

class EstimationResult:
    parameters: dict[str, float]
    covariance: np.ndarray
    fim: np.ndarray
    objective: float
    confidence_intervals: dict[str, tuple[float, float]]  # property
    correlation_matrix: np.ndarray                          # property
    def summary(self) -> str: ...
```

**Implementation strategy:**

The key challenge is that `Parameter` objects are fixed during solve, but
estimation needs to optimize over them. Two options:

**(A) Rebuild the model with parameters as variables.** `estimate_parameters`
calls `experiment.create_model()`, then creates a *new* model where each
`Parameter` is replaced by a `Variable` with bounds. The expression DAG must
be walked and reconstructed. This is clean but requires a DAG copy utility.

**(B) The user defines estimation targets as variables directly.** The
`Experiment.create_model()` always uses `Variable` for unknowns, and
`Parameter` for fixed nominal values used in DoE. The estimation function
solves the model as-is (variables are already optimization targets).

**Recommended: Option B.** Simpler, no DAG rewriting. The `ExperimentModel`
has separate fields:
- `responses` — expressions involving both variables and parameters
- `unknown_parameters` — *variables* that represent the unknowns
- `design_inputs` — *variables* for experimental conditions
- `fixed_parameters` — `Parameter` objects for nominal values used in DoE

The estimation function sets the objective to weighted least squares over
the `unknown_parameters` (which are already variables).

For DoE, the `fixed_parameters` are used for sensitivity analysis via
the parametric compiler.

**Revised ExperimentModel:**

```python
class ExperimentModel:
    model: dm.Model
    unknown_parameters: dict[str, dm.Variable]   # estimated as variables
    design_inputs: dict[str, dm.Variable]         # controlled by experimenter
    responses: dict[str, dm.Expression]           # model predictions
    measurement_error: dict[str, float]           # σ for each response
    nominal_values: dict[str, float] | None       # θ_nominal for FIM
```

### 1.2 Tests (`python/tests/test_estimate.py`)

```python
class TestExperimentModel:
    def test_create_experiment_model(self):
        """ExperimentModel stores all metadata correctly."""

    def test_response_names_match_error_names(self):
        """Raise ValueError if measurement_error keys != response keys."""

    def test_unknown_parameters_are_variables(self):
        """unknown_parameters must be dm.Variable instances."""


class TestEstimateParameters:
    def test_linear_regression(self):
        """y = a*x + b with known data recovers true a, b."""
        # Ground truth: a=2.0, b=1.0
        # Data: x=[1,2,3,4,5], y=[3.0, 5.0, 7.0, 9.0, 11.0]
        # Should recover a≈2.0, b≈1.0 within tolerance

    def test_exponential_decay(self):
        """y = A*exp(-k*t) recovers A, k from synthetic data."""
        # Ground truth: A=5.0, k=0.3
        # Generate data with small noise, verify recovery

    def test_multiresponse_estimation(self):
        """Estimate from multiple simultaneous responses."""
        # e.g., two measured outputs from one model

    def test_weighted_residuals(self):
        """Heteroscedastic errors: different σ per measurement."""
        # Points with smaller σ should have more influence

    def test_bounds_respected(self):
        """Estimated parameters stay within specified bounds."""

    def test_infeasible_data_returns_status(self):
        """If data is incompatible with model, status reflects it."""


class TestEstimationResult:
    def test_confidence_intervals_contain_true_value(self):
        """95% CI from covariance contains the true parameter value."""
        # Use noise-free data so true params are at optimum

    def test_correlation_matrix_diagonal_ones(self):
        """Diagonal of correlation matrix is 1.0."""

    def test_covariance_positive_semidefinite(self):
        """Covariance matrix has non-negative eigenvalues."""

    def test_summary_string_format(self):
        """summary() returns a readable string with all fields."""
```

### 1.3 Analytic verification tests

These use problems with known closed-form solutions.

```python
class TestAnalyticVerification:
    def test_linear_model_fim_equals_XtX(self):
        """For y = Xθ + ε, FIM = X^T Σ^{-1} X (textbook result)."""
        # Build linear regression as discopt model
        # Compare computed FIM to numpy X.T @ diag(1/σ²) @ X

    def test_single_param_variance(self):
        """Single-parameter model: Var(θ) = σ² / Σ(dy/dθ)²."""
        # y = θ*x, data at x=[1,2,3]
        # FIM = sum(x_i²) / σ², Var(θ) = σ² / sum(x_i²)

    def test_fim_symmetry(self):
        """FIM is always symmetric."""

    def test_fim_positive_semidefinite(self):
        """FIM eigenvalues are non-negative."""

    def test_prior_fim_additive(self):
        """FIM(combined) = FIM(experiment 1) + FIM(experiment 2)."""
        # Compute FIM for two datasets separately and together
```

---

## Phase 2: Fisher Information Matrix (`discopt.doe.fim`) (1.5 weeks)

### 2.1 FIM computation via JAX autodiff

**File to create:** `python/discopt/doe/__init__.py`, `python/discopt/doe/fim.py`

```python
def compute_fim(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float],
    *,
    prior_fim: np.ndarray | None = None,
    method: str = "autodiff",         # or "finite_difference" (fallback)
    fd_step: float = 1e-5,            # only used if method="finite_difference"
) -> FIMResult: ...

class FIMResult:
    fim: np.ndarray                      # n_params × n_params
    jacobian: np.ndarray                 # n_responses × n_params
    metrics: dict[str, float]            # log_det, trace_inv, min_eig, cond
    def d_optimal(self) -> float: ...    # log det(FIM)
    def a_optimal(self) -> float: ...    # trace(FIM^{-1})
    def e_optimal(self) -> float: ...    # min eigenvalue
    def me_optimal(self) -> float: ...   # condition number
```

**Implementation detail — autodiff vs finite-difference FIM:**

The autodiff method works when responses are *explicit functions* of
parameters at a fixed design point (no implicit solve needed). This covers:
- Algebraic models: y = f(θ, d)
- DAE models after discretization: responses are algebraic expressions of θ

For models where responses are defined *implicitly* by an optimization
(y = argmin problem), the Jacobian requires implicit differentiation through
the solve. This uses the existing `differentiable_solve` infrastructure.

The finite-difference fallback is provided for comparison and validation.

### 2.2 Tests (`python/tests/test_fim.py`)

```python
class TestFIMComputation:
    def test_linear_model_fim_matches_analytic(self):
        """y = [x1, x2] @ [θ1, θ2] => FIM = X^T Σ^{-1} X."""

    def test_nonlinear_model_fim_matches_finite_difference(self):
        """Autodiff FIM matches central-difference FIM within tolerance."""
        # y = θ1 * exp(-θ2 * t)
        # Compare autodiff J^T Σ^{-1} J to finite-diff version

    def test_fim_with_prior(self):
        """compute_fim with prior_fim adds to base FIM."""

    def test_fim_determinant_positive(self):
        """D-optimal metric is defined (det > 0) for identifiable model."""

    def test_unidentifiable_model_singular_fim(self):
        """Model with redundant parameters has singular FIM (det ≈ 0)."""
        # y = (θ1 * θ2) * x  =>  θ1 and θ2 not separately identifiable

    def test_fim_scales_with_measurements(self):
        """More measurements => larger FIM (more information)."""
        # Same model with 5 vs 10 data points

    def test_fim_scales_with_precision(self):
        """Smaller σ => larger FIM (more precise measurements)."""


class TestFIMMetrics:
    def test_d_optimal_equals_log_det(self):
        """d_optimal() == log(det(FIM))."""

    def test_a_optimal_equals_trace_inv(self):
        """a_optimal() == trace(inv(FIM))."""

    def test_e_optimal_equals_min_eigenvalue(self):
        """e_optimal() == min(eigenvalues(FIM))."""

    def test_me_optimal_equals_condition_number(self):
        """me_optimal() == max_eig / min_eig."""

    def test_metrics_dict_keys(self):
        """All four metrics present in FIMResult.metrics."""


class TestFIMAutodiffVsFiniteDifference:
    """Cross-validate autodiff FIM against finite-difference FIM."""

    @pytest.mark.parametrize("step", [1e-4, 1e-5, 1e-6, 1e-7])
    def test_exponential_model(self, step):
        """y = A*exp(-k*t): autodiff and FD agree."""

    def test_multiresponse_model(self):
        """Multiple outputs: Jacobian columns match FD perturbations."""

    def test_dae_model(self):
        """DAE-discretized model: autodiff FIM matches FD FIM."""
```

### 2.3 Identifiability analysis

Add a convenience function that checks whether parameters are structurally
identifiable from the proposed measurements, using the FIM rank.

```python
def check_identifiability(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float],
) -> IdentifiabilityResult: ...

class IdentifiabilityResult:
    is_identifiable: bool
    fim_rank: int
    n_parameters: int
    problematic_parameters: list[str]  # near-zero sensitivity directions
    condition_number: float
```

**Tests:**

```python
class TestIdentifiability:
    def test_identifiable_model(self):
        """Well-posed model is identifiable."""

    def test_unidentifiable_product(self):
        """y = (a*b)*x: a,b not individually identifiable."""
        result = check_identifiability(...)
        assert not result.is_identifiable
        assert result.fim_rank == 1  # only 1 of 2 params identifiable

    def test_partially_identifiable(self):
        """3 params, 2 identifiable: rank = 2, problematic list has 1."""
```

---

## Phase 3: Optimal Experimental Design (`discopt.doe`) (2 weeks)

### 3.1 Design optimization

**File to create:** `python/discopt/doe/design.py`

```python
class DesignCriterion:
    D_OPTIMAL = "determinant"
    A_OPTIMAL = "trace"
    E_OPTIMAL = "min_eigenvalue"
    ME_OPTIMAL = "condition_number"

def optimal_experiment(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    criterion: str = DesignCriterion.D_OPTIMAL,
    prior_fim: np.ndarray | None = None,
    solver: str = "ipopt",
    solver_options: dict | None = None,
) -> DesignResult: ...

class DesignResult:
    design: dict[str, float]
    fim: np.ndarray
    criterion_value: float
    metrics: dict[str, float]
    parameter_covariance: np.ndarray     # property
    predicted_standard_errors: np.ndarray # property
    def summary(self) -> str: ...
```

**Implementation strategy:**

The design optimization is itself an NLP. Two approaches:

**(A) Direct approach (Phase 3a):** Build a discopt Model where design inputs
are variables, the objective is the FIM criterion, and model constraints
ensure feasibility. The FIM is computed inside the objective via JAX ops.
This requires the objective to call `jax.jacobian` internally, which means
the entire objective is a JAX-traced function. This works because discopt's
DAG compiles to JAX.

**(B) Bilevel approach (Phase 3b, advanced):** For models where responses
are defined by an inner optimization (e.g., equilibrium models), use
`differentiable_solve` to get dy/dθ via implicit differentiation, then
optimize the outer DoE problem. This composes `differentiable_solve` inside
the FIM computation.

Phase 3a handles the common case (algebraic + discretized DAE models).
Phase 3b extends to implicit models later.

### 3.2 Design space exploration

**File:** `python/discopt/doe/exploration.py`

```python
def explore_design_space(
    experiment: Experiment,
    param_values: dict[str, float],
    design_ranges: dict[str, np.ndarray],
    *,
    prior_fim: np.ndarray | None = None,
) -> ExplorationResult: ...

class ExplorationResult:
    grid: dict[str, np.ndarray]
    metrics: dict[str, np.ndarray]   # criterion values at each grid point
    def plot_heatmap(self, criterion, ax=None): ...
    def plot_sensitivity(self, criterion, ax=None): ...
    def best_point(self, criterion) -> dict[str, float]: ...
```

**GPU acceleration:** Use `jax.vmap` to vectorize FIM computation over the
design grid. For a 20×20 grid, this evaluates 400 FIMs in one vectorized
pass instead of 400 sequential solves.

### 3.3 Tests (`python/tests/test_doe.py`)

```python
class TestOptimalExperiment:
    def test_d_optimal_maximizes_determinant(self):
        """Optimal design has higher det(FIM) than random designs."""
        # Compare optimal design to 100 random designs in bounds

    def test_a_optimal_minimizes_trace(self):
        """A-optimal design has lower trace(FIM^{-1}) than random."""

    def test_design_within_bounds(self):
        """Optimal design respects all bounds."""

    def test_single_design_variable(self):
        """1D design: optimal temperature for Arrhenius model."""
        # y = A*exp(-Ea/(R*T)), design variable is T
        # With 2 params (A, Ea), optimal T should be near boundary
        # or at a specific interior point (verifiable analytically)

    def test_prior_fim_shifts_design(self):
        """Adding prior information changes the optimal design."""
        # Prior with strong info about param 1 should push design
        # toward better estimation of param 2

    def test_criterion_options_all_solve(self):
        """All four criteria (D, A, E, ME) produce feasible results."""


class TestExploreDesignSpace:
    def test_1d_exploration(self):
        """Single design variable sweep returns correct grid shape."""

    def test_2d_exploration(self):
        """Two design variables produce 2D grid of metrics."""

    def test_best_point_matches_optimization(self):
        """best_point from grid ≈ optimal_experiment result (coarse)."""
        # Grid optimum should be near NLP optimum (within grid spacing)

    def test_vmap_matches_sequential(self):
        """Vectorized grid evaluation matches sequential loop."""
        # Compute FIM at 10 points via vmap and via for-loop, compare


class TestDesignAnalytic:
    """Problems with known analytic D-optimal designs."""

    def test_linear_2param_optimal_at_endpoints(self):
        """y = θ1 + θ2*x, x in [0,1]: D-optimal puts mass at 0 and 1."""
        # Classic result from optimal design theory

    def test_exponential_single_param(self):
        """y = exp(-θ*t), t in [0,T]: D-optimal at t = 1/θ."""
```

---

## Phase 4: Sequential DoE and Integration (1 week)

### 4.1 Sequential DoE loop

**File:** `python/discopt/doe/sequential.py`

```python
def sequential_doe(
    experiment: Experiment,
    initial_data: dict[str, np.ndarray],
    initial_guess: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    n_rounds: int = 5,
    criterion: str = DesignCriterion.D_OPTIMAL,
    run_experiment: Callable | None = None,
    callback: Callable[[DoERound], None] | None = None,
) -> list[DoERound]: ...

class DoERound:
    round: int
    estimation: EstimationResult
    design: DesignResult
    data_collected: dict[str, np.ndarray] | None
```

### 4.2 Public API exports

**File to modify:** `python/discopt/__init__.py`

Add lazy imports:

```python
# In __init__.py
def estimate_parameters(*args, **kwargs):
    from discopt.estimate import estimate_parameters as _ep
    return _ep(*args, **kwargs)
```

**File to create:** `python/discopt/doe/__init__.py`

```python
from discopt.doe.design import DesignCriterion, DesignResult, optimal_experiment
from discopt.doe.exploration import ExplorationResult, explore_design_space
from discopt.doe.fim import FIMResult, check_identifiability, compute_fim
from discopt.doe.sequential import DoERound, sequential_doe
```

### 4.3 Tests (`python/tests/test_sequential_doe.py`)

```python
class TestSequentialDoE:
    def test_sequential_improves_confidence(self):
        """Each round of DoE narrows confidence intervals."""
        # Use synthetic experiment runner
        # Verify CI width decreases monotonically

    def test_fim_accumulates(self):
        """Prior FIM grows with each round."""
        # det(FIM) should increase each round

    def test_no_runner_returns_recommendation(self):
        """Without run_experiment, returns after first recommendation."""

    def test_callback_called_each_round(self):
        """callback receives DoERound after each iteration."""

    def test_data_merging(self):
        """New data from each round is appended correctly."""
```

---

## Phase 5: Documentation and Examples (1.5 weeks)

### 5.1 Tutorial notebook: Parameter Estimation

**File:** `docs/notebooks/tutorial_estimation.ipynb`

Structure:
1. **Introduction** — What is parameter estimation? Weighted least squares
   and maximum likelihood. Cite {cite:p}`Bard1974` or {cite:p}`Biegler2010`.
2. **Simple example** — Fit y = A·exp(−k·t) to synthetic data
   - Show model building with discopt
   - Run `estimate_parameters()`
   - Inspect `EstimationResult`: parameters, covariance, CI
   - Plot fit vs data with confidence bands
3. **Multi-response example** — Two measured outputs
4. **With DAE model** — Dynamic experiment using `discopt.dae`
   - Batch reactor A→B, estimate rate constant
   - Collocation discretization, then estimation
5. **Comparison** — Side-by-side with scipy.optimize.curve_fit for the
   simple case (same answer, but discopt handles constraints and DAE)

**BibTeX entries to add to `docs/references.bib`:**
```bibtex
@book{Bard1974,
  author = {Bard, Yonathan},
  title = {Nonlinear Parameter Estimation},
  publisher = {Academic Press},
  year = {1974},
}

@article{Wang2022,
  author = {Wang, Jialu and Dowling, Alexander W.},
  title = {Pyomo.DOE: An open-source package for model-based
           design of experiments in Python},
  journal = {AIChE Journal},
  volume = {68},
  number = {12},
  pages = {e17813},
  year = {2022},
  doi = {10.1002/aic.17813},
}

@book{Atkinson2007,
  author = {Atkinson, Anthony and Donev, Alexander and Tobias, Randall},
  title = {Optimum Experimental Designs, with SAS},
  publisher = {Oxford University Press},
  year = {2007},
}

@article{Franceschini2008,
  author = {Franceschini, Gaia and Macchietto, Sandro},
  title = {Model-based design of experiments for parameter precision:
           State of the art},
  journal = {Chemical Engineering Science},
  volume = {63},
  number = {19},
  pages = {4846--4872},
  year = {2008},
  doi = {10.1016/j.ces.2007.11.034},
}
```

### 5.2 Tutorial notebook: Design of Experiments

**File:** `docs/notebooks/tutorial_doe.ipynb`

Structure:
1. **Introduction** — Model-based DoE, Fisher Information Matrix, design
   criteria. Cite {cite:p}`Wang2022`, {cite:p}`Franceschini2008`,
   {cite:p}`Atkinson2007`.
2. **FIM computation** — Build a model, compute FIM, interpret metrics
   - Show D-optimal, A-optimal, E-optimal, ME-optimal
   - Discuss identifiability
3. **Design space exploration** — Heatmap of D-optimality over 2D grid
   - `explore_design_space()` with plots
4. **Optimal experimental design** — Find the best experiment
   - `optimal_experiment()` with D-optimality
   - Compare to grid-search result
5. **Sequential DoE** — Full estimate→design loop with synthetic data
   - Show confidence intervals narrowing over rounds
   - Plot parameter convergence
6. **Advantage: JAX autodiff vs finite differences**
   - Accuracy comparison (autodiff vs FD at various step sizes)
   - Timing comparison
7. **Advanced: mixed-integer design** (optional)
   - Choose which of N candidate measurements to take (binary decisions)

### 5.3 API docstrings

All public classes and functions get NumPy-style docstrings with:
- Parameters / Returns / Raises sections
- Examples section with `>>>` doctests
- Mathematical description where relevant (FIM formula, criteria)
- Cross-references to related functions

### 5.4 Add to `docs/_toc.yml`

```yaml
  - caption: Applications
    chapters:
      - file: notebooks/nn_embedding
      - file: notebooks/decision_focused_learning
      - file: notebooks/tutorial_estimation     # NEW
      - file: notebooks/tutorial_doe            # NEW
```

### 5.5 Docstring examples as tests

Run doctests with pytest:

```python
# In conftest.py or pytest config
collect_ignore_glob = []
doctest_optionflags = ["NORMALIZE_WHITESPACE", "ELLIPSIS"]
```

---

## Phase 6: Advanced Features (2 weeks, optional)

### 6.1 Mixed-integer experimental design

Leverage discopt's MINLP capability for designs with discrete decisions:

- **Measurement selection**: Binary variables for which sensors to install
- **Number of replicates**: Integer variables for how many times to repeat
- **Factorial designs**: Choose from a discrete set of factor levels

```python
def optimal_experiment_milp(
    experiment: Experiment,
    param_values: dict[str, float],
    candidate_designs: list[dict[str, float]],
    n_select: int,
    *,
    criterion: str = DesignCriterion.D_OPTIMAL,
) -> MILPDesignResult: ...
```

This is a unique capability that Pyomo.DoE does not offer.

### 6.2 Robust DoE (parameter uncertainty)

Account for uncertainty in nominal parameter values when designing:

```python
def robust_optimal_experiment(
    experiment: Experiment,
    param_distribution: dict[str, tuple[float, float]],  # (mean, std)
    design_bounds: dict[str, tuple[float, float]],
    *,
    n_scenarios: int = 50,
    criterion: str = DesignCriterion.D_OPTIMAL,
) -> DesignResult: ...
```

Average FIM over parameter scenarios drawn from the uncertainty distribution.

### 6.3 Online DoE with real instrument integration

Callback interface for connecting to lab instruments:

```python
class InstrumentInterface:
    """Abstract base for lab instrument communication."""
    def set_conditions(self, design: dict[str, float]) -> None: ...
    def collect_data(self) -> dict[str, np.ndarray]: ...
    def is_ready(self) -> bool: ...
```

### 6.4 LLM integration

Add `/doe` and `/estimate` Claude Code skills:

- `/doe` — Given a model description, recommend experimental design
- `/estimate` — Given data and model, run estimation and explain results

**Files:** `.claude/commands/doe.md`, `.claude/commands/estimate.md`

---

## Test Correctness Matrix

Each test category and what mathematical property it validates:

| Test | Property | Oracle |
|------|----------|--------|
| Linear model FIM = X^T Σ^{-1} X | FIM definition | Analytic |
| Autodiff J matches finite-diff J | Jacobian accuracy | Cross-validation |
| FIM symmetric | FIM = J^T Σ^{-1} J structure | Matrix property |
| FIM positive semi-definite | Information is non-negative | Eigenvalue check |
| Prior FIM additive | Independent experiments | FIM₁ + FIM₂ = FIM_{1+2} |
| CI contains true value | Covariance correctness | Statistical coverage |
| D-optimal > random designs | Optimality | Comparison |
| Unidentifiable ⇒ singular FIM | Rank deficiency | det(FIM) ≈ 0 |
| Estimation recovers true params | Solver correctness | Known ground truth |
| Sequential DoE improves CI | Information accumulation | Monotone decrease |

---

## File Layout Summary

```
python/discopt/
├── estimate.py                    # Phase 1: parameter estimation
├── doe/
│   ├── __init__.py                # Public re-exports
│   ├── fim.py                     # Phase 2: FIM computation
│   ├── design.py                  # Phase 3: optimal design
│   ├── exploration.py             # Phase 3: design space sweep
│   └── sequential.py              # Phase 4: sequential DoE loop
├── _jax/
│   ├── parametric.py              # Phase 0: response compiler
│   └── differentiable.py          # Phase 0: existing, extend

python/tests/
├── test_parametric.py             # Phase 0 tests
├── test_estimate.py               # Phase 1 tests
├── test_fim.py                    # Phase 2 tests
├── test_doe.py                    # Phase 3 tests
├── test_sequential_doe.py         # Phase 4 tests

docs/
├── notebooks/
│   ├── tutorial_estimation.ipynb  # Phase 5
│   └── tutorial_doe.ipynb         # Phase 5
├── references.bib                 # Add new entries
└── _toc.yml                       # Add new notebooks
```

---

## Dependencies

No new external dependencies required. Everything builds on:
- `jax` (existing) — autodiff, vmap
- `numpy` (existing) — arrays
- `scipy` (existing) — eigenvalues, statistics for CI
- `matplotlib` (existing, optional) — plotting in exploration
- `cyipopt` (existing, optional) — NLP solver backend

---

## Milestones and Review Points

| Milestone | Deliverable | Gate |
|-----------|-------------|------|
| Phase 0 done | `SolveResult.gradient()` works, parametric response compiler | All `test_differentiable.py` pass |
| Phase 1 done | `estimate_parameters()` recovers known params | Analytic tests pass, CI coverage correct |
| Phase 2 done | `compute_fim()` matches analytic and FD | Autodiff ≈ FD within 1e-6, metrics correct |
| Phase 3 done | `optimal_experiment()` beats random designs | D-optimal > 95th percentile of random |
| Phase 4 done | Full sequential loop, public API | Integration test: 5 rounds, CI shrinks |
| Phase 5 done | Notebooks build, citations resolve | `jupyter-book build docs/` zero warnings |
