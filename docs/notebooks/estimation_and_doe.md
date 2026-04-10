# Parameter Estimation & Design of Experiments

discopt provides integrated model-based parameter estimation (`discopt.estimate`)
and optimal design of experiments (`discopt.doe`) using exact JAX autodiff for
Fisher Information Matrix computation.

## Concepts

### The Experiment Interface

Both estimation and DoE share a common `Experiment` base class. You subclass it
and implement `create_model()`, which returns an `ExperimentModel` with labeled
components:

| Component | Type | Purpose |
|-----------|------|---------|
| `unknown_parameters` | `dict[str, Variable]` | Parameters to estimate (optimized as variables) |
| `design_inputs` | `dict[str, Variable]` | Experimental conditions the experimenter controls |
| `responses` | `dict[str, Expression]` | Model predictions at measurement points |
| `measurement_error` | `dict[str, float]` | Standard deviation $\sigma_i$ for each response |

```python
from discopt.estimate import Experiment, ExperimentModel
import discopt.modeling as dm

class MyExperiment(Experiment):
    def create_model(self, **kwargs):
        m = dm.Model("my_exp")
        k = m.continuous("k", lb=0, ub=10)       # unknown parameter
        T = m.continuous("T", lb=300, ub=400)     # design input
        y_pred = k * dm.exp(-1000 / T)            # model prediction
        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={"T": T},
            responses={"y": y_pred},
            measurement_error={"y": 0.05},
        )
```

### Parameter Estimation

`estimate_parameters()` builds a weighted least-squares NLP:

$$\min_\theta \sum_i \left(\frac{y_i^{\text{obs}} - y_i^{\text{model}}(\theta)}{\sigma_i}\right)^2$$

and solves it with discopt's NLP solvers (Ipopt or pure-JAX IPM). The result
includes estimated parameter values, the Fisher Information Matrix, parameter
covariance ($\text{Cov}(\theta) \approx \text{FIM}^{-1}$), and 95% confidence
intervals.

```python
from discopt.estimate import estimate_parameters

result = estimate_parameters(experiment, data, initial_guess={"k": 1.0})
print(result.parameters)            # {"k": 2.34}
print(result.confidence_intervals)   # {"k": (2.12, 2.56)}
print(result.correlation_matrix)     # parameter correlations
```

### Fisher Information Matrix

The FIM quantifies how much information an experiment provides about unknown
parameters {cite:p}`Franceschini2008`:

$$\text{FIM} = J^T \Sigma^{-1} J$$

where $J_{ij} = \partial y_i / \partial \theta_j$ is the sensitivity Jacobian.
discopt computes $J$ via **exact JAX autodiff** — no finite differences, no
step-size tuning, no extra model solves {cite:p}`Wang2022`.

```python
from discopt.doe import compute_fim

fim_result = compute_fim(experiment, {"k": 2.0}, {"T": 350.0})
print(fim_result.d_optimal)     # log det(FIM) — D-optimality
print(fim_result.a_optimal)     # trace(FIM^{-1}) — A-optimality
print(fim_result.e_optimal)     # min eigenvalue — E-optimality
```

### Design Criteria {cite:p}`Atkinson2007`

| Criterion | Formula | Interpretation |
|-----------|---------|----------------|
| **D-optimal** | $\max \log \det(\text{FIM})$ | Minimize volume of confidence ellipsoid |
| **A-optimal** | $\min \text{tr}(\text{FIM}^{-1})$ | Minimize average parameter variance |
| **E-optimal** | $\max \lambda_{\min}(\text{FIM})$ | Minimize worst-case variance |
| **ME-optimal** | $\min \kappa(\text{FIM})$ | Balance information across parameters |

### Optimal Experimental Design

`optimal_experiment()` finds design conditions that maximize the chosen
information criterion:

```python
from discopt.doe import optimal_experiment, DesignCriterion

design = optimal_experiment(
    experiment,
    param_values={"k": 2.0},
    design_bounds={"T": (300, 400)},
    criterion=DesignCriterion.D_OPTIMAL,
)
print(design.design)                    # {"T": 387.2}
print(design.predicted_standard_errors) # predicted SE if experiment is run
```

### Design Space Exploration

Visualize how FIM metrics vary across the design space:

```python
from discopt.doe import explore_design_space
import numpy as np

result = explore_design_space(
    experiment,
    param_values={"k": 2.0},
    design_ranges={"T": np.linspace(300, 400, 20)},
)
result.plot_sensitivity(criterion="log_det_fim")
```

### Sequential DoE

The most powerful workflow alternates estimation and design:

1. **Estimate** parameters from all collected data
2. **Compute FIM** at current estimates
3. **Design** the next experiment to maximize information gain
4. **Run** the experiment and collect new data
5. Repeat — confidence intervals shrink each round

```python
from discopt.doe import sequential_doe

history = sequential_doe(
    experiment=exp,
    initial_data=data,
    initial_guess={"k": 1.0},
    design_bounds={"T": (300, 400)},
    n_rounds=5,
    run_experiment=my_lab_runner,  # callable that runs real experiments
)
for r in history:
    print(f"Round {r.round}: k={r.estimation.parameters['k']:.3f}")
```

### Identifiability Analysis

Before running experiments, check that parameters are structurally identifiable
from the proposed measurements:

```python
from discopt.doe import check_identifiability

result = check_identifiability(experiment, {"k": 2.0, "A": 5.0})
print(result.is_identifiable)          # True/False
print(result.fim_rank)                 # rank of FIM
print(result.problematic_parameters)   # unidentifiable params
```

## API Reference

The full API is auto-generated from docstrings. Key entry points:

**`discopt.estimate`**
- {class}`~discopt.estimate.Experiment` — base class for experiment definitions
- {class}`~discopt.estimate.ExperimentModel` — annotated model with metadata
- {func}`~discopt.estimate.estimate_parameters` — run parameter estimation
- {class}`~discopt.estimate.EstimationResult` — results with CI, FIM, covariance

**`discopt.doe`**
- {func}`~discopt.doe.compute_fim` — Fisher Information Matrix via JAX autodiff
- {func}`~discopt.doe.optimal_experiment` — find best experimental conditions
- {func}`~discopt.doe.explore_design_space` — grid evaluation with plotting
- {func}`~discopt.doe.sequential_doe` — full estimate-design loop
- {func}`~discopt.doe.check_identifiability` — parameter identifiability check
- {class}`~discopt.doe.DesignCriterion` — D/A/E/ME optimality constants
- {class}`~discopt.doe.FIMResult` — FIM with all optimality metrics

## Comparison with Pyomo.DoE

| Feature | Pyomo.DoE {cite:p}`Wang2022` | discopt.doe |
|---------|------------|-------------|
| Sensitivity Jacobian | Finite differences (2N extra solves) | Exact JAX autodiff |
| FIM gradient w.r.t. design | Cholesky variables in NLP | JAX autodiff through FIM |
| GPU acceleration | No | Yes (JAX backend) |
| Parameter estimation | Separate package (parmest) | Integrated |
| Mixed-integer designs | Not supported | Native MINLP |
| Implicit differentiation | Not available | Envelope theorem |

## References

```{bibliography}
:filter: docname in docnames
```
