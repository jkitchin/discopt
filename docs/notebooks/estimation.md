# Parameter Estimation

discopt provides model-based parameter estimation (`discopt.estimate`) using
exact JAX autodiff for Fisher Information Matrix computation. Optimal design
of experiments on the same `Experiment` interface lives in the standalone
[discopt-doe](https://github.com/jkitchin/discopt-doe) plugin
(`pip install discopt-doe`).

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

## Design of Experiments

Optimal experimental design — FIM computation, D/A/E-optimal design, design
space exploration, sequential DoE, and identifiability analysis — moved to the
[discopt-doe](https://github.com/jkitchin/discopt-doe) plugin (#389). It uses
the same `Experiment` subclass you write for estimation:

```bash
pip install discopt-doe
```

See the [discopt-doe documentation](https://github.com/jkitchin/discopt-doe)
for the design workflow.

## API Reference

The full API is auto-generated from docstrings. Key entry points:

**`discopt.estimate`**
- {class}`~discopt.estimate.Experiment` — base class for experiment definitions
- {class}`~discopt.estimate.ExperimentModel` — annotated model with metadata
- {func}`~discopt.estimate.estimate_parameters` — run parameter estimation
- {class}`~discopt.estimate.EstimationResult` — results with CI, FIM, covariance
