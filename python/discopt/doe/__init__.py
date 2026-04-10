"""discopt.doe -- Model-based Design of Experiments.

Optimal experimental design using Fisher Information Matrix analysis
with exact JAX autodiff for sensitivity computation.

Quick Start
-----------
>>> from discopt.doe import compute_fim, optimal_experiment, DesignCriterion
>>> fim_result = compute_fim(experiment, param_values, design_values)
>>> design = optimal_experiment(experiment, param_values, design_bounds)
>>> print(design.summary())

See Also
--------
discopt.estimate : Parameter estimation using the same Experiment interface.
"""

from discopt.doe.design import (
    DesignCriterion,
    DesignResult,
    optimal_experiment,
)
from discopt.doe.exploration import (
    ExplorationResult,
    explore_design_space,
)
from discopt.doe.fim import (
    FIMResult,
    IdentifiabilityResult,
    check_identifiability,
    compute_fim,
)
from discopt.doe.sequential import (
    DoERound,
    sequential_doe,
)

__all__ = [
    "DesignCriterion",
    "DesignResult",
    "ExplorationResult",
    "FIMResult",
    "IdentifiabilityResult",
    "check_identifiability",
    "compute_fim",
    "explore_design_space",
    "optimal_experiment",
    "DoERound",
    "sequential_doe",
]
