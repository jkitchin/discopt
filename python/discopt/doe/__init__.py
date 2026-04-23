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
    BatchDesignResult,
    BatchStrategy,
    DesignCriterion,
    DesignResult,
    batch_optimal_experiment,
    optimal_experiment,
)
from discopt.doe.discrimination import (
    DiscriminationCriterion,
    DiscriminationDesignResult,
    discriminate_compound,
    discriminate_design,
)
from discopt.doe.discrimination_sequential import (
    DiscriminationRound,
    sequential_discrimination,
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
from discopt.doe.selection import (
    ModelSelectionResult,
    likelihood_ratio_test,
    model_selection,
    vuong_test,
)
from discopt.doe.sequential import (
    DoERound,
    sequential_doe,
)

__all__ = [
    "BatchDesignResult",
    "BatchStrategy",
    "DesignCriterion",
    "DesignResult",
    "DiscriminationCriterion",
    "DiscriminationDesignResult",
    "DiscriminationRound",
    "DoERound",
    "ExplorationResult",
    "FIMResult",
    "IdentifiabilityResult",
    "ModelSelectionResult",
    "batch_optimal_experiment",
    "check_identifiability",
    "compute_fim",
    "discriminate_compound",
    "discriminate_design",
    "explore_design_space",
    "likelihood_ratio_test",
    "model_selection",
    "optimal_experiment",
    "sequential_discrimination",
    "sequential_doe",
    "vuong_test",
]
