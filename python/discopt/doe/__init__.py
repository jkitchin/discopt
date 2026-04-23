"""discopt.doe -- Model-based Design of Experiments.

Optimal experimental design using Fisher Information Matrix analysis
with exact JAX autodiff for sensitivity computation.

Quick Start
-----------
>>> from discopt.doe import compute_fim, optimal_experiment, DesignCriterion
>>> fim_result = compute_fim(experiment, param_values, design_values)
>>> design = optimal_experiment(experiment, param_values, design_bounds)
>>> print(design.summary())

Identifiability and estimability diagnostics
--------------------------------------------
>>> from discopt.doe import diagnose_identifiability, estimability_rank, profile_likelihood
>>> diag = diagnose_identifiability(experiment, param_values)
>>> est = estimability_rank(experiment, param_values)
>>> profile = profile_likelihood(experiment, data, "k")

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
from discopt.doe.estimability import (
    EstimabilityResult,
    collinearity_index,
    d_optimal_subset,
    estimability_rank,
)
from discopt.doe.exploration import (
    ExplorationResult,
    explore_design_space,
)
from discopt.doe.fim import (
    FIMResult,
    IdentifiabilityDiagnostics,
    IdentifiabilityResult,
    check_identifiability,
    compute_fim,
    diagnose_identifiability,
)
from discopt.doe.profile import (
    ProfileLikelihoodResult,
    profile_all,
    profile_likelihood,
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
    "DoERound",
    "EstimabilityResult",
    "ExplorationResult",
    "FIMResult",
    "IdentifiabilityDiagnostics",
    "IdentifiabilityResult",
    "ProfileLikelihoodResult",
    "batch_optimal_experiment",
    "check_identifiability",
    "collinearity_index",
    "compute_fim",
    "d_optimal_subset",
    "diagnose_identifiability",
    "estimability_rank",
    "explore_design_space",
    "optimal_experiment",
    "profile_all",
    "profile_likelihood",
    "sequential_doe",
]
