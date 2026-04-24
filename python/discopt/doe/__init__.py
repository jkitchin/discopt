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
    "EstimabilityResult",
    "ExplorationResult",
    "FIMResult",
    "IdentifiabilityDiagnostics",
    "IdentifiabilityResult",
    "ModelSelectionResult",
    "ProfileLikelihoodResult",
    "batch_optimal_experiment",
    "check_identifiability",
    "collinearity_index",
    "compute_fim",
    "d_optimal_subset",
    "diagnose_identifiability",
    "discriminate_compound",
    "discriminate_design",
    "estimability_rank",
    "explore_design_space",
    "likelihood_ratio_test",
    "model_selection",
    "optimal_experiment",
    "profile_all",
    "profile_likelihood",
    "sequential_discrimination",
    "sequential_doe",
    "vuong_test",
]
