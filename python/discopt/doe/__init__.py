"""discopt.doe -- Design of Experiments.

This package has three complementary entry points, each tailored to
a different question:

1. **"What is the best operating condition?"**
   :func:`optimize_round` runs one active-learning round: fit a
   surrogate to the experiments completed so far, recommend the
   next batch via an acquisition function (expected improvement,
   UCB, steepest ascent), append the batch to a workbook for
   execution. See the ``active-learning`` tutorial notebook.

2. **"Does this factor matter?"**
   :func:`factorial_2level_design` builds 2-level full factorial
   screening designs; :func:`effects_estimates` gives signed
   main-effect estimates; :func:`anova_report` produces the
   F-table.

3. **"How precisely can I estimate the model parameters?"**
   :func:`optimal_experiment` and :func:`batch_optimal_experiment`
   solve for an exact D/A/E-optimal design using the Fisher
   Information Matrix computed with JAX autodiff;
   :func:`diagnose_identifiability` and
   :func:`estimability_rank` warn when the chosen experiments
   cannot identify the parameters.

Quick start (FIM-based parameter-estimation design)
---------------------------------------------------
>>> from discopt.doe import compute_fim, optimal_experiment, DesignCriterion
>>> fim_result = compute_fim(experiment, param_values, design_values)
>>> design = optimal_experiment(experiment, param_values, design_bounds)
>>> print(design.summary())

Quick start (active-learning optimization)
------------------------------------------
>>> from discopt.doe import optimize_round, OptimizationCriterion
>>> result = optimize_round(
...     workbook="opt.xlsx",
...     criterion=OptimizationCriterion.MAXIMIZE,
...     surrogate="gp",                       # or sklearn estimator, or Surrogate
...     acquisition="expected_improvement",
...     batch_size=4,
... )
>>> print(result.next_designs)

Identifiability and estimability diagnostics
--------------------------------------------
>>> from discopt.doe import diagnose_identifiability, estimability_rank, profile_likelihood
>>> diag = diagnose_identifiability(experiment, param_values)
>>> est = estimability_rank(experiment, param_values)
>>> profile = profile_likelihood(experiment, data, "k")

See Also
--------
discopt.estimate : Parameter estimation using the same Experiment interface.
discopt.doe.surrogate : Surrogate model protocol + sklearn adapter.
discopt.doe.acquisition : Acquisition functions (EI, UCB, steepest ascent).
"""

from discopt.doe.acquisition import (
    ACQUISITIONS,
    confidence_bound,
    expected_improvement,
    resolve_acquisition,
    steepest_ascent,
)
from discopt.doe.anova import AnovaEffect, AnovaTable, anova_report
from discopt.doe.design import (
    BatchDesignResult,
    BatchStrategy,
    DesignConstraint,
    DesignCriterion,
    DesignResult,
    batch_optimal_experiment,
    optimal_experiment,
    project_to_simplex,
    sample_simplex,
    sum_constraint,
)
from discopt.doe.discrimination import (
    DiscriminationCriterion,
    DiscriminationDesignResult,
    discriminate_compound,
    discriminate_design,
    evaluate_discrimination_criterion,
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
from discopt.doe.fractional import (
    fractional_factorial_design,
)
from discopt.doe.latin import (
    LatinDesign,
    graeco_latin_square,
    hyper_graeco_latin_square,
    latin_square,
    latin_square_design,
)
from discopt.doe.model_based import (
    ModelBasedRoundResult,
    ParametricSurrogate,
    model_based_optimize_round,
)
from discopt.doe.optimize import (
    OptimizationCriterion,
    OptimizationRoundResult,
    optimize_round,
)
from discopt.doe.profile import (
    ProfileLikelihoodResult,
    profile_all,
    profile_likelihood,
)
from discopt.doe.screening import (
    FactorialDesign,
    effects_estimates,
    factorial_2level_design,
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
from discopt.doe.surrogate import (
    PRESETS as SURROGATE_PRESETS,
)
from discopt.doe.surrogate import (
    Surrogate,
    coerce_surrogate,
)
from discopt.doe.templates import (
    TEMPLATE_NAMES,
    build_template,
    linear_template,
    polynomial_1d_template,
    response_surface_template,
    scheffe_linear_template,
    scheffe_quadratic_template,
    scheffe_special_cubic_template,
    simplex_centroid_points,
    simplex_lattice_points,
)

__all__ = [
    "AnovaEffect",
    "AnovaTable",
    "BatchDesignResult",
    "BatchStrategy",
    "DesignConstraint",
    "DesignCriterion",
    "DesignResult",
    "FactorialDesign",
    "LatinDesign",
    "DiscriminationCriterion",
    "DiscriminationDesignResult",
    "DiscriminationRound",
    "DoERound",
    "EstimabilityResult",
    "ExplorationResult",
    "FIMResult",
    "IdentifiabilityDiagnostics",
    "IdentifiabilityResult",
    "ModelBasedRoundResult",
    "ModelSelectionResult",
    "OptimizationCriterion",
    "OptimizationRoundResult",
    "ParametricSurrogate",
    "ProfileLikelihoodResult",
    "SURROGATE_PRESETS",
    "Surrogate",
    "ACQUISITIONS",
    "TEMPLATE_NAMES",
    "anova_report",
    "batch_optimal_experiment",
    "build_template",
    "coerce_surrogate",
    "confidence_bound",
    "expected_improvement",
    "optimize_round",
    "resolve_acquisition",
    "steepest_ascent",
    "check_identifiability",
    "graeco_latin_square",
    "hyper_graeco_latin_square",
    "latin_square",
    "latin_square_design",
    "collinearity_index",
    "compute_fim",
    "d_optimal_subset",
    "diagnose_identifiability",
    "discriminate_compound",
    "effects_estimates",
    "factorial_2level_design",
    "fractional_factorial_design",
    "discriminate_design",
    "estimability_rank",
    "evaluate_discrimination_criterion",
    "explore_design_space",
    "likelihood_ratio_test",
    "linear_template",
    "model_based_optimize_round",
    "model_selection",
    "optimal_experiment",
    "polynomial_1d_template",
    "project_to_simplex",
    "response_surface_template",
    "profile_all",
    "profile_likelihood",
    "sample_simplex",
    "scheffe_linear_template",
    "scheffe_quadratic_template",
    "scheffe_special_cubic_template",
    "sequential_discrimination",
    "sequential_doe",
    "simplex_centroid_points",
    "simplex_lattice_points",
    "sum_constraint",
    "vuong_test",
]
