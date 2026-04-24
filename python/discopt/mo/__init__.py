"""discopt.mo -- Multi-objective optimization via scalarization.

Wraps discopt's single-objective MINLP solver in deterministic scalarization
loops (weighted sum, AUGMECON2 ε-constraint, augmented weighted Tchebycheff,
Normal Boundary Intersection, Normalized Normal Constraint) and returns a
:class:`ParetoFront` result object with built-in dominance filtering,
hypervolume / IGD / spread / epsilon indicators, and 2-D / 3-D plots.

Quick Start
-----------

>>> import discopt.modeling as dm
>>> from discopt.mo import weighted_sum, epsilon_constraint
>>>
>>> m = dm.Model("biobj")
>>> x = m.continuous("x", shape=(2,), lb=0, ub=5)
>>> f1 = x[0] ** 2 + x[1] ** 2
>>> f2 = (x[0] - 2) ** 2 + (x[1] - 1) ** 2
>>> front = epsilon_constraint(m, [f1, f2], n_points=11)
>>> print(front.summary())
>>> hv = front.hypervolume()

When to use which method
------------------------

* :func:`weighted_sum` -- fast baseline, complete for convex fronts only.
* :func:`epsilon_constraint` (AUGMECON2) -- complete for general (including
  nonconvex) fronts; sensitive to the ε-grid.
* :func:`weighted_tchebycheff` -- complete for nonconvex fronts; based on
  L_inf distance from the ideal.
* :func:`normal_boundary_intersection` -- uniform-spacing geometric
  scalarization; works well on convex fronts.
* :func:`normalized_normal_constraint` -- robust geometric scalarization
  with explicit objective normalization.

See the crucible articles under ``.crucible/wiki/methods/`` for algorithmic
background.

Side effects
------------

Scalarizers mutate the input ``Model`` (add auxiliary parameters, variables,
and constraints; restore the objective on exit). Create a fresh model if
you intend to reuse it for other solves.
"""

from discopt.mo.indicators import (
    epsilon_indicator,
    hypervolume,
    igd,
    spread,
)
from discopt.mo.nbi import (
    normal_boundary_intersection,
    normalized_normal_constraint,
)
from discopt.mo.pareto import (
    ParetoFront,
    ParetoPoint,
    filter_nondominated,
)
from discopt.mo.scalarization import (
    epsilon_constraint,
    weighted_sum,
    weighted_tchebycheff,
)
from discopt.mo.utils import (
    ideal_point,
    nadir_point,
    normalize_objectives,
)

__all__ = [
    # Result types
    "ParetoPoint",
    "ParetoFront",
    # Utilities
    "ideal_point",
    "nadir_point",
    "normalize_objectives",
    "filter_nondominated",
    # Scalarizations
    "weighted_sum",
    "epsilon_constraint",
    "weighted_tchebycheff",
    "normal_boundary_intersection",
    "normalized_normal_constraint",
    # Indicators
    "hypervolume",
    "igd",
    "spread",
    "epsilon_indicator",
]
