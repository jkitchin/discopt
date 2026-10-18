"""Generalized disjunctive programming: one normalized view of a model's GDP.

See :mod:`discopt.gdp.tree` (#1775 G1.1). The lowerings themselves live in
:mod:`discopt._relax.gdp_reformulate`.
"""

from discopt.gdp.state import (
    DisjunctiveStateToken,
    fix_disjuncts,
    reduced_subproblem,
    reverse_current_disjunctive_state,
    transform_current_disjunctive_state,
)
from discopt.gdp.tree import DisjunctionNode, DisjunctNode, GDPTree, gdp_tree

__all__ = [
    "DisjunctNode",
    "DisjunctionNode",
    "DisjunctiveStateToken",
    "GDPTree",
    "fix_disjuncts",
    "gdp_tree",
    "reduced_subproblem",
    "reverse_current_disjunctive_state",
    "transform_current_disjunctive_state",
]
