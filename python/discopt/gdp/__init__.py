"""Generalized disjunctive programming: one normalized view of a model's GDP.

See :mod:`discopt.gdp.tree` (#1775 G1.1). The lowerings themselves live in
:mod:`discopt._relax.gdp_reformulate`.
"""

from discopt.gdp.tree import DisjunctionNode, DisjunctNode, GDPTree, gdp_tree

__all__ = ["DisjunctNode", "DisjunctionNode", "GDPTree", "gdp_tree"]
