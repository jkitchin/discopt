"""
discopt.export -- Export optimization models to standard file formats.

Supports MPS (Mathematical Programming System) and CPLEX LP format
for interoperability with external solvers such as Gurobi, CPLEX,
MOSEK, HiGHS, and SCIP.

Only linear and quadratic models are supported. Nonlinear expressions
raise ``ValueError``.

Examples
--------
>>> import discopt.modeling as dm
>>> m = dm.Model("example")
>>> x = m.continuous("x", lb=0, ub=10)
>>> y = m.continuous("y", lb=0, ub=10)
>>> m.minimize(3 * x + 2 * y)
>>> m.subject_to(x + y >= 5)
>>> mps_string = m.to_mps()
>>> m.to_lp("example.lp")
"""

from discopt.export.gams import to_gams
from discopt.export.lp import to_lp
from discopt.export.mps import to_mps
from discopt.export.nl import to_nl

__all__ = ["to_mps", "to_lp", "to_gams", "to_nl"]
