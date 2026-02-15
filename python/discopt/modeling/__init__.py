"""
discopt.modeling -- Modeling API for Mixed-Integer Nonlinear Programs.

This module provides the classes and functions for building optimization
models: :class:`Model`, :class:`Variable`, :class:`Expression`,
:class:`Constraint`, :class:`SolveResult`, and math functions
(:func:`exp`, :func:`log`, :func:`sin`, :func:`cos`, etc.).

Examples
--------
>>> import discopt.modeling as dm
>>> m = dm.Model("my_problem")
>>> x = m.continuous("x", shape=(3,), lb=0, ub=10)
>>> y = m.binary("y", shape=(2,))
>>> m.minimize(cost @ x + fixed_cost @ y)
>>> m.subject_to(A @ x <= b, name="capacity")
>>> result = m.solve()
"""

from discopt.modeling.core import (
    Constraint,
    # Expressions (for isinstance checks, rarely needed)
    Expression,
    # Model
    Model,
    Parameter,
    # Results
    SolveResult,
    SolveUpdate,
    # Variable types (for isinstance checks, rarely needed)
    Variable,
    VarType,
    acosh,
    asinh,
    atanh,
    cos,
    erf,
    # Mathematical functions
    exp,
    from_description,
    from_gams,
    from_nl,
    # Import functions
    from_pyomo,
    log,
    log1p,
    log2,
    log10,
    maximum,
    minimum,
    norm,
    prod,
    sign,
    sin,
    sqrt,
    # Aggregation
    sum,
    tan,
)
from discopt.modeling.core import (
    abs_ as abs,
)

__all__ = [
    "Model",
    "Variable",
    "VarType",
    "Parameter",
    "Expression",
    "Constraint",
    "exp",
    "log",
    "log1p",
    "log2",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "abs",
    "sign",
    "minimum",
    "maximum",
    "sum",
    "prod",
    "norm",
    "SolveResult",
    "SolveUpdate",
    "from_pyomo",
    "from_nl",
    "from_gams",
    "from_description",
]
