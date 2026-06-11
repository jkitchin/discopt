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
    BooleanVar,
    BooleanVarArray,
    Constraint,
    # Opaque AD-only user function node (for isinstance checks)
    CustomCall,
    Disjunct,
    # Expressions (for isinstance checks, rarely needed)
    Expression,
    LogicalExpression,
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
    # Logical functions
    atleast,
    atmost,
    cos,
    custom,
    erf,
    exactly,
    # Mathematical functions
    exp,
    from_description,
    from_gams,
    from_nl,
    # Import functions
    from_pyomo,
    # Conditional / user-defined function
    if_else,
    land,
    lnot,
    log,
    log1p,
    log2,
    log10,
    lor,
    maximum,
    minimum,
    norm,
    prod,
    sigmoid,
    sign,
    sin,
    softplus,
    sqrt,
    # Aggregation
    sum,
    tan,
    tanh,
    udf,
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
    "sigmoid",
    "sign",
    "softplus",
    "minimum",
    "maximum",
    "if_else",
    "udf",
    "custom",
    "CustomCall",
    "sum",
    "prod",
    "norm",
    "tanh",
    "SolveResult",
    "SolveUpdate",
    "from_pyomo",
    "from_nl",
    "from_gams",
    "from_description",
    "BooleanVar",
    "BooleanVarArray",
    "Disjunct",
    "LogicalExpression",
    "land",
    "lor",
    "lnot",
    "atleast",
    "atmost",
    "exactly",
]
