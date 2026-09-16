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

from discopt.batch import solve_batch
from discopt.modeling.argmin import (
    argmin,
    argmin_kkt,
    argmin_layer,
)
from discopt.modeling.core import (
    BooleanVar,
    BooleanVarArray,
    Constraint,
    # Opaque AD-only user function node (for isinstance checks)
    CustomCall,
    DiscontinuousIntrinsicError,
    Disjunct,
    # Disjunction semantics (issue #1124)
    DisjunctionSemantics,
    # Expressions (for isinstance checks, rarely needed)
    Expression,
    LogicalExpression,
    # Model
    Model,
    Parameter,
    SelectorActivation,
    SelectorCardinality,
    # Results
    SolveResult,
    SolveUpdate,
    # Variable types (for isinstance checks, rarely needed)
    Variable,
    VarType,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    # Logical functions
    atleast,
    atmost,
    bulk_construction_gc,
    ceil,
    concatenate,
    cos,
    cosh,
    custom,
    erf,
    exactly,
    # Mathematical functions
    exp,
    floor,
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
    round_,
    sigmoid,
    sign,
    sin,
    sinh,
    softplus,
    sqrt,
    stack,
    # Aggregation
    sum,
    tan,
    tanh,
    trunc,
    udf,
    xlogx,
)
from discopt.modeling.core import (
    abs_ as abs,
)
from discopt.modeling.implicit import (
    implicit,
    implicit_full_space,
)
from discopt.modeling.indexed import (
    IndexedConstraint,
    IndexedParam,
    IndexedVar,
    Skip,
)
from discopt.modeling.sets import ProductSet, RangeSet, Set
from discopt.operators import register_function, registered_names
from discopt.serialize import dumps, load, loads

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
    "atan",
    "asin",
    "acos",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "abs",
    "sigmoid",
    "sign",
    "softplus",
    "xlogx",
    # Recognised but unimplemented discontinuous intrinsics: these are
    # exported so the refusal is explicit and documented (issue #1237).
    # Calling one raises DiscontinuousIntrinsicError.
    "floor",
    "ceil",
    "round_",
    "trunc",
    "DiscontinuousIntrinsicError",
    "minimum",
    "maximum",
    "if_else",
    "udf",
    "custom",
    "argmin",
    "argmin_kkt",
    "argmin_layer",
    "implicit",
    "implicit_full_space",
    "CustomCall",
    "sum",
    "bulk_construction_gc",
    "prod",
    "norm",
    "concatenate",
    "stack",
    "tanh",
    "SolveResult",
    "solve_batch",
    "register_function",
    "registered_names",
    "SolveUpdate",
    "from_pyomo",
    "from_nl",
    "from_gams",
    "load",
    "loads",
    "dumps",
    "from_description",
    "BooleanVar",
    "BooleanVarArray",
    "Disjunct",
    "DisjunctionSemantics",
    "SelectorActivation",
    "SelectorCardinality",
    "LogicalExpression",
    "land",
    "lor",
    "lnot",
    "atleast",
    "atmost",
    "exactly",
    "Set",
    "RangeSet",
    "ProductSet",
    "IndexedVar",
    "IndexedParam",
    "IndexedConstraint",
    "Skip",
]
