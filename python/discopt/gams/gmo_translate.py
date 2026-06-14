"""Build a discopt :class:`~discopt.modeling.core.Model` from a GMO model view.

The translation reads the model purely through the small :class:`GmoView`
protocol below -- column bounds/types, the objective (constant + linear gradient
+ nonlinear instruction list), and each constraint row (constant + linear row +
nonlinear instruction list + sense + rhs).  Keeping the interface this thin lets
the whole translation be unit-tested with an in-memory fake, with the only
GAMS-library-specific code living in the adapter in :mod:`discopt.gams.link`.
"""

from __future__ import annotations

import warnings
from typing import Protocol

from discopt.modeling import core as dm
from discopt.modeling.core import Expression, Model

from .instructions import translate_instructions

# GMO variable-type codes (gmomcc: gmovar_X / _B / _I / _SC / _SI / _S1 / _S2).
GMO_VAR_CONT = 0
GMO_VAR_BINARY = 1
GMO_VAR_INTEGER = 2


class GmoView(Protocol):
    """Minimal read interface over a GAMS Modeling Object."""

    def name(self) -> str: ...
    def num_vars(self) -> int: ...
    def num_rows(self) -> int: ...
    def minimize(self) -> bool: ...
    def constants(self) -> list[float]: ...

    def var_lower(self, j: int) -> float: ...
    def var_upper(self, j: int) -> float: ...
    def var_type(self, j: int) -> int: ...
    def var_name(self, j: int) -> str: ...
    def var_level(self, j: int) -> float: ...

    def obj_constant(self) -> float: ...
    def obj_linear(self) -> dict[int, float]: ...
    def obj_nl(self) -> tuple[list[int], list[int]]: ...

    def row_name(self, i: int) -> str: ...
    def row_sense(self, i: int) -> str: ...
    def row_rhs(self, i: int) -> float: ...
    def row_constant(self, i: int) -> float: ...
    def row_linear(self, i: int) -> dict[int, float]: ...
    def row_nl(self, i: int) -> tuple[list[int], list[int]]: ...


def model_from_gmo(g: GmoView) -> Model:
    """Translate a GMO model view into a discopt :class:`Model`."""
    m = Model(g.name() or "gams_model")
    n = g.num_vars()
    constants = list(g.constants())

    # 1. Columns. Each GMO column is a scalar discopt variable.
    variables: list[Expression] = []
    initial: dict[str, float] = {}
    for j in range(n):
        name = g.var_name(j) or f"x{j + 1}"
        lo = g.var_lower(j)
        hi = g.var_upper(j)
        vtype = g.var_type(j)
        if vtype == GMO_VAR_BINARY:
            var = m.binary(name)
        elif vtype == GMO_VAR_INTEGER:
            var = m.integer(name, lb=_finite(lo, 0.0), ub=_finite(hi, 1e9))
        else:
            if vtype not in (GMO_VAR_CONT,):
                warnings.warn(
                    f"GMO variable '{name}' has unsupported type code {vtype}; "
                    "treated as continuous with its declared bounds.",
                    stacklevel=2,
                )
            var = m.continuous(name, lb=_finite(lo, -1e20), ub=_finite(hi, 1e20))
        variables.append(var)
        lvl = g.var_level(j)
        if lvl not in (0.0, None):
            initial[name] = float(lvl)

    # 2. Objective: constant + linear gradient + nonlinear part.
    obj = _const(g.obj_constant())
    obj = _add_linear(obj, g.obj_linear(), variables)
    obj_op, obj_fld = g.obj_nl()
    if obj_op:
        obj = obj + translate_instructions(obj_op, obj_fld, variables, constants)
    if g.minimize():
        m.minimize(obj)
    else:
        m.maximize(obj)

    # 3. Constraint rows.
    for i in range(g.num_rows()):
        body = _const(g.row_constant(i))
        body = _add_linear(body, g.row_linear(i), variables)
        row_op, row_fld = g.row_nl(i)
        if row_op:
            body = body + translate_instructions(row_op, row_fld, variables, constants)
        rhs = g.row_rhs(i)
        sense = g.row_sense(i)
        name = g.row_name(i) or f"e{i + 1}"
        m.subject_to(_build_constraint(body, sense, rhs), name=name)

    if initial:
        # Surfaced for warm-starting; mirrors gams_parser's convention.
        m._gams_initial_values = initial  # type: ignore[attr-defined]
    return m


def _build_constraint(body: Expression, sense: str, rhs: float):
    rhs_c = _const(rhs)
    if sense == "<=":
        return body <= rhs_c
    if sense == ">=":
        return body >= rhs_c
    if sense == "==":
        return body == rhs_c
    raise ValueError(f"unknown constraint sense {sense!r}")


def _add_linear(expr: Expression, linear: dict[int, float], variables) -> Expression:
    for j, coef in linear.items():
        if coef == 0.0:
            continue
        term = variables[j] if coef == 1.0 else _const(coef) * variables[j]
        expr = term if _is_zero_const(expr) else expr + term
    return expr


def _const(value: float) -> Expression:
    return dm.Constant(float(value))


def _is_zero_const(expr: Expression) -> bool:
    return isinstance(expr, dm.Constant) and float(expr.value) == 0.0


def _finite(value: float, default: float) -> float:
    """Clamp GAMS +/-inf sentinels to a default for bound construction."""
    if value is None:
        return default
    v = float(value)
    if v <= -1e19:
        return default if default < 0 else -1e20
    if v >= 1e19:
        return default if default > 0 else 1e20
    return v
