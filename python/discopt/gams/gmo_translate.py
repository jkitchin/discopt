"""Build a discopt :class:`~discopt.modeling.core.Model` from a GMO model view.

The translation reads the model purely through the small :class:`GmoView`
protocol below -- column bounds/types, the objective (constant + linear gradient
+ nonlinear instruction list), and each constraint row (constant + linear row +
nonlinear instruction list + sense + rhs).  Keeping the interface this thin lets
the whole translation be unit-tested with an in-memory fake, with the only
GAMS-library-specific code living in the adapter in :mod:`discopt.gams.link`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from discopt.modeling import core as dm
from discopt.modeling.core import Expression, Model

from .instructions import GamsTranslationError, translate_instructions

# GMO variable-type codes (gamsapi ``gmovar_*``, checked against GAMS 53).
GMO_VAR_CONT = 0
GMO_VAR_BINARY = 1
GMO_VAR_INTEGER = 2
GMO_VAR_SOS1 = 3
GMO_VAR_SOS2 = 4
GMO_VAR_SEMICONT = 5
GMO_VAR_SEMIINT = 6

# Bounds at or beyond this magnitude are GAMS's +/-INF (discopt's 1e20 sentinel).
_INF = 1e19


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
    def obj_nl_sign(self) -> float: ...

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
    semi: list[tuple] = []
    for j in range(n):
        name = g.var_name(j) or f"x{j + 1}"
        lo = g.var_lower(j)
        hi = g.var_upper(j)
        vtype = g.var_type(j)
        if vtype in (GMO_VAR_SOS1, GMO_VAR_SOS2):
            # Read as continuous this silently dropped the SOS restriction; the
            # .gms reader refuses the same way (#1613).
            raise GamsTranslationError(
                f"GMO variable '{name}' is an SOS{vtype - 2} member; the discopt GAMS "
                "link does not translate SOS sets, and reading it as continuous would "
                "drop the restriction. Model it with binaries instead."
            )
        if vtype == GMO_VAR_BINARY:
            # A fixed binary (x.fx = 1) keeps its bounds in GMO; m.binary alone
            # would widen it back to [0, 1].
            var = m.binary(name)
            var.lb = np.asarray(max(0.0, float(lo)))
            var.ub = np.asarray(min(1.0, float(hi)))
        elif vtype in (GMO_VAR_INTEGER, GMO_VAR_SEMIINT):
            # GAMS's default integer .up is +INF (GAMS 53); it stays infinite
            # rather than being clamped to an invented finite box.
            var = m.integer(name, lb=_bound(lo), ub=_bound(hi))
        elif vtype in (GMO_VAR_CONT, GMO_VAR_SEMICONT):
            var = m.continuous(name, lb=_bound(lo), ub=_bound(hi))
        else:
            raise GamsTranslationError(f"GMO variable '{name}' has unknown type code {vtype}")
        if vtype in (GMO_VAR_SEMICONT, GMO_VAR_SEMIINT):
            semi.append((var, name, float(lo), float(hi)))
        variables.append(var)
        lvl = g.var_level(j)
        if lvl not in (0.0, None):
            initial[name] = float(lvl)

    # The model's first n variables are the GMO columns, in order; the solution
    # write-back relies on that, so auxiliary variables are created only after.
    m._gams_columns = list(variables)  # type: ignore[attr-defined]
    for var, name, lo, hi in semi:
        _lower_semi(m, var, name, lo, hi)

    # 2. Objective: constant + linear gradient + nonlinear part.
    obj = _const(g.obj_constant())
    obj = _add_linear(obj, g.obj_linear(), variables)
    obj_op, obj_fld = g.obj_nl()
    if obj_op:
        # GMO stores the objective *equation* residual (e.g. ``objvar - f(x)``),
        # so its nonlinear part decodes to a sign-flipped ``f``; obj_nl_sign()
        # carries the ``-1 / objVarJacobian`` correction (1.0 for views that
        # already supply the objective directly).
        sign = _obj_nl_sign(g)
        nl = translate_instructions(obj_op, obj_fld, variables, constants)
        obj = obj + (nl if sign == 1.0 else _const(sign) * nl)
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
        if sense == "free":
            # An =N= row imposes no relation (GAMS "no constraint"), so it adds
            # nothing to the feasible set.
            continue
        m.subject_to(_build_constraint(body, sense, rhs), name=name)

    if initial:
        # Surfaced for warm-starting; mirrors gams_parser's convention.
        m._gams_initial_values = initial  # type: ignore[attr-defined]
    return m


def _obj_nl_sign(g: GmoView) -> float:
    """Sign correction for the objective nonlinear part (1.0 if not supplied).

    A view that supplies the method must answer: a failure here used to be
    swallowed and replaced by ``1.0``, which flips the sign of the objective's
    nonlinear part for a GMO objective row.
    """
    fn = getattr(g, "obj_nl_sign", None)
    if fn is None:
        return 1.0
    return float(fn())


def _lower_semi(m: Model, var, name: str, lo: float, hi: float) -> None:
    """Impose GAMS ``semicont``/``semiint`` semantics exactly: ``x = 0`` or ``lo <= x <= hi``.

    Mirrors the .gms reader's lowering (#1613 C-03c): where ``0`` lies in
    ``[lo, hi]`` the union is the box itself; otherwise an on/off binary ``z``
    gives ``lo*z <= x <= hi*z`` on the hull ``[min(0, lo), max(0, hi)]``.
    Integrality, for ``semiint``, stays on ``x``. An infinite bound would need an
    invented big-M, so it is refused rather than approximated.
    """
    if lo <= 0.0 <= hi:
        return
    if not (abs(lo) < _INF and abs(hi) < _INF):
        raise GamsTranslationError(
            f"semicontinuous/semi-integer variable '{name}' has bounds [{lo}, {hi}] "
            "excluding 0 and an infinite bound; `x = 0 or lo <= x <= up` has no exact "
            "bounded lowering then. Give it a finite .lo and .up."
        )
    z = m.binary(f"{name}__semi_on")
    m.subject_to(var >= lo * z, name=f"{name}__semi_lo")
    m.subject_to(var <= hi * z, name=f"{name}__semi_up")
    var.lb = np.asarray(min(0.0, lo))
    var.ub = np.asarray(max(0.0, hi))


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


def _bound(value: float) -> float:
    """A GMO bound as a discopt bound: GAMS +/-INF becomes the +/-1e20 sentinel."""
    v = float(value)
    if v <= -_INF:
        return -1e20
    if v >= _INF:
        return 1e20
    return v
