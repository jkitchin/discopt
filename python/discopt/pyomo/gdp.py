"""Pyomo GDP model -> native discopt :class:`~discopt.modeling.core.Model` (#1775 G0.4).

The ``discopt.pyomo`` solver plugin reaches discopt only through an ``.nl`` file,
which Pyomo can write only after a GDP transformation (``gdp.bigm``, ``gdp.hull``)
has already lowered every disjunction. That throws away exactly the structure a
GDP algorithm works on. :func:`from_pyomo_gdp` instead walks the Pyomo model and
rebuilds it with discopt's own modeling API, so the disjunctive structure arrives
intact:

* **Expressions** -- Pyomo's numeric expression tree (sums, linear and monomial
  terms, products, division, powers, negation, the unary functions, ``abs``,
  ``max``/``min``, named ``Expression`` components) maps onto discopt operators.
  A subtree with no free variable (``Param`` values, fixed variables, numeric
  constants) is evaluated to a constant at conversion time. A constant base with
  a variable exponent, ``c ** f``, becomes the identical ``exp(f * ln c)`` (the
  interval evaluator has no ``c ** f`` rule); ``c <= 0`` there refuses.
* **pyomo.dae** -- a discretized ``ContinuousSet`` is a finite set and a fully
  discretized ``DerivativeVar`` an ordinary variable tied by the discretization's
  rows; either one undiscretized refuses.
* **Variables** -- one discopt column per Pyomo ``VarData``, created on first
  reference, with its bounds and integrality (``domain.get_interval()``). A
  variable that no active row or objective references is not part of the
  optimization problem and is not created, as Pyomo's own ``.nl`` writer omits it.
* **GDP structure** -- every active ``Disjunction`` becomes
  :meth:`Model.add_disjunction` over :meth:`Model.make_disjunct` blocks; a
  ``Disjunction`` declared inside a ``Disjunct`` becomes
  :meth:`Disjunct.add_disjunction` on that block (#1775 G1.2); ``xor=False``
  becomes :attr:`DisjunctionSemantics.OR` (G1.4). A variable declared on a
  ``Disjunct`` and referenced only inside it and the disjunctions nested in it
  becomes a :meth:`Disjunct.variable` local; one referenced elsewhere stays an
  ordinary column, which is how Pyomo's own transformations treat it.
* **Indicators and logic** -- a top-level disjunct's ``indicator_var`` maps to
  the discopt block's :attr:`Disjunct.indicator` and its ``binary_indicator_var``
  to that indicator's binary column, so a cost term written over Pyomo's
  indicator binaries prices the same selector. A *nested* disjunct's indicator
  does not mean the same thing in the two systems: discopt's tree selector
  implies its parent's (the inner selectors sum to the parent's), while Pyomo's
  is free when the parent is False (the inner disjunction and its rows are then
  relaxed). When anything outside the disjunction tree references a nested
  indicator -- a row, the objective, a logical constraint, a fixed value -- it
  becomes its own binary ``W`` and the tree selector is tied to it as
  ``y = W AND y_parent``, which is Pyomo's semantics exactly; an unreferenced
  nested indicator needs no column, since the tree selector is then a valid value
  for it. Free ``BooleanVar`` components map to :meth:`Model.boolean`;
  ``LogicalConstraint`` maps to :meth:`Model.logical`. A fixed indicator, a fixed
  ``BooleanVar``, and a deactivated disjunct inside an active disjunction
  (Pyomo's spelling of "fixed to False") are pinned with a row.

Anything else refuses loudly, naming the component: nothing is dropped. The
refusals are the cases where a faithful rebuild needs a discopt feature that does
not exist (``Expr_if``, external functions, discrete non-integer domains,
``Piecewise``, ``SOSConstraint``, a logical constraint inside a disjunct, a
logical constraint that folds to False, more than one active objective) or
where the Pyomo model is itself ill-formed (a disjunct in no disjunction, or in
two).

:func:`convert_pyomo_gdp` returns the discopt model together with the maps back
to the Pyomo components; :func:`check_evaluation_parity` uses them to compare
every row and the objective at random points, through discopt's interval
evaluator rather than the converter's own code.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np

import discopt.modeling as dm
from discopt.modeling.core import (
    BooleanVar,
    Constant,
    Constraint,
    Disjunct,
    DisjunctionSemantics,
    Expression,
    LogicalExpression,
    Model,
    Variable,
)


class PyomoGDPConversionError(ValueError):
    """The Pyomo model holds something :func:`from_pyomo_gdp` cannot rebuild faithfully."""


@dataclass
class RowRecord:
    """One discopt row emitted for one bound of one Pyomo ``ConstraintData``.

    ``kind`` is ``"eq"``, ``"lb"`` or ``"ub"``; ``bound`` the Pyomo bound's value;
    ``body`` the converted Pyomo body; ``row`` the discopt constraint built from
    them; ``disjunct`` the Pyomo disjunct holding the constraint (``None`` at the
    top level).
    """

    component: Any
    kind: str
    bound: float
    body: Any
    row: Constraint
    disjunct: Any = None


@dataclass
class PyomoGDPConversion:
    """The result of :func:`convert_pyomo_gdp`."""

    model: Model
    # The three maps and ``local`` are Pyomo ``ComponentMap`` / ``ComponentSet``:
    # Pyomo components are keyed by identity, and some are not hashable.
    #: Pyomo ``VarData`` -> discopt scalar column.
    variables: Any = None
    #: Pyomo ``DisjunctData`` -> discopt :class:`Disjunct` block.
    disjuncts: Any = None
    #: Pyomo ``BooleanVarData`` -> discopt :class:`BooleanVar`.
    booleans: Any = None
    #: Pyomo ``DisjunctionData`` in the order they were emitted.
    disjunctions: list[Any] = field(default_factory=list)
    #: Pyomo ``LogicalConstraintData`` in the order they were emitted.
    logical: list[Any] = field(default_factory=list)
    rows: list[RowRecord] = field(default_factory=list)
    objective: Any = None
    objective_expr: Any = None
    #: Pyomo variables that became ``Disjunct.variable`` locals.
    local: Any = None


def from_pyomo_gdp(pyomo_model: Any) -> Model:
    """Rebuild a Pyomo (GDP) model as a native discopt :class:`Model`.

    See the module docstring for what maps to what. Raises
    :class:`PyomoGDPConversionError` naming the first component that cannot be
    rebuilt faithfully.
    """
    return convert_pyomo_gdp(pyomo_model).model


def convert_pyomo_gdp(pyomo_model: Any, name: Optional[str] = None) -> PyomoGDPConversion:
    """Like :func:`from_pyomo_gdp`, returning the maps back to the Pyomo components."""
    return _Converter(pyomo_model, name).run()


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------

_UNARY = {
    "exp": dm.exp,
    "log": dm.log,
    "log10": dm.log10,
    "sqrt": dm.sqrt,
    "sin": dm.sin,
    "cos": dm.cos,
    "tan": dm.tan,
    "asin": dm.asin,
    "acos": dm.acos,
    "atan": dm.atan,
    "sinh": dm.sinh,
    "cosh": dm.cosh,
    "tanh": dm.tanh,
    "asinh": dm.asinh,
    "acosh": dm.acosh,
    "atanh": dm.atanh,
}


def _pyomo():
    try:
        import pyomo.environ as pyo
        from pyomo.core.base.boolean_var import BooleanVar as PyoBooleanVar
        from pyomo.core.base.logical_constraint import LogicalConstraint
        from pyomo.core.expr import logical_expr as le
        from pyomo.core.expr import numeric_expr as ne
        from pyomo.gdp import Disjunct as PyoDisjunct
        from pyomo.gdp import Disjunction as PyoDisjunction
    except ImportError as e:  # pragma: no cover - exercised only without pyomo
        raise ImportError("from_pyomo_gdp needs Pyomo: pip install discopt[pyomo]") from e
    return pyo, ne, le, PyoDisjunct, PyoDisjunction, PyoBooleanVar, LogicalConstraint


class _Converter:
    def __init__(self, pyomo_model: Any, name: Optional[str]) -> None:
        (
            self.pyo,
            self.ne,
            self.le,
            self.PDisjunct,
            self.PDisjunction,
            self.PBooleanVar,
            self.PLogical,
        ) = _pyomo()
        from pyomo.common.collections import ComponentMap, ComponentSet

        self.pm = pyomo_model
        self.out = PyomoGDPConversion(
            Model(name or pyomo_model.name),
            variables=ComponentMap(),
            disjuncts=ComponentMap(),
            booleans=ComponentMap(),
            local=ComponentSet(),
        )
        self._ComponentMap = ComponentMap
        self.m = self.out.model
        # id(VarData) -> discopt column; id(BooleanVarData) -> BooleanVar
        self._vars: dict[int, Variable] = {}
        self._bools: dict[int, BooleanVar] = {}
        # id(DisjunctData) -> discopt block, for indicator lookups
        self._dj: dict[int, Disjunct] = {}
        # id(VarData) -> (var, its home DisjunctData) for variables local to a block
        self._local_home: dict[int, Any] = {}
        # id(indicator_var / binary_indicator_var) -> its nested DisjunctData
        self._nested: dict[int, Any] = {}
        self._named: dict[int, Expression] = {}
        ne = self.ne
        self._known = (
            ne.SumExpression,
            ne.LinearExpression,
            ne.MonomialTermExpression,
            ne.ProductExpression,
            ne.DivisionExpression,
            ne.PowExpression,
            ne.NegationExpression,
            ne.UnaryFunctionExpression,
            ne.MaxExpression,
            ne.MinExpression,
        )

    # -- refusal -----------------------------------------------------------

    @staticmethod
    def _refuse(what: str, component: Any = None) -> PyomoGDPConversionError:
        where = f" ({_name(component)})" if component is not None else ""
        return PyomoGDPConversionError(f"from_pyomo_gdp cannot convert {what}{where}")

    # -- the walk ------------------------------------------------------------

    def run(self) -> PyomoGDPConversion:
        pyo = self.pyo
        self._check_components()
        top_disjunctions = list(self._data(self.pm, self.PDisjunction))
        all_disjuncts = list(
            self.pm.component_data_objects(
                self.PDisjunct, active=True, descend_into=(pyo.Block, self.PDisjunct)
            )
        )
        # Which disjunction (if any) claims each disjunct; refuse orphans and doubles.
        owner: dict[int, Any] = {}
        tree_order: list[Any] = []

        def claim(dj: Any, parent: Any) -> None:
            for d in dj.disjuncts:
                if id(d) in owner:
                    raise self._refuse(
                        f"disjunct {d.name!r} in two disjunctions "
                        f"({owner[id(d)].name!r} and {dj.name!r})"
                    )
                owner[id(d)] = dj
                tree_order.append(d)
                if d.active:
                    for inner in self._data(d, self.PDisjunction):
                        claim(inner, d)

        for dj in top_disjunctions:
            claim(dj, None)
        for d in all_disjuncts:
            if id(d) not in owner:
                raise self._refuse("an active Disjunct that belongs to no active Disjunction", d)

        self._find_locals(top_disjunctions)

        # Discopt blocks first: local variables are declared on them, and every
        # row must exist before the top-level add_disjunction lowers its tree.
        for d in tree_order:
            self._make_block(d)
        for d in tree_order:
            if d.active:
                self._fill_block(d)

        for c in self._data(self.pm, pyo.Constraint):
            for rec in self._rows(c, None):
                self.m.subject_to(rec.row)

        for dj in top_disjunctions:
            self._emit_disjunction(dj, parent=None)

        for lc in self._data(self.pm, self.PLogical, also_disjuncts=True):
            if self._enclosing_disjunct(lc) is not None:
                raise self._refuse(
                    "a LogicalConstraint inside a Disjunct (discopt Disjunct blocks "
                    "hold algebraic rows only)",
                    lc,
                )
            formula = self._logic(lc.expr, lc)
            if formula is False:
                raise self._refuse(
                    "a logical constraint that is always false (the model is "
                    "infeasible as written)",
                    lc,
                )
            if formula is not True:  # an always-true constraint constrains nothing
                self.m.logical(formula, name=lc.name)
            self.out.logical.append(lc)

        self._objective()
        self.out.variables = self._ComponentMap(
            (v, self._vars[id(v)]) for v in self._var_objs.values() if id(v) in self._vars
        )
        return self.out

    # -- components ------------------------------------------------------------

    def _data(self, block: Any, ctype: Any, also_disjuncts: bool = False) -> Iterable[Any]:
        descend = (self.pyo.Block, self.PDisjunct) if also_disjuncts else self.pyo.Block
        data: Iterable[Any] = block.component_data_objects(ctype, active=True, descend_into=descend)
        return data

    def _check_components(self) -> None:
        pyo = self.pyo
        from pyomo.core.base.suffix import Suffix

        allowed = {
            pyo.Var,
            pyo.Param,
            pyo.Set,
            pyo.RangeSet,
            pyo.Constraint,
            pyo.Objective,
            pyo.Expression,
            pyo.Block,
            self.PDisjunct,
            self.PDisjunction,
            self.PBooleanVar,
            self.PLogical,
            Suffix,
        }
        try:  # pyomo.dae: a discretized model is ordinary sets, variables and rows
            from pyomo.dae import ContinuousSet, DerivativeVar
        except ImportError:  # pragma: no cover - pyomo.dae ships with pyomo
            ContinuousSet = DerivativeVar = None
        for comp in self.pm.component_objects(
            active=True, descend_into=(pyo.Block, self.PDisjunct)
        ):
            if comp.ctype is ContinuousSet and comp.ctype is not None:
                if not comp.get_discretization_info():
                    raise self._refuse(
                        "a ContinuousSet that is not discretized (apply a pyomo.dae "
                        "discretization first)",
                        comp,
                    )
                continue
            if comp.ctype is DerivativeVar and comp.ctype is not None:
                if not comp.is_fully_discretized():
                    raise self._refuse(
                        "a DerivativeVar that is not fully discretized (apply a "
                        "pyomo.dae discretization first)",
                        comp,
                    )
                continue
            if comp.ctype not in allowed:
                raise self._refuse(f"a {comp.ctype.__name__} component", comp)

    def _enclosing_disjunct(self, component: Any) -> Any:
        blk = component.parent_block()
        while blk is not None:
            if blk.ctype is self.PDisjunct:
                return blk
            blk = blk.parent_block()
        return None

    # -- locality ---------------------------------------------------------------

    def _find_locals(self, top: list[Any]) -> None:
        """A variable declared on disjunct ``H`` is local iff every reference to it
        sits in ``H`` or a disjunct nested under ``H``."""
        from pyomo.core.expr.visitor import identify_variables

        pyo = self.pyo

        def ancestry(d: Any) -> list[int]:
            out = []
            while d is not None:
                out.append(id(d))
                d = self._parent_disjunct.get(id(d))
            return out

        self._parent_disjunct: dict[int, Any] = {}

        def record(dj: Any, parent: Any) -> None:
            for d in dj.disjuncts:
                self._parent_disjunct[id(d)] = parent
                if d.active:
                    for inner in self._data(d, self.PDisjunction):
                        record(inner, d)

        for dj in top:
            record(dj, None)

        refs: dict[int, list[Optional[int]]] = {}
        self._var_objs: dict[int, Any] = {}

        def visit(expr: Any, ctx: Any) -> None:
            for v in identify_variables(expr, include_fixed=False):
                self._var_objs.setdefault(id(v), v)
                refs.setdefault(id(v), []).append(id(ctx) if ctx is not None else None)

        for c in self._data(self.pm, pyo.Constraint):
            visit(c.body, None)
        for obj in self._data(self.pm, pyo.Objective, also_disjuncts=True):
            visit(obj.expr, None)
        for d in self._disjuncts_by_id().values():
            if not d.active:
                continue
            for c in self._data(d, pyo.Constraint):
                visit(c.body, d)
        for v_id, contexts in refs.items():
            v = self._var_objs[v_id]
            home = self._enclosing_disjunct(v)
            if home is None or id(home) not in self._parent_disjunct or not home.active:
                continue
            if v is getattr(home, "binary_indicator_var", None):
                continue
            ok = all(ctx is not None and id(home) in ancestry(self._by_id[ctx]) for ctx in contexts)
            if ok:
                self._local_home[v_id] = home

    def _disjuncts_by_id(self) -> dict[int, Any]:
        if not hasattr(self, "_by_id"):
            self._by_id: dict[int, Any] = {}
            stack = list(self._data(self.pm, self.PDisjunction))
            while stack:
                dj = stack.pop()
                for d in dj.disjuncts:
                    self._by_id[id(d)] = d
                    if d.active:
                        stack.extend(self._data(d, self.PDisjunction))
        return self._by_id

    # -- disjunct blocks ---------------------------------------------------------

    def _make_block(self, d: Any) -> None:
        block = self.m.make_disjunct(_name(d))
        self._dj[id(d)] = block
        self.out.disjuncts[d] = block
        # Until something references it, an unreferenced nested indicator's value
        # can be read off the tree selector (see _nested_indicator).
        self.out.booleans[d.indicator_var] = block.indicator
        if self._parent_disjunct.get(id(d)) is not None:
            self._nested[id(d.indicator_var)] = d
            self._nested[id(d.binary_indicator_var)] = d
            return
        self._bools[id(d.indicator_var)] = block.indicator
        self._vars[id(d.binary_indicator_var)] = block.indicator.variable
        self._var_objs_add(d.binary_indicator_var)

    def _nested_indicator(self, d: Any) -> BooleanVar:
        """Pyomo's indicator of the nested disjunct ``d`` as its own Boolean ``W``.

        Pyomo leaves a nested indicator free when the parent disjunct is False;
        discopt's tree selector ``y`` implies the parent's ``y_p``. Tying
        ``y = W AND y_p`` (three rows) gives every reference to ``W`` Pyomo's
        meaning while the tree keeps its own.
        """
        got = self._bools.get(id(d.indicator_var))
        if got is not None:
            return got
        w = cast(BooleanVar, self.m.boolean(_name(d.indicator_var)))
        y = self._dj[id(d)].indicator.variable
        y_p = self._dj[id(self._parent_disjunct[id(d)])].indicator.variable
        wv = w.variable
        self.m.subject_to(
            [y <= wv, y <= y_p, y >= wv + y_p - 1],
            name=f"{_name(d)}_indicator_link",
        )
        self._bools[id(d.indicator_var)] = w
        self.out.booleans[d.indicator_var] = w
        self._vars[id(d.binary_indicator_var)] = wv
        self._var_objs_add(d.binary_indicator_var)
        return w

    def _var_objs_add(self, v: Any) -> None:
        if not hasattr(self, "_var_objs"):
            self._var_objs = {}
        self._var_objs.setdefault(id(v), v)

    def _fill_block(self, d: Any) -> None:
        block = self._dj[id(d)]
        for v_id, home in self._local_home.items():
            if home is d:
                v = self._var_objs[v_id]
                self._vars[v_id] = self._declare(v, block)
        rows = []
        for c in self._data(d, self.pyo.Constraint):
            for rec in self._rows(c, d, block):
                rows.append(rec.row)
        if rows:
            block.subject_to(rows)

    def _emit_disjunction(self, dj: Any, parent: Any) -> None:
        children = []
        for d in dj.disjuncts:
            if d.active:
                for inner in self._data(d, self.PDisjunction):
                    self._emit_disjunction(inner, parent=d)
            else:
                # Pyomo reads a deactivated disjunct as "indicator fixed False" and
                # its transformations refuse one whose indicator says otherwise.
                if _fixed_indicator(d) != 0:
                    raise self._refuse(
                        "a deactivated Disjunct whose indicator_var is not fixed to "
                        "False (Pyomo's transformations refuse it too)",
                        d,
                    )
                self._pin(d, 0)
            children.append(self._dj[id(d)])
            if d.active:
                fixed = _fixed_indicator(d)
                if fixed is not None:
                    self._pin(d, fixed)
        semantics = DisjunctionSemantics.SELECT_ONE if dj.xor else DisjunctionSemantics.OR
        if parent is None:
            self.m.add_disjunction(children, name=_name(dj), semantics=semantics)
        else:
            self._dj[id(parent)].add_disjunction(children, name=_name(dj), semantics=semantics)
        self.out.disjunctions.append(dj)

    def _pin(self, d: Any, value: int) -> None:
        """Fix Pyomo's indicator of ``d``: a nested disjunct's is ``W``, not the
        tree selector (fixing that to True would force the parent True too)."""
        if self._parent_disjunct.get(id(d)) is not None:
            col = self._nested_indicator(d).variable
        else:
            col = self._dj[id(d)].indicator.variable
        self.m.subject_to(col == float(value), name=f"{_name(d)}_fixed")

    # -- variables ---------------------------------------------------------------

    def _var(self, v: Any) -> Any:
        if v.fixed:
            if v.value is None:
                raise self._refuse("a fixed variable without a value", v)
            return float(v.value)
        got = self._vars.get(id(v))
        if got is not None:
            return got
        if id(v) in self._nested:
            return self._nested_indicator(self._nested[id(v)]).variable
        if id(v) in self._local_home:
            # A local variable is declared when its block is filled, which happens
            # before any row that references it is converted.
            raise AssertionError(f"local variable {v.name!r} referenced before its block")
        col = self._declare(v, None)
        self._vars[id(v)] = col
        self._var_objs_add(v)
        return col

    def _declare(self, v: Any, block: Optional[Disjunct]) -> Variable:
        interval = v.domain.get_interval()
        if interval is None or interval[2] not in (0, 1):
            raise self._refuse(f"a variable whose domain {v.domain} is not an interval", v)
        lb, ub = v.bounds
        lb = None if lb is None else float(lb)
        ub = None if ub is None else float(ub)
        name = _name(v)
        if interval[2] == 0:
            kind, kwargs = "continuous", {"lb": lb, "ub": ub}
        elif lb is not None and ub is not None and lb == 0.0 and ub == 1.0:
            kind, kwargs = "binary", {}
        else:
            kind, kwargs = "integer", {"lb": lb, "ub": ub}
        if block is not None:
            self.out.local.add(v)
            return block.variable(name, kind=kind, **kwargs)
        if kind == "continuous":
            return self.m.continuous(name, lb=lb, ub=ub)
        if kind == "binary":
            return self.m.binary(name)
        return self.m.integer(name, lb=lb, ub=ub)

    # -- rows ------------------------------------------------------------------

    def _rows(self, c: Any, disjunct: Any, block: Optional[Disjunct] = None) -> list[RowRecord]:
        body = self._expr(c.body)
        lb, ub = c.lb, c.ub
        out: list[RowRecord] = []
        if isinstance(body, float):
            # Every variable in it is fixed: a constant, not a row. One that holds
            # is dropped, as Pyomo's .nl writer drops it. One that is violated
            # makes a disjunct holding it unselectable -- its tree selector is
            # pinned to 0, as Pyomo's transformations relax the row only when the
            # disjunct is off -- and a top-level model infeasible, which refuses.
            tol = 1e-8 * max(1.0, abs(body))
            lo_ok = lb is None or body >= float(lb) - tol
            hi_ok = ub is None or body <= float(ub) + tol
            if not (lo_ok and hi_ok) and block is not None:
                self.m.subject_to(block.indicator.variable == 0.0, name=f"{_name(c)}_infeasible")
                return out
            if not (lo_ok and hi_ok):
                raise self._refuse(
                    f"a constraint that is constant ({body!r}) after fixed values are "
                    f"substituted and violates its bounds [{lb}, {ub}]: the model is "
                    "infeasible as given",
                    c,
                )
            return out
        if c.equality:
            row = body == float(lb)
            out.append(RowRecord(c, "eq", float(lb), body, _row(row, c), disjunct))
        else:
            if lb is not None:
                row = body >= float(lb)
                out.append(RowRecord(c, "lb", float(lb), body, _row(row, c), disjunct))
            if ub is not None:
                row = body <= float(ub)
                out.append(RowRecord(c, "ub", float(ub), body, _row(row, c), disjunct))
        if not out:
            # An unbounded constraint holds everywhere; there is nothing to emit.
            return out
        for rec in out:
            rec.row.name = rec.row.name or _name(c)
        self.out.rows.extend(out)
        return out

    def _objective(self) -> None:
        objs = list(self._data(self.pm, self.pyo.Objective, also_disjuncts=True))
        if len(objs) > 1:
            raise self._refuse(f"{len(objs)} active objectives (discopt solves one)", objs[1])
        if not objs:
            self.m.minimize(Constant(0.0))
            return
        (obj,) = objs
        if self._enclosing_disjunct(obj) is not None:
            raise self._refuse("an Objective declared inside a Disjunct", obj)
        expr = self._expr(obj.expr)
        self.out.objective, self.out.objective_expr = obj, expr
        if obj.sense == self.pyo.maximize:
            self.m.maximize(expr)
        else:
            self.m.minimize(expr)

    # -- expressions ------------------------------------------------------------

    def _expr(self, e: Any) -> Any:
        """Convert a Pyomo numeric expression; a float when it has no free variable.

        Constants fold bottom-up: a node is evaluated by Pyomo only once every child
        has converted to a number. Pyomo's own ``is_fixed()`` is not used for this:
        it calls ``0 * x`` and ``x ** 0`` fixed, and evaluating them then fails on an
        ``x`` that has no value (GDPlib's ``batch_processing``).
        """
        ne = self.ne
        if isinstance(e, _NATIVE_NUMBERS):
            return float(e)
        if not getattr(e, "is_potentially_variable", lambda: False)():
            return self._value(e)
        if e.is_variable_type():
            return self._var(e)
        if e.is_named_expression_type():
            key = id(e)
            if key not in self._named:
                self._named[key] = self._expr(e.expr)
            return self._named[key]
        if not e.is_expression_type():
            raise self._refuse(f"an expression leaf of type {type(e).__name__}", None)
        if not isinstance(e, self._known):
            # Classified before its arguments are touched, so the refusal names
            # this node (``Expr_if``), not something inside it (its condition).
            raise self._refuse(f"a {type(e).__name__} expression", None)
        if isinstance(e, ne.UnaryFunctionExpression) and not isinstance(e, ne.AbsExpression):
            fn = _UNARY.get(e.getname())
            if fn is None:
                raise self._refuse(f"the function {e.getname()!r}", None)
        args = [self._expr(a) for a in e.args]
        if all(isinstance(a, float) for a in args):
            return self._value(e)
        if isinstance(e, (ne.SumExpression, ne.LinearExpression)):
            return args[0] if len(args) == 1 else dm.sum(args)
        if isinstance(e, (ne.MonomialTermExpression, ne.ProductExpression)):
            return args[0] * args[1]
        if isinstance(e, ne.DivisionExpression):
            return args[0] / args[1]
        if isinstance(e, ne.PowExpression):
            base, exponent = args
            if isinstance(base, float):
                # A constant base with a variable exponent, c ** f, is exp(f ln c):
                # the same function, in an atom every discopt evaluator and
                # relaxation supports (the interval evaluator has no c ** f rule).
                # Off c > 0 it is undefined for non-integer f.
                if base <= 0.0:
                    raise self._refuse(
                        f"a constant base {base!r} <= 0 raised to a variable power", None
                    )
                return dm.exp(exponent * math.log(base))
            return base**exponent
        if isinstance(e, ne.NegationExpression):
            return -args[0]
        if isinstance(e, ne.AbsExpression):
            return abs(args[0])
        if isinstance(e, ne.UnaryFunctionExpression):
            return _UNARY[e.getname()](args[0])
        if isinstance(e, ne.MaxExpression):
            return dm.maximum(*args)
        if isinstance(e, ne.MinExpression):
            return dm.minimum(*args)
        raise self._refuse(f"a {type(e).__name__} expression", None)

    def _value(self, e: Any) -> float:
        try:
            return float(self.pyo.value(e))
        except (ValueError, ZeroDivisionError, OverflowError) as err:
            raise self._refuse(f"the constant subexpression {e} ({err})", None) from err

    # -- logic -----------------------------------------------------------------

    def _logic(self, e: Any, lc: Any) -> "LogicalExpression | bool":
        """``e`` as a discopt logical formula, with constants folded away.

        discopt has no logical constant, so a Boolean literal -- or an empty
        ``lor()``/``land()``, which Pyomo builds from a comprehension over an
        empty set (``x.implies(lor(y[s] for s in S if s < first))``) -- is folded
        into the formula around it. The result is a formula, or ``True``/``False``
        when the whole formula is constant.
        """
        le = self.le
        if isinstance(e, _NATIVE_NUMBERS) or (
            not e.is_expression_type() and getattr(e, "is_constant", lambda: False)()
        ):
            value = self.pyo.value(e)
            if value not in (0, 1):
                raise self._refuse(f"the Boolean literal {value!r}", lc)
            return bool(value)
        if not e.is_expression_type():
            return self._boolean(e, lc)
        args = e.args
        if isinstance(e, le.NotExpression):
            a = self._logic(args[0], lc)
            return (not a) if isinstance(a, bool) else dm.lnot(a)
        if isinstance(e, (le.AndExpression, le.OrExpression)):
            is_and = isinstance(e, le.AndExpression)
            terms = []
            for x in args:
                v = self._logic(x, lc)
                if isinstance(v, bool):
                    if v is not is_and:  # False in a conjunction, True in a disjunction
                        return v
                    continue
                terms.append(v)
            if not terms:  # every operand was the identity, or there were none
                return is_and
            return dm.land(*terms) if is_and else dm.lor(*terms)
        if isinstance(e, le.XorExpression):
            if len(args) != 2:
                raise self._refuse(f"an {len(args)}-ary xor", lc)
            a, b = self._logic(args[0], lc), self._logic(args[1], lc)
            if isinstance(a, bool) and isinstance(b, bool):
                return a != b
            if isinstance(a, bool):
                return dm.lnot(b) if a else b  # type: ignore[arg-type]
            if isinstance(b, bool):
                return dm.lnot(a) if b else a
            return dm.xor(a, b)
        if isinstance(e, le.ImplicationExpression):
            a, b = self._logic(args[0], lc), self._logic(args[1], lc)
            if a is False or b is True:
                return True
            if a is True:
                return b
            if b is False:
                return dm.lnot(a)  # type: ignore[arg-type]
            return a.implies(b)  # type: ignore[union-attr]
        if isinstance(e, le.EquivalenceExpression):
            if len(args) != 2:
                raise self._refuse(f"an {len(args)}-ary equivalence", lc)
            a, b = self._logic(args[0], lc), self._logic(args[1], lc)
            if isinstance(a, bool) and isinstance(b, bool):
                return a == b
            if isinstance(a, bool):
                return b if a else dm.lnot(b)  # type: ignore[arg-type]
            if isinstance(b, bool):
                return a if b else dm.lnot(a)
            return a.equivalent_to(b)
        for cls, ctor, holds in (
            (le.ExactlyExpression, dm.exactly, lambda t, n, k: t == k if n == 0 else None),
            (le.AtMostExpression, dm.atmost, lambda t, n, k: t <= k if n == 0 else None),
            (le.AtLeastExpression, dm.atleast, lambda t, n, k: t >= k if n == 0 else None),
        ):
            if isinstance(e, cls):
                k = self.pyo.value(args[0])
                if float(k) != int(k):
                    raise self._refuse(f"a cardinality {k!r} that is not an integer", lc)
                vals = [self._logic(a, lc) for a in args[1:]]
                terms = [v for v in vals if not isinstance(v, bool)]
                # Each True operand counts toward k; a False one never does.
                k_left = int(k) - sum(v is True for v in vals)
                decided = holds(0, len(terms), k_left)
                if decided is not None:
                    return bool(decided)
                if k_left < 0:  # more True operands than allowed already
                    return cls is le.AtLeastExpression
                return ctor(k_left, *terms)
        raise self._refuse(f"a {type(e).__name__} logical expression", lc)

    def _boolean(self, b: Any, lc: Any) -> BooleanVar:
        got = self._bools.get(id(b))
        if got is not None:
            return got
        if id(b) in self._nested:
            return self._nested_indicator(self._nested[id(b)])
        if b.ctype is not self.PBooleanVar:
            raise self._refuse(f"a logical leaf of type {type(b).__name__}", lc)
        binary = b.get_associated_binary()
        if binary is not None and id(binary) in self._vars:
            # Its binary was already referenced algebraically: reuse that column.
            col = self._vars[id(binary)]
            boolean = BooleanVar(col)
        else:
            boolean = cast(BooleanVar, self.m.boolean(_name(b)))
            if binary is not None:
                self._vars[id(binary)] = boolean.variable
                self._var_objs_add(binary)
        if b.fixed:
            if b.value is None:
                raise self._refuse("a fixed BooleanVar without a value", b)
            # discopt has no logical constant: pin the column instead.
            self.m.subject_to(boolean.variable == float(bool(b.value)), name=f"{_name(b)}_fixed")
        self._bools[id(b)] = boolean
        self.out.booleans[b] = boolean
        return boolean


_NATIVE_NUMBERS = (int, float, np.number)


def _name(component: Any) -> str:
    return str(component.name)


def _row(row: Any, c: Any) -> Constraint:
    if not isinstance(row, Constraint):
        raise PyomoGDPConversionError(
            f"from_pyomo_gdp cannot convert constraint {_name(c)!r}: its body has "
            "no variable after fixed values were substituted, so it is not a row "
            f"(it evaluates to {row!r})"
        )
    return row


def _fixed_indicator(d: Any) -> Optional[int]:
    ind = d.indicator_var
    if ind.fixed:
        return int(bool(ind.value))
    b = d.binary_indicator_var
    if b.fixed:
        return int(round(float(b.value)))
    return None


# ---------------------------------------------------------------------------
# Evaluation parity
# ---------------------------------------------------------------------------


@dataclass
class ParityReport:
    points: int = 0
    row_checks: int = 0
    objective_checks: int = 0
    max_abs_error: float = 0.0


def check_evaluation_parity(
    conversion: PyomoGDPConversion,
    n_points: int = 5,
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> ParityReport:
    """Evaluate every converted row and the objective at random points, in Pyomo and
    in discopt, and raise on any disagreement.

    discopt.s side goes through :func:`discopt.mpec_report.evaluate_on_point_box` (the
    interval evaluator at a degenerate box, which refuses an inexact result), so it
    shares no code with the converter. Points are drawn uniformly inside each
    column's bounds (infinite bounds are clipped to ``[-10, 10]`` around the
    finite one); integer columns are rounded, and a fixed variable keeps its value.
    A point where Pyomo cannot evaluate a row (outside a function's domain) skips
    that row at that point and is counted in neither total. Returns the counts;
    the caller asserts they are non-zero.
    """
    from discopt.mpec_report import point_box

    rng = np.random.default_rng(seed)
    pyo = _pyomo()[0]
    report = ParityReport()
    cols = list(conversion.variables.items())
    saved = [(v, v.value) for v, _ in cols]
    try:
        for _ in range(n_points):
            point: dict[int, np.ndarray] = {}
            for v, col in cols:
                if v.fixed:
                    # A fixed indicator binary keeps its column (the pin row reads
                    # it) while rows substitute its value: draw it at that value.
                    point[id(col)] = np.array(float(v.value))
                    continue
                lo = -10.0 if v.lb is None else float(v.lb)
                hi = (lo + 20.0) if v.ub is None else float(v.ub)
                if v.lb is None and v.ub is not None:
                    lo = hi - 20.0
                val = rng.uniform(lo, hi)
                if not v.is_continuous():
                    val = float(np.clip(np.round(val), math.ceil(lo), math.floor(hi)))
                v.set_value(val, skip_validation=True)
                point[id(col)] = np.array(val)
            report.points += 1
            box = point_box(conversion.model, point)
            for rec in conversion.rows:
                try:
                    expected = float(pyo.value(rec.component.body))
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
                if not math.isfinite(expected):
                    continue
                got = _discopt_value(conversion, rec.row.body, box, f"row {rec.component.name}")
                want = {
                    "eq": expected - rec.bound,
                    "ub": expected - rec.bound,
                    "lb": rec.bound - expected,
                }[rec.kind]
                if rec.kind == "eq" and not math.isclose(got, want, rel_tol=rtol, abs_tol=atol):
                    want = -want
                _compare(got, want, rtol, atol, f"row {rec.component.name} ({rec.kind})", report)
                report.row_checks += 1
            if conversion.objective is not None:
                try:
                    expected = float(pyo.value(conversion.objective.expr))
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
                expr = conversion.objective_expr
                got = (
                    float(expr)
                    if isinstance(expr, float)
                    else _discopt_value(conversion, expr, box, "objective")
                )
                _compare(got, expected, rtol, atol, "objective", report)
                report.objective_checks += 1
    finally:
        for v, val in saved:
            v.set_value(val, skip_validation=True)
    return report


def _discopt_value(conversion: PyomoGDPConversion, expr: Any, box: dict, what: str) -> float:
    from discopt.mpec_report import evaluate_on_point_box

    try:
        return _scalar(evaluate_on_point_box(conversion.model, expr, box))
    except ValueError as err:
        # Pyomo evaluated it; discopt could not, so parity is not established.
        raise AssertionError(
            f"evaluation parity: {what}: discopt cannot evaluate it: {err}"
        ) from err


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 1:
        raise AssertionError(f"evaluation parity: a scalar row evaluated to shape {arr.shape}")
    return float(arr[0])


def _compare(got: float, want: float, rtol: float, atol: float, what: str, r: ParityReport) -> None:
    err = abs(got - want)
    r.max_abs_error = max(r.max_abs_error, err)
    if not math.isclose(got, want, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"evaluation parity: {what}: discopt {got!r} vs pyomo {want!r}")


__all__ = [
    "ParityReport",
    "PyomoGDPConversion",
    "PyomoGDPConversionError",
    "RowRecord",
    "check_evaluation_parity",
    "convert_pyomo_gdp",
    "from_pyomo_gdp",
]
