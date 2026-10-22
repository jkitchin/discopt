"""SCIP whole-model backend via PySCIPOpt (issue #1784).

``Model.solve(solver="scip")`` hands the *whole* model to SCIP and bypasses
discopt's own solvers, the way ``solver="gurobi"`` does for the quadratic
classes. Unlike the Gurobi route it is not limited to LP..MIQCQP: SCIP is a
global MINLP solver, so any model whose expressions have an exact SCIP spelling
is accepted.

The model is translated **in memory** into PySCIPOpt expressions -- no ``.nl``
file and no subprocess. The walk reuses the exporters' shared array expansion
(:mod:`discopt.export._arrays`) and row iteration (:mod:`discopt.export._common`),
so vectorised constraints, matrix products, fast-API builder rows and builder
objectives reach SCIP as the same scalar rows every other writer sees.

Every node either has an exact SCIP spelling or is refused by name
(:class:`ScipUnsupportedError`); nothing is approximated. In particular:

* ``atan``/``asin``/``acos``/the inverse hyperbolics/``erf``/``sign``/``min``/
  ``max``/``atan2``/``dm.custom`` have no SCIP expression handler exposed through
  PySCIPOpt and are refused.
* ``a ** b`` with a non-constant exponent is rewritten as ``exp(b * log(a))``
  -- discopt's own definition -- only when ``a > 0`` is
  proven over the declared box; otherwise it is refused, because the rewrite
  would silently drop points where ``a <= 0``.
* ``xlogx`` (``entropy``) is ``x*log(x)`` only where ``x > 0``; at ``x = 0`` discopt
  defines it as 0 while SCIP's ``log`` is undefined, so it is translated only when
  the argument is proven positive.

* A fractional power or ``sqrt`` of a base not proven nonnegative gets the row
  ``base >= 0``. discopt evaluates both to NaN on a negative base, so no point
  there is feasible, and the row is exact; without it SCIP returns such points.

SCIP's optimality claim is never certified (``solver._solve_scip``): its bound is
reported as information under ``bound_source="external_solver"``, because SCIP's
presolve was measured certifying a bound past the true optimum of MINLPLib
``ex1252a`` on a faithfully translated model.

SCIP's answer is not taken on trust: the returned point is re-checked against
the model with discopt's own evaluator (constraint rows, bounds, integrality) and
its objective is recomputed there, before ``solver.py`` reports it. A point that
sits on SCIP's own numerical limits -- within :data:`_SINGULAR_DISTANCE` of a
singularity the declared box admits (a denominator, the base of a negative power,
``cos`` under ``tan``, a ``log`` argument), or with a value past SCIP's
``numerics/hugeval`` -- is reported as such (``raw["numerics"]``) and is never
certified: on ``min 1/x`` over ``[-2, 2]`` SCIP stops at ``x = -1e-9`` and calls
``-1e9`` optimal, but the problem is unbounded.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from discopt.constants import CONSTRAINT_INF
from discopt.export import _arrays
from discopt.export._common import builder_objective, iter_all_rows
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    ObjectiveSense,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

# "No bound" on a variable: ``Model.continuous`` declares its default as
# +/-9.999e19, just UNDER the LP layer's ``CONSTRAINT_INF`` (1e20), so a test at
# 1e20 hands SCIP a finite 1e20-wide box and an unbounded model comes back
# "feasible" at x = 9.999e19. Same threshold as ``mpec._UNBOUNDED_SENTINEL``:
# anything at or past the declared default is "no bound", anything below it is
# the finite bound the user wrote.
_UNBOUNDED_SENTINEL = 0.9999 * CONSTRAINT_INF


#: A returned point this close to a singularity the box admits is SCIP stopping at
#: its feasibility tolerance in front of a pole, not an optimum (``min 1/x`` on
#: ``[-2, 2]`` returns ``x = -1e-9``). Distances are absolute, in the units of the
#: singular expression (denominator, ``cos`` under ``tan``, ``log`` argument).
_SINGULAR_DISTANCE = 1e-6


class ScipUnsupportedError(NotImplementedError):
    """The model contains something with no exact SCIP spelling."""


def _load_scip():
    try:
        import pyscipopt
    except ImportError as exc:
        raise ImportError(
            "pyscipopt is required for solver='scip'. Install it with "
            "`pip install discopt[scip]` (or `pip install pyscipopt`); the wheel "
            "bundles SCIP itself."
        ) from exc
    return pyscipopt


def scip_version() -> str:
    ps = _load_scip()
    return f"SCIP {ps.Model().version()} (PySCIPOpt {ps.__version__})"


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float))


def _enclosure(expr: Expression) -> tuple[float, float]:
    """Sound interval enclosure of ``expr`` over the declared box.

    ``(-inf, inf)`` on anything the enclosure does not understand, so every test
    below errs towards a refusal, a domain row or a singular site -- never towards
    a wrong model or an unearned certificate.
    """
    from discopt._relax.binary_multilinear_reform import _interval_of_dag

    lo, hi = _interval_of_dag(expr)
    return float(lo), float(hi)


def _proven_positive(expr: Expression) -> bool:
    """True when ``expr > 0`` holds on the whole declared box."""
    lo, _ = _enclosure(expr)
    return bool(np.isfinite(lo) and lo > 0.0)


def _tan_pole_possible(arg: Expression) -> bool:
    """Whether ``tan(arg)`` can meet a pole ``pi/2 + k*pi`` on the declared box."""
    lo, hi = _enclosure(arg)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return True
    k = math.ceil((lo - math.pi / 2) / math.pi)
    return math.pi / 2 + k * math.pi <= hi


class _ScipBuilder:
    """Build a PySCIPOpt model from a discopt :class:`Model`.

    ``cols[k]`` is the SCIP variable for flat slot ``k`` in the standard discopt
    layout (``model._variables`` order, ``v.size`` slots each), which is the
    layout ``solver._unpack_solution`` reads back.
    """

    def __init__(self, model: Model, ps):
        self.model = model
        self.ps = ps
        self.scip = ps.Model(model.name or "discopt")
        self.cols: list = []
        self._offset: dict[int, int] = {}
        self.epigraph = None
        #: ``(description, scip_expr)`` for every singularity the declared box
        #: admits; evaluated at SCIP's point by :func:`solve_scip`.
        self.singular_sites: list[tuple[str, Any]] = []
        self._n_domain_rows = 0

    # ---- model ------------------------------------------------------------
    def build(self):
        if self.model._objective is None:
            raise ValueError("solver='scip' needs an objective; call minimize() or maximize().")
        for var in self.model._variables:
            self._add_variable(var)
        rows, builder_rows = iter_all_rows(self.model)
        for i, con in enumerate(rows):
            bodies = _arrays.scalarize_body(con.body, self._resolve)
            for j, body in enumerate(bodies):
                name = con.name or f"c{i}"
                self._add_row(self.expr(body), con.sense, float(con.rhs), f"{name}_{j}")
        for row in builder_rows:
            lhs = self.ps.quicksum(a * self.cols[k] for k, a in row.coeffs.items())
            self._add_row(lhs, row.sense, row.rhs, row.name)
        self._set_objective()
        return self.scip

    def _add_variable(self, var: Variable) -> None:
        self._offset[id(var)] = len(self.cols)
        size = int(var.size)
        lb = np.broadcast_to(np.asarray(var.lb, dtype=np.float64), var.shape).ravel()
        ub = np.broadcast_to(np.asarray(var.ub, dtype=np.float64), var.shape).ravel()
        vtype = {VarType.BINARY: "B", VarType.INTEGER: "I"}.get(var.var_type, "C")
        for k in range(size):
            lo = None if lb[k] <= -_UNBOUNDED_SENTINEL else float(lb[k])
            hi = None if ub[k] >= _UNBOUNDED_SENTINEL else float(ub[k])
            self.cols.append(self.scip.addVar(name=f"{var.name}[{k}]", vtype=vtype, lb=lo, ub=hi))

    def _add_row(self, lhs, sense: str, rhs: float, name: str) -> None:
        if _is_number(lhs):
            # A row with no variable left in it (every term folded to a constant).
            # SCIP's addCons wants an expression; handing it the constant row keeps
            # the judgement on SCIP's own feasibility tolerance instead of a second
            # one here, and an infeasible constant row is SCIP's to prove.
            lhs = self.ps.Expr() + float(lhs)
        if sense == "<=":
            self.scip.addCons(lhs <= rhs, name=name)
        elif sense == ">=":
            self.scip.addCons(lhs >= rhs, name=name)
        elif sense == "==":
            self.scip.addCons(lhs == rhs, name=name)
        else:
            raise ValueError(f"Unknown constraint sense {sense!r}")

    def _set_objective(self) -> None:
        ps = self.ps
        bobj = builder_objective(self.model)
        if bobj is not None:
            lin, quad, const, sense = bobj
            obj = ps.quicksum(a * self.cols[k] for k, a in lin.items()) + const
            if quad:
                obj = obj + ps.quicksum(
                    q * self.cols[i] * self.cols[j] for (i, j), q in quad.items()
                )
            maximize = sense == "maximize"
        else:
            o = self.model._objective
            assert o is not None
            obj = self.expr(_arrays.scalarize_objective(o.expression, self._resolve))
            maximize = o.sense == ObjectiveSense.MAXIMIZE
        if _is_number(obj):
            obj = ps.Expr() + float(obj)
        if not isinstance(obj, ps.Expr) or obj.degree() > 1:
            # SCIP's objective must be linear: move a nonlinear one into an
            # epigraph row. The dual bound on `t` is a dual bound on the objective.
            t = self.scip.addVar(name="_discopt_objective", lb=None, ub=None)
            self.scip.addCons(obj >= t if maximize else obj <= t, name="_discopt_objective_def")
            self.epigraph = t
            obj = t
        self.scip.setObjective(obj, "maximize" if maximize else "minimize")

    def _nonnegative_domain(self, base: Expression, base_expr) -> None:
        """Add ``base >= 0`` unless the box already proves it (see module docstring)."""
        lo, _ = _enclosure(base)
        if np.isfinite(lo) and lo >= 0.0:
            return
        self._n_domain_rows += 1
        self.scip.addCons(base_expr >= 0.0, name=f"_discopt_domain_{self._n_domain_rows}")

    def _singular_if(self, possible: bool, what: str, node: Expression, site_expr) -> None:
        if possible and not _is_number(site_expr):
            self.singular_sites.append((f"{what} in {node!r}", site_expr))

    def _resolve(self, ie: IndexExpression) -> Optional[int]:
        """Flat slot of ``x[i]``/``x[i, j]`` within its variable, else ``None``."""
        if not isinstance(ie.base, Variable):
            return None
        shape = ie.base.shape
        idx = ie.index if isinstance(ie.index, tuple) else (ie.index,)
        if shape in ((), (1,)) and idx in ((0,),):
            return 0
        if len(idx) != len(shape) or not all(isinstance(i, (int, np.integer)) for i in idx):
            return None
        if not all(0 <= int(i) < d for i, d in zip(idx, shape)):
            return None
        return int(np.ravel_multi_index(tuple(int(i) for i in idx), shape))

    # ---- expressions ------------------------------------------------------
    def expr(self, node: Expression):
        """Translate a scalar discopt expression into a PySCIPOpt expression."""
        if isinstance(node, BinaryOp) and node.op in ("+", "-"):
            # The scalarizer folds `dm.sum(x)` into a left-deep chain of depth n.
            # Walk the left spine iteratively (no RecursionError on long sums) and
            # sum once with quicksum (in-place; `a + b` copies, which is O(n^2)).
            terms = []
            cur: Expression = node
            while isinstance(cur, BinaryOp) and cur.op in ("+", "-"):
                right = self.expr(cur.right)
                terms.append(right if cur.op == "+" else -1.0 * right)
                cur = cur.left
            terms.append(self.expr(cur))
            consts = [t for t in terms if _is_number(t)]
            exprs = [t for t in terms if not _is_number(t)]
            if not exprs:
                return float(sum(consts))
            out = self.ps.quicksum(reversed(exprs))
            return out + float(sum(consts)) if consts else out
        return self._expr(node)

    def _expr(self, n: Expression):
        if isinstance(n, Constant):
            if np.asarray(n.value).ndim != 0:
                raise ValueError(
                    f"Array constant of shape {np.asarray(n.value).shape} in a scalar position."
                )
            return float(n.value)
        if isinstance(n, Parameter):
            v = np.asarray(n.value, dtype=np.float64)
            if v.size != 1:
                raise ValueError(f"Array parameter {n.name!r} used without indexing.")
            return float(v.reshape(()))
        if isinstance(n, Variable):
            if n.shape not in ((), (1,)):
                raise ValueError(f"Array variable {n.name!r} in a scalar position.")
            return self.cols[self._offset[id(n)]]
        if isinstance(n, IndexExpression):
            if isinstance(n.base, Parameter):
                return float(np.asarray(n.base.value, dtype=np.float64)[n.index])
            k = self._resolve(n)
            if k is None or id(n.base) not in self._offset:
                raise ValueError(f"Cannot resolve the index expression {n!r}.")
            return self.cols[self._offset[id(n.base)] + k]
        if isinstance(n, UnaryOp):
            a = self.expr(n.operand)
            if n.op == "neg":
                return -a
            if n.op == "abs":
                return abs(a)
            raise ScipUnsupportedError(f"Unary operator {n.op!r} has no SCIP spelling.")
        if isinstance(n, BinaryOp):
            return self._binary(n)
        if isinstance(n, FunctionCall):
            return self._function(n)
        if isinstance(n, (SumExpression, SumOverExpression, MatMulExpression)):
            # A scalar-valued array node the body-level expansion left intact.
            arr = _arrays.scalarize(n)
            if arr.ndim != 0:
                raise ValueError(f"{type(n).__name__} of shape {arr.shape} in a scalar position.")
            return self.expr(_arrays.elem(arr, ()))
        raise ScipUnsupportedError(
            f"Cannot translate {type(n).__name__} to SCIP (an opaque dm.custom body has "
            "no algebraic form SCIP can relax)."
        )

    def _binary(self, n: BinaryOp):
        ps = self.ps
        a, b = self.expr(n.left), self.expr(n.right)
        if n.op == "*":
            return a * b
        if n.op == "/":
            if _is_number(b) and b == 0.0:
                raise ValueError(f"Division by the constant 0 in {n!r}.")
            lo, hi = _enclosure(n.right)
            self._singular_if(lo <= 0.0 <= hi, "the denominator", n, b)
            return a / b
        if n.op == "**":
            if _is_number(b):
                p = float(b)
                fractional = p != math.floor(p)
                if _is_number(a):
                    if fractional and a < 0:
                        raise ValueError(
                            f"{n!r} folds to the constant {a!r} ** {p!r}, which is not a "
                            "real number (discopt evaluates it to NaN)."
                        )
                    return float(a) ** p
                if fractional:
                    self._nonnegative_domain(n.left, a)
                if p < 0:
                    lo, hi = _enclosure(n.left)
                    self._singular_if(lo <= 0.0 <= hi, "the base of a negative power", n, a)
                return a**p
            # Non-constant exponent: exp(b*log(a)) is discopt's own definition,
            # and is the same function only where a > 0.
            if _is_number(a) and a > 0:
                return ps.exp(math.log(a) * b)
            if _proven_positive(n.left):
                return ps.exp(b * ps.log(a))
            raise ScipUnsupportedError(
                f"Variable exponent in {n!r}: SCIP has no x**y operator, and the exact "
                "rewrite exp(y*log(x)) needs the base proven positive over the "
                "declared bounds, which it is not. Tighten the base's bounds."
            )
        raise ScipUnsupportedError(f"Binary operator {n.op!r} has no SCIP spelling.")

    # Exact rewrites in terms of SCIP's exp/log/sin/cos/sqrt/abs.
    _UNARY: dict[str, Callable[[Any, Any], Any]] = {
        "exp": lambda ps, x: ps.exp(x),
        "log": lambda ps, x: ps.log(x),
        "sqrt": lambda ps, x: ps.sqrt(x),
        "sin": lambda ps, x: ps.sin(x),
        "cos": lambda ps, x: ps.cos(x),
        "abs": lambda ps, x: abs(x),
        "log2": lambda ps, x: ps.log(x) * (1.0 / math.log(2.0)),
        "log10": lambda ps, x: ps.log(x) * (1.0 / math.log(10.0)),
        "log1p": lambda ps, x: ps.log(1.0 + x),
        "tan": lambda ps, x: ps.sin(x) / ps.cos(x),
        "sinh": lambda ps, x: 0.5 * (ps.exp(x) - ps.exp(-1.0 * x)),
        "cosh": lambda ps, x: 0.5 * (ps.exp(x) + ps.exp(-1.0 * x)),
        "tanh": lambda ps, x: 1.0 - 2.0 / (ps.exp(2.0 * x) + 1.0),
        "sigmoid": lambda ps, x: 1.0 / (1.0 + ps.exp(-1.0 * x)),
        "softplus": lambda ps, x: ps.log(1.0 + ps.exp(x)),
    }
    # Same functions applied to a plain float (a folded constant subtree).
    _UNARY_NUM: dict[str, Callable[[float], float]] = {
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "abs": abs,
        "log2": math.log2,
        "log10": math.log10,
        "log1p": math.log1p,
        "tan": math.tan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
        "softplus": lambda x: math.log1p(math.exp(x)),
    }

    def _function(self, n: FunctionCall):
        fn = n.func_name.lower()
        if fn.startswith("norm") or fn == "prod":
            return self.expr(_arrays.scalarize_reduction(n))
        if fn == "entropy":
            (arg,) = n.args
            if not _proven_positive(arg):
                raise ScipUnsupportedError(
                    "xlogx(x) is defined as 0 at x = 0, but SCIP's log is undefined "
                    "there, so x*log(x) is only the same function where x > 0. The "
                    "argument is not proven positive over the declared bounds; "
                    "give it a positive lower bound."
                )
            x = self.expr(arg)
            return x * math.log(x) if _is_number(x) else x * self.ps.log(x)
        if fn in self._UNARY and len(n.args) == 1:
            (arg,) = n.args
            x = self.expr(arg)
            if _is_number(x):
                return float(self._UNARY_NUM[fn](float(x)))
            if fn == "sqrt":
                self._nonnegative_domain(arg, x)
            elif fn in ("log", "log2", "log10"):
                self._singular_if(_enclosure(arg)[0] <= 0.0, "the log argument", n, x)
            elif fn == "log1p":
                self._singular_if(_enclosure(arg)[0] <= -1.0, "the log1p argument", n, 1.0 + x)
            elif fn == "tan":
                self._singular_if(_tan_pole_possible(arg), "cos under tan", n, self.ps.cos(x))
            return self._UNARY[fn](self.ps, x)
        raise ScipUnsupportedError(
            f"Function {n.func_name!r} has no exact SCIP expression (SCIP supports "
            "exp, log, sqrt, sin, cos, abs and powers; discopt also rewrites log2, "
            "log10, log1p, tan, sinh, cosh, tanh, sigmoid, softplus, xlogx, norm and "
            "prod exactly). Refusing rather than approximating it."
        )


def build_scip_model(model: Model):
    """Translate ``model`` into ``(pyscipopt.Model, cols, epigraph_var, singular_sites)``.

    ``model`` must already be algebraic (GDP/indicator/SOS/logical relations
    lowered); :func:`discopt.solver` does that before calling here.
    """
    ps = _load_scip()
    b = _ScipBuilder(model, ps)
    scip = b.build()
    return scip, b.cols, b.epigraph, b.singular_sites


# SCIP status string -> (discopt status with a point, discopt status without one),
# for the exits that carry no proof. "feasible" claims a point, so an exit that
# has none never maps to it.
_LIMIT_STATUS = {
    "timelimit": ("time_limit", "time_limit"),
    "nodelimit": ("node_limit", "node_limit"),
    "totalnodelimit": ("node_limit", "node_limit"),
    "stallnodelimit": ("node_limit", "node_limit"),
    "restartlimit": ("iteration_limit", "iteration_limit"),
    "memlimit": ("feasible", "error"),
    "userinterrupt": ("feasible", "error"),
    "sollimit": ("feasible", "unknown"),
    "bestsollimit": ("feasible", "unknown"),
    "primallimit": ("feasible", "unknown"),
    "duallimit": ("feasible", "unknown"),
}


def solve_scip(
    model: Model,
    *,
    time_limit: Optional[float],
    gap_tolerance: float,
    abs_gap_tolerance: Optional[float],
    initial_point: Optional[np.ndarray],
    options: Optional[dict],
) -> dict[str, Any]:
    """Run SCIP on ``model`` and return its raw outcome.

    Returns a dict with ``scip_status``, ``x`` (flat, discopt layout, or
    ``None``), ``scip_objective``, ``dual_bound`` (model sense), ``gap``,
    ``node_count``, ``version`` and ``numerics`` -- the reasons, if any, that the
    answer rests on SCIP's numerical limits rather than on the problem (see the
    module docstring). Interpreting and verifying it is the caller's job
    (``solver._solve_scip``).

    SCIP's log is hidden; ``options={"display/verblevel": 4}`` shows it, since the
    options are applied after the default is set.
    """
    scip, cols, _, sites = build_scip_model(model)
    scip.hideOutput()
    if time_limit is not None and np.isfinite(time_limit):
        scip.setParam("limits/time", max(float(time_limit), 0.0))
    scip.setParam("limits/gap", float(gap_tolerance))
    if abs_gap_tolerance is not None:
        scip.setParam("limits/absgap", float(abs_gap_tolerance))
    for key, value in (options or {}).items():
        # A misspelt parameter raises KeyError from SCIP -- surfaced, not ignored.
        scip.setParam(key, value)

    if initial_point is not None:
        x0 = np.asarray(initial_point, dtype=np.float64).ravel()
        if x0.shape[0] != len(cols):
            raise ValueError(
                f"initial_point has {x0.shape[0]} entries; the model has {len(cols)} "
                "variable slots."
            )
        sol = scip.createPartialSol()
        for var, val in zip(cols, x0):
            if np.isfinite(val):
                scip.setSolVal(sol, var, float(val))
        scip.addSol(sol)

    scip.optimize()
    status = scip.getStatus()
    x = None
    scip_obj = None
    numerics: list[str] = []
    hugeval = float(scip.getParam("numerics/hugeval"))
    if scip.getNSols() > 0:
        best = scip.getBestSol()
        x = np.array([scip.getSolVal(best, v) for v in cols], dtype=np.float64)
        scip_obj = float(scip.getObjVal())
        for what, site in sites:
            val = float(scip.getSolVal(best, site))
            if abs(val) <= _SINGULAR_DISTANCE:
                numerics.append(f"the point sits at {what} = {val:.3g}, at a singularity")
        if np.any(np.abs(x) >= hugeval):
            k = int(np.argmax(np.abs(x)))
            numerics.append(f"variable slot {k} = {x[k]:.3g} is past numerics/hugeval")
        if abs(scip_obj) >= hugeval:
            numerics.append(f"the objective {scip_obj:.3g} is past numerics/hugeval")
    dual = float(scip.getDualbound())
    if abs(dual) < _UNBOUNDED_SENTINEL and abs(dual) >= hugeval:
        numerics.append(f"the dual bound {dual:.3g} is past numerics/hugeval")
    return {
        "scip_status": status,
        "x": x,
        "scip_objective": scip_obj,
        "dual_bound": dual if abs(dual) < _UNBOUNDED_SENTINEL else None,
        "gap": float(scip.getGap()) if x is not None else None,
        "node_count": int(scip.getNTotalNodes()),
        "n_scip_vars": int(scip.getNVars()),
        "n_scip_conss": int(scip.getNConss()),
        "version": f"SCIP {scip.version()}",
        "numerics": numerics,
        "singular_sites": [what for what, _ in sites],
    }


def limit_status(scip_status: str, has_point: bool) -> Optional[str]:
    """discopt status for a SCIP limit exit, or ``None`` if not a limit exit."""
    pair = _LIMIT_STATUS.get(scip_status)
    if pair is None:
        return None
    return pair[0] if has_point else pair[1]
