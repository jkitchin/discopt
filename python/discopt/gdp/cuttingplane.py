"""``gdp.cuttingplane``: big-M plus hull-separation cuts (#1775 G2.7).

Pyomo's ``gdp.cuttingplane`` (Sawaya & Grossmann 2005). The output is the big-M
lowering of the model plus linear cuts that move its continuous relaxation toward
the hull relaxation's, which is tighter:

1. Solve the continuous relaxation of the big-M model (rBigM); call its point x*.
2. Project x* onto the continuous relaxation of the hull model (rHull): x̂ is the
   rHull point nearest x* (2-norm, or ∞-norm), over the model's own variables,
   found by a *local* NLP solve. If x* is (nearly) in rHull, no cut separates it:
   stop.
3. With ``g = x̂ - x*``, the cut is ``g·x >= v`` where ``v`` is a *certified* lower
   bound on ``min g·x`` over a set that contains every feasible point of the
   model, so the cut is valid whatever the accuracy of x̂: an inexact projection
   only weakens the cut, never cuts a feasible point. This is Pyomo's
   ``back_off_constraint_with_calculated_cut_violation``, with the back-off read
   off a certified bound rather than a float optimum. The set is rHull when every
   disjunct row is linear (exactly the Sawaya-Grossmann back-off; an LP when the
   global rows are linear too, a convex NLP when they are not), and the model
   itself otherwise (the dual bound of solving it with objective ``g·x``,
   a convex MINLP here, which is at least as tight as rHull's). The reason is
   speed: discopt does not prove the perspective rows of a nonlinear hull convex,
   so it solves rHull by spatial branch and bound (measured ~25 s per solve on a
   two-disc disjunction, against ~1.7 s for the model itself).
4. Add the cut to rBigM and to the output, and repeat (at most ``max_cuts``) while
   the cut separates x* by more than ``tol``.

The cut is only as strong as rHull is tight, and rHull is the convex hull of a
disjunction only when its disjuncts are convex: like Pyomo, the transformation is
**refused** for a disjunct row that is not linear or proven convex (a convex body
for ``<=``, a concave one for ``>=``; a nonlinear equality is refused), and for an
OR disjunction, which the hull lowering refuses.

Also refused:

- a model that declares a ``Parameter``. Each cut is computed at the parameter
  values of the call, but the output keeps the live parameters, so a later
  ``p.value = ...`` would leave a cut that removes feasible points of the new
  model while its solve still reports a certified optimum. The cuts depend on the
  feasible set and, through x*, on the objective, so a parameter anywhere is
  refused (unused ones included: refusing is simpler than proving one unused).
- a fast-API model (``add_linear_constraints``, ``add_linear_objective``, ...):
  cloning its lowering raises (#1854).

The cuts are equally frozen to the variable bounds of the call: changing a bound
of ``model`` (or of the output, which shares its variables) afterwards can make a
cut cut off points that have become feasible. Re-run the transformation instead.

A disjunction whose hull lowering falls back to big-M (``if_then``, and an
``add_disjunction`` nested in another) has the big-M relaxation as its "hull", so
no cut separates x* from it: such models get 0 cuts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from discopt.modeling.core import (
    Constraint,
    Model,
    Variable,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
)


@dataclass
class CuttingPlaneReport:
    """What :func:`cuttingplane` did: the cuts added (``cut_rows`` holds each as
    ``(g, v)``, the cut ``sum_n g[n] . x[n] >= v`` by variable name), the rBigM
    relaxation value before and after them, and why it stopped."""

    cuts: int = 0
    cut_rows: list[tuple[dict[str, np.ndarray], float]] = field(default_factory=list)
    relaxation_before: float | None = None
    relaxation_after: float | None = None
    stopped: str = ""
    separations: list[float] = field(default_factory=list)


def _require_convex_disjuncts(model: Model) -> bool:
    """Refuse a disjunct row that is not linear or proven convex; return whether
    every one is linear."""
    from discopt._relax.convexity.rules import classify_constraint
    from discopt._relax.gdp_reformulate import _is_linear

    cache: dict = {}
    all_linear = [True]

    def check(con: Constraint, where: str) -> None:
        if not _is_linear(con.body):
            all_linear[0] = False
        if not classify_constraint(con, model, cache):
            raise ValueError(
                f"gdp.cuttingplane: {where} ({con.name or str(con)[:60]}) is not linear or "
                "proven convex. The hull relaxation is the convex hull of a disjunction "
                "only when its disjuncts are convex, so its separating cuts are not the "
                "Sawaya-Grossmann cuts there; Pyomo's gdp.cuttingplane has the same "
                "restriction (#1775 G2.7). Use gdp.bigm or gdp.hull instead."
            )

    def walk(dc: _DisjunctiveConstraint) -> None:
        for k, d in enumerate(dc.disjuncts):
            for item in d:
                if isinstance(item, _DisjunctiveConstraint):
                    walk(item)
                else:
                    check(item, f"a row of disjunct {k} of {dc.name or '<anon>'!r}")

    for c in model._constraints:
        if isinstance(c, _DisjunctiveConstraint):
            walk(c)
        elif isinstance(c, _IndicatorConstraint):
            check(c.constraint, "an indicator row")
    return all_linear[0]


def _by_name(m: Model) -> dict[str, Variable]:
    return {v.name: v for v in m._variables}


def _linear(m: Model, names: list[str], coef: dict[str, np.ndarray]):
    import discopt.modeling as dm

    vars_ = _by_name(m)
    total: Any = None
    for n in names:
        g = coef[n]
        if not np.any(g):
            continue
        v = vars_[n]
        term = float(g) * v if v.shape == () else dm.sum(v * g.reshape(v.shape))
        total = term if total is None else total + term
    return total


def _point(r, names: list[str], shapes: dict[str, tuple]) -> dict[str, np.ndarray] | None:
    if r.x is None:
        return None
    out = {}
    for n in names:
        if n not in r.x:
            return None
        out[n] = np.asarray(r.x[n], dtype=np.float64).reshape(shapes[n])
    return out


def _local_point(m: Model, names: list[str], shapes: dict[str, tuple], time_limit: float):
    """A local NLP solve of ``m``; its point on ``names``, or ``None``."""
    from discopt.solvers.nlp_pounce import solve_nlp_from_model

    res = solve_nlp_from_model(m, options={"max_wall_time": float(time_limit)})
    if res.x is None or not np.all(np.isfinite(res.x)):
        return None
    flat = np.asarray(res.x, dtype=np.float64)
    out, off = {}, 0
    for v in m._variables:
        if v.name in shapes:
            out[v.name] = flat[off : off + v.size].reshape(shapes[v.name])
        off += v.size
    return out if len(out) == len(names) else None


def _quiet_solve(m: Model, time_limit: float):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return m.solve(time_limit=time_limit)


def cuttingplane(
    model: Model,
    *,
    max_cuts: int = 20,
    norm: Literal[2, "inf"] = 2,
    tol: float = 1e-4,
    time_limit: float = 30.0,
) -> tuple[Model, CuttingPlaneReport]:
    """The big-M lowering of ``model`` plus hull-separation cuts, and a report.

    ``model`` is not modified; the returned model shares its variables (as the
    other GDP lowerings do). ``time_limit`` bounds each of the three solves of an
    iteration (the separation NLP through its ``max_wall_time``). ``max_cuts=0``
    returns the big-M lowering without solving anything. The cuts are frozen to
    the variable bounds of the call; see the module docstring for that, the
    algorithm and the refusals.
    """
    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt.modeling import _relax_integrality

    if norm not in (2, "inf"):
        raise ValueError(f"gdp.cuttingplane: norm must be 2 or 'inf', got {norm!r}")
    if max_cuts < 0:
        raise ValueError(f"gdp.cuttingplane: max_cuts must be >= 0, got {max_cuts!r}")
    if model._parameters:
        raise ValueError(
            "gdp.cuttingplane: the model declares Parameters "
            f"({', '.join(p.name for p in model._parameters)}). The cuts would be "
            "computed at their current values, but the output keeps the live "
            "parameters, so changing one afterwards leaves cuts that can remove "
            "feasible points while the solve reports a certified optimum. Fix the "
            "values as constants first, or use gdp.bigm or gdp.hull."
        )
    if model._builder is not None:
        raise NotImplementedError(
            "gdp.cuttingplane: fast-API models (add_linear_constraints, "
            "add_linear_objective, ...) are not supported yet: cloning their GDP "
            "lowering raises (#1854). Build the model with subject_to/minimize, or "
            "use gdp.bigm or gdp.hull."
        )
    linear = _require_convex_disjuncts(model)

    out = reformulate_gdp(model, "big-m")
    report = CuttingPlaneReport()
    if out is model:  # nothing disjunctive: nothing to separate
        report.stopped = "the model has no GDP constraints"
        return model, report
    if model._objective is None:
        raise ValueError("gdp.cuttingplane: the model has no objective to relax over")
    if max_cuts == 0:
        report.stopped = "max_cuts=0 reached"
        return out, report

    names = [v.name for v in model._variables]
    shapes = {v.name: v.shape for v in model._variables}
    rbm = reformulate_gdp(model.clone(), "big-m")
    _relax_integrality.relax_integrality(rbm)
    rhull = reformulate_gdp(model.clone(), "hull")
    _relax_integrality.relax_integrality(rhull)

    import discopt.modeling as dm

    for i in range(max_cuts + 1):
        r = _quiet_solve(rbm, time_limit)
        if r.status != "optimal":
            report.stopped = f"the big-M relaxation ended {r.status}"
            break
        if report.relaxation_before is None:
            report.relaxation_before = float(r.objective)
        report.relaxation_after = float(r.objective)
        if i == max_cuts:
            report.stopped = f"max_cuts={max_cuts} reached"
            break
        xs = _point(r, names, shapes)
        if xs is None:
            report.stopped = "the big-M relaxation returned no point"
            break

        # 2. project x* onto rHull.
        sep = rhull.clone()
        sv = _by_name(sep)
        if norm == 2:
            dist = None
            for n in names:
                v = sv[n]
                term = (
                    (v - float(xs[n])) ** 2
                    if v.shape == ()
                    else dm.sum((v - xs[n].reshape(v.shape)) ** 2)
                )
                dist = term if dist is None else dist + term
            assert dist is not None  # names is non-empty: the model has variables
            sep.minimize(dist)
        else:
            t = sep.continuous("_gdp_cp_t", lb=0)
            for n in names:
                v = sv[n]
                c = float(xs[n]) if v.shape == () else xs[n].reshape(v.shape)
                sep.subject_to(v - c <= t)
                sep.subject_to(c - v <= t)
            sep.minimize(t)
        xh = _local_point(sep, names, shapes, time_limit)
        if xh is None:
            report.stopped = "the separation problem returned no point"
            break
        g = {n: xh[n] - xs[n] for n in names}
        gap = max((float(np.max(np.abs(g[n]))) for n in names), default=0.0)
        scale = 1.0 + max((float(np.max(np.abs(xs[n]))) for n in names), default=0.0)
        report.separations.append(gap)
        if gap <= tol * scale:
            report.stopped = "x* is in the hull relaxation (no separating cut)"
            break

        # 3. the back-off: a certified lower bound on min g.x over a set holding
        # every feasible point (rHull when the disjuncts are linear, else the model).
        back = rhull.clone() if linear else model.clone()
        back.minimize(_linear(back, names, g))
        rb = _quiet_solve(back, time_limit)
        if rb.bound is None or not rb.bound_valid or not np.isfinite(rb.bound):
            report.stopped = f"no certified bound for the cut (status {rb.status})"
            break
        v_lb = float(rb.bound)
        gx_star = sum(float(np.sum(g[n] * xs[n])) for n in names)
        if gx_star >= v_lb - tol * max(1.0, abs(v_lb)):
            report.stopped = "the certified cut does not separate x*"
            break

        # 4. add the cut g.x >= v_lb.
        for m in (rbm, out):
            body = _linear(m, names, g)
            m._constraints.append(
                Constraint(body=body - v_lb, sense=">=", rhs=0.0, name=f"_gdp_cp_cut_{i}")
            )
        report.cuts += 1
        report.cut_rows.append((g, v_lb))
    return out, report


def _cuttingplane_model(model: Model, **options: Any) -> Model:
    """The registry's functional entry point (``transformations``)."""
    return cuttingplane(model, **options)[0]


__all__ = ["CuttingPlaneReport", "cuttingplane"]
