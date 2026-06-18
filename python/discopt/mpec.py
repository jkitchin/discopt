"""Complementarity / MPEC handler.

Mathematical Programs with Equilibrium Constraints (MPECs) carry complementarity
conditions of the form

    0 <= f(x)  ⊥  g(x) >= 0          (i.e. f >= 0, g >= 0, f·g = 0)

which are nonconvex and violate standard constraint qualifications, so a naive
NLP solve stalls. This module provides the two standard reformulations from the
SOTA rule inventory:

* **Scholtes regularization** — replace ``f·g = 0`` with ``f·g <= t`` and drive
  ``t -> 0`` through a homotopy of *local* NLP solves. Smooth, fast, and the
  workhorse for continuous MPECs.
* **SOS1 / disjunctive** — encode "at most one of ``f``, ``g`` is nonzero"
  exactly with a Special Ordered Set of type 1, solved by the global MINLP
  branch-and-bound. Exact, at the cost of discrete search.

Both build a standard discopt model, so no solver-core changes are required and
the global optimality machinery is reused unchanged.

Example
-------
>>> import discopt.modeling.core as dm
>>> from discopt.mpec import complementarity, solve_mpec
>>> m = dm.Model("toy")
>>> x = m.continuous("x", lb=0, ub=10)
>>> y = m.continuous("y", lb=0, ub=10)
>>> m.minimize((x - 1) ** 2 + (y - 1) ** 2)
>>> pairs = [complementarity(x, y)]          # 0 <= x ⊥ y >= 0
>>> res = solve_mpec(m, pairs, method="scholtes")
>>> round(res.objective, 3)
1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.modeling.core import Expression, Model, Variable

__all__ = [
    "Complementarity",
    "complementarity",
    "reformulate_gdp",
    "reformulate_scholtes",
    "reformulate_sos1",
    "tighten_complementarity_bounds",
    "solve_mpec",
]


@dataclass
class Complementarity:
    """A single complementarity condition ``0 <= f ⊥ g >= 0`` (``f·g = 0``).

    ``f`` and ``g`` are discopt expressions that are required to be nonnegative;
    at a feasible point at most one of them is nonzero.
    """

    f: Expression
    g: Expression
    name: Optional[str] = None


def complementarity(f: Expression, g: Expression, name: Optional[str] = None) -> Complementarity:
    """Construct a :class:`Complementarity` condition ``0 <= f ⊥ g >= 0``."""
    return Complementarity(f, g, name)


# ─────────────────────────── reformulations ───────────────────────────


def reformulate_scholtes(model: Model, pairs: list[Complementarity], t) -> None:
    """Add Scholtes constraints ``f >= 0``, ``g >= 0``, ``f·g <= t`` for each pair.

    ``t`` may be a float or a :class:`~discopt.modeling.core.Parameter`; using a
    parameter lets :func:`solve_mpec` drive the homotopy without rebuilding the
    model.
    """
    for i, p in enumerate(pairs):
        tag = p.name or f"compl{i}"
        model.subject_to(p.f >= 0, name=f"{tag}_f_nonneg")
        model.subject_to(p.g >= 0, name=f"{tag}_g_nonneg")
        model.subject_to(p.f * p.g <= t, name=f"{tag}_reg")


def reformulate_sos1(model: Model, pairs: list[Complementarity]) -> None:
    """Encode each complementarity exactly with an SOS1 set.

    Requires ``f >= 0`` and ``g >= 0`` and that at most one is nonzero. When
    ``f``/``g`` are plain variables they enter the SOS1 set directly; otherwise
    nonnegative auxiliary variables ``af == f``, ``ag == g`` are introduced.
    """
    for i, p in enumerate(pairs):
        tag = p.name or f"compl{i}"
        members: list[Variable] = []
        for side, expr in (("f", p.f), ("g", p.g)):
            if isinstance(expr, Variable) and expr.size == 1:
                model.subject_to(expr >= 0, name=f"{tag}_{side}_nonneg")
                members.append(expr)
            else:
                aux = model.continuous(f"{tag}_{side}_aux", lb=0.0, ub=np.inf)
                model.subject_to(aux == expr, name=f"{tag}_{side}_link")
                members.append(aux)
        model.sos1(members, name=f"{tag}_sos1")


def reformulate_gdp(model: Model, pairs: list[Complementarity]) -> None:
    """Encode each complementarity exactly as a disjunction ``(f==0) ∨ (g==0)``.

    With ``f, g >= 0`` this is equivalent to ``f·g == 0``. The disjunction is
    added via :meth:`Model.either_or` and lowered to a big-M model with a
    selector binary by the GDP pass at solve time, so branch-and-bound branches
    on the finite either/or choice and the integrality-aware FBBT can infer one
    side is zero when the other is bounded away from zero (the backward
    complementarity rule).

    Nonlinear or vector operands are lifted into a scalar auxiliary variable
    ``u == expr`` (with finite interval bounds) so the disjunct ``u == 0`` stays
    linear and big-M is exact; plain scalar variables enter the disjunction
    directly. Lifting also keeps the nonlinear part as an ordinary smooth
    equality handled natively by the NLP relaxation — avoiding the perspective
    of a nonlinear equality, whose big-M relaxation is bounded unreliably and
    whose hull form often cannot be linearized.
    """
    for i, p in enumerate(pairs):
        tag = p.name or f"compl{i}"
        fv = _gdp_operand(model, p.f, tag, "f")
        gv = _gdp_operand(model, p.g, tag, "g")
        model.subject_to(fv >= 0, name=f"{tag}_f_nonneg")
        model.subject_to(gv >= 0, name=f"{tag}_g_nonneg")
        model.either_or([[fv == 0], [gv == 0]], name=tag)


def _gdp_operand(model: Model, expr: Expression, tag: str, side: str) -> Expression:
    """Return a linear operand for a GDP complementarity disjunct.

    Linear expressions are used directly; nonlinear bodies are lifted to a
    scalar auxiliary variable with finite interval bounds.
    """
    from discopt._jax.gdp_reformulate import _is_linear

    if _is_linear(expr):
        return expr
    lo, hi = model._branch_bounds(expr, expr)
    model._aux_counter += 1
    u = model.continuous(f"_{tag}_{side}_{model._aux_counter}", lb=lo, ub=hi)
    model.subject_to(u == expr, name=f"{tag}_{side}_lift")
    return u


def tighten_complementarity_bounds(model: Model, pairs: list[Complementarity]) -> int:
    """Complementarity bound propagation for plain variable pairs.

    If one side of ``0 <= x ⊥ y >= 0`` is provably strictly positive
    (``lb > 0``), the other side must be zero. Applied only to single-variable
    sides, where the implication is sound and exact. Returns the number of
    variables fixed to zero.
    """
    fixed = 0
    for p in pairs:
        f_var = p.f if isinstance(p.f, Variable) and p.f.size == 1 else None
        g_var = p.g if isinstance(p.g, Variable) and p.g.size == 1 else None
        if f_var is not None and float(f_var.lb) > 0.0 and g_var is not None:
            g_var.ub = np.zeros_like(np.array(g_var.ub))
            g_var.lb = np.zeros_like(np.array(g_var.lb))
            fixed += 1
        elif g_var is not None and float(g_var.lb) > 0.0 and f_var is not None:
            f_var.ub = np.zeros_like(np.array(f_var.ub))
            f_var.lb = np.zeros_like(np.array(f_var.lb))
            fixed += 1
    return fixed


# ─────────────────────────────── solve ───────────────────────────────


def solve_mpec(
    model: Model,
    pairs: list[Complementarity],
    *,
    method: str = "scholtes",
    t0: float = 1.0,
    sigma: float = 0.1,
    t_min: float = 1e-8,
    max_iter: int = 16,
    x0: Optional[np.ndarray] = None,
    nlp_options: Optional[dict] = None,
    **solve_kwargs,
):
    """Solve an MPEC by reformulating its complementarity conditions.

    Parameters
    ----------
    model : Model
        Model carrying the objective and any ordinary constraints. It is
        augmented in place with the reformulation constraints.
    pairs : list[Complementarity]
        The complementarity conditions ``0 <= f ⊥ g >= 0``.
    method : {"scholtes", "sos1", "gdp"}
        ``"scholtes"`` runs a homotopy of *local* NLP solves with the
        regularization ``t`` shrinking ``t0 -> t0·sigma -> ...`` until ``t_min``.
        ``"sos1"`` encodes each pair as a Special Ordered Set of type 1 and
        ``"gdp"`` as the exact disjunction ``(f==0) ∨ (g==0)``; both build a
        standard model and call the global MINLP solver (``solve_kwargs``
        forwarded to :meth:`Model.solve`). ``"gdp"`` typically explores far
        fewer nodes than the smooth bilinear encoding.

    Returns
    -------
    The solver result. For ``"scholtes"`` this is the final NLP result
    (with ``.x`` and ``.objective``); for ``"sos1"`` / ``"gdp"`` it is the
    :meth:`Model.solve` result.
    """
    if method == "sos1":
        reformulate_sos1(model, pairs)
        return model.solve(**solve_kwargs)

    if method == "gdp":
        reformulate_gdp(model, pairs)
        return model.solve(**solve_kwargs)

    if method != "scholtes":
        raise ValueError(f"unknown MPEC method {method!r}; use 'scholtes', 'sos1', or 'gdp'")

    # Scholtes regularization homotopy via local NLP solves.
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.solvers.nlp_backend import get_nlp_solver

    t = model.parameter("_mpec_t", value=t0)
    reformulate_scholtes(model, pairs, t)

    backend = get_nlp_solver("auto")
    opts = dict(nlp_options) if nlp_options else {}
    opts.setdefault("print_level", 0)

    lb, ub = _flat_bounds(model)
    x_cur = (
        np.clip(np.asarray(x0, dtype=np.float64), lb, ub) if x0 is not None else _midpoint(lb, ub)
    )

    result = None
    tv = t0
    for _ in range(max_iter):
        t.value = np.asarray(tv, dtype=np.float64)
        # Rebuild the evaluator so the updated parameter value is compiled in.
        evaluator = NLPEvaluator(model)
        try:
            result = backend(evaluator, x_cur, options=opts)
        except BaseException:
            break
        if result.x is not None:
            x_cur = np.asarray(result.x, dtype=np.float64)
        if tv <= t_min:
            break
        tv = max(tv * sigma, t_min)
    return result


def _flat_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    lbs, ubs = [], []
    for v in model._variables:
        lbs.append(np.asarray(v.lb).flatten())
        ubs.append(np.asarray(v.ub).flatten())
    return np.concatenate(lbs), np.concatenate(ubs)


def _midpoint(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    lo = np.clip(lb, -1e6, 1e6)
    hi = np.clip(ub, -1e6, 1e6)
    return np.asarray(0.5 * (lo + hi), dtype=np.float64)
