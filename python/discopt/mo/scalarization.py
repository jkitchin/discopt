"""Scalarization methods for multi-objective optimization.

Each function takes a discopt ``Model`` and a list of objective ``Expression``
objects, sweeps a scalarization parameter, and returns a
:class:`~discopt.mo.pareto.ParetoFront`.

Scalarizers mutate the input model: they add auxiliary ``Parameter`` objects
(and, for AUGMECON / Tchebycheff, auxiliary ``Variable`` and constraint
entries). The model's original objective is restored before return; added
parameters / variables / constraints remain on the model as side effects.
Create a fresh ``Model`` if that is undesirable.
"""

from __future__ import annotations

import time
from itertools import count
from typing import Iterable, Optional

import numpy as np

from discopt.mo.pareto import ParetoFront, ParetoPoint
from discopt.mo.utils import (
    _as_senses,
    _x_to_var_dict,
    evaluate_expression,
    ideal_point,
    nadir_point,
)

# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────


def _unique_name(model, base: str) -> str:
    """Return a model-unique name of the form ``{base}_{i}``."""
    existing = {v.name for v in model._variables} | {p.name for p in model._parameters}
    for i in count():
        candidate = f"{base}_{i}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("unreachable")


def _default_names(
    objectives: list,
    objective_names: Optional[Iterable[str]],
) -> list[str]:
    if objective_names is None:
        return [f"f{i + 1}" for i in range(len(objectives))]
    out = list(objective_names)
    if len(out) != len(objectives):
        raise ValueError(f"objective_names has length {len(out)}, expected {len(objectives)}")
    return out


def _bi_weights(n: int) -> np.ndarray:
    """Uniform grid on the 1-simplex for bi-objective problems.

    Returns an ``(n, 2)`` array of convex weights ``[(1, 0), ..., (0, 1)]``.
    """
    if n < 2:
        raise ValueError("n_weights must be >= 2")
    alpha = np.linspace(0.0, 1.0, n)
    return np.column_stack([alpha, 1.0 - alpha])


def _simplex_lattice(k: int, n_approx: int) -> np.ndarray:
    """Das-Dennis simplex-lattice weights on the (k-1)-simplex.

    Produces ``C(H + k - 1, k - 1)`` grid points where ``H`` is chosen so the
    count is close to ``n_approx``. For ``k == 2`` this reduces to
    :func:`_bi_weights` with ``H + 1`` points.
    """
    if k == 2:
        return _bi_weights(n_approx)
    # Solve C(H + k - 1, k - 1) ~ n_approx for H.
    from math import comb

    h = 1
    while comb(h + k - 1, k - 1) < n_approx:
        h += 1

    def recurse(remaining: int, slots: int) -> list[list[int]]:
        if slots == 1:
            return [[remaining]]
        out: list[list[int]] = []
        for val in range(remaining + 1):
            for tail in recurse(remaining - val, slots - 1):
                out.append([val] + tail)
        return out

    grid = np.array(recurse(h, k), dtype=np.float64) / float(h)
    return grid


def _collect_objectives_at_x(
    objectives: list,
    model,
    x_dict: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate all component objectives at a solution."""
    return np.array(
        [evaluate_expression(e, model, x_dict) for e in objectives],
        dtype=np.float64,
    )


# ─────────────────────────────────────────────────────────────
# Weighted sum
# ─────────────────────────────────────────────────────────────


def weighted_sum(
    model,
    objectives: list,
    *,
    senses: Optional[Iterable[str]] = None,
    objective_names: Optional[Iterable[str]] = None,
    weights: Optional[np.ndarray] = None,
    n_weights: int = 21,
    normalize: bool = True,
    warm_start: bool = True,
    filter: bool = True,
    ideal: Optional[np.ndarray] = None,
    nadir: Optional[np.ndarray] = None,
    anchors: Optional[list[dict[str, np.ndarray]]] = None,
    **solve_kwargs,
) -> ParetoFront:
    r"""Weighted-sum scalarization sweep.

    Minimizes ``sum_i w_i * g_i(x)`` across a grid of weights, where
    ``g_i = (f_i - z*_i) / (z^N_i - z*_i)`` when ``normalize=True`` (default)
    or ``g_i = f_i`` otherwise. All maximize objectives are negated
    internally; returned objective values are in the original senses.

    .. note::

       Weighted sum is **complete** only for convex Pareto fronts. On
       nonconvex fronts it cannot recover concave pieces regardless of the
       weight grid; use :func:`epsilon_constraint` or
       :func:`weighted_tchebycheff` in that case.

    Parameters
    ----------
    model : discopt.modeling.Model
        Discopt model whose objective is replaced for each scalarized solve.
    objectives : list of Expression
        Objective expressions, one per criterion.
    senses : iterable of {"min", "max"}, optional
        Per-objective sense; defaults to all-``"min"``.
    objective_names : iterable of str, optional
        Display names attached to the resulting :class:`ParetoFront`.
    weights : numpy.ndarray, optional
        ``(n, k)`` array of convex weights. If omitted, a uniform simplex-
        lattice grid of approximately ``n_weights`` points is used.
    n_weights : int, default 21
        Target grid size (exact for ``k == 2``, approximate otherwise).
    normalize : bool, default True
        Scale objectives to ``[0, 1]`` using ideal/nadir before weighting.
    warm_start : bool, default True
        Re-use the previous solve's primal as the next initial point.
    filter : bool, default True
        Strip weakly dominated points from the returned front.
    ideal, nadir : numpy.ndarray, optional
        Pre-computed reference points; re-used if provided, else computed
        via :func:`~discopt.mo.utils.ideal_point` /
        :func:`~discopt.mo.utils.nadir_point`.
    anchors : list of dict, optional
        Pre-computed anchor solutions (from a prior ``ideal_point`` call).
    \*\*solve_kwargs : dict
        Forwarded to :meth:`discopt.modeling.Model.solve`.
    """
    senses_list = _as_senses(senses, len(objectives))
    names = _default_names(objectives, objective_names)
    saved_obj = model._objective

    try:
        if ideal is None or anchors is None:
            ideal_arr, anchors = ideal_point(
                model,
                objectives,
                senses=senses_list,
                warm_start=warm_start,
                **solve_kwargs,
            )
        else:
            ideal_arr = np.asarray(ideal, dtype=np.float64)

        if normalize:
            if nadir is None:
                nadir_arr = nadir_point(model, objectives, anchors, senses=senses_list)
            else:
                nadir_arr = np.asarray(nadir, dtype=np.float64)
        else:
            nadir_arr = None

        if weights is None:
            weights = _simplex_lattice(len(objectives), n_weights)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.ndim == 1:
                weights = weights[None, :]

        # Build sense signs for the internal min-convention expression.
        signs = np.array([1.0 if s == "min" else -1.0 for s in senses_list], dtype=np.float64)
        # Normalization spans.
        if normalize:
            span = signs * (nadir_arr - ideal_arr)
            span = np.where(np.abs(span) < 1e-12, 1.0, span)
        else:
            span = np.ones(len(objectives), dtype=np.float64)

        points: list[ParetoPoint] = []
        last_x: Optional[dict[str, np.ndarray]] = None

        for w in weights:
            # Build scalarized expression: sum_i w_i * signs_i * (f_i - ideal_i) / span_i
            terms = []
            for i, expr in enumerate(objectives):
                coef = float(w[i] * signs[i] / span[i])
                if coef == 0.0:
                    continue
                terms.append(coef * (expr - float(ideal_arr[i])))
            if not terms:
                # Degenerate all-zero weight: skip.
                continue
            scalarized = terms[0]
            for t in terms[1:]:
                scalarized = scalarized + t
            model.minimize(scalarized)

            kwargs = dict(solve_kwargs)
            if warm_start and last_x is not None and "initial_solution" not in kwargs:
                kwargs["initial_solution"] = _x_to_var_dict(model, last_x)

            t0 = time.perf_counter()
            result = model.solve(**kwargs)
            wall = time.perf_counter() - t0

            if result.x is None:
                continue  # skip infeasible / failed
            obj_vec = _collect_objectives_at_x(objectives, model, result.x)
            points.append(
                ParetoPoint(
                    x={k: np.asarray(v).copy() for k, v in result.x.items()},
                    objectives=obj_vec,
                    status=result.status,
                    wall_time=wall,
                    scalarization_params={"weights": w.tolist()},
                )
            )
            last_x = result.x
    finally:
        model._objective = saved_obj

    front = ParetoFront(
        points=points,
        method="weighted_sum",
        objective_names=names,
        senses=senses_list,
        ideal=ideal_arr,
        nadir=nadir_arr,
    )
    return front.filtered() if filter else front


# ─────────────────────────────────────────────────────────────
# Epsilon-constraint (AUGMECON2)
# ─────────────────────────────────────────────────────────────


def epsilon_constraint(
    model,
    objectives: list,
    *,
    primary: int = 0,
    senses: Optional[Iterable[str]] = None,
    objective_names: Optional[Iterable[str]] = None,
    n_points: int = 21,
    augmented: bool = True,
    delta: float = 1e-3,
    warm_start: bool = True,
    filter: bool = True,
    ideal: Optional[np.ndarray] = None,
    nadir: Optional[np.ndarray] = None,
    anchors: Optional[list[dict[str, np.ndarray]]] = None,
    slack_ub: float = 1e6,
    **solve_kwargs,
) -> ParetoFront:
    """Epsilon-constraint sweep with optional AUGMECON2 augmentation.

    Minimizes the *primary* objective subject to
    ``f_i <= epsilon_i`` (for minimize-sense non-primary objectives;
    inequality is flipped for maximize-sense). When ``augmented=True``
    (default), adds non-negative slack variables and a small penalty
    ``-delta * sum(s_i / r_i)`` on them so every returned point is strictly
    Pareto-optimal [Mavrotas 2009].

    Epsilon grids are uniform over ``[ideal_i, nadir_i]`` of the non-primary
    objectives, using ``n_points`` divisions per axis (total
    ``n_points ** (k - 1)`` subproblems for ``k`` objectives). For ``k = 2``,
    this is simply ``n_points`` solves.

    Parameters follow the same conventions as :func:`weighted_sum`.
    """
    senses_list = _as_senses(senses, len(objectives))
    names = _default_names(objectives, objective_names)
    k = len(objectives)
    if k < 2:
        raise ValueError("Need at least 2 objectives")
    if not 0 <= primary < k:
        raise ValueError(f"primary={primary} not in range [0, {k})")

    saved_obj = model._objective
    saved_n_cons = len(model._constraints)

    try:
        if ideal is None or anchors is None:
            ideal_arr, anchors = ideal_point(
                model,
                objectives,
                senses=senses_list,
                warm_start=warm_start,
                **solve_kwargs,
            )
        else:
            ideal_arr = np.asarray(ideal, dtype=np.float64)
        if nadir is None:
            nadir_arr = nadir_point(model, objectives, anchors, senses=senses_list)
        else:
            nadir_arr = np.asarray(nadir, dtype=np.float64)

        non_primary = [i for i in range(k) if i != primary]
        # Parameters for each non-primary epsilon.
        eps_params = []
        for i in non_primary:
            name = _unique_name(model, f"_mo_eps_{i}")
            eps_params.append(model.parameter(name, value=float(nadir_arr[i])))

        # Slack variables for AUGMECON2.
        slacks = []
        if augmented:
            for i in non_primary:
                name = _unique_name(model, f"_mo_slack_{i}")
                slacks.append(model.continuous(name, lb=0.0, ub=slack_ub))

        # Add constraints: f_i + (sign_i * s_i) == eps_i (for min) or >=
        # for max. Use "=="/">="/"<=" to encode; easier: reframe with
        # min-convention: sign_i * f_i + s_i <= sign_i * eps_i.
        signs = np.array([1.0 if s == "min" else -1.0 for s in senses_list], dtype=np.float64)
        for idx, i in enumerate(non_primary):
            lhs = signs[i] * objectives[i]
            if augmented:
                lhs = lhs + slacks[idx]
            # rhs = signs[i] * eps_param ; use Parameter on the right
            # Constraint: lhs <= signs[i] * eps_param
            constraint = lhs <= signs[i] * eps_params[idx]
            model.subject_to(constraint, name=f"_mo_eps_{i}_ub")

        # Build the scalarized objective.
        # Primary: sign_primary * f_primary  (min convention)
        scalarized = signs[primary] * objectives[primary]
        if augmented:
            # Penalty term: -delta * sum(s_i / r_i)
            r_vals = np.abs(nadir_arr - ideal_arr)
            r_vals = np.where(r_vals < 1e-12, 1.0, r_vals)
            penalty = None
            for idx, i in enumerate(non_primary):
                term = (float(delta) / float(r_vals[i])) * slacks[idx]
                penalty = term if penalty is None else penalty + term
            if penalty is not None:
                scalarized = scalarized - penalty

        # Epsilon grid. For each non-primary objective, a grid of n_points
        # values in [ideal_i, nadir_i] (in original sense).
        grids = []
        for i in non_primary:
            lo, hi = float(ideal_arr[i]), float(nadir_arr[i])
            if senses_list[i] == "max":
                lo, hi = hi, lo  # so that grid goes from worst to best
            grid = np.linspace(lo, hi, n_points)
            grids.append(grid)
        if len(non_primary) == 1:
            eps_grid = grids[0][:, None]
        else:
            mesh = np.meshgrid(*grids, indexing="ij")
            eps_grid = np.column_stack([m.reshape(-1) for m in mesh])

        points: list[ParetoPoint] = []
        last_x: Optional[dict[str, np.ndarray]] = None

        for eps_vec in eps_grid:
            for idx, i in enumerate(non_primary):
                eps_params[idx].value = np.asarray(float(eps_vec[idx]))
            model.minimize(scalarized)

            kwargs = dict(solve_kwargs)
            if warm_start and last_x is not None and "initial_solution" not in kwargs:
                kwargs["initial_solution"] = _x_to_var_dict(model, last_x)

            t0 = time.perf_counter()
            result = model.solve(**kwargs)
            wall = time.perf_counter() - t0

            if result.x is None:
                continue
            obj_vec = _collect_objectives_at_x(objectives, model, result.x)
            params_record = {
                "epsilon": {f"f{i + 1}": float(e) for i, e in zip(non_primary, eps_vec)}
            }
            points.append(
                ParetoPoint(
                    x={k: np.asarray(v).copy() for k, v in result.x.items()},
                    objectives=obj_vec,
                    status=result.status,
                    wall_time=wall,
                    scalarization_params=params_record,
                )
            )
            last_x = result.x
    finally:
        model._objective = saved_obj
        # Trim constraints we added (leaves aux vars and parameters in place).
        del model._constraints[saved_n_cons:]

    front = ParetoFront(
        points=points,
        method="augmecon2" if augmented else "epsilon_constraint",
        objective_names=names,
        senses=senses_list,
        ideal=ideal_arr,
        nadir=nadir_arr,
    )
    return front.filtered() if filter else front


# ─────────────────────────────────────────────────────────────
# Weighted Tchebycheff
# ─────────────────────────────────────────────────────────────


def weighted_tchebycheff(
    model,
    objectives: list,
    *,
    senses: Optional[Iterable[str]] = None,
    objective_names: Optional[Iterable[str]] = None,
    weights: Optional[np.ndarray] = None,
    n_weights: int = 21,
    rho: float = 1e-4,
    reference: Optional[np.ndarray] = None,
    warm_start: bool = True,
    filter: bool = True,
    ideal: Optional[np.ndarray] = None,
    nadir: Optional[np.ndarray] = None,
    anchors: Optional[list[dict[str, np.ndarray]]] = None,
    t_ub: float = 1e6,
    **solve_kwargs,
) -> ParetoFront:
    """Augmented weighted Tchebycheff sweep [Steuer & Choo 1983].

    For each weight ``w`` on the simplex, solves

    .. math::

       \\min \\; t + \\rho \\sum_i w_i (f_i(x) - z^*_i)
       \\quad \\text{s.t.} \\quad t \\ge w_i (f_i(x) - z^*_i)  \\;\\; \\forall i,

    where ``z*`` is the ideal point (or user-supplied reference). The small
    positive ``rho`` restores strict Pareto-optimality of the returned
    points. Unlike weighted sum, Tchebycheff is complete on **nonconvex**
    fronts.
    """
    senses_list = _as_senses(senses, len(objectives))
    names = _default_names(objectives, objective_names)
    k = len(objectives)
    saved_obj = model._objective
    saved_n_cons = len(model._constraints)

    try:
        if reference is not None:
            ideal_arr = np.asarray(reference, dtype=np.float64)
            if anchors is None:
                anchors = []
        elif ideal is None or anchors is None:
            ideal_arr, anchors = ideal_point(
                model,
                objectives,
                senses=senses_list,
                warm_start=warm_start,
                **solve_kwargs,
            )
        else:
            ideal_arr = np.asarray(ideal, dtype=np.float64)
        # Nadir for normalization inside the Tchebycheff form.
        if nadir is None and anchors:
            nadir_arr = nadir_point(model, objectives, anchors, senses=senses_list)
        elif nadir is not None:
            nadir_arr = np.asarray(nadir, dtype=np.float64)
        else:
            nadir_arr = None

        if weights is None:
            weights = _simplex_lattice(k, n_weights)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.ndim == 1:
                weights = weights[None, :]

        # Pre-normalize (f_i - ideal_i) / span_i in min-convention.
        signs = np.array([1.0 if s == "min" else -1.0 for s in senses_list], dtype=np.float64)
        if nadir_arr is not None:
            span = signs * (nadir_arr - ideal_arr)
            span = np.where(np.abs(span) < 1e-12, 1.0, span)
        else:
            span = np.ones(k, dtype=np.float64)

        # Weight parameters (so constraints can be built once).
        w_params = []
        for i in range(k):
            name = _unique_name(model, f"_mo_tchw_{i}")
            w_params.append(model.parameter(name, value=1.0 / k))

        # Auxiliary t variable.
        t_name = _unique_name(model, "_mo_tcht")
        t_var = model.continuous(t_name, lb=-t_ub, ub=t_ub)

        # Pre-build g_i = signs_i * (f_i - ideal_i) / span_i (min-convention,
        # scaled into [0, 1]-ish). Constraint: t >= w_i * g_i  i.e.
        # w_i * g_i - t <= 0.
        for i in range(k):
            g_expr = signs[i] * (objectives[i] - float(ideal_arr[i])) / float(span[i])
            model.subject_to(
                w_params[i] * g_expr - t_var <= 0,
                name=f"_mo_tcheb_max_{i}",
            )

        # Scalarized objective: t + rho * sum_i w_i * g_i
        scalarized = t_var
        for i in range(k):
            g_expr = signs[i] * (objectives[i] - float(ideal_arr[i])) / float(span[i])
            scalarized = scalarized + float(rho) * (w_params[i] * g_expr)

        points: list[ParetoPoint] = []
        last_x: Optional[dict[str, np.ndarray]] = None

        for w in weights:
            w_normed = w / max(w.sum(), 1e-12)
            for i in range(k):
                w_params[i].value = np.asarray(float(w_normed[i]))
            model.minimize(scalarized)

            kwargs = dict(solve_kwargs)
            if warm_start and last_x is not None and "initial_solution" not in kwargs:
                kwargs["initial_solution"] = _x_to_var_dict(model, last_x)

            t0 = time.perf_counter()
            result = model.solve(**kwargs)
            wall = time.perf_counter() - t0

            if result.x is None:
                continue
            obj_vec = _collect_objectives_at_x(objectives, model, result.x)
            points.append(
                ParetoPoint(
                    x={k: np.asarray(v).copy() for k, v in result.x.items()},
                    objectives=obj_vec,
                    status=result.status,
                    wall_time=wall,
                    scalarization_params={"weights": w_normed.tolist()},
                )
            )
            last_x = result.x
    finally:
        model._objective = saved_obj
        del model._constraints[saved_n_cons:]

    front = ParetoFront(
        points=points,
        method="weighted_tchebycheff",
        objective_names=names,
        senses=senses_list,
        ideal=ideal_arr,
        nadir=nadir_arr,
    )
    return front.filtered() if filter else front
