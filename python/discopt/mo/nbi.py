"""Normal Boundary Intersection (NBI) and Normalized Normal Constraint (NNC).

Geometric scalarizations that produce approximately uniform spacing of
Pareto-front samples, irrespective of front curvature
[Das & Dennis 1998; Messac et al. 2003]. Both methods build the CHIM
(convex hull of individual minima) simplex in objective space, generate
evenly-spaced points on it, and map each back to a Pareto point.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

import numpy as np

from discopt.mo.pareto import ParetoFront, ParetoPoint
from discopt.mo.scalarization import (
    _collect_objectives_at_x,
    _default_names,
    _simplex_lattice,
    _unique_name,
)
from discopt.mo.utils import (
    _as_senses,
    _x_to_var_dict,
    ideal_point,
    nadir_point,
)


def _payoff_matrix(
    model,
    objectives: list,
    anchors: list[dict[str, np.ndarray]],
    senses_list: list[str],
) -> np.ndarray:
    """Payoff matrix: ``Phi[i, j]`` = normalized ``g_j`` at anchor ``i``.

    Objectives are mapped into the minimization convention and normalized
    by the ideal/nadir range (columns are in ``[0, 1]``-ish).
    """
    from discopt.mo.utils import evaluate_expression

    k = len(objectives)
    raw = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            raw[i, j] = evaluate_expression(objectives[j], model, anchors[i])
    return raw


# ─────────────────────────────────────────────────────────────
# Normal Boundary Intersection (NBI)
# ─────────────────────────────────────────────────────────────


def normal_boundary_intersection(
    model,
    objectives: list,
    *,
    senses: Optional[Iterable[str]] = None,
    objective_names: Optional[Iterable[str]] = None,
    n_points: int = 21,
    warm_start: bool = True,
    filter: bool = True,
    ideal: Optional[np.ndarray] = None,
    nadir: Optional[np.ndarray] = None,
    anchors: Optional[list[dict[str, np.ndarray]]] = None,
    t_ub: float = 1e6,
    **solve_kwargs,
) -> ParetoFront:
    """NBI scalarization sweep [Das & Dennis 1998].

    Anchor points ``F(x^{(i)}*)`` are computed, the CHIM (convex hull of
    individual minima) built, and weights ``w`` on its simplex produce
    subproblems

    .. math::

       \\max_{x, t}\\; t \\quad \\text{s.t.} \\quad
       \\Phi w - t \\Phi \\mathbf{1} = g(x) - g^*,

    where ``g`` is the sense-normalized objective vector (min-form,
    ideal/nadir-scaled). Solutions lie on the Pareto boundary and are
    approximately uniformly spaced for convex fronts. A post-hoc dominance
    filter removes points that the quasi-normal ray intersects in a
    dominated region.
    """
    senses_list = _as_senses(senses, len(objectives))
    names = _default_names(objectives, objective_names)
    k = len(objectives)
    if k < 2:
        raise ValueError("Need at least 2 objectives")

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

        signs = np.array([1.0 if s == "min" else -1.0 for s in senses_list], dtype=np.float64)
        span = signs * (nadir_arr - ideal_arr)
        span = np.where(np.abs(span) < 1e-12, 1.0, span)

        # Payoff in normalized min-form. At the i-th anchor, g_i = 0 and the
        # other g_j are in [0, 1]-ish.
        raw_payoff = _payoff_matrix(model, objectives, anchors, senses_list)
        phi = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                phi[i, j] = signs[j] * (raw_payoff[i, j] - ideal_arr[j]) / span[j]
        # Quasi-normal direction: negative of column-sum (points inward from
        # the CHIM into the dominated region).
        n_hat = -phi.sum(axis=1)  # shape (k,)

        # Weight grid on simplex (convex combinations of anchors).
        weights = _simplex_lattice(k, n_points)

        # Auxiliary t variable (scalar).
        t_name = _unique_name(model, "_mo_nbi_t")
        t_var = model.continuous(t_name, lb=-t_ub, ub=t_ub)

        # Parameters for the CHIM target point b = phi @ w.
        b_params = [
            model.parameter(_unique_name(model, f"_mo_nbi_b_{j}"), value=0.0) for j in range(k)
        ]

        # Equality constraints: b_j + t * n_hat[j] == g_j(x)
        # i.e. g_j(x) - t * n_hat[j] - b_j == 0.
        for j in range(k):
            g_expr = signs[j] * (objectives[j] - float(ideal_arr[j])) / float(span[j])
            model.subject_to(
                g_expr - float(n_hat[j]) * t_var - b_params[j] == 0,
                name=f"_mo_nbi_eq_{j}",
            )

        # Objective: max t  →  minimize -t
        scalarized = -t_var
        model.minimize(scalarized)

        points: list[ParetoPoint] = []
        last_x: Optional[dict[str, np.ndarray]] = None

        for w in weights:
            w_normed = w / max(w.sum(), 1e-12)
            b_vec = phi.T @ w_normed  # shape (k,)
            for j in range(k):
                b_params[j].value = np.asarray(float(b_vec[j]))

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
                    scalarization_params={"w_chim": w_normed.tolist()},
                )
            )
            last_x = result.x
    finally:
        model._objective = saved_obj
        del model._constraints[saved_n_cons:]

    front = ParetoFront(
        points=points,
        method="nbi",
        objective_names=names,
        senses=senses_list,
        ideal=ideal_arr,
        nadir=nadir_arr,
    )
    return front.filtered() if filter else front


# ─────────────────────────────────────────────────────────────
# Normalized Normal Constraint (NNC)
# ─────────────────────────────────────────────────────────────


def normalized_normal_constraint(
    model,
    objectives: list,
    *,
    senses: Optional[Iterable[str]] = None,
    objective_names: Optional[Iterable[str]] = None,
    n_points: int = 21,
    warm_start: bool = True,
    filter: bool = True,
    ideal: Optional[np.ndarray] = None,
    nadir: Optional[np.ndarray] = None,
    anchors: Optional[list[dict[str, np.ndarray]]] = None,
    **solve_kwargs,
) -> ParetoFront:
    """NNC scalarization sweep [Messac et al. 2003].

    Generates evenly-spaced points ``X^p`` on the normalized utopia
    hyperplane (CHIM) and, for each, minimizes the last objective subject
    to normal-to-hyperplane inequality cuts that restrict the search to the
    Pareto-adjacent strip.

    Returned points are sense-correct; normalization is handled internally
    by the ideal/nadir range.
    """
    senses_list = _as_senses(senses, len(objectives))
    names = _default_names(objectives, objective_names)
    k = len(objectives)
    if k < 2:
        raise ValueError("Need at least 2 objectives")

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

        signs = np.array([1.0 if s == "min" else -1.0 for s in senses_list], dtype=np.float64)
        span = signs * (nadir_arr - ideal_arr)
        span = np.where(np.abs(span) < 1e-12, 1.0, span)

        # Normalized anchors in min-form (k anchor vectors of length k).
        raw_payoff = _payoff_matrix(model, objectives, anchors, senses_list)
        norm_anchors = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                norm_anchors[i, j] = signs[j] * (raw_payoff[i, j] - ideal_arr[j]) / span[j]

        # Normal directions: N_i = anchor_last - anchor_i for i = 0..k-2
        # (vectors from each other anchor toward the last anchor), following
        # Messac et al. 2003 eq. (11).
        last = norm_anchors[-1]
        normals = np.array([last - norm_anchors[i] for i in range(k - 1)])

        # Weight grid on the (k-1)-simplex.
        weights = _simplex_lattice(k, n_points)

        # CHIM target X^p parameters (one per objective).
        xp_params = [
            model.parameter(_unique_name(model, f"_mo_nnc_xp_{j}"), value=0.0) for j in range(k)
        ]

        # Per-iteration CHIM inequality constraints:
        # N_i . (g(x) - X^p) <= 0   for i = 0..k-2
        # Compose g_j(x) once.
        g_exprs = [
            signs[j] * (objectives[j] - float(ideal_arr[j])) / float(span[j]) for j in range(k)
        ]
        for i in range(k - 1):
            lhs = float(normals[i, 0]) * (g_exprs[0] - xp_params[0])
            for j in range(1, k):
                lhs = lhs + float(normals[i, j]) * (g_exprs[j] - xp_params[j])
            model.subject_to(lhs <= 0, name=f"_mo_nnc_normal_{i}")

        # Objective: minimize g_{k-1}(x).
        model.minimize(g_exprs[-1])

        points: list[ParetoPoint] = []
        last_x: Optional[dict[str, np.ndarray]] = None

        for w in weights:
            w_normed = w / max(w.sum(), 1e-12)
            xp = (w_normed[:, None] * norm_anchors).sum(axis=0)  # shape (k,)
            for j in range(k):
                xp_params[j].value = np.asarray(float(xp[j]))

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
                    scalarization_params={"w_chim": w_normed.tolist()},
                )
            )
            last_x = result.x
    finally:
        model._objective = saved_obj
        del model._constraints[saved_n_cons:]

    front = ParetoFront(
        points=points,
        method="nnc",
        objective_names=names,
        senses=senses_list,
        ideal=ideal_arr,
        nadir=nadir_arr,
    )
    return front.filtered() if filter else front
