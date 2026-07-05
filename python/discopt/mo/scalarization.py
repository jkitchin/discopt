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
import warnings
from itertools import count
from typing import Iterable, Optional

import numpy as np

from discopt.mo.pareto import ParetoFront, ParetoPoint
from discopt.mo.utils import (
    ObjectiveEvaluator,
    _as_senses,
    _x_to_var_dict,
    evaluate_expression,
    ideal_point,
    lexicographic_payoff,
    nadir_from_payoff,
    nadir_point,
)

# Warn when a sweep's subproblem count exceeds this, since each subproblem is a
# full global solve with its own ``time_limit`` (MO7).
_GRID_WARN_THRESHOLD = 200

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


def _warn_large_grid(n_subproblems: int, method: str) -> None:
    """Warn when a sweep will issue a large number of global solves (MO7)."""
    if n_subproblems > _GRID_WARN_THRESHOLD:
        warnings.warn(
            f"{method}: sweep will issue up to {n_subproblems} global solves, "
            f"each with the full per-solve time_limit. Consider fewer grid "
            f"points, a coarser n_points/n_weights, or a total_time_limit "
            f"budget.",
            stacklevel=3,
        )


class _TimeBudget:
    """Wall-clock budget for a sweep; ``expired()`` once ``total`` s elapse (MO7).

    ``total=None`` disables the budget (``expired()`` always False). Sweeps that
    honor it return the partial front collected so far and tag the front method
    ``"<method>/truncated"`` so callers can tell the sweep did not complete.
    """

    def __init__(self, total: Optional[float]) -> None:
        self.total = None if total is None else float(total)
        self._t0 = time.perf_counter()
        self.truncated = False

    def expired(self) -> bool:
        if self.total is None:
            return False
        if time.perf_counter() - self._t0 >= self.total:
            self.truncated = True
            return True
        return False


def _tag(method: str, budget: "_TimeBudget") -> str:
    """Append ``/truncated`` to *method* when the budget was hit (MO7)."""
    return f"{method}/truncated" if budget.truncated else method


def _add_tracked(model, constraint, name, tracked: list) -> None:
    """Add a constraint, recording it by identity for later exact removal.

    :meth:`Model.subject_to` returns ``None``, so scalarizers cannot get a
    handle to the constraint they add. This helper appends the constraint via
    ``subject_to`` and also stores the object in ``tracked`` so
    :func:`_remove_tracked` can excise exactly the rows this scalarizer added —
    rather than a positional ``del model._constraints[saved_n:]`` slice that
    would also drop any rows a callback / reformulation appended during the
    sweep (MO8).
    """
    model.subject_to(constraint, name=name)
    tracked.append(constraint)


def _remove_tracked(model, tracked: list) -> None:
    """Remove exactly the constraints in ``tracked`` (by identity)."""
    ids = {id(c) for c in tracked}
    model._constraints[:] = [c for c in model._constraints if id(c) not in ids]


def _collect_objectives_at_x(
    objectives: list,
    model,
    x_dict: dict[str, np.ndarray],
    evaluator: Optional[ObjectiveEvaluator] = None,
) -> np.ndarray:
    """Evaluate all component objectives at a solution.

    Pass a shared :class:`~discopt.mo.utils.ObjectiveEvaluator` to reuse the
    objectives' compiled callables across a sweep instead of recompiling each
    objective on every point (MO5). If omitted, falls back to a per-call
    recompile via :func:`~discopt.mo.utils.evaluate_expression`.
    """
    if evaluator is not None:
        return evaluator.vector_at(x_dict)
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
    total_time_limit: Optional[float] = None,
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
        :func:`~discopt.mo.utils.nadir_point`. A supplied ``ideal`` is honored
        without recomputation when anchors are not otherwise required — i.e.
        when ``normalize=False`` or a ``nadir`` is also supplied (MO6). If
        anchors *are* required (``normalize=True`` and no ``nadir``) but not
        provided, they are computed and the supplied ``ideal`` is still honored.
    anchors : list of dict, optional
        Pre-computed anchor solutions (from a prior ``ideal_point`` call).
        Only needed to estimate the nadir; omit them when passing ``ideal``
        with ``normalize=False`` or with an explicit ``nadir``.
    total_time_limit : float, optional
        Wall-clock budget in seconds for the *whole* sweep. When exceeded, the
        sweep stops and returns the partial front collected so far, with the
        front method tagged ``"weighted_sum/truncated"`` (MO7). ``None``
        (default) means no overall budget — note each subproblem still has its
        own per-solve ``time_limit``.
    \*\*solve_kwargs : dict
        Forwarded to :meth:`discopt.modeling.Model.solve`.
    """
    senses_list = _as_senses(senses, len(objectives))
    names = _default_names(objectives, objective_names)
    saved_obj = model._objective
    budget = _TimeBudget(total_time_limit)
    # Compile each objective once and reuse across the payoff table and every
    # accepted point (MO5). weighted_sum adds no parameters, so the parameter
    # index captured here stays valid for the whole sweep.
    evaluator = ObjectiveEvaluator(model, objectives)

    try:
        # Anchors are only needed to estimate the nadir (when normalizing and no
        # nadir was supplied). A caller who passes ``ideal`` but no ``anchors``
        # and does not need a computed nadir should NOT trigger k anchor solves
        # that discard their ideal (MO6).
        need_anchors = normalize and nadir is None
        if ideal is None or (anchors is None and need_anchors):
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
                # need_anchors was True in this branch, so anchors were computed.
                assert anchors is not None
                nadir_arr = nadir_point(
                    model, objectives, anchors, senses=senses_list, evaluator=evaluator
                )
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
        _warn_large_grid(len(weights), "weighted_sum")

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
            if budget.expired():
                break
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
            obj_vec = _collect_objectives_at_x(objectives, model, result.x, evaluator)
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
        method=_tag("weighted_sum", budget),
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
    payoff: str = "lexicographic",
    bypass: bool = True,
    ideal: Optional[np.ndarray] = None,
    nadir: Optional[np.ndarray] = None,
    anchors: Optional[list[dict[str, np.ndarray]]] = None,
    slack_ub: float = 1e6,
    total_time_limit: Optional[float] = None,
    **solve_kwargs,
) -> ParetoFront:
    """Epsilon-constraint sweep — AUGMECON2 [Mavrotas & Florios 2013].

    Minimizes the *primary* objective subject to
    ``f_i <= epsilon_i`` (for minimize-sense non-primary objectives;
    inequality is flipped for maximize-sense). When ``augmented=True``
    (default), adds non-negative slack variables and a small penalty
    ``-delta * sum(s_i / r_i)`` on them so every returned point is strictly
    Pareto-optimal [Mavrotas 2009].

    The two features that make this AUGMECON2 rather than plain AUGMECON are
    both on by default:

    * **Lexicographic payoff table** (``payoff="lexicographic"``): the
      ideal/nadir range is computed from a lexicographically-optimized payoff
      table (:func:`~discopt.mo.utils.lexicographic_payoff`), whose rows are
      Pareto-optimal, avoiding the inflated / truncated ε-grid a plain
      single-objective payoff produces under alternative optima
      [Miettinen 1999]. Pass ``payoff="simple"`` for the original
      :func:`~discopt.mo.utils.ideal_point` / ``nadir_point`` behavior.
    * **Bypass / jump acceleration** (``bypass=True``): after each solve, the
      innermost objective's optimal slack is used to skip the ε-grid cells that
      would provably reproduce the same solution, sweeping the innermost axis
      from best to worst. This is exact — it only skips subproblems whose
      optimum is already known — and yields fewer solves on flat front regions.
      Requires ``augmented=True``. Pass ``bypass=False`` for the exhaustive
      grid.

    Epsilon grids are uniform over ``[ideal_i, nadir_i]`` of the non-primary
    objectives, using ``n_points`` divisions per axis (up to
    ``n_points ** (k - 1)`` subproblems for ``k`` objectives; fewer with
    ``bypass``). For ``k = 2``, this is at most ``n_points`` solves.

    Parameters follow the same conventions as :func:`weighted_sum`, including
    ``total_time_limit`` (overall wall-clock budget; on expiry the sweep returns
    the partial front tagged ``".../truncated"``). A warning is emitted when the
    grid exceeds ~200 subproblems (MO7).
    """
    senses_list = _as_senses(senses, len(objectives))
    names = _default_names(objectives, objective_names)
    k = len(objectives)
    if k < 2:
        raise ValueError("Need at least 2 objectives")
    if not 0 <= primary < k:
        raise ValueError(f"primary={primary} not in range [0, {k})")
    if payoff not in ("lexicographic", "simple"):
        raise ValueError(f"payoff must be 'lexicographic' or 'simple', got {payoff!r}")
    use_bypass = bool(bypass) and augmented
    _warn_large_grid(n_points ** (k - 1), "epsilon_constraint")

    saved_obj = model._objective
    tracked_cons: list = []
    budget = _TimeBudget(total_time_limit)

    try:
        if ideal is not None and nadir is not None:
            ideal_arr = np.asarray(ideal, dtype=np.float64)
            nadir_arr = np.asarray(nadir, dtype=np.float64)
        elif payoff == "lexicographic":
            ideal_arr, payoff_mat, anchors = lexicographic_payoff(
                model,
                objectives,
                senses=senses_list,
                warm_start=warm_start,
                **solve_kwargs,
            )
            if ideal is not None:
                ideal_arr = np.asarray(ideal, dtype=np.float64)
            nadir_arr = (
                np.asarray(nadir, dtype=np.float64)
                if nadir is not None
                else nadir_from_payoff(payoff_mat, senses=senses_list)
            )
        else:
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
            nadir_arr = (
                np.asarray(nadir, dtype=np.float64)
                if nadir is not None
                else nadir_point(model, objectives, anchors, senses=senses_list)
            )

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
            _add_tracked(model, constraint, f"_mo_eps_{i}_ub", tracked_cons)

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

        # Epsilon grid per non-primary axis, oriented from WORST to BEST value
        # (nadir -> ideal in min-convention). AUGMECON2 sweeps the innermost
        # axis in this direction so that a subproblem's slack tells us how many
        # of the *next* (tighter) ε cells reproduce the same solution and can be
        # skipped ("bypass"). Grid values are stored in min-convention units
        # (sign_i * f_i) so step sizes are positive and comparable to slacks.
        min_grids = []  # min-convention grid values, worst -> best
        for i in non_primary:
            lo_min = float(signs[i] * ideal_arr[i])
            hi_min = float(signs[i] * nadir_arr[i])
            # In min-convention, ideal <= nadir; worst is the larger value.
            worst, best = max(lo_min, hi_min), min(lo_min, hi_min)
            min_grids.append(np.linspace(worst, best, n_points))

        # Step size of the innermost axis (last non-primary), in min-convention.
        inner_axis = len(non_primary) - 1
        inner_grid = min_grids[inner_axis]
        inner_step = abs(float(inner_grid[0] - inner_grid[1])) if n_points > 1 else 0.0

        # Compile each objective once for reuse across every accepted point
        # (MO5). Built after all auxiliary parameters (eps, slacks) are added so
        # the parameter index is final; objectives do not reference the aux
        # parameters, so their compiled callables are unaffected by them.
        evaluator = ObjectiveEvaluator(model, objectives)

        points: list[ParetoPoint] = []
        last_x: Optional[dict[str, np.ndarray]] = None

        def _run_solve(min_eps_vec):
            """Set ε parameters from min-convention values and solve.

            Returns ``(result, wall, obj_vec, slack_inner)`` or
            ``(None, wall, None, None)`` on infeasibility.
            """
            nonlocal last_x
            for idx in range(len(non_primary)):
                i = non_primary[idx]
                # Convert min-convention grid value back to original sense.
                eps_params[idx].value = np.asarray(float(signs[i] * min_eps_vec[idx]))
            model.minimize(scalarized)
            kwargs = dict(solve_kwargs)
            if warm_start and last_x is not None and "initial_solution" not in kwargs:
                kwargs["initial_solution"] = _x_to_var_dict(model, last_x)
            t0 = time.perf_counter()
            result = model.solve(**kwargs)
            wall = time.perf_counter() - t0
            if result.x is None:
                return None, wall, None, None
            obj_vec = _collect_objectives_at_x(objectives, model, result.x, evaluator)
            # Innermost slack (min-convention): eps_inner - sign*f_inner >= 0.
            i_inner = non_primary[inner_axis]
            slack_inner = float(min_eps_vec[inner_axis] - signs[i_inner] * obj_vec[i_inner])
            last_x = result.x
            return result, wall, obj_vec, slack_inner

        def _record(result, wall, obj_vec, min_eps_vec):
            params_record = {
                "epsilon": {
                    f"f{i + 1}": float(signs[i] * min_eps_vec[idx])
                    for idx, i in enumerate(non_primary)
                }
            }
            points.append(
                ParetoPoint(
                    x={kk: np.asarray(v).copy() for kk, v in result.x.items()},
                    objectives=obj_vec,
                    status=result.status,
                    wall_time=wall,
                    scalarization_params=params_record,
                )
            )

        # Iterate the outer axes exhaustively; sweep the innermost axis with the
        # bypass jump when enabled.
        from itertools import product

        outer_axes = list(range(len(non_primary) - 1))
        outer_ranges = [range(len(min_grids[a])) for a in outer_axes]
        for outer_idx in product(*outer_ranges) if outer_axes else [()]:
            if budget.expired():
                break
            inner_pos = 0
            while inner_pos < len(inner_grid):
                if budget.expired():
                    break
                min_eps_vec = [0.0] * len(non_primary)
                for a, gi in zip(outer_axes, outer_idx):
                    min_eps_vec[a] = float(min_grids[a][gi])
                min_eps_vec[inner_axis] = float(inner_grid[inner_pos])

                result, wall, obj_vec, slack_inner = _run_solve(min_eps_vec)
                if result is None:
                    # Infeasible: tightening the innermost ε further stays
                    # infeasible, so abandon the rest of this inner sweep.
                    break
                _record(result, wall, obj_vec, min_eps_vec)

                if use_bypass and inner_step > 0 and slack_inner is not None:
                    # The slack covers floor(slack/step) additional cells that
                    # would reproduce this solution; jump past them.
                    jump = int(np.floor(max(slack_inner, 0.0) / inner_step))
                    inner_pos += max(jump, 0) + 1
                else:
                    inner_pos += 1
    finally:
        model._objective = saved_obj
        # Remove exactly the constraints we added, by identity (MO8); leaves aux
        # vars and parameters (and any rows other code appended) in place.
        _remove_tracked(model, tracked_cons)

    front = ParetoFront(
        points=points,
        method=_tag("augmecon2" if augmented else "epsilon_constraint", budget),
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
    total_time_limit: Optional[float] = None,
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

    Supports ``total_time_limit`` (overall wall-clock budget; on expiry the
    sweep returns the partial front tagged ``"weighted_tchebycheff/truncated"``)
    and warns when the weight grid exceeds ~200 subproblems (MO7).
    """
    senses_list = _as_senses(senses, len(objectives))
    names = _default_names(objectives, objective_names)
    k = len(objectives)
    saved_obj = model._objective
    tracked_cons: list = []
    budget = _TimeBudget(total_time_limit)

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
        # Compile each objective once for reuse across the payoff table and
        # every accepted point (MO5). Built before the aux t/weight parameters;
        # objectives do not reference those, so the compiled callables are
        # unaffected when they are appended.
        evaluator = ObjectiveEvaluator(model, objectives)

        # Nadir for normalization inside the Tchebycheff form.
        if nadir is None and anchors:
            nadir_arr = nadir_point(
                model, objectives, anchors, senses=senses_list, evaluator=evaluator
            )
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
        _warn_large_grid(len(weights), "weighted_tchebycheff")

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
            _add_tracked(
                model,
                w_params[i] * g_expr - t_var <= 0,
                f"_mo_tcheb_max_{i}",
                tracked_cons,
            )

        # Scalarized objective: t + rho * sum_i w_i * g_i
        scalarized = t_var
        for i in range(k):
            g_expr = signs[i] * (objectives[i] - float(ideal_arr[i])) / float(span[i])
            scalarized = scalarized + float(rho) * (w_params[i] * g_expr)

        points: list[ParetoPoint] = []
        last_x: Optional[dict[str, np.ndarray]] = None

        for w in weights:
            if budget.expired():
                break
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
            obj_vec = _collect_objectives_at_x(objectives, model, result.x, evaluator)
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
        _remove_tracked(model, tracked_cons)

    front = ParetoFront(
        points=points,
        method=_tag("weighted_tchebycheff", budget),
        objective_names=names,
        senses=senses_list,
        ideal=ideal_arr,
        nadir=nadir_arr,
    )
    return front.filtered() if filter else front
