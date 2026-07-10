"""Multi-trajectory fitting glue for hybrid physics+ML models.

:func:`fit_trajectories` builds one collocation transcription
(:class:`~discopt.dae.collocation.DAEBuilder`) per experimental trajectory on a
*shared* model, wiring them all to the same right-hand side. When that RHS closes
over a trainable surrogate's weight ``Variable``s (see
:mod:`discopt.nn.trainable`), the weights are shared across every trajectory and
trained jointly — the simultaneous multi-experiment neural-DAE setup measured in
``scripts/hybrid_ml/exp_c_paper_scale.py``.

Everything here is composition over existing primitives (``DAEBuilder``,
``least_squares``, ``state_at``, ``align_time_grid``); no new solver math.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from discopt.dae.collocation import ContinuousSet, DAEBuilder, align_time_grid
from discopt.modeling.core import Model, Variable


@dataclass
class Trajectory:
    """One observed trajectory (experiment).

    Parameters
    ----------
    t_data : np.ndarray
        Measurement times, shape ``(n_obs,)``.
    y_data : dict[str, np.ndarray]
        Observations keyed by state name, each shape ``(n_obs,)``. States absent
        here are unobserved (they still exist in the model, just not fit).
    initial : dict[str, float or np.ndarray]
        Initial condition per state at ``t0``.
    weights : dict[str, float or np.ndarray], optional
        Per-state least-squares weights. A scalar weights that state's whole
        residual sum; a length-``n_obs`` array weights each observation. Absent
        states default to weight 1.
    """

    t_data: np.ndarray
    y_data: dict[str, np.ndarray]
    initial: dict[str, Union[float, np.ndarray]]
    weights: Union[dict[str, Union[float, np.ndarray]], None] = None

    def __post_init__(self) -> None:
        self.t_data = np.asarray(self.t_data, dtype=np.float64)
        self.y_data = {k: np.asarray(v, dtype=np.float64) for k, v in self.y_data.items()}
        for name, y in self.y_data.items():
            if y.shape != self.t_data.shape:
                raise ValueError(
                    f"y_data['{name}'] has shape {y.shape}, expected {self.t_data.shape} "
                    "(one observation per measurement time)"
                )


@dataclass
class TrajectoryFit:
    """Result of :func:`fit_trajectories`: the per-trajectory builders + helpers."""

    builders: list[DAEBuilder]
    trajectories: list[Trajectory]
    state_names: list[str]

    def least_squares(self, interpolate: bool = True):
        """Weighted sum-of-squares over every trajectory and observed state.

        Uses exact collocation-polynomial interpolation at the measurement times
        by default (see :meth:`DAEBuilder.state_at`).
        """
        terms = []
        for traj, builder in zip(self.trajectories, self.builders):
            for name, y in traj.y_data.items():
                w = None if traj.weights is None else traj.weights.get(name)
                terms.append(_weighted_state_lsq(builder, name, traj.t_data, y, w, interpolate))
        if not terms:
            raise ValueError("no observed states across the trajectories; nothing to fit")
        out = terms[0]
        for t in terms[1:]:
            out = out + t
        return out

    def warm_start(self) -> dict[Variable, np.ndarray]:
        """Data-interpolated initial values for every state variable.

        Observed states are interpolated from their measurements onto the
        collocation grid; unobserved states are held flat at their initial
        condition. Merge with a surrogate's ``initial_values()`` via ``|`` to
        form a complete warm start.
        """
        init: dict[Variable, np.ndarray] = {}
        for traj, builder in zip(self.trajectories, self.builders):
            tp = builder._element_points()  # (nfe, ncp+1)
            for name in self.state_names:
                var = builder.get_state(name)
                if name in traj.y_data:
                    y = traj.y_data[name]
                    x0 = float(traj.initial.get(name, y[0]))
                    grid = np.interp(
                        tp,
                        np.concatenate([[builder._cs.bounds[0]], traj.t_data]),
                        np.concatenate([[x0], y]),
                    )
                else:
                    grid = np.full(tp.shape, float(traj.initial.get(name, 0.0)))
                init[var] = grid
        return init

    def extract(self, result, k: int, state: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract ``(t, x)`` for trajectory ``k``'s ``state`` from a solve result.

        Accepts either a ``SolveResult`` (has ``.value``) or an ``NLPResult``
        from :func:`discopt.nn.train` (has a flat ``.x``).
        """
        builder = self.builders[k]
        if not callable(getattr(result, "value", None)):
            x = getattr(result, "x", None)
            if x is None:
                raise TypeError(
                    f"cannot extract from {type(result).__name__}: expected a "
                    "SolveResult (.value) or an NLPResult (.x)"
                )
            result = _ValueResultAdapter(builder._model, x)
        return builder.extract_solution(result, state)


class _ValueResultAdapter:
    """Wrap an ``NLPResult`` (flat ``.x``) with a ``.value(var)`` accessor.

    ``DAEBuilder.extract_solution`` expects a ``SolveResult``; the local
    :func:`discopt.nn.train` returns an ``NLPResult``. This adapter bridges the
    two by splitting the flat solution with the model's variable ordering.
    """

    def __init__(self, model, x_flat):
        from discopt.warm_start import unflatten_solution

        self._values = unflatten_solution(model, np.asarray(x_flat))

    def value(self, var):
        return self._values[var]


def _weighted_state_lsq(builder, name, t_data, y_data, weight, interpolate):
    """Build ``sum_i w_i (x(t_i) - y_i)^2`` for one state, respecting weights."""
    if weight is None:
        return builder.least_squares(name, t_data, y_data, interpolate=interpolate)
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim == 0:
        # Scalar per-state weight: scale the whole (unweighted) residual sum.
        return float(w) * builder.least_squares(name, t_data, y_data, interpolate=interpolate)
    if w.shape != np.asarray(t_data).shape:
        raise ValueError(
            f"per-point weights for state '{name}' have shape {w.shape}, expected "
            f"{np.asarray(t_data).shape}"
        )
    # Per-observation weights: build residuals directly via exact interpolation.
    terms = []
    for i in range(len(t_data)):
        x_expr = builder.state_at(name, float(t_data[i]))
        terms.append(float(w[i]) * (x_expr - float(y_data[i])) ** 2)
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def fit_trajectories(
    model: Model,
    *,
    trajectories: Sequence[Trajectory],
    states: Sequence[tuple[str, dict]],
    rhs: Callable,
    t_span: tuple[float, float],
    nfe: int,
    ncp: int = 2,
    scheme: str = "radau",
    algebraics: Sequence[tuple[str, dict]] = (),
    controls: Sequence[tuple[str, dict]] = (),
    align_grid: bool = False,
    name: str = "traj",
) -> TrajectoryFit:
    """Transcribe one collocation block per trajectory on a shared model.

    Every block uses the *same* ``rhs`` callable, so any surrogate weights it
    closes over are shared and trained jointly. Returns a :class:`TrajectoryFit`
    exposing the joint least-squares objective and a data-interpolated warm
    start.

    Parameters
    ----------
    model : Model
        Shared model; the surrogate's weight variables should already be created
        on it before calling this.
    trajectories : sequence of Trajectory
        The experiments to fit.
    states : sequence of (name, spec)
        State declarations, ``spec`` a dict passed to
        :meth:`DAEBuilder.add_state` (e.g. ``{"bounds": (0, 1.5)}``); the initial
        condition is taken per-trajectory from ``Trajectory.initial``.
    rhs : callable
        ``rhs(t, states, algebraics, controls) -> dict[str, Expression]`` — the
        shared dynamics (see :meth:`DAEBuilder.set_ode`).
    t_span : (float, float)
        Time domain ``(t0, tf)``, shared across trajectories.
    nfe, ncp, scheme : int, int, str
        Collocation parameters (per :class:`ContinuousSet`).
    algebraics, controls : sequence of (name, spec)
        Optional algebraic-variable and control declarations.
    align_grid : bool
        If True, snap each trajectory's element boundaries to its measurement
        times via :func:`~discopt.dae.collocation.align_time_grid`.
    name : str
        Prefix for the per-trajectory ``ContinuousSet`` names.

    Returns
    -------
    TrajectoryFit
    """
    if not trajectories:
        raise ValueError("fit_trajectories requires at least one trajectory")
    state_names = [s[0] for s in states]

    builders: list[DAEBuilder] = []
    for k, traj in enumerate(trajectories):
        if align_grid:
            eb = align_time_grid(t_span, nfe, traj.t_data)
            cs = ContinuousSet(
                f"{name}{k}", t_span, nfe=nfe, ncp=ncp, scheme=scheme, element_boundaries=eb
            )
        else:
            cs = ContinuousSet(f"{name}{k}", t_span, nfe=nfe, ncp=ncp, scheme=scheme)
        builder = DAEBuilder(model, cs)

        for sname, spec in states:
            spec = dict(spec)
            spec.pop("initial", None)  # initial comes from the trajectory
            builder.add_state(sname, initial=traj.initial.get(sname), **spec)
        for aname, spec in algebraics:
            builder.add_algebraic(aname, **dict(spec))
        for cname, spec in controls:
            builder.add_control(cname, **dict(spec))

        builder.set_ode(rhs)
        builder.discretize()
        builders.append(builder)

    return TrajectoryFit(
        builders=builders,
        trajectories=list(trajectories),
        state_names=state_names,
    )
