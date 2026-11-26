"""Post-solve mesh diagnostics and refinement for collocation transcriptions (#2365).

A collocation solve enforces the ODE only at the collocation points. Between them
the state is the element's interpolating polynomial, and nothing in the solve
status says whether that polynomial follows the true trajectory. On a stiff
problem a fast transient can sit inside one element: the NLP still converges, and
the error shows only as undershoot between the nodes, or as a distorted fit once
the states carry physical bounds.

This module measures that error after the solve.

* :class:`MeshErrorEstimate` (from :meth:`DAEBuilder.error_estimate
  <discopt.dae.collocation.DAEBuilder.error_estimate>`) holds the *defect* of
  the collocation polynomial ``r(t) = x_p'(t) - f(t, x_p(t), z_p(t), u)`` at check
  points between the nodes of every element, and the per-element, per-state
  estimate ``h_i * max |r|``. The defect is zero at the collocation points by
  construction; at the check points it is what the solve did not enforce. To
  first order (a Gronwall argument on ``x_p - x``) the state error an element adds
  is bounded by the integral of ``|r|`` over it, which ``h_i * max |r|`` bounds in
  turn. This is the residual (defect) control used to assess and refine a
  collocation mesh in :cite:t:`Biegler2010`, ch. 10. It also lists every check
  point where the polynomial leaves the declared state bounds although the nodes
  stay inside them, the cheap signature of undershoot.
* :meth:`MeshErrorEstimate.refined_boundaries` splits the flagged elements. It
  only *adds* boundaries, so every boundary the user placed (observation times,
  input discontinuities) survives refinement.
* :func:`refine_mesh` runs build, solve, estimate, refine until the estimate is
  within tolerance, warm-starting each round from the previous solution's
  polynomials (:meth:`DAEBuilder.warm_start_from
  <discopt.dae.collocation.DAEBuilder.warm_start_from>`).

No solver math is touched: the estimate only reads a solution.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class MeshAdequacyWarning(UserWarning):
    """The collocation mesh is too coarse for the solution it carries (#2365)."""


@dataclass
class BoundExcursion:
    """A check point where a state polynomial leaves its declared bounds.

    The nodes of the element are inside the bounds (they are bounded variables);
    the polynomial between them is not. ``value`` is the polynomial's value at
    time ``t``; ``bound`` is the bound it crosses.
    """

    state: str
    component: int | None
    element: int
    t: float
    value: float
    bound: float


@dataclass
class MeshErrorEstimate:
    """Per-element error estimate of a collocation solution (#2365).

    Attributes
    ----------
    element_boundaries : np.ndarray
        The mesh the estimate was taken on, shape ``(nfe + 1,)``.
    ncp : int
        Collocation points per element.
    error : dict[str, np.ndarray]
        Per state, ``h_i * max |defect|`` over the check points of element ``i``:
        shape ``(nfe,)`` for a scalar state, ``(nfe, n_components)`` for a vector
        state. It has the units of the state.
    scaled : dict[str, np.ndarray]
        ``error / (atol + rtol * scale)``, where ``scale`` is the largest
        ``|x|`` the state component takes over the whole horizon. A value above 1
        means the element is out of tolerance for that state.
    element_error : np.ndarray
        Largest ``scaled`` value over every state and component, per element,
        shape ``(nfe,)``.
    bound_excursions : list[BoundExcursion]
        Check points where the polynomial leaves the declared state bounds.
    atol, rtol : float
        The tolerances ``scaled`` was formed with.
    n_check : int
        Number of check points evaluated per element.
    """

    element_boundaries: np.ndarray
    ncp: int
    error: dict[str, np.ndarray]
    scaled: dict[str, np.ndarray]
    element_error: np.ndarray
    bound_excursions: list[BoundExcursion] = field(default_factory=list)
    atol: float = 1e-6
    rtol: float = 1e-3
    n_check: int = 0

    @property
    def nfe(self) -> int:
        """Number of finite elements."""
        return len(self.element_boundaries) - 1

    @property
    def max_error(self) -> float:
        """Largest scaled error over all elements (``> 1`` is out of tolerance)."""
        return float(np.max(self.element_error)) if self.element_error.size else 0.0

    @property
    def worst_element(self) -> int:
        """Index of the element with the largest scaled error."""
        return int(np.argmax(self.element_error))

    @property
    def worst_state(self) -> tuple[str, int | None]:
        """``(state, component)`` that sets the error of :attr:`worst_element`."""
        i = self.worst_element
        best: tuple[str, int | None] = ("", None)
        best_val = -math.inf
        for name, arr in self.scaled.items():
            row = arr[i]
            if np.ndim(row) == 0:
                if float(row) > best_val:
                    best_val, best = float(row), (name, None)
            else:
                c = int(np.argmax(row))
                if float(row[c]) > best_val:
                    best_val, best = float(row[c]), (name, c)
        return best

    @property
    def flagged_elements(self) -> np.ndarray:
        """Elements out of tolerance or with a bound excursion, sorted."""
        out = set(np.flatnonzero(self.element_error > 1.0).tolist())
        out.update(e.element for e in self.bound_excursions)
        return np.array(sorted(out), dtype=int)

    @property
    def adequate(self) -> bool:
        """True when no element is flagged."""
        return self.flagged_elements.size == 0

    def summary(self) -> str:
        """One human-readable paragraph describing the estimate."""
        eb = self.element_boundaries
        i = self.worst_element
        name, comp = self.worst_state
        label = name if comp is None else f"{name}[{comp}]"
        lines = [
            f"collocation mesh: {self.nfe} elements, ncp={self.ncp}, "
            f"atol={self.atol:g}, rtol={self.rtol:g}",
            f"  worst element {i} on [{eb[i]:.6g}, {eb[i + 1]:.6g}]: scaled error "
            f"{self.element_error[i]:.3g} (state {label})",
            f"  flagged elements: {self.flagged_elements.size}",
        ]
        if self.bound_excursions:
            worst = max(self.bound_excursions, key=lambda e: abs(e.value - e.bound))
            wl = worst.state if worst.component is None else f"{worst.state}[{worst.component}]"
            lines.append(
                f"  {len(self.bound_excursions)} check point(s) leave the state bounds; worst "
                f"{wl} = {worst.value:.3g} at t={worst.t:.6g} (bound {worst.bound:g})"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def refined_boundaries(
        self,
        *,
        max_split: int = 8,
        min_split: int = 2,
    ) -> np.ndarray:
        """A refined mesh: each flagged element split into equal pieces.

        The number of pieces for element ``i`` is
        ``ceil(element_error[i] ** (1 / (ncp + 1)))``, clipped to
        ``[min_split, max_split]``: the estimate scales like ``h ** (ncp + 1)``
        for a smooth solution, so that many pieces bring it to the tolerance in
        one round when the asymptotic rate holds. Inside a stiff transient it
        does not hold, and :func:`refine_mesh` simply takes another round.

        Every existing boundary is kept, so boundaries the user placed on purpose
        (observation times, input discontinuities) stay boundaries.

        Returns
        -------
        np.ndarray
            Sorted, strictly increasing boundaries, a superset of
            :attr:`element_boundaries`.
        """
        if min_split < 2 or max_split < min_split:
            raise ValueError(
                f"need 2 <= min_split <= max_split, got min_split={min_split}, "
                f"max_split={max_split}"
            )
        eb = self.element_boundaries
        new = [eb]
        for i in self.flagged_elements:
            ratio = max(float(self.element_error[i]), 1.0)
            k = math.ceil(ratio ** (1.0 / (self.ncp + 1)))
            k = int(min(max(k, min_split), max_split))
            new.append(np.linspace(eb[i], eb[i + 1], k + 1)[1:-1])
        out = np.unique(np.concatenate(new))
        # Keep the original boundaries bit-exact (np.unique preserves them, but a
        # split point could land within rounding of one; drop such near-duplicates).
        keep = np.ones(out.size, dtype=bool)
        tiny = 1e-12 * max(1.0, float(np.max(np.abs(eb))))
        orig = set(eb.tolist())
        for j in range(1, out.size):
            if out[j] - out[j - 1] < tiny and out[j] not in orig:
                keep[j] = False
        return out[keep]


# ── evaluation helpers ──


def _flat_solution(model: Any, result: Any) -> np.ndarray:
    """The flat point, in ``model._variables`` order, of ``result``."""
    if result.x is None:
        raise ValueError("The result carries no solution; there is nothing to estimate.")
    parts: list[np.ndarray] = []
    for v in model._variables:
        if v.name not in result.x:
            raise KeyError(
                f"Variable {v.name!r} of the model is missing from the solution; the "
                "result does not belong to this model."
            )
        arr = np.asarray(result.x[v.name], dtype=np.float64).reshape(-1)
        if arr.size != v.size:
            raise ValueError(f"Variable {v.name!r}: expected {v.size} values, got {arr.size}")
        parts.append(arr)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _evaluate(value: Any, model: Any, x_flat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Numeric value of an RHS entry (an expression or a plain number), broadcast."""
    from discopt.modeling.core import Expression

    if isinstance(value, Expression):
        from discopt._point_eval import evaluate_at

        arr = np.asarray(evaluate_at(value, model, x_flat), dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
    return np.broadcast_to(arr, shape)


def check_points(nodes: np.ndarray, per_interval: int) -> np.ndarray:
    """Points strictly between consecutive entries of ``nodes`` (on ``[0, 1]``).

    ``per_interval`` equally spaced points are placed inside each gap of
    ``nodes ∪ {1}``, so no check point is a collocation point.
    """
    if per_interval < 1:
        raise ValueError(f"per_interval must be >= 1, got {per_interval}")
    grid = np.unique(np.concatenate([nodes, [0.0, 1.0]]))
    frac = np.arange(1, per_interval + 1) / (per_interval + 1)
    pts = [a + frac * (b - a) for a, b in zip(grid[:-1], grid[1:])]
    return np.concatenate(pts)


def lagrange_matrices(nodes: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(L, dL)``: Lagrange basis through ``nodes`` and its derivative at ``pts``.

    ``L[p, k] = L_k(pts[p])`` and ``dL[p, k] = L_k'(pts[p])``.
    """
    from discopt.dae.polynomials import _lagrange_deriv, lagrange_basis

    L = np.array([[lagrange_basis(nodes, float(t), k) for k in range(len(nodes))] for t in pts])
    dL = np.array([[_lagrange_deriv(nodes, float(t), k) for k in range(len(nodes))] for t in pts])
    return L, dL


def warn_if_inadequate(est: MeshErrorEstimate, stacklevel: int = 3) -> None:
    """Emit a :class:`MeshAdequacyWarning` when ``est`` flags any element."""
    if est.adequate:
        return
    warnings.warn(
        "The collocation mesh does not resolve this solution; the solve status does "
        "not reflect it.\n"
        + est.summary()
        + "\nRefine the flagged elements (MeshErrorEstimate.refined_boundaries() or "
        "discopt.dae.refine_mesh) and re-solve.",
        MeshAdequacyWarning,
        stacklevel=stacklevel,
    )


# ── refinement loop ──


@dataclass
class MeshRefinement:
    """Outcome of :func:`refine_mesh`.

    Attributes
    ----------
    result : SolveResult
        The final solve.
    model : Model
        The final model.
    builder : DAEBuilder or MOLBuilder
        The final builder.
    estimate : MeshErrorEstimate
        Error estimate of the final solve.
    history : list[MeshErrorEstimate]
        The estimate of every round, first to last.
    adequate : bool
        True when the final estimate is within tolerance.
    """

    result: Any
    model: Any
    builder: Any
    estimate: MeshErrorEstimate
    history: list[MeshErrorEstimate]
    adequate: bool

    @property
    def element_boundaries(self) -> np.ndarray:
        """The final mesh."""
        return self.estimate.element_boundaries


def _warm_start(new_model: Any, new_builder: Any, old_builder: Any, old_result: Any) -> dict:
    """Initial point for ``new_model``: polynomial interpolation for the builder's
    variables, a same-name, same-shape copy for every other variable."""
    guess = dict(new_builder.warm_start_from(old_builder, old_result))
    taken = {getattr(v, "name", None) for v in guess}
    for v in new_model._variables:
        if v.name in taken or old_result.x is None or v.name not in old_result.x:
            continue
        val = np.asarray(old_result.x[v.name], dtype=np.float64)
        if val.shape == tuple(v.shape):
            guess[v] = val.copy()
    return guess


def _by_name(model: Any, guess: dict) -> dict:
    """Re-key ``guess`` onto ``model``'s variables by name.

    ``build`` makes a fresh model each round, so a caller cannot hold its
    variables in advance; keys may be variables of any model (or plain names).
    """
    by_name = {v.name: v for v in model._variables}
    out: dict = {}
    for key, val in guess.items():
        name = key if isinstance(key, str) else getattr(key, "name", None)
        if name not in by_name:
            raise KeyError(f"initial_solution names {name!r}, which the built model does not have")
        out[by_name[name]] = val
    return out


def refine_mesh(
    build: Callable[[np.ndarray], tuple[Any, Any]],
    element_boundaries: np.ndarray,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    max_rounds: int = 5,
    max_elements: int = 2000,
    max_split: int = 8,
    initial_solution: dict | None = None,
    check_per_interval: int = 2,
    **solve_kwargs: Any,
) -> MeshRefinement:
    """Solve, estimate the mesh error, refine the flagged elements, repeat (#2365).

    Parameters
    ----------
    build : callable
        ``build(element_boundaries) -> (model, builder)``. It must construct the
        whole model (variables, the discretized ``builder``, constraints and
        objective) on the given mesh. ``builder`` is a :class:`DAEBuilder` or a
        collocation :class:`MOLBuilder`. The mesh handed to ``build`` always
        contains every boundary of ``element_boundaries``.
    element_boundaries : np.ndarray
        Starting mesh. Its boundaries are never removed.
    atol, rtol : float
        Tolerances of the estimate (see :meth:`DAEBuilder.error_estimate`).
    max_rounds : int
        Maximum number of refinement rounds after the first solve.
    max_elements : int
        Refinement stops (with ``adequate=False``) rather than exceed this many
        elements.
    max_split : int
        Largest number of pieces one element is split into per round.
    initial_solution : dict, optional
        Starting point for the first solve, keyed by variable (of any model built
        on the starting mesh) or by variable name; keys are matched by name to the
        model ``build`` returns. Later solves are warm-started from
        the previous solution: builder variables by evaluating the previous
        collocation polynomials at the new nodes, any other variable by name when
        its shape is unchanged.
    check_per_interval : int
        Check points per gap between nodes (see :meth:`DAEBuilder.error_estimate`).
    **solve_kwargs
        Passed to ``model.solve`` on every round.

    Returns
    -------
    MeshRefinement
        The final solve and the estimate of every round. When the mesh could not
        be made adequate within the limits, ``adequate`` is False and a
        :class:`MeshAdequacyWarning` is emitted.
    """
    mesh = np.asarray(element_boundaries, dtype=np.float64)
    history: list[MeshErrorEstimate] = []
    prev: tuple[Any, Any] | None = None  # (builder, result) of the previous round
    for round_ in range(max_rounds + 1):
        model, builder = build(mesh)
        kwargs = dict(solve_kwargs)
        if prev is not None:
            kwargs["initial_solution"] = _warm_start(model, builder, *prev)
        elif initial_solution is not None:
            kwargs["initial_solution"] = _by_name(model, initial_solution)
        result = model.solve(**kwargs)
        if result.x is None:
            raise RuntimeError(
                f"refine_mesh: the solve on {len(mesh) - 1} elements (round {round_}) "
                f"returned no solution (status {result.status!r}); there is nothing to "
                "estimate."
            )
        est = builder.error_estimate(
            result, atol=atol, rtol=rtol, check_per_interval=check_per_interval, warn=False
        )
        history.append(est)
        if est.adequate:
            return MeshRefinement(result, model, builder, est, history, True)
        if round_ == max_rounds:
            break
        new_mesh = est.refined_boundaries(max_split=max_split)
        if len(new_mesh) - 1 > max_elements:
            break
        prev = (builder, result)
        mesh = new_mesh

    warn_if_inadequate(history[-1], stacklevel=2)
    return MeshRefinement(result, model, builder, history[-1], history, False)
