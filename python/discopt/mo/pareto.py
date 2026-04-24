"""Pareto point and front result types for multi-objective optimization.

A ``ParetoPoint`` captures one solution of a scalarized subproblem (decision
values, objective vector, solve status, wall time, and which scalarization
parameters produced it). A ``ParetoFront`` is a sequence of such points plus
front-level metadata (ideal / nadir estimates, objective names, method tag)
and methods for filtering, indicator evaluation, and plotting.

All objectives are stored in their **original sense** (minimize or maximize);
sense handling happens in the scalarization routines that construct the
front, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ParetoPoint:
    """A single Pareto-optimal (or candidate) point from a scalarization solve.

    Attributes
    ----------
    x : dict[str, numpy.ndarray]
        Decision variable values keyed by variable name, as returned by
        :attr:`discopt.modeling.SolveResult.x`.
    objectives : numpy.ndarray
        Objective vector of shape ``(k,)``, in the original sense of each
        objective (no sign-flipping applied).
    status : str
        Solve status string from :class:`discopt.modeling.SolveResult`.
    wall_time : float
        Subproblem wall-clock time in seconds.
    scalarization_params : dict
        Parameters that generated this point (e.g.
        ``{"weights": [0.25, 0.75]}``, ``{"epsilon": [2.1, 3.4]}``). Used for
        reproducibility and plotting.
    """

    x: dict[str, np.ndarray]
    objectives: np.ndarray
    status: str
    wall_time: float
    scalarization_params: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        """Number of objectives."""
        return int(self.objectives.shape[0])


def _dominates(a: np.ndarray, b: np.ndarray, senses: np.ndarray) -> bool:
    """Return True if *a* dominates *b* under the given senses.

    ``senses`` is an array of +1 (minimize) / -1 (maximize). Dominance means
    *a* is no worse in every component and strictly better in at least one.
    """
    diff = senses * (a - b)  # negative means "a better than b" in that slot
    return bool(np.all(diff <= 0.0) and np.any(diff < 0.0))


def filter_nondominated(
    objectives: np.ndarray,
    senses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a boolean mask of the nondominated rows of *objectives*.

    Parameters
    ----------
    objectives : numpy.ndarray
        Shape ``(n, k)`` objective-value array. Each row is one point.
    senses : numpy.ndarray, optional
        Length-``k`` array of +1 (minimize) / -1 (maximize). Defaults to all
        +1.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n,)``; ``mask[i]`` is True iff row ``i`` is
        not strictly dominated by any other row.
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    if objectives.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {objectives.shape}")
    n = objectives.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if senses is None:
        senses = np.ones(objectives.shape[1], dtype=np.float64)
    senses = np.asarray(senses, dtype=np.float64)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if _dominates(objectives[j], objectives[i], senses):
                mask[i] = False
                break
    return mask


@dataclass
class ParetoFront:
    """A collection of Pareto points plus front-level metadata.

    Attributes
    ----------
    points : list[ParetoPoint]
        Raw scalarization solutions, in the order produced by the generator.
        May contain weakly dominated points; call :meth:`filtered` to remove
        them.
    method : str
        Which scalarization produced the front, e.g. ``"weighted_sum"``,
        ``"augmecon2"``, ``"weighted_tchebycheff"``, ``"nbi"``, ``"nnc"``.
    objective_names : list[str]
        Labels for each objective, used in summaries and plots.
    senses : list[str]
        ``"min"`` / ``"max"`` per objective, as supplied by the caller.
    ideal : numpy.ndarray or None
        Per-objective best values (in original senses). ``None`` if not
        computed.
    nadir : numpy.ndarray or None
        Per-objective worst values on the Pareto set (payoff-table estimate).
        ``None`` if not computed.
    """

    points: list[ParetoPoint]
    method: str
    objective_names: list[str]
    senses: list[str]
    ideal: Optional[np.ndarray] = None
    nadir: Optional[np.ndarray] = None

    # ── Accessors ──

    @property
    def k(self) -> int:
        """Number of objectives."""
        return len(self.objective_names)

    @property
    def n(self) -> int:
        """Number of points."""
        return len(self.points)

    def objectives(self) -> np.ndarray:
        """Stack all objective vectors into an ``(n, k)`` array."""
        if not self.points:
            return np.zeros((0, self.k), dtype=np.float64)
        return np.vstack([p.objectives for p in self.points])

    def _senses_array(self) -> np.ndarray:
        return np.array(
            [1.0 if s == "min" else -1.0 for s in self.senses],
            dtype=np.float64,
        )

    # ── Filtering ──

    def filtered(self) -> "ParetoFront":
        """Return a new front containing only strictly nondominated points."""
        objs = self.objectives()
        if objs.size == 0:
            return ParetoFront(
                points=[],
                method=self.method,
                objective_names=list(self.objective_names),
                senses=list(self.senses),
                ideal=self.ideal,
                nadir=self.nadir,
            )
        mask = filter_nondominated(objs, senses=self._senses_array())
        kept = [p for p, keep in zip(self.points, mask) if keep]
        return ParetoFront(
            points=kept,
            method=self.method,
            objective_names=list(self.objective_names),
            senses=list(self.senses),
            ideal=self.ideal,
            nadir=self.nadir,
        )

    # ── Indicators ──

    def hypervolume(self, reference: Optional[np.ndarray] = None) -> float:
        """Hypervolume dominated by this front.

        Convenience delegate to :func:`discopt.mo.indicators.hypervolume`.
        """
        from discopt.mo.indicators import hypervolume

        return hypervolume(self, reference=reference)

    # ── Presentation ──

    def summary(self) -> str:
        """Human-readable summary of the front."""
        lines = [
            f"Pareto Front ({self.method}, {self.n} points, k={self.k})",
            "=" * 60,
        ]
        if self.ideal is not None:
            ideal_str = ", ".join(
                f"{name}={v:.4g}" for name, v in zip(self.objective_names, self.ideal)
            )
            lines.append(f"  ideal: {ideal_str}")
        if self.nadir is not None:
            nadir_str = ", ".join(
                f"{name}={v:.4g}" for name, v in zip(self.objective_names, self.nadir)
            )
            lines.append(f"  nadir: {nadir_str}")
        if self.points:
            objs = self.objectives()
            lines.append("")
            header = "  " + "  ".join(f"{name:>12s}" for name in self.objective_names)
            lines.append(header)
            lines.append("  " + "  ".join("-" * 12 for _ in self.objective_names))
            for row in objs:
                lines.append("  " + "  ".join(f"{v:>12.6g}" for v in row))
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Scatter plot of the front (2-D or 3-D).

        Requires matplotlib. For ``k == 2`` returns a 2-D scatter; for
        ``k == 3`` a 3-D scatter. Higher dimensions raise ``ValueError``.
        Extra ``**kwargs`` are forwarded to ``ax.scatter``.
        """
        import matplotlib.pyplot as plt

        objs = self.objectives()
        if self.k == 2:
            if ax is None:
                _, ax = plt.subplots()
            ax.scatter(objs[:, 0], objs[:, 1], **kwargs)
            ax.set_xlabel(self.objective_names[0])
            ax.set_ylabel(self.objective_names[1])
            ax.set_title(f"Pareto front ({self.method})")
            return ax
        if self.k == 3:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
            ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], **kwargs)
            ax.set_xlabel(self.objective_names[0])
            ax.set_ylabel(self.objective_names[1])
            ax.set_zlabel(self.objective_names[2])
            ax.set_title(f"Pareto front ({self.method})")
            return ax
        raise ValueError(f"plot() supports k in {{2, 3}}; this front has k={self.k}")
