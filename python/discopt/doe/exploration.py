"""Design space exploration via grid evaluation of FIM criteria.

Evaluates FIM metrics over a grid of design conditions to visualize
the information landscape and identify promising experimental regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from discopt.doe.fim import compute_fim
from discopt.estimate import Experiment


@dataclass
class ExplorationResult:
    """Result of design space exploration.

    Attributes
    ----------
    grid : dict[str, numpy.ndarray]
        Grid coordinates for each design variable.
    metrics : dict[str, numpy.ndarray]
        Criterion values at each grid point. Keys include
        ``"log_det_fim"``, ``"trace_fim_inv"``, ``"min_eigenvalue"``,
        ``"condition_number"``.
    design_names : list[str]
        Ordered design variable names.
    """

    grid: dict[str, np.ndarray]
    metrics: dict[str, np.ndarray]
    design_names: list[str]

    def best_point(self, criterion: str = "log_det_fim") -> dict[str, float]:
        """Find the grid point with the best criterion value.

        Parameters
        ----------
        criterion : str, default "log_det_fim"
            Metric key to optimize.

        Returns
        -------
        dict[str, float]
            Design variable values at the best grid point.
        """
        values = self.metrics[criterion]
        if criterion in ("log_det_fim", "min_eigenvalue"):
            # Maximize
            best_idx = np.unravel_index(np.argmax(values), values.shape)
        else:
            # Minimize
            best_idx = np.unravel_index(np.argmin(values), values.shape)

        idx_tuple = best_idx if isinstance(best_idx, tuple) else (best_idx,)

        result = {}
        for i, name in enumerate(self.design_names):
            if i < len(idx_tuple):
                result[name] = float(self.grid[name][idx_tuple[i]])
            else:
                result[name] = float(self.grid[name][idx_tuple[0]])
        return result

    def plot_heatmap(
        self,
        criterion: str = "log_det_fim",
        ax=None,
        **kwargs,
    ):
        """Plot 2D heatmap of a design criterion.

        Requires exactly two design variables.

        Parameters
        ----------
        criterion : str, default "log_det_fim"
            Metric key to plot.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new figure if None.
        **kwargs
            Passed to ``ax.pcolormesh``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if len(self.design_names) != 2:
            raise ValueError("plot_heatmap requires exactly 2 design variables")

        if ax is None:
            _, ax = plt.subplots()

        n1 = self.design_names[0]
        n2 = self.design_names[1]
        X, Y = np.meshgrid(self.grid[n1], self.grid[n2])
        Z = self.metrics[criterion]

        pcm = ax.pcolormesh(X, Y, Z.T, shading="auto", **kwargs)
        ax.set_xlabel(n1)
        ax.set_ylabel(n2)
        ax.set_title(criterion)
        ax.figure.colorbar(pcm, ax=ax)
        return ax

    def plot_sensitivity(
        self,
        criterion: str = "log_det_fim",
        ax=None,
    ):
        """Plot 1D sensitivity curves.

        For each design variable, plot the criterion value as the
        variable is swept while others are fixed at their midpoint.

        Parameters
        ----------
        criterion : str, default "log_det_fim"
            Metric key to plot.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        values = self.metrics[criterion]
        if values.ndim == 1:
            name = self.design_names[0]
            ax.plot(self.grid[name], values, "o-", label=name)
        else:
            # For multi-D, take slices through the midpoint
            for i, name in enumerate(self.design_names):
                mid_idx = [s // 2 for s in values.shape]
                slc = [mid_idx[j] if j != i else slice(None) for j in range(values.ndim)]
                ax.plot(self.grid[name], values[tuple(slc)], "o-", label=name)

        ax.set_ylabel(criterion)
        ax.legend()
        return ax


def explore_design_space(
    experiment: Experiment,
    param_values: dict[str, float],
    design_ranges: dict[str, np.ndarray],
    *,
    prior_fim: np.ndarray | None = None,
) -> ExplorationResult:
    """Evaluate FIM metrics over a grid of design conditions.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal parameter values.
    design_ranges : dict[str, numpy.ndarray]
        Grid values for each design variable. The full Cartesian
        product of all ranges is evaluated.
    prior_fim : numpy.ndarray, optional
        Prior FIM from previous experiments.

    Returns
    -------
    ExplorationResult
        FIM metrics at each grid point.
    """
    design_names = list(design_ranges.keys())
    grid_arrays = [design_ranges[name] for name in design_names]
    grid_shape = tuple(len(arr) for arr in grid_arrays)

    # Initialize metric arrays
    metric_keys = ["log_det_fim", "trace_fim_inv", "min_eigenvalue", "condition_number"]
    metrics = {key: np.full(grid_shape, np.nan) for key in metric_keys}

    # Evaluate FIM at each grid point
    for idx in product(*(range(n) for n in grid_shape)):
        design_point = {name: float(grid_arrays[i][idx[i]]) for i, name in enumerate(design_names)}
        try:
            fim_result = compute_fim(experiment, param_values, design_point, prior_fim=prior_fim)
            m = fim_result.metrics
            for key in metric_keys:
                metrics[key][idx] = m[key]
        except Exception:
            # Leave as NaN for infeasible points
            continue

    return ExplorationResult(
        grid={name: grid_arrays[i] for i, name in enumerate(design_names)},
        metrics=metrics,
        design_names=design_names,
    )
