"""Input/output scaling for neural network formulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OffsetScaling:
    """Affine scaling: ``x_scaled = (x - offset) / factor``.

    Parameters
    ----------
    x_offset : np.ndarray
        Offset subtracted from inputs before feeding to the network.
    x_factor : np.ndarray
        Scale factor dividing inputs.
    y_offset : np.ndarray
        Offset added to network outputs.
    y_factor : np.ndarray
        Scale factor multiplying network outputs.
    """

    x_offset: np.ndarray
    x_factor: np.ndarray
    y_offset: np.ndarray
    y_factor: np.ndarray

    def __post_init__(self) -> None:
        self.x_offset = np.asarray(self.x_offset, dtype=np.float64)
        self.x_factor = np.asarray(self.x_factor, dtype=np.float64)
        self.y_offset = np.asarray(self.y_offset, dtype=np.float64)
        self.y_factor = np.asarray(self.y_factor, dtype=np.float64)
        if np.any(self.x_factor == 0):
            raise ValueError("x_factor must not contain zeros")
        if np.any(self.y_factor == 0):
            raise ValueError("y_factor must not contain zeros")
