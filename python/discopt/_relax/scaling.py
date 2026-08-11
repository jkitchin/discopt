"""Problem scaling for improved IPM conditioning.

Applies diagonal scaling to the objective and constraints so that
the KKT system has well-conditioned entries.  Scaling is a preprocessing
step: the IPM solves the scaled problem and the solution is unscaled
before returning.

The approach follows Ipopt's gradient-based scaling (default
``nlp_scaling_method=gradient-based``): scale the objective so that
``max(|grad_f|) ~ 1`` and scale each constraint row so that
``max(|J_i|) ~ 1``.

For LP/QP problems, this simplifies to:
- Objective scale: ``1 / max(|c|, 1)``
- Constraint row scale: ``1 / max(|A_i|, 1)`` for each row i
- Variable scale (optional): ``1 / max(column norms)``
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class LPScaleFactors:
    """Scaling factors for an LP/QP problem."""

    obj_scale: float
    row_scale: np.ndarray  # (m,) per-constraint scaling
    applied: bool  # whether scaling was actually applied


def compute_lp_scaling(
    c: np.ndarray | jnp.ndarray,
    A: np.ndarray | jnp.ndarray,
) -> LPScaleFactors:
    """Compute scaling factors for an LP: min c'x s.t. Ax = b.

    Returns factors such that the scaled problem has
    ``max(|c_scaled|) <= 1`` and ``max(|A_i_scaled|) <= 1``.
    """
    c_np = np.asarray(c)
    A_np = np.asarray(A)

    # Objective scaling: 1 / max(|c|)
    c_max = np.max(np.abs(c_np))
    obj_scale = 1.0 / max(c_max, 1.0)

    # Row scaling: 1 / max(|A_i|) per row
    m = A_np.shape[0]
    if m > 0:
        row_norms = np.max(np.abs(A_np), axis=1)
        row_scale = np.where(row_norms > 1.0, 1.0 / row_norms, 1.0)
    else:
        row_scale = np.ones(0, dtype=np.float64)

    # Only apply if there's meaningful variation
    applied = c_max > 10.0 or (m > 0 and np.max(row_norms) > 10.0)

    return LPScaleFactors(obj_scale=obj_scale, row_scale=row_scale, applied=applied)


def scale_lp(c, A, b, factors: LPScaleFactors):
    """Apply scaling to LP data. Returns (c_s, A_s, b_s)."""
    if not factors.applied:
        return c, A, b
    c_s = c * factors.obj_scale
    if A.shape[0] > 0:
        D = jnp.asarray(factors.row_scale)
        A_s = D[:, None] * A
        b_s = D * b
    else:
        A_s, b_s = A, b
    return c_s, A_s, b_s


def unscale_lp_solution(x, y, z_l, z_u, factors: LPScaleFactors):
    """Unscale LP solution back to original problem.

    The primal x is unchanged by row/objective scaling.
    The dual y is scaled by row_scale / obj_scale.
    The bound multipliers z_l, z_u are scaled by obj_scale.
    """
    if not factors.applied:
        return x, y, z_l, z_u
    # x is unchanged (no variable scaling)
    # y_original = row_scale * y_scaled / obj_scale
    if y.shape[0] > 0:
        D = jnp.asarray(factors.row_scale)
        y_out = D * y / factors.obj_scale
    else:
        y_out = y
    z_l_out = z_l / factors.obj_scale
    z_u_out = z_u / factors.obj_scale
    return x, y_out, z_l_out, z_u_out
