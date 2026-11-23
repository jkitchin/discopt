"""Interval arithmetic bound propagation through neural networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.ml.activations import activation_interval
from discopt.ml.network import Activation, NetworkDefinition

# Sentinel "unbounded" values matching Model.continuous defaults; used when no
# layer bounds were propagated so downstream vars stay effectively free.
_DEFAULT_LB = -9.999e19
_DEFAULT_UB = 9.999e19


@dataclass
class LayerBounds:
    """Pre- and post-activation bounds for a single layer."""

    pre_lb: np.ndarray
    pre_ub: np.ndarray
    post_lb: np.ndarray
    post_ub: np.ndarray


def _apply_activation_bounds(
    activation: Activation, pre_lb: np.ndarray, pre_ub: np.ndarray, beta: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Interval image of ``[pre_lb, pre_ub]`` under the activation.

    Exact (up to endpoint rounding, which is not outward) for every supported
    activation, including the non-monotone SWISH and GELU, whose lower end is
    the global minimum when the interval contains the minimizer; see
    :func:`discopt.ml.activations.activation_interval`.
    """
    return activation_interval(activation, pre_lb, pre_ub, beta)


def propagate_bounds(
    network: NetworkDefinition,
    *,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[LayerBounds]:
    """Propagate input bounds through the network via interval arithmetic.

    Uses natural interval extension: split weights into positive and negative
    parts to compute tight affine bounds, then take the interval image of
    each activation (monotone or not).

    Parameters
    ----------
    network : NetworkDefinition
        Network with ``input_bounds`` set.
    input_bounds : tuple of (np.ndarray, np.ndarray), optional
        ``(lb, ub)`` box to propagate instead of ``network.input_bounds``. When
        given, layer 1 consumes *this* box; use it to propagate over the domain
        the layers actually see (e.g. the scaled input box when an
        :class:`~discopt.ml.scaling.OffsetScaling` is applied — see T-N0.2 /
        finding F1). Defaults to ``network.input_bounds`` (behavior-preserving).

    Returns
    -------
    list of LayerBounds
        Bounds for each layer (pre- and post-activation).

    Raises
    ------
    ValueError
        If neither ``input_bounds`` nor ``network.input_bounds`` is set.
    """
    if input_bounds is not None:
        lb, ub = input_bounds
    elif network.input_bounds is not None:
        lb, ub = network.input_bounds
    else:
        raise ValueError("input_bounds must be set for bound propagation")

    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)

    result: list[LayerBounds] = []

    for layer in network.layers:
        W = layer.weights
        b = layer.biases

        # Natural interval extension for affine: zhat = W^T @ x + b
        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)

        pre_lb = W_pos.T @ lb + W_neg.T @ ub + b
        pre_ub = W_pos.T @ ub + W_neg.T @ lb + b

        post_lb, post_ub = _apply_activation_bounds(layer.activation, pre_lb, pre_ub, layer.beta)

        result.append(LayerBounds(pre_lb, pre_ub, post_lb, post_ub))

        # Post-activation bounds become input bounds for the next layer
        lb, ub = post_lb, post_ub

    return result


def scaled_output_bounds(
    layer_bounds: list[LayerBounds] | None,
    y_offset: np.ndarray,
    y_factor: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sign-safe bounds for affinely scaled network outputs (T-N0.4).

    The scaled output is ``outputs[j] = y_factor[j] * z[j] + y_offset[j]`` where
    ``z`` is the last layer's post-activation. Given that layer's propagated
    ``post_lb``/``post_ub``, return element-wise ``(lb, ub)`` accounting for the
    sign of ``y_factor``. When ``layer_bounds is None`` (no propagation), return
    the effectively-unbounded sentinel box, preserving prior behavior.

    This is a pure strengthening: it can never exclude a reachable prediction
    because it maps the exact interval hull of the last layer's outputs.
    """
    if layer_bounds is None:
        return (
            np.full(output_size, _DEFAULT_LB, dtype=np.float64),
            np.full(output_size, _DEFAULT_UB, dtype=np.float64),
        )
    last = layer_bounds[-1]
    yf = np.asarray(y_factor, dtype=np.float64)
    yo = np.asarray(y_offset, dtype=np.float64)
    a = yf * last.post_lb
    b = yf * last.post_ub
    lb = np.minimum(a, b) + yo
    ub = np.maximum(a, b) + yo
    return lb, ub


# --------------------------------------------------------------------------- #
# LP-based tightening of ReLU pre-activation bounds (T-N3.2,
# ``DISCOPT_NN_LP_BIGM``).
#
# Interval arithmetic treats every neuron of a layer as independent, so the
# big-M constants it gives a deep network are loose; minimising and maximising
# each pre-activation over the LP (triangle) relaxation of the layers before it
# is the standard tightening (Tjeng, Xiao & Tedrake 2019; Grimstad & Andersson
# 2019). Measured as entry experiment E1 (docs/dev/nn-module-plan.md §6, prompted
# by PWLNet, Sanyal et al. 2026, which uses interval big-Ms): median deep-layer
# big-M shrinkage 45-66% on 10 nets, and faster on every net both arms certified.
#
# Each bound is the Neumaier-Shcherbina lower bound of that LP from its own duals
# (``_relax.obbt._ns_safe_lp_lower_bound``, which also takes the min with the
# Rust enclosed evaluation), so it holds for the LP whatever the solver's
# accuracy; the LP's boxes and its upper triangle row are widened outward so the
# LP contains every point the network can reach. A result is intersected with
# the interval bound, so tightening can only shrink the box.
# --------------------------------------------------------------------------- #

#: Deterministic allowance for :func:`lp_tighten_bounds`, in LP nonzeros summed
#: over every LP solved. Not a clock: the bounds become big-M constants in the
#: model, so a wall deadline would make the model a function of machine speed
#: (#912). A layer whose LPs would exceed what is left keeps its interval bound,
#: and so does every layer after it.
LP_TIGHTEN_WORK_BUDGET = 50_000_000

_EPS = np.finfo(np.float64).eps


def _widen(lo: np.ndarray, hi: np.ndarray, scale: np.ndarray, n_terms: int):
    """``[lo, hi]`` widened by a float64 rounding margin for an ``n_terms``-term sum."""
    pad = 4.0 * (n_terms + 2) * _EPS * (np.abs(scale) + 1.0)
    return lo - pad, hi + pad


def _interval_layer(layer, lb: np.ndarray, ub: np.ndarray):
    """Interval pre-activation bounds of ``layer`` over ``[lb, ub]`` and their scale."""
    W = np.asarray(layer.weights, dtype=np.float64)
    b = np.asarray(layer.biases, dtype=np.float64)
    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)
    pre_lb = W_pos.T @ lb + W_neg.T @ ub + b
    pre_ub = W_pos.T @ ub + W_neg.T @ lb + b
    scale = np.abs(W).T @ np.maximum(np.abs(lb), np.abs(ub)) + np.abs(b)
    return pre_lb, pre_ub, scale


class _LayerLP:
    """The LP relaxation of layers ``0 .. K-1``, grown one layer at a time.

    Columns: the input, then per layer its pre-activation ``zhat_k`` and (for a
    non-linear activation) its post-activation ``z_k``. Rows: ``zhat_k = W_k^T
    z_{k-1} + b_k`` (equalities), and per ReLU neuron either ``z = zhat``
    (stable active), nothing beyond the box ``z in [0, 0]`` (stable inactive), or
    the triangle ``z >= zhat`` and ``z <= s (zhat - l)``, ``s = u / (u - l)``,
    with its right-hand side widened outward. Any other activation contributes
    only its interval box, which is a valid (weak) relaxation.
    """

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo = [float(v) for v in lo]
        self.hi = [float(v) for v in hi]
        self.prev = list(range(len(lo)))  # columns holding the previous layer's output
        self.ub_rows: list[tuple[dict, float]] = []
        self.eq_rows: list[tuple[dict, float]] = []

    def _col(self, lo: float, hi: float) -> int:
        self.lo.append(float(lo))
        self.hi.append(float(hi))
        return len(self.lo) - 1

    def nnz(self) -> int:
        return sum(len(r) for r, _ in self.ub_rows) + sum(len(r) for r, _ in self.eq_rows)

    def add_layer(self, layer, bounds: LayerBounds, scale: np.ndarray) -> None:
        W = np.asarray(layer.weights, dtype=np.float64)
        b = np.asarray(layer.biases, dtype=np.float64)
        p_lo, p_hi = _widen(bounds.pre_lb, bounds.pre_ub, scale, W.shape[0])
        nxt = []
        for j in range(W.shape[1]):
            zh = self._col(p_lo[j], p_hi[j])
            row = {zh: 1.0}
            for i, col in enumerate(self.prev):
                if W[i, j] != 0.0:
                    row[col] = row.get(col, 0.0) - float(W[i, j])
            self.eq_rows.append((row, float(b[j])))
            if layer.activation == Activation.LINEAR:
                nxt.append(zh)
                continue
            # The chord uses the widened column box, so it is valid over every
            # value the zhat column may take, not only the unwidened interval.
            lo_j, hi_j = float(p_lo[j]), float(p_hi[j])
            if layer.activation == Activation.RELU:
                z = self._col(max(0.0, lo_j), max(0.0, hi_j))
                if lo_j >= 0.0:
                    self.eq_rows.append(({z: 1.0, zh: -1.0}, 0.0))
                elif hi_j > 0.0:
                    self.ub_rows.append(({zh: 1.0, z: -1.0}, 0.0))  # z >= zhat
                    s = hi_j / (hi_j - lo_j)
                    rhs = -s * lo_j
                    # z - s zhat <= -s lo, widened outward: the line through
                    # (lo, 0) and (hi, hi) is computed in float64.
                    rhs += 16.0 * _EPS * (abs(rhs) + s * max(abs(lo_j), hi_j) + hi_j + 1.0)
                    self.ub_rows.append(({z: 1.0, zh: -s}, rhs))
                # hi <= 0: the column box [0, 0] already pins z.
            else:
                post_lo, post_hi = _widen(
                    bounds.post_lb[j : j + 1], bounds.post_ub[j : j + 1], np.array([1.0]), 1
                )
                z = self._col(post_lo[0], post_hi[0])
            nxt.append(z)
        self.prev = nxt

    def bound(self, weights: np.ndarray, bias: float, sense: int):
        """Proved bound on ``sense * min(sense * (weights . prev + bias))``, or None."""
        import scipy.sparse as sp
        from scipy.optimize import linprog

        from discopt._relax.obbt import _ns_safe_lp_lower_bound

        n = len(self.lo)
        c = np.zeros(n)
        for i, col in enumerate(self.prev):
            c[col] += sense * float(weights[i])

        def matrix(rows):
            data, ri, ci = [], [], []
            for r, (coef, _) in enumerate(rows):
                for col, v in coef.items():
                    ri.append(r)
                    ci.append(col)
                    data.append(v)
            return sp.csr_matrix((data, (ri, ci)), shape=(len(rows), n))

        a_ub = matrix(self.ub_rows) if self.ub_rows else None
        b_ub = np.array([r for _, r in self.ub_rows]) if self.ub_rows else None
        a_eq = matrix(self.eq_rows)
        b_eq = np.array([r for _, r in self.eq_rows])
        res = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=list(zip(self.lo, self.hi, strict=True)),
            method="highs",
        )
        if res.status != 0:
            return None  # no optimal dual to certify with: keep the interval bound
        duals = np.concatenate(
            [res.ineqlin.marginals if a_ub is not None else np.zeros(0), res.eqlin.marginals]
        )
        a_all = sp.vstack([a_ub, a_eq]) if a_ub is not None else a_eq
        b_all = np.concatenate([b_ub, b_eq]) if a_ub is not None else b_eq
        g = _ns_safe_lp_lower_bound(
            c, duals, a_all, b_all, np.array(self.lo), np.array(self.hi), n_eq=len(self.eq_rows)
        )
        if g is None:
            return None
        if sense > 0:
            return float(np.nextafter(g + bias, -np.inf))
        return float(np.nextafter(-g + bias, np.inf))


def lp_tighten_bounds(
    network: NetworkDefinition,
    *,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    work_budget: int | None = None,
    stats: dict | None = None,
) -> list[LayerBounds]:
    """Layer bounds tightened by LP over the triangle relaxation (T-N3.2).

    Same contract as :func:`propagate_bounds`, and never looser: every
    pre-activation interval is the intersection of the interval-arithmetic one
    (over the already tightened previous layer) with the proved LP bounds.
    Layer 0 is exact under interval arithmetic and is not re-solved. ``stats``,
    when given, receives ``lps`` (solved), ``declined`` (LPs that returned no
    certified bound), ``tightened`` (bounds that moved) and ``stopped_at`` (the
    first layer the work budget left on its interval bounds, or ``None``).
    """
    if input_bounds is not None:
        lb, ub = input_bounds
    elif network.input_bounds is not None:
        lb, ub = network.input_bounds
    else:
        raise ValueError("input_bounds must be set for bound propagation")
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    budget = LP_TIGHTEN_WORK_BUDGET if work_budget is None else int(work_budget)
    st = {"lps": 0, "declined": 0, "tightened": 0, "stopped_at": None}

    lp = _LayerLP(lb, ub)
    result: list[LayerBounds] = []
    cur_lb, cur_ub = lb, ub
    for k, layer in enumerate(network.layers):
        pre_lb, pre_ub, scale = _interval_layer(layer, cur_lb, cur_ub)
        n_out = pre_lb.shape[0]
        cost = 2 * n_out * (lp.nnz() + len(lp.lo))
        if k > 0 and st["stopped_at"] is None and cost <= budget:
            budget -= cost
            W = np.asarray(layer.weights, dtype=np.float64)
            b = np.asarray(layer.biases, dtype=np.float64)
            for j in range(n_out):
                for sense in (1, -1):
                    v = lp.bound(W[:, j], float(b[j]), sense)
                    st["lps"] += 1
                    if v is None:
                        st["declined"] += 1
                    elif sense > 0 and v > pre_lb[j]:
                        pre_lb[j] = v
                        st["tightened"] += 1
                    elif sense < 0 and v < pre_ub[j]:
                        pre_ub[j] = v
                        st["tightened"] += 1
                if pre_lb[j] > pre_ub[j]:
                    raise RuntimeError(
                        f"lp_tighten_bounds: layer {k} neuron {j} bounds crossed "
                        f"({pre_lb[j]!r} > {pre_ub[j]!r}); a proved LP bound cannot do "
                        "this, so the relaxation or its certificate is wrong"
                    )
        elif k > 0 and st["stopped_at"] is None:
            st["stopped_at"] = k
        post_lb, post_ub = _apply_activation_bounds(layer.activation, pre_lb, pre_ub, layer.beta)
        bounds = LayerBounds(pre_lb, pre_ub, post_lb, post_ub)
        result.append(bounds)
        if k < len(network.layers) - 1:
            lp.add_layer(layer, bounds, scale)
        cur_lb, cur_ub = post_lb, post_ub
    if stats is not None:
        stats.update(st)
    return result
