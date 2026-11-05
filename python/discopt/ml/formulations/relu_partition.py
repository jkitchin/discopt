"""Stronger MILP encodings of ReLU networks than big-M (#2134).

Two formulations, both exact for the MILP (same integer-feasible set as big-M)
and both at least as tight in their LP relaxation:

* :class:`ReluPartitionFormulation` -- the partition-based formulation of
  Tsay, Kronqvist, Thebelt and Misener (NeurIPS 2021, arXiv:2102.04373; see
  also Kronqvist, Misener and Tsay, CPAIOR 2021). Each unstable neuron's
  pre-activation sum ``w.x`` is split into ``P`` partial sums
  ``s_p = sum_{i in S_p} w_i x_i``, each disaggregated over the two ReLU branches.
  ``P = 1`` is exactly big-M (and is emitted as big-M); ``P = n_in`` is the
  convex hull of the neuron over its input box.
* :class:`ReluExtendedFormulation` -- the extended (ideal) formulation of a
  single neuron from Anderson, Huchette, Ma, Tjandraatmadja and Vielma (Math.
  Prog. 2020, arXiv:1811.08359): the disjunctive hull with one copy of every
  input per branch. It is the ``P = n_in`` partition (``z_i = w_i x_i`` is a
  rescaling of the input copy ``x_i``), and is built as such.

And the projected alternative to the extended formulation:

* :class:`AndersonCutSeparator` -- Anderson et al.'s exponential family of ideal
  inequalities for big-M, separated exactly in ``O(n_in)`` per neuron, exposed as
  a :class:`~discopt.callbacks.CutCallback` (``NNFormulation(...,
  ideal_cuts=True).cut_callback``).

Soundness. Every constraint here is derived from the box the layer's input
variable carries (``in_lb``/``in_ub``: the propagated interval bounds, which are
the variable bounds the formulation itself declares), so each is valid for every
point of the embedded network -- the ideal cuts are therefore *global* cuts, not
node-local ones.

Notation for one unstable neuron ``y = max(0, w.x + b)`` with ``L < 0 < U``
(the interval bounds of ``w.x + b``) and indicator ``q``::

    LB_p = sum_{i in S_p} min(w_i l_i, w_i u_i)      UB_p = (same with max)
    q LB_p        <= zb_p        <= q UB_p             (active branch, q = 1)
    (1 - q) LB_p  <= s_p - zb_p  <= (1 - q) UB_p       (inactive branch, q = 0)
    y = sum_p zb_p + q b,   y >= 0,   y >= w.x + b

At ``q = 1`` the second row pins ``zb_p = s_p`` so ``y = w.x + b >= 0``; at
``q = 0`` the first row pins ``zb_p = 0`` so ``y = 0 >= w.x + b``: the MILP is
exactly ``y = relu(w.x + b)``. The inactive branch's ``sum_p (s_p - zb_p) +
(1 - q) b <= 0`` is identically ``y >= w.x + b`` and is written that way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, Union

import numpy as np

from discopt.ml.bounds import LayerBounds
from discopt.ml.formulations.relu_bigm import ReluBigMFormulation, ReluLayerRecord
from discopt.ml.network import NetworkDefinition
from discopt.ml.scaling import OffsetScaling

if TYPE_CHECKING:
    from discopt.callbacks import CutResult, NodeCutContext
    from discopt.modeling.core import Model, Variable

PartitionFn = Callable[[np.ndarray, int], Sequence[Sequence[int]]]

_PARTITION_STRATEGIES = ("equal_size", "contiguous")


def partition_indices(
    weights: np.ndarray,
    n_partitions: int | None,
    strategy: Union[str, PartitionFn] = "equal_size",
) -> list[np.ndarray]:
    """Split the nonzero-weight inputs of one neuron into partitions.

    Inputs whose weight is exactly zero contribute nothing to ``w.x`` and are
    left out of every partition.

    Parameters
    ----------
    weights : np.ndarray
        The neuron's incoming weights, shape ``(n_in,)``.
    n_partitions : int or None
        Requested number of partitions ``P``; clipped to the number of nonzero
        weights. ``None`` means one partition per nonzero weight (the convex
        hull / Anderson extended formulation).
    strategy : str or callable
        ``"equal_size"`` (default): sort the inputs by weight and cut the sorted
        order into ``P`` near-equal contiguous blocks, so each partition gathers
        weights of similar value (the default split of Tsay et al. 2021 and of
        OMLT). ``"contiguous"``: cut the inputs in index order. A callable
        ``f(weights, P) -> list of index arrays`` supplies its own partition; it
        must cover every nonzero-weight input exactly once.

    Returns
    -------
    list of np.ndarray
        Non-empty, disjoint index arrays whose union is the nonzero-weight
        inputs.
    """
    w = np.asarray(weights, dtype=np.float64)
    nz = np.flatnonzero(w != 0.0)
    if nz.size == 0:
        return []
    if n_partitions is None:
        return [np.array([i]) for i in nz]
    if int(n_partitions) < 1:
        raise ValueError(f"n_partitions must be >= 1, got {n_partitions}")
    P = min(int(n_partitions), int(nz.size))

    if callable(strategy):
        parts = [np.asarray(p, dtype=np.int64).ravel() for p in strategy(w, P)]
        parts = [p[w[p] != 0.0] for p in parts]
        parts = [p for p in parts if p.size]
        flat = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        if flat.size != np.unique(flat).size or set(flat.tolist()) != set(nz.tolist()):
            raise ValueError(
                "partition_strategy callable must return disjoint index sets covering "
                "every nonzero-weight input exactly once"
            )
        return parts
    if strategy == "equal_size":
        order = nz[np.argsort(w[nz], kind="stable")]
    elif strategy == "contiguous":
        order = nz
    else:
        raise ValueError(
            f"Unknown partition_strategy {strategy!r}; choose from "
            f"{list(_PARTITION_STRATEGIES)} or pass a callable"
        )
    return [p for p in np.array_split(order, P) if p.size]


class ReluPartitionFormulation(ReluBigMFormulation):
    """Partition-based ReLU formulation (Tsay et al., NeurIPS 2021).

    Identical to :class:`~discopt.ml.formulations.relu_bigm.ReluBigMFormulation`
    except for unstable ReLU neurons, which get ``P`` disaggregated partial sums
    instead of the two big-M rows (see the module docstring). Stable neurons
    (``pre_lb >= 0`` or ``pre_ub <= 0``, as :func:`~discopt.ml.detect_dead_relus`
    classifies them) get no binary, as in big-M. A neuron whose effective ``P``
    is 1 is emitted as big-M, which is the same formulation.

    Parameters
    ----------
    n_partitions : int or None
        ``P`` per neuron (default 2), clipped to the neuron's number of nonzero
        weights. ``None`` gives the convex hull (:class:`ReluExtendedFormulation`).
    partition_strategy : str or callable
        See :func:`partition_indices`.
    """

    def __init__(
        self,
        model: Model,
        network: NetworkDefinition,
        prefix: str,
        scaling: OffsetScaling | None,
        n_partitions: int | None = 2,
        partition_strategy: Union[str, PartitionFn] = "equal_size",
    ) -> None:
        super().__init__(model, network, prefix, scaling)
        if n_partitions is not None and int(n_partitions) < 1:
            raise ValueError(f"n_partitions must be >= 1, got {n_partitions}")
        if not callable(partition_strategy) and partition_strategy not in _PARTITION_STRATEGIES:
            raise ValueError(
                f"Unknown partition_strategy {partition_strategy!r}; choose from "
                f"{list(_PARTITION_STRATEGIES)} or pass a callable"
            )
        self._n_partitions = n_partitions
        self._partition_strategy = partition_strategy

    def _relu_layer(
        self,
        m: Model,
        zhat: Variable,
        prev_z: Variable,
        W: np.ndarray,
        b: np.ndarray,
        in_lb: np.ndarray,
        in_ub: np.ndarray,
        bounds: LayerBounds,
        k: int,
    ) -> Variable:
        pfx = self._prefix
        n_out = W.shape[1]
        pre_lb = bounds.pre_lb
        pre_ub = bounds.pre_ub
        z = m.continuous(
            f"{pfx}_z_{k}",
            shape=(n_out,),
            lb=np.maximum(pre_lb, 0.0),
            ub=np.maximum(pre_ub, 0.0),
        )
        in_lb = np.asarray(in_lb, dtype=np.float64)
        in_ub = np.asarray(in_ub, dtype=np.float64)

        for j in range(n_out):
            lb_j = float(pre_lb[j])
            ub_j = float(pre_ub[j])
            if lb_j >= 0:
                m.subject_to(z[j] == zhat[j], name=f"{pfx}_relu_active_{k}_{j}")
                continue
            if ub_j <= 0:
                m.subject_to(z[j] == 0, name=f"{pfx}_relu_inactive_{k}_{j}")
                continue

            q = m.binary(f"{pfx}_q_{k}_{j}")
            m.subject_to(z[j] >= 0, name=f"{pfx}_relu_lb_{k}_{j}")
            m.subject_to(z[j] >= zhat[j], name=f"{pfx}_relu_ge_{k}_{j}")

            w = W[:, j]
            parts = partition_indices(w, self._n_partitions, self._partition_strategy)
            if len(parts) <= 1:
                # P = 1 IS big-M (Tsay et al. 2021): same rows, no auxiliaries.
                m.subject_to(z[j] <= ub_j * q, name=f"{pfx}_relu_ub_{k}_{j}")
                m.subject_to(
                    z[j] <= zhat[j] - lb_j * (1 - q),
                    name=f"{pfx}_relu_bigm_{k}_{j}",
                )
                continue

            wl = w * in_lb
            wu = w * in_ub
            term_lo = np.minimum(wl, wu)
            term_hi = np.maximum(wl, wu)
            LB = np.array([float(term_lo[p].sum()) for p in parts])
            UB = np.array([float(term_hi[p].sum()) for p in parts])
            zb = m.continuous(
                f"{pfx}_zb_{k}_{j}",
                shape=(len(parts),),
                lb=np.minimum(LB, 0.0),
                ub=np.maximum(UB, 0.0),
            )
            for p, idx in enumerate(parts):
                s_p = sum(float(w[i]) * prev_z[int(i)] for i in idx)
                tag = f"{k}_{j}_{p}"
                m.subject_to(zb[p] >= LB[p] * q, name=f"{pfx}_part_on_lo_{tag}")
                m.subject_to(zb[p] <= UB[p] * q, name=f"{pfx}_part_on_hi_{tag}")
                m.subject_to(s_p - zb[p] >= LB[p] * (1 - q), name=f"{pfx}_part_off_lo_{tag}")
                m.subject_to(s_p - zb[p] <= UB[p] * (1 - q), name=f"{pfx}_part_off_hi_{tag}")
            m.subject_to(
                z[j] == sum(zb[p] for p in range(len(parts))) + float(b[j]) * q,
                name=f"{pfx}_part_out_{k}_{j}",
            )

        self._record_relu_layer(k, z, prev_z, W, b, in_lb, in_ub, bounds)
        return z


class ReluExtendedFormulation(ReluPartitionFormulation):
    """Anderson et al. (2020) extended formulation: the per-neuron convex hull.

    The ``P = n_in`` partition: one disaggregated term per nonzero weight. Its
    LP relaxation of each unstable neuron is the convex hull of
    ``{(x, y): y = relu(w.x + b), x in box}``, the tightest any single-neuron
    formulation can be; the price is ``n_in`` auxiliaries and ``4 n_in`` rows per
    unstable neuron. Big-M plus :class:`AndersonCutSeparator` reaches the same
    hull in the original space, adding only the inequalities a node violates.
    """

    def __init__(
        self,
        model: Model,
        network: NetworkDefinition,
        prefix: str,
        scaling: OffsetScaling | None,
    ) -> None:
        super().__init__(model, network, prefix, scaling, n_partitions=None)


class AndersonCutSeparator:
    """Exact separation of Anderson et al.'s (2020) ideal ReLU inequalities.

    For an unstable neuron ``y = max(0, w.x + b)`` over the box ``l <= x <= u``,
    with ``Lc_i = l_i`` and ``Uc_i = u_i`` when ``w_i >= 0`` (swapped when
    ``w_i < 0``), Theorem 1 of Anderson et al. gives the convex hull as
    ``y >= 0``, ``y >= w.x + b``, ``0 <= q <= 1`` and, for every ``I`` subset of
    the inputs::

        y <= sum_{i in I} w_i (x_i - Lc_i (1 - q)) + (b + sum_{i not in I} w_i Uc_i) q

    ``I = all`` is the big-M row ``y <= w.x + b - L (1 - q)`` and ``I = {}`` is
    ``y <= U q``. Given a point, the most violated member is found term by term
    (``i in I`` iff ``w_i (x_i - Lc_i (1 - q)) < w_i Uc_i q``), so separation is
    ``O(n_in)`` per neuron. The box is the one the input variable carries, so
    every cut is globally valid -- which is what a ``CutResult`` must be.

    Use as ``model.solve(cut_callback=nn.cut_callback)``. On a pure MILP the
    solver recognises this separator (:meth:`native_milp_spec`) and separates the
    same inequalities natively in Rust (#2157): on the default HiGHS route over
    repeated root LP solves, the violated rows then handed to HiGHS with the
    MILP; on the monolithic Rust MILP engine at every fractional node. Any other
    ``cut_callback`` -- including this object wrapped in a function -- still
    moves the MILP onto discopt's spatial branch and bound, which calls it from
    Python at every node.

    Parameters
    ----------
    layers : list of ReluLayerRecord
        The embedded ReLU layers (``ReluBigMFormulation.relu_layers``).
    tol : float
        Minimum violation ``y* - rhs(x*, q*)`` for a cut to be returned,
        relative to ``max(1, |rhs|)``.
    """

    def __init__(self, layers: Sequence[ReluLayerRecord], tol: float = 1e-6) -> None:
        self._layers = list(layers)
        self._tol = float(tol)
        self.n_calls = 0
        self.n_cuts = 0

    @staticmethod
    def _offsets(model: Model) -> dict[int, int]:
        offsets: dict[int, int] = {}
        off = 0
        for v in model._variables:
            offsets[id(v)] = off
            off += v.size
        return offsets

    def separate(self, x_flat: np.ndarray, model: Model) -> list[CutResult]:
        """Return the most violated ideal inequality of every neuron that has one."""
        from discopt.callbacks import CutResult

        x_flat = np.asarray(x_flat, dtype=np.float64).ravel()
        offsets = self._offsets(model)
        cuts: list[CutResult] = []
        for rec in self._layers:
            zo = offsets[id(rec.z)]
            xo = offsets[id(rec.x)]
            n_in = rec.weights.shape[0]
            xs = x_flat[xo : xo + n_in]
            for j, q in enumerate(rec.binaries):
                if q is None:
                    continue  # stable neuron: encoded exactly, nothing to cut
                w = rec.weights[:, j]
                bj = float(rec.biases[j])
                pos = w >= 0
                Lc = np.where(pos, rec.in_lb, rec.in_ub)
                Uc = np.where(pos, rec.in_ub, rec.in_lb)
                qs = float(x_flat[offsets[id(q)]])
                ys = float(x_flat[zo + j])
                in_term = w * (xs - Lc * (1.0 - qs))
                out_term = w * Uc * qs
                in_I = in_term < out_term
                rhs_val = float(np.where(in_I, in_term, out_term).sum()) + bj * qs
                if ys - rhs_val <= self._tol * max(1.0, abs(rhs_val)):
                    continue
                # y - sum_I w_i x_i - q (b + sum_{~I} w_i Uc_i + sum_I w_i Lc_i)
                #   <= - sum_I w_i Lc_i
                q_coef = -(bj + float((w * Uc)[~in_I].sum()) + float((w * Lc)[in_I].sum()))
                terms: list = [(rec.z[j], 1.0), (q, q_coef)]
                for i in np.flatnonzero(in_I & (w != 0.0)):
                    terms.append((rec.x[int(i)], -float(w[i])))
                rhs = -float((w * Lc)[in_I].sum())
                cuts.append(CutResult(terms=terms, sense="<=", rhs=rhs))
        self.n_cuts += len(cuts)
        return cuts

    def __call__(self, ctx: NodeCutContext, model: Model) -> list[CutResult]:
        self.n_calls += 1
        return self.separate(ctx.x_relaxation, model)

    def native_milp_spec(self, model: Model) -> dict[str, np.ndarray] | None:
        """The neurons as flat column-index arrays for the Rust separator (#2157).

        A pure MILP whose ``cut_callback`` is this separator has the inequalities
        separated by the native ``discopt._rust.ReluIdealSeparator`` -- at the
        root on the HiGHS route, at every fractional node on the Rust MILP
        engine -- instead of moving to the Python spatial tree. Columns are
        offsets into the model's flattened variables, which are the engines'
        structural columns. The input box is
        deliberately *not* included: the solver takes it from the columns' own
        bounds at solve time, so a cut can never rest on a box the model does not
        enforce.

        Returns ``None`` when a recorded variable is not in ``model`` (a
        separator built for a different model must not address this one's
        columns); a network with no unstable neuron gives empty arrays.
        """
        offsets = self._offsets(model)
        y_cols: list[int] = []
        q_cols: list[int] = []
        bias: list[float] = []
        ptr = [0]
        x_cols: list[int] = []
        w_all: list[float] = []
        for rec in self._layers:
            if id(rec.z) not in offsets or id(rec.x) not in offsets:
                return None
            zo = offsets[id(rec.z)]
            xo = offsets[id(rec.x)]
            n_in = rec.weights.shape[0]
            for j, q in enumerate(rec.binaries):
                if q is None:
                    continue
                if id(q) not in offsets:
                    return None
                w = np.asarray(rec.weights[:, j], dtype=np.float64)
                nz = np.flatnonzero(w != 0.0)
                y_cols.append(zo + j)
                q_cols.append(offsets[id(q)])
                bias.append(float(rec.biases[j]))
                x_cols.extend(int(xo + i) for i in nz if i < n_in)
                w_all.extend(float(w[i]) for i in nz)
                ptr.append(len(x_cols))
        return {
            "y": np.asarray(y_cols, dtype=np.int64),
            "q": np.asarray(q_cols, dtype=np.int64),
            "bias": np.asarray(bias, dtype=np.float64),
            "ptr": np.asarray(ptr, dtype=np.int64),
            "x": np.asarray(x_cols, dtype=np.int64),
            "w": np.asarray(w_all, dtype=np.float64),
        }

    def record_native_counts(self, calls: int, cuts: int) -> None:
        """Fold the Rust separator's executed-call / added-cut counts in (#2157)."""
        self.n_calls += int(calls)
        self.n_cuts += int(cuts)


__all__ = [
    "AndersonCutSeparator",
    "ReluExtendedFormulation",
    "ReluPartitionFormulation",
    "partition_indices",
]
