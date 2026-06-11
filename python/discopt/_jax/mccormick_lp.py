"""LP-form McCormick relaxation: bilinears lifted to aux columns, solved by HiGHS.

This is the "spatial-BB" relaxation: a polyhedral underestimator of the
nonconvex feasible set. Unlike the McCormick-NLP path in
:mod:`discopt._jax.mccormick_nlp`, it returns a globally optimal value of
the linear relaxation in one LP solve — no local minima, no warm-start
sensitivity. The trade-off is that bound information per node is loose
until spatial branching tightens variable domains.

The heavy lifting (term classification, McCormick row construction, aux
column bookkeeping) lives in :func:`build_milp_relaxation`. This module
is a thin wrapper that strips integrality on aux columns so the result is
a pure LP and exposes a simple ``compute(model, node_lb, node_ub)`` API
that fits the per-node call shape in :mod:`discopt.solver`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import NonlinearTerms, classify_nonlinear_terms
from discopt.modeling.core import Model, VarType


@dataclass
class MccormickLPResult:
    """Outcome of one LP-form McCormick relaxation solve."""

    status: str
    lower_bound: Optional[float] = None
    x: Optional[np.ndarray] = None  # first ``n_orig`` columns of the LP solution


class MccormickLPRelaxer:
    """Reusable per-node LP-form McCormick relaxation.

    Term classification and the empty :class:`DiscretizationState` are built
    once at construction time. Each call to :meth:`solve_at_node` rebuilds
    the lifted LP with the node's bound box and solves it via HiGHS.
    """

    def __init__(self, model: Model, *, superposition: bool = False) -> None:
        self._model = model
        self._terms: NonlinearTerms = classify_nonlinear_terms(model)
        # Opt-in M8 superposition cuts for bilinear-of-nonlinear products.
        self._superposition = superposition
        # Spatial-BB uses standard McCormick globally — no partitioning here.
        self._disc = DiscretizationState(partitions={})
        self._n_orig = sum(v.size for v in model._variables)
        # Pre-compute which original columns are integer/binary so that
        # integrality is preserved (only aux columns get relaxed).
        flags: list[int] = []
        for v in model._variables:
            flag = 1 if v.var_type in (VarType.BINARY, VarType.INTEGER) else 0
            flags.extend([flag] * v.size)
        self._orig_integrality = np.asarray(flags, dtype=np.int32)

    @property
    def has_bilinear(self) -> bool:
        """True if the model has any bilinear / trilinear / multilinear product."""
        return bool(self._terms.bilinear or self._terms.trilinear or self._terms.multilinear)

    def solve_at_node(
        self,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
        time_limit: Optional[float] = None,
    ) -> MccormickLPResult:
        """Solve the McCormick LP relaxation restricted to the given bound box.

        Returns a :class:`MccormickLPResult`. ``lower_bound`` is a valid lower
        bound on the original problem within this box (for minimization).
        ``x`` is the LP solution projected to the original variable columns.
        On any solver failure, ``status != "optimal"`` and the LB is ``None``.
        """
        try:
            milp, _ = build_milp_relaxation(
                self._model,
                self._terms,
                self._disc,
                bound_override=(
                    np.asarray(node_lb, dtype=np.float64),
                    np.asarray(node_ub, dtype=np.float64),
                ),
                superposition=self._superposition,
            )
        except Exception:
            return MccormickLPResult(status="error")

        # Pad original integrality flags to the lifted column count; aux cols
        # remain continuous (flag 0). If the model has no integers this is a
        # pure LP anyway.
        n_total = len(milp._c)
        if n_total > self._n_orig:
            pad = np.zeros(n_total - self._n_orig, dtype=np.int32)
            milp._integrality = np.concatenate([self._orig_integrality, pad])
        else:
            milp._integrality = self._orig_integrality
        if not int(milp._integrality.sum()):
            milp._integrality = None

        res = milp.solve(time_limit=time_limit)
        if res.objective is None or res.x is None:
            return MccormickLPResult(status=res.status)
        x_orig = np.asarray(res.x)[: self._n_orig].copy()
        return MccormickLPResult(
            status=res.status,
            lower_bound=float(res.objective),
            x=x_orig,
        )
