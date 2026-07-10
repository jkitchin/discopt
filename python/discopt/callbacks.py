"""
Callback and cut generation API for discopt's Branch & Bound solver.

Provides callback protocols that users can implement to interact with
the B&B search: adding lazy constraints, filtering incumbents, and
monitoring node processing.

Example
-------
>>> import discopt
>>> from discopt.callbacks import CallbackContext, CutResult, NodeCallback
>>>
>>> def my_logger(ctx: CallbackContext, model: discopt.Model) -> None:
...     bb = "n/a" if ctx.best_bound is None else f"{ctx.best_bound:.4f}"
...     print(f"Node {ctx.node_count}: bound={bb}")
>>>
>>> result = m.solve(node_callback=my_logger)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from discopt.modeling.core import Variable

if TYPE_CHECKING:
    from discopt.modeling.core import Model

logger = logging.getLogger(__name__)


@dataclass
class CallbackContext:
    """Information passed to callbacks during Branch & Bound.

    Attributes
    ----------
    node_count : int
        Total number of nodes explored so far.
    incumbent_obj : float or None
        Best feasible objective value found so far, or None if no
        incumbent exists yet.
    best_bound : float or None
        The certified global dual bound of the search so far (a lower bound for
        a MINIMIZE, an upper bound for a MAXIMIZE), or ``None`` when no such
        bound has been certified yet. ``None`` is reported whenever the tree
        bound is not (yet) a rigorous global bound — e.g. a node was fathomed
        non-rigorously (an NLP failure with no infeasibility proof), an
        unbounded/free root leaves the bound at ``-inf``, or the relaxation
        omits rows so no dual bound exists. Never over-reports: this value never
        exceeds what the final ``SolveResult.bound`` would certify (A1).
    gap : float or None
        Relative optimality gap, or None if no incumbent exists — and also None
        whenever ``best_bound`` is None, since a gap against a non-bound (a
        tainted tree or the failure sentinel) is meaningless (A2).
    elapsed_time : float
        Wall-clock seconds since solve started.
    x_relaxation : numpy.ndarray
        Current node's NLP relaxation solution (flat vector).
    node_bound : float
        Current node's lower bound from the NLP relaxation.
    """

    node_count: int
    incumbent_obj: float | None
    best_bound: float | None
    gap: float | None
    elapsed_time: float
    x_relaxation: np.ndarray
    node_bound: float


@dataclass
class CutResult:
    """A linear cut returned by a lazy constraint callback.

    The cut is expressed as: sum(coeff * var for var, coeff in terms) sense rhs.

    Attributes
    ----------
    terms : list of (Variable or IndexExpression, float) tuples
        Each tuple pairs a variable (or indexed variable element) with
        its coefficient in the cut. For array variables, use ``x[i]``
        as the variable. For scalar variables, use the variable directly.
    sense : str
        One of ``"<="``, ``">="``, or ``"=="``.
    rhs : float
        Right-hand side value of the cut.
    """

    terms: list  # list of (Variable or IndexExpression, float)
    sense: str
    rhs: float

    def __post_init__(self):
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(f"Invalid cut sense: {self.sense!r}. Must be '<=', '>=', or '=='.")


class LazyConstraintCallback(Protocol):
    """Protocol for lazy constraint (cut) callbacks.

    Called at each integer-feasible node. Return a list of
    :class:`CutResult` objects to add as linear constraints, or an
    empty list to accept the solution.
    """

    def __call__(
        self,
        ctx: CallbackContext,
        model: "Model",  # noqa: F821
    ) -> list[CutResult]: ...


class IncumbentCallback(Protocol):
    """Protocol for incumbent callbacks.

    Called when a new incumbent (best feasible solution) is about to be
    accepted. Return ``False`` to reject it.
    """

    def __call__(
        self,
        ctx: CallbackContext,
        model: "Model",  # noqa: F821
        solution: dict[str, np.ndarray],
    ) -> bool: ...


class NodeCallback(Protocol):
    """Protocol for node callbacks.

    Called after each batch of nodes is processed. Useful for logging
    and monitoring B&B progress.
    """

    def __call__(
        self,
        ctx: CallbackContext,
        model: "Model",  # noqa: F821
    ) -> None: ...


def cut_result_to_dense(
    cut: CutResult,
    model: "Model",  # noqa: F821
) -> tuple[np.ndarray, float, str]:
    """Convert a CutResult with Variable keys to a dense coefficient vector.

    Parameters
    ----------
    cut : CutResult
        The cut with Variable/IndexExpression -> coefficient mapping.
    model : Model
        The model whose variable ordering defines the flat layout.

    Returns
    -------
    coeffs : numpy.ndarray
        Dense coefficient vector of length ``model.num_variables``.
    rhs : float
        Right-hand side value.
    sense : str
        Constraint sense (``"<="``, ``">="``, or ``"=="``).

    Raises
    ------
    ValueError
        If a variable in the cut is not found in the model, or if an
        index is out of bounds.
    """
    from discopt.modeling.core import IndexExpression

    n_vars = model.num_variables
    coeffs = np.zeros(n_vars, dtype=np.float64)

    # Build offset map: variable id -> flat offset
    offsets: dict[int, int] = {}
    offset = 0
    for v in model._variables:
        offsets[id(v)] = offset
        offset += v.size

    for key, coeff in cut.terms:
        if isinstance(key, Variable):
            var = key
            if id(var) not in offsets:
                raise ValueError(f"Variable '{var.name}' not found in model '{model.name}'")
            flat_start = offsets[id(var)]
            # For scalar variables, set the single coefficient
            if var.size == 1:
                coeffs[flat_start] = coeff
            else:
                # For array variables with a single coefficient, broadcast
                coeffs[flat_start : flat_start + var.size] = coeff
        elif isinstance(key, IndexExpression):
            base = key.base
            if not isinstance(base, Variable):
                raise ValueError(f"IndexExpression base must be a Variable, got {type(base)}")
            if id(base) not in offsets:
                raise ValueError(f"Variable '{base.name}' not found in model '{model.name}'")
            flat_start = offsets[id(base)]
            idx = key.index
            # Compute flat index within the variable
            if isinstance(idx, (int, np.integer)):
                flat_idx = int(idx)
            elif isinstance(idx, tuple):
                # A sliced/partial subscript addresses many scalars; a cut term
                # key must name exactly one. Refuse loudly rather than let
                # np.ravel_multi_index raise a bare TypeError.
                if len(idx) != len(base.shape) or not all(
                    isinstance(i, (int, np.integer)) for i in idx
                ):
                    raise ValueError(
                        f"Non-scalar subscript {idx!r} on variable '{base.name}' "
                        f"(shape {base.shape}) cannot be a cut term key; only a "
                        "single scalar element is allowed."
                    )
                flat_idx = int(np.ravel_multi_index(idx, base.shape))
            else:
                flat_idx = int(idx)
            if flat_idx < 0 or flat_idx >= base.size:
                raise ValueError(
                    f"Index {idx} out of bounds for variable '{base.name}' with size {base.size}"
                )
            coeffs[flat_start + flat_idx] = coeff
        else:
            raise ValueError(f"Cut term key must be Variable or IndexExpression, got {type(key)}")

    return coeffs, cut.rhs, cut.sense
