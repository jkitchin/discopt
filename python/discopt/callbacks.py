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
    #: ``"global"`` (the default and, today, the only accepted value) means the cut
    #: is valid everywhere in the feasible region. ``"local"`` would mean "valid
    #: only in the node's box" — and is REFUSED, because discopt's cut pool is
    #: applied to every node through ``_AugmentedEvaluator``, so there is nowhere
    #: to put a subtree-scoped cut. Silently treating a local cut as global is a
    #: false-certificate generator, which is why this refuses instead (#1278 D).
    scope: str = "global"

    def __post_init__(self):
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(f"Invalid cut sense: {self.sense!r}. Must be '<=', '>=', or '=='.")
        if self.scope == "local":
            raise ValueError(
                "CutResult(scope='local'): subtree-scoped cuts are not supported. "
                "discopt's cut pool is applied at EVERY node, so a local cut has "
                "nowhere to live and treating it as global would cut a feasible "
                "region out of the whole tree. Return the cut only where it is "
                "globally valid, or tighten the model instead."
            )
        if self.scope != "global":
            raise ValueError(
                f"Invalid cut scope: {self.scope!r}. Must be 'global' (the only supported scope)."
            )


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


# --------------------------------------------------------------------------- #
# Node cut callback (#1248 component D, via #1278)
# --------------------------------------------------------------------------- #
class CutValidationError(ValueError):
    """A user cut was shown to exclude a point known to be feasible.

    Raised, not logged-and-dropped. A GLOBAL cut applies at every node, so an
    invalid one removes the optimum from the whole tree and the solve returns a
    false ``optimal`` — the single worst failure this codebase has. Dropping it
    silently would also leave the caller believing their cut was applied. See
    :func:`validate_cut_against_witnesses`.
    """


@dataclass
class NodeCutContext:
    """What a :class:`CutCallback` is told about the node it is cutting.

    Unlike :class:`CallbackContext`, which describes the search, this describes
    ONE node — including its box, which is what a spatial cut needs. A plugin
    supplying tangent-plane cuts of a Gibbs energy surface, for instance, wants
    the box to pick the linearization point.

    Attributes
    ----------
    node_id : int
        The B&B tree's id for this node.
    node_lb, node_ub : numpy.ndarray
        The node's variable box, flat, length ``model.num_variables``. A cut
        valid only on this box is a *local* cut — see :class:`CutResult.scope`,
        which does not accept one yet.
    x_relaxation : numpy.ndarray
        The node's relaxation solution, flat. This is the point a separating cut
        would normally be asked to cut off.
    node_bound : float
        The node's relaxation objective, in the solver's INTERNAL (minimize)
        sense — the same convention :attr:`CallbackContext.node_bound` uses, so
        a maximize model reports the negated value. ``incumbent_obj`` and
        ``best_bound`` are in the model's own sense.
    incumbent_obj : float or None
        Best feasible objective so far, or None.
    best_bound : float or None
        Certified global dual bound so far, or None when none is certified (the
        same contract as :attr:`CallbackContext.best_bound`).
    node_count : int
        Nodes explored so far.
    elapsed_time : float
        Wall-clock seconds since the solve started.
    """

    node_id: int
    node_lb: np.ndarray
    node_ub: np.ndarray
    x_relaxation: np.ndarray
    node_bound: float
    incumbent_obj: float | None
    best_bound: float | None
    node_count: int
    elapsed_time: float


class CutCallback(Protocol):
    """Protocol for a cut callback invoked at **every** node.

    Unlike :class:`LazyConstraintCallback`, which runs only at integer-feasible
    nodes, this runs at every node the search evaluates, spatial nodes included.
    Return a list of :class:`CutResult` (possibly empty).

    **Every returned cut is validated before it is accepted** (see
    :func:`validate_cut_against_witnesses`); a cut that excludes a known feasible
    point raises :class:`CutValidationError` and aborts the solve.
    """

    def __call__(
        self,
        ctx: NodeCutContext,
        model: "Model",  # noqa: F821
    ) -> list[CutResult]: ...


def validate_cut_against_witnesses(
    cut: CutResult,
    model: "Model",  # noqa: F821
    witnesses,
    tol: float = 1e-6,
) -> tuple[int, float]:
    """Check ``cut`` against points already known to be feasible for the model.

    ``witnesses`` is an iterable of flat solution vectors the solver has verified
    feasible (incumbents, and relaxation solutions that passed the constraint
    feasibility test). Each is a point the true optimum may legitimately be at or
    beyond, so a valid global cut must not exclude any of them.

    What this is, and what it is not
    --------------------------------
    This is a **filter, not a proof**. A cut the solver cannot derive is an
    assertion by the caller; no finite set of witnesses can establish that it is
    valid everywhere. What the gate does is catch the errors that actually happen
    — a sign flip, a sense the wrong way round, an rhs off by the objective
    offset — before they poison the tree. #1248 asked for validation against
    sampled feasible points and for a loud refusal on a violator; it is mandatory
    here rather than opt-in, because a global cut is exactly the flag that can
    poison the whole search.

    Returns
    -------
    (n_checked, worst_slack)
        How many witnesses were actually tested, and the worst signed violation
        seen (``<= 0`` means every witness satisfied the cut). **A caller must
        look at ``n_checked``**: zero means the gate was vacuous, not that the cut
        is sound (CLAUDE.md §6).

    Raises
    ------
    CutValidationError
        If any witness violates the cut by more than ``tol`` scaled by the row's
        magnitude.
    """
    coeffs, rhs, sense = cut_result_to_dense(cut, model)
    n_checked = 0
    worst = -np.inf
    for x in witnesses:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != coeffs.shape[0] or not np.all(np.isfinite(x)):
            continue
        lhs = float(coeffs @ x)
        scale = max(1.0, float(np.abs(coeffs) @ np.abs(x)), abs(rhs))
        if sense == "<=":
            slack = lhs - rhs
        elif sense == ">=":
            slack = rhs - lhs
        else:  # "=="
            slack = abs(lhs - rhs)
        slack /= scale
        worst = max(worst, slack)
        n_checked += 1
        if slack > tol:
            raise CutValidationError(
                f"a {sense!r} cut with rhs={rhs!r} excludes a point this solve has "
                f"already verified FEASIBLE (violation {slack:.3e} relative, "
                f"tolerance {tol:.1e}). A global cut applies at every node, so "
                "accepting this one would remove a feasible region from the whole "
                "tree and the solve could return a false optimum. Fix the cut, or "
                "return it only where it is valid."
            )
    return n_checked, (worst if n_checked else float("-inf"))
