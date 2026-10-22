"""Pattern-based nonlinear bound tightening shared across solver frontends.

These rules complement the existing linear FBBT and LP-based OBBT paths.
Each rule must be sound with respect to the current variable box and may only
tighten bounds. The runner clips every rule's output against the current box so
future rules can be added without changing solver-side plumbing.
"""

from __future__ import annotations

import inspect
import logging
import os
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, NoReturn, Optional, Sequence, TypeVar

import numpy as np

from discopt._flat_index import flat_index_in_shape
from discopt._relax._directed import (
    add_down,
    add_up,
    affine_preimage,
    div_down,
    div_up,
    lib_down,
    lib_up,
    mul_down,
    mul_up,
    sqrt_down,
    sqrt_up,
    sub_down,
    sub_up,
)
from discopt._relax._numeric import (
    ROUNDOFF_OPS,
    roundoff_slack,
    roundoff_slack_arr,
)
from discopt._relax._numeric import (
    is_effectively_finite as _is_effectively_finite,
)
from discopt._relax.quadratic_form import (
    extract_quadratic_support,
    polynomial_degree_bound,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    CustomCall,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

logger = logging.getLogger(__name__)

is_effectively_finite = _is_effectively_finite

_ScaledMatch = TypeVar("_ScaledMatch")

# Feasibility tolerance for declaring a tightened variable box empty. Bound
# tightening must never certify infeasibility from a crossover smaller than the
# solver's absolute feasibility tolerance: such a box is feasible *within
# tolerance*, and pruning it can discard a genuine optimum. This matters for
# approximate reformulations such as the GDP hull perspective, whose eps-clamped
# form leaves an O(eps) residual that would otherwise read as a hard
# infeasibility (issue #27a). A box that is empty by less than this tolerance is
# snapped to a degenerate (lb == ub) box and left for the node NLP to validate;
# a crossover larger than the tolerance is a genuine infeasibility and is pruned.
_EMPTY_INTERVAL_FEAS_TOL = 1e-6
#: Integrality tolerance (conftest ``integrality=1e-5``) for the empty-integer-box proof.
_INTEGER_BOX_TOL = 1e-5
_FLOAT_EPS = float(np.finfo(np.float64).eps)

# #1397: the round-off bound and its operation count now live in ``_numeric`` so the
# other scale-exposed yardsticks in this layer (``factorable_reform``'s denominator
# sign gate, the reduced-cost deadbands) share one definition and one justification.
# The module-private aliases are kept because this file's rules read better with them
# and the #1397 regression suite pins them.
_CROSSOVER_OPS = ROUNDOFF_OPS
_roundoff_slack = roundoff_slack
_roundoff_slack_arr = roundoff_slack_arr


@dataclass(frozen=True)
class FlatVariableMetadata:
    """Flat indexing metadata shared by nonlinear tightening rules."""

    base_offsets: dict[int, int]
    flat_var_types: tuple[VarType, ...]

    def scalar_flat_index(self, expr) -> Optional[int]:
        """Return the flat scalar index for a scalar variable expression."""
        if isinstance(expr, Variable):
            if expr.size != 1:
                return None
            return self.base_offsets[id(expr)]

        if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
            base = expr.base
            base_offset = self.base_offsets[id(base)]
            idx = expr.index
            if base.shape == ():
                flat_idx = 0
            else:
                if not isinstance(idx, tuple):
                    idx = (idx,)
                # Only a fully scalar subscript (one integer per axis) addresses a
                # single flat entry. Slices / None / partial indices (e.g. the
                # vectorized ``var[:, 1:]`` expressions the DAE transcriber emits)
                # cover many scalars, so no flat index exists for them.
                if len(idx) != len(base.shape) or not all(
                    isinstance(i, (int, np.integer)) for i in idx
                ):
                    return None
                # #941: `% s` normalizes in-range negatives correctly but also
                # *wraps* an out-of-range index onto a valid slot (7 % 4 == 3),
                # and accepts `bool` as an index. Refusing is the sound answer.
                normalized_flat = flat_index_in_shape(tuple(idx), tuple(base.shape))
                if normalized_flat is None:
                    return None
                flat_idx = normalized_flat
            return base_offset + flat_idx

        return None


def build_flat_variable_metadata(model: Model) -> FlatVariableMetadata:
    """Build flat-variable indexing metadata for a model."""
    base_offsets: dict[int, int] = {}
    flat_var_types: list[VarType] = []
    offset = 0
    for var in model._variables:
        base_offsets[id(var)] = offset
        flat_var_types.extend([var.var_type] * var.size)
        offset += var.size
    return FlatVariableMetadata(base_offsets=base_offsets, flat_var_types=tuple(flat_var_types))


def _model_structural_token(model: Model) -> tuple[int, int, int, int]:
    """Identity token for a model's constraint set, used to validate caches.

    Nonlinear tightening runs once per branch-and-bound node on the *same* model
    with only the numeric variable box changing; the symbolic constraint DAG is
    invariant. This token detects the rare case of the model being mutated
    (constraints rebuilt, added, or removed) between calls so that structural
    caches are dropped rather than reused stale.
    """
    constraints = model._constraints
    # #1514: the variable list is part of the structure too. ``metadata`` maps each
    # variable to its flat offset, and a caller that adds a variable and then
    # restores the constraint list (``epsilon_constraint`` does) left a token equal
    # to the cached one over a stale offset map -> ``KeyError`` in a rule. A
    # catch-all at the call site used to turn that into "no tightening".
    variables = model._variables
    return (id(constraints), len(constraints), id(variables), len(variables))


def _get_struct_cache(model: Model) -> dict:
    """Return (building if needed) the per-model structural cache.

    Holds bound-independent decompositions reused across B&B nodes: the flat
    variable metadata and the additive flattening of constraint bodies. The cache
    is keyed to a structural token and rebuilt if the model's constraint set
    changes. If the model cannot store an attribute, a fresh (non-persistent)
    cache is returned so callers degrade gracefully to recomputation.
    """
    token = _model_structural_token(model)
    cache = getattr(model, "_nl_struct_cache", None)
    if cache is None or cache.get("token") != token:
        cache = {
            "token": token,
            "flatten": {},
            "rowstruct": {},
            "metadata": build_flat_variable_metadata(model),
        }
        # #1520: narrowed. ``setattr`` refuses an object that cannot carry the
        # attribute with ``AttributeError`` -- the docstring's documented
        # degradation (a fresh cache per call, same results). Anything else is a
        # defect and propagates.
        try:
            setattr(model, "_nl_struct_cache", cache)
        except AttributeError as exc:
            # Non-persistent cache -> recomputation per call, which is a real cost
            # the profile would otherwise attribute to the decomposition itself.
            logger.debug("FBBT structural cache not persisted: %s: %s", type(exc).__name__, exc)
    return cache


def _cached_flat_metadata(model: Model) -> FlatVariableMetadata:
    """Return per-model flat-variable metadata, cached across B&B nodes."""
    metadata: FlatVariableMetadata = _get_struct_cache(model)["metadata"]
    return metadata


def _cached_flat_terms(model: Model, expr) -> list[tuple[float, object]]:
    """Return the additive flattening of ``expr`` at unit scale, cached per model.

    ``_flatten_sum`` walks the symbolic expression DAG with isinstance dispatch on
    every node. That structure is invariant across B&B nodes (only the numeric box
    changes), so the result is memoized keyed by expression identity. The returned
    list is identical to a fresh ``_flatten_sum`` call — exact-math-preserving.

    Callers MUST treat the returned list as read-only; it is shared with the cache.
    All current rules iterate it read-only.
    """
    cache = _get_struct_cache(model)
    flatten = cache["flatten"]
    key = id(expr)
    terms: Optional[list[tuple[float, object]]] = flatten.get(key)
    if terms is None:
        terms = []
        _flatten_sum(expr, 1.0, terms)
        flatten[key] = terms
    return terms


_ROWSTRUCT_MISS = object()


def _cached_row_structure(model: Model, rule_name: str, constraint, metadata, build):
    """Memoize a rule's *bound-independent* decomposition of one constraint row.

    A rule's row scan splits into two halves: deriving the row's algebraic
    shape (which terms are squares, which are linear, their coefficients) and
    then intersecting that shape against the current box. Only the second half
    reads ``flat_lb``/``flat_ub``; the first depends solely on the expression
    DAG and the flat-variable metadata, both invariant across B&B nodes -- the
    same invariant ``_cached_flat_terms`` already relies on.

    Measured on portfol_classical050_1 (#1066), that first half dominated:
    10.5M ``_constant_value`` calls and 55.6M ``isinstance`` calls, 15.9 s of a
    46.6 s solve, redone identically at every one of 160 tightening passes.

    The whole decomposition is cached rather than the individual term matches,
    because a per-term cache would have to re-apply each term's scale *after*
    the lookup. Threading the scale through the recursion computes
    ``(s * c1) * c2`` while a unit-scale cache would compute ``s * (c1 * c2)``;
    those differ in the last ulp, which would move a bound and violate the
    CLAUDE.md 5 bound-neutrality requirement. Caching the finished dicts
    reproduces the original arithmetic bit-for-bit.

    ``build`` must return the decomposition, or ``None`` when the row does not
    match the rule -- a non-match is cached too, since it is just as invariant.

    The decomposition depends on ``metadata`` as well as on the row, and
    ``metadata`` reaches a rule as a *parameter*. The production driver always
    passes ``_cached_flat_metadata(model)`` -- the very object this cache is
    stored beside, invalidated by the same structural token -- but a caller that
    builds its own metadata (a test exercising a rule directly) would otherwise
    read entries derived from a different one. Such a caller bypasses the cache
    rather than being trusted to match it.
    """
    body = getattr(constraint, "body", None)
    if body is None:
        # #1514: a non-algebraic row (a disjunction, a logical or SOS constraint)
        # has no body to decompose. Every rule using this cache is row-independent
        # (``row_scan_is_anytime``), so skipping the row -- ``None`` is each rule's
        # "no structure here" -- only forgoes a tightening. It used to raise
        # ``AttributeError`` and abort the whole pass.
        return None
    cache = _get_struct_cache(model)
    if metadata is not cache["metadata"]:
        return build()
    rowstruct = cache["rowstruct"]
    key = (rule_name, id(body))
    value = rowstruct.get(key, _ROWSTRUCT_MISS)
    if value is _ROWSTRUCT_MISS:
        value = build()
        rowstruct[key] = value
    return value


@dataclass(frozen=True)
class NonlinearBoundTighteningStats:
    """Summary of a nonlinear tightening pass."""

    n_tightened: int
    applied_rules: tuple[str, ...]
    infeasible: bool = False
    infeasibility_reason: Optional[str] = None
    #: True when the pass stopped early because its ``deadline`` passed, so the
    #: returned box is a valid but *incomplete* tightening (fewer rules ran, or a
    #: rule saw only a prefix of the constraints). Never set when no deadline was
    #: given. Purely informational — the box itself is sound either way.
    deadline_reached: bool = False


# Constraints between two ``time.perf_counter()`` reads inside a rule's row scan.
# The read is ~50 ns and a row's match attempt is orders of magnitude more, so the
# stride is free; it exists so the poll cannot be starved by a cheap-row model.
# Granularity matters more than the poll's existence: #868 found ``probing`` polling
# its deadline once per binary on an instance with 7 binaries and 107k constraints,
# which is a poll that never fires in time.
_DEADLINE_POLL_STRIDE = 64


def _cached_model_structure(model: Model, key: str, metadata, build):
    """Memoize a rule's *bound-independent* scan of the whole model.

    The row-level sibling :func:`_cached_row_structure` keys on one constraint
    body; a rule whose conclusion depends on what the model does *not* contain
    (``PeriodicVariableBoundRule``) must scan the objective and every row before
    it can conclude anything, so its unit of caching is the model.

    Same metadata guard, for the same reason: the scan resolves expressions to
    flat indices through ``metadata``, so a caller supplying its own bypasses the
    cache rather than being trusted to match it.
    """
    cache = _get_struct_cache(model)
    if metadata is not cache["metadata"]:
        return build()
    value = cache.get(key, _ROWSTRUCT_MISS)
    if value is _ROWSTRUCT_MISS:
        value = build()
        cache[key] = value
    return value


def _rows_until(model: Model, deadline: Optional[float]):
    """Iterate ``model._constraints``, stopping once ``deadline`` passes.

    ``deadline`` is an absolute ``time.perf_counter()`` value, or ``None`` for the
    unbounded legacy behaviour (the constraint list is then returned as-is, so a
    budget-free call has *zero* added cost and is byte-identical to before).

    Every rule in this module only ever *intersects* the incoming box, and each
    constraint's inference is independently valid, so consuming a prefix of the rows
    yields a valid — merely looser — box. Stopping early is a "do less" bail, never
    a "do wrong" one (issue #875; same contract as the Rust presolve passes' `_until`
    variants from #868).
    """
    if deadline is None:
        return model._constraints
    return _iter_rows_until(model._constraints, deadline)


def _iter_rows_until(rows, deadline: float):
    for i, row in enumerate(rows):
        if i % _DEADLINE_POLL_STRIDE == 0 and time.perf_counter() >= deadline:
            return
        yield row


class NonlinearBoundTighteningInfeasible(ValueError):
    """Raised internally when a sound tightening rule proves infeasibility."""


class _ReciprocalIntervalInfeasible:
    """Sentinel for reciprocal interval propagation that proves infeasibility."""


_RECIPROCAL_INTERVAL_INFEASIBLE = _ReciprocalIntervalInfeasible()


class NonlinearBoundTighteningRule:
    """Base class for extensible nonlinear bound tightening rules.

    A rule MAY declare a trailing ``deadline: Optional[float] = None`` parameter on
    :meth:`tighten`; :func:`tighten_nonlinear_bounds` detects that and passes the
    absolute ``time.perf_counter()`` budget through, so the rule can stop scanning
    rows (via :func:`_rows_until`) instead of running the whole constraint list. A
    rule without the parameter keeps the four-argument contract exactly and is
    simply called without it — external rules do not need to change.
    """

    name = "unnamed_rule"

    #: True only when consuming a PREFIX of ``model._constraints`` yields a valid
    #: (merely looser) box — i.e. every row's inference stands on its own and the
    #: rule draws no conclusion from a row being *absent*.
    #:
    #: Default ``False``, deliberately: a rule that accumulates across rows and then
    #: acts on the complement (``PeriodicVariableBoundRule`` restricts a variable to
    #: one period precisely because nothing *else* in the model uses it) would cut
    #: feasible points if a truncated scan missed the disqualifying row. Under a
    #: budget such a rule is skipped whole rather than run on a prefix, so the safe
    #: default is the one a new rule inherits without thinking about it.
    row_scan_is_anytime: bool = False

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


_RULE_ACCEPTS_DEADLINE: dict[type, bool] = {}


def _rule_accepts_deadline(rule: NonlinearBoundTighteningRule) -> bool:
    """True when ``rule.tighten`` declares a ``deadline`` parameter (memoized).

    Signature inspection rather than a try/except around the call: a ``TypeError``
    raised from *inside* a rule must not be mistaken for "this rule has the old
    signature" and silently retried unbudgeted.
    """
    cls = type(rule)
    hit = _RULE_ACCEPTS_DEADLINE.get(cls)
    if hit is None:
        try:
            params = inspect.signature(cls.tighten).parameters
            hit = "deadline" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError) as exc:
            logger.debug("cannot inspect %s.tighten, assuming no deadline: %s", cls.__name__, exc)
            hit = False
        _RULE_ACCEPTS_DEADLINE[cls] = hit
    return hit


def _constant_value(expr) -> Optional[float]:
    if not isinstance(expr, Constant):
        return None
    values = np.asarray(expr.value, dtype=np.float64).ravel()
    if values.size != 1:
        return None
    return float(values[0])


class _ScaledExpressionMatcher:
    """Shared scale/negation matcher for rule-specific expression leaves."""

    @staticmethod
    def match(
        expr,
        scale: float,
        leaf_match: Callable[[object, float], Optional[_ScaledMatch]],
    ) -> Optional[_ScaledMatch]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return _ScaledExpressionMatcher.match(expr.operand, -scale, leaf_match)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return _ScaledExpressionMatcher.match(expr.right, scale * left_const, leaf_match)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return _ScaledExpressionMatcher.match(expr.left, scale * right_const, leaf_match)
            # Neither operand is constant, so there is no scale left to peel — but
            # the product may still be a leaf this rule recognizes (``x*x`` is a
            # square; #1395). Offer it to ``leaf_match`` instead of rejecting it
            # here. A leaf hook that does not want a product returns ``None``, which
            # is exactly the old behaviour.
            return leaf_match(expr, scale)

        return leaf_match(expr, scale)


def _match_scaled_linear_var(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[int, float]]:
    def leaf_match(leaf: object, leaf_scale: float) -> Optional[tuple[int, float]]:
        flat_idx = metadata.scalar_flat_index(leaf)
        if flat_idx is None:
            return None
        return flat_idx, leaf_scale

    return _ScaledExpressionMatcher.match(expr, scale, leaf_match)


def _match_scaled_square_var(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[int, float]]:
    """Match ``c * x**2`` — and the identical ``c * x*x`` — in one flat variable.

    Both spellings are accepted (#1395). ``x*x`` is exactly ``x**2``, so every
    rule consuming this matcher (the sum-of-squares, sqrt-sum-of-squares,
    separable-quadratic and univariate-quadratic rules) is as sound on the ``Mul``
    spelling as it already is on the ``Pow`` one — but until #1395 only the ``Pow``
    node was recognized, so the same model written ``x*x`` lost every
    square-based inference. Measured on ``x`` declared over ``[-5, 5]``:
    ``x**2 <= 4`` tightened the box to ``[-2, 2]`` while ``x*x <= 4`` left it at
    ``[-5, 5]``, and ``x**2 <= -1`` was refuted by ``sum_of_squares_upper_bound``
    while ``x*x <= -1`` fell through to the continuous NLP and returned
    ``status="error"`` instead of ``infeasible``.

    The ``Mul`` arm requires both operands to resolve to the *same* flat scalar
    index. Anything else — two different variables (a genuine bilinear term), a
    non-scalar operand, or a nested product like ``(2*x)*(3*x)`` whose operands are
    not bare variables — stays unmatched, exactly as before.
    """

    def leaf_match(leaf: object, leaf_scale: float) -> Optional[tuple[int, float]]:
        if isinstance(leaf, BinaryOp) and leaf.op == "**":
            exponent = _constant_value(leaf.right)
            if exponent is None or abs(exponent - 2.0) > 1e-12:
                return None
            flat_idx = metadata.scalar_flat_index(leaf.left)
            if flat_idx is None:
                return None
            return flat_idx, leaf_scale
        if isinstance(leaf, BinaryOp) and leaf.op == "*":
            # ``_ScaledExpressionMatcher`` has already peeled any constant factor,
            # so a ``*`` arriving here has two non-constant operands: a square iff
            # they are the same scalar variable.
            left_idx = metadata.scalar_flat_index(leaf.left)
            if left_idx is None:
                return None
            right_idx = metadata.scalar_flat_index(leaf.right)
            if right_idx is None or right_idx != left_idx:
                return None
            return left_idx, leaf_scale
        return None

    return _ScaledExpressionMatcher.match(expr, scale, leaf_match)


def _match_scaled_bilinear(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[int, int, float]]:
    """Match ``c * (a*x) * (b*y)`` as ``(x, y, c*a*b)`` for scalar variables x, y.

    The constant factors are peeled by ``_ScaledExpressionMatcher``; a product of
    two non-constant operands then reaches the leaf, and each operand must itself be
    a scaled scalar variable (``2*x8`` in ``ball_mk4``'s ``(2*x8)*x7``). ``x == y``
    is returned too (a square spelled as a product of scaled copies); anything else
    is unmatched (#1867).
    """

    def leaf_match(leaf: object, leaf_scale: float) -> Optional[tuple[int, int, float]]:
        if not (isinstance(leaf, BinaryOp) and leaf.op == "*"):
            return None
        left = _match_scaled_linear_var(leaf.left, 1.0, metadata)
        right = _match_scaled_linear_var(leaf.right, 1.0, metadata)
        if left is None or right is None:
            return None
        return left[0], right[0], leaf_scale * left[1] * right[1]

    return _ScaledExpressionMatcher.match(expr, scale, leaf_match)


def _merge_affine_matches(
    left: tuple[Optional[int], float, float],
    right: tuple[Optional[int], float, float],
) -> Optional[tuple[Optional[int], float, float]]:
    left_idx, left_coeff, left_offset = left
    right_idx, right_coeff, right_offset = right
    if left_idx is not None and right_idx is not None and left_idx != right_idx:
        return None
    flat_idx = left_idx if left_idx is not None else right_idx
    return flat_idx, left_coeff + right_coeff, left_offset + right_offset


def _match_affine_var(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[Optional[int], float, float]]:
    """Match an affine scalar expression a*x + b in one flat variable."""
    const_val = _constant_value(expr)
    if const_val is not None:
        return None, 0.0, scale * const_val

    flat_idx = metadata.scalar_flat_index(expr)
    if flat_idx is not None:
        return flat_idx, scale, 0.0

    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_affine_var(expr.operand, -scale, metadata)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_affine_var(expr.right, scale * left_const, metadata)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_affine_var(expr.left, scale * right_const, metadata)
        return None

    if isinstance(expr, BinaryOp) and expr.op in ("+", "-"):
        left = _match_affine_var(expr.left, scale, metadata)
        right_scale = scale if expr.op == "+" else -scale
        right = _match_affine_var(expr.right, right_scale, metadata)
        if left is None or right is None:
            return None
        return _merge_affine_matches(left, right)

    return None


def _merge_single_variable_affine(
    current: Optional[tuple[Optional[int], float, float]],
    candidate: tuple[Optional[int], float, float],
) -> Optional[tuple[Optional[int], float, float]]:
    if current is None:
        return candidate
    return _merge_affine_matches(current, candidate)


def _apply_integrality(
    lb: float,
    ub: float,
    var_type: VarType,
) -> tuple[float, float]:
    if var_type == VarType.BINARY:
        return max(lb, 0.0), min(ub, 1.0)
    if var_type == VarType.INTEGER:
        return float(np.ceil(lb - 1e-9)), float(np.floor(ub + 1e-9))
    return lb, ub


def _constraint_label(constraint) -> str:
    name = getattr(constraint, "name", None)
    return str(name) if name else repr(constraint)


def _prove_infeasible(rule_name: str, constraint, reason: str) -> NoReturn:
    raise NonlinearBoundTighteningInfeasible(
        f"{rule_name} proved infeasibility for {_constraint_label(constraint)}: {reason}"
    )


def _tighten_affine_argument_interval(
    tightened_lb: np.ndarray,
    tightened_ub: np.ndarray,
    metadata: FlatVariableMetadata,
    flat_idx: int,
    coeff: float,
    offset: float,
    arg_lb: Optional[float] = None,
    arg_ub: Optional[float] = None,
) -> None:
    """Intersect a flat variable box with L <= coeff*x + offset <= U."""
    if abs(coeff) <= 1e-12:
        return

    new_lb = float(tightened_lb[flat_idx])
    new_ub = float(tightened_ub[flat_idx])

    # #1537 D: the preimage is rounded OUTWARD. In round-to-nearest one ulp of a
    # 4e8-magnitude quotient (6e-8) exceeds the 1e-9 integrality slack below, and
    # the rounded bound cut the only feasible integer (certified false infeasible).
    coeff = float(coeff)
    offset = float(offset)
    if arg_lb is not None and np.isfinite(arg_lb):
        if coeff > 0.0:
            new_lb = max(new_lb, affine_preimage(coeff, offset, float(arg_lb), lower=True))
        else:
            new_ub = min(new_ub, affine_preimage(coeff, offset, float(arg_lb), lower=False))

    if arg_ub is not None and np.isfinite(arg_ub):
        if coeff > 0.0:
            new_ub = min(new_ub, affine_preimage(coeff, offset, float(arg_ub), lower=False))
        else:
            new_lb = max(new_lb, affine_preimage(coeff, offset, float(arg_ub), lower=True))

    new_lb, new_ub = _apply_integrality(
        new_lb,
        new_ub,
        metadata.flat_var_types[flat_idx],
    )
    if new_lb <= new_ub:
        tightened_lb[flat_idx] = new_lb
        tightened_ub[flat_idx] = new_ub
        return

    if new_lb - new_ub <= _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack(new_lb, new_ub):
        # Sub-tolerance crossover (floating-point rounding on a degenerate box):
        # snap to a degenerate point and let the node NLP validate, rather than
        # certify a false infeasibility (issue #27a; hda fixed-value at ~68930).
        mid = 0.5 * (new_lb + new_ub)
        tightened_lb[flat_idx] = mid
        tightened_ub[flat_idx] = mid
        return

    raise NonlinearBoundTighteningInfeasible(
        f"tightened interval is empty for flat variable {flat_idx}: [{new_lb}, {new_ub}]"
    )


def _linearize_affine_expr(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
    n_vars: int,
) -> Optional[tuple[np.ndarray, float]]:
    const_val = _constant_value(expr)
    if const_val is not None:
        return np.zeros(n_vars, dtype=np.float64), scale * const_val

    flat_idx = metadata.scalar_flat_index(expr)
    if flat_idx is not None:
        coeff = np.zeros(n_vars, dtype=np.float64)
        coeff[flat_idx] = scale
        return coeff, 0.0

    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _linearize_affine_expr(expr.operand, -scale, metadata, n_vars)

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            left = _linearize_affine_expr(expr.left, scale, metadata, n_vars)
            right = _linearize_affine_expr(expr.right, scale, metadata, n_vars)
            if left is None or right is None:
                return None
            return left[0] + right[0], left[1] + right[1]
        if expr.op == "-":
            left = _linearize_affine_expr(expr.left, scale, metadata, n_vars)
            right = _linearize_affine_expr(expr.right, -scale, metadata, n_vars)
            if left is None or right is None:
                return None
            return left[0] + right[0], left[1] + right[1]
        if expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return _linearize_affine_expr(expr.right, scale * left_const, metadata, n_vars)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return _linearize_affine_expr(expr.left, scale * right_const, metadata, n_vars)
            return None
        if expr.op == "/":
            right_const = _constant_value(expr.right)
            if right_const is not None and abs(right_const) > 1e-12:
                return _linearize_affine_expr(expr.left, scale / right_const, metadata, n_vars)
            return None
        if expr.op == "**":
            exponent = _constant_value(expr.right)
            if exponent == 1.0:
                return _linearize_affine_expr(expr.left, scale, metadata, n_vars)
            if exponent == 0.0:
                return np.zeros(n_vars, dtype=np.float64), scale
            return None

    if isinstance(expr, SumOverExpression):
        coeff = np.zeros(n_vars, dtype=np.float64)
        const = 0.0
        for term in expr.terms:
            piece = _linearize_affine_expr(term, scale, metadata, n_vars)
            if piece is None:
                return None
            coeff += piece[0]
            const += piece[1]
        return coeff, const

    return None


def _affine_bounds(
    coeff: np.ndarray,
    const: float,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, float]:
    lower = float(const)
    upper = float(const)
    for c_i, lb_i, ub_i in zip(coeff, lb, ub):
        c = float(c_i)
        if c >= 0.0:
            lower += c * float(lb_i)
            upper += c * float(ub_i)
        else:
            lower += c * float(ub_i)
            upper += c * float(lb_i)
    return lower, upper


def _tighten_affine_upper_bound(
    tightened_lb: np.ndarray,
    tightened_ub: np.ndarray,
    metadata: FlatVariableMetadata,
    coeff: np.ndarray,
    const: float,
    rhs: float,
) -> None:
    """Intersect a box with ``coeff @ x + const <= rhs`` using interval FBBT."""
    for idx, c_i in enumerate(coeff):
        c = float(c_i)
        if abs(c) <= 1e-12:
            continue

        # #1537 D: the other terms' minimum is a directed (downward) sum and the
        # residual quotient is rounded outward, so a cancelling large-magnitude row
        # cannot cut a feasible point.
        other_min = float(const)
        for j, c_j in enumerate(coeff):
            if j == idx:
                continue
            cj = float(c_j)
            if cj >= 0.0:
                other_min = add_down(other_min, mul_down(cj, float(tightened_lb[j])))
            else:
                other_min = add_down(other_min, mul_down(cj, float(tightened_ub[j])))

        resid = sub_up(float(rhs), other_min)
        bound = div_up(resid, c) if c > 0.0 else div_down(resid, c)
        new_lb = float(tightened_lb[idx])
        new_ub = float(tightened_ub[idx])
        if c > 0.0:
            new_ub = min(new_ub, bound)
        else:
            new_lb = max(new_lb, bound)

        new_lb, new_ub = _apply_integrality(new_lb, new_ub, metadata.flat_var_types[idx])
        if new_lb <= new_ub:
            tightened_lb[idx] = new_lb
            tightened_ub[idx] = new_ub
            continue

        if new_lb - new_ub <= _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack(new_lb, new_ub):
            # Sub-tolerance crossover: snap to a degenerate point rather than
            # certify a false infeasibility (issue #27a).
            mid = 0.5 * (new_lb + new_ub)
            tightened_lb[idx] = mid
            tightened_ub[idx] = mid
            continue

        raise NonlinearBoundTighteningInfeasible(
            f"tightened interval is empty for flat variable {idx}: [{new_lb}, {new_ub}]"
        )


def _flatten_sum(expr, scale: float, out: list[tuple[float, object]]) -> None:
    """Flatten an additive expression into ``(scale, leaf)`` terms.

    The walk is iterative over an explicit ``(expr, scale)`` stack rather than
    recursive: a long linear row parses to a left-deep ``((a+b)+c)+d…`` chain
    whose depth equals the term count, so a recursive walk exhausted the Python
    stack on models with a few thousand terms (``t1000``) and killed the whole
    tightening pass with a ``RecursionError`` (#1198). Depth is now bounded only
    by the heap.

    Children are pushed in reverse so they pop left-to-right, which makes the
    emitted term order identical to the recursive walk's; the scale carried on a
    stack entry is the same value the recursion passed down (sign flips are
    exact), so the term list is bit-for-bit unchanged.
    """
    stack: list[tuple[object, float]] = [(expr, scale)]
    while stack:
        node, node_scale = stack.pop()

        if isinstance(node, SumOverExpression):
            for term in reversed(node.terms):
                stack.append((term, node_scale))
            continue

        if isinstance(node, BinaryOp) and node.op == "+":
            stack.append((node.right, node_scale))
            stack.append((node.left, node_scale))
            continue
        if isinstance(node, BinaryOp) and node.op == "-":
            stack.append((node.right, -node_scale))
            stack.append((node.left, node_scale))
            continue
        # Distribute negation over its operand: neg(a + b) == (-a) + (-b) is an
        # exact identity, so pushing the sign inward lets the additive walk see
        # the full structure underneath a `neg` (e.g. a `neg(sum-of-squares)`
        # term whose squares would otherwise be hidden inside one opaque leaf,
        # invisible to the downstream quadratic/interval bounding rules). This
        # mirrors the negation handling already in `_flatten_sum_terms`
        # (convexity/patterns.py) and is a sound normalization — it never removes
        # a feasible point (#777).
        if isinstance(node, UnaryOp) and node.op == "neg":
            stack.append((node.operand, -node_scale))
            continue
        out.append((node_scale, node))


def _min_univariate_quadratic(a: float, b: float, lb: float, ub: float) -> float:
    """Return min of a*x^2 + b*x over [lb, ub] for a >= 0."""
    if a < -1e-12:
        raise ValueError("nonconvex univariate quadratic is not supported")

    if abs(a) <= 1e-12:
        return min(b * lb, b * ub)

    x_star = -b / (2.0 * a)
    x_eval = min(max(x_star, lb), ub)
    return min(a * lb * lb + b * lb, a * x_eval * x_eval + b * x_eval, a * ub * ub + b * ub)


# #1867: the integrality tolerance a solution is accepted at. An integer column's
# value may sit up to this far from an integer, so the integer-domain minimum below
# is taken over [k - tol, k + tol] around each candidate integer, not at k exactly.
_INTEGER_DOMAIN_TOL = 1e-5


def integer_separable_quadratic_enabled() -> bool:
    """``DISCOPT_INTEGER_SEPARABLE_QUAD`` gate for the #1867 integer-domain minimum.

    With it, :class:`SeparableQuadraticUpperBoundRule` takes each integer/binary
    column's minimum contribution over its *integer* points instead of over the
    continuous interval. ``a*(x**2 - x)`` on ``x in {-1,...,2}`` has continuous
    minimum ``-a/4`` at ``x = 0.5`` but integer minimum ``0``, which is what proves
    ``ball_mk3_*`` infeasible at presolve. (This docstring used to say
    ``ball_mk3_*`` / ``ball_mk4_*``; ``ball_mk4_*`` has integer cross terms, which
    this rule abstains on. It is :class:`IntegerBlockQuadraticRule`'s.) Bound-changing (a
    strictly tighter minimum feeds both the infeasibility test and the leave-one-out
    tightening), so gated per CLAUDE.md section 5. Read at call time.

    **Graduated (default ON, 2026-10-22).** Firing scan of the rule at the root box,
    flag OFF vs ON, over 394 MINLPLib instances (<= 200 vars/cons, the 66 vendored
    included): output changes on 5, never looser -- ``ball_mk3_10/20/30`` proven
    infeasible (MINLPLib ``=inf=``), ``ball_mk2_10/30`` 10/30 bounds tighter. Panel
    on the 5, 120 s, interleaved, load <= 1.7: cert-clean (no bound past an optimum,
    no false infeasible, both incumbents pass ``verify_point`` on a fresh model) and
    net-positive -- certified 1 -> 5 (``ball_mk3_*`` infeasible in 0.1-0.2 s, time
    limit OFF; ``ball_mk2_30`` optimal in 39.8 s, 140 nodes, time limit OFF;
    ``ball_mk2_10`` 53.0 -> 39.4 s). ``DISCOPT_INTEGER_SEPARABLE_QUAD=0`` restores the
    continuous minimum. Not covered: rows with integer cross terms (``ball_mk4_*``).
    """
    return os.environ.get("DISCOPT_INTEGER_SEPARABLE_QUAD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _min_univariate_quadratic_integer(a: float, b: float, lb: float, ub: float) -> float:
    """Min of ``a*x^2 + b*x`` (``a >= 0``) over the integer points of ``[lb, ub]``,
    each widened by ``_INTEGER_DOMAIN_TOL`` (#1867).

    Convexity puts the integer minimiser at ``floor(x*)`` or ``ceil(x*)`` of the
    vertex ``x* = -b/(2a)``, clamped to the integer range (an endpoint when
    ``a == 0``); each candidate's contribution is the continuous minimum over its
    tolerance window intersected with ``[lb, ub]``. Never below the continuous
    minimum, so it can only tighten. An empty integer range (``ceil(lb) >
    floor(ub)``) abstains to the continuous minimum: emptiness is the box's
    business, not this bound's.
    """
    # The range is rounded with the SAME tolerance a solution is accepted at: an
    # integer k with lb in (k, k + tol] still has accepted points (x = lb is within tol
    # of k), so it must stay a candidate. Rounding with a 1e-9 slack dropped it and
    # overestimated the minimum -- e.g. (x - 1)**2 on [1 + 5e-6, 3] gave 0.99998 where
    # the accepted point x = lb gives 2.5e-11 -- which could prove a false infeasible
    # or cut an accepted point (#1881 review). Each window below is intersected with
    # [lb, ub], so a candidate never reaches outside the box.
    tol = _INTEGER_DOMAIN_TOL
    lo = float(np.ceil(lb - tol))
    hi = float(np.floor(ub + tol))
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        return _min_univariate_quadratic(a, b, lb, ub)
    if abs(a) <= 1e-12:
        candidates = {lo, hi}
    else:
        x_star = -b / (2.0 * a)
        candidates = {
            min(max(float(np.floor(x_star)), lo), hi),
            min(max(float(np.ceil(x_star)), lo), hi),
        }
    return min(
        _min_univariate_quadratic(a, b, max(k - tol, lb), min(k + tol, ub)) for k in candidates
    )


def _tighten_univariate_quadratic_interval(
    a: float,
    b: float,
    rhs: float,
    lb: float,
    ub: float,
) -> Optional[tuple[float, float]]:
    """Intersect [lb, ub] with the feasible set of a*x^2 + b*x <= rhs for a >= 0.

    Infeasibility is declared (``None``) only when the constraint is violated by
    more than the feasibility tolerance. A sub-tolerance violation — e.g. the
    O(eps) residual of an approximate GDP hull perspective — must not manufacture
    a hard infeasibility here, because the ``None`` return is pruned upstream
    without any tolerance check. The threshold is translated into each branch's
    native units: a constant constraint compares ``rhs`` directly, while the
    quadratic branch compares the discriminant, since the constraint violation of
    ``a*x^2 + b*x <= rhs`` at its vertex equals ``-discriminant / (4a)`` (issue
    #27a). Such residuals are deferred to the node NLP for exact validation.

    **Caller contract (#1397).** ``rhs`` must already be widened *outward* by a
    bound on its own accumulation error — the caller's ``float_err``. The constant
    branch below compares ``rhs`` directly and does no arithmetic of its own, so
    its soundness rests entirely on that: an outward-widened ``rhs`` can only make
    the branch declare feasibility more readily, never infeasibility. The
    discriminant branch cannot lean on the caller, because ``b*b + 4*a*rhs`` is a
    *fresh* cancellation built here: at large ``|b|`` the two terms are big and
    nearly equal, so the difference carries an error of a few ulps of the larger,
    which at ``|b| ~ 1e8`` already exceeds the whole 1e-6 tolerance. It therefore
    subtracts :func:`_roundoff_slack` of its own two terms, in *discriminant*
    units — unmultiplied by ``4a``, unlike the tolerance, because it bounds the
    error of the discriminant itself rather than a constraint violation.
    """
    if abs(a) <= 1e-12:
        # Dropping a tiny ``a >= 0`` is a relaxation (``a*x^2 >= 0``). Dropping a tiny
        # nonzero ``b`` is NOT: ``5 - 1e-13*y <= 0`` over ``y <= 1e14`` was read as
        # ``5 <= 0`` and a feasible model was proved infeasible (#1617). Only an exact
        # zero makes the row a constant; any other ``b`` is divided through.
        if b == 0.0:
            return (lb, ub) if rhs >= -_EMPTY_INTERVAL_FEAS_TOL else None
        bound = rhs / b
        if b > 0.0:
            return (lb, min(ub, bound))
        return (max(lb, bound), ub)

    discriminant = b * b + 4.0 * a * rhs
    # Vertex violation = -discriminant / (4a); infeasible only beyond tolerance,
    # and only beyond the round-off of the cancellation that built the
    # discriminant (#1397 -- see the caller contract in the docstring).
    if discriminant < -4.0 * a * _EMPTY_INTERVAL_FEAS_TOL - _roundoff_slack(b * b, 4.0 * a * rhs):
        return None
    discriminant = max(discriminant, 0.0)
    sqrt_disc = float(np.sqrt(discriminant))
    root_lo = (-b - sqrt_disc) / (2.0 * a)
    root_hi = (-b + sqrt_disc) / (2.0 * a)
    return (max(lb, root_lo), min(ub, root_hi))


class SumOfSquaresUpperBoundRule(NonlinearBoundTighteningRule):
    """Tighten bounds from constraints like sum(a_i * x_i^2) <= c."""

    name = "sum_of_squares_upper_bound"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _match_scaled_square(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[int, float]]:
        return _match_scaled_square_var(expr, scale, metadata)

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in _rows_until(model, deadline):
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms = _cached_flat_terms(model, constraint.body)

            constant_term = 0.0
            # #1397: running bound on the magnitude of the terms ``constant_term`` is
            # a signed sum of. The sum's round-off is a few ulps of THIS, not of the
            # (possibly cancelled-to-nothing) result, so the result cannot supply its
            # own yardstick and it has to be accumulated alongside.
            constant_absum = 0.0
            square_coeffs: dict[int, float] = {}
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    constant_absum += abs(scale * const_val)
                    continue

                match = self._match_scaled_square(term, scale, metadata)
                if match is None:
                    matches_pattern = False
                    break

                flat_idx, coeff = match
                if coeff <= 0.0:
                    matches_pattern = False
                    break
                square_coeffs[flat_idx] = square_coeffs.get(flat_idx, 0.0) + coeff

            if not matches_pattern or not square_coeffs:
                continue

            rhs = -constant_term
            rhs_slack = _roundoff_slack(constant_absum)
            # A negative upper bound on a nonnegative sum of squares is genuine
            # infeasibility only beyond the feasibility tolerance; a sub-tolerance
            # excess (e.g. an eps-scale hull-perspective residual) is feasible
            # within tolerance and must not be pruned (issue #27a). It is also
            # genuine only beyond the round-off of the sum that produced it
            # (#1397): at ``constant_absum ~ 1e10`` one ulp already exceeds the
            # whole tolerance, so without this the rule proves infeasibility from
            # rounding noise and the node is pruned.
            if rhs < -_EMPTY_INTERVAL_FEAS_TOL - rhs_slack:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "nonnegative sum of squares has a negative upper bound",
                )
            # #1397: widen outward before the clamp. ``max(0.0, rhs)`` alone turns a
            # spurious negative into ``radius = 0``, i.e. it FIXES every x_i in the
            # row to zero -- a tightening built from round-off. Adding the bound
            # first makes the radius an over-estimate, which can only enlarge the
            # box and so can never remove a feasible point.
            rhs = max(0.0, rhs + rhs_slack)
            for flat_idx, coeff in square_coeffs.items():
                if coeff <= 0.0:
                    continue

                radius = float(np.sqrt(rhs / coeff))
                new_lb = max(float(tightened_lb[flat_idx]), -radius)
                new_ub = min(float(tightened_ub[flat_idx]), radius)

                var_type = metadata.flat_var_types[flat_idx]
                if var_type == VarType.BINARY:
                    new_lb = max(new_lb, 0.0)
                    new_ub = min(new_ub, 1.0)
                elif var_type == VarType.INTEGER:
                    new_lb = float(np.ceil(new_lb - 1e-9))
                    new_ub = float(np.floor(new_ub + 1e-9))

                if new_lb <= new_ub:
                    tightened_lb[flat_idx] = new_lb
                    tightened_ub[flat_idx] = new_ub
                else:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"required interval for flat variable {flat_idx} is empty",
                    )

        return tightened_lb, tightened_ub


class SqrtSumOfSquaresUpperBoundRule(NonlinearBoundTighteningRule):
    """Tighten bounds from constraints like sqrt(sum(a_i * x_i^2)) <= c."""

    name = "sqrt_sum_of_squares_upper_bound"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _match_scaled_square(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[int, float]]:
        return _match_scaled_square_var(expr, scale, metadata)

    def _match_sum_of_squares(
        self,
        expr,
        metadata: FlatVariableMetadata,
    ) -> Optional[dict[int, float]]:
        terms: list[tuple[float, object]] = []
        _flatten_sum(expr, 1.0, terms)

        constant_term = 0.0
        square_coeffs: dict[int, float] = {}
        for scale, term in terms:
            const_val = _constant_value(term)
            if const_val is not None:
                constant_term += scale * const_val
                continue

            match = self._match_scaled_square(term, scale, metadata)
            if match is None:
                return None
            flat_idx, coeff = match
            if coeff <= 0.0:
                return None
            square_coeffs[flat_idx] = square_coeffs.get(flat_idx, 0.0) + coeff

        if abs(constant_term) > 1e-12 or not square_coeffs:
            return None
        return square_coeffs

    def _match_scaled_sqrt_sum_of_squares(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[float, dict[int, float]]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_sqrt_sum_of_squares(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_sqrt_sum_of_squares(
                    expr.right, scale * left_const, metadata
                )
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_sqrt_sum_of_squares(
                    expr.left, scale * right_const, metadata
                )
            return None

        if not isinstance(expr, FunctionCall) or expr.func_name != "sqrt" or len(expr.args) != 1:
            return None

        square_coeffs = self._match_sum_of_squares(expr.args[0], metadata)
        if square_coeffs is None:
            return None
        return scale, square_coeffs

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in _rows_until(model, deadline):
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms = _cached_flat_terms(model, constraint.body)

            constant_term = 0.0
            # #1397: see the same accumulator in SumOfSquaresUpperBoundRule.
            constant_absum = 0.0
            sqrt_match: Optional[tuple[float, dict[int, float]]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    constant_absum += abs(scale * const_val)
                    continue

                match = self._match_scaled_sqrt_sum_of_squares(term, scale, metadata)
                if match is None or sqrt_match is not None:
                    matches_pattern = False
                    break
                sqrt_match = match

            if not matches_pattern or sqrt_match is None:
                continue

            sqrt_coeff, square_coeffs = sqrt_match
            if sqrt_coeff <= 0.0:
                continue

            rhs = -constant_term / sqrt_coeff
            # #1397: the accumulation bound divided by the same coefficient, so the
            # slack lands in the same units as ``rhs``.
            rhs_slack = _roundoff_slack(constant_absum) / sqrt_coeff
            # Infeasible only beyond the feasibility tolerance (see issue #27a);
            # a sub-tolerance excess is feasible within tolerance and deferred.
            # And only beyond the round-off of the sum behind it (#1397).
            if rhs < -_EMPTY_INTERVAL_FEAS_TOL - rhs_slack:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "nonnegative sqrt sum of squares has a negative upper bound",
                )
            # #1397: widen outward before the clamp -- see the sibling rule.
            rhs = max(0.0, rhs + rhs_slack)
            squared_rhs = rhs * rhs

            for flat_idx, coeff in square_coeffs.items():
                radius = float(np.sqrt(squared_rhs / coeff))
                new_lb = max(float(tightened_lb[flat_idx]), -radius)
                new_ub = min(float(tightened_ub[flat_idx]), radius)
                new_lb, new_ub = _apply_integrality(
                    new_lb,
                    new_ub,
                    metadata.flat_var_types[flat_idx],
                )
                if new_lb <= new_ub:
                    tightened_lb[flat_idx] = new_lb
                    tightened_ub[flat_idx] = new_ub
                else:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"required interval for flat variable {flat_idx} is empty",
                    )

        return tightened_lb, tightened_ub


def affine_implied_values(model: Model) -> dict[int, float]:
    """Flat index -> value for each column an ``==`` row pins on its own (#1714).

    A row qualifies when every non-constant term is a scaled copy of ONE scalar
    variable, so ``a*x + c == 0`` (``z == 1``, ``x[0] == 0``, ``2*v + 1 == 3``) and
    the row's only solution is ``x = -c/a``. Rows that mix two variables, carry a
    nonlinear term, or address a slice are skipped. When several rows pin the same
    column the first one wins; a conflict is an infeasible model, which the NLP
    reports on its own.

    This is a START-POINT helper, not a bound inference: callers use the values to
    place an initial iterate and never narrow a box with them.
    """
    metadata = build_flat_variable_metadata(model)
    out: dict[int, float] = {}
    for constraint in model._constraints:
        if getattr(constraint, "sense", None) != "==":
            continue
        constant = 0.0
        flat_idx: Optional[int] = None
        coeff = 0.0
        for scale, term in _cached_flat_terms(model, constraint.body):
            value = _constant_value(term)
            if value is not None:
                constant += scale * value
                continue
            match = _match_scaled_linear_var(term, scale, metadata)
            if match is None or (flat_idx is not None and match[0] != flat_idx):
                flat_idx = None
                break
            flat_idx = match[0]
            coeff += match[1]
        else:
            if flat_idx is not None and coeff != 0.0:
                implied = -constant / coeff
                if np.isfinite(implied):
                    out.setdefault(flat_idx, float(implied))
    return out


class SeparableQuadraticUpperBoundRule(NonlinearBoundTighteningRule):
    """Tighten bounds from separable convex quadratic constraints like x + y^2 <= c."""

    name = "separable_quadratic_upper_bound"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _match_scaled_square(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[int, float]]:
        return _match_scaled_square_var(expr, scale, metadata)

    def _decompose_row(
        self,
        model: Model,
        constraint,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[float, dict[int, tuple[float, float]], float]]:
        """Bound-independent half of the row scan; see ``_cached_row_structure``.

        Returns ``(constant_term, {flat_idx: (quad_coeff, linear_coeff)},
        constant_absum)``, or ``None`` when the row is not a separable convex
        quadratic this rule can use. The arithmetic is exactly the loop this
        replaced, in the same order.

        ``constant_absum`` is the magnitude of the constant leaves ``constant_term``
        is a signed sum of (#1397). ``tighten`` needs it because ``constant_term``
        cannot describe its own error: ``-0.4e + 1e16 - 1e16 + 0.3e`` evaluates to
        ``+0.3e`` where exact arithmetic gives ``-0.1e``, and ``abs(constant_term)``
        reads that as an O(1e-1) quantity with an O(1e-17) error rather than the
        residue of a 2e16-magnitude cancellation.
        """
        terms = _cached_flat_terms(model, constraint.body)

        constant_term = 0.0
        constant_absum = 0.0
        quad_coeffs: dict[int, float] = {}
        linear_coeffs: dict[int, float] = {}

        for scale, term in terms:
            const_val = _constant_value(term)
            if const_val is not None:
                constant_term += scale * const_val
                constant_absum += abs(scale * const_val)
                continue

            square_match = self._match_scaled_square(term, scale, metadata)
            if square_match is not None:
                flat_idx, coeff = square_match
                quad_coeffs[flat_idx] = quad_coeffs.get(flat_idx, 0.0) + coeff
                continue

            linear_match = _match_scaled_linear_var(term, scale, metadata)
            if linear_match is not None:
                flat_idx, coeff = linear_match
                linear_coeffs[flat_idx] = linear_coeffs.get(flat_idx, 0.0) + coeff
                continue

            return None

        if not quad_coeffs and not linear_coeffs:
            return None

        coeffs = {
            flat_idx: (
                quad_coeffs.get(flat_idx, 0.0),
                linear_coeffs.get(flat_idx, 0.0),
            )
            for flat_idx in set(quad_coeffs) | set(linear_coeffs)
        }
        if any(a < -1e-12 for a, _ in coeffs.values()):
            return None
        return constant_term, coeffs, constant_absum

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in _rows_until(model, deadline):
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            structure = _cached_row_structure(
                model,
                self.name,
                constraint,
                metadata,
                lambda c=constraint: self._decompose_row(model, c, metadata),
            )
            if structure is None:
                continue
            constant_term, coeffs, constant_absum = structure

            min_contribs: dict[int, float] = {}
            integer_domain = integer_separable_quadratic_enabled()
            for flat_idx, (a, b) in coeffs.items():
                # #1867: an integer/binary column's contribution is minimised over
                # its integer points (see ``_min_univariate_quadratic_integer``).
                is_int = metadata.flat_var_types[flat_idx] in (VarType.INTEGER, VarType.BINARY)
                minimiser = (
                    _min_univariate_quadratic_integer
                    if integer_domain and is_int
                    else _min_univariate_quadratic
                )
                min_contribs[flat_idx] = minimiser(
                    a,
                    b,
                    float(tightened_lb[flat_idx]),
                    float(tightened_ub[flat_idx]),
                )

            total_min = float(sum(min_contribs.values()))
            # A bound on the floating-point error of ``total_min`` and of every
            # leave-one-out ``total_min - min_contribs[j]`` below. Without it a huge
            # contribution absorbs the small ones: on x1 in the default ±9.999e19 box,
            # -5·x1 + x2 <= 2 with x2 = -1 summed to -4.9995e20 + -1 == -4.9995e20, the
            # leave-one-out rest for x1 came out 0 instead of -1, x1 >= -0.4 replaced
            # x1 >= -0.6, and a feasible LP at x1 = -0.5 was certified infeasible.
            # Widening outward by it keeps every inference valid in exact arithmetic.
            #
            # #1397: the constant's share of this bound is ``constant_absum``, the
            # magnitude of the constant leaves, NOT ``abs(constant_term)`` as it was
            # written. The two differ exactly when the constant is itself a cancelled
            # sum, which is the case this bound exists to cover: with a single square
            # over a box straddling zero every ``min_contribs`` value is 0, so
            # ``abs(constant_term)`` was the ONLY scale in the product, and a constant
            # that cancelled from 1e16 down to 0.6 contributed a float_err of 1e-16
            # instead of 35. Measured on this tree: ``x**2 - 0.4e + M - M + 0.3e <= 0``
            # (exact row ``x**2 <= 0.1*spacing(M)``, feasible at x = 0) was proved
            # infeasible at every M from 1e11 to 1e16.
            float_err = (
                (len(min_contribs) + 4.0)
                * _FLOAT_EPS
                * (sum(abs(v) for v in min_contribs.values()) + constant_absum)
            )
            # Declare infeasibility only when the minimum separable activity
            # exceeds the upper bound by more than the feasibility tolerance;
            # a sub-tolerance excess is feasible within tolerance (e.g. the
            # eps-scale residual of an approximate GDP hull perspective) and
            # must not be pruned (issue #27a).
            if constant_term + total_min > _EMPTY_INTERVAL_FEAS_TOL + float_err:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "minimum separable quadratic activity exceeds the upper bound",
                )

            for flat_idx, (a, b) in coeffs.items():
                rhs = -constant_term - (total_min - min_contribs[flat_idx]) + float_err
                if not np.isfinite(rhs):
                    continue

                interval = _tighten_univariate_quadratic_interval(
                    a,
                    b,
                    rhs,
                    float(tightened_lb[flat_idx]),
                    float(tightened_ub[flat_idx]),
                )
                if interval is None:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"required interval for flat variable {flat_idx} is empty",
                    )

                new_lb, new_ub = interval
                var_type = metadata.flat_var_types[flat_idx]
                if var_type == VarType.BINARY:
                    new_lb = max(new_lb, 0.0)
                    new_ub = min(new_ub, 1.0)
                elif var_type == VarType.INTEGER:
                    new_lb = float(np.ceil(new_lb - 1e-9))
                    new_ub = float(np.floor(new_ub + 1e-9))

                if new_lb <= new_ub:
                    tightened_lb[flat_idx] = new_lb
                    tightened_ub[flat_idx] = new_ub
                elif new_lb <= new_ub + _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack(new_lb, new_ub):
                    # Sub-tolerance crossover: snap to a degenerate box instead
                    # of pruning, leaving validation to the node NLP.
                    tightened_lb[flat_idx] = new_ub
                    tightened_ub[flat_idx] = new_ub
                else:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"required interval for flat variable {flat_idx} is empty",
                    )

        return tightened_lb, tightened_ub


def _safe_exp(value: float) -> float:
    if value > 709.0:
        return float("inf")
    if value < -745.0:
        return 0.0
    return float(np.exp(value))


def _monotone_function_value(func_name: str, value: float) -> float:
    if func_name == "exp":
        return _safe_exp(value)
    if func_name == "log":
        if value <= 0.0:
            return -float("inf")
        return float(np.log(value))
    if func_name == "log2":
        if value <= 0.0:
            return -float("inf")
        return float(np.log2(value))
    if func_name == "log10":
        if value <= 0.0:
            return -float("inf")
        return float(np.log10(value))
    if func_name == "log1p":
        if value <= -1.0:
            return -float("inf")
        return float(np.log1p(value))
    if func_name == "sqrt":
        if value < 0.0:
            return float("nan")
        return float(np.sqrt(value))
    raise ValueError(f"Unsupported monotone function: {func_name}")


#: Points at which a monotone function's value is an exact float, by definition of
#: the function (``exp(0) = 1``, ``log(1) = 0``, ...). The value is returned as the
#: mathematical constant, never through libm, so it needs no outward widening -- the
#: same "exact results stay exact" contract the ``_directed`` helpers keep for
#: ``+ - * /`` (#1572).
_MONOTONE_EXACT_VALUES: dict[str, dict[float, float]] = {
    "exp": {0.0: 1.0},
    "log": {1.0: 0.0},
    "log2": {1.0: 0.0},
    "log10": {1.0: 0.0},
    "log1p": {0.0: 0.0},
}


def _monotone_value_directed(func_name: str, value: float, *, lower: bool) -> float:
    """Directed enclosure of ``f(value)`` for an increasing monotone ``f`` (#1572).

    A lower bound on the exact ``f(value)`` when ``lower`` is true, an upper bound
    otherwise. ``sqrt`` is correctly rounded (IEEE 754), so it gets true directed
    rounding; the libm functions are widened by :func:`lib_down` / :func:`lib_up`
    (``LIB_ULPS + 1`` ulps), except at their exact points.
    """
    exact = _MONOTONE_EXACT_VALUES.get(func_name, {}).get(value)
    if exact is not None:
        return exact
    if func_name == "sqrt":
        if value < 0.0:
            return float("nan")
        return sqrt_down(value) if lower else sqrt_up(value)
    raw = _monotone_function_value(func_name, value)
    if not lower:
        return lib_up(raw)
    out = lib_down(raw)
    # exp is positive everywhere: an underflowed ``exp(-800) = 0`` widened down to
    # ``-5e-324`` is a valid but needlessly negative bound; 0 is exact and sound.
    return max(out, 0.0) if func_name == "exp" else out


def _affine_value_directed(coeff: float, offset: float, value: float, *, lower: bool) -> float:
    """Directed enclosure of ``coeff * value + offset`` (#1572)."""
    if lower:
        return add_down(mul_down(coeff, value), offset)
    return add_up(mul_up(coeff, value), offset)


def _inverse_monotone_upper(func_name: str, rhs: float) -> Optional[float]:
    """Return U such that f(arg) <= rhs implies arg <= U.

    #1537 D: the inverse is rounded UP -- libm results widened by ``lib_up``, the
    ``sqrt`` preimage by a directed multiply -- so ``U`` encloses the exact preimage.
    """
    if func_name == "exp":
        if rhs <= 0.0:
            return None
        return lib_up(float(np.log(rhs)))
    if func_name == "log":
        return lib_up(_safe_exp(rhs))
    if func_name == "log2":
        return float("inf") if rhs > 1024.0 else lib_up(float(2.0**rhs))
    if func_name == "log10":
        return float("inf") if rhs > 308.0 else lib_up(float(10.0**rhs))
    if func_name == "log1p":
        return sub_up(lib_up(_safe_exp(rhs)), 1.0)
    if func_name == "sqrt":
        if rhs < 0.0:
            return None
        return mul_up(rhs, rhs)
    return None


def _inverse_monotone_lower(func_name: str, rhs: float) -> Optional[float]:
    """Return L such that f(arg) >= rhs implies arg >= L.

    #1537 D: rounded DOWN, mirroring :func:`_inverse_monotone_upper`.
    """
    if func_name == "exp":
        if rhs <= 0.0:
            return None
        return lib_down(float(np.log(rhs)))
    if func_name == "log":
        return lib_down(_safe_exp(rhs))
    if func_name == "log2":
        return 0.0 if rhs < -1074.0 else lib_down(float(2.0**rhs))
    if func_name == "log10":
        return 0.0 if rhs < -324.0 else lib_down(float(10.0**rhs))
    if func_name == "log1p":
        return sub_down(lib_down(_safe_exp(rhs)), 1.0)
    if func_name == "sqrt":
        if rhs <= 0.0:
            return None
        return mul_down(rhs, rhs)
    return None


_MONOTONE_DOMAINS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "exp": (None, None),
    "log": (0.0, None),
    "log2": (0.0, None),
    "log10": (0.0, None),
    "log1p": (-1.0, None),
    "sqrt": (0.0, None),
}


class MonotoneFunctionEqualityRule(NonlinearBoundTighteningRule):
    """Propagate equalities like y == exp(a*x + b) in both directions."""

    name = "monotone_function_equality"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _match_scaled_function(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[str, float, int, float, float]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_function(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_function(expr.right, scale * left_const, metadata)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_function(expr.left, scale * right_const, metadata)
            return None

        if (
            not isinstance(expr, FunctionCall)
            or expr.func_name not in _MONOTONE_DOMAINS
            or len(expr.args) != 1
        ):
            return None

        affine_match = _match_affine_var(expr.args[0], 1.0, metadata)
        if affine_match is None:
            return None
        flat_idx, arg_coeff, arg_offset = affine_match
        if flat_idx is None or abs(arg_coeff) <= 1e-12:
            return None
        return expr.func_name, scale, flat_idx, arg_coeff, arg_offset

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in _rows_until(model, deadline):
            if getattr(constraint, "sense", None) != "==":
                continue

            terms = _cached_flat_terms(model, constraint.body)

            constant_term = 0.0
            affine_match: Optional[tuple[Optional[int], float, float]] = None
            function_match: Optional[tuple[str, float, int, float, float]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = self._match_scaled_function(term, scale, metadata)
                if match is not None:
                    if function_match is not None:
                        matches_pattern = False
                        break
                    function_match = match
                    continue

                affine_term = _match_affine_var(term, scale, metadata)
                if affine_term is None:
                    matches_pattern = False
                    break
                affine_match = _merge_single_variable_affine(affine_match, affine_term)
                if affine_match is None:
                    matches_pattern = False
                    break

            if (
                not matches_pattern
                or affine_match is None
                or function_match is None
                or affine_match[0] is None
            ):
                continue

            linear_idx, linear_coeff, linear_offset = affine_match
            assert linear_idx is not None
            if abs(linear_coeff) <= 1e-12:
                continue

            func_name, func_coeff, arg_idx, arg_coeff, arg_offset = function_match
            if abs(func_coeff) <= 1e-12:
                continue

            domain_lb, domain_ub = _MONOTONE_DOMAINS[func_name]
            arg_endpoint_a = arg_coeff * float(tightened_lb[arg_idx]) + arg_offset
            arg_endpoint_b = arg_coeff * float(tightened_ub[arg_idx]) + arg_offset
            current_arg_lb = min(arg_endpoint_a, arg_endpoint_b)
            current_arg_ub = max(arg_endpoint_a, arg_endpoint_b)
            # #1537: the argument endpoints are affine images of the box, rounded at
            # the magnitude of what was summed -- ulp(1e6) ~ 1.2e-10 once the model is
            # written in ``x = y - 1e6``. Every PROOF below is taken on outward-widened
            # intervals and judged against the row's own feasibility tolerance (in
            # function units), as #1397/#1415 do for the square rules; the bare 1e-12
            # here certified hda infeasible after translation (a sqrt range touching
            # its required range at one point read as empty).
            arg_slack = _roundoff_slack(arg_endpoint_a, arg_endpoint_b, arg_offset)
            if domain_lb is not None:
                current_arg_lb = max(current_arg_lb, domain_lb)
            if domain_ub is not None:
                current_arg_ub = min(current_arg_ub, domain_ub)
            if current_arg_lb > current_arg_ub + _EMPTY_INTERVAL_FEAS_TOL + arg_slack:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"{func_name} argument domain is empty on the current box",
                )

            # Outward function range for the proof: f on the widened argument
            # interval (clipped to the domain), plus the function's own rounding.
            wide_arg_lb = current_arg_lb - arg_slack
            wide_arg_ub = current_arg_ub + arg_slack
            if domain_lb is not None:
                wide_arg_lb = max(wide_arg_lb, domain_lb)
            if domain_ub is not None:
                wide_arg_ub = min(wide_arg_ub, domain_ub)
            wide_func_min = _monotone_function_value(func_name, wide_arg_lb)
            wide_func_max = _monotone_function_value(func_name, wide_arg_ub)
            wide_func_min -= _roundoff_slack(wide_func_min)
            wide_func_max += _roundoff_slack(wide_func_max)

            # #1572: the TIGHTENINGS take a directed enclosure of the function range,
            # not the proof's widened one above. ``arg_slack`` is a tolerance-style
            # bound (``8 eps`` times the SUM of both endpoint magnitudes, applied to
            # each endpoint) and ``_roundoff_slack`` adds ``8 eps |f|`` more; fed to a
            # tightening it moved an exact ``exp(0) = 1`` lower bound to
            # ``1 - 8.4e-14`` -- ~380 ulps, scaled by the *other* endpoint (46) -- on
            # top of the directed rounding below. The directed enclosure is exact
            # where the arithmetic is exact and one rounding step outward where not.
            if arg_coeff > 0.0:
                dir_arg_lo_src = float(tightened_lb[arg_idx])
                dir_arg_hi_src = float(tightened_ub[arg_idx])
            else:
                dir_arg_lo_src = float(tightened_ub[arg_idx])
                dir_arg_hi_src = float(tightened_lb[arg_idx])
            dir_arg_lo = _affine_value_directed(arg_coeff, arg_offset, dir_arg_lo_src, lower=True)
            dir_arg_hi = _affine_value_directed(arg_coeff, arg_offset, dir_arg_hi_src, lower=False)
            if domain_lb is not None:
                dir_arg_lo = max(dir_arg_lo, domain_lb)
            if domain_ub is not None:
                dir_arg_hi = min(dir_arg_hi, domain_ub)
            if dir_arg_lo <= dir_arg_hi:
                tight_func_min = _monotone_value_directed(func_name, dir_arg_lo, lower=True)
                tight_func_max = _monotone_value_directed(func_name, dir_arg_hi, lower=False)
            else:
                # The domain clip emptied the directed interval: the box is
                # domain-infeasible by less than the proof tolerance accepted above.
                # Keep the proof's widened range (the pre-#1572 behaviour) rather
                # than tighten from an empty enclosure.
                tight_func_min = wide_func_min
                tight_func_max = wide_func_max

            # #1537 D: the TIGHTENINGS below are outward enclosures too, not only the
            # proofs. The linear side's target is ``-c0 - fc * f`` over the directed
            # function range, each endpoint in directed rounding; the extracted
            # coefficients and constant are model data, the arithmetic on them is
            # what is enclosed.
            neg_c0 = -constant_term
            if func_coeff > 0.0:
                target_lb = sub_down(neg_c0, mul_up(func_coeff, tight_func_max))
                target_ub = sub_up(neg_c0, mul_down(func_coeff, tight_func_min))
            else:
                target_lb = sub_down(neg_c0, mul_up(func_coeff, tight_func_min))
                target_ub = sub_up(neg_c0, mul_down(func_coeff, tight_func_max))
            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                linear_idx,
                linear_coeff,
                linear_offset,
                arg_lb=target_lb,
                arg_ub=target_ub,
            )

            linear_endpoint_a = linear_coeff * float(tightened_lb[linear_idx]) + linear_offset
            linear_endpoint_b = linear_coeff * float(tightened_ub[linear_idx]) + linear_offset
            linear_expr_lb = min(linear_endpoint_a, linear_endpoint_b)
            linear_expr_ub = max(linear_endpoint_a, linear_endpoint_b)
            required_values = (
                (-constant_term - linear_expr_lb) / func_coeff,
                (-constant_term - linear_expr_ub) / func_coeff,
            )
            required_func_lb = min(required_values)
            required_func_ub = max(required_values)

            # Outward enclosure of the function values the linear side allows, for
            # the argument tightening (#1537 D). The linear expression's range is
            # rebuilt in directed rounding; the round-to-nearest values above stay
            # what the infeasibility proof below has always used.
            lin_t_lo = float(tightened_lb[linear_idx])
            lin_t_hi = float(tightened_ub[linear_idx])
            if linear_coeff > 0.0:
                lin_lo = add_down(mul_down(linear_coeff, lin_t_lo), linear_offset)
                lin_hi = add_up(mul_up(linear_coeff, lin_t_hi), linear_offset)
            else:
                lin_lo = add_down(mul_down(linear_coeff, lin_t_hi), linear_offset)
                lin_hi = add_up(mul_up(linear_coeff, lin_t_lo), linear_offset)
            if func_coeff > 0.0:
                out_func_lb = div_down(sub_down(neg_c0, lin_hi), func_coeff)
                out_func_ub = div_up(sub_up(neg_c0, lin_lo), func_coeff)
            else:
                out_func_lb = div_down(sub_up(neg_c0, lin_lo), func_coeff)
                out_func_ub = div_up(sub_down(neg_c0, lin_hi), func_coeff)

            feasible_func_lb = max(out_func_lb, tight_func_min)
            feasible_func_ub = min(out_func_ub, tight_func_max)
            required_slack = _roundoff_slack(
                abs(constant_term), linear_expr_lb, linear_expr_ub
            ) / abs(func_coeff)
            if max(required_func_lb - required_slack, wide_func_min) > min(
                required_func_ub + required_slack, wide_func_max
            ) + _EMPTY_INTERVAL_FEAS_TOL / abs(func_coeff):
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"{func_name}(argument) range cannot satisfy linked equality",
                )

            arg_lb = domain_lb
            arg_ub = domain_ub
            if np.isfinite(feasible_func_lb):
                lower = _inverse_monotone_lower(func_name, feasible_func_lb)
                if lower is not None:
                    arg_lb = lower if arg_lb is None else max(arg_lb, lower)
            if np.isfinite(feasible_func_ub):
                upper = _inverse_monotone_upper(func_name, feasible_func_ub)
                if upper is None:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"{func_name}(argument) cannot be <= {feasible_func_ub}",
                    )
                arg_ub = upper if arg_ub is None else min(arg_ub, upper)

            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                arg_idx,
                arg_coeff,
                arg_offset,
                arg_lb=arg_lb,
                arg_ub=arg_ub,
            )

        return tightened_lb, tightened_ub


class MonotoneFunctionBoundsRule(NonlinearBoundTighteningRule):
    """Tighten affine arguments of monotone unary function constraints."""

    name = "monotone_function_bounds"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _match_scaled_function(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[str, float, int, float, float]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_function(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_function(expr.right, scale * left_const, metadata)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_function(expr.left, scale * right_const, metadata)
            return None

        if (
            not isinstance(expr, FunctionCall)
            or expr.func_name not in _MONOTONE_DOMAINS
            or len(expr.args) != 1
        ):
            return None

        affine_match = _match_affine_var(expr.args[0], 1.0, metadata)
        if affine_match is None:
            return None
        flat_idx, arg_coeff, arg_offset = affine_match
        if flat_idx is None or abs(arg_coeff) <= 1e-12:
            return None
        return expr.func_name, scale, flat_idx, arg_coeff, arg_offset

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in _rows_until(model, deadline):
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms = _cached_flat_terms(model, constraint.body)

            constant_term = 0.0
            function_match: Optional[tuple[str, float, int, float, float]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = self._match_scaled_function(term, scale, metadata)
                if match is None or function_match is not None:
                    matches_pattern = False
                    break
                function_match = match

            if not matches_pattern or function_match is None:
                continue

            func_name, func_coeff, flat_idx, arg_coeff, arg_offset = function_match
            if abs(func_coeff) <= 1e-12:
                continue

            domain_lb, domain_ub = _MONOTONE_DOMAINS[func_name]
            arg_lb = domain_lb
            arg_ub = domain_ub
            arg_endpoint_a = arg_coeff * float(tightened_lb[flat_idx]) + arg_offset
            arg_endpoint_b = arg_coeff * float(tightened_ub[flat_idx]) + arg_offset
            current_arg_lb = min(arg_endpoint_a, arg_endpoint_b)
            current_arg_ub = max(arg_endpoint_a, arg_endpoint_b)
            # #1537: proofs on outward-widened intervals; see the equality rule.
            arg_slack = _roundoff_slack(arg_endpoint_a, arg_endpoint_b, arg_offset)
            if domain_lb is not None:
                current_arg_lb = max(current_arg_lb, domain_lb)
            if domain_ub is not None:
                current_arg_ub = min(current_arg_ub, domain_ub)
            if current_arg_lb > current_arg_ub + _EMPTY_INTERVAL_FEAS_TOL + arg_slack:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"{func_name} argument domain is empty on the current box",
                )

            wide_lo = current_arg_lb - arg_slack
            wide_hi = current_arg_ub + arg_slack
            if domain_lb is not None:
                wide_lo = max(wide_lo, domain_lb)
            if domain_ub is not None:
                wide_hi = min(wide_hi, domain_ub)
            wide_min = _monotone_function_value(func_name, wide_lo)
            wide_max = _monotone_function_value(func_name, wide_hi)
            rhs = -constant_term / func_coeff
            proof_tol = (_EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack(constant_term)) / abs(
                func_coeff
            )
            if func_coeff > 0.0:
                if rhs < wide_min - _roundoff_slack(wide_min) - proof_tol:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"{func_name}(argument) cannot be <= {rhs}",
                    )
                # #1537 D: invert an UPPER enclosure of the rhs quotient.
                upper = _inverse_monotone_upper(func_name, div_up(-constant_term, func_coeff))
                if upper is not None:
                    arg_ub = upper if arg_ub is None else min(arg_ub, upper)
            else:
                if rhs > wide_max + _roundoff_slack(wide_max) + proof_tol:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"{func_name}(argument) cannot be >= {rhs}",
                    )
                # func_coeff < 0: f(arg) >= rhs; invert a LOWER enclosure (#1537 D).
                lower = _inverse_monotone_lower(func_name, div_down(-constant_term, func_coeff))
                if lower is not None:
                    arg_lb = lower if arg_lb is None else max(arg_lb, lower)

            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                flat_idx,
                arg_coeff,
                arg_offset,
                arg_lb=arg_lb,
                arg_ub=arg_ub,
            )

        return tightened_lb, tightened_ub


class QuadraticEqualityBoundsRule(NonlinearBoundTighteningRule):
    """Propagate equalities like x == y**2 in both directions."""

    name = "quadratic_equality_bounds"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    @staticmethod
    def _square_interval(lb: float, ub: float) -> tuple[float, float]:
        if lb <= 0.0 <= ub:
            return 0.0, max(lb * lb, ub * ub)
        return min(lb * lb, ub * ub), max(lb * lb, ub * ub)

    @staticmethod
    def _intersect_square_preimage(
        current_lb: float,
        current_ub: float,
        square_lb: float,
        square_ub: float,
    ) -> Optional[tuple[float, float]]:
        """Intersect ``[current_lb, current_ub]`` with the preimage of the square range.

        **Caller contract (#1397).** ``square_lb`` and ``square_ub`` must already be
        widened *outward* by a bound on their own accumulation error -- lower side
        reduced, upper side raised. This routine holds no scale information of its
        own: the constant that produced the range was a signed sum computed by the
        caller, and a range that cancelled to ~0 from 1e12-magnitude terms is
        indistinguishable here from a genuine tight range. Both call sites widen
        before calling.
        """
        # A negative upper bound on x**2 is infeasible only beyond the
        # feasibility tolerance; a sub-tolerance residual (e.g. eps-scale hull
        # perspective) is feasible within tolerance and deferred (issue #27a).
        if square_ub < -_EMPTY_INTERVAL_FEAS_TOL:
            return None
        square_lb = max(0.0, square_lb)
        square_ub = max(0.0, square_ub)

        radius = float(np.sqrt(square_ub))
        new_lb = max(current_lb, -radius)
        new_ub = min(current_ub, radius)

        if square_lb > 1e-12:
            inner = float(np.sqrt(square_lb))
            if new_lb >= 0.0:
                new_lb = max(new_lb, inner)
            elif new_ub <= 0.0:
                new_ub = min(new_ub, -inner)

        # #1397: the crossover of two bounds in the VARIABLE's units, so 1e-12 is a
        # yardstick only for an O(1) box. ``new_lb``/``new_ub`` come out of a sqrt and
        # a max/min against the incoming box; at |x| ~ 1e6 one ulp already exceeds
        # 1e-12, and the ``None`` return is pruned upstream with no further check.
        if new_lb > new_ub + 1e-12 + _roundoff_slack(new_lb, new_ub):
            return None
        return new_lb, new_ub

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in _rows_until(model, deadline):
            if getattr(constraint, "sense", None) != "==":
                continue

            terms = _cached_flat_terms(model, constraint.body)

            constant_term = 0.0
            # #1397: magnitude of the constant leaves, for the round-off bound below.
            constant_absum = 0.0
            affine_match: Optional[tuple[Optional[int], float, float]] = None
            square_match: Optional[tuple[int, float]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    constant_absum += abs(scale * const_val)
                    continue

                match = _match_scaled_square_var(term, scale, metadata)
                if match is not None:
                    if square_match is not None:
                        matches_pattern = False
                        break
                    square_match = match
                    continue

                affine_term = _match_affine_var(term, scale, metadata)
                if affine_term is None:
                    matches_pattern = False
                    break
                affine_match = _merge_single_variable_affine(affine_match, affine_term)
                if affine_match is None:
                    matches_pattern = False
                    break

            if (
                not matches_pattern
                or affine_match is None
                or square_match is None
                or affine_match[0] is None
            ):
                continue

            linear_idx, linear_coeff, linear_offset = affine_match
            assert linear_idx is not None
            square_idx, square_coeff = square_match
            if abs(linear_coeff) <= 1e-12 or abs(square_coeff) <= 1e-12:
                continue

            square_min, square_max = self._square_interval(
                float(tightened_lb[square_idx]),
                float(tightened_ub[square_idx]),
            )
            # #1397: each endpoint is a difference of ``constant_term`` (itself a
            # signed sum) and the scaled square activity, so widen it outward by the
            # round-off of the terms THAT endpoint was built from before it becomes a
            # bound on the affine variable. Without any widening the row
            # ``y - x**2 + 0.4e + M - M - 0.3e == 0`` -- exactly
            # ``y = x**2 + 0.1*spacing(M)``, satisfied by y = 0, x ~ 0 -- pushed y's
            # lower bound up to ``0.3*spacing(M)`` and the box against a fixed y = 0
            # was reported empty, at every M from 1e12 up.
            #
            # #1415: the two endpoints get their OWN bounds, not one bound over both.
            # ``square_min`` and ``square_max`` are separate quantities from separate
            # arithmetic; lumping them charged each endpoint the other's magnitude,
            # which on a wide box is the only magnitude in the row. Measured on
            # ``y - exp(x) == 0, x - y**2 == 0`` over the +-9.999e19 default box: at
            # ``square_min = 0`` (a box straddling zero, so the lower endpoint is
            # exact) the lumped bound still widened it by ``8*eps*square_max``, which
            # at ``y_ub ~ 1.8e5`` is 5.6e-5 -- so ``x >= 0`` came out ``x >= -5.6e-5``
            # and the ``y = exp(x) >= 1`` that follows from it came out 0.99994.
            # Per-endpoint is the same bound restricted to the terms each endpoint
            # actually differenced, so it is no less sound and never wider.
            linear_target_min_value = -constant_term - square_coeff * square_min
            linear_target_max_value = -constant_term - square_coeff * square_max
            linear_target_min_slack = _roundoff_slack(constant_absum, square_coeff * square_min)
            linear_target_max_slack = _roundoff_slack(constant_absum, square_coeff * square_max)
            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                linear_idx,
                linear_coeff,
                linear_offset,
                arg_lb=min(
                    linear_target_min_value - linear_target_min_slack,
                    linear_target_max_value - linear_target_max_slack,
                ),
                arg_ub=max(
                    linear_target_min_value + linear_target_min_slack,
                    linear_target_max_value + linear_target_max_slack,
                ),
            )

            linear_endpoint_a = linear_coeff * float(tightened_lb[linear_idx]) + linear_offset
            linear_endpoint_b = linear_coeff * float(tightened_ub[linear_idx]) + linear_offset
            linear_expr_lb = min(linear_endpoint_a, linear_endpoint_b)
            linear_expr_ub = max(linear_endpoint_a, linear_endpoint_b)
            # #1397: the round-off of ``(-constant_term - linear_expr) / square_coeff``,
            # in square units -- the row-unit accumulation error divided by the same
            # coefficient the value was.
            #
            # #1415: per endpoint, for the reason given above. The two required-square
            # values are built from ``linear_expr_lb`` and ``linear_expr_ub``
            # separately, and the squared box error that used to be lumped in with
            # them belongs to ``square_min``/``square_max`` -- so it is carried by
            # those two candidates below instead of being charged to both sides of
            # the required range.
            required_square_lo_value = (-constant_term - linear_expr_lb) / square_coeff
            required_square_hi_value = (-constant_term - linear_expr_ub) / square_coeff
            required_square_lo_slack = _roundoff_slack(constant_absum, linear_expr_lb) / abs(
                square_coeff
            )
            required_square_hi_slack = _roundoff_slack(constant_absum, linear_expr_ub) / abs(
                square_coeff
            )
            required_square_lb = min(
                required_square_lo_value - required_square_lo_slack,
                required_square_hi_value - required_square_hi_slack,
            )
            required_square_ub = max(
                required_square_lo_value + required_square_lo_slack,
                required_square_hi_value + required_square_hi_slack,
            )
            # Widen outward before these become inferences: lower side down, upper
            # side up. Raising the lower side on rounding noise would lift the inner
            # radius in the preimage below and carve out feasible points. ``square_min``
            # and ``square_max`` come out of one multiply each, so each carries its own
            # magnitude's round-off and no other's.
            feasible_square_lb = max(
                required_square_lb,
                square_min - _roundoff_slack(square_min),
                0.0,
            )
            feasible_square_ub = min(
                required_square_ub,
                square_max + _roundoff_slack(square_max),
            )
            if feasible_square_lb > feasible_square_ub + 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "square term range cannot satisfy linked equality",
                )

            interval = self._intersect_square_preimage(
                float(tightened_lb[square_idx]),
                float(tightened_ub[square_idx]),
                feasible_square_lb,
                feasible_square_ub,
            )
            if interval is None:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"required interval for flat variable {square_idx} is empty",
                )
            new_lb, new_ub = _apply_integrality(
                interval[0],
                interval[1],
                metadata.flat_var_types[square_idx],
            )
            if new_lb <= new_ub:
                tightened_lb[square_idx] = new_lb
                tightened_ub[square_idx] = new_ub
            else:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"required interval for flat variable {square_idx} is empty",
                )

        return tightened_lb, tightened_ub


class SquareDifferenceLowerBoundRule(NonlinearBoundTighteningRule):
    """Tighten the leading square in equalities like ``a*x^2 = b*y^2 + c*z^2``."""

    name = "square_difference_lower_bound"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _collect_square_sum(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
        square_coeffs: dict[int, float],
        const_absum: list[float],
    ) -> Optional[float]:
        """Walk a square-sum row, returning its signed constant.

        ``const_absum`` is a one-element accumulator, threaded the same way
        ``square_coeffs`` is: it collects ``sum(|scale * const|)`` over the constant
        leaves, so the caller knows the *magnitude of the terms* the returned
        constant is a signed sum of. The return value alone cannot supply that --
        a constant that cancelled to ~0 from two 1e12 leaves is indistinguishable
        from a genuine 0 -- and #1397 needs it to bound the sum's round-off.
        """
        const_val = _constant_value(expr)
        if const_val is not None:
            const_absum[0] += abs(scale * const_val)
            return scale * const_val

        match = _match_scaled_square_var(expr, scale, metadata)
        if match is not None:
            flat_idx, coeff = match
            square_coeffs[flat_idx] = square_coeffs.get(flat_idx, 0.0) + coeff
            return 0.0

        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._collect_square_sum(
                expr.operand, -scale, metadata, square_coeffs, const_absum
            )

        if isinstance(expr, BinaryOp):
            if expr.op == "+":
                left = self._collect_square_sum(
                    expr.left, scale, metadata, square_coeffs, const_absum
                )
                right = self._collect_square_sum(
                    expr.right, scale, metadata, square_coeffs, const_absum
                )
                if left is None or right is None:
                    return None
                return left + right
            if expr.op == "-":
                left = self._collect_square_sum(
                    expr.left, scale, metadata, square_coeffs, const_absum
                )
                right = self._collect_square_sum(
                    expr.right, -scale, metadata, square_coeffs, const_absum
                )
                if left is None or right is None:
                    return None
                return left + right
            if expr.op == "*":
                left_const = _constant_value(expr.left)
                if left_const is not None:
                    return self._collect_square_sum(
                        expr.right, scale * left_const, metadata, square_coeffs, const_absum
                    )
                right_const = _constant_value(expr.right)
                if right_const is not None:
                    return self._collect_square_sum(
                        expr.left, scale * right_const, metadata, square_coeffs, const_absum
                    )
            if expr.op == "/":
                right_const = _constant_value(expr.right)
                if right_const is not None:
                    return self._collect_square_sum(
                        expr.left, scale / right_const, metadata, square_coeffs, const_absum
                    )

        return None

    def _decompose_row(
        self,
        constraint,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[float, int, float, list[tuple[int, float]], float]]:
        """Bound-independent half of the row scan; see ``_cached_row_structure``.

        Returns ``(constant_term, target_idx, target_scale, positive, const_absum)``
        -- the negative square this rule tightens, the positive squares whose box
        activity bounds it, and the magnitude of the constant leaves
        ``constant_term`` is a signed sum of (#1397) -- or ``None`` when the row is
        not of that shape. The arithmetic is exactly the prologue this replaced, in
        the same order, and ``positive`` keeps its original list order because the
        caller sums over it in floating point.
        """
        if getattr(constraint, "sense", None) != "==":
            return None

        square_coeffs: dict[int, float] = {}
        const_absum = [0.0]
        constant = self._collect_square_sum(
            constraint.body, 1.0, metadata, square_coeffs, const_absum
        )
        if constant is None:
            return None
        constant_term = float(constant)

        square_coeffs = {idx: coeff for idx, coeff in square_coeffs.items() if abs(coeff) > 1e-12}
        negative = [(idx, coeff) for idx, coeff in square_coeffs.items() if coeff < -1e-12]
        positive = [(idx, coeff) for idx, coeff in square_coeffs.items() if coeff > 1e-12]
        if len(negative) != 1 or not positive:
            return None

        target_idx, target_coeff = negative[0]
        return constant_term, target_idx, -target_coeff, positive, const_absum[0]

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in _rows_until(model, deadline):
            structure = _cached_row_structure(
                model,
                self.name,
                constraint,
                metadata,
                lambda c=constraint: self._decompose_row(c, metadata),
            )
            if structure is None:
                continue
            constant_term, target_idx, target_scale, positive, const_absum = structure

            rhs_lb = constant_term
            rhs_ub = constant_term
            # #1397: these bound the magnitude of the terms ``rhs_lb``/``rhs_ub`` are
            # signed sums of, so their round-off can be bounded. Seeded from the
            # constant leaves' own abs-sum rather than from ``abs(constant_term)``,
            # which would read a cancelled constant as carrying no scale.
            #
            # #1415: one accumulator per side. ``rhs_lb`` sums the ``sq_lb`` terms and
            # ``rhs_ub`` the ``sq_ub`` ones; a single accumulator over both charged
            # each side the other's magnitude, which on a wide box (``sq_ub`` the
            # square of a 1e10 bound, ``sq_lb`` exactly 0) is the whole bound.
            rhs_lb_absum = const_absum
            rhs_ub_absum = const_absum
            for flat_idx, coeff in positive:
                sq_lb, sq_ub = QuadraticEqualityBoundsRule._square_interval(
                    float(tightened_lb[flat_idx]),
                    float(tightened_ub[flat_idx]),
                )
                rhs_lb += coeff * sq_lb
                rhs_ub += coeff * sq_ub
                rhs_lb_absum += abs(coeff * sq_lb)
                rhs_ub_absum += abs(coeff * sq_ub)

            rhs_lb_slack = _roundoff_slack(rhs_lb_absum)
            rhs_ub_slack = _roundoff_slack(rhs_ub_absum)
            # Infeasible only beyond the feasibility tolerance (issue #27a); a
            # sub-tolerance residual is feasible within tolerance and deferred.
            # And only beyond the round-off of the activity sum (#1397): squares of
            # a 1e7-wide box already reach 1e14, where one ulp of the sum dwarfs the
            # 1e-6 tolerance and a rounding residual reads as proof of infeasibility.
            if rhs_ub < -_EMPTY_INTERVAL_FEAS_TOL - rhs_ub_slack:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "positive square activity cannot balance the negative square",
                )

            # #1397: widen each side outward, in square units, before it becomes an
            # inference. A lower bound on x**2 must be UNDER-estimated and an upper
            # bound OVER-estimated; the previous ``max(0.0, .../target_scale)`` did
            # neither, so a rounding residual could raise the inner radius and carve
            # a hole that excludes a feasible point.
            feasible_square_lb = max(0.0, (rhs_lb - rhs_lb_slack) / target_scale)
            feasible_square_ub = max(0.0, (rhs_ub + rhs_ub_slack) / target_scale)
            if feasible_square_lb > feasible_square_ub + _EMPTY_INTERVAL_FEAS_TOL:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "required square interval is empty",
                )

            interval = QuadraticEqualityBoundsRule._intersect_square_preimage(
                float(tightened_lb[target_idx]),
                float(tightened_ub[target_idx]),
                feasible_square_lb,
                feasible_square_ub,
            )
            if interval is None:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"required interval for flat variable {target_idx} is empty",
                )

            new_lb, new_ub = _apply_integrality(
                interval[0],
                interval[1],
                metadata.flat_var_types[target_idx],
            )
            if new_lb <= new_ub:
                tightened_lb[target_idx] = new_lb
                tightened_ub[target_idx] = new_ub
            else:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"required interval for flat variable {target_idx} is empty",
                )

        return tightened_lb, tightened_ub


def _match_scaled_constant_division(expr, scale: float) -> Optional[tuple[float, object]]:
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_scaled_constant_division(expr.operand, -scale)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_scaled_constant_division(expr.right, scale * left_const)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_scaled_constant_division(expr.left, scale * right_const)
        return None

    if not isinstance(expr, BinaryOp) or expr.op != "/":
        return None
    numerator = _constant_value(expr.left)
    if numerator is None or abs(numerator) <= 1e-12:
        return None
    return scale * numerator, expr.right


def _match_scaled_negative_power(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[int, float, float]]:
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_scaled_negative_power(expr.operand, -scale, metadata)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_scaled_negative_power(expr.right, scale * left_const, metadata)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_scaled_negative_power(expr.left, scale * right_const, metadata)
        return None

    if not isinstance(expr, BinaryOp) or expr.op != "**":
        return None
    exponent = _constant_value(expr.right)
    if exponent is None or exponent >= 0.0:
        return None
    flat_idx = metadata.scalar_flat_index(expr.left)
    if flat_idx is None:
        return None
    return flat_idx, float(exponent), scale


class PositiveAffineReciprocalBoundsRule(NonlinearBoundTighteningRule):
    """Propagate ``c / positive_affine >= rhs`` into affine upper bounds."""

    name = "positive_affine_reciprocal_bounds"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()
        n_vars = len(flat_lb)

        for constraint in _rows_until(model, deadline):
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms = _cached_flat_terms(model, constraint.body)

            constant_term = 0.0
            reciprocal_match: Optional[tuple[float, object]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = _match_scaled_constant_division(term, scale)
                if match is None or reciprocal_match is not None:
                    matches_pattern = False
                    break
                reciprocal_match = match

            if not matches_pattern or reciprocal_match is None or constant_term <= 0.0:
                continue

            scaled_numerator, denominator = reciprocal_match
            if scaled_numerator >= 0.0:
                continue

            affine = _linearize_affine_expr(denominator, 1.0, metadata, n_vars)
            if affine is None:
                continue
            coeff, const = affine
            arg_lb, arg_ub = _affine_bounds(coeff, const, tightened_lb, tightened_ub)
            if arg_lb <= 0.0:
                continue

            rhs = -scaled_numerator / constant_term
            if not np.isfinite(rhs):
                continue
            # #1537: arg_lb is an affine image of the box (rounded at the magnitude of
            # ``const`` and the box), rhs a quotient; prove only past both roundings.
            if arg_lb > rhs + _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack(arg_lb, const, rhs):
                _prove_infeasible(
                    self.name,
                    constraint,
                    "positive affine denominator exceeds reciprocal upper bound",
                )
            if arg_ub <= rhs + 1e-12:
                continue
            _tighten_affine_upper_bound(tightened_lb, tightened_ub, metadata, coeff, const, rhs)

        return tightened_lb, tightened_ub


class NegativePowerBoundsRule(NonlinearBoundTighteningRule):
    """Infer strict positive lower bounds from ``x**p <= affine`` for ``p < 0``."""

    name = "negative_power_bounds"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _decompose_row(
        self,
        model: Model,
        constraint,
        metadata: FlatVariableMetadata,
        n_vars: int,
    ) -> Optional[tuple[np.ndarray, float, int, float, float]]:
        """Bound-independent half of the row scan; see ``_cached_row_structure``.

        Returns ``(affine_coeff, affine_const, base_idx, exponent, power_scale)``
        for a row of the form ``power_scale * x**exponent + affine <= rhs`` with
        ``exponent < 0``, or ``None``. The arithmetic is exactly the prologue this
        replaced, in the same order.

        ``affine_coeff`` is shared by every node that reads this row, so it is
        marked read-only: a downstream in-place write would otherwise corrupt the
        cached row silently, and the box it produced with it (CLAUDE.md 3 --
        refuse loudly rather than approximate).
        """
        if getattr(constraint, "sense", None) not in ("<=", "=="):
            return None

        terms = _cached_flat_terms(model, constraint.body)

        power_match: Optional[tuple[int, float, float]] = None
        affine_coeff = np.zeros(n_vars, dtype=np.float64)
        affine_const = 0.0

        for scale, term in terms:
            match = _match_scaled_negative_power(term, scale, metadata)
            if match is not None:
                if power_match is not None:
                    return None
                power_match = match
                continue

            affine = _linearize_affine_expr(term, scale, metadata, n_vars)
            if affine is None:
                return None
            affine_coeff += affine[0]
            affine_const += affine[1]

        if power_match is None:
            return None

        base_idx, exponent, power_scale = power_match
        if power_scale <= 0.0:
            return None
        if abs(float(affine_coeff[base_idx])) > 1e-12:
            return None

        affine_coeff.setflags(write=False)
        return affine_coeff, affine_const, base_idx, exponent, power_scale

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()
        n_vars = len(flat_lb)

        for constraint in _rows_until(model, deadline):
            structure = _cached_row_structure(
                model,
                self.name,
                constraint,
                metadata,
                lambda c=constraint: self._decompose_row(model, c, metadata, n_vars),
            )
            if structure is None:
                continue
            affine_coeff, affine_const, base_idx, exponent, power_scale = structure

            if tightened_lb[base_idx] < -1e-12 or tightened_ub[base_idx] <= 0.0:
                continue

            affine_min, _affine_max = _affine_bounds(
                affine_coeff,
                affine_const,
                tightened_lb,
                tightened_ub,
            )
            rhs_ub = -affine_min / power_scale
            if rhs_ub <= 0.0:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "negative power has no positive upper allowance",
                )
            if not np.isfinite(rhs_ub):
                continue

            lower = float(rhs_ub ** (1.0 / exponent))
            if not np.isfinite(lower):
                continue
            new_lb = max(float(tightened_lb[base_idx]), lower)
            new_ub = float(tightened_ub[base_idx])
            new_lb, new_ub = _apply_integrality(
                new_lb,
                new_ub,
                metadata.flat_var_types[base_idx],
            )
            if new_lb <= new_ub:
                tightened_lb[base_idx] = new_lb
                tightened_ub[base_idx] = new_ub
                continue
            _prove_infeasible(
                self.name,
                constraint,
                f"required positive lower bound {new_lb} exceeds upper bound {new_ub}",
            )

        return tightened_lb, tightened_ub


class ReciprocalBoundsRule(NonlinearBoundTighteningRule):
    """Tighten sign-stable affine denominators in simple reciprocal constraints."""

    name = "reciprocal_bounds"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _match_scaled_reciprocal(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[float, int, float, float]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_reciprocal(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_reciprocal(expr.right, scale * left_const, metadata)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_reciprocal(expr.left, scale * right_const, metadata)
            return None

        if not isinstance(expr, BinaryOp) or expr.op != "/":
            return None

        numerator = _constant_value(expr.left)
        if numerator is None or abs(numerator) <= 1e-12:
            return None

        denominator = _match_affine_var(expr.right, 1.0, metadata)
        if denominator is None:
            return None
        flat_idx, denom_coeff, denom_offset = denominator
        if flat_idx is None or abs(denom_coeff) <= 1e-12:
            return None

        return scale * numerator, flat_idx, denom_coeff, denom_offset

    @staticmethod
    def _argument_interval_for_leq(
        numerator: float,
        rhs: float,
        arg_lo: float,
        arg_hi: float,
    ) -> Optional[tuple[Optional[float], Optional[float]] | _ReciprocalIntervalInfeasible]:
        vals = (numerator / arg_lo, numerator / arg_hi)
        min_val = min(vals)
        max_val = max(vals)
        if rhs >= max_val - 1e-12:
            return None
        if rhs < min_val - 1e-12:
            return _RECIPROCAL_INTERVAL_INFEASIBLE

        # numerator > 0 always yields a LOWER bound on the argument (either sign of a
        # sign-definite argument), numerator < 0 an UPPER one; round outward (#1537 D).
        if numerator > 0.0:
            return div_down(numerator, rhs), None
        return None, div_up(numerator, rhs)

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in _rows_until(model, deadline):
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms = _cached_flat_terms(model, constraint.body)

            constant_term = 0.0
            reciprocal_match: Optional[tuple[float, int, float, float]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = self._match_scaled_reciprocal(term, scale, metadata)
                if match is None or reciprocal_match is not None:
                    matches_pattern = False
                    break
                reciprocal_match = match

            if not matches_pattern or reciprocal_match is None:
                continue

            numerator, flat_idx, denom_coeff, denom_offset = reciprocal_match
            arg_endpoint_a = denom_coeff * float(tightened_lb[flat_idx]) + denom_offset
            arg_endpoint_b = denom_coeff * float(tightened_ub[flat_idx]) + denom_offset
            arg_lo = min(arg_endpoint_a, arg_endpoint_b)
            arg_hi = max(arg_endpoint_a, arg_endpoint_b)
            if arg_lo <= 0.0 <= arg_hi:
                continue

            rhs = -constant_term
            arg_interval = self._argument_interval_for_leq(
                numerator,
                rhs,
                arg_lo,
                arg_hi,
            )
            if isinstance(arg_interval, _ReciprocalIntervalInfeasible):
                _prove_infeasible(
                    self.name,
                    constraint,
                    "reciprocal activity exceeds the upper bound on the sign-stable box",
                )
            elif arg_interval is None:
                continue
            else:
                arg_lb, arg_ub = arg_interval
            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                flat_idx,
                denom_coeff,
                denom_offset,
                arg_lb=arg_lb,
                arg_ub=arg_ub,
            )

        return tightened_lb, tightened_ub


def _accumulate_affine(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
    coeffs: dict[int, float],
    const: list[float],
) -> bool:
    """Accumulate ``scale * expr`` into ``coeffs``/``const`` as an affine form.

    Returns ``False`` (and leaves the accumulators in an undefined partial state,
    which the caller discards) the moment a non-affine node is encountered — a
    variable*variable product, a power, a function call, or a division by a
    non-constant. This is deliberately conservative so the bilinear rule never
    mis-reads a nonlinear term as affine.
    """
    const_val = _constant_value(expr)
    if const_val is not None:
        const[0] += scale * const_val
        return True

    flat_idx = metadata.scalar_flat_index(expr)
    if flat_idx is not None:
        coeffs[flat_idx] = coeffs.get(flat_idx, 0.0) + scale
        return True

    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _accumulate_affine(expr.operand, -scale, metadata, coeffs, const)

    if isinstance(expr, BinaryOp) and expr.op in ("+", "-"):
        if not _accumulate_affine(expr.left, scale, metadata, coeffs, const):
            return False
        right_scale = scale if expr.op == "+" else -scale
        return _accumulate_affine(expr.right, right_scale, metadata, coeffs, const)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _accumulate_affine(expr.right, scale * left_const, metadata, coeffs, const)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _accumulate_affine(expr.left, scale * right_const, metadata, coeffs, const)
        return False

    if isinstance(expr, BinaryOp) and expr.op == "/":
        denom = _constant_value(expr.right)
        if denom is not None and abs(denom) > 1e-12:
            return _accumulate_affine(expr.left, scale / denom, metadata, coeffs, const)
        return False

    return False


def _affine_box_interval(
    coeffs: dict[int, float],
    const: float,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, float]:
    """Interval of an affine form over the variable box (``0 * inf = nan`` safe)."""
    lo = const
    hi = const
    for idx, c in coeffs.items():
        if abs(c) <= 1e-12:
            # A (numerically) zero coefficient contributes nothing; skipping it
            # avoids ``0 * inf = nan`` when the variable is unbounded.
            continue
        if c > 0.0:
            lo += c * float(lb[idx])
            hi += c * float(ub[idx])
        else:
            lo += c * float(ub[idx])
            hi += c * float(lb[idx])
    return lo, hi


class BilinearProductEqualityRule(NonlinearBoundTighteningRule):
    """Reverse-divide ``v * affine = c`` equalities to bound an otherwise-free ``v``.

    A constraint whose body is ``s·(v · F) + R == 0`` — where ``v`` is a single
    variable appearing *only* in the bilinear product, ``F`` is an affine form in
    the other variables that is sign-stable (its interval excludes 0), and ``R``
    is the affine remainder — pins ``v = (-R/s) / F``. Interval-dividing the
    numerator box by the (bounded-away-from-zero) ``F`` box yields a sound
    enclosure of ``v``, which is intersected with its current bounds.

    This is the only mechanism that can bound a variable appearing solely in
    nonlinear constraints (e.g. the mole-fraction ratios ``x = a/(affine)`` in
    ex6_1_4), where McCormick cannot even build an envelope without a prior
    finite bound and linear-constraint OBBT has nothing to act on.
    """

    name = "bilinear_product_equality"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def _match_product_term(
        self, term, metadata: FlatVariableMetadata
    ) -> Optional[tuple[float, int, dict[int, float], float]]:
        """Match ``s · (v · F)``; return ``(s, v_idx, F_coeffs, F_const)`` or None."""
        scale = 1.0
        expr = term
        # Peel off leading negations and constant factors.
        while True:
            if isinstance(expr, UnaryOp) and expr.op == "neg":
                scale = -scale
                expr = expr.operand
                continue
            if isinstance(expr, BinaryOp) and expr.op == "*":
                left_const = _constant_value(expr.left)
                if left_const is not None:
                    scale *= left_const
                    expr = expr.right
                    continue
                right_const = _constant_value(expr.right)
                if right_const is not None:
                    scale *= right_const
                    expr = expr.left
                    continue
            break

        if not (isinstance(expr, BinaryOp) and expr.op == "*"):
            return None

        for var_side, affine_side in ((expr.left, expr.right), (expr.right, expr.left)):
            v_idx = metadata.scalar_flat_index(var_side)
            if v_idx is None:
                continue
            coeffs: dict[int, float] = {}
            const = [0.0]
            if not _accumulate_affine(affine_side, 1.0, metadata, coeffs, const):
                continue
            # F must be a genuine multi-/single-variable affine form NOT containing
            # v (else the product is quadratic in v, a different rule's job).
            coeffs = {i: c for i, c in coeffs.items() if abs(c) > 1e-12}
            if not coeffs or v_idx in coeffs:
                continue
            return scale, v_idx, coeffs, const[0]
        return None

    def _decompose_row(
        self,
        model: Model,
        constraint,
        metadata: FlatVariableMetadata,
    ):
        """Bound-independent half of the row scan; see ``_cached_row_structure``.

        Returns ``(scale_p, v_idx, f_coeffs, f_const, rest_coeffs, rest_const)``
        for a row ``s·(v·F) + R == 0``, or ``None``. The arithmetic is exactly the
        prologue this replaced, in the same order.

        The two coefficient maps are shared by every node that reads this row, so
        they are wrapped read-only: ``_affine_box_interval`` only iterates them,
        and a future in-place write would otherwise corrupt the cached row and
        every box derived from it silently (CLAUDE.md 3 -- refuse loudly).
        """
        if getattr(constraint, "sense", None) != "==":
            return None

        terms = _cached_flat_terms(model, constraint.body)
        product: Optional[tuple[float, int, dict[int, float], float]] = None
        rest_coeffs: dict[int, float] = {}
        rest_const = [0.0]

        for scale, term in terms:
            const_val = _constant_value(term)
            if const_val is not None:
                rest_const[0] += scale * const_val
                continue
            match = self._match_product_term(term, metadata)
            if match is not None:
                if product is not None:
                    return None  # more than one bilinear product: not this pattern
                p_scale, v_idx, f_coeffs, f_const = match
                product = (scale * p_scale, v_idx, f_coeffs, f_const)
                continue
            if not _accumulate_affine(term, scale, metadata, rest_coeffs, rest_const):
                return None  # a second nonlinear term: bail

        if product is None:
            return None

        scale_p, v_idx, f_coeffs, f_const = product
        # v must appear nowhere else (its presence in the remainder would make
        # the isolation ``v = -R/(s·F)`` circular/unsound).
        if abs(rest_coeffs.get(v_idx, 0.0)) > 1e-12:
            return None

        return (
            scale_p,
            v_idx,
            MappingProxyType(f_coeffs),
            f_const,
            MappingProxyType(rest_coeffs),
            rest_const[0],
        )

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()
        eps = 1e-9

        for constraint in _rows_until(model, deadline):
            structure = _cached_row_structure(
                model,
                self.name,
                constraint,
                metadata,
                lambda c=constraint: self._decompose_row(model, c, metadata),
            )
            if structure is None:
                continue
            scale_p, v_idx, f_coeffs, f_const, rest_coeffs, rest_const_val = structure

            f_lo, f_hi = _affine_box_interval(f_coeffs, f_const, tightened_lb, tightened_ub)
            if not (f_lo > eps or f_hi < -eps):
                continue  # F not sign-stable: division by an interval straddling 0

            r_lo, r_hi = _affine_box_interval(
                rest_coeffs, rest_const_val, tightened_lb, tightened_ub
            )
            # s·(v·F) + R == 0  ->  v·F == -R/s. Build the numerator box P.
            num_endpoints = [-r_hi / scale_p, -r_lo / scale_p]
            p_lo = min(num_endpoints)
            p_hi = max(num_endpoints)
            if not (np.isfinite(p_lo) and np.isfinite(p_hi)):
                continue  # unbounded remainder: nothing to divide

            # v == P / F with F bounded away from 0; enclose by interval division.
            candidates = [p_lo / f_lo, p_lo / f_hi, p_hi / f_lo, p_hi / f_hi]
            if any(not np.isfinite(c) or np.isnan(c) for c in candidates):
                continue
            v_lo = min(candidates)
            v_hi = max(candidates)

            if v_hi < float(tightened_ub[v_idx]) - 1e-10:
                tightened_ub[v_idx] = max(float(tightened_lb[v_idx]), v_hi)
            if v_lo > float(tightened_lb[v_idx]) + 1e-10:
                tightened_lb[v_idx] = min(float(tightened_ub[v_idx]), v_lo)

        return tightened_lb, tightened_ub


class FunctionDomainBoundRule(NonlinearBoundTighteningRule):
    """Apply a monotone function's natural domain to its single-variable argument.

    ``log``/``log2``/``log10`` require a strictly positive argument, ``sqrt`` a
    non-negative one, ``log1p`` an argument ``>= -1`` (see ``_MONOTONE_DOMAINS``).
    Wherever such a function appears in the model — objective *or* constraint, and
    inside *any* surrounding expression (e.g. the ``log(x)`` in ``x*log(x)``) — its
    argument must lie in the domain at every evaluable point, so the implied bound
    on the argument variable holds globally.

    :class:`MonotoneFunctionBoundsRule` only derives domain bounds for a function
    that is the *whole* body of a ``<=``/``==`` constraint; it misses functions in
    the objective or nested in products. A free variable fed to ``log`` (e.g.
    MINLPLib ``ex8_5_4``'s ``log(x0)*x0`` objective with all-free variables) then
    keeps an unbounded, sign-unconstrained box: the local NLP wanders into
    ``x <= 0`` where ``log`` is undefined, every node solve fails, and the search
    exhausts with no incumbent — falsely reported ``infeasible`` (issue #265).

    This rule closes that gap for any single-variable affine argument. Soundness:
    the domain bound never removes a point where the function is defined, so no
    feasible solution is cut; multi-variable arguments (``log(x1 - x2)``) and
    non-affine ones are conservatively skipped.
    """

    name = "function_domain_bound"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def tighten(self, model, flat_lb, flat_ub, metadata, deadline=None):
        lb = np.asarray(flat_lb, dtype=np.float64).copy()
        ub = np.asarray(flat_ub, dtype=np.float64).copy()

        def _subexprs(expr):
            if isinstance(expr, BinaryOp):
                return [expr.left, expr.right]
            if isinstance(expr, UnaryOp):
                return [expr.operand]
            if isinstance(expr, (FunctionCall, CustomCall)):
                return list(expr.args)
            if isinstance(expr, IndexExpression):
                return [expr.base]
            if isinstance(expr, SumExpression):
                return [expr.operand]
            if isinstance(expr, SumOverExpression):
                return list(expr.terms)
            if isinstance(expr, MatMulExpression):
                return [expr.left, expr.right]
            return []

        def visit(expr):
            if (frac_base := _fractional_power_base(expr)) is not None:
                # ``b ** p`` with a constant NON-integer ``p`` is undefined (NaN in
                # every evaluator) for ``b < 0`` -- the same ``b >= 0`` domain as
                # ``sqrt(b)``, which ``x ** 0.5`` is. Without this the two
                # spellings of one model disagreed: ``sqrt(x) <= 1.5`` on
                # ``[-5, 5]`` certified ``min x = 0`` while ``x ** 0.5 <= 1.5``
                # returned an uncertified 1.25 with bound -5 (#1493).
                match = _match_affine_var(frac_base, 1.0, metadata)
                if match is not None:
                    flat_idx, coeff, offset = match
                    if flat_idx is not None and abs(coeff) > 1e-12:
                        _tighten_affine_argument_interval(
                            lb, ub, metadata, flat_idx, coeff, offset, arg_lb=0.0
                        )
            if (
                isinstance(expr, FunctionCall)
                and expr.func_name in _MONOTONE_DOMAINS
                and len(expr.args) == 1
            ):
                domain_lb, domain_ub = _MONOTONE_DOMAINS[expr.func_name]
                if domain_lb is not None or domain_ub is not None:
                    match = _match_affine_var(expr.args[0], 1.0, metadata)
                    if match is not None:
                        flat_idx, coeff, offset = match
                        if flat_idx is not None and abs(coeff) > 1e-12:
                            _tighten_affine_argument_interval(
                                lb,
                                ub,
                                metadata,
                                flat_idx,
                                coeff,
                                offset,
                                arg_lb=domain_lb,
                                arg_ub=domain_ub,
                            )

        def walk(root):
            # Iterative over an explicit stack (#1520): a long ``.nl`` row is a
            # left-deep ``+`` chain as deep as its term count, and the recursive
            # walk hit ``RecursionError`` on a few thousand terms -- which the old
            # blanket handler turned into "no domain tightening" for the whole
            # model. Each visit only intersects the box, so visit order does not
            # change the result, and a shared subexpression need be visited once.
            seen: set[int] = set()
            stack = [root]
            while stack:
                expr = stack.pop()
                if id(expr) in seen:
                    continue
                seen.add(id(expr))
                visit(expr)
                stack.extend(_subexprs(expr))

        # #1520: no except. The walk declines by skipping (a non-affine or
        # multi-variable argument is left alone), and an empty argument domain raises
        # ``NonlinearBoundTighteningInfeasible``, which must reach the caller. Any
        # other exception is a defect in the rule; the old handler turned it into
        # "no tightening".
        if model._objective is not None:
            walk(model._objective.expression)
        for c in _rows_until(model, deadline):
            body = getattr(c, "body", None)
            if body is None:
                # A non-algebraic row (disjunction, logical, SOS) has no body. Its
                # functions hold only inside a disjunct, so their domains imply
                # nothing about the global box: skipping forgoes no valid
                # tightening. It used to raise AttributeError, which the removed
                # handler turned into "no tightening" for the WHOLE model (#1520).
                continue
            walk(body)
        return lb, ub


def _fractional_power_base(expr):
    """The base ``b`` of ``b ** p`` for a constant scalar non-integer ``p``, else None."""
    if not (isinstance(expr, BinaryOp) and expr.op == "**"):
        return None
    rexp = expr.right
    if not isinstance(rexp, Constant) or np.asarray(rexp.value).ndim != 0:
        return None
    p = float(rexp.value)
    if not np.isfinite(p) or float(p).is_integer():
        return None
    return expr.left


_PERIODIC_FUNCS = frozenset({"sin", "cos"})
_TWO_PI = 2.0 * np.pi


class PeriodicVariableBoundRule(NonlinearBoundTighteningRule):
    """Restrict a periodic-only variable to a single period (lossless).

    A continuous variable that appears in the model *solely* as the bare argument
    of ``sin``/``cos`` enters the objective and constraints only through
    ``(sin x, cos x)``, which is fully covered by any ``2*pi``-wide window. So the
    box can be shrunk to one period with no loss of feasible behaviour — and a
    finite, single-period box is what lets spatial branch-and-bound and the local
    sub-NLP actually converge on otherwise-unbounded angular variables (e.g.
    ``cos(y)`` over a free ``y``; AC-OPF voltage angles).

    Soundness: the reduction only ever *shrinks* the box (an unbounded side to
    ``[-pi, pi]`` / one period from the finite anchor; a ``> 2*pi`` finite box to
    its first period), and a full period preserves every ``(sin, cos)`` value the
    original box could take. A variable used anywhere outside ``sin``/``cos`` (or
    inside a non-bare argument such as ``cos(2*y)``) is conservatively skipped.
    """

    name = "periodic_variable_bound"

    # NOT anytime (the base-class default, restated because this rule is the reason
    # the default exists). The reduction is justified by what the model does *not*
    # contain: a variable stays in ``periodic - disqualified`` only while no scanned
    # row uses it outside ``sin``/``cos``. A scan cut short at a deadline could miss
    # exactly the disqualifying row and shrink a variable that is genuinely used
    # elsewhere to one period — cutting feasible points. Under a budget this rule is
    # therefore skipped whole; ``deadline`` is accepted and deliberately unused.
    row_scan_is_anytime = False

    def tighten(self, model, flat_lb, flat_ub, metadata, deadline=None):
        del deadline  # see row_scan_is_anytime above: this scan must be complete
        lb = np.asarray(flat_lb, dtype=np.float64).copy()
        ub = np.asarray(flat_ub, dtype=np.float64).copy()
        # #1520: no except. The scan abstains by value (an unrecognized node or a
        # non-algebraic row -> ``None`` -> the box comes back unchanged); any
        # exception is a defect in the rule, which the old handler turned into
        # "no tightening".
        periodic: set[int] = set()
        disqualified: set[int] = set()
        complete = [True]  # cleared if an unrecognized node is encountered

        def _subexprs(expr):
            """Child sub-expressions, or ``None`` if the node type is unknown."""
            if isinstance(expr, BinaryOp):
                return [expr.left, expr.right]
            if isinstance(expr, UnaryOp):
                return [expr.operand]
            if isinstance(expr, (FunctionCall, CustomCall)):
                return list(expr.args)
            if isinstance(expr, IndexExpression):
                return [expr.base]
            if isinstance(expr, SumExpression):
                return [expr.operand]
            if isinstance(expr, SumOverExpression):
                return list(expr.terms)
            if isinstance(expr, MatMulExpression):
                return [expr.left, expr.right]
            return None

        def visit(expr):
            """Classify one node; return its children to scan (none for a leaf)."""
            # A bare scalar-variable leaf reached here is a *non-periodic* use
            # (the sin/cos branch below returns before recursing into its arg).
            idx = metadata.scalar_flat_index(expr)
            if idx is not None:
                disqualified.add(idx)
                return ()
            if isinstance(expr, Variable):  # bare vector variable: non-periodic use
                off = metadata.base_offsets.get(id(expr))
                if off is not None:
                    for k in range(expr.size):
                        disqualified.add(off + k)
                return ()
            if isinstance(expr, (Constant, Parameter)):
                return ()  # childless, no variables
            if (
                isinstance(expr, FunctionCall)
                and expr.func_name in _PERIODIC_FUNCS
                and len(expr.args) == 1
            ):
                aidx = metadata.scalar_flat_index(expr.args[0])
                if aidx is not None:
                    periodic.add(aidx)
                    return ()  # bare var inside sin/cos: accounted for
            subs = _subexprs(expr)
            if subs is None:
                # Unknown node type: cannot prove the variable is periodic-only,
                # so abstain from the whole rule (sound).
                complete[0] = False
                return ()
            return subs

        def walk(root):
            # Iterative over an explicit stack (#1520): the recursive walk hit
            # ``RecursionError`` on a long left-deep ``.nl`` row, which the old
            # blanket handler absorbed. ``periodic``/``disqualified`` are sets, so
            # visit order does not change the verdict, and a node reached twice
            # contributes nothing new the second time.
            seen: set[int] = set()
            stack = [root]
            while stack and complete[0]:
                expr = stack.pop()
                if id(expr) in seen:
                    continue
                seen.add(id(expr))
                stack.extend(visit(expr))

        def _scan():
            """The whole bound-independent scan: which variables are
            periodic-only, or ``None`` if an unrecognized node forbids the
            conclusion. Depends on the expression graph, never on the box."""
            if model._objective is not None:
                walk(model._objective.expression)
            for c in model._constraints:  # must be complete; see row_scan_is_anytime
                body = getattr(c, "body", None)
                if body is None:
                    # A non-algebraic row (disjunction, logical, SOS) may use a
                    # variable outside sin/cos in a form this walk cannot see,
                    # so the scan cannot be complete: abstain from the whole
                    # rule (sound). It used to raise AttributeError, which the
                    # removed blanket handler absorbed into the same abstention
                    # (#1520).
                    return None
                walk(body)
            if not complete[0]:
                return None
            return frozenset(periodic - disqualified)

        periodic_only = _cached_model_structure(model, "periodic_only", metadata, _scan)
        if periodic_only is None:
            return lb, ub  # saw an unrecognized node -> change nothing

        for idx in periodic_only:
            if idx >= len(metadata.flat_var_types):
                continue
            if metadata.flat_var_types[idx] != VarType.CONTINUOUS:
                continue
            lo, hi = float(lb[idx]), float(ub[idx])
            finite_lo, finite_hi = lo > -1e15, hi < 1e15
            if finite_lo and finite_hi:
                if hi - lo > _TWO_PI + 1e-9:
                    hi = lo + _TWO_PI
            elif finite_lo:
                hi = lo + _TWO_PI
            elif finite_hi:
                lo = hi - _TWO_PI
            else:
                lo, hi = -np.pi, np.pi
            lb[idx], ub[idx] = lo, hi
        return lb, ub


_EXPR_CHILD_ATTRS = ("left", "right", "operand", "base")
_EXPR_CHILD_SEQS = ("args", "operands", "terms")


def _collect_expr_flat_indices(expr, metadata: FlatVariableMetadata, out: set) -> None:
    """Collect the flat variable indices referenced anywhere in ``expr``."""
    si = metadata.scalar_flat_index(expr)
    if si is not None:
        out.add(si)
        return
    if isinstance(expr, Variable):
        off = metadata.base_offsets.get(id(expr))
        if off is not None:
            out.update(range(off, off + expr.size))
        return
    if isinstance(expr, (Constant, Parameter)):
        return
    for attr in _EXPR_CHILD_ATTRS:
        child = getattr(expr, attr, None)
        if isinstance(child, _EXPR_TYPES):
            _collect_expr_flat_indices(child, metadata, out)
    for attr in _EXPR_CHILD_SEQS:
        seq = getattr(expr, attr, None)
        if seq:
            for child in seq:
                if isinstance(child, _EXPR_TYPES):
                    _collect_expr_flat_indices(child, metadata, out)


class DefinedVariableForwardRule(NonlinearBoundTighteningRule):
    """Forward-substitution FBBT for variables *defined* by an equality.

    When a scalar variable appears linearly and in isolation in an equality —
    ``c·x_def + g(others) == rhs`` with ``x_def`` absent from ``g`` — it is fully
    determined: ``x_def = (rhs - g)/c``. Bounding ``x_def`` by the interval
    enclosure of ``(rhs - g)/c`` over the current box turns an *unbounded* auxiliary
    (e.g. a division/sqrt slack ``x4 = 4243.28/(x0·x1)``, gear4/nvs05-class) into a
    finite range. That is what lets the McCormick relaxation over the variable's
    subtree stay bounded, so the spatial dual bound becomes *certifiable* instead of
    being dropped as un-rigorous on an unbounded node (issue: nvs/gear unbounded
    auxiliaries). Sound: an interval enclosure of the defining expression is a valid
    enclosure of the variable; bounds are only ever intersected, never loosened --
    the one exception being a sub-tolerance crossing, which is snapped *outward* to
    the enclosure's own endpoint (see ``tighten``) so eps-scale rounding cannot
    manufacture an empty box. The fixpoint loop in :func:`tighten_nonlinear_bounds`
    resolves chained definitions (``x7`` defined via ``x6`` resolves once ``x6`` is
    bounded).
    """

    name = "defined_variable_forward"
    # Each row's inference stands alone (bounds derived from THIS constraint,
    # intersected into the box), so a prefix of the rows is a looser-but-valid
    # tightening -- safe to stop at a deadline mid-scan (#875).
    row_scan_is_anytime = True

    def tighten(self, model, flat_lb, flat_ub, metadata, deadline=None):
        from discopt._relax.convexity.interval import Interval
        from discopt._relax.convexity.interval_eval import evaluate_interval

        out_lb = flat_lb.copy()
        out_ub = flat_ub.copy()

        # Reverse map: flat scalar index -> Variable (for in-call box updates).
        idx_to_var: dict[int, Variable] = {}
        box: dict = {}
        for var in model._variables:
            off = metadata.base_offsets[id(var)]
            sz = var.size
            # #1617: an array variable's enclosure is an ARRAY interval, as
            # ``interval_eval`` reads it (``box[x].lo[i]`` for ``x[i]``). The scalar
            # min/max this used to store crashed every indexed lookup
            # (``IndexError`` on a 0-d array), and an empty variable (``shape=(0,)``)
            # crashed the ``np.min`` itself.
            if var.shape == ():
                box[var] = Interval(float(out_lb[off]), float(out_ub[off]))
                idx_to_var[off] = var
            else:
                box[var] = Interval(
                    np.array(out_lb[off : off + sz], dtype=np.float64).reshape(var.shape),
                    np.array(out_ub[off : off + sz], dtype=np.float64).reshape(var.shape),
                )

        for con in _rows_until(model, deadline):
            if getattr(con, "sense", None) != "==":
                continue
            rhs = float(getattr(con, "rhs", 0.0) or 0.0)
            if not np.isfinite(rhs):
                continue
            terms = _cached_flat_terms(model, con.body)
            if len(terms) < 2:
                continue
            iso = self._isolated_defined_var(terms, metadata, out_lb, out_ub)
            if iso is None:
                continue
            iso_pos, iso_idx, c_iso = iso

            # Interval of the rest = sum_{k != iso_pos} c_k * I(leaf_k).
            rest_lo = 0.0
            rest_hi = 0.0
            ok = True
            for k, (c_k, leaf_k) in enumerate(terms):
                if k == iso_pos:
                    continue
                iv = evaluate_interval(leaf_k, model, box)
                if iv is None or not (np.isfinite(iv.lo) and np.isfinite(iv.hi)):
                    ok = False
                    break
                a, b = c_k * iv.lo, c_k * iv.hi
                rest_lo += min(a, b)
                rest_hi += max(a, b)
            if not ok:
                continue

            # x_iso = (rhs - rest) / c_iso.
            num_lo, num_hi = rhs - rest_hi, rhs - rest_lo
            if c_iso > 0:
                new_lo, new_hi = num_lo / c_iso, num_hi / c_iso
            else:
                new_lo, new_hi = num_hi / c_iso, num_lo / c_iso
            if not (np.isfinite(new_lo) and np.isfinite(new_hi)):
                continue

            upd_lo = max(out_lb[iso_idx], new_lo)
            upd_hi = min(out_ub[iso_idx], new_hi)
            tightened = (
                not np.isfinite(out_lb[iso_idx])
                or not np.isfinite(out_ub[iso_idx])
                or upd_lo > out_lb[iso_idx] + 1e-12
                or upd_hi < out_ub[iso_idx] - 1e-12
            )
            # Decide here what a crossed (``upd_lo > upd_hi``) interval means,
            # rather than constructing the ``Interval`` and letting its dataclass
            # invariant raise out of the whole tightening pass (#1197).
            if upd_lo > upd_hi + _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack(upd_lo, upd_hi):
                # Beyond any rounding slack the defining expression's enclosure
                # and the variable's box share no point. That is a genuine empty
                # interval, reported the way every sibling rule reports one --
                # the caller turns it into ``stats.infeasible``.
                _prove_infeasible(
                    self.name,
                    con,
                    f"defining interval [{new_lo}, {new_hi}] for flat variable "
                    f"{iso_idx} is disjoint from its box "
                    f"[{out_lb[iso_idx]}, {out_ub[iso_idx]}]",
                )
            if upd_lo > upd_hi:
                # Sub-tolerance crossing: floating-point noise on an effectively
                # degenerate range (ex7_3_6 crosses by 3e-15 on a variable whose
                # true range is the single point 1.0), not an infeasibility proof.
                # Collapse to the degenerate box at the enclosure's endpoint, the
                # same repair ``_snap_tolerant_crossovers`` applies box-wide.
                upd_lo = upd_hi
            if tightened:
                out_lb[iso_idx] = upd_lo
                out_ub[iso_idx] = upd_hi
                var = idx_to_var.get(iso_idx)
                if var is not None:  # let later constraints in this pass use it
                    box[var] = Interval(upd_lo, upd_hi)
        return out_lb, out_ub

    @staticmethod
    def _isolated_defined_var(terms, metadata, lb, ub):
        """Find a term whose leaf is a scalar variable that (a) is currently
        unbounded on a side and (b) appears in no other term. Returns
        ``(term_pos, flat_idx, coeff)`` or ``None``."""
        for pos, (coeff, leaf) in enumerate(terms):
            if abs(coeff) < 1e-12:
                continue
            idx = metadata.scalar_flat_index(leaf)
            if idx is None:
                continue
            if np.isfinite(lb[idx]) and np.isfinite(ub[idx]):
                continue  # already bounded; this rule targets free aux variables
            others: set = set()
            for k, (_c, other_leaf) in enumerate(terms):
                if k == pos:
                    continue
                _collect_expr_flat_indices(other_leaf, metadata, others)
            if idx in others:
                continue  # not isolated (appears elsewhere) → cannot forward-solve
            return pos, idx, coeff
        return None


# --- Convex-quadratic (ellipsoid) bound tightening (issue #1193) -------------
#
# A row ``x'Qx + c'x + d <= 0`` whose ``Q`` is positive definite describes an
# ELLIPSOID, and the exact range of every coordinate over an ellipsoid has a
# closed form. Interval FBBT cannot see that: it evaluates the row one term at a
# time, so every cross term ``x_i x_j`` is replaced by the product of two
# independent intervals and the coupling that actually bounds the set is thrown
# away. Measured on ``nvs19`` (8 integer variables, 8 dense quadratic rows, four
# of them positive definite): the root box after Rust FBBT + non-cutoff OBBT is
# ``x_i <= [49, 55, 55, 54, 61, 58, 53, 58]``, while the ellipsoid of the single
# row ``x'Q5 x <= 930`` already gives ``x_i <= [14, 19, 19, 21, 17, 15, 21, 15]``
# -- 2.5x to 4x tighter in every one of the eight dimensions, from one row.
#
# Derivation. With ``Q > 0`` and ``x* = -Q^-1 c / 2``,
#
#     x'Qx + c'x + d <= 0   <=>   (x - x*)' Q (x - x*) <= r2,
#     r2 = (1/4) c' Q^-1 c - d,
#
# and for ``y = x - x*``, Cauchy-Schwarz in the ``Q``-norm gives
# ``|y_i| = |e_i' y| <= sqrt(e_i' Q^-1 e_i) * ||y||_Q <= sqrt(r2 * (Q^-1)_ii)``,
# attained -- so the bound is exact for the row in isolation.
_ELLIPSOID_ENV = "DISCOPT_ELLIPSOID_BOUNDS"

#: Largest row support the rule will factor. The eigendecomposition is O(m^3)
#: and runs once per row per model (the result is bound-independent and cached),
#: but a single dense 10k-variable row would still cost minutes. Rows above the
#: cap are skipped, which only leaves the box looser.
_ELLIPSOID_MAX_SUPPORT = 250

#: Conditioning gate. ``(Q^-1)_ii`` carries a relative error of order
#: ``eps * cond(Q)``; at ``1e8`` that is ~2e-8, comfortably inside the slack
#: below. A worse-conditioned row is refused rather than tightened from digits
#: that are not there (CLAUDE.md 3: no silent approximations).
_ELLIPSOID_MAX_COND = 1e8

#: Safety margin on the computed radius, applied outward. Two orders of
#: magnitude above the round-off the conditioning gate admits, and utterly
#: negligible against a radius of any practical size -- a bound-tightening error
#: in the *tight* direction cuts off feasible points, so the margin is one-sided.
_ELLIPSOID_RADIUS_REL_SLACK = 1e-6
_ELLIPSOID_RADIUS_ABS_SLACK = 1e-9


def _ellipsoid_bounds_enabled() -> bool:
    """True when the convex-quadratic ellipsoid rule is enabled.

    Bound-changing (CLAUDE.md 5), so it ships behind a flag and default-OFF
    until the corpus-wide differential panel passes. Read per call rather than
    at import so a test can flip it without reloading the module.
    """
    return os.environ.get(_ELLIPSOID_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


class ConvexQuadraticEllipsoidRule(NonlinearBoundTighteningRule):
    """Tighten bounds from a positive-definite quadratic row, cross terms included.

    Complements :class:`SeparableQuadraticUpperBoundRule`, which handles the
    diagonal case ``sum_i (a_i x_i^2 + b_i x_i) <= c`` term by term and abstains
    as soon as a row contains a bilinear product. This rule takes exactly those
    rows: it recovers the row's ``(Q, c, d)`` exactly (or abstains -- see
    :func:`~discopt._relax.quadratic_form.extract_quadratic_support`), and when
    ``Q`` is positive definite reads the coordinate ranges straight off the
    ellipsoid.
    """

    name = "convex_quadratic_ellipsoid"
    # Each row's ellipsoid is derived from THAT row alone and intersected into
    # the incoming box, so a prefix of the rows is a looser-but-valid tightening.
    row_scan_is_anytime = True

    def _row_ellipsoids(self, model: Model, constraint, metadata: FlatVariableMetadata):
        """Bound-independent half of the row scan; see ``_cached_row_structure``.

        Returns ``(support, le_box, ge_box)`` where ``le_box`` is the coordinate
        box implied by ``body <= 0`` and ``ge_box`` the one implied by
        ``body >= 0``; either is ``None`` when that direction is not a usable
        ellipsoid. Returns ``None`` when the row is of no use at all. Nothing
        here reads the variable box, so the whole result is cached per model.
        """
        # Cheap structural reject before the expensive expansion. Recognizing a
        # row means expanding it into monomials, and on a high-degree row that
        # expansion is the whole cost: st_e36 row 0 is a quadratic times a
        # squared quadratic (degree 6) and takes 554 ms to expand before the
        # degree check throws it away -- measured as +31.8% wall on that
        # instance (5 interleaved reps, sd 0.16/0.10 s, node_count identical at
        # 83) with zero coordinates tightened. The degree BOUND walks the DAG
        # without expanding, so the same row is rejected in microseconds.
        # The bound is conservative (it can exceed the true degree when leading
        # terms cancel), so skipping on ``> 2`` can only make this rule tighten
        # LESS, never more -- soundness is unaffected. ``None`` means the walk
        # could not bound the degree, and we fall through to exact extraction.
        degree = polynomial_degree_bound(constraint.body)
        if degree is not None and degree > 2:
            return None
        reduced = extract_quadratic_support(constraint.body, model)
        if reduced is None:
            return None
        support, Q, c, d = reduced

        # A single-variable row is the separable rule's job and it does it
        # exactly; below two variables there is no cross term to exploit.
        if not (2 <= len(support) <= _ELLIPSOID_MAX_SUPPORT):
            return None
        if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(c)) and np.isfinite(d)):
            return None

        sym = 0.5 * (Q + Q.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(sym)
        except np.linalg.LinAlgError as exc:
            # Not swallowed silently (CLAUDE.md 7): a failed decomposition means
            # this row yields no bound, and that is recorded rather than guessed.
            logger.debug("%s: eigendecomposition failed on a row: %s", self.name, exc)
            return None

        # ``-body <= 0`` is the same problem with every eigenvalue negated, so a
        # single decomposition serves both senses; at most one of the two can be
        # positive definite.
        le_box = self._ellipsoid_box(eigvals, eigvecs, c, d)
        ge_box = self._ellipsoid_box(-eigvals[::-1], eigvecs[:, ::-1], -c, -d)
        if le_box is None and ge_box is None:
            return None
        return support, le_box, ge_box

    @staticmethod
    def _ellipsoid_box(eigvals, eigvecs, c, d):
        """Coordinate box of ``{x : x'Qx + c'x + d <= 0}`` for ``Q = V diag(w) V'``.

        ``None`` unless ``Q`` is positive definite, well enough conditioned to
        trust, and the ellipsoid is non-degenerate.
        """
        lam_min, lam_max = float(eigvals[0]), float(eigvals[-1])
        if lam_min <= 0.0 or lam_max > _ELLIPSOID_MAX_COND * lam_min:
            return None

        # x* = -Q^-1 c / 2 and r2 = (1/4) c' Q^-1 c - d, both through the
        # decomposition rather than an explicit inverse.
        rotated = eigvecs.T @ c
        qinv_c = eigvecs @ (rotated / eigvals)
        r2 = 0.25 * float(c @ qinv_c) - d
        if not np.isfinite(r2) or r2 <= 0.0:
            # r2 < 0 means the row alone is infeasible. That is a real inference,
            # but a *false* one -- from round-off on a near-degenerate row --
            # would prune a feasible model, so this rule declines to draw it and
            # leaves infeasibility to the search. r2 == 0 is a single point and
            # is left to the node NLP for the same reason.
            return None

        center = -0.5 * qinv_c
        diag_qinv = (eigvecs**2) @ (1.0 / eigvals)
        if not (np.all(np.isfinite(center)) and np.all(diag_qinv >= 0.0)):
            return None
        radius = np.sqrt(r2 * diag_qinv)
        radius = radius * (1.0 + _ELLIPSOID_RADIUS_REL_SLACK) + _ELLIPSOID_RADIUS_ABS_SLACK
        lo = center - radius
        hi = center + radius
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            return None
        return lo, hi

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()
        if not _ellipsoid_bounds_enabled():
            return tightened_lb, tightened_ub

        for constraint in _rows_until(model, deadline):
            sense = getattr(constraint, "sense", None)
            if sense not in ("<=", ">=", "=="):
                continue

            structure = _cached_row_structure(
                model,
                self.name,
                constraint,
                metadata,
                lambda c=constraint: self._row_ellipsoids(model, c, metadata),
            )
            if structure is None:
                continue
            support, le_box, ge_box = structure

            boxes = []
            if sense in ("<=", "==") and le_box is not None:
                boxes.append(le_box)
            if sense in (">=", "==") and ge_box is not None:
                boxes.append(ge_box)
            if not boxes:
                continue

            for lo, hi in boxes:
                for pos, flat_idx in enumerate(support):
                    new_lb = max(float(tightened_lb[flat_idx]), float(lo[pos]))
                    new_ub = min(float(tightened_ub[flat_idx]), float(hi[pos]))

                    var_type = metadata.flat_var_types[flat_idx]
                    if var_type == VarType.BINARY:
                        new_lb = max(new_lb, 0.0)
                        new_ub = min(new_ub, 1.0)
                    if var_type in (VarType.BINARY, VarType.INTEGER):
                        new_lb = float(np.ceil(new_lb - 1e-9))
                        new_ub = float(np.floor(new_ub + 1e-9))

                    if new_lb <= new_ub:
                        tightened_lb[flat_idx] = new_lb
                        tightened_ub[flat_idx] = new_ub
                    elif new_lb <= new_ub + _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack(
                        new_lb, new_ub
                    ):
                        # Sub-tolerance crossover: snap to a degenerate box
                        # instead of pruning, leaving validation to the node NLP.
                        tightened_lb[flat_idx] = new_ub
                        tightened_ub[flat_idx] = new_ub
                    else:
                        _prove_infeasible(
                            self.name,
                            constraint,
                            f"required interval for flat variable {flat_idx} is empty",
                        )

        return tightened_lb, tightened_ub


#: Largest block (connected component of a row's ``Q``) whose integer minimum is
#: found by enumeration, and the most lattice points one enumeration may visit. A
#: block over either limit is bounded by its continuous minimum instead, which is
#: valid and merely weaker.
_INT_BLOCK_MAX_SIZE = 6
_INT_BLOCK_MAX_POINTS = 20_000


def integer_block_quadratic_enabled() -> bool:
    """``DISCOPT_INTEGER_BLOCK_QUAD`` gate for :class:`IntegerBlockQuadraticRule` (#1867).

    Bound-changing (the rule can prove a row infeasible), so gated per CLAUDE.md
    section 5. Read at call time.

    **Graduated (default ON, 2026-10-22).** Firing scan at the declared box, OFF vs
    ON, over 388 local MINLPLib instances (<= 200 vars/cons; 2 more could not be
    scanned -- one missing file, one ``RecursionError`` in the unrelated
    ``_model_contains_nonsmooth_node`` walker): the rule reads 17 rows and proves
    exactly ``ball_mk4_05/10/15`` infeasible, changing nothing else (it never
    tightens a bound). MINLPLib lists 05/10 ``=inf=`` and 15 open (best dual 58.83,
    no primal); 15's row is the same pairs ``100x^2 + 100y^2 - 98x - 98y - 4xy =
    98(x^2 - x) + 98(y^2 - y) + 2(x - y)^2 >= 0`` on Z^2 against ``rhs -1``, so it
    is infeasible by that identity, which the regression test checks exactly.
    Panel, 120 s, interleaved, load < 2: cert-clean (6 rows, 0 problems) and
    net-positive -- OFF certifies none of the three (``time_limit`` at ~78 s, no
    incumbent, 2-6k nodes), ON proves all three infeasible at presolve in 0.1 s.
    SCIP 10 at 60 s: 05 infeasible in 0.6 s, 10 and 15 at its time limit. ``=0`` is
    the opt-out.
    """
    return os.environ.get("DISCOPT_INTEGER_BLOCK_QUAD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _quad_components(Q: np.ndarray) -> list[np.ndarray]:
    """Connected components of the graph whose edges are ``Q``'s nonzero off-diagonals."""
    m = Q.shape[0]
    label = np.full(m, -1, dtype=np.int64)
    comps: list[np.ndarray] = []
    for start in range(m):
        if label[start] >= 0:
            continue
        label[start] = len(comps)
        stack, members = [start], [start]
        while stack:
            i = stack.pop()
            for j in np.nonzero(Q[i])[0]:
                if label[j] < 0:
                    label[j] = len(comps)
                    stack.append(int(j))
                    members.append(int(j))
        comps.append(np.array(sorted(members), dtype=np.int64))
    return comps


def _min_quadratic_block_integer(
    Q: np.ndarray, c: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> tuple[float, float]:
    """Lower bound on ``x'Qx + c'x`` (``Q`` positive definite) over integer points of
    ``[lb, ub]``, each widened by ``_INTEGER_DOMAIN_TOL``. Returns ``(bound, absmag)``,
    ``absmag`` the magnitude the bound was summed from (for the caller's float error).

    With ``x*`` the unconstrained minimiser and ``U`` the value at an integer point
    ``k0`` of the box, every accepted point ``x`` (within tol of an integer ``k``)
    with ``q(x) <= U`` lies in the ellipsoid ``{q <= U}``, so its ``k`` lies in that
    ellipsoid's coordinate box widened by tol; those ``k`` are enumerated. Over a
    window ``|x - k|_inf <= tol``, ``q(x) >= q(k) - tol * ||2Qk + c||_1`` because
    ``Q`` is PSD. Points outside the ellipsoid have ``q > U >= `` the enumerated
    minimum, so the minimum over windows is a bound over every accepted point.
    Falls back to the continuous minimum ``-c'Q^-1c/4`` (valid for any box) when the
    integer range is empty or the enumeration would exceed the point cap.
    """
    tol = _INTEGER_DOMAIN_TOL
    qinv = np.linalg.inv(Q)
    x_star = -0.5 * (qinv @ c)
    q_star = float(-0.25 * (c @ qinv @ c))
    cont = (q_star, abs(q_star) + float(np.abs(c) @ np.abs(x_star)))
    lo = np.ceil(lb - tol)
    hi = np.floor(ub + tol)
    if np.any(lo > hi) or not np.all(np.isfinite(x_star)):
        return cont
    k0 = np.minimum(np.maximum(np.round(x_star), lo), hi)
    if not np.all(np.isfinite(k0)):
        return cont
    u_val = float(k0 @ Q @ k0 + c @ k0)
    gap = max(u_val - q_star, 0.0) * (1.0 + 1e-9) + 1e-12 * (1.0 + abs(u_val))
    diag_qinv = np.diag(qinv)
    if not np.all(diag_qinv > 0.0):
        return cont
    radius = np.sqrt(gap * diag_qinv) * (1.0 + _ELLIPSOID_RADIUS_REL_SLACK)
    radius = radius + _ELLIPSOID_RADIUS_ABS_SLACK
    k_lo = np.maximum(np.ceil(x_star - radius - tol), lo)
    k_hi = np.minimum(np.floor(x_star + radius + tol), hi)
    if np.any(k_lo > k_hi) or not (np.all(np.isfinite(k_lo)) and np.all(np.isfinite(k_hi))):
        return cont
    sizes = (k_hi - k_lo + 1.0).astype(np.float64)
    if float(np.prod(sizes)) > _INT_BLOCK_MAX_POINTS:
        return cont
    axes = [np.arange(a, b + 1.0) for a, b in zip(k_lo, k_hi)]
    pts = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    vals = np.einsum("pi,ij,pj->p", pts, Q, pts) + pts @ c
    grad_l1 = np.abs(2.0 * (pts @ Q) + c).sum(axis=1)
    window = vals - tol * grad_l1
    best = int(np.argmin(window))
    kb = np.abs(pts[best])
    absmag = float(kb @ np.abs(Q) @ kb + np.abs(c) @ kb)
    # Never report below what the continuous minimum already guarantees, nor above
    # the enumerated window bound.
    return max(float(window[best]), q_star), absmag


class IntegerBlockQuadraticRule(NonlinearBoundTighteningRule):
    """Prove a convex quadratic row infeasible over its integer points, cross terms
    included (#1867).

    The row's ``(Q, c, d)`` is recovered exactly (or the row is skipped), and ``Q``
    is split into connected blocks. A block of integer/binary columns that is
    positive definite and small is bounded below by its **integer** minimum
    (:func:`_min_quadratic_block_integer`); a singleton by the univariate minimum
    (integer-domain for an integer column); any other positive-definite block by its
    continuous minimum. A row with a block that is not positive definite (or a
    concave singleton) is skipped. When the bounds' sum exceeds the row's upper
    limit the row is infeasible.

    This is the cross-term generalisation of the integer-domain minimum in
    :class:`SeparableQuadraticUpperBoundRule`: ``ball_mk4_*``'s pairs
    ``100x^2 + 100y^2 - 98x - 98y - 4xy`` reach ``-49`` at ``(0.5, 0.5)`` but never
    go below ``0`` on Z^2, so ``sum <= -1`` is infeasible. It proves; it does not
    tighten, and it only runs on rows with at least one multi-column integer block,
    which neither the separable rule (it abstains on any cross term) nor the
    ellipsoid rule (continuous) can decide.
    """

    name = "integer_block_quadratic"
    # Each row's proof stands alone, so a prefix of the rows is a valid scan.
    row_scan_is_anytime = True

    def _row_blocks(self, model: Model, constraint, metadata: FlatVariableMetadata):
        """Bound-independent half: ``(support, d, blocks)`` with ``blocks`` a list of
        ``(positions, Q_b, c_b, kind)``, or ``None`` when the rule cannot use the row.

        Built from the iterative flattener (``_cached_flat_terms``, #1198) and the
        rule-family matchers, not ``extract_quadratic_support``: that expansion
        recurses, and a 20k-term row raised ``RecursionError`` out of the whole
        pass. Each term must be a constant, ``c*x``, ``c*x**2``/``c*x*x`` or a
        product of two scaled variables; anything else abstains the row. ``Q`` stays
        sparse until the blocks are known, so a long separable row costs one scan.
        """
        terms = _cached_flat_terms(model, constraint.body)
        d = 0.0
        lin: dict[int, float] = {}
        quad: dict[tuple[int, int], float] = {}
        has_cross = False
        for scale, term in terms:
            const_val = _constant_value(term)
            if const_val is not None:
                d += scale * const_val
                continue
            sq = _match_scaled_square_var(term, scale, metadata)
            if sq is not None:
                quad[(sq[0], sq[0])] = quad.get((sq[0], sq[0]), 0.0) + sq[1]
                continue
            li = _match_scaled_linear_var(term, scale, metadata)
            if li is not None:
                lin[li[0]] = lin.get(li[0], 0.0) + li[1]
                continue
            bi = _match_scaled_bilinear(term, scale, metadata)
            if bi is None:
                return None
            i, j, coeff = bi
            key = (min(i, j), max(i, j))
            quad[key] = quad.get(key, 0.0) + coeff
            has_cross = has_cross or i != j
        if not has_cross:
            return None  # separable: SeparableQuadraticUpperBoundRule's row
        if not np.isfinite(d) or not all(np.isfinite(v) for v in (*lin.values(), *quad.values())):
            return None
        support = tuple(sorted({k for pair in quad for k in pair} | set(lin)))
        position = {j: p for p, j in enumerate(support)}
        adj: dict[int, set[int]] = {p: set() for p in range(len(support))}
        for (i, j), v in quad.items():
            if i != j and v != 0.0:
                adj[position[i]].add(position[j])
                adj[position[j]].add(position[i])
        is_int = [
            metadata.flat_var_types[int(j)] in (VarType.INTEGER, VarType.BINARY) for j in support
        ]
        seen = [False] * len(support)
        blocks = []
        has_int_block = False
        for start in range(len(support)):
            if seen[start]:
                continue
            seen[start] = True
            stack, members = [start], [start]
            while stack:
                for nb in adj[stack.pop()]:
                    if not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
                        members.append(nb)
            pos = np.array(sorted(members), dtype=np.int64)
            if pos.size > _ELLIPSOID_MAX_SUPPORT:
                return None  # a dense block this large is not worth decomposing here
            cols = [support[p] for p in pos]
            Qb = np.zeros((pos.size, pos.size), dtype=np.float64)
            for a, ia in enumerate(cols):
                for b, ib in enumerate(cols):
                    if a == b:
                        Qb[a, a] = quad.get((ia, ia), 0.0)
                    elif a < b:
                        v = 0.5 * quad.get((min(ia, ib), max(ia, ib)), 0.0)
                        Qb[a, b] = Qb[b, a] = v
            cb = np.array([lin.get(j, 0.0) for j in cols], dtype=np.float64)
            if pos.size == 1:
                if Qb[0, 0] < -1e-12:
                    return None
                blocks.append((pos, Qb, cb, "int1" if is_int[pos[0]] else "cont1"))
                continue
            eig = np.linalg.eigvalsh(Qb)
            if eig[0] <= 0.0 or eig[-1] > _ELLIPSOID_MAX_COND * eig[0]:
                return None
            if all(is_int[p] for p in pos) and pos.size <= _INT_BLOCK_MAX_SIZE:
                blocks.append((pos, Qb, cb, "intk"))
                has_int_block = True
            else:
                blocks.append((pos, Qb, cb, "contk"))
        if not has_int_block:
            return None
        return support, float(d), blocks

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
        deadline: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not integer_block_quadratic_enabled():
            return flat_lb.copy(), flat_ub.copy()
        for constraint in _rows_until(model, deadline):
            # ``>=`` rows are ``-body <= 0`` with a negated Q, which is never PD when
            # Q is, so only ``<=`` and ``==`` (its ``<=`` half) can be decided here.
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue
            structure = _cached_row_structure(
                model,
                self.name,
                constraint,
                metadata,
                lambda c=constraint: self._row_blocks(model, c, metadata),
            )
            if structure is None:
                continue
            support, d, blocks = structure
            sup = np.asarray(support, dtype=np.int64)
            total = d
            absmag = abs(d)
            n_terms = 1
            for pos, Qb, cb, kind in blocks:
                cols = sup[pos]
                lb_b = np.asarray(flat_lb[cols], dtype=np.float64)
                ub_b = np.asarray(flat_ub[cols], dtype=np.float64)
                if kind in ("int1", "cont1"):
                    a, b = float(Qb[0, 0]), float(cb[0])
                    minimiser = (
                        _min_univariate_quadratic_integer
                        if kind == "int1"
                        else _min_univariate_quadratic
                    )
                    val = minimiser(a, b, float(lb_b[0]), float(ub_b[0]))
                    mag = abs(val)
                elif kind == "intk":
                    val, mag = _min_quadratic_block_integer(Qb, cb, lb_b, ub_b)
                else:
                    qinv = np.linalg.inv(Qb)
                    val = float(-0.25 * (cb @ qinv @ cb))
                    mag = abs(val) + float(np.abs(cb) @ np.abs(0.5 * (qinv @ cb)))
                if not np.isfinite(val):
                    total = -np.inf
                    break
                total += val
                absmag += mag
                n_terms += pos.size
            if not np.isfinite(total):
                continue
            # The same outward error bound the separable rule uses, over every
            # magnitude the sum was built from; the PD/conditioning gate above keeps
            # the per-block solves' own error far inside it.
            float_err = (n_terms + 4.0) * 1e3 * _FLOAT_EPS * (1.0 + absmag)
            if total > _EMPTY_INTERVAL_FEAS_TOL + float_err:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "minimum quadratic activity over the integer points of its blocks "
                    "exceeds the upper bound",
                )
        return flat_lb.copy(), flat_ub.copy()


_EXPR_TYPES = (
    Variable,
    Constant,
    Parameter,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    CustomCall,
    IndexExpression,
    MatMulExpression,
    SumExpression,
    SumOverExpression,
)


DEFAULT_NONLINEAR_BOUND_RULES: tuple[NonlinearBoundTighteningRule, ...] = (
    DefinedVariableForwardRule(),
    MonotoneFunctionEqualityRule(),
    QuadraticEqualityBoundsRule(),
    SquareDifferenceLowerBoundRule(),
    MonotoneFunctionBoundsRule(),
    PositiveAffineReciprocalBoundsRule(),
    NegativePowerBoundsRule(),
    ReciprocalBoundsRule(),
    BilinearProductEqualityRule(),
    SqrtSumOfSquaresUpperBoundRule(),
    SumOfSquaresUpperBoundRule(),
    SeparableQuadraticUpperBoundRule(),
    ConvexQuadraticEllipsoidRule(),
    IntegerBlockQuadraticRule(),
    PeriodicVariableBoundRule(),
)


def _snap_tolerant_crossovers(lb: np.ndarray, ub: np.ndarray) -> None:
    """Collapse sub-tolerance ``lb > ub`` crossovers to a degenerate box in place.

    Where ``ub < lb <= ub + _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack_arr(lb, ub)``
    the lower bound is pulled
    down to the (exact, more constraining) upper bound, yielding ``lb == ub``
    rather than an empty box. This keeps an approximate reformulation's eps-scale
    residual from manufacturing a hard infeasibility while leaving the genuine
    bound intact for the node NLP to validate.
    """
    crossed = (lb > ub) & (lb <= ub + _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack_arr(lb, ub))
    if np.any(crossed):
        lb[crossed] = ub[crossed]


def tighten_nonlinear_bounds(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    rules: Sequence[NonlinearBoundTighteningRule] = DEFAULT_NONLINEAR_BOUND_RULES,
    max_rounds: int = 5,
    deadline: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, NonlinearBoundTighteningStats]:
    """Run registered nonlinear tightening rules on a variable box.

    ``deadline`` is an absolute ``time.perf_counter()`` value bounding the whole
    pass, or ``None`` (the default) for the unbounded legacy behaviour, which is
    byte-identical to before — the polls are the only difference, and none of them
    can fire.

    Why it exists (issue #875). This is up to ``max_rounds`` sweeps over every rule,
    and every rule walks every constraint; nothing downstream bounded it, because it
    runs *before* the first branch-and-bound node exists. Measured on
    ``watercontamination0202`` (106,711 vars / 107,209 rows), the three root-setup
    calls cost **80.9 s against a 30 s ``time_limit``**. Same class as the LP-engine
    loops of #858 and the Rust presolve passes of #868, one layer up each time.

    The budget is honoured at three granularities, coarsest first: between rounds,
    between rules, and — for a rule that declares ``row_scan_is_anytime`` — inside
    its constraint scan (:func:`_rows_until`). The innermost one is what makes the
    poll real rather than nominal: a single rule's sweep over 107k rows already
    exceeds a tight budget, so a between-rules-only poll would be #868's ``probing``
    all over again.

    Soundness. Every early exit does strictly *less* work: rules only intersect the
    incoming box, so a dropped round, a skipped rule, or an unscanned row leaves the
    box **looser**, never wrong. A rule whose conclusion depends on a row being
    *absent* is skipped whole rather than truncated (see
    :attr:`NonlinearBoundTighteningRule.row_scan_is_anytime`). The one thing a
    budget can cost is an infeasibility *proof* the full pass would have found —
    ``infeasible`` stays unset and branch-and-bound discovers it instead, which is
    weaker, never false.
    """
    tightened_lb = np.asarray(flat_lb, dtype=np.float64).copy()
    tightened_ub = np.asarray(flat_ub, dtype=np.float64).copy()
    initial_lb = tightened_lb.copy()
    initial_ub = tightened_ub.copy()
    metadata = _cached_flat_metadata(model)

    applied_rules: list[str] = []

    def _mark_rule(rule_name: str) -> None:
        if rule_name not in applied_rules:
            applied_rules.append(rule_name)

    def _count_changed(new: np.ndarray, old: np.ndarray) -> int:
        finite = np.isfinite(new) & np.isfinite(old)
        changed = np.zeros(new.shape, dtype=bool)
        changed[finite] = np.abs(new[finite] - old[finite]) > 1e-12
        changed[~finite] = new[~finite] != old[~finite]
        return int(np.count_nonzero(changed))

    def _count_tightened(lb: np.ndarray, ub: np.ndarray) -> int:
        return int(_count_changed(lb, initial_lb) + _count_changed(ub, initial_ub))

    empty_initial = np.flatnonzero(
        tightened_lb
        > tightened_ub + _EMPTY_INTERVAL_FEAS_TOL + _roundoff_slack_arr(tightened_lb, tightened_ub)
    )
    if empty_initial.size > 0:
        first_idx = int(empty_initial[0])
        return (
            tightened_lb,
            tightened_ub,
            NonlinearBoundTighteningStats(
                n_tightened=0,
                applied_rules=(),
                infeasible=True,
                infeasibility_reason=f"initial interval is empty for flat variable {first_idx}",
            ),
        )
    # An integer column whose box holds no integer. Rules round only the bounds they
    # tighten, so with no row to tighten (``x`` integer in [1.2, 1.8], no constraints)
    # nothing saw it, and the Rust MILP route raised on the clipped non-integer point.
    # The 1e-5 slack is the integrality tolerance, so no tolerance-feasible point is cut.
    is_int = np.fromiter(
        (t in (VarType.INTEGER, VarType.BINARY) for t in metadata.flat_var_types),
        dtype=bool,
        count=len(metadata.flat_var_types),
    )
    if is_int.shape == tightened_lb.shape and is_int.any():
        with np.errstate(invalid="ignore"):
            no_integer = is_int & (
                np.ceil(tightened_lb - _INTEGER_BOX_TOL) > np.floor(tightened_ub + _INTEGER_BOX_TOL)
            )
        if no_integer.any():
            first_idx = int(np.flatnonzero(no_integer)[0])
            return (
                tightened_lb,
                tightened_ub,
                NonlinearBoundTighteningStats(
                    n_tightened=0,
                    applied_rules=(),
                    infeasible=True,
                    infeasibility_reason=(
                        f"integer flat variable {first_idx} has no integer in "
                        f"[{tightened_lb[first_idx]!r}, {tightened_ub[first_idx]!r}]"
                    ),
                ),
            )
    # Snap sub-tolerance crossovers (lb slightly above ub) to a degenerate box
    # so they are explored and validated by the node NLP rather than pruned.
    _snap_tolerant_crossovers(tightened_lb, tightened_ub)

    deadline_reached = False

    def _out_of_budget() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    for _ in range(max(1, int(max_rounds))):
        if _out_of_budget():
            deadline_reached = True
            break
        round_changed = False
        for rule in rules:
            # Between-rules poll. A rule that is not ``row_scan_is_anytime`` cannot
            # be cut mid-scan without risking an unsound conclusion, so this is the
            # only place it can be declined — never started rather than truncated,
            # which bounds the overrun to one in-flight rule.
            if _out_of_budget():
                deadline_reached = True
                break
            prev_lb = tightened_lb.copy()
            prev_ub = tightened_ub.copy()
            rule_kwargs = {}
            if (
                deadline is not None
                and getattr(rule, "row_scan_is_anytime", False)
                and _rule_accepts_deadline(rule)
            ):
                rule_kwargs["deadline"] = deadline
            try:
                cand_lb, cand_ub = rule.tighten(model, prev_lb, prev_ub, metadata, **rule_kwargs)
            except NonlinearBoundTighteningInfeasible as exc:
                _mark_rule(rule.name)
                return (
                    prev_lb,
                    prev_ub,
                    NonlinearBoundTighteningStats(
                        n_tightened=_count_tightened(prev_lb, prev_ub),
                        applied_rules=tuple(applied_rules),
                        infeasible=True,
                        infeasibility_reason=str(exc),
                        deadline_reached=deadline_reached,
                    ),
                )

            cand_lb_arr = np.asarray(cand_lb, dtype=np.float64)
            cand_ub_arr = np.asarray(cand_ub, dtype=np.float64)
            tightened_lb = np.maximum(prev_lb, cand_lb_arr)
            tightened_ub = np.minimum(prev_ub, cand_ub_arr)
            empty_indices = np.flatnonzero(
                tightened_lb
                > tightened_ub
                + _EMPTY_INTERVAL_FEAS_TOL
                + _roundoff_slack_arr(tightened_lb, tightened_ub)
            )
            if empty_indices.size > 0:
                _mark_rule(rule.name)
                first_idx = int(empty_indices[0])
                return (
                    prev_lb,
                    prev_ub,
                    NonlinearBoundTighteningStats(
                        n_tightened=_count_tightened(prev_lb, prev_ub),
                        applied_rules=tuple(applied_rules),
                        infeasible=True,
                        infeasibility_reason=(
                            f"{rule.name} returned an empty interval for flat variable {first_idx}"
                        ),
                        deadline_reached=deadline_reached,
                    ),
                )
            # Snap sub-tolerance crossovers introduced by intersecting this rule's
            # box with the prior box (e.g. an eps-scale perspective residual vs a
            # bound-linking zero) to a degenerate box rather than pruning.
            _snap_tolerant_crossovers(tightened_lb, tightened_ub)

            n_changed = _count_changed(tightened_lb, prev_lb) + _count_changed(
                tightened_ub, prev_ub
            )
            if n_changed > 0:
                _mark_rule(rule.name)
                round_changed = True
        if deadline_reached or not round_changed:
            break

    return (
        tightened_lb,
        tightened_ub,
        NonlinearBoundTighteningStats(
            n_tightened=_count_tightened(tightened_lb, tightened_ub),
            applied_rules=tuple(applied_rules),
            deadline_reached=deadline_reached,
        ),
    )
