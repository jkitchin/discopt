"""
Warm-start utilities for discopt.

Validates and flattens user-provided initial solutions so they can be
injected into the NLP solver (as starting point) and the B&B tree
(as initial incumbent / upper bound).
"""

from __future__ import annotations

import logging
import warnings
from typing import Union

import numpy as np

from discopt.modeling.core import Model, Variable, VarType

logger = logging.getLogger(__name__)


def validate_initial_solution(
    model: Model,
    initial_solution: dict[Variable, Union[float, np.ndarray, list]],
    *,
    tol_bounds: float = 1e-6,
    tol_integrality: float = 1e-5,
) -> np.ndarray:
    """Validate an initial solution and return it as a flat numpy vector.

    Parameters
    ----------
    model : Model
        The optimization model whose variables the solution refers to.
    initial_solution : dict
        Mapping from Variable objects to their values (scalars, lists, or
        numpy arrays).
    tol_bounds : float
        Tolerance for variable-bound violations.  Values outside
        ``[lb - tol, ub + tol]`` trigger a warning and are clamped.
    tol_integrality : float
        Tolerance for integrality violations on INTEGER / BINARY variables.
        Non-integer values trigger a warning and are rounded.

    Returns
    -------
    np.ndarray
        Flat solution vector (length = total number of scalar variables
        in the model) suitable for passing to the NLP evaluator.

    Raises
    ------
    TypeError
        If *initial_solution* is not a dict or contains non-Variable keys.
    ValueError
        If a Variable does not belong to the model, or a value has the
        wrong shape or non-finite entries.
    """
    if not isinstance(initial_solution, dict):
        raise TypeError(
            "initial_solution must be a dict mapping Variable objects to values, "
            f"got {type(initial_solution).__name__}"
        )

    # Build a set of model variables for membership checks
    model_vars = {id(v): v for v in model._variables}

    # Validate keys
    for key in initial_solution:
        if not isinstance(key, Variable):
            raise TypeError(
                "initial_solution keys must be Variable objects, "
                f"got {type(key).__name__} for key {key!r}"
            )
        if id(key) not in model_vars:
            raise ValueError(
                f"Variable '{key.name}' is not part of this model. "
                "Use the Variable objects returned by m.continuous(), "
                "m.binary(), or m.integer()."
            )

    # Build flat vector: start from midpoint of bounds, then overlay
    # provided values.
    n_vars = sum(v.size for v in model._variables)
    x_flat = np.zeros(n_vars, dtype=np.float64)
    offset = 0
    provided_mask = np.zeros(n_vars, dtype=bool)

    for v in model._variables:
        size = v.size
        if v in initial_solution:
            val = np.asarray(initial_solution[v], dtype=np.float64)

            # Shape validation
            expected_shape = v.shape
            if expected_shape == ():
                # Scalar variable: accept scalar or (1,) array
                if val.shape not in ((), (1,)):
                    raise ValueError(
                        f"Variable '{v.name}' is scalar but got value with shape {val.shape}"
                    )
                val = val.flatten()
            else:
                if val.shape != expected_shape:
                    # Try to reshape from flat
                    try:
                        val = val.reshape(expected_shape)
                    except ValueError:
                        raise ValueError(
                            f"Variable '{v.name}' has shape {expected_shape} "
                            f"but got value with shape {val.shape}"
                        )
                val = val.flatten()

            # Bounds checking and clamping
            lb_flat = v.lb.flatten()
            ub_flat = v.ub.flatten()

            if not np.all(np.isfinite(val)):
                raise ValueError(
                    f"Variable '{v.name}' initial_solution contains non-finite value(s)"
                )

            below = val < lb_flat - tol_bounds
            above = val > ub_flat + tol_bounds
            if np.any(below) or np.any(above):
                n_viol = int(np.sum(below) + np.sum(above))
                warnings.warn(
                    f"Variable '{v.name}': {n_viol} value(s) outside bounds. Clamping to [lb, ub].",
                    stacklevel=3,
                )
                val = np.clip(val, lb_flat, ub_flat)

            # Integrality checking and rounding
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                frac = np.abs(val - np.round(val))
                if np.any(frac > tol_integrality):
                    n_frac = int(np.sum(frac > tol_integrality))
                    warnings.warn(
                        f"Variable '{v.name}': {n_frac} value(s) are not "
                        "integer-valued. Rounding to nearest integer.",
                        stacklevel=3,
                    )
                    val = np.round(val)
                    val = np.clip(val, lb_flat, ub_flat)

            x_flat[offset : offset + size] = val
            provided_mask[offset : offset + size] = True
        else:
            # Default: midpoint of bounds (clipped for unbounded vars)
            lb_flat = v.lb.flatten()
            ub_flat = v.ub.flatten()
            lb_clip = np.clip(lb_flat, -1e4, 1e4)
            ub_clip = np.clip(ub_flat, -1e4, 1e4)
            x_flat[offset : offset + size] = 0.5 * (lb_clip + ub_clip)

        offset += size

    n_provided = int(np.sum(provided_mask))
    n_total_vars = len(model._variables)
    if n_provided > 0 and n_provided < n_vars:
        logger.info(
            "Warm start: %d of %d variables provided, using midpoint for the rest",
            len(initial_solution),
            n_total_vars,
        )

    return x_flat


def unflatten_solution(
    model: Model,
    x_flat: np.ndarray,
) -> dict[Variable, np.ndarray]:
    """Split a flat solution vector back into per-Variable arrays.

    Inverse of :func:`validate_initial_solution`'s flattening: it uses the same
    ordering contract — variables in ``model._variables`` declaration order, each
    occupying ``v.size`` contiguous entries — so
    ``unflatten_solution(m, validate_initial_solution(m, d))`` reproduces ``d``
    (up to the bound clamping / integrality rounding that validation applies).

    Parameters
    ----------
    model : Model
        The model whose variables define the flat layout.
    x_flat : np.ndarray
        Flat solution vector of length ``sum(v.size for v in model._variables)``,
        e.g. an ``NLPResult.x`` from :func:`discopt.solvers.nlp_pounce.solve_nlp`.

    Returns
    -------
    dict[Variable, np.ndarray]
        Maps each model Variable to its value, reshaped to ``v.shape``. Scalar
        variables (``shape == ()``) map to a 0-d array.
    """
    x_flat = np.asarray(x_flat, dtype=np.float64).ravel()
    n_vars = sum(v.size for v in model._variables)
    if x_flat.shape != (n_vars,):
        raise ValueError(
            f"x_flat has length {x_flat.shape[0]}, expected {n_vars} "
            "(sum of variable sizes in this model)"
        )

    out: dict[Variable, np.ndarray] = {}
    offset = 0
    for v in model._variables:
        size = v.size
        chunk = x_flat[offset : offset + size]
        out[v] = chunk.reshape(v.shape) if v.shape != () else chunk.reshape(())
        offset += size
    return out


def check_feasibility(
    model: Model,
    x_flat: np.ndarray,
    *,
    tol: float = 1e-4,
) -> tuple[bool, list[str]]:
    """Check whether a flat solution vector is feasible for the model.

    Returns a tuple ``(is_feasible, violations)`` where *violations* is a
    list of human-readable strings describing any constraint or bound
    violations found.
    """
    violations: list[str] = []

    # Variable bounds
    offset = 0
    for v in model._variables:
        size = v.size
        vals = x_flat[offset : offset + size]
        lb_flat = v.lb.flatten()
        ub_flat = v.ub.flatten()

        below = vals < lb_flat - tol
        above = vals > ub_flat + tol
        if np.any(below):
            violations.append(
                f"Variable '{v.name}': {int(np.sum(below))} value(s) below lower bound"
            )
        if np.any(above):
            violations.append(
                f"Variable '{v.name}': {int(np.sum(above))} value(s) above upper bound"
            )
        offset += size

    # Constraint feasibility (requires evaluator)
    try:
        from discopt._tape_nlp_evaluator import make_evaluator
        from discopt.modeling.core import Constraint

        evaluator = make_evaluator(model)  # #1063: canonical funnel, not the JAX ctor
        if evaluator.n_constraints > 0:
            cons = evaluator.evaluate_constraints(x_flat)
            idx = 0
            for c in model._constraints:
                if not isinstance(c, Constraint):
                    continue
                val = cons[idx]
                if c.sense == "<=":
                    if val > tol:
                        name = c.name or f"constraint_{idx}"
                        violations.append(f"Constraint '{name}': value {val:.6g} > 0 (sense <=)")
                elif c.sense == "==":
                    if abs(val) > tol:
                        name = c.name or f"constraint_{idx}"
                        violations.append(
                            f"Constraint '{name}': |value| = {abs(val):.6g} != 0 (sense ==)"
                        )
                elif c.sense == ">=":
                    if val < -tol:
                        name = c.name or f"constraint_{idx}"
                        violations.append(f"Constraint '{name}': value {val:.6g} < 0 (sense >=)")
                idx += 1
    except Exception as e:
        logger.debug("Feasibility check skipped (evaluator error): %s", e)

    return len(violations) == 0, violations


#: Most reform-added discrete columns a completion will search over. The repair
#: costs ``O(new_binaries)`` constraint evaluations per sweep; beyond this the
#: padded point is returned unrepaired and the caller's own feasibility gate
#: decides its fate.
_MAX_REPAIRED_DISCRETE = 64
#: Most moves the repair will take before giving up.
_MAX_REPAIR_MOVES = 32
#: Hard cap on constraint evaluations spent completing one warm start.
_MAX_REPAIR_EVALS = 512


def _flat_size(model: Model) -> int:
    return int(sum(int(v.size) for v in model._variables))


def _tail_default(v: Variable) -> np.ndarray:
    """The value :func:`validate_initial_solution` uses for a variable nobody
    supplied — midpoint of the (clipped) bounds, snapped to the nearest feasible
    integer for a discrete variable."""
    lb = np.asarray(v.lb, dtype=np.float64).flatten()
    ub = np.asarray(v.ub, dtype=np.float64).flatten()
    mid = 0.5 * (np.clip(lb, -1e4, 1e4) + np.clip(ub, -1e4, 1e4))
    if v.var_type in (VarType.BINARY, VarType.INTEGER):
        mid = np.clip(np.round(mid), np.ceil(lb - 1e-9), np.floor(ub + 1e-9))
    return mid


def complete_initial_point(model: Model, x_head, *, evaluator=None):
    """Extend a warm start over a PREFIX of ``model``'s variables to the whole vector.

    Solve-time reformulations *append* variables: the GDP pass lowers a
    disjunction into selector binaries and big-M rows, the factorable lift adds
    monomial auxiliaries, and so on. The user's ``initial_solution`` was flattened
    against the variables they declared, so after such a pass it is SHORTER than
    the working model's vector — and the warm-start site fed it to the evaluator
    anyway, which raised ``ValueError: objective: x: expected length 5, got 3``
    out of a solve that works fine without a warm start (#1255).

    The completion fills the appended tail with the same defaults
    :func:`validate_initial_solution` uses for a variable nobody supplied, then
    repairs the appended DISCRETE columns by a bounded best-neighbour search on
    total constraint violation. That is what recovers a GDP selector: at the
    given point one
    disjunct holds, its big-M rows are satisfied only with its own selector at 1,
    and every other assignment leaves either the ``sum(y) == 1`` row or an
    activated disjunct row violated — so the search lands on the indicator the
    point implies, with no per-pass metadata and no knowledge of which pass added
    the column.

    Purely primal, and never trusted: the caller re-checks the completed point
    against the model's own feasibility gate before it may seed anything, so a
    completion that lands anywhere wrong is dropped rather than believed.

    Returns ``None`` when the head does not line up with a prefix of the model's
    variables, or is not finite — the caller then simply does not warm-start.
    """
    n = _flat_size(model)
    x_head = np.asarray(x_head, dtype=np.float64).ravel()
    if x_head.size == n:
        return x_head.astype(np.float64, copy=True)
    if x_head.size > n or not np.all(np.isfinite(x_head)):
        return None

    # The head must cover a whole prefix of the variable list: reform passes
    # append, they do not interleave, so a head that ends mid-variable means the
    # point does not describe this model's columns at all.
    tail_vars: list[tuple[int, Variable]] = []
    off = 0
    head_ok = x_head.size == 0
    for v in model._variables:
        if off == x_head.size:
            head_ok = True
        if off >= x_head.size:
            tail_vars.append((off, v))
        off += int(v.size)
    if not head_ok:
        return None

    x = np.empty(n, dtype=np.float64)
    x[: x_head.size] = x_head
    discrete_cols: list[int] = []
    for start, v in tail_vars:
        size = int(v.size)
        x[start : start + size] = _tail_default(v)
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            discrete_cols.extend(range(start, start + size))

    if not discrete_cols:
        return x
    if len(discrete_cols) > _MAX_REPAIRED_DISCRETE:
        logger.debug(
            "warm start: %d reform-added discrete columns exceeds the repair budget; "
            "returning the padded point unrepaired",
            len(discrete_cols),
        )
        return x

    try:
        repaired = _repair_discrete(model, x, discrete_cols, evaluator)
    except Exception as exc:  # noqa: BLE001 - reported; the padded point still stands
        logger.debug("warm start: discrete repair unavailable (%s)", exc)
        return x
    return repaired


def _repair_discrete(model: Model, x: np.ndarray, cols: list[int], evaluator):
    """Best-neighbour search on total constraint violation over *cols*.

    Plain 1-flip descent is not enough, and the GDP case is exactly why: the
    selector rows are ``sum(y) == 1``, so moving from the wrong disjunct to the
    right one is a SWAP — two flips, with the first one uphill. Measured on
    #1255's model, descent stopped at ``y = (1, 0)`` (total violation 0.9, the
    activated wrong disjunct) because every single flip from there is worse.

    So the search takes the best neighbour even when it is uphill, forbids
    immediately undoing the column it just moved (a tabu of one), and remembers
    the best point it has seen. The swap then takes two moves and lands on total
    violation 0. Bounded by an evaluation budget, and purely primal either way —
    the caller re-verifies whatever comes back.
    """
    from discopt._relax.primal_heuristics import row_violations

    if evaluator is None:
        from discopt._tape_nlp_evaluator import make_evaluator

        evaluator = make_evaluator(model)

    budget = [_MAX_REPAIR_EVALS]

    def total_violation(xx: np.ndarray) -> float:
        budget[0] -= 1
        v = row_violations(evaluator, xx)
        return float(np.sum(v)) if v.size else 0.0

    lb: list[float] = []
    ub: list[float] = []
    for v in model._variables:
        lb.extend(np.asarray(v.lb, dtype=np.float64).flatten().tolist())
        ub.extend(np.asarray(v.ub, dtype=np.float64).flatten().tolist())
    lo_arr = np.asarray(lb, dtype=np.float64)
    hi_arr = np.asarray(ub, dtype=np.float64)

    # Candidate values per column: the integers in its box. A wide integer range
    # is not what this repair is for — a selector is binary.
    candidates: dict[int, list[float]] = {}
    for j in cols:
        lo = int(np.ceil(lo_arr[j] - 1e-9))
        hi = int(np.floor(hi_arr[j] + 1e-9))
        if hi < lo or hi - lo > 4:
            continue
        candidates[j] = [float(c) for c in range(lo, hi + 1)]
    if not candidates:
        return x

    cur = x.copy()
    cur_viol = total_violation(cur)
    best_x, best_viol = cur.copy(), cur_viol
    tabu = -1
    moves = 0
    while best_viol > 0.0 and moves < _MAX_REPAIR_MOVES and budget[0] > 0:
        move = None
        move_viol = np.inf
        for j, vals in candidates.items():
            if j == tabu:
                continue
            keep = cur[j]
            for cand in vals:
                if cand == keep or budget[0] <= 0:
                    continue
                cur[j] = cand
                trial = total_violation(cur)
                if trial < move_viol:
                    move_viol, move = trial, (j, cand)
            cur[j] = keep
        if move is None:
            break
        j, cand = move
        cur[j] = cand
        cur_viol = move_viol
        tabu = j
        moves += 1
        if cur_viol < best_viol:
            best_x, best_viol = cur.copy(), cur_viol
    return best_x


# --------------------------------------------------------------------------- #
# Primal-dual warm start (#1247)
# --------------------------------------------------------------------------- #
def primal_point_from_result(model: Model, result) -> np.ndarray:
    """Flatten a :class:`~discopt.modeling.core.SolveResult`'s ``x`` for ``model``.

    Uses the same ordering contract as :func:`validate_initial_solution` and
    :func:`unflatten_solution` — variables in ``model._variables`` declaration
    order, each occupying ``v.size`` contiguous entries — but keys off
    ``result.x``, which is a dict of *variable names*.

    Raises ``ValueError`` when the result does not describe this model (a missing
    name or a wrong shape). It refuses rather than filling a default: a warm
    start silently completed from a different model is a warm start that quietly
    does nothing, which is the failure this function exists to make impossible.
    """
    if getattr(result, "x", None) is None:
        raise ValueError(
            "warm start: the previous result carries no solution vector "
            f"(status={getattr(result, 'status', '?')!r}), so there is no point to start from"
        )
    x_by_name = result.x
    chunks: list[np.ndarray] = []
    for v in model._variables:
        if v.name not in x_by_name:
            raise ValueError(
                f"warm start: the previous result has no value for variable {v.name!r}. "
                "It must come from a solve of a model with the same variables."
            )
        arr = np.asarray(x_by_name[v.name], dtype=np.float64).ravel()
        if arr.size != v.size:
            raise ValueError(
                f"warm start: variable {v.name!r} has {v.size} scalar entries in this model "
                f"but {arr.size} in the previous result."
            )
        chunks.append(arr)
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks)


def bound_duals_from_result(model: Model, duals: "dict[str, np.ndarray] | None"):
    """Flatten ``SolveResult.bound_duals_lower`` / ``_upper`` in variable order.

    Returns ``None`` when ``duals`` is ``None`` (the previous solve reported no
    bound multipliers) so the caller can start without them — POUNCE fills the
    unseeded multipliers from the supplied point. A dict that is present but does
    not cover the model raises, for the reason in
    :func:`primal_point_from_result`.
    """
    if duals is None:
        return None
    chunks: list[np.ndarray] = []
    for v in model._variables:
        if v.name not in duals:
            raise ValueError(
                f"warm start: the previous result's bound multipliers have no entry for "
                f"variable {v.name!r}."
            )
        arr = np.asarray(duals[v.name], dtype=np.float64).ravel()
        if arr.size != v.size:
            raise ValueError(
                f"warm start: variable {v.name!r} has {v.size} scalar entries in this model "
                f"but {arr.size} bound multipliers in the previous result."
            )
        chunks.append(arr)
    if not chunks:
        return None
    return np.concatenate(chunks)


def constraint_duals_from_result(evaluator, duals: "dict[str, np.ndarray] | None"):
    """Flatten ``SolveResult.constraint_duals`` into the evaluator's row order.

    The exact inverse of ``solver._unpack_constraint_duals``: it walks the
    evaluator's ``_source_constraints`` / ``_constraint_flat_sizes`` — the layout
    source of truth — and keys by ``Constraint.name`` (or ``c{idx}`` when
    anonymous), so a vector body's multipliers land back on their own rows.

    Returns ``None`` when ``duals`` is ``None``.
    """
    if duals is None:
        return None
    chunks: list[np.ndarray] = []
    for idx, (c, size) in enumerate(
        zip(evaluator._source_constraints, evaluator._constraint_flat_sizes)
    ):
        size = int(size)
        key = c.name if c.name else f"c{idx}"
        if key not in duals:
            raise ValueError(
                f"warm start: the previous result's constraint multipliers have no entry for "
                f"constraint {key!r}."
            )
        arr = np.asarray(duals[key], dtype=np.float64).ravel()
        if arr.size != size:
            raise ValueError(
                f"warm start: constraint {key!r} has {size} rows in this model but "
                f"{arr.size} multipliers in the previous result."
            )
        chunks.append(arr)
    if not chunks:
        return None
    return np.concatenate(chunks)
