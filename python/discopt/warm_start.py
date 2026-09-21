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

    **This verifier fails closed (#1402).** It is used as the *independent*
    feasibility check behind a reported incumbent, so "I could not evaluate the
    point" must never be reported as "the point is feasible". Every way of not
    reaching a verdict — a wrong-length vector, a non-finite entry, an evaluator
    that raises, a constraint that evaluates to NaN — yields ``False`` with a
    violation naming the reason. Two mechanisms previously returned
    ``(True, [])`` for a point that was never checked:

    * the constraint arm's ``except Exception`` left ``violations`` empty, so an
      evaluator error read as a clean bill of health (measured: a length-1 vector
      for a 2-variable model, whose evaluator raises ``ValueError``, was reported
      feasible for a model infeasible everywhere in its box);
    * every violation test is a *strict* comparison, and every strict comparison
      against NaN is ``False``, so an all-NaN point passed both the bounds loop
      and all three sense branches with no exception raised at all.

    A third mechanism of the same class reached the same verdict without either
    an exception or a NaN: the rows were **mis-attributed**. The walk advanced one
    index per :class:`~discopt.modeling.core.Constraint` *object* while the
    evaluator emits one row per flat *element*, so an array-valued body left its
    rows after the first unread and desynchronised every constraint behind it, and
    ``model._builder_linear_constraints()`` rows (#840) were never examined at all.
    Rows are now enumerated from
    :meth:`~discopt._relax.nlp_evaluator.NLPEvaluator.constraint_row_map` — the
    evaluator's own map, built from the same ``_source_constraints`` /
    ``_constraint_flat_sizes`` as the row stream — which makes that class
    structurally impossible rather than merely fixed. This is the same fix #908
    applied to the two *in-solver* incumbent verifiers, which
    :mod:`discopt.validation.feasibility` now owns; this function is the verifier
    the *benchmark* correctness gate uses and was not migrated with them.
    """
    violations: list[str] = []

    # #1402 mechanism A, first half: a length mismatch is why the evaluator
    # raises, so name it here rather than letting it surface as an opaque error.
    x_flat = np.asarray(x_flat, dtype=float).ravel()
    n_expected = sum(int(v.size) for v in model._variables)
    if x_flat.size != n_expected:
        violations.append(
            f"solution vector has {x_flat.size} entries but the model has "
            f"{n_expected} variable entries: the point cannot be checked"
        )
        return False, violations

    # #1402 mechanism B: NaN/inf fail every strict comparison below, so gate on
    # finiteness explicitly instead of letting the comparisons pass them through.
    if not np.all(np.isfinite(x_flat)):
        n_bad = int(np.sum(~np.isfinite(x_flat)))
        violations.append(
            f"solution vector has {n_bad} non-finite entr{'y' if n_bad == 1 else 'ies'} "
            "(NaN or inf): the point cannot be checked"
        )
        return False, violations

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

        evaluator = make_evaluator(model)  # #1063: canonical funnel, not the JAX ctor
        n_rows = int(evaluator.n_constraints)
        if n_rows > 0:
            cons = np.asarray(evaluator.evaluate_constraints(x_flat), dtype=float).ravel()
            if cons.size != n_rows:
                # Fail closed rather than index into a short row stream: an
                # IndexError here would be a refusal too, but a SHORT-by-design
                # stream would silently leave the tail rows unchecked.
                violations.append(
                    f"evaluator returned {cons.size} constraint values for "
                    f"{n_rows} rows; feasibility is NOT verified"
                )
            else:
                # #1402 mechanism C: enumerate rows from the evaluator's OWN map,
                # never by walking `model._constraints` with one index per object.
                # See `constraint_row_map`'s docstring and
                # `discopt/validation/feasibility.py` (#908): the per-object walk
                # this replaces was wrong in the wrongly-ACCEPT direction twice
                # over, and neither way raises or produces a NaN, so #1402's other
                # two mechanisms leave it untouched:
                #
                #   * an array-valued body is ONE `Constraint` and MANY rows
                #     (`x <= 1` on a 3-vector is one object, three rows), so rows
                #     1..k-1 of every vector constraint went unread and every
                #     constraint after the first vector one read the WRONG row;
                #   * the evaluator's row set is `model._constraints` PLUS
                #     `model._builder_linear_constraints()` (#840), so the
                #     builder-resident rows were never examined at all.
                #
                # Measured on this tree by `scripts/audit_1402_row_attribution.py`:
                # three separate points, each violating a row by 4.0, were all
                # reported FEASIBLE.
                checked = np.zeros(n_rows, dtype=bool)
                for start, stop, c in evaluator.constraint_row_map():
                    sense = c.sense if isinstance(c.sense, str) else c.sense.value
                    base = c.name or f"constraint_{start}"
                    for r in range(start, stop):
                        checked[r] = True
                        val = float(cons[r])
                        # A vector constraint's rows need distinguishable names;
                        # a scalar one keeps the bare name it always had.
                        name = base if stop - start == 1 else f"{base}[{r - start}]"
                        if not np.isfinite(val):
                            # #1402 mechanism B in the constraint arm: a NaN body
                            # fails `val > tol`, `abs(val) > tol` and `val < -tol`
                            # alike, so without this the row is silently treated
                            # as satisfied.
                            violations.append(
                                f"Constraint '{name}': body evaluated to {val!r}, so the row "
                                "cannot be checked"
                            )
                        elif sense == "<=":
                            if val > tol:
                                violations.append(
                                    f"Constraint '{name}': value {val:.6g} > 0 (sense <=)"
                                )
                        elif sense == "==":
                            if abs(val) > tol:
                                violations.append(
                                    f"Constraint '{name}': |value| = {abs(val):.6g} != 0 (sense ==)"
                                )
                        elif sense == ">=":
                            if val < -tol:
                                violations.append(
                                    f"Constraint '{name}': value {val:.6g} < 0 (sense >=)"
                                )
                        else:
                            # An unrecognised sense matched none of the three
                            # branches and fell out of the old walk silently,
                            # leaving the row unchecked with no trace.
                            violations.append(
                                f"Constraint '{name}': unrecognised sense {c.sense!r}, so the "
                                "row cannot be checked"
                            )
                n_unchecked = int(np.sum(~checked))
                if n_unchecked:
                    # The map is built from the same `_source_constraints` /
                    # `_constraint_flat_sizes` as the row stream, so this cannot
                    # fire today. It is the guard that keeps a future drift from
                    # degrading back into a silent partial check.
                    violations.append(
                        f"constraint_row_map covered {n_rows - n_unchecked} of {n_rows} "
                        f"evaluator rows; {n_unchecked} row(s) were never checked"
                    )
    except Exception as e:
        # #1402 mechanism A: this arm used to log at DEBUG and fall through to
        # `return len(violations) == 0`, so an evaluator error produced
        # `(True, [])` — "verified feasible" for a point whose constraints were
        # never evaluated. The log message said "skipped"; the return value said
        # "verified". A verifier that cannot verify must answer no (CLAUDE.md §3,
        # §7): the error becomes a violation rather than a silence.
        logger.debug("Feasibility check could not evaluate constraints: %s", e)
        violations.append(
            f"constraints could not be evaluated ({type(e).__name__}: {e}); "
            "feasibility is NOT verified"
        )

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


def prepare_warm_start(model: Model, x0, *, route: str, evaluator=None):
    """Make a warm start usable by *model*, or drop it. Never raises.

    #1324 set the contract that a warm start is a HINT: it may fail to help, it
    may be ignored, but it must never be able to fail a solve that succeeds
    without it. #1334 found two routes that still broke it, and they break it in
    two different ways, so this is the one place that answers both:

    * **Width.** Solve-time reformulations APPEND columns (the GDP pass lowers a
      disjunction into selector binaries; the factorable lift adds monomial
      auxiliaries), and the point was flattened against the variables the *user*
      declared. Three routes already completed it inline and three did not --
      ``solver="amp"`` raised ``AMP initial_point has length 1; expected 3`` and
      ``solver="mip-nlp"`` raised ``NLP initial point has shape (1,); expected
      (3,)`` on a GDP model that solves fine with no warm start at all.
    * **Finiteness.** A NaN reaches here from ``primal_point_from_result``
      whenever the previous solve left one in a column (``np.clip`` passes NaN
      through), and the NLP-BB injection block rounds before it checks anything:
      ``round(nan)`` raises ``ValueError: cannot convert float NaN to integer``
      out of the middle of the solve. Filter first, round later.

    Returns a finite ``float64`` vector of exactly ``model``'s column count, or
    ``None`` when the point cannot describe this model -- in which case the
    caller simply solves without it. Purely primal and never trusted either way:
    every injection site re-checks the point against the model's own feasibility
    gate before it may seed anything.

    Parameters
    ----------
    model : Model
        The model as the route will actually solve it, i.e. AFTER any
        reformulation pass that may have appended columns.
    x0 : array-like or None
        The candidate warm start. ``None`` passes straight through.
    route : str
        Human-readable route name for the log line, so a dropped warm start says
        which path dropped it (``"AMP"``, ``"MIP-NLP"``, ``"NLP-BB"``, ...).
    evaluator : optional
        Constraint evaluator, used to repair reform-added discrete columns.
    """
    if x0 is None:
        return None

    x = np.asarray(x0, dtype=np.float64).ravel()

    if not np.all(np.isfinite(x)):
        logger.warning(
            "%s warm start dropped: the initial point has %d non-finite value(s). "
            "The solve continues without it.",
            route,
            int(np.count_nonzero(~np.isfinite(x))),
        )
        return None

    n_cols = _flat_size(model)
    if x.size == n_cols:
        return x

    completed = complete_initial_point(model, x, evaluator=evaluator)
    if completed is None:
        logger.warning(
            "%s warm start dropped: the initial solution covers %d columns and the "
            "model the solver built has %d (a reformulation added variables). "
            "The solve continues without it.",
            route,
            int(x.size),
            n_cols,
        )
        return None

    logger.info(
        "%s warm start extended from %d to %d columns across a solve-time reformulation",
        route,
        int(x.size),
        n_cols,
    )
    return completed


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
def primal_point_from_result(
    model: Model,
    result,
    *,
    tol_bounds: float = 1e-6,
    clamp: bool = True,
) -> np.ndarray:
    """Flatten a :class:`~discopt.modeling.core.SolveResult`'s ``x`` for ``model``.

    Uses the same ordering contract as :func:`validate_initial_solution` and
    :func:`unflatten_solution` — variables in ``model._variables`` declaration
    order, each occupying ``v.size`` contiguous entries — but keys off
    ``result.x``, which is a dict of *variable names*.

    Raises ``ValueError`` when the result does not describe this model (a missing
    name or a wrong shape). It refuses rather than filling a default: a warm
    start silently completed from a different model is a warm start that quietly
    does nothing, which is the failure this function exists to make impossible.

    #1316: values outside the model's **current** bounds are clamped and warned
    about, exactly as :func:`validate_initial_solution` already did for
    ``initial_solution``. A warm start comes from an *earlier* solve, so a bound
    that moved in between (the per-condition bounds of a phase-diagram trace, say)
    leaves the previous point outside the box — and this vector is not only a
    starting point, it is offered to the B&B tree as an incumbent. An unclamped
    one was accepted as a certified optimum whose value is unreachable inside the
    real feasible region. ``clamp=False`` returns the raw point for a caller that
    means to inspect it rather than solve from it.

    #1334: NaN gets the same treatment, one step earlier. ``np.clip`` passes NaN
    through -- every comparison against it is False -- so a NaN column survived
    the #1316 clamp untouched and crashed the NLP-BB injection block on
    ``round()``. A NaN says the previous solve did not determine that variable,
    so it starts from the default a variable nobody supplied gets.
    """
    if getattr(result, "x", None) is None:
        raise ValueError(
            "warm start: the previous result carries no solution vector "
            f"(status={getattr(result, 'status', '?')!r}), so there is no point to start from"
        )
    x_by_name = result.x
    chunks: list[np.ndarray] = []
    n_clamped = 0
    clamped_names: list[str] = []
    n_nan = 0
    nan_names: list[str] = []
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
        if clamp:
            lb_flat = np.asarray(v.lb, dtype=np.float64).ravel()
            ub_flat = np.asarray(v.ub, dtype=np.float64).ravel()
            # #1334: NaN must be handled BEFORE the clamp, because ``np.clip``
            # passes it through -- every comparison against NaN is False, so it
            # survives both the out-of-bounds test and the clip and reaches the
            # injection sites, where NLP-BB's ``round(x[j])`` raised ``ValueError:
            # cannot convert float NaN to integer`` out of the middle of a solve.
            # (``±inf`` needs nothing extra: it IS outside the bounds, so the
            # clamp below already lands it on the bound.) A NaN column carries no
            # information about where to start, which is exactly the state of a
            # variable the caller never supplied, so it gets the same value
            # ``validate_initial_solution`` gives that variable -- and the point
            # is a hint either way, re-checked against the model's feasibility
            # gate before it may seed anything.
            nan_mask = np.isnan(arr)
            if np.any(nan_mask):
                n_nan += int(np.count_nonzero(nan_mask))
                nan_names.append(v.name)
                arr = np.where(nan_mask, _tail_default(v), arr)
            outside = int(
                np.count_nonzero((arr < lb_flat - tol_bounds) | (arr > ub_flat + tol_bounds))
            )
            if outside:
                n_clamped += outside
                clamped_names.append(v.name)
                arr = np.clip(arr, lb_flat, ub_flat)
        chunks.append(arr)
    if n_clamped:
        warnings.warn(
            f"warm start: {n_clamped} value(s) from the previous solve lie outside this "
            f"model's current bounds ({', '.join(clamped_names)}); clamping to [lb, ub]. "
            "A bound changed between the two solves, so the previous point is no longer "
            "feasible here.",
            stacklevel=3,
        )
    if n_nan:
        warnings.warn(
            f"warm start: {n_nan} value(s) from the previous solve are NaN "
            f"({', '.join(nan_names)}); starting those variables from the default "
            "the solver uses for a variable nobody supplied. The previous result "
            "did not determine them.",
            stacklevel=3,
        )
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
