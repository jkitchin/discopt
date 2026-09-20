"""
NLP solver wrapper using pounce (pure-Rust Ipopt port).

Mirrors :mod:`discopt.solvers.nlp_ipopt` exactly. ``pounce.Problem`` is
shape-compatible with ``cyipopt.Problem`` (same constructor signature,
same ``add_option`` method, same ``(x, info)`` return from ``solve``,
same Ipopt status codes), so the callback adapter and bound inference
from the cyipopt backend are reused unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from discopt import _timing
from discopt.modeling.core import Model
from discopt.solvers import NLPResult, SolveStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    # #75: importing this at module scope pulls jax on every solve, because
    # ``nlp_backend`` imports this module to probe for POUNCE. It is only ever
    # used as an annotation here (this module has ``from __future__ import
    # annotations``), so it costs nothing at runtime.
    from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.solvers.nlp_ipopt import (
    _IPOPT_STATUS_MAP,
    _infer_constraint_bounds,
    _IpoptCallbacks,
)

try:
    import pounce as _pounce  # noqa: F401

    POUNCE_AVAILABLE = True
except ImportError:
    POUNCE_AVAILABLE = False

_logger = logging.getLogger(__name__)

#: Option keys this process has already warned about (see the ``add_option``
#: loop in :func:`solve_nlp`). Module-level so the warning is once per key per
#: process rather than once per solve.
_WARNED_REJECTED_OPTIONS: set[str] = set()


def _run_solve(problem, x0: np.ndarray, warm_start: Optional[object]):
    """Call ``pounce.Problem.solve``, with or without a warm start.

    ``x0`` still wins over ``warm_start.x`` inside pounce, and the caller has
    already clipped it into the box this NLP is solved on, so both are passed:
    the point from the box in force now, the multipliers and barrier parameter
    from the previous solve.
    """
    with _timing.charge("pounce"):
        if warm_start is not None:
            return problem.solve(x0.astype(np.float64), warm_start=warm_start)
        return problem.solve(x0.astype(np.float64))


def _kkt_from_info(info: dict) -> Optional[dict[str, float]]:
    """Terminal KKT residuals from POUNCE's ``info``, or ``None`` if absent.

    The four names #1247 specifies map to POUNCE's ``final_*`` entries, which are
    measured on the solver's internally **scaled** problem — that is what its own
    convergence test runs on, and what ``kkt_error`` means there. The
    ``*_unscaled`` entries beside them are the same residuals in the model's own
    units, which is what a certificate stated in problem units (a CALPHAD
    tangent-plane bound, say) must be built from. Both are reported rather than
    one silently standing in for the other; ``barrier_parameter`` is POUNCE's
    terminal ``mu``, which is what a subsequent warm start seeds ``mu_init``
    with.

    Missing entries are omitted rather than filled with a sentinel, so a consumer
    reading a key it did not get sees ``KeyError``/``None`` instead of a number
    the solver never reported.
    """
    mapping = {
        "primal_infeasibility": "final_constr_viol",
        "dual_infeasibility": "final_dual_inf",
        "complementarity": "final_compl",
        "kkt_error": "final_kkt_error",
        "primal_infeasibility_unscaled": "final_unscaled_constr_viol",
        "dual_infeasibility_unscaled": "final_unscaled_dual_inf",
        "complementarity_unscaled": "final_unscaled_compl",
        "kkt_error_unscaled": "final_unscaled_kkt_error",
        "barrier_parameter": "mu",
    }
    out: dict[str, float] = {}
    for name, key in mapping.items():
        val = info.get(key)
        if val is None:
            continue
        try:
            out[name] = float(val)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    return out or None


def _linear_solver_from_info(info: dict) -> Optional[dict]:
    """What the backend reports about its linear solver, or ``None`` (#1370).

    POUNCE reports the block-structured factorization under ``linear_solver``
    (``blocks``, ``n_blocks``, ``border_dim``) once pounce#955 ships; 0.12.0, the
    current pin, reports no such key at all. Absent is carried as ``None`` rather
    than as an empty dict that a caller could read as "the block path ran and
    found nothing" — the same rule ``_kkt_from_info`` follows for a residual the
    solver never reported.
    """
    raw = info.get("linear_solver")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    return {"report": raw}


def solve_nlp(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]] = None,
    options: Optional[dict] = None,
    kkt_schur_block: Optional[Sequence[int]] = None,
    ordering: Optional[Sequence[int]] = None,
    block_structure: Optional[tuple[Sequence[int], Sequence[int]]] = None,
    warm_start: Optional[object] = None,
) -> NLPResult:
    """Solve an NLP using pounce with the NLPEvaluator callbacks.

    Same signature and semantics as :func:`discopt.solvers.nlp_ipopt.solve_nlp`,
    plus two optional structure-aware passthroughs to the underlying
    ``pounce.Problem`` (see pounce#180).

    Args:
        kkt_schur_block: Optional sequence of **KKT-space indices** (``0..dim``,
            block order ``x, slack, eq-dual, ineq-dual``) identifying a
            block-triangular / Schur partition of the KKT system. Handed to
            ``pounce.Problem.set_kkt_schur_block`` before solving; pounce falls
            back to the full-space path transparently when the partition is
            unsuitable, so the solution is unchanged and only factorization time
            differs (Parker, Garcia & Bent, arXiv:2602.17968). Honored only on
            the default FERAL + exact-Hessian path.
        ordering: Optional sequence of KKT-space indices giving a custom
            factorization ordering, handed to ``pounce.Problem.set_ordering``.
            Correctness-safe for the same reason as ``kkt_schur_block``.
        block_structure: Optional ``(var_blocks, con_blocks)`` in **NLP index
            space** — one block id per column and one per row, negative for the
            shared border — handed to ``pounce.Problem.set_block_structure``,
            which maps the declaration onto the KKT layout itself (which columns
            survived fixing, how rows split into equalities and inequalities) and
            factorizes the blocks in parallel over a shared border (#1370,
            pounce#955). Unlike ``kkt_schur_block``, which takes KKT-space
            indices a modelling layer cannot reliably produce, this is the space
            the model knows; build it with
            :func:`discopt.block_structure.resolve_block_structure` rather than by
            hand, so the labels are checked against the emitted problem's own
            sparsity. Correctness-safe in the same sense as the two above: the
            factorization changes, the solution does not.
        warm_start: Optional ``pounce.WarmStart`` carrying a previous solve's
            primal point, constraint and bound multipliers, and barrier
            parameter (#1247). Handed to ``pounce.Problem.solve``, which derives
            the warm-start options (``warm_start_init_point``, ``mu_init``, the
            bound pushes) from it. Convergence-affecting only: where the solver
            starts cannot change what it certifies at termination — though on a
            *nonconvex* NLP it can change which local stationary point is
            reached.
    """
    if not POUNCE_AVAILABLE:
        raise ImportError(
            "pounce is required for this backend. Install it with:\n"
            "  pip install -e /path/to/pounce/python\n"
            "Or pick a different NLP backend (e.g. 'cyipopt')."
        )

    import pounce

    opts = dict(options) if options else {}
    opts.setdefault("print_level", 0)
    # NEITHER of the two shared POUNCE requests is seeded here, and both omissions
    # are measured rather than assumed (#945):
    #
    # `bound_relax_factor = 0` is requested by the call sites whose returned POINT
    # is the product (`solver._solve_continuous`, `oa._solve_nlp_attempt`,
    # `gdpopt_loa._solve_nlp_subproblem`) — see `solvers.pounce_incumbent_options`.
    # Backend-wide it also reaches Benders and GBD recourse, whose product is the
    # MULTIPLIERS, and a degenerate feasible set has no finite multiplier without
    # Ipopt's relaxation (#940: two correctness-lane tests, 1.6s -> 79s; #946).
    #
    # `constr_viol_tol = 1e-8` is a backend-wide default for the matrix-form
    # backends but is deliberately NOT routed here. Measured on nvs05 at a 20 s
    # budget, 4 arms interleaved, 3 reps, ZERO within-arm spread
    # (scratchpad/issue945/nvs05_attribution.py):
    #
    #     print_level only           objective  8.731956987   bound 3.542469244
    #     + constr_viol_tol=1e-8     objective 12.589492574   bound 3.513753544
    #     + incumbent options only    objective  8.731956987   bound 3.542469244
    #
    # so it costs a 31%-worse incumbent AND a looser bound, entirely on its own.
    # The incumbent it displaces is genuinely feasible — worst row violation
    # 1.8e-12, box 0, integrality 0 (nvs05_feasibility.py) — so this is a real
    # solution lost, not a false one rejected, which is what I had assumed and had
    # to retract. Against that: it fixes nothing here. The out-of-box defect #945 is
    # about is `bound_relax_factor`'s, and the arm above shows constr_viol_tol alone
    # does not touch it (mindtpy_cq stays at 2.9999000025 either way). Sound but not
    # helpful stays out (CLAUDE.md §5). Route it here only with a measurement
    # showing a benefit that outweighs nvs05.

    n = evaluator.n_variables
    m = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds
    # Snap tiny floating-point bound inversions (lb just above ub) so POUNCE's
    # IPM does not reject the problem as Invalid_Problem_Definition. These arise
    # from relaxation / bound-tightening rounding (e.g. an AMP integer-fixed
    # subproblem whose continuous bounds were tightened to lb=ub+1e-11); the
    # mirror of the same guard on the LP path (lp_pounce._snap_inverted_bounds).
    from discopt.solvers.lp_pounce import _snap_inverted_bounds

    lb, ub = _snap_inverted_bounds(
        np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)
    )

    if constraint_bounds is not None:
        cl = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        cu = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    elif m > 0:
        cl, cu = _infer_constraint_bounds(evaluator)
    else:
        cl = np.empty(0, dtype=np.float64)
        cu = np.empty(0, dtype=np.float64)

    callbacks = _IpoptCallbacks(evaluator)

    problem = pounce.Problem(
        n=n,
        m=m,
        problem_obj=callbacks,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    for key, value in opts.items():
        try:
            if isinstance(value, (np.floating, float)):
                problem.add_option(key, float(value))
            elif isinstance(value, (np.integer, int)):
                problem.add_option(key, int(value))
            else:
                problem.add_option(key, value)
        except (TypeError, ValueError, RuntimeError) as exc:
            # #1247: a rejected option used to vanish at DEBUG, so a misspelled
            # or unsupported key silently did nothing — the caller believed it
            # was solving under an option the solver never saw. Warn, once per
            # key per process so a backend-wide default that this pounce build
            # does not know cannot turn into per-solve spam.
            if key not in _WARNED_REJECTED_OPTIONS:
                _WARNED_REJECTED_OPTIONS.add(key)
                _logger.warning(
                    "pounce rejected the option %r (value %r): %s: %s. It is NOT in effect "
                    "for this solve [pounce-option-rejected].",
                    key,
                    value,
                    type(exc).__name__,
                    exc,
                )

    # Structure-aware KKT passthroughs (pounce#180). Both are correctness-safe:
    # pounce transparently falls back to the full-space path when the partition
    # or ordering is unsuitable, so only factorization time changes. Guarded so
    # an older pounce without these methods degrades gracefully to the
    # full-space solve rather than raising.
    if kkt_schur_block is not None:
        block = [int(i) for i in kkt_schur_block]
        if hasattr(problem, "set_kkt_schur_block"):
            try:
                problem.set_kkt_schur_block(block)
            except (TypeError, ValueError, RuntimeError):
                _logger.debug("pounce rejected kkt_schur_block, using full space")
        else:
            _logger.debug("pounce has no set_kkt_schur_block; ignoring passthrough")
    if ordering is not None:
        order = [int(i) for i in ordering]
        if hasattr(problem, "set_ordering"):
            try:
                problem.set_ordering(order)
            except (TypeError, ValueError, RuntimeError):
                _logger.debug("pounce rejected ordering, using default")
        else:
            _logger.debug("pounce has no set_ordering; ignoring passthrough")
    if block_structure is not None:
        var_blocks, con_blocks = block_structure
        vb = [int(b) for b in var_blocks]
        cb = [int(b) for b in con_blocks]
        if len(vb) != n or len(cb) != m:
            # Not pounce's fallback territory: a length mismatch means the labels
            # were built against a different problem than the one being solved,
            # and a partition that is merely in-range names the wrong columns.
            raise ValueError(
                f"block_structure has {len(vb)} variable and {len(cb)} constraint labels, "
                f"but this NLP has {n} columns and {m} rows. Build the labels from the "
                "evaluator that serves the solve (discopt.block_structure."
                "resolve_block_structure)."
            )
        if hasattr(problem, "set_block_structure"):
            try:
                problem.set_block_structure(vb, cb)
            except (TypeError, ValueError, RuntimeError):
                _logger.debug("pounce rejected block_structure, using full space")
        else:
            # pounce-solver 0.12.0, the current pin, has no such method: the
            # block-structured KKT path is pounce#955 and unreleased. Degrade to
            # the full-space solve rather than raising (#394's pattern).
            _logger.debug("pounce has no set_block_structure; ignoring passthrough")

    t0 = time.perf_counter()
    try:
        x, info = _run_solve(problem, x0, warm_start)
    except RuntimeError as exc:
        # #1247: pounce validates option NAMES at solve time, not at
        # ``add_option`` time, so a misspelled key reaches here as a raw Rust
        # ``OPTION_INVALID`` out of the middle of the solver. Name the option and
        # where it came from instead. The solve still fails — an option the
        # caller asked for that the solver will not honour is a refusal, not
        # something to drop and carry on with (CLAUDE.md §3).
        msg = str(exc)
        if "OPTION_INVALID" in msg or "Unknown option" in msg:
            raise ValueError(
                f"POUNCE rejected a solver option: {msg.splitlines()[0]}. Options reaching the "
                f"NLP backend on this solve: {sorted(opts)}. Fix or drop the offending key "
                "(Model.solve(ipopt_options={...}))."
            ) from exc
        raise
    wall_time = time.perf_counter() - t0

    status_code = info.get("status", -100)
    status = _IPOPT_STATUS_MAP.get(status_code, SolveStatus.ERROR)

    # (The interim cyipopt-retry-on-UNBOUNDED guard was removed once pounce#258 fixed
    # the actual root cause. jit1's B&B nodes converged to the node optimum and then
    # returned Ipopt status 3 "Search_Direction_Becomes_Too_Small", which this file's
    # _IPOPT_STATUS_MAP mis-mapped to UNBOUNDED — the false verdict the guard papered
    # over. pounce#258 makes the strict certificate reachable under strong objective
    # scaling, and the code-3 mis-map is fixed in nlp_ipopt._IPOPT_STATUS_MAP. Verified
    # post-#258: jit1's 26 node solves all return OPTIMAL, jit1 solves to 173982.61 on
    # pure POUNCE with no retry.)

    multipliers = info.get("mult_g", None)
    if multipliers is not None and len(multipliers) == 0:
        multipliers = None
    mult_x_L = info.get("mult_x_L", None)
    if mult_x_L is not None and len(mult_x_L) == 0:
        mult_x_L = None
    mult_x_U = info.get("mult_x_U", None)
    if mult_x_U is not None and len(mult_x_U) == 0:
        mult_x_U = None

    return NLPResult(
        status=status,
        x=np.asarray(x),
        objective=float(info.get("obj_val", np.nan)),
        kkt=_kkt_from_info(info),
        linear_solver=_linear_solver_from_info(info),
        multipliers=np.asarray(multipliers) if multipliers is not None else None,
        bound_multipliers_lower=np.asarray(mult_x_L) if mult_x_L is not None else None,
        bound_multipliers_upper=np.asarray(mult_x_U) if mult_x_U is not None else None,
        iterations=int(info.get("iter_count", 0)),
        wall_time=wall_time,
        raw_status=int(status_code),
    )


def solve_nlp_from_model(
    model: Model,
    x0: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
    kkt_schur_block: Optional[Sequence[int]] = None,
    ordering: Optional[Sequence[int]] = None,
    block_structure: object = "auto",
) -> NLPResult:
    """Convenience: create an NLPEvaluator from a model and solve with POUNCE.

    Same signature and semantics as
    :func:`discopt.solvers.nlp_ipopt.solve_nlp_from_model`.

    ``kkt_schur_block`` and ``ordering`` are optional structure-aware
    passthroughs to :func:`solve_nlp` (see its docstring for the full contract
    and how to construct the KKT-space indices by hand). Both are
    correctness-safe: pounce falls back to the full-space path transparently, so
    the solution is unchanged and only factorization time differs. They require a
    pounce build exposing ``Problem.set_kkt_schur_block`` / ``set_ordering``
    (absent in pounce-solver 0.7.0, the current pin); until then they are silently
    no-ops.

    Args:
        model: A Model with objective and constraints set.
        x0: Initial point (n,). If None, uses midpoint of bounds clipped to [-100, 100].
        options: POUNCE/Ipopt options dict.
        kkt_schur_block: Optional Schur/block-triangular KKT partition (see above).
        ordering: Optional custom KKT-space factorization ordering (see above).
        block_structure: ``"auto"`` (the default) resolves the model's own
            declaration — :meth:`Model.set_block` /
            :meth:`Model.set_constraint_block` — against the evaluator that
            serves this solve, and passes the labels on; a model that declares
            nothing is solved exactly as before. An explicit
            ``(var_blocks, con_blocks)`` pair is validated against the same
            evaluator and then passed; ``None`` skips the feature entirely. An
            inconsistent declaration raises
            :class:`~discopt.block_structure.BlockStructureError` rather than
            being downgraded to a full-space solve: it means the model says
            something about itself that is not true.

    Returns:
        NLPResult with solution.
    """
    from discopt._tape_nlp_evaluator import make_evaluator

    evaluator = make_evaluator(model)
    labels: Optional[tuple[Sequence[int], Sequence[int]]] = None
    if isinstance(block_structure, str):
        if block_structure != "auto":
            raise ValueError(
                f"block_structure={block_structure!r}; pass 'auto', None, or an explicit "
                "(var_blocks, con_blocks) pair."
            )
        from discopt.block_structure import block_structure_for_model

        resolved = block_structure_for_model(model, evaluator)
        labels = resolved.as_pair() if resolved is not None else None
    elif block_structure is not None:
        from discopt.block_structure import validate_block_labels

        labels = validate_block_labels(block_structure, evaluator).as_pair()

    if x0 is None:
        lb, ub = evaluator.variable_bounds
        lb_clipped = np.clip(lb, -100.0, 100.0)
        ub_clipped = np.clip(ub, -100.0, 100.0)
        x0 = 0.5 * (lb_clipped + ub_clipped)

    return solve_nlp(
        evaluator,
        x0,
        options=options,
        kkt_schur_block=kkt_schur_block,
        ordering=ordering,
        block_structure=labels,
    )
