"""
Multi-start primal heuristics for MINLP.

Finds good feasible solutions by launching NLP solves from diverse starting
points. Includes a feasibility pump that rounds fractional integer variables
and re-solves the resulting NLP.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator, cached_evaluator
from discopt.modeling.core import Model, VarType
from discopt.solvers import NLPResult, SolveStatus

# Iteration cap for the *sub-NLP* solves inside the primal heuristics (issue #268).
# These solves only need an approximately feasible point (the heuristic then checks
# integrality + constraints and lets B&B certify); they do NOT need a tight optimum.
# Left uncapped, a single pump/local-search projection can grind to the backend's
# full iteration limit (one ex1263 solve hit ~1225 IPM iterations) and burn the
# wall-clock budget that the branch-and-bound search needs. A generous cap bounds
# the pathological cases while leaving normal projections (well under it) untouched;
# the caller can still override via ``ipopt_options``/``nlp_options``. Sound: a
# capped, unconverged point simply fails the feasibility check and yields no
# incumbent — it can never inject a wrong one (inject_incumbent re-verifies).
_HEURISTIC_NLP_MAX_ITER = 300

# VOLUME-1 (docs/dev/nlp-solve-volume-2026-07-06.md): the objective-improvement
# coordinate descent inside ``integer_local_search`` (``_objective_improve``) is
# the single dominant NLP-solve SOURCE on the easy-panel instances BARON solves
# sub-second — nvs06 runs 888 sub-NLP solves there (of 911 total), nvs08 779,
# ex1224 217 — and its measured incumbent-improvement HIT RATE is 0 % on every
# one of them (the incumbent is already found by the root multistart's first
# start). Left uncapped, the descent keeps re-sweeping ``int_idx × {±1,±2}``
# until its wall deadline (~9 s), issuing hundreds of no-op sub-NLPs. This flag
# caps the number of sub-NLP solves the descent may issue to
# ``_ILS_SOLVE_CAP_MULT × n_int`` (a small multiple of the integer dimension —
# enough for a full first-improvement sweep or two, which is where any real gain
# lands, but not the hundreds-of-solves plateau). Default OFF (0 ⇒ unlimited =
# current behavior); set ``DISCOPT_ILS_SOLVE_CAP`` to a positive integer to arm
# it. Sound: capping this descent only ever *weakens* the incumbent it might
# find (the descent injects sub-NLP-verified points that B&B still re-verifies),
# and it never touches the dual bound or the certificate.
_ILS_SOLVE_CAP_MULT = int(os.environ.get("DISCOPT_ILS_SOLVE_CAP", "0"))


@dataclass
class MultiStartResult:
    """Result of multi-start NLP solving."""

    best_objective: Optional[float] = None
    best_solution: Optional[np.ndarray] = None
    n_starts: int = 0
    n_feasible: int = 0
    n_integer_feasible: int = 0
    all_objectives: list[float] = field(default_factory=list)


def _get_integer_mask(model: Model) -> np.ndarray:
    """Return a boolean mask over the flat variable vector: True where integer/binary."""
    parts: list[np.ndarray] = []
    for v in model._variables:
        is_int = v.var_type in (VarType.BINARY, VarType.INTEGER)
        parts.append(np.full(v.size, is_int, dtype=bool))
    return np.concatenate(parts) if parts else np.array([], dtype=bool)


def _get_variable_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return (lb, ub) flat arrays for all variables."""
    lbs: list[np.ndarray] = []
    ubs: list[np.ndarray] = []
    for v in model._variables:
        lbs.append(v.lb.flatten())
        ubs.append(v.ub.flatten())
    return np.concatenate(lbs), np.concatenate(ubs)


def _generate_starts(
    lb: np.ndarray,
    ub: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate diverse starting points within bounds.

    Uses stratified random sampling: divide [0, 1] into n_starts strata along
    each dimension and sample uniformly within each stratum, then scale to
    [lb, ub]. This gives better coverage than pure uniform random.
    """
    n = len(lb)
    # Clip infinite bounds for sampling purposes
    lb_safe = np.clip(lb, -1e6, 1e6)
    ub_safe = np.clip(ub, -1e6, 1e6)

    # Stratified sampling along a single axis with random permutation
    starts = np.empty((n_starts, n), dtype=np.float64)
    for j in range(n):
        # Divide [0,1] into n_starts strata, sample within each
        strata = (np.arange(n_starts) + rng.uniform(size=n_starts)) / n_starts
        rng.shuffle(strata)
        starts[:, j] = lb_safe[j] + strata * (ub_safe[j] - lb_safe[j])

    return starts


def _is_nlp_feasible(result: NLPResult) -> bool:
    """Check whether an NLP result represents a feasible solution."""
    return result.status in (SolveStatus.OPTIMAL,) and result.x is not None


def _is_integer_feasible(
    x: np.ndarray,
    int_mask: np.ndarray,
    tol: float = 1e-5,
) -> bool:
    """Check if integer variables are within tolerance of integer values."""
    if not np.any(int_mask):
        return True
    frac = np.abs(x[int_mask] - np.round(x[int_mask]))
    return bool(np.all(frac <= tol))


class MultiStartNLP:
    """Multi-start NLP solver for finding good feasible solutions.

    Generates diverse starting points and solves an NLP from each.
    Tracks the best feasible (and integer-feasible) solution found.
    """

    def __init__(
        self,
        model: Model,
        n_starts: int = 64,
        seed: int = 42,
    ) -> None:
        self._model = model
        self._n_starts = n_starts
        self._seed = seed

    def solve(
        self,
        ipopt_options: Optional[dict] = None,
        backend: Optional[Callable] = None,
    ) -> MultiStartResult:
        """Run multi-start NLP solving.

        Args:
            ipopt_options: Options passed to the NLP backend (e.g. max_iter, tol).
            backend: ``solve_nlp(evaluator, x0, options=...)`` callable. If None,
                resolves to ``get_nlp_solver("auto")``
                (POUNCE-preferred, falling back to cyipopt).

        Returns:
            MultiStartResult with best solution and statistics.
        """
        model = self._model
        evaluator = cached_evaluator(model)
        lb, ub = evaluator.variable_bounds
        int_mask = _get_integer_mask(model)
        has_integers = np.any(int_mask)

        rng = np.random.default_rng(self._seed)
        starts = _generate_starts(lb, ub, self._n_starts, rng)

        opts = dict(ipopt_options) if ipopt_options else {}
        opts.setdefault("print_level", 0)
        opts.setdefault("max_iter", _HEURISTIC_NLP_MAX_ITER)
        if backend is None:
            from discopt.solvers.nlp_backend import get_nlp_solver

            backend = get_nlp_solver("auto")

        result = MultiStartResult(n_starts=self._n_starts)
        best_obj = float("inf")

        for i in range(self._n_starts):
            x0 = starts[i]
            nlp_result = backend(evaluator, x0, options=opts)
            if not _is_nlp_feasible(nlp_result):
                continue

            result.n_feasible += 1
            assert nlp_result.objective is not None
            assert nlp_result.x is not None

            int_feas = _is_integer_feasible(nlp_result.x, int_mask)
            if int_feas:
                result.n_integer_feasible += 1

            result.all_objectives.append(nlp_result.objective)

            # For MINLP, only update incumbent if integer-feasible
            if has_integers and not int_feas:
                continue

            if nlp_result.objective < best_obj:
                best_obj = nlp_result.objective
                result.best_objective = nlp_result.objective
                result.best_solution = nlp_result.x.copy()

        return result


def feasibility_pump(
    model: Model,
    x_nlp: np.ndarray,
    max_rounds: int = 5,
    ipopt_options: Optional[dict] = None,
    backend: Optional[Callable] = None,
    evaluator: Optional[NLPEvaluator] = None,
    deadline: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Try to find an integer-feasible solution via rounding + re-solve.

    Given an NLP solution with fractional integer variables:
    1. Round integer variables to nearest integer.
    2. Fix integer variables and re-solve NLP for continuous variables.
    3. If feasible, return. Otherwise perturb and retry.

    Args:
        model: The optimization model.
        x_nlp: An NLP relaxation solution (may have fractional integers).
        max_rounds: Maximum rounding + re-solve attempts.
        ipopt_options: Options passed to the NLP backend.
        backend: ``solve_nlp(evaluator, x0, options=...)`` callable. If None,
            resolves to ``get_nlp_solver("auto")``
            (POUNCE-preferred, falling back to cyipopt).
        deadline: Optional ``time.perf_counter()`` wall-clock deadline. When the
            current time reaches it, the pump stops at the start of the next
            round and returns the best feasible solution found so far (or None).
            Keeps a tight global ``time_limit`` from being overrun by the root
            heuristic's per-round NLP solves.
        evaluator: Optional prebuilt :class:`NLPEvaluator` for ``model``. Reusing
            the caller's evaluator avoids rebuilding (and recompiling, ~3s) the
            JAX sparse-Hessian/Jacobian kernels for the same model structure.
            The evaluator reads variable bounds from the model on each solve, so
            the integer-pinning below is honoured regardless of which evaluator
            is used. If None, a fresh one is constructed.

    Returns:
        An integer-feasible solution vector, or None if not found.
    """
    int_mask = _get_integer_mask(model)
    if not np.any(int_mask):
        # No integer variables, the NLP solution is already integer-feasible.
        return x_nlp.copy()

    lb, ub = _get_variable_bounds(model)
    if evaluator is None:
        evaluator = cached_evaluator(model)

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", _HEURISTIC_NLP_MAX_ITER)
    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        backend = get_nlp_solver("auto")

    rng = np.random.default_rng(42)

    for round_idx in range(max_rounds):
        # Always run the first round (a feasible incumbent is the primary goal,
        # worth a small overrun); only the *extra* perturbation rounds are
        # deadline-gated so the pump cannot loop well past a tight ``time_limit``.
        if deadline is not None and round_idx > 0 and time.perf_counter() >= deadline:
            break
        x_try = x_nlp.copy()

        # Round integer variables
        x_try[int_mask] = np.round(x_try[int_mask])

        # Perturb on rounds > 0 by randomly flipping some integer values
        if round_idx > 0:
            flip_mask = int_mask & (rng.random(len(x_try)) < 0.3)
            perturbation = rng.choice([-1, 0, 1], size=len(x_try))
            x_try[flip_mask] = x_try[flip_mask] + perturbation[flip_mask]

        # Clip to bounds
        x_try = np.clip(x_try, lb, ub)
        x0 = x_try.copy()

        # Actually FIX the integer variables at their rounded values by pinning
        # lb = ub = rounded value on the model (NLPEvaluator reads bounds from the
        # model each call), then re-solve only the continuous variables. Without
        # this, the re-solve uses the original (open) bounds and drifts the
        # integers straight back to their fractional relaxation values, so the
        # rounding — and the perturbation below — accomplishes nothing. Bounds are
        # always restored in the finally so the search tree is left untouched.
        saved_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        try:
            offset = 0
            for v in model._variables:
                sz = v.size
                saved_bounds.append((v.lb.copy(), v.ub.copy()))
                if v.var_type in (VarType.BINARY, VarType.INTEGER):
                    fixed = x0[offset : offset + sz].reshape(v.lb.shape)
                    v.lb = fixed.copy()
                    v.ub = fixed.copy()
                offset += sz
            try:
                nlp_result = backend(evaluator, x0, options=opts)
            except BaseException:
                # Some NLP backends (pounce via PyO3) raise PanicException, which
                # is not a subclass of Exception; treat any failure as this round
                # producing no point and perturb on the next round.
                continue
        finally:
            for v, (lb_v, ub_v) in zip(model._variables, saved_bounds):
                v.lb = lb_v
                v.ub = ub_v

        if not _is_nlp_feasible(nlp_result) or nlp_result.x is None:
            continue

        x_cand = np.asarray(nlp_result.x).copy()
        # Snap any tiny drift on the pinned integers, then require BOTH integer
        # and constraint feasibility. Checking constraints here (not just
        # integrality, which is trivially satisfied once integers are pinned) is
        # what makes the perturbation loop useful: an infeasible rounding is
        # rejected and the next round tries a perturbed neighbour instead of
        # returning a point the caller will only discard.
        x_cand[int_mask] = np.round(x_cand[int_mask])
        if not _is_integer_feasible(x_cand, int_mask):
            continue
        if not _check_constraint_feasibility(evaluator, x_cand):
            continue
        return x_cand

    return None


def _check_constraint_feasibility(
    evaluator: NLPEvaluator,
    x: np.ndarray,
    tol: float = 1e-6,
    rtol: float = 1e-9,
) -> bool:
    """Check that ``x`` satisfies the model's constraints to within tolerance.

    A pure ABSOLUTE tolerance (``tol``) is too strict for constraints built from
    large-magnitude nonlinear terms: an objective-linking row such as
    ``592*x1**0.65 + ... - objvar <= 0`` evaluates as the difference of two
    quantities near 1.5e5, so its floating-point cancellation noise alone is
    ~1e-6 -- exactly the absolute tolerance. discopt then rejects a genuinely
    optimal point (prob07: the true global basin at obj 154990 carries a 2.4e-6
    residual on that one row and was discarded, leaving a worse 162070 incumbent)
    while BARON, which scales feasibility by constraint magnitude, accepts it.

    Use the conventional combined test ``|viol| <= tol + rtol*scale`` where the
    per-row ``scale`` is the absolute linearized magnitude ``sum_j |J_ij|*|x_j|``
    -- the size of the row's additive terms, derived from the Jacobian and NOT
    from the (possibly +/-1e20 sentinel) bound values, so an unbounded row cannot
    inflate the tolerance. ``rtol`` is kept extremely tight (1e-9) so this only
    ever forgives cancellation noise proportional to the terms; any violation of
    real consequence is still rejected. The absolute test is tried first and the
    Jacobian (the only added cost) is evaluated only for rows that fail it, so
    well-scaled feasible points keep the original cheap path unchanged.
    """
    if evaluator.n_constraints == 0:
        return True
    g = np.asarray(evaluator.evaluate_constraints(x))
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    cl, cu = (np.asarray(b, dtype=np.float64) for b in _infer_constraint_bounds(evaluator))
    viol = np.maximum(np.maximum(cl - g, 0.0), np.maximum(g - cu, 0.0))
    if bool(np.all(viol <= tol)):
        return True
    # Some row exceeds the absolute tolerance: re-test those rows against a
    # term-magnitude-scaled tolerance before declaring infeasibility.
    try:
        jac = np.abs(np.asarray(evaluator.evaluate_jacobian(x), dtype=np.float64))
        scale = jac @ np.abs(np.asarray(x, dtype=np.float64))
    except Exception:
        return False
    return bool(np.all(viol <= tol + rtol * scale))


def subnlp(
    model: Model,
    x_relax: np.ndarray,
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    integer_tol: float = 1e-5,
    feas_tol: float = 1e-6,
    evaluator: Optional[NLPEvaluator] = None,
    time_budget: Optional[float] = None,
) -> Optional[tuple[np.ndarray, float]]:
    """SubNLP-style primal heuristic: fix integers, re-solve continuous NLP.

    Given a relaxation point ``x_relax``:
    1. Round integer variables to their nearest integer value and tighten
       their bounds to that single value (fixed). For pure-continuous models,
       this step is a no-op.
    2. Solve the resulting continuous NLP from ``x_relax`` as warm start.
    3. Verify that the returned point is integer-feasible (trivial when
       integers are fixed) and constraint-feasible.

    Args:
        model: The optimization model.
        x_relax: Relaxation point (NLP-relaxed solution at a B&B node).
        backend: ``solve_nlp(evaluator, x0, options=...)`` callable. If None,
            uses :func:`discopt.solvers.nlp_backend.get_nlp_solver('auto')`.
        nlp_options: Options dict forwarded to the NLP backend.
        integer_tol: Tolerance for declaring integer feasibility.
        feas_tol: Tolerance for declaring constraint feasibility.
        evaluator: Pre-built NLPEvaluator; one is constructed if omitted.
        time_budget: Optional wall-clock cap (seconds) for the inner NLP solve.
            When set (and positive), it is forwarded to the backend as the
            ``max_wall_time`` option so a single subNLP solve cannot run past the
            caller's deadline. Unaccepted by backends that ignore the key (it is
            silently skipped there). An explicit ``max_wall_time`` already in
            ``nlp_options`` takes precedence.

    Returns:
        ``(x, obj)`` if the heuristic produced a usable integer- and
        constraint-feasible point, else ``None``.
    """
    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        backend = get_nlp_solver("auto")

    if evaluator is None:
        evaluator = cached_evaluator(model)

    int_mask = _get_integer_mask(model)
    lb_orig, ub_orig = _get_variable_bounds(model)

    x0 = np.asarray(x_relax, dtype=np.float64).copy()

    # Fix integer variables by rounding and clamping bounds to that value.
    # We mutate the model variables' bounds in-place and restore afterwards
    # since NLPEvaluator.variable_bounds reads from the model on each call.
    saved_bounds: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        if np.any(int_mask):
            x0[int_mask] = np.round(x0[int_mask])
            x0 = np.clip(x0, lb_orig, ub_orig)

            # Save and tighten bounds on integer variables.
            offset = 0
            for v in model._variables:
                sz = v.size
                saved_bounds.append((v.lb.copy(), v.ub.copy()))
                if v.var_type in (VarType.BINARY, VarType.INTEGER):
                    fixed = x0[offset : offset + sz].reshape(v.lb.shape)
                    v.lb = fixed.copy()
                    v.ub = fixed.copy()
                offset += sz

        opts = dict(nlp_options) if nlp_options else {}
        opts.setdefault("print_level", 0)
        opts.setdefault("max_iter", _HEURISTIC_NLP_MAX_ITER)
        if time_budget is not None and time_budget > 0.0:
            opts.setdefault("max_wall_time", float(time_budget))

        try:
            nlp_result = backend(evaluator, x0, options=opts)
        except BaseException:
            # Catch BaseException — some NLP backends (e.g. pounce via PyO3)
            # raise PanicException, which is not a subclass of Exception.
            return None
    finally:
        for v, (lb_v, ub_v) in zip(model._variables, saved_bounds):
            v.lb = lb_v
            v.ub = ub_v

    # Accept either a converged (OPTIMAL) solve or an ITERATION_LIMIT one: an
    # interior-point solver routinely caps out one step short of its convergence
    # test at a point that is already constraint-feasible (prob07's true-global
    # basin terminates at ITERATION_LIMIT, obj 154990). The shared
    # ``_is_nlp_feasible`` gate accepts OPTIMAL only and so discarded that point,
    # leaving a worse incumbent. Trusting the returned point is sound here only
    # because subnlp re-verifies genuine constraint- and integer-feasibility
    # below (``_check_constraint_feasibility`` / ``_is_integer_feasible``), and
    # ``inject_incumbent`` enforces strict improvement -- this mirrors the
    # acceptance set ``_solve_root_node_multistart`` already uses.
    if nlp_result.status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
        return None
    if nlp_result.x is None or nlp_result.objective is None:
        return None

    x_out = np.asarray(nlp_result.x)

    # Clip integer slots back to the rounded value (the fixed-bounds solve
    # should already yield this, but guard against tiny drifts).
    if np.any(int_mask):
        x_out = x_out.copy()
        x_out[int_mask] = np.round(x_out[int_mask])
        if not _is_integer_feasible(x_out, int_mask, tol=integer_tol):
            return None

    if not _check_constraint_feasibility(evaluator, x_out, tol=feas_tol):
        return None

    # Recompute objective at the snapped point to keep it consistent.
    obj = float(evaluator.evaluate_objective(x_out))
    return x_out, obj


def integer_local_search(
    model: Model,
    x_relax: np.ndarray,
    backend: Optional[Callable] = None,
    evaluator: Optional[NLPEvaluator] = None,
    nlp_options: Optional[dict] = None,
    max_restarts: int = 24,
    max_steps: int = 60,
    pair_cap: int = 40,
    time_budget: float = 3.0,
    feas_tol: float = 1e-6,
    seed: int = 0,
) -> Optional[tuple[np.ndarray, float]]:
    """Constraint-violation-guided integer local search (1-opt + 2-opt).

    The round-and-repair heuristics (:func:`feasibility_pump`, :func:`subnlp`)
    only repair the *continuous* variables — they take the relaxation's integer
    assignment as given. For integer-heavy nonconvex problems the relaxation's
    integers are optimal for the *relaxed* (e.g. McCormick) constraints yet
    violate the TRUE constraints, and no continuous re-solve fixes that. This
    heuristic instead searches the integer lattice directly: it descends the
    total true-constraint violation by unit moves (1-opt steepest descent; with
    pairwise 2-opt moves when 1-opt stalls — essential for bilinear constraints
    such as ``x*y >= c`` where a single variable cannot move the product), then
    repairs the continuous variables and verifies true feasibility via
    :func:`subnlp` at each local minimum. A few perturbation restarts escape
    shallow local minima.

    Sound by construction: only points that pass subnlp's integer- and
    constraint-feasibility checks are returned, so the caller may inject them as
    incumbents without affecting any dual bound or certification. The cost is
    bounded by ``time_budget`` (wall-clock), ``max_restarts`` and ``max_steps``;
    2-opt is skipped when the integer count exceeds ``pair_cap`` to avoid the
    O(n^2) neighbourhood blowing up on large models.

    Args:
        model: The optimization model.
        x_relax: Relaxation point (a B&B node's relaxed solution).
        backend: NLP backend for the continuous repair; resolved via
            ``get_nlp_solver('auto')`` when None.
        evaluator: Pre-built NLPEvaluator; one is constructed if omitted.
        nlp_options: Options forwarded to the NLP backend during repair.
        max_restarts: Number of perturbation restarts.
        max_steps: Max descent steps per restart.
        pair_cap: Max integer count for which 2-opt is enabled.
        time_budget: Wall-clock budget in seconds (the whole call returns early
            once exceeded).
        feas_tol: Constraint feasibility tolerance.
        seed: RNG seed for reproducible perturbations.

    Returns:
        ``(x, obj)`` for the best feasible point found, else ``None``.
    """
    import time

    if evaluator is None:
        evaluator = cached_evaluator(model)
    int_mask = _get_integer_mask(model)
    if not np.any(int_mask) or evaluator.n_constraints == 0:
        # Pure-continuous or unconstrained: nothing for an integer lattice
        # search to do — the continuous repair heuristics cover those.
        return None

    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        backend = get_nlp_solver("auto")

    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    lb, ub = _get_variable_bounds(model)
    cl, cu = _infer_constraint_bounds(evaluator)
    cl = np.asarray(cl, dtype=np.float64)
    cu = np.asarray(cu, dtype=np.float64)
    int_idx = np.where(int_mask)[0]
    n_int = int(int_idx.size)
    use_2opt = n_int <= pair_cap

    # Scale the search budget to the integer dimensionality. ``max_restarts`` and
    # ``max_steps`` default to constants sized for large lattices; on a small
    # integer space every restart re-descends to the same handful of points, so
    # each surplus restart is a wasted ``subnlp`` NLP solve — the dominant fixed
    # cost on tiny instances (see the SCIP head-to-head). One descent step can
    # move each integer at most ±1, so a local minimum is reached within O(range)
    # steps; restarts grow ~linearly with the dimension that perturbations
    # explore. Effort is only ever *reduced* below the caller's cap (never raised
    # past it), and this is a pure incumbent heuristic — subnlp-verified points
    # only, injected as candidates — so fewer restarts can only weaken the
    # incumbent (which B&B then closes), never the dual bound or certification.
    eff_restarts = min(max_restarts, max(3, 3 * n_int))
    eff_steps = min(max_steps, max(8, 4 * n_int))

    def violation(x: np.ndarray) -> float:
        g = np.asarray(evaluator.evaluate_constraints(x))
        return float(np.sum(np.maximum(0.0, cl - g)) + np.sum(np.maximum(0.0, g - cu)))

    deadline = time.perf_counter() + max(0.0, time_budget)
    has_continuous = bool(np.any(~int_mask))

    def _objective_improve(x_feas: np.ndarray, obj_feas: float) -> tuple[np.ndarray, float]:
        """Descend the OBJECTIVE over feasible integer neighbours from a feasible
        point. The violation descent above only reaches *a* feasible integer
        assignment; its objective can sit well above optimal (nvs24: a feasible
        -1022 vs the optimum -1033, two integer moves away). This first-improvement
        coordinate search over ±1/±2 integer steps — keeping only feasible,
        objective-improving moves — bridges that gap. Pure-integer models evaluate
        the objective directly; mixed models repair the continuous block via
        ``subnlp`` at each candidate so the returned point stays truly feasible.
        Sound: every returned point is feasible, so it is only ever an incumbent
        candidate and never affects the dual bound or certification."""
        bx = _round_clip(x_feas)
        best_x, best_obj = np.asarray(x_feas, dtype=np.float64).copy(), float(obj_feas)
        # VOLUME-1 sub-NLP solve cap (default OFF). When armed, cap the number of
        # continuous-repair sub-NLP solves this descent may issue to
        # ``_ILS_SOLVE_CAP_MULT × n_int`` — a full first-improvement sweep or two
        # — instead of re-sweeping until the wall deadline. Only the *extra*
        # no-op solves past a couple of sweeps are cut (measured 0 % hit rate on
        # the easy panel); the descent still injects any better point it finds.
        _solve_cap = _ILS_SOLVE_CAP_MULT * max(1, n_int) if _ILS_SOLVE_CAP_MULT > 0 else None
        _solves_used = 0
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for j in int_idx:
                for d in (-1.0, 1.0, -2.0, 2.0):
                    if time.perf_counter() >= deadline:
                        break
                    if _solve_cap is not None and _solves_used >= _solve_cap:
                        return best_x, best_obj
                    nv = bx[j] + d
                    if nv < lb[j] - 1e-9 or nv > ub[j] + 1e-9:
                        continue
                    xt = bx.copy()
                    xt[j] = nv
                    if has_continuous:
                        _solves_used += 1
                        cand = subnlp(
                            model,
                            xt,
                            backend=backend,
                            nlp_options=nlp_options,
                            evaluator=evaluator,
                            feas_tol=feas_tol,
                        )
                        if cand is None:
                            continue
                        cx, cobj = np.asarray(cand[0], dtype=np.float64), float(cand[1])
                    else:
                        if violation(xt) > feas_tol:
                            continue
                        cx, cobj = xt, float(evaluator.evaluate_objective(xt))
                    if cobj < best_obj - 1e-9:
                        best_x, best_obj = cx.copy(), cobj
                        bx = _round_clip(best_x)
                        improved = True
                        break
                if improved:
                    break
        return best_x, best_obj

    def _round_clip(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).copy()
        y[int_mask] = np.round(y[int_mask])
        return np.clip(y, lb, ub)

    # Seed pool. The caller's relaxation point can be a degenerate vertex of a
    # *different* relaxation (e.g. a McCormick node solution that parks integer
    # multipliers at a bound, killing every bilinear product) — a dead basin for
    # local moves. So also seed from the model's own continuous relaxation: the
    # NLPEvaluator treats integers as continuous in [lb, ub], so a single NLP
    # solve from the box midpoint yields a balanced fractional point that rounds
    # into a far better basin. Both are rounded and used as restart bases.
    seeds: list[np.ndarray] = [_round_clip(x_relax)]
    try:
        # Clip the bounds to a finite window BEFORE averaging: unbounded
        # variables (lb=-inf and/or ub=+inf, common once a model has free
        # continuous vars like nvs05's x4..x7) make ``0.5*(lb+ub)`` evaluate to
        # +/-inf or NaN, which poisons the whole midpoint seed and silently
        # discards this second (relaxation) restart base. Clip first so every
        # coordinate is a finite, usable start.
        mid = np.clip(0.5 * (np.clip(lb, -1e3, 1e3) + np.clip(ub, -1e3, 1e3)), -1e3, 1e3)
        relax_opts = dict(nlp_options) if nlp_options else {}
        relax_opts.setdefault("print_level", 0)
        relax_opts.setdefault("max_iter", _HEURISTIC_NLP_MAX_ITER)
        relax_res = backend(evaluator, mid, options=relax_opts)
        if relax_res is not None and relax_res.x is not None:
            seeds.append(_round_clip(np.asarray(relax_res.x)))
    except BaseException:
        # Backend may panic (pounce/PyO3); fall back to the caller's seed alone.
        pass

    rng = np.random.default_rng(seed)
    best: Optional[tuple[np.ndarray, float]] = None
    n_seeds = len(seeds)

    for restart in range(max(eff_restarts, n_seeds)):
        if time.perf_counter() >= deadline:
            break
        # Try each seed clean first (descent alone often suffices), then spend
        # remaining restarts perturbing — from a feasible base once one is found,
        # else cycling the seed pool to diversify the basin.
        if restart < n_seeds:
            xc = seeds[restart].copy()
        else:
            base = best[0] if best is not None else seeds[restart % n_seeds]
            xc = _round_clip(base)
            sel = rng.choice(int_idx, size=int(rng.integers(1, n_int + 1)), replace=False)
            if rng.random() < 0.5:
                # Local ±1 nudge: explore the current basin's neighbourhood.
                step = rng.choice([-1.0, 0.0, 1.0], size=sel.size)
                xc[sel] = np.clip(xc[sel] + step, lb[sel], ub[sel])
            else:
                # Full-domain random resample: a global jump that can reach a
                # disconnected feasible well. Essential when feasibility lives on
                # an isolated discrete set — e.g. integers pinned by a high-degree
                # equality (i-1)(i-2)...(i-k)=0, where every ±1 step off a root
                # explodes the violation, so local moves can never hop roots.
                # Descent from the random point then drives each integer to its
                # nearest root, so diverse random draws cover diverse root combos.
                for j in sel:
                    xc[j] = float(rng.integers(int(lb[j]), int(ub[j]) + 1))

        cur = violation(xc)
        for _ in range(eff_steps):
            if cur <= feas_tol or time.perf_counter() >= deadline:
                break
            best_v = cur
            best_move: Optional[tuple[tuple[int, float], ...]] = None
            # 1-opt steepest descent.
            for j in int_idx:
                for d in (-1.0, 1.0):
                    nv = xc[j] + d
                    if nv < lb[j] - 1e-9 or nv > ub[j] + 1e-9:
                        continue
                    xt = xc.copy()
                    xt[j] = nv
                    v = violation(xt)
                    if v < best_v - 1e-9:
                        best_v, best_move = v, ((j, nv),)
            # 2-opt fallback only when 1-opt cannot improve (bilinear coupling).
            if best_move is None and use_2opt and time.perf_counter() < deadline:
                for a in range(n_int):
                    ja = int_idx[a]
                    for b in range(a + 1, n_int):
                        jb = int_idx[b]
                        for da in (-1.0, 1.0):
                            na = xc[ja] + da
                            if na < lb[ja] - 1e-9 or na > ub[ja] + 1e-9:
                                continue
                            for db in (-1.0, 1.0):
                                nb = xc[jb] + db
                                if nb < lb[jb] - 1e-9 or nb > ub[jb] + 1e-9:
                                    continue
                                xt = xc.copy()
                                xt[ja] = na
                                xt[jb] = nb
                                v = violation(xt)
                                if v < best_v - 1e-9:
                                    best_v, best_move = v, ((ja, na), (jb, nb))
            if best_move is None:
                break  # local minimum
            for j, nv in best_move:
                xc[j] = nv
            cur = best_v

        # Repair the continuous variables at this (locally good) integer
        # assignment and verify TRUE feasibility. subnlp only returns feasible,
        # integer-consistent points, so anything it gives back is a valid
        # incumbent candidate.
        repaired = subnlp(
            model,
            xc,
            backend=backend,
            nlp_options=nlp_options,
            evaluator=evaluator,
            feas_tol=feas_tol,
        )
        if repaired is not None:
            x_ok, obj_ok = repaired
            # Turn "a feasible point" into "the locally objective-best feasible
            # point" before recording it (and before perturbed restarts dive from
            # it), so the heuristic returns the strong incumbent the dual bound is
            # already tight enough to certify (nvs24: -1022 -> the optimum -1033).
            x_ok, obj_ok = _objective_improve(np.asarray(x_ok), float(obj_ok))
            if best is None or obj_ok < best[1]:
                best = (np.asarray(x_ok).copy(), float(obj_ok))
            # Once feasible, later perturbed restarts dive from this point's
            # neighbourhood (see the restart-base selection above) to improve it.

    return best


def integer_box_search(
    model: Model,
    x_incumbent: np.ndarray,
    *,
    radius: int = 2,
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    evaluator: Optional[NLPEvaluator] = None,
    max_int_vars: int = 3,
    max_combos: int = 128,
    integer_tol: float = 1e-5,
    feas_tol: float = 1e-6,
    time_budget: float = 4.0,
) -> Optional[tuple[np.ndarray, float]]:
    """Objective-improving integer *box* search around an incumbent.

    :func:`local_branching` only flips *binary* variables, and
    :func:`integer_local_search` descends constraint *violation* — it stops at
    the first feasible integer point and never makes an objective-improving move
    among feasible neighbours. For general-integer models a feasible incumbent
    can sit next to a far better feasible assignment that no unit (1-opt/2-opt)
    move reaches, because the connecting lattice path is objective-*increasing*
    or threads through infeasible points. Concretely, nvs05 parks at the feasible
    ``(i1=3, i2=2) -> 7.75`` while the global ``(i1=5, i2=1) -> 5.47`` is two
    coupled integer steps away over an objective-increasing ridge, with
    ``(3,1)``/``(4,1)`` infeasible (``i1**2*i2 >= 16.8``).

    This enumerates the ``+/-radius`` integer box around the incumbent's integer
    assignment, fixes each combination, and re-solves the continuous sub-NLP via
    :func:`subnlp`, returning the best *strictly improving* feasible point (or
    ``None``). It is the general-integer analogue of local branching.

    Bounded and sound: it only fires for a small integer count
    (``n_int <= max_int_vars``) and a small grid (``<= max_combos`` cells, capped
    further by each variable's own ``[lb, ub]``), every returned point is
    subnlp-verified feasible, and the caller injects it only on strict
    improvement — so the dual bound and certification are untouched.
    """
    import time

    int_mask = _get_integer_mask(model)
    int_idx = np.where(int_mask)[0]
    n_int = int(int_idx.size)
    if n_int == 0 or n_int > max_int_vars:
        return None

    lb, ub = _get_variable_bounds(model)
    x_inc = np.asarray(x_incumbent, dtype=np.float64)
    if x_inc.size <= int(int_idx.max()):
        return None
    centers = np.round(x_inc[int_idx])

    # Per-variable candidate values, clamped to the box AND to [lb, ub].
    axes: list[list[float]] = []
    for k, j in enumerate(int_idx):
        lo = max(int(np.ceil(lb[j] - 1e-9)), int(centers[k]) - radius)
        hi = min(int(np.floor(ub[j] + 1e-9)), int(centers[k]) + radius)
        if hi < lo:
            return None
        axes.append([float(v) for v in range(lo, hi + 1)])

    n_combos = 1
    for ax in axes:
        n_combos *= len(ax)
    # A single-cell grid means the incumbent is pinned with no neighbours to try.
    if n_combos <= 1 or n_combos > max_combos:
        return None

    if evaluator is None:
        evaluator = cached_evaluator(model)

    # Warm-start propagation. The continuous sub-NLP at a *neighbour* integer
    # assignment often has a narrow nonconvex feasible basin that the incumbent's
    # continuous values (a poor start once the integers shift by more than a step)
    # or a generic midpoint/random start miss entirely — so a better feasible
    # assignment a few integer steps away is never reached. Instead, expand the
    # box in rings of increasing Chebyshev distance from the incumbent and seed
    # each cell from an ALREADY-SOLVED feasible lattice-neighbour's continuous
    # values. Every hop is then a single integer step from a feasible point, so
    # the NLP stays in-basin and walks outward one ring at a time (e.g. nvs05
    # (3,2) -> (4,2) -> (5,2) -> (5,1) reaches the global 5.47). One NLP per cell,
    # deterministic, deadline-bounded. Sound: subnlp-verified feasible points only.
    center_key = tuple(int(centers[k]) for k in range(n_int))
    cont_at: dict[tuple[int, ...], np.ndarray] = {center_key: x_inc.copy()}

    def cheby(combo: tuple[int, ...]) -> int:
        return max(abs(combo[k] - center_key[k]) for k in range(n_int))

    combos = sorted(
        (tuple(int(v) for v in c) for c in itertools.product(*axes)),
        key=lambda c: (cheby(c), sum(abs(c[k] - center_key[k]) for k in range(n_int))),
    )

    deadline = time.perf_counter() + max(0.0, time_budget)
    best: Optional[tuple[np.ndarray, float]] = None
    for combo in combos:
        if time.perf_counter() >= deadline:
            break
        if combo == center_key:
            continue  # the incumbent's own cell — nothing to improve on
        # Seed from the nearest already-solved feasible neighbour (smallest L1
        # gap), falling back to the incumbent's continuous values.
        seed_src = x_inc
        best_gap = None
        for key, cont in cont_at.items():
            gap = sum(abs(combo[k] - key[k]) for k in range(n_int))
            if best_gap is None or gap < best_gap:
                best_gap, seed_src = gap, cont
        seed = seed_src.copy()
        for k, j in enumerate(int_idx):
            seed[j] = float(combo[k])
        found = subnlp(
            model,
            seed,
            backend=backend,
            nlp_options=nlp_options,
            evaluator=evaluator,
            integer_tol=integer_tol,
            feas_tol=feas_tol,
        )
        if found is not None:
            cont_at[combo] = np.asarray(found[0]).copy()
            if best is None or found[1] < best[1]:
                best = (np.asarray(found[0]).copy(), float(found[1]))
    return best


def enumerate_binary_seeds_subnlp(
    model: Model,
    x_relax: np.ndarray,
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    evaluator: Optional[NLPEvaluator] = None,
    max_binaries: int = 4,
    integer_tol: float = 1e-5,
) -> list[tuple[np.ndarray, float]]:
    """Root primal heuristic: enumerate every 0/1 assignment of the binaries.

    A single nearest-rounding :func:`subnlp` seed lands in whichever disjunct
    the relaxation's selector points at — and for a nonconvex disjunctive (GDP)
    model the relaxation can return an *integer-feasible but only locally
    optimal* selector (e.g. the wrong branch of an ``if_else``). Which branch it
    settles on is decided by tiny, platform-dependent floating-point differences
    in the relaxation solution, so the global optimum is found on one platform
    and missed on another — unwanted nondeterminism. Critically the offending
    selector is usually *not* fractional: the relaxation reports it at a clean
    0/1, so rounding heuristics never reconsider it.

    This heuristic removes that dependence at the root: it enumerates *all* 0/1
    assignments over the (capped) set of binary variables — fractional or not —
    and solves a fixed-integer sub-NLP from each. For a single disjunction the
    enumeration covers every disjunct, so the optimal one is always tried
    regardless of which branch the relaxation happened to lock onto. Seeds whose
    fixing is infeasible simply yield no sub-NLP solution and are dropped.

    Each disjunct is attempted from two continuous starts — the relaxation point
    and a neutral bound midpoint. The disaggregated perspective variables in the
    relaxation point carry the *settled* disjunct's values, a poor (and
    platform-sensitively convergent) start for the others; the second start
    makes escaping to the right disjunct robust across platforms.

    The cost is bounded to ``2 ** (max_binaries + 1)`` sub-NLP solves; when the
    model has more than ``max_binaries`` binaries the enumeration is skipped (an
    empty list is returned) to avoid combinatorial blow-up, leaving the regular
    rounding/pump heuristics in charge. Intended for a single root invocation.

    Args:
        model: The optimization model.
        x_relax: Relaxation point at the root node (used as the continuous seed).
        backend: ``solve_nlp(evaluator, x0, options=...)`` callable; resolved to
            ``get_nlp_solver("auto")`` if None.
        nlp_options: Options forwarded to the NLP backend.
        evaluator: Pre-built evaluator; one is constructed if omitted.
        max_binaries: Maximum number of binaries to enumerate over; above this
            the enumeration is skipped entirely.
        integer_tol: Integrality tolerance forwarded to :func:`subnlp`.

    Returns:
        Every feasible ``(x, obj)`` found across the enumerated seeds (possibly
        empty). The caller injects each as an incumbent candidate and lets the
        B&B tree keep the best, so this is agnostic to the objective sense.
    """
    int_mask = _get_integer_mask(model)
    if not np.any(int_mask):
        return []

    lb, ub = _get_variable_bounds(model)
    x_relax = np.asarray(x_relax, dtype=np.float64)

    # Binary variables: integer-typed with [0, 1] bounds. Enumerate over *all*
    # of them, not just the fractional ones — the selector that traps the
    # relaxation in the wrong disjunct is typically reported at a clean 0/1.
    binary_idx = [i for i in np.nonzero(int_mask)[0] if lb[i] >= -1e-9 and ub[i] <= 1.0 + 1e-9]
    if not binary_idx or len(binary_idx) > max_binaries:
        return []

    # Continuous (non-binary) base seeds. The relaxation point is a good start
    # for whichever disjunct it settled in, but a *bad* one for the others: the
    # disaggregated perspective variables carry the settled disjunct's (nonzero)
    # values, so for the other disjuncts the NLP must restore them toward 0 —
    # an ill-conditioned step (the hull perspective divides by y_k + eps) where
    # the solver can stall nondeterministically under load. A zero-continuous
    # start sidesteps that: the inactive disaggregated variables begin at their
    # feasible 0, leaving only the active (convex) disjunct to solve, which
    # converges robustly regardless of platform or CPU contention.
    cont_mask = ~np.isin(np.arange(len(x_relax)), binary_idx)
    zero_start = x_relax.copy()
    zero_start[cont_mask] = np.clip(0.0, lb, ub)[cont_mask]
    base_seeds = [zero_start, x_relax]

    results: list[tuple[np.ndarray, float]] = []
    for combo in itertools.product((0.0, 1.0), repeat=len(binary_idx)):
        for base in base_seeds:
            seed = base.copy()
            for idx, value in zip(binary_idx, combo):
                seed[idx] = value
            found = subnlp(
                model,
                seed,
                backend=backend,
                nlp_options=nlp_options,
                evaluator=evaluator,
                integer_tol=integer_tol,
            )
            if found is not None:
                results.append(found)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Improvement heuristics: diving, RINS, local branching
#
# These follow the SOTA rule inventory's "incumbent search" component. They only
# ever *propose* feasible incumbents; they never alter the dual (lower) bound, so
# they cannot weaken the global optimality certificate. All bound mutations on
# the model are temporary and restored in a ``finally`` block.
# ─────────────────────────────────────────────────────────────────────────────


def _flat_slot_map(model: Model) -> list[tuple[object, int]]:
    """Map each flat variable index to ``(variable, local_offset)``.

    Lets a heuristic fix a single scalar slot of a (possibly vector) variable by
    writing into ``v.lb.flat[local]`` / ``v.ub.flat[local]``.
    """
    slots: list[tuple[object, int]] = []
    for v in model._variables:
        for local in range(v.size):
            slots.append((v, local))
    return slots


def _resolve_backend(backend: Optional[Callable]) -> Callable:
    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        return get_nlp_solver("auto")
    return backend


def _fix_slot(v: object, local: int, value: float) -> None:
    """Fix scalar slot ``local`` of variable ``v`` to ``value``.

    Variable bound arrays are read-only, so we replace them with writable copies
    (the caller saves/restores the originals).
    """
    new_lb = np.array(v.lb, dtype=np.float64)  # type: ignore[attr-defined]
    new_ub = np.array(v.ub, dtype=np.float64)  # type: ignore[attr-defined]
    new_lb.flat[local] = value
    new_ub.flat[local] = value
    v.lb = new_lb  # type: ignore[attr-defined]
    v.ub = new_ub  # type: ignore[attr-defined]


def _finalize_candidate(
    evaluator: NLPEvaluator,
    x: np.ndarray,
    int_mask: np.ndarray,
    integer_tol: float,
    feas_tol: float,
) -> Optional[tuple[np.ndarray, float]]:
    """Snap integers, verify integer + constraint feasibility, return (x, obj)."""
    x_out = np.asarray(x, dtype=np.float64).copy()
    if np.any(int_mask):
        x_out[int_mask] = np.round(x_out[int_mask])
        if not _is_integer_feasible(x_out, int_mask, tol=integer_tol):
            return None
    if not _check_constraint_feasibility(evaluator, x_out, tol=feas_tol):
        return None
    obj = float(evaluator.evaluate_objective(x_out))
    return x_out, obj


def diving(
    model: Model,
    x_relax: np.ndarray,
    *,
    mode: str = "fractional",
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    max_dives: Optional[int] = None,
    integer_tol: float = 1e-5,
    feas_tol: float = 1e-6,
    evaluator: Optional[NLPEvaluator] = None,
    deadline: Optional[float] = None,
) -> Optional[tuple[np.ndarray, float]]:
    """Diving heuristic: progressively fix one fractional integer and re-solve.

    Starting from a relaxation point, each dive step selects one fractional
    integer variable, fixes it to a rounded value, and re-solves the continuous
    NLP relaxation under the accumulated fixings. The dive ends when all integers
    are integral (success) or a sub-NLP is infeasible (failure).

    ``mode`` selects the variable/direction rule:

    * ``"fractional"`` — fix the most fractional integer (closest to 0.5),
      rounding to the nearest integer.
    * ``"objective"`` — fix the most fractional integer, rounding in the
      direction the objective gradient prefers (down where ``dF/dx_i > 0`` for a
      minimization, i.e. toward the cheaper neighbour).

    Returns ``(x, obj)`` for a feasible incumbent, else ``None``. The dual bound
    is never touched.
    """
    int_mask = _get_integer_mask(model)
    int_idx = np.nonzero(int_mask)[0]
    if int_idx.size == 0:
        return None

    backend = _resolve_backend(backend)
    if evaluator is None:
        evaluator = cached_evaluator(model)
    lb0, ub0 = _get_variable_bounds(model)
    slot_map = _flat_slot_map(model)

    opts = dict(nlp_options) if nlp_options else {}
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", _HEURISTIC_NLP_MAX_ITER)

    fixed = np.zeros(int_mask.shape[0], dtype=bool)
    x_cur = np.clip(np.asarray(x_relax, dtype=np.float64), lb0, ub0)
    saved = [(v.lb.copy(), v.ub.copy()) for v in model._variables]
    budget = max_dives if max_dives is not None else int(int_idx.size) + 1

    try:
        for _ in range(budget):
            # Each dive step is a full continuous NLP solve. On the no-relaxation
            # flowsheet class those solves are seconds each and (worse) overrun
            # their own ``max_wall_time`` because each IPM iteration's exact
            # Hessian is expensive — so ``budget`` unpolled dives blow a tight
            # ``time_limit`` (heatexch_gen3: diving alone ran tens of seconds past
            # the deadline, F4). Poll the absolute deadline before launching each
            # sub-NLP and stop the dive when it has passed. Skipping the remaining
            # dive steps is always sound: diving is a primal heuristic and never
            # affects the dual bound.
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            try:
                res = backend(evaluator, x_cur, options=opts)
            except BaseException:
                return None
            if not _is_nlp_feasible(res):
                return None
            x = np.asarray(res.x, dtype=np.float64)

            frac = np.abs(x - np.round(x))
            cand = [i for i in int_idx if not fixed[i] and frac[i] > integer_tol]
            if not cand:
                return _finalize_candidate(evaluator, x, int_mask, integer_tol, feas_tol)

            # Select the most fractional unfixed integer (closest to 0.5).
            sel = min(cand, key=lambda i: abs(frac[i] - 0.5))

            if mode == "objective":
                try:
                    grad = np.asarray(evaluator.evaluate_gradient(x))
                    # Minimization: round toward the cheaper neighbour.
                    rounded = np.floor(x[sel]) if grad[sel] > 0 else np.ceil(x[sel])
                except BaseException:
                    rounded = np.round(x[sel])
            else:
                rounded = np.round(x[sel])
            rounded = float(np.clip(rounded, lb0[sel], ub0[sel]))

            v, local = slot_map[sel]
            _fix_slot(v, local, rounded)
            fixed[sel] = True
            x_cur = x.copy()
            x_cur[sel] = rounded
            x_cur = np.clip(x_cur, lb0, ub0)
        return None
    finally:
        for v, (lb_v, ub_v) in zip(model._variables, saved):
            v.lb = lb_v
            v.ub = ub_v


def fractional_diving(
    model: Model, x_relax: np.ndarray, **kwargs
) -> Optional[tuple[np.ndarray, float]]:
    """Diving that fixes the most fractional integer, rounding to nearest."""
    return diving(model, x_relax, mode="fractional", **kwargs)


def objective_diving(
    model: Model, x_relax: np.ndarray, **kwargs
) -> Optional[tuple[np.ndarray, float]]:
    """Diving that rounds toward the objective-preferred neighbour."""
    return diving(model, x_relax, mode="objective", **kwargs)


def rins(
    model: Model,
    x_incumbent: np.ndarray,
    x_relax: np.ndarray,
    *,
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    integer_tol: float = 1e-5,
    feas_tol: float = 1e-6,
    evaluator: Optional[NLPEvaluator] = None,
    deadline: Optional[float] = None,
) -> Optional[tuple[np.ndarray, float]]:
    """RINS (Relaxation Induced Neighborhood Search).

    Fix every integer variable on which the incumbent and the relaxation agree,
    then dive on the remaining (disagreeing) integers. This searches the
    neighbourhood "between" the incumbent and the relaxation — often where better
    incumbents hide — at the cost of one restricted dive. Returns ``(x, obj)`` or
    ``None``; the dual bound is never touched.
    """
    int_mask = _get_integer_mask(model)
    int_idx = np.nonzero(int_mask)[0]
    if int_idx.size == 0:
        return None

    if evaluator is None:
        evaluator = cached_evaluator(model)
    lb0, ub0 = _get_variable_bounds(model)
    slot_map = _flat_slot_map(model)

    x_inc = np.asarray(x_incumbent, dtype=np.float64)
    x_rel = np.asarray(x_relax, dtype=np.float64)

    agree = [
        i
        for i in int_idx
        if abs(np.round(x_inc[i]) - np.round(x_rel[i])) <= integer_tol
        and abs(x_rel[i] - np.round(x_rel[i])) <= integer_tol
    ]
    # Nothing fixed (full disagreement) degenerates to a plain dive; nothing free
    # (full agreement) means RINS has no neighbourhood to explore.
    if len(agree) == int_idx.size:
        return None

    saved = [(v.lb.copy(), v.ub.copy()) for v in model._variables]
    try:
        for i in agree:
            val = float(np.clip(np.round(x_inc[i]), lb0[i], ub0[i]))
            v, local = slot_map[i]
            _fix_slot(v, local, val)
        # Dive on the restricted model (fresh evaluator reads the tightened bounds).
        return diving(
            model,
            x_rel,
            mode="fractional",
            backend=backend,
            nlp_options=nlp_options,
            integer_tol=integer_tol,
            feas_tol=feas_tol,
            deadline=deadline,
        )
    finally:
        for v, (lb_v, ub_v) in zip(model._variables, saved):
            v.lb = lb_v
            v.ub = ub_v


def _restrict_slot(v: object, local: int, lo: float, hi: float) -> None:
    """Restrict scalar slot ``local`` of ``v`` to the range ``[lo, hi]``.

    Like :func:`_fix_slot` but sets a (possibly non-degenerate) bound range; the
    caller saves/restores the originals.
    """
    new_lb = np.array(v.lb, dtype=np.float64)  # type: ignore[attr-defined]
    new_ub = np.array(v.ub, dtype=np.float64)  # type: ignore[attr-defined]
    new_lb.flat[local] = lo
    new_ub.flat[local] = hi
    v.lb = new_lb  # type: ignore[attr-defined]
    v.ub = new_ub  # type: ignore[attr-defined]


def rens(
    model: Model,
    x_relax: np.ndarray,
    *,
    sub_solver: Callable[[Model], Optional[tuple[np.ndarray, float]]],
    integer_tol: float = 1e-5,
    max_free: int = 24,
) -> Optional[tuple[np.ndarray, float]]:
    """RENS (Relaxation Enforced Neighborhood Search).

    Fix every integer that is (near-)integral in the relaxation ``x_relax`` and
    restrict each *fractional* integer to its ``{floor, ceil}`` unit box. The
    resulting sub-MINLP — far smaller than the original, since only the fractional
    integers stay free — is solved *exactly* by ``sub_solver(model)``, which sees
    the tightened bounds and returns ``(x_flat, obj)`` or ``None``.

    RENS thus lands the **optimal** integer assignment in the relaxation's
    rounding neighbourhood, where all-at-once rounding (the feasibility pump) and
    greedy single-direction diving settle for a feasible-but-suboptimal one. On a
    near-integral convex relaxation (the typical MIQP case) the neighbourhood is
    tiny and its optimum is usually the global optimum, so injecting it early
    collapses the surrounding branch-and-bound search to a quick optimality proof.

    Returns ``None`` (cheaply, after only a fractionality count) when more than
    ``max_free`` integers are fractional — the neighbourhood is then too large to
    be worth an exact sub-solve, and the caller should fall back to the pump /
    diving. The model's bounds are always restored before returning; the dual
    bound is never touched (the caller injects the result only on improvement).

    Reference: Berthold, "RENS — the optimal rounding", Math. Prog. Comp. 2014.
    """
    int_mask = _get_integer_mask(model)
    int_idx = np.nonzero(int_mask)[0]
    if int_idx.size == 0:
        return None
    x = np.asarray(x_relax, dtype=np.float64)
    if x.size <= int(int_idx.max()):
        return None
    lb0, ub0 = _get_variable_bounds(model)
    frac = np.abs(x[int_idx] - np.round(x[int_idx]))
    if int((frac > integer_tol).sum()) > max_free:
        return None

    slot_map = _flat_slot_map(model)
    saved = [(v.lb.copy(), v.ub.copy()) for v in model._variables]
    try:
        for k, i in enumerate(int_idx):
            xi = float(x[i])
            if frac[k] <= integer_tol:
                lo = hi = float(np.clip(np.round(xi), lb0[i], ub0[i]))
            else:
                lo = float(np.clip(np.floor(xi), lb0[i], ub0[i]))
                hi = float(np.clip(np.ceil(xi), lb0[i], ub0[i]))
            v, local = slot_map[i]
            _restrict_slot(v, local, lo, hi)
        return sub_solver(model)
    finally:
        for v, (lb_v, ub_v) in zip(model._variables, saved):
            v.lb = lb_v
            v.ub = ub_v


def _binary_slot_term(model: Model, flat_idx: int):
    """Build a scalar modeling Expression for binary flat slot ``flat_idx``.

    Maps the flat index to its backing :class:`Variable` and component and
    returns either the scalar variable itself (``size == 1``) or the indexed
    component ``v[unravel(local)]``, suitable for assembling a linear cut.
    """
    slot_map = _flat_slot_map(model)
    v, local = slot_map[flat_idx]
    if v.size == 1:  # type: ignore[attr-defined]
        return v
    shape = v.shape  # type: ignore[attr-defined]
    if len(shape) <= 1:
        return v[local]  # type: ignore[index]
    return v[tuple(int(i) for i in np.unravel_index(local, shape))]  # type: ignore[index]


def _local_branching_submip(
    model: Model,
    x_incumbent: np.ndarray,
    binary_idx: list[int],
    *,
    k: int,
    backend: Optional[Callable],
    nlp_options: Optional[dict],
    integer_tol: float,
    feas_tol: float,
    evaluator: NLPEvaluator,
    time_limit: float,
    max_nodes: int,
    gap_tolerance: float,
) -> Optional[tuple[np.ndarray, float]]:
    """Scalable local branching via a bounded sub-MIP (Fischetti–Lodi 2003).

    Adds the Hamming-distance cut

        ``sum_{j: xbar_j=0} x_j + sum_{j: xbar_j=1} (1 - x_j) <= k``

    over the binary variables as a single linear constraint, then re-solves the
    restricted problem with a SMALL budget. Unlike the enumeration variant in
    :func:`local_branching`, the cost is independent of the binary count, so this
    works for the ``graphpart`` family (108 binaries) where enumerating ``C(n,k)``
    flip sets is hopeless.

    The cut is appended to ``model._constraints`` and removed in a ``finally`` so
    the caller's model is left byte-for-byte unchanged. The sub-solve is launched
    with ``_lns_enabled=False`` so it can NEVER re-enter this LNS layer (recursion
    guard), and is bounded by ``time_limit`` / ``max_nodes``.

    Returns the best feasible ``(x, obj)`` strictly improving the incumbent's
    objective, else ``None``. Heuristic only: the returned point is re-verified
    integer- and constraint-feasible and the dual bound is never touched.
    """
    import discopt.modeling as dm

    x_inc = np.asarray(x_incumbent, dtype=np.float64)
    incumbent_bits = {i: float(np.round(x_inc[i])) for i in binary_idx}

    # Assemble the symbolic Hamming-distance expression over the binaries.
    terms = []
    for i in binary_idx:
        term = _binary_slot_term(model, i)
        if incumbent_bits[i] >= 0.5:
            terms.append(1 - term)
        else:
            terms.append(term)
    cut = dm.sum(terms) <= float(k)

    inc_obj = float(evaluator.evaluate_objective(x_inc))

    n_constraints_before = len(model._constraints)
    model._constraints.append(cut)
    try:
        from discopt.solver import solve_model

        result = solve_model(
            model,
            time_limit=max(0.0, float(time_limit)),
            gap_tolerance=float(gap_tolerance),
            max_nodes=int(max_nodes),
            # Seed the sub-solve at the incumbent so it starts feasible for the cut.
            initial_point=x_inc.copy(),
            # CRITICAL recursion guard: the sub-solve must not re-enter this layer.
            _lns_enabled=False,
        )
    except Exception:
        return None
    finally:
        # Restore the model exactly: drop the appended cut (append-then-pop).
        del model._constraints[n_constraints_before:]

    x_dict = getattr(result, "x", None)
    if not isinstance(x_dict, dict):
        return None
    # SolveResult.x is keyed by variable name; flatten back to the model's flat
    # variable order to match the incumbent / evaluator layout.
    chunks: list[np.ndarray] = []
    for v in model._variables:
        if v.name not in x_dict:
            return None
        chunks.append(np.asarray(x_dict[v.name], dtype=np.float64).reshape(-1))
    x_out = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    if x_out.shape[0] != x_inc.shape[0]:
        return None

    cand = _finalize_candidate(evaluator, x_out, _get_integer_mask(model), integer_tol, feas_tol)
    if cand is None:
        return None
    _, obj_cand = cand
    # Strict improvement only — never propose a non-improving incumbent.
    if not np.isfinite(obj_cand) or obj_cand >= inc_obj - 1e-9:
        return None
    return cand


# Fallback estimate (seconds) for one enumeration sub-NLP before any round has
# been measured. The profiled sub-NLP mean on the 12-binary flay/fac class is
# ~14 ms (bottleneck-profile-2026-07-05 §1.1); 15 ms is the conservative prior
# used to predict a round's cost when no measurement exists yet.
_LB_SUBNLP_PRIOR_S = 0.015

# Minimum remaining budget (seconds) before it is worth dispatching the truncated
# neighbourhood to the bounded sub-MIP. A nested ``solve_model`` re-pays a fixed
# setup/JIT/root tax measured at ~1.6-2 s on this class (F4 territory); launching
# it with less than this is nearly all tax and no search, so below the threshold
# we truncate the enumeration outright rather than blow the slice on startup cost.
_LB_SUBMIP_MIN_BUDGET_S = 2.5


def local_branching(
    model: Model,
    x_incumbent: np.ndarray,
    *,
    k: int = 2,
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    integer_tol: float = 1e-5,
    feas_tol: float = 1e-6,
    evaluator: Optional[NLPEvaluator] = None,
    max_binaries: int = 12,
    submip_time_limit: float = 2.0,
    submip_max_nodes: int = 1000,
    submip_gap_tolerance: float = 1e-4,
    deadline: Optional[float] = None,
    node_bound: Optional[float] = None,
    incumbent_obj: Optional[float] = None,
    gap_tolerance: float = 1e-4,
) -> Optional[tuple[np.ndarray, float]]:
    """Local branching: search the Hamming-radius-``k`` neighbourhood of a binary
    incumbent for a better feasible point.

    Classic local branching adds the constraint ``sum_{j: x*_j=0} x_j +
    sum_{j: x*_j=1}(1 - x_j) <= k`` and re-solves a sub-MIP. For up to
    ``max_binaries`` binaries we realise the same neighbourhood directly by
    enumeration: every flip of up to ``k`` binaries is fixed and the continuous
    sub-NLP (via :func:`subnlp`) is solved for each — exact and self-contained.

    For MORE than ``max_binaries`` binaries (e.g. the ``graphpart`` family's 108)
    the enumeration is hopeless, so we dispatch to :func:`_local_branching_submip`,
    which adds the Hamming cut as a single linear constraint and re-solves the
    restricted problem with a bounded budget (with a recursion guard so the
    sub-solve never re-enters the LNS layer).

    Budget enforcement (F1, bottleneck-profile-2026-07-05 §1.1). The enumeration
    branch issues ``sum_r C(n_bin, r<=k)`` sub-NLPs — 79 at k=2, 1586 at k=5 for
    12 binaries — which historically ignored its ``submip_time_limit`` slice and
    the solver's absolute ``deadline`` entirely (fac2: 1665 sub-NLPs = 84 % of
    wall; flay03m: 3330 = 96 %). It now:

    1. honours a hard absolute ``deadline`` in addition to the per-call slice,
       polling before every sub-NLP (~14 ms each — polling is free);
    2. predicts each radius round's cost as ``C(n_bin, r) x measured_mean`` and,
       when the round cannot fit the remaining budget, truncates the enumeration
       rather than blowing past it — dispatching the *unexplored* neighbourhood
       to the bounded :func:`_local_branching_submip` so the search is not simply
       abandoned; and
    3. skips the whole search when the incumbent already matches the node
       relaxation ``node_bound`` within ``gap_tolerance`` (nothing to improve).

    This is budget enforcement only: the neighbourhood, the k-schedule policy,
    and the soundness of every proposed point are unchanged. Only proposes
    incumbents — the dual bound is untouched.

    Returns the best feasible ``(x, obj)`` found in the neighbourhood, or
    ``None``.
    """
    int_mask = _get_integer_mask(model)
    lb0, ub0 = _get_variable_bounds(model)
    binary_idx = [
        int(i) for i in np.nonzero(int_mask)[0] if lb0[i] >= -1e-9 and ub0[i] <= 1.0 + 1e-9
    ]
    if not binary_idx:
        return None

    k = max(1, min(k, len(binary_idx)))

    # (3) Nothing to improve: the incumbent already sits at the node relaxation
    # bound within tolerance, so no point in the Hamming ball can beat it. Skip
    # the whole search (heuristic-only; the dual bound is untouched either way).
    if (
        node_bound is not None
        and incumbent_obj is not None
        and np.isfinite(node_bound)
        and np.isfinite(incumbent_obj)
    ):
        abs_gap = incumbent_obj - float(node_bound)
        denom = max(abs(incumbent_obj), abs(float(node_bound)), 1e-10)
        if abs_gap <= 1e-9 or abs_gap / denom <= gap_tolerance:
            return None

    # Absolute wall past which no further sub-NLP may start. The effective budget
    # is the tighter of the caller's per-call slice and the solver's deadline.
    slice_deadline = time.perf_counter() + max(0.0, float(submip_time_limit))
    if deadline is not None and np.isfinite(deadline):
        effective_deadline = min(slice_deadline, float(deadline))
    else:
        effective_deadline = slice_deadline

    # Scalable sub-MIP variant for large binary blocks.
    if len(binary_idx) > max_binaries:
        if evaluator is None:
            evaluator = cached_evaluator(model)
        remaining = max(0.0, effective_deadline - time.perf_counter())
        return _local_branching_submip(
            model,
            x_incumbent,
            binary_idx,
            k=k,
            backend=backend,
            nlp_options=nlp_options,
            integer_tol=integer_tol,
            feas_tol=feas_tol,
            evaluator=evaluator,
            time_limit=min(float(submip_time_limit), remaining),
            max_nodes=submip_max_nodes,
            gap_tolerance=submip_gap_tolerance,
        )

    if evaluator is None:
        evaluator = cached_evaluator(model)

    x_inc = np.asarray(x_incumbent, dtype=np.float64)
    incumbent_bits = {i: float(np.round(x_inc[i])) for i in binary_idx}
    k = max(1, min(k, len(binary_idx)))

    best: Optional[tuple[np.ndarray, float]] = None
    # Rolling mean sub-NLP wall used to predict the next round's cost. Seeded with
    # the profiled prior; refined from every measured sub-NLP.
    mean_subnlp_s = _LB_SUBNLP_PRIOR_S
    n_measured = 0
    # Highest radius whose full enumeration we could afford. If the budget runs
    # out mid-schedule we hand the *unexplored* radii to the bounded sub-MIP so
    # the neighbourhood is still searched, just not by brute force.
    truncated_at: Optional[int] = None

    # Enumerate flip sets of size 0..k (size 0 re-evaluates the incumbent itself).
    for radius in range(k + 1):
        # (2) Predict this round's cost and stop enumerating if it cannot fit the
        # remaining budget. C(n, 0)=1 (re-evaluate incumbent) is always cheap and
        # always worth doing; larger radii are gated.
        remaining = effective_deadline - time.perf_counter()
        if remaining <= 0.0:
            truncated_at = radius
            break
        round_calls = math.comb(len(binary_idx), radius)
        predicted = round_calls * mean_subnlp_s
        if radius >= 1 and predicted > remaining:
            # Cannot afford the full round; hand the rest to the bounded sub-MIP.
            truncated_at = radius
            break

        for flip in itertools.combinations(binary_idx, radius):
            # (1) Poll the deadline before every sub-NLP (they are ~14 ms, so
            # per-iteration polling is free). Never start one past the budget.
            if time.perf_counter() >= effective_deadline:
                truncated_at = radius
                break
            seed = x_inc.copy()
            for i in binary_idx:
                seed[i] = incumbent_bits[i]
            for i in flip:
                seed[i] = 1.0 - incumbent_bits[i]
            _t0 = time.perf_counter()
            found = subnlp(
                model,
                seed,
                backend=backend,
                nlp_options=nlp_options,
                evaluator=evaluator,
                integer_tol=integer_tol,
                feas_tol=feas_tol,
            )
            # Refine the rolling mean from the measured sub-NLP wall.
            n_measured += 1
            mean_subnlp_s += (time.perf_counter() - _t0 - mean_subnlp_s) / n_measured
            if found is not None and (best is None or found[1] < best[1]):
                best = found
        else:
            # Inner loop completed without a budget break; continue the schedule.
            continue
        # Inner loop broke on the deadline: stop the whole enumeration.
        truncated_at = radius
        break

    # If the budget cut the enumeration short, search the unexplored Hamming
    # ball (radius >= truncated_at, up to k) via the bounded sub-MIP, which adds
    # the Hamming cut as one linear constraint instead of enumerating C(n, r)
    # flips. This keeps the neighbourhood covered without blowing the deadline.
    if truncated_at is not None and truncated_at <= k:
        remaining = effective_deadline - time.perf_counter()
        if remaining >= _LB_SUBMIP_MIN_BUDGET_S:
            submip = _local_branching_submip(
                model,
                x_inc,
                binary_idx,
                k=k,
                backend=backend,
                nlp_options=nlp_options,
                integer_tol=integer_tol,
                feas_tol=feas_tol,
                evaluator=evaluator,
                time_limit=min(float(submip_time_limit), remaining),
                max_nodes=submip_max_nodes,
                gap_tolerance=submip_gap_tolerance,
            )
            if submip is not None and (best is None or submip[1] < best[1]):
                best = submip
    return best
