"""
Multi-start primal heuristics for MINLP.

Finds good feasible solutions by launching NLP solves from diverse starting
points. Includes a feasibility pump that rounds fractional integer variables
and re-solves the resulting NLP.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model, VarType
from discopt.solvers import NLPResult, SolveStatus


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
        evaluator = NLPEvaluator(model)
        lb, ub = evaluator.variable_bounds
        int_mask = _get_integer_mask(model)
        has_integers = np.any(int_mask)

        rng = np.random.default_rng(self._seed)
        starts = _generate_starts(lb, ub, self._n_starts, rng)

        opts = dict(ipopt_options) if ipopt_options else {}
        opts.setdefault("print_level", 0)
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
        evaluator = NLPEvaluator(model)

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)
    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        backend = get_nlp_solver("auto")

    rng = np.random.default_rng(42)

    for round_idx in range(max_rounds):
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

    Returns:
        ``(x, obj)`` if the heuristic produced a usable integer- and
        constraint-feasible point, else ``None``.
    """
    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        backend = get_nlp_solver("auto")

    if evaluator is None:
        evaluator = NLPEvaluator(model)

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
        evaluator = NLPEvaluator(model)
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

    def violation(x: np.ndarray) -> float:
        g = np.asarray(evaluator.evaluate_constraints(x))
        return float(np.sum(np.maximum(0.0, cl - g)) + np.sum(np.maximum(0.0, g - cu)))

    deadline = time.perf_counter() + max(0.0, time_budget)

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
        mid = np.clip(0.5 * (lb + ub), -1e3, 1e3)
        relax_opts = dict(nlp_options) if nlp_options else {}
        relax_opts.setdefault("print_level", 0)
        relax_res = backend(evaluator, mid, options=relax_opts)
        if relax_res is not None and relax_res.x is not None:
            seeds.append(_round_clip(np.asarray(relax_res.x)))
    except BaseException:
        # Backend may panic (pounce/PyO3); fall back to the caller's seed alone.
        pass

    rng = np.random.default_rng(seed)
    best: Optional[tuple[np.ndarray, float]] = None
    n_seeds = len(seeds)

    for restart in range(max(max_restarts, n_seeds)):
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
        for _ in range(max_steps):
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
            if best is None or obj_ok < best[1]:
                best = (np.asarray(x_ok).copy(), float(obj_ok))
            # Once feasible, later perturbed restarts dive from this point's
            # neighbourhood (see the restart-base selection above) to improve it.

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
