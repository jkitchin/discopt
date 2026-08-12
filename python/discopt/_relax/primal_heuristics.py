"""
Multi-start primal heuristics for MINLP.

Finds good feasible solutions by launching NLP solves from diverse starting
points. Includes a feasibility pump that rounds fractional integer variables
and re-solves the resulting NLP.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from discopt._tape_nlp_evaluator import make_evaluator as cached_evaluator
from discopt._work_budget import EVAL, NLP_SOLVE, WorkBudget

if TYPE_CHECKING:  # pragma: no cover - typing only
    # #75: annotation only. ``cached_evaluator`` above is now the backend
    # dispatcher, which keeps the jax import inside its fallback -- importing
    # ``_relax.nlp_evaluator`` here put JAX on every solve regardless of backend.
    from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model, VarType

# #843: the QUBO/Ising local-search primal moved to ``discopt.qubo_primal`` when
# it graduated default-ON — the seed's gate AND search must be JAX-free (they run
# on every solve now), and this module's import pulls the JAX evaluator stack.
# Re-exported here because callers/docs reference ``primal_heuristics.is_qubo`` /
# ``qubo_local_search``; ``solve_model`` imports the JAX-free module directly.
from discopt.qubo_primal import is_qubo, qubo_local_search  # noqa: F401
from discopt.solvers import NLPResult, SolveStatus, pounce_incumbent_options

logger = logging.getLogger(__name__)


def _now() -> float:
    """The single point where this module reads the wall clock (#950).

    Every deadline in these heuristics — the caller's absolute ``deadline`` and
    each heuristic's own per-call slice — is stamped and polled through this
    function, and every :class:`WorkBudget` built here is handed it as its
    ``clock``. In production it is :func:`time.perf_counter` and nothing else;
    the indirection is a *seam*, so a test of deadline-edge behaviour can pin
    the schedule it means to test instead of racing the machine.

    Why it is needed: a heuristic that stamps ``deadline = perf_counter() +
    slice`` and then polls it decides how much of the search runs from wall
    time, so a stall *outside* the process — an xdist worker descheduled on a
    loaded CI runner — changes what the code under test does. That is exactly
    how ``test_deadline_expiring_mid_round_truncates`` failed as
    ``assert 0 >= 2``: a 300 ms stall inside local branching's 250 ms slice
    retired the whole budget before the first sub-NLP, which is *correct*
    production behaviour (#912: never start a solve past the deadline) and
    indistinguishable from broken truncation logic. Tests monkeypatch this
    symbol; production code must never pass a clock of its own. A caller's
    absolute ``deadline`` argument is measured against this clock, so a test
    that replaces it must supply any ``deadline`` in the same time base (or,
    like the test above, none at all).
    """
    return time.perf_counter()


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


def _heuristic_nlp_options(caller_options: Optional[dict] = None) -> dict:
    """Base NLP options for a sub-solve in this module (#945).

    Every NLP solve here exists to produce a candidate **incumbent**, so every one
    of them takes :func:`discopt.solvers.pounce_incumbent_options`: the returned
    point has to lie inside the box the model declared, or the incumbent it becomes
    is not a solution of the model the user wrote. Ipopt's default
    ``bound_relax_factor`` relaxes every bound — including the slack bounds standing
    in for inequality rows — by ``1e-8*(1 + |bound|)``, and a squared row takes the
    square root of that: ``(x-3)^2 <= 0`` admits every ``x`` within 1e-4 of 3.
    Measured on the MindtPy constraint-qualification fixture, whose exact optimum is
    3.0, ``feasibility_pump`` returned ``x = 2.9999`` and the default ``m.solve()``
    path certified it ``optimal`` with ``gap = 0.0``; with this seed the same run
    returns 2.999999998 (``scratchpad/issue945/heuristic_incumbent_box.py``).

    :func:`discopt.solvers.pounce_option_defaults` is deliberately **not** seeded
    here. The two halves are separable and only this one is wanted: its
    ``constr_viol_tol = 1e-8`` costs a 31%-worse incumbent and a looser bound on
    nvs05, entirely on its own (see the measurement in ``nlp_pounce.solve_nlp``).

    Routed through one helper rather than spelled at each call site so a seventh
    heuristic cannot silently keep Ipopt's default — the #940 lesson, one level up.
    Caller options are merged *after* the seed, so an explicit request still wins.
    """
    opts = pounce_incumbent_options()
    if caller_options:
        opts.update(caller_options)
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", _HEURISTIC_NLP_MAX_ITER)
    return opts


# Wall-clock floor/cap for a single heuristic sub-NLP launched against a caller
# deadline (#966). ``solver.py`` already clamps the root relaxation exactly this
# way (``max(_DEADLINE_NODE_FLOOR_S, min(3.0, deadline - now))``); these give the
# primal heuristics the same contract, and ``continuous_multistart`` below has
# used the ``min(3.0, remaining)`` form since F4.
#
# WHY A CAP AND NOT JUST A POLL: the deadline polls in this module gate whether a
# solve *starts*, never how long it runs, so polling alone bounds the *number* of
# overshooting solves to one and says nothing about that one's duration. Measured
# on a 20 s budget, bchoco07/bchoco08/heatexch_gen3 ran 2.4–4.7 s past the
# deadline in every arm of the #966 panel (all three coupled flags ON and OFF
# alike, within ~0.5 s of each other — i.e. a default-configuration defect, not a
# flag effect), and a post-deadline stack sampler put 100 % of its 73 samples
# inside one ``nlp_pounce.solve_nlp`` call: 91.8 % reached from
# ``feasibility_pump``, 8.2 % from ``integer_local_search``. An ITERATION cap
# (``_HEURISTIC_NLP_MAX_ITER``) is not a TIME cap — on the no-relaxation
# flowsheet class a single IPM iteration carrying an exact Hessian is itself
# seconds long.
#
# NOT A HARD GUARANTEE: POUNCE tests its own ``max_wall_time`` between IPM
# iterations, so one expensive iteration still overruns it (``diving`` below and
# ``test_f4_root_budget_gate`` both record the same observation). The cap turns
# an UNBOUNDED overrun into a roughly one-iteration one; entry gating (F4) and
# this duration cap are complementary, not alternatives.
_DEADLINE_NLP_FLOOR_S = 0.1
_DEADLINE_NLP_CAP_S = 3.0


def _deadline_wall_cap(deadline: Optional[float]) -> Optional[float]:
    """Wall-clock cap for one heuristic sub-NLP, derived from ``deadline``.

    Returns ``None`` when there is no finite deadline, so a caller that never
    passed one keeps its existing unclamped behaviour bit-for-bit. The floor
    keeps a just-expired deadline from handing the backend a zero/negative
    ``max_wall_time`` (whose meaning is backend-defined) — a solve started at the
    edge gets a small positive slice rather than an undefined one.
    """
    if deadline is None or not np.isfinite(deadline):
        return None
    return max(_DEADLINE_NLP_FLOOR_S, min(_DEADLINE_NLP_CAP_S, float(deadline) - _now()))


# VOLUME-1 (docs/dev/nlp-solve-volume-2026-07-06.md) + ILS-DEFAULT
# (docs/dev/ils-default-validation-2026-07-06.md): the objective-improvement
# coordinate descent inside ``integer_local_search`` (``_objective_improve``) is
# the single dominant NLP-solve SOURCE on the easy-panel instances BARON solves
# sub-second — nvs06 runs 888 sub-NLP solves there (of 911 total), nvs08 779,
# ex1224 217 — and its measured incumbent-improvement HIT RATE is 0 % on every
# one of them (the incumbent is already found by the root multistart's first
# start). Left uncapped, the descent keeps re-sweeping ``int_idx × {±1,±2}``
# until its wall deadline (~9 s), issuing hundreds of no-op sub-NLPs. The cap
# limits the number of sub-NLP solves a single descent may issue to
# ``ils_solve_cap × n_int`` (a small multiple of the integer dimension — enough
# for a full first-improvement sweep or two, which is where any real gain lands,
# but not the hundreds-of-solves plateau). The value lives on
# :class:`~discopt.solver_tuning.SolverTuning` (``ils_solve_cap``, **default 2 =
# ON** as of ILS-DEFAULT, broad-validated on a held-out integer sample), with the
# legacy ``DISCOPT_ILS_SOLVE_CAP`` env var as its default source; set it to 0 to
# restore the old UNCAPPED behavior (the debugging escape hatch). Read per-solve
# via ``solver_tuning.current()`` so it is per-``Model`` and thread-safe. Sound:
# capping this descent only ever *weakens* the incumbent it might find (the
# descent injects sub-NLP-verified points that B&B still re-verifies), and it
# never touches the dual bound or the certificate.


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


# Statuses ``feasibility_pump`` accepts from its fix-and-solve round.
#
# TIME_LIMIT is in the set *because* the pump now caps each projection from the
# caller's deadline (``_deadline_wall_cap``): a cap whose own time-limited points
# are then thrown away would trade the overshoot for a lost incumbent, which is
# not an improvement — it is a different regression. ``nlp_ipopt``'s status map
# turns Ipopt −5 ``Maximum_WallTime_Exceeded`` into exactly this status, so this
# is the status a clamped solve actually returns (verified, not assumed).
#
# Trusting a non-converged point here is sound ONLY because the pump re-verifies
# it independently of the status immediately below — ``_is_integer_feasible``
# after snapping the pinned integers, then ``_check_constraint_feasibility`` on
# the real constraint bodies — and ``inject_incumbent`` enforces strict
# improvement on top. That is the same footing on which ``subnlp`` and
# ``_solve_root_node_multistart`` already accept ITERATION_LIMIT. The status is
# a hint about convergence, never the feasibility evidence.
#
# Deliberately NOT folded into ``_is_nlp_feasible``: that gate is shared with
# call sites where no such re-verification follows, and widening it there would
# be weakening a check rather than relocating one.
_PUMP_ACCEPT_STATUSES = (SolveStatus.OPTIMAL, SolveStatus.TIME_LIMIT)


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

        opts = _heuristic_nlp_options(ipopt_options)
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
        deadline: Optional ``time.perf_counter()`` wall-clock deadline. Bounds the
            pump two ways: the extra perturbation rounds stop once it has passed,
            AND each round's NLP solve is capped by ``_deadline_wall_cap`` so a
            projection already under way cannot run past it unbounded (#966 — the
            poll alone left a 2.4–4.7 s overrun on a 20 s budget). Returns the
            best feasible solution found so far (or None). Keeps a tight global
            ``time_limit`` from being overrun by the root heuristic's NLP solves.
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

    opts = _heuristic_nlp_options(ipopt_options)
    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        backend = get_nlp_solver("auto")

    rng = np.random.default_rng(42)

    for round_idx in range(max_rounds):
        # Always run the first round (a feasible incumbent is the primary goal,
        # worth a small overrun); only the *extra* perturbation rounds are
        # deadline-gated so the pump cannot loop well past a tight ``time_limit``.
        if deadline is not None and round_idx > 0 and _now() >= deadline:
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
            solve_opts = dict(opts)
            _wall_cap = _deadline_wall_cap(deadline)
            if _wall_cap is not None:
                # Bound THIS solve, not merely the gap between rounds. The poll at
                # the top of the loop cannot stop a projection that has already
                # started, and round 0 is not polled at all — which is how a pump
                # launched just under the deadline ran seconds past it (#966). An
                # explicit caller ``max_wall_time`` still wins (setdefault).
                solve_opts.setdefault("max_wall_time", _wall_cap)
            try:
                nlp_result = backend(evaluator, x0, options=solve_opts)
            except BaseException as exc:
                # Some NLP backends (pounce via PyO3) raise PanicException, which
                # is not a subclass of Exception; treat any failure as this round
                # producing no point and perturb on the next round.
                logger.debug("fix-and-solve NLP round failed: %s: %s", type(exc).__name__, exc)
                continue
        finally:
            for v, (lb_v, ub_v) in zip(model._variables, saved_bounds):
                v.lb = lb_v
                v.ub = ub_v

        if nlp_result.status not in _PUMP_ACCEPT_STATUSES or nlp_result.x is None:
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

        opts = _heuristic_nlp_options(nlp_options)
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


def continuous_multistart(
    model: Model,
    n_starts: int = 32,
    seed: int = 42,
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    evaluator: Optional[NLPEvaluator] = None,
    deadline: Optional[float] = None,
    feas_tol: float = 1e-6,
    incumbent_obj: Optional[float] = None,
) -> Optional[tuple[np.ndarray, float]]:
    """Stratified continuous multistart for pure-continuous nonconvex models.

    Basin diversification for the one model class the primal-heuristic suite
    otherwise leaves bare (issue #188): with no integers to round, flip or dive
    on, ``feasibility_pump``/``integer_local_search``/``fractional_diving``/
    RINS/RENS all no-op, and on the McCormick-LP spatial path the node NLPs
    warm-start from the parent point — so every local solve stays locked in the
    first basin the relaxation vertex happens to fall into
    (kall_congruentcircles_c51: parks at the two-row packing 1.5371 while the
    single-row global basin at 1.0730 is reachable by 3/32 stratified starts,
    every seed tried).

    Draws ``n_starts`` stratified samples over the variable box (the
    :func:`_generate_starts` Latin-hypercube-style sampler MultiStartNLP uses)
    and runs one local NLP from each, keeping the best *constraint-verified*
    point. Deadline-gated between starts; each solve is individually capped so
    one pathological start cannot eat the remaining budget.

    Scope guard: returns ``None`` untried when the model has any integer
    variable — integer models have a full heuristic arsenal already, and a
    relaxed-integer local optimum is useless as an incumbent there.

    Sound (heuristic-policy regime, CLAUDE.md §5): a primal-side finder only.
    Every returned point is re-verified with ``_check_constraint_feasibility``
    and the caller's ``inject_incumbent`` enforces strict improvement, so the
    dual bound and the certificate math are untouched by construction.

    Args:
        model: The optimization model (must be pure-continuous to act).
        n_starts: Number of stratified starting points.
        seed: RNG seed for the stratified sampler (fixed for determinism).
        backend: ``solve_nlp(evaluator, x0, options=...)`` callable. If None,
            uses :func:`discopt.solvers.nlp_backend.get_nlp_solver('auto')`.
        nlp_options: Options dict forwarded to the NLP backend.
        evaluator: Pre-built NLPEvaluator; one is constructed if omitted.
        deadline: Absolute ``time.perf_counter()`` deadline. Starts are skipped
            once it has passed; the loop never begins a solve it cannot fit.
        feas_tol: Constraint-feasibility tolerance for accepting a point.
        incumbent_obj: Current incumbent objective, if any. Purely
            informational for early exit bookkeeping — points are kept only if
            they beat it, but the caller's injection still re-checks.

    Returns:
        ``(x, obj)`` for the best constraint-feasible local optimum found that
        beats ``incumbent_obj`` (when given), else ``None``.
    """
    int_mask = _get_integer_mask(model)
    if np.any(int_mask):
        return None
    if n_starts <= 0:
        return None

    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        backend = get_nlp_solver("auto")
    if evaluator is None:
        evaluator = cached_evaluator(model)

    lb, ub = _get_variable_bounds(model)
    rng = np.random.default_rng(seed)
    starts = _generate_starts(lb, ub, n_starts, rng)

    opts = _heuristic_nlp_options(nlp_options)

    best_obj = float("inf") if incumbent_obj is None else float(incumbent_obj)
    best_x: Optional[np.ndarray] = None

    for i in range(n_starts):
        remaining = np.inf if deadline is None else deadline - _now()
        if remaining <= 0.05:
            break
        solve_opts = dict(opts)
        if np.isfinite(remaining):
            # Cap the single solve so a stiff start cannot eat the whole budget,
            # while still letting a typical (sub-second) local solve converge.
            solve_opts.setdefault("max_wall_time", float(min(3.0, remaining)))
        try:
            res = backend(evaluator, starts[i], options=solve_opts)
        except BaseException as exc:
            # PyO3 backends can raise PanicException (not an Exception subclass).
            logger.debug("multistart NLP start %d failed: %s: %s", i, type(exc).__name__, exc)
            continue
        # Accept OPTIMAL or ITERATION_LIMIT — the point is independently
        # re-verified below, mirroring subnlp's acceptance set.
        if res.status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
            continue
        if res.x is None or res.objective is None:
            continue
        obj = float(res.objective)
        if not np.isfinite(obj) or obj >= best_obj:
            continue
        x_out = np.asarray(res.x, dtype=np.float64)
        if not _check_constraint_feasibility(evaluator, x_out, tol=feas_tol):
            continue
        # Keep the verified objective consistent with the evaluator.
        obj = float(evaluator.evaluate_objective(x_out))
        if obj < best_obj:
            best_obj = obj
            best_x = x_out.copy()

    if best_x is None:
        return None
    return best_x, best_obj


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
    eval_budget: Optional[int] = None,
    solve_budget: Optional[int] = None,
    deadline: Optional[float] = None,
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
    bounded by ``eval_budget``/``solve_budget`` (deterministic operation counts),
    ``max_restarts`` and ``max_steps``; 2-opt is skipped when the integer count
    exceeds ``pair_cap`` to avoid the O(n^2) neighbourhood blowing up on large
    models.

    **Determinism (issue #912).** How far this search gets used to be decided by
    a wall clock (``time_budget``), which made the incumbent it returns — and
    therefore the whole B&B tree below it — a function of machine speed: the
    descent routinely never converges, so on the measured cliff case ``gear2``
    closed in 3 nodes with a 5 s budget and 91 nodes with 3 s. The extent is now
    counted in deterministic operations (:mod:`discopt._work_budget`): whichever
    of the evaluation cap and the sub-NLP-solve cap is reached first ends the
    search. The two are counted separately because they differ in cost by four
    orders of magnitude and by a 27x-varying ratio, so no single currency prices
    both (see the module docstring). ``deadline`` remains a *backstop* for the
    caller's ``time_limit`` — it decides when to stop, never how much work to do.
    Setting both budgets to 0 (``DISCOPT_ILS_EVAL_BUDGET=0
    DISCOPT_ILS_SOLVE_BUDGET=0``) restores the legacy wall-clock gate on
    ``time_budget``.

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
        time_budget: Legacy wall-clock budget in seconds. Used **only** when both
            deterministic budgets are disabled (set to 0).
        eval_budget: Max constraint/objective evaluations for the whole call.
            ``None`` resolves it from ``SolverTuning.ils_eval_budget``.
        solve_budget: Max continuous-repair sub-NLP solves for the whole call.
            ``None`` resolves it from ``SolverTuning.ils_solve_budget``.
        deadline: Absolute ``time.perf_counter()`` timestamp of the caller's
            overall solve deadline. A backstop only — it stops the search when
            the user's ``time_limit`` is up, and never decides how much work a
            within-limit search does.
        feas_tol: Constraint feasibility tolerance.
        seed: RNG seed for reproducible perturbations.

    Returns:
        ``(x, obj)`` for the best feasible point found, else ``None``.
    """
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

    # Deterministic extent gate (issue #912). ``budget`` counts the model-level
    # operations this search issues; the caller's solve deadline rides along as
    # a backstop so the heuristic still honours ``time_limit``. With
    # both budgets 0 the old wall-clock extent gate is restored: an unlimited
    # counter gated on ``time_budget`` exactly as before, with one deliberate
    # difference — a caller-supplied solve deadline still applies (as the
    # earlier of the two), so the escape hatch cannot overrun ``time_limit`` the
    # way the pre-#912 path could. It is the *extent* gate that is restored, not
    # the overrun.
    if eval_budget is None or solve_budget is None:
        from discopt import solver_tuning as _st

        _tuning = _st.current()
        if eval_budget is None:
            eval_budget = _tuning.ils_eval_budget
        if solve_budget is None:
            solve_budget = _tuning.ils_solve_budget
    if int(eval_budget) > 0 or int(solve_budget) > 0:
        budget = WorkBudget(
            {EVAL: int(eval_budget), NLP_SOLVE: int(solve_budget)},
            deadline=deadline,
            clock=_now,
        )
    else:
        _wall = _now() + max(0.0, time_budget)
        budget = WorkBudget(
            None, deadline=_wall if deadline is None else min(_wall, deadline), clock=_now
        )

    def violation(x: np.ndarray) -> float:
        budget.charge(EVAL)
        g = np.asarray(evaluator.evaluate_constraints(x))
        return float(np.sum(np.maximum(0.0, cl - g)) + np.sum(np.maximum(0.0, g - cu)))

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
        # VOLUME-1 / ILS-DEFAULT sub-NLP solve cap (default ON, mult 2). Cap the
        # number of continuous-repair sub-NLP solves this descent may issue to
        # ``ils_solve_cap × n_int`` — a full first-improvement sweep or two —
        # instead of re-sweeping until the wall deadline. Only the *extra* no-op
        # solves past a couple of sweeps are cut (measured 0 % hit rate); the
        # descent still injects any better point it finds. ``ils_solve_cap=0``
        # restores the old uncapped behavior (escape hatch). Read per-solve.
        from discopt import solver_tuning as _st

        _ils_cap_mult = _st.current().ils_solve_cap
        _solve_cap = _ils_cap_mult * max(1, n_int) if _ils_cap_mult > 0 else None
        _solves_used = 0
        improved = True
        while improved and not budget.exhausted():
            improved = False
            for j in int_idx:
                for d in (-1.0, 1.0, -2.0, 2.0):
                    if budget.exhausted():
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
                        budget.charge(NLP_SOLVE)
                        cand = subnlp(
                            model,
                            xt,
                            backend=backend,
                            nlp_options=nlp_options,
                            evaluator=evaluator,
                            feas_tol=feas_tol,
                            # #966: cap the repair from whichever deadline the
                            # budget is actually enforcing. ``budget.exhausted()``
                            # is polled BETWEEN solves, so without this a descent
                            # step launched just under the wall runs past it.
                            time_budget=_deadline_wall_cap(budget.deadline),
                        )
                        if cand is None:
                            continue
                        cx, cobj = np.asarray(cand[0], dtype=np.float64), float(cand[1])
                    else:
                        if violation(xt) > feas_tol:
                            continue
                        budget.charge(EVAL)
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
        relax_opts = _heuristic_nlp_options(nlp_options)
        _relax_cap = _deadline_wall_cap(budget.deadline)  # #966
        if _relax_cap is not None:
            relax_opts.setdefault("max_wall_time", _relax_cap)
        budget.charge(NLP_SOLVE)
        relax_res = backend(evaluator, mid, options=relax_opts)
        if relax_res is not None and relax_res.x is not None:
            seeds.append(_round_clip(np.asarray(relax_res.x)))
    except BaseException as exc:
        # Backend may panic (pounce/PyO3); fall back to the caller's seed alone.
        logger.debug("relaxation restart seed unavailable: %s: %s", type(exc).__name__, exc)

    rng = np.random.default_rng(seed)
    best: Optional[tuple[np.ndarray, float]] = None
    n_seeds = len(seeds)

    for restart in range(max(eff_restarts, n_seeds)):
        if budget.exhausted():
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
            if cur <= feas_tol or budget.exhausted():
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
            if best_move is None and use_2opt and not budget.exhausted():
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
        budget.charge(NLP_SOLVE)
        repaired = subnlp(
            model,
            xc,
            backend=backend,
            nlp_options=nlp_options,
            evaluator=evaluator,
            feas_tol=feas_tol,
            time_budget=_deadline_wall_cap(budget.deadline),  # #966
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

    # Rule 6/#912: report what actually decided the extent of this search.
    # ``stopped_on="deadline"`` means the machine's speed cut it short, so the
    # incumbent it returns (and the tree below it) is NOT reproducible; the
    # determinism panel reads this line.
    logger.debug(
        "integer_local_search: evals=%d solves=%d limits=%s stopped_on=%s seeds=%d",
        budget.spent(EVAL),
        budget.spent(NLP_SOLVE),
        budget.limits,
        budget.stopped_on,
        n_seeds,
    )
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
    solve_budget: Optional[int] = None,
    deadline: Optional[float] = None,
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

    **Determinism (issue #912).** Which cells get enumerated is bounded by
    ``solve_budget`` (one sub-NLP per cell), not by a wall clock, so the search
    stops at the same cell on every machine. ``deadline`` is the caller's
    ``time_limit`` backstop. ``solve_budget=0`` restores the legacy
    ``time_budget`` wall gate.
    """
    # As in ``one_hot_swap_search``: a non-positive ``time_budget`` is the caller
    # saying there is no budget at all, and #912's deterministic cell budget must
    # not override that.
    if float(time_budget) <= 0.0:
        return None
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

    # #912: the enumeration's extent is a deterministic sub-NLP count, not a wall
    # budget. One solve per cell and the cell list is already capped by
    # ``max_combos``, so this only ever trims a large box — but it trims it at the
    # same cell on every machine. ``deadline`` (the caller's ``time_limit``) stays
    # as a backstop. Both budgets off => the legacy ``time_budget`` wall gate.
    if solve_budget is None:
        from discopt import solver_tuning as _st

        solve_budget = max(1, round(_BOX_BUDGET_RATIO * _st.current().ils_solve_budget))
    if int(solve_budget) > 0:
        budget = WorkBudget({NLP_SOLVE: int(solve_budget)}, deadline=deadline, clock=_now)
    else:
        _wall = _now() + max(0.0, time_budget)
        budget = WorkBudget(
            None, deadline=_wall if deadline is None else min(_wall, deadline), clock=_now
        )
    best: Optional[tuple[np.ndarray, float]] = None
    for combo in combos:
        if budget.exhausted():
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
        budget.charge(NLP_SOLVE)
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


def _residual_assignments(
    residual: list[int], x_relax: np.ndarray, limit: int
) -> list[tuple[float, ...]]:
    """Candidate 0/1 fixings for binaries outside every one-hot group.

    Ordered by how much information backs them: three *anchors* first — the
    relaxation's own nearest rounding, then the two homogeneous fixings (a big-M
    indicator's "off" and "on" values) — and then the rest by increasing **Hamming
    distance from an anchor**, one distance wave at a time. Deterministic in every
    branch: a primal heuristic that varies run to run makes a node-count comparison
    unreproducible.

    The distance ordering is what makes the list usable, not just complete. Raw
    ``itertools.product`` order is a *binary counter*, which has nothing to do with
    plausibility: measured on cstr, every feasible residual assignment sat at index
    **30 or 31 of 32** — one bit away from all-ones — so a budget that stopped
    anywhere short of the full enumeration missed all of them, while under this
    ordering those same fixings land in the first wave after the anchors. The
    anchors are the informative points on a big-M GDP (an uncovered indicator is
    usually at a bound), so "near an anchor" is where to spend the budget first.

    Always returns at least one assignment (the empty tuple when there is no
    residual), so the caller's cross product is never empty.
    """
    k = len(residual)
    if k == 0:
        return [()]

    nearest = tuple(float(np.clip(np.round(x_relax[j]), 0.0, 1.0)) for j in residual)
    anchors: list[tuple[float, ...]] = [nearest, (0.0,) * k, (1.0,) * k]
    out: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for cand in anchors:
        if cand not in seen:
            seen.add(cand)
            out.append(cand)

    # Wave d: everything exactly d flips from some anchor, anchors in order.
    for dist in range(1, k + 1):
        if len(out) >= limit:
            break
        for anchor in anchors:
            for flips in itertools.combinations(range(k), dist):
                if len(out) >= limit:
                    break
                bits = list(anchor)
                for j in flips:
                    bits[j] = 1.0 - bits[j]
                cand = tuple(bits)
                if cand not in seen:
                    seen.add(cand)
                    out.append(cand)
            if len(out) >= limit:
                break
    return out[:limit]


def one_hot_config_subnlp(
    model: Model,
    x_relax: np.ndarray,
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    evaluator: Optional[NLPEvaluator] = None,
    max_configs: int = 256,
    integer_tol: float = 1e-5,
    deadline: Optional[float] = None,
) -> list[tuple[np.ndarray, float]]:
    """Root primal *constructor* for disjunctive (GDP) models: pick one disjunct
    per disjunction, then solve the fixed-integer sub-NLP.

    Motivation (#823). On a big-M reformulated GDP the indicator binaries are
    partitioned by ``sum_k y_k == 1`` rows, one per disjunction. Two existing
    paths both decline this structure:

    * :func:`enumerate_binary_seeds_subnlp` enumerates *all* 0/1 assignments and
      so self-gates off above ``max_binaries`` (4). Real GDPs carry one indicator
      per disjunct — measured 6..138 binaries across the GDPlib small set — so the
      root disjunct cover never runs on this class at all.
    * Plain :func:`subnlp` rounds each binary independently to nearest. Independent
      rounding does not respect ``sum_k y_k == 1``: a disjunction whose relaxed
      indicators read ``(0.4, 0.35, 0.25)`` rounds to *all zeros*, and one reading
      ``(0.6, 0.55)`` rounds to *two ones*. Either way the fixed sub-NLP is
      infeasible on a constraint the model states outright, so the heuristic
      returns nothing and the node contributes no incumbent.

    This constructor instead selects, per group, the indicator with the largest
    relaxed value (a valid configuration *by construction*), fixes the rest of the
    group to 0, and solves one sub-NLP. It then tries a bounded number of
    alternates by flipping the least-confident groups — those with the smallest
    margin between their best and runner-up indicator — to their runner-up, since
    those are where the relaxation's preference carries the least information.

    Binaries outside every group are searched too, by
    :func:`_residual_assignments` — measured, that is what separates a model this
    helps from one it does not (see that function and the ``max_configs`` note).

    Cost is bounded to ``max_configs`` sub-NLP solves in total across both
    searches (not ``2**k``), so it scales to the 138-binary models the enumerator
    skips, and the whole path is additionally deadline-bounded — in practice the
    deadline, not ``max_configs``, is what stops it.

    ``max_configs`` is deliberately large. Measured on cstr, a fixed-integer
    sub-NLP here costs **0.018 s** (384 plans in 7 s), and the feasible plans sit
    at ``ci + ri`` = 17, 23 and 33 in the configuration x residual grid — so a
    32-solve budget cannot reach them under *any* ordering, and an earlier default
    of 32 returned nothing on a model where three feasible plans exist (one at the
    true optimum). A budget too small to reach the answer is not a cheap heuristic,
    it is a heuristic that does not work.

    Soundness: every returned point comes from :func:`subnlp`, which fixes the
    integers and then verifies both integer- and constraint-feasibility before
    returning. This proposes incumbents only; it never touches the dual bound, so
    it cannot weaken the optimality certificate. Binaries outside any detected
    group keep their nearest-rounded value from *x_relax* (subnlp's own rule).

    Args:
        model: The optimization model.
        x_relax: Relaxation point at the node (drives the per-group argmax).
        backend: ``solve_nlp(evaluator, x0, options=...)`` callable.
        nlp_options: Options forwarded to the NLP backend.
        evaluator: Pre-built evaluator; one is constructed if omitted.
        max_configs: Maximum number of plans (sub-NLP solves) to try, across
            both the configuration and residual searches combined.
        integer_tol: Integrality tolerance forwarded to :func:`subnlp`.
        deadline: Absolute ``perf_counter`` deadline; the loop stops before
            starting a configuration that cannot finish inside it.

    Returns:
        Every feasible ``(x, obj)`` found (possibly empty). The caller injects
        each as an incumbent candidate, so this is agnostic to objective sense.
    """
    int_mask = _get_integer_mask(model)
    if not np.any(int_mask):
        return []

    lb, ub = _get_variable_bounds(model)
    n_vars = int(int_mask.size)
    x_relax = np.asarray(x_relax, dtype=np.float64)

    groups = _scan_one_hot_rows(model, int_mask, n_vars)
    if not groups:
        return []

    # Per group: the relaxation's preference order, and how strongly it prefers.
    ranked: list[list[int]] = []
    margins: list[float] = []
    for g in groups:
        order = sorted(g, key=lambda j: -float(x_relax[j]))
        ranked.append(order)
        top = float(x_relax[order[0]])
        second = float(x_relax[order[1]]) if len(order) > 1 else 0.0
        margins.append(top - second)

    # Least-confident groups first: those are the flips most likely to matter.
    flip_order = sorted(range(len(groups)), key=lambda gi: margins[gi])

    # Configurations, by increasing number of demotions from the pure argmax and,
    # within a wave, least-confident groups first. Same principle as the residual
    # ordering: spend the budget near the informative anchor, outward.
    #
    # This has to be a distance enumeration, not a ladder. A cumulative chain
    # (config k demotes the first k groups of ``flip_order``) reaches |groups|+1 of
    # the 2**|groups| configurations, and never demotes group g alone unless g is
    # first in that order. Measured on cstr against the real in-solve relaxation
    # point, an exhaustive scan of the rank-0/rank-1 space found 6 feasible
    # configurations — best 3.0620146, the true optimum — at multi-flip positions
    # (configs 1, 17, 32, 33, 48, 49 of 64). Not one of them lies on the chain.
    flippable = [gi for gi in flip_order if len(ranked[gi]) >= 2]
    configs: list[dict[int, int]] = []
    for dist in range(len(flippable) + 1):
        if len(configs) >= max_configs:
            break
        for combo in itertools.combinations(flippable, dist):
            if len(configs) >= max_configs:
                break
            configs.append({gi: 1 for gi in combo})

    # Binaries that sit outside EVERY disjunction still have to be fixed, and
    # nearest-rounding them can be invalid on its own. Measured on the GDPlib
    # small set this is decisive, not a detail: valid disjunct selection alone
    # gives a feasible sub-NLP on batch_processing (one-hot rows cover 138/138
    # binaries) but not on cstr (15/20) — and on cstr, searching the 5 uncovered
    # binaries turns "no point at all" into a feasible 3.13020 against a true
    # optimum of 3.06201. So the residual gets its own bounded search.
    covered = {j for g in groups for j in g}
    residual = [
        j
        for j in np.nonzero(int_mask)[0].tolist()
        if j not in covered and lb[j] >= -1e-9 and ub[j] <= 1.0 + 1e-9
    ]
    residual_assigns = _residual_assignments(residual, x_relax, max_configs)

    # A zero-continuous start: inactive disaggregated variables begin at their
    # feasible 0, leaving only the active disjunct to solve. Same reasoning as
    # ``enumerate_binary_seeds_subnlp`` — a relaxation point carries the settled
    # disjunct's values, an ill-conditioned start for any other configuration.
    cont = ~int_mask
    zero_start = x_relax.copy()
    zero_start[cont] = np.clip(0.0, lb, ub)[cont]

    # Interleave the two searches DIAGONALLY, so neither can starve the other.
    # Concatenating them (all residual variants, then all group demotions) does not
    # interleave: it merely orders. Measured on cstr — 6 groups, 5 uncovered
    # binaries — the residual enumeration is 2**5 = 32 assignments, which exactly
    # fills ``max_configs``, so every slot went to a residual variant of
    # configuration 0 and NO alternative disjunct configuration was ever tried. With
    # the relaxation's argmax configuration infeasible there (0/32 feasible), the
    # search could not reach the demoted configuration that IS feasible at 3.0620146
    # — the true optimum — and returned nothing.
    #
    # Enumerating by increasing ``ci + ri`` keeps the most-informed fixing first
    # ((0,0) = argmax configuration with the relaxation's own residual rounding) and
    # thereafter spends the budget on both axes, whatever their relative sizes.
    plans: list[tuple[int, int]] = []
    for total in range(len(configs) + len(residual_assigns) - 1):
        for ci in range(min(total, len(configs) - 1), -1, -1):
            ri = total - ci
            if ri < len(residual_assigns):
                plans.append((ci, ri))
        if len(plans) >= max_configs:
            break

    results: list[tuple[np.ndarray, float]] = []
    attempted = 0
    stop = "exhausted"
    for ci, ri in plans[:max_configs]:
        if deadline is not None and _now() >= deadline:
            stop = "deadline"
            break
        seed = zero_start.copy()
        choice = configs[ci]
        for gi, g in enumerate(groups):
            pick = ranked[gi][choice.get(gi, 0)]
            for j in g:
                seed[j] = 0.0
            seed[pick] = 1.0
        for j, v in zip(residual, residual_assigns[ri]):
            seed[j] = v
        budget = None if deadline is None else max(0.0, deadline - _now())
        if budget is not None and budget <= 0.0:
            stop = "deadline"
            break
        attempted += 1
        found = subnlp(
            model,
            seed,
            backend=backend,
            nlp_options=nlp_options,
            evaluator=evaluator,
            integer_tol=integer_tol,
            time_budget=budget,
        )
        if found is not None:
            results.append(found)
    # An empty return has several distinct causes — no sub-NLP was even started,
    # the deadline cut the search short, or every fixing was genuinely infeasible.
    # Reporting the count and the stop reason is what makes those distinguishable
    # after the fact; without it "found nothing" is unfalsifiable (CLAUDE.md §6).
    logger.debug(
        "one_hot_config_subnlp: %d group(s), %d residual binaries, %d/%d sub-NLP(s) "
        "attempted (%s), %d feasible",
        len(groups),
        len(residual),
        attempted,
        len(plans[:max_configs]),
        stop,
        len(results),
    )
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

    opts = _heuristic_nlp_options(nlp_options)

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
            if deadline is not None and _now() >= deadline:
                return None
            dive_opts = dict(opts)
            _dive_cap = _deadline_wall_cap(deadline)
            if _dive_cap is not None:
                # The poll above bounds how many dive steps START, not how long
                # the one that started runs — which is why the overrun this
                # comment describes survived the poll (#966). Cap the step too.
                dive_opts.setdefault("max_wall_time", _dive_cap)
            try:
                res = backend(evaluator, x_cur, options=dive_opts)
            except BaseException:
                return None
            # A capped step that hits TIME_LIMIT ends the dive, exactly as the
            # poll above would have one step later. Sound either way: diving is a
            # primal heuristic and never touches the dual bound.
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


# #912: each converted primal heuristic gets a share of the root ILS budget in
# proportion to the wall slice it used to be given, so the deterministic budgets
# preserve the *relative* effort the tuned wall budgets encoded. ILS had 5 s,
# ``integer_box_search`` 4 s, ``local_branching`` a 2 s sub-MIP slice, and
# ``one_hot_swap_search`` 1 s. Handing all four the ILS number instead was
# measured (interleaved, 3 rounds/arm) at 1.38x wall on syn05hfsg and 1.18x on
# fac2 for identical node counts — sound but harmful, which under CLAUDE.md §5
# does not ship.
_BOX_BUDGET_RATIO = 0.8  # 4 s / 5 s
_LB_BUDGET_RATIO = 0.4  # 2 s / 5 s
_SWAP_BUDGET_RATIO = 0.2  # 1 s / 5 s

# (#912 removed ``_LB_SUBNLP_PRIOR_S``, the ~15 ms prior for one enumeration
# sub-NLP. It existed only to convert a round's *solve count* into a predicted
# wall time so the round could be compared against a remaining wall budget. Local
# branching now compares the solve count against a remaining *solve* budget, so
# there is nothing left to convert and no measured mean to drift with the
# machine. Deleted rather than left dangling — CLAUDE.md §3, no dead constants.)

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
    solve_budget: Optional[int] = None,
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
        # Deliberately NOT the shared certification tolerance from
        # ``solvers._gap`` (1e-6), despite looking like the same test. This gate
        # decides whether to SKIP a heuristic, so its safe direction is the
        # opposite one: a floor tighter than the certificate tolerance runs the
        # search more often, which costs time and can only ever find a better
        # incumbent. Widening it here to 1e-6 for consistency would skip more and
        # is a search-behaviour change owing its own net-positive panel (#945).
        abs_gap = incumbent_obj - float(node_bound)
        denom = max(abs(incumbent_obj), abs(float(node_bound)), 1e-10)
        if abs_gap <= 1e-9 or abs_gap / denom <= gap_tolerance:
            return None

    # Absolute wall past which no further sub-NLP may start. The effective budget
    # is the tighter of the caller's per-call slice and the solver's deadline.
    slice_deadline = _now() + max(0.0, float(submip_time_limit))
    if deadline is not None and np.isfinite(deadline):
        effective_deadline = min(slice_deadline, float(deadline))
    else:
        effective_deadline = slice_deadline

    # Scalable sub-MIP variant for large binary blocks.
    if len(binary_idx) > max_binaries:
        if evaluator is None:
            evaluator = cached_evaluator(model)
        remaining = max(0.0, effective_deadline - _now())
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
    # #912: the enumeration's extent is a deterministic sub-NLP count.
    #
    # This is the site where the wall clock did the most damage. The round-cost
    # prediction used to be ``C(n, r) x mean_subnlp_s`` against ``deadline - now``
    # — a *measured* mean wall time against a *measured* remaining wall — so which
    # radius the enumeration reached, and therefore which neighbourhood was
    # searched by brute force versus handed to the sub-MIP, was decided by how
    # fast the machine happened to be running at that moment. The same quantity
    # in solve counts (``C(n, r)`` against the remaining sub-NLP budget) answers
    # the same question — "can I afford this round?" — as a function of the model
    # alone. ``deadline`` stays as the ``time_limit`` backstop.
    if solve_budget is None:
        from discopt import solver_tuning as _st

        solve_budget = max(1, round(_LB_BUDGET_RATIO * _st.current().ils_solve_budget))
    if int(solve_budget) > 0:
        budget = WorkBudget({NLP_SOLVE: int(solve_budget)}, deadline=effective_deadline, clock=_now)
    else:
        budget = WorkBudget(None, deadline=effective_deadline, clock=_now)
    # Highest radius whose full enumeration we could afford. If the budget runs
    # out mid-schedule we hand the *unexplored* radii to the bounded sub-MIP so
    # the neighbourhood is still searched, just not by brute force.
    truncated_at: Optional[int] = None

    # Enumerate flip sets of size 0..k (size 0 re-evaluates the incumbent itself).
    for radius in range(k + 1):
        # (2) Cost this round in sub-NLP solves and stop enumerating if it cannot
        # fit the remaining budget. C(n, 0)=1 (re-evaluate incumbent) is always
        # cheap and always worth doing; larger radii are gated.
        if budget.exhausted():
            truncated_at = radius
            break
        round_calls = math.comb(len(binary_idx), radius)
        affordable = budget.remaining(NLP_SOLVE)
        if radius >= 1 and affordable is not None and round_calls > affordable:
            # Cannot afford the full round; hand the rest to the bounded sub-MIP.
            truncated_at = radius
            break

        for flip in itertools.combinations(binary_idx, radius):
            # (1) Poll the budget before every sub-NLP. Never start one past it.
            if budget.exhausted():
                truncated_at = radius
                break
            seed = x_inc.copy()
            for i in binary_idx:
                seed[i] = incumbent_bits[i]
            for i in flip:
                seed[i] = 1.0 - incumbent_bits[i]
            budget.charge(NLP_SOLVE)
            found = subnlp(
                model,
                seed,
                backend=backend,
                nlp_options=nlp_options,
                evaluator=evaluator,
                integer_tol=integer_tol,
                feas_tol=feas_tol,
            )
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
        remaining = effective_deadline - _now()
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


def _scan_one_hot_rows(model: Model, binary_mask: np.ndarray, n_vars: int) -> list[list[int]]:
    """Scan for disjoint ``sum(binaries) == 1`` rows, of any sizes.

    This is the raw structural scan shared by two consumers with *different*
    requirements:

    * :func:`_detect_one_hot_groups` (the #280 swap move) additionally demands at
      least two groups, all of equal size — the swap pairs members by sorted
      position, which is only well defined for a clean equal-size partition.
    * :func:`one_hot_config_subnlp` (disjunct selection) has no such need: picking
      one indicator per disjunction is well defined for groups of any sizes, and
      for a single group. Requiring equal sizes there would discard exactly the
      GDP models whose disjunctions have differing numbers of disjuncts.

    Returns the disjoint groups in scan order (each a sorted list of flat binary
    indices), or ``[]`` when the model has no such rows.
    """
    from discopt._relax.milp_relaxation import _linearize_affine_expr_sparse

    groups: list[list[int]] = []
    seen: set[int] = set()
    for c in model._constraints:
        if getattr(c, "sense", None) != "==":
            continue
        try:
            # Sparse: this scan touches EVERY constraint, and the dense
            # linearization costs O(n_vars) per row to allocate and zero — the
            # #875 shape (~460 s of root setup on a 106,711-var instance from the
            # identical pattern in ``_fix_single_var_equalities``).
            terms, const = _linearize_affine_expr_sparse(c.body, model, n_vars)
        except Exception as exc:  # noqa: BLE001 - a non-affine row is not a one-hot row
            logger.debug("one-hot row scan skipped a body: %s: %s", type(exc).__name__, exc)
            continue
        if not np.isfinite(const) or abs(float(const) + 1.0) > 1e-9:
            continue  # not ``... == 1``
        nz = sorted(j for j, v in terms.items() if abs(v) > 1e-9)
        if len(nz) < 2:
            continue
        if nz[0] < 0 or nz[-1] >= n_vars:
            continue  # out of the flat range the dense array bounded by raising
        if any(abs(terms[j] - 1.0) > 1e-9 for j in nz):
            continue  # non-unit coefficients — not a plain one-hot sum
        if nz[-1] >= binary_mask.size or not np.all(binary_mask[nz]):
            continue  # support is not entirely binary
        g = [int(i) for i in nz]
        if seen.intersection(g):
            continue  # overlapping groups: not a clean partition
        seen.update(g)
        groups.append(sorted(g))
    return groups


def _detect_one_hot_groups(model: Model, binary_mask: np.ndarray, n_vars: int) -> list[list[int]]:
    """Detect disjoint, equal-size one-hot groups (``sum of binaries == 1``).

    Scans the model's ``==`` constraints for the assignment / set-partition
    pattern ``sum_k x[i,k] == 1`` (each *item* i assigned to exactly one *slot*
    k). A row qualifies only when its affine form is ``sum(x_g) - 1`` with unit
    coefficients over an all-binary support; per-slot cardinality/balance rows
    (``sum == c`` with ``c != 1``) are naturally excluded (their constant is
    ``-c``). Overlapping or unequal-size groups are rejected — the swap move pairs
    members by sorted position, which is only well defined for a clean partition
    into equal-size groups.

    Returns the list of groups (each a sorted list of flat binary indices, one
    entry per slot), or ``[]`` when no such structure is present.
    """
    groups = _scan_one_hot_rows(model, binary_mask, n_vars)

    if len(groups) < 2:
        return []
    size = len(groups[0])
    if size < 2 or any(len(g) != size for g in groups):
        return []
    return groups


def one_hot_swap_search(
    model: Model,
    x_incumbent: np.ndarray,
    *,
    evaluator: Optional[NLPEvaluator] = None,
    integer_tol: float = 1e-5,
    feas_tol: float = 1e-6,
    max_restarts: int = 30,
    max_passes: int = 40,
    time_budget: float = 1.0,
    eval_budget: Optional[int] = None,
    deadline: Optional[float] = None,
    seed: int = 0,
) -> Optional[tuple[np.ndarray, float]]:
    """Assignment-aware *swap* local search for one-hot (set-partition) MIQPs.

    Many combinatorial MIQPs — graph partitioning, QAP, clustering, assignment —
    constrain disjoint binary groups to be one-hot (``sum_k x[i,k] == 1``): each
    item ``i`` occupies exactly one slot ``k``. On such models a single bit flip
    always breaks a one-hot row, so the generic constraint-violation search,
    RINS, and local-branching neighbourhoods make little progress and the solver
    settles for a *sound but poor* incumbent while the dual bound is already tight
    (issue #280). The feasibility-preserving move here is a **swap**: exchange the
    slots of two items. A swap leaves every one-hot row satisfied AND leaves every
    per-slot cardinality/balance row unchanged (each slot's count is preserved),
    so the search stays on the feasible manifold with no sub-solve at all.

    Greedy first-improving swap descent with perturbation restarts (a light
    Kernighan–Lin). Each candidate assignment is scored by the model
    :class:`NLPEvaluator` — the objective is never special-cased, so the move
    works for any objective over the one-hot structure. The best assignment found
    is re-verified integer- and constraint-feasible via :func:`_finalize_candidate`
    (this also catches any *other* constraint a swap might violate) and returned
    only on strict improvement.

    General (gated purely on detected one-hot structure, never on a problem name
    or shape — CLAUDE.md §2) and sound (heuristic-policy regime, CLAUDE.md §5:
    only a re-verified, strictly-improving incumbent is proposed; the dual bound
    and certificate are never touched). Returns ``(x, obj)`` or ``None``.
    """
    # A non-positive ``time_budget`` is the caller saying "there is no budget at
    # all", not "use the default": honour it in both arms. #912 replaced the wall
    # gate with an evaluation count, which would otherwise have quietly turned
    # ``time_budget=0`` into a full-effort search.
    if float(time_budget) <= 0.0:
        return None
    int_mask = _get_integer_mask(model)
    if not np.any(int_mask):
        return None
    lb0, ub0 = _get_variable_bounds(model)
    n_vars = int(int_mask.size)
    binary_mask = int_mask & (lb0 >= -1e-9) & (ub0 <= 1.0 + 1e-9)
    if not np.any(binary_mask):
        return None

    groups = _detect_one_hot_groups(model, binary_mask, n_vars)
    if not groups:
        return None

    if evaluator is None:
        evaluator = cached_evaluator(model)

    x_inc = np.asarray(x_incumbent, dtype=np.float64).copy()
    n_groups = len(groups)
    group_arr = np.asarray(groups, dtype=np.int64)  # (n_groups, group_size)

    # Decode the incumbent's active slot per group (the ~1 member).
    assign0 = np.empty(n_groups, dtype=np.int64)
    for gi in range(n_groups):
        assign0[gi] = int(np.argmax(x_inc[group_arr[gi]]))

    def _reconstruct(assign: np.ndarray) -> np.ndarray:
        x = x_inc.copy()
        x[group_arr.ravel()] = 0.0
        for gi in range(n_groups):
            x[int(group_arr[gi, int(assign[gi])])] = 1.0
        return x

    # #912: this descent is pure objective evaluation (no sub-solve), so its
    # extent is an evaluation count. A wall budget here made the swap sequence —
    # and therefore the incumbent — a function of machine speed. ``deadline`` (the
    # caller's ``time_limit``) remains as the backstop; ``eval_budget=0`` restores
    # the legacy ``time_budget`` wall gate.
    if eval_budget is None:
        from discopt import solver_tuning as _st

        eval_budget = max(1, round(_SWAP_BUDGET_RATIO * _st.current().ils_eval_budget))
    if int(eval_budget) > 0:
        budget = WorkBudget({EVAL: int(eval_budget)}, deadline=deadline, clock=_now)
    else:
        t_end = _now() + max(0.0, float(time_budget))
        if deadline is not None and np.isfinite(deadline):
            t_end = min(t_end, float(deadline))
        budget = WorkBudget(None, deadline=t_end, clock=_now)

    def _obj(assign: np.ndarray) -> float:
        budget.charge(EVAL)
        return float(evaluator.evaluate_objective(_reconstruct(assign)))

    inc_obj = _obj(assign0)  # incumbent's own objective on the reconstructed point

    def _expired() -> bool:
        return budget.exhausted()

    def _descend(assign: np.ndarray) -> tuple[np.ndarray, float]:
        """First-improving swap descent to a local minimum (budget-bounded)."""
        a = assign.copy()
        cur = _obj(a)
        for _ in range(max_passes):
            if _expired():
                break
            improved = False
            for gi in range(n_groups):
                if _expired():
                    break
                for gj in range(gi + 1, n_groups):
                    if a[gi] == a[gj]:
                        continue
                    a[gi], a[gj] = a[gj], a[gi]
                    o = _obj(a)
                    if o < cur - 1e-9:
                        cur = o
                        improved = True  # accept (first improvement)
                    else:
                        a[gi], a[gj] = a[gj], a[gi]  # revert
            if not improved:
                break
        return a, cur

    best_a, best_obj = _descend(assign0)

    rng = np.random.default_rng(seed)
    restarts = 0
    while restarts < max_restarts and not _expired():
        restarts += 1
        a = best_a.copy()
        # Perturb: a few random slot swaps between differing groups.
        for _ in range(int(rng.integers(1, 4))):
            gi, gj = (int(v) for v in rng.integers(0, n_groups, size=2))
            if a[gi] != a[gj]:
                a[gi], a[gj] = a[gj], a[gi]
        a, o = _descend(a)
        if o < best_obj - 1e-9:
            best_obj, best_a = o, a.copy()

    if not np.isfinite(best_obj) or best_obj >= inc_obj - 1e-9:
        return None

    cand = _finalize_candidate(evaluator, _reconstruct(best_a), int_mask, integer_tol, feas_tol)
    if cand is None:
        return None
    _, obj_cand = cand
    if not np.isfinite(obj_cand) or obj_cand >= inc_obj - 1e-9:
        return None
    return cand
