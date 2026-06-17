"""
Solver orchestrator: end-to-end Model.solve() via NLP-based spatial Branch & Bound.

Connects:
  - PyTreeManager (Rust B&B engine) for node management / branching / pruning
  - NLPEvaluator (JAX) for objective/gradient/Hessian/constraint/Jacobian
  - solve_nlp (cyipopt) for continuous relaxation solves at each node
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Callable, Optional, cast

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from discopt._jax.alphabb import estimate_alpha as _estimate_alpha_jax
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds
from discopt._rust import PyTreeManager
from discopt.constants import INFEASIBILITY_SENTINEL as _INFEASIBILITY_SENTINEL
from discopt.constants import SENTINEL_THRESHOLD as _SENTINEL_THRESHOLD
from discopt.constants import STARTING_POINT_CLIP as _SPC
from discopt.modeling.core import (
    Constraint,
    CustomCall,
    Model,
    SolveResult,
    VarType,
)
from discopt.solvers import SolveStatus
from discopt.solvers.nlp_ipopt import solve_nlp

logger = logging.getLogger(__name__)


def _branch_priority_integer_vars(model: Model) -> frozenset[int]:
    """Integer/binary flat indices that *gate* the model's nonlinear terms.

    Branching these before other integers drives the global dual bound up on
    problems whose relaxation bound is structurally 0 until a *set* of binaries
    is jointly fixed. The objective products are nonnegative and zeroable on the
    box boundary; a product's value is forced away from 0 only once the binaries
    that select/activate it are all pinned (ex1252's line-selection binaries
    ``x36/x37/x38``, tied to the product factors ``x18/x19/x20`` by the equality
    constraints ``x18=x36`` ...). Because the bound moves only on the *joint*
    fixing, no single-variable fractionality or pseudocost score points at these
    variables, so most-fractional branching wanders with the bound pinned at 0.

    The set is the integer variables that either (a) appear directly in a
    nonlinear term, or (b) are tied to a nonlinear-term variable by a
    two-variable linear *equality* constraint (``x_int = c·x_nl``). This is
    branching-order metadata only — it never enters a bound or feasibility test,
    so it cannot affect soundness.
    """
    from discopt._jax.term_classifier import _get_flat_index, classify_nonlinear_terms
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        IndexExpression,
        UnaryOp,
        Variable,
    )

    n = sum(v.size for v in model._variables)
    is_int = [False] * n
    fi = 0
    for v in model._variables:
        flag = v.var_type in (VarType.BINARY, VarType.INTEGER)
        for _ in range(v.size):
            if fi < n:
                is_int[fi] = flag
            fi += 1

    # Variables appearing in any lifted nonlinear term (objective or constraint).
    terms = classify_nonlinear_terms(model)
    nl: set[int] = set()
    for group in (terms.bilinear, terms.trilinear, terms.multilinear):
        for term in group or []:
            nl.update(int(x) for x in term)
    for term in terms.monomial or []:
        nl.add(int(term[0]))

    priority: set[int] = {j for j in nl if 0 <= j < n and is_int[j]}

    # Affine reduction of an expression to ``{idx: coeff}, const`` (None if the
    # expression is not affine). Used to spot two-variable equality linkages.
    def _affine(expr: Any, s: float = 1.0):
        if isinstance(expr, Constant):
            return ({}, s * float(expr.value))
        if isinstance(expr, (Variable, IndexExpression)):
            flat = _get_flat_index(expr, model)
            if flat is None:
                return None
            return ({int(flat): s}, 0.0)
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return _affine(expr.operand, -s)
        if isinstance(expr, BinaryOp):
            if expr.op in ("+", "-"):
                left = _affine(expr.left, s)
                right = _affine(expr.right, s if expr.op == "+" else -s)
                if left is None or right is None:
                    return None
                d = dict(left[0])
                for k, val in right[0].items():
                    d[k] = d.get(k, 0.0) + val
                return (d, left[1] + right[1])
            if expr.op == "*":
                if isinstance(expr.left, Constant):
                    return _affine(expr.right, s * float(expr.left.value))
                if isinstance(expr.right, Constant):
                    return _affine(expr.left, s * float(expr.right.value))
        return None

    for c in model._constraints:
        if getattr(c, "sense", None) != "==":
            continue
        reduced = _affine(c.body)
        if reduced is None:
            continue
        nz = [(k, v) for k, v in reduced[0].items() if abs(v) > 1e-12]
        if len(nz) != 2:
            continue
        (i, _), (j, _) = nz
        if is_int[i] and j in nl:
            priority.add(i)
        elif is_int[j] and i in nl:
            priority.add(j)

    return frozenset(priority)


def _select_priority_branch_var(
    solution: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    priority_vars: frozenset[int],
    tol: float = 1e-5,
) -> Optional[int]:
    """Most-fractional priority var that is still a viable branch at this node.

    Returns the flat index of the priority variable whose relaxation value is
    most fractional and that is not already pinned by the node box, or ``None``
    when no priority variable is branchable here (all fixed or integral — the
    standard selector then applies). A hint is honoured by the Rust tree only
    when the variable is fractional, matching this filter.
    """
    sol = np.asarray(solution)
    best: Optional[int] = None
    best_frac = tol
    for j in priority_vars:
        if j >= sol.shape[0] or node_ub[j] - node_lb[j] <= tol:
            continue
        val = float(sol[j])
        frac = val - math.floor(val)
        f = min(frac, 1.0 - frac)
        if f > best_frac:
            best_frac = f
            best = int(j)
    return best


# --- POUNCE batched-NLP node solver tuning (Phase A, discopt#97) ---
# The callback-based POUNCE batch path runs node NLPs through Python/JAX
# callbacks, so every objective/gradient/Jacobian/Hessian call re-acquires the
# GIL and the Python share of each solve serializes across Rayon workers. The
# batch therefore only amortizes its overhead on larger per-node problems.
# Benchmarks (14 cores): net loss at n_vars=6 (pooling: 8.4s serial vs 9.4s
# batch), ~4% win at n_vars=105 (facility: 109.6s vs 105.2s). Below this many
# variables we stay on the serial per-node path. (The real fix is a native-Rust
# node evaluator — Phase B — which removes the GIL bottleneck entirely.)
_POUNCE_BATCH_MIN_VARS = 50
# Floor on the per-node NLP wall-time budget. The B&B loop re-derives each
# node's budget from the live remaining time to the global deadline; this floor
# keeps a node that straddles the deadline from being handed a zero/negative
# budget. Nodes that start fully past the deadline are skipped (see the serial
# loop), so at most one node per batch can overrun by this floor.
_DEADLINE_NODE_FLOOR_S = 0.1
# Multistart runs 3 starts per nonconvex node (warm/midpoint/random) and keeps
# the best — better incumbents on nonconvex models, but ~3x the node-solve cost.
# Off by default; flip to opt in (or pass multistart=True to _solve_batch_pounce).
_POUNCE_BATCH_MULTISTART = False


class _AugmentedEvaluator:
    """Wraps an NLPEvaluator with additional linear cut constraints.

    When cuts are injected, the constraint function becomes:
        [original_constraints; A_cut @ x - b_cut]
    where each cut a^T x <= b becomes a^T x - b <= 0 (upper bounded by 0).
    For >= cuts, a^T x >= b becomes b - a^T x <= 0 (negated).
    """

    def __init__(self, evaluator, cut_pool):
        self._ev = evaluator
        self._cut_pool = cut_pool
        A, b, senses = cut_pool.to_constraint_arrays()
        self._n_cuts = A.shape[0]
        if self._n_cuts > 0:
            # Normalize: convert all cuts to <= form (a^T x - rhs <= 0)
            self._A = A.copy()
            self._b = b.copy()
            for k in range(self._n_cuts):
                if senses[k] == ">=":
                    self._A[k] = -self._A[k]
                    self._b[k] = -self._b[k]
                # "==" treated as <= (conservative)
        else:
            self._A = None
            self._b = None

    @property
    def n_constraints(self):
        return self._ev.n_constraints + self._n_cuts

    @property
    def n_variables(self):
        return self._ev.n_variables

    @property
    def variable_bounds(self):
        return self._ev.variable_bounds

    def evaluate_objective(self, x):
        return self._ev.evaluate_objective(x)

    def evaluate_gradient(self, x):
        return self._ev.evaluate_gradient(x)

    def evaluate_hessian(self, x):
        return self._ev.evaluate_hessian(x)

    def evaluate_constraints(self, x):
        orig = self._ev.evaluate_constraints(x)
        if self._n_cuts == 0:
            return orig
        cut_vals = self._A @ x - self._b
        return np.concatenate([orig, cut_vals])

    def evaluate_jacobian(self, x):
        orig = self._ev.evaluate_jacobian(x)
        if self._n_cuts == 0:
            return orig
        return np.vstack([orig, self._A])

    def evaluate_lagrangian_hessian(self, x, obj_factor, lambda_):
        # Cut constraints are linear so their Hessian contribution is zero
        m_orig = self._ev.n_constraints
        return self._ev.evaluate_lagrangian_hessian(x, obj_factor, lambda_[:m_orig])

    def get_augmented_constraint_bounds(self, original_bounds):
        """Return constraint bounds extended with cut bounds (all <= 0)."""
        if self._n_cuts == 0:
            return original_bounds
        if original_bounds is None:
            original_bounds = []
        cut_bounds = [(-1e20, 0.0)] * self._n_cuts
        return list(original_bounds) + cut_bounds

    def get_augmented_jax_bounds(self, g_l_jax, g_u_jax):
        """Return JAX constraint bound arrays extended with cut bounds."""
        import jax.numpy as jnp

        if self._n_cuts == 0:
            return g_l_jax, g_u_jax
        cut_gl = jnp.full(self._n_cuts, -1e20, dtype=jnp.float64)
        cut_gu = jnp.zeros(self._n_cuts, dtype=jnp.float64)
        if g_l_jax is not None:
            new_gl = jnp.concatenate([g_l_jax, cut_gl])
            new_gu = jnp.concatenate([g_u_jax, cut_gu])
        else:
            new_gl = cut_gl
            new_gu = cut_gu
        return new_gl, new_gu

    @property
    def _obj_fn(self):
        return self._ev._obj_fn

    @property
    def _cons_fn(self):
        if self._n_cuts == 0:
            return self._ev._cons_fn

        import jax.numpy as jnp

        orig_cons_fn = self._ev._cons_fn
        A_jax = jnp.array(self._A, dtype=jnp.float64)
        b_jax = jnp.array(self._b, dtype=jnp.float64)

        if orig_cons_fn is not None:

            def augmented_con(x):
                orig = orig_cons_fn(x)
                cut_vals = A_jax @ x - b_jax
                return jnp.concatenate([orig, cut_vals])
        else:

            def augmented_con(x):
                return A_jax @ x - b_jax

        return augmented_con


class _BoundOverrideEvaluator:
    """Proxy an evaluator while overriding the variable bounds exposed to backends."""

    def __init__(self, evaluator, lb: np.ndarray, ub: np.ndarray):
        self._evaluator = evaluator
        self._lb = np.asarray(lb, dtype=np.float64)
        self._ub = np.asarray(ub, dtype=np.float64)

    def __getattr__(self, name):
        return getattr(self._evaluator, name)

    @property
    def variable_bounds(self):
        return self._lb, self._ub


def _evaluator_fingerprint(model: Model) -> tuple:
    """Structural fingerprint of a model for evaluator-cache validity.

    Captures identity of the objective, constraints, variables, and parameters.
    Mutating ``Parameter.value`` does NOT change the fingerprint, so repeated
    solves that only rebind parameter values reuse the same JITed callables
    and hit the XLA cache.
    """
    return (
        id(model._objective),
        tuple(id(c) for c in model._constraints),
        tuple(id(v) for v in model._variables),
        tuple(id(p) for p in model._parameters),
        bool(getattr(model, "_gauss_newton_hessian", False)),
    )


def _make_evaluator(model: Model):
    """Create or reuse a cached NLPEvaluator for the model.

    The first call builds a fresh ``NLPEvaluator`` (which JITs obj/grad/hess/
    cons/jac/lag_hess). Subsequent calls return the same evaluator as long as
    the model's structural fingerprint is unchanged, so the underlying jit
    objects (and their XLA caches) are preserved across solves. Parameter
    value changes are threaded through at call time as a runtime pytree.
    """
    fingerprint = _evaluator_fingerprint(model)
    cached = getattr(model, "_nlp_evaluator_cache", None)
    if cached is not None:
        ev, cached_fp = cached
        if cached_fp == fingerprint:
            return ev
    ev = NLPEvaluator(model, gauss_newton=getattr(model, "_gauss_newton_hessian", False))
    model._nlp_evaluator_cache = (ev, fingerprint)
    return ev


def _estimate_alpha_fd(evaluator, lb, ub, n_samples=30):
    """Estimate alphaBB convexification parameters via finite-difference Hessians.

    Samples random points in [lb, ub], computes the FD Hessian at each,
    finds the most negative eigenvalue, and returns alpha = max(0, -lambda_min/2 * 1.5 + 1e-6).
    """
    n = len(lb)
    rng = np.random.RandomState(123)

    # Clip infinite bounds for sampling
    lb_clip = np.clip(lb, -1e4, 1e4)
    ub_clip = np.clip(ub, -1e4, 1e4)
    span = ub_clip - lb_clip
    # Avoid zero-width dimensions
    span = np.maximum(span, 1e-8)

    eps = 1e-6
    global_min_eig = 0.0

    for _ in range(n_samples):
        x = lb_clip + rng.uniform(size=n) * span
        # Central-difference Hessian
        hess = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                fpp = evaluator.evaluate_objective(x_pp)
                fpm = evaluator.evaluate_objective(x_pm)
                fmp = evaluator.evaluate_objective(x_mp)
                fmm = evaluator.evaluate_objective(x_mm)
                h = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
                hess[i, j] = h
                hess[j, i] = h
        eigs = np.linalg.eigvalsh(hess)
        global_min_eig = min(global_min_eig, float(eigs[0]))

    alpha_scalar = max(0.0, -global_min_eig / 2.0 * 1.5 + 1e-6)
    return np.full(n, alpha_scalar)


def _compute_alphabb_bound(evaluator, node_lb, node_ub, alpha):
    """Compute a valid lower bound by minimizing the alphaBB underestimator.

    L(x) = f(x) - sum_i alpha_i * (x_i - lb_i) * (ub_i - x_i)

    The perturbation term is non-negative for x in [node_lb, node_ub], so
    L(x) <= f(x) there and the minimum of L over the box is a valid lower
    bound on f. The minimization domain MUST be exactly the true node box and
    the perturbation MUST use the same lb/ub arrays as the optimizer bounds:
    evaluating L at any x OUTSIDE [node_lb, node_ub] makes the perturbation
    NEGATIVE, which flips L into an over-estimator and yields an invalid
    (too-high) "lower bound".

    Returns the minimum of L over [node_lb, node_ub], or -inf when the box is
    unbounded / so wide that alphaBB is numerically meaningless (in which case
    alphaBB abstains and the caller's interval / LP relaxation bounds stand).
    """
    node_lb = np.asarray(node_lb, dtype=np.float64)
    node_ub = np.asarray(node_ub, dtype=np.float64)

    # alphaBB is only valid on a finite box. Unbounded or huge dimensions make
    # the quadratic perturbation astronomically large and scipy's
    # finite-difference gradient unreliable, so abstain instead of risking an
    # invalid bound.
    #
    # NOTE: a previous implementation instead CLIPPED the optimizer domain to
    # [-1e4, 1e4] while leaving node_lb/node_ub unclipped in the perturbation.
    # On any node with a bound outside that range (e.g. a big-M ~1e19 on an
    # unbounded variable) the clip pushed the optimizer outside the true box,
    # turned the perturbation negative, and produced a spurious ~1e17 "lower
    # bound" that fathomed the optimum region and falsely certified suboptimal
    # incumbents as global. Optimizing over the true box keeps L <= f, so any
    # value returned here is a sound lower bound.
    _ALPHABB_BOX_LIMIT = 1e8
    if not (np.all(np.isfinite(node_lb)) and np.all(np.isfinite(node_ub))):
        return -np.inf
    if np.any(np.abs(node_lb) > _ALPHABB_BOX_LIMIT) or np.any(np.abs(node_ub) > _ALPHABB_BOX_LIMIT):
        return -np.inf
    if np.any(node_ub < node_lb):
        return -np.inf

    def underestimator(x):
        f_val = evaluator.evaluate_objective(x)
        perturbation = np.sum(alpha * (x - node_lb) * (node_ub - x))
        return f_val - perturbation

    # Multiple starting points for robustness, all strictly inside the box.
    width = node_ub - node_lb
    mid = 0.5 * (node_lb + node_ub)
    bounds = list(zip(node_lb, node_ub))

    best_val = np.inf
    for x0 in (mid, node_lb + 0.25 * width, node_lb + 0.75 * width):
        try:
            result = scipy_minimize(underestimator, x0, method="L-BFGS-B", bounds=bounds)
            if result.fun < best_val:
                best_val = result.fun
        except (ValueError, ArithmeticError, RuntimeError):
            continue

    return best_val if np.isfinite(best_val) else -np.inf


def _compute_interval_bound(model, node_lb, node_ub, negate):
    """Sound interval-arithmetic lower bound on the objective over a node box.

    Builds a ``{Variable: Interval}`` box from the flat node bounds and
    evaluates the objective expression with outward-rounded interval
    arithmetic. The lower endpoint of the resulting enclosure is always a
    valid (if loose) lower bound on ``f`` over the box; for a maximization
    model the internal minimization objective is ``-f``, so its valid lower
    bound is ``-hi``.

    Unlike the McCormick-NLP bound this is cheap and unconditional, so it lets
    every open nonconvex node carry a finite valid bound on every iteration
    (instead of ``-inf`` on iterations where the periodic McCormick-NLP solve
    is skipped). Returns ``-inf`` when the enclosure is not finite or
    evaluation fails — never an invalid (too-high) bound.
    """
    if model._objective is None:
        return -np.inf
    try:
        from discopt._jax.convexity.interval import Interval
        from discopt._jax.convexity.interval_eval import evaluate_interval

        box = {}
        offset = 0
        for v in model._variables:
            sz = v.size
            lo = np.asarray(node_lb[offset : offset + sz], dtype=np.float64).reshape(v.shape)
            hi = np.asarray(node_ub[offset : offset + sz], dtype=np.float64).reshape(v.shape)
            box[v] = Interval(lo, hi)
            offset += sz
        iv = evaluate_interval(model._objective.expression, model, box)
        lo = float(np.min(np.asarray(iv.lo)))
        hi = float(np.max(np.asarray(iv.hi)))
    except (ValueError, ArithmeticError, RuntimeError, TypeError, KeyError, IndexError) as e:
        logger.debug("interval objective bound failed: %s", e)
        return -np.inf
    bound = -hi if negate else lo
    return bound if np.isfinite(bound) else -np.inf


def _extract_variable_info(model: Model):
    """Extract flat variable bounds and integer variable group info from a model.

    Returns:
        n_vars: total number of scalar decision variables
        lb: flat lower bounds array
        ub: flat upper bounds array
        int_var_offsets: list of flat offsets for integer/binary variable groups
        int_var_sizes: list of sizes for integer/binary variable groups
    """
    lb_parts = []
    ub_parts = []
    int_var_offsets = []
    int_var_sizes = []
    offset = 0

    for v in model._variables:
        lb_parts.append(v.lb.flatten())
        ub_parts.append(v.ub.flatten())
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            int_var_offsets.append(offset)
            int_var_sizes.append(v.size)
        offset += v.size

    n_vars = offset
    lb = np.concatenate(lb_parts) if lb_parts else np.array([], dtype=np.float64)
    ub = np.concatenate(ub_parts) if ub_parts else np.array([], dtype=np.float64)

    return n_vars, lb, ub, int_var_offsets, int_var_sizes


def _check_lp_solution_feasibility(A_eq, b_eq, x_full, tol=1e-4):
    """Check that an LP/QP solution satisfies A_eq @ x = b_eq within tolerance.

    Returns True if the maximum absolute constraint residual is within *tol*.
    Used by MILP/MIQP B&B to reject LP/QP relaxation solutions where the IPM
    converged to a constraint-violating point.
    """
    if A_eq.shape[0] == 0:
        return True
    residual = np.asarray(A_eq) @ np.asarray(x_full) - np.asarray(b_eq)
    return float(np.max(np.abs(residual))) <= tol


def _check_constraint_feasibility(evaluator, x, cl_list, cu_list, tol=1e-4):
    """Return True if x satisfies all constraints within tolerance.

    Parameters
    ----------
    evaluator : NLPEvaluator or _AugmentedEvaluator
        Constraint evaluator with ``evaluate_constraints`` and ``n_constraints``.
    x : np.ndarray
        Candidate solution vector.
    cl_list : list[float]
        Lower bounds on constraints (use -1e20 for no lower bound).
    cu_list : list[float]
        Upper bounds on constraints (use 1e20 for no upper bound).
    tol : float
        Feasibility tolerance.

    Returns
    -------
    bool
        True if all constraints are satisfied within *tol*.
    """
    if evaluator.n_constraints == 0:
        return True
    cons = evaluator.evaluate_constraints(np.asarray(x, dtype=np.float64))
    cl = np.array(cl_list, dtype=np.float64)
    cu = np.array(cu_list, dtype=np.float64)
    # The evaluator may have more constraints than cl/cu (e.g., augmented with
    # cutting planes).  Only check the original constraints.
    n_check = min(len(cons), len(cl))
    if n_check == 0:
        return True
    max_viol = max(
        float(np.max(cons[:n_check] - cu[:n_check])), float(np.max(cl[:n_check] - cons[:n_check]))
    )
    return max_viol <= tol


def _is_integer_feasible_solution(x, int_offsets, int_sizes, tol=1e-5):
    """Return True if all discrete variables are integral within tolerance."""
    for off, sz in zip(int_offsets, int_sizes):
        for j in range(off, off + sz):
            xj = x[j]
            if not np.isfinite(xj) or abs(xj - round(xj)) > tol:
                return False
    return True


def _structural_linear_row_mask(model, sizes, m):
    """Boolean mask (length ``m``) of Jacobian rows whose body is structurally affine.

    The numeric Jacobian-difference test for linearity samples the constraint
    gradient at two interior points. A saturating nonlinearity such as
    ``max(c, f(x))`` whose two samples both land in the same flat piece reads as
    constant-gradient and is misclassified as linear; FBBT then extrapolates that
    local flat behaviour across the whole box and can exclude feasible regions
    (e.g. pinning a GDP big-M disjunct-output variable to its kink value, which
    fathoms the subtree holding the true optimum — issue #27a). Gating linear
    FBBT on a *structural* affineness check as well makes the classification
    rigorous: a row is treated as linear only when its source constraint body
    contains no nonlinear operator. Conservative — a structural false negative
    only forgoes a tightening, it can never cause a false prune.

    Returns ``None`` if the per-constraint row expansion cannot be aligned with
    ``m`` (caller then keeps the purely numeric classification).
    """
    from discopt._jax.gdp_reformulate import _is_linear

    flags: list[bool] = []
    k = 0
    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        n = int(sizes[k]) if sizes is not None and k < len(sizes) else 1
        flags.extend([bool(_is_linear(c.body))] * n)
        k += 1
    if len(flags) != m:
        return None
    return np.asarray(flags, dtype=bool)


def _cached_structural_linear_mask(evaluator, m):
    """Return the structural linear-row mask for ``evaluator``'s model, cached.

    The mask (a row is structurally affine iff its body has no nonlinear
    operator) depends only on the constraint bodies, which are invariant across
    B&B nodes. Recomputing it every node re-walks every constraint DAG, so it is
    memoized on the evaluator and keyed by the row count ``m`` to stay robust if
    the row layout ever differs. Returns ``None`` when the mask is unavailable or
    cannot be aligned to ``m`` rows, so callers fall back to the numeric
    two-point Jacobian classification.
    """
    cached = getattr(evaluator, "_structural_linear_mask_cache", None)
    if cached is not None and cached[0] == m:
        return cached[1]
    mask = None
    try:
        sizes = getattr(evaluator, "_constraint_flat_sizes", None)
        mask = _structural_linear_row_mask(evaluator._model, sizes, m)
    except Exception:
        mask = None
    try:
        evaluator._structural_linear_mask_cache = (m, mask)
    except Exception:
        pass
    return mask


def _tighten_node_bounds_with_status(evaluator, node_lb, node_ub, cl_list, cu_list, max_rounds=3):
    """Constraint-based bound tightening (FBBT) for a single B&B node.

    Uses the constraint Jacobian to propagate implied variable bounds.
    For linear constraints (e.g., x_i <= M * y_i with y_i fixed), this
    is exact and eliminates degenerate variable bounds that cause IPM
    convergence failures.

    Parameters
    ----------
    evaluator : NLPEvaluator
        Provides evaluate_constraints and evaluate_jacobian.
    node_lb, node_ub : np.ndarray
        Variable bounds at this node.
    cl_list, cu_list : list[float]
        Constraint bounds.
    max_rounds : int
        Maximum propagation rounds.

    Returns
    -------
    lb, ub : np.ndarray
        Tightened variable bounds.
    infeasible : bool
        True if nonlinear tightening proved the node infeasible.
    """
    lb = node_lb.copy()
    ub = node_ub.copy()

    if evaluator.n_constraints == 0 or not cl_list:
        return _apply_nonlinear_tightening_with_status(evaluator._model, lb, ub)

    n = len(lb)
    m = len(cl_list)
    cu = np.array(cu_list, dtype=np.float64)
    cl = np.array(cl_list, dtype=np.float64)

    # Detect which constraints are linear by checking if the Jacobian
    # changes between two distinct evaluation points.  FBBT via Jacobian
    # linearization is only sound for linear constraints; applying it to
    # nonlinear constraints (e.g. x^1.5) can over-tighten and exclude
    # feasible regions, causing false infeasibility (issue #6).
    try:
        # Clip first: unbounded vars give inf-(-inf)=NaN under unclipped subtract.
        lb_c = np.clip(lb, -_SPC, _SPC)
        ub_c = np.clip(ub, -_SPC, _SPC)
        pt_a = lb_c + 0.25 * (ub_c - lb_c)
        pt_b = lb_c + 0.75 * (ub_c - lb_c)
        J_a = evaluator.evaluate_jacobian(pt_a)
        J_b = evaluator.evaluate_jacobian(pt_b)
        is_linear = np.all(np.abs(J_a - J_b) < 1e-8, axis=1)  # (m,) bool
    except Exception:
        return _apply_nonlinear_tightening_with_status(evaluator._model, lb, ub)

    # The two-point Jacobian test is fooled by saturating nonlinearities
    # (max/abs/clamped exprs) when both samples fall in one locally-flat piece;
    # require structural affineness as well so FBBT never linearizes a nonlinear
    # body and over-tightens it (issue #27a).
    _structural = _cached_structural_linear_mask(evaluator, len(is_linear))
    if _structural is not None:
        is_linear = is_linear & _structural

    if not np.any(is_linear):
        return _apply_nonlinear_tightening_with_status(evaluator._model, lb, ub)

    for _ in range(max_rounds):
        changed = False

        tightened_lb, tightened_ub, infeasible = _apply_nonlinear_tightening_with_status(
            evaluator._model, lb, ub
        )
        if infeasible:
            return tightened_lb, tightened_ub, True
        if np.any(np.abs(tightened_lb - lb) > 1e-12) or np.any(np.abs(tightened_ub - ub) > 1e-12):
            lb = tightened_lb
            ub = tightened_ub
            changed = True

        # Evaluate Jacobian at midpoint of current bounds
        mid = np.clip(lb, -_SPC, _SPC)
        span = np.clip(ub, -_SPC, _SPC) - mid
        mid = mid + 0.5 * span
        try:
            J = evaluator.evaluate_jacobian(mid)  # (m, n)
            g = evaluator.evaluate_constraints(mid)  # (m,)
        except Exception:
            break

        for j in range(m):
            if not is_linear[j]:
                continue
            # For constraint g_j(x) <= cu_j:
            # Linear approx: g_j(mid) + J[j,:] @ (x - mid) <= cu_j
            # To find max x_i, set other vars to MINIMIZE g (most room):
            #   J[j,k] > 0 → use lb[k];  J[j,k] < 0 → use ub[k]
            if cu[j] < 1e19:
                for i in range(n):
                    if abs(J[j, i]) < 1e-12 or lb[i] == ub[i]:
                        continue
                    residual = cu[j] - g[j]
                    for k in range(n):
                        if k == i:
                            continue
                        if abs(J[j, k]) < 1e-12:
                            # Zero Jacobian entry contributes nothing; skip to
                            # avoid ``0 * inf = nan`` when var k is unbounded.
                            continue
                        if J[j, k] > 0:
                            residual -= J[j, k] * (lb[k] - mid[k])
                        else:
                            residual -= J[j, k] * (ub[k] - mid[k])
                    # J[j,i] * (x_i - mid_i) <= residual
                    if J[j, i] > 1e-12:
                        new_ub = mid[i] + residual / J[j, i]
                        if new_ub < ub[i] - 1e-10:
                            ub[i] = max(lb[i], new_ub)
                            changed = True
                    elif J[j, i] < -1e-12:
                        new_lb = mid[i] + residual / J[j, i]
                        if new_lb > lb[i] + 1e-10:
                            lb[i] = min(ub[i], new_lb)
                            changed = True

            # For constraint g_j(x) >= cl_j:
            # To find min x_i, set other vars to MAXIMIZE g (most room):
            #   J[j,k] > 0 → use ub[k];  J[j,k] < 0 → use lb[k]
            if cl[j] > -1e19:
                for i in range(n):
                    if abs(J[j, i]) < 1e-12 or lb[i] == ub[i]:
                        continue
                    residual = cl[j] - g[j]
                    for k in range(n):
                        if k == i:
                            continue
                        if abs(J[j, k]) < 1e-12:
                            # Zero Jacobian entry contributes nothing; skip to
                            # avoid ``0 * inf = nan`` when var k is unbounded.
                            continue
                        if J[j, k] > 0:
                            residual -= J[j, k] * (ub[k] - mid[k])
                        else:
                            residual -= J[j, k] * (lb[k] - mid[k])
                    # J[j,i] * (x_i - mid_i) >= residual
                    if J[j, i] > 1e-12:
                        new_lb = mid[i] + residual / J[j, i]
                        if new_lb > lb[i] + 1e-10:
                            lb[i] = min(ub[i], new_lb)
                            changed = True
                    elif J[j, i] < -1e-12:
                        new_ub = mid[i] + residual / J[j, i]
                        if new_ub < ub[i] - 1e-10:
                            ub[i] = max(lb[i], new_ub)
                            changed = True

        if not changed:
            break

    return lb, ub, False


def _tighten_node_bounds(evaluator, node_lb, node_ub, cl_list, cu_list, max_rounds=3):
    """Constraint-based bound tightening without the infeasibility status."""
    lb, ub, _infeasible = _tighten_node_bounds_with_status(
        evaluator, node_lb, node_ub, cl_list, cu_list, max_rounds=max_rounds
    )
    return lb, ub


def _apply_nonlinear_tightening_with_status(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply opportunistic nonlinear bound tightening without aborting node processing."""
    try:
        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(model, lb, ub)
    except Exception as exc:
        logger.debug("Skipping nonlinear tightening after error: %s", exc)
        return lb, ub, False

    if stats.infeasible:
        logger.debug("Nonlinear tightening proved infeasibility: %s", stats.infeasibility_reason)
        return lb, ub, True

    return tightened_lb, tightened_ub, False


def _apply_nonlinear_tightening(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply opportunistic nonlinear bound tightening without the infeasibility status."""
    tightened_lb, tightened_ub, _infeasible = _apply_nonlinear_tightening_with_status(model, lb, ub)
    return tightened_lb, tightened_ub


def _infer_constraint_bounds(model: Model, evaluator=None):
    """Infer (cl, cu) arrays from model constraint senses.

    The NLPEvaluator compiles constraints as `body - rhs`, so:
      - '<=' constraints: body - rhs <= 0 => cl = -inf, cu = 0
      - '==' constraints: body - rhs == 0 => cl = 0, cu = 0
      - '>=' constraints: body - rhs >= 0 => cl = 0, cu = inf

    If an ``evaluator`` is passed, its per-constraint flat sizes are used
    to expand each source Constraint's bounds to ``flat_size`` rows so
    vector-valued bodies (e.g. DAEBuilder's vectorized collocation
    residuals) line up with cyipopt's row count. Without an evaluator,
    each source Constraint contributes one row (legacy scalar behavior).
    """
    cl_list = []
    cu_list = []
    sizes = None
    if evaluator is not None and hasattr(evaluator, "_constraint_flat_sizes"):
        sizes = evaluator._constraint_flat_sizes

    k = 0
    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        if c.sense == "<=":
            lo, hi = -1e20, 0.0
        elif c.sense == "==":
            lo, hi = 0.0, 0.0
        elif c.sense == ">=":
            lo, hi = 0.0, 1e20
        else:
            raise ValueError(f"Unknown constraint sense: {c.sense}")
        n = int(sizes[k]) if sizes is not None else 1
        cl_list.extend([lo] * n)
        cu_list.extend([hi] * n)
        k += 1

    return cl_list, cu_list


def _gams_initial_seed(model, node_lb, node_ub):
    """Build a flat start vector from a model's GAMS-provided initial values.

    ``from_gams`` parses ``x.l`` assignments into ``model._gams_initial_values``
    (``var_name -> float`` for scalars, ``var_name -> {flat_idx: float}`` for
    indexed variables) but nothing in the solver consumed them, so a modeler's
    (often near-optimal) starting point was ignored -- discopt re-derived its
    primal solely from relaxation/midpoint seeds. For nonconvex models whose good
    basin is hard to reach from those generic seeds (prob07's true global at
    154990 sits in a basin only the published start lands in), this left a worse
    incumbent. Return the provided point, filled with the bound midpoint for any
    variable the file did not initialize and clipped into ``[node_lb, node_ub]``;
    return ``None`` when the model carries no initial values (the common case),
    so callers add no work for models without a start.
    """
    iv = getattr(model, "_gams_initial_values", None)
    if not iv:
        return None
    lb_c = np.clip(node_lb, -_SPC, _SPC)
    ub_c = np.clip(node_ub, -_SPC, _SPC)
    x0 = 0.5 * (lb_c + ub_c)
    offset = 0
    found = False
    for v in model._variables:
        sz = v.size
        entry = iv.get(v.name)
        if entry is not None:
            if isinstance(entry, dict):
                for flat_idx, val in entry.items():
                    if 0 <= int(flat_idx) < sz:
                        x0[offset + int(flat_idx)] = float(val)
                        found = True
            else:
                x0[offset : offset + sz] = float(entry)
                found = True
        offset += sz
    if not found:
        return None
    return np.clip(x0, node_lb, node_ub)


def _generate_starting_points(node_lb, node_ub, n_random=2):
    """Generate diverse starting points for multi-start NLP at root node."""
    lb_clipped = np.clip(node_lb, -_SPC, _SPC)
    ub_clipped = np.clip(node_ub, -_SPC, _SPC)
    span = ub_clipped - lb_clipped

    points = [
        0.5 * (lb_clipped + ub_clipped),  # midpoint
        lb_clipped + 0.25 * span,  # lower-quarter
        lb_clipped + 0.75 * span,  # upper-quarter
    ]

    rng = np.random.RandomState(42)
    for _ in range(n_random):
        points.append(lb_clipped + rng.uniform(size=lb_clipped.shape) * span)

    return points


def _solve_root_node_multistart(
    evaluator,
    node_lb,
    node_ub,
    constraint_bounds,
    options,
    nlp_solver,
    n_random=2,
    convex=False,
):
    """Solve root NLP relaxation from multiple starting points.

    On nonconvex problems, different starting points can converge to
    different local minima. Multi-start at the root increases the
    chance of finding the global optimum for the initial bound/incumbent.

    On *convex* problems the relaxation has a unique optimum, so any start that
    converges to a feasible KKT point has already found the global optimum and
    further starts only re-find it. With ``convex=True`` the search short-circuits
    on the first feasible OPTIMAL result (the midpoint start, tried first), while
    still falling through to additional starts when a start fails to converge or
    lands constraint-infeasible — so robustness is preserved and only the
    redundant convex re-solves are skipped.
    """
    starting_points = _generate_starting_points(node_lb, node_ub, n_random=n_random)

    cl = cu = None
    if constraint_bounds:
        cl = [b[0] for b in constraint_bounds]
        cu = [b[1] for b in constraint_bounds]

    best_result = None
    best_feasible = False
    best_obj = np.inf
    last_result = None

    for x0 in starting_points:
        nlp_result = _solve_node_nlp(
            evaluator,
            x0,
            node_lb,
            node_ub,
            constraint_bounds,
            options,
            nlp_solver=nlp_solver,
        )
        last_result = nlp_result
        if nlp_result.status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
            continue
        feasible = cl is None or _check_constraint_feasibility(evaluator, nlp_result.x, cl, cu)
        obj = nlp_result.objective
        # Prefer a constraint-FEASIBLE iterate over any infeasible one; within the
        # same feasibility class, prefer the lower objective. An interior-point
        # solver can hit ITERATION_LIMIT at a low-objective but constraint-
        # infeasible point (e.g. division constraints from an interior start).
        # Selecting that over a feasible higher-objective iterate would mark the
        # root constraint-infeasible and unsoundly fathom the whole tree as
        # "infeasible" on a feasible problem.
        if (
            best_result is None
            or (feasible and not best_feasible)
            or (feasible == best_feasible and obj < best_obj)
        ):
            best_result = nlp_result
            best_feasible = feasible
            best_obj = obj

        # Convex relaxation: a feasible KKT (OPTIMAL) point is the unique global
        # optimum, so the remaining starts would only re-converge to it. Stop.
        if convex and feasible and nlp_result.status == SolveStatus.OPTIMAL:
            break

    if best_result is not None:
        return best_result
    # All failed — return the last result
    return last_result


def _invoke_pre_import_callbacks(
    *,
    model,
    tree,
    t_start,
    result_ids,
    result_lbs,
    result_sols,
    result_feas,
    n_batch,
    int_offsets,
    int_sizes,
    n_vars,
    lazy_constraints,
    incumbent_callback,
    _cut_pool,
):
    """Check lazy constraints and incumbent callbacks before importing results.

    For each integer-feasible solution in the batch:
    1. Call ``lazy_constraints`` callback. If it returns cuts, add them to the
       cut pool and mark the node as infeasible (preventing it from becoming
       an incumbent). The cuts will tighten subsequent relaxations.
    2. Call ``incumbent_callback``. If it returns False, mark the node as
       infeasible.
    """
    from discopt._jax.cutting_planes import LinearCut
    from discopt.callbacks import CallbackContext, cut_result_to_dense

    incumbent_info = tree.incumbent()
    inc_obj = None
    if incumbent_info is not None:
        _, inc_obj = incumbent_info
        if inc_obj >= _SENTINEL_THRESHOLD:
            inc_obj = None

    stats = tree.stats()
    elapsed = time.perf_counter() - t_start

    for i in range(n_batch):
        if result_lbs[i] >= _SENTINEL_THRESHOLD:
            continue  # skip infeasible nodes

        # Check integrality
        sol_is_int_feas = True
        for off, sz in zip(int_offsets, int_sizes):
            for j in range(off, off + sz):
                if abs(result_sols[i, j] - round(result_sols[i, j])) > 1e-5:
                    sol_is_int_feas = False
                    break
            if not sol_is_int_feas:
                break

        if not sol_is_int_feas:
            continue

        ctx = CallbackContext(
            node_count=stats["total_nodes"],
            incumbent_obj=inc_obj,
            best_bound=stats.get("global_lower_bound", -np.inf),
            gap=stats.get("gap"),
            elapsed_time=elapsed,
            x_relaxation=result_sols[i].copy(),
            node_bound=float(result_lbs[i]),
        )

        # --- Lazy constraints ---
        if lazy_constraints is not None:
            try:
                cuts = lazy_constraints(ctx, model)
                if cuts:
                    for cut in cuts:
                        coeffs, rhs, sense = cut_result_to_dense(cut, model)
                        _cut_pool.add(LinearCut(coeffs=coeffs, rhs=rhs, sense=sense))
                    # Mark as infeasible so it does not become incumbent.
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    logger.info(
                        "Lazy constraint callback added %d cut(s) at node %d",
                        len(cuts),
                        int(result_ids[i]),
                    )
                    continue  # skip incumbent callback for cut-separated nodes
            except Exception as e:
                logger.warning("Lazy constraint callback raised an exception: %s", e)

        # --- Incumbent callback ---
        if incumbent_callback is not None:
            try:
                solution = _unpack_solution(model, result_sols[i])
                accept = incumbent_callback(ctx, model, solution)
                if accept is False:
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    logger.info(
                        "Incumbent callback rejected solution at node %d",
                        int(result_ids[i]),
                    )
            except Exception as e:
                logger.warning("Incumbent callback raised an exception: %s", e)


def _unpack_solution(model: Model, x_flat: np.ndarray):
    """Convert flat solution vector to {var_name: array} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        size = v.size
        val = x_flat[offset : offset + size]
        if v.shape == () or v.shape == (1,):
            result[v.name] = val.reshape(v.shape) if v.shape == () else val
        else:
            result[v.name] = val.reshape(v.shape)
        offset += size
    return result


def _unpack_constraint_duals(
    evaluator, mult_g: Optional[np.ndarray]
) -> Optional[dict[str, np.ndarray]]:
    """Slice a flat constraint-multiplier vector into a dict keyed by
    Constraint.name (or ``c{idx}`` when anonymous), using the evaluator's
    per-source-constraint flat sizes as the source of truth for layout.
    Returns ``None`` if the input is missing or empty.
    """
    if mult_g is None or len(mult_g) == 0:
        return None
    out: dict[str, np.ndarray] = {}
    offset = 0
    for idx, (c, sz) in enumerate(
        zip(evaluator._source_constraints, evaluator._constraint_flat_sizes)
    ):
        sz = int(sz)
        chunk = np.asarray(mult_g[offset : offset + sz], dtype=float)
        key = c.name if c.name else f"c{idx}"
        out[key] = chunk if sz > 1 else chunk.reshape(())
        offset += sz
    if offset != len(mult_g):
        return None
    return out


def _unpack_bound_duals(
    model: Model, mult_x: Optional[np.ndarray]
) -> Optional[dict[str, np.ndarray]]:
    """Slice a flat bound-multiplier vector into a dict keyed by Variable.name.
    Returns ``None`` if the input is missing or empty.
    """
    if mult_x is None or len(mult_x) == 0:
        return None
    out: dict[str, np.ndarray] = {}
    offset = 0
    for v in model._variables:
        size = v.size
        chunk = np.asarray(mult_x[offset : offset + size], dtype=float)
        if v.shape == () or v.shape == (1,):
            out[v.name] = chunk.reshape(v.shape) if v.shape == () else chunk
        else:
            out[v.name] = chunk.reshape(v.shape)
        offset += size
    if offset != len(mult_x):
        return None
    return out


def _strong_branch_lp(
    evaluator,
    solution: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    candidate_var_indices: np.ndarray,
    parent_lb: float,
    max_candidates: int = 5,
    time_limit: float = 1.0,
    prefer_pounce: bool = False,
) -> Optional[int]:
    """Perform strong branching via LP relaxations for unreliable candidates.

    For each candidate variable, solves two LP relaxations (down-branch and
    up-branch) and returns the variable index with the best product score.

    Uses the NLP evaluator's gradient at the current solution as LP objective
    (first-order Taylor approximation), with node bounds as variable bounds.

    Parameters
    ----------
    evaluator : NLPEvaluator or _AugmentedEvaluator
        Evaluator for gradient/constraint computation.
    solution : np.ndarray
        Current relaxation solution at this node.
    node_lb, node_ub : np.ndarray
        Variable bounds for this node.
    candidate_var_indices : np.ndarray
        Flat indices of candidate variables to evaluate.
    parent_lb : float
        Parent node's relaxation lower bound.
    max_candidates : int
        Maximum number of candidates to evaluate (most fractional first).
    time_limit : float
        Total time budget for all LP solves.

    Returns
    -------
    int or None
        Best variable index to branch on, or None if no valid candidate.
    """
    try:
        from discopt.solvers.lp_backend import get_lp_solver

        solve_lp = get_lp_solver(prefer_pounce)
    except ImportError:
        return None  # strong branching is optional; fall back to pseudocosts

    n_vars = len(solution)
    n_candidates = len(candidate_var_indices)
    if n_candidates == 0:
        return None

    # Limit candidates — prioritize those closest to 0.5 fractionality
    if n_candidates > max_candidates:
        fracs = np.array([solution[i] - np.floor(solution[i]) for i in candidate_var_indices])
        closeness_to_half = 0.5 - np.abs(fracs - 0.5)
        top_k = np.argsort(-closeness_to_half)[:max_candidates]
        candidate_var_indices = candidate_var_indices[top_k]

    # LP objective: gradient of the objective at the current solution.
    try:
        c = np.asarray(evaluator.evaluate_gradient(solution), dtype=np.float64).ravel()
    except Exception:
        return None

    # LP constraints from the evaluator's Jacobian (linearized).
    A_ub = None
    b_ub = None
    try:
        if evaluator.n_constraints > 0:
            g_vals = np.asarray(evaluator.evaluate_constraints(solution), dtype=np.float64).ravel()
            J = np.asarray(evaluator.evaluate_jacobian(solution), dtype=np.float64)
            if J.ndim == 1:
                J = J.reshape(1, -1)
            # Linearized constraints: J @ (x - x0) + g(x0) <= 0
            # => J @ x <= J @ x0 - g(x0)
            A_ub = J
            b_ub = J @ solution - g_vals
    except Exception:
        pass  # Proceed without constraints (just variable bounds)

    bounds_list = [(float(node_lb[j]), float(node_ub[j])) for j in range(n_vars)]

    best_var = None
    best_score = -np.inf
    t_start = time.perf_counter()
    per_solve_limit = max(0.05, time_limit / (2 * len(candidate_var_indices) + 1))

    for var_idx in candidate_var_indices:
        if time.perf_counter() - t_start > time_limit:
            break

        var_idx = int(var_idx)
        val = solution[var_idx]
        floor_val = np.floor(val)

        # Down branch: x_i <= floor(val)
        down_bounds = list(bounds_list)
        down_bounds[var_idx] = (down_bounds[var_idx][0], floor_val)
        try:
            down_result = solve_lp(
                c, A_ub=A_ub, b_ub=b_ub, bounds=down_bounds, time_limit=per_solve_limit
            )
            down_obj = down_result.objective
            down_lb = (
                float(down_obj)
                if down_result.status == SolveStatus.OPTIMAL and down_obj is not None
                else np.inf
            )
        except Exception:
            down_lb = np.inf

        # Up branch: x_i >= ceil(val)
        up_bounds = list(bounds_list)
        up_bounds[var_idx] = (floor_val + 1.0, up_bounds[var_idx][1])
        try:
            up_result = solve_lp(
                c, A_ub=A_ub, b_ub=b_ub, bounds=up_bounds, time_limit=per_solve_limit
            )
            up_obj = up_result.objective
            up_lb = (
                float(up_obj)
                if up_result.status == SolveStatus.OPTIMAL and up_obj is not None
                else np.inf
            )
        except Exception:
            up_lb = np.inf

        # Product score: improvement in each direction
        down_gain = max(0.0, down_lb - parent_lb) if np.isfinite(down_lb) else 1e6
        up_gain = max(0.0, up_lb - parent_lb) if np.isfinite(up_lb) else 1e6
        score = (1e-6 + down_gain) * (1e-6 + up_gain)

        if score > best_score:
            best_score = score
            best_var = var_idx

    return best_var


_BOUND_WARN_THRESHOLD = 1e15


def _optimal_relative_gap(objective: float) -> Optional[float]:
    """Return the relative gap for a certified optimum."""
    return None if abs(float(objective)) <= 1e-10 else 0.0


def _format_bad_bound_entries(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> list[str]:
    """Return human-readable entries for scalar variables with problematic bounds."""
    bad_vars: list[str] = []
    offset = 0
    for v in model._variables:
        lb_flat = np.asarray(flat_lb[offset : offset + v.size], dtype=np.float64)
        ub_flat = np.asarray(flat_ub[offset : offset + v.size], dtype=np.float64)
        for j in range(v.size):
            lo, hi = float(lb_flat[j]), float(ub_flat[j])
            if (
                not np.isfinite(lo)
                or not np.isfinite(hi)
                or abs(lo) > _BOUND_WARN_THRESHOLD
                or abs(hi) > _BOUND_WARN_THRESHOLD
            ):
                name = v.name if v.size == 1 else f"{v.name}[{j}]"
                bad_vars.append(f"{name} (lb={lo:.2g}, ub={hi:.2g})")
        offset += v.size
    return bad_vars


def _check_finite_bounds(model: Model) -> None:
    """Warn if any variable has very large or infinite declared bounds.

    Interior point methods use barrier terms that require reasonably sized
    bounds. Bounds beyond 1e15 cause numerical difficulties (NaN gradients,
    ill-conditioned KKT systems) and the solver silently produces NaN
    objectives or reports iteration_limit. Nonlinear tightening is consumed by
    some solver paths, but this warning remains conservative because not every
    path applies the tightened box to the actual NLP solve.
    """
    raw_lb, raw_ub = flat_variable_bounds(model)
    raw_bad_vars = _format_bad_bound_entries(model, raw_lb, raw_ub)
    if not raw_bad_vars:
        return

    tightening_note = ""
    try:
        _tightened_lb, _tightened_ub, bt_stats = tighten_nonlinear_bounds(model, raw_lb, raw_ub)
        if bt_stats.infeasible:
            logger.info(
                "Nonlinear tightening proved infeasibility before large-bound warning: %s",
                bt_stats.infeasibility_reason,
            )
            return
        if bt_stats.n_tightened > 0:
            tightening_note = (
                f" Nonlinear tightening can adjust {bt_stats.n_tightened} bounds"
                f" via {', '.join(bt_stats.applied_rules)}."
            )
    except Exception as exc:
        logger.debug("Skipping nonlinear tightening before large-bound warning: %s", exc)

    bad_vars = raw_bad_vars
    if bad_vars:
        import warnings

        warnings.warn(
            f"Variables with very large or infinite declared bounds: "
            f"{', '.join(bad_vars[:5])}. "
            f"{tightening_note} "
            f"NLP solvers may fail (NaN, iteration_limit) when bounds "
            f"exceed ~1e15. Add tighter explicit bounds, e.g. "
            f"m.continuous('x', lb=0, ub=1000).",
            stacklevel=3,
        )


def _detect_nonlinear_bound_infeasibility(model: Model) -> Optional[str]:
    """Return a nonlinear bound-tightening infeasibility proof when available."""
    flat_lb, flat_ub = flat_variable_bounds(model)
    try:
        _tightened_lb, _tightened_ub, stats = tighten_nonlinear_bounds(model, flat_lb, flat_ub)
    except Exception as exc:
        logger.debug("Skipping nonlinear infeasibility precheck after error: %s", exc)
        return None
    if stats.infeasible:
        return stats.infeasibility_reason or "nonlinear bound tightening proved infeasibility"
    return None


def _is_pure_continuous(model: Model) -> bool:
    """Check if model has no integer/binary variables."""
    return all(v.var_type == VarType.CONTINUOUS for v in model._variables)


def _model_contains_custom_call(model: Model) -> bool:
    """True if any objective/constraint body contains a ``CustomCall`` node.

    A ``CustomCall`` wraps an opaque AD-only user function (``dm.custom``) that
    the relaxation compiler, Rust presolve, and ``.nl`` export cannot reason
    about. The solver uses this to force the local NLP path and to refuse global
    branch-and-bound (see ``solve_model``). See issue #27b.
    """

    def _walk(expr) -> bool:
        if isinstance(expr, CustomCall):
            return True
        # Generic child traversal mirroring the DAG node fields used elsewhere
        # (e.g. discopt._jax.cutting_planes): BinaryOp/MatMul -> left/right,
        # UnaryOp/SumExpression -> operand, FunctionCall/CustomCall -> args,
        # SumOverExpression -> terms, IndexExpression -> base.
        left = getattr(expr, "left", None)
        if left is not None and _walk(left):
            return True
        right = getattr(expr, "right", None)
        if right is not None and _walk(right):
            return True
        operand = getattr(expr, "operand", None)
        if operand is not None and _walk(operand):
            return True
        base = getattr(expr, "base", None)
        if base is not None and _walk(base):
            return True
        for a in getattr(expr, "args", ()) or ():
            if _walk(a):
                return True
        for t in getattr(expr, "terms", ()) or ():
            if _walk(t):
                return True
        return False

    obj = getattr(model, "_objective", None)
    if obj is not None and getattr(obj, "expression", None) is not None and _walk(obj.expression):
        return True
    for c in model._constraints:
        body = getattr(c, "body", None)
        if body is not None and _walk(body):
            return True
    return False


def _classify_model_convexity(
    model: Model,
    *,
    failure_label: str = "Convexity detection failed",
    log_nonconvex_continuous: bool = False,
) -> tuple[bool, bool, list[bool] | None]:
    """Run the sound model convexity classifier once for solver dispatch."""
    try:
        from discopt._jax.convexity import classify_model as _classify_convexity

        # use_certificate=True enables the sound interval-Hessian fallback
        # for constraints/objective the syntactic walker leaves unproven.
        is_convex, constraint_mask = _classify_convexity(model, use_certificate=True)
        if log_nonconvex_continuous and not is_convex:
            logger.info("Nonconvex continuous model detected — using spatial Branch and Bound")
        return True, bool(is_convex), list(constraint_mask)
    except Exception as exc:
        logger.debug("%s: %s", failure_label, exc)
        return False, False, None


# Size gate for the auto cut policy: above this many scalar variables the
# per-node cut-separation overhead (eigh / extra LP re-solves) outweighs the
# typical bound gain, so the policy declines to enable cuts (sound: a no-op).
_AUTO_CUTS_MAX_VARS = 40


def _apply_auto_cut_policy(model: "Model", relaxer) -> None:
    """Choose at most one QCQP cut family by structure + size, in place.

    Data-driven policy (see the W2 A/B sweep): on QCQP *with* linear constraints
    targeted RLT is the strongest lever and PSD can even add nodes, so prefer
    RLT; on box-QP (no linear constraints) RLT cannot fire, so use PSD. Never
    stack the two. Declines (no cuts) on oversize models. Mutates the relaxer's
    ``_psd_cuts`` / ``_rlt_cuts`` flags; purely a performance choice — every cut
    family is sound, so this never affects correctness.
    """
    from discopt._jax.milp_relaxation import _linear_constraint_forms

    try:
        n = sum(v.size for v in model._variables)
        if n > _AUTO_CUTS_MAX_VARS:
            return  # size gate: leave cuts off
        has_linear_constraints = bool(_linear_constraint_forms(model, n))
        if has_linear_constraints:
            relaxer._rlt_cuts = True
            relaxer._psd_cuts = False
        else:
            relaxer._psd_cuts = True
            relaxer._rlt_cuts = False
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("auto cut policy skipped: %s", exc)


def _root_relaxation_lower_bound(
    model: "Model",
    root_lb: np.ndarray,
    root_ub: np.ndarray,
    time_limit: float,
    psd_cuts: bool = False,
) -> Optional[float]:
    """Solve the root MILP relaxation once and return its LP value as a rigorous
    global lower bound, or ``None`` if unavailable.

    The MILP relaxation built by ``build_milp_relaxation`` is a convex outer
    approximation of *model* over the box ``[root_lb, root_ub]`` (the same one AMP
    uses). When its objective is fully linearized (``objective_bound_valid``), the
    relaxation LP's optimal value is a valid lower bound on the original objective
    over that box — hence over the whole feasible set — for a *minimization*. This
    mirrors the root-LP-bound seeding ``_solve_milp_bb`` already performs, but for
    the spatial path, whose tree ``global_lower_bound`` can be tainted up to the
    incumbent on a nonconvex model and is dropped on an uncertified exit.

    Numerically catastrophic envelopes (e.g. cleared-division equalities over
    wide-ranged defined variables) are sanitized away first so the backend can
    return a bound instead of failing on the conditioning. Only ever called for a
    MINIMIZE objective; defensively returns ``None`` on any failure so it can
    never make a previously-bounded solve worse.
    """
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import (
        build_milp_relaxation,
        sanitize_relaxation_for_conditioning,
    )
    from discopt._jax.term_classifier import classify_nonlinear_terms

    try:
        terms = classify_nonlinear_terms(model)
        relax, _relax_info = build_milp_relaxation(
            model,
            terms,
            DiscretizationState(),
            bound_override=(root_lb, root_ub),
        )
        if not relax._objective_bound_valid:
            return None
        relax = sanitize_relaxation_for_conditioning(relax)

        # PSD (moment) cuts strengthen the McCormick relaxation toward the SDP
        # bound on nonconvex QCQP. `sanitize_*` only drops rows, so the column
        # map `_relax_info` still matches `relax`. Each cut is valid for the whole
        # feasible region, so the strengthened LP value is a *valid* lower bound
        # (>= the plain bound); it joins the candidates below and `max` keeps the
        # tightest. Opt-in via `psd_cuts=True`; any failure is a sound no-op.
        psd_bound: Optional[float] = None
        if psd_cuts:
            try:
                from discopt._jax.psd_cuts import psd_strengthen_relaxation_bound

                _zb, _za, _nc = psd_strengthen_relaxation_bound(relax, _relax_info)
                if _nc and _za is not None and np.isfinite(_za):
                    psd_bound = float(_za)
            except Exception as psd_exc:  # pragma: no cover - defensive
                logger.debug("root PSD-strengthened bound skipped: %s", psd_exc)
        budget = min(10.0, max(1.0, time_limit * 0.1))
        result = relax.solve(time_limit=budget, gap_tolerance=1e-6)
        # Only an OPTIMAL relaxation solve yields a valid lower bound. An
        # "unbounded" verdict means the relaxation (e.g. a McCormick envelope over
        # a box where a nonlinear-term variable is still unbounded) has no finite
        # lower bound, yet the backend still reports ``bound = 0.0`` — a finite
        # value that is NOT a valid bound. On himmel16 the root relaxation is
        # unbounded and 0.0 > the true optimum -0.866; surfacing it as the
        # fallback bound would publish an invalid (above-incumbent) dual bound.
        # Gate on optimality so an unbounded/limit solve returns no bound instead.
        plain_bound: Optional[float] = None
        if result.status == "optimal" and result.bound is not None and np.isfinite(result.bound):
            plain_bound = float(result.bound)

        # The raw ``relax.solve`` above carries only the static envelope cuts; the
        # per-node spatial relaxation additionally separates the multilinear hull,
        # edge-concave blocks, and (issue #114) the univariate-square tangents the
        # static envelope leaves slack deep inside a wide box. Routing the root
        # box through ``solve_at_node`` applies that same on-demand separation, so
        # the surfaced fallback bound matches the tree's tight per-node bounds
        # instead of the loose static value (ex9_2_6: -201.5 -> ~-1.7). Each
        # separated cut is a supporting hyperplane, so the result is still a
        # rigorous global lower bound; the relaxer's own guards (himmel16
        # unbounded cross-check, infeasible/limit re-verify) return no bound on
        # any unsound solve. ``_objective_bound_valid`` above already certified
        # the objective is fully linearized, the precondition both paths share.
        sep_bound: Optional[float] = None
        try:
            from discopt._jax.mccormick_lp import MccormickLPRelaxer

            node_res = MccormickLPRelaxer(model).solve_at_node(root_lb, root_ub, time_limit=budget)
            if node_res.lower_bound is not None and np.isfinite(node_res.lower_bound):
                sep_bound = float(node_res.lower_bound)
        except Exception as sep_exc:  # pragma: no cover - defensive
            logger.debug("root separated-relaxation bound skipped: %s", sep_exc)

        # Both values are valid lower bounds for a minimization, so the larger
        # (tighter) one is the better rigorous bound.
        candidates = [b for b in (plain_bound, sep_bound, psd_bound) if b is not None]
        if candidates:
            return max(candidates)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("root MILP-relaxation bound skipped: %s", exc)
    return None


def solve_model(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    threads: int = 1,
    deterministic: bool = True,
    batch_size: int = 16,
    strategy: str = "best_first",
    max_nodes: int = 100_000,
    ipopt_options: Optional[dict] = None,
    nlp_solver: str = "ipm",
    sparse: Optional[bool] = None,
    cutting_planes: bool = False,
    psd_cuts: bool = False,
    rlt_cuts: bool = False,
    cuts: str = "auto",
    partitions: int = 0,
    branching_policy: str = "fractional",
    use_learned_relaxations: bool = False,
    mccormick_bounds: str = "auto",
    gdp_method: str = "big-m",
    initial_point: Optional[np.ndarray] = None,
    skip_convex_check: bool = False,
    nlp_bb: Optional[bool] = None,
    lazy_constraints=None,
    incumbent_callback=None,
    node_callback=None,
    solver: Optional[str] = None,
    use_highs_milp: bool = True,
    presolve: bool = True,
    presolve_polynomial: bool = False,
    presolve_reverse_ad: bool = False,
    in_tree_presolve_stride: int = 0,
    eigenvalue_root_bound: bool = False,
    relaxation_arithmetic: str = "mccormick",
    subnlp_enabled: bool = True,
    subnlp_backend: str = "auto",
    subnlp_frequency: int = 20,
    subnlp_max_calls: int = 200,
    subnlp_options: Optional[dict] = None,
    **kwargs,
) -> SolveResult:
    """
    Solve a Model via NLP-based spatial Branch & Bound.

    At each B&B node the solver: (1) solves a continuous NLP relaxation
    with node-tightened bounds, (2) optionally generates OA cutting planes,
    (3) prunes if infeasible, (4) fathoms and updates incumbent if
    integer-feasible, or (5) branches on the most fractional integer variable.

    This function is called by :meth:`Model.solve` and is not typically
    invoked directly.

    Parameters
    ----------
    model : Model
        A Model with objective and constraints set.
    time_limit : float, default 3600.0
        Wall-clock time limit in seconds.
    gap_tolerance : float, default 1e-4
        Relative optimality gap tolerance for termination.
    threads : int, default 1
        Number of CPU threads (reserved for future use).
    deterministic : bool, default True
        Ensure deterministic results.
    batch_size : int, default 16
        Number of B&B nodes to export per iteration.
    strategy : str, default "best_first"
        Node selection strategy: ``"best_first"`` (lowest dual bound),
        ``"depth_first"``, or ``"best_estimate"`` (lowest pseudocost-based
        estimate of the subtree optimum — finds good incumbents earlier while
        ``best_first`` still drives the proof of optimality).
    max_nodes : int, default 100_000
        Maximum number of B&B nodes before stopping.
    ipopt_options : dict, optional
        Options passed to cyipopt (only used when ``nlp_solver="ipopt"``).
    nlp_solver : str, default "ipm"
        NLP solver backend: ``"pounce"`` (POUNCE — pure-Rust Ipopt port),
        ``"ipopt"`` (cyipopt), ``"ipm"`` (pure-JAX IPM), or
        ``"sparse_ipm"`` (sparse KKT + scipy direct solve). For single
        continuous solves the ``"ipm"`` default is promoted to a KKT-valid
        backend, resolving to POUNCE when available and falling back to cyipopt.
    sparse : bool or None, default None
        Force sparse (True) or dense (False) Jacobian evaluation.
        If None, auto-selects based on problem size and density.
    cutting_planes : bool, default False
        Enable outer-approximation cut generation after NLP relaxation solves.
    partitions : int, default 0
        Number of piecewise McCormick partitions (0 = standard convex
        relaxation, k > 0 = k partitions for tighter relaxations).
    branching_policy : str, default "fractional"
        Variable selection: ``"fractional"`` (most-fractional, default)
        or ``"gnn"`` (GNN scoring hook; Rust handles actual branching).
    use_learned_relaxations : bool, default False
        Use ICNN-based learned convex relaxations instead of standard
        McCormick. Requires ``pip install discopt[gnn]`` (equinox + optax).
        Falls back to standard McCormick for unsupported operations.
    mccormick_bounds : str, default "auto"
        McCormick relaxation lower-bounding strategy:
        ``"auto"`` resolves to ``"none"`` — the McCormick ``"nlp"`` bound is
        only valid for convex models (see below), and convex models already
        get valid bounds from the NLP relaxation, so ``"auto"`` relies on the
        NLP/alphaBB path in both cases,
        ``"nlp"`` solves an NLP over the McCormick objective relaxation.
        This is a valid lower bound **only for convex models**: the bound
        solver evaluates the relaxation at ``x_cv == x_cc`` where every
        McCormick rule is tight, so it minimizes the original objective
        *locally* and a nonconvex local optimum is not a valid bound
        (issue #120). For nonconvex models it is automatically downgraded to
        ``"none"``,
        ``"lp"`` solves an LP over the full McCormick reformulation —
        gives a *valid global* lower bound and tends to find provable
        global optima much faster on nonconvex problems with bilinears
        (pooling, polynomial NLP, QCQP). Recommended when your model
        is nonconvex and contains bilinear/multilinear terms; pair with
        ``subnlp_frequency=1`` to turn each LP primal into an incumbent.
        Requires at least one continuous variable; falls back to
        ``"none"`` for pure-integer nonconvex models (the ``"nlp"`` fallback
        would be unsound — issue #120),
        ``"midpoint"`` evaluates the convex underestimator at midpoint
        (heuristic, not a valid global lower bound — use with caution),
        ``"none"`` disables.
    gdp_method : str, default "big-m"
        Reformulation method for disjunctive constraints:
        ``"big-m"`` (default) or ``"hull"`` (convex hull).
    solver : str or None, default None
        Optional global-solver selector. Use ``"amp"`` to dispatch to
        Adaptive Multivariate Partitioning instead of branch-and-bound.
        Use ``"gp"`` to dispatch to the geometric-programming fast path:
        the model is checked for GP structure (posynomial/monomial
        objective and constraints over strictly-positive continuous
        variables) and, if it qualifies, solved exactly via the log-space
        convex reformulation (``y = log x``). Raises ``ValueError`` if the
        model is not a geometric program. See :mod:`discopt.gp`.
        When ``solver`` is left ``None``, a recognised GP is **automatically**
        routed through this same exact log-space convex solve (a single NLP,
        global optimum) instead of branch-and-bound; pass ``"bb"`` to opt out
        and force the classic branch-and-bound path. The automatic route is
        skipped when branch-and-bound streaming callbacks (``incumbent_callback``,
        ``node_callback``, ``iteration_callback``) are attached, or when
        ``skip_convex_check=True``.
    solver="amp" options
        The AMP backend also accepts ``rel_gap``, ``abs_tol``, ``max_iter``,
        ``n_init_partitions``, ``partition_method``, ``milp_time_limit``,
        ``milp_gap_tolerance``, ``presolve_bt``, ``presolve_bt_algo``,
        ``presolve_bt_time_limit``, ``presolve_bt_mip_time_limit``,
        ``apply_partitioning``, ``disc_var_pick``, ``partition_scaling_factor``,
        ``partition_scaling_factor_update``, ``disc_add_partition_method``,
        ``disc_abs_width_tol``, ``convhull_formulation``, ``convhull_ebd``,
        ``convhull_ebd_encoding``, ``use_start_as_incumbent``, ``obbt_at_root``,
        ``obbt_with_cutoff``, ``alphabb_cutoff_obbt``, and ``obbt_time_limit``.

    Returns
    -------
    SolveResult
        Contains solution values, objective, gap, node count, and
        per-layer profiling times (Rust, JAX, Python).
    """
    # --- Enforce float64 precision ---
    # JAX defaults to float32 unless JAX_ENABLE_X64=1 is set *before* importing
    # JAX.  All solver tolerances assume float64; float32 silently degrades
    # convergence and may return incorrect solutions.
    import jax.numpy as jnp

    if jnp.zeros(1).dtype != jnp.float64:
        import warnings

        warnings.warn(
            "JAX is running in float32 mode.  Set the environment variable "
            "JAX_ENABLE_X64=1 *before* importing JAX for full solver precision.  "
            "Results may be inaccurate.",
            stacklevel=2,
        )

    _valid_nlp_solvers = {"ipm", "pounce", "ipopt", "cyipopt", "sparse_ipm", "simplex"}
    if nlp_solver not in _valid_nlp_solvers:
        raise ValueError(
            f"Unknown nlp_solver={nlp_solver!r}. Choose one of "
            f"{sorted(_valid_nlp_solvers)}. (The 'ripopt' backend was replaced "
            "by 'pounce', a pure-Rust port of Ipopt.)"
        )

    # --- AMP (Adaptive Multivariate Partitioning) global solver ---
    _solver = solver if solver is not None else kwargs.pop("solver", None)
    # Recognised global-solver selectors: ``None`` (default branch-and-bound,
    # with the automatic GP fast path below), ``"amp"``, ``"gp"`` (force the GP
    # log-space path), and ``"bb"`` (force classic branch-and-bound, opting out
    # of the automatic GP fast path). Reject anything else rather than silently
    # falling through to B&B.
    if _solver not in (None, "amp", "gp", "bb"):
        raise ValueError(f"Unknown solver={_solver!r}. Choose one of None, 'amp', 'gp', 'bb'.")
    if _solver == "amp":
        import warnings

        from discopt.solvers.amp import solve_amp

        amp_kwargs = {}
        amp_option_keys = (
            "rel_gap",
            "abs_tol",
            "max_iter",
            "n_init_partitions",
            "partition_method",
            "iteration_callback",
            "milp_time_limit",
            "milp_gap_tolerance",
            "presolve_bt",
            "presolve_bt_algo",
            "presolve_bt_time_limit",
            "presolve_bt_mip_time_limit",
            "apply_partitioning",
            "disc_var_pick",
            "partition_scaling_factor",
            "partition_scaling_factor_update",
            "disc_add_partition_method",
            "disc_abs_width_tol",
            "convhull_formulation",
            "convhull_ebd",
            "convhull_ebd_encoding",
            "use_start_as_incumbent",
            "obbt_at_root",
            "obbt_with_cutoff",
            "alphabb_cutoff_obbt",
            "obbt_time_limit",
            "milp_solver",
        )
        for key in amp_option_keys:
            if key in kwargs:
                amp_kwargs[key] = kwargs.pop(key)
        if initial_point is not None:
            amp_kwargs["initial_point"] = initial_point

        ignored_amp_options = []

        def _note_ignored(name: str, should_warn: bool) -> None:
            if should_warn:
                ignored_amp_options.append(name)

        _note_ignored("threads", threads != 1)
        _note_ignored("deterministic", deterministic is not True)
        _note_ignored("batch_size", batch_size != 16)
        _note_ignored("strategy", strategy != "best_first")
        _note_ignored("max_nodes", max_nodes != 100_000)
        _note_ignored("ipopt_options", ipopt_options is not None)
        _note_ignored("sparse", sparse is not None)
        _note_ignored("cutting_planes", cutting_planes is not False)
        _note_ignored("partitions", partitions != 0)
        _note_ignored("branching_policy", branching_policy != "fractional")
        _note_ignored("use_learned_relaxations", use_learned_relaxations is not False)
        _note_ignored("mccormick_bounds", mccormick_bounds != "auto")
        _note_ignored("gdp_method", gdp_method != "big-m")
        _note_ignored("nlp_bb", nlp_bb is not None)
        _note_ignored("lazy_constraints", lazy_constraints is not None)
        _note_ignored("incumbent_callback", incumbent_callback is not None)
        _note_ignored("node_callback", node_callback is not None)
        _note_ignored("use_highs_milp", use_highs_milp is not True)
        if kwargs:
            ignored_amp_options.extend(sorted(kwargs))
        if ignored_amp_options:
            warnings.warn(
                "AMP ignores solve_model options: "
                + ", ".join(sorted(dict.fromkeys(ignored_amp_options))),
                stacklevel=2,
            )

        # rel_gap defaults to gap_tolerance if not separately provided
        if "rel_gap" not in amp_kwargs:
            amp_kwargs["rel_gap"] = gap_tolerance

        return solve_amp(
            model,
            time_limit=time_limit,
            nlp_solver=nlp_solver,
            skip_convex_check=skip_convex_check,
            **amp_kwargs,
        )

    # --- GP (geometric programming) log-space convex fast path ---
    if _solver == "gp":
        import warnings

        from discopt.gp import classify_gp, solve_gp

        if classify_gp(model) is None:
            raise ValueError(
                "solver='gp' was requested but the model is not a geometric "
                "program. A GP needs a posynomial (or monomial) objective, "
                "constraints of the form posynomial <= monomial or monomial "
                "== monomial, and strictly-positive continuous variables. "
                "See discopt.gp.classify_gp for the exact preconditions."
            )

        ignored_gp_options = []

        def _note_ignored_gp(name: str, should_warn: bool) -> None:
            if should_warn:
                ignored_gp_options.append(name)

        _note_ignored_gp("threads", threads != 1)
        _note_ignored_gp("deterministic", deterministic is not True)
        _note_ignored_gp("batch_size", batch_size != 16)
        _note_ignored_gp("strategy", strategy != "best_first")
        _note_ignored_gp("max_nodes", max_nodes != 100_000)
        _note_ignored_gp("partitions", partitions != 0)
        _note_ignored_gp("branching_policy", branching_policy != "fractional")
        _note_ignored_gp("use_learned_relaxations", use_learned_relaxations is not False)
        _note_ignored_gp("mccormick_bounds", mccormick_bounds != "auto")
        _note_ignored_gp("gdp_method", gdp_method != "big-m")
        _note_ignored_gp("cutting_planes", cutting_planes is not False)
        _note_ignored_gp("nlp_bb", nlp_bb is not None)
        _note_ignored_gp("lazy_constraints", lazy_constraints is not None)
        _note_ignored_gp("incumbent_callback", incumbent_callback is not None)
        _note_ignored_gp("node_callback", node_callback is not None)
        if kwargs:
            ignored_gp_options.extend(sorted(kwargs))
        if ignored_gp_options:
            warnings.warn(
                "GP fast path ignores solve_model options: "
                + ", ".join(sorted(dict.fromkeys(ignored_gp_options))),
                stacklevel=2,
            )

        result = solve_gp(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            nlp_solver=nlp_solver,
            ipopt_options=ipopt_options,
        )
        if result is None:  # pragma: no cover - guarded by classify_gp above
            raise RuntimeError("GP reformulation failed unexpectedly.")
        return result

    # --- Auto GP fast path: a recognised geometric program solves exactly via
    # its log-space convex reformulation (y = log x), which is strictly better
    # than branch-and-bound (a single convex NLP gives the global optimum). This
    # fires only when no global solver was explicitly requested and the user did
    # not attach branch-and-bound streaming callbacks (which the GP path cannot
    # honour — falling through to B&B keeps them firing). ``solver="bb"`` is an
    # explicit opt-out that forces the classic path. ``classify_gp`` bails on the
    # first integer variable / non-positive lower bound, so the probe is cheap on
    # the common (non-GP) path.
    _has_bb_callbacks = (
        incumbent_callback is not None
        or node_callback is not None
        or kwargs.get("iteration_callback") is not None
    )
    if _solver is None and not _has_bb_callbacks and not skip_convex_check:
        from discopt.gp import classify_gp, solve_gp

        if classify_gp(model) is not None:
            gp_result = solve_gp(
                model,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                nlp_solver=nlp_solver,
                ipopt_options=ipopt_options,
            )
            if gp_result is not None:
                return gp_result

    # --- OA decomposition: general-purpose Outer Approximation ---
    if gdp_method == "oa":
        from discopt.solvers.oa import solve_oa

        # Extract OA-specific kwargs that solve_model doesn't understand
        oa_kwargs = {}
        for key in ("equality_relaxation", "ecp_mode", "feasibility_cuts"):
            if key in kwargs:
                oa_kwargs[key] = kwargs.pop(key)

        return solve_oa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_nodes,
            nlp_solver=nlp_solver,
            **oa_kwargs,
        )

    # --- LOA decomposition: intercept before GDP reformulation ---
    if gdp_method == "loa":
        from discopt.solvers.gdpopt_loa import solve_gdpopt_loa

        return solve_gdpopt_loa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_nodes,
            nlp_solver=nlp_solver,
        )

    # Capture any modeler-provided GAMS start (``x.l`` -> _gams_initial_values)
    # BEFORE the reform passes below rebuild ``model`` into fresh objects that
    # drop this dynamically-attached attribute. It is re-attached by name to the
    # working model just before the B&B loop so the root subnlp can seed from it.
    _captured_gams_initial_values = getattr(model, "_gams_initial_values", None)

    # --- GDP reformulation: convert indicator/disjunctive/SOS to standard MINLP ---
    from discopt._jax.gdp_reformulate import reformulate_gdp

    model = reformulate_gdp(model, method=gdp_method)

    # --- Entropy-family canonicalization: recover the ``entropy(x) = x*log(x)``
    # and ``centropy(x, y) = x*log(x/y)`` intrinsics from the raw products that
    # AMPL/GAMS emit when they lower those opcodes into a ``.nl`` file. The
    # MILP/McCormick-LP relaxer cannot decompose ``x*log(x)`` / ``x*log(x/y)``
    # and falls back to a constant separable objective floor that never tightens
    # under branching, so an entropy / Gibbs / relative-entropy objective (e.g.
    # ``ex6_1_4``) is solved to the global optimum but cannot be certified
    # (issue #207). discopt carries dedicated convexity support for both
    # intrinsics (``entropy`` is convex, ``centropy`` is jointly convex on
    # x≥0,y>0); recovering them lets a separable entropy / relative-entropy
    # objective be detected as convex and certify on the convex fast path. The
    # rewrites are exact and convexity-preserving, so they run unconditionally
    # and return the model unchanged when nothing matches.
    from discopt._jax.factorable_reform import canonicalize_entropy

    model = canonicalize_entropy(model)

    # --- Objective-defining-equality relaxation (the SUSPECT "objective
    # constraint"). When the model is `min/max z` with z a free scalar that
    # appears only in one equality `z = g(x)` (affinely), relax that equality
    # to the inequality the objective binds against (`z >= g(x)` for min). The
    # rewrite is EXACT at the optimum — lowering z to g(x) is always feasible
    # and improves the objective, so the relaxed optimum satisfies the original
    # equality — and turns a convex-defining equality (non-convex *as an
    # equality*) into a convex inequality, unlocking the convex solve path with
    # a valid lower bound instead of erratic nonconvex NLP-BB (issue: du-opt).
    # The transform is structurally gated and abstains conservatively, so it is
    # general and never alters the optimum. Skipped when streaming B&B callbacks
    # are attached so node/incumbent indices stay aligned with the user's model.
    if not _has_bb_callbacks:
        from discopt._jax.objective_epigraph import relax_objective_defining_equality

        model, _epi_changed = relax_objective_defining_equality(model)
        if _epi_changed:
            logger.debug("relaxed objective-defining equality to binding inequality")

    # --- Factorable reformulation: clear sign-definite denominators and lift
    # mixed repeated-factor products (e.g. x*x*y) into bilinear form via
    # monomial aux variables, so terms the relaxation pipeline would otherwise
    # drop to general_nl get a valid outer approximation (issue #130). The pass
    # is value-preserving, but clearing a denominator (e.g. x**2/z, convex for
    # z>0) or distributing a product can *destroy* convexity — so it is gated to
    # provably-nonconvex models only, preserving the convex fast path for the
    # rest. The cheap structural scan runs first so only models that actually
    # have liftable terms pay for convexity detection.
    from discopt._jax.factorable_reform import (
        factorable_reformulate,
        has_factorable_work,
    )

    # Whether the model has continuous DECISION variables, captured *before* the
    # factorable lift introduces continuous *auxiliary* variables (w == x**k).
    # A model whose original variables are all discrete is a combinatorial
    # problem: the bilinear lift adds dependent continuous aux vars, but spatial
    # branching on them does not help and can trap the incumbent search on an
    # objective plateau (nvs16 stalls at the x1==1 plateau obj=14.203 instead of
    # the true 0.703 — issue #120 primal convergence). Route those models to the
    # integer-B&B + alphaBB path (mc_mode "none") rather than the McCormick LP
    # spatial path, mirroring the existing "integer-only models have nothing to
    # spatial-branch on" fallback below (which the aux vars otherwise defeat).
    _origin_has_continuous_var = any(v.var_type == VarType.CONTINUOUS for v in model._variables)

    # Whether any *original* continuous decision variable has a finite box to
    # spatial-branch on, captured (like ``_origin_has_continuous_var``) before the
    # factorable lift adds dependent continuous aux vars. A model whose only
    # continuous variables are *unbounded* slacks (gear4's ``x4, x5``) has nothing
    # to bisect on the spatial McCormick LP path: the tree dead-ends after the
    # root and a no-incumbent exhaustion would be falsely certified infeasible
    # (issue #185). Such models route to the NLP/alphaBB path instead.
    _origin_lb_chk, _origin_ub_chk = flat_variable_bounds(model)
    # Periodic-variable reduction: a free continuous variable used only inside
    # sin/cos is restricted to one period [-pi, pi]. Spatial B&B cannot partition
    # an infinite domain, so an angular variable like cos(y) over a free y never
    # converges (nlp_001: 237 nodes -> 1). Surgical: only this rule runs here, so
    # variables the relaxation already handles analytically (e.g. x*exp(x) over a
    # free x) are left untouched. Sound: the reduction only shrinks the box.
    try:
        from discopt._jax.nonlinear_bound_tightening import PeriodicVariableBoundRule

        _per_lb, _per_ub, _per_stats = tighten_nonlinear_bounds(
            model, _origin_lb_chk, _origin_ub_chk, rules=(PeriodicVariableBoundRule(),)
        )
        if _per_stats.n_tightened > 0:
            from discopt.solvers.amp import _apply_flat_bounds_to_model

            _apply_flat_bounds_to_model(model, _per_lb, _per_ub)
            _origin_lb_chk, _origin_ub_chk = _per_lb, _per_ub
    except Exception as _per_exc:  # pragma: no cover - defensive
        logger.debug("periodic-variable bound reduction skipped: %s", _per_exc)
    _origin_has_finite_continuous_var = False
    _origin_chk_off = 0
    for _ov in model._variables:
        if _ov.var_type == VarType.CONTINUOUS:
            _ov_seg = slice(_origin_chk_off, _origin_chk_off + _ov.size)
            if np.all(np.isfinite(_origin_lb_chk[_ov_seg])) and np.all(
                np.isfinite(_origin_ub_chk[_ov_seg])
            ):
                _origin_has_finite_continuous_var = True
                break
        _origin_chk_off += _ov.size

    # Pre-reform model + original variable count, set only when the factorable
    # lift actually fires (see below). Used by the per-node interval bound to
    # recover the un-distributed objective's tight enclosure over the original
    # variables. ``None`` means no lift happened, so the live model IS original.
    _prereform_model = None
    _prereform_nvars = 0

    if has_factorable_work(model):
        # Tighten variable bounds with FBBT *before* the reform so its interval
        # checks see finite bounds. A fractional-power-of-product lift (issue
        # #138) only fires when the lifted base has a finite interval; constraint
        # propagation supplies that for models whose denominator variables are
        # *declared* unbounded but are pinned finitely by other constraints (e.g.
        # ex1233's geometric vars x12..x20, bounded by the assignment rows). Cheap
        # and only run when the reform is about to fire; sound (FBBT only removes
        # infeasible regions, so the tightened box still contains every feasible
        # point). The later root presolve re-tightens, so this never loosens.
        try:
            from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

            _fr_off: list[int] = []
            _fr_sz: list[int] = []
            _fr_o = 0
            for _v in model._variables:
                if _v.var_type in (VarType.BINARY, VarType.INTEGER):
                    _fr_off.append(_fr_o)
                    _fr_sz.append(_v.size)
                _fr_o += _v.size
            _, _fr_lb, _fr_ub, _, _ = _extract_variable_info(model)
            _fr_lb, _fr_ub, _fr_infeas, _fr_changed = tighten_root_bounds_with_fbbt(
                model, _fr_lb, _fr_ub, _fr_off, _fr_sz
            )
            if not _fr_infeas and _fr_changed:
                from discopt.solvers.amp import _apply_flat_bounds_to_model

                _apply_flat_bounds_to_model(model, _fr_lb, _fr_ub)
        except Exception as _fr_fbbt_exc:  # pragma: no cover - defensive
            logger.debug("pre-reform FBBT skipped: %s", _fr_fbbt_exc)

        _fr_ok, _fr_convex, _ = _classify_model_convexity(model)
        if _fr_ok and not _fr_convex:
            # Retain the pre-reform model and its original variable count. The
            # factorable lift distributes products and replaces repeated factors
            # with aux columns (``100*(x1-x0**2)**2`` becomes
            # ``100*x1**2 - 200*x1*_fr_aux + 100*x0**4`` with ``_fr_aux = x1*x0**2``),
            # which DESTROYS the sum-of-squares structure: naive interval
            # arithmetic on the distributed/lifted objective is hopelessly loose
            # (a square that is provably >= 0 enclosed as a large-negative
            # interval), so the cheap per-node interval bound can no longer prune.
            # Aux columns are appended after the originals, so the first
            # ``_prereform_nvars`` flat entries of every node box are exactly the
            # original-variable sub-box; evaluating the ORIGINAL objective over it
            # is a valid (and, for sum-of-squares, tight) lower bound. See the
            # per-node bound loops below.
            _prereform_model = model
            _prereform_nvars = sum(v.size for v in model._variables)
            model = factorable_reformulate(model)
        # A *convex* model with a clearable denominator is deliberately left
        # untouched here: many such divisions (e.g. the rotated-SOC ``x**2/z``)
        # are solved exactly by the convex NLP fast path, and clearing would
        # needlessly destroy that structure. Only if the convex NLP later *fails*
        # to certify do we clear and fall back to spatial B&B (see the convex
        # fast-path block below).

    # --- Build Rust model representation for FBBT ---
    _model_repr = None
    try:
        from discopt._rust import model_to_repr

        _builder = getattr(model, "_builder", None)
        _model_repr = model_to_repr(model, _builder)
    except Exception:
        pass  # FBBT bindings unavailable; skip

    # --- Root presolve: M10 variable elimination + (opt-in) M4+M5
    # polynomial reformulation, then FBBT for bound propagation.
    # Tightened bounds are pushed back into the Python `model` so that
    # the relaxation compiler / B&B initialisation see them. See
    # discopt._jax.presolve_pipeline for the sequencing rationale.
    if _model_repr is not None and presolve:
        try:
            from discopt._jax.presolve_pipeline import (
                propagate_bounds_to_model,
                run_root_presolve,
            )

            _model_repr, _presolve_stats = run_root_presolve(
                _model_repr,
                eliminate=True,
                polynomial=presolve_polynomial,
                fbbt=True,
            )
            n_tightened = propagate_bounds_to_model(model, _model_repr)
            elim = _presolve_stats.get("elimination", {})
            poly = _presolve_stats.get("polynomial", {})
            if elim.get("variables_fixed", 0) > 0 or n_tightened > 0:
                logger.info(
                    "Presolve: fixed %d vars, removed %d eqs, tightened %d "
                    "bounds (poly aux vars: %d)",
                    elim.get("variables_fixed", 0),
                    elim.get("constraints_removed", 0),
                    n_tightened,
                    poly.get("aux_variables_introduced", 0),
                )
        except Exception as e:
            logger.debug("Root presolve failed: %s", e)

    # --- Reverse-AD interval tightening (M9 of #51, opt-in) ---
    # Iterates Gauss-Seidel reverse-mode interval AD over every
    # constraint to a fixed point and writes back tighter scalar bounds.
    # Disabled by default because it walks the Python expression DAG and
    # can be slow on very large models.
    if presolve and presolve_reverse_ad:
        try:
            from discopt._jax.presolve_pipeline import run_reverse_ad_tightening

            n_rad = run_reverse_ad_tightening(model)
            if n_rad > 0:
                logger.info("Reverse-AD presolve tightened %d variable bounds", n_rad)
        except Exception as e:
            logger.debug("Reverse-AD tightening failed: %s", e)

    # --- Eigenvalue root bound on quadratic objectives (M6 of #51, opt-in) ---
    # For models with a quadratic objective, compute a sound root-node
    # bound via spectral decomposition. Used only as an informational
    # diagnostic at the root; does not affect the B&B tree directly.
    if eigenvalue_root_bound:
        try:
            from discopt._jax.convexity.eigenvalue_arith import (
                QuadraticForm,
                quadratic_form_bound,
            )
            from discopt._jax.problem_classifier import (
                ProblemClass,
                classify_problem,
                extract_qp_data,
            )

            pcls = classify_problem(model)
            if pcls in (ProblemClass.QP, ProblemClass.MIQP):
                qp = extract_qp_data(model)
                Q_qf = 0.5 * np.asarray(qp.Q, dtype=np.float64)
                b_qf = np.asarray(qp.c, dtype=np.float64)
                qf = QuadraticForm(Q=Q_qf, b=b_qf, c=float(qp.obj_const))
                x_lo = np.asarray(qp.x_l, dtype=np.float64)
                x_hi = np.asarray(qp.x_u, dtype=np.float64)
                if np.all(np.isfinite(x_lo)) and np.all(np.isfinite(x_hi)):
                    eig_bound = quadratic_form_bound(qf, x_lo, x_hi)
                    logger.info(
                        "Eigenvalue root bound on quadratic objective: [%g, %g]",
                        float(eig_bound.lo),
                        float(eig_bound.hi),
                    )
        except Exception as e:
            logger.debug("Eigenvalue root bound failed: %s", e)

    # --- Learned relaxation registry (opt-in) ---
    import warnings

    _learned_registry = None
    _relax_mode = "standard"
    if use_learned_relaxations:
        try:
            from discopt._jax.learned_relaxations import load_pretrained_registry

            _learned_registry = load_pretrained_registry()
            if len(_learned_registry) > 0:
                _relax_mode = "learned"
            else:
                warnings.warn(
                    "No pretrained learned relaxation models found. "
                    "Falling back to standard McCormick.",
                    stacklevel=2,
                )
        except ImportError:
            warnings.warn(
                "Learned relaxations require pip install discopt[gnn] "
                "(equinox + optax). Falling back to standard McCormick.",
                stacklevel=2,
            )

    t_start = time.perf_counter()
    rust_time = 0.0
    jax_time = 0.0

    # --- AD-only user functions (dm.custom): force the local NLP path ---
    # A CustomCall wraps an opaque JAX-traceable callable. discopt can autodiff
    # it (so the local NLP path works), but the relaxation compiler, Rust
    # presolve, .nl export, and nonlinear bound tightening cannot reason about
    # it — global branch-and-bound would have no valid node relaxation. So we
    # solve locally only (no global optimality certificate) and refuse when
    # integer/binary variables force B&B. Placed before the DAG-walking
    # presolve/infeasibility checks so they never see a CustomCall. See #27b.
    if _model_contains_custom_call(model):
        if not _is_pure_continuous(model):
            raise ValueError(
                "Model contains a dm.custom(...) AD-only user function together "
                "with integer/binary variables. Global branch-and-bound needs a "
                "valid relaxation at each node, which an opaque callable cannot "
                "provide. Rebuild the function from dm.* primitives (see dm.udf), "
                "or remove the integer/binary variables."
            )
        logger.info(
            "Model contains a dm.custom(...) AD-only user function — solving on "
            "the local NLP path only (no global optimality certificate)."
        )
        result = _solve_continuous(
            model,
            time_limit,
            ipopt_options,
            t_start,
            nlp_solver,
            initial_point=initial_point,
        )
        result.gap_certified = False
        return result

    if nlp_solver == "pounce":
        logger.info("Using POUNCE (pure-Rust Ipopt port)")
    elif nlp_solver in ("sparse_ipm", "ipm"):
        # The JAX IPM (and its sparse variant) was retired; these names are
        # back-compat aliases that resolve to POUNCE on the NLP/MINLP path.
        logger.info("Using POUNCE (pure-Rust Ipopt port; %r is a deprecated alias)", nlp_solver)
    else:
        logger.info("Using Ipopt (via cyipopt)")

    # --- Check for very large variable bounds ---
    # All solver paths (LP IPM, QP IPM, NLP) use barrier methods that
    # struggle with bounds beyond ~1e15. Check once before any dispatch.
    _check_finite_bounds(model)
    nonlinear_infeasibility = _detect_nonlinear_bound_infeasibility(model)
    if nonlinear_infeasibility is not None:
        logger.info(
            "Nonlinear bound tightening proved model infeasible: %s", nonlinear_infeasibility
        )
        return SolveResult(
            status="infeasible",
            wall_time=time.perf_counter() - t_start,
            gap_certified=True,
        )

    _pure_continuous = _is_pure_continuous(model)
    _pure_continuous_convexity_known = False
    _pure_continuous_is_convex = False
    _pure_continuous_constraint_mask = None
    if _pure_continuous and not skip_convex_check:
        # For QPs this must run before QP dispatch so indefinite
        # pure-continuous QPs do not use the convex QP solver.
        (
            _pure_continuous_convexity_known,
            _pure_continuous_is_convex,
            _pure_continuous_constraint_mask,
        ) = _classify_model_convexity(
            model,
            failure_label="Convex fast path detection failed",
            log_nonconvex_continuous=True,
        )
    _root_convexity_known = _pure_continuous_convexity_known
    _root_is_convex = _pure_continuous_is_convex
    _root_constraint_mask = _pure_continuous_constraint_mask

    # --- Explicit NLP-BB override: bypass specialized solvers ---
    if nlp_bb is True and not _pure_continuous:
        return _solve_nlp_bb(
            model,
            time_limit,
            gap_tolerance,
            batch_size,
            strategy,
            max_nodes,
            t_start,
            nlp_solver,
            skip_convex_check=skip_convex_check,
            initial_point=initial_point,
            lazy_constraints=lazy_constraints,
            incumbent_callback=incumbent_callback,
            node_callback=node_callback,
            in_tree_presolve_stride=in_tree_presolve_stride,
            in_tree_presolve_repr=_model_repr,
        )

    # --- Problem classification: dispatch LP/QP to specialized solvers ---
    try:
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        problem_class = classify_problem(model)
    except Exception as e:
        logger.debug("Problem classification failed: %s", e)
        problem_class = None

    _pure_continuous_force_spatial = False
    if problem_class is not None:
        if problem_class == ProblemClass.LP:
            return _solve_lp(model, t_start, time_limit, prefer_pounce=nlp_solver == "pounce")
        elif problem_class == ProblemClass.QP:
            if _pure_continuous:
                if _pure_continuous_convexity_known and _pure_continuous_is_convex:
                    return _solve_qp(model, t_start, prefer_pounce=nlp_solver == "pounce")
                _pure_continuous_force_spatial = True
            else:
                return _solve_qp(model, t_start, prefer_pounce=nlp_solver == "pounce")
        elif problem_class == ProblemClass.MILP:
            # Warm-started-simplex engine (nlp_solver="simplex"): the whole MILP
            # B&B runs in Rust with dual-warm-started simplex node solves. Opt-in;
            # falls through to the default path if unavailable.
            if nlp_solver == "simplex":
                _simplex_res = _solve_milp_simplex(
                    model, time_limit, gap_tolerance, max_nodes, t_start
                )
                if _simplex_res is not None:
                    return _simplex_res
            # POUNCE-only mode (nlp_solver="pounce") routes to the self-hosted
            # B&B and bypasses HiGHS entirely (Phase 1 increment 5); the
            # self-hosted path is sound (incr 1), POUNCE-recovers stalled
            # nodes (2), purifies incumbents (3), and reduced-cost-fixes (4).
            _pounce_only = nlp_solver == "pounce"
            if use_highs_milp and not _pounce_only:
                highs_result = _solve_milp_highs(model, t_start, time_limit, gap_tolerance)
                if highs_result is not None:
                    return highs_result
            return _solve_milp_bb(
                model,
                time_limit,
                gap_tolerance,
                batch_size,
                strategy,
                max_nodes,
                t_start,
                prefer_pounce=_pounce_only,
            )
        elif problem_class == ProblemClass.MIQP:
            # Try HiGHS MIQP first (unless POUNCE-only), fall back to B&B.
            _pounce_only = nlp_solver == "pounce"
            if not _pounce_only:
                highs_result = _solve_qp_highs(model, t_start, time_limit)
                if highs_result is not None:
                    return highs_result
            return _solve_miqp_bb(
                model,
                time_limit,
                gap_tolerance,
                batch_size,
                strategy,
                max_nodes,
                t_start,
                prefer_pounce=_pounce_only,
            )

    # The pure-JAX interior-point method ("ipm") and its sparse variant
    # ("sparse_ipm") are retired as NLP solvers. From here on (the NLP/MINLP
    # path only) route them to POUNCE, the pure-Rust Ipopt port. MILP/MIQP/LP/QP
    # already returned above with the original nlp_solver, so their HiGHS-vs-
    # POUNCE routing (gated by _pounce_only == "pounce") is unaffected. JAX
    # remains the autodiff substrate (McCormick relaxations + differentiable
    # layers). "ipm" is the historical default, so this resolution is silent.
    if nlp_solver in ("ipm", "sparse_ipm"):
        nlp_solver = "pounce"

    # --- Convex NLP fast path: skip B&B for convex continuous problems ---
    if _pure_continuous and _pure_continuous_convexity_known and _pure_continuous_is_convex:
        logger.info("Convex NLP detected — solving with single NLP (global optimality guaranteed)")
        result = _solve_continuous(
            model,
            time_limit,
            ipopt_options,
            t_start,
            nlp_solver,
            initial_point=initial_point,
        )
        result.convex_fast_path = True
        if result.status == "optimal":
            return result
        # The convex NLP did not certify (e.g. an ill-conditioned division whose
        # denominator reaches toward zero, like st_e17's 0.2458*x0**2/x1 with
        # x1 down to 1e-5). If the model has a sign-definite non-constant
        # denominator — which the relaxation otherwise drops to general_nl —
        # clear it (exact, value-preserving) and fall back to the sound spatial
        # B&B, which can certify it. Only clearing is applied (no
        # convexity-destroying mixed-product lift). When nothing is clearable,
        # return the NLP result unchanged (no regression).
        from discopt._jax.factorable_reform import has_clearable_denominator

        if has_clearable_denominator(model):
            cleared = factorable_reformulate(model, clear_only=True)
            if cleared is not model:
                logger.info(
                    "Convex NLP did not certify (status=%s); clearing sign-definite "
                    "denominator and retrying via spatial B&B",
                    result.status,
                )
                model = cleared
                _pure_continuous_is_convex = False
                _root_is_convex = False
                _root_constraint_mask = None  # stale: cleared constraint is nonconvex
                try:
                    from discopt._rust import model_to_repr

                    _model_repr = model_to_repr(model, getattr(model, "_builder", None))
                except Exception:
                    _model_repr = None
                # Fall through to the spatial B&B below (rebuilt from `model`).
            else:
                return result
        else:
            return result

    # --- Pure continuous: solve directly only when spatial search was not requested ---
    if (
        _pure_continuous
        and not _pure_continuous_force_spatial
        and (skip_convex_check or not _pure_continuous_convexity_known)
    ):
        return _solve_continuous(
            model,
            time_limit,
            ipopt_options,
            t_start,
            nlp_solver,
            initial_point=initial_point,
        )

    # --- NLP-BB auto-select for convex MINLPs (nlp_bb=None) ---
    # Placed after problem classifier so MILP/MIQP use their specialized
    # (faster) solvers. Only genuinely nonlinear convex MINLPs reach here.
    # Also skip when lazy constraints are provided (they need the cut pool
    # infrastructure from the full spatial B&B loop).
    if nlp_bb is None and lazy_constraints is None:
        if not _root_convexity_known:
            _root_convexity_known, _root_is_convex, _root_constraint_mask = (
                _classify_model_convexity(
                    model,
                    failure_label="Convex MINLP detection failed",
                )
            )
        if _root_convexity_known and _root_is_convex:
            logger.info("Convex MINLP detected, using NLP-BB (nonlinear Branch and Bound)")
            return _solve_nlp_bb(
                model,
                time_limit,
                gap_tolerance,
                batch_size,
                strategy,
                max_nodes,
                t_start,
                nlp_solver,
                skip_convex_check=skip_convex_check,
                initial_point=initial_point,
                lazy_constraints=lazy_constraints,
                incumbent_callback=incumbent_callback,
                node_callback=node_callback,
                in_tree_presolve_stride=in_tree_presolve_stride,
                in_tree_presolve_repr=_model_repr,
            )

    # --- Extract variable info ---
    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)

    # --- Root presolve: FBBT + integer-bound rounding before tree creation ---
    t_rust_start = time.perf_counter()
    from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

    lb, ub, root_infeasible, _ = tighten_root_bounds_with_fbbt(
        model,
        lb,
        ub,
        int_offsets,
        int_sizes,
        model_repr=_model_repr,
    )
    rust_time += time.perf_counter() - t_rust_start
    if root_infeasible:
        wall_time = time.perf_counter() - t_start
        return SolveResult(
            status="infeasible",
            objective=None,
            bound=None,
            gap=None,
            x=None,
            wall_time=wall_time,
            node_count=0,
            rust_time=rust_time,
            jax_time=jax_time,
            python_time=wall_time - rust_time - jax_time,
        )

    # --- Root OBBT over the McCormick relaxation (range reduction) ---
    # For nonconvex models the spatial B&B solves an LP-form McCormick
    # relaxation at every node; tightening the root box here strengthens that
    # envelope for the *entire* tree. Unlike the per-incumbent linear-only OBBT
    # below, this min/max-es each variable over the full relaxation polytope,
    # so it reduces ranges even when the only constraints are nonlinear (the LP
    # polytope is a valid outer approximation, so every tightening is sound).
    # Skipped for known-convex models (handled by the convex/NLP path) and for
    # pure-integer models (no continuous variable to spatial-branch on).
    _obbt_has_continuous = any(v.var_type == VarType.CONTINUOUS for v in model._variables)
    _obbt_known_convex = _root_convexity_known and _root_is_convex
    if (
        bool(kwargs.get("obbt_at_root", True))
        and model._objective is not None
        and _obbt_has_continuous
        and not _obbt_known_convex
        and n_vars <= 500
    ):
        # OBBT wall time falls into python_time (computed as the remainder at
        # the end of the solve), so no separate timer is tracked here.
        try:
            from discopt._jax.obbt import obbt_tighten_root

            _obbt_budget = min(max(time_limit * 0.1, 2.0), 15.0)
            _obbt_res = obbt_tighten_root(
                model,
                lb,
                ub,
                rounds=3,
                deadline=time.perf_counter() + _obbt_budget,
                superposition=(relaxation_arithmetic == "superposition"),
                prefer_pounce=nlp_solver == "pounce",
            )
            if _obbt_res.infeasible:
                wall_time = time.perf_counter() - t_start
                return SolveResult(
                    status="infeasible",
                    objective=None,
                    bound=None,
                    gap=None,
                    x=None,
                    wall_time=wall_time,
                    node_count=0,
                    rust_time=rust_time,
                    jax_time=jax_time,
                    python_time=wall_time - rust_time - jax_time,
                )
            if _obbt_res.n_tightened > 0:
                lb = np.maximum(lb, _obbt_res.lb)
                ub = np.minimum(ub, _obbt_res.ub)
                logger.info(
                    "Root OBBT tightened %d bounds over %d sweep(s) (%.2fs)",
                    _obbt_res.n_tightened,
                    _obbt_res.n_rounds,
                    _obbt_res.total_lp_time,
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Root OBBT failed: %s", e)

    # Snapshot the root-global box (root FBBT + non-cutoff root OBBT only) so a
    # rigorous root MILP-relaxation fallback bound can be computed at the end if
    # the spatial tree yields nothing usable. Taken here, before the search
    # loop's incumbent-cutoff OBBT, whose bounds are not valid for a global bound.
    _root_lb_snapshot = np.asarray(lb, dtype=np.float64).copy()
    _root_ub_snapshot = np.asarray(ub, dtype=np.float64).copy()

    # --- Create PyTreeManager (Rust) ---
    t_rust_start = time.perf_counter()
    tree = PyTreeManager(
        n_vars,
        lb.tolist(),
        ub.tolist(),
        int_offsets,
        int_sizes,
        strategy,
    )
    tree.initialize()
    rust_time += time.perf_counter() - t_rust_start

    # --- Compile NLP evaluator ---
    t_jax_start = time.perf_counter()
    evaluator = _make_evaluator(model)
    jax_time += time.perf_counter() - t_jax_start

    # --- Infer constraint bounds ---
    cl_list, cu_list = _infer_constraint_bounds(model, evaluator)
    constraint_bounds = list(zip(cl_list, cu_list)) if cl_list else None

    # --- Prepare cut generation if enabled ---
    _generate_cuts = None
    _bilinear_terms = None
    _constraint_senses = None
    _cut_pool = None
    if cutting_planes:
        from discopt._jax.cutting_planes import (
            CutPool,
            detect_bilinear_terms,
            generate_cuts_at_node,
        )

        _generate_cuts = generate_cuts_at_node
        _bilinear_terms = detect_bilinear_terms(model)
        _cut_pool = CutPool(max_cuts=500)
        _constraint_senses = [c.sense for c in model._constraints if isinstance(c, Constraint)]

    # --- Lazy constraint callback requires a cut pool ---
    if lazy_constraints is not None and _cut_pool is None:
        from discopt._jax.cutting_planes import CutPool

        _cut_pool = CutPool(max_cuts=500)

    # --- Convexity detection (Phase E) ---
    # Use the expression DAG convexity detector to:
    # (E2) Skip relaxation overhead for fully convex subproblems
    # (E3) Enable OA cuts per-constraint (not just for affine constraints)
    _model_is_convex = False
    _oa_enabled = False
    _convex_constraint_mask = None
    if _root_convexity_known:
        _model_is_convex = _root_is_convex
        _convex_constraint_mask = _root_constraint_mask or []
        if _model_is_convex:
            logger.info("Model detected as convex — NLP solutions are valid lower bounds")
        if cutting_planes and any(_convex_constraint_mask):
            _oa_enabled = True
    else:
        _root_convexity_known, _root_is_convex, _root_constraint_mask = _classify_model_convexity(
            model
        )
        if _root_convexity_known:
            _model_is_convex = _root_is_convex
            _convex_constraint_mask = _root_constraint_mask or []
            if _model_is_convex:
                logger.info("Model detected as convex — NLP solutions are valid lower bounds")
            if cutting_planes and any(_convex_constraint_mask):
                _oa_enabled = True
        else:
            if cutting_planes and model._constraints:
                _convex_constraint_mask = [False] * len(model._constraints)

    # Per-node certificate refresh imported lazily so the main
    # classification import above stays the single-point-of-truth for
    # "convexity is available at all". ``None`` disables per-node
    # refresh below (the root mask is then used verbatim).
    _refresh_mask: Any = None
    try:
        from discopt._jax.convexity import refresh_convex_mask as _refresh_mask_import

        _refresh_mask = _refresh_mask_import
    except Exception:
        pass

    # Enable nonconvex spatial branching so integer-feasible nodes are not
    # prematurely fathomed.  The NLP local optimum at such a node may not
    # be the global optimum of the continuous subproblem.
    if not _model_is_convex:
        tree.set_nonconvex(True)
    _gap_certified = True

    # Sense-derived negation flag for the internal (minimization) B&B. Unlike
    # ``_mc_negate`` below — which is only assigned correctly inside the
    # McCormick "nlp"/"midpoint" setup — this is valid in every relaxation
    # mode, so the interval bound stays sound for maximization models too.
    from discopt.modeling.core import ObjectiveSense as _ObjectiveSense

    _obj_negate = (
        model._objective is not None and model._objective.sense == _ObjectiveSense.MAXIMIZE
    )

    # --- Default Ipopt options ---
    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", 3000)
    opts.setdefault("tol", 1e-7)

    # --- Augmented constraint function with cuts (updated each iteration) ---
    _augmented_evaluator = None

    # --- AlphaBB convexification for nonconvex models ---
    # (E2) Skip alphaBB entirely for convex models — NLP gives valid bounds.
    # Lever 3 (issue #194): the alpha estimate (jax.hessian over a sample grid)
    # is a ~2s one-time root cost. It is only needed when alphaBB is the bound
    # source — i.e. when the McCormick LP relaxer is NOT active. So we DEFER it
    # here and compute it just below, gated on ``_mc_lp_relaxer is None`` (set by
    # the relaxer block). When the LP relaxer supplies every node's valid dual
    # bound, alphaBB would only ever be a fallback on nodes the LP relaxer
    # declines; on the corpus that never changes the certified result (A/B: 0
    # regressions), so skipping the estimate is sound and removes the ~2s setup.
    _alphabb_alpha = None
    _use_alphabb = False
    _alphabb_eligible = n_vars <= 50 and not _model_is_convex and hasattr(evaluator, "_obj_fn")

    # --- McCormick relaxation bounds ---
    _mc_obj_eval = None  # BatchRelaxationEvaluator for midpoint bounds
    _mc_obj_relax_fn = None  # raw relaxation fn for NLP bounds
    _mc_con_relax_fns: list[Callable] | None = None
    _mc_obj_relax_fn_np = None  # numpy-backed relaxation, when supported
    _mc_con_relax_fns_np: list[Callable] | None = None
    _mc_con_senses = None
    _mc_negate = False
    _mc_mode = mccormick_bounds
    _mc_lp_relaxer = None  # MccormickLPRelaxer instance when _mc_mode == "lp"

    if _mc_mode == "auto":
        # The McCormick "nlp" objective bound is a *valid* dual bound only when
        # the objective relaxation is a convex underestimator. The bound solver
        # evaluates the compiled relaxation at x_cv == x_cc (see
        # mccormick_nlp.solve_mccormick_relaxation_nlp / McCormickRelaxation-
        # Evaluator), where every McCormick rule is tight, so the minimized
        # surface coincides with the original objective and is convex iff the
        # model is convex. For a nonconvex model a *local* NLP solve of that
        # surface can return a value ABOVE the true optimum and certify it as a
        # lower bound — a silent false-optimal (issue #120: nvs16 certified the
        # local 14.20 over the true optimum 0.70).
        #
        # For a nonconvex model with continuous variables, the LP-form McCormick
        # relaxation IS a rigorous valid dual bound: it is a polyhedral OUTER
        # approximation of the nonconvex feasible set, so its LP optimum is a
        # true lower bound, and an infeasible relaxation is a rigorous fathom.
        # Combined with spatial branching this closes the gap and *certifies*
        # optimality on bilinear/multilinear models (e.g. min u+v s.t. u*v>=5),
        # where the "none" path finds the optimum but cannot prove it. The "lp"
        # setup below falls back to "none"/"nlp" (and the nonconvex-"nlp" guard
        # to "none") when the model has no bilinear terms or the relaxer cannot
        # be built, and any per-node term it cannot relax safely yields an LP
        # "error" (no bound) rather than an invalid one — so this never trades
        # soundness for the tighter bound. Pure-integer nonconvex models have
        # nothing to spatial-branch on, so they keep the rigorous alphaBB
        # underestimator ("none"). The flag is taken over the *original*
        # decision variables (captured before the factorable lift) so dependent
        # continuous lift aux vars do not spuriously route a combinatorial model
        # onto the spatial path and stall its incumbent search (nvs16).
        _has_continuous_var = _origin_has_continuous_var
        if not _model_is_convex and _has_continuous_var and model._objective is not None:
            _mc_mode = "lp"
        else:
            _mc_mode = "none"

    if _mc_mode == "lp" and model._objective is not None:
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        # The spatial path needs at least one variable it can branch on: a
        # finite-box continuous variable to bisect, or (issue #194) an integer
        # variable in a nonlinear term whose domain can be partitioned to close a
        # McCormick gap. A model with neither — e.g. integer-only or
        # unbounded-slack-only where the integers occur only linearly — would
        # dead-end after the root and a no-incumbent exhaustion would be *falsely*
        # certified (the worst failure class, issue #185), so it falls back to the
        # NLP/alphaBB path below.
        try:
            _mc_lp_relaxer = MccormickLPRelaxer(
                model,
                superposition=(relaxation_arithmetic == "superposition"),
                psd_cuts=psd_cuts,
                rlt_cuts=rlt_cuts,
            )
        except Exception as e:
            logger.warning("McCormick LP relaxer setup failed: %s", e)
            _mc_lp_relaxer = None
            _mc_mode = "none"
        else:
            if not _mc_lp_relaxer.has_relaxable_nonlinearity:
                # No nonlinear term the LP relaxer can bound (products,
                # monomials, or fractional powers) → standard LP relaxation
                # = the model itself for linear parts. Drop back to NLP.
                # NB: monomial/fractional-power-only nonconvex models DO get
                # a valid LP dual bound here (issue #120 fix); gating on
                # has_bilinear alone wrongly routed them to the unsound "nlp"
                # bound.
                _mc_lp_relaxer = None
                _mc_mode = "nlp"
            else:
                # Structure-gated cut policy (cuts="auto", the default): the A/B
                # sweep showed RLT dominates on QCQP *with* linear constraints, PSD
                # on box-QP (no constraints), and stacking the two is
                # counter-productive. So pick exactly one by structure, gated by
                # size, never both. An explicit psd_cuts/rlt_cuts flag takes
                # precedence (the user opted into a specific family); cuts="manual"
                # disables the policy entirely.
                if cuts == "auto" and not psd_cuts and not rlt_cuts:
                    _apply_auto_cut_policy(model, _mc_lp_relaxer)
                # A node is spatial-branchable if there is a finite-box continuous
                # variable to bisect, OR an integer variable in a nonlinear term
                # whose domain can be partitioned (issue #194). Register those
                # integer columns so the Rust tree spatial-branches on them
                # instead of dead-ending and falsely certifying.
                _int_cols = {
                    j for off, sz in zip(int_offsets, int_sizes) for j in range(off, off + int(sz))
                }
                _nl_int_cols = sorted(_int_cols & set(_mc_lp_relaxer.nonlinear_columns))
                _has_branchable = _origin_has_finite_continuous_var or bool(_nl_int_cols)
                if not _has_branchable:
                    logger.info(
                        "McCormick LP requested but model has no spatial-branchable "
                        "variable (no finite-box continuous var, no nonlinear-term "
                        "integer); falling back to NLP relaxation."
                    )
                    _mc_lp_relaxer = None
                    _mc_mode = "nlp"
                else:
                    if _nl_int_cols:
                        tree.set_spatial_integer_cols(np.asarray(_nl_int_cols, dtype=np.int64))
                    # Root probe: keep the LP relaxer only if it actually yields
                    # a valid objective bound (or a rigorous infeasibility proof)
                    # at the root box. When the objective is not LP-linearizable
                    # the relaxer falls back to a feasibility objective and
                    # returns no bound; engaging it would then SKIP the root NLP
                    # multistart and suppress the alphaBB/interval floor, losing
                    # a bound that path would otherwise produce. Falling back to
                    # "none" here preserves the rigorous alphaBB underestimator
                    # for those models while keeping the LP bound for the ones it
                    # can actually relax.
                    try:
                        _probe_lb, _probe_ub = flat_variable_bounds(model)
                        # Bound the probe's MILP relaxation solve to a SMALL slice
                        # of the budget. Without any limit it inherited
                        # solve_at_node's default time_limit=None -> the Rust MILP
                        # B&B ran unbounded (up to max_nodes=1e6), solving the root
                        # relaxation to optimality; on a hard MINLP root (e.g.
                        # du-opt) that single *discarded* probe consumed the whole
                        # wall-clock (~77s vs a 25s limit) before the spatial search
                        # even began. The probe only needs to learn whether the
                        # relaxer yields a usable bound / infeasibility proof — the
                        # Rust solver returns a valid dual bound even on an early
                        # timeout — so a brief cap suffices and leaves the bulk of
                        # the budget for the actual B&B. Mirror the OBBT root-budget
                        # heuristic above, never exceeding the live remaining time.
                        _probe_remaining = time_limit - (time.perf_counter() - t_start)
                        _probe_budget = min(
                            max(time_limit * 0.1, 2.0),
                            max(_probe_remaining, _DEADLINE_NODE_FLOOR_S),
                        )
                        _probe = _mc_lp_relaxer.solve_at_node(
                            _probe_lb, _probe_ub, time_limit=_probe_budget
                        )
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug("McCormick LP root probe failed: %s", e)
                        _probe = None
                    _probe_useful = _probe is not None and (
                        _probe.status == "infeasible" or _probe.lower_bound is not None
                    )
                    if not _probe_useful:
                        _mc_lp_relaxer = None
                        _mc_mode = "none"

    # AlphaBB alpha estimate (lever 3, issue #194), deferred from above: compute
    # it only when the LP relaxer is NOT the bound source. When the LP relaxer is
    # active it supplies every node's valid dual bound, so the ~2s alpha estimate
    # (and the per-node alphaBB it enables) is skipped. ``DISCOPT_ALPHABB_WITH_LP=1``
    # forces the estimate even under the LP relaxer (A/B / fallback safety).
    _alphabb_force = os.environ.get("DISCOPT_ALPHABB_WITH_LP", "0") == "1"
    if _alphabb_eligible and (_mc_lp_relaxer is None or _alphabb_force):
        try:
            _alphabb_alpha = np.asarray(
                _estimate_alpha_jax(evaluator._obj_fn, lb, ub, n_samples=100)
            )
            _use_alphabb = bool(np.any(_alphabb_alpha > 1e-8))
        except (ValueError, ArithmeticError, RuntimeError) as e:
            logger.debug("JAX alphaBB estimation failed: %s", e)

    # Soundness guard (issue #120): the McCormick "nlp" objective bound is a
    # valid dual bound only for convex models. The bound solver evaluates the
    # compiled relaxation at x_cv == x_cc, where every McCormick rule is tight,
    # so it minimizes the original objective *locally*; for a nonconvex model
    # that local optimum can lie ABOVE the true optimum and be certified as a
    # lower bound (silent false-optimal). "auto" already avoids "nlp" for
    # nonconvex models above; this also catches an explicit
    # mccormick_bounds="nlp" and the lp→nlp fallbacks (e.g. pure-integer
    # nonconvex models). Fall back to the rigorous alphaBB underestimator.
    if _mc_mode == "nlp" and not _model_is_convex:
        logger.warning(
            "McCormick 'nlp' objective bound is not a valid dual bound for "
            "nonconvex models (issue #120); falling back to the alphaBB "
            "underestimator. Use mccormick_bounds='lp' for a valid spatial "
            "relaxation on models with continuous variables."
        )
        _mc_mode = "none"

    if _mc_mode in ("midpoint", "nlp") and model._objective is not None:
        from discopt._jax.batch_evaluator import BatchRelaxationEvaluator
        from discopt._jax.relaxation_compiler import (
            compile_constraint_relaxation,
            compile_objective_relaxation,
        )
        from discopt.modeling.core import ObjectiveSense

        try:
            _mc_obj_relax_fn = compile_objective_relaxation(
                model,
                partitions=partitions,
                mode=_relax_mode,
                learned_registry=_learned_registry,
                arithmetic=relaxation_arithmetic,
            )
            _mc_obj_eval = BatchRelaxationEvaluator(_mc_obj_relax_fn, n_vars)
            _mc_negate = model._objective.sense == ObjectiveSense.MAXIMIZE

            if _mc_mode == "nlp" and model._constraints:
                _mc_con_relax_fns = []
                _mc_con_senses = []
                for c in model._constraints:
                    if isinstance(c, Constraint):
                        _mc_con_relax_fns.append(
                            compile_constraint_relaxation(
                                c,
                                model,
                                partitions=partitions,
                                mode=_relax_mode,
                                learned_registry=_learned_registry,
                                arithmetic=relaxation_arithmetic,
                            )
                        )
                        _mc_con_senses.append(c.sense)

            # Build numpy-backed relaxations alongside JAX when the model
            # uses only supported ops. Skip when piecewise/learned/
            # chebyshev/taylor modes are active — those paths produce
            # tighter relaxations that the numpy backend doesn't replicate.
            if (
                _mc_mode == "nlp"
                and partitions == 0
                and _relax_mode == "standard"
                and relaxation_arithmetic == "mccormick"
            ):
                try:
                    from discopt._numpy.relaxation_compiler import (
                        compile_constraint_relaxation as _np_compile_con,
                    )
                    from discopt._numpy.relaxation_compiler import (
                        compile_objective_relaxation as _np_compile_obj,
                    )
                    from discopt._numpy.relaxation_compiler import (
                        supported_for_model,
                    )

                    if supported_for_model(model):
                        _mc_obj_relax_fn_np = _np_compile_obj(model)
                        if model._constraints:
                            _mc_con_relax_fns_np = []
                            for c in model._constraints:
                                if isinstance(c, Constraint):
                                    _mc_con_relax_fns_np.append(_np_compile_con(c, model))
                except (NotImplementedError, Exception) as e:
                    logger.debug("numpy McCormick backend unavailable: %s", e)
                    _mc_obj_relax_fn_np = None
                    _mc_con_relax_fns_np = None
        except Exception as e:
            logger.warning("McCormick relaxation setup failed: %s", e)
            _mc_obj_eval = None
            _mc_obj_relax_fn = None
            _mc_obj_relax_fn_np = None
            _mc_con_relax_fns_np = None

    # --- Warm-start: inject user-provided initial solution as incumbent ---
    if initial_point is not None:
        ws_obj = float(evaluator.evaluate_objective(initial_point))
        # Check integer feasibility of the warm-start point
        ws_int_feas = True
        for off, sz in zip(int_offsets, int_sizes):
            for j in range(off, off + sz):
                if abs(initial_point[j] - round(initial_point[j])) > 1e-5:
                    ws_int_feas = False
                    break
            if not ws_int_feas:
                break
        if ws_int_feas and np.isfinite(ws_obj) and ws_obj < _SENTINEL_THRESHOLD:
            ws_con_feas = not cl_list or _check_constraint_feasibility(
                evaluator, initial_point, cl_list, cu_list
            )
            if ws_con_feas:
                tree.inject_incumbent(initial_point, ws_obj)
                logger.info("Warm-start incumbent injected: obj=%.6g", ws_obj)
            else:
                logger.info(
                    "Warm-start point is integer-feasible but violates "
                    "constraints, using as NLP starting point only"
                )
        else:
            logger.info(
                "Warm-start point is not integer-feasible, using as NLP starting point only"
            )

    # --- Feasibility pump at root ---
    # Try to find an integer-feasible incumbent before B&B starts.
    _fp_ran = False

    # Re-attach the captured modeler start (by name) to the post-reform working
    # model so ``_gams_initial_seed`` can build a root subnlp seed from it. Only
    # set when the reforms did not already carry it forward; a no-op when no
    # start was provided.
    if _captured_gams_initial_values and not getattr(model, "_gams_initial_values", None):
        model._gams_initial_values = _captured_gams_initial_values

    # --- SubNLP primal heuristic state ---
    _subnlp_backend_fn = None
    _subnlp_calls = 0
    _subnlp_feasible = 0
    _subnlp_incumbent_updates = 0
    # Best incumbent value the cutoff-tightening phases (C/C3) have already acted
    # on. They fire whenever the incumbent strictly improves below this — from
    # ANY source, including the sub-NLP / binary-seed heuristics that inject
    # directly and are not counted in proc_stats["incumbent_updates"].
    _last_tighten_inc = np.inf
    if subnlp_enabled:
        try:
            from typing import cast

            from discopt.solvers.nlp_backend import Backend, get_nlp_solver

            _subnlp_backend_fn = get_nlp_solver(cast(Backend, subnlp_backend))
        except ImportError as _e:
            logger.debug("SubNLP backend unavailable, disabling: %s", _e)
            _subnlp_backend_fn = None

    # --- B&B loop ---
    # McCormick NLP is expensive (one IPM per node). Run it every N iterations
    # and use cheap midpoint bounds in between. Period=1 means every iteration.
    _mc_nlp_period = 5  # run McCormick NLP every 5th iteration
    iteration = 0
    _deadline = t_start + time_limit

    # Objective-gating priority branching (issue #184). Opt-in via
    # ``DISCOPT_OBJ_BRANCH_PRIORITY=1``: branch the binaries that gate the
    # objective's nonlinear terms first so the global bound can climb off a
    # structural 0 (see _branch_priority_integer_vars). Empty when disabled or
    # when the model has no such gating integers, leaving branching unchanged.
    _branch_priority_vars: frozenset[int] = frozenset()
    if os.environ.get("DISCOPT_OBJ_BRANCH_PRIORITY", "0") == "1":
        try:
            _branch_priority_vars = _branch_priority_integer_vars(model)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("objective-gating priority detection failed: %s", e)
        if _branch_priority_vars:
            logger.info(
                "Objective-gating priority branching: %d integer var(s) %s",
                len(_branch_priority_vars),
                sorted(_branch_priority_vars),
            )

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Update per-iteration time budget for NLP subproblem solves (issue #5).
        remaining = time_limit - elapsed
        opts["max_wall_time"] = max(remaining, _DEADLINE_NODE_FLOOR_S)

        # Export batch from Rust tree
        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        node_infeasible_mask = np.zeros(n_batch, dtype=bool)

        # Apply the current global box to each exported node (issue: cutoff
        # tightening was dead). Phase C (incumbent-cutoff OBBT) and Phase C3
        # (incumbent-cutoff FBBT) shrink the global lb/ub once an incumbent is
        # found, but the Rust tree has no bound-update API and re-exports nodes
        # with their original (wide) boxes. Intersecting each node box with the
        # current global lb/ub here is what makes that cutoff tightening
        # actually prune: a region cut away only holds solutions no better than
        # the incumbent, so dropping it is sound (open nodes keep lb < inc, so
        # best_bound stays a valid lower bound <= the true optimum). Before any
        # incumbent, lb/ub equal the root box and this is a no-op.
        _gl = np.asarray(lb, dtype=np.float64)
        _gu = np.asarray(ub, dtype=np.float64)
        for i in range(n_batch):
            bl = np.maximum(np.asarray(batch_lb[i], dtype=np.float64), _gl)
            bu = np.minimum(np.asarray(batch_ub[i], dtype=np.float64), _gu)
            if np.any(bl > bu + 1e-9):
                # Empty intersection: node lies entirely outside the cutoff box,
                # so its subtree cannot improve on the incumbent — fathom it.
                node_infeasible_mask[i] = True
                continue
            batch_lb[i] = bl.tolist()
            batch_ub[i] = bu.tolist()

        # Tighten node bounds via constraint propagation (FBBT).
        if cl_list:
            for i in range(n_batch):
                if node_infeasible_mask[i]:
                    continue
                node_lb_i = np.array(batch_lb[i])
                node_ub_i = np.array(batch_ub[i])
                t_lb, t_ub, node_infeasible = _tighten_node_bounds_with_status(
                    evaluator, node_lb_i, node_ub_i, cl_list, cu_list
                )
                if node_infeasible:
                    node_infeasible_mask[i] = True
                    continue
                batch_lb[i] = t_lb.tolist()
                batch_ub[i] = t_ub.tolist()

        # Solve NLP relaxation for each node in the batch
        t_jax_start = time.perf_counter()

        # Use augmented evaluator with cuts if available
        _active_evaluator = evaluator
        _active_cb = constraint_bounds
        if _cut_pool is not None and len(_cut_pool) > 0:
            _augmented_evaluator = _AugmentedEvaluator(evaluator, _cut_pool)
            _active_evaluator = _augmented_evaluator
            _active_cb = _augmented_evaluator.get_augmented_constraint_bounds(constraint_bounds)

        # POUNCE batches node NLPs via solve_nlp_batch (Phase A, discopt#97).
        # It uses Python callbacks, so it needs no JAX _obj_fn on the evaluator.
        # The callback path is GIL-bound, so only batch when the per-node problem
        # is large enough to amortize it (_POUNCE_BATCH_MIN_VARS); smaller nodes
        # stay on the serial path.
        _use_pounce_batch = nlp_solver == "pounce" and n_vars >= _POUNCE_BATCH_MIN_VARS
        if _use_pounce_batch and n_batch > 1:
            result_ids, result_lbs, result_sols, result_feas, _batch_trusted = _solve_batch_pounce(
                _active_evaluator,
                batch_lb,
                batch_ub,
                batch_ids,
                n_vars,
                _active_cb,
                opts,
                batch_psols=batch_psols,
                multistart=_POUNCE_BATCH_MULTISTART,
                convex=_model_is_convex,
            )
            # A convex node whose relaxation objective is not KKT-valid (and
            # could not be polished) is not a valid lower bound; decertify the
            # gap rather than trust it (roadmap P0.3). Bounds are left as-is.
            if _model_is_convex and not bool(np.all(_batch_trusted)):
                _gap_certified = False
            # Constraint feasibility post-check for batch IPM results.
            # When the IPM solution violates constraints (e.g. due to hitting
            # the iteration limit), mark the node as infeasible (SENTINEL).
            # This prevents the invalid solution from becoming the incumbent
            # and causes the Rust tree to prune the node.
            if cl_list:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        if not _check_constraint_feasibility(
                            _active_evaluator,
                            result_sols[i],
                            cl_list,
                            cu_list,
                        ):
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            logger.debug(
                                "Batch node %d: IPM solution violates "
                                "constraints, marking infeasible",
                                int(batch_ids[i]),
                            )
            # For nonconvex problems, NLP objective is NOT a valid lower
            # bound (local minima can exceed the global optimum).  Reset ALL
            # non-sentinel nodes to -inf so only convex relaxation bounds are
            # used.  For integer-feasible nodes, inject the NLP solution as
            # an incumbent candidate via tree.inject_incumbent() and let the
            # Rust tree continue spatial branching on continuous variables.
            if not _model_is_convex:
                _nlp_obj_backup = result_lbs.copy()
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        sol_is_int_feas = _is_integer_feasible_solution(
                            result_sols[i], int_offsets, int_sizes
                        )
                        if sol_is_int_feas:
                            # Inject NLP solution as incumbent candidate.
                            # The Rust tree will update its incumbent if this
                            # objective improves on the current best.
                            nlp_obj = float(_nlp_obj_backup[i])
                            if np.isfinite(nlp_obj):
                                tree.inject_incumbent(result_sols[i].copy(), nlp_obj)
                        # Reset ALL nonconvex nodes to -inf; convex bounds
                        # computed below will provide valid lower bounds.
                        result_lbs[i] = -np.inf
            # Tighten lower bounds with alphaBB underestimator
            if _use_alphabb:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        try:
                            node_lb_i = np.array(batch_lb[i])
                            node_ub_i = np.array(batch_ub[i])
                            relax_lb = _compute_alphabb_bound(
                                evaluator, node_lb_i, node_ub_i, _alphabb_alpha
                            )
                            result_lbs[i] = max(result_lbs[i], relax_lb)
                        except (ValueError, ArithmeticError, RuntimeError) as e:
                            logger.debug("alphaBB bound failed at node %d: %s", i, e)
            # Cheap, always-valid interval-arithmetic bound. Runs every
            # iteration so nonconvex nodes never sit at -inf between the
            # periodic McCormick-NLP solves (which only fire every
            # _mc_nlp_period iterations). max() of valid bounds stays valid,
            # so this only ever tightens the global lower bound — enabling
            # certified "optimal" status without weakening soundness.
            if not _model_is_convex and model._objective is not None:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        iv_lb = _compute_interval_bound(
                            model, batch_lb[i], batch_ub[i], _obj_negate
                        )
                        if np.isfinite(iv_lb):
                            result_lbs[i] = max(result_lbs[i], iv_lb)
                        # Original (un-distributed) objective over the original-
                        # variable sub-box — a tight enclosure the lifted model's
                        # distributed bilinear form cannot give (see serial path).
                        if _prereform_model is not None:
                            pr_lb = _compute_interval_bound(
                                _prereform_model, batch_lb[i], batch_ub[i], _obj_negate
                            )
                            if np.isfinite(pr_lb):
                                result_lbs[i] = max(result_lbs[i], pr_lb)
            # Tighten lower bounds with McCormick relaxation
            if _mc_obj_eval is not None:
                try:
                    import jax.numpy as jnp

                    lb_jax = jnp.array(batch_lb, dtype=jnp.float64)
                    ub_jax = jnp.array(batch_ub, dtype=jnp.float64)
                    # Use NLP mode only periodically to avoid per-node IPM overhead.
                    # Midpoint mode is NOT a valid lower bound (cv(mid) >= min f(x)
                    # is not guaranteed), so skip McCormick on non-NLP iterations.
                    _use_mc_nlp = (
                        _mc_mode == "nlp"
                        and _mc_obj_relax_fn is not None
                        and (iteration == 0 or iteration % _mc_nlp_period == 0)
                    )
                    if _use_mc_nlp:
                        from discopt._jax.mccormick_nlp import solve_mccormick_batch

                        assert _mc_obj_relax_fn is not None
                        mc_lbs = np.asarray(
                            solve_mccormick_batch(
                                _mc_obj_relax_fn,
                                _mc_con_relax_fns,
                                _mc_con_senses,
                                lb_jax,
                                ub_jax,
                                negate=_mc_negate,
                                deadline=t_start + time_limit,
                                obj_relax_fn_numpy=_mc_obj_relax_fn_np,
                                con_relax_fns_numpy=_mc_con_relax_fns_np,
                            )
                        )
                    elif _mc_mode != "nlp":
                        from discopt._jax.mccormick_nlp import (
                            evaluate_midpoint_bound_batch,
                        )

                        assert _mc_obj_relax_fn is not None
                        mc_lbs = np.asarray(
                            evaluate_midpoint_bound_batch(
                                _mc_obj_relax_fn,
                                lb_jax,
                                ub_jax,
                                negate=_mc_negate,
                            )
                        )
                    else:
                        mc_lbs = None
                    if mc_lbs is not None:
                        for i in range(n_batch):
                            if result_lbs[i] < _SENTINEL_THRESHOLD and np.isfinite(mc_lbs[i]):
                                result_lbs[i] = max(result_lbs[i], float(mc_lbs[i]))
                except (ValueError, ArithmeticError, RuntimeError) as e:
                    logger.debug("Batch McCormick bound failed: %s", e)
            # LP-form McCormick: lift bilinears, solve as LP via HiGHS.
            # Per-node, ~20ms for problems with tens of bilinear terms.
            if _mc_lp_relaxer is not None:
                for i in range(n_batch):
                    if node_infeasible_mask[i]:
                        continue
                    # Deadline enforcement (time_limit overrun fix). A single
                    # McCormick LP solve has an irreducible ~1s floor on a large
                    # lifted relaxation, so clamping the per-node budget to
                    # _DEADLINE_NODE_FLOOR_S still lets a batch of N nodes run
                    # ~N s past the cap. Once the deadline has passed, stop
                    # solving the rest of the batch: leave each remaining node's
                    # bound at -inf (unpruned, it is the default) and decertify
                    # the gap, mirroring the serial path. -inf never fabricates a
                    # false "optimal"; the loop exits at the next batch top.
                    _node_remaining = _deadline - time.perf_counter()
                    if _node_remaining <= 0.0:
                        if not _model_is_convex:
                            _gap_certified = False
                        continue
                    nlp_failed = result_lbs[i] >= _SENTINEL_THRESHOLD
                    try:
                        mc_res = _mc_lp_relaxer.solve_at_node(
                            np.asarray(batch_lb[i]),
                            np.asarray(batch_ub[i]),
                            time_limit=max(_node_remaining, _DEADLINE_NODE_FLOOR_S),
                        )
                    except Exception as e:
                        logger.debug("McCormick LP failed at node %d: %s", i, e)
                        continue
                    if mc_res.status == "infeasible":
                        # Rigorous fathom: the McCormick LP is a valid outer
                        # relaxation, so an empty relaxed feasible set proves the
                        # node's subtree infeasible. Mark it (sentinel + mask) so
                        # the finalize block below does not decertify the gap.
                        node_infeasible_mask[i] = True
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        result_feas[i] = False
                        continue
                    if mc_res.lower_bound is not None and np.isfinite(mc_res.lower_bound):
                        if nlp_failed:
                            # A failed / locally-infeasible NLP solve is not an
                            # infeasibility proof. Adopt the LP bound + LP point
                            # so the node branches instead of being pruned via
                            # the sentinel (mirrors the serial path).
                            result_lbs[i] = float(mc_res.lower_bound)
                            if mc_res.x is not None:
                                result_sols[i] = mc_res.x
                            result_feas[i] = False
                        else:
                            result_lbs[i] = max(result_lbs[i], float(mc_res.lower_bound))
            if not _model_is_convex:
                for i in range(n_batch):
                    if result_lbs[i] == -np.inf:
                        _gap_certified = False
                    elif not np.isfinite(result_lbs[i]):
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                    elif result_lbs[i] >= _SENTINEL_THRESHOLD and not node_infeasible_mask[i]:
                        # Soundness guard (issue #27a, batch parity with the
                        # serial path): a node carrying the failure sentinel
                        # without an FBBT infeasibility proof is pruned without
                        # being proven suboptimal — the NLP merely failed or
                        # was locally infeasible. Decertify the gap so the
                        # result downgrades to "feasible" instead of claiming
                        # a certified optimum.
                        _gap_certified = False
        else:
            result_ids = np.empty(n_batch, dtype=np.int64)
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.empty(n_batch, dtype=bool)

            # Lever 2 (#194): when the LP relaxer supplies each node's dual bound
            # and the model is nonconvex, the per-node NLP is purely a primal
            # heuristic — its objective is NOT a valid bound there (the McCormick
            # LP is). Gate it to a stride so it runs on a fraction of nodes; the
            # LP relaxer still bounds every node, and the SubNLP heuristic (below)
            # plus the strided NLP find incumbents. DISCOPT_NODE_NLP_STRIDE=1
            # restores the per-node NLP. Convex / no-LP-relaxer paths are
            # unaffected (the NLP bound matters there, so it runs every node).
            _nlp_stride = int(os.environ.get("DISCOPT_NODE_NLP_STRIDE", "4"))
            _gate_node_nlp = _mc_lp_relaxer is not None and not _model_is_convex and _nlp_stride > 1

            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                # Deadline enforcement (time_limit overrun fix). The budget set
                # at the batch top is the WHOLE remaining time; reusing it for
                # every node lets a batch of N nodes run ~N x over the cap. Re-
                # derive the live remaining budget before each node so the cap
                # is honored. Once past the deadline, skip the solve and record a
                # trivial (-inf) lower bound: -inf leaves the node unpruned (it
                # is the default node bound) and decertifies the gap below, so we
                # stop early without ever fabricating a false "optimal".
                _node_remaining = _deadline - time.perf_counter()
                if _node_remaining <= 0.0:
                    result_ids[i] = int(batch_ids[i])
                    result_lbs[i] = -np.inf
                    lb_clipped = np.clip(node_lb, -_SPC, _SPC)
                    ub_clipped = np.clip(node_ub, -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_clipped + ub_clipped)
                    result_feas[i] = False
                    if not _model_is_convex:
                        _gap_certified = False
                    continue
                opts["max_wall_time"] = max(_node_remaining, _DEADLINE_NODE_FLOOR_S)

                # Strided primal NLP (lever 2): run the per-node NLP only on a
                # fraction of nodes in the gated regime; every node still gets the
                # LP relaxer's bound + primal below.
                _node_nlp_due = (not _gate_node_nlp) or (int(batch_ids[i]) % _nlp_stride == 0)
                nlp_result = None
                if iteration == 0 and _mc_lp_relaxer is None:
                    # The root multistart NLP is the only bound source we have.
                    # When the LP relaxer is active, skip it: the LP block below
                    # supplies the bound + primal and SubNLP turns that into
                    # the incumbent — running multistart NLP here is wasted work.
                    nlp_result = _solve_root_node_multistart(
                        _active_evaluator,
                        node_lb,
                        node_ub,
                        _active_cb,
                        opts,
                        nlp_solver,
                    )
                elif iteration > 0 and _node_nlp_due:
                    # Warm-start from parent solution if available
                    psol_i = np.array(batch_psols[i])
                    if not np.any(np.isnan(psol_i)):
                        # Clip parent solution into child's bounds
                        x0 = np.clip(psol_i, node_lb, node_ub)
                    else:
                        lb_clipped = np.clip(node_lb, -_SPC, _SPC)
                        ub_clipped = np.clip(node_ub, -_SPC, _SPC)
                        x0 = 0.5 * (lb_clipped + ub_clipped)
                    nlp_result = _solve_node_nlp(
                        _active_evaluator,
                        x0,
                        node_lb,
                        node_ub,
                        _active_cb,
                        opts,
                        nlp_solver=nlp_solver,
                    )

                result_ids[i] = int(batch_ids[i])

                if nlp_result is not None and nlp_result.status in (
                    SolveStatus.OPTIMAL,
                    SolveStatus.ITERATION_LIMIT,
                ):
                    nlp_obj = float(nlp_result.objective)
                    nlp_lb = nlp_obj
                    convex_lb = -np.inf  # accumulate valid convex lower bound

                    if _use_alphabb:
                        try:
                            relax_lb = _compute_alphabb_bound(
                                evaluator, node_lb, node_ub, _alphabb_alpha
                            )
                            if _model_is_convex:
                                nlp_lb = max(nlp_lb, relax_lb)
                            convex_lb = max(convex_lb, relax_lb)
                        except (ValueError, ArithmeticError, RuntimeError) as e:
                            logger.debug("alphaBB bound failed: %s", e)
                    # McCormick relaxation bound
                    if _mc_obj_relax_fn is not None:
                        try:
                            import jax.numpy as jnp

                            lb_j = jnp.array(node_lb, dtype=jnp.float64)
                            ub_j = jnp.array(node_ub, dtype=jnp.float64)
                            _use_mc_nlp_serial = _mc_mode == "nlp" and (
                                iteration == 0 or iteration % _mc_nlp_period == 0
                            )
                            if _use_mc_nlp_serial:
                                from discopt._jax.mccormick_nlp import (
                                    solve_mccormick_relaxation_nlp,
                                )

                                mc_lb = solve_mccormick_relaxation_nlp(
                                    _mc_obj_relax_fn,
                                    _mc_con_relax_fns,
                                    _mc_con_senses,
                                    lb_j,
                                    ub_j,
                                    negate=_mc_negate,
                                    deadline=t_start + time_limit,
                                    obj_relax_fn_numpy=_mc_obj_relax_fn_np,
                                    con_relax_fns_numpy=_mc_con_relax_fns_np,
                                )
                            elif _mc_mode != "nlp":
                                from discopt._jax.mccormick_nlp import (
                                    evaluate_midpoint_bound,
                                )

                                mc_lb = evaluate_midpoint_bound(
                                    _mc_obj_relax_fn,
                                    lb_j,
                                    ub_j,
                                    negate=_mc_negate,
                                )
                            else:
                                mc_lb = -np.inf
                            if np.isfinite(mc_lb):
                                if _model_is_convex:
                                    nlp_lb = max(nlp_lb, mc_lb)
                                convex_lb = max(convex_lb, mc_lb)
                        except (ValueError, ArithmeticError, RuntimeError) as e:
                            logger.debug("McCormick bound failed: %s", e)

                    # Cheap, always-valid interval-arithmetic bound. Keeps
                    # nonconvex nodes off -inf between the periodic
                    # McCormick-NLP solves so the global lower bound tightens
                    # every iteration. max() of valid bounds stays valid.
                    if not _model_is_convex:
                        iv_lb = _compute_interval_bound(model, node_lb, node_ub, _obj_negate)
                        if np.isfinite(iv_lb):
                            convex_lb = max(convex_lb, iv_lb)
                        # Tighter enclosure from the un-distributed objective: the
                        # factorable lift turns sum-of-squares into a distributed
                        # bilinear form whose interval bound is uselessly loose,
                        # but the ORIGINAL objective over the original-variable
                        # sub-box (the first _prereform_nvars flat entries) is a
                        # valid, far tighter bound. This is what lets Rosenbrock-
                        # style problems certify (obj = sum of squares >= 0).
                        if _prereform_model is not None:
                            pr_lb = _compute_interval_bound(
                                _prereform_model, node_lb, node_ub, _obj_negate
                            )
                            if np.isfinite(pr_lb):
                                convex_lb = max(convex_lb, pr_lb)

                    # For nonconvex problems: NLP local min is NOT a valid
                    # lower bound (can exceed global opt → premature pruning).
                    # Inject integer-feasible NLP solutions as incumbent
                    # candidates, but import only convex-relaxation lower
                    # bounds into the tree.
                    con_feasible = not cl_list or _check_constraint_feasibility(
                        _active_evaluator, nlp_result.x, cl_list, cu_list
                    )
                    if not con_feasible:
                        nlp_lb = _INFEASIBILITY_SENTINEL
                        logger.debug(
                            "Node %d: NLP solution violates constraints, marking infeasible",
                            int(batch_ids[i]),
                        )
                    elif not _model_is_convex:
                        if _is_integer_feasible_solution(nlp_result.x, int_offsets, int_sizes):
                            if np.isfinite(nlp_obj) and nlp_obj < _SENTINEL_THRESHOLD:
                                tree.inject_incumbent(
                                    np.asarray(nlp_result.x).copy(), float(nlp_obj)
                                )
                        nlp_lb = convex_lb if convex_lb > -np.inf else -np.inf

                    # Guard: NaN and +inf lower bounds corrupt the Rust B&B
                    # tree.  Keep -inf for nonconvex nodes without a valid
                    # relaxation bound; the post-LP finalize block below
                    # decertifies the gap only if no valid bound source (NLP,
                    # alphaBB, McCormick, or the LP relaxer) supplied a finite
                    # lower bound.  Deferring the -inf decertification until
                    # after the LP relaxer mirrors the batch path and lets a
                    # valid LP McCormick bound certify optimality on bilinears.
                    if np.isnan(nlp_lb) or nlp_lb == np.inf:
                        nlp_lb = _INFEASIBILITY_SENTINEL
                    result_lbs[i] = nlp_lb
                    result_sols[i] = nlp_result.x
                    result_feas[i] = False
                else:
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    lb_clipped = np.clip(node_lb, -_SPC, _SPC)
                    ub_clipped = np.clip(node_ub, -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_clipped + ub_clipped)
                    result_feas[i] = False

                # LP-form McCormick bound (lifted bilinears, HiGHS LP).
                # Runs independently of the NLP: provides a valid lower bound
                # and a feasible LP point usable for spatial branching even
                # when the NLP was skipped (root) or returned infeasible /
                # iteration_limit (any node).
                if _mc_lp_relaxer is not None:
                    try:
                        mc_lp_res = _mc_lp_relaxer.solve_at_node(
                            node_lb,
                            node_ub,
                            time_limit=max(_deadline - time.perf_counter(), _DEADLINE_NODE_FLOOR_S),
                        )
                    except Exception as e:
                        logger.debug("McCormick LP failed at node %d: %s", int(batch_ids[i]), e)
                        mc_lp_res = None
                    if mc_lp_res is not None and mc_lp_res.status == "infeasible":
                        # The McCormick LP is a valid OUTER relaxation of this
                        # node's subtree: if the (larger) relaxed feasible set is
                        # empty, the original is too.  This is a rigorous
                        # infeasibility proof, so the node is fathomed — mark it
                        # infeasible (sentinel + mask) and DO NOT decertify the
                        # gap.  Without this, an LP-infeasible node keeps the
                        # failure sentinel and the finalize block below would
                        # wrongly downgrade the run to "feasible".
                        node_infeasible_mask[i] = True
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        result_feas[i] = False
                    elif (
                        mc_lp_res is not None
                        and mc_lp_res.lower_bound is not None
                        and np.isfinite(mc_lp_res.lower_bound)
                    ):
                        mc_lp_lb = float(mc_lp_res.lower_bound)
                        cur = result_lbs[i]
                        if cur >= _SENTINEL_THRESHOLD or not np.isfinite(cur):
                            # No NLP bound (skipped or failed): adopt LP point
                            # + LP bound so the tree has a valid relaxation
                            # to spatial-branch on.
                            result_lbs[i] = mc_lp_lb
                            if mc_lp_res.x is not None:
                                result_sols[i] = mc_lp_res.x
                            result_feas[i] = False
                        else:
                            result_lbs[i] = max(cur, mc_lp_lb)

                # Nonconvex finalize (mirrors the batch path at the top of this
                # iteration).  Runs AFTER every bound source (NLP, alphaBB,
                # McCormick, LP relaxer) so a valid relaxation bound is given
                # the chance to certify optimality before we decide to
                # decertify the gap.
                #   * -inf  : no bound source produced a finite lower bound, so
                #             the node carries only the trivial -inf bound. The
                #             gap cannot be certified; downgrade to "feasible".
                #   * non-finite (and not -inf): coerce to the infeasibility
                #             sentinel so the Rust tree prunes it cleanly.
                #   * >= sentinel (issue #27a): a node pruned with no rigorous
                #             lower bound and no rigorous (FBBT) infeasibility
                #             proof is NOT proven suboptimal — the local NLP
                #             merely failed/diverged or its optimum violated the
                #             original constraints, neither of which rules out a
                #             better solution in the subtree. Certifying anyway
                #             lets big-M/mbigm GDP relaxations of nonlinear
                #             disjuncts silently fathom an unsolved subtree and
                #             report a wrong "optimal". Decertify so the result
                #             downgrades to "feasible" instead of lying.
                if not _model_is_convex and not node_infeasible_mask[i]:
                    if result_lbs[i] == -np.inf:
                        _gap_certified = False
                    elif not np.isfinite(result_lbs[i]):
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                    elif result_lbs[i] >= _SENTINEL_THRESHOLD:
                        _gap_certified = False
        jax_time += time.perf_counter() - t_jax_start

        if np.any(node_infeasible_mask):
            for idx in np.flatnonzero(node_infeasible_mask):
                i = int(idx)
                result_lbs[i] = _INFEASIBILITY_SENTINEL
                result_feas[i] = False

        # --- Objective-gating priority branching (issue #184) ---
        # The global dual bound is the minimum over the open frontier, so it stays
        # pinned at a structural 0 until *every* shallow node has had the binaries
        # gating its objective products fixed. No single-variable fractionality or
        # pseudocost score sees that joint jump (the bound moves only when a whole
        # set of binaries is pinned), so most-fractional branching can wander for
        # thousands of nodes at bound 0. Hinting the B&B to branch the gating
        # binaries first reaches the depth where the per-node relaxation (lifted-LP
        # FBBT) lifts each leaf off 0 — which is what raises the global bound. Pure
        # search reordering over already-fractional integer candidates: it can
        # never change a bound or a feasibility verdict.
        priority_hinted: set[int] = set()
        if _branch_priority_vars:
            pb_ids: list[int] = []
            pb_vars: list[int] = []
            for i in range(n_batch):
                if result_lbs[i] >= _SENTINEL_THRESHOLD:
                    continue
                pick = _select_priority_branch_var(
                    result_sols[i], batch_lb[i], batch_ub[i], _branch_priority_vars
                )
                if pick is not None:
                    nid = int(batch_ids[i])
                    pb_ids.append(nid)
                    pb_vars.append(pick)
                    priority_hinted.add(nid)
            if pb_ids:
                tree.set_branch_hints(
                    np.array(pb_ids, dtype=np.int64),
                    np.array(pb_vars, dtype=np.int64),
                )

        # --- Optional GNN branching scoring ---
        # GNN computes variable scores and passes hints to Rust TreeManager,
        # which uses them instead of most-fractional branching.
        if branching_policy == "gnn":
            from discopt._jax.gnn_policy import select_branch_variable_gnn
            from discopt._jax.problem_graph import build_graph

            hint_node_ids = []
            hint_var_indices = []
            for i in range(n_batch):
                if result_lbs[i] < _SENTINEL_THRESHOLD and int(batch_ids[i]) not in priority_hinted:
                    node_lb_i = np.array(batch_lb[i])
                    node_ub_i = np.array(batch_ub[i])
                    graph = build_graph(model, result_sols[i], node_lb_i, node_ub_i)
                    var_idx = select_branch_variable_gnn(graph, params=None)
                    if var_idx is not None:
                        hint_node_ids.append(int(batch_ids[i]))
                        hint_var_indices.append(var_idx)
            if hint_node_ids:
                tree.set_branch_hints(
                    np.array(hint_node_ids, dtype=np.int64),
                    np.array(hint_var_indices, dtype=np.int64),
                )

        # --- Strong branching for unreliable pseudocost candidates ---
        # For nodes without GNN hints, use LP-based strong branching when
        # pseudocost observations are insufficient for reliable branching.
        if branching_policy != "gnn" and iteration > 0:
            sb_hint_ids: list[int] = []
            sb_hint_vars = []
            rel_thresh = tree.reliability_threshold()
            for i in range(n_batch):
                if result_lbs[i] >= _SENTINEL_THRESHOLD:
                    continue  # skip infeasible nodes
                node_id = int(batch_ids[i])
                if node_id in priority_hinted:
                    continue  # gating-binary hint already set for this node
                sol_i = result_sols[i]
                var_indices, _frac_parts, obs_counts, _scores = tree.score_candidates(sol_i)
                if len(var_indices) == 0:
                    continue
                # Identify unreliable candidates
                unreliable_mask = obs_counts < rel_thresh
                if not np.any(unreliable_mask):
                    continue  # all candidates are reliable, pseudocosts will work
                unreliable_vars = np.asarray(var_indices)[unreliable_mask]
                try:
                    best_var = _strong_branch_lp(
                        _active_evaluator,
                        sol_i,
                        np.array(batch_lb[i]),
                        np.array(batch_ub[i]),
                        unreliable_vars,
                        parent_lb=float(result_lbs[i]),
                        max_candidates=5,
                        time_limit=0.5,
                        prefer_pounce=nlp_solver == "pounce",
                    )
                    if best_var is not None:
                        sb_hint_ids.append(node_id)
                        sb_hint_vars.append(best_var)
                except Exception as e:
                    logger.debug("Strong branching failed for node %d: %s", node_id, e)
            if sb_hint_ids:
                tree.set_branch_hints(
                    np.array(sb_hint_ids, dtype=np.int64),
                    np.array(sb_hint_vars, dtype=np.int64),
                )

        # --- Optional cut generation (OA + RLT + lift-and-project) ---
        if cutting_planes and _generate_cuts is not None and _cut_pool is not None:
            for i in range(n_batch):
                if result_lbs[i] < _SENTINEL_THRESHOLD:  # skip infeasible nodes
                    node_lb_i = np.array(batch_lb[i])
                    node_ub_i = np.array(batch_ub[i])
                    # Refresh the convex-constraint mask on this node's
                    # tightened box: a constraint UNKNOWN at the root
                    # may be provably convex on the subtree. The
                    # refresh only flips False -> True (soundness
                    # preserved) and is skipped when every root entry
                    # is already True (nothing to tighten).
                    node_mask = _convex_constraint_mask
                    node_oa_enabled = _oa_enabled
                    if (
                        _refresh_mask is not None
                        and _convex_constraint_mask is not None
                        and not all(_convex_constraint_mask)
                    ):
                        try:
                            node_mask = _refresh_mask(
                                model, _convex_constraint_mask, node_lb_i, node_ub_i
                            )
                        except Exception as exc:
                            logger.debug("Per-node convexity refresh failed: %s", exc)
                            node_mask = _convex_constraint_mask
                        if (
                            node_mask is not _convex_constraint_mask
                            and node_mask is not None
                            and any(node_mask)
                        ):
                            node_oa_enabled = True
                    new_cuts = _generate_cuts(
                        evaluator,
                        model,
                        result_sols[i],
                        node_lb_i,
                        node_ub_i,
                        constraint_senses=_constraint_senses,
                        bilinear_terms=_bilinear_terms,
                        oa_enabled=node_oa_enabled,
                        convex_constraint_mask=node_mask,
                    )
                    _cut_pool.add_many(new_cuts)
                    # Age and purge stale cuts
                    _cut_pool.age_cuts(result_sols[i])
            _cut_pool.purge_inactive(max_age=15)

        # --- Feasibility pump after root node ---
        if iteration == 0 and not _fp_ran:
            _fp_ran = True
            # Find the best relaxation solution from this batch
            best_root_idx = None
            best_root_obj = np.inf
            for i in range(n_batch):
                if result_lbs[i] < _SENTINEL_THRESHOLD and result_lbs[i] < best_root_obj:
                    best_root_obj = result_lbs[i]
                    best_root_idx = i
            if best_root_idx is not None:
                try:
                    from discopt._jax.primal_heuristics import feasibility_pump

                    fp_sol = feasibility_pump(
                        model,
                        result_sols[best_root_idx],
                        max_rounds=5,
                        backend=_resolve_heuristic_backend(nlp_solver),
                        evaluator=evaluator,
                    )
                    if fp_sol is not None:
                        fp_obj = float(evaluator.evaluate_objective(fp_sol))
                        fp_feas = not cl_list or _check_constraint_feasibility(
                            evaluator, fp_sol, cl_list, cu_list
                        )
                        if np.isfinite(fp_obj) and fp_obj < _SENTINEL_THRESHOLD and fp_feas:
                            tree.inject_incumbent(fp_sol, fp_obj)
                            logger.info("Feasibility pump found incumbent: obj=%.6g", fp_obj)
                except Exception as e:
                    logger.debug("Feasibility pump failed: %s", e)

                # --- Integer local search (1-opt + 2-opt) ---
                # Two roles, both at the root:
                #   (1) Feasibility recovery — if round-and-repair found no
                #       incumbent, the relaxation's integer assignment is likely
                #       true-infeasible (common for nonconvex integer-heavy
                #       problems: the McCormick relaxation's integers satisfy the
                #       relaxed but not the true constraints).
                #   (2) Incumbent improvement — round-and-repair (feasibility
                #       pump / nearest-rounding subnlp) only repairs the
                #       *continuous* variables around ONE relaxation rounding, so
                #       it routinely lands on a far-suboptimal integer assignment
                #       (e.g. nvs23: it parks at obj=-287 while the global is
                #       -1125). The lattice search seeds from the model's own
                #       continuous relaxation and descends the integer lattice, so
                #       it reaches much better integer assignments. Running it
                #       even when an incumbent already exists lets it IMPROVE that
                #       incumbent, not just recover feasibility.
                # Sound either way: only subnlp-verified feasible points are
                # injected, and inject_incumbent accepts a point only if it
                # strictly beats the current incumbent.
                if not _model_is_convex and best_root_idx is not None:
                    try:
                        from discopt._jax.primal_heuristics import integer_local_search

                        ils = integer_local_search(
                            model,
                            result_sols[best_root_idx],
                            backend=_resolve_heuristic_backend(nlp_solver),
                            evaluator=evaluator,
                            time_budget=min(5.0, 0.15 * max(1.0, time_limit)),
                        )
                        if ils is not None:
                            _x_ils, _obj_ils = ils
                            if np.isfinite(_obj_ils) and _obj_ils < _SENTINEL_THRESHOLD:
                                tree.inject_incumbent(_x_ils.copy(), float(_obj_ils))
                                logger.info(
                                    "Integer local search found incumbent: obj=%.6g",
                                    _obj_ils,
                                )
                    except Exception as e:
                        logger.debug("Integer local search failed: %s", e)

        # --- SubNLP primal heuristic ---
        # Fix integers in the best relaxation solution, then solve the
        # resulting continuous NLP. Useful for nonconvex problems whose
        # relaxation solver returns a local optimum that violates either
        # integrality or constraints (so the existing inject_incumbent
        # path above declines it). Runs at root and on a schedule after,
        # capped per solve. Skipped for convex models (no benefit).
        if (
            _subnlp_backend_fn is not None
            and not _model_is_convex
            and _subnlp_calls < subnlp_max_calls
            and (iteration == 0 or iteration % max(1, subnlp_frequency) == 0)
        ):
            from discopt._jax.primal_heuristics import subnlp as _subnlp

            # Root only: seed one subnlp from the modeler's GAMS-provided start
            # (parsed into model._gams_initial_values). For nonconvex models the
            # published point often lands in the global basin that generic
            # relaxation/midpoint seeds miss (prob07: 162070 -> 154990). No-op
            # when the model carries no initial values. Sound: subnlp re-verifies
            # feasibility and inject_incumbent enforces strict improvement.
            if iteration == 0 and _subnlp_calls < subnlp_max_calls:
                _gseed = _gams_initial_seed(model, lb, ub)
                if _gseed is not None:
                    _subnlp_calls += 1
                    try:
                        _sn = _subnlp(
                            model,
                            _gseed,
                            backend=_subnlp_backend_fn,
                            nlp_options=subnlp_options,
                            evaluator=evaluator,
                        )
                    except Exception as _e:
                        logger.debug("subnlp (gams seed) raised: %s", _e)
                        _sn = None
                    if _sn is not None:
                        _x_sn, _obj_sn = _sn
                        _subnlp_feasible += 1
                        if np.isfinite(_obj_sn) and _obj_sn < _SENTINEL_THRESHOLD:
                            tree.inject_incumbent(_x_sn.copy(), float(_obj_sn))
                            _subnlp_incumbent_updates += 1
                            logger.info("SubNLP incumbent (gams seed): obj=%.6g", _obj_sn)

            _cands_sn = [
                (i, float(result_lbs[i]))
                for i in range(n_batch)
                if result_lbs[i] < _SENTINEL_THRESHOLD and np.isfinite(result_lbs[i])
            ]
            _cands_sn.sort(key=lambda t: t[1])
            # Fall back to bound midpoint if no relaxation produced a usable point.
            if not _cands_sn and iteration == 0:
                _lb_c = np.clip(lb, -_SPC, _SPC)
                _ub_c = np.clip(ub, -_SPC, _SPC)
                _x_seed = 0.5 * (_lb_c + _ub_c)
                _subnlp_calls += 1
                try:
                    _sn = _subnlp(
                        model,
                        _x_seed,
                        backend=_subnlp_backend_fn,
                        nlp_options=subnlp_options,
                        evaluator=evaluator,
                    )
                except Exception as _e:
                    logger.debug("subnlp raised: %s", _e)
                    _sn = None
                if _sn is not None:
                    _x_sn, _obj_sn = _sn
                    _subnlp_feasible += 1
                    if np.isfinite(_obj_sn) and _obj_sn < _SENTINEL_THRESHOLD:
                        tree.inject_incumbent(_x_sn.copy(), float(_obj_sn))
                        _subnlp_incumbent_updates += 1
                        logger.info("SubNLP incumbent (seed): obj=%.6g", _obj_sn)
            else:
                _try_idxs = (
                    [i for i, _ in _cands_sn] if iteration == 0 else [i for i, _ in _cands_sn[:1]]
                )
                for _i in _try_idxs:
                    if _subnlp_calls >= subnlp_max_calls:
                        break
                    _subnlp_calls += 1
                    try:
                        _sn = _subnlp(
                            model,
                            result_sols[_i],
                            backend=_subnlp_backend_fn,
                            nlp_options=subnlp_options,
                            evaluator=evaluator,
                        )
                    except Exception as _e:
                        logger.debug("subnlp raised: %s", _e)
                        _sn = None
                    if _sn is None:
                        continue
                    _x_sn, _obj_sn = _sn
                    _subnlp_feasible += 1
                    if np.isfinite(_obj_sn) and _obj_sn < _SENTINEL_THRESHOLD:
                        tree.inject_incumbent(_x_sn.copy(), float(_obj_sn))
                        _subnlp_incumbent_updates += 1
                        logger.info("SubNLP incumbent: obj=%.6g (iter=%d)", _obj_sn, iteration)

        # --- Root binary-seed enumeration (deterministic disjunct cover) ---
        # A single nearest-rounding subnlp lands in whichever disjunct the
        # relaxation's fractional selector happens to round toward, a choice
        # decided by platform-dependent floating point for nonconvex GDP
        # models. At the root only, enumerate every 0/1 assignment over the
        # (capped) set of fractional binaries and solve a fixed-integer subnlp
        # from each, so the optimal disjunct is always tried regardless of
        # platform FP. Bounded to 2**max_binaries solves; skipped above the cap.
        if (
            iteration == 0
            and _subnlp_backend_fn is not None
            and not _model_is_convex
            and _subnlp_calls < subnlp_max_calls
        ):
            from discopt._jax.primal_heuristics import enumerate_binary_seeds_subnlp

            # Seed the enumeration from the best root relaxation point when one
            # produced a usable bound; otherwise fall back to the bound midpoint.
            # The enumeration sets the binaries explicitly and starts each
            # disjunct's continuous NLP from zero, so it does not actually need a
            # good relaxation point — and on some platforms the root relaxation
            # returns only sentinel bounds (no usable candidate), which must not
            # silently disable this heuristic.
            _enum_cands = [
                (i, float(result_lbs[i]))
                for i in range(n_batch)
                if result_lbs[i] < _SENTINEL_THRESHOLD and np.isfinite(result_lbs[i])
            ]
            _enum_cands.sort(key=lambda t: t[1])
            if _enum_cands:
                _enum_seed = result_sols[_enum_cands[0][0]]
            else:
                _enum_seed = 0.5 * (np.clip(lb, -_SPC, _SPC) + np.clip(ub, -_SPC, _SPC))
            try:
                _enum_results = enumerate_binary_seeds_subnlp(
                    model,
                    _enum_seed,
                    backend=_subnlp_backend_fn,
                    nlp_options=subnlp_options,
                    evaluator=evaluator,
                )
            except Exception as _e:
                logger.debug("enumerate_binary_seeds_subnlp raised: %s", _e)
                _enum_results = []
            _subnlp_calls += len(_enum_results)
            for _x_en, _obj_en in _enum_results:
                _subnlp_feasible += 1
                if np.isfinite(_obj_en) and _obj_en < _SENTINEL_THRESHOLD:
                    tree.inject_incumbent(_x_en.copy(), float(_obj_en))
                    _subnlp_incumbent_updates += 1
                    logger.info("SubNLP enum incumbent: obj=%.6g", _obj_en)

        # --- User callbacks: lazy constraints and incumbent filtering ---
        if lazy_constraints is not None or incumbent_callback is not None:
            _invoke_pre_import_callbacks(
                model=model,
                tree=tree,
                t_start=t_start,
                result_ids=result_ids,
                result_lbs=result_lbs,
                result_sols=result_sols,
                result_feas=result_feas,
                n_batch=n_batch,
                int_offsets=int_offsets,
                int_sizes=int_sizes,
                n_vars=n_vars,
                lazy_constraints=lazy_constraints,
                incumbent_callback=incumbent_callback,
                _cut_pool=_cut_pool,
            )

        # Import results back to Rust tree
        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        # Did the incumbent strictly improve this iteration, from ANY source?
        # proc_stats["incumbent_updates"] counts only incumbents found in the
        # batch evaluation — NOT the sub-NLP / binary-seed heuristics, which
        # inject directly via tree.inject_incumbent. For small nonconvex MINLPs
        # the optimum is routinely found by those heuristics, so gating the
        # cutoff-tightening phases on the batch counter alone left the tightened
        # global box (and thus all pruning) on the table and the gap never
        # closed. Tracking the incumbent value across iterations fires the
        # tightening once per genuine improvement regardless of who found it.
        _inc_now = tree.incumbent()
        _inc_obj_now = (
            float(_inc_now[1]) if _inc_now is not None and np.isfinite(_inc_now[1]) else np.inf
        )
        _incumbent_improved = _inc_obj_now < _last_tighten_inc - 1e-9
        if _incumbent_improved:
            _last_tighten_inc = _inc_obj_now

        # --- Periodic OBBT with incumbent cutoff (Phase C) ---
        # When a new incumbent is found and bounds are still wide,
        # re-run OBBT with the incumbent objective as a cutoff.
        if _incumbent_improved and n_vars <= 200:
            incumbent_info = tree.incumbent()
            if incumbent_info is not None:
                inc_sol, inc_obj = incumbent_info
                if inc_obj < _SENTINEL_THRESHOLD:
                    try:
                        from discopt._jax.obbt import obbt_tighten_root

                        # Tighten against the McCormick *relaxation* (not just the
                        # model's linear rows) with the incumbent as a cutoff. This
                        # runs duality-based bound tightening (one objective LP whose
                        # reduced costs bound every variable) followed by
                        # optimality-based OBBT, so a new incumbent shrinks the box
                        # via the nonlinear envelopes too — both are rigorous
                        # tightenings of an outer approximation.
                        obbt_result = obbt_tighten_root(
                            model,
                            np.array(lb),
                            ub=np.array(ub),
                            rounds=2,
                            incumbent_cutoff=float(inc_obj),
                            deadline=time.perf_counter() + 5.0,
                            time_limit_per_lp=0.1,
                            prefer_pounce=nlp_solver == "pounce",
                        )
                        if not obbt_result.infeasible and obbt_result.n_tightened > 0:
                            lb = obbt_result.lb
                            ub = obbt_result.ub
                            logger.info(
                                "Relaxation OBBT/DBBT tightened %d bounds (incumbent=%.6g)",
                                obbt_result.n_tightened,
                                inc_obj,
                            )
                    except Exception as e:
                        logger.debug("Periodic OBBT failed: %s", e)

        # --- FBBT with incumbent cutoff (Phase C3) ---
        # Cheap bound tightening via Rust FBBT (no LP solves).
        # Run on every incumbent improvement, complementing OBBT.
        if _model_repr is not None and _incumbent_improved:
            incumbent_info = tree.incumbent()
            if incumbent_info is not None:
                inc_sol, inc_obj = incumbent_info
                if inc_obj < _SENTINEL_THRESHOLD:
                    try:
                        fbbt_lbs, fbbt_ubs = _model_repr.fbbt_with_cutoff(
                            max_iter=10, tol=1e-8, incumbent_bound=float(inc_obj)
                        )
                        fbbt_lbs = np.asarray(fbbt_lbs)
                        fbbt_ubs = np.asarray(fbbt_ubs)
                        # Map per-block bounds to flat bounds array
                        n_tightened = 0
                        flat_idx = 0
                        for bi, vinfo in enumerate(model._variables):
                            if vinfo.size != 1:
                                flat_idx += vinfo.size
                                continue
                            for j in range(vinfo.size):
                                new_lo = fbbt_lbs[bi]
                                new_hi = fbbt_ubs[bi]
                                if new_lo > lb[flat_idx] + 1e-10:
                                    lb[flat_idx] = new_lo
                                    n_tightened += 1
                                if new_hi < ub[flat_idx] - 1e-10:
                                    ub[flat_idx] = new_hi
                                    n_tightened += 1
                                flat_idx += 1
                        if n_tightened > 0:
                            logger.info(
                                "FBBT tightened %d bounds (incumbent=%.6g)",
                                n_tightened,
                                inc_obj,
                            )
                    except Exception as e:
                        logger.debug("FBBT with cutoff failed: %s", e)

        # --- Node callback: notify user after each batch ---
        if node_callback is not None:
            try:
                stats_snap = tree.stats()
                incumbent_info_cb = tree.incumbent()
                inc_obj_cb = None
                if incumbent_info_cb is not None:
                    _, inc_obj_cb = incumbent_info_cb
                    if inc_obj_cb >= _SENTINEL_THRESHOLD:
                        inc_obj_cb = None
                best_idx = 0
                for i in range(n_batch):
                    if result_lbs[i] < result_lbs[best_idx]:
                        best_idx = i
                from discopt.callbacks import CallbackContext

                ctx = CallbackContext(
                    node_count=stats_snap["total_nodes"],
                    incumbent_obj=inc_obj_cb,
                    best_bound=stats_snap.get("global_lower_bound", -np.inf),
                    gap=stats_snap.get("gap"),
                    elapsed_time=time.perf_counter() - t_start,
                    x_relaxation=result_sols[best_idx].copy(),
                    node_bound=float(result_lbs[best_idx]),
                )
                node_callback(ctx, model)
            except Exception as e:
                logger.warning("Node callback raised an exception: %s", e)

        iteration += 1

        # Check termination
        if tree.is_finished():
            break
        if tree.gap() <= gap_tolerance:
            break

        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    # --- Build result ---
    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time

    stats = tree.stats()
    incumbent = tree.incumbent()

    if incumbent is not None:
        sol_array, obj_val = incumbent
        # Filter out bogus incumbents from infeasible NLP relaxations
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        x_dict = _unpack_solution(model, sol_flat)

        # Refine the incumbent's continuous variables with a KKT-accurate
        # re-solve (integers fixed at their incumbent values). The batched JAX
        # IPM can leave continuous variables a few digits short of full
        # precision, which inflates active-constraint complementarity at
        # validation time; POUNCE (or cyipopt) converges them tightly. This is
        # a single solve at the end of the search. Best-effort and guarded so a
        # divergent re-solve can never change the reported optimum.
        try:
            _fix_lb = lb.copy()
            _fix_ub = ub.copy()
            for _off, _sz in zip(int_offsets, int_sizes):
                for _k in range(int(_sz)):
                    _val = float(round(float(sol_flat[_off + _k])))
                    _fix_lb[_off + _k] = _val
                    _fix_ub[_off + _k] = _val
            _polish_opts = dict(opts)
            _polish_opts["max_wall_time"] = max(
                0.1, min(5.0, time_limit - (time.perf_counter() - t_start))
            )
            _polish_opts.setdefault("print_level", 0)
            _polished = _solve_node_nlp_kkt(
                evaluator, sol_flat, _fix_lb, _fix_ub, constraint_bounds, _polish_opts
            )
            if (
                _polished.status == SolveStatus.OPTIMAL
                and _polished.x is not None
                and np.all(np.isfinite(_polished.x))
                and _polished.objective is not None
                and abs(float(_polished.objective) - obj_val) <= 1e-4 * (1.0 + abs(obj_val))
            ):
                _refined = np.asarray(_polished.x, dtype=float).copy()
                for _off, _sz in zip(int_offsets, int_sizes):
                    for _k in range(int(_sz)):
                        _refined[_off + _k] = round(float(sol_flat[_off + _k]))
                sol_flat = _refined
                x_dict = _unpack_solution(model, sol_flat)
                obj_val = float(_polished.objective)
        except Exception as _exc:
            logger.debug("Incumbent KKT polish failed: %s", _exc)

        # Negate objective back for maximization (B&B tree tracks minimization)
        from discopt.modeling.core import ObjectiveSense

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        search_closed = tree.gap() <= gap_tolerance or tree.is_finished()
        if search_closed and _gap_certified:
            status = "optimal"
        else:
            status = "feasible"
    else:
        x_dict = None
        obj_val = None
        if stats["total_nodes"] >= max_nodes:
            status = "node_limit"
            # No incumbent and a resource limit: nothing is certified. Without an
            # incumbent there is no gap to certify, and a leftover tree bound
            # describes an unexplored search, not a proven optimum — a certified
            # exit here would claim optimality with no solution at all.
            _gap_certified = False
        elif wall_time >= time_limit:
            status = "time_limit"
            _gap_certified = False
        else:
            # Tree exhausted with no feasible node: infeasibility *is* a certified
            # conclusion, so leave _gap_certified untouched.
            status = "infeasible"

    from discopt.modeling.core import ObjectiveSense

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    assert model._objective is not None
    if bound_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        bound_val = -bound_val
    gap_val = stats["gap"]

    # A *feasible* exit never inherits the tree's validity flag as a certificate.
    # The flag means no node invalidated the tree bound — NOT that the gap is
    # closed; a budget/node-limited feasible exit leaves that valid bound strictly
    # below the incumbent (open gap). Reporting gap_certified=True there would
    # falsely claim global optimality (max_nodes=1 root-only exit: obj 19029,
    # bound -583, gap 1.03 — yet previously "certified"). Drop the flag here and
    # re-earn it below only if a rigorous global bound actually closes the gap.
    # "optimal" already implies a closed certified search; infeasible/limit exits
    # are handled above and keep their own certification semantics.
    if status == "feasible":
        _gap_certified = False

    if not _gap_certified:
        bound_val = None
        gap_val = None

    # When the tree produced no usable dual bound (uncertified exit, or a
    # relaxation the LP backend could not solve), fall back to a rigorous global
    # lower bound from the root MILP relaxation over the root box, so a
    # minimization reports a finite sound bound instead of None (issue #138). The
    # MILP path already seeds such a root bound; this brings the spatial path to
    # parity. Surfaced lazily — only when the search yielded nothing — and never
    # overrides an existing finite bound.
    if (bound_val is None or not np.isfinite(bound_val)) and (
        model._objective.sense == ObjectiveSense.MINIMIZE
    ):
        _rr = _root_relaxation_lower_bound(
            model, _root_lb_snapshot, _root_ub_snapshot, time_limit, psd_cuts=psd_cuts
        )
        if _rr is not None and np.isfinite(_rr):
            bound_val = _rr
            if obj_val is not None and np.isfinite(obj_val):
                gap_val = max(0.0, obj_val - bound_val) / max(1.0, abs(obj_val))

    # Re-earn certification on a feasible exit iff a *rigorous* global lower bound
    # (here the root-relaxation fallback, itself only returned when the objective
    # is validly linearized) meets the incumbent within tolerance — then the
    # incumbent is provably global and the honest status is "optimal". The
    # bound <= incumbent guard stops a spurious above-incumbent bound (which the
    # max(0, …) gap clamp would otherwise read as gap 0) from certifying.
    if (
        status == "feasible"
        and obj_val is not None
        and bound_val is not None
        and np.isfinite(bound_val)
        and bound_val <= obj_val + 1e-9
        and gap_val is not None
        and gap_val <= gap_tolerance
    ):
        _gap_certified = True
        status = "optimal"

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=bound_val,
        gap=gap_val,
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
        gap_certified=_gap_certified,
        subnlp_calls=_subnlp_calls,
        subnlp_feasible=_subnlp_feasible,
        subnlp_incumbent_updates=_subnlp_incumbent_updates,
    )


def _default_nlp_solver() -> str:
    """Resolve the default KKT-valid NLP backend.

    Prefers POUNCE (pure-Rust Ipopt port), falls back to cyipopt, then to the
    pure-JAX IPM as a last resort. Mirrors the preference order of
    :func:`discopt.solvers.nlp_backend.get_nlp_solver`.
    """
    from discopt.solvers.nlp_backend import available_backends

    avail = available_backends()
    if "pounce" in avail:
        return "pounce"
    if "cyipopt" in avail:
        return "ipopt"
    return "ipm"


def _resolve_heuristic_backend(nlp_solver: str) -> Optional[Callable]:
    """Resolve the NLP backend for primal heuristics, honoring ``nlp_solver``.

    The pump / rounding heuristics project onto the continuous feasible set with
    a standalone ``solve_nlp`` callable. Map the caller's choice to a KKT-valid
    backend (POUNCE-preferred): explicit ``"pounce"`` / ``"ipopt"`` are honored,
    while the pure-JAX ``"ipm"`` / ``"sparse_ipm"`` selections (which have no
    standalone NLP entry point) fall back to ``"auto"`` (POUNCE if importable,
    else cyipopt). If the preferred backend is unimportable we degrade to
    ``"auto"``, and return ``None`` only when no backend exists at all (letting
    the heuristic default to its own ``"auto"`` resolution / skip).
    """
    from discopt.solvers.nlp_backend import Backend, get_nlp_solver

    preferred = {"pounce": "pounce", "ipopt": "cyipopt", "cyipopt": "cyipopt"}.get(
        nlp_solver, "auto"
    )
    for backend in (preferred, "auto"):
        try:
            return get_nlp_solver(cast(Backend, backend))
        except ImportError:
            continue
    return None


def _solve_continuous(
    model: Model,
    time_limit: float,
    ipopt_options: Optional[dict],
    t_start: float,
    nlp_solver: str = "ipopt",
    initial_point: Optional[np.ndarray] = None,
) -> SolveResult:
    """Solve a purely continuous model directly with NLP solver (no B&B)."""
    # Single-NLP solves need reliable KKT convergence. The pure-JAX IPM's
    # acceptable-tolerance check only covers bound complementarity, so on
    # problems with unbounded variables and inequality constraints it can
    # terminate at a non-KKT point and report OPTIMAL (false optimality).
    # B&B subproblems tolerate this because the tree catches it, but single
    # solves don't, so promote the default ipm -> a KKT-valid solver (POUNCE,
    # falling back to cyipopt). Users who explicitly requested
    # ipm/pounce/sparse_ipm still get what they asked for.
    if nlp_solver == "ipm":
        nlp_solver = _default_nlp_solver()

    t_jax_start = time.perf_counter()
    evaluator = _make_evaluator(model)
    jax_time = time.perf_counter() - t_jax_start

    raw_lb, raw_ub = evaluator.variable_bounds
    lb = np.asarray(raw_lb, dtype=np.float64).copy()
    ub = np.asarray(raw_ub, dtype=np.float64).copy()
    tightened_lb, tightened_ub, nonlinear_infeasible = _apply_nonlinear_tightening_with_status(
        model, lb, ub
    )
    if nonlinear_infeasible:
        wall_time = time.perf_counter() - t_start
        return SolveResult(
            status="infeasible",
            wall_time=wall_time,
            gap_certified=True,
            jax_time=jax_time,
            python_time=wall_time - jax_time,
        )
    lb, ub = tightened_lb, tightened_ub
    lb_clipped = np.clip(lb, -_SPC, _SPC)
    ub_clipped = np.clip(ub, -_SPC, _SPC)
    if initial_point is not None:
        x0 = np.clip(initial_point, lb, ub)
        logger.info("Using warm-start point for continuous NLP")
    else:
        x0 = 0.5 * (lb_clipped + ub_clipped)
        # Variables that are effectively unbounded on both sides can collapse
        # to midpoint 0 even after nonlinear tightening produces a compact
        # box. Zero is a stationary or nonsmooth point for common atoms
        # (sin/cos, sqrt(x^2), abs), so keep the original sentinel-bound
        # signal when nudging starts away from that pathological point.
        fully_unbounded = (raw_lb <= -_BOUND_WARN_THRESHOLD) & (raw_ub >= _BOUND_WARN_THRESHOLD)
        x0 = np.where(fully_unbounded, 0.5, x0)
        # On problems with one-sided large bounds (e.g. x >= 1e-5 with no
        # upper bound), the midpoint of the clipped [-_SPC, _SPC] range
        # lands at ~50, which sends exp/log NLPs into overflow territory
        # and crashes ipopt. Tighten the starting-point range to keep
        # initial iterates in a numerically safe zone while still
        # respecting actual bounds.
        _X0_CLIP = 10.0
        x0 = np.clip(x0, np.maximum(lb, -_X0_CLIP), np.minimum(ub, _X0_CLIP))

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)

    # Pass remaining time budget to NLP solver so stalled subproblems
    # don't run unbounded (see issue #5).
    remaining = time_limit - (time.perf_counter() - t_start)
    opts["max_wall_time"] = max(remaining, 0.1)

    constraint_bounds = None
    backend_evaluator = cast(NLPEvaluator, _BoundOverrideEvaluator(evaluator, lb, ub))

    t_jax_start = time.perf_counter()
    if nlp_solver == "pounce":
        from discopt.solvers.nlp_pounce import solve_nlp as solve_nlp_pounce

        nlp_result = solve_nlp_pounce(
            backend_evaluator, x0, constraint_bounds=constraint_bounds, options=opts
        )
    else:
        # "ipm"/"sparse_ipm" resolve to POUNCE upstream (the JAX IPM is retired);
        # only "ipopt"/"cyipopt" reach this branch.
        nlp_result = solve_nlp(
            backend_evaluator, x0, constraint_bounds=constraint_bounds, options=opts
        )
    jax_time += time.perf_counter() - t_jax_start

    wall_time = time.perf_counter() - t_start
    python_time = wall_time - jax_time

    if nlp_result.status == SolveStatus.OPTIMAL:
        status = "optimal"
    elif nlp_result.status == SolveStatus.INFEASIBLE:
        status = "infeasible"
    else:
        status = nlp_result.status.value

    x_dict = _unpack_solution(model, nlp_result.x) if nlp_result.x is not None else None

    # Negate objective back for maximization (NLPEvaluator solves minimization of -f)
    from discopt.modeling.core import ObjectiveSense

    assert model._objective is not None
    obj_val = nlp_result.objective
    if obj_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        obj_val = -obj_val

    constraint_duals = _unpack_constraint_duals(evaluator, nlp_result.multipliers)
    bound_duals_lower = _unpack_bound_duals(model, nlp_result.bound_multipliers_lower)
    bound_duals_upper = _unpack_bound_duals(model, nlp_result.bound_multipliers_upper)

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=obj_val if status == "optimal" else None,
        gap=_optimal_relative_gap(obj_val) if status == "optimal" and obj_val is not None else None,
        x=x_dict,
        constraint_duals=constraint_duals,
        bound_duals_lower=bound_duals_lower,
        bound_duals_upper=bound_duals_upper,
        wall_time=wall_time,
        node_count=0,
        rust_time=0.0,
        jax_time=jax_time,
        python_time=python_time,
    )


def _solve_nlp_bb(
    model: Model,
    time_limit: float,
    gap_tolerance: float,
    batch_size: int,
    strategy: str,
    max_nodes: int,
    t_start: float,
    nlp_solver: str,
    skip_convex_check: bool = False,
    initial_point: Optional[np.ndarray] = None,
    lazy_constraints=None,
    incumbent_callback=None,
    node_callback=None,
    in_tree_presolve_stride: int = 0,
    in_tree_presolve_repr=None,
) -> SolveResult:
    """Solve a MINLP via nonlinear Branch & Bound (NLP-BB).

    Instead of solving convex relaxations (McCormick/alphaBB) at each node,
    NLP-BB solves the original continuous NLP with discrete variables fixed
    via bound tightening.  For convex MINLPs, the NLP objective at each node
    is a valid lower bound, giving certified optimality gaps without any
    relaxation overhead.

    For nonconvex problems the NLP objective is NOT a valid lower bound;
    the solver runs in heuristic mode and reports gap_certified=False.
    """
    from discopt._jax.gdp_reformulate import reformulate_gdp
    from discopt.modeling.core import ObjectiveSense

    model = reformulate_gdp(model, method="big-m")

    rust_time = 0.0
    jax_time = 0.0

    # --- Convexity gate ---
    _gap_certified = True
    try:
        from discopt._jax.convexity import classify_model as _classify_model

        _model_is_convex, _ = _classify_model(model, use_certificate=True)
    except Exception:
        _model_is_convex = False

    if not _model_is_convex and not skip_convex_check:
        logger.warning(
            "NLP-BB on nonconvex model: running in heuristic mode "
            "(gap not certified). Pass skip_convex_check=True to suppress."
        )
        _gap_certified = False

    # --- Extract variable info and create tree ---
    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)

    # --- Root presolve: FBBT + integer-bound rounding before tree creation ---
    t_rust_start = time.perf_counter()
    from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

    lb, ub, root_infeasible, _ = tighten_root_bounds_with_fbbt(
        model,
        lb,
        ub,
        int_offsets,
        int_sizes,
    )
    rust_time += time.perf_counter() - t_rust_start
    if root_infeasible:
        wall_time = time.perf_counter() - t_start
        return SolveResult(
            status="infeasible",
            objective=None,
            bound=None,
            gap=None,
            x=None,
            wall_time=wall_time,
            node_count=0,
            rust_time=rust_time,
            jax_time=jax_time,
            python_time=wall_time - rust_time - jax_time,
            nlp_bb=True,
            gap_certified=_gap_certified,
        )

    t_rust_start = time.perf_counter()
    tree = PyTreeManager(
        n_vars,
        lb.tolist(),
        ub.tolist(),
        int_offsets,
        int_sizes,
        strategy,
    )
    tree.initialize()
    rust_time += time.perf_counter() - t_rust_start

    # --- Compile NLP evaluator ---
    t_jax_start = time.perf_counter()
    evaluator = _make_evaluator(model)
    jax_time += time.perf_counter() - t_jax_start

    # --- Infer constraint bounds ---
    cl_list, cu_list = _infer_constraint_bounds(model, evaluator)
    constraint_bounds = list(zip(cl_list, cu_list)) if cl_list else None

    opts: dict = {}
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", 3000)
    opts.setdefault("tol", 1e-7)

    # --- Warm-start: inject user-provided initial solution as incumbent ---
    if initial_point is not None:
        ws_obj = float(evaluator.evaluate_objective(initial_point))
        ws_int_feas = True
        for off, sz in zip(int_offsets, int_sizes):
            for j in range(off, off + sz):
                if abs(initial_point[j] - round(initial_point[j])) > 1e-5:
                    ws_int_feas = False
                    break
            if not ws_int_feas:
                break
        if ws_int_feas and np.isfinite(ws_obj) and ws_obj < _SENTINEL_THRESHOLD:
            ws_con_feas = not cl_list or _check_constraint_feasibility(
                evaluator, initial_point, cl_list, cu_list
            )
            if ws_con_feas:
                tree.inject_incumbent(initial_point, ws_obj)
                logger.info("NLP-BB warm-start incumbent: obj=%.6g", ws_obj)

    # --- Feasibility pump flag ---
    _fp_ran = False

    # --- Soundness: distinguish a PROVEN-infeasible fathom from a merely
    # UNCONVERGED one. A node whose NLP returns SolveStatus.INFEASIBLE (or whose
    # FBBT interval arithmetic proves the box empty) is a valid infeasibility
    # certificate. A node whose NLP merely hit ITERATION_LIMIT / ERROR with a
    # constraint-violating iterate is NON-convergence, NOT a proof — an
    # interior-point method can stall at an infeasible point on a perfectly
    # feasible convex NLP (e.g. division constraints 40/x7 <= x5 need ~5k iters
    # to reach feasibility; the 3k default stalls). If the tree later empties
    # with no incumbent, an "infeasible" verdict is only sound when NO node was
    # fathomed on non-convergence; otherwise feasibility is genuinely UNKNOWN.
    _unconverged_fathom = False

    # --- NLP-BB loop ---
    # POUNCE batches node NLPs via solve_nlp_batch (Phase A, discopt#97); it is
    # a true KKT solver, so its converged objective is a reliable lower bound
    # for convex MINLPs. The callback path is GIL-bound, so only batch when the
    # per-node problem is large enough to amortize it; smaller nodes stay on the
    # serial path.
    _use_pounce_batch = nlp_solver == "pounce" and n_vars >= _POUNCE_BATCH_MIN_VARS
    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Update per-iteration time budget for NLP subproblem solves (issue #5).
        remaining = time_limit - elapsed
        opts["max_wall_time"] = max(remaining, 0.1)

        # Export batch from Rust tree
        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        # Tighten node bounds via constraint propagation (FBBT).
        # This resolves degenerate bounds (e.g., x <= M*y with y fixed at 0)
        # that cause IPM convergence failures.
        node_infeasible_mask = np.zeros(n_batch, dtype=bool)
        if cl_list:
            for i in range(n_batch):
                node_lb_i = np.array(batch_lb[i])
                node_ub_i = np.array(batch_ub[i])
                t_lb, t_ub, node_infeasible = _tighten_node_bounds_with_status(
                    evaluator, node_lb_i, node_ub_i, cl_list, cu_list
                )
                if node_infeasible:
                    node_infeasible_mask[i] = True
                    continue
                batch_lb[i] = t_lb.tolist()
                batch_ub[i] = t_ub.tolist()

        # B3: persistent in-tree FBBT via the Rust kernel, gated by
        # depth-stride. Best-effort — silently skipped if shape doesn't
        # match (models with array variable blocks aren't supported by
        # the kernel yet).
        if in_tree_presolve_stride and in_tree_presolve_repr is not None:
            try:
                n_blocks = in_tree_presolve_repr.n_var_blocks
                for i in range(n_batch):
                    if len(batch_lb[i]) != n_blocks:
                        continue
                    delta = in_tree_presolve_repr.in_tree_presolve(
                        np.asarray(batch_lb[i], dtype=np.float64),
                        np.asarray(batch_ub[i], dtype=np.float64),
                        node_depth=0,
                        depth_stride=in_tree_presolve_stride,
                    )
                    if delta["ran"] and not delta["infeasible"]:
                        batch_lb[i] = list(delta["lb"])
                        batch_ub[i] = list(delta["ub"])
            except Exception as _e:
                logger.debug("in-tree presolve skipped: %s", _e)

        # Solve NLP at each node (no relaxation, no multistart for convex)
        t_jax_start = time.perf_counter()

        if _use_pounce_batch and n_batch > 1:
            result_ids, result_lbs, result_sols, result_feas, _batch_trusted = _solve_batch_pounce(
                evaluator,
                batch_lb,
                batch_ub,
                batch_ids,
                n_vars,
                constraint_bounds,
                opts,
                batch_psols=batch_psols,
                multistart=_POUNCE_BATCH_MULTISTART,
                convex=_model_is_convex,
            )
            # Convex MINLP: the NLP objective is the node lower bound. A node
            # whose relaxation did not reach KKT (and could not be polished) is
            # not a valid lower bound, so decertify the gap rather than trust
            # it — leaving bound/incumbent untouched (roadmap P0.3).
            if _model_is_convex and not bool(np.all(_batch_trusted)):
                _gap_certified = False
            # Constraint feasibility post-check
            if cl_list:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        if not _check_constraint_feasibility(
                            evaluator, result_sols[i], cl_list, cu_list
                        ):
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            # Batch IPM returned an objective (not a clean
                            # infeasibility verdict) but the iterate violates
                            # constraints: a stall, not a proof. Untrusted nodes
                            # are unconverged by definition; tainting on any
                            # constraint-infeasible batch fathom is conservative
                            # and sound.
                            _unconverged_fathom = True
            # For nonconvex: NLP objective is NOT a valid lower bound.
            # Keep it for integer-feasible nodes (incumbent candidates),
            # but reset to -inf for others so we don't prune incorrectly.
            if not _model_is_convex:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        sol_is_int_feas = True
                        for off, sz in zip(int_offsets, int_sizes):
                            for j in range(off, off + sz):
                                frac = abs(result_sols[i, j] - round(result_sols[i, j]))
                                if frac > 1e-5:
                                    sol_is_int_feas = False
                                    break
                            if not sol_is_int_feas:
                                break
                        if not sol_is_int_feas:
                            result_lbs[i] = -np.inf
        else:
            # Serial fallback (batch_size=1 or non-IPM solver)
            result_ids = np.empty(n_batch, dtype=np.int64)
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.empty(n_batch, dtype=bool)

            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                if iteration == 0:
                    nlp_result = _solve_root_node_multistart(
                        evaluator,
                        node_lb,
                        node_ub,
                        constraint_bounds,
                        opts,
                        nlp_solver,
                        convex=_model_is_convex,
                    )
                else:
                    psol_i = np.array(batch_psols[i])
                    if not np.any(np.isnan(psol_i)):
                        x0 = np.clip(psol_i, node_lb, node_ub)
                    else:
                        lb_c = np.clip(node_lb, -_SPC, _SPC)
                        ub_c = np.clip(node_ub, -_SPC, _SPC)
                        x0 = 0.5 * (lb_c + ub_c)
                    nlp_result = _solve_node_nlp(
                        evaluator,
                        x0,
                        node_lb,
                        node_ub,
                        constraint_bounds,
                        opts,
                        nlp_solver=nlp_solver,
                        convex=_model_is_convex,
                    )

                result_ids[i] = int(batch_ids[i])
                if nlp_result.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
                    nlp_lb = nlp_result.objective
                    # Convex node bound is only valid if the relaxation reached
                    # KKT. ITERATION_LIMIT here means non-KKT and unpolished
                    # (the serial IPM path polishes; POUNCE does not), so the
                    # objective is not a valid lower bound — decertify the gap
                    # (roadmap P0.3) while still using it as a branching point.
                    if _model_is_convex and nlp_result.status == SolveStatus.ITERATION_LIMIT:
                        _gap_certified = False
                    # Constraint feasibility check
                    if cl_list and not _check_constraint_feasibility(
                        evaluator, nlp_result.x, cl_list, cu_list
                    ):
                        nlp_lb = _INFEASIBILITY_SENTINEL
                        # The solver returned OPTIMAL/ITERATION_LIMIT yet the
                        # iterate violates constraints — this is non-convergence
                        # (a stall), not a SolveStatus.INFEASIBLE proof. Fathoming
                        # it cannot certify global infeasibility.
                        _unconverged_fathom = True
                    # For nonconvex: reset non-integer-feasible to -inf
                    elif not _model_is_convex:
                        sol_is_int_feas = True
                        for off, sz in zip(int_offsets, int_sizes):
                            for j in range(off, off + sz):
                                xj = nlp_result.x[j]
                                if not np.isfinite(xj) or abs(xj - round(xj)) > 1e-5:
                                    sol_is_int_feas = False
                                    break
                            if not sol_is_int_feas:
                                break
                        if not sol_is_int_feas:
                            nlp_lb = -np.inf
                    # Guard: NaN lower bounds corrupt the Rust B&B tree.
                    if not np.isfinite(nlp_lb):
                        nlp_lb = _INFEASIBILITY_SENTINEL
                    result_lbs[i] = nlp_lb
                    result_sols[i] = nlp_result.x
                    result_feas[i] = False
                else:
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    # A clean SolveStatus.INFEASIBLE is a valid infeasibility
                    # certificate (for a convex node); ERROR/TIME_LIMIT/UNBOUNDED
                    # are not — they are solver failures that must not masquerade
                    # as a proof of global infeasibility.
                    if nlp_result.status != SolveStatus.INFEASIBLE:
                        _unconverged_fathom = True
                    lb_c = np.clip(node_lb, -_SPC, _SPC)
                    ub_c = np.clip(node_ub, -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)
                    result_feas[i] = False

        jax_time += time.perf_counter() - t_jax_start

        if np.any(node_infeasible_mask):
            for idx in np.flatnonzero(node_infeasible_mask):
                i = int(idx)
                result_lbs[i] = _INFEASIBILITY_SENTINEL
                result_feas[i] = False

        # --- Feasibility pump after root node ---
        if iteration == 0 and not _fp_ran:
            _fp_ran = True
            best_root_idx = None
            best_root_obj = np.inf
            for i in range(n_batch):
                if result_lbs[i] < _SENTINEL_THRESHOLD and result_lbs[i] < best_root_obj:
                    best_root_obj = result_lbs[i]
                    best_root_idx = i
            if best_root_idx is not None:
                try:
                    from discopt._jax.primal_heuristics import feasibility_pump

                    fp_sol = feasibility_pump(
                        model,
                        result_sols[best_root_idx],
                        max_rounds=5,
                        backend=_resolve_heuristic_backend(nlp_solver),
                        evaluator=evaluator,
                    )
                    if fp_sol is not None:
                        fp_obj = float(evaluator.evaluate_objective(fp_sol))
                        fp_feas = not cl_list or _check_constraint_feasibility(
                            evaluator, fp_sol, cl_list, cu_list
                        )
                        if np.isfinite(fp_obj) and fp_obj < _SENTINEL_THRESHOLD and fp_feas:
                            tree.inject_incumbent(fp_sol, fp_obj)
                            logger.info("NLP-BB feasibility pump incumbent: obj=%.6g", fp_obj)
                except Exception as e:
                    logger.debug("Feasibility pump failed: %s", e)

        # --- User callbacks ---
        if lazy_constraints is not None or incumbent_callback is not None:
            _invoke_pre_import_callbacks(
                model=model,
                tree=tree,
                t_start=t_start,
                result_ids=result_ids,
                result_lbs=result_lbs,
                result_sols=result_sols,
                result_feas=result_feas,
                n_batch=n_batch,
                int_offsets=int_offsets,
                int_sizes=int_sizes,
                n_vars=n_vars,
                lazy_constraints=lazy_constraints,
                incumbent_callback=incumbent_callback,
                _cut_pool=None,
            )

        # Import results back to Rust tree
        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        # --- Node callback ---
        if node_callback is not None:
            try:
                stats_snap = tree.stats()
                incumbent_info_cb = tree.incumbent()
                inc_obj_cb = None
                if incumbent_info_cb is not None:
                    _, inc_obj_cb = incumbent_info_cb
                    if inc_obj_cb >= _SENTINEL_THRESHOLD:
                        inc_obj_cb = None
                best_idx = 0
                for i in range(n_batch):
                    if result_lbs[i] < result_lbs[best_idx]:
                        best_idx = i
                from discopt.callbacks import CallbackContext

                ctx = CallbackContext(
                    node_count=stats_snap["total_nodes"],
                    incumbent_obj=inc_obj_cb,
                    best_bound=stats_snap.get("global_lower_bound", -np.inf),
                    gap=stats_snap.get("gap"),
                    elapsed_time=time.perf_counter() - t_start,
                    x_relaxation=result_sols[best_idx].copy(),
                    node_bound=float(result_lbs[best_idx]),
                )
                node_callback(ctx, model)
            except Exception as e:
                logger.warning("Node callback raised an exception: %s", e)

        iteration += 1

        # Check termination
        if tree.is_finished():
            break
        if tree.gap() <= gap_tolerance:
            break
        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    # --- Build result ---
    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time

    stats = tree.stats()
    incumbent = tree.incumbent()

    if incumbent is not None:
        sol_array, obj_val = incumbent
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    constraint_duals = None
    bound_duals_lower = None
    bound_duals_upper = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        x_dict = _unpack_solution(model, sol_flat)

        # Recover relaxation duals at the incumbent by re-solving the NLP with
        # integer variables fixed. Best-effort — failures leave duals as None
        # and the examiner falls back to its LSQ recovery.
        try:
            fix_lb = lb.copy()
            fix_ub = ub.copy()
            for off, sz in zip(int_offsets, int_sizes):
                for k in range(int(sz)):
                    val = float(round(float(sol_flat[off + k])))
                    fix_lb[off + k] = val
                    fix_ub[off + k] = val
            recover_opts = dict(opts)
            recover_opts["max_wall_time"] = max(
                0.1, min(5.0, time_limit - (time.perf_counter() - t_start))
            )
            recover_opts.setdefault("print_level", 0)
            nlp_recovered = _solve_node_nlp_kkt(
                evaluator, sol_flat, fix_lb, fix_ub, constraint_bounds, recover_opts
            )
            if nlp_recovered.status in (
                SolveStatus.OPTIMAL,
                SolveStatus.ITERATION_LIMIT,
            ):
                constraint_duals = _unpack_constraint_duals(evaluator, nlp_recovered.multipliers)
                bound_duals_lower = _unpack_bound_duals(
                    model, nlp_recovered.bound_multipliers_lower
                )
                bound_duals_upper = _unpack_bound_duals(
                    model, nlp_recovered.bound_multipliers_upper
                )
                # Zero bound multipliers on integer columns: they reflect the
                # cost of fixing the integer, not bound activity in the
                # original model. The examiner already drops integer columns
                # from stationarity, so zeros are safe here.
                if bound_duals_lower is not None or bound_duals_upper is not None:
                    for v in model._variables:
                        if v.var_type in (VarType.BINARY, VarType.INTEGER):
                            if bound_duals_lower is not None and v.name in bound_duals_lower:
                                bound_duals_lower[v.name] = np.zeros_like(bound_duals_lower[v.name])
                            if bound_duals_upper is not None and v.name in bound_duals_upper:
                                bound_duals_upper[v.name] = np.zeros_like(bound_duals_upper[v.name])

                # Adopt the refined primal from the KKT re-solve. It solves the
                # same integer-fixed subproblem to tighter precision than the
                # batched JAX IPM, so the continuous variables (and thus
                # active-constraint residuals) stay consistent with the
                # recovered duals — without this, complementarity (mu * residual)
                # can exceed validation tolerances. Guarded: only adopt an
                # OPTIMAL re-solve whose objective matches the incumbent, so a
                # divergent recover can never corrupt the reported solution.
                if (
                    nlp_recovered.status == SolveStatus.OPTIMAL
                    and nlp_recovered.x is not None
                    and np.all(np.isfinite(nlp_recovered.x))
                    and nlp_recovered.objective is not None
                    and abs(float(nlp_recovered.objective) - obj_val) <= 1e-4 * (1.0 + abs(obj_val))
                ):
                    refined = np.asarray(nlp_recovered.x, dtype=float).copy()
                    # Keep integer columns pinned at their (rounded) incumbent
                    # values; only the continuous variables are refined.
                    for off, sz in zip(int_offsets, int_sizes):
                        for k in range(int(sz)):
                            refined[off + k] = round(float(sol_flat[off + k]))
                    sol_flat = refined
                    x_dict = _unpack_solution(model, sol_flat)
                    obj_val = float(nlp_recovered.objective)
        except Exception as _exc:
            logger.debug("NLP-BB dual recovery failed: %s", _exc)

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        # "optimal" requires both a closed search AND a certified gap: a node
        # whose convex relaxation was not KKT-valid (roadmap P0.3) leaves the
        # bound uncertified, so the search closing does not prove optimality.
        if (tree.gap() <= gap_tolerance or tree.is_finished()) and _gap_certified:
            status = "optimal"
        else:
            status = "feasible"
    else:
        x_dict = None
        obj_val = None
        if stats["total_nodes"] >= max_nodes:
            status = "node_limit"
            # No incumbent and a resource limit: nothing is certified. Without an
            # incumbent there is no gap to certify, and a leftover tree bound
            # describes an unexplored search, not a proven optimum — a certified
            # exit here would claim optimality with no solution at all.
            _gap_certified = False
        elif wall_time >= time_limit:
            status = "time_limit"
            _gap_certified = False
        elif _unconverged_fathom:
            # Tree exhausted with no incumbent, but at least one node was fathomed
            # on NLP NON-convergence (a stall), not a valid infeasibility
            # certificate. We cannot soundly claim the problem is infeasible — its
            # feasibility is genuinely undetermined. Report "unknown" rather than a
            # false "infeasible" (the worst-class error: a feasible problem
            # declared infeasible). Conservative by design: if any fathom was
            # unconverged we forgo certifying infeasibility we might otherwise have
            # proven — soundness over capability.
            status = "unknown"
            _gap_certified = False
        else:
            # Tree exhausted with no feasible node and every fathom was a valid
            # certificate (FBBT-empty box or SolveStatus.INFEASIBLE): infeasibility
            # *is* a certified conclusion, so leave _gap_certified untouched.
            status = "infeasible"

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    assert model._objective is not None
    if bound_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        bound_val = -bound_val

    # An uncertified gap is not a rigorous dual bound; do not present one.
    gap_val = stats["gap"]

    # A *feasible* exit never inherits the tree's validity flag as a certificate:
    # the flag attests bound validity, not gap closure, and a node/budget-limited
    # feasible exit leaves the tree gap open. See the spatial-path note above
    # (max_nodes=1 false-certification regression). This path carries no rigorous
    # root-relaxation fallback, so a dropped certificate is not re-earned here.
    if status == "feasible":
        _gap_certified = False

    if not _gap_certified:
        bound_val = None
        gap_val = None

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=bound_val,
        gap=gap_val,
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
        nlp_bb=True,
        gap_certified=_gap_certified,
        constraint_duals=constraint_duals,
        bound_duals_lower=bound_duals_lower,
        bound_duals_upper=bound_duals_upper,
    )


def _solve_node_nlp(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
    nlp_solver: str = "ipopt",
    convex: bool = False,
):
    """Solve the NLP relaxation at a single B&B node with tightened bounds.

    We override variable bounds to use the node-specific bounds
    rather than the global bounds.
    """
    # Pre-screen: detect trivially infeasible nodes by evaluating constraints
    # at the midpoint. When the feasible region is very narrow (most variables
    # pinned) and constraints are violated, NLP solvers like POUNCE can stall
    # for thousands of iterations instead of quickly returning infeasible.
    if constraint_bounds is not None and evaluator.n_constraints > 0:
        from discopt.solvers import NLPResult

        x_mid = np.clip(x0, node_lb, node_ub)
        # Clip first: unbounded vars produce inf-(-inf)=NaN under raw subtract,
        # which then disables this pre-screen on every node with free vars.
        lb_c = np.clip(node_lb, -_SPC, _SPC)
        ub_c = np.clip(node_ub, -_SPC, _SPC)
        span = ub_c - lb_c
        n_pinned = np.sum(span < 1e-10)
        if n_pinned >= len(span) - 1:
            # Nearly all variables pinned: evaluate constraints at midpoint
            try:
                g = evaluator.evaluate_constraints(x_mid)
                infeasible = False
                for k, (cl, cu) in enumerate(constraint_bounds):
                    if g[k] < cl - 1e-6 or g[k] > cu + 1e-6:
                        infeasible = True
                        break
                if infeasible:
                    # Verify at the bounds midpoint too
                    x_check = 0.5 * (lb_c + ub_c)
                    g2 = evaluator.evaluate_constraints(x_check)
                    still_infeasible = False
                    for k, (cl, cu) in enumerate(constraint_bounds):
                        if g2[k] < cl - 1e-6 or g2[k] > cu + 1e-6:
                            still_infeasible = True
                            break
                    if still_infeasible:
                        return NLPResult(
                            status=SolveStatus.INFEASIBLE,
                            x=x_mid,
                            objective=_INFEASIBILITY_SENTINEL,
                        )
            except Exception:
                pass  # If evaluation fails, fall through to NLP solver

    if nlp_solver in ("pounce", "ipm", "sparse_ipm"):
        # "ipm"/"sparse_ipm" are deprecated aliases — the JAX IPM is retired as a
        # node NLP solver, so all route to POUNCE (the pure-Rust Ipopt port).
        return _solve_node_nlp_pounce(
            evaluator, x0, node_lb, node_ub, constraint_bounds, options, convex=convex
        )
    return _solve_node_nlp_ipopt(evaluator, x0, node_lb, node_ub, constraint_bounds, options)


def _solve_node_nlp_pounce(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
    convex: bool = False,
):
    """Solve node NLP with POUNCE (pure-Rust Ipopt port).

    Robustness layer (roadmap P0.2/P0.3):

    - A failed solve (error / divergence / *local* infeasibility — which on a
      nonconvex node is not an infeasibility proof) is retried from up to two
      alternative deterministic starts (box midpoint, then an off-center
      point) before the failure is reported. Without this a single bad start
      costs the node its bound and decertifies the gap.
    - When ``convex=True`` and the result is ITERATION_LIMIT (non-KKT, so the
      objective is not a valid lower bound), one polish re-solve at a boosted
      iteration budget is attempted from the stalled iterate; only an OPTIMAL
      polish restores trust in the bound.
    """
    from discopt.solvers import NLPResult
    from discopt.solvers.nlp_pounce import solve_nlp as solve_nlp_pounce

    proxy = _BoundOverrideEvaluator(evaluator, node_lb, node_ub)

    # Guard against POUNCE stalling on degenerate/infeasible subproblems
    # by enforcing a per-node wall time limit. Cap at 30s per node, but
    # also respect the remaining global budget passed via options (issue #5).
    opts = dict(options)
    caller_limit = opts.get("max_wall_time", 30.0)
    if caller_limit <= 0:
        caller_limit = 30.0
    opts["max_wall_time"] = min(30.0, caller_limit)

    def _attempt(start: np.ndarray, attempt_opts: dict):
        try:
            return solve_nlp_pounce(
                proxy,  # type: ignore[arg-type]
                start,
                constraint_bounds=constraint_bounds,
                options=attempt_opts,
            )
        except Exception as e:
            logger.debug("POUNCE solver failed: %s", e)
            return NLPResult(status=SolveStatus.ERROR, x=start, objective=_INFEASIBILITY_SENTINEL)

    lb_c = np.clip(np.asarray(node_lb, dtype=np.float64), -_SPC, _SPC)
    ub_c = np.clip(np.asarray(node_ub, dtype=np.float64), -_SPC, _SPC)
    midpoint = 0.5 * (lb_c + ub_c)
    # Deterministic off-center fallback start (no RNG: determinism by default).
    off_center = lb_c + 0.382 * (ub_c - lb_c)

    result = _attempt(np.asarray(x0, dtype=np.float64), opts)
    if result.status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
        for alt in (midpoint, off_center):
            if np.allclose(alt, x0, atol=1e-12):
                continue
            retry = _attempt(alt, opts)
            if retry.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
                result = retry
                break

    # Convex polish-retry: a non-KKT objective is not a valid LB; one boosted
    # re-solve from the stalled iterate often reaches KKT (trusted again).
    if convex and result.status == SolveStatus.ITERATION_LIMIT:
        polish_opts = dict(opts)
        polish_opts["max_iter"] = max(3 * int(opts.get("max_iter", 1000) or 1000), 3000)
        start = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(start)):
            start = midpoint
        polished = _attempt(np.clip(start, node_lb, node_ub), polish_opts)
        if (
            polished.status == SolveStatus.OPTIMAL
            and polished.objective is not None
            and np.isfinite(polished.objective)
        ):
            return polished

    return result


def _solve_batch_pounce(
    evaluator,
    batch_lb,
    batch_ub,
    batch_ids,
    n_vars,
    constraint_bounds,
    options,
    batch_psols=None,
    multistart=False,
    convex=False,
):
    """Solve a batch of node NLP relaxations in parallel with POUNCE.

    Phase A of discopt#97: the general-NLP analog of :func:`_solve_batch_ipm`,
    using POUNCE's ``solve_nlp_batch`` (pounce#126). Each (node, start) pair
    becomes one callback ``pounce.Problem`` differing only in its tightened
    variable bounds and initial point; POUNCE runs one instance per Rayon
    worker (the GIL is released except during the per-evaluation callbacks).
    Siblings share KKT sparsity, so ``share_structure=True`` reuses each
    worker's symbolic factorization across the instances it handles.

    Returns the same ``(result_ids, result_lbs, result_sols, result_feas)``
    contract as :func:`_solve_batch_ipm`: ``result_lbs`` holds the NLP
    objective per node (or ``_INFEASIBILITY_SENTINEL`` for failed / infeasible
    nodes), and ``result_feas`` is all-``False`` (the Rust tree checks
    integrality). POUNCE is a true filter-IPM (KKT-valid), so unlike the JAX
    IPM its converged objective needs no polish pass on convex models; the
    caller still applies the nonconvex / constraint-feasibility post-checks.

    Starting points per node:

    * ``multistart=False`` or ``convex=True`` → a single warm start (the
      parent solution clipped into the child bounds, else the bound midpoint).
      A convex node has a unique optimum, so one start reaches it — POUNCE
      needs no multistart there, unlike the JAX IPM.
    * ``multistart=True`` and ``convex=False`` → three starts per node
      (warm, midpoint, deterministic-random), keeping the best converged
      objective. Mirrors :func:`_solve_batch_ipm`'s scheme; on nonconvex nodes
      a better local solution becomes a stronger incumbent and prunes the
      tree faster.
    """
    import pounce

    from discopt.solvers.nlp_ipopt import (
        _IPOPT_STATUS_MAP,
        _infer_constraint_bounds,
        _IpoptCallbacks,
    )

    n_batch = len(batch_ids)
    m = evaluator.n_constraints

    # Constraint bounds are shared across nodes; only variable bounds vary.
    if constraint_bounds is not None:
        cl = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        cu = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    elif m > 0:
        cl, cu = _infer_constraint_bounds(evaluator)
    else:
        cl = np.empty(0, dtype=np.float64)
        cu = np.empty(0, dtype=np.float64)

    # Batch-level options. Whitelist keys POUNCE understands and enforce the
    # per-node wall-time guard (issue #5) used by the serial pounce path.
    batch_opts: dict = {"print_level": 0}
    for k in ("max_iter", "tol", "acceptable_tol"):
        if options.get(k) is not None:
            batch_opts[k] = options[k]
    caller_limit = options.get("max_wall_time", 30.0)
    if not caller_limit or caller_limit <= 0:
        caller_limit = 30.0
    batch_opts["max_wall_time"] = min(30.0, caller_limit)

    # Multistart only helps escape weak local minima on nonconvex nodes.
    do_multistart = bool(multistart) and not convex
    n_starts = 3 if do_multistart else 1
    rng = np.random.RandomState(42) if do_multistart else None

    # Flatten (node, start) into one problem list; node i occupies the slice
    # [i * n_starts : (i + 1) * n_starts].
    problems = []
    x0s = []
    node_bounds = []
    for i in range(n_batch):
        node_lb = np.asarray(batch_lb[i], dtype=np.float64)
        node_ub = np.asarray(batch_ub[i], dtype=np.float64)
        node_bounds.append((node_lb, node_ub))

        lb_c = np.clip(node_lb, -_SPC, _SPC)
        ub_c = np.clip(node_ub, -_SPC, _SPC)
        midpoint = 0.5 * (lb_c + ub_c)

        # Warm start: parent solution clipped into child bounds, else midpoint.
        if batch_psols is not None:
            psol_i = np.asarray(batch_psols[i], dtype=np.float64)
            warm = np.clip(psol_i, node_lb, node_ub) if not np.any(np.isnan(psol_i)) else midpoint
        else:
            warm = midpoint

        if do_multistart:
            span = np.maximum(ub_c - lb_c, 0.0)
            rand = lb_c + rng.uniform(size=n_vars) * span  # type: ignore[union-attr]
            node_starts = [warm, midpoint, rand]
        else:
            node_starts = [warm]

        for x0 in node_starts:
            # One callbacks proxy per problem so concurrent Rayon workers never
            # share mutable Python state. The JAX evaluator is pure/reentrant.
            proxy = _BoundOverrideEvaluator(evaluator, node_lb, node_ub)
            callbacks = _IpoptCallbacks(proxy)
            problems.append(
                pounce.Problem(
                    n=n_vars,
                    m=m,
                    problem_obj=callbacks,
                    lb=node_lb,
                    ub=node_ub,
                    cl=cl,
                    cu=cu,
                )
            )
            x0s.append(np.asarray(x0, dtype=np.float64))

    result_ids = np.array(batch_ids, dtype=np.int64)
    result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
    result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
    result_feas = np.zeros(n_batch, dtype=bool)  # Let Rust check integrality
    # trusted[i] is False when result_lbs[i] is a numeric bound that is not a
    # valid lower bound: for a convex model, a non-KKT (ITERATION_LIMIT)
    # objective. POUNCE has no polish loop, so any non-optimal-but-usable
    # convex result is untrusted. The caller decertifies the gap on these
    # without touching the bound/incumbent (roadmap P0.3). Irrelevant for
    # nonconvex models (objective discarded by the caller) → stays True.
    trusted = np.ones(n_batch, dtype=bool)

    try:
        results = pounce.solve_nlp_batch(
            problems,
            x0s=x0s,
            options=batch_opts,
            share_structure=True,
        )
    except Exception as e:
        # Whole-batch failure: fall back to per-node serial solves so one bad
        # instance can't sink the iteration (single warm start per node).
        logger.debug("Batch POUNCE failed (%s); falling back to serial nodes", e)
        for i in range(n_batch):
            node_lb, node_ub = node_bounds[i]
            res = _solve_node_nlp_pounce(
                evaluator,
                x0s[i * n_starts],
                node_lb,
                node_ub,
                constraint_bounds,
                options,
                convex=convex,
            )
            result_sols[i] = np.asarray(res.x, dtype=np.float64)
            if res.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
                obj = float(res.objective)
                if np.isfinite(obj):
                    result_lbs[i] = obj
                if convex and res.status != SolveStatus.OPTIMAL:
                    trusted[i] = False
        return result_ids, result_lbs, result_sols, result_feas, trusted

    # Reduce the starts of each node to its best accepted result. The evaluator
    # objective is in minimization sense (maximize models are negated), so the
    # lowest objective is best — matching _solve_batch_ipm's argmin.
    for i in range(n_batch):
        best_obj = None
        best_x = None
        best_status = None
        for s in range(n_starts):
            x, info = results[i * n_starts + s]
            x_arr = np.asarray(x, dtype=np.float64)
            if best_x is None:
                best_x = x_arr  # placeholder if no start is accepted
            status = _IPOPT_STATUS_MAP.get(info.get("status", -100), SolveStatus.ERROR)
            # Accept the same statuses as the serial pounce node path; anything
            # else (infeasible, restoration failure, errors) is not usable.
            if status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
                obj = float(info.get("obj_val", np.nan))
                if np.isfinite(obj) and (best_obj is None or obj < best_obj):
                    best_obj = obj
                    best_x = x_arr
                    best_status = status
        result_sols[i] = best_x
        if best_obj is not None:
            result_lbs[i] = best_obj
            # A non-KKT (ITERATION_LIMIT) convex objective is not a valid LB.
            # Try one boosted polish re-solve from the stalled iterate before
            # giving up trust (P0.3 polish-retry); only OPTIMAL restores it.
            if convex and best_status != SolveStatus.OPTIMAL:
                node_lb_i, node_ub_i = node_bounds[i]
                polish = _solve_node_nlp_pounce(
                    evaluator,
                    np.clip(best_x, node_lb_i, node_ub_i),
                    node_lb_i,
                    node_ub_i,
                    constraint_bounds,
                    {**options, "max_iter": max(3 * int(options.get("max_iter") or 1000), 3000)},
                )
                if (
                    polish.status == SolveStatus.OPTIMAL
                    and polish.objective is not None
                    and np.isfinite(polish.objective)
                ):
                    result_lbs[i] = float(polish.objective)
                    result_sols[i] = np.asarray(polish.x, dtype=np.float64)
                else:
                    trusted[i] = False

    return result_ids, result_lbs, result_sols, result_feas, trusted


def _solve_node_nlp_ipopt(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
):
    """Solve node NLP with cyipopt (Ipopt)."""
    try:
        import cyipopt
    except ImportError:
        raise ImportError("cyipopt is required. Install it with: pip install cyipopt")

    from discopt.solvers.nlp_ipopt import _IpoptCallbacks

    n = evaluator.n_variables
    m = evaluator.n_constraints
    callbacks = _IpoptCallbacks(evaluator)

    if constraint_bounds is not None:
        cl = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        cu = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    else:
        cl = np.empty(0, dtype=np.float64)
        cu = np.empty(0, dtype=np.float64)

    problem = cyipopt.Problem(
        n=n,
        m=m,
        problem_obj=callbacks,
        lb=node_lb.astype(np.float64),
        ub=node_ub.astype(np.float64),
        cl=cl,
        cu=cu,
    )

    # cyipopt requires native Python types (rejects numpy scalars).
    # Some options (e.g. max_wall_time) may not exist in older Ipopt versions.
    for key, value in options.items():
        try:
            if isinstance(value, (np.floating, float)):
                problem.add_option(key, float(value))
            elif isinstance(value, (np.integer, int)):
                problem.add_option(key, int(value))
            else:
                problem.add_option(key, value)
        except TypeError:
            logger.debug("Ipopt option '%s' not accepted, skipping", key)

    from discopt.solvers import NLPResult

    try:
        x, info = problem.solve(x0.astype(np.float64))
    except Exception as e:
        logger.debug("Ipopt solver failed: %s", e)
        return NLPResult(
            status=SolveStatus.ERROR,
            x=x0,
            objective=_INFEASIBILITY_SENTINEL,
        )

    from discopt.solvers.nlp_ipopt import _IPOPT_STATUS_MAP

    status_code = info["status"]
    status = _IPOPT_STATUS_MAP.get(status_code, SolveStatus.ERROR)

    mult_g = info.get("mult_g")
    mult_x_L = info.get("mult_x_L")
    mult_x_U = info.get("mult_x_U")

    return NLPResult(
        status=status,
        x=np.asarray(x),
        objective=float(info["obj_val"]),
        multipliers=np.asarray(mult_g) if mult_g is not None else None,
        bound_multipliers_lower=np.asarray(mult_x_L) if mult_x_L is not None else None,
        bound_multipliers_upper=np.asarray(mult_x_U) if mult_x_U is not None else None,
    )


def _solve_node_nlp_kkt(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
):
    """Solve a node NLP with a KKT-valid solver for B&B polish passes.

    Prefers POUNCE (pure-Rust Ipopt port); falls back to cyipopt when POUNCE
    is unavailable. Used to re-solve nodes whose IPM iterate stalled at a
    non-KKT point, where the objective is not a valid lower bound.
    """
    if _default_nlp_solver() == "pounce":
        return _solve_node_nlp_pounce(evaluator, x0, node_lb, node_ub, constraint_bounds, options)
    return _solve_node_nlp_ipopt(evaluator, x0, node_lb, node_ub, constraint_bounds, options)


# ---------------------------------------------------------------------------
# Specialized LP/QP solvers
# ---------------------------------------------------------------------------


def _scalar_constraint_layout(
    model: Model,
) -> Optional[tuple[list[str], list[tuple[str, str]]]]:
    """Return per-row identity for the LP/QP fast paths, when all constraints
    are scalar.

    The algebraic and repr-based extractors used by ``extract_lp_data`` /
    ``extract_qp_data`` partition rows into ``[equalities..., inequalities...]``
    in the order constraints appear in ``model._constraints``. This helper
    returns parallel lists ``(eq_names, ub_info)``:

      - ``eq_names[i]`` is the constraint name for the ``i``-th equality row.
      - ``ub_info[j] = (name, original_sense)`` for the ``j``-th inequality
        row, where ``original_sense`` is ``"<="`` or ``">="`` (used to flip
        the dual sign for ``">="`` constraints that the extractor negated).

    Returns ``None`` if any constraint body is non-scalar — vector-bodied
    constraints don't have a one-row-per-constraint mapping the LP path can
    rely on, and the LP fast path's algebraic/repr extractors don't handle
    them either.
    """
    eq_names: list[str] = []
    ub_info: list[tuple[str, str]] = []
    for idx, con in enumerate(model._constraints):
        if not isinstance(con, Constraint):
            continue
        body_shape = getattr(con.body, "shape", ())
        if body_shape not in ((), (1,)):
            return None
        name = con.name if con.name else f"c{idx}"
        if con.sense == "==":
            eq_names.append(name)
        elif con.sense in ("<=", ">="):
            ub_info.append((name, con.sense))
        else:
            return None
    return eq_names, ub_info


def _lp_qp_unpack_duals(
    model: Model,
    *,
    row_dual: Optional[np.ndarray],
    col_dual: Optional[np.ndarray],
    n_eq: int,
    n_ub: int,
    n_orig: int,
) -> tuple[
    Optional[dict[str, np.ndarray]],
    Optional[dict[str, np.ndarray]],
    Optional[dict[str, np.ndarray]],
]:
    """Translate HiGHS row duals + reduced costs into discopt's named-dual
    convention.

    HiGHS row dual layout matches the constraint matrix the wrapper assembles:
    ``A_ub`` rows first (multipliers ≥ 0), then ``A_eq`` rows (free).
    ``_decompose_eq_slack_form`` emits inequalities to ``A_ub`` in declared
    order, flipping the row sign for ``">="`` so HiGHS sees ``-body ≤ 0``;
    we flip the multiplier back so the returned dual reflects the original
    ``">="`` body (giving ``μ ≤ 0`` in the examiner convention).

    Reduced costs are split into lower- and upper-bound multipliers using the
    examiner's convention ``λ_lb = max(rc, 0)``, ``λ_ub = max(-rc, 0)``.

    Returns ``(constraint_duals, bound_duals_lower, bound_duals_upper)``;
    any entry may be ``None`` if the underlying solver did not return that
    family of multipliers, or the row layout cannot be mapped (vector body,
    extractor mismatch, etc.).
    """
    layout = _scalar_constraint_layout(model)
    if layout is None:
        return None, None, None
    eq_names, ub_info = layout
    if len(eq_names) != n_eq or len(ub_info) != n_ub:
        return None, None, None

    constraint_duals: Optional[dict[str, np.ndarray]] = None
    if row_dual is not None and row_dual.size == n_ub + n_eq:
        # HiGHS reports row duals as ∂obj/∂b. The examiner uses the Lagrangian
        # convention μ s.t. ∇f + ∇body·μ = 0, with μ ≥ 0 for "<=" and μ ≤ 0
        # for ">=", so we negate for "<=" and "==" rows. For ">=" the
        # extractor already flipped the row to "-body ≤ const", which (after
        # composing the two negations) leaves the HiGHS row_dual unflipped.
        out: dict[str, np.ndarray] = {}
        for j, (name, original_sense) in enumerate(ub_info):
            rd = float(row_dual[j])
            mu = rd if original_sense == ">=" else -rd
            out[name] = np.asarray(mu, dtype=float).reshape(())
        for i, name in enumerate(eq_names):
            mu = -float(row_dual[n_ub + i])
            out[name] = np.asarray(mu, dtype=float).reshape(())
        constraint_duals = out

    bound_duals_lower: Optional[dict[str, np.ndarray]] = None
    bound_duals_upper: Optional[dict[str, np.ndarray]] = None
    if col_dual is not None and col_dual.size >= n_orig:
        rc = np.asarray(col_dual[:n_orig], dtype=float)
        lam_lb = np.maximum(rc, 0.0)
        lam_ub = np.maximum(-rc, 0.0)
        lo: dict[str, np.ndarray] = {}
        up: dict[str, np.ndarray] = {}
        offset = 0
        for v in model._variables:
            sz = int(v.size)
            chunk_lo = lam_lb[offset : offset + sz]
            chunk_up = lam_ub[offset : offset + sz]
            if v.shape == () or v.shape == (1,):
                lo[v.name] = chunk_lo.reshape(v.shape) if v.shape == (1,) else chunk_lo.reshape(())
                up[v.name] = chunk_up.reshape(v.shape) if v.shape == (1,) else chunk_up.reshape(())
            else:
                lo[v.name] = chunk_lo.reshape(v.shape)
                up[v.name] = chunk_up.reshape(v.shape)
            offset += sz
        bound_duals_lower = lo
        bound_duals_upper = up

    return constraint_duals, bound_duals_lower, bound_duals_upper


def _mip_recover_relaxation_duals(
    model: Model,
    *,
    lp_data,
    x_flat: np.ndarray,
    n_orig: int,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    time_limit: Optional[float] = None,
    Q_orig: Optional[np.ndarray] = None,
    prefer_pounce: bool = False,
) -> tuple[
    Optional[dict[str, np.ndarray]],
    Optional[dict[str, np.ndarray]],
    Optional[dict[str, np.ndarray]],
]:
    """Re-solve the relaxation with integer variables fixed at their MIP
    incumbent so the LP/QP solver returns row duals + reduced costs we can map
    back to discopt's named-dual convention.

    Pass ``Q_orig`` for QP-style relaxations; omit it for LP-style. With
    ``prefer_pounce`` the fix-and-resolve uses POUNCE (HiGHS-free path);
    otherwise HiGHS. Returns ``(None, None, None)`` if recovery is unavailable
    (solver missing, the fix-and-resolve LP/QP itself fails, or layout
    mismatch).

    Bound multipliers on the fixing bounds for integer columns are zeroed in
    the returned dicts — they reflect the act of fixing, not feasibility of
    the original integer-feasible point.
    """
    try:
        if prefer_pounce:
            if Q_orig is None:
                from discopt.solvers.lp_pounce import solve_lp as _highs_solve_lp
            else:
                from discopt.solvers.qp_pounce import solve_qp as _highs_solve_qp
        elif Q_orig is None:
            from discopt.solvers.lp_highs import (  # type: ignore[assignment]
                solve_lp as _highs_solve_lp,
            )
        else:
            from discopt.solvers.qp_highs import (  # type: ignore[assignment]
                solve_qp as _highs_solve_qp,
            )
    except ImportError:
        return None, None, None

    from discopt.solvers import SolveStatus

    bounds_fixed: list[tuple[float, float]] = []
    is_integer_col = np.zeros(n_orig, dtype=bool)
    offset = 0
    for v in model._variables:
        sz = int(v.size)
        is_int = v.var_type in (VarType.BINARY, VarType.INTEGER)
        for k in range(sz):
            if is_int:
                val = float(round(float(x_flat[offset + k])))
                bounds_fixed.append((val, val))
                is_integer_col[offset + k] = True
            else:
                lb_k = float(np.asarray(lp_data.x_l[offset + k]))
                ub_k = float(np.asarray(lp_data.x_u[offset + k]))
                bounds_fixed.append((lb_k, ub_k))
        offset += sz

    c_orig = np.asarray(lp_data.c[:n_orig])

    try:
        relax: Any
        if Q_orig is None:
            relax = _highs_solve_lp(
                c=c_orig,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds_fixed,
                time_limit=time_limit,
            )
        else:
            relax = _highs_solve_qp(
                Q=Q_orig,
                c=c_orig,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds_fixed,
                integrality=None,
                time_limit=time_limit,
            )
    except Exception as exc:
        logger.debug("MIP-relaxation dual recovery failed: %s", exc)
        return None, None, None

    if relax.status != SolveStatus.OPTIMAL:
        return None, None, None

    n_eq_rows = A_eq.shape[0] if A_eq is not None else 0
    n_ub_rows = A_ub.shape[0] if A_ub is not None else 0
    cd, bdl, bdu = _lp_qp_unpack_duals(
        model,
        row_dual=relax.dual_values,
        col_dual=relax.reduced_costs,
        n_eq=n_eq_rows,
        n_ub=n_ub_rows,
        n_orig=n_orig,
    )

    # Zero out bound multipliers on the fix-bounds of integer columns; they
    # quantify the cost of fixing, not bound activity in the original model.
    if (bdl is not None or bdu is not None) and is_integer_col.any():
        offset = 0
        for v in model._variables:
            sz = int(v.size)
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                if bdl is not None and v.name in bdl:
                    bdl[v.name] = np.zeros_like(bdl[v.name])
                if bdu is not None and v.name in bdu:
                    bdu[v.name] = np.zeros_like(bdu[v.name])
            offset += sz

    return cd, bdl, bdu


def _decompose_eq_slack_form(
    A_eq_full: np.ndarray,
    b_eq_full: np.ndarray,
    n_orig: int,
    n_slack: int,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Reconstruct (A_ub, b_ub, A_eq, b_eq) from an equality-plus-slack form.

    `extract_lp_data` / `extract_qp_data` convert inequalities to equalities
    with non-negative slacks. Rows whose slack column is nonzero are the
    original inequalities; rows with no slack are true equalities. This
    helper projects back to inequality/equality form over the original
    variables so HiGHS LP/MILP/QP solvers can consume it.
    """
    if A_eq_full.shape[0] == 0:
        return None, None, None, None

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    for i in range(A_eq_full.shape[0]):
        slack_part = A_eq_full[i, n_orig:]
        has_slack = n_slack > 0 and np.any(np.abs(slack_part) > 1e-15)
        if has_slack:
            slack_idx = np.argmax(np.abs(slack_part))
            slack_coef = slack_part[slack_idx]
            orig_row = A_eq_full[i, :n_orig]
            rhs = b_eq_full[i]
            if slack_coef > 0:
                ub_rows.append(orig_row)
                ub_rhs.append(rhs)
            else:
                ub_rows.append(-orig_row)
                ub_rhs.append(-rhs)
        else:
            eq_rows.append(A_eq_full[i, :n_orig])
            eq_rhs.append(b_eq_full[i])

    A_ub = np.array(ub_rows, dtype=np.float64) if ub_rows else None
    b_ub = np.array(ub_rhs, dtype=np.float64) if ub_rows else None
    A_eq = np.array(eq_rows, dtype=np.float64) if eq_rows else None
    b_eq = np.array(eq_rhs, dtype=np.float64) if eq_rows else None
    return A_ub, b_ub, A_eq, b_eq


def _solve_lp(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
    prefer_pounce: bool = False,
) -> SolveResult:
    """Solve an LP through the first available engine, then the JAX LP IPM.

    Engine order is HiGHS -> POUNCE -> JAX IPM, or POUNCE -> HiGHS -> JAX IPM
    when ``prefer_pounce`` is set (the user passed ``nlp_solver="pounce"``,
    i.e. asked for POUNCE everywhere; roadmap P0.4). The pure-JAX IPM
    struggles on problems whose declared bounds exceed ~1e15 (it returns NaN
    via Newton blow-up on unbounded variables); HiGHS and POUNCE both handle
    unbounded columns natively, so the IPM is the last resort.
    """
    engines = [_solve_lp_highs, _solve_lp_pounce]
    if prefer_pounce:
        engines.reverse()
    for engine in engines:
        result = engine(model, t_start, time_limit)
        if result is not None:
            return result

    from discopt._jax.lp_ipm import lp_ipm_solve
    from discopt._jax.problem_classifier import extract_lp_data

    t_jax_start = time.perf_counter()
    lp_data = extract_lp_data(model)
    state = lp_ipm_solve(lp_data.c, lp_data.A_eq, lp_data.b_eq, lp_data.x_l, lp_data.x_u)
    jax_time = time.perf_counter() - t_jax_start
    wall_time = time.perf_counter() - t_start

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    x_flat = np.asarray(state.x[:n_orig])
    obj_val = float(state.obj) + lp_data.obj_const

    # Negate objective back for maximization (LP solver always minimizes)
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        obj_val = -obj_val

    conv = int(state.converged)
    if conv in (1, 2):
        status = "optimal"
    elif conv == 3:
        status = "iteration_limit"
    else:
        status = "error"

    sr = SolveResult(
        status=status,
        objective=obj_val,
        bound=obj_val if status == "optimal" else None,
        gap=_optimal_relative_gap(obj_val) if status == "optimal" else None,
        x=_unpack_solution(model, x_flat),
        wall_time=wall_time,
        node_count=0,
        rust_time=0.0,
        jax_time=jax_time,
        python_time=wall_time - jax_time,
    )
    # LPs are convex by definition; mark for parity with the QP/NLP fast paths.
    sr.convex_fast_path = True
    return sr


def _solve_lp_highs(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
) -> SolveResult | None:
    """Solve an LP using HiGHS. Returns None when HiGHS is unavailable or
    the HiGHS wrapper fails, so the caller can fall back to another engine."""
    try:
        from discopt.solvers.lp_highs import solve_lp as _highs_solve_lp
    except ImportError:
        return None
    return _solve_lp_matrix(model, t_start, time_limit, _highs_solve_lp, "HiGHS")


def _solve_lp_pounce(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
) -> SolveResult | None:
    """Solve an LP using POUNCE (pure-Rust IPM). Returns None when POUNCE is
    unavailable or fails, so the caller can fall back to another engine."""
    import functools

    from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
    from discopt.solvers.lp_pounce import solve_lp as _pounce_solve_lp

    if not POUNCE_AVAILABLE:
        return None
    # Request an infeasibility certificate so an infeasible model-level LP
    # surfaces *why* via SolveResult.infeasibility_certificate (roadmap P0.2).
    solve_fn = functools.partial(_pounce_solve_lp, certificate=True)
    return _solve_lp_matrix(model, t_start, time_limit, solve_fn, "POUNCE")


def _solve_lp_matrix(
    model: Model,
    t_start: float,
    time_limit: float | None,
    solve_lp_fn,
    engine: str,
) -> SolveResult | None:
    """Solve a pure LP through a matrix-form ``solve_lp`` backend.

    ``solve_lp_fn`` must follow the shared LP contract (lp_highs / lp_pounce):
    same signature, same ``LPResult`` with HiGHS-convention duals.
    """
    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.modeling.core import ObjectiveSense
    from discopt.solvers import SolveStatus

    lp_data = extract_lp_data(model)
    n_orig = sum(v.size for v in model._variables)

    bounds = list(
        zip(
            np.asarray(lp_data.x_l[:n_orig]).tolist(),
            np.asarray(lp_data.x_u[:n_orig]).tolist(),
        )
    )

    n_total = lp_data.A_eq.shape[1] if lp_data.A_eq.shape[0] > 0 else n_orig
    n_slack = n_total - n_orig
    A_eq_full = np.asarray(lp_data.A_eq)
    b_eq_full = np.asarray(lp_data.b_eq)
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(A_eq_full, b_eq_full, n_orig, n_slack)

    try:
        result = solve_lp_fn(
            c=np.asarray(lp_data.c[:n_orig]),
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            time_limit=time_limit,
        )
    except Exception as e:
        logger.debug("%s LP solve failed: %s", engine, e)
        return None

    wall_time = time.perf_counter() - t_start

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and result.objective is not None
        if not _matrix_solution_feasible(result.x[:n_orig], A_ub, b_ub, A_eq, b_eq, bounds):
            logger.warning(
                "%s LP returned an infeasible point labeled optimal; "
                "falling back to the next engine.",
                engine,
            )
            return None
        obj_val = float(result.objective) + float(lp_data.obj_const)
        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        n_eq_rows = A_eq.shape[0] if A_eq is not None else 0
        n_ub_rows = A_ub.shape[0] if A_ub is not None else 0
        cd, bdl, bdu = _lp_qp_unpack_duals(
            model,
            row_dual=result.dual_values,
            col_dual=result.reduced_costs,
            n_eq=n_eq_rows,
            n_ub=n_ub_rows,
            n_orig=n_orig,
        )

        sr = SolveResult(
            status="optimal",
            objective=obj_val,
            bound=obj_val,
            gap=_optimal_relative_gap(obj_val),
            x=_unpack_solution(model, np.asarray(result.x[:n_orig])),
            wall_time=wall_time,
            node_count=0,
            rust_time=0.0,
            jax_time=0.0,
            python_time=wall_time,
            constraint_duals=cd,
            bound_duals_lower=bdl,
            bound_duals_upper=bdu,
        )
        sr.convex_fast_path = True
        return sr
    if result.status == SolveStatus.INFEASIBLE:
        return SolveResult(
            status="infeasible",
            wall_time=wall_time,
            infeasibility_certificate=getattr(result, "infeasibility_certificate", None),
        )
    if result.status == SolveStatus.UNBOUNDED:
        return SolveResult(status="unbounded", wall_time=wall_time)
    if result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(status="time_limit", wall_time=wall_time)
    return None


def _solve_qp(model: Model, t_start: float, prefer_pounce: bool = False) -> SolveResult:
    """Solve a QP through the first available engine, then the JAX QP IPM.

    Engine order is HiGHS -> POUNCE, or POUNCE -> HiGHS when ``prefer_pounce``
    is set (the user passed ``nlp_solver="pounce"``; roadmap P0.4). The POUNCE
    engine handles pure-continuous QPs only — MIQPs stay on HiGHS or fall
    through to the JAX path / B&B.
    """
    engines = [_solve_qp_highs, _solve_qp_pounce]
    if prefer_pounce:
        engines.reverse()
    for engine in engines:
        result = engine(model, t_start)
        if result is not None:
            return result
    return _solve_qp_jax(model, t_start)


def _solve_qp_highs(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
) -> SolveResult | None:
    """Solve a QP/MIQP using HiGHS. Returns None if HiGHS is unavailable."""
    try:
        from discopt.solvers.qp_highs import solve_qp as _highs_solve_qp
    except ImportError:
        return None
    return _solve_qp_matrix(model, t_start, time_limit, _highs_solve_qp, "HiGHS")


def _solve_qp_pounce(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
) -> SolveResult | None:
    """Solve a pure-continuous QP using POUNCE. Returns None when POUNCE is
    unavailable, the model has integer variables (no MIQP in an IPM), or the
    solve fails — so the caller can fall back to another engine."""
    import functools

    from discopt.solvers.qp_pounce import POUNCE_AVAILABLE
    from discopt.solvers.qp_pounce import solve_qp as _pounce_solve_qp

    if not POUNCE_AVAILABLE:
        return None
    if any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables):
        return None
    solve_fn = functools.partial(_pounce_solve_qp, certificate=True)
    return _solve_qp_matrix(model, t_start, time_limit, solve_fn, "POUNCE")


def _matrix_solution_feasible(x, A_ub, b_ub, A_eq, b_eq, bounds, tol=1e-6) -> bool:
    """Check a matrix-form LP/QP solution against its own constraints.

    Engines can mislabel results: HiGHS's QP solver has been observed to
    return a constraint-violating point flagged kOptimal (violation ~7.5 on a
    small random strictly convex QP). A point failing its own constraints is
    never accepted as optimal — the caller falls through to the next engine.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return False
    scale = 1.0 + float(np.max(np.abs(x)))
    if A_ub is not None and b_ub is not None and len(b_ub):
        if np.max(np.asarray(A_ub) @ x - np.asarray(b_ub)) > tol * scale:
            return False
    if A_eq is not None and b_eq is not None and len(b_eq):
        if np.max(np.abs(np.asarray(A_eq) @ x - np.asarray(b_eq))) > tol * scale:
            return False
    if bounds is not None:
        for xi, (lo, hi) in zip(x, bounds):
            if xi < lo - tol * scale or xi > hi + tol * scale:
                return False
    return True


def _solve_qp_matrix(
    model: Model,
    t_start: float,
    time_limit: float | None,
    solve_qp_fn,
    engine: str,
) -> SolveResult | None:
    """Solve a QP/MIQP through a matrix-form ``solve_qp`` backend.

    ``solve_qp_fn`` must follow the shared QP contract (qp_highs / qp_pounce):
    same signature, same ``QPResult`` with HiGHS-convention duals.
    """
    from discopt._jax.problem_classifier import extract_qp_data
    from discopt.modeling.core import ObjectiveSense
    from discopt.solvers import SolveStatus

    qp_data = extract_qp_data(model)
    n_orig = sum(v.size for v in model._variables)

    # Build bounds list (original variables only, no slacks)
    bounds = list(
        zip(
            np.asarray(qp_data.x_l[:n_orig]).tolist(),
            np.asarray(qp_data.x_u[:n_orig]).tolist(),
        )
    )

    n_total = qp_data.A_eq.shape[1] if qp_data.A_eq.shape[0] > 0 else n_orig
    n_slack = n_total - n_orig
    A_eq_full = np.asarray(qp_data.A_eq)
    b_eq_full = np.asarray(qp_data.b_eq)
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(A_eq_full, b_eq_full, n_orig, n_slack)

    # Build integrality array for MIQP
    integrality = None
    has_integer = any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables)
    if has_integer:
        int_arr = np.zeros(n_orig, dtype=np.int32)
        offset = 0
        for v in model._variables:
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                int_arr[offset : offset + v.size] = 1
            offset += v.size
        integrality = int_arr

    # Q matrix: only the original variable part (no slacks)
    Q_orig = np.asarray(qp_data.Q[:n_orig, :n_orig])
    c_orig = np.asarray(qp_data.c[:n_orig])

    try:
        result = solve_qp_fn(
            Q=Q_orig,
            c=c_orig,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            integrality=integrality,
            time_limit=time_limit,
        )
    except Exception as e:
        logger.debug("%s QP solve failed: %s", engine, e)
        return None

    wall_time = time.perf_counter() - t_start

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and result.objective is not None
        if not _matrix_solution_feasible(result.x[:n_orig], A_ub, b_ub, A_eq, b_eq, bounds):
            logger.warning(
                "%s QP returned an infeasible point labeled optimal; "
                "falling back to the next engine.",
                engine,
            )
            return None
        x_flat = result.x[:n_orig]
        obj_val = result.objective + qp_data.obj_const

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        n_eq_rows = A_eq.shape[0] if A_eq is not None else 0
        n_ub_rows = A_ub.shape[0] if A_ub is not None else 0
        if integrality is None:
            cd, bdl, bdu = _lp_qp_unpack_duals(
                model,
                row_dual=result.dual_values,
                col_dual=result.reduced_costs,
                n_eq=n_eq_rows,
                n_ub=n_ub_rows,
                n_orig=n_orig,
            )
        else:
            # MIQP: HiGHS doesn't expose MIP duals. Recover by re-solving the
            # QP relaxation with integers fixed at the MIP incumbent.
            cd, bdl, bdu = _mip_recover_relaxation_duals(
                model,
                lp_data=qp_data,
                x_flat=np.asarray(x_flat, dtype=float),
                n_orig=n_orig,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                time_limit=time_limit,
                Q_orig=Q_orig,
            )

        sr = SolveResult(
            status="optimal",
            objective=obj_val,
            bound=obj_val,
            gap=_optimal_relative_gap(obj_val),
            x=_unpack_solution(model, x_flat),
            wall_time=wall_time,
            node_count=result.node_count,
            rust_time=0.0,
            jax_time=0.0,
            python_time=wall_time,
            constraint_duals=cd,
            bound_duals_lower=bdl,
            bound_duals_upper=bdu,
        )
        # A detected QP with PSD Q is a convex problem solved directly without
        # B&B -- semantically the same as the convex NLP fast path.
        if integrality is None:
            sr.convex_fast_path = True
        return sr
    elif result.status == SolveStatus.INFEASIBLE:
        return SolveResult(
            status="infeasible",
            wall_time=wall_time,
            infeasibility_certificate=getattr(result, "infeasibility_certificate", None),
        )
    elif result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(status="time_limit", wall_time=wall_time)

    return None


def _solve_milp_highs(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
    gap_tolerance: float = 1e-4,
) -> SolveResult | None:
    """Solve a MILP using HiGHS MIP. Returns None if HiGHS is unavailable."""
    try:
        from discopt.solvers.milp_highs import solve_milp as _highs_solve_milp
    except ImportError:
        return None

    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.modeling.core import ObjectiveSense
    from discopt.solvers import SolveStatus

    lp_data = extract_lp_data(model)
    n_orig = sum(v.size for v in model._variables)

    bounds = list(
        zip(
            np.asarray(lp_data.x_l[:n_orig]).tolist(),
            np.asarray(lp_data.x_u[:n_orig]).tolist(),
        )
    )

    n_total = lp_data.A_eq.shape[1] if lp_data.A_eq.shape[0] > 0 else n_orig
    n_slack = n_total - n_orig
    A_eq_full = np.asarray(lp_data.A_eq)
    b_eq_full = np.asarray(lp_data.b_eq)
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(A_eq_full, b_eq_full, n_orig, n_slack)

    int_arr = np.zeros(n_orig, dtype=np.int32)
    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            int_arr[offset : offset + v.size] = 1
        offset += v.size

    c_orig = np.asarray(lp_data.c[:n_orig])

    try:
        result = _highs_solve_milp(
            c=c_orig,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            integrality=int_arr,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
        )
    except Exception as e:
        logger.debug("HiGHS MILP solve failed: %s", e)
        return None

    wall_time = time.perf_counter() - t_start

    assert model._objective is not None
    sense = model._objective.sense

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and result.objective is not None
        x_flat = result.x[:n_orig]
        obj_val = result.objective + lp_data.obj_const
        if sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val
        cd, bdl, bdu = _mip_recover_relaxation_duals(
            model,
            lp_data=lp_data,
            x_flat=np.asarray(x_flat, dtype=float),
            n_orig=n_orig,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            time_limit=time_limit,
        )
        return SolveResult(
            status="optimal",
            objective=obj_val,
            bound=obj_val,
            gap=result.gap if result.gap is not None else 0.0,
            x=_unpack_solution(model, x_flat),
            wall_time=wall_time,
            node_count=result.node_count,
            rust_time=0.0,
            jax_time=0.0,
            python_time=wall_time,
            constraint_duals=cd,
            bound_duals_lower=bdl,
            bound_duals_upper=bdu,
        )
    elif result.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=wall_time, node_count=result.node_count)
    elif result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(status="time_limit", wall_time=wall_time, node_count=result.node_count)

    return None


def _solve_qp_jax(model: Model, t_start: float) -> SolveResult:
    """Solve a QP using the pure-JAX QP IPM."""
    from discopt._jax.problem_classifier import extract_qp_data
    from discopt._jax.qp_ipm import qp_ipm_solve

    t_jax_start = time.perf_counter()
    qp_data = extract_qp_data(model)
    state = qp_ipm_solve(
        qp_data.Q,
        qp_data.c,
        qp_data.A_eq,
        qp_data.b_eq,
        qp_data.x_l,
        qp_data.x_u,
    )
    jax_time = time.perf_counter() - t_jax_start
    wall_time = time.perf_counter() - t_start

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    x_flat = np.asarray(state.x[:n_orig])
    obj_val = float(state.obj) + qp_data.obj_const

    # Negate objective back for maximization (QP solver always minimizes)
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        obj_val = -obj_val

    conv = int(state.converged)
    if conv in (1, 2):
        status = "optimal"
    elif conv == 3:
        status = "iteration_limit"
    else:
        status = "error"

    sr = SolveResult(
        status=status,
        objective=obj_val,
        bound=obj_val if status == "optimal" else None,
        gap=_optimal_relative_gap(obj_val) if status == "optimal" else None,
        x=_unpack_solution(model, x_flat),
        wall_time=wall_time,
        node_count=0,
        rust_time=0.0,
        jax_time=jax_time,
        python_time=wall_time - jax_time,
    )
    # QP dispatch only reaches this function for detected convex QPs.
    sr.convex_fast_path = True
    return sr


def _pounce_recover_node_bound(
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    c: np.ndarray,
    obj_const: float,
    A_ub,
    b_ub,
    A_eq,
    b_eq,
    t_start: float,
    time_limit: float,
    Q: Optional[np.ndarray] = None,
):
    """Re-solve a stalled (non-KKT) MILP/MIQP node relaxation with POUNCE.

    The JAX LP/QP IPM reports ``converged==3`` when it hits the iteration
    limit; that objective is not a valid node lower bound. POUNCE is a true
    filter IPM, so its OPTIMAL objective is KKT-valid and its INFEASIBLE is
    Phase-1-certified — either recovers the node instead of decertifying the
    whole gap (Phase 1 increment 2; mirrors the P0.3 polish-retry).

    Returns ``("optimal", bound, x)``, ``("infeasible", None, None)``, or
    ``None`` when POUNCE is unavailable / could not settle the node either.
    """
    try:
        if Q is None:
            from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
            from discopt.solvers.lp_pounce import solve_lp as _pounce_solve
        else:
            from discopt.solvers.qp_pounce import POUNCE_AVAILABLE
            from discopt.solvers.qp_pounce import (  # type: ignore[assignment]
                solve_qp as _pounce_solve,
            )
    except ImportError:
        return None
    if not POUNCE_AVAILABLE:
        return None

    time_left = max(0.5, time_limit - (time.perf_counter() - t_start))
    kwargs = dict(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=list(zip(np.asarray(node_lb).tolist(), np.asarray(node_ub).tolist())),
        time_limit=min(30.0, time_left),
    )
    if Q is not None:
        kwargs["Q"] = Q
    try:
        res = _pounce_solve(**kwargs)
    except Exception as e:
        logger.debug("POUNCE node-bound recovery failed: %s", e)
        return None
    if (
        res.status == SolveStatus.OPTIMAL
        and res.objective is not None
        and np.isfinite(res.objective)
    ):
        return ("optimal", float(res.objective) + float(obj_const), np.asarray(res.x))
    if res.status == SolveStatus.INFEASIBLE:
        return ("infeasible", None, None)
    return None


def _solve_node_lp_pounce(lp_data, node_lb, node_ub, n_vars, n_orig, t_start, time_limit):
    """Solve one MILP-B&B node relaxation with POUNCE on the augmented
    standard-form LP (Path B: POUNCE as the *primary* node engine in POUNCE
    mode, not just recovery).

    POUNCE (Rust) takes the cut-augmented LP at any shape with no per-shape JAX
    recompilation — the reason root cuts are cheap here — and its ``OPTIMAL``
    objective is KKT-valid, so no trust-gate decertification is needed. Solving
    the *augmented* ``lp_data`` directly preserves the root cuts' bound
    tightening (node bounds apply to the structural columns; the slack columns
    keep their ``lp_data`` bounds).

    Returns ``(lower_bound, solution_over_n_vars, "optimal")``,
    ``(sentinel, None, "infeasible")``, or ``None`` when POUNCE is unavailable
    or could not settle the node (the caller falls back / decertifies)."""
    try:
        import pounce

        from discopt.solvers.lp_pounce import POUNCE_AVAILABLE, _snap_inverted_bounds
    except ImportError:
        return None
    if not POUNCE_AVAILABLE:
        return None
    # Solve the node relaxation with POUNCE's *structured convex* LP/QP engine
    # (``pounce.solve_qp`` over the pounce-convex IPM, with presolve), not the
    # generic callback TNLP path (``solve_lp``). The callback path hides the
    # linear structure, so POUNCE's presolve cannot engage and its IPM takes
    # ~100 iterations (~3 s) on these degenerate, equality-pair-encoded
    # relaxations; the structured convex path presolves + scales and converges
    # in ~20 iterations (~0.03 s) on the same LP (~100x), which is what lets the
    # MILP B&B visit enough nodes to bound the tree. It needs the native
    # inequality form, so decompose the slack-expanded standard form back to
    # A_ub/A_eq over the structural columns.
    n_slack = int(lp_data.A_eq.shape[1]) - n_orig
    try:
        A_ub_m, b_ub_m, A_eq_m, b_eq_m = _decompose_eq_slack_form(
            np.asarray(lp_data.A_eq, dtype=np.float64),
            np.asarray(lp_data.b_eq, dtype=np.float64),
            n_orig,
            n_slack,
        )
        lb_n = np.asarray(node_lb, dtype=np.float64)
        ub_n = np.asarray(node_ub, dtype=np.float64)
        lb_n, ub_n = _snap_inverted_bounds(lb_n, ub_n)
        res = pounce.solve_qp(
            P=None,  # P=0 -> LP
            c=np.asarray(lp_data.c[:n_orig], dtype=np.float64),
            G=A_ub_m,
            h=b_ub_m,
            A=A_eq_m,
            b=b_eq_m,
            lb=lb_n,
            ub=ub_n,
        )
    except Exception as e:
        logger.debug("POUNCE convex node solve failed: %s", e)
        return None
    if res.status == "optimal" and res.x is not None and np.isfinite(res.obj):
        x_sol = np.asarray(res.x, dtype=np.float64)
        # Soundness gate (mirrors the JAX-IPM path's _check_lp_solution_feasibility):
        # reject a node point that violates its own rows so a slightly-infeasible
        # relaxation solution cannot seed a spurious incumbent or node bound.
        tol = 1e-5
        feasible = True
        if A_ub_m is not None and A_ub_m.shape[0]:
            feasible = bool(np.all(A_ub_m @ x_sol <= b_ub_m + tol * (1.0 + np.abs(b_ub_m))))
        if feasible and A_eq_m is not None and A_eq_m.shape[0]:
            feasible = bool(np.all(np.abs(A_eq_m @ x_sol - b_eq_m) <= tol * (1.0 + np.abs(b_eq_m))))
        if not feasible:
            logger.debug("POUNCE convex node solution violates its rows; rejecting")
            return None
        bound = float(res.obj) + float(lp_data.obj_const)
        return (bound, x_sol[:n_vars], "optimal")
    if res.status == "primal_infeasible":
        return (_INFEASIBILITY_SENTINEL, None, "infeasible")
    return None


# Interior-point relaxation optima are interior: integer coordinates come
# back smeared (e.g. 0.99996) — beyond the tree's 1e-5 integrality tolerance
# but clearly integral. Coordinates within this tolerance are candidates for
# the snap-fix-resolve purification below.
_SNAP_TOL = 1e-4


def _pounce_snap_incumbent(
    x_relax: np.ndarray,
    int_offsets,
    int_sizes,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    c: np.ndarray,
    obj_const: float,
    A_ub,
    b_ub,
    A_eq,
    b_eq,
    t_start: float,
    time_limit: float,
    Q: Optional[np.ndarray] = None,
):
    """Purify a near-integral relaxation point into an exact incumbent.

    Naively rounding the integer coordinates of an interior point breaks
    equality rows by ~the snap distance, so instead: round each integer
    coordinate that is within ``_SNAP_TOL`` of an integer, *fix* it there, and
    re-solve the continuous relaxation with POUNCE (Phase 1 increment 3).
    An OPTIMAL result is an exactly feasible integer point with an exact
    objective, suitable for ``tree.inject_incumbent``. Returns
    ``(objective, x)`` in minimization form, or ``None``.
    """
    idx = [j for off, sz in zip(int_offsets, int_sizes) for j in range(off, off + int(sz))]
    if not idx:
        return None
    vals = np.asarray(x_relax, dtype=np.float64)[idx]
    snapped = np.round(vals)
    if not np.all(np.isfinite(vals)) or float(np.max(np.abs(vals - snapped))) > _SNAP_TOL:
        return None
    fl = np.asarray(node_lb, dtype=np.float64).copy()
    fu = np.asarray(node_ub, dtype=np.float64).copy()
    # The snapped value must lie inside the *original* node box (pinning both
    # bounds afterwards would mask an out-of-box snap).
    if np.any(snapped < fl[idx] - 1e-9) or np.any(snapped > fu[idx] + 1e-9):
        return None
    fl[idx] = snapped
    fu[idx] = snapped
    rec = _pounce_recover_node_bound(
        fl, fu, c, obj_const, A_ub, b_ub, A_eq, b_eq, t_start, time_limit, Q=Q
    )
    if rec is not None and rec[0] == "optimal":
        return rec[1], rec[2]
    return None


# Reduced costs below this are treated as zero (basic / degenerate -> no fix).
_RCF_RC_TOL = 1e-7


def _reduced_cost_fixing(lb, ub, int_idx, reduced_costs, z_lp, z_inc):
    """Tighten integer variable bounds by LP reduced-cost fixing.

    For a minimization relaxation with optimum ``z_lp`` (a valid lower bound),
    reduced costs ``d = c - A^T y``, and an incumbent objective ``z_inc`` (a
    valid upper bound), every improving feasible point satisfies, term by
    term (all terms are non-negative at the LP optimum),

        d_j * (x_j - bound_j) <= z_inc - z_lp =: gap

    which caps how far each non-basic integer variable can move from the
    bound it is pressed against:

        d_j >  tol (pressed to lb): x_j <= lb_j + floor(gap / d_j)
        d_j < -tol (pressed to ub): x_j >= ub_j - floor(gap / |d_j|)

    The true optimum ``x*`` has objective ``<= z_inc``, so it satisfies these
    bounds — RCF never cuts it. ``gap`` is inflated by a small relative
    margin so interior-point dual tolerance cannot over-tighten. Returns
    tightened ``(lb, ub)`` copies and the number of bound changes.
    """
    lb = np.array(lb, dtype=np.float64).copy()
    ub = np.array(ub, dtype=np.float64).copy()
    gap = float(z_inc) - float(z_lp)
    if not np.isfinite(gap) or gap < 0:
        return lb, ub, 0
    # Safety margin for IPM dual tolerance: never tighten so hard we risk the
    # optimum (correctness over aggressiveness).
    gap += 1e-6 * (1.0 + abs(float(z_inc)))
    changes = 0
    for j in int_idx:
        d = float(reduced_costs[j])
        if d > _RCF_RC_TOL:
            new_ub = lb[j] + float(np.floor(gap / d + 1e-9))
            if new_ub < ub[j] - 0.5:
                ub[j] = max(lb[j], new_ub)
                changes += 1
        elif d < -_RCF_RC_TOL:
            new_lb = ub[j] - float(np.floor(gap / (-d) + 1e-9))
            if new_lb > lb[j] + 0.5:
                lb[j] = min(ub[j], new_lb)
                changes += 1
    return lb, ub, changes


def _root_reduced_cost_fixing(lp_data, n_orig, lb, ub, int_offsets, int_sizes, t_start, time_limit):
    """Best-effort root reduced-cost fixing for a MILP (increment 4).

    Solves the root LP relaxation with POUNCE (for KKT-valid duals), purifies
    a near-integral point into an incumbent, and applies
    :func:`_reduced_cost_fixing`. Returns ``(lb, ub, incumbent_or_None)``,
    unchanged (and ``None``) whenever POUNCE is unavailable, the root LP is
    not optimal, or no incumbent is recoverable — so it can never weaken the
    search, only tighten it.
    """
    int_idx = [j for off, sz in zip(int_offsets, int_sizes) for j in range(off, off + int(sz))]
    if not int_idx:
        return lb, ub, None
    try:
        from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
        from discopt.solvers.lp_pounce import solve_lp as _pounce_solve_lp
    except ImportError:
        return lb, ub, None
    if not POUNCE_AVAILABLE:
        return lb, ub, None

    n_total = lp_data.A_eq.shape[1] if lp_data.A_eq.shape[0] > 0 else n_orig
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(
        np.asarray(lp_data.A_eq), np.asarray(lp_data.b_eq), n_orig, n_total - n_orig
    )
    c_m = np.asarray(lp_data.c[:n_orig])
    obj_const = float(lp_data.obj_const)
    try:
        res = _pounce_solve_lp(
            c=c_m,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=list(zip(np.asarray(lb).tolist(), np.asarray(ub).tolist())),
            time_limit=min(30.0, max(0.5, time_limit - (time.perf_counter() - t_start))),
        )
    except Exception as e:
        logger.debug("root RCF: POUNCE LP failed: %s", e)
        return lb, ub, None
    if res.status != SolveStatus.OPTIMAL or res.reduced_costs is None or res.x is None:
        return lb, ub, None

    z_lp = float(res.objective) + obj_const
    inc = _pounce_snap_incumbent(
        np.asarray(res.x),
        int_offsets,
        int_sizes,
        lb,
        ub,
        c_m,
        obj_const,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        t_start,
        time_limit,
    )
    if inc is None:
        return lb, ub, None
    z_inc, x_inc = inc
    new_lb, new_ub, n_changes = _reduced_cost_fixing(
        lb, ub, int_idx, np.asarray(res.reduced_costs), z_lp, z_inc
    )
    if n_changes:
        logger.info("root reduced-cost fixing tightened %d integer bound(s)", n_changes)
    return new_lb, new_ub, (z_inc, x_inc)


def _binary_mask(model: Model, n_orig: int) -> np.ndarray:
    """Length-``n_orig`` mask of which flat columns are binary variables."""
    mask = np.zeros(n_orig, dtype=bool)
    off = 0
    for v in model._variables:
        sz = int(v.size)
        if v.var_type == VarType.BINARY:
            mask[off : off + sz] = True
        off += sz
    return mask


def _augment_lpdata_with_cover_cuts(lp_data, n_orig: int, cuts):
    """Add ``sum_{j in C} x_j <= rhs`` rows to the standard-form LP, each with
    its own non-negative slack column (``sum x_C + s = rhs``).

    Augments the relaxation only (extra rows + slack columns); the tree's
    original-variable structure is untouched, so the B&B branches exactly as
    before but on a tighter relaxation. Cover cuts are valid, so the optimum
    is preserved (cannot affect ``incorrect_count``)."""
    A = np.asarray(lp_data.A_eq, dtype=np.float64)
    b = np.asarray(lp_data.b_eq, dtype=np.float64)
    c = np.asarray(lp_data.c, dtype=np.float64)
    xl = np.asarray(lp_data.x_l, dtype=np.float64)
    xu = np.asarray(lp_data.x_u, dtype=np.float64)
    m_rows, n_cols = A.shape
    k = len(cuts)
    newA = np.zeros((m_rows + k, n_cols + k), dtype=np.float64)
    newA[:m_rows, :n_cols] = A
    newb = np.concatenate([b, np.zeros(k, dtype=np.float64)])
    for i, (cover, rhs) in enumerate(cuts):
        for j in cover:
            newA[m_rows + i, j] = 1.0
        newA[m_rows + i, n_cols + i] = 1.0  # slack: sum x_C + s = rhs, s >= 0
        newb[m_rows + i] = rhs
    return lp_data._replace(
        A_eq=newA,
        b_eq=newb,
        c=np.concatenate([c, np.zeros(k, dtype=np.float64)]),
        x_l=np.concatenate([xl, np.zeros(k, dtype=np.float64)]),
        x_u=np.concatenate([xu, np.full(k, 1e20, dtype=np.float64)]),
    )


# Gomory mixed-integer cuts are correct (structurally projected, soundness
# guarded). Their cost depends entirely on the relaxation engine: adding cut
# rows changes the LP shape, which forces a JAX recompile but is free for
# POUNCE (Rust, Path B). GMI is therefore a **POUNCE-mode feature** — enabled
# automatically when node relaxations are solved by POUNCE (no recompile), and
# left off under the JAX IPM where the per-shape recompile would dominate.
#
# ``GOMORY_CUTS_ENABLED`` is a hard override: ``True``/``False`` force GMI on/off
# for every solve; ``None`` (the default) defers to the engine gate below.
GOMORY_CUTS_ENABLED: bool | None = None


def _gomory_enabled(prefer_pounce: bool) -> bool:
    """Whether to run Gomory cuts. Honors the ``GOMORY_CUTS_ENABLED`` override;
    otherwise GMI is on exactly when the node relaxations are solved by POUNCE
    (``prefer_pounce``), where cut-augmented shapes cost no JAX recompile (Path
    B). Under the JAX IPM, adding cuts would recompile per shape, so GMI is off."""
    if GOMORY_CUTS_ENABLED is not None:
        return bool(GOMORY_CUTS_ENABLED)
    return prefer_pounce


def _augment_lpdata_with_gomory_cuts(lp_data, coeffs: np.ndarray, rhs: np.ndarray):
    """Add Gomory cuts ``coeffs[i] · x >= rhs[i]`` to the standard-form LP, each
    with a non-negative surplus column (``coeffs·x - s = rhs``).

    ``coeffs`` are the *structurally projected* GMI coefficients (see
    :func:`_separate_gomory_cuts`), so they are O(1) and reference only the
    structural columns — keeping the augmented relaxation well-conditioned. A
    small rhs margin scaled to the coefficient magnitude absorbs the residual
    numerical error in the refined basis, so a cut can never exclude a true
    integer point (preserving ``incorrect_count == 0``) while still separating
    the fractional vertex."""
    A = np.asarray(lp_data.A_eq, dtype=np.float64)
    b = np.asarray(lp_data.b_eq, dtype=np.float64)
    c = np.asarray(lp_data.c, dtype=np.float64)
    xl = np.asarray(lp_data.x_l, dtype=np.float64)
    xu = np.asarray(lp_data.x_u, dtype=np.float64)
    m_rows, n_cols = A.shape
    k = coeffs.shape[0]
    newA = np.zeros((m_rows + k, n_cols + k), dtype=np.float64)
    newA[:m_rows, :n_cols] = A
    newb = np.concatenate([b, np.zeros(k, dtype=np.float64)])
    for i in range(k):
        row = np.asarray(coeffs[i], dtype=np.float64)
        margin = 1e-7 * (1.0 + float(np.abs(row).sum()))  # safe-GMI rhs relaxation
        newA[m_rows + i, :n_cols] = row
        newA[m_rows + i, n_cols + i] = -1.0  # surplus: coeffs·x - s = rhs, s >= 0
        newb[m_rows + i] = float(rhs[i]) - margin
    return lp_data._replace(
        A_eq=newA,
        b_eq=newb,
        c=np.concatenate([c, np.zeros(k, dtype=np.float64)]),
        x_l=np.concatenate([xl, np.zeros(k, dtype=np.float64)]),
        x_u=np.concatenate([xu, np.full(k, 1e20, dtype=np.float64)]),
    )


def _project_cut_to_structural(coeffs, rhs, A, b, n_orig):
    """Project a standard-form cut ``coeffs·x >= rhs`` onto the structural
    variables by substituting each slack ``x_s = (b_r - sum_k A[r,k] x_k)/c_s``
    via its (singleton) defining row ``r``.

    This is an exact substitution through ``A_eq x = b_eq`` — true for every
    feasible point — so validity and the separation of the current vertex are
    preserved, but the result references only structural columns with O(1)
    coefficients (no coupling to the wide-range row slacks that diverge the IPM
    on cut-augmented relaxations). Returns a length-``n`` coefficient vector
    (slack entries zero) and rhs, or ``None`` if a slack column is not a clean
    singleton."""
    n_cols = A.shape[1]
    out = np.zeros(n_cols, dtype=np.float64)
    out[:n_orig] = np.asarray(coeffs[:n_orig], dtype=np.float64)
    r_rhs = float(rhs)
    for s in range(n_orig, n_cols):
        gs = float(coeffs[s])
        if gs == 0.0:
            continue
        rows = np.nonzero(A[:, s])[0]
        if rows.size != 1:
            return None  # not a singleton slack — cannot project cleanly
        r = int(rows[0])
        cs = A[r, s]
        if abs(cs) < 1e-12:
            return None
        factor = gs / cs
        out[:n_orig] -= factor * A[r, :n_orig]
        r_rhs -= factor * b[r]
    if not np.all(np.isfinite(out)) or not np.isfinite(r_rhs):
        return None
    if np.max(np.abs(out)) < 1e-12:
        return None  # projected to a trivial cut
    return out, r_rhs


def _separate_gomory_cuts(lp_data, x_vertex, n_orig, int_idx, max_cuts: int = 8):
    """Separate GMI cuts at the crossover vertex via the Rust ``_rust`` binding,
    then project each onto the structural variables.

    Returns ``(coeffs, rhs)`` arrays (up to ``max_cuts`` projected rows) or
    ``None`` when the binding is unavailable, ``x_vertex`` is not a basic
    feasible solution, or no cut survives. Only the genuine integer-constrained
    structural columns (``int_idx``) are marked integral — slacks stay
    continuous — so the cuts are sound for the original problem."""
    try:
        from discopt._rust import gomory_cuts_py
    except ImportError:
        return None
    A = np.asarray(lp_data.A_eq, dtype=np.float64)
    b = np.asarray(lp_data.b_eq, dtype=np.float64)
    n_cur = A.shape[1]
    integrality = np.zeros(n_cur, dtype=bool)
    integrality[[j for j in int_idx if j < n_cur]] = True
    res = gomory_cuts_py(
        np.ascontiguousarray(x_vertex, dtype=np.float64),
        np.ascontiguousarray(A),
        np.ascontiguousarray(b),
        np.ascontiguousarray(lp_data.c, dtype=np.float64),
        np.ascontiguousarray(lp_data.x_l, dtype=np.float64),
        np.ascontiguousarray(lp_data.x_u, dtype=np.float64),
        integrality,
    )
    if res is None:
        return None
    coeffs, rhs = np.asarray(res[0], dtype=np.float64), np.asarray(res[1], dtype=np.float64)
    proj_coeffs, proj_rhs = [], []
    for ci in range(min(coeffs.shape[0], max_cuts)):
        projected = _project_cut_to_structural(coeffs[ci], float(rhs[ci]), A, b, n_orig)
        if projected is not None:
            proj_coeffs.append(projected[0])
            proj_rhs.append(projected[1])
    if not proj_coeffs:
        return None
    return np.array(proj_coeffs, dtype=np.float64), np.array(proj_rhs, dtype=np.float64)


def _augment_lpdata_with_mir_cuts(lp_data, coeffs: np.ndarray, rhs: np.ndarray):
    """Add MIR cuts ``coeffs[i] · x <= rhs[i]`` to the standard-form LP, each
    with a non-negative slack column (``coeffs·x + s = rhs``).

    MIR cuts are over the structural columns with O(1) coefficients (no slack
    coupling), so the augmented relaxation stays well-conditioned. A small rhs
    relaxation guards against floating-point error in the separation point so a
    cut cannot exclude a true integer point."""
    A = np.asarray(lp_data.A_eq, dtype=np.float64)
    b = np.asarray(lp_data.b_eq, dtype=np.float64)
    c = np.asarray(lp_data.c, dtype=np.float64)
    xl = np.asarray(lp_data.x_l, dtype=np.float64)
    xu = np.asarray(lp_data.x_u, dtype=np.float64)
    m_rows, n_cols = A.shape
    k = coeffs.shape[0]
    newA = np.zeros((m_rows + k, n_cols + k), dtype=np.float64)
    newA[:m_rows, :n_cols] = A
    newb = np.concatenate([b, np.zeros(k, dtype=np.float64)])
    for i in range(k):
        row = np.asarray(coeffs[i], dtype=np.float64)
        margin = 1e-7 * (1.0 + float(np.abs(row).sum()))  # safe-cut rhs relaxation
        newA[m_rows + i, :n_cols] = row
        newA[m_rows + i, n_cols + i] = 1.0  # slack: coeffs·x + s = rhs, s >= 0
        newb[m_rows + i] = float(rhs[i]) + margin
    return lp_data._replace(
        A_eq=newA,
        b_eq=newb,
        c=np.concatenate([c, np.zeros(k, dtype=np.float64)]),
        x_l=np.concatenate([xl, np.zeros(k, dtype=np.float64)]),
        x_u=np.concatenate([xu, np.full(k, 1e20, dtype=np.float64)]),
    )


def _separate_mir_cuts(lp_data, x_vertex, n_orig, int_idx, a_ub_orig, b_ub_orig, max_cuts: int = 8):
    """Separate MIR cuts from the original ``<=`` rows at the crossover vertex
    via the Rust ``_rust`` binding, embedded into the current columns.

    Returns ``(coeffs, rhs)`` over the current standard-form columns (structural
    entries set, slacks zero) or ``None`` when the binding is unavailable, the
    structural lower bounds are not finite (the MIR shift needs them), or no cut
    is produced. Only the integer-constrained structural columns are marked
    integral, so the cuts are sound."""
    if a_ub_orig is None or np.asarray(a_ub_orig).shape[0] == 0:
        return None
    try:
        from discopt._rust import mir_cuts_py
    except ImportError:
        return None
    lo = np.asarray(lp_data.x_l, dtype=np.float64)[:n_orig]
    if not np.all(np.isfinite(lo)):
        return None  # MIR's lower-bound shift requires finite lower bounds
    integ = np.zeros(n_orig, dtype=bool)
    integ[[j for j in int_idx if j < n_orig]] = True
    res = mir_cuts_py(
        np.ascontiguousarray(a_ub_orig, dtype=np.float64),
        np.ascontiguousarray(b_ub_orig, dtype=np.float64),
        np.ascontiguousarray(lo),
        integ,
        np.ascontiguousarray(np.asarray(x_vertex, dtype=np.float64)[:n_orig]),
    )
    if res is None:
        return None
    coeffs, rhs = np.asarray(res[0], dtype=np.float64), np.asarray(res[1], dtype=np.float64)
    n_cur = int(np.asarray(lp_data.A_eq).shape[1])
    embedded = np.zeros((coeffs.shape[0], n_cur), dtype=np.float64)
    embedded[:, :n_orig] = coeffs[:, :n_orig]
    return embedded[:max_cuts], rhs[:max_cuts]


def _extract_clique_edges(model: Model) -> list[tuple[int, int]]:
    """Conflict-graph 2-clique edges from the Rust presolve clique pass.

    Each edge ``(i, j)`` (flat variable indices) is a pair of binaries that
    cannot both be 1. Best-effort: returns ``[]`` if the bridge/pass is
    unavailable."""
    try:
        from discopt._jax.presolve_pipeline import run_root_presolve
        from discopt._rust import model_to_repr

        repr_ = model_to_repr(model, getattr(model, "_builder", None))
        _, stats = run_root_presolve(
            repr_,
            cliques=True,
            eliminate=False,
            aggregate=False,
            redundancy=False,
            implied_bounds=False,
            coefficient_strengthening=False,
            factorable_elim=False,
            fbbt=False,
            simplify=False,
            probing=False,
        )
        return list(stats.get("cliques", {}).get("edges", []) or [])
    except Exception as e:
        logger.debug("clique edge extraction skipped: %s", e)
        return []


def _cut_loop_relaxation_x(lp_data, prefer_pounce: bool):
    """Solve the (cut-augmented) root relaxation for the cut loop, returning the
    optimum ``x`` or ``None``.

    In POUNCE mode the solve goes through POUNCE so the cut-augmented shape costs
    no JAX recompile (consistent with the Path-B node engine); otherwise the JAX
    IPM is used. Either way the returned point is just a separation seed —
    crossover and cut validity do not depend on which engine produced it."""
    if prefer_pounce:
        try:
            from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
            from discopt.solvers.lp_pounce import solve_lp as _pounce_solve
        except ImportError:
            POUNCE_AVAILABLE = False
        if POUNCE_AVAILABLE:
            try:
                res = _pounce_solve(
                    c=np.asarray(lp_data.c, dtype=np.float64),
                    A_eq=np.asarray(lp_data.A_eq, dtype=np.float64),
                    b_eq=np.asarray(lp_data.b_eq, dtype=np.float64),
                    bounds=list(
                        zip(
                            np.asarray(lp_data.x_l, dtype=np.float64).tolist(),
                            np.asarray(lp_data.x_u, dtype=np.float64).tolist(),
                        )
                    ),
                )
            except Exception as exc:
                logger.debug("cut-loop POUNCE solve failed: %s", exc)
                return None
            if res.status == SolveStatus.OPTIMAL and res.x is not None:
                return np.asarray(res.x, dtype=np.float64)
            return None
    import jax.numpy as jnp

    from discopt._jax.lp_ipm import lp_ipm_solve

    state = lp_ipm_solve(
        jnp.asarray(lp_data.c),
        jnp.asarray(lp_data.A_eq),
        jnp.asarray(lp_data.b_eq),
        jnp.asarray(lp_data.x_l),
        jnp.asarray(lp_data.x_u),
    )
    if int(state.converged) != 1:
        return None
    return np.asarray(state.x, dtype=np.float64)


def _root_cover_cut_loop(
    lp_data,
    n_orig: int,
    is_binary: np.ndarray,
    A_ub_orig,
    b_ub_orig,
    t_start: float,
    time_limit: float,
    clique_edges=(),
    int_idx=(),
    prefer_pounce: bool = False,
    max_rounds: int = 5,
    max_total_cuts: int = 500,
):
    """Round-based root cut separation: cover + clique + Gomory cuts (Phase 2/3).

    Solves the root LP relaxation, crosses the interior optimum over to a
    vertex, separates violated cover cuts (from the *original* knapsack rows),
    clique cuts (from the presolve conflict-graph edges), and structurally
    projected Gomory mixed-integer cuts (from the iteratively-refined basis at
    the vertex), augments ``lp_data``, and repeats. Returns the (possibly
    augmented) ``lp_data`` and the number of cuts added. A no-op when there are
    no binary-knapsack rows, clique edges, or integer variables."""
    from discopt._jax.cover_cuts import (
        has_binary_knapsack_rows,
        separate_clique_cuts,
        separate_cover_cuts,
    )

    has_cover = A_ub_orig is not None and has_binary_knapsack_rows(A_ub_orig, b_ub_orig, is_binary)
    has_clique = bool(clique_edges)
    has_gomory = bool(len(int_idx))
    if not has_cover and not has_clique and not has_gomory:
        return lp_data, 0

    total = 0
    for _round in range(max_rounds):
        if time.perf_counter() - t_start >= time_limit:
            break
        # Solve the (possibly cut-augmented) root relaxation. In POUNCE mode use
        # POUNCE so adding cut rows costs no JAX recompile (matches the Path-B
        # node engine); otherwise the JAX IPM.
        x_relax = _cut_loop_relaxation_x(lp_data, prefer_pounce)
        if x_relax is None:  # need a usable optimum to separate
            break
        # Cross over the interior optimum to a vertex of the optimal face
        # (Phase 2): cover/clique cuts separate a vertex sharply but the
        # interior analytic center weakly. Separation stays valid regardless,
        # so a failed crossover only costs cut effectiveness, never soundness.
        from discopt._jax.crossover import crossover_to_vertex

        try:
            x_vertex = crossover_to_vertex(
                x_relax,
                np.asarray(lp_data.A_eq),
                np.asarray(lp_data.b_eq),
                np.asarray(lp_data.c),
                np.asarray(lp_data.x_l),
                np.asarray(lp_data.x_u),
            )
        except Exception as _xo_exc:
            logger.debug("crossover skipped: %s", _xo_exc)
            x_vertex = x_relax
        x_star = x_vertex[:n_orig]
        cuts: list[tuple[frozenset, float]] = []
        seen: set[frozenset] = set()
        sources = []
        if has_cover:
            sources.append(separate_cover_cuts(A_ub_orig, b_ub_orig, x_star, is_binary))
        if has_clique:
            sources.append(separate_clique_cuts(clique_edges, x_star))
        for found in sources:
            for cover, rhs in found:
                if cover not in seen:
                    seen.add(cover)
                    cuts.append((cover, rhs))

        round_added = 0
        # Gomory cuts on the first round only: derived from the original
        # (well-conditioned) standard-form basis and projected onto the
        # structural variables. Re-separating GMI on the cut-augmented system
        # compounds ill-conditioning, so later rounds keep only the
        # combinatorial (cover/clique) cuts. Added before cover/clique grow the
        # matrix so the projected coefficients stay sized to the structural
        # columns; soundness-guarded by the rhs margin in the augmentation.
        # Cheap pre-check: only attempt the (basis-recovery + GMI) work when an
        # integer variable is actually fractional at the vertex; otherwise GMI
        # would find nothing and the basis solve is wasted.
        _int_fractional = (
            has_gomory
            and _round == 0
            and any(
                abs(x_vertex[j] - round(float(x_vertex[j]))) > 1e-6
                for j in int_idx
                if j < len(x_vertex)
            )
        )
        if _int_fractional:
            try:
                gom = _separate_gomory_cuts(lp_data, x_vertex, n_orig, int_idx)
            except Exception as _gom_exc:
                logger.debug("gomory separation skipped: %s", _gom_exc)
                gom = None
            if gom is not None:
                gc, gr = gom
                lp_data = _augment_lpdata_with_gomory_cuts(lp_data, gc, gr)
                round_added += int(gc.shape[0])
        # MIR cuts from the original <= rows (basis-free; complements GMI). Same
        # POUNCE-mode gate (has_gomory) and round-0-only policy.
        if has_gomory and _round == 0:
            try:
                mir = _separate_mir_cuts(lp_data, x_vertex, n_orig, int_idx, A_ub_orig, b_ub_orig)
            except Exception as _mir_exc:
                logger.debug("mir separation skipped: %s", _mir_exc)
                mir = None
            if mir is not None:
                mc, mr = mir
                lp_data = _augment_lpdata_with_mir_cuts(lp_data, mc, mr)
                round_added += int(mc.shape[0])
        if cuts:  # cover/clique reference original columns (< n_orig), still valid
            lp_data = _augment_lpdata_with_cover_cuts(lp_data, n_orig, cuts)
            round_added += len(cuts)

        if round_added == 0:
            break
        total += round_added
        if total >= max_total_cuts:
            break
        # GMI is round-0 only; with no cover/clique to re-separate, further
        # rounds would just re-solve the LP for nothing.
        if not has_cover and not has_clique:
            break
    return lp_data, total


def _root_dive(lp_data, n_orig, int_idx, t_start, time_limit, max_steps=None, prefer_pounce=False):
    """Fractional diving from the root LP to find an early incumbent (Phase 3).

    Repeatedly solve the LP relaxation, fix the most-fractional unfixed integer
    variable to its nearest integer, and re-solve, until every integer is
    integral (an incumbent) or a fix makes the LP non-optimal (dive abandoned).
    Returns ``(objective, x_orig)`` in minimization sense, or ``None``. An early
    incumbent front-loads pruning and reduced-cost fixing.

    In POUNCE-only mode (``prefer_pounce``) the dive LPs are solved with POUNCE
    on the native inequality form instead of the pure-JAX IPM (Phase 8: no JAX
    IPM on the POUNCE LP/MILP path; the JAX IPM stalls on the slack-expanded
    standard form anyway).
    """
    if not int_idx:
        return None

    steps = max_steps if max_steps is not None else len(int_idx) + 1

    if prefer_pounce:
        # POUNCE-only mode (Phase 8: no pure-JAX IPM on this path). The dive is
        # an optional early-incumbent heuristic; a faithful POUNCE port would
        # need one LP solve per fixed integer (dozens of sequential solves),
        # which can starve the main B&B budget on larger relaxations. Skipping
        # it is sound — node solves and small-integer recovery still find
        # incumbents — and removes the last JAX-IPM call from the POUNCE path.
        return None

    import jax.numpy as jnp

    from discopt._jax.lp_ipm import lp_ipm_solve

    xl = np.asarray(lp_data.x_l, dtype=np.float64).copy()
    xu = np.asarray(lp_data.x_u, dtype=np.float64).copy()
    c = jnp.asarray(lp_data.c)
    A = jnp.asarray(lp_data.A_eq)
    b = jnp.asarray(lp_data.b_eq)
    steps = max_steps if max_steps is not None else len(int_idx) + 1
    for _ in range(steps):
        if time.perf_counter() - t_start >= time_limit:
            return None
        state = lp_ipm_solve(c, A, b, jnp.asarray(xl), jnp.asarray(xu))
        if int(state.converged) != 1:
            return None  # infeasible/stalled fix -> abandon the dive
        x = np.asarray(state.x)
        fracs = [
            (j, abs(x[j] - round(x[j])))
            for j in int_idx
            if xl[j] != xu[j] and abs(x[j] - round(x[j])) > 1e-6
        ]
        if not fracs:
            return float(state.obj) + float(lp_data.obj_const), x[:n_orig]
        j = max(fracs, key=lambda t: t[1])[0]
        v = float(round(x[j]))
        xl[j] = v
        xu[j] = v
    return None


def _solve_milp_simplex(
    model: Model,
    time_limit: float,
    gap_tolerance: float,
    max_nodes: int,
    t_start: float,
) -> Optional[SolveResult]:
    """Solve a pure MILP with the Rust-internal warm-started-simplex B&B
    (``nlp_solver="simplex"``).

    The whole search runs in Rust: the existing tree manager with each node's LP
    solved by the bounded simplex (root cold, children dual-warm-started from the
    inherited basis). Returns ``None`` to defer to the default path when the
    binding is unavailable or the model has no constraints. Pure-MILP only;
    MINLP/MIQP keep the POUNCE/IPM path."""
    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.modeling.core import ObjectiveSense

    try:
        from discopt._rust import solve_milp_py
    except ImportError:
        return None

    lp_data = extract_lp_data(model)
    A = np.ascontiguousarray(lp_data.A_eq, dtype=np.float64)
    if A.shape[0] == 0:
        return None  # no constraints — let the default path handle it
    n_orig = sum(v.size for v in model._variables)
    _, _, _, int_offsets, int_sizes = _extract_variable_info(model)
    int_idx = [j for off, sz in zip(int_offsets, int_sizes) for j in range(off, off + int(sz))]

    status, x_struct, obj, bound, nodes, _lp_iters = solve_milp_py(
        np.ascontiguousarray(lp_data.c, dtype=np.float64),
        A,
        np.ascontiguousarray(lp_data.b_eq, dtype=np.float64),
        np.ascontiguousarray(lp_data.x_l, dtype=np.float64),
        np.ascontiguousarray(lp_data.x_u, dtype=np.float64),
        np.ascontiguousarray(np.asarray(int_idx, dtype=np.int64)),
        n_orig,
        float(lp_data.obj_const),
        int(max_nodes),
        float(gap_tolerance),
    )
    wall_time = time.perf_counter() - t_start
    maximize = model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE

    if status in ("optimal", "feasible"):
        x_dict = _unpack_solution(model, np.asarray(x_struct, dtype=np.float64))
        obj_val = -obj if maximize else obj
        bound_val = None
        gap_val = None
        if np.isfinite(bound):
            bound_val = -bound if maximize else bound
            gap_val = abs(obj_val - bound_val) / (abs(obj_val) + 1e-10)
        return SolveResult(
            status=status,
            objective=obj_val,
            bound=bound_val,
            gap=gap_val,
            x=x_dict,
            wall_time=wall_time,
            node_count=nodes,
        )
    if status == "unbounded":
        return SolveResult(status="unbounded", wall_time=wall_time, node_count=nodes)
    if status == "node_limit":
        return SolveResult(status="node_limit", wall_time=wall_time, node_count=nodes)
    return SolveResult(status="infeasible", wall_time=wall_time, node_count=nodes)


def _solve_milp_bb(
    model: Model,
    time_limit: float,
    gap_tolerance: float,
    batch_size: int,
    strategy: str,
    max_nodes: int,
    t_start: float,
    prefer_pounce: bool = False,
) -> SolveResult:
    """Solve a MILP via B&B with LP relaxation solves at each node.

    ``prefer_pounce`` (POUNCE-only mode) routes the incumbent dual recovery
    through POUNCE instead of HiGHS so the whole solve is HiGHS-free.
    """
    import jax.numpy as jnp

    from discopt._jax.lp_ipm import lp_ipm_solve
    from discopt._jax.problem_classifier import extract_lp_data

    rust_time = 0.0
    jax_time = 0.0

    t_jax_start = time.perf_counter()
    lp_data = extract_lp_data(model)
    jax_time += time.perf_counter() - t_jax_start

    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)
    n_orig = sum(v.size for v in model._variables)

    # Root reduced-cost fixing (increment 4): best-effort integer-bound
    # tightening from the root LP duals + a purified incumbent. Sound and
    # graceful -- only ever tightens, skipped entirely if POUNCE is absent or
    # no incumbent is recoverable.
    _root_incumbent = None
    try:
        lb, ub, _root_incumbent = _root_reduced_cost_fixing(
            lp_data, n_orig, lb, ub, int_offsets, int_sizes, t_start, time_limit
        )
    except Exception as _rcf_exc:
        logger.debug("root RCF skipped: %s", _rcf_exc)

    # --- Root knapsack cover cuts (Phase 3) ---
    # Separate valid cover inequalities from the *original* knapsack rows and
    # augment the relaxation, tightening every node's LP bound. Cover cuts are
    # valid, so the optimum is preserved; the tree variable structure is
    # untouched (extra rows/slacks only).
    # Decompose the ORIGINAL problem once: reused for cover separation and for
    # node/dual recovery, which must reference the user's constraints (not the
    # auxiliary cover rows). The B&B node LP relaxations below use the
    # cover-augmented ``lp_data``; recovery uses ``lp_data_orig`` / these.
    lp_data_orig = lp_data
    _n_total0 = lp_data.A_eq.shape[1] if lp_data.A_eq.shape[0] > 0 else n_orig
    _A_ub_m, _b_ub_m, _A_eq_m, _b_eq_m = _decompose_eq_slack_form(
        np.asarray(lp_data.A_eq), np.asarray(lp_data.b_eq), n_orig, _n_total0 - n_orig
    )
    try:
        _is_bin = _binary_mask(model, n_orig)
        # Conflict-graph clique edges (only worth extracting if binaries exist).
        _clique_edges = _extract_clique_edges(model) if bool(_is_bin.any()) else []
        # Gomory cuts gated on the relaxation engine (see _gomory_enabled):
        # passing no integer indices disables the GMI branch, so under the JAX
        # IPM the loop runs exactly as before GMI (cover/clique only, no
        # recompile). In POUNCE mode the node solves (Path B) take cut-augmented
        # shapes for free, so GMI is enabled.
        _cut_int_idx = (
            [j for off, sz in zip(int_offsets, int_sizes) for j in range(off, off + int(sz))]
            if _gomory_enabled(prefer_pounce)
            else []
        )
        lp_data, _n_cuts = _root_cover_cut_loop(
            lp_data,
            n_orig,
            _is_bin,
            _A_ub_m,
            _b_ub_m,
            t_start,
            time_limit,
            clique_edges=_clique_edges,
            int_idx=_cut_int_idx,
            prefer_pounce=prefer_pounce,
        )
        if _n_cuts:
            logger.info("root cuts added %d valid inequalities (cover + clique + gomory)", _n_cuts)
    except Exception as _cc_exc:
        logger.debug("root cuts skipped: %s", _cc_exc)

    # Seed a rigorous root lower bound from the (cut-augmented) LP relaxation,
    # solved once via the fast convex engine (~0.03 s). This is always a valid
    # dual bound and is surfaced even on an uncertified / time-limited exit, so
    # a solve that cannot finish the tree — e.g. AMP's short per-iteration MILP
    # budget, which otherwise returns bound=-inf and leaves AMP's LB stuck at
    # -inf — still yields a finite lower bound. Tightens across AMP iterations
    # as the relaxation gains partitions/cuts. POUNCE-only (the convex engine);
    # left at -inf otherwise so behavior is unchanged on the JAX-IPM path.
    _root_lp_bound = -np.inf
    if prefer_pounce:
        try:
            _root_out = _solve_node_lp_pounce(lp_data, lb, ub, n_vars, n_orig, t_start, time_limit)
            if _root_out is not None and _root_out[2] == "optimal" and np.isfinite(_root_out[0]):
                _root_lp_bound = float(_root_out[0])
        except Exception as _rlb_exc:
            logger.debug("root LP bound seeding skipped: %s", _rlb_exc)

    t_rust_start = time.perf_counter()
    tree = PyTreeManager(n_vars, lb.tolist(), ub.tolist(), int_offsets, int_sizes, strategy)
    tree.initialize()
    if _root_incumbent is not None:
        _z_inc, _x_inc = _root_incumbent
        tree.inject_incumbent(np.asarray(_x_inc[:n_vars], dtype=np.float64).copy(), float(_z_inc))
    # Root fractional diving (Phase 3): an early incumbent front-loads pruning
    # and reduced-cost fixing. The tree keeps the best of any injected points.
    try:
        _int_idx = [j for off, sz in zip(int_offsets, int_sizes) for j in range(off, off + int(sz))]
        _dive = _root_dive(
            lp_data, n_orig, _int_idx, t_start, time_limit, prefer_pounce=prefer_pounce
        )
        if _dive is not None:
            _dz, _dx = _dive
            tree.inject_incumbent(np.asarray(_dx[:n_vars], dtype=np.float64).copy(), float(_dz))
    except Exception as _dive_exc:
        logger.debug("root dive skipped: %s", _dive_exc)
    rust_time += time.perf_counter() - t_rust_start

    # A node relaxation that stalled at the iteration limit (converged==3) is
    # not at KKT, so its objective is not a valid lower bound for the node
    # (f(x~) >= f*); trusting it can prune the true integer optimum. Such
    # nodes are first re-solved with POUNCE (KKT-valid bound or certified
    # infeasibility); only unrecoverable ones decertify the gap (mirrors the
    # P0.3 trust-gate + polish-retry; bounds are left untouched).
    _gap_certified = True
    # _A_ub_m/_b_ub_m/_A_eq_m/_b_eq_m (original problem) were decomposed above,
    # before cover augmentation; node recovery uses them so its bound is for
    # the user's constraints.
    _c_m = np.asarray(lp_data_orig.c[:n_orig])

    def _recover_or_decertify(i, lbs, sols, node_lb_i, node_ub_i):
        nonlocal _gap_certified
        rec = _pounce_recover_node_bound(
            node_lb_i,
            node_ub_i,
            _c_m,
            float(lp_data.obj_const),
            _A_ub_m,
            _b_ub_m,
            _A_eq_m,
            _b_eq_m,
            t_start,
            time_limit,
        )
        if rec is None:
            _gap_certified = False
        elif rec[0] == "optimal":
            lbs[i] = rec[1]
            sols[i] = rec[2][:n_vars]
        else:  # Phase-1-certified infeasible node: prune is rigorous.
            lbs[i] = _INFEASIBILITY_SENTINEL

    def _maybe_inject_snapped(x_row, node_lb_i, node_ub_i):
        # Purification (increment 3): near-integral interior points become
        # exact incumbents via snap-fix-resolve.
        inc = _pounce_snap_incumbent(
            x_row,
            int_offsets,
            int_sizes,
            node_lb_i,
            node_ub_i,
            _c_m,
            float(lp_data.obj_const),
            _A_ub_m,
            _b_ub_m,
            _A_eq_m,
            _b_eq_m,
            t_start,
            time_limit,
        )
        if inc is not None:
            tree.inject_incumbent(
                np.asarray(inc[1][:n_vars], dtype=np.float64).copy(), float(inc[0])
            )

    # Path B: in POUNCE-only mode, POUNCE solves node relaxations directly
    # (no JAX recompile on cut-augmented shapes). Checked once here.
    _pounce_nodes_avail = False
    if prefer_pounce:
        try:
            from discopt.solvers.lp_pounce import POUNCE_AVAILABLE as _PNA

            _pounce_nodes_avail = bool(_PNA)
        except ImportError:
            _pounce_nodes_avail = False

    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, _batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        t_jax_start = time.perf_counter()
        result_ids = np.array(batch_ids, dtype=np.int64)
        n_slack = lp_data.x_l.shape[0] - n_orig

        if prefer_pounce and _pounce_nodes_avail:
            # Path B: solve each node's relaxation with POUNCE (Rust) instead of
            # the JAX IPM. POUNCE takes the cut-augmented LP at any shape with no
            # per-shape recompile, and its OPTIMAL bound is KKT-valid. The rare
            # stall/unavailable node defers to the existing recovery/decertify.
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.zeros(n_batch, dtype=bool)
            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])
                _mid = 0.5 * (np.clip(node_lb, -_SPC, _SPC) + np.clip(node_ub, -_SPC, _SPC))
                out = _solve_node_lp_pounce(
                    lp_data, node_lb, node_ub, n_vars, n_orig, t_start, time_limit
                )
                if out is not None and out[2] == "optimal":
                    result_lbs[i] = out[0]
                    result_sols[i] = out[1]
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        _maybe_inject_snapped(result_sols[i], node_lb, node_ub)
                elif out is not None:  # POUNCE-certified infeasible: rigorous prune
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    result_sols[i] = _mid
                else:  # unavailable/stalled: original-problem recovery or decertify
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    result_sols[i] = _mid
                    _recover_or_decertify(i, result_lbs, result_sols, node_lb, node_ub)
        elif n_batch > 1:
            # Batch LP solve via vmap
            from discopt._jax.lp_ipm import lp_ipm_solve_batch

            xl_arr = jnp.array(batch_lb, dtype=jnp.float64)
            xu_arr = jnp.array(batch_ub, dtype=jnp.float64)
            slack_l = jnp.zeros((n_batch, n_slack), dtype=jnp.float64)
            slack_u = jnp.full((n_batch, n_slack), 1e20, dtype=jnp.float64)
            xl_full = jnp.concatenate([xl_arr, slack_l], axis=1)
            xu_full = jnp.concatenate([xu_arr, slack_u], axis=1)

            try:
                state = lp_ipm_solve_batch(lp_data.c, lp_data.A_eq, lp_data.b_eq, xl_full, xu_full)
                converged = np.asarray(state.converged)
                obj_vals = np.asarray(state.obj)
                x_vals = np.asarray(state.x)

                ok = (converged == 1) | (converged == 2) | (converged == 3)
                result_lbs = np.asarray(
                    np.where(ok, obj_vals + float(lp_data.obj_const), _INFEASIBILITY_SENTINEL),
                    dtype=np.float64,
                )
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    if ok[i]:
                        result_sols[i] = x_vals[i, :n_vars]
                        # Reject LP solutions that violate constraints
                        if not _check_lp_solution_feasibility(
                            lp_data.A_eq, lp_data.b_eq, x_vals[i]
                        ):
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                            ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                            result_sols[i] = 0.5 * (lb_c + ub_c)
                        else:
                            _maybe_inject_snapped(
                                result_sols[i],
                                np.array(batch_lb[i]),
                                np.array(batch_ub[i]),
                            )
                    else:
                        lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                        ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
                # Non-KKT (max-iter) LP bounds are recovered via POUNCE;
                # unrecoverable ones decertify the gap.
                for i in np.where((converged == 3) & (result_lbs < _SENTINEL_THRESHOLD))[0]:
                    _recover_or_decertify(
                        int(i),
                        result_lbs,
                        result_sols,
                        np.array(batch_lb[i]),
                        np.array(batch_ub[i]),
                    )
            except Exception as e:
                logger.debug("Batch LP solve failed: %s", e)
                result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                    ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)
            result_feas = np.zeros(n_batch, dtype=bool)
        else:
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.zeros(n_batch, dtype=bool)

            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                x_l_node = jnp.array(node_lb, dtype=jnp.float64)
                x_u_node = jnp.array(node_ub, dtype=jnp.float64)

                x_l_full = jnp.concatenate([x_l_node, jnp.zeros(n_slack)])
                x_u_full = jnp.concatenate([x_u_node, jnp.full(n_slack, 1e20)])

                try:
                    state = lp_ipm_solve(lp_data.c, lp_data.A_eq, lp_data.b_eq, x_l_full, x_u_full)
                    conv = int(state.converged)
                    if conv in (1, 2, 3):
                        # Reject LP solutions that violate constraints
                        if _check_lp_solution_feasibility(lp_data.A_eq, lp_data.b_eq, state.x):
                            result_lbs[i] = float(state.obj) + lp_data.obj_const
                            result_sols[i] = np.asarray(state.x[:n_vars])
                            if conv == 3:  # non-KKT: recover via POUNCE
                                _recover_or_decertify(i, result_lbs, result_sols, node_lb, node_ub)
                            if result_lbs[i] < _SENTINEL_THRESHOLD:
                                _maybe_inject_snapped(result_sols[i], node_lb, node_ub)
                        else:
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            lb_c = np.clip(node_lb, -_SPC, _SPC)
                            ub_c = np.clip(node_ub, -_SPC, _SPC)
                            result_sols[i] = 0.5 * (lb_c + ub_c)
                    else:
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        lb_c = np.clip(node_lb, -_SPC, _SPC)
                        ub_c = np.clip(node_ub, -_SPC, _SPC)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
                except Exception as e:
                    logger.debug("Per-node LP/QP solve failed: %s", e)
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    lb_c = np.clip(node_lb, -_SPC, _SPC)
                    ub_c = np.clip(node_ub, -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)

        jax_time += time.perf_counter() - t_jax_start

        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        iteration += 1
        if tree.is_finished():
            break
        if tree.gap() <= gap_tolerance:
            break
        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time
    stats = tree.stats()
    incumbent = tree.incumbent()

    if incumbent is not None:
        sol_array, obj_val = incumbent
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    constraint_duals = None
    bound_duals_lower = None
    bound_duals_upper = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        x_dict = _unpack_solution(model, sol_flat)

        # Recover relaxation duals at the integer-feasible incumbent by
        # re-solving the LP relaxation with integer variables fixed. Use the
        # ORIGINAL problem (lp_data_orig / its decomposition); the duals are
        # for the user's named constraints, not the auxiliary cover rows.
        try:
            constraint_duals, bound_duals_lower, bound_duals_upper = _mip_recover_relaxation_duals(
                model,
                lp_data=lp_data_orig,
                x_flat=np.asarray(sol_flat[:n_orig], dtype=float),
                n_orig=n_orig,
                A_ub=_A_ub_m,
                b_ub=_b_ub_m,
                A_eq=_A_eq_m,
                b_eq=_b_eq_m,
                time_limit=max(0.1, time_limit - (time.perf_counter() - t_start)),
                prefer_pounce=prefer_pounce,
            )
        except Exception as _exc:
            logger.debug("MILP-BB dual recovery failed: %s", _exc)

        # Negate objective back for maximization (B&B tree tracks minimization)
        from discopt.modeling.core import ObjectiveSense

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        # "optimal" needs a closed search AND a certified gap: a stalled
        # (non-KKT) node bound leaves optimality unproven even when the tree
        # appears finished.
        if (tree.gap() <= gap_tolerance or tree.is_finished()) and _gap_certified:
            status = "optimal"
        else:
            status = "feasible"
    else:
        x_dict = None
        obj_val = None
        if stats["total_nodes"] >= max_nodes:
            status = "node_limit"
            # No incumbent and a resource limit: nothing is certified. Without an
            # incumbent there is no gap to certify, and a leftover tree bound
            # describes an unexplored search, not a proven optimum — a certified
            # exit here would claim optimality with no solution at all.
            _gap_certified = False
        elif wall_time >= time_limit:
            status = "time_limit"
            _gap_certified = False
        else:
            # Tree exhausted with no feasible node: infeasibility *is* a certified
            # conclusion, so leave _gap_certified untouched.
            status = "infeasible"

    from discopt.modeling.core import ObjectiveSense

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    assert model._objective is not None
    _maximize = model._objective.sense == ObjectiveSense.MAXIMIZE
    if bound_val is not None and _maximize:
        bound_val = -bound_val

    gap_val = stats["gap"]

    # A *feasible* exit never inherits the tree's validity flag as a certificate:
    # the flag attests bound validity, not gap closure, and a node/budget-limited
    # feasible exit leaves the tree gap open. See the spatial-path note above
    # (max_nodes=1 false-certification regression). This path carries no rigorous
    # root-relaxation fallback, so a dropped certificate is not re-earned here.
    if status == "feasible":
        _gap_certified = False

    if not _gap_certified:
        # Tree bound/gap are not rigorous on an uncertified exit (a node may
        # have been pruned without proof); drop them.
        bound_val = None
        gap_val = None
    # The root LP relaxation bound is always a valid dual bound. Surface it
    # whenever the tree did not yield a usable finite bound — an uncertified
    # exit (dropped above) or a certified-but-time-limited solve that never
    # closed a node (global_lower_bound still -inf). This lets AMP's short
    # per-iteration MILP budget return a finite lower bound instead of None.
    if (bound_val is None or not np.isfinite(bound_val)) and np.isfinite(_root_lp_bound):
        bound_val = -_root_lp_bound if _maximize else _root_lp_bound

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=bound_val,
        gap=gap_val,
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
        constraint_duals=constraint_duals,
        bound_duals_lower=bound_duals_lower,
        bound_duals_upper=bound_duals_upper,
        gap_certified=_gap_certified,
    )


def _solve_miqp_bb(
    model: Model,
    time_limit: float,
    gap_tolerance: float,
    batch_size: int,
    strategy: str,
    max_nodes: int,
    t_start: float,
    prefer_pounce: bool = False,
) -> SolveResult:
    """Solve a MIQP via B&B with QP relaxation solves at each node.

    ``prefer_pounce`` (POUNCE-only mode) routes the incumbent dual recovery
    through POUNCE instead of HiGHS so the whole solve is HiGHS-free.
    """
    import jax.numpy as jnp

    from discopt._jax.problem_classifier import extract_qp_data
    from discopt._jax.qp_ipm import qp_ipm_solve

    rust_time = 0.0
    jax_time = 0.0

    t_jax_start = time.perf_counter()
    qp_data = extract_qp_data(model)
    jax_time += time.perf_counter() - t_jax_start

    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)
    n_orig = sum(v.size for v in model._variables)

    # --- Root presolve: FBBT before tree creation ---
    # The node QP IPM diverges to NaN on variables with infinite bounds (e.g.
    # alan's x0..x3 are [0, inf), bounded only implicitly by the equalities).
    # A NaN solve was then mis-pruned as rigorously infeasible, yielding a
    # false-infeasible verdict on a feasible model (issue #127). FBBT tightens
    # such bounds (here to [0, 1]/[0, 0.833]) so every node relaxation is
    # well-posed; a True root_infeasible from FBBT is itself a sound proof.
    t_rust_start = time.perf_counter()
    from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

    lb, ub, root_infeasible, _ = tighten_root_bounds_with_fbbt(
        model, lb, ub, int_offsets, int_sizes
    )
    rust_time += time.perf_counter() - t_rust_start
    if root_infeasible:
        wall_time = time.perf_counter() - t_start
        return SolveResult(
            status="infeasible",
            objective=None,
            bound=None,
            gap=None,
            x=None,
            wall_time=wall_time,
            node_count=0,
            rust_time=rust_time,
            jax_time=jax_time,
            python_time=wall_time - rust_time - jax_time,
        )

    t_rust_start = time.perf_counter()
    tree = PyTreeManager(n_vars, lb.tolist(), ub.tolist(), int_offsets, int_sizes, strategy)
    tree.initialize()
    rust_time += time.perf_counter() - t_rust_start

    # A node relaxation that stalled at the iteration limit (converged==3) is
    # not at KKT, so its objective is not a valid lower bound for the node
    # (f(x~) >= f*); trusting it can prune the true integer optimum. Such
    # nodes are first re-solved with POUNCE; only unrecoverable ones
    # decertify the gap (mirrors the P0.3 trust-gate + polish-retry).
    _gap_certified = True
    _n_total0 = qp_data.A_eq.shape[1] if qp_data.A_eq.shape[0] > 0 else n_orig
    _A_ub_m, _b_ub_m, _A_eq_m, _b_eq_m = _decompose_eq_slack_form(
        np.asarray(qp_data.A_eq), np.asarray(qp_data.b_eq), n_orig, _n_total0 - n_orig
    )
    _c_m = np.asarray(qp_data.c[:n_orig])
    _Q_m = np.asarray(qp_data.Q[:n_orig, :n_orig])

    def _maybe_inject_snapped(x_row, node_lb_i, node_ub_i):
        # Purification (increment 3): near-integral interior points become
        # exact incumbents via snap-fix-resolve.
        inc = _pounce_snap_incumbent(
            x_row,
            int_offsets,
            int_sizes,
            node_lb_i,
            node_ub_i,
            _c_m,
            float(qp_data.obj_const),
            _A_ub_m,
            _b_ub_m,
            _A_eq_m,
            _b_eq_m,
            t_start,
            time_limit,
            Q=_Q_m,
        )
        if inc is not None:
            tree.inject_incumbent(
                np.asarray(inc[1][:n_vars], dtype=np.float64).copy(), float(inc[0])
            )

    def _handle_nonclean(i, lbs, sols, x_full, obj_val, node_lb_i, node_ub_i):
        # A node whose QP relaxation did not cleanly converge (non-KKT, solver
        # failure, or NaN iterate) is not a valid lower bound, and — crucially —
        # is not a proof of infeasibility. Pruning it as rigorously infeasible is
        # the false-infeasible bug (issue #127).
        #
        # ``x_full`` is the returned iterate (or None). When it is finite and
        # constraint-feasible it is a genuine feasible point: its objective
        # ``obj_val`` is exact there, so it is a valid incumbent (upper bound).
        # We keep it (the tree harvests an integer-feasible point) and decertify
        # the gap, since as a *lower* bound a non-KKT objective is untrusted.
        # Otherwise there is no usable iterate, so we re-solve with POUNCE: a
        # Phase-1-certified-infeasible verdict prunes soundly, an optimal one
        # restores a trusted bound, and an inconclusive one keeps the node open
        # (lb=-inf — branched, never fathomed by a bogus bound) and decertifies.
        nonlocal _gap_certified
        finite_feas = (
            x_full is not None
            and bool(np.all(np.isfinite(x_full)))
            and _check_lp_solution_feasibility(qp_data.A_eq, qp_data.b_eq, x_full)
        )
        if finite_feas:
            # Try first to recover a trusted (KKT) lower bound; if POUNCE
            # certifies one, the node certifies normally. Otherwise keep the
            # non-KKT objective as an (untrusted) bound and decertify — the
            # feasible iterate is still a valid incumbent either way.
            rec = _pounce_recover_node_bound(
                node_lb_i,
                node_ub_i,
                _c_m,
                float(qp_data.obj_const),
                _A_ub_m,
                _b_ub_m,
                _A_eq_m,
                _b_eq_m,
                t_start,
                time_limit,
                Q=_Q_m,
            )
            if rec is not None and rec[0] == "optimal":
                lbs[i] = rec[1]
                sols[i] = np.asarray(rec[2][:n_vars], dtype=np.float64)
            else:
                sols[i] = np.asarray(x_full[:n_vars], dtype=np.float64)
                lbs[i] = float(obj_val)
                _gap_certified = False
            if lbs[i] < _SENTINEL_THRESHOLD:
                _maybe_inject_snapped(sols[i], node_lb_i, node_ub_i)
            return
        lb_c = np.clip(node_lb_i, -_SPC, _SPC)
        ub_c = np.clip(node_ub_i, -_SPC, _SPC)
        sols[i] = 0.5 * (lb_c + ub_c)
        rec = _pounce_recover_node_bound(
            node_lb_i,
            node_ub_i,
            _c_m,
            float(qp_data.obj_const),
            _A_ub_m,
            _b_ub_m,
            _A_eq_m,
            _b_eq_m,
            t_start,
            time_limit,
            Q=_Q_m,
        )
        if rec is not None and rec[0] == "optimal":
            lbs[i] = rec[1]
            sols[i] = np.asarray(rec[2][:n_vars], dtype=np.float64)
            if lbs[i] < _SENTINEL_THRESHOLD:
                _maybe_inject_snapped(sols[i], node_lb_i, node_ub_i)
        elif rec is not None:  # Phase-1-certified infeasible: rigorous prune.
            lbs[i] = _INFEASIBILITY_SENTINEL
        else:  # inconclusive — keep the node open, never a false-infeasible.
            lbs[i] = -np.inf
            _gap_certified = False

    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, _batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        t_jax_start = time.perf_counter()
        result_ids = np.array(batch_ids, dtype=np.int64)
        n_slack = qp_data.x_l.shape[0] - n_orig

        if n_batch > 1:
            # Batch QP solve via vmap
            from discopt._jax.qp_ipm import qp_ipm_solve_batch

            xl_arr = jnp.array(batch_lb, dtype=jnp.float64)
            xu_arr = jnp.array(batch_ub, dtype=jnp.float64)
            slack_l = jnp.zeros((n_batch, n_slack), dtype=jnp.float64)
            slack_u = jnp.full((n_batch, n_slack), 1e20, dtype=jnp.float64)
            xl_full = jnp.concatenate([xl_arr, slack_l], axis=1)
            xu_full = jnp.concatenate([xu_arr, slack_u], axis=1)

            try:
                state = qp_ipm_solve_batch(
                    qp_data.Q,
                    qp_data.c,
                    qp_data.A_eq,
                    qp_data.b_eq,
                    xl_full,
                    xu_full,
                )
                converged = np.asarray(state.converged)
                obj_vals = np.asarray(state.obj)
                x_vals = np.asarray(state.x)

                # Only a cleanly converged (KKT) QP with a finite iterate is a
                # trustworthy bound or infeasibility verdict. conv==3 (max-iter)
                # and non-finite iterates are routed to POUNCE recovery instead
                # of being pruned as infeasible (issue #127).
                finite_rows = np.all(np.isfinite(x_vals), axis=1)
                clean = ((converged == 1) | (converged == 2)) & finite_rows
                result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                    ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                    if clean[i] and _check_lp_solution_feasibility(
                        qp_data.A_eq, qp_data.b_eq, x_vals[i]
                    ):
                        result_lbs[i] = obj_vals[i] + float(qp_data.obj_const)
                        result_sols[i] = x_vals[i, :n_vars]
                        _maybe_inject_snapped(
                            result_sols[i], np.array(batch_lb[i]), np.array(batch_ub[i])
                        )
                    elif clean[i]:
                        # Converged + finite but the node box is genuinely
                        # infeasible: a sound prune.
                        result_sols[i] = 0.5 * (lb_c + ub_c)
                    else:
                        # Non-KKT / non-finite: untrusted bound, never a
                        # false-infeasible prune (issue #127).
                        _handle_nonclean(
                            i,
                            result_lbs,
                            result_sols,
                            x_vals[i],
                            obj_vals[i] + float(qp_data.obj_const),
                            np.array(batch_lb[i]),
                            np.array(batch_ub[i]),
                        )
            except Exception as e:
                logger.debug("Batch QP solve failed: %s", e)
                result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                    ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)
            result_feas = np.zeros(n_batch, dtype=bool)
        else:
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.zeros(n_batch, dtype=bool)

            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                x_l_node = jnp.array(node_lb, dtype=jnp.float64)
                x_u_node = jnp.array(node_ub, dtype=jnp.float64)

                x_l_full = jnp.concatenate([x_l_node, jnp.zeros(n_slack)])
                x_u_full = jnp.concatenate([x_u_node, jnp.full(n_slack, 1e20)])

                try:
                    state = qp_ipm_solve(
                        qp_data.Q,
                        qp_data.c,
                        qp_data.A_eq,
                        qp_data.b_eq,
                        x_l_full,
                        x_u_full,
                    )
                    conv = int(state.converged)
                    x_np = np.asarray(state.x)
                    # Only a cleanly converged (KKT) QP with a finite iterate is
                    # trustworthy as a bound or an infeasibility verdict. conv==3
                    # (max-iter) and non-finite iterates go to POUNCE recovery
                    # rather than being pruned as infeasible (issue #127).
                    clean = conv in (1, 2) and bool(np.all(np.isfinite(x_np)))
                    lb_c = np.clip(node_lb, -_SPC, _SPC)
                    ub_c = np.clip(node_ub, -_SPC, _SPC)
                    if clean and _check_lp_solution_feasibility(
                        qp_data.A_eq, qp_data.b_eq, state.x
                    ):
                        result_lbs[i] = float(state.obj) + qp_data.obj_const
                        result_sols[i] = x_np[:n_vars]
                        if result_lbs[i] < _SENTINEL_THRESHOLD:
                            _maybe_inject_snapped(result_sols[i], node_lb, node_ub)
                    elif clean:
                        # Converged, finite, yet the node box admits no point
                        # satisfying the equalities: a sound infeasibility prune.
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        result_sols[i] = 0.5 * (lb_c + ub_c)
                    else:
                        # Non-KKT / non-finite: untrusted bound, never a
                        # false-infeasible prune (issue #127).
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        _handle_nonclean(
                            i,
                            result_lbs,
                            result_sols,
                            x_np,
                            float(state.obj) + qp_data.obj_const,
                            node_lb,
                            node_ub,
                        )
                except Exception as e:
                    logger.debug("Per-node LP/QP solve failed: %s", e)
                    # No usable iterate from a crashed solve: recover or keep
                    # open, never a false-infeasible prune (issue #127).
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    _handle_nonclean(i, result_lbs, result_sols, None, np.nan, node_lb, node_ub)

        jax_time += time.perf_counter() - t_jax_start

        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        iteration += 1
        if tree.is_finished():
            break
        if tree.gap() <= gap_tolerance:
            break
        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time
    stats = tree.stats()
    incumbent = tree.incumbent()

    if incumbent is not None:
        sol_array, obj_val = incumbent
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    constraint_duals = None
    bound_duals_lower = None
    bound_duals_upper = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        x_dict = _unpack_solution(model, sol_flat)

        # Recover relaxation duals at the integer-feasible incumbent by
        # re-solving the QP relaxation with integer variables fixed.
        try:
            n_total = qp_data.A_eq.shape[1] if qp_data.A_eq.shape[0] > 0 else n_orig
            n_slack_local = n_total - n_orig
            A_eq_full = np.asarray(qp_data.A_eq)
            b_eq_full = np.asarray(qp_data.b_eq)
            A_ub_, b_ub_, A_eq_, b_eq_ = _decompose_eq_slack_form(
                A_eq_full, b_eq_full, n_orig, n_slack_local
            )
            Q_orig = np.asarray(qp_data.Q[:n_orig, :n_orig])
            constraint_duals, bound_duals_lower, bound_duals_upper = _mip_recover_relaxation_duals(
                model,
                lp_data=qp_data,
                x_flat=np.asarray(sol_flat[:n_orig], dtype=float),
                n_orig=n_orig,
                A_ub=A_ub_,
                b_ub=b_ub_,
                A_eq=A_eq_,
                b_eq=b_eq_,
                time_limit=max(0.1, time_limit - (time.perf_counter() - t_start)),
                Q_orig=Q_orig,
                prefer_pounce=prefer_pounce,
            )
        except Exception as _exc:
            logger.debug("MIQP-BB dual recovery failed: %s", _exc)

        # Negate objective back for maximization (B&B tree tracks minimization)
        from discopt.modeling.core import ObjectiveSense

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        # "optimal" needs a closed search AND a certified gap: a stalled
        # (non-KKT) node bound leaves optimality unproven even when the tree
        # appears finished.
        if (tree.gap() <= gap_tolerance or tree.is_finished()) and _gap_certified:
            status = "optimal"
        else:
            status = "feasible"
    else:
        x_dict = None
        obj_val = None
        if stats["total_nodes"] >= max_nodes:
            status = "node_limit"
            # No incumbent and a resource limit: nothing is certified. Without an
            # incumbent there is no gap to certify, and a leftover tree bound
            # describes an unexplored search, not a proven optimum — a certified
            # exit here would claim optimality with no solution at all.
            _gap_certified = False
        elif wall_time >= time_limit:
            status = "time_limit"
            _gap_certified = False
        else:
            # Tree exhausted with no feasible node: infeasibility *is* a certified
            # conclusion, so leave _gap_certified untouched.
            status = "infeasible"

    from discopt.modeling.core import ObjectiveSense

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    assert model._objective is not None
    if bound_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        bound_val = -bound_val

    # An uncertified gap is not a rigorous dual bound; do not present one.
    gap_val = stats["gap"]

    # A *feasible* exit never inherits the tree's validity flag as a certificate:
    # the flag attests bound validity, not gap closure, and a node/budget-limited
    # feasible exit leaves the tree gap open. See the spatial-path note above
    # (max_nodes=1 false-certification regression). This path carries no rigorous
    # root-relaxation fallback, so a dropped certificate is not re-earned here.
    if status == "feasible":
        _gap_certified = False

    if not _gap_certified:
        bound_val = None
        gap_val = None

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=bound_val,
        gap=gap_val,
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
        constraint_duals=constraint_duals,
        bound_duals_lower=bound_duals_lower,
        bound_duals_upper=bound_duals_upper,
        gap_certified=_gap_certified,
    )
