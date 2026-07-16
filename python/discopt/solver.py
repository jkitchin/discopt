"""
Solver orchestrator: end-to-end Model.solve() via NLP-based spatial Branch & Bound.

Connects:
  - PyTreeManager (Rust B&B engine) for node management / branching / pruning
  - NLPEvaluator (JAX) for objective/gradient/Hessian/constraint/Jacobian
  - solve_nlp (cyipopt) for continuous relaxation solves at each node
"""

from __future__ import annotations

import functools
import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

import numpy as np

# ``flat_variable_bounds`` lives in a JAX-free helper module, so importing it
# does not pull in JAX. The JAX-dependent helpers (NLPEvaluator, alphaBB,
# nonlinear bound tightening) are imported lazily at their nonlinear-path call
# sites, so a pure LP/MILP/MIQP solve never pays JAX/XLA cold-start.
from discopt._jax.model_utils import flat_variable_bounds

if TYPE_CHECKING:
    from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._rust import PyTreeManager
from discopt.constants import INFEASIBILITY_SENTINEL as _INFEASIBILITY_SENTINEL
from discopt.constants import SENTINEL_THRESHOLD as _SENTINEL_THRESHOLD
from discopt.constants import STARTING_POINT_CLIP as _SPC
from discopt.debug import outermost_solve as _debug_outermost_solve
from discopt.modeling.core import (
    Constraint,
    CustomCall,
    Model,
    SolveResult,
    VarType,
)
from discopt.solver_tuning import current as _tuning
from discopt.solver_tuning import reset_current as _reset_tuning
from discopt.solver_tuning import set_current as _set_tuning
from discopt.solvers import SolveStatus

# R3a measurement sink (temporary, behavior-neutral). When set to a mutable
# dict by an experiment harness, the nonconvex B&B path stores the Rust tree's
# per-variable branch-frequency vector under key "branch_var_counts" just before
# building its result. This is instrumentation only: it never influences a
# branching, bounding, or feasibility decision. Left as ``None`` in all normal
# solves so there is zero overhead and no observable behavior change.
_R3A_BRANCH_COUNT_SINK: Optional[dict] = None

# ``solve_nlp`` (cyipopt) is imported lazily at its nonlinear-path call site in
# ``_solve_continuous`` so a pure LP/MILP/MIQP solve does not pull in the JAX-
# backed NLPEvaluator that ``nlp_ipopt`` imports at module scope.

logger = logging.getLogger(__name__)


def _get_heuristic_governor():
    """Return the process-lifetime heuristic governor (G2).

    Lazy-imported so the module has no import-time coupling to the governor and
    so ``heuristic_governor`` (a small leaf module) stays independently testable.
    The governor is inert unless ``DISCOPT_HEURISTIC_GOVERNOR`` is set (default
    behaviour byte-identical).
    """
    from discopt.heuristic_governor import governor

    return governor()


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
        if not math.isfinite(val):
            # An ill-conditioned node LP can return a non-finite relaxation value
            # for a column (issue #640: a >1e12-spread lifted relaxation on which
            # the simplex ratio test overflows). Such a value carries no branching
            # signal — skip it rather than crash ``math.floor(nan)``; the standard
            # selector then branches on a finite column (sound: a hint is only ever
            # an accelerator, never a correctness dependency).
            continue
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
# Dense-Jacobian compilation guard. ``NLPEvaluator.evaluate_jacobian`` uses
# ``jax.jit(jax.jacfwd(...))`` — a forward-mode dense Jacobian whose compiled XLA
# program replicates the whole constraint system once per input variable. For a
# large model (n inputs x m constraints) that jaxpr explodes during MLIR
# lowering and XLA aborts the *process* with a native SIGBUS/SIGILL — not a
# catchable Python exception. Above this many dense entries (n * m), code paths
# that only need the Jacobian as an optimization skip the dense compile and use
# a Jacobian-free fallback. rsyn0810m03hfsg (1185 x 1935 ~ 2.3M) crashed here
# once presolve was fast enough to reach node bound-tightening; tln6 (~1.7k) and
# the broad corpus sit far below the cap, so this is inert on normal models.
_MAX_DENSE_JACOBIAN_ELEMS = 1_000_000
# Per-node OBBT (Lever A) gates. Per-node optimization-based bound tightening is
# powerful but costs O(n_vars) LPs per node, so it is enabled only for the
# functionally-dependent-intermediate structural class and on small models, and
# its cumulative wall time is capped to a fraction of the solve budget. These
# defaults keep it inert on the broad corpus while letting the welded-beam
# (nvs05) class certify (see ``dependent_vars`` and the batch loop).
_PER_NODE_OBBT_MAX_VARS = 100
_PER_NODE_OBBT_BUDGET_FRAC = 0.6
_PER_NODE_OBBT_PER_NODE_S = 3.0
_PER_NODE_OBBT_PER_LP_S = 0.3
_PER_NODE_OBBT_ROUNDS = 3
# T2.5 (flag-gated, default OFF): scored top-k OBBT de-gate. BARON/SCIP reduce at
# every node but affordably, by scoring variables and probing only the most
# promising ones. discopt's per-node OBBT probes all columns in index order —
# O(n) LPs/node — so it is size-gated off at n>_PER_NODE_OBBT_MAX_VARS, which
# skips large stall-class spatial models entirely (casctanks n=560 never runs
# per-node OBBT; F12). With ``DISCOPT_OBBT_TOPK=1`` the gate instead runs OBBT on
# the top-k ``width × |reduced cost|`` variables for models above the size cap,
# bounding the per-sweep probe count so the existing per-node/cumulative budgets
# can hold. Selecting *which* variables to tighten is a pure affordability lever;
# every surviving tightening is still NS-safe (soundness unchanged). Default OFF
# until the differential + panel gates are green on consecutive nightlies.
_PER_NODE_OBBT_TOPK = 20


def _obbt_topk_enabled() -> bool:
    """Whether the T2.5 scored top-k OBBT de-gate is on (env flag, default OFF)."""
    return os.environ.get("DISCOPT_OBBT_TOPK", "").strip().lower() in ("1", "true", "yes", "on")


# P3 branch-and-reduce: per-node probing (issue #632). When the in-tree FBBT
# pass runs (``in_tree_presolve_stride > 0``), also probe discrete variables at
# the node — tentatively fix each at a bound, re-run cutoff-FBBT, and contract
# on a proven-infeasible fixing (binaries forced, integer endpoints peeled).
# Sound by construction (contracts only on proven infeasibility); default OFF
# because it costs O(discrete) extra FBBT solves per firing.
def _node_probing_enabled() -> bool:
    """Whether P3 per-node probing is on (``DISCOPT_NODE_PROBING``, default OFF)."""
    return os.environ.get("DISCOPT_NODE_PROBING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _node_probe_max_vars() -> int:
    """Per-node probing budget (discrete vars probed per firing); default 32."""
    try:
        return max(0, int(os.environ.get("DISCOPT_NODE_PROBE_MAX_VARS", "32")))
    except ValueError:
        return 32


# PF1 (issue #632): telemetry counting how many times the in-tree presolve
# (FBBT / branch-and-reduce) kernel actually fired on the GLOBAL spatial B&B
# node loop. The spike found the kernel was wired only into ``_solve_nlp_bb``
# and this count was 0 on spatial instances. A regression test resets this and
# asserts it is > 0 after a stride>=1 solve of such an instance, pinning the
# wiring so it cannot silently disconnect again. Process-global, reset at the
# top of ``solve_model``; read via ``_in_tree_presolve_global_calls()``.
_IN_TREE_PRESOLVE_GLOBAL_CALLS = 0


def _in_tree_presolve_global_calls() -> int:
    """Firings of the in-tree presolve kernel on the global spatial B&B path."""
    return _IN_TREE_PRESOLVE_GLOBAL_CALLS


# Root branch-and-reduce fixpoint (cert:T2.3) no-offtarget gate: skip the loop when
# the relative gap between the root dual bound and the incumbent cutoff is at/below
# this — an already-tight root has nothing to close, so running it would only add
# wall cost on the already-fast class (the §14 T2.4 ≤1.05 no-offtarget guard).
_ROOT_FIXPOINT_MIN_GAP = 1e-4
# Auto build-time level-1 RLT gate. ``rlt="auto"`` (the default) leaves build-time
# level-1 RLT off, but its root-bound tightening certifies several small nonconvex
# instances the per-node-only policy leaves open (nvs05: a 6.94 incumbent with a
# 2.02 bound -> certified 5.4709). The constraint×bound product rows make the
# per-node LP catastrophically slow on large models (casctanks, 500 vars: 359 s
# for one node), so it is auto-engaged only at or below this lifted-variable count.
_AUTO_RLT_LEVEL1_MAX_VARS = 50
# Convex-objective node bound (the supporting-hyperplane lower bound for a model
# whose minimized objective is a convex quadratic). No size cap: the bound is a
# deterministic projected-gradient solve on the constant Hessian and is valid at
# any iterate (see ``_convex_objective_lower_bound``). Gated only on a PSD Hessian
# with this margin, so the convexity verdict (hence bound soundness) is never
# borderline.
_CONVEX_OBJ_PSD_TOL = 1e-6
# Lazy re-separation, global-bound-stall governor (C-42 Part 2, THRU-4
# follow-on; the relaxer-side stride net is ``_LAZY_RESEP_STRIDE`` in
# ``mccormick_lp.py``). Active only under pool inheritance
# (``DISCOPT_CUT_INHERIT`` with a captured root pool). All three constants are
# GLOBAL — never tuned per instance:
#   * ``_LAZY_RESEP_STALL_WINDOW``: node solves without any global-lower-bound
#     improvement before re-separation is probed. Small enough to catch the
#     tspn05-class freeze well inside its budget, large enough that the
#     steadily-closing nvs19-class (bound moves every few nodes) never probes.
#   * ``_LAZY_RESEP_PROBE_BUDGET``: node solves the probe may re-separate
#     before concluding separation is bound-inert here (the nvs24 signature)
#     and muting until the bound next moves — this caps the throughput cost of
#     a wrong probe at ``budget / window`` of the node solves.
#   * ``_LAZY_RESEP_GLB_EPS``: relative improvement that counts as progress;
#     near-zero because the stall signature is an EXACTLY frozen bound.
# Per-node alternatives (parent-bound stall, LP-gain productivity) were tried
# and falsified by measurement — see ``docs/dev/c42-cut-inherit-fix-2026-07-07.md``.
_LAZY_RESEP_STALL_WINDOW = 24
_LAZY_RESEP_PROBE_BUDGET = 8
_LAZY_RESEP_GLB_EPS = 1e-9
# Floor on the time budget handed to the end-of-solve root-relaxation fallback
# bound (issue #138). On a hard nonconvex minimize the B&B loop can consume the
# entire `time_limit` and exit uncertified, leaving no time for the rigorous
# root MILP-relaxation bound — so a sound dual bound is dropped to None. This
# floor lets the fallback still run a small, bounded solve so an uncertified exit
# reports a finite *sound* bound instead of None. It is paid only when the search
# produced no usable bound at all (never on a clean/certified solve), and the
# fallback's own internal budget (~10% of this) keeps the overrun small.
_ROOT_FALLBACK_FLOOR_S = 3.0
# Multistart runs 3 starts per nonconvex node (warm/midpoint/random) and keeps
# the best — better incumbents on nonconvex models, but ~3x the node-solve cost.
# Off by default; flip to opt in (or pass multistart=True to _solve_batch_pounce).
_POUNCE_BATCH_MULTISTART = False

# Native-AD node NLP solves (discopt#281): route the per-node NLP through
# POUNCE's own AD on the .nl problem instead of the JAX callback bridge. Opt-in
# (DISCOPT_NLP_NATIVE / options["nlp_native"]); falls back to the JAX path
# automatically whenever a native base cannot be built/validated for the model
# (see solvers.nlp_native). Default OFF: POUNCE's PyNlProblem is unsendable
# (pyo3), so caching it on the model and using it across the batch/parallel paths
# trips "unsendable ... dropped on another thread" under pytest-xdist and can
# perturb MIQP-batch certification; and the speedup is neutral-to-modest. Enable
# explicitly once PyNlProblem is made Send-safe.
_NLP_NATIVE_DEFAULT = os.environ.get("DISCOPT_NLP_NATIVE", "0").lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _native_nlp_enabled(options: dict) -> bool:
    """Whether to attempt the POUNCE-native node NLP path for this solve."""
    val = options.get("nlp_native") if options else None
    return _NLP_NATIVE_DEFAULT if val is None else bool(val)


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

    Thin alias for the canonical :func:`nlp_evaluator.evaluator_fingerprint`; kept
    here for the existing importers (e.g. ``solvers.nlp_native``).
    """
    from discopt._jax.nlp_evaluator import evaluator_fingerprint

    return evaluator_fingerprint(model)


def _make_evaluator(model: Model):
    """Create or reuse a cached NLPEvaluator for the model.

    Delegates to the canonical :func:`nlp_evaluator.cached_evaluator` so the B&B
    loop, the primal heuristics, and the POUNCE node solves all share one cache
    (and one set of compiled callables) instead of each rebuilding the evaluator.
    """
    from discopt._jax.nlp_evaluator import cached_evaluator

    return cached_evaluator(model)


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


def _alphabb_node_box(model, node_lb, node_ub):
    """Build the ``{Variable: Interval}`` box ``rigorous_alpha`` expects.

    Mirrors the flat->box translation used by :func:`_compute_interval_bound`
    so the interval Hessian is enclosed over *this node's* box rather than the
    root box.
    """
    from discopt._jax.convexity.interval import Interval

    box = {}
    offset = 0
    for v in model._variables:
        sz = v.size
        lo = np.asarray(node_lb[offset : offset + sz], dtype=np.float64).reshape(v.shape)
        hi = np.asarray(node_ub[offset : offset + sz], dtype=np.float64).reshape(v.shape)
        box[v] = Interval(lo, hi)
        offset += sz
    return box


def _compute_alphabb_bound(evaluator, model, alphabb_expr, node_lb, node_ub):
    """Compute a valid lower bound by minimizing the alphaBB underestimator.

    L(x) = f(x) - sum_i alpha_i * (x_i - lb_i) * (ub_i - x_i)

    The perturbation term is non-negative for x in [node_lb, node_ub], so
    L(x) <= f(x) there and the minimum of L over the box is a valid lower
    bound on f. The minimization domain MUST be exactly the true node box and
    the perturbation MUST use the same lb/ub arrays as the optimizer bounds:
    evaluating L at any x OUTSIDE [node_lb, node_ub] makes the perturbation
    NEGATIVE, which flips L into an over-estimator and yields an invalid
    (too-high) "lower bound".

    Soundness (C-17). ``alpha`` is derived from :func:`rigorous_alpha`, which
    uses a *sound* interval enclosure of the Hessian over ``[node_lb, node_ub]``
    and a rigorous per-row interval-Gershgorin eigenvalue bound. That guarantees
    ``Hessian(f) + 2 diag(alpha)`` is PSD over the WHOLE box, so L is provably
    convex there and its supporting hyperplane is a valid underestimator. The
    previous implementation instead used a SAMPLED alpha (Hessian evaluated at a
    fixed set of interior points) and checked convexity only at the box center;
    a negative-curvature band narrower than the sample spacing passed the gate
    with alpha too small, making L nonconvex over the box and the "lower bound"
    exceed ``min_box f`` -> false optimal. We now ABSTAIN whenever
    ``rigorous_alpha`` cannot certify convexity (unbounded/indefinite interval
    Hessian -> a ``+inf`` entry), returning ``-inf`` so the caller falls back to
    whatever other VALID bound exists rather than a guessed one.

    Returns a valid lower bound on min f over [node_lb, node_ub] (a supporting
    hyperplane of the convex L; see below), or -inf when the box is unbounded / so
    wide that alphaBB is numerically meaningless, the evaluator cannot supply
    derivatives, or ``rigorous_alpha`` abstains (in which case alphaBB emits no
    bound and the caller's interval / LP relaxation bounds stand).
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

    # Rigorous, per-node alpha from a SOUND interval Hessian over THIS box
    # (C-17). ``rigorous_alpha`` returns ``+inf`` for any variable whose row of
    # the interval Hessian is unbounded/indefinite -> it cannot certify
    # convexity, so we ABSTAIN rather than emit a guessed (possibly invalid)
    # bound. Building the enclosure over the node box (not the root box) also
    # tightens alpha as B&B subdivides.
    from discopt._jax.alphabb import rigorous_alpha

    try:
        box = _alphabb_node_box(model, node_lb, node_ub)
        alpha = np.asarray(rigorous_alpha(alphabb_expr, model, box), dtype=np.float64)
    except (ValueError, ArithmeticError, RuntimeError, KeyError, IndexError, TypeError) as e:
        logger.debug("rigorous alphaBB alpha failed: %s", e)
        return -np.inf
    if not np.all(np.isfinite(alpha)):
        # Interval Hessian unbounded/indefinite -> convexity uncertifiable.
        return -np.inf

    n = node_lb.shape[0]
    center = 0.5 * (node_lb + node_ub)

    # The bound is a SUPPORTING HYPERPLANE of the convex underestimator
    #   L(x) = f(x) - sum_i alpha_i (x_i-lb_i)(ub_i-x_i),
    # which underestimates f on the box and is convex by alphaBB construction.
    # The hyperplane min over the box is valid at ANY anchor x0 (so this is sound
    # regardless of how well x0 is optimized), and exact at x0 = argmin_box L. We
    # find x0 with a deterministic projected-gradient solve on L's own (analytic)
    # derivatives -- no SciPy optimizer, no random restarts. grad/Hessian of the
    # *perturbation* are closed form; f's come from the evaluator. If the evaluator
    # cannot supply them, abstain (sound: the caller's interval/LP bounds stand).
    grad_f = getattr(evaluator, "evaluate_gradient", None)
    hess_f = getattr(evaluator, "evaluate_hessian", None)
    if grad_f is None:
        return -np.inf

    def L(x):
        x = np.asarray(x, dtype=np.float64)
        pert = float(np.sum(alpha * (x - node_lb) * (node_ub - x)))
        return float(evaluator.evaluate_objective(x)) - pert

    def grad_L(x):
        # d/dx_i [alpha_i (x_i-lb_i)(ub_i-x_i)] = alpha_i (lb_i + ub_i - 2 x_i).
        return np.asarray(grad_f(x), dtype=np.float64) - alpha * (node_lb + node_ub - 2.0 * x)

    # Convexity of L is GUARANTEED by construction: ``rigorous_alpha`` bounds the
    # interval Hessian over the whole box so Hf + 2 diag(alpha) is PSD everywhere
    # in [node_lb, node_ub]. We therefore do NOT re-gate on a center-only Hessian
    # (that check was the C-17 bug — it passed on sampled alpha that left L
    # nonconvex off-center). The center Hessian is used only to pick a FISTA step
    # size for the anchor search (tightness, never validity): the supporting
    # hyperplane of a convex L is a valid underestimator at ANY anchor, so a poor
    # step at worst loosens the bound, never invalidates it.
    lipschitz = 0.0
    try:
        Hf = np.asarray(hess_f(center), dtype=np.float64) if hess_f is not None else None
        if Hf is not None and Hf.shape == (n, n) and np.all(np.isfinite(Hf)):
            eigs = np.linalg.eigvalsh(0.5 * (Hf + Hf.T) + 2.0 * np.diag(alpha))
            lipschitz = float(np.abs(eigs).max())  # >= ||Hessian(L)||_2 at the center
    except (ValueError, ArithmeticError, RuntimeError):
        lipschitz = 0.0

    # Deterministic FISTA projected gradient toward argmin_box L (tightness only).
    x_hat = center.copy()
    if np.isfinite(lipschitz) and lipschitz > 0.0:
        step = 1.0 / lipschitz
        x = center.copy()
        y = center.copy()
        t = 1.0
        for _ in range(200):
            try:
                gy = grad_L(y)
            except (ValueError, ArithmeticError, RuntimeError):
                break
            if not np.all(np.isfinite(gy)):
                break
            x_new = np.clip(y - step * gy, node_lb, node_ub)
            if np.max(np.abs(x_new - x)) <= 1e-12 * (1.0 + np.max(np.abs(x_new))):
                x = x_new
                break
            t_new = 0.5 * (1.0 + float(np.sqrt(1.0 + 4.0 * t * t)))
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x, t = x_new, t_new
        x_hat = np.clip(x, node_lb, node_ub)

    # min over the box of L(x_hat) + grad L(x_hat).(x - x_hat) <= L(x) <= f(x).
    try:
        g_hat = grad_L(x_hat)
        L_hat = L(x_hat)
    except (ValueError, ArithmeticError, RuntimeError):
        return -np.inf
    if not (np.all(np.isfinite(g_hat)) and np.isfinite(L_hat)):
        return -np.inf
    tangent_min = L_hat + float(
        np.sum(np.where(g_hat >= 0.0, g_hat * (node_lb - x_hat), g_hat * (node_ub - x_hat)))
    )
    if not np.isfinite(tangent_min):
        return -np.inf
    # Magnitude-scaled margin so the float64 evaluation stays a valid lower bound.
    return float(tangent_min - 1e-9 * (1.0 + abs(L_hat) + abs(tangent_min)))


def _objective_is_convex_quadratic(
    model: Model, evaluator, n_vars: int, remaining_budget: float | None = None
) -> bool:
    """Whether the internally-minimized objective is a convex quadratic.

    True iff (a) no term in the model is higher than bilinear/square — so the
    objective is at most quadratic — and (b) the objective Hessian is PSD. A
    quadratic has a *constant* Hessian, so a PSD verdict at one point holds on the
    whole space, hence on every B&B node box; the objective is then convex
    everywhere and its supporting-hyperplane underestimator (see
    :func:`_convex_objective_lower_bound`) is a rigorous lower bound.

    This is the structural fact the spatial McCormick relaxation throws away: it
    linearizes a convex x^2 with two tangents (a gap of width^2/4 at the midpoint —
    on a [0,200] integer range that is ~10^4 *per square*), so the LP bound is
    hopelessly loose (nvs17 root -2522 vs the convex bound -1106 ~ the optimum
    -1100). Keeping the convex objective exact recovers an almost-tight bound.

    The eigenvalue test runs on the *evaluator's* objective Hessian, which is the
    internally-minimized objective (negated for a maximize), so PSD here means the
    minimized objective is convex regardless of the user's sense.

    ``remaining_budget`` (seconds): when given, the PSD test is *skipped* (returns
    False, abstaining to the McCormick bound) if the estimated first-time dense
    objective-Hessian XLA compile would not fit the remaining time budget. That
    compile is uninterruptible and super-linear in the objective's quadratic term
    count, so on a large quadratic form (e.g. qap: 21 424 terms, ~48 s objective
    compile) it would otherwise blow the ``time_limit`` (#654). The convex bound is
    a *tightening only*, so abstaining is always sound — the dual bound stands.
    """
    if model._objective is None or n_vars == 0:
        return False
    try:
        from discopt._jax.term_classifier import classify_nonlinear_terms

        t = classify_nonlinear_terms(model)
        # Reject anything above degree two anywhere in the model. ``monomial`` maps
        # ``var -> power``, so a cubic ``x**3`` shows up as power 3 here even though
        # it leaves ``general_nl`` empty — without this guard it would be mistaken
        # for a quadratic and the single-point Hessian below would not characterize
        # the (non-constant) curvature, an unsound over-claim.
        # ``monomial`` is an iterable of ``(var, power)`` (a list of pairs or a
        # dict); pull the powers robustly across either shape.
        _monos = list(t.monomial.items() if hasattr(t.monomial, "items") else t.monomial)
        if (
            t.trilinear
            or t.multilinear
            or t.fractional_power
            or t.bilinear_with_fp
            or t.ratio_of_products
            or t.general_nl
            or any(int(deg) > 2 for _, deg in _monos)
        ):
            return False

        # #654 budget gate: the PSD test below forces the dense objective-Hessian
        # compile (``jacfwd∘jacfwd``), whose XLA codegen is super-linear in the
        # objective's quadratic term count and uninterruptible once entered. On a
        # large quadratic form that single compile dwarfs the whole time budget, so
        # skip the (tightening-only) convex bound when it will not fit. The term
        # count is a model-wide upper bound on the objective's quadratic nnz —
        # conservative, so over-estimating only abstains (sound).
        if remaining_budget is not None:
            from discopt._jax.nlp_evaluator import estimate_dense_obj_hessian_compile_s

            _obj_quad_nnz = len(t.bilinear) + sum(1 for _, deg in _monos if int(deg) == 2)
            _compile_est = estimate_dense_obj_hessian_compile_s(_obj_quad_nnz)
            if _compile_est > remaining_budget:
                logger.info(
                    "convex-objective node bound skipped: dense obj-Hessian compile "
                    "~%.1fs (%d quad terms) exceeds remaining budget %.1fs (#654)",
                    _compile_est,
                    _obj_quad_nnz,
                    remaining_budget,
                )
                return False
        lb = np.array([v.lb for v in model._variables for _ in range(v.size)], dtype=np.float64)
        ub = np.array([v.ub for v in model._variables for _ in range(v.size)], dtype=np.float64)
        lb_f = np.where(np.isfinite(lb), lb, -1.0)
        ub_f = np.where(np.isfinite(ub), ub, 1.0)
        # Evaluate the Hessian at two distinct points: a genuine quadratic has a
        # CONSTANT Hessian, so a PSD verdict at one point holds on every node box.
        # Requiring the two to agree is a belt-and-suspenders guard that the
        # objective really is quadratic (rejecting any non-quadratic that slipped
        # past the structural check above) before trusting the constant-Hessian
        # convexity argument.
        H1 = np.asarray(evaluator.evaluate_hessian(0.5 * (lb_f + ub_f)), dtype=np.float64)
        H2 = np.asarray(evaluator.evaluate_hessian(0.25 * lb_f + 0.75 * ub_f), dtype=np.float64)
        if H1.shape != (n_vars, n_vars) or not np.all(np.isfinite(H1)):
            return False
        if not np.allclose(H1, H2, atol=1e-7, rtol=1e-7):
            return False
        # A pure-linear objective has a (near-)zero Hessian; its box bound is
        # already exact via interval arithmetic, so only engage on genuine
        # curvature. The threshold is well above float noise for a well-scaled
        # Hessian, keeping the PSD verdict (and thus the bound) sound.
        eig_min = float(np.linalg.eigvalsh(0.5 * (H1 + H1.T)).min())
        return eig_min >= _CONVEX_OBJ_PSD_TOL
    except Exception:
        return False


def _convex_objective_lower_bound(evaluator, node_lb, node_ub) -> float:
    """Rigorous lower bound on a convex quadratic objective over ``[node_lb, node_ub]``.

    For convex ``f`` and ANY ``x0`` in the box, the supporting hyperplane
    ``f(x0) + grad f(x0) . (x - x0)`` underestimates ``f`` everywhere, so its box
    minimum (separable: each coordinate at ``lb`` or ``ub`` by the gradient's sign)
    is a valid lower bound on ``min_box f`` and hence on the node optimum (the box
    contains the node's feasible set). The bound is *exact* when ``x0 = argmin_box
    f`` — there KKT complementarity zeroes the box-min of the gradient term — and
    only loosens, never becomes invalid, for any other ``x0``.

    So the right ``x0`` is not a tuned heuristic and not something to hunt for with
    random-start local solves: it is the box-constrained minimizer of the *constant*
    quadratic, which we approach with a deterministic projected-gradient (FISTA)
    solve. Because every iterate yields a valid bound, the iteration count is a
    tightness knob, never a soundness one — there is no need for a size cap or a
    cohort-tuned start (both of which the old random-restart L-BFGS version
    required). Caller establishes convexity via
    :func:`_objective_is_convex_quadratic`. Returns ``-inf`` (abstain) on an
    open/empty box or non-finite data.
    """
    nlb = np.asarray(node_lb, dtype=np.float64)
    nub = np.asarray(node_ub, dtype=np.float64)
    if not (np.all(np.isfinite(nlb)) and np.all(np.isfinite(nub))) or np.any(nub < nlb):
        return -np.inf
    n = nlb.shape[0]
    center = 0.5 * (nlb + nub)

    def f(x):
        return float(evaluator.evaluate_objective(np.asarray(x, dtype=np.float64)))

    # Quadratic ==> grad f(x) = H x + g with H *constant*. Recover both from the
    # evaluator at the box center: H is point-independent; g = grad(center) - H@center.
    try:
        H = np.asarray(evaluator.evaluate_hessian(center), dtype=np.float64)
        g0 = np.asarray(evaluator.evaluate_gradient(center), dtype=np.float64)
    except (ValueError, ArithmeticError, RuntimeError):
        return -np.inf
    if H.shape != (n, n) or not (np.all(np.isfinite(H)) and np.all(np.isfinite(g0))):
        return -np.inf
    H = 0.5 * (H + H.T)  # symmetrize the (symmetric) convex Hessian
    g = g0 - H @ center

    # Deterministic FISTA projected gradient for argmin over the box of the convex
    # quadratic. Step 1/L with L an upper bound on the gradient's Lipschitz constant
    # ||H||_2; for symmetric H, ||H||_2 <= ||H||_inf, so 1/||H||_inf is a valid
    # (conservative) step. A fixed iteration budget is safe: under-convergence only
    # loosens the bound. No randomness, no tuned start, no size cap.
    L = float(np.abs(H).sum(axis=1).max())  # ||H||_inf >= ||H||_2
    # Warm-start at the box-projected unconstrained minimizer ``clip(H^-1(-g))``:
    # for a positive-definite H this is the exact ``argmin_box`` whenever it lands
    # in the box (the common, tightest case) and an excellent start otherwise.
    # Deterministic; falls back to the box center if H is singular.
    x_hat = np.clip(center, nlb, nub)
    try:
        xstar = np.linalg.solve(H, -g)
        if np.all(np.isfinite(xstar)):
            x_hat = np.clip(xstar, nlb, nub)
    except np.linalg.LinAlgError:
        pass
    if np.isfinite(L) and L > 0.0:
        step = 1.0 / L
        x = x_hat.copy()
        y = x_hat.copy()
        t = 1.0
        for _ in range(200):
            x_new = np.clip(y - step * (H @ y + g), nlb, nub)
            if np.max(np.abs(x_new - x)) <= 1e-12 * (1.0 + np.max(np.abs(x_new))):
                x = x_new
                break
            t_new = 0.5 * (1.0 + float(np.sqrt(1.0 + 4.0 * t * t)))
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x, t = x_new, t_new
        x_hat = np.clip(x, nlb, nub)

    # Supporting hyperplane at x_hat: f(x_hat) + min_box grad.(x - x_hat).
    grad = H @ x_hat + g
    if not np.all(np.isfinite(grad)):
        return -np.inf
    fx = f(x_hat)
    tangent_min = fx + float(
        np.sum(np.where(grad >= 0.0, grad * (nlb - x_hat), grad * (nub - x_hat)))
    )
    if not np.isfinite(tangent_min):
        return -np.inf
    # Magnitude-scaled margin so the float64 hyperplane evaluation stays a valid
    # (never-too-high) lower bound despite rounding — the same safe-bound discipline
    # as ``obbt._ns_safe_lp_lower_bound``.
    margin = 1e-9 * (1.0 + abs(fx) + abs(tangent_min))
    return float(tangent_min - margin)


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


def _round_incumbent_integers(
    sol_flat,
    int_offsets,
    int_sizes,
    evaluator=None,
    cl_list=None,
    cu_list=None,
    integrality_tol=1e-5,
    feas_tol=1e-4,
):
    """Snap near-integral discrete coordinates of a reported incumbent to exact
    integers, verifying the rounded point stays feasible.

    C-3: the batched JAX IPM leaves integer columns a few digits short of exact
    (e.g. ``2.999997``); such a point still passes the ``1e-5`` integrality gate
    at injection and is stored verbatim as the tree incumbent. The terminal KKT
    polish / dual-recovery re-solve normally rounds and re-solves it, but that
    step is *best-effort and guarded* — if it raises, is skipped, or returns a
    non-adopted result, the raw near-integral coordinates are reported as-is,
    yielding a certified "optimal" whose integer variables are fractional.

    This rounds every discrete coordinate that is within ``integrality_tol`` of
    an integer to that integer (a perturbation no larger than the integrality
    tolerance the point already satisfied) and, when an evaluator + constraint
    bounds are supplied, checks the rounded point against the constraints at
    ``feas_tol``. A coordinate that is *not* within tolerance of any integer is
    left untouched — snapping a genuinely fractional value would fabricate a
    point the search never proved feasible.

    Returns ``(rounded, feasible)``:
      * ``rounded`` — a copy of ``sol_flat`` with in-tolerance integer columns
        snapped exactly. ``sol_flat`` is never mutated.
      * ``feasible`` — ``True`` if no evaluator was supplied (nothing to check)
        or the rounded point satisfies the constraints within ``feas_tol``;
        ``False`` if rounding broke feasibility. The caller must not certify a
        rounded point that reports ``False``.
    """
    rounded = np.asarray(sol_flat, dtype=float).copy()
    changed = False
    for off, sz in zip(int_offsets, int_sizes):
        for j in range(off, off + int(sz)):
            xj = rounded[j]
            if not np.isfinite(xj):
                continue
            nearest = round(float(xj))
            if xj != nearest and abs(xj - nearest) <= integrality_tol:
                rounded[j] = float(nearest)
                changed = True
    if not changed:
        return rounded, True
    if evaluator is None or not cl_list:
        return rounded, True
    feasible = _check_constraint_feasibility(evaluator, rounded, cl_list, cu_list, tol=feas_tol)
    return rounded, feasible


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


def _reduce_node_and_stage(
    reduce_node_fn,
    model,
    i,
    batch_lb,
    batch_ub,
    lp_result,
    tree,
    cutoff,
    pending,
):
    """Run per-node reduce (cert:T2.4b) and stage the tightened child box.

    Calls ``reduce_node`` on node ``i``'s box using the just-solved node LP's
    marginals (``lp_result`` carries ``reduced_costs``/``safe_bound`` when the LP
    requested them) plus cutoff-FBBT. On a strictly smaller box it updates
    ``batch_lb[i]``/``batch_ub[i]`` (so downstream branching/hints see the tighter
    box) and records it in ``pending[i]`` for the ``set_node_bounds`` child export.
    Returns True iff the reduction proved the node infeasible under the cutoff (a
    rigorous fathom). Tighten-only: any failure leaves the box unchanged."""
    try:
        cur_lb = np.asarray(batch_lb[i], dtype=np.float64)
        cur_ub = np.asarray(batch_ub[i], dtype=np.float64)
        res = reduce_node_fn(model, cur_lb, cur_ub, lp_result, cutoff)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("reduce_node failed at node %d: %s", i, exc)
        return False
    if res.infeasible:
        return True
    if res.n_tightened > 0:
        new_lb = np.maximum(cur_lb, res.lb)
        new_ub = np.minimum(cur_ub, res.ub)
        # Guard against an empty box from float noise (fathom).
        if np.any(new_lb > new_ub + 1e-9):
            return True
        batch_lb[i] = new_lb.tolist()
        batch_ub[i] = new_ub.tolist()
        pending[i] = (new_lb, new_ub)
    return False


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

    # Dense-Jacobian compilation guard (see _MAX_DENSE_JACOBIAN_ELEMS). The
    # two-point linearity test below calls evaluate_jacobian, which JIT-compiles
    # a forward-mode dense Jacobian; on a large model that XLA lowering aborts
    # the process with a native SIGBUS/SIGILL the try/except cannot catch. Above
    # the cap, skip the linearity test entirely and fall back to the structural
    # / interval nonlinear tightening (the same fallback taken when no
    # constraint is detected linear) — it needs no monolithic Jacobian compile,
    # so it only forgoes an optimization and never changes a bound unsoundly.
    if n * m > _MAX_DENSE_JACOBIAN_ELEMS:
        return _apply_nonlinear_tightening_with_status(evaluator._model, lb, ub)

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
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

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


def _adaptive_root_random_starts(n_vars):
    """Random multi-start count scaled to the problem dimension.

    The three deterministic anchors (midpoint, lower/upper quarter) already cover
    a low-dimensional box well; extra *random* starts only buy basin diversity as
    the nonconvex landscape grows. Each start is a full NLP solve, so on tiny
    models the fixed ``n_random=2`` is pure per-solve overhead (the dominant cost
    on small instances — see the SCIP head-to-head). Effort grows with dimension:
    none for tiny boxes, the full two random starts once the model is large enough
    to harbour distinct local minima the anchors miss. This only affects the root
    *incumbent* (on nonconvex problems the multistart NLP value is injected as an
    incumbent candidate, never as a dual bound — so the gap/certification is
    untouched; a weaker incumbent only means the B&B tree closes it instead).
    """
    if n_vars <= 5:
        return 0
    if n_vars <= 12:
        return 1
    return 2


def _generate_starting_points(node_lb, node_ub, n_random=2):
    """Generate diverse starting points for multi-start NLP at root node.

    ``n_random=None`` selects :func:`_adaptive_root_random_starts` for the count.
    """
    if n_random is None:
        n_random = _adaptive_root_random_starts(int(np.size(node_lb)))
    lb_clipped = np.clip(node_lb, -_SPC, _SPC)
    ub_clipped = np.clip(node_ub, -_SPC, _SPC)
    span = ub_clipped - lb_clipped

    points = [
        0.5 * (lb_clipped + ub_clipped),  # midpoint
        lb_clipped + 0.25 * span,  # lower-quarter
        lb_clipped + 0.75 * span,  # upper-quarter
    ]

    # Near-lower-bound start (small *absolute* offset, not a span fraction).
    # Half-bounded variables (finite lb, +inf ub) get clipped to ``[lb, _SPC]``,
    # so every span-fraction start (midpoint, quarters, randoms) lands tens of
    # units above the lower bound -- a poor interior-point seed for flows /
    # areas / temperatures whose natural feasible scale is O(1) near their lower
    # bound (heatexch_gen3: the clip-midpoint seed plateaus constraint-
    # infeasible at viol~3e-4, while a seed 0.1 above the lower bound converges
    # to viol~4e-7). The absolute offset reaches feasibility where a 0.25-span
    # offset (== lb+25 here) does not. Added once, tried before the random
    # starts; multistart keeps the best constraint-feasible iterate, so this
    # only ever adds a start -- it cannot worsen the returned point.
    _near_lb_offset = np.minimum(0.1, 0.25 * np.where(span > 0.0, span, 0.1))
    points.append(lb_clipped + _near_lb_offset)

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
    n_random=None,
    convex=False,
    deadline=None,
    observe_cost=None,
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
    # Rolling mean of observed per-start wall (self-calibrating). On the
    # expensive-Hessian class a single start can run many seconds and OVERRUN its
    # own ``max_wall_time`` clamp (the exact Hessian per IPM iteration is slow), so
    # a start launched with a few seconds nominally left still runs ~10 s past the
    # deadline (F4, heatexch_gen3). Refusing to *launch* a further start when the
    # time left cannot absorb one typical start is the only reliable cap. The FIRST
    # start always runs, so the relaxation bound this routine provides is preserved
    # — only the *extra* diversification starts (a primal-quality heuristic: a
    # better local optimum of the SAME relaxation) are capped. Sound: capping extra
    # starts never changes the dual bound.
    _max_start_wall = 0.0

    for _start_idx, x0 in enumerate(starting_points):
        # Deadline enforcement: each NLP start is a full local solve (seconds on
        # large models). Always run the first start (we must return some iterate),
        # then stop launching new starts once the deadline has passed OR the time
        # left cannot absorb another WORST-CASE-so-far start, and shrink each
        # start's own wall budget to the time actually left. The worst-case (max)
        # rather than the mean is used because a single start can OVERRUN its own
        # ``max_wall_time`` clamp by ~10 s on the expensive-Hessian class, so once
        # one long start is observed, no further start is launched unless the time
        # left can absorb one of that size — that is what keeps the ``time_limit``
        # contract (heatexch_gen3, F4).
        if deadline is not None and _start_idx > 0:
            _ms_remaining = deadline - time.perf_counter()
            if _ms_remaining <= max(_DEADLINE_NODE_FLOOR_S, _max_start_wall):
                break
            options = dict(options)
            options["max_wall_time"] = max(_ms_remaining, _DEADLINE_NODE_FLOOR_S)
        _t_start_nlp = time.perf_counter()
        nlp_result = _solve_node_nlp(
            evaluator,
            x0,
            node_lb,
            node_ub,
            constraint_bounds,
            options,
            nlp_solver=nlp_solver,
        )
        _start_wall = time.perf_counter() - _t_start_nlp
        _max_start_wall = max(_max_start_wall, _start_wall)
        # Publish each start's wall to the caller's budget-gate cost tracker so
        # downstream root heuristics (subnlp, diving) know how expensive one NLP
        # solve is on this model even when no relaxation candidate ever ran the
        # feasibility pump (the no-relaxation flowsheet class, F4).
        if observe_cost is not None:
            observe_cost(_start_wall)
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


def _nonrigorous_sentinel_fathom(node_lower_bound: float, node_infeasible: bool) -> bool:
    """Whether a nonconvex node is being fathomed *non-rigorously* (issue #27a).

    A node carrying the failure sentinel (``result_lbs[i] >= SENTINEL_THRESHOLD``)
    but *without* a rigorous infeasibility proof (``node_infeasible`` False — no
    empty McCormick/LP relaxation over the finite box, no
    ``SolveStatus.INFEASIBLE``) is pruned without being proven suboptimal: its
    local NLP merely failed / diverged, or its optimum violated the original
    constraints, neither of which rules out a better point in the subtree. The
    Rust tree still mechanically cuts that subtree by bound (``1e30 >=``
    incumbent, ``tree_manager.rs``), so the *gap* it reports is not certified.
    When this returns True the caller must set ``_gap_certified = False`` — the
    bare frontier gap can no longer certify: the fathom itself proves nothing
    about the removed subtree.

    Terminal accounting (SPATIAL-CERT, #604 parity): the removal is not
    unaccounted — the C-1 sweep floors it at its pop-time bound (proved at an
    ancestor; valid over the subtree forever, #603). The terminal decision
    re-earns certification iff the floor-inclusive bound
    ``min(frontier, taint floor)`` closes the certification gap — i.e. every
    unproven removal is *provably* within tolerance of the incumbent. That is a
    rigorous certificate (every term of the closed gap is a proved bound), so
    "a non-rigorous fathom can never certify global optimality" is preserved:
    the fathom never certifies anything; only the proved floors do. When the
    floor-inclusive gap stays open, the verdict remains "feasible", exactly as
    before.

    A rigorously-infeasible node (``node_infeasible`` True) does NOT decertify:
    an empty relaxation over the box is a valid emptiness proof, so pruning it is
    sound. Extracted from the two twin call sites (serial + batch nonconvex
    finalize) as a pure predicate so the decertification decision is unit
    testable; the behaviour is byte-for-byte the previous inline logic
    (bound-neutral).
    """
    return node_lower_bound >= _SENTINEL_THRESHOLD and not node_infeasible


def _certified_callback_bound(
    global_lower_bound: Optional[float],
    tree_bound_valid: bool,
    is_maximize: bool,
) -> Optional[float]:
    """The certified global dual bound to surface on the callback API.

    A1 (correctness): the callback's ``best_bound`` must never exceed the bound
    the final :class:`SolveResult` would certify. The Rust tree's raw
    ``global_lower_bound`` is the minimum over the *surviving* open frontier, but
    a non-rigorously-fathomed node (an NLP that merely failed/diverged and was
    sentinel-pruned with no infeasibility proof — solver.py:5814/5842) removes an
    unproven subtree from that frontier. Its subtree is not proven suboptimal, so
    the surviving-frontier minimum may sit strictly *above* the true certified
    global bound. Reporting it then over-reports a dual bound the search never
    proved (nvs05: tree ``global_lower_bound = 5.32`` on a tainted tree whose
    rigorous bound is 1.35 — reported bound must stay 1.35). The final
    result-assembly path already drops the tree bound in exactly this case
    (``_tree_bound_valid``); the callback must mirror it.

    Returns the sense-corrected certified bound (for a MAXIMIZE the internal
    minimization tracks ``-obj``, so a lower bound ``L`` on ``-obj`` is an upper
    bound ``-L`` on ``obj``), or ``None`` when no certified bound exists: the
    tree is tainted (``tree_bound_valid`` False), or the bound is the failure
    sentinel / non-finite (``-inf`` root, or the ``1e30`` no-relaxation
    sentinel). ``None`` matches the ``Optional`` convention already used by
    ``incumbent_obj``/``gap``.
    """
    if not tree_bound_valid:
        return None
    if global_lower_bound is None or not np.isfinite(global_lower_bound):
        return None
    if abs(global_lower_bound) >= _SENTINEL_THRESHOLD:
        return None
    return -global_lower_bound if is_maximize else global_lower_bound


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
    tree_bound_valid=True,
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

    from discopt.modeling.core import ObjectiveSense as _ObjectiveSense

    _cb_is_max = model._objective is not None and model._objective.sense == _ObjectiveSense.MAXIMIZE
    _cb_bound = _certified_callback_bound(
        stats.get("global_lower_bound"), tree_bound_valid, _cb_is_max
    )

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
            best_bound=_cb_bound,
            # A2: the gap is only meaningful against a certified dual bound. When
            # ``best_bound`` is None (tainted tree / failure sentinel), the raw
            # ``stats["gap"]`` was computed from that same non-bound, so surface
            # None rather than a gap derived from the sentinel.
            gap=(stats.get("gap") if _cb_bound is not None else None),
            elapsed_time=elapsed,
            x_relaxation=result_sols[i].copy(),
            node_bound=float(result_lbs[i]),
        )

        # --- Lazy constraints ---
        if lazy_constraints is not None:
            # Invoking the user's callback is the only step allowed to fail
            # softly: a user-code error is logged and the node proceeds as
            # normal (no cut, no rejection). Everything AFTER the callback —
            # rejecting the node and inserting the cut — is our own code and
            # must NOT be swallowed. INT-1/INT-4 (#413): the previous broad
            # ``except`` wrapped the ``_cut_pool.add`` too, so an
            # ``AttributeError`` (e.g. a ``None`` cut pool) was downgraded to a
            # warning AND the rejection that followed it was lost, letting the
            # excluded integer-feasible point become the incumbent.
            try:
                cuts = lazy_constraints(ctx, model)
            except Exception as e:
                logger.warning("Lazy constraint callback raised an exception: %s", e)
                cuts = None
            if cuts:
                # Reject the node FIRST: even if cut insertion below were to
                # fail, the point the cuts exclude must never be accepted as an
                # incumbent. This assignment precedes any fallible cut math.
                result_lbs[i] = _INFEASIBILITY_SENTINEL
                for cut in cuts:
                    coeffs, rhs, sense = cut_result_to_dense(cut, model)
                    _cut_pool.add(LinearCut(coeffs=coeffs, rhs=rhs, sense=sense))
                logger.info(
                    "Lazy constraint callback added %d cut(s) at node %d",
                    len(cuts),
                    int(result_ids[i]),
                )
                continue  # skip incumbent callback for cut-separated nodes

        # --- Incumbent callback ---
        if incumbent_callback is not None:
            solution = _unpack_solution(model, result_sols[i])
            # Only the user callback may fail softly; the rejection that acts on
            # its verdict is our code and stays OUTSIDE the swallow (INT-1, #413).
            try:
                accept = incumbent_callback(ctx, model, solution)
            except Exception as e:
                logger.warning("Incumbent callback raised an exception: %s", e)
                accept = None
            if accept is False:
                result_lbs[i] = _INFEASIBILITY_SENTINEL
                logger.info(
                    "Incumbent callback rejected solution at node %d",
                    int(result_ids[i]),
                )


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


def _pack_solution(model: Model, x_dict: dict, n_vars: int) -> np.ndarray:
    """Inverse of :func:`_unpack_solution`: pack a ``{var_name: array}`` dict into
    the flat solution vector, using the same per-variable layout (each variable
    occupies ``v.size`` consecutive slots in ``model._variables`` order)."""
    flat = np.zeros(int(n_vars), dtype=np.float64)
    offset = 0
    for v in model._variables:
        size = int(v.size)
        flat[offset : offset + size] = np.asarray(x_dict[v.name], dtype=np.float64).reshape(-1)
        offset += size
    return flat


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

        # Phase-D lever (perf-d1): the strong-branch score reads only the child LP
        # *objective* bound (never its vertex), so route these repeated LPs through
        # the in-house warm simplex instead of a cold POUNCE IPM solve per call.
        # Controlled by DISCOPT_SEPARATION_LP_SIMPLEX (default ON); the off-switch
        # ("0") restores the caller-selected backend (POUNCE under nlp_solver=
        # "pounce"). Node-neutrality is validated on the cert baseline: the simplex
        # vertex optimum and the IPM analytic-center objective agree to LP
        # tolerance, so the argmax branch decision is unchanged.
        _sb_simplex = os.environ.get("DISCOPT_SEPARATION_LP_SIMPLEX", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        solve_lp = get_lp_solver(prefer_pounce=prefer_pounce and not _sb_simplex)
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


def _relative_gap_from_objective_bound(
    objective: Optional[float],
    bound: Optional[float],
) -> Optional[float]:
    """Return the public relative gap between a mapped incumbent and bound."""
    if objective is None or bound is None:
        return None
    obj = float(objective)
    bnd = float(bound)
    if not np.isfinite(obj) or not np.isfinite(bnd):
        return None
    return abs(obj - bnd) / max(abs(obj), abs(bnd), 1e-10)


# Absolute B&B gap tolerance, decoupled from the relative ``gap_tolerance``.
# Matches the AMP path's ``abs_tol`` (and SCIP's default absolute gap). The tree's
# hybrid ``gap()`` floors its denominator at 1.0, so for an optimum with magnitude
# below 1 the relative ``gap_tolerance`` silently degenerates into an *absolute*
# tolerance — letting a coarse 1e-4 certify a near-zero optimum that is nowhere
# near the true value (gear: returned 3.7e-05 against a trivial 0 bound while the
# true optimum is ~4e-12). A separate, tighter absolute tolerance keeps the search
# going until the absolute gap genuinely closes.
_DEFAULT_ABS_GAP_TOL = 1e-6


def _gap_converged(tree, gap_tolerance: float, abs_gap_tol: float = _DEFAULT_ABS_GAP_TOL) -> bool:
    """B&B gap convergence with decoupled absolute and relative tolerances.

    Converges when the *relative* gap ``(UB-LB)/max(|UB|,|LB|,eps) <= gap_tolerance``
    OR the *absolute* gap ``UB-LB <= abs_gap_tol``. This replaces the bare
    ``tree.gap() <= gap_tolerance`` check: ``tree.gap()`` floors its denominator at
    1.0, collapsing both tolerances into one, so a loose relative ``gap_tolerance``
    certified a near-zero optimum sitting on a trivial 0 bound (the gear pathology).
    Computing the relative gap without the 1.0 floor keeps it meaningful near zero,
    and the independent absolute tolerance provides the only sound way to certify a
    genuinely-zero optimum. ``tree.is_finished()`` remains the exhaustive-search
    terminator at the call sites.
    """
    stats = tree.stats()
    ub = float(stats.get("incumbent_value", float("inf")))
    lb = float(stats.get("global_lower_bound", float("-inf")))
    return _gap_values_converged(ub, lb, gap_tolerance, abs_gap_tol)


def _gap_values_converged(
    ub: float, lb: float, gap_tolerance: float, abs_gap_tol: float = _DEFAULT_ABS_GAP_TOL
) -> bool:
    """Value-form of :func:`_gap_converged` (same arithmetic, explicit bounds).

    Extracted so the terminal certification accounting can run the *identical*
    convergence test on the floor-inclusive rigorous bound
    ``min(frontier, taint floor)`` (SPATIAL-CERT, mirroring the MILP driver's
    #604 discipline) — a second, subtly different gap formula here would be a
    soundness bug of its own.
    """
    if not np.isfinite(ub) or not np.isfinite(lb):
        return False
    abs_gap = max(0.0, ub - lb)
    if abs_gap <= abs_gap_tol:
        return True
    denom = max(abs(ub), abs(lb), 1e-10)
    return abs_gap / denom <= gap_tolerance


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
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

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
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

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

    def _walk(root) -> bool:
        # Explicit-stack DFS (not Python recursion): ``from_nl`` can build a
        # single body tens of thousands of nodes deep, which a per-node recursive
        # walk would overflow the default recursion limit on (issue #271).
        stack = [root]
        while stack:
            expr = stack.pop()
            if isinstance(expr, CustomCall):
                return True
            # Generic child traversal mirroring the DAG node fields used
            # elsewhere (e.g. discopt._jax.cutting_planes): BinaryOp/MatMul ->
            # left/right, UnaryOp/SumExpression -> operand,
            # FunctionCall/CustomCall -> args, SumOverExpression -> terms,
            # IndexExpression -> base.
            left = getattr(expr, "left", None)
            if left is not None:
                stack.append(left)
            right = getattr(expr, "right", None)
            if right is not None:
                stack.append(right)
            operand = getattr(expr, "operand", None)
            if operand is not None:
                stack.append(operand)
            base = getattr(expr, "base", None)
            if base is not None:
                stack.append(base)
            stack.extend(getattr(expr, "args", ()) or ())
            stack.extend(getattr(expr, "terms", ()) or ())
        return False

    obj = getattr(model, "_objective", None)
    if obj is not None and getattr(obj, "expression", None) is not None and _walk(obj.expression):
        return True
    for c in model._constraints:
        body = getattr(c, "body", None)
        if body is not None and _walk(body):
            return True
    return False


def _model_contains_nonsmooth_node(model: Model) -> bool:
    """True if any objective/constraint body contains a non-smooth node.

    ``abs``/``min``/``max`` are convex (or concave) but non-differentiable at
    their kink. A smooth gradient-based NLP can oscillate there and fail to
    certify (e.g. ``min |x|`` over an interval straddling 0 returns the wrong
    point with ``iteration_limit``). The solver uses this to route such models
    to the spatial McCormick B&B, whose exact piecewise relaxation certifies
    them. ``abs`` is either ``UnaryOp("abs")`` or ``FunctionCall("abs")``;
    ``min``/``max`` are ``FunctionCall("min"|"max")``.
    """
    from discopt.modeling.core import FunctionCall, UnaryOp

    _nonsmooth_funcs = {"abs", "min", "max"}

    def _walk(expr) -> bool:
        if isinstance(expr, UnaryOp) and expr.op == "abs":
            return True
        if isinstance(expr, FunctionCall) and expr.func_name in _nonsmooth_funcs:
            return True
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
    """Run the sound model convexity classifier once for solver dispatch.

    The result is memoized on the model (``solve_model`` clears the cache at the
    start of every solve), so the classifier — which is eigenvalue-heavy and can
    cost tens of seconds on large quadratic models — runs at most once per solve
    instead of at each of its several dispatch call sites.

    Classification is also bounded by ``model._convexity_time_budget`` (a fraction
    of ``time_limit``, set by ``solve_model``): if the per-constraint walk would
    overrun that budget it is abandoned and the model is reported as
    convexity-unknown, which routes to the sound spatial Branch and Bound. This
    keeps a tight ``time_limit`` from being blown by classification alone.
    """
    cached = getattr(model, "_convexity_classification_cache", None)
    if cached is not None:
        return cast("tuple[bool, bool, list[bool] | None]", cached)

    from discopt._jax.convexity.rules import ConvexityBudgetExceeded

    # Default cap (15 s) bounds classification even when called outside a budgeted
    # solve_model (e.g. on a model produced by factorable reformulation).
    budget = getattr(model, "_convexity_time_budget", 15.0)
    deadline = (time.perf_counter() + budget) if budget else None
    result: tuple[bool, bool, list[bool] | None]
    try:
        from discopt._jax.convexity import classify_model as _classify_convexity

        # use_certificate=True enables the sound interval-Hessian fallback
        # for constraints/objective the syntactic walker leaves unproven.
        is_convex, constraint_mask = _classify_convexity(
            model, use_certificate=True, deadline=deadline
        )
        if log_nonconvex_continuous and not is_convex:
            logger.info("Nonconvex continuous model detected — using spatial Branch and Bound")
        result = (True, bool(is_convex), list(constraint_mask))
    except ConvexityBudgetExceeded as exc:
        logger.info("Convexity classification skipped (time budget): %s", exc)
        result = (False, False, None)
    except Exception as exc:
        logger.debug("%s: %s", failure_label, exc)
        result = (False, False, None)
    try:
        model._convexity_classification_cache = result
    except Exception:
        pass
    return result


# Size gate for the auto cut policy: above this many scalar variables the
# per-node cut-separation overhead (eigh / extra LP re-solves) outweighs the
# typical bound gain, so the policy declines to enable cuts (sound: a no-op).
_AUTO_CUTS_MAX_VARS = 40

# Structure-cut presolve size gate: the symbolic recognizer runs `model_to_sympy`
# (and SymPy `solve`/`simplify` when objective product terms are present)
# unconditionally, so it is only auto-engaged on models small enough that this is
# cheap. Counted as (sum of scalar variable sizes) + (number of constraints).
_STRUCTURE_CUTS_MAX_SIZE = 100

# RENS root primal heuristic (#281): fraction of the remaining wall budget granted
# to the bound-restricted sub-MINLP solve, and an absolute cap so it can never
# starve the surrounding proof on a long overall budget.
_RENS_BUDGET_FRAC = 0.5
_RENS_BUDGET_CAP_S = 8.0

# Root cut pool (P3). Rounds of spectral PSD separation to run once at the root
# to build the inherited pool; more rounds drive the root bound toward the Shor
# SDP bound (nvs17: 8 rounds -> -2453, ~60 -> -1300, ~150 -> -1221). The pool is
# appended (sound — every PSD cut is globally valid) to each node's relaxation IN
# ADDITION to its own per-node separation, so it only ever tightens the bound.
#
# DEFAULT OFF (rounds=0): on the small dense integer QCQP that motivated it
# (nvs17, 7 vars) the gate that now routes pure-integer nonconvex models onto the
# McCormick-LP path already certifies via per-node separation (~41s); the
# inherited pool cuts the node count ~5x but its per-node LP overhead makes
# wall-clock ~30% worse there. Retained as an opt-in lever (the ``root_cut_rounds``
# / ``root_cut_max`` arguments to :func:`solve_model`, defaulting to the
# ``DISCOPT_ROOT_CUT_ROUNDS`` / ``DISCOPT_ROOT_CUT_MAX`` env vars) for larger
# instances where per-node re-separation, not LP size, dominates.
#
# These env reads are *defaults only*: they are resolved per call inside
# ``solve_model`` (kwarg wins when given), so unlike module-frozen constants they
# can be set after ``import discopt`` and differ per Model/per solve.
_ROOT_CUT_POOL_ROUNDS_ENV = int(os.environ.get("DISCOPT_ROOT_CUT_ROUNDS", "0"))
_ROOT_CUT_POOL_MAX_ENV = int(os.environ.get("DISCOPT_ROOT_CUT_MAX", "200"))

# Marchand-Wolsey aggregation c-MIR separator (cert:P3). DEFAULT-OFF, bound-
# changing per CLAUDE.md §5: it ships dark behind this flag until proven on
# nightlies. Read per-solve (below) so it can be toggled after ``import discopt``.
# The separator is validity-gated (nonnegative row combination + valid MIR ⇒
# valid cut; Rust ``aggregation_validity_random_systems`` property test), so
# enabling it can only add valid cuts, never a false certificate.
_CMIR_AGGREGATION_ENV_DEFAULT = os.environ.get("DISCOPT_CMIR_AGGREGATION", "0").lower() not in (
    "0",
    "",
    "false",
    "no",
    "off",
)


def _cmir_aggregation_enabled() -> bool:
    """Whether the aggregation c-MIR separator is enabled for this solve.

    Re-reads ``DISCOPT_CMIR_AGGREGATION`` each call (default-off) so tests and
    callers can toggle it after import; falls back to the import-time default."""
    val = os.environ.get("DISCOPT_CMIR_AGGREGATION")
    if val is None:
        return _CMIR_AGGREGATION_ENV_DEFAULT
    return val.lower() not in ("0", "", "false", "no", "off")


def _p3_force_cut_path_enabled() -> bool:
    """cert:P3.1c experiment toggle (``DISCOPT_P3_FORCE_CUT_PATH``, default-OFF).

    The Phase 3 1b measurement (``certification-gap-plan.md`` §7) found the
    integer-product / graphpart class routes *away* from any cut seam: the
    big-M reformulation rewrites ``nlp_solver`` from ``"pounce"`` to
    ``"simplex"`` (``:3327``), sending the model to the monolithic Rust
    ``_solve_milp_simplex`` engine, which has no root/per-node cut loop — so the
    aggregation c-MIR / Gomory / cover separators never run. This flag is a
    **cut-reachability entry-experiment lever only**: when set, it SKIPS that
    ``nlp_solver→"simplex"`` reroute, keeping the model on the self-hosted
    ``_solve_milp_bb`` path (``prefer_pounce=True``, which enables the integer
    cut loop via ``_gomory_enabled``). It measures whether making cuts *reachable*
    closes the 0b root gap (GO), before any invasive Rust cut-seam refactor.

    Default-OFF and math-neutral when off: with the flag unset the reroute is
    unchanged, so default dispatch/behavior is bit-for-bit identical."""
    val = os.environ.get("DISCOPT_P3_FORCE_CUT_PATH")
    if val is None:
        return False
    return val.lower() not in ("0", "", "false", "no", "off")


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


_F = TypeVar("_F", bound=Callable[..., Any])


def _scoped_tuning(fn: _F) -> _F:
    """Publish the ``tuning`` kwarg as the active :class:`SolverTuning` for the
    call, then restore the previous context. Relaxer read sites consult
    ``solver_tuning.current()`` instead of ``os.environ`` — so the levers are
    per-call and typed. ``tuning=None`` resolves to a fresh env-default instance
    (the prior global behavior), and the reset prevents one solve's overrides from
    leaking into a later relaxer built outside any solve (e.g. in tests).

    Typed as ``(_F) -> _F`` so the decorated function keeps its original
    signature/return type for callers and the type checker.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = _set_tuning(kwargs.pop("tuning", None))
        try:
            return fn(*args, **kwargs)
        finally:
            _reset_tuning(token)

    return cast(_F, wrapper)


# Several walkers along the solve path (factorable reformulation, the relaxation
# compiler's ``_compile_node``, FBBT, ...) recurse one Python frame per
# expression node. ``from_nl`` can rebuild a *single* constraint body tens of
# thousands of nodes deep (watercontamination0202r ~53k, graphpart_clique-30
# ~7.6k), overrunning CPython's default 1000-frame recursion limit and raising an
# uncaught ``RecursionError`` (issue #271). Mirroring the convexity walk's fix
# (issue #266), the whole solve runs with a depth-scaled recursion limit on a
# large-stack worker thread when — and only when — the model has such a deep
# expression. Shallow models (the overwhelming majority) run inline at the
# default limit, byte-for-byte unchanged.
_DEEP_SOLVE_DEPTH_GATE = 700


def _scoped_deep_recursion(fn: _F) -> _F:
    """Run ``fn(model, ...)`` with size-scaled recursion headroom so a model with
    a very deep expression graph cannot crash the solve with ``RecursionError``.

    The headroom is gated on the deepest single expression in the model, so the
    common shallow case keeps the default limit and runs inline with no worker
    thread (zero behavioural change). Deep models run on a large-stack worker
    thread with a raised limit, the proven mechanism from issue #266.
    """

    @functools.wraps(fn)
    def wrapper(model, *args, **kwargs):
        from discopt._jax.convexity.rules import _run_with_deep_recursion
        from discopt._jax.factorable_reform import _max_expr_node_count

        try:
            depth = _max_expr_node_count(model)
        except Exception:
            depth = 0
        if depth <= _DEEP_SOLVE_DEPTH_GATE:
            return fn(model, *args, **kwargs)
        # A handful of Python frames are entered per node along the deepest path
        # (factorable walk, then the compiler's node walk); cushion generously and
        # cap so a pathological size can't request an unsatisfiable limit.
        depth_need = min(4000 + 8 * depth, 1_000_000)
        return _run_with_deep_recursion(lambda: fn(model, *args, **kwargs), depth_need=depth_need)

    return cast(_F, wrapper)


# Backend-specific keyword arguments that ``solve_model`` forwards through its
# own ``**kwargs`` to an optional backend (AMP / gurobi / mip-nlp / GP / the
# lp-spatial path). These are NOT named parameters of ``solve_model`` but are
# legitimately accepted — the kwarg-validation guard (M6) must allow them so a
# real backend option is never rejected as a typo. Sourced from every
# ``kwargs.get``/``kwargs.pop`` site in this module plus the AMP option list.
_BACKEND_PASSTHROUGH_KWARGS: frozenset[str] = frozenset(
    {
        # lp-spatial diagnostic path
        "lp_spatial",
        "lp_spatial_cut_rounds",
        # gurobi backend
        "gurobi_options",
        # mip-nlp (OA/ECP/LOA) backend
        "mip_nlp_method",
        "mip_nlp_options",
        "equality_relaxation",
        "ecp_mode",
        "feasibility_cuts",
        "init_strategy",
        "heuristic_nonconvex",
        "add_slack",
        "max_slack",
        "oa_penalty_factor",
        "OA_penalty_factor",
        "add_no_good_cuts",
        "integer_to_binary",
        "feasibility_norm",
        "add_regularization",
        "level_coef",
        "stalling_limit",
        "cycling_check",
        "milp_solver",
        "solution_pool",
        "num_solution_iteration",
        "mip_nlp_profile",
        "tree_strategy",
        "cut_strategy",
        "objective_epigraph",
        "anti_epigraph",
        "nonlinear_partitioning",
        "quadratic_partitioning",
        "absolute_value_auxiliaries",
        "monomial_extraction",
        "signomial_extraction",
        "integer_bilinear_strategy",
        "integer_bilinear_max_bits",
        "quadratic_extraction",
        "direct_quadratic_routing",
        "rootsearch_strategy",
        "fixed_nlp_strategy",
        "solution_pool_capacity",
        "hyperplane_max_per_iter",
        "hyperplane_selection_factor",
        "relaxation_phase",
        "mip_solution_limit_strategy",
        "convex_bounding",
        "master_repair",
        "reduction_cuts",
        "fp_iteration_limit",
        "fp_cutoffdecr",
        "fp_projcuts",
        "fp_transfercuts",
        "fp_projzerotol",
        "fp_mipgap",
        "fp_discrete_only",
        "fp_main_norm",
        "fp_norm_constraint",
        "fp_norm_constraint_coef",
        # AMP backend option keys (see ``amp_option_keys`` in solve_model)
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
    }
)


@functools.lru_cache(maxsize=1)
def solve_model_accepted_kwargs() -> frozenset[str]:
    """The complete set of keyword names ``Model.solve`` may forward to the solver.

    Union of (a) ``solve_model``'s own named parameters and (b) the curated
    backend-passthrough keys forwarded through its ``**kwargs``. Used by
    ``Model.solve`` to reject a misspelled/unknown keyword loudly (M6) instead of
    silently swallowing it — a swallowed ``gap_tolerence=…`` leaves the solver at
    the default gap while the user believes it was tightened (a results-integrity
    hazard). ``inspect.signature`` follows ``functools.wraps`` through the two
    decorators to the real signature.
    """
    import inspect

    params = inspect.signature(solve_model).parameters
    named = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    # ``model`` is the positional target, not a forwardable option; drop it.
    named.discard("model")
    return frozenset(named | _BACKEND_PASSTHROUGH_KWARGS)


@_scoped_deep_recursion
@_scoped_tuning
@_debug_outermost_solve
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
    nlp_solver: str = "pounce",
    sparse: Optional[bool] = None,
    cutting_planes: bool = False,
    psd_cuts: bool = False,
    rlt_cuts: bool = False,
    rlt: Union[bool, str] = "auto",
    cuts: str = "auto",
    partitions: int = 0,
    branching_policy: str = "fractional",
    use_learned_relaxations: bool = False,
    mccormick_bounds: str = "auto",
    gdp_method: str = "big-m",
    decomposition: Optional[str] = None,
    decomposition_structure=None,
    record_decomposition: bool = False,
    lagrangian_bound: bool = False,
    lagrangian_frequency: int = 1,
    initial_point: Optional[np.ndarray] = None,
    skip_convex_check: bool = False,
    nlp_bb: Optional[bool] = None,
    lazy_constraints=None,
    incumbent_callback=None,
    node_callback=None,
    solver: Optional[str] = None,
    presolve: bool = True,
    presolve_polynomial: bool = False,
    presolve_reverse_ad: bool = False,
    # PF1 (issue #632): default 1 = FBBT-only in-tree presolve at every node,
    # on BOTH the NLP-BB and global spatial B&B paths. Probing stays OFF
    # (DISCOPT_NODE_PROBING). Spike-supported (m3 proof, fac2 -43% nodes, no
    # losses); sound-by-construction cutoff-aware FBBT (see the node loops).
    in_tree_presolve_stride: int = 1,
    eigenvalue_root_bound: bool = False,
    relaxation_arithmetic: str = "mccormick",
    subnlp_enabled: bool = True,
    subnlp_backend: str = "auto",
    subnlp_frequency: int = 20,
    subnlp_max_calls: int = 200,
    subnlp_options: Optional[dict] = None,
    structure_cuts: bool = True,
    root_cut_rounds: Optional[int] = None,
    root_cut_max: Optional[int] = None,
    _lns_enabled: bool = True,
    rens: bool = True,
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
    nlp_solver : str, default "pounce"
        Numerical engine for every problem class. The default ``"pounce"``
        (POUNCE — pure-Rust Ipopt port) is now the universal default: it
        routes LP/QP/MILP/MIQP/NLP/MINLP through POUNCE (and the self-hosted
        B&B for the integer classes). HiGHS has been removed entirely from the
        LP/MILP path (issue #356). Other values: ``"ipopt"`` (cyipopt for the
        NLP node/continuous solves), or ``"ipm"`` / ``"sparse_ipm"`` (back-compat
        aliases — these select the simplex-first matrix routing for LP/MILP
        and resolve to POUNCE for NLP/MINLP). ``"simplex"`` selects the
        pure-Rust warm-started-simplex MILP B&B.
    sparse : bool or None, default None
        Force sparse (True) or dense (False) Jacobian evaluation.
        If None, auto-selects based on problem size and density.
    cutting_planes : bool, default False
        Enable outer-approximation cut generation after NLP relaxation solves.
    rlt : bool or str, default "auto"
        Reformulation-Linearization Technique (RLT) control for the McCormick
        LP relaxation. ``"auto"`` lets the structure-gated cut policy decide
        per-node RLT cut separation (RLT on QCQP with linear constraints, PSD on
        box-QP) and leaves build-time level-1 RLT off. ``True`` (or ``"on"``)
        engages RLT in full: build-time level-1 constraint×bound products tighten
        the root bound *and* per-node RLT cuts are separated, overriding the auto
        policy. ``False`` (or ``"off"``) forces every RLT lever off. This is the
        first-class replacement for the legacy ``DISCOPT_RLT=1`` environment
        variable (still honored as a force-on override). Every RLT family is
        sound — a constraint×bound product is non-negative at every feasible
        point — so this only trades bound tightness for relaxation size.
    partitions : int, default 0
        Number of piecewise McCormick partitions (0 = standard convex
        relaxation, k > 0 = k partitions for tighter relaxations).
    root_cut_rounds : int, optional
        Opt-in root PSD cut-pool separation rounds: when > 0 and the relaxer
        carries PSD cuts, a strong cut pool is separated once at the root box
        and inherited at every node (sound; never removes a feasible point).
        ``None`` (default) falls back to the ``DISCOPT_ROOT_CUT_ROUNDS`` env
        var (default 0 = off). Resolved per call, so it can be set after import
        and can differ per solve.
    root_cut_max : int, optional
        Cap on the number of inherited root-pool cut rows (keeps per-node LP
        size bounded). ``None`` (default) falls back to ``DISCOPT_ROOT_CUT_MAX``
        (default 200).
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
        ``"none"`` disables.
        (The former ``"midpoint"`` mode was removed in correctness issue C-18:
        it returned the underestimator's value at the box midpoint, which is not
        a valid lower bound on the box minimum, so it could certify a wrong
        optimum. Selecting it now raises ``ValueError``.)
    gdp_method : str, default "big-m"
        Reformulation method for disjunctive constraints:
        ``"big-m"`` (default) or ``"hull"`` (convex hull).
    solver : str or None, default None
        Optional global-solver selector. Use ``"amp"`` to dispatch to
        Adaptive Multivariate Partitioning instead of branch-and-bound.
        Use ``"gurobi"`` to dispatch LP, MILP, QP, MIQP, QCP, QCQP, MIQCP,
        and MIQCQP models to the optional Gurobi backend. General NLP/MINLP
        expressions are not translated into Gurobi nonlinear expressions by
        this selector; unsupported classes raise a clear ``NotImplementedError``
        instead of falling back silently. For global MINLP through discopt's
        AMP algorithm with Gurobi as the MILP-master subsolver, use
        ``solver="amp", milp_solver="gurobi"``.
        Use ``"mip-nlp"`` to dispatch to the MIP-NLP decomposition family
        (OA/ECP/FP/GOA/LP-NLP-BB now; ROA is a reserved method selector).
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
        ``obbt_with_cutoff``, ``alphabb_cutoff_obbt``, ``obbt_time_limit``, and
        ``milp_solver``. ``milp_solver`` accepts ``"auto"``, ``"pounce"``,
        ``"simplex"``, or ``"gurobi"`` (HiGHS was removed, issue #356).
    solver="mip-nlp" options
        The MIP-NLP backend accepts ``mip_nlp_method`` and
        ``mip_nlp_options``. Current implemented methods are ``"oa"``,
        ``"ecp"``, ``"fp"``, ``"goa"``, and ``"lp_nlp_bb"``; ``"roa"`` raises
        ``NotImplementedError`` until its dedicated implementation lands.
        Existing OA options ``equality_relaxation``, ``ecp_mode``,
        ``feasibility_cuts``, ``heuristic_nonconvex``, ``add_slack``,
        ``max_slack``, ``oa_penalty_factor``, ``add_no_good_cuts``,
        ``integer_to_binary``, ``feasibility_norm``, ``add_regularization``,
        ``level_coef``, ``stalling_limit``, ``cycling_check``,
        ``solution_pool``, ``num_solution_iteration``, and ``milp_solver``
        plus initialization option ``init_strategy`` may be passed as top-level
        aliases and take precedence over duplicate keys in ``mip_nlp_options``.
        ``solution_pool``
        currently requires ``milp_solver="gurobi"``. For
        ``mip_nlp_method="lp_nlp_bb"``, ``milp_solver="gurobi"`` is also
        required because the single-tree LP/NLP branch-and-bound variant uses
        lazy master callbacks.
        Experimental SHOT-parity controls are accepted only with
        ``mip_nlp_profile="shot"`` and include ``tree_strategy``,
        ``cut_strategy``, ``objective_epigraph``, ``anti_epigraph``,
        ``nonlinear_partitioning``, ``quadratic_partitioning``,
        ``absolute_value_auxiliaries``, ``monomial_extraction``,
        ``signomial_extraction``, ``integer_bilinear_strategy``,
        ``integer_bilinear_max_bits``, ``quadratic_extraction``,
        ``direct_quadratic_routing``, ``rootsearch_strategy``,
        ``fixed_nlp_strategy``, ``solution_pool_capacity``,
        ``hyperplane_max_per_iter``, ``hyperplane_selection_factor``,
        ``relaxation_phase``, ``mip_solution_limit_strategy``,
        ``convex_bounding``, ``master_repair``, and ``reduction_cuts``.
        MIP-NLP runs attach a structured ``result.mip_nlp_trace`` payload.
        For ``mip_nlp_method="goa"``, convexity-certified MINLPs use OA's
        valid master bounds and other models use AMP/global relaxations.
        AMP options such as ``rel_gap``, ``abs_tol``, ``max_iter``,
        ``n_init_partitions``, ``partition_method``, ``milp_time_limit``,
        ``milp_gap_tolerance``, ``presolve_bt``, and
        ``convhull_formulation`` may also be passed as top-level aliases;
        AMP-only options apply only on the nonconvex AMP path and are ignored
        with a warning when GOA automatically hands a convexity-certified model
        to OA.
        Supported ``add_regularization`` values are ``"level_L1"``,
        ``"level_L2"``, ``"level_L_infinity"``, ``"grad_lag"``,
        ``"hess_lag"``, ``"hess_only_lag"``, and ``"sqp_lag"``.
        Supported ``init_strategy`` values are ``"rNLP"``,
        ``"initial_binary"``, ``"max_binary"``, and ``"fp"``. The
        ``mip_nlp_method`` selector determines the effective ``ecp_mode`` and
        cannot be overridden by ``mip_nlp_options``; a conflicting top-level
        ``ecp_mode`` and explicit ``mip_nlp_method`` raises ``ValueError``.

    Returns
    -------
    SolveResult
        Contains solution values, objective, gap, node count, and
        per-layer profiling times (Rust, JAX, Python).
    """
    # --- Enforce float64 precision ---
    # JAX defaults to float32 unless JAX_ENABLE_X64=1 is set *before* importing
    # JAX.  ``discopt/__init__.py`` sets that env var at import time, so x64 is
    # guaranteed whenever JAX is first imported through discopt.  We only need to
    # *warn* about the pathological case where the user imported JAX themselves
    # (in float32) before discopt — which means JAX is already in ``sys.modules``.
    # Gating on that avoids forcing a JAX/XLA import (and its cold-start cost) on
    # pure LP/MILP/MIQP solves that never touch JAX.
    import sys

    if "jax" in sys.modules:
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

    # C-18: the "midpoint" mode returned the McCormick underestimator's VALUE at
    # the box midpoint, u(mid), and fed it into the node lower bound as if it were
    # a bound. But u(mid) <= f(mid) does NOT imply u(mid) <= min_box f: for x**2
    # on [1, 3] the mode returns 3.0 while the true box minimum is 1.0, so it can
    # fathom the node holding the true optimum -> false "optimal". The only sound
    # cheap way to turn the convex underestimator into a bound is to *minimize* it
    # over the box, which is exactly what mccormick_bounds="nlp" already does.
    # Rather than ship a non-bound that claims to be a bound, the mode is removed
    # and rejected loudly. Use "lp" for a valid global spatial bound on nonconvex
    # models, or "nlp" for the convex-model relaxation bound. See
    # docs/dev/correctness-issues.md (C-18).
    _valid_mccormick_bounds = {"auto", "nlp", "lp", "none"}
    if mccormick_bounds not in _valid_mccormick_bounds:
        if mccormick_bounds == "midpoint":
            raise ValueError(
                "mccormick_bounds='midpoint' has been removed (correctness issue "
                "C-18): it evaluated the convex underestimator at the box midpoint "
                "and returned that value as a lower bound, but u(midpoint) is NOT a "
                "valid lower bound on the box minimum (e.g. for x**2 on [1, 3] it "
                "returns 3.0 while the true minimum is 1.0), so it could certify a "
                "wrong optimum. Use mccormick_bounds='lp' for a valid global "
                "spatial bound on nonconvex models with continuous variables, or "
                "'nlp' for the convex-model relaxation bound."
            )
        raise ValueError(
            f"Unknown mccormick_bounds={mccormick_bounds!r}. Choose one of "
            f"{sorted(_valid_mccormick_bounds)}."
        )

    # Root PSD cut-pool levers: resolve the explicit kwargs against the env-var
    # defaults (kwarg wins). Threading them here — instead of reading frozen
    # module constants — means they can be set after ``import discopt`` and can
    # differ per Model/per solve, while ``DISCOPT_ROOT_CUT_ROUNDS`` /
    # ``DISCOPT_ROOT_CUT_MAX`` remain as deprecated process-wide defaults.
    _root_cut_rounds = (
        _ROOT_CUT_POOL_ROUNDS_ENV if root_cut_rounds is None else int(root_cut_rounds)
    )
    _root_cut_max = _ROOT_CUT_POOL_MAX_ENV if root_cut_max is None else int(root_cut_max)
    if _root_cut_rounds < 0:
        raise ValueError(f"root_cut_rounds must be >= 0, got {_root_cut_rounds}")
    if _root_cut_max < 1:
        raise ValueError(f"root_cut_max must be >= 1, got {_root_cut_max}")

    # Anchor the whole-solve clock HERE, before any reformulation / presolve /
    # relaxation-build work — not at the B&B loop entry below. Preprocessing
    # (root presolve, OBBT, relaxation build) can take tens of seconds on large
    # models; counting it from a clock that only starts at the loop let a
    # ``solve(time_limit=N)`` overrun to several×N. ``t_start`` is set to this
    # anchor at the loop entry so every downstream deadline check
    # (``now - t_start > time_limit``) and the reported wall time include the
    # preprocessing, and ``_remaining_budget()`` lets each preprocessing phase
    # clamp its own internal budget to the time actually left.
    _solve_t0 = time.perf_counter()

    def _remaining_budget() -> float:
        return max(0.0, float(time_limit) - (time.perf_counter() - _solve_t0))

    def _deadline_exhausted(floor: float = _DEADLINE_NODE_FLOOR_S) -> bool:
        """True once the whole-solve budget is spent past a small floor.

        Optional root-setup phases (presolve, reverse-AD, eigenvalue diagnostic,
        root OBBT, root cut pool) check this to avoid *launching* further
        bound-strengthening work once ``time_limit`` is already blown — the fix
        for issue #654's "budget effectively ignored" symptom. This never
        truncates an in-flight bound-producing op (that would drop a valid bound;
        docs/dev/baron-gap-plan.md §8): each phase runs to completion once started
        and only new phases are skipped, so the overrun is bounded to at most one
        in-flight uninterruptible op and the wall scales with ``time_limit``.
        Skipping a phase only ever *weakens* the (still-valid) bound — a larger
        box or fewer cuts — never makes it unsound.
        """
        return _remaining_budget() <= floor

    # Reset the per-solve convexity-classification memo (a previous solve, or an
    # IIS feasibility probe, may have cached a verdict for a different constraint
    # set), and budget classification to a fraction of the time limit so it
    # cannot, on its own, overrun ``time_limit`` (issue: tens of seconds of
    # eigenvalue work on large quadratic models before search even starts).
    model._convexity_classification_cache = None
    _convexity_time_budget = min(max(0.2 * float(time_limit), 0.5), 20.0)
    model._convexity_time_budget = _convexity_time_budget
    # #654: absolute solve deadline, anchored at ``_solve_t0`` (before all
    # preprocessing), so an uninterruptible pre-B&B phase can decline to *start*
    # new work once the budget is spent rather than blow past ``time_limit``. Read
    # by ``IncrementalMcCormickLP._validate`` to bound its row-for-row
    # cold-vs-patched self-check — tens of seconds on large factorable models
    # (sonet*, qap): the dominant #654 overrun — leaving ``ok=False`` so the engine
    # falls back to the sound per-node cold build. Never truncates an in-flight op.
    model._solve_deadline = _solve_t0 + float(time_limit)

    # Opt-in LP-node spatial branch-and-cut engine (discopt#280) for pure-integer
    # product MINLPs. The default NLP-per-node spatial path freezes its dual bound
    # on dense integer-bilinear problems (e.g. nvs17: stuck at -65842, times out);
    # this engine solves a pure McCormick LP per node (incremental + warm-started),
    # branches on integers/products and runs a feasibility-pump primal, closing
    # nvs17 to proven optimality. Opt-in via ``solve(lp_spatial=True)``; returns
    # ``None`` (falls through to the default path, no behavior change) for any model
    # out of its scope (non-pure-integer, maximize, unbounded box) or on any error.
    if kwargs.get("lp_spatial", False):
        try:
            from discopt._jax.lp_spatial_bb import solve_lp_spatial_bb

            _lps = solve_lp_spatial_bb(
                model,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                max_nodes=max_nodes,
                root_cut_rounds=int(kwargs.get("lp_spatial_cut_rounds", 0)),
            )
        except Exception as _lps_exc:  # pragma: no cover - defensive
            logger.debug("lp_spatial engine failed, falling back: %s", _lps_exc)
            _lps = None
        if _lps is not None:
            return SolveResult(
                status=_lps.status,
                objective=_lps.objective,
                bound=_lps.bound,
                gap=_lps.gap,
                x=(None if _lps.x is None else _unpack_solution(model, np.asarray(_lps.x))),
                wall_time=time.perf_counter() - _solve_t0,
                node_count=_lps.node_count,
                gap_certified=(_lps.status == "optimal"),
            )

    # --- Structure-cut presolve (auto-engaged square-difference-network cuts) ---
    # The symbolic cut recognizer auto-derives sound coupling cuts for the
    # square-difference (Weymouth gas/water) network structure directly from the
    # model graph, collapsing the otherwise-catastrophic McCormick relaxation gap
    # on those models (gas-network MINLP, issue #15: bound 1.0 vs optimum 3.0026 ->
    # bound = optimum; 2821 nodes / 182 s -> 5 nodes / ~3 s). Each injected cut is a
    # verified valid underestimator (sound), and the pass is a genuine no-op on a
    # model without the structure. It is auto-engaged but gated, because it runs
    # SymPy analysis: opt out with ``structure_cuts=False``; only when SymPy is
    # importable; only on small models (so ``model_to_sympy`` stays cheap); only
    # with budget remaining. Runs *before* the relaxation build so the tightened
    # model flows through the normal pipeline. Only the square-difference detector
    # is auto-engaged here — the broader ``inject_all_patterns`` battery stays
    # opt-in. Any failure is swallowed so the recognizer can never break a solve.
    if structure_cuts and model._objective is not None and _remaining_budget() > 1.0:
        _struct_size = sum(v.size for v in model._variables) + len(model._constraints)
        if _struct_size <= _STRUCTURE_CUTS_MAX_SIZE:
            try:
                from discopt._jax.symbolic.cut_recognizer import recognize_and_inject

                _n_struct_cuts = recognize_and_inject(model)
                if _n_struct_cuts:
                    logger.info(
                        "Structure-cut presolve: injected %d square-difference-network "
                        "coupling cut(s) (auto-engaged; opt out with structure_cuts=False)",
                        _n_struct_cuts,
                    )
            except ImportError:
                pass  # optional [sympy] extra not installed -> skip silently
            except Exception as _sc_exc:  # pragma: no cover - defensive
                logger.debug("structure-cut presolve skipped: %s", _sc_exc)

    # --- Solver-family dispatch ---
    _solver = solver if solver is not None else kwargs.pop("solver", None)
    # Recognised global-solver selectors: ``None`` (default branch-and-bound,
    # with the automatic GP fast path below), ``"amp"``, ``"gurobi"``,
    # ``"mip-nlp"``,
    # ``"gp"`` (force the GP log-space path), and ``"bb"`` (force classic
    # branch-and-bound, opting out of the automatic GP fast path). Reject
    # anything else rather than silently falling through to B&B.
    if _solver not in (None, "amp", "gurobi", "mip-nlp", "gp", "bb"):
        raise ValueError(
            f"Unknown solver={_solver!r}. Choose one of None, 'amp', 'gurobi', "
            "'mip-nlp', 'gp', 'bb'."
        )
    gurobi_options = kwargs.pop("gurobi_options", None) if _solver == "gurobi" else None

    # --- MIP-NLP decomposition solver family ---
    if _solver == "mip-nlp":
        import warnings

        from discopt.solvers.mip_nlp import solve_mip_nlp
        from discopt.solvers.mip_nlp_options import (
            FP_OPTION_KEYS,
            GOA_OPTION_KEYS,
            MIP_NLP_PROFILE_OPTION_KEYS,
            SHOT_OPTION_KEYS,
        )

        mip_nlp_method = kwargs.pop("mip_nlp_method", None)
        mip_nlp_options = kwargs.pop("mip_nlp_options", None)
        mip_nlp_kwargs: dict[str, Any] = {}
        for key in (
            "equality_relaxation",
            "ecp_mode",
            "feasibility_cuts",
            "init_strategy",
            "heuristic_nonconvex",
            "add_slack",
            "max_slack",
            "oa_penalty_factor",
            "OA_penalty_factor",
            "add_no_good_cuts",
            "integer_to_binary",
            "feasibility_norm",
            "add_regularization",
            "level_coef",
            "stalling_limit",
            "cycling_check",
            "milp_solver",
            "solution_pool",
            "num_solution_iteration",
            *MIP_NLP_PROFILE_OPTION_KEYS,
            *SHOT_OPTION_KEYS,
            *FP_OPTION_KEYS,
        ):
            if key in kwargs:
                mip_nlp_kwargs[key] = kwargs.pop(key)
        if mip_nlp_method is None:
            mip_nlp_method = "ecp" if bool(mip_nlp_kwargs.get("ecp_mode", False)) else "oa"

        mip_nlp_method_key = (
            mip_nlp_method.strip().lower().replace("-", "_")
            if isinstance(mip_nlp_method, str)
            else mip_nlp_method
        )
        if mip_nlp_method_key == "goa":
            for key in sorted(GOA_OPTION_KEYS):
                if key in kwargs:
                    mip_nlp_kwargs[key] = kwargs.pop(key)

        gdp_methods = {"big-m", "hull", "mbigm", "auto"}
        native_gdp_methods = {"loa"}
        if gdp_method == "oa":
            warnings.warn(
                "gdp_method='oa' is deprecated for selecting MINLP OA. Use "
                "solver='mip-nlp', mip_nlp_method='oa'. Interpreting gdp_method "
                "as 'big-m' for GDP reformulation in this solve.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_gdp_method = "big-m"
        elif gdp_method in gdp_methods:
            resolved_gdp_method = gdp_method
        elif gdp_method in native_gdp_methods:
            allowed = ", ".join(sorted(gdp_methods))
            raise ValueError(
                f"gdp_method={gdp_method!r} conflicts with solver='mip-nlp'. "
                "Use mip_nlp_method to select the MIP-NLP algorithm and reserve "
                f"gdp_method for GDP reformulation methods: {allowed}."
            )
        else:
            allowed = ", ".join(sorted(gdp_methods | {"oa"}))
            raise ValueError(
                f"Unknown gdp_method={gdp_method!r} for solver='mip-nlp'. Choose one of: {allowed}."
            )

        ignored_mip_nlp_options = []

        def _note_ignored_mip_nlp(name: str, should_warn: bool) -> None:
            if should_warn:
                ignored_mip_nlp_options.append(name)

        _note_ignored_mip_nlp("threads", threads != 1)
        _note_ignored_mip_nlp("deterministic", deterministic is not True)
        _note_ignored_mip_nlp("batch_size", batch_size != 16)
        _note_ignored_mip_nlp("strategy", strategy != "best_first")
        _note_ignored_mip_nlp("ipopt_options", ipopt_options is not None)
        _note_ignored_mip_nlp("sparse", sparse is not None)
        _note_ignored_mip_nlp("cutting_planes", cutting_planes is not False)
        _note_ignored_mip_nlp("psd_cuts", psd_cuts is not False)
        _note_ignored_mip_nlp("rlt_cuts", rlt_cuts is not False)
        _note_ignored_mip_nlp("rlt", rlt != "auto")
        _note_ignored_mip_nlp("cuts", cuts != "auto")
        _note_ignored_mip_nlp("partitions", partitions != 0)
        _note_ignored_mip_nlp("branching_policy", branching_policy != "fractional")
        _note_ignored_mip_nlp("use_learned_relaxations", use_learned_relaxations is not False)
        _note_ignored_mip_nlp("mccormick_bounds", mccormick_bounds != "auto")
        _note_ignored_mip_nlp("decomposition", decomposition is not None)
        _note_ignored_mip_nlp("lagrangian_bound", lagrangian_bound is not False)
        _note_ignored_mip_nlp("lagrangian_frequency", lagrangian_frequency != 1)
        _note_ignored_mip_nlp("skip_convex_check", skip_convex_check is not False)
        _note_ignored_mip_nlp("nlp_bb", nlp_bb is not None)
        _note_ignored_mip_nlp("lazy_constraints", lazy_constraints is not None)
        _note_ignored_mip_nlp("incumbent_callback", incumbent_callback is not None)
        _note_ignored_mip_nlp("node_callback", node_callback is not None)
        _note_ignored_mip_nlp("presolve", presolve is not True)
        _note_ignored_mip_nlp("presolve_polynomial", presolve_polynomial is not False)
        _note_ignored_mip_nlp("presolve_reverse_ad", presolve_reverse_ad is not False)
        # PF1 (#632): default is now 1; warn only when the user overrode it, so a
        # default solve on the mip-nlp path stays quiet.
        _note_ignored_mip_nlp("in_tree_presolve_stride", in_tree_presolve_stride != 1)
        _note_ignored_mip_nlp("eigenvalue_root_bound", eigenvalue_root_bound is not False)
        _note_ignored_mip_nlp("relaxation_arithmetic", relaxation_arithmetic != "mccormick")
        _note_ignored_mip_nlp("subnlp_enabled", subnlp_enabled is not True)
        _note_ignored_mip_nlp("subnlp_backend", subnlp_backend != "auto")
        _note_ignored_mip_nlp("subnlp_frequency", subnlp_frequency != 20)
        _note_ignored_mip_nlp("subnlp_max_calls", subnlp_max_calls != 200)
        _note_ignored_mip_nlp("subnlp_options", subnlp_options is not None)
        _note_ignored_mip_nlp("root_cut_rounds", root_cut_rounds is not None)
        _note_ignored_mip_nlp("root_cut_max", root_cut_max is not None)
        if kwargs:
            ignored_mip_nlp_options.extend(sorted(kwargs))
        if ignored_mip_nlp_options:
            warnings.warn(
                "MIP-NLP solver ignores solve_model options: "
                + ", ".join(sorted(dict.fromkeys(ignored_mip_nlp_options))),
                stacklevel=2,
            )

        from discopt._jax.gdp_reformulate import reformulate_gdp

        model = reformulate_gdp(model, method=resolved_gdp_method)

        return solve_mip_nlp(
            model,
            method=mip_nlp_method,
            mip_nlp_options=mip_nlp_options,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_nodes,
            nlp_solver=nlp_solver,
            initial_point=initial_point,
            **mip_nlp_kwargs,
        )

    # --- AMP (Adaptive Multivariate Partitioning) global solver ---
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
        _note_ignored("nlp_bb", nlp_bb is not None)
        _note_ignored("lazy_constraints", lazy_constraints is not None)
        _note_ignored("incumbent_callback", incumbent_callback is not None)
        _note_ignored("node_callback", node_callback is not None)
        amp_gdp_methods = {"big-m", "hull", "mbigm", "auto"}
        amp_gdp_method = gdp_method if gdp_method in amp_gdp_methods else "big-m"
        _note_ignored("gdp_method", gdp_method not in amp_gdp_methods)
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

        from discopt._jax.gdp_reformulate import reformulate_gdp

        model = reformulate_gdp(
            model,
            method=amp_gdp_method,
            respect_disjunction_methods=False,
        )

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

    # --- Benders / Lagrangian decomposition: opt-in, structure-exploiting ---
    if decomposition is not None:
        if decomposition == "benders":
            from discopt.decomposition.benders import solve_benders

            return solve_benders(
                model,
                structure=decomposition_structure,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                max_iterations=max_nodes,
                nlp_solver=nlp_solver,
            )
        if decomposition == "lagrangian":
            from discopt.decomposition.lagrangian import solve_lagrangian

            return solve_lagrangian(
                model,
                structure=decomposition_structure,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                max_iterations=max_nodes,
                nlp_solver=nlp_solver,
            )
        if decomposition == "auto":
            # Consult the advisor, log its reasoning, and dispatch its
            # recommendation. A NONE / no-benefit recommendation falls through to
            # the normal monolithic solve below (W1).
            from discopt.decomposition import analyze_decomposition
            from discopt.decomposition.advisor.types import MethodKind
            from discopt.decomposition.learning import record_outcome
            from discopt.decomposition.learning.store import RecordStore

            _store_path = os.environ.get("DISCOPT_DECOMP_STORE")
            _store = (
                RecordStore(path=_store_path) if (_store_path or record_decomposition) else None
            )
            advisor = analyze_decomposition(model, store=_store)
            expl = advisor.recommendation()
            logger.info("decomposition='auto' recommendation:\n%s", expl.render("text"))
            if expl.recommendation is not MethodKind.NONE:
                decomposed = advisor.decompose()
                result = decomposed.solve(
                    time_limit=time_limit,
                    gap_tolerance=gap_tolerance,
                    max_iterations=max_nodes,
                    nlp_solver=nlp_solver,
                )
                if _store is not None:
                    try:
                        record_outcome(advisor, _store, chosen=decomposed.method)
                    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort
                        logger.debug("decomposition telemetry failed: %s", exc)
                assert isinstance(result, SolveResult)
                return result
            # else: fall through to the monolithic solve path.
        elif decomposition not in ("benders", "lagrangian"):
            raise ValueError(
                f"Unknown decomposition={decomposition!r}; choose 'benders', "
                "'lagrangian', or 'auto'."
            )

    # --- Deprecated compatibility route: OA is a MINLP solver strategy, not a GDP method. ---
    if gdp_method == "oa":
        import warnings

        from discopt._jax.gdp_reformulate import reformulate_gdp
        from discopt.solvers.mip_nlp import solve_mip_nlp
        from discopt.solvers.mip_nlp_options import (
            FP_OPTION_KEYS,
            MIP_NLP_PROFILE_OPTION_KEYS,
            SHOT_OPTION_KEYS,
        )

        warnings.warn(
            "gdp_method='oa' is deprecated for selecting MINLP OA. Use "
            "solver='mip-nlp', mip_nlp_method='oa' instead. Interpreting "
            "gdp_method as 'big-m' for GDP reformulation in this solve.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Extract OA-specific kwargs that solve_model doesn't understand
        mip_nlp_kwargs = {}
        for key in (
            "equality_relaxation",
            "ecp_mode",
            "feasibility_cuts",
            "init_strategy",
            "heuristic_nonconvex",
            "add_slack",
            "max_slack",
            "oa_penalty_factor",
            "OA_penalty_factor",
            "add_no_good_cuts",
            "feasibility_norm",
            "add_regularization",
            "level_coef",
            "stalling_limit",
            "cycling_check",
            "milp_solver",
            "solution_pool",
            "num_solution_iteration",
            *MIP_NLP_PROFILE_OPTION_KEYS,
            *SHOT_OPTION_KEYS,
            *FP_OPTION_KEYS,
        ):
            if key in kwargs:
                mip_nlp_kwargs[key] = kwargs.pop(key)

        model = reformulate_gdp(model, method="big-m")

        return solve_mip_nlp(
            model,
            method="ecp" if mip_nlp_kwargs.get("ecp_mode", False) else "oa",
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_nodes,
            nlp_solver=nlp_solver,
            **mip_nlp_kwargs,
        )

    # --- LOA decomposition: intercept before GDP reformulation ---
    if gdp_method == "loa":
        from discopt.solvers.gdpopt_loa import solve_gdpopt_loa

        loa_kwargs = {}
        if "milp_solver" in kwargs:
            loa_kwargs["milp_solver"] = kwargs.pop("milp_solver")

        return solve_gdpopt_loa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_nodes,
            nlp_solver=nlp_solver,
            **loa_kwargs,
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
    # A model with no continuous decision variable is pure-discrete: when it
    # routes onto the McCormick-LP path and the relaxer turns out unusable, its
    # fallback must be the sound alphaBB ("none"), never the integer-unsound
    # nonconvex "nlp" objective bound. Defined here (not just in the auto block)
    # so the lp-path fallbacks see it even under an explicit mccormick_bounds="lp".
    _pure_discrete = not _origin_has_continuous_var

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
        from discopt._jax.nonlinear_bound_tightening import (
            FunctionDomainBoundRule,
            PeriodicVariableBoundRule,
            tighten_nonlinear_bounds,
        )

        # Two domain/period reductions that hand the spatial+NLP paths a valid
        # box on otherwise-free nonlinear variables: restrict sin/cos-only angles
        # to one period, and clamp log/sqrt arguments to their natural domain so
        # the local NLP never wanders into the undefined region (issue #265's
        # false-infeasible from a free log argument).
        _per_lb, _per_ub, _per_stats = tighten_nonlinear_bounds(
            model,
            _origin_lb_chk,
            _origin_ub_chk,
            rules=(PeriodicVariableBoundRule(), FunctionDomainBoundRule()),
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

    # Functionally-dependent continuous variables (each pinned by a nonlinear
    # defining equality x_i = f(others)). Detected on the ORIGINAL model, before
    # the factorable reform rewrites those equalities into product form (which
    # would hide the affine isolation of the output). The dependency is
    # invariant under that rewrite, so the names captured here remain correct to
    # deprioritize in the lifted/solved model. Spatial branching deprioritizes
    # these — branching on a pinned output wastes effort the McCormick gap on
    # the independent inputs would not (welded-beam / nvs05). Names are mapped
    # to flat columns of the live model at tree setup; deprioritization carries
    # a completeness-preserving fallback, so an empty or imperfect set is safe.
    _dependent_var_names: set = set()
    try:
        from discopt._jax.dependent_vars import find_functionally_dependent_names

        _dependent_var_names = find_functionally_dependent_names(model)
    except Exception as _dep_exc:  # pragma: no cover - defensive
        logger.debug("functional-dependency detection skipped: %s", _dep_exc)

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
            # factorable_reformulate builds a FRESH model object that does not
            # carry the convexity-classification budget attribute. Without this
            # the dispatch classify below reads the 15 s default instead of the
            # intended fraction of ``time_limit`` and can overrun a tight budget
            # on the (larger) lifted model — heatexch_gen3: 12 s classify under a
            # 15 s solve budget. Re-assert the budget (and clear any stale cache).
            model._convexity_classification_cache = None
            model._convexity_time_budget = _convexity_time_budget
            model._solve_deadline = _solve_t0 + float(time_limit)  # #654 (see above)
        # A *convex* model with a clearable denominator is deliberately left
        # untouched here: many such divisions (e.g. the rotated-SOC ``x**2/z``)
        # are solved exactly by the convex NLP fast path, and clearing would
        # needlessly destroy that structure. Only if the convex NLP later *fails*
        # to certify do we clear and fall back to spatial B&B (see the convex
        # fast-path block below).

    # --- Integer-bilinear exact reformulation ---
    # When a bilinear term ``x_i*x_j`` has an integer (declared or implied)
    # factor, binary-expand it and big-M-linearize the resulting binary*var
    # products, turning the bilinear MINLP into an *equivalent pure MILP* whose
    # relaxation is exact. discopt's single McCormick envelope over a wide integer
    # box is loose (its integer optimum can sit below the true optimum — the
    # ``ex126x`` trim-loss family: 19.1 vs 19.6); the exact linearization closes
    # that gap and routes through the MILP branch-and-bound (cover/clique/Gomory/
    # MIR cuts). Value-preserving and gated to integer-bilinear models, so it is a
    # no-op everywhere else.
    try:
        from discopt._jax.integer_product_reform import (
            has_nonconvex_integer_bilinear,
            reformulate_integer_bilinear,
        )

        # Gate on a *distinct-variable* integer-bilinear term ``x_i*x_j`` (i != j).
        # Its Hessian is indefinite, so this is a cheap, sound *nonconvexity*
        # witness — and exactly the loose-relaxation structure the pass fixes.
        # Convex MIQPs (only ``x**2`` squares, PSD curvature) have no such term
        # and are left to the convex QP/NLP fast paths: binary-expanding them
        # would merely bloat the model and divert it off those paths (which broke
        # the MIQP-batch certification path). This witness is far cheaper than a
        # full convexity classification (~6s on ex1263), so the common path and
        # the reformulated path both stay fast.
        if has_nonconvex_integer_bilinear(model):
            _ipx = reformulate_integer_bilinear(model)
            # Adopt the reformulation ONLY when it eliminates *all* nonlinearity,
            # i.e. yields an equivalent pure MILP. If other nonlinear terms remain
            # (e.g. the transcendentals in gear), the model would still go through
            # the spatial path — now merely carrying the extra big-M variables for
            # no benefit — so the reformulation is discarded and the original model
            # is solved unchanged. This keeps the pass a strict improvement.
            from discopt._jax.problem_classifier import ProblemClass, classify_problem
            from discopt._jax.term_classifier import classify_nonlinear_terms

            # Require the reformulation to be a *genuinely* pure MILP: classify_problem
            # is largely extract_lp_data-based and can report MILP while nonlinear
            # terms remain (which that linear projection silently drops). Confirm with
            # the DAG-walking term classifier, else adopting + routing to the MILP
            # engine would solve a lossy linear projection — falsely unbounded when a
            # dropped nonlinear constraint was what bounded an open variable
            # (carton7, issue #286).
            _ipx_nl = classify_nonlinear_terms(_ipx) if _ipx is not model else None
            _ipx_pure_milp = _ipx_nl is not None and not (
                _ipx_nl.bilinear
                or _ipx_nl.trilinear
                or _ipx_nl.multilinear
                or _ipx_nl.monomial
                or _ipx_nl.fractional_power
                or _ipx_nl.bilinear_with_fp
                or _ipx_nl.ratio_of_products
                or _ipx_nl.general_nl
            )
            if _ipx_pure_milp and classify_problem(_ipx) == ProblemClass.MILP:
                model = _ipx
                model._convexity_classification_cache = None
                model._convexity_time_budget = _convexity_time_budget
                model._solve_deadline = _solve_t0 + float(time_limit)  # #654 (see above)
                # The reformulated big-M MILP is best handled by a real MILP
                # engine. discopt's FBBT root presolve is both redundant (the MILP
                # engines presolve internally) and pathologically slow on the
                # lifted big-M structure (ex1263: ~10s presolve vs ~1s solve), so
                # skip it; and route off the self-hosted IPM B&B (no MILP cuts,
                # ~60s) onto the monolithic Rust simplex MILP engine (~1s,
                # pure-Rust), which falls back to HiGHS / the IPM path if the
                # simplex binding is unavailable.
                presolve = False
                if nlp_solver == "pounce":
                    # cert:P3.1c cut-reachability experiment (default-OFF): keep the
                    # reformulated MILP on the self-hosted _solve_milp_bb path (which
                    # has the root cut loop) instead of rerouting to the cut-less
                    # monolithic Rust _solve_milp_simplex engine, so the aggregation
                    # c-MIR / Gomory / cover separators can actually fire on this
                    # class. Math-neutral when the flag is unset. See
                    # _p3_force_cut_path_enabled / certification-gap-plan.md §7.
                    if not _p3_force_cut_path_enabled():
                        nlp_solver = "simplex"
    except Exception as _ipx_exc:  # pragma: no cover - defensive
        logger.debug("integer-bilinear reformulation skipped: %s", _ipx_exc)

    # --- Build Rust model representation for FBBT ---
    global _IN_TREE_PRESOLVE_GLOBAL_CALLS
    _IN_TREE_PRESOLVE_GLOBAL_CALLS = 0  # PF1 telemetry reset (issue #632)
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
    # Skipped if the budget is already blown (e.g. a large-model reformulation
    # above overran ``time_limit``): presolve only tightens bounds, so declining
    # it leaves a looser-but-valid box and lets the wall track ``time_limit``
    # (#654). ``propagate_bounds_to_model`` is a no-op when skipped.
    if _model_repr is not None and presolve and not _deadline_exhausted():
        try:
            from discopt._jax.presolve_pipeline import (
                propagate_bounds_to_model,
                run_root_presolve,
            )

            # Cap presolve to a fraction of the time limit, further clamped to
            # the time actually left. Without a cap the Rust orchestrator runs
            # to its iteration cap (16 sweeps, ~1s each on large models), which
            # on its own can overrun a tight ``time_limit`` before search starts
            # (e.g. contvar: 17.5s of presolve under a 15s budget). The Rust
            # side honours ``time_limit_ms`` between sweeps, so the overrun is
            # bounded by a single sweep.
            _presolve_budget_s = min(
                min(max(0.25 * float(time_limit), 2.0), 30.0), _remaining_budget()
            )
            _model_repr, _presolve_stats = run_root_presolve(
                _model_repr,
                eliminate=True,
                polynomial=presolve_polynomial,
                fbbt=True,
                time_limit_ms=int(_presolve_budget_s * 1000),
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
    # can be slow on very large models. Skipped once the budget is blown (#654):
    # it only tightens bounds, so declining it leaves a valid looser box.
    if presolve and presolve_reverse_ad and not _deadline_exhausted():
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
    # diagnostic at the root; does not affect the B&B tree directly. Skipped
    # once the budget is blown (#654): purely diagnostic, safe to omit.
    if eigenvalue_root_bound and not _deadline_exhausted():
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

    # Anchor at the whole-solve clock (set before reformulation/presolve above)
    # so every ``now - t_start > time_limit`` deadline check and the reported
    # wall time account for the preprocessing already spent.
    t_start = _solve_t0
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
        # INT-1 (#413): the NLP-BB path cannot honor a callback that REJECTS an
        # integer-feasible point. It has no per-node cut-application mechanism
        # (the augmented-evaluator cut pool is spatial-B&B-only) and its primal
        # heuristics (feasibility pump, RENS, diving, sub-NLP) inject incumbents
        # directly via ``tree.inject_incumbent`` WITHOUT consulting the callback
        # — so a lazy constraint's cut or an incumbent-callback's ``False`` would
        # be silently dropped and the excluded point accepted as the optimum.
        # Refuse loudly rather than return a wrong answer (CLAUDE.md §3); these
        # callbacks are supported on the spatial-B&B path (``nlp_bb=False``).
        if lazy_constraints is not None or incumbent_callback is not None:
            _rejected = []
            if lazy_constraints is not None:
                _rejected.append("lazy_constraints")
            if incumbent_callback is not None:
                _rejected.append("incumbent_callback")
            raise ValueError(
                f"{' and '.join(_rejected)} cannot be used with nlp_bb=True: the "
                "NLP-BB path cannot enforce a callback that rejects an integer-"
                "feasible point (its primal heuristics inject incumbents without "
                "consulting the callback, and it has no per-node cut application). "
                "Use nlp_bb=False (spatial branch-and-bound) for these callbacks, "
                "or omit nlp_bb to auto-select a path that honors them."
            )
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
            rens_enabled=rens,
            _lns_enabled=_lns_enabled,
        )

    # --- Problem classification: dispatch LP/QP to specialized solvers ---
    try:
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        problem_class = classify_problem(model)
    except Exception as e:
        logger.debug("Problem classification failed: %s", e)
        problem_class = None

    if _solver == "gurobi":
        if problem_class is None:
            raise NotImplementedError(
                "solver='gurobi' requires problem classification and currently "
                "supports LP, MILP, QP, MIQP, QCP, QCQP, MIQCP, and MIQCQP models only."
            )
        if problem_class == ProblemClass.LP:
            return _solve_lp_gurobi(model, t_start, time_limit, threads, gurobi_options)
        if problem_class == ProblemClass.MILP:
            return _solve_milp_gurobi(
                model, t_start, time_limit, gap_tolerance, threads, gurobi_options
            )
        if problem_class == ProblemClass.QP:
            return _solve_qp_gurobi(
                model, t_start, time_limit, gap_tolerance, threads, gurobi_options
            )
        if problem_class == ProblemClass.MIQP:
            return _solve_qp_gurobi(
                model, t_start, time_limit, gap_tolerance, threads, gurobi_options
            )
        if problem_class in (
            ProblemClass.QCP,
            ProblemClass.QCQP,
            ProblemClass.MIQCP,
            ProblemClass.MIQCQP,
        ):
            return _solve_qcp_gurobi(
                model, t_start, time_limit, gap_tolerance, threads, gurobi_options
            )
        raise NotImplementedError(
            f"solver='gurobi' supports LP, MILP, QP, MIQP, QCP, QCQP, MIQCP, "
            f"and MIQCQP models only; "
            f"classified this model as {problem_class.value!r}."
        )

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
                if lagrangian_bound:
                    logger.warning(
                        "lagrangian_bound is ignored with nlp_solver='simplex' (the "
                        "monolithic Rust MILP engine has no per-node hook); use the "
                        "default nlp_solver='pounce' MILP path to enable it."
                    )
                _simplex_res = _solve_milp_simplex(
                    model, time_limit, gap_tolerance, max_nodes, t_start
                )
                if _simplex_res is not None:
                    return _simplex_res
            # POUNCE-only mode (nlp_solver="pounce", the universal default)
            # routes to the self-hosted B&B and bypasses HiGHS entirely. An
            # interior-point method is the wrong tool for the *linear* node LPs
            # (no warm-start across branches, interior smearing, no basis), so
            # the per-node engine defaults to the exact-vertex warm-started
            # **simplex** (node_engine="simplex"); it degrades to the POUNCE IPM
            # node path inside _solve_milp_bb if the simplex binding is absent.
            # The B&B itself is sound, runs the continuous-repair root dive for
            # an early incumbent, recovers stalled nodes, and reduced-cost-fixes.
            _pounce_only = nlp_solver == "pounce"
            return _solve_milp_bb(
                model,
                time_limit,
                gap_tolerance,
                batch_size,
                strategy,
                max_nodes,
                t_start,
                prefer_pounce=_pounce_only,
                # Node LP relaxations are linear — always solve them with the
                # structured engine (exact-vertex Rust simplex, degrading to
                # POUNCE), regardless of nlp_solver. nlp_solver governs only the
                # NLP subproblem solver. The JAX LP-IPM node path was retired (#370).
                node_engine="simplex",
                lagrangian_bound=lagrangian_bound,
                lagrangian_frequency=lagrangian_frequency,
            )
        elif problem_class == ProblemClass.MIQP:
            # A convex MIQP may use the convex MIQP B&B; a NONCONVEX one must
            # NOT. `_solve_miqp_bb` assumes a convex node QP (a convex relaxation
            # solved to global optimality), so
            # on an indefinite or concave-maximize objective they return a local
            # stationary point and certify it as global — a false-optimal (e.g.
            # `max x**2` over integer [-3,3] returned 0 instead of 9). The
            # pure-continuous QP path already guards this (it forces the spatial
            # path on an indefinite QP); MIQP did not. Mirror it: classify
            # convexity (eigenvalue-sound, sense-aware, memoized) and use the
            # convex solvers only when the model is KNOWN convex. Otherwise fall
            # through (no return) to the sound spatial McCormick Branch-and-Bound
            # below, which branches the integers and bounds each node with a valid
            # outer relaxation.
            (
                _root_convexity_known,
                _root_is_convex,
                _root_constraint_mask,
            ) = _classify_model_convexity(model, failure_label="MIQP convexity detection failed")
            if _root_convexity_known and _root_is_convex:
                # Convex MIQP via discopt's own self-hosted B&B (POUNCE node QP
                # relaxations) — HiGHS-free by design (issue #359 / pure-Rust
                # goal). The convex node QP is solved to global optimality, so the
                # B&B bound is valid.
                return _solve_miqp_bb(
                    model,
                    time_limit,
                    gap_tolerance,
                    batch_size,
                    strategy,
                    max_nodes,
                    t_start,
                    prefer_pounce=True,
                )
            logger.info(
                "Nonconvex MIQP detected — routing to spatial Branch-and-Bound "
                "(convex MIQP solvers would certify a local stationary point)"
            )
            # Fall through to the spatial/McCormick path below.

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
        # The convex NLP may fail to certify for two reasons handled here.
        from discopt._jax.factorable_reform import has_clearable_denominator

        if _model_contains_nonsmooth_node(model):
            # (1) Non-smooth objective/constraints (abs/min/max): a smooth
            # gradient-based solver oscillates at the kink (e.g. min |x| over
            # [-1, 1] returns ~0.99 with iteration_limit). The spatial McCormick
            # B&B has an *exact* piecewise relaxation for these nodes and
            # certifies them, so fall back to it rather than returning the
            # unconverged point.
            logger.info(
                "Convex NLP did not certify (status=%s) on a non-smooth model; "
                "retrying via spatial B&B (exact piecewise relaxation)",
                result.status,
            )
            _pure_continuous_is_convex = False
            _root_is_convex = False
            _pure_continuous_force_spatial = True
            # Fall through to the spatial B&B below.
        elif has_clearable_denominator(model):
            # (2) An ill-conditioned division whose denominator reaches toward
            # zero (like st_e17's 0.2458*x0**2/x1 with x1 down to 1e-5). If the
            # model has a sign-definite non-constant denominator — which the
            # relaxation otherwise drops to general_nl — clear it (exact,
            # value-preserving) and fall back to the sound spatial B&B, which can
            # certify it. Only clearing is applied (no convexity-destroying
            # mixed-product lift).
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
            # Nothing clearable and the model is smooth: return the NLP result
            # unchanged (no regression).
            return result

    # --- Pure continuous: solve directly only when spatial search was not requested ---
    if (
        _pure_continuous
        and not _pure_continuous_force_spatial
        and (skip_convex_check or not _pure_continuous_convexity_known)
    ):
        _cont_result = _solve_continuous(
            model,
            time_limit,
            ipopt_options,
            t_start,
            nlp_solver,
            initial_point=initial_point,
        )
        # A local NLP on a model we could NOT certify convex (classification
        # failed/timed out, leaving convexity unknown) is best-effort only. If it
        # errors outright (e.g. the NLP backend failed on a large/ill-conditioned
        # problem), don't surface a bare status="error": fall through to the sound
        # spatial McCormick B&B below, which returns a valid bound / feasible point
        # under the time limit. The user-requested skip_convex_check path keeps its
        # local-only result on any non-error status. (issue #266)
        if _cont_result.status != "error":
            # C-33/SC-1 (P0, DEFAULT path): reaching this fallback means convexity
            # was NOT established — either the user set skip_convex_check, or the
            # classifier abstained (`not _pure_continuous_convexity_known`). The
            # KNOWN-convex case already returned via the convex fast path above
            # (line ~3745), so a *convex* certificate can never be lost here.
            # On a nonconvex model the single NLP finds only a LOCAL optimum, which
            # is not global — emitting it with gap_certified=True (and bound set to
            # the local objective) is a FALSE optimality certificate (a nonconvex
            # double-well returned obj=-50 certified while the true min was -78).
            # Withhold the certificate: keep the feasible incumbent (objective, x)
            # but strip the fabricated dual bound/gap and mark it uncertified. Do
            # NOT weaken this into "trust the NLP" — refuse to certify (CLAUDE.md
            # §1, §3). Genuine infeasibility from _solve_continuous is a rigorous
            # nonlinear-tightening / NLP-infeasibility claim, not a gap, so leave it.
            if _cont_result.status != "infeasible":
                _cont_result.gap_certified = False
                _cont_result.bound = None
                _cont_result.root_bound = None
                _cont_result.gap = None
                _cont_result.root_gap = None
            return _cont_result
        logger.info(
            "Local NLP on convexity-unknown continuous model returned error; "
            "falling back to spatial Branch-and-Bound (issue #266)"
        )

    # --- NLP-BB auto-select for convex MINLPs (nlp_bb=None) ---
    # Placed after problem classifier so MILP/MIQP use their specialized
    # (faster) solvers. Only genuinely nonlinear convex MINLPs reach here.
    # Also skip when a rejecting user callback is provided: the NLP-BB path
    # cannot honor a lazy constraint or an incumbent-callback rejection (no
    # per-node cut application; its primal heuristics inject incumbents without
    # consulting the callback — INT-1, #413). Fall through to the spatial-B&B
    # loop, which enforces both correctly, rather than silently drop the
    # rejection and accept the excluded point.
    if nlp_bb is None and lazy_constraints is None and incumbent_callback is None:
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
                rens_enabled=rens,
                _lns_enabled=_lns_enabled,
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

    # --- Python nonlinear forward-substitution FBBT (on top of the Rust FBBT) ---
    # The Rust FBBT above does not forward-*define* unbounded auxiliary variables —
    # division/sqrt slacks of the form ``x = f(others)`` (gear4/nvs05 class). The
    # DefinedVariableForwardRule in tighten_nonlinear_bounds bounds them by the
    # interval enclosure of their defining expression. This is what keeps the
    # per-node McCormick relaxation bounded over the whole tree: an unbounded-
    # relaxation node is otherwise sentinel-pruned, which taints the spatial dual
    # bound so the (already-found) optimum cannot be certified. Runs before the root
    # OBBT below so OBBT's min/max LPs over the relaxation are themselves bounded.
    lb, ub, _nl_root_infeasible = _apply_nonlinear_tightening_with_status(model, lb, ub)
    if _nl_root_infeasible:
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
    # Skipped only for known-convex models (handled by the convex/NLP path).
    # Pure-INTEGER models with nonlinear terms still benefit: the McCormick
    # envelope of a product over a wide integer range (nvs17/19/23/24 span
    # [0,200]) is catastrophically loose, and integer branching alone cannot close
    # that relaxation gap in any reasonable node budget (the frontier dual bound
    # crawls). OBBT range reduction — rounded inward for integers, so still sound —
    # shrinks the envelope for the whole tree. ``obbt_tighten_root`` self-gates to
    # a no-op when there is no relaxable nonlinearity, so a pure-linear (MILP)
    # model pays nothing here.
    _obbt_has_continuous = any(v.var_type == VarType.CONTINUOUS for v in model._variables)
    _obbt_has_nonlinear = False
    if not _obbt_has_continuous:
        try:
            from discopt._jax.term_classifier import classify_nonlinear_terms as _cnt

            _ot = _cnt(model)
            _obbt_has_nonlinear = bool(
                _ot.bilinear
                or _ot.trilinear
                or _ot.multilinear
                or _ot.monomial
                or _ot.fractional_power
                or _ot.bilinear_with_fp
                or _ot.ratio_of_products
                or _ot.general_nl
            )
        except Exception:
            _obbt_has_nonlinear = False
    _obbt_known_convex = _root_convexity_known and _root_is_convex
    if (
        bool(kwargs.get("obbt_at_root", True))
        and model._objective is not None
        and not _obbt_known_convex
        # Skip once the budget is blown (#654): OBBT only shrinks the box, so
        # declining it leaves a valid looser envelope and lets the wall track
        # ``time_limit`` instead of paying a full clamped OBBT sweep past the
        # deadline. (When time remains, the budget below still clamps it.)
        and not _deadline_exhausted()
        # Continuous (or mixed) models keep the original ≤500-var reach. The
        # pure-integer-nonlinear path is newer and capped tighter (≤50 vars, the
        # ``_AUTO_RLT_LEVEL1_MAX_VARS`` scale): there OBBT reaches a fixpoint in a
        # fraction of a second, so it cannot burn the root budget on a large model
        # where the 2·n projection LPs would not pay for themselves.
        and (
            (_obbt_has_continuous and n_vars <= 500)
            or (_obbt_has_nonlinear and n_vars <= _AUTO_RLT_LEVEL1_MAX_VARS)
        )
    ):
        # OBBT wall time falls into python_time (computed as the remainder at
        # the end of the solve), so no separate timer is tracked here.
        try:
            from discopt._jax.obbt import obbt_tighten_root

            _obbt_budget = min(min(max(time_limit * 0.1, 2.0), 15.0), _remaining_budget())
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

    # --- B2-FIX (task #89): per-node taint accounting for the reported dual bound.
    # A non-rigorous sentinel fathom (a node whose local NLP merely failed and that
    # was pruned with no infeasibility proof) rightly decertifies *optimality*
    # (issue #27a) — but the node's POP-TIME lower bound (inherited from its
    # parent's rigorous relaxation solve; the Rust import floors every child at
    # its parent's bound) remains a valid lower bound for that node's whole
    # subtree forever. The rigorous global dual bound of a tainted tree is
    # therefore  min(surviving-frontier bound, min over tainted nodes of their
    # pop-time bound)  — NOT the bare frontier minimum (which may over-report:
    # the unproven subtree was removed from it), and NOT nothing (discarding the
    # whole tree bound under-reports: nvs05 reported 1.348 where the search had
    # proven 4.87 — DECOMP-1 §3, "decertify-and-discard").
    #   _taint_floor_internal: running min (internal minimization sense) of the
    #     pop-time bounds of every node fathomed without a rigorous proof. +inf
    #     while no such node exists. A -inf floor (root fathomed non-rigorously,
    #     or a node never bounded) makes the recovery below discard the tree
    #     bound — exactly the pre-fix conservative behavior.
    #   _tree_bound_poisoned: True when a possibly-INVALID bound value entered
    #     the tree itself (convex batch node whose non-KKT objective was kept as
    #     its bound, roadmap P0.3). Then no frontier arithmetic is trustworthy
    #     and the tree bound is discarded wholesale, as before.
    _taint_floor_internal = np.inf
    _tree_bound_poisoned = False
    #   _convex_bound_untrusted: True when a convex node's NLP objective was not
    #     a trusted (KKT) lower bound (C-13 serial / P0.3 batch). The node's own
    #     bound was abstained or poisoned, but — unlike a sentinel fathom — there
    #     is no pop-time floor re-representing what the certification would rest
    #     on, so the SPATIAL-CERT terminal accounting below must never re-earn
    #     certification over this cause.
    _convex_bound_untrusted = False

    # --- Soundness (C-1): distinguish a PROVEN-infeasible fathom from a merely
    # non-rigorous one, exactly as ``_solve_nlp_bb`` does with
    # ``_unconverged_fathom``. A node fathomed on an empty McCormick/LP
    # relaxation over a finite box (``node_infeasible_mask``) or on a
    # ``SolveStatus.INFEASIBLE`` certificate is a valid infeasibility proof. A
    # node that merely carries the failure sentinel because its local NLP failed
    # / diverged / returned a constraint-violating iterate is NON-rigorous — it
    # does not prove the subtree empty. If the tree later exhausts with no
    # incumbent, an "infeasible" verdict is only sound when NO node was fathomed
    # non-rigorously; otherwise feasibility is genuinely UNKNOWN and reporting
    # "infeasible" would be a false certificate (the worst-class error). Set by a
    # single authoritative per-batch sweep below (any sentinel-without-proof node)
    # and consumed in the finalize else-branch.
    _nonrigorous_fathom = False
    # #467 sub-bug #3: set True when the ROOT batch (iteration 0 — the whole
    # feasible region) is rigorously proven infeasible (every root node carries a
    # ``node_infeasible_mask`` empty-box / empty-relaxation certificate — the same
    # rigor the terminal ``infeasible`` verdict already trusts). A soft incumbent
    # accepted only within tolerance that lies inside a region the search RIGOROUSLY
    # proved infeasible must NOT be certified ``optimal``; the terminal logic
    # discards such an incumbent (unless it is itself rigorously feasible, which
    # would mean the root proof over-tightened — then the incumbent stands and no
    # false ``infeasible`` is emitted).
    _root_rigorously_infeasible = False

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
    _alphabb_expr = None
    _use_alphabb = False
    _alphabb_eligible = n_vars <= 50 and not _model_is_convex and hasattr(evaluator, "_obj_fn")

    # Convex-objective node bound. When the (minimized) objective is a convex
    # quadratic but the model is nonconvex (nonconvex constraints), the spatial
    # path McCormick-linearizes the convex objective and loses a huge amount of
    # bound (nvs17/19/23/24 span [0,200]: root McCormick -2522 vs convex bound
    # -1106 ~ optimum -1100). Keeping the objective exact via its supporting
    # hyperplane recovers an almost-tight, rigorous bound at each node. Engaged
    # even when a McCormick LP relaxer is present (the LP is the loose source).
    _use_convex_obj_bound = (not _model_is_convex) and _objective_is_convex_quadratic(
        model, evaluator, n_vars, remaining_budget=_remaining_budget()
    )
    if _use_convex_obj_bound:
        logger.debug("convex-objective node bound enabled (n_vars=%d)", n_vars)

    # --- McCormick relaxation bounds ---
    _mc_obj_eval = None  # BatchRelaxationEvaluator for the McCormick "nlp" bound
    _mc_obj_relax_fn = None  # raw relaxation fn for NLP bounds
    _mc_con_relax_fns: list[Callable] | None = None
    _mc_con_senses = None
    _mc_negate = False
    _mc_mode = mccormick_bounds
    _mc_lp_relaxer = None  # MccormickLPRelaxer instance when _mc_mode == "lp"
    # Global root cut pool (P3): separated once at the root, then inherited at
    # every node so each node reproduces the strong root bound for the cost of a
    # warm LP solve instead of re-separating the PSD/RLT cuts from scratch. Stays
    # None unless the relaxer carries PSD cuts (the nvs* / box-QP regime).
    _root_cut_pool = None
    # CUT-INHERIT-GRAD diagnostic: the fraction of the ONE root pool separation
    # solve's wall spent in the square+PSD point-separation loops. Surfaced in
    # ``solver_stats`` for observability; NOT the gate predicate (the entry
    # experiment showed it does not separate win-from-neutral — the pool-fires
    # signal does). ``None`` when no pool is separated.
    _root_sqpsd_frac: Optional[float] = None
    # Root-cut-pool inheritance (THRU-4, ``DISCOPT_CUT_INHERIT`` /
    # ``SolverTuning.cut_inherit``, **opt-in, default force-off**,
    # CUT-INHERIT-GRAD). Tri-state resolution:
    #   * ``_cut_inherit_mode is True``  — force ON  (env ``=1`` / ``cut_inherit=True``)
    #   * ``_cut_inherit_mode is False`` — force OFF (env unset/``=0`` / ``cut_inherit=False``)
    #     — the SHIPPED DEFAULT: byte-identical to pre-THRU-4 behaviour.
    #   * ``_cut_inherit_mode is None``  — STRUCTURE-GATED opt-in (env ``=gated`` /
    #     ``cut_inherit=None``): the pool is captured optimistically at the root and
    #     inheritance engages iff a non-empty pool actually populates
    #     (``_root_cut_pool is not None``). Validated broadly beneficial where it
    #     fires, but NOT the default — a flag-path false-optimal on the MINLP
    #     cold-path class (nvs22) blocks the flip (CLAUDE.md §1; see
    #     ``docs/dev/cut-inherit-grad-2026-07-08.md``).
    # When active: (a) the general root cut pool below is captured even when the
    # incremental engine is unavailable (cold-path instances — nvs19/24 — are
    # exactly where THRU-3 measured the per-node separation drag), and (b) every
    # node solve passes ``skip_pool_separators=True`` so the square/PSD
    # point-separation loops — up to 8 full MILP re-solves per node each — are
    # replaced by the inherited pool. Root behaviour is unchanged. When the model
    # carries no liftable square/PSD structure the pool stays empty and the gated
    # path is byte-identical to force-OFF.
    _cut_inherit_mode = _tuning().cut_inherit
    # "Not forced off" — the condition under which the root pool is captured and
    # (given a populated pool) inheritance is allowed to engage. Forced-ON and
    # structure-gated both capture; only an explicit ``=0`` suppresses it.
    _cut_inherit_enabled = _cut_inherit_mode is not False
    # The dual bound the root cut-pool relaxation proves over the whole feasible
    # region (a valid global lower bound for a MINIMIZE). The pool is separated for
    # its CUT ROWS, but the strengthened relaxation also yields a far tighter root
    # bound than the cut-less tree path (nvs19: -1156 vs the McCormick -88237);
    # captured here so the final bound / certification can use it instead of
    # discarding it. ``None`` when no pool is separated.
    _root_pool_bound = None

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
        # A nonconvex model qualifies for the LP-form McCormick path when it has
        # something to branch on that the relaxation can tighten:
        #   * a continuous variable (spatial branching), OR
        #   * (nvs* gap) a pure-/mixed-integer model whose integers appear in
        #     nonlinear terms — the integers are branched by the standard B&B
        #     while the McCormick+PSD relaxation supplies a valid, far tighter
        #     per-node dual bound than the interval alphaBB floor. The prototype
        #     for the global cut-pool proved this closes nvs17 (frozen -65842 ->
        #     14 nodes, bound -1110), the SCIP regime.
        # Pure-integer models were historically pinned to alphaBB ("none") on the
        # premise they had "nothing to spatial-branch on" — that conflates spatial
        # with integer branching. Admit them and let the downstream
        # has_relaxable_nonlinearity / _has_branchable / probe gates fall a model
        # back when the LP relaxer is unusable. ``_pure_discrete`` routes those
        # fallbacks to the sound alphaBB "none" rather than the integer-unsound
        # "nlp" objective bound (issue #120: a local NLP solve of a nonconvex
        # surface can certify a value ABOVE the true optimum).
        if not _model_is_convex and model._objective is not None:
            _mc_mode = "lp"
        else:
            _mc_mode = "none"

    if _mc_mode == "lp" and model._objective is not None:
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        # Resolve the high-level ``rlt`` switch into the two concrete RLT levers:
        # build-time level-1 RLT (``rlt_level1``, which tightens the root bound)
        # and per-node RLT cut separation (``rlt_cuts``). ``rlt`` is the
        # user-facing control that replaces the legacy ``DISCOPT_RLT`` env gate:
        #   * "auto" (default): defer to the structure-gated cut policy for the
        #     per-node cuts; build-time level-1 stays off unless the env var
        #     forces it (back-compat). This is the historical default behaviour.
        #   * True / "on": engage RLT in full — build-time level-1 *and* per-node
        #     cuts — overriding the auto policy (an explicit opt-in to a family).
        #   * False / "off": force every RLT lever off, even the per-node cuts the
        #     auto policy would otherwise pick.
        # Every RLT family is sound (a constraint×bound product is non-negative at
        # every feasible point), so this switch only ever trades bound tightness
        # for relaxation size — never correctness.
        _rlt_on: Optional[bool]
        if rlt is True or (isinstance(rlt, str) and rlt.lower() in ("on", "true")):
            _rlt_on = True
        elif rlt is False or (isinstance(rlt, str) and rlt.lower() in ("off", "false")):
            _rlt_on = False
        else:  # "auto" (or any unrecognized value) → let the policy decide
            _rlt_on = None
        _eff_rlt_cuts = True if _rlt_on else rlt_cuts
        # Under "auto", additionally engage build-time level-1 RLT (root-bound
        # tightening) on small models — it certifies instances the per-node cut
        # policy alone leaves open (nvs05) at negligible LP cost, while staying off
        # large models where the product rows blow up the per-node LP (casctanks).
        # The per-node cut family is untouched (this only adds the root tightening),
        # and RLT is sound, so this trades bound tightness for size, never
        # correctness. An explicit rlt=True/False still wins.
        _auto_rlt_level1 = _rlt_on is None and n_vars <= _AUTO_RLT_LEVEL1_MAX_VARS
        _eff_rlt_level1 = bool(_rlt_on) or _auto_rlt_level1

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
                rlt_cuts=_eff_rlt_cuts,
                rlt_level1=_eff_rlt_level1,
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
                # bound. For a pure-discrete model fall to the sound alphaBB
                # "none" instead — the nonconvex "nlp" objective bound is not
                # valid there (issue #120).
                _mc_lp_relaxer = None
                _mc_mode = "none" if _pure_discrete else "nlp"
            else:
                # Structure-gated cut policy (cuts="auto", the default): the A/B
                # sweep showed RLT dominates on QCQP *with* linear constraints, PSD
                # on box-QP (no constraints), and stacking the two is
                # counter-productive. So pick exactly one by structure, gated by
                # size, never both. An explicit psd_cuts/rlt_cuts flag, or an
                # explicit rlt=True/False, takes precedence (the user opted into or
                # out of a specific family); cuts="manual" disables the policy
                # entirely. ``rlt=True`` already forced rlt_cuts on above, so the
                # policy is skipped; ``rlt=False`` skips it and pins RLT cuts off.
                if _rlt_on is False:
                    _mc_lp_relaxer._rlt_cuts = False
                elif cuts == "auto" and not psd_cuts and not rlt_cuts and _rlt_on is None:
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
                        "integer); falling back to %s relaxation.",
                        "alphaBB" if _pure_discrete else "NLP",
                    )
                    _mc_lp_relaxer = None
                    _mc_mode = "none" if _pure_discrete else "nlp"
                else:
                    if _nl_int_cols:
                        tree.set_spatial_integer_cols(np.asarray(_nl_int_cols, dtype=np.int64))
                    # Deprioritize functionally-dependent continuous columns in
                    # spatial branching. Two disjoint sources, both of which are
                    # *outputs* pinned by the independent variables, so bisecting
                    # them never tightens the McCormick gap (which is driven by the
                    # independent inputs):
                    #   (a) lifted auxiliaries introduced by the factorable reform
                    #       (every aux is ``w = g(x)`` by construction — appended
                    #       after the originals, so columns ``>= _prereform_nvars``).
                    #       Branching a product aux ``w = x_i*x_j`` cannot shrink
                    #       that product's envelope; only bisecting ``x_i``/``x_j``
                    #       can. Without this, the relaxer's wide-domain aux columns
                    #       win the relative-width competition and the integer
                    #       product never gets partitioned (welded-beam / nvs05:
                    #       bound frozen at 2.022 vs the true 5.471).
                    #   (b) original continuous outputs pinned by a nonlinear
                    #       equality ``x_i = f(others)`` (detected pre-reform).
                    # Restricting spatial branching to the original *independent*
                    # variables matches BARON/Couenne practice and is complete: when
                    # every independent domain is a point, all dependents are exact
                    # and the relaxation is tight. The Rust selector's last-resort
                    # fallback still branches a deprioritized column if no
                    # independent one remains, preserving completeness.
                    try:
                        _dep_cols_set: set = set()
                        if _prereform_model is not None and n_vars > _prereform_nvars:
                            _dep_cols_set.update(range(_prereform_nvars, n_vars))
                        # R4 (DISCOPT_LIFT_ZERO_SPANNING_FACTORS): a lifted aux
                        # ``w = f(x)`` for a *product factor* whose interval spans 0
                        # is branch-responsive — splitting w at 0 flips the factor's
                        # sign and tightens the product's McCormick envelope, the one
                        # move that un-pins the bound (st_e36). Keep those columns
                        # branchable by removing them from the deprioritized set. The
                        # flag gates the reform's tagging, so this set is empty (no
                        # behaviour change) unless the flag is on.
                        _zsf_names = getattr(model, "_zero_spanning_factor_auxes", None)
                        if _zsf_names:
                            _name_to_col = {}
                            _col = 0
                            for _v in model._variables:
                                _name_to_col[_v.name] = _col
                                _col += _v.size
                            _zsf_cols = {
                                _name_to_col[_nm] for _nm in _zsf_names if _nm in _name_to_col
                            }
                            _dep_cols_set.difference_update(_zsf_cols)
                        if _dependent_var_names:
                            from discopt._jax.dependent_vars import (
                                dependent_columns_for_model,
                            )

                            _dep_cols_set.update(
                                dependent_columns_for_model(model, _dependent_var_names)
                            )
                        _dep_cols = sorted(_dep_cols_set)
                        if _dep_cols:
                            tree.set_branch_deprioritized(np.asarray(_dep_cols, dtype=np.int64))
                            logger.debug(
                                "spatial branching deprioritizes %d "
                                "functionally-dependent continuous columns "
                                "(%d lifted aux + %d original outputs)",
                                len(_dep_cols),
                                max(0, n_vars - _prereform_nvars)
                                if _prereform_model is not None
                                else 0,
                                len(_dependent_var_names),
                            )
                    except Exception as _dep_exc:  # pragma: no cover - defensive
                        logger.debug("branch-deprioritization wiring skipped: %s", _dep_exc)
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
                    elif _deadline_exhausted():
                        # Budget already blown (#654): skip the one-time root cut
                        # pool separation. Its only value is a stronger per-node
                        # bound during the search, but a past-deadline node loop
                        # finalizes almost immediately, so separating the pool
                        # here would just overrun the wall for no realized bound.
                        # Nodes still separate their own cuts if any run. Sound:
                        # the relaxer/bound are unchanged — only the inherited-pool
                        # speedup is forgone.
                        pass
                    elif _root_cut_rounds > 0 and getattr(_mc_lp_relaxer, "_psd_cuts", False):
                        # Root cut pool (P3, opt-in): the relaxer carries PSD cuts, so
                        # separate a strong pool ONCE at the root box (many
                        # rounds, near the Shor SDP bound) and capture its rows.
                        # Every node then inherits the pool for a warm LP solve
                        # instead of re-separating from scratch — the standalone
                        # prototype showed this drops nvs17's per-node cost ~30x
                        # (0.5s -> 16ms) while reproducing the strong root bound.
                        # Sound: each PSD cut is valid for the whole feasible set,
                        # so an inherited row never cuts off a feasible point.
                        try:
                            _pool_chunks: list = []
                            _root_remaining = time_limit - (time.perf_counter() - t_start)
                            _pool_budget = min(
                                max(time_limit * 0.25, 5.0),
                                max(_root_remaining, _DEADLINE_NODE_FLOOR_S),
                            )
                            _pool_res = _mc_lp_relaxer.solve_at_node(
                                _probe_lb,
                                _probe_ub,
                                time_limit=_pool_budget,
                                out_cuts=_pool_chunks,
                                psd_max_rounds=_root_cut_rounds,
                            )
                            if _pool_chunks and _pool_chunks[0] is not None:
                                # solve_at_node captures each separated chunk as a
                                # (A, b, col_idents) triple of upper-bound rows
                                # ``A x <= b`` over the root's lifted column space
                                # plus the per-column identity vector (C-44); a
                                # node remaps the rows onto its own layout by those
                                # identities. inherited_cuts takes the same form.
                                _A_pool, _b_pool, _idents_pool = _pool_chunks[0]
                                _n_pool = _A_pool.shape[0]
                                if _n_pool > _root_cut_max:
                                    # Keep the last (most-recently separated, i.e.
                                    # deepest-round) rows; they target the tightest
                                    # residual violation. Capping bounds per-node LP
                                    # size so inheritance stays cheap.
                                    _A_pool = _A_pool[-_root_cut_max:]
                                    _b_pool = _b_pool[-_root_cut_max:]
                                _root_cut_pool = (_A_pool, _b_pool, _idents_pool)
                                # Keep the strengthened root bound: it is a valid
                                # global lower bound (the pool relaxation holds over
                                # the whole feasible region) and is far tighter than
                                # the cut-less tree path — so the final certificate
                                # should use it rather than recomputing from scratch.
                                if (
                                    _pool_res is not None
                                    and _pool_res.lower_bound is not None
                                    and np.isfinite(_pool_res.lower_bound)
                                ):
                                    _root_pool_bound = float(_pool_res.lower_bound)
                                logger.info(
                                    "Root PSD cut pool: %d cuts (of %d separated, "
                                    "%d rounds), root bound %s — inherited at every node",
                                    _A_pool.shape[0],
                                    _n_pool,
                                    _root_cut_rounds,
                                    f"{_pool_res.lower_bound:.4g}"
                                    if _pool_res is not None and _pool_res.lower_bound is not None
                                    else "n/a",
                                )
                        except Exception as _pool_exc:  # pragma: no cover - defensive
                            logger.debug("root cut pool separation skipped: %s", _pool_exc)
                            _root_cut_pool = None
                    elif getattr(_mc_lp_relaxer, "_inc", None) is not None or _cut_inherit_enabled:
                        # Root cut pool for the GENERAL spatial path (cert:T1.3).
                        # When the incremental engine is active but PSD cuts are
                        # off, the fast path (which skips per-node separation) would
                        # otherwise solve base McCormick only, collapsing the bound
                        # on separation-reliant models (measured: dispatch 3 → 9843
                        # nodes). Capture the general root separation chain ONCE
                        # (multilinear / edge-concave / univariate-square / convex /
                        # RLT) — ``out_cuts`` forces the cold, separating path (see
                        # ``solve_at_node``) — and inherit it at every fast-path
                        # node. Sound: a cut valid over the root box is valid over
                        # every sub-box, so an inherited row never removes a
                        # feasible point.
                        #
                        # THRU-4/CUT-INHERIT-GRAD (``_cut_inherit_enabled``):
                        # additionally capture this pool
                        # when the incremental engine is unavailable — cold-path
                        # nodes are exactly where the per-node square/PSD point
                        # separators dominate (nvs24: 73%+12% of the solve wall) —
                        # so the node call sites below can skip those loops in
                        # favour of the inherited pool (``skip_pool_separators``).
                        try:
                            _pool_chunks = []
                            _root_remaining = time_limit - (time.perf_counter() - t_start)
                            _pool_budget = min(
                                max(time_limit * 0.25, 5.0),
                                max(_root_remaining, _DEADLINE_NODE_FLOOR_S),
                            )
                            # CUT-INHERIT-GRAD structure predicate: snapshot the
                            # per-family separation timers BEFORE the root pool solve
                            # so the square+PSD wall this ONE root separation spends is
                            # isolable (no node solves have run yet, so the delta is
                            # root-only). The fraction of the root pool solve's wall
                            # consumed by the two point-separation loops is the
                            # cheap, general, root-time feature that discriminates the
                            # dense-integer-QP win class (loops dominate the node wall,
                            # THRU-3: nvs24 73%+12%, nvs19 42%+20%) from the neutral
                            # quadratic slice (loops fire but are not the bottleneck,
                            # THRU-4-graduate wall ratio 1.004x). Keys on measured cost
                            # only — never on instance name/shape (CLAUDE.md §2).
                            _sep_before = dict(getattr(_mc_lp_relaxer, "_sep_timers", {}))
                            _sqpsd_before = _sep_before.get("univariate_square", 0.0) + (
                                _sep_before.get("psd", 0.0)
                            )
                            _pool_solve_t0 = time.perf_counter()
                            _pool_res = _mc_lp_relaxer.solve_at_node(
                                _probe_lb,
                                _probe_ub,
                                time_limit=_pool_budget,
                                out_cuts=_pool_chunks,
                            )
                            _pool_solve_wall = time.perf_counter() - _pool_solve_t0
                            _sep_after = getattr(_mc_lp_relaxer, "_sep_timers", {})
                            _sqpsd_root_wall = (
                                _sep_after.get("univariate_square", 0.0)
                                + _sep_after.get("psd", 0.0)
                                - _sqpsd_before
                            )
                            if _pool_solve_wall > 1e-9:
                                _root_sqpsd_frac = max(0.0, _sqpsd_root_wall / _pool_solve_wall)
                            else:
                                _root_sqpsd_frac = 0.0
                            if _pool_chunks and _pool_chunks[0] is not None:
                                _A_pool, _b_pool, _idents_pool = _pool_chunks[0]
                                _n_pool = _A_pool.shape[0]
                                if _n_pool > _root_cut_max:
                                    _A_pool = _A_pool[-_root_cut_max:]
                                    _b_pool = _b_pool[-_root_cut_max:]
                                _root_cut_pool = (_A_pool, _b_pool, _idents_pool)
                                if (
                                    _pool_res is not None
                                    and _pool_res.lower_bound is not None
                                    and np.isfinite(_pool_res.lower_bound)
                                ):
                                    _root_pool_bound = float(_pool_res.lower_bound)
                                logger.info(
                                    "Root cut pool (general spatial): %d cuts (of %d "
                                    "separated), inherited at every fast-path node",
                                    _A_pool.shape[0],
                                    _n_pool,
                                )
                        except Exception as _pool_exc:  # pragma: no cover - defensive
                            logger.debug("general root cut pool skipped: %s", _pool_exc)
                            _root_cut_pool = None

    # AlphaBB alpha estimate (lever 3, issue #194), deferred from above: compute
    # it only when the LP relaxer is NOT the bound source. When the LP relaxer is
    # active it supplies every node's valid dual bound, so the ~2s alpha estimate
    # (and the per-node alphaBB it enables) is skipped. ``DISCOPT_ALPHABB_WITH_LP=1``
    # forces the estimate even under the LP relaxer (A/B / fallback safety).
    _alphabb_force = _tuning().alphabb_with_lp
    if (
        _alphabb_eligible
        and (_mc_lp_relaxer is None or _alphabb_force)
        and model._objective is not None
    ):
        # C-17: the node bound is derived from ``rigorous_alpha`` (sound interval
        # Hessian) per node box, NOT a sampled root alpha. We only need the
        # internally-minimized objective EXPRESSION here; the per-node alpha is
        # (re)computed rigorously inside ``_compute_alphabb_bound``.
        # ``evaluate_objective``/``_obj_fn`` minimize ``-f`` for a maximize model,
        # so the expression whose Hessian must be convexified is likewise negated.
        _alphabb_expr = -model._objective.expression if _obj_negate else model._objective.expression
        _use_alphabb = True

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

    if _mc_mode == "nlp" and model._objective is not None:
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
        except Exception as e:
            logger.warning("McCormick relaxation setup failed: %s", e)
            _mc_obj_eval = None
            _mc_obj_relax_fn = None

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
    # Best incumbent value the integer-neighbourhood box search has already been
    # run from. The box search re-enumerates the integer lattice around the
    # incumbent, so it is only worth re-running when the incumbent itself moved
    # to a new integer assignment; tracking its objective avoids redundant
    # re-enumeration of the same neighbourhood every scheduled iteration.
    _last_box_inc_obj = np.inf
    # --- Adaptive LNS primal-improvement layer state (issue #267) ---
    # Counts of LNS improver calls, used to escalate the local-branching radius
    # k across calls and to throttle node-diving. The whole layer is disabled
    # when ``_lns_enabled`` is False (the recursion guard for sub-MIP re-solves).
    _lns_lb_calls = 0
    _lns_dive_calls = 0
    _lns_k_schedule = (2, 5, 10)
    _lns_has_integers = bool(int_sizes) and int(np.sum(int_sizes)) > 0
    # Best incumbent value the cutoff-tightening phases (C/C3) have already acted
    # on. They fire whenever the incumbent strictly improves below this — from
    # ANY source, including the sub-NLP / binary-seed heuristics that inject
    # directly and are not counted in proc_stats["incumbent_updates"].
    _last_tighten_inc = np.inf

    # --- SCIP-style heuristic effort budget (#330) ---
    # The heaviest standalone-strength primal heuristics — the root binary-seed
    # *enumeration* phase and the sub-MIP LNS (RINS, local branching) — dominate
    # wall time on easy instances when run unconditionally inside a global solver
    # (40–104 sub-NLP solves on ≤7-node models; issue #330) without changing the
    # proven optimum. Following SCIP's ``heur_subnlp`` budgeting, gate *those* by a
    # *contingent* that grows with the search effort already spent (B&B nodes) and
    # is weighted by their demonstrated success — the same shape as SCIP's
    # ``iterquot·nodes·[gain·(found+1)/(calls+1)] + offset``. So they are deferred
    # on trivially-easy models and only fire once the search is hard enough to
    # afford them; a productive improver stays funded while one that keeps finding
    # nothing is smoothly defunded. NOT gated: the cheap incumbent *finders*
    # (feasibility pump, a single root sub-NLP) that supply the first incumbent for
    # pruning, and the lighter integer lattice searches (integer local / box
    # search) that discopt's often-weak McCormick relaxation genuinely leans on to
    # reach the global assignment (nvs05/nvs23). The MINLP literature endorses
    # exactly this trade for in-solver heuristics: "it is often worth sacrificing
    # success on a small number of instances for a significant saving in average
    # running time" (e.g. the FP enumeration phase is dropped inside a global
    # solver). Soundness is unaffected: B&B remains exhaustive, so a deferred or
    # skipped improver can never yield a wrong optimum — at worst a handful of
    # instances take more nodes. ``DISCOPT_HEUR_BUDGET=0`` restores the prior
    # always-on behaviour.
    _heur_budget_on = os.environ.get("DISCOPT_HEUR_BUDGET", "1") != "0"
    _HEUR_BUDGET_OFFSET = float(os.environ.get("DISCOPT_HEUR_OFFSET", "0"))  # root contingent
    _HEUR_BUDGET_QUOT = float(os.environ.get("DISCOPT_HEUR_QUOT", "0.5"))  # per processed node
    _HEUR_SUCCESS_GAIN = 3.0  # SCIP's 3·(found+1)/(calls+1) success weighting
    # Cost (sub-NLP-solve-equivalents) of each *budgeted* improver — the heavier
    # standalone-strength components: the binary-seed *enumeration* phase and the
    # sub-MIP LNS (RINS, local branching). The lighter lattice searches
    # (integer local / box search) are left ungated above.
    _HEUR_COST = {"enumerate": 12.0, "rins": 5.0, "lbranch": 10.0}
    # Mutable container so the nested gate/record helpers can update it without a
    # ``nonlocal`` dance. ``cost`` is sub-NLP-solve-equivalents already spent on
    # the (improver-role) heuristics; ``found`` counts those that strictly
    # improved the incumbent — the success signal in the contingent.
    _heur_state = {"calls": 0, "found": 0, "cost": 0.0}
    # G2: the hit-rate-adaptive governor (default-OFF; see heuristic_governor.py).
    _heuristic_governor = _get_heuristic_governor()

    def _improver_allowed(cost: float) -> bool:
        """Whether an *improver*-role heuristic costing ``cost`` may run now.

        The finder role is never gated: when there is no incumbent yet, securing
        the first one (for pruning) takes priority. Once an incumbent exists the
        call is an improver and must fit the success-weighted, node-proportional
        contingent (SCIP ``heur_subnlp`` shape)."""
        if not _heur_budget_on or tree.incumbent() is None:
            return True
        _nodes = float(tree.stats().get("total_nodes", 0))
        _weight = _HEUR_SUCCESS_GAIN * (_heur_state["found"] + 1) / (_heur_state["calls"] + 1)
        _contingent = _HEUR_BUDGET_OFFSET + _HEUR_BUDGET_QUOT * _nodes * _weight
        return (_heur_state["cost"] + cost) <= _contingent

    def _record_improver(cost: float, improved: bool) -> None:
        """Charge an improver-role run against the contingent and note success."""
        _heur_state["calls"] += 1
        _heur_state["cost"] += cost
        if improved:
            _heur_state["found"] += 1

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

    # --- TX1: adaptive back-off for the strided in-tree node NLP -------------
    # DISCOPT_ADAPTIVE_NLP (default ON since G2, flag-graduation; =0 restores the
    # fixed stride). The strided node-NLP is
    # a pure primal heuristic (fires only in the nonconvex + LP-relaxer regime,
    # where the LP gives the bound); TX0 measured it as idle waste on integer-heavy
    # models (nvs09 14.3 s, identical proof). When ON we grow the *effective*
    # stride whenever the node-NLP fired but found no incumbent improvement, and
    # reset it to the base stride the instant it does. This state persists across
    # B&B iterations; it never touches the convex / no-LP-relaxer bound-source path.
    _adaptive_nlp = _tuning().adaptive_nlp
    _ADAPTIVE_NLP_PATIENCE = 2  # non-improving fired batches before a doubling
    _ADAPTIVE_NLP_STRIDE_CAP = 256  # ceiling on the effective stride
    _adaptive_nlp_state = {"eff_stride": 0, "no_improve": 0}

    # --- F4: root-heuristic NLP/compile budget gate --------------------------
    # ``solve(time_limit=T)`` is a contract. On the no-relaxation flowsheet class
    # (contvar, heatexch_gen3, hda) the root PRIMAL-HEURISTIC phase blows past it
    # two ways (bottleneck-profile-2026-07-05 §4):
    #   (a) the *first* heuristic NLP forces an uninterruptible first-time XLA
    #       compile of the sparse Lagrangian Hessian — up to ~3x the whole budget
    #       on a deep DAG, and no deadline poll can fire inside a jit compile;
    #   (b) once compiled, each per-node/heuristic NLP solve still runs many
    #       seconds and OVERRUNS its own ``max_wall_time`` clamp (the exact
    #       Hessian per IPM iteration is expensive), so families that launch an
    #       NLP without first checking the deadline (diving, extra pump rounds,
    #       node-diving) accumulate tens of seconds past T.
    # Gate ENTRY (compiles cannot be interrupted). Every gated call is a primal
    # heuristic, so skipping one is always sound: it can change which incumbent is
    # found and when (node counts may shift) but never the dual bound or the
    # returned optimum. Off switch: ``DISCOPT_ROOT_BUDGET_GATE=0``.
    _root_budget_gate_on = os.environ.get("DISCOPT_ROOT_BUDGET_GATE", "1") != "0"
    # Worst-case observed root/heuristic NLP wall (self-calibrating per model +
    # machine); seeds a default until the first solve is measured. Used to refuse
    # launching a new heuristic NLP when the time left cannot absorb another
    # solve of the largest size seen so far. The *max* (not mean) is deliberate:
    # a heuristic NLP can OVERRUN its own ``max_wall_time`` clamp by ~10 s on the
    # expensive-Hessian class, so once one long solve is observed we must not
    # launch another unless that much budget remains. A dict so the nested
    # closures can mutate it without a ``nonlocal`` dance.
    _heur_nlp_cost = {"max": 0.0, "default": 2.0}

    def _observe_heur_nlp(wall: float) -> None:
        """Record an observed heuristic/root NLP wall for the entry gate."""
        if wall >= 0.0 and wall > _heur_nlp_cost["max"]:
            _heur_nlp_cost["max"] = float(wall)

    def _mean_heur_nlp_cost() -> float:
        if _heur_nlp_cost["max"] <= 0.0:
            return _heur_nlp_cost["default"]
        return _heur_nlp_cost["max"]

    def _root_heur_nlp_entry_ok(_ev=None) -> bool:
        """Whether a compile-/solve-triggering root heuristic NLP may start now.

        Returns False when either the deadline has effectively passed (no room to
        absorb even one typical solve) or the evaluator's Hessian kernel is not
        yet compiled and the estimated first-time compile does not fit the
        remaining budget. Both cases skip a *primal heuristic* only — never the
        dual-bound path — so refusing is always sound (§0.3 heuristic-policy).
        """
        if not _root_budget_gate_on:
            return True
        _remaining = _deadline - time.perf_counter()
        # No budget left to absorb even one typical (already-compiled) solve.
        if _remaining <= max(_DEADLINE_NODE_FLOOR_S, _mean_heur_nlp_cost()):
            return False
        # First-time compile risk: an uninterruptible XLA compile can dwarf the
        # whole budget and cannot be polled once entered, so only enter when the
        # (conservative, measured) estimate fits the time left.
        if _ev is not None:
            try:
                _compile_est = _ev.hessian_compile_estimate_s()
            except Exception:
                _compile_est = 0.0
            if _compile_est > 0.0 and _remaining < _compile_est:
                return False
        return True

    # Root-node certification instrumentation (cert:T0.1). Snapshot the tree's
    # global lower bound (internal minimization sense) and the elapsed wall
    # clock at the moment the root node has been fully processed (end of
    # iteration 0), before any branching lifts the frontier. ``root_gap`` is
    # derived from these at result-build time against the final incumbent.
    _root_time: Optional[float] = None
    _root_glb_internal: Optional[float] = None

    # Per-node reduction timers (cert:T0.3). Accumulated across the spatial B&B
    # loop and surfaced on SolveResult.solver_stats. Pure instrumentation.
    _reduce_timers = {"fbbt": 0.0, "obbt": 0.0}

    # Objective-gating priority branching (issue #184). Opt-in via
    # ``DISCOPT_OBJ_BRANCH_PRIORITY=1``: branch the binaries that gate the
    # objective's nonlinear terms first so the global bound can climb off a
    # structural 0 (see _branch_priority_integer_vars). Empty when disabled or
    # when the model has no such gating integers, leaving branching unchanged.
    _branch_priority_vars: frozenset[int] = frozenset()
    if _tuning().obj_branch_priority:
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

    # --- Per-node OBBT (Lever A) setup ---
    # Tighten each node's box against its McCormick relaxation (with the
    # incumbent objective as a cutoff when available). Per-node bounds are
    # subtree-local and rigorous. This is gated to the SAME structural class as
    # branch-deprioritization — models with functionally-dependent continuous
    # intermediates (defining equalities x_i = f(others)) — so models without
    # that structure incur zero overhead. The two levers are complementary:
    # deprioritization stops branching from wasting effort on dependent outputs,
    # while per-node OBBT pins those outputs from the relaxation + cutoff so a
    # node can fathom once its independent drivers are fixed. Neither alone
    # certifies the welded-beam (nvs05) class; together they do (~23 nodes).
    # A cumulative time budget caps total cost so a deep tree can never let
    # per-node OBBT dominate wall clock.
    _pn_obbt_structural = _mc_lp_relaxer is not None and bool(_dependent_var_names)
    _pn_obbt_small = _pn_obbt_structural and n_vars <= _PER_NODE_OBBT_MAX_VARS
    # T2.5 de-gate: with the scored top-k flag on, the same structural class runs
    # per-node OBBT above the size cap too — but only on the top-k
    # ``width × |reduced cost|`` variables, so the per-sweep probe count is bounded
    # (O(top_k) LPs/node instead of O(n_vars)) and the existing per-node /
    # cumulative budgets can cap wall. Below the cap we keep the legacy
    # all-columns behavior byte-for-byte (top_k stays None) so the flag is inert
    # on the already-certifying small class.
    _pn_obbt_degated = (
        _pn_obbt_structural and n_vars > _PER_NODE_OBBT_MAX_VARS and _obbt_topk_enabled()
    )
    _per_node_obbt_enabled = _pn_obbt_small or _pn_obbt_degated
    _pn_obbt_topk = _PER_NODE_OBBT_TOPK if _pn_obbt_degated else None
    _pn_obbt_spent = 0.0
    _pn_obbt_budget_total = time_limit * _PER_NODE_OBBT_BUDGET_FRAC
    if _per_node_obbt_enabled:
        from discopt._jax.obbt import obbt_tighten_root

        logger.debug(
            "per-node OBBT enabled (n_vars=%d, dependent=%d, budget=%.1fs, top_k=%s)",
            n_vars,
            len(_dependent_var_names),
            _pn_obbt_budget_total,
            _pn_obbt_topk,
        )

    # --- Per-node cheap reduction (cert:T2.4b, flag default OFF) ---
    # After each node LP solve, run reduce_node (cutoff-FBBT + free DBBT from the
    # node LP reduced costs + integer RC-fixing) and feed the tightened box to the
    # child nodes via ``tree.set_node_bounds`` before the tree branches. Gated to
    # the LP-relaxer spatial path (the only path exposing node-LP marginals) and
    # behind the flag, default OFF until T2.6.
    _node_reduce_enabled = _tuning().node_reduce and _mc_lp_relaxer is not None
    _node_reduce_fn: Any = None
    if _node_reduce_enabled:
        try:
            from discopt._jax.node_reduce import reduce_node as _node_reduce_fn

            logger.debug("per-node reduce_node enabled (cert:T2.4b)")
        except Exception as _nr_exc:  # pragma: no cover - defensive
            logger.debug("reduce_node import failed; disabling node reduce: %s", _nr_exc)
            _node_reduce_enabled = False
            _node_reduce_fn = None

    # --- Reduced-space McCormick per-node bounding (MAiNGO-parity §2 P2.3) ---
    # ``DISCOPT_RELAX_SPACE=reduced`` swaps the lifted per-node LP for a Kelley
    # cutting-plane bound over the ORIGINAL variables only (no auxiliary columns —
    # the #557 dense-lifted FT storm is avoided by construction). The bound is
    # certifying: an ``optimal`` status is a VALID node dual lower bound and an
    # ``infeasible`` status is a rigorous fathom (the relaxed feasible set is
    # empty). ``lifted``/``auto`` preserve today's path byte-for-byte.
    #
    # Sound-or-refuse (plan §0.3): the reduced evaluator is *probed* once here on
    # the root box. If the model is outside the sound MCBox scope
    # (``UnsupportedRelaxation``), the WHOLE solve silently falls back to the
    # lifted path — never an error to the user. Per node, ``unsupported`` /
    # ``unbounded`` yields no reduced bound (the lifted LP still runs as fallback,
    # so no node is ever left without a bound source it would otherwise have had).
    _relax_space = _tuning().relax_space
    if _relax_space == "hybrid":
        raise NotImplementedError(
            "DISCOPT_RELAX_SPACE=hybrid is reserved for MAiNGO-parity plan P2.5 "
            "(MC<->AVM per-term lift) and is not implemented yet. Use 'lifted' "
            "(default), 'auto', or 'reduced'."
        )
    _reduced_space_active = _relax_space == "reduced" and model._objective is not None
    # The reduced-space evaluator relaxes over the ORIGINAL degrees of freedom only
    # (its whole premise — no auxiliary columns). Two adjustments are REQUIRED for a
    # sound bound (task #69, P2.3 root cause #2):
    #
    #   1. Build the relaxation on the PRE-REFORMULATION model. When
    #      ``factorable_reformulate`` ran (line ~3864), ``model`` is already the
    #      LIFTED model: its ``_variables`` include the added dependent aux columns
    #      (e.g. nvs22: 8 originals -> 15 lifted) and its constraints include their
    #      defining equalities. Relaxing THAT through MCBox rebuilds the lifted
    #      formulation — defeating the reduced-space purpose AND feeding the aux
    #      columns' huge bounds (~1e8 on nvs22) into the Kelley LP, whose resulting
    #      ~1e9-scaled basis the in-house simplex can mis-solve as spuriously
    #      "infeasible" -> wrong node fathom -> FALSE OPTIMAL. Use ``_prereform_model``
    #      (the true DOF) when it exists.
    #   2. Slice every node box to the original columns ``[:_reduced_n_orig]``. Aux
    #      columns are appended after the originals (comment at the reformulation
    #      site), so the first ``_reduced_n_orig`` flat entries of every tree-exported
    #      node box are exactly the original-variable sub-box.
    _reduced_model = _prereform_model if _prereform_model is not None else model
    if _prereform_model is not None:
        _reduced_n_orig = int(_prereform_nvars)
    else:
        _reduced_n_orig = int(sum(v.size for v in model._variables))
    _reduced_bound_fn: Any = None
    if _reduced_space_active:
        try:
            from discopt._jax.mccormick_subgradient import (
                UnsupportedRelaxation as _RSUnsupported,
            )
            from discopt._jax.mccormick_subgradient import (
                reduced_mccormick_lp_bound as _reduced_bound_fn,
            )

            # Probe buildability once on the root box: a model out of MCBox scope
            # surfaces here (status "unsupported") and we fall back for the whole
            # solve rather than paying the probe cost at every node.
            _rs_probe = _reduced_bound_fn(
                _reduced_model,
                np.asarray(lb, dtype=np.float64)[:_reduced_n_orig],
                np.asarray(ub, dtype=np.float64)[:_reduced_n_orig],
                max_rounds=1,
            )
            if _rs_probe.status == "unsupported":
                logger.info(
                    "DISCOPT_RELAX_SPACE=reduced: model outside sound reduced-space "
                    "(MCBox) scope; falling back to the lifted McCormick path for "
                    "this solve."
                )
                _reduced_space_active = False
                _reduced_bound_fn = None
        except _RSUnsupported as _rs_exc:
            logger.info(
                "DISCOPT_RELAX_SPACE=reduced: reduced-space build refused (%s); "
                "falling back to the lifted McCormick path for this solve.",
                _rs_exc,
            )
            _reduced_space_active = False
            _reduced_bound_fn = None
        except Exception as _rs_exc:  # pragma: no cover - defensive
            logger.warning(
                "DISCOPT_RELAX_SPACE=reduced: reduced-space setup failed (%s); "
                "falling back to the lifted McCormick path for this solve.",
                _rs_exc,
            )
            _reduced_space_active = False
            _reduced_bound_fn = None
        if _reduced_space_active:
            logger.debug("reduced-space McCormick per-node bounding active (P2.3)")

    def _reduced_node_bound(_node_lb, _node_ub, _i):
        """Compute the reduced-space node dual bound over the given node box.

        Returns ``(kind, value)`` where ``kind`` is:
          * ``"infeasible"`` — rigorous fathom (relaxed feasible set empty);
          * ``"bound"`` — ``value`` is a VALID node dual lower bound (min sense);
          * ``None`` — no reduced bound available (unsupported/unbounded/error);
            the caller must not fathom and should keep any other bound source.

        Correctness (plan §0.3, CLAUDE.md §1): only ``optimal``/``infeasible`` are
        acted on; a non-finite ``optimal`` bound is dropped (never trusted). This
        can only raise a node bound toward — never above — the true box optimum.

        The node box spans the lifted space; slice to the original variables
        (``[:_reduced_n_orig]``) — the reduced evaluator is defined over the
        original variables only (see the setup block's rationale).
        """
        try:
            rb = _reduced_bound_fn(
                _reduced_model,
                np.asarray(_node_lb, dtype=np.float64)[:_reduced_n_orig],
                np.asarray(_node_ub, dtype=np.float64)[:_reduced_n_orig],
            )
        except Exception as _rb_exc:  # pragma: no cover - defensive
            logger.debug("reduced-space bound failed at node %d: %s", _i, _rb_exc)
            return None, None
        if rb.status == "infeasible":
            return "infeasible", None
        if rb.status == "optimal" and rb.bound is not None and np.isfinite(rb.bound):
            return "bound", float(rb.bound)
        return None, None

    from discopt import debug as _debug

    def _debug_validate_candidate(
        x,
        _ev=evaluator,
        _cl=cl_list,
        _cu=cu_list,
        _ioff=int_offsets,
        _isz=int_sizes,
    ):
        """Validate a debugger-injected candidate against the ORIGINAL problem.

        ``inject_incumbent`` trusts its caller (no feasibility re-check), so
        the debugger's ``inject`` steer must verify integrality + constraint
        feasibility here and hand the tree the point's true evaluated
        objective — never a relaxation bound (CLAUDE.md §1). Returns
        ``(feasible, x_validated, obj)``; pure reads only.
        """
        xv = np.asarray(x, dtype=np.float64).copy()
        if _ioff and not _is_integer_feasible_solution(xv, _ioff, _isz):
            return False, xv, float("nan")
        for _o, _s in zip(_ioff, _isz):
            xv[_o : _o + _s] = np.round(xv[_o : _o + _s])
        if _cl and not _check_constraint_feasibility(_ev, xv, _cl, _cu):
            return False, xv, float("nan")
        _obj = float(_ev.evaluate_objective(xv))
        if not np.isfinite(_obj) or _obj >= _SENTINEL_THRESHOLD:
            return False, xv, float("nan")
        return True, xv, _obj

    # Lazy re-separation governor state (C-42 Part 2; see the module-level
    # ``_LAZY_RESEP_*`` constants). Touched only under active pool inheritance.
    _lazy_glb_ref: Optional[float] = None
    _lazy_armed = False
    _lazy_stagnant_solves = 0
    _lazy_probe_spent = 0
    _lazy_mode = "idle"
    _lazy_resep_fires = 0

    # Set when the interactive debugger's `quit` breaks the search loop: a
    # user-interrupted exit proves nothing, so the status decision below must
    # not fall through to a certified "infeasible"/"optimal".
    _debug_quit = False
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Interactive debugger: top-of-iteration checkpoint (no-op when detached).
        if _debug.fire(
            _debug.Checkpoint.ITER_START,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=elapsed,
        ):
            _debug_quit = True
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

        # C-42 Part 2 — lazy re-separation, global-bound-stall governor. Under
        # pool inheritance the square/PSD point separators are skipped at
        # nodes; on the tspn05 class that skip is load-bearing for closure
        # (#552: the tree's bound freezes and the certificate is lost at
        # budget). The governor watches the tree's GLOBAL lower bound:
        #   * ``idle``    — the bound improved within the last
        #     ``_LAZY_RESEP_STALL_WINDOW`` node solves: keep the cheap
        #     pool-only path (this is what preserves #551's 2–5× throughput
        #     win on nvs19/nvs24);
        #   * ``probing`` — the bound had been moving (the governor is ARMED
        #     by a genuine in-tree improvement) and then stagnated for a full
        #     window: re-enable the full separation pass for up to
        #     ``_LAZY_RESEP_PROBE_BUDGET`` node solves;
        #   * ``muted``   — the probe did not move the bound either (the
        #     nvs24 signature: separation is measured bound-inert there):
        #     return to pool-only until the bound next improves, which resets
        #     the cycle (self-healing on phase changes).
        # An improvement while ``probing`` REFRESHES the probe (separation is
        # demonstrably moving the bound right now, so keep it engaged — this
        # is what lets tspn05 close instead of rationing separation to an
        # 8-in-32 duty cycle); a probe still exits after a full budget of
        # improvement-free separated solves, which caps the cost of a
        # misjudged lock-in. An improvement in any other mode resets to
        # ``idle``. Enabling separation only ever TIGHTENS a node's relaxation
        # (every cut is valid), so the governor is performance-only —
        # soundness is unaffected in every state. Evaluated only under active
        # pool inheritance: the default path reads no extra state and is
        # byte-identical.
        _lazy_probing = False
        if _cut_inherit_enabled and _root_cut_pool is not None:
            try:
                _glb_now = float(tree.stats()["global_lower_bound"])
            except Exception:  # pragma: no cover - defensive
                _glb_now = -np.inf
            if np.isfinite(_glb_now) and (
                _lazy_glb_ref is None
                or _glb_now > _lazy_glb_ref + _LAZY_RESEP_GLB_EPS * max(1.0, abs(_lazy_glb_ref))
            ):
                if _lazy_glb_ref is not None:
                    # A genuine in-tree improvement (not the first finite
                    # reference) ARMS the governor: a stall is only a
                    # meaningful signal once the bound has demonstrably been
                    # moving. A bound that has never moved since the root is
                    # the pool-at-fixed-point signature (nvs24: the root pool
                    # is separated to convergence and per-node re-separation
                    # is measured bound-inert), where a probe only burns the
                    # most expensive separation wall in the corpus; the
                    # stride net remains the unconditional prober there.
                    _lazy_armed = True
                _lazy_glb_ref = _glb_now
                _lazy_stagnant_solves = 0
                _lazy_probe_spent = 0
                if _lazy_mode != "probing":
                    _lazy_mode = "idle"
            if (
                _lazy_mode == "idle"
                and _lazy_armed
                and _lazy_stagnant_solves >= _LAZY_RESEP_STALL_WINDOW
            ):
                _lazy_mode = "probing"
            elif _lazy_mode == "probing" and _lazy_probe_spent >= _LAZY_RESEP_PROBE_BUDGET:
                _lazy_mode = "muted"
            _lazy_probing = _lazy_mode == "probing"
            _lazy_stagnant_solves += n_batch
            if _lazy_probing:
                _lazy_probe_spent += n_batch
                _lazy_resep_fires += n_batch

        # Interactive debugger: nodes selected — boxes/ids now available.
        if _debug.fire(
            _debug.Checkpoint.AFTER_SELECT,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=elapsed,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
        ):
            _debug_quit = True
            break

        node_infeasible_mask = np.zeros(n_batch, dtype=bool)

        # Per-node reduce (cert:T2.4b) staging for THIS batch: the incumbent cutoff
        # and a {batch_index: (lb, ub)} map of reduced child boxes, applied via
        # set_node_bounds just before the tree branches (below).
        _nr_pending: dict = {}
        _nr_cutoff = None
        if _node_reduce_enabled:
            _nr_inc = tree.incumbent()
            _nr_cutoff = (
                float(_nr_inc[1])
                if _nr_inc is not None
                and np.isfinite(_nr_inc[1])
                and _nr_inc[1] < _SENTINEL_THRESHOLD
                else None
            )

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
            _t_fbbt = time.perf_counter()
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
            _reduce_timers["fbbt"] += time.perf_counter() - _t_fbbt

        # --- In-tree presolve / branch-and-reduce (PF1, issue #632) ---
        # Wire the persistent Rust FBBT(+reduce) kernel into the GLOBAL spatial
        # B&B node loop, gated by ``in_tree_presolve_stride`` (same kwarg the
        # NLP-BB path uses). Before this the kernel fired only on ``_solve_nlp_bb``
        # (convex path); every unproved spatial instance lives here and never saw
        # cutoff-aware in-tree FBBT / branch-and-reduce. The kernel:
        #   * runs cutoff-aware FBBT with the incumbent as a valid upper bound, so
        #     it discards only points that cannot improve the incumbent (sound
        #     inside B&B), then
        #   * optionally (probing, default OFF) contracts discrete domains on
        #     proven-infeasible fixings.
        # SOUNDNESS: every contraction is outward-rounded interval propagation
        # applied as an INTERSECTION with the current box (``run_in_tree_presolve``
        # never loosens); a proven-empty box fathoms the node. A reduced box thus
        # always still contains the entire feasible region — no feasible point,
        # and never the optimum, is ever excluded. Best-effort: silently skipped
        # when the repr's block count doesn't match the node box (e.g. a
        # polynomial-reformulated repr with extra aux vars) — the intersect-only
        # floor keeps it sound even then.
        if in_tree_presolve_stride and _model_repr is not None:
            try:
                _itp_n_blocks = _model_repr.n_var_blocks
                _itp_probing = _node_probing_enabled()
                _itp_probe_max = _node_probe_max_vars()
                _itp_inc = tree.incumbent()
                _itp_cutoff = (
                    float(_itp_inc[1])
                    if _itp_inc is not None
                    and np.isfinite(_itp_inc[1])
                    and _itp_inc[1] < _SENTINEL_THRESHOLD
                    else None
                )
                _itp_depths = tree.node_depths(np.asarray(batch_ids, dtype=np.int64))
                _t_itp = time.perf_counter()
                for i in range(n_batch):
                    if node_infeasible_mask[i]:
                        continue
                    if len(batch_lb[i]) != _itp_n_blocks:
                        continue
                    _itp_delta = _model_repr.in_tree_presolve(
                        np.asarray(batch_lb[i], dtype=np.float64),
                        np.asarray(batch_ub[i], dtype=np.float64),
                        node_depth=int(_itp_depths[i]),
                        depth_stride=in_tree_presolve_stride,
                        incumbent=_itp_cutoff,
                        probing=_itp_probing,
                        probe_max_vars=_itp_probe_max,
                    )
                    if not _itp_delta["ran"]:
                        continue
                    _IN_TREE_PRESOLVE_GLOBAL_CALLS += 1
                    if _itp_delta["infeasible"]:
                        # Rigorous fathom: the node box is empty (FBBT/probing
                        # proof), so its subtree holds no feasible point.
                        node_infeasible_mask[i] = True
                    else:
                        batch_lb[i] = list(_itp_delta["lb"])
                        batch_ub[i] = list(_itp_delta["ub"])
                _reduce_timers["fbbt"] += time.perf_counter() - _t_itp
            except Exception as _itp_exc:  # pragma: no cover - defensive
                logger.debug("global in-tree presolve skipped: %s", _itp_exc)

        # --- Per-node OBBT (Lever A) ---
        # Tighten each surviving node's box against its own McCormick relaxation
        # (optionally with the incumbent objective as a cutoff). The relaxation
        # is built over the node box, so the resulting bounds are valid for the
        # node's subtree — rigorous, not heuristic. For the dependent-
        # intermediate class this pins the functionally-dependent outputs from
        # the relaxation, which is what lets a node fathom once its independent
        # drivers are branched (welded-beam / nvs05). Gated + budgeted at setup;
        # here we additionally stop as soon as the cumulative budget is spent.
        if _per_node_obbt_enabled and _pn_obbt_spent < _pn_obbt_budget_total:
            _pn_inc = tree.incumbent()
            _pn_cutoff = (
                float(_pn_inc[1])
                if _pn_inc is not None
                and np.isfinite(_pn_inc[1])
                and _pn_inc[1] < _SENTINEL_THRESHOLD
                else None
            )
            for i in range(n_batch):
                # Skip per-node OBBT on the ROOT batch (iteration 0). It is redundant
                # there with the global root OBBT just run, and — at up to 60% of the
                # time budget (4.8s on an 8s limit) — it runs *before* the root
                # primal heuristic, pushing time-to-first-incumbent past the limit on
                # nonconvex models (issue #287: kall's first incumbent landed at
                # 10.4s on an 8s budget). Its value is on *branched* nodes, where it
                # pins the functionally-dependent outputs once the independent
                # drivers are fixed (welded-beam / nvs05 fathoming) — those run from
                # iteration 1 on, so deferring the root costs nothing there.
                if iteration == 0:
                    break
                if node_infeasible_mask[i]:
                    continue
                if _pn_obbt_spent >= _pn_obbt_budget_total:
                    break
                _t_pn = time.perf_counter()
                if time_limit - (_t_pn - t_start) < _DEADLINE_NODE_FLOOR_S:
                    break
                try:
                    _pn_res = obbt_tighten_root(
                        model,
                        np.asarray(batch_lb[i], dtype=np.float64),
                        ub=np.asarray(batch_ub[i], dtype=np.float64),
                        rounds=_PER_NODE_OBBT_ROUNDS,
                        incumbent_cutoff=_pn_cutoff,
                        deadline=_t_pn + _PER_NODE_OBBT_PER_NODE_S,
                        time_limit_per_lp=_PER_NODE_OBBT_PER_LP_S,
                        prefer_pounce=nlp_solver == "pounce",
                        top_k=_pn_obbt_topk,
                    )
                except Exception as _pn_exc:  # pragma: no cover - defensive
                    logger.debug("per-node OBBT failed: %s", _pn_exc)
                    _pn_res = None
                finally:
                    _pn_obbt_spent += time.perf_counter() - _t_pn
                if _pn_res is None:
                    continue
                if _pn_res.infeasible:
                    node_infeasible_mask[i] = True
                    continue
                if _pn_res.n_tightened > 0:
                    batch_lb[i] = np.asarray(_pn_res.lb, dtype=np.float64).tolist()
                    batch_ub[i] = np.asarray(_pn_res.ub, dtype=np.float64).tolist()

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
            # B2-FIX (task #89): the possibly-invalid value stays in result_lbs
            # and enters the tree as the node's bound, so the frontier minimum
            # itself is no longer trustworthy — poison the tree bound (full
            # discard at result build; the taint-floor recovery must not apply).
            if _model_is_convex and not bool(np.all(_batch_trusted)):
                _gap_certified = False
                _tree_bound_poisoned = True
                _convex_bound_untrusted = True
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
                                evaluator, model, _alphabb_expr, node_lb_i, node_ub_i
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
                    # (The removed "midpoint" mode returned cv(mid), which is NOT a
                    # valid lower bound on the box minimum — correctness issue C-18;
                    # "nlp" is now the only McCormick objective-bound mode here.)
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
            # Reduced-space McCormick per-node bound (P2.3). Replaces the lifted
            # LP for nodes where it applies; nodes it cannot bound (None) fall
            # through to the lifted block below unchanged.
            _reduced_done = np.zeros(n_batch, dtype=bool)
            if _reduced_space_active:
                for i in range(n_batch):
                    if node_infeasible_mask[i]:
                        continue
                    if _deadline - time.perf_counter() <= 0.0:
                        break
                    _rk, _rv = _reduced_node_bound(batch_lb[i], batch_ub[i], i)
                    if _rk == "infeasible":
                        # Rigorous fathom: the reduced-space relaxation is a valid
                        # outer relaxation, so an empty relaxed feasible set proves
                        # the node's subtree infeasible (mirrors the lifted path).
                        node_infeasible_mask[i] = True
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        result_feas[i] = False
                        _reduced_done[i] = True
                    elif _rk == "bound":
                        # Valid node dual lower bound. Combine soundly (max) with
                        # any bound already present; reduced REPLACES the lifted
                        # LP solve for this node (the storm-avoidance win).
                        cur = result_lbs[i]
                        if cur >= _SENTINEL_THRESHOLD or not np.isfinite(cur):
                            result_lbs[i] = _rv
                            result_feas[i] = False
                        else:
                            result_lbs[i] = max(cur, _rv)
                        _reduced_done[i] = True
                    # _rk is None: no reduced bound (unsupported/unbounded); leave
                    # the node for the lifted LP fallback below.

            # LP-form McCormick: lift bilinears, solve as LP via HiGHS.
            # Per-node, ~20ms for problems with tens of bilinear terms.
            if _mc_lp_relaxer is not None:
                for i in range(n_batch):
                    if node_infeasible_mask[i]:
                        continue
                    if _reduced_done[i]:
                        # Reduced-space already produced this node's bound/fathom.
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
                        # Past the per-node deadline: skip this node's relaxation
                        # solve. It stays OPEN carrying its inherited (valid) parent
                        # bound — the Rust import floors every node at its parent's
                        # lower bound — so the tree's dual bound stays valid. Leaving
                        # a node unbounded never fathoms it, so do NOT decertify (that
                        # only discarded the bound the parent already proved, #138).
                        continue
                    nlp_failed = result_lbs[i] >= _SENTINEL_THRESHOLD
                    try:
                        mc_res = _mc_lp_relaxer.solve_at_node(
                            np.asarray(batch_lb[i]),
                            np.asarray(batch_ub[i]),
                            time_limit=max(_node_remaining, _DEADLINE_NODE_FLOOR_S),
                            inherited_cuts=_root_cut_pool,
                            separate=True,
                            want_marginals=_node_reduce_enabled,
                            # THRU-4: with the root pool inherited, skip the per-node
                            # square/PSD point-separation loops (sound: their cut
                            # families are box-independent and already in the pool)
                            # — unless the global-stall governor is probing (C-42).
                            skip_pool_separators=(
                                _cut_inherit_enabled
                                and _root_cut_pool is not None
                                and not _lazy_probing
                            ),
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
                    # --- Per-node reduce (cert:T2.4b) ---
                    # Reduce the node box from THIS solve's marginals (no extra LP)
                    # plus cutoff-FBBT, and stage the tightened box for the child
                    # export (applied via set_node_bounds before process_evaluated).
                    if _node_reduce_enabled and _node_reduce_fn is not None:
                        _nr_res = _reduce_node_and_stage(
                            _node_reduce_fn,
                            model,
                            i,
                            batch_lb,
                            batch_ub,
                            mc_res,
                            tree,
                            _nr_cutoff,
                            _nr_pending,
                        )
                        if _nr_res:
                            node_infeasible_mask[i] = True
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            result_feas[i] = False
                            continue
            if not _model_is_convex:
                for i in range(n_batch):
                    if result_lbs[i] == -np.inf:
                        # A node left unbounded (no valid relaxation bound this
                        # round) stays OPEN and is floored at its inherited parent
                        # bound on import — still a valid global lower bound. It does
                        # not fathom anything, so it does NOT taint the tree (#138).
                        pass
                    elif not np.isfinite(result_lbs[i]):
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                    elif _nonrigorous_sentinel_fathom(result_lbs[i], node_infeasible_mask[i]):
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
            _nlp_stride = _tuning().node_nlp_stride
            _gate_node_nlp = _mc_lp_relaxer is not None and not _model_is_convex and _nlp_stride > 1

            # TX1: effective stride. When adaptive back-off is ON and we are in the
            # throttle-safe gated regime, use the (persisted, back-off-grown)
            # effective stride instead of the fixed one; seed it from the base
            # stride on first use. Adaptive-off, or outside the gated regime, this
            # is byte-identical to today's fixed-stride behavior.
            _eff_nlp_stride = _nlp_stride
            if _adaptive_nlp and _gate_node_nlp:
                if _adaptive_nlp_state["eff_stride"] <= 0:
                    _adaptive_nlp_state["eff_stride"] = _nlp_stride
                _eff_nlp_stride = _adaptive_nlp_state["eff_stride"]

            # TX1: snapshot the incumbent so we can attribute any improvement over
            # this batch's per-node loop to the strided node-NLP (the ONLY
            # inject_incumbent site inside the loop below); track whether it fired.
            _adapt_inc_pre = tree.incumbent()
            _adapt_inc_pre_val = float(_adapt_inc_pre[1]) if _adapt_inc_pre is not None else None
            _adapt_nlp_fired = False

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
                    # The node stays OPEN at -inf and is floored at its inherited
                    # parent bound on import (a valid global lower bound). It fathoms
                    # nothing, so do NOT decertify — that only discarded the parent's
                    # proven bound and forced the weak root fallback (#138).
                    continue
                opts["max_wall_time"] = max(_node_remaining, _DEADLINE_NODE_FLOOR_S)

                # Strided primal NLP (lever 2): run the per-node NLP only on a
                # fraction of nodes in the gated regime; every node still gets the
                # LP relaxer's bound + primal below.
                _node_nlp_due = (not _gate_node_nlp) or (int(batch_ids[i]) % _eff_nlp_stride == 0)
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
                        deadline=_deadline,
                        observe_cost=_observe_heur_nlp,
                    )
                elif iteration > 0 and _node_nlp_due:
                    if _gate_node_nlp:
                        _adapt_nlp_fired = True
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
                        convex=_model_is_convex,
                    )

                result_ids[i] = int(batch_ids[i])

                # C-13: default a convex node to "trusted" (valid NLP lower bound);
                # the ITERATION_LIMIT branch below clears it when the node NLP did
                # not converge to a KKT point, which decertifies the gap after every
                # bound source has had its chance (mirrors the nonconvex finalize).
                _serial_nlp_trusted = True

                if nlp_result is not None and nlp_result.status in (
                    SolveStatus.OPTIMAL,
                    SolveStatus.ITERATION_LIMIT,
                ):
                    nlp_obj = float(nlp_result.objective)
                    # C-13: for a CONVEX model the node NLP objective is used as a
                    # rigorous lower bound — but that is legitimate only when the
                    # solve actually converged to a KKT point (SolveStatus.OPTIMAL).
                    # An interior-point iterate stopped at ITERATION_LIMIT can sit
                    # strictly ABOVE the true node minimum (non-KKT, unconverged
                    # duals), so its objective is NOT a valid lower bound; trusting
                    # it can fathom the subtree holding the optimum while the gap
                    # stays certified → false "optimal". The polish-retry inside
                    # _solve_node_nlp (convex=True above) already tried to reach KKT;
                    # if it still returned ITERATION_LIMIT, ABSTAIN from the NLP
                    # bound (fall back to the valid relaxation/interval bounds
                    # accumulated in convex_lb, or -inf → node stays open at its
                    # inherited parent bound, fathoming nothing) and decertify the
                    # gap. This mirrors the batch path's _batch_trusted guard
                    # (roadmap P0.3) and _solve_nlp_bb's ITERATION_LIMIT decertify.
                    _serial_nlp_trusted = (not _model_is_convex) or (
                        nlp_result.status == SolveStatus.OPTIMAL
                    )
                    nlp_lb = nlp_obj if _serial_nlp_trusted else -np.inf
                    convex_lb = -np.inf  # accumulate valid convex lower bound

                    if _use_alphabb:
                        try:
                            relax_lb = _compute_alphabb_bound(
                                evaluator, model, _alphabb_expr, node_lb, node_ub
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
                                )
                            else:
                                # The removed "midpoint" mode returned cv(mid),
                                # which is NOT a valid lower bound (C-18); "nlp" is
                                # the only McCormick objective-bound mode.
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

                # Reduced-space McCormick per-node bound (P2.3, serial path).
                # Replaces the lifted LP where it applies; a node it cannot bound
                # (None) falls through to the lifted block unchanged.
                _reduced_done_serial = False
                if _reduced_space_active and not node_infeasible_mask[i]:
                    _rk, _rv = _reduced_node_bound(node_lb, node_ub, i)
                    if _rk == "infeasible":
                        node_infeasible_mask[i] = True
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        result_feas[i] = False
                        _reduced_done_serial = True
                    elif _rk == "bound":
                        cur = result_lbs[i]
                        if cur >= _SENTINEL_THRESHOLD or not np.isfinite(cur):
                            result_lbs[i] = _rv
                            result_feas[i] = False
                        else:
                            result_lbs[i] = max(cur, _rv)
                        _reduced_done_serial = True

                # LP-form McCormick bound (lifted bilinears, HiGHS LP).
                # Runs independently of the NLP: provides a valid lower bound
                # and a feasible LP point usable for spatial branching even
                # when the NLP was skipped (root) or returned infeasible /
                # iteration_limit (any node).
                if _mc_lp_relaxer is not None and not _reduced_done_serial:
                    try:
                        mc_lp_res = _mc_lp_relaxer.solve_at_node(
                            node_lb,
                            node_ub,
                            time_limit=max(_deadline - time.perf_counter(), _DEADLINE_NODE_FLOOR_S),
                            inherited_cuts=_root_cut_pool,
                            separate=True,
                            want_marginals=_node_reduce_enabled,
                            # THRU-4: with the root pool inherited, skip the per-node
                            # square/PSD point-separation loops (sound: their cut
                            # families are box-independent and already in the pool)
                            # — unless the global-stall governor is probing (C-42).
                            skip_pool_separators=(
                                _cut_inherit_enabled
                                and _root_cut_pool is not None
                                and not _lazy_probing
                            ),
                        )
                    except Exception as e:
                        logger.debug("McCormick LP failed at node %d: %s", int(batch_ids[i]), e)
                        mc_lp_res = None
                    if (
                        _node_reduce_enabled
                        and _node_reduce_fn is not None
                        and mc_lp_res is not None
                        and mc_lp_res.status != "infeasible"
                    ):
                        # Per-node reduce (cert:T2.4b): tighten this node's box from
                        # the marginals + cutoff-FBBT and stage the child box.
                        if _reduce_node_and_stage(
                            _node_reduce_fn,
                            model,
                            i,
                            batch_lb,
                            batch_ub,
                            mc_lp_res,
                            tree,
                            _nr_cutoff,
                            _nr_pending,
                        ):
                            node_infeasible_mask[i] = True
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            result_feas[i] = False
                            continue
                        # The serial path uses ``node_lb``/``node_ub`` locals for
                        # the subsequent branching/feasibility logic; refresh them
                        # from the (possibly) reduced batch box.
                        node_lb = np.asarray(batch_lb[i], dtype=np.float64)
                        node_ub = np.asarray(batch_ub[i], dtype=np.float64)
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
                #   * -inf  : no bound source produced a finite lower bound this
                #             round. A NON-ROOT node stays OPEN at -inf and is
                #             floored at its inherited parent bound on import (still
                #             a valid global lower bound); it fathoms nothing, so it
                #             does NOT taint the tree — leave the gap certifiable
                #             (#138). The ROOT has no parent to floor against: if it
                #             also has no finite spatial-branch direction (an
                #             unbounded-below / free-variable root), the Rust tree
                #             cannot branch it and would previously fathom it and
                #             collapse `global_lower_bound` to the incumbent, falsely
                #             certifying a local/near-feasible point as optimal. That
                #             collapse is now blocked in the Rust tree
                #             (`bound_unresolved` in `tree_manager.rs`, issue #467):
                #             the global bound is pinned at -inf, so the gap is
                #             infinite and the run downgrades to "feasible" via the
                #             existing status logic. No Python change is needed here.
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
                        # Non-root: stays open, floored at the parent bound on
                        # import — does not taint. Root with no branch direction:
                        # the Rust tree pins the global bound at -inf instead of
                        # collapsing to the incumbent (#467), so this remains sound.
                        pass
                    elif not np.isfinite(result_lbs[i]):
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                    elif _nonrigorous_sentinel_fathom(result_lbs[i], node_infeasible_mask[i]):
                        _gap_certified = False

                # C-13: convex node whose NLP objective was NOT a valid lower bound
                # (non-KKT ITERATION_LIMIT, unrescued by the polish-retry).  The
                # bound was already abstained above (nlp_lb=-inf, so the node imports
                # at its inherited parent bound — no unsound fathom), but the gap can
                # no longer be certified optimal: an under-converged relaxation
                # objective proves nothing about the subtree, and for a convex model
                # the NLP objective is typically the ONLY bound source (alphaBB is
                # nonconvex-only; McCormick/LP relaxers are usually absent).
                # Decertify unconditionally on any untrusted convex node — the same
                # conservative guard the batch path applies via _batch_trusted
                # (roadmap P0.3) and that _solve_nlp_bb applies on ITERATION_LIMIT.
                if _model_is_convex and not node_infeasible_mask[i] and not _serial_nlp_trusted:
                    _gap_certified = False
                    _convex_bound_untrusted = True

            # TX1: adaptive back-off update. Only when the strided node-NLP fired
            # this batch (gated regime): if it did NOT improve the incumbent for
            # PATIENCE consecutive fired batches, double the effective stride (up to
            # the cap); reset to the base stride the moment it improves. Purely a
            # heuristic-schedule change — the dual bound / gap are untouched.
            if _adaptive_nlp and _gate_node_nlp and _adapt_nlp_fired:
                _adapt_inc_post = tree.incumbent()
                _adapt_inc_post_val = (
                    float(_adapt_inc_post[1]) if _adapt_inc_post is not None else None
                )
                _adapt_improved = _adapt_inc_post_val is not None and (
                    _adapt_inc_pre_val is None
                    or _adapt_inc_post_val < _adapt_inc_pre_val - _DEFAULT_ABS_GAP_TOL
                )
                if _adapt_improved:
                    if _adaptive_nlp_state["eff_stride"] != _nlp_stride:
                        logger.debug(
                            "TX1 adaptive node-NLP: incumbent improved, reset stride %d -> %d",
                            _adaptive_nlp_state["eff_stride"],
                            _nlp_stride,
                        )
                    _adaptive_nlp_state["eff_stride"] = _nlp_stride
                    _adaptive_nlp_state["no_improve"] = 0
                else:
                    _adaptive_nlp_state["no_improve"] += 1
                    if _adaptive_nlp_state["no_improve"] >= _ADAPTIVE_NLP_PATIENCE:
                        _new_stride = min(_eff_nlp_stride * 2, _ADAPTIVE_NLP_STRIDE_CAP)
                        if _new_stride != _adaptive_nlp_state["eff_stride"]:
                            logger.debug(
                                "TX1 adaptive node-NLP: no improvement, back off stride %d -> %d",
                                _adaptive_nlp_state["eff_stride"],
                                _new_stride,
                            )
                        _adaptive_nlp_state["eff_stride"] = _new_stride
                        _adaptive_nlp_state["no_improve"] = 0
        jax_time += time.perf_counter() - t_jax_start

        # C-1 (path-agnostic, covers convex + nonconvex, batch + serial): any node
        # entering the tree with the failure sentinel but WITHOUT a rigorous
        # infeasibility certificate (``node_infeasible_mask`` — an empty McCormick/
        # LP relaxation over the finite box) is being fathomed non-rigorously. Its
        # subtree is not proven empty, so if the tree later exhausts with no
        # incumbent we must not declare the model "infeasible". The per-site flags
        # above already cover the nonconvex decertify branches; this final sweep is
        # the authoritative guard and also catches the convex path (whose local NLP
        # objective is a valid bound but whose constraint-violating / failed nodes
        # still get sentinelled with no proof).
        #
        # B2-FIX (task #89): every such node also contributes its POP-TIME lower
        # bound (still stored in the Rust pool — import_results has not run yet
        # for this batch, so ``node_lower_bounds`` returns the bound the node was
        # popped with, proved at its parent) to ``_taint_floor_internal``. That
        # floor keeps the unproven subtree represented in the reported global
        # dual bound after the sentinel import removes it from the frontier,
        # instead of discarding the entire tree bound (DECOMP-1
        # "decertify-and-discard"). Fetched lazily — zero Rust calls on a clean
        # batch — and read-only, so node processing is byte-identical.
        _taint_pop_lbs = None
        for i in range(n_batch):
            if result_lbs[i] >= _SENTINEL_THRESHOLD and not node_infeasible_mask[i]:
                _nonrigorous_fathom = True
                if _taint_pop_lbs is None:
                    _taint_pop_lbs = np.asarray(
                        tree.node_lower_bounds(np.asarray(result_ids, dtype=np.int64)),
                        dtype=np.float64,
                    )
                _taint_floor_internal = min(_taint_floor_internal, float(_taint_pop_lbs[i]))
                logger.debug(
                    "Non-rigorous sentinel fathom at node %d (iteration %d): "
                    "pop-time bound %.6g kept as a floor of the reported global bound",
                    int(result_ids[i]),
                    iteration,
                    float(_taint_pop_lbs[i]),
                )

        if np.any(node_infeasible_mask):
            for idx in np.flatnonzero(node_infeasible_mask):
                i = int(idx)
                result_lbs[i] = _INFEASIBILITY_SENTINEL
                result_feas[i] = False

        # #467 sub-bug #3: the ROOT batch covers the whole feasible region. If
        # EVERY root node is rigorously infeasible-masked (empty box / empty
        # relaxation over a finite box — the rigorous certificate, not a soft
        # failure sentinel), the model is rigorously proven infeasible at the root.
        # A within-tolerance-only incumbent found by the primal heuristic then lies
        # inside a region the search proved infeasible; the terminal logic must not
        # certify it ``optimal`` (ex7_3_6: FBBT proves the root empty by ~2e-6 > the
        # 1e-6 FBBT tolerance, yet a pump point at ~1.2e-4 residual was certified).
        # Gated on iteration 0 and n_batch >= 1 so it reflects the root, not a deep
        # subtree; ``_nonrigorous_fathom`` guarantees only rigorous certificates are
        # trusted for the eventual ``infeasible`` verdict.
        if (
            iteration == 0
            and n_batch >= 1
            and bool(np.all(node_infeasible_mask[:n_batch]))
            and not _nonrigorous_fathom
        ):
            _root_rigorously_infeasible = True

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

        # --- Root-optimality short-circuit for primal heuristics (#330) ---
        # At the root the minimum finite relaxation bound in this batch is a
        # valid *global* dual bound. Once a heuristic has injected an incumbent
        # that meets it (same gap test as ``_gap_converged``), the optimum is
        # certified and every remaining root primal heuristic is provably-wasted
        # work — no feasible point exists below the bound, so none can improve
        # the incumbent. Skipping them changes neither the returned optimum nor
        # its certification, and only fires on instances already solved at the
        # root (the trivially-easy ones); on harder instances the root gap stays
        # open and every heuristic runs exactly as before, so there is no
        # large-instance regression. Restricted to ``iteration == 0`` because the
        # batch minimum is a valid global bound only there.
        _batch_relax_lb = np.inf
        if iteration == 0:
            for _ii in range(n_batch):
                _lb_ii = result_lbs[_ii]
                if _lb_ii >= _SENTINEL_THRESHOLD or not np.isfinite(_lb_ii):
                    continue
                if _lb_ii < _batch_relax_lb:
                    _batch_relax_lb = float(_lb_ii)

        def _root_optimum_proven() -> bool:
            if iteration != 0 or not np.isfinite(_batch_relax_lb):
                return False
            _inc = tree.incumbent()
            if _inc is None or not np.isfinite(_inc[1]):
                return False
            _ub = float(_inc[1])
            _abs_gap = max(0.0, _ub - _batch_relax_lb)
            if _abs_gap <= _DEFAULT_ABS_GAP_TOL:
                return True
            _denom = max(abs(_ub), abs(_batch_relax_lb), 1e-10)
            return _abs_gap / _denom <= gap_tolerance

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
            if best_root_idx is not None and _root_heur_nlp_entry_ok(evaluator):
                try:
                    from discopt._jax.primal_heuristics import feasibility_pump

                    _t_fp = time.perf_counter()
                    fp_sol = feasibility_pump(
                        model,
                        result_sols[best_root_idx],
                        max_rounds=5,
                        backend=_resolve_heuristic_backend(nlp_solver),
                        evaluator=evaluator,
                        deadline=_deadline,
                    )
                    _observe_heur_nlp(time.perf_counter() - _t_fp)
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

                # --- NLP-relaxation-seeded feasibility pump ---
                # The pump above rounds ``result_sols[best_root_idx]``. When the
                # relaxation bound is supplied by the McCormick *LP* relaxer that
                # seed is an LP vertex, and for integer-heavy nonconvex models an
                # LP vertex can round into a far-suboptimal integer assignment even
                # though the genuine continuous NLP relaxation rounds straight into
                # the global basin (st_e31: LP-vertex rounding parks the incumbent
                # at obj=-1.0 for >100 s while a single NLP-relaxation rounding
                # reaches the true optimum -2.0 immediately). When the LP relaxer is
                # active the root multistart NLP is otherwise skipped (it is not a
                # bound source there), so the pump never sees that point. Solve one
                # root NLP relaxation now purely to supply a better rounding seed.
                # General: any integer-heavy nonconvex MINLP on the LP-relaxer path
                # benefits; no-op for convex models and when no NLP backend exists.
                # Sound: only pump-verified feasible incumbents that strictly
                # improve the current incumbent are injected (``inject_incumbent``
                # enforces strict improvement), so a worse seed can never regress
                # the reported solution.
                if (
                    _mc_lp_relaxer is not None
                    and not _model_is_convex
                    and _root_heur_nlp_entry_ok(_active_evaluator)
                ):
                    try:
                        from discopt._jax.primal_heuristics import feasibility_pump

                        _t_relax = time.perf_counter()
                        _relax_opts = dict(opts)
                        _relax_opts["max_wall_time"] = max(
                            _DEADLINE_NODE_FLOOR_S,
                            min(3.0, _deadline - time.perf_counter()),
                        )
                        _root_relax = _solve_root_node_multistart(
                            _active_evaluator,
                            np.array(batch_lb[best_root_idx]),
                            np.array(batch_ub[best_root_idx]),
                            _active_cb,
                            _relax_opts,
                            nlp_solver,
                            n_random=0,
                            deadline=_deadline,
                            observe_cost=_observe_heur_nlp,
                        )
                        if (
                            _root_relax is not None
                            and _root_relax.x is not None
                            and _root_relax.status
                            in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT)
                        ):
                            fp_sol2 = feasibility_pump(
                                model,
                                np.asarray(_root_relax.x, dtype=float),
                                max_rounds=5,
                                backend=_resolve_heuristic_backend(nlp_solver),
                                evaluator=evaluator,
                                deadline=_deadline,
                            )
                            if fp_sol2 is not None:
                                fp_obj2 = float(evaluator.evaluate_objective(fp_sol2))
                                fp_feas2 = not cl_list or _check_constraint_feasibility(
                                    evaluator, fp_sol2, cl_list, cu_list
                                )
                                if (
                                    np.isfinite(fp_obj2)
                                    and fp_obj2 < _SENTINEL_THRESHOLD
                                    and fp_feas2
                                ):
                                    tree.inject_incumbent(fp_sol2, fp_obj2)
                                    logger.info(
                                        "NLP-relaxation feasibility pump found incumbent: obj=%.6g",
                                        fp_obj2,
                                    )
                    except Exception as e:
                        logger.debug("NLP-relaxation feasibility pump failed: %s", e)
                    finally:
                        _observe_heur_nlp(time.perf_counter() - _t_relax)

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
                # NOT budget-gated: the 1-opt/2-opt lattice search is one of the
                # lighter improvers discopt's (often weak) McCormick relaxation
                # genuinely relies on to reach the global integer assignment
                # (nvs23: -287 → -1125), so it stays on at the root. The budget
                # targets the heavier standalone-strength components below.
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

                # Fractional-diving fallback (issue #268). When neither pump nor
                # local search produced an incumbent, dive: fix one fractional
                # integer at a time and re-solve the NLP between fixings. This
                # finds feasible integer assignments that all-at-once rounding
                # misses on combinatorial MINLPs (m7 and the syn/clay/flay family),
                # so the search returns a feasible incumbent instead of exhausting
                # with nothing. Only when nothing else found one and time remains;
                # bounded by ~n_int sub-NLP solves. Sound: diving re-verifies
                # integer + constraint feasibility and inject_incumbent enforces
                # strict improvement, so the dual bound / gap are untouched.
                if (
                    best_root_idx is not None
                    and tree.incumbent() is None
                    and _root_heur_nlp_entry_ok(evaluator)
                ):
                    try:
                        from discopt._jax.primal_heuristics import fractional_diving

                        _t_dive = time.perf_counter()
                        dv = fractional_diving(
                            model,
                            result_sols[best_root_idx],
                            backend=_resolve_heuristic_backend(nlp_solver),
                            evaluator=evaluator,
                            deadline=_deadline,
                        )
                        _observe_heur_nlp(time.perf_counter() - _t_dive)
                        if dv is not None:
                            _x_dv, _obj_dv = dv
                            _obj_dv = float(_obj_dv)
                            _dv_feas = not cl_list or _check_constraint_feasibility(
                                evaluator, _x_dv, cl_list, cu_list
                            )
                            if np.isfinite(_obj_dv) and _obj_dv < _SENTINEL_THRESHOLD and _dv_feas:
                                tree.inject_incumbent(np.asarray(_x_dv).copy(), _obj_dv)
                                logger.info("Fractional diving found incumbent: obj=%.6g", _obj_dv)
                    except Exception as e:
                        logger.debug("Fractional diving failed: %s", e)

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
            and not _root_optimum_proven()
            # F4: the SubNLP heuristic launches one or more full NLP solves that
            # overrun their own ``max_wall_time`` on the expensive-Hessian class;
            # do not even start it when the budget is exhausted (it is a primal
            # heuristic, so skipping is sound and never touches the dual bound).
            and _root_heur_nlp_entry_ok(evaluator)
        ):
            from discopt._jax.primal_heuristics import subnlp as _subnlp

            # Root only: seed one subnlp from the modeler's GAMS-provided start
            # (parsed into model._gams_initial_values). For nonconvex models the
            # published point often lands in the global basin that generic
            # relaxation/midpoint seeds miss (prob07: 162070 -> 154990). No-op
            # when the model carries no initial values. Sound: subnlp re-verifies
            # feasibility and inject_incumbent enforces strict improvement.
            if (
                iteration == 0
                and _subnlp_calls < subnlp_max_calls
                and _root_heur_nlp_entry_ok(evaluator)
            ):
                _gseed = _gams_initial_seed(model, lb, ub)
                if _gseed is not None:
                    _subnlp_calls += 1
                    _t_sn_g = time.perf_counter()
                    try:
                        _sn = _subnlp(
                            model,
                            _gseed,
                            backend=_subnlp_backend_fn,
                            nlp_options=subnlp_options,
                            evaluator=evaluator,
                            # Single root attempt (no loop): give it a fair budget
                            # so it can converge to a feasible incumbent even if
                            # preprocessing already consumed the deadline.
                            time_budget=min(3.0, float(time_limit)),
                        )
                    except Exception as _e:
                        logger.debug("subnlp (gams seed) raised: %s", _e)
                        _sn = None
                    _observe_heur_nlp(time.perf_counter() - _t_sn_g)
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
            # F4: gate this single fallback solve on the budget too — on the
            # no-relaxation class ``_cands_sn`` is always empty (every bound is a
            # sentinel), so this is the subnlp that fires, and it OVERRUNS its
            # nominal budget on the expensive-Hessian models. Skipping is sound
            # (primal heuristic).
            if not _cands_sn and iteration == 0 and _root_heur_nlp_entry_ok(evaluator):
                _lb_c = np.clip(lb, -_SPC, _SPC)
                _ub_c = np.clip(ub, -_SPC, _SPC)
                _x_seed = 0.5 * (_lb_c + _ub_c)
                _subnlp_calls += 1
                _t_sn_mid = time.perf_counter()
                try:
                    _sn = _subnlp(
                        model,
                        _x_seed,
                        backend=_subnlp_backend_fn,
                        nlp_options=subnlp_options,
                        evaluator=evaluator,
                        # Single root fallback attempt (no loop): give it a fair
                        # budget so it can converge even if preprocessing already
                        # consumed the deadline.
                        time_budget=min(3.0, float(time_limit)),
                    )
                except Exception as _e:
                    logger.debug("subnlp raised: %s", _e)
                    _sn = None
                _observe_heur_nlp(time.perf_counter() - _t_sn_mid)
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
                for _loop_idx, _i in enumerate(_try_idxs):
                    if _subnlp_calls >= subnlp_max_calls:
                        break
                    # The root tries every relaxation candidate to cover all
                    # disjuncts; once one has certified the optimum (incumbent ==
                    # global root bound) the rest are wasted (#330).
                    if _loop_idx > 0 and _root_optimum_proven():
                        break
                    # Deadline enforcement: each subnlp is a round-and-repair NLP
                    # search (seconds on large models) that also OVERRUNS its own
                    # ``max_wall_time`` clamp on the expensive-Hessian class (a
                    # nominal 3 s budget ran ~15 s on heatexch_gen3, F4). At the
                    # root ``_try_idxs`` can hold many candidates, so without a stop
                    # the loop runs well past a tight ``time_limit``. Gate EVERY
                    # candidate (including the first) on the shared budget check:
                    # once the time left cannot absorb another worst-case solve,
                    # stop launching. Skipping subnlp is always sound (primal
                    # heuristic; the dual bound is untouched).
                    _sn_remaining = _deadline - time.perf_counter()
                    if not _root_heur_nlp_entry_ok(evaluator):
                        break
                    _subnlp_calls += 1
                    # The first candidate gets a full (un-clamped) budget so one
                    # NLP solve can actually converge to a feasible incumbent even
                    # if the deadline just passed; later candidates are clamped to
                    # the remaining time so the loop cannot keep overrunning.
                    _sn_budget = (
                        3.0
                        if _loop_idx == 0
                        else min(3.0, max(_DEADLINE_NODE_FLOOR_S, _sn_remaining))
                    )
                    _t_sn = time.perf_counter()
                    try:
                        _sn = _subnlp(
                            model,
                            result_sols[_i],
                            backend=_subnlp_backend_fn,
                            nlp_options=subnlp_options,
                            evaluator=evaluator,
                            time_budget=_sn_budget,
                        )
                    except Exception as _e:
                        logger.debug("subnlp raised: %s", _e)
                        _sn = None
                    _observe_heur_nlp(time.perf_counter() - _t_sn)
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
            and not _root_optimum_proven()
            and _improver_allowed(_HEUR_COST["enumerate"])
            # G2 governor: gate the enumeration only in its improver role (an
            # incumbent already exists); as a finder (no incumbent) gap is open
            # so allowed() lets it through — securing the first incumbent wins.
            and _heuristic_governor.allowed(
                "enumerate", gap_open=tree.incumbent() is None or not _root_optimum_proven()
            )
            # F4: the binary-seed enumeration issues up to 2**k full sub-NLPs; do
            # not start it when the budget is exhausted (primal heuristic — sound).
            and _root_heur_nlp_entry_ok(evaluator)
        ):
            _enum_inc0 = tree.incumbent()
            _enum_had_inc = _enum_inc0 is not None and np.isfinite(_enum_inc0[1])
            _enum_obj0 = float(_enum_inc0[1]) if _enum_had_inc else np.inf
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
            if _enum_had_inc:
                _enum_inc1 = tree.incumbent()
                _enum_improved = _enum_inc1 is not None and float(_enum_inc1[1]) < _enum_obj0 - 1e-9
                _record_improver(_HEUR_COST["enumerate"], _enum_improved)
                _heuristic_governor.record("enumerate", _enum_improved)

        # --- Incumbent integer-neighbourhood search (general-integer LB) ---
        # A feasible incumbent of a nonconvex general-integer model can sit next
        # to a far better feasible integer assignment that no unit (1-opt/2-opt)
        # violation-descent move reaches, because the connecting lattice path is
        # objective-increasing or threads through infeasible cells (nvs05 parks at
        # (3,2)->7.75 while the global (5,1)->5.47 is two coupled steps away over an
        # objective-increasing ridge). When the relaxation drops the cross-terms
        # that would supply a lower bound, B&B has nothing to steer it there in
        # time. integer_box_search re-solves the fixed-integer sub-NLP over the
        # +/-radius box around the incumbent with warm-start propagation, so the
        # better assignment is found directly. Fire whenever the incumbent strictly
        # improves to a new value (the ``_last_box_inc_obj`` guard is the throttle —
        # NOT the subnlp iteration schedule, which can skip the window in which the
        # improving incumbent first appears and was leaving the better assignment on
        # the table on small instances that finish before the next scheduled tick).
        if _subnlp_backend_fn is not None and not _model_is_convex and not _root_optimum_proven():
            # NOT budget-gated: like the lattice search above, the integer box
            # search is a light improver discopt leans on where the relaxation
            # drops the cross-terms that would otherwise bound the search
            # (nvs05: (3,2)→7.75 vs the global (5,1)→5.47). It already self-
            # throttles on ``_last_box_inc_obj`` (one run per new incumbent value).
            _inc_box = tree.incumbent()
            if (
                _inc_box is not None
                and np.isfinite(_inc_box[1])
                and _inc_box[1] < _last_box_inc_obj - 1e-9
                # F4: don't launch the box-search sub-NLPs past the budget.
                and _root_heur_nlp_entry_ok(evaluator)
            ):
                _last_box_inc_obj = float(_inc_box[1])
                from discopt._jax.primal_heuristics import integer_box_search

                _box_budget = min(4.0, max(_DEADLINE_NODE_FLOOR_S, _deadline - time.perf_counter()))
                try:
                    _bx = integer_box_search(
                        model,
                        _inc_box[0],
                        backend=_subnlp_backend_fn,
                        nlp_options=subnlp_options,
                        evaluator=evaluator,
                        time_budget=_box_budget,
                    )
                except Exception as _e:
                    logger.debug("integer_box_search raised: %s", _e)
                    _bx = None
                if _bx is not None and np.isfinite(_bx[1]) and _bx[1] < _inc_box[1] - 1e-9:
                    tree.inject_incumbent(_bx[0].copy(), float(_bx[1]))
                    _subnlp_incumbent_updates += 1
                    _last_box_inc_obj = float(_bx[1])
                    logger.info("Box-search incumbent: obj=%.6g (iter=%d)", _bx[1], iteration)

        # --- Adaptive LNS primal-improvement layer (issue #267) ---
        # On nonconvex MIQPs (the graphpart family) the dual bound is already
        # tight while the incumbent is poor: the optimum is a different INTEGER
        # assignment that root rounding/diving never reaches. This scheduler runs
        # at NON-root nodes too, where tree branching has produced diverse node
        # relaxations, and applies three structured neighbourhoods:
        #   * node-diving (diversify) — dive from this node's best relaxation;
        #   * RINS (improve) — search between incumbent and node relaxation;
        #   * local branching (improve) — Hamming-ball sub-MIP around the
        #     incumbent, escalating k across calls.
        # Adaptive gating keeps it inert where it cannot help: it is skipped for
        # convex / no-integer models and whenever the relative gap is already
        # closed (near-optimal), and every call is bounded by a small time slice
        # and the remaining wall budget. ``_lns_enabled`` is the recursion guard —
        # the local-branching sub-solve sets it False so this layer never nests.
        # Sound: every candidate is re-verified integer- and constraint-feasible
        # and injected only on strict improvement; the dual bound is untouched.
        if (
            _lns_enabled
            and _subnlp_backend_fn is not None
            and not _model_is_convex
            and _lns_has_integers
            and iteration > 0
        ):
            _lns_remaining = _deadline - time.perf_counter()
            if _lns_remaining > _DEADLINE_NODE_FLOOR_S:
                # Best relaxation point at the current batch (lowest node bound).
                _lns_best_idx = None
                _lns_best_obj = np.inf
                for _i in range(n_batch):
                    if result_lbs[_i] < _SENTINEL_THRESHOLD and result_lbs[_i] < _lns_best_obj:
                        _lns_best_obj = result_lbs[_i]
                        _lns_best_idx = _i

                _lns_inc = tree.incumbent()
                _lns_have_inc = (
                    _lns_inc is not None
                    and np.isfinite(_lns_inc[1])
                    and _lns_inc[1] < _SENTINEL_THRESHOLD
                )
                # Is the gap still open? Compute the relative gap against the
                # current dual bound; only run the IMPROVERS when it is open.
                _lns_stats = tree.stats()
                _lns_lb = float(_lns_stats.get("global_lower_bound", float("-inf")))
                _lns_gap_open = True
                if _lns_have_inc and np.isfinite(_lns_lb):
                    _ub = float(_lns_inc[1])
                    _abs_gap = max(0.0, _ub - _lns_lb)
                    _denom = max(abs(_ub), abs(_lns_lb), 1e-10)
                    _lns_gap_open = (_abs_gap > _DEFAULT_ABS_GAP_TOL) and (
                        _abs_gap / _denom > gap_tolerance
                    )

                _lns_backend = _resolve_heuristic_backend(nlp_solver)

                # (1) Node-diving (diversify/find). Dive from this node's best
                # relaxation on a frequency schedule; tree branching supplies the
                # diversity that root-only diving cannot.
                if (
                    _lns_best_idx is not None
                    and iteration % max(1, subnlp_frequency) == 0
                    and _root_heur_nlp_entry_ok(evaluator)
                ):
                    _lns_dive_calls += 1
                    try:
                        from discopt._jax.primal_heuristics import fractional_diving

                        _t_ndive = time.perf_counter()
                        _dv = fractional_diving(
                            model,
                            result_sols[_lns_best_idx],
                            backend=_lns_backend,
                            evaluator=evaluator,
                            deadline=_deadline,
                        )
                        _observe_heur_nlp(time.perf_counter() - _t_ndive)
                        if _dv is not None:
                            _x_dv, _obj_dv = _dv
                            _obj_dv = float(_obj_dv)
                            _dv_feas = not cl_list or _check_constraint_feasibility(
                                evaluator, _x_dv, cl_list, cu_list
                            )
                            if np.isfinite(_obj_dv) and _obj_dv < _SENTINEL_THRESHOLD and _dv_feas:
                                tree.inject_incumbent(np.asarray(_x_dv).copy(), _obj_dv)
                                logger.info("LNS node-diving incumbent: obj=%.6g", _obj_dv)
                    except Exception as _e:
                        logger.debug("LNS node-diving failed: %s", _e)

                # (2) RINS (improve). Only when an incumbent exists and the gap is
                # open — search the neighbourhood between incumbent and the node
                # relaxation.
                if (
                    _lns_have_inc
                    and _lns_gap_open
                    and _lns_best_idx is not None
                    and (_deadline - time.perf_counter()) > _DEADLINE_NODE_FLOOR_S
                    and _improver_allowed(_HEUR_COST["rins"])
                    and _heuristic_governor.allowed("rins", gap_open=_lns_gap_open)
                ):
                    _rins_obj0 = float(_lns_inc[1])
                    _rins_improved = False
                    try:
                        from discopt._jax.primal_heuristics import rins

                        _ri = rins(
                            model,
                            np.asarray(_lns_inc[0], dtype=np.float64),
                            result_sols[_lns_best_idx],
                            backend=_lns_backend,
                            evaluator=evaluator,
                            deadline=_deadline,
                        )
                        if _ri is not None:
                            _x_ri, _obj_ri = _ri
                            _obj_ri = float(_obj_ri)
                            _ri_feas = not cl_list or _check_constraint_feasibility(
                                evaluator, _x_ri, cl_list, cu_list
                            )
                            if np.isfinite(_obj_ri) and _obj_ri < _SENTINEL_THRESHOLD and _ri_feas:
                                tree.inject_incumbent(np.asarray(_x_ri).copy(), _obj_ri)
                                _rins_improved = _obj_ri < _rins_obj0 - 1e-9
                                logger.info("LNS RINS incumbent: obj=%.6g", _obj_ri)
                    except Exception as _e:
                        logger.debug("LNS RINS failed: %s", _e)
                    _record_improver(_HEUR_COST["rins"], _rins_improved)
                    _heuristic_governor.record("rins", _rins_improved)

                # (3) Local branching (improve). Scalable Hamming-ball sub-MIP
                # around the incumbent, escalating k across calls. Bounded by a
                # small per-call time slice and the remaining budget. The sub-solve
                # carries the recursion guard internally (``_lns_enabled=False``).
                if (
                    _lns_have_inc
                    and _lns_gap_open
                    and (_deadline - time.perf_counter()) > 1.0
                    and _improver_allowed(_HEUR_COST["lbranch"])
                    and _heuristic_governor.allowed("lbranch", gap_open=_lns_gap_open)
                ):
                    _lb_k = _lns_k_schedule[min(_lns_lb_calls, len(_lns_k_schedule) - 1)]
                    _lns_lb_calls += 1
                    _lb_slice = min(2.0, max(0.5, _deadline - time.perf_counter() - 0.2))
                    _lb_obj0 = float(_lns_inc[1])
                    _lb_improved = False
                    try:
                        from discopt._jax.primal_heuristics import local_branching

                        _lb = local_branching(
                            model,
                            np.asarray(_lns_inc[0], dtype=np.float64),
                            k=_lb_k,
                            backend=_lns_backend,
                            evaluator=evaluator,
                            submip_time_limit=_lb_slice,
                            submip_max_nodes=1000,
                            submip_gap_tolerance=gap_tolerance,
                            # F1: hand the enumeration a hard budget (profile
                            # §1.1) so it cannot blow past the slice/deadline.
                            deadline=_deadline,
                            node_bound=_lns_lb,
                            incumbent_obj=float(_lns_inc[1]),
                            gap_tolerance=gap_tolerance,
                        )
                        if _lb is not None:
                            _x_lb, _obj_lb = _lb
                            _obj_lb = float(_obj_lb)
                            _lb_feas = not cl_list or _check_constraint_feasibility(
                                evaluator, _x_lb, cl_list, cu_list
                            )
                            if np.isfinite(_obj_lb) and _obj_lb < _SENTINEL_THRESHOLD and _lb_feas:
                                tree.inject_incumbent(np.asarray(_x_lb).copy(), _obj_lb)
                                _lb_improved = _obj_lb < _lb_obj0 - 1e-9
                                logger.info(
                                    "LNS local-branching incumbent: obj=%.6g (k=%d)",
                                    _obj_lb,
                                    _lb_k,
                                )
                    except Exception as _e:
                        logger.debug("LNS local-branching failed: %s", _e)
                    _record_improver(_HEUR_COST["lbranch"], _lb_improved)
                    _heuristic_governor.record("lbranch", _lb_improved)

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
                tree_bound_valid=_gap_certified,
            )

        # Convex-objective node bound (applied at the single point every node's
        # bound funnels through, so it covers all upstream paths). When the
        # minimized objective is a convex quadratic, the supporting-hyperplane
        # lower bound over the node box is rigorous and far tighter than the
        # McCormick linearization of the convex objective (nvs17 root -2522 LP vs
        # -1106 convex ~ optimum -1100). max() only tightens, so it is always sound.
        if _use_convex_obj_bound:
            for i in range(n_batch):
                if result_lbs[i] >= _SENTINEL_THRESHOLD or node_infeasible_mask[i]:
                    continue
                if time.perf_counter() >= _deadline:
                    break
                cvx_lb = _convex_objective_lower_bound(
                    evaluator, np.asarray(batch_lb[i]), np.asarray(batch_ub[i])
                )
                if np.isfinite(cvx_lb):
                    result_lbs[i] = max(result_lbs[i], cvx_lb)

        # Completeness guard (soundness). In nonconvex mode the Rust tree never
        # promotes a node's relaxation bound to the incumbent, and the per-node NLP
        # that normally injects feasible points is strided — so a node whose
        # relaxation solution is already an integer- AND constraint-feasible point
        # (e.g. the true optimum at a fully-branched leaf) can be fathomed without
        # its objective EVER being recorded. The tree then exhausts while missing
        # that point and falsely certifies a worse incumbent (nvs19: certified
        # -1098.0 with -1098.4 feasible). Inject every such verified point here,
        # ungated: ``inject_incumbent`` accepts only a strictly-improving feasible
        # point and never touches the dual bound, so this only ever tightens the
        # incumbent — it cannot make the search unsound, only complete.
        if not _model_is_convex and int_offsets:
            _cl = [c[0] for c in constraint_bounds] if constraint_bounds else None
            _cu = [c[1] for c in constraint_bounds] if constraint_bounds else None
            for i in range(n_batch):
                if node_infeasible_mask[i] or result_lbs[i] >= _SENTINEL_THRESHOLD:
                    continue
                xi = np.asarray(result_sols[i], dtype=np.float64)
                if not _is_integer_feasible_solution(xi, int_offsets, int_sizes):
                    continue
                xr = xi.copy()
                for _off, _sz in zip(int_offsets, int_sizes):
                    xr[_off : _off + _sz] = np.round(xr[_off : _off + _sz])
                if _cl is not None and not _check_constraint_feasibility(evaluator, xr, _cl, _cu):
                    continue
                _obj_i = float(evaluator.evaluate_objective(xr))
                if np.isfinite(_obj_i) and _obj_i < _SENTINEL_THRESHOLD:
                    tree.inject_incumbent(xr, _obj_i)

        # Interactive debugger: steer point — relaxations solved, results not
        # yet imported. Safe-steer (inject incumbent / branch hint) applies here;
        # `inject` candidates are validated by _debug_validate_candidate.
        if _debug.fire(
            _debug.Checkpoint.BEFORE_IMPORT,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=time.perf_counter() - t_start,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
            result_lbs=result_lbs,
            result_sols=result_sols,
            result_feas=result_feas,
            validator=_debug_validate_candidate,
        ):
            _debug_quit = True
            break

        # Feed per-node reduced boxes forward to the children (cert:T2.4c). Applied
        # BEFORE process_evaluated so the tree branches from the contracted box and
        # every child inherits the reduction. Only nodes still open (not fathomed
        # this round) are updated; each staged box is a subset of the node's box, so
        # the contraction removes no feasible integer point (tree_manager.rs:792).
        if _node_reduce_enabled and _nr_pending:
            t_rust_start = time.perf_counter()
            for _bi, (_nlb, _nub) in _nr_pending.items():
                if node_infeasible_mask[_bi] or result_lbs[_bi] >= _SENTINEL_THRESHOLD:
                    continue
                try:
                    tree.set_node_bounds(
                        int(batch_ids[_bi]),
                        np.asarray(_nlb, dtype=np.float64),
                        np.asarray(_nub, dtype=np.float64),
                    )
                except Exception as _sb_exc:  # pragma: no cover - defensive
                    logger.debug("set_node_bounds failed at node %d: %s", _bi, _sb_exc)
            rust_time += time.perf_counter() - t_rust_start

        # Import results back to Rust tree
        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        # Interactive debugger: prune/branch/fathom applied by the tree.
        if _debug.fire(
            _debug.Checkpoint.AFTER_PROCESS,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=time.perf_counter() - t_start,
        ):
            _debug_quit = True
            break

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
            # Interactive debugger: new-incumbent event checkpoint.
            if _debug.fire(
                _debug.Checkpoint.INCUMBENT_FOUND,
                tree=tree,
                model=model,
                iteration=iteration,
                elapsed=time.perf_counter() - t_start,
            ):
                _debug_quit = True
                break

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
                        fbbt_lbs = np.asarray(fbbt_lbs, dtype=np.float64)
                        fbbt_ubs = np.asarray(fbbt_ubs, dtype=np.float64)
                        # C-40: apply cutoff-FBBT bounds only when the Rust repr's
                        # variable layout provably aligns 1:1 with the flat B&B
                        # columns — i.e. it returns exactly ``n_vars`` intervals and
                        # every model block is scalar (block index == flat column).
                        # ``_model_repr`` can carry a reformulated/eliminated variable
                        # set whose length differs from ``model._variables`` (measured
                        # 144 vs 145 on ``util``); the old ``fbbt_lbs[bi]`` → ``lb[flat]``
                        # map then reads a *misaligned* variable's bound, writes a
                        # crossed ``lb>ub`` box, and — with the write done in place and
                        # the OOB swallowed — leaves the GLOBAL box corrupted. That
                        # corrupted box empties the intersection at the child-node
                        # cutoff clamp below, fathoming the optimum-containing node and
                        # certifying a false optimal (C-40). Forgoing this *optional*
                        # tightening on a misaligned repr keeps a valid, looser box —
                        # sound by construction (CLAUDE.md §3), never a lost bound.
                        _all_scalar = all(v.size == 1 for v in model._variables)
                        _aligned = (
                            fbbt_lbs.shape == (n_vars,)
                            and fbbt_ubs.shape == (n_vars,)
                            and _all_scalar
                        )
                        if _aligned:
                            # Intersect into a candidate box; commit only if it stays
                            # consistent (no crossed bound). FBBT lower bounds and
                            # upper bounds are each valid, so ``max``/``min`` against
                            # the current box only tightens — a crossing would mean the
                            # region is empty under the cutoff, which the tree proves
                            # via node relaxations, not via an in-place box write.
                            cand_lb = np.maximum(lb, fbbt_lbs)
                            cand_ub = np.minimum(ub, fbbt_ubs)
                            if not np.any(cand_lb > cand_ub + 1e-9):
                                n_tightened = int(
                                    np.count_nonzero(cand_lb > lb + 1e-10)
                                    + np.count_nonzero(cand_ub < ub - 1e-10)
                                )
                                lb = cand_lb
                                ub = cand_ub
                                if n_tightened > 0:
                                    logger.info(
                                        "FBBT tightened %d bounds (incumbent=%.6g)",
                                        n_tightened,
                                        inc_obj,
                                    )
                        else:
                            logger.debug(
                                "Cutoff-FBBT skipped: repr layout misaligned "
                                "(returned %d intervals, flat n_vars=%d, all_scalar=%s)",
                                fbbt_lbs.shape[0] if fbbt_lbs.ndim == 1 else -1,
                                n_vars,
                                _all_scalar,
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
                from discopt.modeling.core import ObjectiveSense as _ObjSense

                _cb_is_max = (
                    model._objective is not None and model._objective.sense == _ObjSense.MAXIMIZE
                )
                _cb_bound = _certified_callback_bound(
                    stats_snap.get("global_lower_bound"), _gap_certified, _cb_is_max
                )
                ctx = CallbackContext(
                    node_count=stats_snap["total_nodes"],
                    incumbent_obj=inc_obj_cb,
                    best_bound=_cb_bound,
                    # A2: no certified bound -> no meaningful gap (the raw gap was
                    # derived from the same non-bound). See the pre-import site.
                    gap=(stats_snap.get("gap") if _cb_bound is not None else None),
                    elapsed_time=time.perf_counter() - t_start,
                    x_relaxation=result_sols[best_idx].copy(),
                    node_bound=float(result_lbs[best_idx]),
                )
                node_callback(ctx, model)
            except Exception as e:
                logger.warning("Node callback raised an exception: %s", e)

        # Root-node certification snapshot (cert:T0.1). After the root batch has
        # been processed but before the first branch, the tree's global lower
        # bound reflects the root relaxation alone. Record it and the elapsed
        # wall clock; ``root_gap`` is finalized against the incumbent below.
        if iteration == 0:
            _root_time = time.perf_counter() - t_start
            _root_glb_snap = tree.stats().get("global_lower_bound")
            if _root_glb_snap is not None and np.isfinite(_root_glb_snap):
                _root_glb_internal = float(_root_glb_snap)

        # --- Root branch-and-reduce fixpoint (cert:T2.3, flag default OFF) ---
        # At the END of iteration 0 the root heuristics have run, so an incumbent
        # cutoff may exist. Iterate the R1 reduce stages {S2 cutoff-FBBT, S3
        # cutoff-OBBT} to a fixpoint on the ROOT box, then intersect the tightened
        # box into the global lb/ub (which every in-tree node inherits, 5425) and
        # re-capture the root cut pool from the tightened box. Tighten-only and
        # deadline-bounded; a failure leaves the search unchanged. Unlocked by the
        # R1 GO (cert-gap-plan §14 "T2.1-revisit … 2026-07-06"); default OFF until
        # T2.6's nightly-green flip.
        if (
            iteration == 0
            and _tuning().root_fixpoint
            and _mc_lp_relaxer is not None
            and model._objective is not None
        ):
            try:
                from discopt._jax.root_reduce import run_root_fixpoint

                _rf_inc = tree.incumbent()
                _rf_cutoff = (
                    float(_rf_inc[1])
                    if _rf_inc is not None
                    and np.isfinite(_rf_inc[1])
                    and _rf_inc[1] < _SENTINEL_THRESHOLD
                    else None
                )
                # No-offtarget gate (§14 T2.4 ≤1.05 wall guard): skip the fixpoint
                # when the root is already tight — the relative gap between the root
                # dual bound and the incumbent cutoff is below a small threshold, so
                # reduction has nothing to close and the loop would only add wall
                # cost on the already-fast class (e.g. instances that close at ≤10
                # nodes). When there is no incumbent yet OR no root bound, run it
                # (the structural/finitizing value can still help).
                _rf_gap_ok = True
                if _rf_cutoff is not None and _root_glb_internal is not None:
                    _rf_lo = float(_root_glb_internal)
                    if np.isfinite(_rf_lo):
                        _rf_rel_gap = abs(_rf_cutoff - _rf_lo) / (1.0 + abs(_rf_cutoff))
                        _rf_gap_ok = _rf_rel_gap > _ROOT_FIXPOINT_MIN_GAP
                # R1 budget: ~10% of the time limit (loop converges in <=2 iters,
                # S3 OBBT gets ~85% of it inside the loop). Hard deadline-bounded.
                _rf_budget = min(max(time_limit * 0.10, 1.0), max(_remaining_budget(), 0.0))
                if _rf_gap_ok and _rf_budget > _DEADLINE_NODE_FLOOR_S:
                    _rf_res = run_root_fixpoint(
                        model,
                        np.asarray(lb, dtype=np.float64),
                        np.asarray(ub, dtype=np.float64),
                        incumbent_cutoff=_rf_cutoff,
                        deadline=time.perf_counter() + _rf_budget,
                        tol=1e-6,
                        prefer_pounce=nlp_solver == "pounce",
                        superposition=(relaxation_arithmetic == "superposition"),
                        measure_bound=False,
                    )
                    if _rf_res.infeasible:
                        # The relaxation excludes every point better than the
                        # incumbent cutoff -> the incumbent is optimal. Nothing to
                        # tighten; the tree certifies on the next termination check.
                        logger.info(
                            "Root fixpoint proved no improving point below the "
                            "incumbent cutoff (root reduce)"
                        )
                    elif _rf_res.n_tightened > 0:
                        # Intersect tighten-only into the global box.
                        lb = np.maximum(np.asarray(lb, dtype=np.float64), _rf_res.lb)
                        ub = np.minimum(np.asarray(ub, dtype=np.float64), _rf_res.ub)
                        logger.info(
                            "Root fixpoint tightened %d bounds over %d round(s) (cutoff=%s)",
                            _rf_res.n_tightened,
                            _rf_res.n_rounds,
                            "none" if _rf_cutoff is None else f"{_rf_cutoff:.6g}",
                        )
                        # The existing root cut pool (captured on the WIDER root box)
                        # stays valid after tightening — a cut valid over a box is
                        # valid over any sub-box — so every in-tree node keeps
                        # inheriting sound cuts with NO refresh needed for soundness.
                        # A refresh on the tightened box can only *strengthen* the
                        # pool, but it costs a full separating solve (irreducible ~1s
                        # floor on a large lifted relaxation, not bounded by the LP
                        # time_limit — measured +3.7s on pooling_adhya1stp), which is
                        # a no-offtarget-guard violation (§14 T2.4 ≤1.05 wall). So the
                        # refresh is opt-in only (DISCOPT_ROOT_FIXPOINT_REPOOL=1) for a
                        # future strength A/B; the default flagged path skips it and
                        # keeps the still-valid pool.
                        if (
                            getattr(_mc_lp_relaxer, "_inc", None) is not None
                            and os.environ.get("DISCOPT_ROOT_FIXPOINT_REPOOL") == "1"
                            and _remaining_budget() > _DEADLINE_NODE_FLOOR_S
                        ):
                            try:
                                _rf_chunks: list = []
                                _mc_lp_relaxer.solve_at_node(
                                    np.asarray(lb, dtype=np.float64),
                                    np.asarray(ub, dtype=np.float64),
                                    time_limit=max(_remaining_budget(), _DEADLINE_NODE_FLOOR_S),
                                    out_cuts=_rf_chunks,
                                )
                                if _rf_chunks and _rf_chunks[0] is not None:
                                    _A_rf, _b_rf, _idents_rf = _rf_chunks[0]
                                    if _A_rf is not None and _A_rf.shape[0] > 0:
                                        if _A_rf.shape[0] > _root_cut_max:
                                            _A_rf = _A_rf[-_root_cut_max:]
                                            _b_rf = _b_rf[-_root_cut_max:]
                                        _root_cut_pool = (_A_rf, _b_rf, _idents_rf)
                            except Exception as _rf_pool_exc:  # pragma: no cover
                                logger.debug(
                                    "root-fixpoint cut-pool refresh skipped: %s",
                                    _rf_pool_exc,
                                )
            except Exception as _rf_exc:  # pragma: no cover - defensive
                logger.debug("root fixpoint reduce skipped: %s", _rf_exc)

        iteration += 1

        # Check termination
        if tree.is_finished():
            break
        if _gap_converged(tree, gap_tolerance):
            break

        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    # --- Build result ---
    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time

    stats = tree.stats()
    incumbent = tree.incumbent()

    # SPATIAL-CERT: the floor-inclusive rigorous bound adopted when the terminal
    # accounting below re-earns certification over soundly-floored sentinel
    # removals (None on every other exit — then the result build is unchanged).
    _taint_rig_bound_internal: Optional[float] = None

    # R3a instrumentation: export the per-variable branch-frequency vector when
    # an experiment harness has armed the sink. Behavior-neutral; guarded so a
    # missing accessor (older extension build) degrades silently.
    if _R3A_BRANCH_COUNT_SINK is not None:
        try:
            _R3A_BRANCH_COUNT_SINK["branch_var_counts"] = np.asarray(
                tree.branch_var_counts()
            ).tolist()
        except Exception as _e:  # pragma: no cover - diagnostics only
            logger.debug("R3a branch_var_counts capture failed: %s", _e)

    if incumbent is not None:
        sol_array, obj_val = incumbent
        # Filter out bogus incumbents from infeasible NLP relaxations
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    # #467 sub-bug #3: rigorous infeasibility proof wins over a soft incumbent.
    # When the root (whole feasible region) was rigorously proven infeasible, an
    # incumbent is only trustworthy if it is ITSELF rigorously feasible (at the
    # solver's constraint-feasibility tolerance, evaluated over ALL constraints).
    # If it is not, it is a spurious primal-heuristic point inside a region the
    # search proved empty — discard it so control falls to the terminal
    # infeasibility/unknown logic below (which reports ``infeasible`` for a
    # rigorous proof, ``unknown`` for a non-rigorous one). If the incumbent IS
    # rigorously feasible, the root proof must have over-tightened: keep the
    # incumbent and NEVER report ``infeasible`` (guards the worst-class error, a
    # false infeasible on a truly-feasible model).
    if incumbent is not None and _root_rigorously_infeasible:
        # Verify against the ORIGINAL (pre-reform) constraints. The factorable
        # reform rewrites the model in terms of lifted aux variables; an incumbent
        # can satisfy every reformed constraint (the aux defining equalities plus
        # the rewritten bodies) while violating an ORIGINAL constraint by more than
        # tolerance (ex7_3_6: reformed set feasible at 1e-4, but original C0 is
        # violated by ~1.2e-4). Checking the reformed ``evaluator`` would mask that,
        # so use the pre-reform model over the incumbent's original-variable slice
        # (the first ``_prereform_nvars`` flat entries — aux columns are appended
        # after the originals). When no reform ran, ``evaluator``/``cl_list`` ARE
        # the original constraints, so fall back to them.
        _inc_feasible = True
        try:
            if _prereform_model is not None:
                _pf_eval = _make_evaluator(_prereform_model)
                _pf_cl, _pf_cu = _infer_constraint_bounds(_prereform_model, _pf_eval)
                if _pf_cl:
                    _inc_feasible = _check_constraint_feasibility(
                        _pf_eval,
                        np.array(sol_array)[:_prereform_nvars],
                        _pf_cl,
                        _pf_cu,
                    )
            elif cl_list:
                _inc_feasible = _check_constraint_feasibility(
                    evaluator, np.array(sol_array), cl_list, cu_list
                )
        except Exception as _pf_exc:  # pragma: no cover - defensive
            # If the original-model re-check cannot be built, be conservative and
            # do NOT discard the incumbent (never risk a false infeasible).
            logger.debug("prereform incumbent re-check skipped: %s", _pf_exc)
            _inc_feasible = True
        if not _inc_feasible:
            logger.debug(
                "Discarding within-tolerance-only incumbent (obj=%.6g): the root was "
                "rigorously proven infeasible and this point violates the ORIGINAL "
                "constraints beyond tolerance (#467 sub-bug #3).",
                obj_val,
            )
            incumbent = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        # C-3: snap near-integral discrete coordinates to exact integers before
        # reporting. The tree incumbent can carry a coordinate a few digits shy
        # of integral (it passed the 1e-5 injection gate); the terminal polish
        # below rounds+re-solves it, but is best-effort — if it raises or is not
        # adopted the raw fractional value would be certified. Round up front so
        # the reported point is exactly integral regardless of the polish
        # outcome, only when the rounded point stays feasible.
        _rounded_inc, _rounded_feas = _round_incumbent_integers(
            sol_flat, int_offsets, int_sizes, evaluator, cl_list, cu_list
        )
        if _rounded_feas:
            sol_flat = _rounded_inc
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
            ):
                _pobj = float(_polished.objective)
                _refined = np.asarray(_polished.x, dtype=float).copy()
                for _off, _sz in zip(int_offsets, int_sizes):
                    for _k in range(int(_sz)):
                        _refined[_off + _k] = round(float(sol_flat[_off + _k]))
                # Two reasons to adopt the integer-fixed continuous completion:
                #   (a) purification — objective ~unchanged, just tighter continuous
                #       values (the original behavior, always safe), and
                #   (b) improvement (#281) — it is strictly better than the B&B
                #       incumbent's continuous completion. The smallinvDAX-style MIQPs
                #       reach the right integer assignment but the batched IPM leaves
                #       the continuous part short of optimal, so the reported gap sits
                #       a fraction of a percent (sometimes more) above the optimum; a
                #       KKT-accurate completion closes it.
                # An improvement is adopted ONLY after verifying the refined point is
                # genuinely feasible (fixed integers + a feasible continuous
                # completion is a valid MINLP point), so an improving-but-divergent
                # re-solve can never report a false optimum.
                _unchanged = abs(_pobj - obj_val) <= 1e-4 * (1.0 + abs(obj_val))
                # An objective improvement from the re-solve is adopted ONLY for
                # convex models, where the integer-fixed continuous relaxation is
                # exact (no spatial-envelope slack) so a KKT completion is a genuine
                # MINLP point — the smallinvDAX MIQP case #281 targets. For
                # nonconvex/spatial models the terminal re-solve runs against loose
                # McCormick envelopes, so an "improved" point can satisfy the relaxed
                # constraint system yet be infeasible for the true model (st_e35:
                # fractional power of a ratio jumped 176.17 -> 0.0). There we keep
                # only the always-safe purification (unchanged objective).
                _improved = _model_is_convex and _pobj < obj_val - 1e-9 * (1.0 + abs(obj_val))
                _accept = _unchanged or (
                    _improved
                    and (
                        not cl_list
                        or _check_constraint_feasibility(evaluator, _refined, cl_list, cu_list)
                    )
                )
                if _accept:
                    sol_flat = _refined
                    x_dict = _unpack_solution(model, sol_flat)
                    obj_val = _pobj
        except Exception as _exc:
            logger.debug("Incumbent KKT polish failed: %s", _exc)

        # Negate objective back for maximization (B&B tree tracks minimization)
        from discopt.modeling.core import ObjectiveSense

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        # An unresolved tree bound (a node fathomed with no branch direction and no
        # valid finite dual bound — an unbounded-below / free-variable root) means
        # the search cannot certify optimality even though `is_finished()` is True
        # (no node can be branched further). The Rust tree pins `global_lower_bound`
        # at -inf in that case (issue #467); do not let `is_finished()` alone
        # certify. `_gap_converged` already returns False on the -inf bound, so
        # require it (not the bare `is_finished()`) whenever the bound is unresolved.
        _bound_unresolved = bool(tree.stats().get("bound_unresolved", False))
        if _bound_unresolved:
            _gap_certified = False

        # SPATIAL-CERT (gap-closing plan P0; MILP-driver parity with #604): a
        # non-rigorous sentinel fathom decertifies the gap mid-loop (issue #27a,
        # sites above), but the removed subtree is not unaccounted — its POP-TIME
        # lower bound (proved at an ancestor's rigorous relaxation solve; a bound
        # over a fixed box stays valid forever) is accumulated in
        # ``_taint_floor_internal`` by the authoritative C-1 sweep (#603). The
        # rigorous global dual bound of such a tree is therefore
        #     min(surviving-frontier bound, taint floor)
        # — every term of which is a *proved* bound. When that floor-inclusive
        # bound closes the certification gap against the incumbent, every
        # unproven removal is provably within tolerance of the incumbent — the
        # exact criterion #604 certifies with in the MILP driver (its
        # ``unresolved_floor`` permanently seeds the frontier minimum and
        # ``Optimal`` requires the floor-inclusive ``gap_closed``). Withholding
        # the label there is pure over-conservatism: certification must track
        # genuine search truncation and UNACCOUNTED removals, not
        # soundly-floored node failures (measured: nvs22 under
        # DISCOPT_LU_DENSITY_ROUTE=1 — one node whose LP soundly declined an
        # uncertifiable vertex bound was sentinel-fathomed carrying pop-time
        # floor 7.4035, far above the incumbent 6.05822; frontier gap 4.9e-9,
        # yet the exit was downgraded to "feasible").
        #
        # Certification is re-earned here ONLY when:
        #   * the sole decertification cause was sentinel fathoms — an untrusted
        #     bound VALUE in the tree (``_tree_bound_poisoned``), an untrusted
        #     convex node bound (``_convex_bound_untrusted``), or an unresolved
        #     -inf pin (``_bound_unresolved``) can never re-earn: those leave no
        #     rigorous per-removal floor to account with;
        #   * the floor-inclusive bound is finite and non-sentinel (a -inf floor
        #     — an unbounded removal — keeps the conservative downgrade); and
        #   * the floor-inclusive gap converges under the IDENTICAL arithmetic
        #     the search itself certifies with (``_gap_values_converged``).
        # The reported bound is then the floor-inclusive value (never the bare
        # frontier, which may over-report the removed subtrees) — byte-identical
        # to the bound #603's recovery already reports on these exits today;
        # only the label changes. This does not touch #603's #27a gate at the
        # late re-certification block: a taint-recovered bound on a solve whose
        # floor-inclusive gap did NOT close still never upgrades to "optimal".
        if (
            not _gap_certified
            and _nonrigorous_fathom
            and not _tree_bound_poisoned
            and not _convex_bound_untrusted
            and not _bound_unresolved
        ):
            _glb_int = stats.get("global_lower_bound")
            _inc_int = stats.get("incumbent_value")
            if (
                _glb_int is not None
                and np.isfinite(_glb_int)
                and _inc_int is not None
                and np.isfinite(_inc_int)
            ):
                _rig_int = min(float(_glb_int), _taint_floor_internal)
                if (
                    np.isfinite(_rig_int)
                    and abs(_rig_int) < _SENTINEL_THRESHOLD
                    and _gap_values_converged(float(_inc_int), _rig_int, gap_tolerance)
                ):
                    _gap_certified = True
                    _taint_rig_bound_internal = _rig_int

        search_closed = _gap_converged(tree, gap_tolerance) or (
            tree.is_finished() and not _bound_unresolved
        )
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
        elif _nonrigorous_fathom:
            # C-1: the tree exhausted with no incumbent, but at least one node was
            # fathomed on a NON-rigorous failure (its local NLP failed / diverged /
            # returned a constraint-violating iterate and it was sentinelled with no
            # empty-relaxation or SolveStatus.INFEASIBLE certificate). A non-rigorous
            # failure does NOT prove a subtree empty, so we cannot soundly claim the
            # model is infeasible — its feasibility is genuinely UNDETERMINED.
            # Reporting "infeasible" here would be a false certificate (the
            # worst-class error: a feasible model declared infeasible). Report
            # "unknown" instead, exactly as the _solve_nlp_bb path does with
            # _unconverged_fathom. Conservative by design: soundness over capability.
            status = "unknown"
            _gap_certified = False
        elif _debug_quit:
            # Interactive debugger `quit`: the user interrupted the search, so
            # the tree is NOT exhausted and neither infeasibility nor optimality
            # was proven (a false "infeasible" here is the worst-class error).
            # Report "unknown", uncertified.
            status = "unknown"
            _gap_certified = False
        else:
            # Tree exhausted with no feasible node and every fathom was a valid
            # certificate (empty McCormick/LP relaxation over a finite box or
            # SolveStatus.INFEASIBLE): infeasibility *is* a certified conclusion, so
            # leave _gap_certified untouched.
            status = "infeasible"

    # Interactive debugger: terminal checkpoint. Fired after the status
    # decision so ``debug="on-error"`` sessions can key on the outcome:
    # ``error`` is the non-"optimal" status ("time_limit", "unknown",
    # "infeasible", ...), or None on a certified optimum.
    _debug.fire(
        _debug.Checkpoint.TERMINATED,
        tree=tree,
        model=model,
        iteration=iteration,
        elapsed=wall_time,
        error=None if status == "optimal" else status,
    )

    from discopt.modeling.core import ObjectiveSense

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    # SPATIAL-CERT: on an exit certified over soundly-floored sentinel removals,
    # the honest rigorous bound is the floor-inclusive value min(frontier, taint
    # floor) — the bare frontier may over-report the removed subtrees. This is
    # the same value #603's recovery reports on the uncertified version of these
    # exits; adopting it here keeps "certified" and "reported" bounds identical.
    if _taint_rig_bound_internal is not None:
        bound_val = _taint_rig_bound_internal
    assert model._objective is not None
    if bound_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        bound_val = -bound_val
    gap_val = stats["gap"]
    if (
        _taint_rig_bound_internal is not None
        and obj_val is not None
        and np.isfinite(obj_val)
        and bound_val is not None
        and np.isfinite(bound_val)
    ):
        # Recompute the reported gap from the adopted floor-inclusive bound so
        # the surfaced gap matches the surfaced bound (the tree's ``gap`` was
        # computed from the bare frontier).
        gap_val = abs(obj_val - bound_val) / max(1.0, abs(obj_val))

    # A *feasible* exit never inherits the tree's validity flag as a certificate.
    # The flag means no node invalidated the tree bound — NOT that the gap is
    # closed; a budget/node-limited feasible exit leaves that valid bound strictly
    # below the incumbent (open gap). Reporting gap_certified=True there would
    # falsely claim global optimality (max_nodes=1 root-only exit: obj 19029,
    # bound -583, gap 1.03 — yet previously "certified"). Drop the flag here and
    # re-earn it below only if a rigorous global bound actually closes the gap.
    # "optimal" already implies a closed certified search; infeasible/limit exits
    # are handled above and keep their own certification semantics.
    # Separate the two things the validity flag conflates: (1) the tree bound is
    # *untainted* — no node was fathomed without a soundness proof, so the frontier
    # minimum is a valid global dual bound — vs (2) the gap is *closed* (optimality).
    # ``_gap_certified`` here still reflects (1); the feasible-reset below clears it
    # for (2). A budget/node-limited feasible exit does not close the gap, but its
    # untainted tree bound is still the best rigorous dual bound we have and must NOT
    # be dropped to None (which forced the far weaker root-relaxation fallback, #138).
    _tree_bound_valid = _gap_certified
    if status == "feasible":
        _gap_certified = False

    _bound_from_taint_recovery = False
    if not _gap_certified:
        gap_val = None
        # Keep the untainted tree bound on a feasible exit; recompute its gap. Only
        # a tainted or non-finite tree bound is discarded (then the root fallback
        # below supplies a sound bound). A bound that meets the incumbent re-earns
        # certification in the block further down.
        if _tree_bound_valid and bound_val is not None and np.isfinite(bound_val):
            if obj_val is not None and np.isfinite(obj_val):
                gap_val = abs(obj_val - bound_val) / max(1.0, abs(obj_val))
        else:
            bound_val = None
            # B2-FIX (task #89): decertify-and-discard repair. The tree bound was
            # decertified, but unless a possibly-invalid bound VALUE entered the
            # tree (``_tree_bound_poisoned``), every number in the frontier is
            # still a rigorous per-subtree bound — the only unsound step was
            # *removing* sentinel-fathomed nodes without proof. Those subtrees
            # are re-represented by ``_taint_floor_internal`` (the min of their
            # pop-time bounds, each proved at the node's parent), so
            #   min(frontier bound, taint floor)
            # is a rigorous global dual bound: the frontier term covers every
            # surviving open node, the floor covers every unproven removed
            # subtree, and rigorously-fathomed regions need no term (their
            # proofs stand). Report it instead of discarding the search's proof
            # wholesale (DECOMP-1: nvs05 reported 1.348 where the tree had
            # proved ~4.87). Stays UNCERTIFIED: the gap is recomputed for
            # honesty but ``_gap_certified`` remains False and the
            # re-certification block below is gated off for this bound — a
            # tainted tree never upgrades to "optimal" via its own recovered
            # bound (issue #27a's contract), only via an independently rigorous
            # bound (root pool / root relaxation), exactly as before.
            # The frontier term must itself be finite: a -inf frontier (e.g. the
            # Rust tree pinned ``global_lower_bound`` at -inf because a node was
            # fathomed unresolved WITHOUT passing through the sentinel sweep —
            # issue #467's ``bound_unresolved``) means an unproven subtree has
            # no floor entry, so the floor alone would over-report. -inf is then
            # the honest frontier value and the recovery must stand down.
            _glb_int = stats["global_lower_bound"]
            if not _tree_bound_poisoned and _glb_int is not None and np.isfinite(_glb_int):
                _rig_int = min(_taint_floor_internal, float(_glb_int))
                if np.isfinite(_rig_int) and abs(_rig_int) < _SENTINEL_THRESHOLD:
                    bound_val = (
                        -_rig_int if model._objective.sense == ObjectiveSense.MAXIMIZE else _rig_int
                    )
                    _bound_from_taint_recovery = True
                    if obj_val is not None and np.isfinite(obj_val):
                        gap_val = abs(obj_val - bound_val) / max(1.0, abs(obj_val))

    # Root cut-pool bound: a rigorous global lower bound the strengthened root
    # relaxation already proved during setup (nvs19: -1156 vs the cut-less tree's
    # -88237). The tree's per-node cuts prune children but never lift the frontier
    # minimum, so an uncertified feasible exit reports the loose McCormick bound
    # and a ~99% gap while we have a far tighter one in hand. Adopt it here (for a
    # MINIMIZE, mirroring the recompute-fallback's soundness gate) before that
    # fallback recomputes from scratch — it is both stronger and free. If it meets
    # the incumbent, the re-certification below upgrades the exit to "optimal".
    if (
        _root_pool_bound is not None
        and np.isfinite(_root_pool_bound)
        and model._objective.sense == ObjectiveSense.MINIMIZE
        and (bound_val is None or not np.isfinite(bound_val) or _root_pool_bound > bound_val)
    ):
        bound_val = _root_pool_bound
        # The pool bound is rigorous independently of the tree's taint, so it
        # may re-certify below exactly as before (B2-FIX gate does not apply).
        _bound_from_taint_recovery = False
        if obj_val is not None and np.isfinite(obj_val):
            gap_val = max(0.0, obj_val - bound_val) / max(1.0, abs(obj_val))

    # When the tree produced no usable dual bound (uncertified exit, or a
    # relaxation the LP backend could not solve), fall back to a rigorous global
    # bound from the root MILP relaxation over the root box, so the search reports
    # a finite sound bound instead of None (issue #138). The MILP path already
    # seeds such a root bound; this brings the spatial path to parity. Surfaced
    # lazily — only when the search yielded nothing — and never overrides an
    # existing finite bound. ``_root_relaxation_lower_bound`` always returns a
    # valid lower bound of the *internally minimized* objective: for a MINIMIZE it
    # is the dual lower bound directly; for a MAXIMIZE the builder minimizes
    # ``-obj`` so a lower bound ``L`` on ``-obj`` means ``obj <= -L`` — a valid
    # *upper* bound, surfaced by negating (issue #267: inscribedsquare02, whose
    # objective square is finite only after FBBT bounds the free auxiliaries, used
    # to report ``bound=None`` because no maximize-side fallback existed).
    # B2-FIX (task #89): a taint-recovered tree bound must not SHORT-CIRCUIT this
    # fallback — pre-fix, every tainted exit reached it (the tree bound had been
    # discarded to None) and could report a root-relaxation bound STRONGER than
    # the taint floor (tanksize: floor 0.847 vs root relaxation 0.868). Run it in
    # that case too and keep the tighter of the two rigorous bounds.
    _rr_needed = bound_val is None or not np.isfinite(bound_val) or _bound_from_taint_recovery
    _rr_remaining = time_limit - (time.perf_counter() - t_start)
    if _rr_needed and _rr_remaining <= 0.0:
        # B&B consumed the whole limit and surfaced no bound. Spend a small bounded
        # slice on the rigorous root-relaxation fallback anyway so an uncertified
        # exit reports a sound dual bound instead of None (issue #138). Only ever
        # reached when the search produced nothing usable; the floor's own ~10%
        # internal budget keeps the wall-time overrun small.
        _rr_remaining = _ROOT_FALLBACK_FLOOR_S
    if _rr_needed and _rr_remaining > 0.0:
        _is_maximize = model._objective.sense == ObjectiveSense.MAXIMIZE
        # Budget the fallback to the time actually left: it runs *after* the B&B
        # loop already consumed the limit, so passing the full ``time_limit`` here
        # would let it overrun by a second whole budget.
        _rr = _root_relaxation_lower_bound(
            model, _root_lb_snapshot, _root_ub_snapshot, _rr_remaining, psd_cuts=psd_cuts
        )
        if _rr is not None and np.isfinite(_rr):
            # Negate for MAXIMIZE: a lower bound on ``-obj`` is an upper bound on
            # ``obj``. The relaxation is a valid outer approximation either way, so
            # the surfaced bound is rigorous and on the correct side of the
            # incumbent (a maximize upper bound is ``>= obj``).
            _rr_signed = -_rr if _is_maximize else _rr
            # Keep the tighter rigorous bound (a MINIMIZE wants the larger lower
            # bound, a MAXIMIZE the smaller upper bound). When the independently
            # rigorous fallback wins, the taint-recovery gate lifts (it may
            # re-certify below, exactly the pre-fix behavior).
            if bound_val is None or not np.isfinite(bound_val):
                bound_val = _rr_signed
                _bound_from_taint_recovery = False
            elif (_rr_signed <= bound_val) if _is_maximize else (_rr_signed >= bound_val):
                bound_val = _rr_signed
                _bound_from_taint_recovery = False
            if obj_val is not None and np.isfinite(obj_val):
                gap_val = abs(bound_val - obj_val) / max(1.0, abs(obj_val))

    # Re-earn certification on a feasible exit iff a *rigorous* global bound (here
    # the root-relaxation fallback, itself only returned when the objective is
    # validly linearized) meets the incumbent within tolerance — then the
    # incumbent is provably global and the honest status is "optimal". For a
    # MINIMIZE the bound is a lower bound (``bound <= incumbent``); for a MAXIMIZE
    # it is an upper bound (``bound >= incumbent``). The on-correct-side guard
    # stops a spurious wrong-side bound (which the abs-gap clamp would otherwise
    # read as gap 0) from certifying.
    _is_max = model._objective.sense == ObjectiveSense.MAXIMIZE
    _bound_on_correct_side = (
        bound_val is not None
        and np.isfinite(bound_val)
        and (bound_val >= obj_val - 1e-9 if _is_max else bound_val <= obj_val + 1e-9)
        if obj_val is not None
        else False
    )
    if (
        status == "feasible"
        and obj_val is not None
        and _bound_on_correct_side
        and gap_val is not None
        and gap_val <= gap_tolerance
        # B2-FIX (task #89): a bound recovered from a TAINTED tree (frontier
        # min floored by the tainted nodes' pop-time bounds) is reported but
        # never re-certifies: #27a's contract is that a non-rigorous fathom
        # can never upgrade the exit to "optimal". Independently rigorous
        # bounds (root pool / root relaxation) still re-certify as before.
        and not _bound_from_taint_recovery
    ):
        _gap_certified = True
        status = "optimal"

    # SOUNDNESS INVARIANT (bound <= incumbent). A valid dual bound never crosses a
    # known feasible objective: a MINIMIZE lower bound is <= the incumbent, a
    # MAXIMIZE upper bound is >= it. The Rust tree already enforces this
    # (``update_global_lower_bound`` caps ``global_lower_bound`` at
    # ``incumbent_value``), but the Python bound-adoption fallbacks above — the
    # root-relaxation recompute over a possibly FBBT-tightened *snapshot* box, the
    # taint floor, the root-cut-pool bound — can surface a value that is valid over
    # its own box yet exceeds the GLOBAL incumbent when that box no longer contains
    # the optimum. That is an unsound reported bound (ex14_1_7 / ex14_1_9: the
    # objective is a free variable the uniform engine cannot bound, so the tainted
    # exit ran the root-relaxation fallback over a tightened snapshot and adopted
    # 28.5 / 1.0 as a "tighter" lower bound though the true optimum is 0). Cap the
    # reported bound at the incumbent. This runs AFTER the certification decision —
    # which correctly used the raw, possibly wrong-side bound via
    # ``_bound_on_correct_side`` to REFUSE certification — so it can never
    # manufacture a false "optimal", and it is a no-op for any genuinely valid
    # bound (already on the correct side). The reported gap is recomputed from the
    # capped bound so ``bound`` and ``gap`` stay mutually consistent.
    if (
        bound_val is not None
        and np.isfinite(bound_val)
        and obj_val is not None
        and np.isfinite(obj_val)
    ):
        _capped = max(bound_val, obj_val) if _is_max else min(bound_val, obj_val)
        if _capped != bound_val:
            bound_val = _capped
            gap_val = abs(bound_val - obj_val) / max(1.0, abs(obj_val))

    # Root-node certification metrics (cert:T0.1). Convert the root snapshot to
    # the reported objective sense (mirroring ``bound_val``'s negation for
    # MAXIMIZE) and adopt the strengthened root cut-pool bound when it is the
    # tighter rigorous lower bound (MINIMIZE only, matching the pool-bound
    # adoption above). ``root_gap`` is the relative gap of that bound against
    # the final incumbent, using the same floored abs/rel convention as ``gap``.
    root_bound_val: Optional[float] = None
    root_gap_val: Optional[float] = None
    _root_internal = _root_glb_internal
    if (
        _root_pool_bound is not None
        and np.isfinite(_root_pool_bound)
        and model._objective.sense == ObjectiveSense.MINIMIZE
        and (_root_internal is None or _root_pool_bound > _root_internal)
    ):
        _root_internal = _root_pool_bound
    if _root_internal is not None and np.isfinite(_root_internal):
        root_bound_val = -_root_internal if _is_max else _root_internal
        if obj_val is not None and np.isfinite(obj_val):
            root_gap_val = abs(obj_val - root_bound_val) / max(1.0, abs(obj_val))

    # Per-family reduction/separation timers (cert:T0.3). OBBT time is already
    # accumulated in ``_pn_obbt_spent``; FBBT in ``_reduce_timers``; the
    # separation families on the relaxer instance. Only non-zero families are
    # surfaced; an all-zero result reports None.
    _reduce_timers["obbt"] = _pn_obbt_spent
    _solver_stats: dict[str, float] = {
        f"reduce/{_rfam}": float(_rt) for _rfam, _rt in _reduce_timers.items() if _rt > 0.0
    }
    if _mc_lp_relaxer is not None and getattr(_mc_lp_relaxer, "_sep_timers", None):
        for _sfam, _stime in _mc_lp_relaxer._sep_timers.items():
            if _stime > 0.0:
                _solver_stats[f"separate/{_sfam}"] = float(_stime)
        # THRU-3: surface the square-gate fire count so the default-OFF gate is
        # proven to engage when enabled (0 on the default path).
        _sqf = int(getattr(_mc_lp_relaxer, "_square_gate_fires", 0))
        if _sqf > 0:
            _solver_stats["gate/square_fires"] = float(_sqf)
    # Root-cut-pool inheritance counters (THRU-4). Surfaced whenever a pool was
    # built or inherited so both the fire-proof (pool populates + inherits) and
    # the skip lever are observable on the final result.
    if _root_cut_pool is not None:
        _solver_stats["pool/size"] = float(_root_cut_pool[0].shape[0])
    # CUT-INHERIT-GRAD: surface the structure predicate's root feature so the
    # gate decision is verifiable from the outside (entry experiment + fire-proof).
    if _root_sqpsd_frac is not None:
        _solver_stats["pool/root_sqpsd_frac"] = float(_root_sqpsd_frac)
    # CUT-INHERIT-GRAD gate decision (verifiable firing). ``mode``: 1 force-on,
    # 0 force-off (env unset/``=0`` — the shipped default), -1 structure-gated
    # opt-in (``=gated``/``cut_inherit=None``). ``gate_decision``: 1 iff
    # inheritance actually engaged (not forced off AND a non-empty root pool was
    # separated — the pool-fires predicate); the feature it keys on is the pool
    # size, surfaced as ``pool/size`` above.
    _gate_mode_code = (
        1.0 if _cut_inherit_mode is True else (0.0 if _cut_inherit_mode is False else -1.0)
    )
    _solver_stats["pool/gate_mode"] = _gate_mode_code
    _solver_stats["pool/gate_decision"] = (
        1.0 if (_cut_inherit_enabled and _root_cut_pool is not None) else 0.0
    )
    if _mc_lp_relaxer is not None and getattr(_mc_lp_relaxer, "_pool_stats", None):
        for _pfam, _pcount in _mc_lp_relaxer._pool_stats.items():
            if _pcount > 0:
                _solver_stats[f"pool/{_pfam}"] = float(_pcount)
    # C-42 Part 2: node solves the global-bound-stall governor re-separated
    # (driver-side; the relaxer's ``lazy_reseparations`` counts the stride net).
    if _lazy_resep_fires > 0:
        _solver_stats["pool/stall_reseparations"] = float(_lazy_resep_fires)

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
        root_bound=root_bound_val,
        root_gap=root_gap_val,
        root_time=_root_time,
        solver_stats=_solver_stats or None,
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
    backend_evaluator = cast("NLPEvaluator", _BoundOverrideEvaluator(evaluator, lb, ub))

    t_jax_start = time.perf_counter()
    if nlp_solver == "pounce":
        from discopt.solvers.nlp_pounce import solve_nlp as solve_nlp_pounce

        nlp_result = solve_nlp_pounce(
            backend_evaluator, x0, constraint_bounds=constraint_bounds, options=opts
        )
    else:
        # "ipm"/"sparse_ipm" resolve to POUNCE upstream (the JAX IPM is retired);
        # only "ipopt"/"cyipopt" reach this branch.
        from discopt.solvers.nlp_ipopt import solve_nlp

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

    # Root-node certification metrics (cert:T0.1). This path has no B&B tree —
    # the single NLP solve at the root box is the whole solve, so the root
    # bound/gap/time equal the reported ones.
    _c_bound = obj_val if status == "optimal" else None
    _c_gap = _optimal_relative_gap(obj_val) if status == "optimal" and obj_val is not None else None

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=_c_bound,
        gap=_c_gap,
        x=x_dict,
        constraint_duals=constraint_duals,
        bound_duals_lower=bound_duals_lower,
        bound_duals_upper=bound_duals_upper,
        wall_time=wall_time,
        node_count=0,
        rust_time=0.0,
        jax_time=jax_time,
        python_time=python_time,
        root_bound=_c_bound,
        root_gap=_c_gap,
        root_time=wall_time,
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
    in_tree_presolve_stride: int = 1,  # PF1 (#632): FBBT-only default; see solve_model
    in_tree_presolve_repr=None,
    rens_enabled: bool = True,
    _lns_enabled: bool = True,
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

    # Root-node certification instrumentation (cert:T0.1); see solve_model's
    # spatial-path snapshot for the rationale.
    _root_time: Optional[float] = None
    _root_glb_internal: Optional[float] = None
    # Adaptive LNS primal-improvement state (RINS + local branching). These
    # improvers existed only in solve_model's loop; syn/rsyn/clay take THIS path,
    # so they never got polished past the root incumbent (issue #267/#282). Wire
    # them in here. ``_lns_enabled`` is the recursion guard (local_branching's
    # sub-solve sets it False); soundness is preserved because every candidate is
    # re-verified feasible and injected only on strict improvement.
    _lns_has_integers = bool(int_sizes) and int(np.sum(int_sizes)) > 0
    _lns_k_schedule = (2, 5, 10)
    _lns_lb_calls = 0
    _lns_deadline = t_start + time_limit
    # Adaptive cost/benefit governor for the improver-role LNS heuristics, mirrored
    # from solve_model (see the _improver_allowed/_record_improver pair there). #321
    # ported the improvers into this path but NOT this governor, so on instances
    # where LNS never improves (e.g. clay0303hfsg, whose optimum is found early) the
    # sub-MIPs fired at every node and starved the tree — certified-in-~21 s became a
    # timeout with no bound (issue #347). The contingent is SCIP heur_subnlp-shaped:
    # once an incumbent exists an improver may run only if it fits a success-weighted,
    # node-proportional budget, so improvers that stop paying off shut themselves off.
    # Soundness is untouched: B&B stays exhaustive, so a skipped improver can only
    # cost nodes, never a wrong optimum. DISCOPT_HEUR_BUDGET=0 restores always-on.
    _heur_budget_on = os.environ.get("DISCOPT_HEUR_BUDGET", "1") != "0"
    _HEUR_BUDGET_OFFSET = float(os.environ.get("DISCOPT_HEUR_OFFSET", "0"))
    _HEUR_BUDGET_QUOT = float(os.environ.get("DISCOPT_HEUR_QUOT", "0.5"))
    _HEUR_SUCCESS_GAIN = 3.0
    _HEUR_COST = {"rins": 5.0, "lbranch": 10.0}
    _heur_state = {"calls": 0, "found": 0, "cost": 0.0}
    # G2: the hit-rate-adaptive governor (default-OFF; see heuristic_governor.py).
    _heuristic_governor = _get_heuristic_governor()

    def _improver_allowed(cost: float) -> bool:
        """Whether an improver-role LNS heuristic costing ``cost`` may run now.

        Never gated before the first incumbent (securing one for pruning wins).
        Once an incumbent exists the call must fit the success-weighted,
        node-proportional contingent (SCIP ``heur_subnlp`` shape)."""
        if not _heur_budget_on or tree.incumbent() is None:
            return True
        _nodes = float(tree.stats().get("total_nodes", 0))
        _weight = _HEUR_SUCCESS_GAIN * (_heur_state["found"] + 1) / (_heur_state["calls"] + 1)
        _contingent = _HEUR_BUDGET_OFFSET + _HEUR_BUDGET_QUOT * _nodes * _weight
        return (_heur_state["cost"] + cost) <= _contingent

    def _record_improver(cost: float, improved: bool) -> None:
        """Charge an improver-role run against the contingent and note success."""
        _heur_state["calls"] += 1
        _heur_state["cost"] += cost
        if improved:
            _heur_state["found"] += 1

    from discopt import debug as _debug

    def _debug_validate_candidate(
        x,
        _ev=evaluator,
        _cl=cl_list,
        _cu=cu_list,
        _ioff=int_offsets,
        _isz=int_sizes,
    ):
        """Validate a debugger-injected candidate against the ORIGINAL problem.

        ``inject_incumbent`` trusts its caller (no feasibility re-check), so
        the debugger's ``inject`` steer must verify integrality + constraint
        feasibility here and hand the tree the point's true evaluated
        objective — never a relaxation bound (CLAUDE.md §1). Returns
        ``(feasible, x_validated, obj)``; pure reads only.
        """
        xv = np.asarray(x, dtype=np.float64).copy()
        if _ioff and not _is_integer_feasible_solution(xv, _ioff, _isz):
            return False, xv, float("nan")
        for _o, _s in zip(_ioff, _isz):
            xv[_o : _o + _s] = np.round(xv[_o : _o + _s])
        if _cl and not _check_constraint_feasibility(_ev, xv, _cl, _cu):
            return False, xv, float("nan")
        _obj = float(_ev.evaluate_objective(xv))
        if not np.isfinite(_obj) or _obj >= _SENTINEL_THRESHOLD:
            return False, xv, float("nan")
        return True, xv, _obj

    # Set when the interactive debugger's `quit` breaks the search loop: a
    # user-interrupted exit proves nothing, so the status decision below must
    # not fall through to a certified "infeasible"/"optimal".
    _debug_quit = False
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Interactive debugger: top-of-iteration checkpoint (no-op when detached).
        if _debug.fire(
            _debug.Checkpoint.ITER_START,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=elapsed,
        ):
            _debug_quit = True
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

        # Interactive debugger: nodes selected — boxes/ids now available.
        if _debug.fire(
            _debug.Checkpoint.AFTER_SELECT,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=elapsed,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
        ):
            _debug_quit = True
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
                # P3 branch-and-reduce: enable per-node probing (default OFF) and
                # feed the current incumbent as a cutoff so both the in-tree FBBT
                # and the probing pass are optimality-aware. The incumbent value
                # is a valid upper bound on the optimum, so cutoff-driven
                # contraction never removes an improving feasible point.
                _itp_probing = _node_probing_enabled()
                _itp_probe_max = _node_probe_max_vars()
                _itp_inc = tree.incumbent()
                _itp_cutoff = (
                    float(_itp_inc[1])
                    if _itp_inc is not None
                    and np.isfinite(_itp_inc[1])
                    and _itp_inc[1] < _SENTINEL_THRESHOLD
                    else None
                )
                # Real per-node tree depth so the depth-stride gate is honest
                # (issue #632 / PF1): the old hardcode of 0 made every stride
                # fire at every node (0 % s == 0). At stride 1 this is a no-op
                # (fires everywhere either way); it only matters for stride > 1.
                _itp_depths = tree.node_depths(np.asarray(batch_ids, dtype=np.int64))
                for i in range(n_batch):
                    if len(batch_lb[i]) != n_blocks:
                        continue
                    delta = in_tree_presolve_repr.in_tree_presolve(
                        np.asarray(batch_lb[i], dtype=np.float64),
                        np.asarray(batch_ub[i], dtype=np.float64),
                        node_depth=int(_itp_depths[i]),
                        depth_stride=in_tree_presolve_stride,
                        incumbent=_itp_cutoff,
                        probing=_itp_probing,
                        probe_max_vars=_itp_probe_max,
                    )
                    if delta["ran"] and delta["infeasible"]:
                        # Rigorous fathom: the node box is empty (FBBT/probing
                        # proof). Mark infeasible so the node is pruned soundly.
                        node_infeasible_mask[i] = True
                    elif delta["ran"]:
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
                        deadline=t_start + time_limit,
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
                _root_incumbent = False
                # --- RENS (Relaxation Enforced Neighborhood Search), primary ---
                # Solve the relaxation's rounding neighbourhood EXACTLY (fix the
                # integers already integral in the relaxation; restrict each
                # fractional one to {floor, ceil}; solve the small sub-MINLP). On a
                # near-integral convex relaxation this lands the *optimal* integers
                # at the root, where the feasibility pump's all-at-once rounding
                # settles for a feasible-but-suboptimal assignment (#281:
                # smallinvDAX 0.4004 -> 0.3988). It returns cheaply when too many
                # integers are fractional, so the pump/diving fallback below still
                # covers set-partitioning-style relaxations parked at 0.5.
                #
                # G2 governor: RENS is the single largest NLP-solve source on the
                # easy class (33 % of solve wall) at a measured 0 % incumbent
                # hit-rate — the incumbent already came from the root multistart.
                # The governor throttles it once its class miss-streak proves it is
                # not paying off (default-OFF; heuristic-policy — soundness is
                # untouched since B&B stays exhaustive).
                _rens_gap_open = tree.incumbent() is None or (
                    best_root_obj < _SENTINEL_THRESHOLD
                    and float(tree.incumbent()[1]) - float(best_root_obj)
                    > gap_tolerance * (abs(float(tree.incumbent()[1])) + 1e-10)
                )
                if rens_enabled and _heuristic_governor.allowed("rens", gap_open=_rens_gap_open):
                    _rens_improved = False
                    _rens_obj0 = (
                        float(tree.incumbent()[1]) if tree.incumbent() is not None else float("inf")
                    )
                    try:
                        from discopt._jax.primal_heuristics import rens as _rens_heuristic

                        _rens_budget = max(
                            0.5,
                            min(
                                _RENS_BUDGET_FRAC * (time_limit - (time.perf_counter() - t_start)),
                                _RENS_BUDGET_CAP_S,
                            ),
                        )

                        def _rens_sub_solver(_restricted, _tl=_rens_budget):
                            # Solve the bound-restricted sub-MINLP with the full
                            # engine (rens=False prevents nested RENS; structure_cuts
                            # off — the SymPy presolve cannot help a bound-tightened
                            # MIQP). Returns a flat solution aligned to the evaluator.
                            sub = solve_model(
                                _restricted,
                                time_limit=_tl,
                                gap_tolerance=gap_tolerance,
                                nlp_solver=nlp_solver,
                                rens=False,
                                structure_cuts=False,
                            )
                            if sub.objective is None or sub.x is None:
                                return None
                            return _pack_solution(_restricted, sub.x, n_vars), float(sub.objective)

                        rens_res = _rens_heuristic(
                            model,
                            result_sols[best_root_idx],
                            sub_solver=_rens_sub_solver,
                        )
                        if rens_res is not None:
                            rens_sol = np.asarray(rens_res[0], dtype=np.float64).copy()
                            rens_obj = float(evaluator.evaluate_objective(rens_sol))
                            rens_feas = not cl_list or _check_constraint_feasibility(
                                evaluator, rens_sol, cl_list, cu_list
                            )
                            if (
                                np.isfinite(rens_obj)
                                and rens_obj < _SENTINEL_THRESHOLD
                                and rens_feas
                                and _is_integer_feasible_solution(rens_sol, int_offsets, int_sizes)
                            ):
                                tree.inject_incumbent(rens_sol, rens_obj)
                                _root_incumbent = True
                                _rens_improved = rens_obj < _rens_obj0 - 1e-9
                                logger.info("NLP-BB RENS incumbent: obj=%.6g", rens_obj)
                    except Exception as e:
                        logger.debug("RENS failed: %s", e)
                    _heuristic_governor.record("rens", _rens_improved)

                # --- Feasibility pump (fallback when RENS does not apply) ---
                if not _root_incumbent:
                    try:
                        from discopt._jax.primal_heuristics import feasibility_pump

                        fp_sol = feasibility_pump(
                            model,
                            result_sols[best_root_idx],
                            max_rounds=5,
                            backend=_resolve_heuristic_backend(nlp_solver),
                            evaluator=evaluator,
                            deadline=t_start + time_limit,
                        )
                        if fp_sol is not None:
                            fp_obj = float(evaluator.evaluate_objective(fp_sol))
                            fp_feas = not cl_list or _check_constraint_feasibility(
                                evaluator, fp_sol, cl_list, cu_list
                            )
                            if np.isfinite(fp_obj) and fp_obj < _SENTINEL_THRESHOLD and fp_feas:
                                tree.inject_incumbent(fp_sol, fp_obj)
                                _root_incumbent = True
                                logger.info("NLP-BB feasibility pump incumbent: obj=%.6g", fp_obj)
                    except Exception as e:
                        logger.debug("Feasibility pump failed: %s", e)

                # Fractional-diving fallback (issue #268). The feasibility pump
                # rounds every integer at once, which lands on a constraint-
                # infeasible assignment on facility-location / set-partitioning
                # MINLPs (clay*/flay*/syn*) whose relaxation parks every binary
                # near 0.5 — the pump then returns nothing and the convex NLP-BB
                # exhausts with no incumbent (reported as a no-solution time_limit).
                # Diving fixes integers one at a time, re-solving the NLP between
                # fixings, so it finds a feasible incumbent where all-at-once
                # rounding cannot. Only tried when the pump found no incumbent;
                # bounded by ~n_int sub-NLP solves and the remaining time budget.
                # Sound: diving re-verifies integer + constraint feasibility and
                # inject_incumbent enforces strict improvement, so the dual bound
                # and gap certification are untouched.
                if not _root_incumbent and (time.perf_counter() - t_start) < time_limit:
                    try:
                        from discopt._jax.primal_heuristics import fractional_diving

                        dv = fractional_diving(
                            model,
                            result_sols[best_root_idx],
                            backend=_resolve_heuristic_backend(nlp_solver),
                            evaluator=evaluator,
                            # F4: poll the absolute deadline between dive sub-NLPs so
                            # the ~n_int-solve dive cannot run past a tight limit.
                            deadline=t_start + time_limit,
                        )
                        if dv is not None:
                            dv_sol, dv_obj = dv
                            dv_obj = float(dv_obj)
                            dv_feas = not cl_list or _check_constraint_feasibility(
                                evaluator, dv_sol, cl_list, cu_list
                            )
                            if np.isfinite(dv_obj) and dv_obj < _SENTINEL_THRESHOLD and dv_feas:
                                tree.inject_incumbent(np.asarray(dv_sol).copy(), dv_obj)
                                logger.info("NLP-BB fractional diving incumbent: obj=%.6g", dv_obj)
                    except Exception as e:
                        logger.debug("Fractional diving failed: %s", e)

        # --- User callbacks ---
        # INT-1 (#413): the NLP-BB path cannot honor a callback that rejects an
        # integer-feasible point — it has no per-node cut application and its
        # primal heuristics inject incumbents without consulting the callback.
        # ``solve_model`` refuses these callbacks before routing here (explicit
        # ``nlp_bb=True`` raises; auto-select falls through to spatial B&B). This
        # is a fail-loud backstop: if a rejecting callback ever reaches this path
        # it is a routing bug, and we must NOT silently drop the rejection (which
        # would accept the excluded point as the optimum) by calling
        # ``_invoke_pre_import_callbacks`` with a ``None`` cut pool.
        if lazy_constraints is not None or incumbent_callback is not None:
            raise AssertionError(
                "lazy_constraints/incumbent_callback reached the NLP-BB path, "
                "which cannot enforce them (INT-1, #413). This is a routing bug: "
                "solve_model should have refused or rerouted to spatial B&B."
            )

        # Interactive debugger: steer point — relaxations solved, not imported.
        # `inject` candidates are validated by _debug_validate_candidate.
        if _debug.fire(
            _debug.Checkpoint.BEFORE_IMPORT,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=time.perf_counter() - t_start,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
            result_lbs=result_lbs,
            result_sols=result_sols,
            result_feas=result_feas,
            validator=_debug_validate_candidate,
        ):
            _debug_quit = True
            break

        # Import results back to Rust tree
        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        # Interactive debugger: prune/branch/fathom applied by the tree.
        if _debug.fire(
            _debug.Checkpoint.AFTER_PROCESS,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=time.perf_counter() - t_start,
        ):
            _debug_quit = True
            break

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
                from discopt.modeling.core import ObjectiveSense as _ObjSense

                _cb_is_max = (
                    model._objective is not None and model._objective.sense == _ObjSense.MAXIMIZE
                )
                _cb_bound = _certified_callback_bound(
                    stats_snap.get("global_lower_bound"), _gap_certified, _cb_is_max
                )
                ctx = CallbackContext(
                    node_count=stats_snap["total_nodes"],
                    incumbent_obj=inc_obj_cb,
                    best_bound=_cb_bound,
                    # A2: no certified bound -> no meaningful gap (the raw gap was
                    # derived from the same non-bound). See the pre-import site.
                    gap=(stats_snap.get("gap") if _cb_bound is not None else None),
                    elapsed_time=time.perf_counter() - t_start,
                    x_relaxation=result_sols[best_idx].copy(),
                    node_bound=float(result_lbs[best_idx]),
                )
                node_callback(ctx, model)
            except Exception as e:
                logger.warning("Node callback raised an exception: %s", e)

        # --- LNS primal-improvement layer (RINS + local branching) ---
        # Runs at non-root nodes when an incumbent exists and the gap is still
        # open. Sound by construction: each candidate is re-verified integer- and
        # constraint-feasible and injected only on strict improvement, so the dual
        # bound is never touched. Gated off in the local-branching sub-solve via
        # ``_lns_enabled=False`` so it can never recurse.
        if (
            _lns_enabled
            and _lns_has_integers
            and iteration > 0
            and (_lns_deadline - time.perf_counter()) > _DEADLINE_NODE_FLOOR_S
        ):
            _lns_inc = tree.incumbent()
            if (
                _lns_inc is not None
                and np.isfinite(_lns_inc[1])
                and _lns_inc[1] < _SENTINEL_THRESHOLD
            ):
                _lns_stats = tree.stats()
                _lns_lb = float(_lns_stats.get("global_lower_bound", float("-inf")))
                _lns_ub = float(_lns_inc[1])
                _lns_abs_gap = max(0.0, _lns_ub - _lns_lb)
                _lns_denom = max(abs(_lns_ub), abs(_lns_lb), 1e-10)
                _lns_gap_open = (not np.isfinite(_lns_lb)) or (
                    _lns_abs_gap > _DEFAULT_ABS_GAP_TOL
                    and _lns_abs_gap / _lns_denom > gap_tolerance
                )
                if _lns_gap_open:
                    _lns_backend = _resolve_heuristic_backend(nlp_solver)
                    _lns_best_idx = None
                    _lns_best = np.inf
                    for _i in range(n_batch):
                        if result_lbs[_i] < _SENTINEL_THRESHOLD and result_lbs[_i] < _lns_best:
                            _lns_best = result_lbs[_i]
                            _lns_best_idx = _i
                    # (1) RINS — search between incumbent and the node relaxation.
                    if (
                        _lns_best_idx is not None
                        and (_lns_deadline - time.perf_counter()) > _DEADLINE_NODE_FLOOR_S
                        and _improver_allowed(_HEUR_COST["rins"])
                        and _heuristic_governor.allowed("rins", gap_open=True)
                    ):
                        _rins_obj0 = float(_lns_inc[1])
                        _rins_improved = False
                        try:
                            from discopt._jax.primal_heuristics import rins

                            _ri = rins(
                                model,
                                np.asarray(_lns_inc[0], dtype=np.float64),
                                result_sols[_lns_best_idx],
                                backend=_lns_backend,
                                evaluator=evaluator,
                                deadline=_lns_deadline,
                            )
                            if _ri is not None:
                                _x_ri, _obj_ri = _ri
                                _obj_ri = float(_obj_ri)
                                if (
                                    np.isfinite(_obj_ri)
                                    and _obj_ri < _SENTINEL_THRESHOLD
                                    and (
                                        not cl_list
                                        or _check_constraint_feasibility(
                                            evaluator, _x_ri, cl_list, cu_list
                                        )
                                    )
                                ):
                                    tree.inject_incumbent(np.asarray(_x_ri).copy(), _obj_ri)
                                    _rins_improved = _obj_ri < _rins_obj0 - 1e-9
                                    logger.info("NLP-BB LNS RINS incumbent: obj=%.6g", _obj_ri)
                        except Exception as _e:
                            logger.debug("NLP-BB LNS RINS failed: %s", _e)
                        _record_improver(_HEUR_COST["rins"], _rins_improved)
                        _heuristic_governor.record("rins", _rins_improved)
                    # (2) Local branching — Hamming-ball sub-MIP, escalating k.
                    if (
                        (_lns_deadline - time.perf_counter()) > 1.0
                        and _improver_allowed(_HEUR_COST["lbranch"])
                        and _heuristic_governor.allowed("lbranch", gap_open=True)
                    ):
                        _lb_k = _lns_k_schedule[min(_lns_lb_calls, len(_lns_k_schedule) - 1)]
                        _lns_lb_calls += 1
                        _lb_slice = min(2.0, max(0.5, _lns_deadline - time.perf_counter() - 0.2))
                        _lb_obj0 = float(_lns_inc[1])
                        _lb_improved = False
                        try:
                            from discopt._jax.primal_heuristics import local_branching

                            _lb = local_branching(
                                model,
                                np.asarray(_lns_inc[0], dtype=np.float64),
                                k=_lb_k,
                                backend=_lns_backend,
                                evaluator=evaluator,
                                submip_time_limit=_lb_slice,
                                submip_max_nodes=1000,
                                submip_gap_tolerance=gap_tolerance,
                                # F1: hand the enumeration a hard budget (profile
                                # §1.1) so it cannot blow past the slice/deadline.
                                deadline=_lns_deadline,
                                node_bound=_lns_lb,
                                incumbent_obj=float(_lns_inc[1]),
                                gap_tolerance=gap_tolerance,
                            )
                            if _lb is not None:
                                _x_lb, _obj_lb = _lb
                                _obj_lb = float(_obj_lb)
                                if (
                                    np.isfinite(_obj_lb)
                                    and _obj_lb < _SENTINEL_THRESHOLD
                                    and (
                                        not cl_list
                                        or _check_constraint_feasibility(
                                            evaluator, _x_lb, cl_list, cu_list
                                        )
                                    )
                                ):
                                    tree.inject_incumbent(np.asarray(_x_lb).copy(), _obj_lb)
                                    _lb_improved = _obj_lb < _lb_obj0 - 1e-9
                                    logger.info(
                                        "NLP-BB LNS local-branching incumbent: obj=%.6g (k=%d)",
                                        _obj_lb,
                                        _lb_k,
                                    )
                        except Exception as _e:
                            logger.debug("NLP-BB LNS local-branching failed: %s", _e)
                        _record_improver(_HEUR_COST["lbranch"], _lb_improved)
                        _heuristic_governor.record("lbranch", _lb_improved)

        # Root-node certification snapshot (cert:T0.1).
        if iteration == 0:
            _root_time = time.perf_counter() - t_start
            _root_glb_snap = tree.stats().get("global_lower_bound")
            if _root_glb_snap is not None and np.isfinite(_root_glb_snap):
                _root_glb_internal = float(_root_glb_snap)

        iteration += 1

        # Check termination
        if tree.is_finished():
            break
        if _gap_converged(tree, gap_tolerance):
            break
        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    # --- Build result ---
    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time

    stats = tree.stats()
    incumbent = tree.incumbent()

    # R3a instrumentation (behavior-neutral): export the per-variable branch
    # frequency when the diagnostic sink is armed. Mirrors the capture in the
    # McCormick-LP nonconvex driver so alphaBB/NLP-path instances (e.g. tls2)
    # are covered too.
    if _R3A_BRANCH_COUNT_SINK is not None:
        try:
            _R3A_BRANCH_COUNT_SINK["branch_var_counts"] = np.asarray(
                tree.branch_var_counts()
            ).tolist()
        except Exception as _e:  # pragma: no cover - diagnostics only
            logger.debug("R3a branch_var_counts capture failed: %s", _e)

    if incumbent is not None:
        sol_array, obj_val = incumbent
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    constraint_duals = None
    bound_duals_lower = None
    bound_duals_upper = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        # C-3: snap near-integral discrete coordinates to exact integers before
        # reporting; the dual-recovery re-solve below is best-effort and does
        # not guarantee the reported primal is rounded (see helper docstring).
        _rounded_inc, _rounded_feas = _round_incumbent_integers(
            sol_flat, int_offsets, int_sizes, evaluator, cl_list, cu_list
        )
        if _rounded_feas:
            sol_flat = _rounded_inc
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
        if (_gap_converged(tree, gap_tolerance) or tree.is_finished()) and _gap_certified:
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
        elif _debug_quit:
            # Interactive debugger `quit`: the user interrupted the search, so
            # the tree is NOT exhausted and neither infeasibility nor optimality
            # was proven (a false "infeasible" here is the worst-class error).
            # Report "unknown", uncertified.
            status = "unknown"
            _gap_certified = False
        else:
            # Tree exhausted with no feasible node and every fathom was a valid
            # certificate (FBBT-empty box or SolveStatus.INFEASIBLE): infeasibility
            # *is* a certified conclusion, so leave _gap_certified untouched.
            status = "infeasible"

    # Interactive debugger: terminal checkpoint. Fired after the status
    # decision so ``debug="on-error"`` sessions can key on the outcome:
    # ``error`` is the non-"optimal" status ("time_limit", "unknown",
    # "infeasible", ...), or None on a certified optimum.
    _debug.fire(
        _debug.Checkpoint.TERMINATED,
        tree=tree,
        model=model,
        iteration=iteration,
        elapsed=wall_time,
        error=None if status == "optimal" else status,
    )

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
    # (max_nodes=1 false-certification regression). But the untainted tree bound is
    # itself a valid global dual bound (every node floored at its valid parent
    # bound), so keep it and recompute the gap rather than dropping to None (#138).
    _tree_bound_valid = _gap_certified
    if status == "feasible":
        _gap_certified = False

    if not _gap_certified:
        gap_val = None
        if _tree_bound_valid and bound_val is not None and np.isfinite(bound_val):
            if obj_val is not None and np.isfinite(obj_val):
                gap_val = abs(obj_val - bound_val) / max(1.0, abs(obj_val))
        else:
            bound_val = None

    # Re-earn certification when the retained valid tree bound meets the incumbent
    # within tolerance: the incumbent is then provably global and the honest status
    # is "optimal" (mirrors the spatial path's re-certification).
    if (
        status == "feasible"
        and obj_val is not None
        and bound_val is not None
        and np.isfinite(bound_val)
        and gap_val is not None
        and gap_val <= gap_tolerance
    ):
        _is_max = model._objective.sense == ObjectiveSense.MAXIMIZE
        if (bound_val >= obj_val - 1e-9) if _is_max else (bound_val <= obj_val + 1e-9):
            _gap_certified = True
            status = "optimal"

    # Root-node certification metrics (cert:T0.1); see solve_model for the
    # sense conversion and gap convention.
    _nlpbb_is_max = model._objective.sense == ObjectiveSense.MAXIMIZE
    root_bound_val: Optional[float] = None
    root_gap_val: Optional[float] = None
    if _root_glb_internal is not None and np.isfinite(_root_glb_internal):
        root_bound_val = -_root_glb_internal if _nlpbb_is_max else _root_glb_internal
        if obj_val is not None and np.isfinite(obj_val):
            root_gap_val = abs(obj_val - root_bound_val) / max(1.0, abs(obj_val))

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
        root_bound=root_bound_val,
        root_gap=root_gap_val,
        root_time=_root_time,
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
        # Native path (discopt#281): solve the .nl directly via POUNCE's own AD,
        # bypassing the JAX callbacks. On a non-accept status fall through to the
        # JAX path, which carries the alternative-start retry / convex polish
        # robustness layer; this never loses a usable result.
        if _native_nlp_enabled(options):
            from discopt.solvers import SolveStatus as _SS
            from discopt.solvers.nlp_native import get_native_base, solve_node_native

            nb = get_native_base(evaluator)
            if nb is not None:
                res = solve_node_native(nb, x0, node_lb, node_ub, options)
                if res.status in (_SS.OPTIMAL, _SS.ITERATION_LIMIT):
                    return res
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

    # F4: each ``_attempt`` is a full POUNCE solve that OVERRUNS its own
    # ``max_wall_time`` clamp on the expensive-Hessian class (a nominal few-second
    # budget ran ~10 s on heatexch_gen3), so a first attempt plus two alternative-
    # start retries can push a single node NLP tens of seconds past a tight
    # ``time_limit``. Poll an absolute deadline derived from the caller's remaining
    # budget (``max_wall_time``) between retries and stop once it has passed. This
    # never weakens correctness: a skipped retry only leaves the (already-computed)
    # first-attempt result, and a node whose NLP failed stays OPEN at its inherited
    # parent bound — a valid global lower bound — so the dual bound is untouched.
    _t_node_entry = time.perf_counter()
    _retry_deadline = _t_node_entry + max(_DEADLINE_NODE_FLOOR_S, float(caller_limit))
    result = _attempt(np.asarray(x0, dtype=np.float64), opts)
    if result.status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
        for alt in (midpoint, off_center):
            if np.allclose(alt, x0, atol=1e-12):
                continue
            if time.perf_counter() >= _retry_deadline:
                break
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

    # Native-AD path (discopt#281): when available, each (node, start) becomes a
    # cheap bound variant of one parsed .nl problem solved by POUNCE's own AD —
    # no JAX callbacks. ``native_base`` is None (→ JAX callback Problems) when the
    # model has no usable .nl representation. Result vectors come back in .nl
    # column order and are mapped to evaluator order via ``to_eval_order``.
    native_base = None
    if _native_nlp_enabled(options):
        from discopt.solvers.nlp_native import get_native_base

        native_base = get_native_base(evaluator)

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
            if native_base is not None:
                # Native .nl problem: a per-node bound variant (shares the parsed
                # DAG / AD tapes). Bounds and start go in .nl column order.
                problems.append(native_base.variant(node_lb, node_ub, x0))
                x0s.append(native_base.to_nl_order(np.asarray(x0, dtype=np.float64)))
            else:
                # One callbacks proxy per problem so concurrent Rayon workers
                # never share mutable Python state. The JAX evaluator is
                # pure/reentrant.
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
            # x0s are in .nl column order when native; the JAX serial path wants
            # evaluator order.
            warm0 = x0s[i * n_starts]
            if native_base is not None:
                warm0 = native_base.to_eval_order(warm0)
            res = _solve_node_nlp_pounce(
                evaluator,
                warm0,
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
            # Native results come back in .nl column order; map to evaluator
            # order so downstream bound clips and the returned solution align.
            x_arr = (
                native_base.to_eval_order(x)
                if native_base is not None
                else np.asarray(x, dtype=np.float64)
            )
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

# Reject an interior-point QP "optimal" whose reported final KKT residual exceeds
# this (issue #145 drift guard for the POUNCE-first default). Loose enough to pass
# a normally-converged IPM solve (observed ~1e-9, even on a 1e6-conditioned Q),
# tight enough to catch a stalled one; the caller then degrades to the next
# engine. A backend that reports no residual skips the check.
_QP_KKT_RESIDUAL_TOL = 1e-6


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

    Pass ``Q_orig`` for QP-style relaxations; omit it for LP-style. QP/MIQP
    recovery is HiGHS-free and always uses POUNCE (issue #359); LP/MILP recovery
    uses POUNCE under ``prefer_pounce`` and the pure-Rust simplex otherwise
    (issue #356) — both expose HiGHS-convention duals. This is reporting only:
    the recovered duals populate SolveResult sensitivity, they are not consumed
    for bound tightening. Returns ``(None, None, None)`` if recovery is
    unavailable (solver missing, the fix-and-resolve LP/QP itself fails, or
    layout mismatch).

    Bound multipliers on the fixing bounds for integer columns are zeroed in
    the returned dicts — they reflect the act of fixing, not feasibility of
    the original integer-feasible point.
    """
    try:
        if Q_orig is not None:
            # QP/MIQP dual recovery is HiGHS-free (issue #359): always POUNCE.
            from discopt.solvers.qp_pounce import solve_qp as _recover_qp
        elif prefer_pounce:
            from discopt.solvers.lp_pounce import solve_lp as _recover_lp
        else:
            from discopt.solvers.lp_simplex import (  # type: ignore[assignment]
                solve_lp as _recover_lp,
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
            relax = _recover_lp(
                c=c_orig,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds_fixed,
                time_limit=time_limit,
            )
        else:
            relax = _recover_qp(
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
    """Solve an LP through the pure-Rust simplex (then POUNCE if installed).

    Engine order is Rust simplex -> POUNCE, or POUNCE -> Rust simplex when
    ``prefer_pounce`` is set (the user passed ``nlp_solver="pounce"``). The
    fragile JAX LP-IPM last resort was **retired** in issue #364: the hardened
    pure-Rust simplex (iterative refinement, condition/growth signals, dual
    anti-cycling, EXPAND anti-degeneracy) is the single robust LP engine, so a
    genuine simplex(+POUNCE) failure now reports an honest ``error`` rather than
    falling to the IPM — which returned NaN via Newton blow-up on declared bounds
    exceeding ~1e15, i.e. was never a sound last resort.
    """
    engines = [_solve_lp_simplex, _solve_lp_pounce]
    if prefer_pounce:
        engines.reverse()
    for engine in engines:
        result = engine(model, t_start, time_limit)
        if result is not None:
            return result

    # Single robust engine (issue #364): no JAX LP-IPM fallback. Both the simplex
    # and POUNCE (if installed) returned None — a binding that is unavailable or a
    # genuine numerical failure — so report it honestly instead of trusting the
    # IPM's large-bound NaN. The Rust simplex is always built in, so in practice
    # this is only reached on a true solve failure the IPM could not have fixed.
    wall_time = time.perf_counter() - t_start
    return SolveResult(
        status="error",
        objective=None,
        bound=None,
        wall_time=wall_time,
        node_count=0,
    )


def _solve_lp_simplex(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
) -> SolveResult | None:
    """Solve an LP using the pure-Rust warm-started simplex. Returns None when
    the simplex binding is unavailable or fails, so the caller can fall back to
    another engine."""
    from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE
    from discopt.solvers.lp_simplex import solve_lp as _simplex_solve_lp

    if not SIMPLEX_AVAILABLE:
        return None
    return _solve_lp_matrix(model, t_start, time_limit, _simplex_solve_lp, "simplex")


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


def _solve_lp_gurobi(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
    threads: int | None = None,
    options: Optional[dict] = None,
) -> SolveResult:
    """Solve an LP using the explicit Gurobi backend."""
    import functools

    from discopt.solvers.gurobi import solve_lp as _gurobi_solve_lp

    solve_fn = functools.partial(_gurobi_solve_lp, threads=threads, options=options)
    result = _solve_lp_matrix(
        model,
        t_start,
        time_limit,
        solve_fn,
        "Gurobi",
        strict=True,
    )
    if result is None:  # pragma: no cover - strict mode raises before this
        raise RuntimeError("Gurobi LP solve failed without returning a result.")
    return result


def _solve_lp_matrix(
    model: Model,
    t_start: float,
    time_limit: float | None,
    solve_lp_fn,
    engine: str,
    strict: bool = False,
) -> SolveResult | None:
    """Solve a pure LP through a matrix-form ``solve_lp`` backend.

    ``solve_lp_fn`` must follow the shared LP contract (lp_simplex / lp_pounce):
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
        if strict:
            raise
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
    """Solve a QP with discopt's own engines — POUNCE, then the JAX QP IPM.

    HiGHS-free by design (issue #359 / pure-Rust goal): a continuous QP is solved
    by POUNCE (the pure-Rust Ipopt port), and a POUNCE failure or a non-converged
    solve degrades to discopt's JAX QP interior-point method — never to HiGHS. The
    POUNCE engine handles pure-continuous QPs only; MIQPs return ``None`` from it
    and route to the self-hosted B&B path. ``prefer_pounce`` is retained for
    call-site compatibility but no longer selects between backends (there is only
    one default backend now).

    Soundness: QP duals/reduced costs are reported, never consumed for bound
    tightening (OBBT/DBBT read the LP oracles), so the only hazard is a drifted
    objective on an unconverged solve (#145). ``_solve_qp_matrix`` guards it — the
    returned point is re-checked for primal feasibility and for a stationary KKT
    residual, degrading to the next engine (the JAX IPM) on failure.

    No-rescue tracking: with HiGHS gone the JAX IPM is a weak last resort (it can
    return ``iteration_limit`` even on easy QPs). A POUNCE non-result is therefore
    logged at WARNING with the marker ``qp-pounce-no-result`` so we can measure how
    often the HiGHS-free path has no working engine and decide later whether a
    pure-Rust drift-rescue is warranted (issue #359).
    """
    del prefer_pounce  # no HiGHS fallback to order against; kept for signature compat
    result = _solve_qp_pounce(model, t_start)
    if result is not None:
        return result
    from discopt.solvers.qp_pounce import POUNCE_AVAILABLE

    if POUNCE_AVAILABLE:
        logger.warning(
            "HiGHS-free QP [qp-pounce-no-result]: POUNCE was available but returned "
            "no usable result (solve failure or feasibility/convergence guard "
            "rejection); falling back to the JAX QP IPM last resort, which has no "
            "robust rescue. Track how often this fires (issue #359)."
        )
    else:
        logger.warning(
            "HiGHS-free QP [qp-pounce-unavailable]: pounce-solver is not installed, "
            "so the QP path has no primary engine and will use the JAX QP IPM last "
            "resort. Install pounce-solver for a working QP solver."
        )
    return _solve_qp_jax(model, t_start)


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


def _solve_qp_gurobi(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
    gap_tolerance: float = 1e-4,
    threads: int | None = None,
    options: Optional[dict] = None,
) -> SolveResult:
    """Solve a QP/MIQP using the explicit Gurobi backend."""
    import functools

    from discopt.solvers.gurobi import solve_qp as _gurobi_solve_qp

    solve_fn = functools.partial(
        _gurobi_solve_qp,
        threads=threads,
        options=options,
    )
    result = _solve_qp_matrix(
        model,
        t_start,
        time_limit,
        solve_fn,
        "Gurobi",
        gap_tolerance=gap_tolerance,
        strict=True,
    )
    if result is None:  # pragma: no cover - strict mode raises before this
        raise RuntimeError("Gurobi QP/MIQP solve failed without returning a result.")
    return result


def _quadratic_rows_solution_feasible(x, quadratic_constraints, tol=1e-6) -> bool:
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return False
    x_scale = 1.0 + float(np.max(np.abs(x))) if x.size else 1.0
    for row in quadratic_constraints:
        Q = np.asarray(row.Q, dtype=np.float64)
        c = np.asarray(row.c, dtype=np.float64)
        rhs = float(row.rhs)
        value = float(0.5 * x @ Q @ x + c @ x)
        scale = x_scale + abs(rhs) + abs(value)
        if row.sense == "<=" and value > rhs + tol * scale:
            return False
        if row.sense == ">=" and value < rhs - tol * scale:
            return False
        if row.sense == "==" and abs(value - rhs) > tol * scale:
            return False
    return True


def _solve_qcp_gurobi(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
    gap_tolerance: float = 1e-4,
    threads: int | None = None,
    options: Optional[dict] = None,
) -> SolveResult:
    """Solve a QCP/QCQP/MIQCP/MIQCQP using the explicit Gurobi backend."""
    from discopt._jax.problem_classifier import extract_qcp_data
    from discopt.modeling.core import ObjectiveSense
    from discopt.solvers import SolveStatus
    from discopt.solvers.gurobi import solve_qcp as _gurobi_solve_qcp

    qcp_data = extract_qcp_data(model)
    n_orig = sum(v.size for v in model._variables)

    bounds = list(
        zip(
            np.asarray(qcp_data.x_l[:n_orig]).tolist(),
            np.asarray(qcp_data.x_u[:n_orig]).tolist(),
        )
    )

    A_ub = np.asarray(qcp_data.A_ub, dtype=np.float64)
    b_ub = np.asarray(qcp_data.b_ub, dtype=np.float64)
    A_eq = np.asarray(qcp_data.A_eq, dtype=np.float64)
    b_eq = np.asarray(qcp_data.b_eq, dtype=np.float64)
    A_ub_arg = A_ub if A_ub.shape[0] else None
    b_ub_arg = b_ub if b_ub.shape[0] else None
    A_eq_arg = A_eq if A_eq.shape[0] else None
    b_eq_arg = b_eq if b_eq.shape[0] else None

    integrality = None
    if any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables):
        int_arr = np.zeros(n_orig, dtype=np.int32)
        offset = 0
        for v in model._variables:
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                int_arr[offset : offset + v.size] = 1
            offset += v.size
        integrality = int_arr

    result = _gurobi_solve_qcp(
        Q=np.asarray(qcp_data.Q[:n_orig, :n_orig]),
        c=np.asarray(qcp_data.c[:n_orig]),
        A_ub=A_ub_arg,
        b_ub=b_ub_arg,
        A_eq=A_eq_arg,
        b_eq=b_eq_arg,
        bounds=bounds,
        quadratic_constraints=qcp_data.quadratic_constraints,
        integrality=integrality,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        threads=threads,
        options=options,
    )

    wall_time = time.perf_counter() - t_start
    assert model._objective is not None
    sense = model._objective.sense

    objective = None
    if result.objective is not None:
        objective = float(result.objective) + float(qcp_data.obj_const)
        if sense == ObjectiveSense.MAXIMIZE:
            objective = -objective

    bound = None
    if result.status == SolveStatus.OPTIMAL and objective is not None:
        bound = objective
    elif result.bound is not None:
        bound = float(result.bound) + float(qcp_data.obj_const)
        if sense == ObjectiveSense.MAXIMIZE:
            bound = -bound

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and objective is not None
        x_flat = np.asarray(result.x[:n_orig], dtype=np.float64)
        if not _matrix_solution_feasible(x_flat, A_ub_arg, b_ub_arg, A_eq_arg, b_eq_arg, bounds):
            raise RuntimeError("Gurobi QCP returned an infeasible point labeled optimal.")
        if not _quadratic_rows_solution_feasible(x_flat, qcp_data.quadratic_constraints):
            raise RuntimeError("Gurobi QCP returned a quadratic-row-infeasible optimal point.")
        return SolveResult(
            status="optimal",
            objective=objective,
            bound=objective,
            gap=result.gap if result.gap is not None else _optimal_relative_gap(objective),
            x=_unpack_solution(model, x_flat),
            wall_time=wall_time,
            node_count=result.node_count,
            rust_time=0.0,
            jax_time=0.0,
            python_time=wall_time,
        )
    if result.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=wall_time, node_count=result.node_count)
    if result.status == SolveStatus.UNBOUNDED:
        return SolveResult(status="unbounded", wall_time=wall_time, node_count=result.node_count)
    if result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(
            status="time_limit",
            objective=objective,
            bound=bound,
            gap=_relative_gap_from_objective_bound(objective, bound),
            x=_unpack_solution(model, result.x[:n_orig]) if result.x is not None else None,
            wall_time=wall_time,
            node_count=result.node_count,
        )
    if result.status == SolveStatus.ITERATION_LIMIT:
        return SolveResult(
            status="iteration_limit",
            objective=objective,
            bound=bound,
            gap=_relative_gap_from_objective_bound(objective, bound),
            x=_unpack_solution(model, result.x[:n_orig]) if result.x is not None else None,
            wall_time=wall_time,
            node_count=result.node_count,
        )
    return SolveResult(status="error", wall_time=wall_time, node_count=result.node_count)


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
    gap_tolerance: float = 1e-4,
    strict: bool = False,
) -> SolveResult | None:
    """Solve a QP/MIQP through a matrix-form ``solve_qp`` backend.

    ``solve_qp_fn`` must follow the shared QP contract (qp_pounce, or the
    optional Gurobi wrapper): same signature, same ``QPResult`` with
    HiGHS-convention duals.
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
            gap_tolerance=gap_tolerance,
        )
    except Exception as e:
        if strict:
            raise
        logger.debug("%s QP solve failed: %s", engine, e)
        return None

    wall_time = time.perf_counter() - t_start
    assert model._objective is not None
    sense = model._objective.sense

    objective = None
    if result.objective is not None:
        objective = float(result.objective) + float(qp_data.obj_const)
        if sense == ObjectiveSense.MAXIMIZE:
            objective = -objective

    bound = None
    result_bound = getattr(result, "bound", None)
    if result.status == SolveStatus.OPTIMAL and objective is not None:
        bound = objective
    elif result_bound is not None:
        bound = float(result_bound) + float(qp_data.obj_const)
        if sense == ObjectiveSense.MAXIMIZE:
            bound = -bound

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and result.objective is not None
        if not _matrix_solution_feasible(result.x[:n_orig], A_ub, b_ub, A_eq, b_eq, bounds):
            logger.warning(
                "%s QP returned an infeasible point labeled optimal; "
                "falling back to the next engine.",
                engine,
            )
            return None
        # Convergence guard for the POUNCE-first default: an interior-point
        # backend can label a stalled, drifted point "optimal" (issue #145). When
        # it reports a final KKT residual, reject a non-stationary "optimal" and
        # degrade to the next engine (the JAX QP IPM) rather than trust a drifted
        # objective. ``None`` (a backend that reports no residual) skips the check.
        if result.kkt_error is not None and result.kkt_error > _QP_KKT_RESIDUAL_TOL:
            logger.warning(
                "%s QP reported a non-stationary 'optimal' (KKT residual %.2e > %.0e); "
                "falling back to the next engine.",
                engine,
                result.kkt_error,
                _QP_KKT_RESIDUAL_TOL,
            )
            return None
        x_flat = result.x[:n_orig]
        assert objective is not None

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
            objective=objective,
            bound=objective,
            gap=result.gap if result.gap is not None else _optimal_relative_gap(objective),
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
    elif result.status == SolveStatus.UNBOUNDED:
        return SolveResult(status="unbounded", wall_time=wall_time, node_count=result.node_count)
    elif result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(
            status="time_limit",
            objective=objective,
            bound=bound,
            gap=_relative_gap_from_objective_bound(objective, bound),
            x=_unpack_solution(model, result.x[:n_orig]) if result.x is not None else None,
            wall_time=wall_time,
            node_count=result.node_count,
        )
    elif result.status == SolveStatus.ITERATION_LIMIT:
        return SolveResult(
            status="iteration_limit",
            objective=objective,
            bound=bound,
            gap=_relative_gap_from_objective_bound(objective, bound),
            x=_unpack_solution(model, result.x[:n_orig]) if result.x is not None else None,
            wall_time=wall_time,
            node_count=result.node_count,
        )

    return None


def _solve_milp_gurobi(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
    gap_tolerance: float = 1e-4,
    threads: int | None = None,
    options: Optional[dict] = None,
) -> SolveResult:
    """Solve a MILP using the explicit Gurobi backend."""
    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.modeling.core import ObjectiveSense
    from discopt.solvers import SolveStatus
    from discopt.solvers.gurobi import solve_milp as _gurobi_solve_milp

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

    result = _gurobi_solve_milp(
        c=np.asarray(lp_data.c[:n_orig]),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        integrality=int_arr,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        threads=threads,
        options=options,
    )

    wall_time = time.perf_counter() - t_start
    assert model._objective is not None
    sense = model._objective.sense

    objective = None
    if result.objective is not None:
        objective = float(result.objective) + float(lp_data.obj_const)
        if sense == ObjectiveSense.MAXIMIZE:
            objective = -objective

    # ``result.bound`` is a valid lower bound for the internal minimization.
    # Map it back to the original sense: lower bound for minimize, upper bound
    # for maximize (matching discopt's existing SolveResult convention).
    bound = None
    if result.status == SolveStatus.OPTIMAL and objective is not None:
        bound = objective
    elif result.bound is not None:
        bound = float(result.bound) + float(lp_data.obj_const)
        if sense == ObjectiveSense.MAXIMIZE:
            bound = -bound

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and objective is not None
        cd, bdl, bdu = _mip_recover_relaxation_duals(
            model,
            lp_data=lp_data,
            x_flat=np.asarray(result.x[:n_orig], dtype=float),
            n_orig=n_orig,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            time_limit=time_limit,
        )
        return SolveResult(
            status="optimal",
            objective=objective,
            bound=bound,
            gap=result.gap if result.gap is not None else 0.0,
            x=_unpack_solution(model, result.x[:n_orig]),
            wall_time=wall_time,
            node_count=result.node_count,
            rust_time=0.0,
            jax_time=0.0,
            python_time=wall_time,
            constraint_duals=cd,
            bound_duals_lower=bdl,
            bound_duals_upper=bdu,
        )

    if result.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=wall_time, node_count=result.node_count)
    if result.status == SolveStatus.UNBOUNDED:
        return SolveResult(status="unbounded", wall_time=wall_time, node_count=result.node_count)
    if result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(
            status="time_limit",
            objective=objective,
            bound=bound,
            gap=result.gap,
            x=_unpack_solution(model, result.x[:n_orig]) if result.x is not None else None,
            wall_time=wall_time,
            node_count=result.node_count,
        )
    if result.status == SolveStatus.ITERATION_LIMIT:
        return SolveResult(
            status="iteration_limit",
            objective=objective,
            bound=bound,
            gap=result.gap,
            x=_unpack_solution(model, result.x[:n_orig]) if result.x is not None else None,
            wall_time=wall_time,
            node_count=result.node_count,
        )
    return SolveResult(status="error", wall_time=wall_time, node_count=result.node_count)


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


def _pounce_qp_relaxation_nodes(qp_data, batch_lb, batch_ub, n_orig, t_start, time_limit):
    """Solve a batch of MIQP B&B node QP relaxations with POUNCE (JAX-free).

    Each node solves the augmented standard-form QP
    ``min 0.5 x'Qx + c'x  s.t. A_eq x = b_eq, lb <= x <= ub`` (x includes the
    inequality slacks; their bounds are ``[0, +inf)``) over the node's variable
    box. POUNCE is a true filter IPM on the *convex* relaxation (the same
    convexity assumption the JAX QP IPM made), so:

    * ``OPTIMAL`` -> KKT-valid relaxation optimum -> a valid node lower bound,
    * ``INFEASIBLE`` -> Phase-1-certified empty box -> a sound prune,
    * anything else -> untrusted; the caller keeps the node open (never a
      false-infeasible prune).

    Returns numpy arrays ``(clean, infeasible, obj_vals, x_vals)`` indexed by
    node: ``clean[i]`` marks a trustworthy OPTIMAL solve, ``infeasible[i]`` a
    certified empty box, ``obj_vals[i]`` the augmented objective (without
    ``obj_const``), and ``x_vals[i]`` the full primal iterate (with slacks).

    Node waves are solved in one ``pounce.solve_qp_batch`` call (one parallel
    Rayon wave across nodes) over POUNCE's *structured convex* QP form, which —
    like ``_solve_node_lp_pounce`` for MILP — presolves/scales the native
    ``min ½x'Qx + c'x s.t. A_eq x = b_eq, A_ub x <= b_ub, lb <= x <= ub`` and
    converges in ~20 IPM iterations, versus the callback TNLP path's ~100
    (the slacks are reconstructed, not solved for). Falls back to the serial
    callback path if ``solve_qp_batch`` is unavailable or the wave raises.
    """
    n_batch = len(batch_lb)
    Q = np.asarray(qp_data.Q, dtype=np.float64)
    c = np.asarray(qp_data.c, dtype=np.float64)
    A_eq = np.asarray(qp_data.A_eq, dtype=np.float64)
    b_eq = np.asarray(qp_data.b_eq, dtype=np.float64)
    n_total = int(qp_data.x_l.shape[0])
    n_slack = n_total - n_orig

    clean = np.zeros(n_batch, dtype=bool)
    infeasible = np.zeros(n_batch, dtype=bool)
    obj_vals = np.full(n_batch, np.nan, dtype=np.float64)
    x_vals = np.full((n_batch, n_total), np.nan, dtype=np.float64)

    # --- Batched structured-QP path (pounce-solver >= 0.5) ---
    solve_qp_batch = None
    try:
        import pounce

        solve_qp_batch = getattr(pounce, "solve_qp_batch", None)
    except ImportError:
        solve_qp_batch = None

    if solve_qp_batch is not None:
        try:
            from discopt.solvers.lp_pounce import _snap_inverted_bounds

            # The structural inequality/equality blocks are shared across the
            # wave; only the variable box (lb/ub) varies per node. The
            # decomposition returns ``None`` for an absent block (no pure
            # equalities or no inequalities), which POUNCE's ``solve_qp``
            # accepts directly.
            A_ub_m, b_ub_m, A_eq_m, b_eq_m = _decompose_eq_slack_form(A_eq, b_eq, n_orig, n_slack)
            P_s = Q[:n_orig, :n_orig]
            c_s = c[:n_orig]
            A_struct = A_eq[:, :n_orig]
            # Exact slack reconstruction: the slack columns ``S`` are shared, so
            # given a structural ``x_s`` the original equality-slack iterate has
            # slacks ``z = S^+ (b_eq - A_struct x_s)``. Then
            # ``A_eq_full [x_s, z] = b_eq`` exactly (rhs lies in range(S) for a
            # feasible slack form), so the caller's n_total feasibility check
            # and snapping see a byte-faithful full iterate.
            S_pinv = np.linalg.pinv(A_eq[:, n_orig:]) if n_slack > 0 else None

            problems = []
            for i in range(n_batch):
                lb_n = np.asarray(batch_lb[i], dtype=np.float64)
                ub_n = np.asarray(batch_ub[i], dtype=np.float64)
                lb_n, ub_n = _snap_inverted_bounds(lb_n, ub_n)
                problems.append(
                    {
                        "P": P_s,
                        "c": c_s,
                        "A": A_eq_m,
                        "b": b_eq_m,
                        "G": A_ub_m,
                        "h": b_ub_m,
                        "lb": lb_n,
                        "ub": ub_n,
                    }
                )

            # Q is shared and convex by assumption (same as the JAX QP IPM);
            # skip the per-problem O(n^3) PSD check.
            results = solve_qp_batch(problems, check_psd=False)

            for i in range(n_batch):
                res = results[i]
                if res.status == "optimal" and res.x is not None and np.isfinite(res.obj):
                    x_s = np.asarray(res.x, dtype=np.float64)
                    if x_s.shape[0] == n_orig and np.all(np.isfinite(x_s)):
                        if n_slack > 0:
                            z = S_pinv @ (b_eq - A_struct @ x_s)
                            x_full = np.concatenate([x_s, z])
                        else:
                            x_full = x_s
                        clean[i] = True
                        obj_vals[i] = float(res.obj)
                        x_vals[i] = x_full
                elif res.status == "primal_infeasible":
                    infeasible[i] = True
            return clean, infeasible, obj_vals, x_vals
        except Exception as e:  # noqa: BLE001 - any wave failure -> serial fallback
            logger.debug("Batched POUNCE QP wave failed (%s); serial fallback", e)
            clean[:] = False
            infeasible[:] = False
            obj_vals[:] = np.nan
            x_vals[:] = np.nan

    # --- Serial callback fallback (older wheels / batch failure) ---
    from discopt.solvers.qp_pounce import solve_qp as _pounce_qp

    slack_lb = np.zeros(n_slack, dtype=np.float64)
    slack_ub = np.full(n_slack, 1e20, dtype=np.float64)

    for i in range(n_batch):
        time_left = max(0.5, time_limit - (time.perf_counter() - t_start))
        lb_full = np.concatenate([np.asarray(batch_lb[i], dtype=np.float64), slack_lb])
        ub_full = np.concatenate([np.asarray(batch_ub[i], dtype=np.float64), slack_ub])
        bounds = list(zip(lb_full.tolist(), ub_full.tolist()))
        try:
            res = _pounce_qp(
                Q=Q,
                c=c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                time_limit=min(30.0, time_left),
            )
        except Exception as e:  # noqa: BLE001 - a crashed node solve is "untrusted"
            logger.debug("POUNCE node QP solve failed: %s", e)
            continue
        if (
            res.status == SolveStatus.OPTIMAL
            and res.x is not None
            and res.objective is not None
            and np.isfinite(res.objective)
        ):
            x_full = np.asarray(res.x, dtype=np.float64)
            if x_full.shape[0] == n_total and np.all(np.isfinite(x_full)):
                clean[i] = True
                obj_vals[i] = float(res.objective)
                x_vals[i] = x_full
        elif res.status == SolveStatus.INFEASIBLE:
            infeasible[i] = True
    return clean, infeasible, obj_vals, x_vals


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
        # An LP optimum must respect the node's variable box; an off-bound point
        # can be integral (e.g. a binary at -1) and otherwise pass the row checks,
        # seeding a spurious integer incumbent in the tree.
        if np.any(x_sol < lb_n - tol) or np.any(x_sol > ub_n + tol):
            logger.debug("POUNCE convex node solution violates variable bounds; rejecting")
            return None
        if A_ub_m is not None and b_ub_m is not None and A_ub_m.shape[0]:
            feasible = bool(np.all(A_ub_m @ x_sol <= b_ub_m + tol * (1.0 + np.abs(b_ub_m))))
        if feasible and A_eq_m is not None and b_eq_m is not None and A_eq_m.shape[0]:
            feasible = bool(np.all(np.abs(A_eq_m @ x_sol - b_eq_m) <= tol * (1.0 + np.abs(b_eq_m))))
        if not feasible:
            logger.debug("POUNCE convex node solution violates its rows; rejecting")
            return None
        bound = float(res.obj) + float(lp_data.obj_const)
        return (bound, x_sol[:n_vars], "optimal")
    if res.status == "primal_infeasible":
        return (_INFEASIBILITY_SENTINEL, None, "infeasible")
    return None


def _solve_node_lp_simplex(lp_data, node_lb, node_ub, n_vars, n_orig, t_start, time_limit):
    """Solve one MILP-B&B node relaxation with the pure-Rust warm-started
    simplex (exact vertex) — the default node engine for pure MILP.

    An interior-point method is the wrong tool for the *linear* node LPs in MILP
    B&B: it cannot cheaply warm-start across a branch, returns the analytic
    centre (so integer coordinates come back smeared and need purification), and
    yields no basis. The simplex reaches an exact basic-feasible vertex, so its
    objective is a rigorous node lower bound (no trust-gate decertification),
    integer coordinates are exact, and node throughput is far higher (issue: the
    POUNCE-IPM MILP node path was ~300 nodes/s vs the simplex's tens of
    thousands). Differentiability is unaffected — it is a post-solve IFT layer
    over (x*, λ*), independent of which engine produced them.

    Same contract as :func:`_solve_node_lp_pounce`: returns
    ``(lower_bound, solution_over_n_vars, "optimal")``,
    ``(sentinel, None, "infeasible")``, or ``None`` when the simplex backend is
    unavailable or could not settle the node (caller falls back / decertifies).
    """
    try:
        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE, solve_lp
    except ImportError:
        return None
    if not SIMPLEX_AVAILABLE:
        return None
    # Decompose the slack-expanded standard form back to native A_ub/A_eq over
    # the structural columns (the simplex adapter's matrix form), matching
    # _solve_node_lp_pounce so node bounds apply to the structural columns.
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
        bounds = list(zip(lb_n.tolist(), ub_n.tolist()))
        res = solve_lp(
            np.asarray(lp_data.c[:n_orig], dtype=np.float64),
            A_ub=A_ub_m,
            b_ub=b_ub_m,
            A_eq=A_eq_m,
            b_eq=b_eq_m,
            bounds=bounds,
        )
    except Exception as e:
        logger.debug("simplex node solve failed: %s", e)
        return None
    if res.status == SolveStatus.OPTIMAL and res.x is not None and np.isfinite(res.objective):
        x_sol = np.asarray(res.x, dtype=np.float64)
        # Soundness gate (mirror _solve_node_lp_pounce): reject a point that
        # violates its own rows so a bad relaxation cannot seed a spurious bound.
        tol = 1e-5
        feasible = True
        # An LP optimum must respect the node's variable bounds. The simplex
        # adapter can occasionally return a basic point that violates the box it
        # was given on mixed equality/inequality nodes; such a point is integral
        # off-bound (e.g. a binary at -1) and would otherwise pass the row checks
        # below and be accepted by the tree as a spurious integer incumbent.
        if np.any(x_sol < lb_n - tol) or np.any(x_sol > ub_n + tol):
            logger.debug("simplex node solution violates variable bounds; rejecting")
            return None
        if A_ub_m is not None and b_ub_m is not None and A_ub_m.shape[0]:
            feasible = bool(np.all(A_ub_m @ x_sol <= b_ub_m + tol * (1.0 + np.abs(b_ub_m))))
        if feasible and A_eq_m is not None and b_eq_m is not None and A_eq_m.shape[0]:
            feasible = bool(np.all(np.abs(A_eq_m @ x_sol - b_eq_m) <= tol * (1.0 + np.abs(b_eq_m))))
        if not feasible:
            logger.debug("simplex node solution violates its rows; rejecting")
            return None
        bound = float(res.objective) + float(lp_data.obj_const)
        return (bound, x_sol[:n_vars], "optimal")
    if res.status == SolveStatus.INFEASIBLE:
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
    integral, so the cuts are sound.

    Upper bounds are passed through so the separator can apply upper-bound
    complementation (Marchand-Wolsey bound substitution) on columns whose vertex
    value sits near the upper bound; non-finite upper bounds disable
    complementation for that column and fall back to the lower-shift. The columns
    here are the model's *original* declared variables, so there are no
    product-aux (lifted ``w = x*y``) columns to mark integer at this call site;
    that lift lives in the McCormick relaxation layer (see cmir_cuts.py) and is
    tracked separately."""
    if a_ub_orig is None or np.asarray(a_ub_orig).shape[0] == 0:
        return None
    try:
        from discopt._rust import mir_cuts_py
    except ImportError:
        return None
    lo = np.asarray(lp_data.x_l, dtype=np.float64)[:n_orig]
    if not np.all(np.isfinite(lo)):
        return None  # MIR's lower-bound shift requires finite lower bounds
    # Upper bounds drive per-column upper-bound complementation; a non-finite
    # ub[j] (or one missing) simply disables complementation for that column
    # (the Rust separator treats +inf as "no upper bound"). This is sound: with
    # no complementation the column falls back to the lower-shift substitution.
    hi = np.asarray(lp_data.x_u, dtype=np.float64)[:n_orig].copy()
    hi[~np.isfinite(hi)] = np.inf
    integ = np.zeros(n_orig, dtype=bool)
    integ[[j for j in int_idx if j < n_orig]] = True
    res = mir_cuts_py(
        np.ascontiguousarray(a_ub_orig, dtype=np.float64),
        np.ascontiguousarray(b_ub_orig, dtype=np.float64),
        np.ascontiguousarray(lo),
        np.ascontiguousarray(hi),
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


def _separate_aggregation_mir_cuts(
    lp_data, x_vertex, n_orig, int_idx, a_ub_orig, b_ub_orig, max_cuts: int = 8
):
    """Separate Marchand-Wolsey aggregation c-MIR cuts from the original ``<=``
    rows at the crossover vertex, via the Rust ``aggregation_mir_cuts_py`` binding.

    Pairs ``<=`` rows with nonnegative weights to cancel a continuous variable,
    forms the valid implied aggregate row, and applies the same complemented MIR
    (bound substitution + delta-scan) as :func:`_separate_mir_cuts` to it. A
    nonnegative combination of ``<=`` rows is a valid ``<=`` inequality, and MIR
    on it is valid for the integer hull, so every emitted cut is valid for the
    original feasible set — no integer-feasible point is ever removed (proven by
    the Rust ``aggregation_validity_random_systems`` property test).

    **Default-off**: this is the ``DISCOPT_CMIR_AGGREGATION`` feature-flagged
    path; the caller gates the call, this helper only does the separation. Same
    contract as :func:`_separate_mir_cuts`: returns ``(coeffs, rhs)`` embedded
    into the current standard-form columns, or ``None`` when the binding is
    unavailable, lower bounds are non-finite, or no cut is produced."""
    if a_ub_orig is None or np.asarray(a_ub_orig).shape[0] < 2:
        return None  # aggregation needs at least two rows to combine
    try:
        from discopt._rust import aggregation_mir_cuts_py
    except ImportError:
        return None
    lo = np.asarray(lp_data.x_l, dtype=np.float64)[:n_orig]
    if not np.all(np.isfinite(lo)):
        return None  # the MIR lower-bound shift requires finite lower bounds
    hi = np.asarray(lp_data.x_u, dtype=np.float64)[:n_orig].copy()
    hi[~np.isfinite(hi)] = np.inf
    integ = np.zeros(n_orig, dtype=bool)
    integ[[j for j in int_idx if j < n_orig]] = True
    res = aggregation_mir_cuts_py(
        np.ascontiguousarray(a_ub_orig, dtype=np.float64),
        np.ascontiguousarray(b_ub_orig, dtype=np.float64),
        np.ascontiguousarray(lo),
        np.ascontiguousarray(hi),
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

    The seed always goes through POUNCE so the cut-augmented shape costs no JAX
    recompile (consistent with the Path-B node engine); the JAX-IPM seed was
    retired in #370. The returned point is just a separation seed — crossover and
    cut validity do not depend on which engine produced it — so if POUNCE is
    unavailable the cut loop is simply skipped (``None``). ``prefer_pounce`` is
    kept for call-site symmetry."""
    del prefer_pounce
    try:
        from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
        from discopt.solvers.lp_pounce import solve_lp as _pounce_solve
    except ImportError:
        return None
    if not POUNCE_AVAILABLE:
        return None
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
    augmented) ``lp_data``, the total number of cuts added, and a per-source
    count dict (``cover_clique``/``gomory``/``mir``/``aggregation``) for
    instrumentation. A no-op when there are no binary-knapsack rows, clique
    edges, or integer variables."""
    from discopt._jax.cover_cuts import (
        has_binary_knapsack_rows,
        separate_clique_cuts,
        separate_cover_cuts,
    )

    has_cover = A_ub_orig is not None and has_binary_knapsack_rows(A_ub_orig, b_ub_orig, is_binary)
    has_clique = bool(clique_edges)
    has_gomory = bool(len(int_idx))
    if not has_cover and not has_clique and not has_gomory:
        return lp_data, 0, {"cover_clique": 0, "gomory": 0, "mir": 0, "aggregation": 0}

    total = 0
    # Per-source cut counts (cert:P3.1b instrumentation). Surfaced on the MILP
    # SolveResult's ``solver_stats`` so ON/OFF measurements can confirm the
    # aggregation c-MIR separator actually *fired* on the default path (a cut
    # count of 0 with the flag on means the branch never separated — a wiring or
    # scoping finding, not a bound result). Pure instrumentation; no math change.
    by_source = {"cover_clique": 0, "gomory": 0, "mir": 0, "aggregation": 0}
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
                by_source["gomory"] += int(gc.shape[0])
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
                by_source["mir"] += int(mc.shape[0])
        # Aggregation c-MIR (cert:P3): DEFAULT-OFF, gated by
        # DISCOPT_CMIR_AGGREGATION. Combines pairs of <= rows to cancel a
        # continuous variable, then applies the same complemented MIR as above —
        # valid by construction (nonnegative row combo + valid MIR). Same round-0
        # / POUNCE-mode gate as single-row MIR; reuses the MIR augmentation.
        if has_gomory and _round == 0 and _cmir_aggregation_enabled():
            try:
                agg = _separate_aggregation_mir_cuts(
                    lp_data, x_vertex, n_orig, int_idx, A_ub_orig, b_ub_orig
                )
            except Exception as _agg_exc:
                logger.debug("aggregation c-MIR separation skipped: %s", _agg_exc)
                agg = None
            if agg is not None:
                ac, ar = agg
                lp_data = _augment_lpdata_with_mir_cuts(lp_data, ac, ar)
                round_added += int(ac.shape[0])
                by_source["aggregation"] += int(ac.shape[0])
        if cuts:  # cover/clique reference original columns (< n_orig), still valid
            lp_data = _augment_lpdata_with_cover_cuts(lp_data, n_orig, cuts)
            round_added += len(cuts)
            by_source["cover_clique"] += len(cuts)

        if round_added == 0:
            break
        total += round_added
        if total >= max_total_cuts:
            break
        # GMI is round-0 only; with no cover/clique to re-separate, further
        # rounds would just re-solve the LP for nothing.
        if not has_cover and not has_clique:
            break
    return lp_data, total, by_source


def _root_dive(
    lp_data,
    n_orig,
    int_idx,
    t_start,
    time_limit,
    max_steps=None,
    prefer_pounce=False,
    n_vars=None,
    lb=None,
    ub=None,
    node_engine="pounce",
):
    """Fractional diving from the root LP to find an early incumbent (Phase 3).

    Repeatedly solve the LP relaxation, fix the most-fractional unfixed integer
    variable to its nearest integer, and re-solve, until every integer is
    integral (an incumbent) or a fix makes the LP non-optimal (dive abandoned).
    Returns ``(objective, x_orig)`` in minimization sense, or ``None``. An early
    incumbent front-loads pruning and reduced-cost fixing.

    On the self-hosted path (``prefer_pounce`` and/or ``node_engine="simplex"``)
    the dive LPs are solved with the structured node engine
    (``_solve_node_lp_simplex`` — exact vertex — or ``_solve_node_lp_pounce``)
    on the cut-augmented standard form, fixing one integer per re-solve and
    *re-solving the continuous variables each step*. This continuous repair is
    why the dive finds incumbents where plain rounding fails: on weak-relaxation
    (big-M) models the relaxation's integer coordinates are fractional and a
    rounded point violates the constraints, so without the dive the search runs
    to the time limit with **no incumbent** at all (no bound-based pruning →
    tree explosion). A few dozen fast solves at the root is cheap next to that.
    Requires ``n_vars``/``lb``/``ub`` (the structural node bounds).
    """
    if not int_idx:
        return None

    steps = max_steps if max_steps is not None else len(int_idx) + 1

    if prefer_pounce or node_engine == "simplex":
        if n_vars is None or lb is None or ub is None:
            return None
        node_solve = _solve_node_lp_simplex if node_engine == "simplex" else _solve_node_lp_pounce
        xl = np.asarray(lb, dtype=np.float64).copy()
        xu = np.asarray(ub, dtype=np.float64).copy()
        for _ in range(steps):
            if time.perf_counter() - t_start >= time_limit:
                return None
            out = node_solve(lp_data, xl, xu, n_vars, n_orig, t_start, time_limit)
            if out is None or out[2] != "optimal" or out[1] is None:
                return None  # infeasible/stalled fix -> abandon the dive
            obj, x, _ = out
            x = np.asarray(x, dtype=np.float64)
            fracs = [
                (j, abs(x[j] - round(x[j])))
                for j in int_idx
                if xl[j] != xu[j] and abs(x[j] - round(x[j])) > 1e-6
            ]
            if not fracs:
                # ``obj`` already includes ``lp_data.obj_const`` (added by the
                # node solver); ``x`` spans the structural columns.
                return float(obj), x
            j = max(fracs, key=lambda t: t[1])[0]
            v = float(round(x[j]))
            xl[j] = v
            xu[j] = v
        return None

    # No structured node engine selected: the root dive (an optional
    # early-incumbent heuristic) is skipped. The JAX-IPM dive was retired (#370);
    # on the default path this branch is unreachable — node_engine is always
    # "simplex" (or prefer_pounce is set), so the structured dive above runs.
    return None


# Wall-clock cap (seconds) for the fast Rust simplex MILP engine before it defers
# to the robust fallback. Bounds time wasted on a stalled reformulation regardless
# of the overall time_limit (issue #291).
_SIMPLEX_MILP_BUDGET_CAP_S = 10.0


def _solve_milp_simplex(
    model: Model,
    time_limit: float,
    gap_tolerance: float,
    max_nodes: int,
    t_start: float,
) -> Optional[SolveResult]:
    """Solve a pure MILP with the Rust-internal warm-started-simplex B&B
    (``nlp_solver="simplex"`` and the POUNCE-only default MILP path).

    The whole search runs in Rust: the existing tree manager with each node's LP
    solved by the bounded simplex (root cold, children dual-warm-started from the
    inherited basis), with a continuous-repair root dive for an early incumbent.
    Returns ``None`` to defer to the default path when the binding is unavailable,
    the model has no constraints, or the returned point fails the feasibility
    gate. Pure-MILP only; MINLP/MIQP keep the POUNCE/IPM path."""
    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.modeling.core import ObjectiveSense

    try:
        from discopt._rust import solve_milp_py
    except ImportError:
        return None

    # ``extract_lp_data`` captures only the LINEAR part of the model; any nonlinear
    # term is silently dropped. Solving that linear projection as if exact is sound
    # ONLY for a genuinely linear model. Otherwise a dropped *bounding* nonlinear
    # constraint can make the projection falsely unbounded/optimal — carton7
    # (issue #286): continuous variables with infinite upper bounds, bounded only
    # by the dropped nonlinear constraints, were reported as a false global
    # ``unbounded`` at the root. Defer any model carrying nonlinear terms to the
    # spatial / NLP path (which keeps those constraints).
    from discopt._jax.term_classifier import classify_nonlinear_terms

    _nl = classify_nonlinear_terms(model)
    if (
        _nl.bilinear
        or _nl.trilinear
        or _nl.multilinear
        or _nl.monomial
        or _nl.fractional_power
        or _nl.bilinear_with_fp
        or _nl.ratio_of_products
        or _nl.general_nl
    ):
        return None

    # Authoritative linearity backstop (defense-in-depth). The term classifier
    # above can have blind spots: a power/product over a *non-variable* base —
    # e.g. fac2's ``(x36+…+x41)**2.5`` objective — is missed by both the Rust and
    # Python term classifiers, yet ``extract_lp_data`` still silently drops it,
    # so the engine would certify a wrong 'optimal' on the linear projection
    # (off by 134x on fac2). The degree analysis behind ``is_objective_linear`` /
    # ``is_constraint_linear`` (the same check the router trusts to reach this
    # MILP branch) catches every such term, so defer unless the model is provably
    # linear in its objective and every constraint. For models that reach here
    # through the normal MILP route this is a no-op (the router already proved
    # linearity); it only guards direct calls, future re-routing, and any
    # router/extractor representation discrepancy.
    try:
        from discopt._rust import model_to_repr

        _repr = model_to_repr(model, getattr(model, "_builder", None))
        _fully_linear = bool(_repr.is_objective_linear()) and all(
            _repr.is_constraint_linear(i) for i in range(_repr.n_constraints)
        )
    except Exception:
        _fully_linear = False
    if not _fully_linear:
        return None

    lp_data = extract_lp_data(model)
    A = np.ascontiguousarray(lp_data.A_eq, dtype=np.float64)
    if A.shape[0] == 0:
        return None  # no constraints — let the default path handle it
    n_orig = sum(v.size for v in model._variables)
    _, _, _, int_offsets, int_sizes = _extract_variable_info(model)
    int_idx = [j for off, sz in zip(int_offsets, int_sizes) for j in range(off, off + int(sz))]

    # Pass the remaining wall-clock budget so the Rust B&B's deadline (loop-top,
    # per-node and per-LP) fires. Without it ``time_limit_s`` defaults to 0.0 -> no
    # deadline, so a single node whose simplex fails to converge (or a degenerate
    # cycle) runs unbounded, ignoring the user's ``time_limit`` entirely (issue
    # #291: nvs12's integer-bilinear reformulation hung > 40 s on a 15 s limit).
    # Bound the fast simplex engine to a modest slice of the remaining budget, so a
    # stall on a pathological reformulation (issue #291) defers quickly (status
    # node_limit -> None below) and leaves the rest for the robust spatial/POUNCE
    # fallback. The absolute cap matters: with the default time_limit=3600 a plain
    # fraction would let a stalled solve burn ~1800 s before falling back. The
    # engine is the *fast* path — a well-behaved reformulation (ex126x) solves in
    # ~1 s, far inside this cap — so capping costs nothing on the common case while
    # bounding the wasted time before fallback to a few seconds.
    _remaining = float(time_limit) - (time.perf_counter() - t_start)
    _milp_budget = max(0.5, min(0.5 * _remaining, _SIMPLEX_MILP_BUDGET_CAP_S))
    # Interactive debugger: install the Rust checkpoint hook only when a debugger
    # is attached now (bound-neutral otherwise). This is the pure-Rust MILP
    # fast-path — the hook fires the same node-lifecycle checkpoints as the
    # Python-driven loops, across the PyO3 boundary.
    from discopt import debug as _debug

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
        time_limit_s=float(_milp_budget),
        debug_hook=_debug.rust_hook(),
    )
    wall_time = time.perf_counter() - t_start
    maximize = model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE

    def _debug_stopped_result() -> Optional[SolveResult]:
        """Partial, uncertified result when the interactive debugger's `quit`
        stopped the Rust search. Returned at the defer sites below instead of
        ``None`` so the caller does NOT fall back to another engine — the user
        asked the solve to stop, not to restart elsewhere. ``None`` (the normal
        case) preserves the plain defer-to-fallback behavior."""
        _sess = _debug.current()
        if _sess is None or not _sess.stop_requested:
            return None
        _bv = None
        if np.isfinite(bound):
            _bv = -bound if maximize else bound
        return SolveResult(
            status="unknown",
            bound=_bv,
            wall_time=wall_time,
            node_count=nodes,
            gap_certified=False,
        )

    if status in ("optimal", "feasible"):
        x_arr = np.asarray(x_struct, dtype=np.float64)
        xo = x_arr[:n_orig]
        # Feasibility gate: never take an engine's "optimal"/"feasible" on faith.
        # The Rust whole-search can return an infeasible point — e.g. all-zeros
        # on a zero-objective feasibility MILP, where the simplex can terminate
        # at the (infeasible) starting point instead of driving phase-1 to
        # feasibility. Verify the returned point against the model's own rows,
        # bounds, and integrality; on violation defer (return None) so the
        # caller falls back to a sound engine rather than returning a wrong
        # "optimal".
        n_slack = int(lp_data.A_eq.shape[1]) - n_orig
        _A_ub_m, _b_ub_m, _A_eq_m, _b_eq_m = _decompose_eq_slack_form(
            np.asarray(lp_data.A_eq), np.asarray(lp_data.b_eq), n_orig, n_slack
        )
        _tol = 1e-5
        _feas = True
        if _A_ub_m is not None and _b_ub_m is not None and _A_ub_m.shape[0]:
            _feas = bool(np.all(_A_ub_m @ xo <= _b_ub_m + _tol * (1.0 + np.abs(_b_ub_m))))
        if _feas and _A_eq_m is not None and _b_eq_m is not None and _A_eq_m.shape[0]:
            _feas = bool(np.all(np.abs(_A_eq_m @ xo - _b_eq_m) <= _tol * (1.0 + np.abs(_b_eq_m))))
        if _feas:
            _xl = np.asarray(lp_data.x_l[:n_orig], dtype=np.float64)
            _xu = np.asarray(lp_data.x_u[:n_orig], dtype=np.float64)
            _feas = bool(np.all(xo >= _xl - _tol) and np.all(xo <= _xu + _tol))
        if _feas:
            for _off, _sz in zip(int_offsets, int_sizes):
                seg = xo[_off : _off + int(_sz)]
                if np.any(np.abs(seg - np.round(seg)) > 1e-4):
                    _feas = False
                    break
        if not _feas:
            logger.warning(
                "Rust simplex MILP returned an infeasible point labeled %s; "
                "deferring to a sound engine",
                status,
            )
            return _debug_stopped_result()
        x_dict = _unpack_solution(model, x_arr)
        obj_val = -obj if maximize else obj
        bound_val = None
        gap_val = None
        if np.isfinite(bound):
            bound_val = -bound if maximize else bound
            gap_val = abs(obj_val - bound_val) / (abs(obj_val) + 1e-10)

        # Root-node certification metrics (cert:T0.1/T0.5). The one-shot Rust MILP
        # driver runs the whole B&B internally, so there is no per-iteration root
        # snapshot to read. Recover the root bound by solving the continuous
        # relaxation (integers relaxed) over the root box — one extra LP solve,
        # cheap on this fast path. Pure instrumentation: the returned incumbent /
        # node_count are untouched, so the solve stays bound-neutral.
        root_bound_val = None
        root_gap_val = None
        root_time_val = None
        try:
            _t_root = time.perf_counter()
            _lp_status, _, _lp_obj, _lp_bound, _, _ = solve_milp_py(
                np.ascontiguousarray(lp_data.c, dtype=np.float64),
                A,
                np.ascontiguousarray(lp_data.b_eq, dtype=np.float64),
                np.ascontiguousarray(lp_data.x_l, dtype=np.float64),
                np.ascontiguousarray(lp_data.x_u, dtype=np.float64),
                np.ascontiguousarray(np.empty(0, dtype=np.int64)),  # integers relaxed
                n_orig,
                float(lp_data.obj_const),
                1000,  # a relaxed (integer-free) LP solves at the root; headroom to finalize
                float(gap_tolerance),
                time_limit_s=float(max(0.1, min(_milp_budget, 5.0))),
            )
            root_time_val = time.perf_counter() - _t_root
            if (
                _lp_status in ("optimal", "feasible")
                and _lp_obj is not None
                and np.isfinite(_lp_obj)
            ):
                root_bound_val = -_lp_obj if maximize else _lp_obj
                if obj_val is not None and np.isfinite(obj_val):
                    root_gap_val = abs(obj_val - root_bound_val) / max(1.0, abs(obj_val))
        except Exception as _e:  # pragma: no cover - defensive
            logger.debug("root LP relaxation bound failed: %s", _e)

        return SolveResult(
            status=status,
            objective=obj_val,
            bound=bound_val,
            gap=gap_val,
            x=x_dict,
            wall_time=wall_time,
            node_count=nodes,
            root_bound=root_bound_val,
            root_gap=root_gap_val,
            root_time=root_time_val,
            gap_certified=status == "optimal",
        )
    if status == "unbounded":
        return SolveResult(status="unbounded", wall_time=wall_time, node_count=nodes)
    if status == "node_limit":
        # The simplex MILP engine exhausted its node/time budget without proving
        # optimality and found no usable incumbent here. Rather than surface a
        # bare ``node_limit`` (no solution), defer (return None) so the caller falls
        # back to the robust spatial / POUNCE path — which, e.g., solves nvs12's
        # integer-bilinear reformulation in ~0.4 s where this engine stalls
        # (issue #291). A genuine incumbent is returned via the optimal/feasible
        # branch above, so deferral only discards a no-solution result. On a
        # debugger `quit` this instead surfaces the uncertified partial state
        # (no fallback engine resumes a solve the user stopped).
        return _debug_stopped_result()
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
    node_engine: str = "pounce",
    lagrangian_bound: bool = False,
    lagrangian_frequency: int = 1,
) -> SolveResult:
    """Solve a MILP via B&B with LP relaxation solves at each node.

    ``prefer_pounce`` (POUNCE-only mode) routes the incumbent dual recovery
    through POUNCE instead of the Rust simplex; either way the solve is
    HiGHS-free (HiGHS was removed, issue #356).

    ``node_engine`` selects the per-node LP relaxation engine in POUNCE-only
    mode: ``"simplex"`` (the default for pure MILP — exact-vertex warm-started
    simplex, far faster than the IPM on linear nodes and free of interior
    smearing / trust-gate decertification) or ``"pounce"`` (the POUNCE IPM,
    used as a fallback when the simplex binding is unavailable). The
    auxiliary root cut-loop / dual recovery stay on POUNCE either way; only the
    hot per-node solve changes.
    """

    from discopt._jax.problem_classifier import extract_lp_data

    rust_time = 0.0
    jax_time = 0.0

    t_jax_start = time.perf_counter()
    lp_data = extract_lp_data(model)
    jax_time += time.perf_counter() - t_jax_start

    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)
    n_orig = sum(v.size for v in model._variables)

    # Resolve the per-node LP engine. ``node_engine="simplex"`` (the pure-MILP
    # default) uses the exact-vertex warm-started simplex; it degrades to the
    # POUNCE IPM node path if the simplex binding is unavailable. The chosen
    # engine drives the root LP-bound seed, the root dive, and the node loop.
    _simplex_nodes = False
    if node_engine == "simplex":
        try:
            from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE as _SXA

            _simplex_nodes = bool(_SXA)
        except ImportError:
            _simplex_nodes = False
    _node_solve = _solve_node_lp_simplex if _simplex_nodes else _solve_node_lp_pounce
    _node_engine_resolved = "simplex" if _simplex_nodes else "pounce"
    # The self-hosted structured node path runs when POUNCE-only mode is on
    # (either engine) — i.e. whenever we are not on the legacy JAX-IPM path.
    _use_structured_nodes = prefer_pounce or _simplex_nodes

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
    _cut_by_source = {"cover_clique": 0, "gomory": 0, "mir": 0, "aggregation": 0}
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
        lp_data, _n_cuts, _cut_by_source = _root_cover_cut_loop(
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
            logger.info(
                "root cuts added %d valid inequalities "
                "(cover/clique=%d gomory=%d mir=%d aggregation=%d)",
                _n_cuts,
                _cut_by_source.get("cover_clique", 0),
                _cut_by_source.get("gomory", 0),
                _cut_by_source.get("mir", 0),
                _cut_by_source.get("aggregation", 0),
            )
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
    if _use_structured_nodes:
        try:
            _root_out = _node_solve(lp_data, lb, ub, n_vars, n_orig, t_start, time_limit)
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

    # Opt-in Lagrangian node-bound hook: dualize coupling constraints and combine
    # a valid per-node dual lower bound with the LP relaxation bound (max() never
    # weakens a valid bound). Applies only to linear, minimization models with
    # coupling structure; otherwise it cleanly no-ops. Multipliers are fixed once
    # at the root from the root box.
    _lag_bounder = None
    _lag_freq = max(1, int(lagrangian_frequency))
    if lagrangian_bound:
        try:
            from discopt.decomposition.lagrangian.node_bounder import LagrangianNodeBounder

            _lag_bounder = LagrangianNodeBounder.try_build(model, prefer_pounce=prefer_pounce)
            if _lag_bounder is not None:
                _lag_bounder.solve_root_dual(lb, ub)
            else:
                logger.info(
                    "lagrangian_bound: model is not a linear minimization with coupling "
                    "structure; node-bound hook disabled."
                )
        except Exception as _lag_exc:
            logger.debug("lagrangian_bound setup failed: %s", _lag_exc)
            _lag_bounder = None
    # Root fractional diving (Phase 3): an early incumbent front-loads pruning
    # and reduced-cost fixing. The tree keeps the best of any injected points.
    try:
        _int_idx = [j for off, sz in zip(int_offsets, int_sizes) for j in range(off, off + int(sz))]
        _dive = _root_dive(
            lp_data,
            n_orig,
            _int_idx,
            t_start,
            time_limit,
            prefer_pounce=prefer_pounce,
            n_vars=n_vars,
            lb=lb,
            ub=ub,
            node_engine=_node_engine_resolved,
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

    # Path B: in POUNCE-only mode the structured engine solves node relaxations
    # directly (no JAX recompile on cut-augmented shapes): the exact-vertex
    # simplex by default, or the POUNCE IPM. Checked once here. POUNCE
    # availability is still tracked because the dual recovery path uses it.
    _pounce_nodes_avail = False
    try:
        from discopt.solvers.lp_pounce import POUNCE_AVAILABLE as _PNA

        _pounce_nodes_avail = bool(_PNA)
    except ImportError:
        _pounce_nodes_avail = False
    # The structured engine (Rust simplex, or POUNCE as fallback) is the only node
    # LP engine — the JAX LP-IPM node path was retired (#370). POUNCE now counts as
    # a structured fallback regardless of prefer_pounce, so a no-Rust install still
    # solves nodes with an exact/KKT-valid engine instead of the retired IPM.
    _structured_avail = _simplex_nodes or _pounce_nodes_avail

    iteration = 0
    # Root-node certification instrumentation (cert:T0.1).
    _root_time: Optional[float] = None
    _root_glb_internal: Optional[float] = None
    from discopt import debug as _debug

    # Set when the interactive debugger's `quit` breaks the search loop: a
    # user-interrupted exit proves nothing, so the status decision below must
    # not fall through to a certified "infeasible"/"optimal".
    _debug_quit = False
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Interactive debugger: top-of-iteration checkpoint (no-op when detached).
        if _debug.fire(
            _debug.Checkpoint.ITER_START,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=elapsed,
        ):
            _debug_quit = True
            break

        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, _batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        # Interactive debugger: nodes selected — boxes/ids now available.
        if _debug.fire(
            _debug.Checkpoint.AFTER_SELECT,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=elapsed,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
        ):
            _debug_quit = True
            break

        t_jax_start = time.perf_counter()
        result_ids = np.array(batch_ids, dtype=np.int64)

        if _structured_avail:
            # Path B: solve each node's relaxation with the structured engine
            # (exact-vertex simplex by default, else POUNCE IPM) instead of the
            # JAX IPM. It takes the cut-augmented LP at any shape with no
            # per-shape recompile, and its OPTIMAL bound is exact/KKT-valid. The
            # rare stall/unavailable node defers to the existing recovery path.
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.zeros(n_batch, dtype=bool)
            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])
                _mid = 0.5 * (np.clip(node_lb, -_SPC, _SPC) + np.clip(node_ub, -_SPC, _SPC))
                out = _node_solve(lp_data, node_lb, node_ub, n_vars, n_orig, t_start, time_limit)
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
        else:
            # The JAX LP-IPM node path was retired (#370). Node LP relaxations
            # use the structured engine (Rust simplex, or POUNCE); reaching here
            # means neither is available — an unsupported configuration.
            raise RuntimeError(
                "No structured node LP engine available (Rust simplex or POUNCE). "
                "The JAX LP-IPM node fallback was retired in #370; build the Rust "
                "extension or install POUNCE."
            )

        # Lagrangian node bounds: combine a valid dual lower bound with each
        # node's LP relaxation bound. Gated by cadence and applied only to nodes
        # that produced a real (non-sentinel) bound; max() keeps soundness.
        if _lag_bounder is not None and (iteration % _lag_freq == 0):
            for i in range(n_batch):
                if result_lbs[i] < _SENTINEL_THRESHOLD:
                    _lag_lb = _lag_bounder.node_bound(
                        np.asarray(batch_lb[i], dtype=np.float64),
                        np.asarray(batch_ub[i], dtype=np.float64),
                    )
                    if _lag_lb is not None and np.isfinite(_lag_lb):
                        result_lbs[i] = max(result_lbs[i], _lag_lb)

        jax_time += time.perf_counter() - t_jax_start

        # Interactive debugger: steer point — relaxations solved, not imported.
        if _debug.fire(
            _debug.Checkpoint.BEFORE_IMPORT,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=time.perf_counter() - t_start,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
            result_lbs=result_lbs,
            result_sols=result_sols,
            result_feas=result_feas,
        ):
            _debug_quit = True
            break

        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        # Interactive debugger: prune/branch/fathom applied by the tree.
        if _debug.fire(
            _debug.Checkpoint.AFTER_PROCESS,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=time.perf_counter() - t_start,
        ):
            _debug_quit = True
            break

        # Root-node certification snapshot (cert:T0.1).
        if iteration == 0:
            _root_time = time.perf_counter() - t_start
            _root_glb_snap = tree.stats().get("global_lower_bound")
            if _root_glb_snap is not None and np.isfinite(_root_glb_snap):
                _root_glb_internal = float(_root_glb_snap)

        iteration += 1
        if tree.is_finished():
            break
        if _gap_converged(tree, gap_tolerance):
            break
        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time

    stats = tree.stats()
    incumbent = tree.incumbent()

    # R3a instrumentation (behavior-neutral): export per-variable branch
    # frequency when the diagnostic sink is armed (integer B&B drivers too).
    if _R3A_BRANCH_COUNT_SINK is not None:
        try:
            _R3A_BRANCH_COUNT_SINK["branch_var_counts"] = np.asarray(
                tree.branch_var_counts()
            ).tolist()
        except Exception as _e:  # pragma: no cover - diagnostics only
            logger.debug("R3a branch_var_counts capture failed: %s", _e)

    if incumbent is not None:
        sol_array, obj_val = incumbent
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    constraint_duals = None
    bound_duals_lower = None
    bound_duals_upper = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        # C-3: snap near-integral discrete coordinates to exact integers. In the
        # MILP path each integer was branch-fixed to [k, k] before the node LP
        # solved, so a stored k±ε is a numeric artifact and rounding to k
        # restores exactly the point the LP was solving — it cannot move a
        # linear row by more than the integrality tol. The dual-recovery
        # re-solve does not round the reported primal, so round here.
        _rounded_inc, _rounded_feas = _round_incumbent_integers(sol_flat, int_offsets, int_sizes)
        if _rounded_feas:
            sol_flat = _rounded_inc
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
        if (_gap_converged(tree, gap_tolerance) or tree.is_finished()) and _gap_certified:
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
        elif _debug_quit:
            # Interactive debugger `quit`: the user interrupted the search, so
            # the tree is NOT exhausted and neither infeasibility nor optimality
            # was proven (a false "infeasible" here is the worst-class error).
            # Report "unknown", uncertified.
            status = "unknown"
            _gap_certified = False
        else:
            # Tree exhausted with no feasible node: infeasibility *is* a certified
            # conclusion, so leave _gap_certified untouched.
            status = "infeasible"

    # Interactive debugger: terminal checkpoint. Fired after the status
    # decision so ``debug="on-error"`` sessions can key on the outcome:
    # ``error`` is the non-"optimal" status ("time_limit", "unknown",
    # "infeasible", ...), or None on a certified optimum.
    _debug.fire(
        _debug.Checkpoint.TERMINATED,
        tree=tree,
        model=model,
        iteration=iteration,
        elapsed=wall_time,
        error=None if status == "optimal" else status,
    )

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

    # Root-node certification metrics (cert:T0.1). Prefer the tree's global
    # lower bound snapshot at the end of the root batch; fall back to the root
    # LP relaxation bound. Both are in the internal minimization sense.
    _root_internal = _root_glb_internal
    if _root_internal is None and np.isfinite(_root_lp_bound):
        _root_internal = float(_root_lp_bound)
    root_bound_val: Optional[float] = None
    root_gap_val: Optional[float] = None
    if _root_internal is not None and np.isfinite(_root_internal):
        root_bound_val = -_root_internal if _maximize else _root_internal
        if obj_val is not None and np.isfinite(obj_val):
            root_gap_val = abs(obj_val - root_bound_val) / max(1.0, abs(obj_val))

    # Root cut counts by source (cert:P3.1b instrumentation): lets an ON/OFF
    # measurement confirm the aggregation c-MIR separator actually fired on this
    # default MILP path (a zero count with the flag on is a wiring/scoping
    # finding). Only non-zero sources are surfaced.
    _milp_solver_stats = {
        f"cuts/{_src}": float(_cnt) for _src, _cnt in _cut_by_source.items() if _cnt > 0
    }

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
        root_bound=root_bound_val,
        root_gap=root_gap_val,
        root_time=_root_time,
        constraint_duals=constraint_duals,
        bound_duals_lower=bound_duals_lower,
        bound_duals_upper=bound_duals_upper,
        gap_certified=_gap_certified,
        solver_stats=_milp_solver_stats or None,
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
    through POUNCE instead of the Rust simplex; either way the solve is
    HiGHS-free (HiGHS was removed, issue #356).
    """
    from discopt._jax.problem_classifier import extract_qp_data

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
    # Root-node certification instrumentation (cert:T0.1).
    _root_time: Optional[float] = None
    _root_glb_internal: Optional[float] = None
    from discopt import debug as _debug

    # Set when the interactive debugger's `quit` breaks the search loop: a
    # user-interrupted exit proves nothing, so the status decision below must
    # not fall through to a certified "infeasible"/"optimal".
    _debug_quit = False
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Interactive debugger: top-of-iteration checkpoint (no-op when detached).
        if _debug.fire(
            _debug.Checkpoint.ITER_START,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=elapsed,
        ):
            _debug_quit = True
            break

        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, _batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        # Interactive debugger: nodes selected — boxes/ids now available.
        if _debug.fire(
            _debug.Checkpoint.AFTER_SELECT,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=elapsed,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
        ):
            _debug_quit = True
            break

        t_jax_start = time.perf_counter()
        result_ids = np.array(batch_ids, dtype=np.int64)
        # Solve every node's convex QP relaxation with POUNCE (JAX-free). The
        # batched JAX QP IPM was the last JAX dependency on the MIQP path; POUNCE
        # gives the same KKT-valid bound / Phase-1 infeasibility verdict per node.
        clean, infeasible, obj_vals, x_vals = _pounce_qp_relaxation_nodes(
            qp_data, batch_lb, batch_ub, n_orig, t_start, time_limit
        )
        result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
        result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
        result_feas = np.zeros(n_batch, dtype=bool)

        for i in range(n_batch):
            node_lb = np.array(batch_lb[i])
            node_ub = np.array(batch_ub[i])
            lb_c = np.clip(node_lb, -_SPC, _SPC)
            ub_c = np.clip(node_ub, -_SPC, _SPC)

            if infeasible[i]:
                # POUNCE Phase-1-certified empty box: a sound infeasibility prune.
                result_sols[i] = 0.5 * (lb_c + ub_c)
            elif clean[i] and _check_lp_solution_feasibility(qp_data.A_eq, qp_data.b_eq, x_vals[i]):
                # KKT-valid relaxation optimum -> a valid node lower bound.
                result_lbs[i] = obj_vals[i] + float(qp_data.obj_const)
                result_sols[i] = x_vals[i, :n_vars]
                if result_lbs[i] < _SENTINEL_THRESHOLD:
                    _maybe_inject_snapped(result_sols[i], node_lb, node_ub)
            else:
                # OPTIMAL-but-inconsistent or non-clean (iteration limit / crash):
                # an untrusted bound. Keep the node open (POUNCE recovery), never a
                # false-infeasible prune (issue #127).
                x_seed = x_vals[i] if np.all(np.isfinite(x_vals[i])) else None
                obj_seed = (
                    obj_vals[i] + float(qp_data.obj_const) if np.isfinite(obj_vals[i]) else np.nan
                )
                _handle_nonclean(i, result_lbs, result_sols, x_seed, obj_seed, node_lb, node_ub)

        jax_time += time.perf_counter() - t_jax_start

        # Interactive debugger: steer point — relaxations solved, not imported.
        if _debug.fire(
            _debug.Checkpoint.BEFORE_IMPORT,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=time.perf_counter() - t_start,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
            result_lbs=result_lbs,
            result_sols=result_sols,
            result_feas=result_feas,
        ):
            _debug_quit = True
            break

        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        # Interactive debugger: prune/branch/fathom applied by the tree.
        if _debug.fire(
            _debug.Checkpoint.AFTER_PROCESS,
            tree=tree,
            model=model,
            iteration=iteration,
            elapsed=time.perf_counter() - t_start,
        ):
            _debug_quit = True
            break

        # Root-node certification snapshot (cert:T0.1).
        if iteration == 0:
            _root_time = time.perf_counter() - t_start
            _root_glb_snap = tree.stats().get("global_lower_bound")
            if _root_glb_snap is not None and np.isfinite(_root_glb_snap):
                _root_glb_internal = float(_root_glb_snap)

        iteration += 1
        if tree.is_finished():
            break
        if _gap_converged(tree, gap_tolerance):
            break
        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time

    stats = tree.stats()
    incumbent = tree.incumbent()

    # R3a instrumentation (behavior-neutral): export per-variable branch
    # frequency when the diagnostic sink is armed (integer B&B drivers too).
    if _R3A_BRANCH_COUNT_SINK is not None:
        try:
            _R3A_BRANCH_COUNT_SINK["branch_var_counts"] = np.asarray(
                tree.branch_var_counts()
            ).tolist()
        except Exception as _e:  # pragma: no cover - diagnostics only
            logger.debug("R3a branch_var_counts capture failed: %s", _e)

    if incumbent is not None:
        sol_array, obj_val = incumbent
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    constraint_duals = None
    bound_duals_lower = None
    bound_duals_upper = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        # C-3: snap near-integral discrete coordinates to exact integers (see the
        # MILP-BB path). Integers are branch-fixed before each node QP solve, so
        # rounding a stored k±ε restores the fixed point; the dual-recovery
        # re-solve does not round the reported primal.
        _rounded_inc, _rounded_feas = _round_incumbent_integers(sol_flat, int_offsets, int_sizes)
        if _rounded_feas:
            sol_flat = _rounded_inc
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
        if (_gap_converged(tree, gap_tolerance) or tree.is_finished()) and _gap_certified:
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
        elif _debug_quit:
            # Interactive debugger `quit`: the user interrupted the search, so
            # the tree is NOT exhausted and neither infeasibility nor optimality
            # was proven (a false "infeasible" here is the worst-class error).
            # Report "unknown", uncertified.
            status = "unknown"
            _gap_certified = False
        else:
            # Tree exhausted with no feasible node: infeasibility *is* a certified
            # conclusion, so leave _gap_certified untouched.
            status = "infeasible"

    # Interactive debugger: terminal checkpoint. Fired after the status
    # decision so ``debug="on-error"`` sessions can key on the outcome:
    # ``error`` is the non-"optimal" status ("time_limit", "unknown",
    # "infeasible", ...), or None on a certified optimum.
    _debug.fire(
        _debug.Checkpoint.TERMINATED,
        tree=tree,
        model=model,
        iteration=iteration,
        elapsed=wall_time,
        error=None if status == "optimal" else status,
    )

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
    # (max_nodes=1 false-certification regression). But the untainted tree bound is
    # itself a valid global dual bound (every node floored at its valid parent
    # bound), so keep it and recompute the gap rather than dropping to None (#138).
    _tree_bound_valid = _gap_certified
    if status == "feasible":
        _gap_certified = False

    if not _gap_certified:
        gap_val = None
        if _tree_bound_valid and bound_val is not None and np.isfinite(bound_val):
            if obj_val is not None and np.isfinite(obj_val):
                gap_val = abs(obj_val - bound_val) / max(1.0, abs(obj_val))
        else:
            bound_val = None

    # Re-earn certification when the retained valid tree bound meets the incumbent
    # within tolerance: the incumbent is then provably global and the honest status
    # is "optimal" (mirrors the spatial path's re-certification).
    if (
        status == "feasible"
        and obj_val is not None
        and bound_val is not None
        and np.isfinite(bound_val)
        and gap_val is not None
        and gap_val <= gap_tolerance
    ):
        _is_max = model._objective.sense == ObjectiveSense.MAXIMIZE
        if (bound_val >= obj_val - 1e-9) if _is_max else (bound_val <= obj_val + 1e-9):
            _gap_certified = True
            status = "optimal"

    # Root-node certification metrics (cert:T0.1).
    _miqp_is_max = model._objective.sense == ObjectiveSense.MAXIMIZE
    root_bound_val: Optional[float] = None
    root_gap_val: Optional[float] = None
    if _root_glb_internal is not None and np.isfinite(_root_glb_internal):
        root_bound_val = -_root_glb_internal if _miqp_is_max else _root_glb_internal
        if obj_val is not None and np.isfinite(obj_val):
            root_gap_val = abs(obj_val - root_bound_val) / max(1.0, abs(obj_val))

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
        root_bound=root_bound_val,
        root_gap=root_gap_val,
        root_time=_root_time,
        constraint_duals=constraint_duals,
        bound_duals_lower=bound_duals_lower,
        bound_duals_upper=bound_duals_upper,
        gap_certified=_gap_certified,
    )
