"""#843: QUBO/Ising local-search primal — JAX-free.

An unconstrained binary quadratic model (``chimera_k64ising``: 1192 binary vars,
0 constraints, indefinite MAXIMIZE Ising) returns NO incumbent from the dense
B&B, and the #827 trivial seed only ever produced the useless all-zeros floor.
``qubo_local_search`` (greedy-1opt + tabu on the quadratic form) constructs a
real feasible incumbent that ``solve_model`` injects as ``initial_point``.

This module is deliberately **JAX-free** (numpy + the JAX-free
``problem_classifier`` gate and ``QPData`` extraction ladder): the seed fires by
default (#843 graduation), so both the structural gate *and* the search itself
must not pull the multi-second JAX cold start onto the pure LP/MILP/QP/MIQP
paths that ``test_lazy_jax_linear_path`` pins. The original #846 implementation
built the Hessian through the JAX ``NLPEvaluator``; this rewrite gets the
identical ``½xᵀHx + cᵀx`` internal minimize form from
``problem_classifier.extract_qp_data`` (already sense-negated for MAXIMIZE),
whose only JAX rung is an autodiff last resort behind the algebraic and
Rust-repr extractors.

Soundness: an unconstrained QUBO has no feasibility to violate — any binary
point is a valid incumbent (a MAXIMIZE incumbent can never exceed the optimum)
— and the injection path re-verifies integer + constraint feasibility before
seeding, so the dual bound / certificate are untouched.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from discopt.modeling.core import Model, VarType

logger = logging.getLogger(__name__)

# Poll the deadline every this many tabu iterations (a #863 lesson: every long
# loop polls its deadline). Cheap relative to the O(n) vector work per iteration,
# and bounds the overrun of a pass that started just before the deadline.
_DEADLINE_POLL_STRIDE = 256


def is_qubo(model: Model) -> bool:
    """True if *model* is an unconstrained binary quadratic program (QUBO).

    That is: every variable is binary-valued (BINARY, or INTEGER with a [0, 1]
    box), and there are no constraints — neither general Python constraints nor
    fast-path builder-resident linear rows (#840). This is the
    ``chimera_k64ising`` / Max-Cut structure (#843). Such a model has no
    feasibility to satisfy — *any* binary point is feasible — so a
    quadratic-form local search is a sound, purely-primal constructor.

    Pure Python + numpy — safe to call on the JAX-free LP/MILP cold-start path.
    The *quadratic degree* of the objective is NOT checked here (that requires
    walking the expression DAG); :func:`qubo_local_search` enforces it via the
    algebraic extractor and returns ``None`` for degree > 2.
    """
    if model._constraints or model._num_builder_constraint_rows():
        return False
    if model._objective is None:
        return False
    for v in model._variables:
        if v.var_type not in (VarType.BINARY, VarType.INTEGER):
            return False
        lo = np.asarray(v.lb, dtype=np.float64).ravel()
        hi = np.asarray(v.ub, dtype=np.float64).ravel()
        if not (np.all(lo == 0.0) and np.all(hi == 1.0)):
            return False
    return True


def qubo_local_search(
    model: Model,
    *,
    deadline: Optional[float] = None,
    max_starts: int = 12,
    iters_per_start: int = 4000,
    tenure: int = 20,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Greedy-1opt + tabu local search on a binary QUBO (#843).

    Minimizes the INTERNAL objective ``½xᵀHx + cᵀx`` over ``{0,1}ⁿ`` — exactly
    the internal minimize form the solver uses (``QPData`` is negated already
    for a MAXIMIZE), so the caller can inject the returned binary point as
    ``x0`` / incumbent and the reported objective sense is handled by the
    solver as usual.

    The Hessian ``H`` is constant (quadratic objective), so each 1-flip's
    objective delta is ``δ·g + ½·H_kk`` (δ = ±1) and the gradient updates
    incrementally (``g += δ·H[:,k]``) — O(nnz) per flip. Tabu (with an
    aspiration override on a new global best) escapes the shallow local optima
    plain greedy gets stuck in. Runs from zeros / ones / stratified-random
    starts and returns the best binary point, or ``None`` if the model is not a
    QUBO (the MIQP classification gate excludes a degree > 2 objective, where
    the constant-Hessian delta bookkeeping would be wrong, and a linear
    objective — the MILP path solves that exactly, JAX-free, without help), or
    no point improves on all-zeros.

    ``deadline`` is a ``time.perf_counter()`` instant, polled between starts
    and every :data:`_DEADLINE_POLL_STRIDE` iterations inside a pass.

    Purely primal and sound: an unconstrained QUBO has no feasibility to
    violate, so any binary point is a valid incumbent; the dual bound /
    certificate are untouched. JAX-free end to end.
    """
    if not is_qubo(model):
        return None
    # ``problem_classifier`` is JAX-free (a pinned invariant of the lazy-import
    # architecture). The MIQP gate is the *degree* check ``is_qubo`` cannot do
    # structurally: a linear objective classifies MILP (the JAX-free simplex B&B
    # solves it exactly without help — and must stay JAX-free), degree > 2
    # classifies MINLP (the constant-Hessian delta bookkeeping below would be
    # wrong there). ``extract_qp_data`` then returns min-form ``½xᵀQx + cᵀx + d``
    # (Q the symmetric Hessian, already negated for a MAXIMIZE) via its ladder —
    # builder-repr → algebraic DAG walk → Rust-repr probe — whose only JAX rung
    # is the autodiff last resort, reached only if both the algebraic walk and
    # the Rust evaluator fail. Q may be scipy-sparse on a wide model (#863) —
    # never ``np.asarray`` it; the search below handles both forms.
    from discopt._jax.problem_classifier import (
        ProblemClass,
        classify_problem,
        extract_qp_data,
    )

    try:
        if classify_problem(model) != ProblemClass.MIQP:
            return None
        qp = extract_qp_data(model)
    except RecursionError:
        # A pathologically deep expression chain (thousands of `expr + term`
        # re-bindings) overflows the recursive DAG walkers. Not seedable; the
        # solve proper falls back the same way.
        return None
    c = np.asarray(qp.c, dtype=np.float64)
    n = c.shape[0]
    if n == 0:
        return None

    import scipy.sparse as _sp

    H_sparse = None
    if _sp.issparse(qp.Q):
        H_sparse = _sp.csc_matrix(qp.Q, dtype=np.float64)
        if H_sparse.nnz == 0:
            return None  # no quadratic term survived extraction
        Hdiag = np.asarray(H_sparse.diagonal(), dtype=np.float64)
    else:
        H = np.asarray(qp.Q, dtype=np.float64)
        if H.shape != (n, n) or not np.any(H):
            return None  # malformed or no quadratic term — nothing to search on
        Hdiag = np.diag(H)

    def _col(k: int) -> np.ndarray:
        if H_sparse is not None:
            return np.asarray(H_sparse[:, [k]].toarray(), dtype=np.float64).ravel()
        return np.asarray(H[:, k], dtype=np.float64)

    def _matvec(x: np.ndarray) -> np.ndarray:
        if H_sparse is not None:
            return np.asarray(H_sparse @ x, dtype=np.float64).ravel()
        return np.asarray(H @ x, dtype=np.float64)

    def one_pass(x0: np.ndarray, iters: int) -> tuple[np.ndarray, float]:
        x = x0.astype(np.float64).copy()
        g = _matvec(x) + c
        cur = 0.5 * float(x @ _matvec(x)) + float(c @ x)
        best, bestx = cur, x.copy()
        tabu_until = np.full(n, -1, dtype=np.int64)
        for it in range(iters):
            if (
                deadline is not None
                and it % _DEADLINE_POLL_STRIDE == 0
                and it > 0
                and time.perf_counter() >= deadline
            ):
                break
            delta = 1.0 - 2.0 * x
            dobj = delta * g + 0.5 * Hdiag  # Δ internal objective per flip
            allowed = np.where(tabu_until <= it, dobj, np.inf)
            aspire = np.where(cur + dobj < best - 1e-9, dobj, np.inf)  # override tabu on new best
            pick = np.minimum(allowed, aspire)
            k = int(np.argmin(pick))
            if not np.isfinite(pick[k]):
                k = int(np.argmin(dobj))
            d = delta[k]
            x[k] = 1.0 - x[k]
            cur += dobj[k]
            g = g + d * _col(k)
            tabu_until[k] = it + tenure
            if cur < best - 1e-9:
                best, bestx = cur, x.copy()
        return bestx, best

    rng = np.random.default_rng(seed)
    z = np.zeros(n, dtype=np.float64)
    starts = [z, np.ones(n)]
    starts += [
        np.asarray(rng.random(n) > 0.5, dtype=np.float64) for _ in range(max(0, max_starts - 2))
    ]
    best_x: Optional[np.ndarray] = None
    best_internal = np.inf
    for i, x0 in enumerate(starts):
        # Always run the first (zeros) pass — even against a tight deadline the
        # in-pass poll guarantees at least one short greedy burst, and returning
        # a good incumbent matters more than never overrunning by ~256 cheap
        # iterations.
        if i > 0 and deadline is not None and time.perf_counter() >= deadline:
            break
        xb, ib = one_pass(x0, iters_per_start)
        if ib < best_internal - 1e-9:
            best_internal, best_x = ib, xb
    # Only return a point that beats the trivial all-zeros incumbent (internal 0.0,
    # measured relative to the objective constant, which cancels in comparisons).
    if best_x is None or best_internal >= -1e-9:
        return None
    return best_x
