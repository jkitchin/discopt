"""Backend selection for matrix-form LP/QP solves (HiGHS or POUNCE).

The two engines are signature- and ``LPResult``/``QPResult``-compatible
(``lp_highs``/``qp_highs`` vs ``lp_pounce``/``qp_pounce``), so consumers
(OBBT, relaxation solvers, ...) can pick one through this seam and stay
agnostic. This is what lets discopt run with **only POUNCE installed**
(no HiGHS): the selector falls back to whichever backend is importable.

``prefer_pounce`` flips the preference to POUNCE-first (the POUNCE-only mode,
``nlp_solver="pounce"``); otherwise HiGHS is preferred with POUNCE fallback.
"""

from __future__ import annotations

from typing import Callable


def _lp_highs() -> Callable | None:
    try:
        from discopt.solvers.lp_highs import solve_lp

        return solve_lp
    except ImportError:
        return None


def _lp_pounce() -> Callable | None:
    try:
        from discopt.solvers.lp_pounce import POUNCE_AVAILABLE, solve_lp
    except ImportError:
        return None
    return solve_lp if POUNCE_AVAILABLE else None


def _lp_simplex() -> Callable | None:
    # Pure-Rust warm-started simplex; available iff the binding is built.
    try:
        from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE, solve_lp
    except ImportError:
        return None
    return solve_lp if SIMPLEX_AVAILABLE else None


def _qp_highs() -> Callable | None:
    try:
        from discopt.solvers.qp_highs import solve_qp

        return solve_qp
    except ImportError:
        return None


def _qp_pounce() -> Callable | None:
    try:
        from discopt.solvers.qp_pounce import POUNCE_AVAILABLE, solve_qp
    except ImportError:
        return None
    return solve_qp if POUNCE_AVAILABLE else None


def get_lp_solver(prefer_pounce: bool = False) -> Callable:
    """Return a matrix-form ``solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, ...)``.

    Order is HiGHS -> POUNCE, flipped when ``prefer_pounce``. Raises
    :class:`ImportError` only when neither backend is importable.
    """
    order = (_lp_pounce, _lp_highs) if prefer_pounce else (_lp_highs, _lp_pounce)
    for factory in order:
        solver = factory()
        if solver is not None:
            return solver
    raise ImportError(
        "No LP backend available. Install one of:\n"
        "  pip install pounce-solver   (POUNCE)\n"
        "  pip install highspy         (HiGHS)"
    )


def get_exact_lp_solver() -> Callable | None:
    """Return an *exact* (simplex/vertex) LP oracle, or ``None`` if unavailable.

    Prefers discopt's **own** pure-Rust warm-started simplex, then HiGHS — never
    the POUNCE IPM. OBBT tightens a variable's bound to the optimum of
    ``min``/``max x_i`` over the relaxation polytope, which is sound **only when
    that LP is solved to its true optimum**. POUNCE's interior-point method
    returns the analytic center of the optimal face; its reported objective
    normally matches the simplex optimum but can be grossly wrong on
    ill-conditioned LPs (e.g. a 1e6 coefficient spread) while still reporting
    ``OPTIMAL`` — an over-tightening that cuts off feasible, even
    globally-optimal, points (issue #145). A simplex reaches the exact vertex,
    so its optimum is a rigorous bound; the self-hosted Rust simplex gives this
    without an external HiGHS dependency. Callers that need a *sound* bound
    (OBBT) must use this; when it returns ``None`` they must skip tightening
    rather than fall back to the IPM.
    """
    return _lp_simplex() or _lp_highs()


def get_exact_dual_lp_solver() -> Callable | None:
    """Return an exact LP oracle that also provides **reduced costs**, or ``None``.

    Duality-based bound tightening (DBBT) reads the LP's reduced costs to bound
    how far each variable can move from the bound it is pressed against. That
    requires an exact (vertex) oracle that *exposes* its duals: HiGHS does
    (``col_dual``); discopt's pure-Rust simplex reaches the exact vertex but does
    not expose reduced costs across the binding, and the POUNCE IPM's duals are
    not rigorous (issue #145). So this returns HiGHS when available, else
    ``None`` — DBBT then soundly no-ops rather than tighten from inexact duals.
    """
    return _lp_highs()


def get_qp_solver(prefer_pounce: bool = False) -> Callable:
    """Return a matrix-form ``solve_qp(Q, c, A_ub, ...)``; see
    :func:`get_lp_solver`. POUNCE handles continuous QPs only (MIQPs go
    through HiGHS or the B&B)."""
    order = (_qp_pounce, _qp_highs) if prefer_pounce else (_qp_highs, _qp_pounce)
    for factory in order:
        solver = factory()
        if solver is not None:
            return solver
    raise ImportError(
        "No QP backend available. Install one of:\n"
        "  pip install pounce-solver   (POUNCE)\n"
        "  pip install highspy         (HiGHS)"
    )


def _milp_highs() -> Callable | None:
    try:
        from discopt.solvers.milp_highs import solve_milp

        return solve_milp
    except ImportError:
        return None


def _milp_pounce() -> Callable | None:
    # POUNCE "matrix MILP" is the self-hosted B&B; available iff POUNCE is.
    try:
        from discopt.solvers.lp_pounce import POUNCE_AVAILABLE

        if not POUNCE_AVAILABLE:
            return None
        from discopt.solvers.milp_pounce import solve_milp

        return solve_milp
    except ImportError:
        return None


def _milp_simplex() -> Callable | None:
    # Pure-Rust warm-started-simplex B&B; available iff the binding is built.
    try:
        from discopt._rust import solve_milp_py  # noqa: F401
        from discopt.solvers.milp_simplex import solve_milp

        return solve_milp
    except ImportError:
        return None


def get_milp_solver(prefer_pounce: bool = False, backend: str = "auto") -> Callable:
    """Return a matrix-form ``solve_milp(c, A_ub, ..., integrality, ...)``.

    ``backend`` selects the preferred engine: ``"auto"`` (HiGHS-first, or
    POUNCE-first under ``prefer_pounce``), ``"highs"``, ``"pounce"``, or
    ``"simplex"`` (the pure-Rust warm-started-simplex B&B). The preferred engine
    is tried first and the call falls back to the standard order if it is
    unavailable, so selection never fails when *any* backend is importable.
    Raises :class:`ImportError` only when none is available.
    """
    valid = {"auto", "highs", "pounce", "simplex"}
    if backend not in valid:
        raise ValueError(f"Unknown MILP backend {backend!r}; choose from {sorted(valid)}.")
    base = (_milp_pounce, _milp_highs) if prefer_pounce else (_milp_highs, _milp_pounce)
    if backend == "simplex":
        order: tuple[Callable[[], Callable | None], ...] = (_milp_simplex, *base)
    elif backend == "highs":
        order = (_milp_highs, *base)
    elif backend == "pounce":
        order = (_milp_pounce, *base)
    else:
        order = base
    for factory in order:
        solver = factory()
        if solver is not None:
            return solver
    raise ImportError(
        "No MILP backend available. Install one of:\n"
        "  pip install pounce-solver   (POUNCE, via the self-hosted B&B)\n"
        "  pip install highspy         (HiGHS)"
    )


def available_lp_backends() -> list[str]:
    names = []
    if _lp_highs() is not None:
        names.append("highs")
    if _lp_pounce() is not None:
        names.append("pounce")
    return names
