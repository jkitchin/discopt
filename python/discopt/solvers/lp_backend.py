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


def available_lp_backends() -> list[str]:
    names = []
    if _lp_highs() is not None:
        names.append("highs")
    if _lp_pounce() is not None:
        names.append("pounce")
    return names
