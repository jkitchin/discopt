"""Backend selection for matrix-form LP/QP/MILP solves.

The engines are signature- and ``LPResult``/``QPResult``/``MILPResult``-compatible
(``lp_highs`` vs ``lp_pounce``, ``qp_pounce`` (QP is POUNCE-only since #359), and
the pure-Rust warm-started simplex), so consumers (OBBT, relaxation solvers,
OA/GDP/Benders masters, ...) can pick one through this seam and stay agnostic.
This is what lets discopt run with **only POUNCE installed** (no HiGHS): the
selector falls back to whichever backend is importable.

The matrix-LP and matrix-MILP defaults route to the self-hosted Rust simplex
first (issue #356) — HiGHS-free, exact, and fast on the ill-conditioned lifted
relaxations — with HiGHS then POUNCE as fallbacks. The Rust LP simplex now
surfaces ``dual_values``/``reduced_costs`` in HiGHS's convention, so the
dual-consuming seams (Benders subproblem, DBBT) run on it too. ``prefer_pounce``
flips the preference to POUNCE-first (the POUNCE-only mode, ``nlp_solver="pounce"``).

Exception — the **QP** seam (:func:`get_qp_solver`) is POUNCE-only (issue #359):
the QP path has no HiGHS backend at all (``qp_highs`` was removed). There is no QP
fallback — POUNCE is the QP engine.
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


def _qp_pounce() -> Callable | None:
    try:
        from discopt.solvers.qp_pounce import POUNCE_AVAILABLE, solve_qp
    except ImportError:
        return None
    return solve_qp if POUNCE_AVAILABLE else None


def get_lp_solver(prefer_pounce: bool = False) -> Callable:
    """Return a matrix-form ``solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, ...)``.

    Default order is the self-hosted Rust simplex -> HiGHS -> POUNCE (issue #356):
    the simplex reaches the exact vertex and now exposes ``dual_values`` /
    ``reduced_costs`` in HiGHS's convention, so the dual-consuming seams (Benders
    subproblem, ...) run HiGHS-free. ``prefer_pounce`` keeps the POUNCE-first
    order (the POUNCE-only mode). Raises :class:`ImportError` only when no backend
    is importable.

    Note the simplex returns no warm-start basis (``LPResult.basis is None``);
    callers that warm-start across a cutting-plane loop simply cold-start each
    round — correct, only a speed difference.
    """
    order = (_lp_pounce, _lp_highs) if prefer_pounce else (_lp_simplex, _lp_highs, _lp_pounce)
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
    requires an exact (vertex) oracle that *exposes* its duals. discopt's
    pure-Rust simplex now surfaces ``reduced_costs`` (and ``dual_values``) from
    the optimal basis in HiGHS's convention — exact vertex duals that satisfy
    strong duality (validated against HiGHS) — so it is preferred here, with HiGHS
    as the fallback (issue #356). The POUNCE IPM is never used: its analytic-center
    duals are not rigorous (issue #145). Returns ``None`` only when neither exact
    oracle is importable, and DBBT then soundly no-ops.
    """
    return _lp_simplex() or _lp_highs()


def get_qp_solver(prefer_pounce: bool = False) -> Callable:
    """Return a matrix-form ``solve_qp(Q, c, A_ub, ...)``; see
    :func:`get_lp_solver`. POUNCE handles continuous QPs only (MIQPs go
    through the self-hosted B&B).

    POUNCE-only and HiGHS-free (issue #359): the LP/MILP routing went HiGHS-free
    in #356 and the QP path now has no HiGHS backend (``qp_highs`` was removed),
    so there is no fallback. ``prefer_pounce`` is retained for symmetry with the
    other selectors but has no effect. QP duals are reported, not consumed for
    bound tightening, so this carries none of the LP-dual soundness hazards #356
    guarded against; the top-level ``Model.solve`` QP path adds a
    primal-feasibility + KKT-residual guard."""
    del prefer_pounce  # POUNCE is the only QP backend; kept for selector symmetry
    solver = _qp_pounce()
    if solver is not None:
        return solver
    raise ImportError("No QP backend available. Install POUNCE:\n  pip install pounce-solver")


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


def _milp_gurobi() -> Callable | None:
    try:
        from discopt.solvers.gurobi import solve_milp

        return solve_milp
    except ImportError:
        return None


def get_milp_solver(prefer_pounce: bool = False, backend: str = "auto") -> Callable:
    """Return a matrix-form ``solve_milp(c, A_ub, ..., integrality, ...)``.

    ``backend`` selects the preferred engine: ``"auto"`` (**simplex-first** — the
    pure-Rust warm-started-simplex B&B — then HiGHS, then POUNCE; or POUNCE-first
    under ``prefer_pounce``), ``"highs"``, ``"pounce"``, ``"simplex"``, or
    ``"gurobi"``. The preferred engine is tried first and the call falls back to
    the standard order if it is unavailable, so selection never fails when *any*
    backend is importable. An explicit Gurobi selection returns the optional
    wrapper; a missing ``gurobipy`` installation or license is reported when the
    wrapper is called. Raises :class:`ImportError` only when none is available.

    Routing the default to the self-hosted Rust simplex (issue #356, part B) makes
    the matrix-MILP path HiGHS-free without an external dependency: the simplex
    reaches the exact B&B optimum (a rigorous bound, unlike the POUNCE IPM's
    analytic-center objective that can drift on ill-conditioned LPs — #145) and is
    fast on the lifted, ill-conditioned relaxations where the POUNCE IPM is slow.
    HiGHS/POUNCE remain as fallbacks. ``prefer_pounce`` (the POUNCE-only mode)
    keeps its POUNCE-first order unchanged.
    """
    valid = {"auto", "highs", "pounce", "simplex", "gurobi"}
    if backend not in valid:
        raise ValueError(f"Unknown MILP backend {backend!r}; choose from {sorted(valid)}.")
    base = (
        (_milp_pounce, _milp_highs) if prefer_pounce else (_milp_simplex, _milp_highs, _milp_pounce)
    )
    if backend == "simplex":
        order: tuple[Callable[[], Callable | None], ...] = (_milp_simplex, *base)
    elif backend == "gurobi":
        order = (_milp_gurobi, *base)
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
        "  pip install highspy         (HiGHS)\n"
        "  pip install gurobipy        (Gurobi, requires a working license)"
    )


def available_lp_backends() -> list[str]:
    names = []
    if _lp_highs() is not None:
        names.append("highs")
    if _lp_pounce() is not None:
        names.append("pounce")
    return names
