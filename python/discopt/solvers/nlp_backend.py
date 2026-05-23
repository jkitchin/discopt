"""Backend dispatch for continuous NLP solves.

A single seam used by ``solver.py`` and the primal heuristics so that no
call site needs to know which NLP backend is in use. The chosen backend
must expose ``solve_nlp(evaluator, x0, constraint_bounds=None, options=None)``
returning :class:`discopt.solvers.NLPResult`.
"""

from __future__ import annotations

from typing import Callable, Literal

Backend = Literal["auto", "pounce", "cyipopt"]


def _try_pounce() -> Callable | None:
    try:
        from discopt.solvers import nlp_pounce
    except ImportError:
        return None
    if not getattr(nlp_pounce, "POUNCE_AVAILABLE", False):
        return None
    return nlp_pounce.solve_nlp


def _try_cyipopt() -> Callable | None:
    try:
        import cyipopt  # noqa: F401
    except ImportError:
        return None
    from discopt.solvers import nlp_ipopt

    return nlp_ipopt.solve_nlp


def get_nlp_solver(backend: Backend = "auto") -> Callable:
    """Return a ``solve_nlp(evaluator, x0, ...)`` callable for ``backend``.

    ``"auto"`` resolves to pounce if importable, otherwise cyipopt.
    Raises :class:`ImportError` if no backend is available.
    """
    if backend == "pounce":
        fn = _try_pounce()
        if fn is None:
            raise ImportError("pounce backend requested but pounce is not importable.")
        return fn

    if backend == "cyipopt":
        fn = _try_cyipopt()
        if fn is None:
            raise ImportError("cyipopt backend requested but cyipopt is not importable.")
        return fn

    if backend != "auto":
        raise ValueError(f"Unknown NLP backend: {backend!r}")

    fn = _try_pounce() or _try_cyipopt()
    if fn is None:
        raise ImportError("No NLP backend available. Install pounce or cyipopt.")
    return fn


def available_backends() -> list[str]:
    """List names of NLP backends that import successfully."""
    names: list[str] = []
    if _try_pounce() is not None:
        names.append("pounce")
    if _try_cyipopt() is not None:
        names.append("cyipopt")
    return names
