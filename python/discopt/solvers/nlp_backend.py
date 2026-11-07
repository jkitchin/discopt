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
    from discopt._openmp_guard import check_openmp_compatible

    # Raises (rather than returning None) on a macOS OpenMP-runtime clash: the
    # package is installed, and "auto" must not silently pretend it is not (#2205).
    check_openmp_compatible("cyipopt", feature="the cyipopt NLP backend")
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
    """List names of NLP backends that import successfully.

    cyipopt is left out when importing it would abort this process on a second
    OpenMP runtime (macOS, #2205): it cannot be imported *here*. Asking for it by
    name through :func:`get_nlp_solver` raises the explanation instead.
    """
    from discopt._openmp_guard import OpenMPRuntimeConflictError

    names: list[str] = []
    if _try_pounce() is not None:
        names.append("pounce")
    try:
        cyipopt_fn = _try_cyipopt()
    except OpenMPRuntimeConflictError:
        cyipopt_fn = None
    if cyipopt_fn is not None:
        names.append("cyipopt")
    return names
