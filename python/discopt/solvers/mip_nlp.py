"""MIP-NLP decomposition solver-family facade."""

from __future__ import annotations

from typing import Any, Optional

from discopt.modeling.core import Model, SolveResult

_IMPLEMENTED_METHODS = frozenset({"oa", "ecp"})
_RESERVED_METHOD_ISSUES = {
    "fp": "#115",
    "roa": "#116/#117",
    "goa": "#118",
    "lp_nlp_bb": "#119",
}
SUPPORTED_METHODS = _IMPLEMENTED_METHODS | frozenset(_RESERVED_METHOD_ISSUES)
_METHOD_ALIASES = {
    "lp/nlp-bb": "lp_nlp_bb",
}
_OA_OPTION_KEYS = {"equality_relaxation", "ecp_mode", "feasibility_cuts"}


def _normalize_method(method: Any) -> str:
    if not isinstance(method, str):
        raise ValueError(f"mip_nlp_method must be a string, got {type(method).__name__}.")
    raw = method.strip().lower()
    normalized = _METHOD_ALIASES.get(raw, raw.replace("-", "_"))
    if normalized not in SUPPORTED_METHODS:
        reserved = ", ".join(sorted(_RESERVED_METHOD_ISSUES))
        raise ValueError(
            f"Unknown mip_nlp_method={method!r}. Choose 'oa' or 'ecp'. "
            f"Reserved future methods are: {reserved}."
        )
    return normalized


def solve_mip_nlp(
    model: Model,
    *,
    method: str = "oa",
    mip_nlp_options: Optional[dict[str, Any]] = None,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "pounce",
    **kwargs: Any,
) -> SolveResult:
    """Solve a MINLP with a MIP-NLP decomposition method."""
    method = _normalize_method(method)
    options: dict[str, Any] = {}
    if mip_nlp_options is not None:
        if not isinstance(mip_nlp_options, dict):
            raise TypeError(
                "mip_nlp_options must be a dict of OA/ECP solver options, "
                f"got {type(mip_nlp_options).__name__}."
            )
        options.update(mip_nlp_options)
    options.update(kwargs)

    if method in _IMPLEMENTED_METHODS:
        unexpected = sorted(set(options) - _OA_OPTION_KEYS)
        if unexpected:
            raise ValueError(
                "Unsupported MIP-NLP OA/ECP option(s): "
                + ", ".join(unexpected)
                + ". Supported options are: "
                + ", ".join(sorted(_OA_OPTION_KEYS))
            )

        from discopt.solvers.oa import solve_oa

        if "ecp_mode" in kwargs:
            alias_method = "ecp" if bool(kwargs["ecp_mode"]) else "oa"
            if alias_method != method:
                raise ValueError(
                    "Conflicting MIP-NLP method selectors: "
                    f"mip_nlp_method={method!r} and ecp_mode={kwargs['ecp_mode']!r}."
                )
        options["ecp_mode"] = method == "ecp"

        return solve_oa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_iterations,
            nlp_solver=nlp_solver,
            **options,
        )

    raise NotImplementedError(
        f"mip_nlp_method={method!r} is reserved for a future MIP-NLP "
        f"implementation tracked in {_RESERVED_METHOD_ISSUES[method]}; currently "
        "implemented methods are 'oa' and 'ecp'."
    )
