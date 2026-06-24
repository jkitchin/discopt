"""MIP-NLP decomposition solver-family facade."""

from __future__ import annotations

from typing import Any, Optional

from discopt.modeling.core import Model, SolveResult

SUPPORTED_METHODS = {"oa", "ecp", "goa", "roa", "fp", "lp_nlp_bb"}
_METHOD_ALIASES = {
    "lp-nlp-bb": "lp_nlp_bb",
    "lp/nlp-bb": "lp_nlp_bb",
    "lp_nlp_bb": "lp_nlp_bb",
}
_OA_OPTION_KEYS = {"equality_relaxation", "ecp_mode", "feasibility_cuts"}


def _normalize_method(method: str) -> str:
    normalized = _METHOD_ALIASES.get(method, method)
    if normalized not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown mip_nlp_method={method!r}. Choose one of {sorted(SUPPORTED_METHODS)}."
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
    if mip_nlp_options:
        options.update(mip_nlp_options)
    options.update(kwargs)

    if method in {"oa", "ecp"}:
        unexpected = sorted(set(options) - _OA_OPTION_KEYS)
        if unexpected:
            raise ValueError(
                "Unsupported MIP-NLP OA/ECP option(s): "
                + ", ".join(unexpected)
                + ". Supported options are: "
                + ", ".join(sorted(_OA_OPTION_KEYS))
            )

        from discopt.solvers.oa import solve_oa

        if method == "ecp":
            options["ecp_mode"] = True
        else:
            options.setdefault("ecp_mode", False)

        return solve_oa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_iterations,
            nlp_solver=nlp_solver,
            **options,
        )

    raise NotImplementedError(
        f"mip_nlp_method={method!r} is reserved for the MIP-NLP decomposition "
        "family but is not implemented yet."
    )
