"""MIP-NLP decomposition solver-family facade."""

from __future__ import annotations

from typing import Any, Optional

from discopt.modeling.core import Model, SolveResult

_IMPLEMENTED_METHODS = frozenset({"oa", "ecp", "fp"})
_RESERVED_METHOD_ISSUES = {
    "roa": "#116/#117",
    "goa": "#118",
    "lp_nlp_bb": "#119",
}
SUPPORTED_METHODS = _IMPLEMENTED_METHODS | frozenset(_RESERVED_METHOD_ISSUES)
_METHOD_ALIASES = {
    "lp/nlp-bb": "lp_nlp_bb",
}
_OA_OPTION_KEYS = {
    "equality_relaxation",
    "ecp_mode",
    "feasibility_cuts",
    "init_strategy",
    "heuristic_nonconvex",
    "add_slack",
    "max_slack",
    "oa_penalty_factor",
    "add_no_good_cuts",
    "feasibility_norm",
    "stalling_limit",
    "cycling_check",
}
_OA_OPTION_ALIASES = {
    "OA_penalty_factor": "oa_penalty_factor",
}
_FP_OPTION_KEYS = {
    "add_no_good_cuts",
    "feasibility_norm",
    "init_strategy",
}


def _normalize_method(method: Any) -> str:
    if not isinstance(method, str):
        raise ValueError(f"mip_nlp_method must be a string, got {type(method).__name__}.")
    raw = method.strip().lower()
    normalized = _METHOD_ALIASES.get(raw, raw.replace("-", "_"))
    if normalized not in SUPPORTED_METHODS:
        reserved = ", ".join(sorted(_RESERVED_METHOD_ISSUES))
        raise ValueError(
            f"Unknown mip_nlp_method={method!r}. Choose 'oa', 'ecp', or 'fp'. "
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
    initial_point=None,
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
    for alias, canonical in _OA_OPTION_ALIASES.items():
        if alias in options:
            if canonical not in options:
                options[canonical] = options[alias]
            del options[alias]

    if method in _IMPLEMENTED_METHODS:
        supported_keys = _FP_OPTION_KEYS if method == "fp" else _OA_OPTION_KEYS
        unexpected = sorted(set(options) - supported_keys)
        if unexpected:
            raise ValueError(
                f"Unsupported MIP-NLP {method} option(s): "
                + ", ".join(unexpected)
                + ". Supported options are: "
                + ", ".join(sorted(supported_keys))
            )

        if method == "fp":
            from discopt.solvers.oa import _normalize_init_strategy, solve_feasibility_pump

            if "init_strategy" in options:
                init_strategy = _normalize_init_strategy(options["init_strategy"])
                if init_strategy != "fp":
                    raise ValueError(
                        "mip_nlp_method='fp' only accepts init_strategy='fp' when "
                        "an initialization strategy is supplied."
                    )

            return solve_feasibility_pump(
                model,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                max_iterations=max_iterations,
                nlp_solver=nlp_solver,
                initial_point=initial_point,
                add_no_good_cuts=bool(options.get("add_no_good_cuts", True)),
                feasibility_norm=options.get("feasibility_norm", "L_infinity"),
            )

        from discopt.solvers.oa import _normalize_init_strategy, solve_oa

        if "ecp_mode" in kwargs:
            alias_method = "ecp" if bool(kwargs["ecp_mode"]) else "oa"
            if alias_method != method:
                raise ValueError(
                    "Conflicting MIP-NLP method selectors: "
                    f"mip_nlp_method={method!r} and ecp_mode={kwargs['ecp_mode']!r}."
                )
        options["ecp_mode"] = method == "ecp"
        options["init_strategy"] = _normalize_init_strategy(options.get("init_strategy", "rNLP"))

        return solve_oa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_iterations,
            nlp_solver=nlp_solver,
            initial_point=initial_point,
            **options,
        )

    raise NotImplementedError(
        f"mip_nlp_method={method!r} is reserved for a future MIP-NLP "
        f"implementation tracked in {_RESERVED_METHOD_ISSUES[method]}; currently "
        "implemented methods are 'oa' and 'ecp'."
    )
