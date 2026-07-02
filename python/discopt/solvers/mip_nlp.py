"""MIP-NLP decomposition solver-family facade."""

from __future__ import annotations

from typing import Any, Optional, cast

from discopt.modeling.core import Model, ObjectiveSense, SolveResult
from discopt.solvers.mip_nlp_options import (
    FP_OPTION_KEYS,
    GOA_OPTION_KEYS,
    MIPNLPShotConfig,
    ensure_mip_nlp_trace,
    split_mip_nlp_profile_options,
)

_IMPLEMENTED_METHODS = frozenset({"oa", "ecp", "fp", "goa", "lp_nlp_bb"})
_RESERVED_METHOD_ISSUES = {
    "roa": "#116/#117",
}
SUPPORTED_METHODS = _IMPLEMENTED_METHODS | frozenset(_RESERVED_METHOD_ISSUES)
_METHOD_ALIASES = {
    "lp/nlp-bb": "lp_nlp_bb",
}
_OA_OPTION_KEYS = frozenset(
    {
        "equality_relaxation",
        "ecp_mode",
        "feasibility_cuts",
        "init_strategy",
        "heuristic_nonconvex",
        "add_slack",
        "max_slack",
        "oa_penalty_factor",
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
        *FP_OPTION_KEYS,
    }
)
_OA_OPTION_ALIASES = {
    "OA_penalty_factor": "oa_penalty_factor",
}
_FP_OPTION_KEYS = frozenset(
    {
        "add_no_good_cuts",
        "feasibility_norm",
        "init_strategy",
        *FP_OPTION_KEYS,
    }
)
_LP_NLP_BB_OPTION_KEYS = frozenset(
    {
        "equality_relaxation",
        "feasibility_cuts",
        "init_strategy",
        "heuristic_nonconvex",
        "add_slack",
        "max_slack",
        "oa_penalty_factor",
        "add_no_good_cuts",
        "integer_to_binary",
        "feasibility_norm",
        "milp_solver",
    }
)


def _apply_shot_profile_reformulations(
    model: Model,
    shot_config: MIPNLPShotConfig | None,
) -> Model:
    """Apply SHOT-profile reformulations that already have proven discopt passes."""
    if shot_config is None:
        return model
    objective = getattr(model, "_objective", None)
    if objective is None:
        return model
    if objective.sense == ObjectiveSense.MINIMIZE:
        if shot_config.objective_epigraph == "off":
            return model
    elif shot_config.anti_epigraph == "off":
        return model

    from discopt._jax.objective_epigraph import relax_objective_defining_equality

    reformulated, _changed = relax_objective_defining_equality(model)
    return cast(Model, reformulated)


def _normalize_method(method: Any) -> str:
    if not isinstance(method, str):
        raise ValueError(f"mip_nlp_method must be a string, got {type(method).__name__}.")
    raw = method.strip().lower()
    normalized = _METHOD_ALIASES.get(raw, raw.replace("-", "_"))
    if normalized == "gloa":
        raise ValueError(
            "mip_nlp_method='gloa' is reserved for future GDP logic-based global "
            "outer approximation. Use mip_nlp_method='goa' for MIP-NLP global "
            "outer approximation, and use gdp_method only for GDP reformulation."
        )
    if normalized not in SUPPORTED_METHODS:
        reserved = ", ".join(sorted(_RESERVED_METHOD_ISSUES))
        raise ValueError(
            f"Unknown mip_nlp_method={method!r}. Choose 'oa', 'ecp', 'fp', "
            "'goa', or 'lp_nlp_bb'. "
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
    from discopt._jax.factorable_reform import canonicalize_entropy

    model = canonicalize_entropy(model)
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
    profile, shot_config, options = split_mip_nlp_profile_options(options)
    model = _apply_shot_profile_reformulations(model, shot_config)

    if method in _IMPLEMENTED_METHODS:
        if method == "fp":
            supported_keys = _FP_OPTION_KEYS
        elif method == "lp_nlp_bb":
            supported_keys = _LP_NLP_BB_OPTION_KEYS
        elif method == "goa":
            supported_keys = GOA_OPTION_KEYS
        else:
            supported_keys = _OA_OPTION_KEYS
        unexpected = sorted(set(options) - supported_keys)
        if unexpected:
            raise ValueError(
                f"Unsupported MIP-NLP {method} option(s): "
                + ", ".join(unexpected)
                + ". Supported options are: "
                + ", ".join(sorted(supported_keys))
            )

        if (
            profile == "shot"
            and shot_config is not None
            and shot_config.tree_strategy == "single_tree"
            and method in {"oa", "ecp"}
        ):
            milp_solver = str(options.get("milp_solver", "gurobi")).strip().lower()
            if milp_solver != "gurobi":
                raise RuntimeError(
                    "SHOT tree_strategy='single_tree' requires milp_solver='gurobi' "
                    "because the current single-tree implementation depends on "
                    "Gurobi MIPSOL/MIPNODE callbacks. Use tree_strategy='multi_tree' "
                    "for non-Gurobi MILP backends."
                )
            unsupported_single_tree = sorted(set(options) - _LP_NLP_BB_OPTION_KEYS)
            if unsupported_single_tree:
                raise ValueError(
                    "SHOT tree_strategy='single_tree' routes to the LP/NLP-BB "
                    "callback solver and does not support option(s): "
                    + ", ".join(unsupported_single_tree)
                    + ". Supported single-tree options are: "
                    + ", ".join(sorted(_LP_NLP_BB_OPTION_KEYS))
                )
            from discopt.solvers.oa import _normalize_init_strategy, solve_lp_nlp_bb

            options["milp_solver"] = "gurobi"
            options["init_strategy"] = _normalize_init_strategy(
                options.get("init_strategy", "rNLP")
            )
            options.setdefault("add_no_good_cuts", True)
            if shot_config.integer_bilinear_strategy == "binary_expansion":
                options.setdefault("integer_to_binary", True)
            result = solve_lp_nlp_bb(
                model,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                max_iterations=max_iterations,
                nlp_solver=nlp_solver,
                initial_point=initial_point,
                mip_nlp_profile=profile,
                mip_nlp_shot_config=shot_config,
                **options,
            )
            ensure_mip_nlp_trace(
                result,
                method="lp_nlp_bb",
                profile=profile,
                shot_config=shot_config,
            )
            return result

        if method == "fp":
            from discopt.solvers.oa import _normalize_init_strategy, solve_feasibility_pump

            if "init_strategy" in options:
                init_strategy = _normalize_init_strategy(options["init_strategy"])
                if init_strategy != "fp":
                    raise ValueError(
                        "mip_nlp_method='fp' only accepts init_strategy='fp' when "
                        "an initialization strategy is supplied."
                    )

            result = solve_feasibility_pump(
                model,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                max_iterations=max_iterations,
                nlp_solver=nlp_solver,
                initial_point=initial_point,
                add_no_good_cuts=bool(options.get("add_no_good_cuts", True)),
                feasibility_norm=options.get("feasibility_norm", "L_infinity"),
                fp_iteration_limit=options.get("fp_iteration_limit"),
                fp_cutoffdecr=options.get("fp_cutoffdecr", 0.0),
                fp_projcuts=options.get("fp_projcuts"),
                fp_transfercuts=options.get("fp_transfercuts", False),
                fp_projzerotol=options.get("fp_projzerotol", 0.0),
                fp_mipgap=options.get("fp_mipgap"),
                fp_discrete_only=options.get("fp_discrete_only", True),
                fp_main_norm=options.get("fp_main_norm"),
                fp_norm_constraint=options.get("fp_norm_constraint", False),
                fp_norm_constraint_coef=options.get("fp_norm_constraint_coef", 1.0),
            )
            ensure_mip_nlp_trace(
                result,
                method=method,
                profile=profile,
                shot_config=shot_config,
            )
            return result

        if method == "goa":
            from discopt.solvers.oa import solve_goa

            result = solve_goa(
                model,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                max_iterations=max_iterations,
                nlp_solver=nlp_solver,
                initial_point=initial_point,
                add_no_good_cuts=bool(options.pop("add_no_good_cuts", True)),
                **options,
            )
            ensure_mip_nlp_trace(
                result,
                method=method,
                profile=profile,
                shot_config=shot_config,
            )
            return result

        if method == "lp_nlp_bb":
            from discopt.solvers.oa import _normalize_init_strategy, solve_lp_nlp_bb

            options["init_strategy"] = _normalize_init_strategy(
                options.get("init_strategy", "rNLP")
            )
            if profile == "shot" and shot_config is not None:
                options.setdefault("add_no_good_cuts", True)
                if shot_config.integer_bilinear_strategy == "binary_expansion":
                    options.setdefault("integer_to_binary", True)
            result = solve_lp_nlp_bb(
                model,
                time_limit=time_limit,
                gap_tolerance=gap_tolerance,
                max_iterations=max_iterations,
                nlp_solver=nlp_solver,
                initial_point=initial_point,
                mip_nlp_profile=profile,
                mip_nlp_shot_config=shot_config,
                **options,
            )
            ensure_mip_nlp_trace(
                result,
                method=method,
                profile=profile,
                shot_config=shot_config,
            )
            return result

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

        result = solve_oa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_iterations,
            nlp_solver=nlp_solver,
            initial_point=initial_point,
            mip_nlp_profile=profile,
            mip_nlp_shot_config=shot_config,
            **options,
        )
        ensure_mip_nlp_trace(
            result,
            method=method,
            profile=profile,
            shot_config=shot_config,
        )
        return result

    raise NotImplementedError(
        f"mip_nlp_method={method!r} is reserved for a future MIP-NLP "
        f"implementation tracked in {_RESERVED_METHOD_ISSUES[method]}; currently "
        "implemented methods are 'oa', 'ecp', 'fp', 'goa', and 'lp_nlp_bb'."
    )
