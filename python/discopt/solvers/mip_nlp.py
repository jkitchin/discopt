"""MIP-NLP decomposition solver-family facade."""

from __future__ import annotations

import time
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
_EVENT_HOOK_OPTION_KEYS = frozenset(
    {
        "external_primal_candidate_hook",
        "external_hyperplane_hook",
        "external_dual_bound_hook",
        "termination_hook",
    }
)
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
        *_EVENT_HOOK_OPTION_KEYS,
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
_DIRECT_ROUTABLE_METHODS = frozenset({"oa", "ecp", "goa"})
_DIRECT_BACKENDS = frozenset({"auto", "gurobi", "pounce", "simplex"})


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


def _normalize_direct_backend(options: dict[str, Any]) -> str:
    backend = str(options.get("milp_solver", "auto")).strip().lower().replace("-", "_")
    return backend or "auto"


def _annotate_strategy_trace(
    result: SolveResult,
    *,
    method: str,
    profile: str,
    shot_config: MIPNLPShotConfig | None,
    selected_strategy: str,
    problem_class: str | None = None,
    backend: str | None = None,
    direct_attempt: dict[str, object] | None = None,
) -> SolveResult:
    ensure_mip_nlp_trace(
        result,
        method=method,
        profile=profile,
        shot_config=shot_config,
    )
    selection: dict[str, object] = {"selected_strategy": selected_strategy}
    if problem_class is not None:
        selection["problem_class"] = problem_class
    if backend is not None:
        selection["backend"] = backend
    if direct_attempt is not None:
        selection["direct_attempt"] = direct_attempt

    trace = result.mip_nlp_trace
    if trace is not None:
        trace["selected_strategy"] = selected_strategy
        trace["strategy_selection"] = selection
        summary = trace.setdefault("summary", {})
        if isinstance(summary, dict):
            summary["selected_strategy"] = selected_strategy
            if backend is not None:
                summary["selected_backend"] = backend
            if direct_attempt is not None:
                summary["direct_routing"] = direct_attempt
    return result


def _direct_attempt(
    *,
    problem_class: str,
    candidate_strategy: str,
    backend: str,
    fallback_reason: str,
) -> dict[str, object]:
    return {
        "candidate_strategy": candidate_strategy,
        "problem_class": problem_class,
        "backend": backend,
        "fallback_reason": fallback_reason,
    }


def _requires_known_convexity(
    model: Model,
    *,
    problem_class: str,
    candidate_strategy: str,
    backend: str,
    failure_label: str,
) -> dict[str, object] | None:
    import discopt.solver as solver_module

    known, is_convex, _constraint_mask = solver_module._classify_model_convexity(
        model,
        failure_label=failure_label,
        log_nonconvex_continuous=problem_class == "nlp",
    )
    if not known:
        return _direct_attempt(
            problem_class=problem_class,
            candidate_strategy=candidate_strategy,
            backend=backend,
            fallback_reason="convexity_unknown",
        )
    if not is_convex:
        return _direct_attempt(
            problem_class=problem_class,
            candidate_strategy=candidate_strategy,
            backend=backend,
            fallback_reason="nonconvex_model",
        )
    return None


def _annotate_direct_result(
    result: SolveResult,
    *,
    profile: str,
    shot_config: MIPNLPShotConfig,
    selected_strategy: str,
    problem_class: str,
    backend: str,
) -> SolveResult:
    return _annotate_strategy_trace(
        result,
        method="direct",
        profile=profile,
        shot_config=shot_config,
        selected_strategy=selected_strategy,
        problem_class=problem_class,
        backend=backend,
    )


def _try_direct_nlp(
    model: Model,
    *,
    profile: str,
    shot_config: MIPNLPShotConfig,
    problem_class: str,
    time_limit: float,
    nlp_solver: str,
    initial_point: Any,
    solver_module: Any,
) -> tuple[SolveResult | None, dict[str, object] | None]:
    strategy = "direct_nlp"
    fallback = _requires_known_convexity(
        model,
        problem_class=problem_class,
        candidate_strategy=strategy,
        backend=nlp_solver,
        failure_label="SHOT direct NLP convexity detection failed",
    )
    if fallback is not None:
        return None, fallback
    result = solver_module._solve_continuous(
        model,
        time_limit,
        None,
        time.perf_counter(),
        nlp_solver,
        initial_point=initial_point,
    )
    result.convex_fast_path = True
    return (
        _annotate_direct_result(
            result,
            profile=profile,
            shot_config=shot_config,
            selected_strategy=strategy,
            problem_class=problem_class,
            backend=nlp_solver,
        ),
        None,
    )


def _try_direct_lp(
    model: Model,
    *,
    profile: str,
    shot_config: MIPNLPShotConfig,
    problem_class: str,
    backend: str,
    time_limit: float,
    nlp_solver: str,
    solver_module: Any,
) -> tuple[SolveResult | None, dict[str, object] | None]:
    strategy = "direct_lp"
    t_start = time.perf_counter()
    if backend == "gurobi":
        result = solver_module._solve_lp_gurobi(model, t_start, time_limit)
    elif backend == "pounce":
        result = solver_module._solve_lp_pounce(model, t_start, time_limit)
    elif backend == "simplex":
        result = None
    else:
        result = solver_module._solve_lp(
            model,
            t_start,
            time_limit,
            prefer_pounce=nlp_solver == "pounce",
        )
    if result is None:
        return None, _direct_attempt(
            problem_class=problem_class,
            candidate_strategy=strategy,
            backend=backend,
            fallback_reason="direct_backend_unavailable",
        )
    return (
        _annotate_direct_result(
            result,
            profile=profile,
            shot_config=shot_config,
            selected_strategy=strategy,
            problem_class=problem_class,
            backend=backend,
        ),
        None,
    )


def _try_direct_milp(
    model: Model,
    *,
    profile: str,
    shot_config: MIPNLPShotConfig,
    problem_class: str,
    backend: str,
    time_limit: float,
    gap_tolerance: float,
    max_iterations: int,
    nlp_solver: str,
    solver_module: Any,
) -> tuple[SolveResult | None, dict[str, object] | None]:
    strategy = "direct_milp"
    t_start = time.perf_counter()
    result = None
    if backend == "gurobi":
        result = solver_module._solve_milp_gurobi(model, t_start, time_limit, gap_tolerance)
    elif backend == "simplex":
        result = solver_module._solve_milp_simplex(
            model,
            time_limit,
            gap_tolerance,
            max_iterations,
            t_start,
        )
    elif backend == "pounce":
        result = solver_module._solve_milp_bb(
            model,
            time_limit,
            gap_tolerance,
            16,
            "best_first",
            max_iterations,
            t_start,
            prefer_pounce=True,
            node_engine="simplex",
        )
    else:
        result = solver_module._solve_milp_bb(
            model,
            time_limit,
            gap_tolerance,
            16,
            "best_first",
            max_iterations,
            t_start,
            prefer_pounce=nlp_solver == "pounce",
            node_engine="simplex" if nlp_solver == "pounce" else "pounce",
        )
    if result is None:
        return None, _direct_attempt(
            problem_class=problem_class,
            candidate_strategy=strategy,
            backend=backend,
            fallback_reason="direct_backend_unavailable",
        )
    return (
        _annotate_direct_result(
            result,
            profile=profile,
            shot_config=shot_config,
            selected_strategy=strategy,
            problem_class=problem_class,
            backend=backend,
        ),
        None,
    )


def _try_direct_qp(
    model: Model,
    *,
    profile: str,
    shot_config: MIPNLPShotConfig,
    problem_class: str,
    is_miqp: bool,
    backend: str,
    time_limit: float,
    gap_tolerance: float,
    max_iterations: int,
    nlp_solver: str,
    solver_module: Any,
) -> tuple[SolveResult | None, dict[str, object] | None]:
    strategy = "direct_miqp" if is_miqp else "direct_qp"
    if backend == "simplex":
        return None, _direct_attempt(
            problem_class=problem_class,
            candidate_strategy=strategy,
            backend=backend,
            fallback_reason="direct_backend_unavailable",
        )
    fallback = _requires_known_convexity(
        model,
        problem_class=problem_class,
        candidate_strategy=strategy,
        backend=backend,
        failure_label="SHOT direct quadratic convexity detection failed",
    )
    if fallback is not None:
        return None, fallback

    t_start = time.perf_counter()
    result = None
    if backend == "gurobi":
        result = solver_module._solve_qp_gurobi(model, t_start, time_limit, gap_tolerance)
    elif backend == "pounce" and not is_miqp:
        result = solver_module._solve_qp_pounce(model, t_start, time_limit)
        if result is None:
            return None, _direct_attempt(
                problem_class=problem_class,
                candidate_strategy=strategy,
                backend=backend,
                fallback_reason="direct_backend_unavailable",
            )
    elif is_miqp:
        result = solver_module._solve_miqp_bb(
            model,
            time_limit,
            gap_tolerance,
            16,
            "best_first",
            max_iterations,
            t_start,
            prefer_pounce=nlp_solver == "pounce" or backend == "pounce",
        )
    else:
        result = solver_module._solve_qp(
            model,
            t_start,
            prefer_pounce=nlp_solver == "pounce",
        )

    if result is None:
        return None, _direct_attempt(
            problem_class=problem_class,
            candidate_strategy=strategy,
            backend=backend,
            fallback_reason="direct_backend_unavailable",
        )
    return (
        _annotate_direct_result(
            result,
            profile=profile,
            shot_config=shot_config,
            selected_strategy=strategy,
            problem_class=problem_class,
            backend=backend,
        ),
        None,
    )


def _try_direct_qcp(
    model: Model,
    *,
    profile: str,
    shot_config: MIPNLPShotConfig,
    problem_class: str,
    backend: str,
    time_limit: float,
    gap_tolerance: float,
    solver_module: Any,
) -> tuple[SolveResult | None, dict[str, object] | None]:
    strategy = {
        "qcp": "direct_qcp",
        "qcqp": "direct_qcqp",
        "miqcp": "direct_miqcp",
        "miqcqp": "direct_miqcqp",
    }[problem_class]
    if backend != "gurobi":
        return None, _direct_attempt(
            problem_class=problem_class,
            candidate_strategy=strategy,
            backend=backend,
            fallback_reason="requires_milp_solver_gurobi",
        )
    fallback = _requires_known_convexity(
        model,
        problem_class=problem_class,
        candidate_strategy=strategy,
        backend=backend,
        failure_label="SHOT direct QCP convexity detection failed",
    )
    if fallback is not None:
        return None, fallback
    result = solver_module._solve_qcp_gurobi(
        model,
        time.perf_counter(),
        time_limit,
        gap_tolerance,
    )
    return (
        _annotate_direct_result(
            result,
            profile=profile,
            shot_config=shot_config,
            selected_strategy=strategy,
            problem_class=problem_class,
            backend=backend,
        ),
        None,
    )


def _try_solve_shot_direct_strategy(
    model: Model,
    *,
    method: str,
    profile: str,
    shot_config: MIPNLPShotConfig | None,
    options: dict[str, Any],
    time_limit: float,
    gap_tolerance: float,
    max_iterations: int,
    nlp_solver: str,
    initial_point: Any,
) -> tuple[SolveResult | None, dict[str, object] | None]:
    if (
        profile != "shot"
        or shot_config is None
        or method not in _DIRECT_ROUTABLE_METHODS
        or shot_config.direct_quadratic_routing == "off"
    ):
        return None, None

    import discopt.solver as solver_module
    from discopt._jax.problem_classifier import ProblemClass, classify_problem

    backend = _normalize_direct_backend(options)
    requested_hooks = sorted(key for key in _EVENT_HOOK_OPTION_KEYS if options.get(key) is not None)
    if requested_hooks:
        return None, {
            "candidate_strategy": "direct",
            "backend": backend,
            "fallback_reason": "external_hooks_requested",
            "external_hooks": requested_hooks,
        }
    try:
        problem_class = classify_problem(model)
    except Exception as exc:
        return None, {
            "candidate_strategy": "direct",
            "backend": backend,
            "fallback_reason": f"classification_failed: {type(exc).__name__}",
        }

    problem_value = problem_class.value
    if backend not in _DIRECT_BACKENDS:
        return None, {
            "candidate_strategy": "direct",
            "problem_class": problem_value,
            "backend": backend,
            "fallback_reason": "unsupported_direct_backend",
        }

    if problem_class == ProblemClass.NLP:
        return _try_direct_nlp(
            model,
            profile=profile,
            shot_config=shot_config,
            problem_class=problem_value,
            solver_module=solver_module,
            time_limit=time_limit,
            nlp_solver=nlp_solver,
            initial_point=initial_point,
        )
    if problem_class == ProblemClass.LP:
        return _try_direct_lp(
            model,
            profile=profile,
            shot_config=shot_config,
            problem_class=problem_value,
            solver_module=solver_module,
            backend=backend,
            time_limit=time_limit,
            nlp_solver=nlp_solver,
        )
    if problem_class == ProblemClass.MILP:
        return _try_direct_milp(
            model,
            profile=profile,
            shot_config=shot_config,
            problem_class=problem_value,
            solver_module=solver_module,
            backend=backend,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_iterations,
            nlp_solver=nlp_solver,
        )
    if problem_class in (ProblemClass.QP, ProblemClass.MIQP):
        return _try_direct_qp(
            model,
            profile=profile,
            shot_config=shot_config,
            problem_class=problem_value,
            solver_module=solver_module,
            is_miqp=problem_class == ProblemClass.MIQP,
            backend=backend,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_iterations,
            nlp_solver=nlp_solver,
        )
    if problem_class in (
        ProblemClass.QCP,
        ProblemClass.QCQP,
        ProblemClass.MIQCP,
        ProblemClass.MIQCQP,
    ):
        return _try_direct_qcp(
            model,
            profile=profile,
            shot_config=shot_config,
            problem_class=problem_value,
            solver_module=solver_module,
            backend=backend,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
        )
    return None, None


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
            return _annotate_strategy_trace(
                result,
                method="lp_nlp_bb",
                profile=profile,
                shot_config=shot_config,
                selected_strategy="lp_nlp_bb",
                backend="gurobi",
            )

        direct_result, direct_attempt = _try_solve_shot_direct_strategy(
            model,
            method=method,
            profile=profile,
            shot_config=shot_config,
            options=options,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_iterations,
            nlp_solver=nlp_solver,
            initial_point=initial_point,
        )
        if direct_result is not None:
            return direct_result

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
            return _annotate_strategy_trace(
                result,
                method=method,
                profile=profile,
                shot_config=shot_config,
                selected_strategy=method,
            )

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
            return _annotate_strategy_trace(
                result,
                method=method,
                profile=profile,
                shot_config=shot_config,
                selected_strategy=method,
                direct_attempt=direct_attempt,
            )

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
            return _annotate_strategy_trace(
                result,
                method=method,
                profile=profile,
                shot_config=shot_config,
                selected_strategy=method,
                backend=str(options.get("milp_solver", "gurobi")).strip().lower(),
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
        return _annotate_strategy_trace(
            result,
            method=method,
            profile=profile,
            shot_config=shot_config,
            selected_strategy=method,
            backend=str(options.get("milp_solver", "auto")).strip().lower(),
            direct_attempt=direct_attempt,
        )

    raise NotImplementedError(
        f"mip_nlp_method={method!r} is reserved for a future MIP-NLP "
        f"implementation tracked in {_RESERVED_METHOD_ISSUES[method]}; currently "
        "implemented methods are 'oa', 'ecp', 'fp', 'goa', and 'lp_nlp_bb'."
    )
