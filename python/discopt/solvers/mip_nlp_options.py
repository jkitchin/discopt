"""Shared option names for the MIP-NLP solver-family facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FP_OPTION_KEYS = (
    "fp_iteration_limit",
    "fp_cutoffdecr",
    "fp_projcuts",
    "fp_transfercuts",
    "fp_projzerotol",
    "fp_mipgap",
    "fp_discrete_only",
    "fp_main_norm",
    "fp_norm_constraint",
    "fp_norm_constraint_coef",
)

GOA_OA_FORWARD_OPTION_KEYS = (
    "rel_gap",
    "max_iter",
    "init_strategy",
    "feasibility_norm",
    *FP_OPTION_KEYS,
)

GOA_AMP_OPTION_DEFAULTS: dict[str, Any] = {
    "abs_tol": 1e-6,
    "use_start_as_incumbent": False,
    "n_init_partitions": 2,
    "partition_method": "auto",
    "iteration_callback": None,
    "milp_time_limit": None,
    "milp_gap_tolerance": None,
    "apply_partitioning": True,
    "disc_var_pick": None,
    "partition_scaling_factor": 10.0,
    "partition_scaling_factor_update": None,
    "disc_add_partition_method": "adaptive",
    "disc_abs_width_tol": 1e-3,
    "convhull_formulation": "disaggregated",
    "convhull_ebd": False,
    "convhull_ebd_encoding": "gray",
    "presolve_bt": True,
    "presolve_bt_algo": 1,
    "presolve_bt_time_limit": None,
    "presolve_bt_mip_time_limit": None,
    "obbt_at_root": False,
    "obbt_time_limit": 30.0,
    "obbt_with_cutoff": False,
    "alphabb_cutoff_obbt": True,
    "milp_solver": "auto",
}

GOA_AMP_ONLY_OPTION_KEYS = tuple(GOA_AMP_OPTION_DEFAULTS)
GOA_OPTION_KEYS = frozenset(
    ("add_no_good_cuts", *GOA_OA_FORWARD_OPTION_KEYS, *GOA_AMP_ONLY_OPTION_KEYS)
)

MIP_NLP_PROFILE_OPTION_KEYS = ("mip_nlp_profile",)

SHOT_OPTION_KEYS = (
    "tree_strategy",
    "cut_strategy",
    "rootsearch_strategy",
    "fixed_nlp_strategy",
    "solution_pool_capacity",
    "hyperplane_max_per_iter",
    "hyperplane_selection_factor",
    "relaxation_phase",
    "mip_solution_limit_strategy",
    "convex_bounding",
    "master_repair",
    "reduction_cuts",
)

_PROFILE_ALIASES = {
    "": "default",
    "default": "default",
    "none": "default",
    "shot": "shot",
}

_TREE_STRATEGIES = frozenset({"auto", "multi_tree", "single_tree"})
_CUT_STRATEGIES = frozenset({"auto", "oa", "ecp", "esh"})
_ROOTSEARCH_STRATEGIES = frozenset({"auto", "none", "bisection", "toms748"})
_FIXED_NLP_STRATEGIES = frozenset(
    {"auto", "none", "always", "adaptive", "iteration", "time", "solution_pool"}
)
_RELAXATION_PHASES = frozenset({"auto", "off", "initial", "periodic"})
_MIP_SOLUTION_LIMIT_STRATEGIES = frozenset({"auto", "none", "static", "adaptive", "force_optimal"})


@dataclass(frozen=True)
class MIPNLPShotConfig:
    """Validated experimental controls for the SHOT-parity MIP-NLP profile."""

    tree_strategy: str = "multi_tree"
    cut_strategy: str = "auto"
    rootsearch_strategy: str = "auto"
    fixed_nlp_strategy: str = "adaptive"
    solution_pool_capacity: int | None = None
    hyperplane_max_per_iter: int | None = None
    hyperplane_selection_factor: float = 1.0
    relaxation_phase: str = "auto"
    mip_solution_limit_strategy: str = "adaptive"
    convex_bounding: bool = False
    master_repair: bool = False
    reduction_cuts: bool = False

    def as_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_mip_nlp_profile(value: Any = None) -> str:
    """Normalize the optional MIP-NLP profile selector."""
    if value is None:
        return "default"
    if not isinstance(value, str):
        raise ValueError(f"mip_nlp_profile must be a string, got {type(value).__name__}.")
    key = value.strip().lower().replace("-", "_")
    if key not in _PROFILE_ALIASES:
        raise ValueError("Unknown mip_nlp_profile={!r}. Choose 'default' or 'shot'.".format(value))
    return _PROFILE_ALIASES[key]


def _normalize_enum(name: str, value: Any, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}.")
    key = value.strip().lower().replace("-", "_")
    if key not in allowed:
        raise ValueError(
            f"Unknown {name}={value!r}. Choose one of: " + ", ".join(sorted(allowed)) + "."
        )
    return key


def _normalize_optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}.")
    return out


def _normalize_unit_factor(name: str, value: Any) -> float:
    out = float(value)
    if not 0.0 < out <= 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1], got {value!r}.")
    return out


def _normalize_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "on"}:
            return True
        if key in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def normalize_shot_config(raw_options: dict[str, Any]) -> MIPNLPShotConfig:
    """Validate the SHOT-profile option subset and fill stable defaults."""
    unexpected = sorted(set(raw_options) - set(SHOT_OPTION_KEYS))
    if unexpected:
        raise ValueError(
            "Unsupported SHOT-profile MIP-NLP option(s): "
            + ", ".join(unexpected)
            + ". Supported options are: "
            + ", ".join(SHOT_OPTION_KEYS)
        )

    return MIPNLPShotConfig(
        tree_strategy=_normalize_enum(
            "tree_strategy",
            raw_options.get("tree_strategy", MIPNLPShotConfig.tree_strategy),
            _TREE_STRATEGIES,
        ),
        cut_strategy=_normalize_enum(
            "cut_strategy",
            raw_options.get("cut_strategy", MIPNLPShotConfig.cut_strategy),
            _CUT_STRATEGIES,
        ),
        rootsearch_strategy=_normalize_enum(
            "rootsearch_strategy",
            raw_options.get("rootsearch_strategy", MIPNLPShotConfig.rootsearch_strategy),
            _ROOTSEARCH_STRATEGIES,
        ),
        fixed_nlp_strategy=_normalize_enum(
            "fixed_nlp_strategy",
            raw_options.get("fixed_nlp_strategy", MIPNLPShotConfig.fixed_nlp_strategy),
            _FIXED_NLP_STRATEGIES,
        ),
        solution_pool_capacity=_normalize_optional_positive_int(
            "solution_pool_capacity",
            raw_options.get("solution_pool_capacity", MIPNLPShotConfig.solution_pool_capacity),
        ),
        hyperplane_max_per_iter=_normalize_optional_positive_int(
            "hyperplane_max_per_iter",
            raw_options.get("hyperplane_max_per_iter", MIPNLPShotConfig.hyperplane_max_per_iter),
        ),
        hyperplane_selection_factor=_normalize_unit_factor(
            "hyperplane_selection_factor",
            raw_options.get(
                "hyperplane_selection_factor",
                MIPNLPShotConfig.hyperplane_selection_factor,
            ),
        ),
        relaxation_phase=_normalize_enum(
            "relaxation_phase",
            raw_options.get("relaxation_phase", MIPNLPShotConfig.relaxation_phase),
            _RELAXATION_PHASES,
        ),
        mip_solution_limit_strategy=_normalize_enum(
            "mip_solution_limit_strategy",
            raw_options.get(
                "mip_solution_limit_strategy",
                MIPNLPShotConfig.mip_solution_limit_strategy,
            ),
            _MIP_SOLUTION_LIMIT_STRATEGIES,
        ),
        convex_bounding=_normalize_bool(
            "convex_bounding",
            raw_options.get("convex_bounding", MIPNLPShotConfig.convex_bounding),
        ),
        master_repair=_normalize_bool(
            "master_repair",
            raw_options.get("master_repair", MIPNLPShotConfig.master_repair),
        ),
        reduction_cuts=_normalize_bool(
            "reduction_cuts",
            raw_options.get("reduction_cuts", MIPNLPShotConfig.reduction_cuts),
        ),
    )


def split_mip_nlp_profile_options(
    options: dict[str, Any],
) -> tuple[str, MIPNLPShotConfig | None, dict[str, Any]]:
    """Separate experimental profile options from method-native options."""
    remaining = dict(options)
    profile = normalize_mip_nlp_profile(remaining.pop("mip_nlp_profile", None))
    shot_raw = {key: remaining.pop(key) for key in SHOT_OPTION_KEYS if key in remaining}
    if profile != "shot":
        if shot_raw:
            raise ValueError(
                "SHOT-style MIP-NLP option(s) require mip_nlp_profile='shot': "
                + ", ".join(sorted(shot_raw))
                + "."
            )
        return profile, None, remaining
    return profile, normalize_shot_config(shot_raw), remaining


def ensure_mip_nlp_trace(
    result: Any,
    *,
    method: str,
    profile: str,
    shot_config: MIPNLPShotConfig | None,
) -> None:
    """Attach a minimal trace envelope when a solver path did not populate one."""
    if getattr(result, "mip_nlp_trace", None) is not None:
        return
    result.mip_nlp_trace = {
        "schema_version": 1,
        "solver": "mip-nlp",
        "method": method,
        "profile": profile,
        "shot_options": shot_config.as_trace_dict() if shot_config is not None else {},
        "iterations": [],
        "summary": {
            "mip_count": int(getattr(result, "mip_count", 0) or 0),
            "nlp_subproblem_count": int(getattr(result, "subnlp_calls", 0) or 0),
            "cut_count": None,
            "solution_pool_candidates": 0,
        },
        "termination_reason": getattr(result, "status", None),
        "master_bound_valid": bool(getattr(result, "gap_certified", True)),
        "gap_certified": bool(getattr(result, "gap_certified", True)),
        "bound_validity": "global" if bool(getattr(result, "gap_certified", True)) else "heuristic",
    }
