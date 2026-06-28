"""Shared option names for the MIP-NLP solver-family facade."""

from __future__ import annotations

from typing import Any

GOA_OA_FORWARD_OPTION_KEYS = (
    "rel_gap",
    "max_iter",
    "init_strategy",
    "feasibility_norm",
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
