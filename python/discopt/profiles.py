"""Named solve profiles for the ``discopt solve`` CLI.

A *profile* is a named bundle of :meth:`Model.solve` options (including a nested
``tuning`` dict for :class:`~discopt.solver_tuning.SolverTuning`). Built-in
profiles ship with discopt; users add or override them in a TOML file. Precedence
when resolving the options for a solve is: **explicit CLI flag > profile >
built-in defaults**.

User config is read (later wins) from:
  1. ``~/.config/discopt/profiles.toml``   (or ``$XDG_CONFIG_HOME/discopt/profiles.toml``)
  2. ``./discopt.toml``                     (a ``[profiles.<name>]`` table in cwd)

Schema mirrors the benchmark solver-config style::

    [profiles.fast]
    time_limit = 10
    gap_tolerance = 1e-2
    tuning = { node_bound_mode = "lp", node_nlp_stride = 8 }
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

# Built-in profiles. ``tuning`` is a plain dict (validated against SolverTuning
# fields when the solve is built); everything else is a Model.solve kwarg.
BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    "fast": {
        "time_limit": 10.0,
        "gap_tolerance": 1e-2,
        "tuning": {"node_bound_mode": "lp", "node_nlp_stride": 8},
    },
    "exact": {
        "time_limit": 3600.0,
        "gap_tolerance": 1e-6,
    },
    "feasible": {
        "nlp_bb": True,
        "gap_tolerance": 1e-1,
    },
}


def _user_config_paths() -> list[Path]:
    cfg_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(cfg_home) if cfg_home else Path.home() / ".config"
    return [base / "discopt" / "profiles.toml", Path.cwd() / "discopt.toml"]


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:  # pragma: no cover - py310 fallback
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return {}  # no TOML reader -> user config silently unavailable
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError):
        return {}
    profs = data.get("profiles", data)  # accept a bare table or a [profiles.*] block
    return profs if isinstance(profs, dict) else {}


def load_profiles() -> dict[str, dict[str, Any]]:
    """Built-in profiles overlaid by user TOML (cwd wins over the home config)."""
    profiles = copy.deepcopy(BUILTIN_PROFILES)
    for path in _user_config_paths():
        for name, opts in _load_toml(path).items():
            if isinstance(opts, dict):
                profiles[name] = opts
    return profiles


def resolve_options(profile_name: str | None, cli_overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge a profile with explicit CLI overrides (CLI wins).

    The ``tuning`` sub-dict is merged key-wise, so a single ``--tuning rlt_quad=…``
    flag overrides only that field, not the whole profile bundle. Raises
    ``KeyError`` with the available names if ``profile_name`` is unknown.
    """
    profiles = load_profiles()
    base: dict[str, Any] = {}
    if profile_name is not None:
        if profile_name not in profiles:
            raise KeyError(f"unknown profile {profile_name!r}; available: {sorted(profiles)}")
        base = copy.deepcopy(profiles[profile_name])

    merged = dict(base)
    cli_tuning = cli_overrides.pop("tuning", None)
    for k, v in cli_overrides.items():
        merged[k] = v
    if cli_tuning:
        merged["tuning"] = {**base.get("tuning", {}), **cli_tuning}
    return merged
