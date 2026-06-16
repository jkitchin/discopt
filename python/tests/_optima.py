"""Loader for the shared reference-optima registry (``data/known_optima.toml``).

A single source of truth for the published global optima of benchmark instances
used across the certification suites, so a soundness regression is checked
against one authoritative value rather than a constant duplicated per test file.

Usage::

    from _optima import known_optimum, optima_registry

    opt = known_optimum("ex8_1_1")               # -> -2.021806783
    entry = known_optimum("ex8_1_1", full=True)   # -> {"optimum": ..., "source": ...}
    all_entries = optima_registry()               # -> {name: entry, ...}

``_optima`` is importable directly because pytest puts the ``tests`` directory
on ``sys.path`` (there is no ``tests/__init__.py``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib

_REGISTRY_PATH = Path(__file__).parent / "data" / "known_optima.toml"


@lru_cache(maxsize=1)
def optima_registry() -> dict[str, dict[str, Any]]:
    """Return the full registry mapping instance name -> metadata table.

    The top-level ``schema`` key is stripped; every remaining table is an
    instance entry with at least an ``optimum`` field.
    """
    with _REGISTRY_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    data.pop("schema", None)
    return data


def known_optimum(name: str, *, full: bool = False) -> Any:
    """Return the recorded global optimum for ``name`` (or its full entry).

    Raises ``KeyError`` with the list of known instances if ``name`` is absent,
    so a typo fails loudly rather than silently skipping a soundness check.
    """
    registry = optima_registry()
    try:
        entry = registry[name]
    except KeyError:
        known = ", ".join(sorted(registry))
        raise KeyError(f"no recorded optimum for {name!r}; known instances: {known}") from None
    return entry if full else entry["optimum"]
