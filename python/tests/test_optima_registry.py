"""Integrity locks for the shared reference-optima registry.

``data/known_optima.toml`` is the single source of truth for published global
optima used by the certification suites. These checks ensure it stays
well-formed (every entry has a numeric ``optimum`` and a source) and that the
loader fails loudly on an unknown instance rather than silently skipping a
soundness check.
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import pytest
from _optima import known_optimum, optima_registry


def test_every_entry_is_well_formed():
    registry = optima_registry()
    assert registry, "registry is empty"
    for name, entry in registry.items():
        assert "optimum" in entry, f"{name}: missing 'optimum'"
        assert isinstance(entry["optimum"], (int, float)), f"{name}: optimum not numeric"
        assert entry.get("source"), f"{name}: missing 'source'"
        assert entry.get("status"), f"{name}: missing 'status'"


def test_known_optimum_returns_float_and_full_entry():
    assert known_optimum("mathopt5_2") == pytest.approx(-1.0)
    entry = known_optimum("mathopt5_2", full=True)
    assert entry["optimum"] == pytest.approx(-1.0)
    assert entry["source"] == "MINLPLib"


def test_unknown_instance_raises_loudly():
    with pytest.raises(KeyError, match="no recorded optimum"):
        known_optimum("does_not_exist")
