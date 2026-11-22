"""#2299: the bridge-scan truncation warning must name only remedies that ship.

It used to end "... or install the hypergraph detector." No such module, extra or
package exists (KaHyPar border detection, plan T4.1, is deferred), while the shipped
``.dec`` import (``detect_decomposition(model, dec_file=...)``, T4.2) went unmentioned.
"""

from __future__ import annotations

import inspect
import logging
import re
import tomllib
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt.decomposition import detect_decomposition

pytestmark = pytest.mark.smoke

_LOGGER = "discopt.decomposition.graph.kernels"


def _chain(n: int = 301) -> dm.Model:
    # Scalar variables on purpose: the scan's vertex set is ``model._variables``, so
    # one ``shape=(n,)`` array would be a single vertex and never reach the budget.
    m = dm.Model("chain")
    x = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(n)]
    for i in range(n - 1):
        m.subject_to(x[i] + x[i + 1] <= 1)
    m.minimize(-sum(x[1:], x[0]))
    return m


def _truncation_warning(caplog) -> str:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        s = detect_decomposition(_chain())
    assert s.detection_truncated
    msgs = [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER and "bridge-constraint scan skipped" in r.getMessage()
    ]
    assert len(msgs) == 1, msgs
    return msgs[0]


def test_warning_names_no_nonexistent_detector(caplog):
    msg = _truncation_warning(caplog)
    assert "hypergraph detector" not in msg.lower()


def test_every_named_remedy_exists(caplog):
    msg = _truncation_warning(caplog)
    checked = 0
    # mark_coupling is a Model method.
    assert "mark_coupling(" in msg
    assert callable(getattr(dm.Model, "mark_coupling", None))
    checked += 1
    # dec_file= is a keyword of detect_decomposition.
    assert "dec_file=" in msg
    assert "dec_file" in inspect.signature(detect_decomposition).parameters
    checked += 1
    # Any extra the message tells the user to install must be declared.
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    extras = set(tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"])
    named = re.findall(r"discopt\[([\w,-]+)\]", msg)
    for group in named:
        for extra in group.split(","):
            assert extra in extras, (extra, sorted(extras))
            checked += 1
    if re.search(r"\binstall\b", msg, re.IGNORECASE):
        assert named, f"message says 'install' without naming a declared extra: {msg!r}"
    assert checked >= 2
