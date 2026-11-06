"""#1917: ``DISCOPT_NLP_LADDER_SCOPED`` is retired; the variable must be inert.

The flag turned POUNCE's infeasibility retry ladder off in primal heuristics and in
nonconvex node NLPs (#2161). Its §5 panels were cert-clean with a ~4% wall gain, but
GDPlib hda hull was deterministically worse ON (incumbent 4824 -> 2323, bound stuck at
the root) and syngas big-m, the case it was built for, certifies without it. Flag and
implementation were deleted together; the data is in
``docs/dev/data/2161-ladder-scoped/`` and the record in
``docs/dev/flag-retirement-audit.md``. The implementation is in #2161's history.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import discopt.modeling as dm
import pytest

pytestmark = pytest.mark.smoke

_PKG = Path(__file__).resolve().parents[1] / "discopt"


def test_module_and_flag_reads_are_gone():
    assert importlib.util.find_spec("discopt.solvers._nlp_ladder_scope") is None
    hits = [
        str(p.relative_to(_PKG))
        for p in _PKG.rglob("*.py")
        if "NLP_LADDER_SCOPED" in p.read_text(encoding="utf-8")
        or "_nlp_ladder_scope" in p.read_text(encoding="utf-8")
    ]
    # Prove the scan walked the package (CLAUDE.md §6).
    assert sum(1 for _ in _PKG.rglob("*.py")) > 100
    assert hits == [], hits


def _nonconvex():
    m = dm.Model("ladder_inert")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    z = m.binary("z")
    m.subject_to(x * y >= -1 + z)
    m.subject_to(x + y <= 1.5)
    m.minimize(x**2 - y**2 + x * y + 0.5 * z)
    return m


def test_variable_is_inert(monkeypatch):
    out = []
    for val in ("0", "1"):
        monkeypatch.setenv("DISCOPT_NLP_LADDER_SCOPED", val)
        r = _nonconvex().solve(time_limit=30, deterministic=True)
        out.append((r.status, r.objective, r.bound, r.node_count))
    assert out[0][0] == "optimal"
    assert out[0] == out[1], out
