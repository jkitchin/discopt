"""#2143: the root LOA proposer's proved bound floors the search.

``DISCOPT_GDP_LOA_PROPOSER_BOUND``, graduated default ON 2026-11-05.

The proposer (``DISCOPT_GDP_LOA_PROPOSER``) runs LOA on the source GDP and, until
this flag, kept only its point. On GDPlib ``syngas`` hull LOA proves 4669.0235
optimal at ~67 s (master bound 4668.809), while the hull tree reaches that same
bound only after 27 nodes at 180-224 s, past a 120 s limit. ON installs LOA's bound
with ``PyTreeManager.seed_root_bound``: syngas hull certifies at 71.5 s.

LOA's bound is valid for the lowered model because LOA cuts only certified-convex
rows and solves subproblems globally (its ``bound_valid``), and every GDP lowering
is exact at integer selectors. The guards refuse a bound whose objective does not
map (sign, offset, scale) or that lies above the incumbent.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402
from discopt._relax import primal_heuristics as PH  # noqa: E402

pytestmark = pytest.mark.smoke

FLAG = "DISCOPT_GDP_LOA_PROPOSER_BOUND"
INSTALLED = "installed as the root dual bound"
#: The toy's optimum: x = 2 in the left disjunct (cost (2-3)^2 + 1).
TOY_OPT = 2.0


def _toy(maximize: bool = False):
    m = dm.Model("loa_bound_toy")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    m.either_or([[x <= 2, z >= 1], [x >= 6, z >= 4 + dm.exp(0.1 * x)]], name="d")
    cost = (x - 3) ** 2 + z
    if maximize:
        m.maximize(-cost)
    else:
        m.minimize(cost)
    return m


def _solve(monkeypatch, flag: str, *, maximize: bool = False, method: str = "hull"):
    """The toy at a 1-node budget, the root wave forced to fail so the proposer fires."""
    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv("DISCOPT_GDP_LOA_PROPOSER", "1")
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    monkeypatch.setattr(PH, "one_hot_config_subnlp", lambda *a, **k: [])
    records: list[str] = []

    class H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    h = H()
    lg = logging.getLogger("discopt")
    old = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    try:
        r = _toy(maximize).solve(gdp_method=method, time_limit=60, max_nodes=1)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    return r, records


def test_flag_defaults_on(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert S._gdp_loa_proposer_bound_enabled() is True  # graduated 2026-11-05 (#2143)
    monkeypatch.setenv(FLAG, "0")
    assert S._gdp_loa_proposer_bound_enabled() is False
    monkeypatch.setenv(FLAG, "1")
    assert S._gdp_loa_proposer_bound_enabled() is True


def test_off_never_installs_the_bound(monkeypatch):
    r, records = _solve(monkeypatch, "0")
    assert any("GDP LOA proposer (root): LOA" in m for m in records), records
    assert not any(INSTALLED in m for m in records)
    assert r.objective == pytest.approx(TOY_OPT, abs=1e-5)


def _installed(records: list[str]) -> float:
    """The internal-sense bound the proposer installed (from its log line)."""
    (line,) = [m for m in records if INSTALLED in m]
    return float(line.rsplit("internal ", 1)[1].rstrip(")"))


def test_on_installs_a_valid_floor(monkeypatch):
    """On the toy LOA stops at its first incumbent (status feasible), so its bound is
    valid but open: it must be installed, floor the reported bound, and stay below
    the optimum. Hull only: under big-M the toy is convex and takes the convex MINLP
    route, which never reaches the proposer."""
    r, records = _solve(monkeypatch, "1")
    seed = _installed(records)
    assert seed <= TOY_OPT + 1e-9
    assert r.objective == pytest.approx(TOY_OPT, abs=1e-5)
    assert r.bound is not None and seed - 1e-9 <= r.bound <= TOY_OPT + 1e-6


def test_on_maximize_maps_the_sign(monkeypatch):
    r, records = _solve(monkeypatch, "1", maximize=True)
    seed = _installed(records)  # internal (minimization) sense: -(user bound)
    assert seed <= TOY_OPT + 1e-9
    assert r.objective == pytest.approx(-TOY_OPT, abs=1e-5)
    # When maximizing the bound is an upper bound: never below the optimum, and at
    # most the installed one mapped back to the user's sense.
    assert r.bound is not None and -TOY_OPT - 1e-6 <= r.bound <= -seed + 1e-9


class _Tree:
    def __init__(self, inc):
        self._inc = inc
        self.seeded: list[float] = []

    def incumbent(self):
        return self._inc

    def seed_root_bound(self, b):
        self.seeded.append(b)


def _work():
    m = dm.Model("w")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    return m


def _loa(bound, objective, valid=True):
    return SimpleNamespace(bound=bound, objective=objective, bound_valid=valid, status="optimal")


def test_guard_installs_a_consistent_bound():
    tree = _Tree((None, 10.0))
    assert S._seed_loa_proposer_bound(tree, _work(), _loa(9.5, 10.0), 10.0)
    assert tree.seeded == [9.5]


def test_guard_refuses_an_invalid_bound():
    tree = _Tree((None, 10.0))
    assert not S._seed_loa_proposer_bound(tree, _work(), _loa(9.5, 10.0, valid=False), 10.0)
    assert tree.seeded == []


def test_guard_refuses_an_objective_that_does_not_map():
    """An offset between LOA's objective and the lowered model's: bound not used."""
    tree = _Tree((None, 10.0))
    assert not S._seed_loa_proposer_bound(tree, _work(), _loa(9.5, 10.0), 10.5)
    assert tree.seeded == []


def test_guard_refuses_a_bound_above_the_incumbent(caplog):
    tree = _Tree((None, 9.0))
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        assert not S._seed_loa_proposer_bound(tree, _work(), _loa(9.5, 10.0), 10.0)
    assert tree.seeded == []
    assert any("false certificate" in r.getMessage() for r in caplog.records)
