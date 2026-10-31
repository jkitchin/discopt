"""#1917 / #1775 G5.1 ``DISCOPT_GDP_DISJUNCT_FBBT``: a big-M bounded over the siblings'
disjunct-local FBBT boxes (:mod:`discopt.gdp.bounds`).

Bound-changing on every big-M lowering, so default OFF (CLAUDE.md §5). Pinned:

* flag OFF: no cap, today's M;
* flag ON: every M is ``<=`` today's, and strictly smaller on a model where a
  sibling's own rows confine the box; the lowering is otherwise the same rows;
* the differential battery (no feasible point cut, no infeasible point admitted)
  on every shape under big-M, mbigm and auto with the flag ON;
* the same certified optimum as the hull, with a root bound at least as strong;
* the soundness guards: a reified disjunction takes no cap, a disjunct FBBT calls
  infeasible keeps the global box, and misaligned FBBT output produces no cap.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt._relax import gdp_reformulate as grm
from discopt.gdp.bounds import disjunct_local_big_m_caps
from gdp_differential import run_shape
from test_gdp_differential_battery import SHAPES

FLAG = "DISCOPT_GDP_DISJUNCT_FBBT"
_REAL_COMPUTE_BIG_M = grm._compute_big_m


def _plant():
    """Two process options on a shared throughput x in [0, 100].

    Option A runs only at low throughput (x <= 10) and has output y = 2x; option B
    needs x >= 50 and has output y = x - 40. A's rows are relaxed while B runs, so
    their M is needed over B's box (x >= 50), and vice versa. The declared box
    ([0, 100] for both) gives M up to 100+ for both."""
    m = dm.Model("plant")
    x = m.continuous("x", lb=0, ub=100)
    y = m.continuous("y", lb=0, ub=200)
    m.either_or([[x <= 10, y == 2 * x], [x >= 50, y == x - 40]], name="opt")
    m.subject_to(y >= 15)
    m.minimize(x + 0.5 * y)
    return m


def _ms(monkeypatch, flag: str, build=_plant) -> list[float]:
    monkeypatch.setenv(FLAG, flag)
    seen: list[float] = []

    def spy(*a, **k):
        M = _REAL_COMPUTE_BIG_M(*a, **k)
        seen.append(M)
        return M

    monkeypatch.setattr(grm, "_compute_big_m", spy)
    grm.reformulate_gdp(build(), "big-m")
    assert seen, "no big-M was computed: the probe did not fire"
    return seen


def test_flag_off_is_todays_m(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    off = _ms(monkeypatch, "0")
    assert grm._DISJUNCT_LOCAL_CAPS.get() is None  # nothing leaks out of a lowering
    assert len(off) == 4


def test_flag_on_never_raises_m_and_shrinks_it_here(monkeypatch):
    off = _ms(monkeypatch, "0")
    on = _ms(monkeypatch, "1")
    assert len(on) == len(off)
    assert all(b <= a + 1e-12 for a, b in zip(off, on)), (off, on)
    # B's rows relaxed over A's box (x <= 10): 242.4 -> 53.0 and 50.5 -> 42.9.
    assert sum(b < 0.5 * a for a, b in zip(off, on)) >= 1, (off, on)
    assert sum(b < a - 1e-9 for a, b in zip(off, on)) >= 3, (off, on)
    assert grm._DISJUNCT_LOCAL_CAPS.get() is None


@pytest.mark.parametrize("method", ["big-m", "mbigm", "auto"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_differential_battery_with_the_flag(monkeypatch, shape, method):
    monkeypatch.setenv(FLAG, "1")
    rep = run_shape(shape, method)
    if rep.refused is not None:
        assert method in shape.refuses
        return
    assert rep.feasible_checked > 0 and rep.infeasible_checked > 0, rep
    rep.assert_sound()


def test_the_battery_saw_a_cap(monkeypatch):
    """The battery passes vacuously if no shape gets a cap."""
    monkeypatch.setenv(FLAG, "1")
    capped = [s.name for s in SHAPES if disjunct_local_big_m_caps(s.build())]
    assert len(capped) >= 5, capped


@pytest.mark.correctness
def test_same_certified_optimum_as_hull_and_a_root_bound_no_weaker(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hull = _plant().solve(gdp_method="hull", time_limit=60)
        root_off = _plant().solve(gdp_method="big-m", time_limit=60, max_nodes=1)
        monkeypatch.setenv(FLAG, "1")
        on = _plant().solve(gdp_method="big-m", time_limit=60)
        root_on = _plant().solve(gdp_method="big-m", time_limit=60, max_nodes=1)
    assert hull.status == on.status == "optimal"
    assert on.objective == pytest.approx(hull.objective, abs=1e-6)
    assert on.bound <= on.objective + 1e-9 and on.bound <= hull.objective + 1e-6
    assert root_on.bound >= root_off.bound - 1e-9


def test_a_reified_disjunction_takes_no_cap():
    from discopt.modeling.core import DisjunctionSemantics

    m = dm.Model("reified")
    x = m.continuous("x", lb=0, ub=100)
    m.either_or([[x <= 10], [x >= 50]], name="d", semantics=DisjunctionSemantics.EXACTLY_ONE_TRUE)
    m.minimize(x)
    assert disjunct_local_big_m_caps(m) == {}


def test_a_disjunct_fbbt_calls_infeasible_keeps_the_global_box(monkeypatch):
    """Sibling B is empty under the global rows (x <= 20 globally, B needs x >= 50).
    FBBT's verdict is not a proof, so A's rows are bounded over the global box,
    not dropped as if only an impossible sibling could relax them."""
    m = dm.Model("empty_sibling")
    x = m.continuous("x", lb=0, ub=100)
    y = m.continuous("y", lb=0, ub=200)
    m.subject_to(x <= 20)
    m.either_or([[y >= 3 * x], [x >= 50, y <= 1]], name="d")
    m.minimize(x - y)
    caps = disjunct_local_big_m_caps(m)
    a_row = next(c for c in m._constraints if hasattr(c, "disjuncts")).disjuncts[0][0]
    # Over the global box (x in [0, 20], y in [0, 200]): 3x - y <= 60.
    assert caps[id(a_row)] == pytest.approx(60 * 1.01, rel=1e-6)


def test_misaligned_fbbt_output_produces_no_cap(monkeypatch):
    import discopt.gdp.bounds as b

    monkeypatch.setattr(b, "_fbbt", lambda model: None)
    assert disjunct_local_big_m_caps(_plant()) == {}
