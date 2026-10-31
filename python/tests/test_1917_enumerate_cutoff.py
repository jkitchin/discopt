"""#1917 G4.2 ``DISCOPT_GDP_ENUMERATE_CUTOFF``: enumerate bounds each configuration
against the incumbent (row ``f <= c``, ``c = UB - delta``).

Bound-changing on ``gdp_method="enumerate"``, so default OFF (CLAUDE.md §5). Pinned:

* flag OFF: no cutoff row, no configuration closed by it (today's route);
* flag ON: the same certified optimum on every G4.2 model, min and max, with a bound
  that never passes the optimum; on a model built so that most configurations are
  dominated, the cutoff actually closes them (the probe fires);
* the bound rule: a configuration's bound obtained under the cutoff is recorded as
  ``min(bound, c)`` -- an uncertified restricted solve whose bound overshoots says
  nothing above ``c`` -- and a certified-empty restriction records ``c``.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
import test_1775_g4_2_ric_enumerate as g42
from discopt.gdp.algorithms.enumerate import Enumerate
from discopt.gdp.algorithms.subproblem import SubproblemResult

FLAG = "DISCOPT_GDP_ENUMERATE_CUTOFF"


def _chain(sense: str = "min"):
    """Three independent two-way choices over a shared budget: 8 configurations,
    each a small LP. ``x_i`` costs 1 per unit in the cheap disjunct and 5 in the
    other; ``sum x >= 3`` with ``x_i <= 1`` in the cheap one. Optimum 3 (all cheap)."""
    m = dm.Model(f"chain_{sense}")
    xs = [m.continuous(f"x{i}", lb=0, ub=3) for i in range(3)]
    cs = [m.continuous(f"c{i}", lb=0, ub=15) for i in range(3)]
    for i, (x, c) in enumerate(zip(xs, cs)):
        m.either_or([[c >= x, x <= 1], [c >= 5 * x]], name=f"d{i}")
    m.subject_to(xs[0] + xs[1] + xs[2] >= 3)
    total = cs[0] + cs[1] + cs[2]
    if sense == "min":
        m.minimize(total)
    else:
        m.maximize(-total)
    return m


def _run(mk, **kw):
    return g42._quiet(Enumerate(mk(), time_limit=60, **kw).run)


def test_flag_off_is_todays_route(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    r = _run(_chain)
    assert r.status == "optimal" and r.solver_stats["n_cutoff_dominated"] == 0


@pytest.mark.parametrize("sense", ["min", "max"])
def test_the_cutoff_closes_dominated_configurations(monkeypatch, sense):
    monkeypatch.setenv(FLAG, "0")
    off = _run(lambda: _chain(sense))
    monkeypatch.setenv(FLAG, "1")
    on = _run(lambda: _chain(sense))
    want = 3.0 if sense == "min" else -3.0
    for r in (off, on):
        assert r.status == "optimal" and r.gap_certified
        assert r.objective == pytest.approx(want, abs=1e-6)
    assert off.solver_stats["n_cutoff_dominated"] == 0
    assert on.solver_stats["n_cutoff_dominated"] >= 1, on.solver_stats  # the probe fired
    if sense == "min":
        assert on.bound <= want + 1e-9 and on.bound >= want - 1e-3
    else:
        assert on.bound >= want - 1e-9 and on.bound <= want + 1e-3


@pytest.mark.parametrize(
    "mk", [g42._nonconvex, g42._or_model, g42._nested, g42._nonlinear_objective]
)
def test_same_certified_optimum_as_big_m(monkeypatch, mk):
    monkeypatch.setenv(FLAG, "1")
    ref = g42._quiet(mk().solve, gdp_method="big-m", time_limit=60)
    r = g42._quiet(mk().solve, gdp_method="enumerate", time_limit=60)
    assert ref.status == "optimal" and r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)
    assert r.bound <= r.objective + 1e-9
    assert r.bound <= ref.objective + 1e-6


def test_maximize_bound_stays_on_its_side(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    r = g42._quiet(g42._nonlinear_objective("max").solve, gdp_method="enumerate", time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(-0.5625, abs=1e-5)
    assert r.bound >= r.objective - 1e-9


def test_no_proof_under_the_cutoff_falls_back_to_the_plain_solve(monkeypatch):
    """A restricted solve that proves nothing -- here every one returns
    ``unresolved`` with an overshooting bound of 1e9, which says nothing when the
    restriction is empty -- is never recorded: the configuration is re-solved
    without the cutoff, and the run certifies the true optimum as with the flag
    OFF. (Measured trigger: ``_nested``'s two configurations tie at 0.25, so the
    restriction is infeasible by only ``delta`` and the solve returned a point
    violating the cutoff row by 6.5e-6 with no bound.)"""
    import discopt.gdp.algorithms.subproblem as sp

    monkeypatch.setenv(FLAG, "1")
    real = sp.solve_subproblem
    cutoffs: list[float] = []

    def overshoot(*a, **k):
        if k.get("cutoff") is None:
            return real(*a, **k)
        cutoffs.append(k["cutoff"])
        return SubproblemResult("unresolved", bound=1e9, reason="test")

    monkeypatch.setattr(sp, "solve_subproblem", overshoot)
    alg = Enumerate(_chain(), time_limit=60)
    r = g42._quiet(alg.run)
    assert cutoffs, "no configuration was solved under a cutoff: the probe did not fire"
    assert r.solver_stats["n_cutoff_retried"] == len(cutoffs)
    assert r.solver_stats["n_cutoff_dominated"] == 0
    assert r.status == "optimal" and r.objective == pytest.approx(3.0, abs=1e-6)
    assert all(b is not None and b < 1e9 for b in alg.config_bounds)
    assert r.bound <= 3.0 + 1e-9


def test_a_tie_with_the_incumbent_still_certifies(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    r = g42._quiet(g42._nested().solve, gdp_method="enumerate", time_limit=60)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(0.25, abs=1e-6)
    assert r.bound <= r.objective + 1e-9
