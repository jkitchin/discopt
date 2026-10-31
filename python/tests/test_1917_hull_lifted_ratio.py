"""#1917 ``DISCOPT_HULL_LIFTED_RATIO``: the eps-perspective ratio ``v / yhat`` as a
bounded variable ``t`` (``v - yhat * t = 0``, ``t`` in ``v``'s own box).

Bound-changing, so default OFF (CLAUDE.md §5). What is pinned here:

* flag OFF: today's lowering (a division by ``yhat`` in every nonlinear hull row, no
  ``t``);
* flag ON: no division in any hulled row, one ``t`` per (disjunct, variable) used by
  a nonlinear row, each with its ``v``'s bounds and one linking row;
* the differential battery (no feasible point cut, no infeasible point admitted)
  on every shape under hull with the flag ON;
* the motivating failure: on Trespalacios' Example 6.3.3 (``ex_633_trespalacios``,
  the Pyomo GDPopt test model) the hull node NLPs stop on
  ``Invalid_Number_Detected`` with the flag OFF -- at iterates where ``v / yhat``
  is up to ~1e8 times ``v``'s bound -- and never with it ON, and both arms certify
  the published optimum 4.46.
"""

from __future__ import annotations

import collections

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import _collect_variables, _is_linear, reformulate_gdp
from gdp_differential import run_shape
from test_gdp_differential_battery import SHAPES

FLAG = "DISCOPT_HULL_LIFTED_RATIO"
#: Trespalacios (2015) Ex. 6.3.3 optimum, as stated in Pyomo's model docstring and
#: certified by the #1775 GAMS oracle (``gdplib_runner.panel_oracle``).
EX633_OPT = 4.460367612328062


def _ex633() -> dm.Model:
    """``pyomo/examples/gdp/small_lit/ex_633_trespalacios.py`` in discopt's API."""
    m = dm.Model("ex_633_trespalacios")
    x1 = m.continuous("x1", lb=0, ub=5)
    x2 = m.continuous("x2", lb=0, ub=3)
    m.minimize(5 + 0.2 * x1 - x2)
    m.either_or(
        [
            [
                x2 <= 0.4 * dm.exp(x1 / 2.0),
                x2 <= 0.5 * (x1 - 2.5) ** 2 + 0.3,
                x2 <= 6.5 / (x1 / 0.3 + 2.0) + 1.0,
            ],
            [
                x2 <= 0.3 * dm.exp(x1 / 1.8),
                x2 <= 0.7 * (x1 / 1.2 - 2.1) ** 2 + 0.3,
                x2 <= 6.5 / (x1 / 0.8 + 1.1),
            ],
        ],
        name="d1",
    )
    m.either_or(
        [
            [
                x2 <= 0.9 * dm.exp(x1 / 2.1),
                x2 <= 1.3 * (x1 / 1.5 - 1.8) ** 2 + 0.3,
                x2 <= 6.5 / (x1 / 0.8 + 1.1),
            ],
            [
                x2 <= 0.4 * dm.exp(x1 / 1.5),
                x2 <= 1.2 * (x1 - 2.5) ** 2 + 0.3,
                x2 <= 6.0 / (x1 / 0.6 + 1.0) + 0.5,
            ],
        ],
        name="d2",
    )
    return m


def _hulled_rows(low):
    return [c for c in low._constraints if c.name and "_d" in c.name and "_c" in c.name]


def _ratio_vars(low):
    return [v for v in low._variables if "_t_" in v.name]


def test_flag_off_is_todays_lowering(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    low = reformulate_gdp(_ex633(), "hull")
    rows = [c for c in _hulled_rows(low) if not _is_linear(c.body)]
    assert len(rows) == 12, [c.name for c in rows]
    assert all("/" in str(c.body) for c in rows)
    assert not _ratio_vars(low)


def test_flag_on_lifts_every_ratio(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    low = reformulate_gdp(_ex633(), "hull")
    rows = [c for c in _hulled_rows(low) if not _is_linear(c.body)]
    assert len(rows) == 12, [c.name for c in rows]
    # g is evaluated at t: no 1/yhat left anywhere in a hulled row (the model's own
    # divisions, 6.5 / (t / 0.3 + 2), remain).
    for c in rows:
        names = _collect_variables(c.body)
        assert not any("_v_" in n for n in names), (c.name, names)
        assert any("_t_" in n for n in names), (c.name, names)
    # 2 disjunctions x 2 disjuncts x {x1, x2}
    ts = _ratio_vars(low)
    assert len(ts) == 8, [t.name for t in ts]
    by_name = {v.name: v for v in low._variables}
    links = [c for c in low._constraints if c.name and c.name.startswith("_hull_ratio_")]
    assert len(links) == 8 and all(c.sense == "==" for c in links)
    for t in ts:
        v = by_name[t.name.replace("_t_", "_v_")]
        np.testing.assert_array_equal(t.lb, v.lb)
        np.testing.assert_array_equal(t.ub, v.ub)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_differential_battery_under_hull_with_the_flag(monkeypatch, shape):
    monkeypatch.setenv(FLAG, "1")
    rep = run_shape(shape, "hull")
    if rep.refused is not None:
        assert "hull" in shape.refuses
        return
    assert rep.feasible_checked > 0 and rep.infeasible_checked > 0, rep
    rep.assert_sound()


def test_the_battery_actually_saw_a_lifted_ratio(monkeypatch):
    """The battery would pass vacuously if no shape had a nonlinear disjunct row."""
    monkeypatch.setenv(FLAG, "1")
    lifted = [s.name for s in SHAPES if _ratio_vars(_lower_or_none(s))]
    assert len(lifted) >= 3, lifted


def _lower_or_none(shape):
    try:
        return reformulate_gdp(shape.build(), "hull")
    except Exception:  # noqa: BLE001 - shapes that declare a hull refusal
        assert "hull" in shape.refuses, shape.name
        return dm.Model("refused")


def _solve_counting_nlp_statuses(monkeypatch, flag: str, **kw):
    from discopt.solvers import nlp_pounce

    monkeypatch.setenv(FLAG, flag)
    counts: collections.Counter = collections.Counter()
    orig = nlp_pounce._run_solve

    def wrapped(problem, x0, warm_start, solve_report=False):
        x, info, rep = orig(problem, x0, warm_start, solve_report)
        counts[str(info.get("status_msg"))] += 1
        return x, info, rep

    monkeypatch.setattr(nlp_pounce, "_run_solve", wrapped)
    res = _ex633().solve(gdp_method="hull", **kw)
    assert sum(counts.values()) > 0, "no node NLP was counted: the probe did not fire"
    return res, counts


#: Node-NLP exits that mean the local solve broke down, not that the node is empty.
_BROKEN = (
    "Invalid_Number_Detected",
    "Maximum_Iterations_Exceeded",
    "Restoration_Failed",
)
# ``Maximum_WallTime_Exceeded`` is left out: a node NLP's wall slice is a budget that
# machine load can exhaust, not a breakdown of the local solve.


@pytest.mark.slow
@pytest.mark.correctness
def test_ex633_hull_certifies_with_no_broken_node_nlp(monkeypatch):
    """Without the lifting (measured on main ``4517c7b``) this model, hulled, does
    not certify in 120 s: 105 nodes, bound 4.4256, and 43 of 90 node NLPs broken
    (39 iteration limits, 3 wall limits, 1 restoration failure), every one at an
    iterate where ``v / yhat`` leaves ``v``'s box. The Pyomo-converted twin
    (``pyomo.ex_633_trespalacios``) breaks the same way as ``Invalid_Number_Detected``
    on 165 of 192. Lifted: certified in ~21 nodes with no broken NLP."""
    res, counts = _solve_counting_nlp_statuses(monkeypatch, "1", time_limit=60)
    assert not any(counts[k] for k in _BROKEN), counts
    assert res.status == "optimal", (res.status, counts)
    assert res.objective == pytest.approx(EX633_OPT, abs=1e-6, rel=1e-4)
    assert res.bound <= res.objective + 1e-9
    assert res.bound <= EX633_OPT + 1e-6
