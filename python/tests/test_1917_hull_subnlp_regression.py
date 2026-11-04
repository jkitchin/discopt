"""#1917: heuristic sub-NLPs under hull fail on fixings that have feasible points.

Under ``gdp_method="hull"`` a sub-NLP that fixes an indicator to 0 leaves that
disjunct's disaggregated copies on their whole box, held at 0 only by the linking
rows ``lb*y <= v <= ub*y``. The interior-point solve then approaches ``v = 0``
through eps-perspective rows evaluated at ``yhat ~ eps``. Measured on GDPlib
``syngas`` at bdc54a48, first root sub-NLP with its integers set to the big-M
incumbent's (known-feasible) configuration: infeasible at ITERATION_LIMIT after
300 iterations (4.5 s), and still after 3000 (41 s).

``DISCOPT_SUBNLP_FBBT_BOX`` (re-solve a failed sub-NLP on its node-FBBT box, with a
``Restoration_Failed`` constant-row retry) fixed the toy below and gained hull
incumbents on GDPlib hda and gdp_col. It was RETIRED in the PR that introduced it,
because its 120 s panel was not cert-clean; see the row in
``docs/dev/flag-retirement-audit.md``. These tests keep its fixtures as regression
cases: the defect is recorded as a strict xfail, so a future fix flips it, and the
retry's absence is pinned so node NLPs stay unchanged.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solvers.nlp_pounce as nlp_pounce  # noqa: E402
from discopt._relax import primal_heuristics as PH  # noqa: E402
from discopt._relax.gdp_reformulate import reformulate_gdp  # noqa: E402
from discopt.solver import (  # noqa: E402
    _BoundOverrideEvaluator,
    _extract_variable_info,
    _make_evaluator,
)
from discopt.solvers import SolveStatus  # noqa: E402

_RESTORATION_FAILED = -2


def _chain(n: int = 3):
    """``n`` two-term disjunctions, one with ``exp``, the other with ``log``."""
    m = dm.Model("hull_chain")
    xs = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(n)]
    zs = [m.continuous(f"z{i}", lb=-10, ub=10) for i in range(n)]
    for i in range(n):
        m.either_or(
            [
                [dm.exp(xs[i]) <= 3 + zs[i], xs[i] * xs[i] + zs[i] * zs[i] <= 20],
                [xs[i] >= 6, dm.log(1 + xs[i]) + zs[i] >= 2.5],
            ],
            name=f"d{i}",
        )
    m.minimize(sum((xs[i] - 5) ** 2 + zs[i] for i in range(n)))
    return m


def _configurations(low, n):
    """One start per one-hot configuration, continuous columns at their box midpoint."""
    lb, ub = PH._get_variable_bounds(low)
    names = []
    for v in low._variables:
        names += [v.name] * v.size
    ind = [i for i, nm in enumerate(names) if PH._get_integer_mask(low)[i]]
    assert len(ind) == 2 * n, names
    mid = np.where(np.isfinite(lb) & np.isfinite(ub), 0.5 * (lb + ub), 0.0)
    for pick in itertools.product((0, 1), repeat=n):
        x = mid.copy()
        for k in range(n):
            x[ind[2 * k]] = 1.0 - pick[k]
            x[ind[2 * k + 1]] = float(pick[k])
        yield x


def _feasible_count(n: int = 3) -> tuple[int, int]:
    low = reformulate_gdp(_chain(n), "hull")
    found = tried = 0
    for x in _configurations(low, n):
        tried += 1
        out = PH.subnlp(low, x)
        if out is not None:
            found += 1
            # whatever subnlp returns is verified against the original constraints
            assert np.all(np.isfinite(out[0]))
    return found, tried


@pytest.mark.xfail(
    strict=True,
    reason="#1917: hull sub-NLPs fail at yhat ~ eps; DISCOPT_SUBNLP_FBBT_BOX retired",
)
def test_every_hull_configuration_subnlp_finds_its_point():
    found, tried = _feasible_count()
    assert tried == 8
    assert found == tried


def test_hull_subnlp_fails_on_some_configurations():
    # Pins the measured defect: on the original box POUNCE stops at
    # Invalid_Number_Detected (``exp``/``log`` at yhat ~ eps) on these fixings.
    found, tried = _feasible_count()
    assert tried == 8
    assert found < tried


def _one_constant_row():
    """``x`` fixed at 1 by its box with one row on ``x`` alone; ``y`` free."""
    m = dm.Model("const_row")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((y - 2.0) * (y - 2.0) + x)
    m.subject_to(x == 1.0)
    m.subject_to(x + y <= 4.0)
    ev = _make_evaluator(m)
    _, lb, ub, _, _ = _extract_variable_info(m)
    lb, ub = lb.copy(), ub.copy()
    lb[0] = ub[0] = 1.0
    return _BoundOverrideEvaluator(ev, lb, ub), [(0.0, 0.0), (-np.inf, 0.0)]


def test_restoration_failure_is_not_retried(monkeypatch):
    # The retired flag's constant-row retry on Restoration_Failed is gone: a node
    # NLP that fails restoration is reported as failed, unchanged from before #1917.
    real = nlp_pounce._run_solve
    calls = []

    def fake(problem, x0, warm_start, solve_report=False):
        calls.append(1)
        if len(calls) == 1:
            return x0.copy(), {"status": _RESTORATION_FAILED, "obj_val": np.nan}, None
        return real(problem, x0, warm_start, solve_report)

    monkeypatch.setattr(nlp_pounce, "_run_solve", fake)
    ev, bounds = _one_constant_row()
    r = nlp_pounce.solve_nlp(ev, np.array([1.0, 1.0]), constraint_bounds=bounds)
    assert len(calls) == 1
    assert r.raw_status == _RESTORATION_FAILED and r.status == SolveStatus.ERROR
