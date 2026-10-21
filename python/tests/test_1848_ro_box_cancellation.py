"""#1848: box RO worst-cased cancelling non-affine terms independently.

``x + exp(p) - exp(p) <= 0.5`` over ``p in [0, 2]`` was reported **infeasible**
(each ``exp(p)`` charged its own worst case: ``exp(2) - exp(0)``), a false
certificate; ``<= 10`` gave 3.61 for a truth of 10. Two changes:

* like terms are cancelled before robustifying (``cancel_like_terms``), so the
  issue's rows are robustified exactly;
* where an uncertain component still occurs more than once and sign tracking's
  bound is not attained at a box vertex, the counterpart is only a conservative
  restriction: ``Model.solve`` then withholds the certificate (``optimal`` ->
  ``feasible``, ``infeasible`` -> ``unknown``).

Oracle: a dense grid over the box (exact enough for these scalar rows).
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, RobustCounterpart
from discopt.ro.formulations._common import CONSERVATIVE_ATTR, cancel_like_terms

pytestmark = pytest.mark.smoke


def _robust_solve(row, rhs, sense="<=", pbar=1.0, delta=1.0):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array(pbar))
    body = row(x, p, dm)
    m.subject_to(body <= rhs if sense == "<=" else body == rhs)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=delta)).formulate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m, m.solve(time_limit=60)


ISSUE_ROWS = [
    (lambda x, p, F: x + F.exp(p) - F.exp(p), 10.0, "<=", 10.0),
    (lambda x, p, F: x + p * p - p * p, 10.0, "<=", 10.0),
    (lambda x, p, F: x + F.exp(p) - F.exp(p), 0.5, "<=", 0.5),
    (lambda x, p, F: x + F.exp(p) - F.exp(p), 0.5, "==", 0.5),
    (lambda x, p, F: x + 3 * p - 3 * p + p * p, 6.0, "<=", 2.0),  # 6 - max p^2
]


@pytest.mark.parametrize("case", range(len(ISSUE_ROWS)))
def test_issue_rows_are_exact(case):
    row, rhs, sense, truth = ISSUE_ROWS[case]
    m, r = _robust_solve(row, rhs, sense)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)
    assert not getattr(m, CONSERVATIVE_ATTR, None)


def test_conservative_row_is_not_certified():
    # exp(p) - exp(p/2) does not cancel; per-occurrence worst cases are
    # exp(2) - exp(0) = 6.39 against a true max of exp(2) - exp(1) = 4.67.
    m, r = _robust_solve(lambda x, p, F: x + F.exp(p) - F.exp(p / 2), 10.0)
    assert getattr(m, CONSERVATIVE_ATTR, None) == ["_constraints[0]"]
    assert r.status == "feasible"
    assert r.gap_certified is False and r.bound is None
    assert r.objective <= 10 - (np.exp(2) - np.exp(1)) + 1e-6  # still robust


def test_conservative_infeasible_is_not_certified():
    # Robust truth: x <= 4.8 - 4.67 = 0.13 is feasible; the restriction is not.
    m, r = _robust_solve(lambda x, p, F: x + F.exp(p) - F.exp(p / 2), 4.8)
    assert r.status != "infeasible"


def test_repeated_but_attained_stays_certified():
    # |p| - 2p over [-1, 1]: both occurrences ask for p = -1, a vertex.
    m, r = _robust_solve(lambda x, p, F: x + F.abs(p) - 2 * p, 6.0, pbar=0.0)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(3.0, abs=1e-6)
    assert not getattr(m, CONSERVATIVE_ATTR, None)


def test_cancel_like_terms_is_the_same_function():
    from discopt.modeling.core import Constant
    from discopt.ro.affine_policy import _substitute_var
    from discopt.ro.formulations._common import eval_constant_expr, substitute_param

    m = dm.Model()
    x = m.continuous("x", lb=-1, ub=1)
    p = m.parameter("p", value=np.array(0.3))
    e = x + 2 * dm.exp(p) - (dm.exp(p) + dm.exp(p)) + 3 * p - p / 0.5 + 4.0
    c = cancel_like_terms(e)
    assert c is not e
    assert "exp" not in repr(c)
    evaluated = 0
    for xv in (-1.0, 0.2, 1.0):
        for pv in (-2.0, 0.3):

            def at(expr):
                expr = substitute_param(expr, "p", Constant(np.array(pv)))
                return float(eval_constant_expr(_substitute_var(expr, x, Constant(np.array(xv)))))

            assert at(c) == pytest.approx(at(e), abs=1e-12)
            evaluated += 1
    assert evaluated == 6
    assert cancel_like_terms(x + p) is not None


TERMS = {
    "exp": (lambda p, F: F.exp(p), np.exp),
    "sq": (lambda p, F: p * p, lambda v: v * v),
    "abs": (lambda p, F: F.abs(p), np.abs),
    "lin": (lambda p, F: p, lambda v: v),
    "cube": (lambda p, F: p**3, lambda v: v**3),
}


def test_random_battery_against_grid_oracle():
    rng = np.random.default_rng(1848)
    names = sorted(TERMS)
    checked = certified = 0
    for _ in range(30):
        k = int(rng.integers(2, 5))
        picks = [names[int(i)] for i in rng.integers(0, len(names), size=k)]
        coefs = [float(c) for c in rng.choice([-3.0, -1.0, 1.0, 2.0], size=k)]
        if rng.random() < 0.5:  # force an exact cancellation pair
            picks.append(picks[0])
            coefs.append(-coefs[0])
        pbar, delta = float(rng.uniform(-1, 1)), float(rng.uniform(0.2, 1.0))
        rhs = float(rng.uniform(0, 12))

        def row(x, p, F, picks=picks, coefs=coefs):
            out = x
            for nm, c in zip(picks, coefs):
                out = out + c * TERMS[nm][0](p, F)
            return out

        grid = np.linspace(pbar - delta, pbar + delta, 4001)
        h = sum(c * TERMS[nm][1](grid) for nm, c in zip(picks, coefs))
        x_true = rhs - float(np.max(h))  # grid max <= true max
        m, r = _robust_solve(row, rhs, pbar=pbar, delta=delta)
        if r.status == "infeasible":
            assert x_true < -1e-6, (picks, coefs, pbar, delta, rhs)
        if r.x is not None:
            xv = float(r.x["x"])
            assert xv <= min(20.0, x_true) + 1e-6  # robust on the grid
        if r.status == "optimal":
            certified += 1
            assert r.objective == pytest.approx(min(20.0, x_true), abs=2e-3), (picks, coefs)
        checked += 1
    assert checked == 30 and certified > 0
