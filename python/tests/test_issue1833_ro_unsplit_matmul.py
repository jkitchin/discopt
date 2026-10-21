"""#1833: a vector ``@`` row reached the polyhedral counterpart unsplit.

``_split_vector_uncertain_rows`` let a body with no static shape through
whole, so ``x + M @ p <= 10`` got one set of duals for both rows and a feasible
model was reported infeasible (truth: per-row worst cases 6 and 1, x = 4). The
static shape analysis now covers matmul (#1831); a body whose shape it still
cannot determine is split by its evaluated shape, or refused.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, PolyhedralUncertaintySet, RobustCounterpart
from discopt.ro.counterpart import _split_vector_uncertain_rows

pytestmark = pytest.mark.smoke

M = np.array([[3.0, -1.0], [-2.0, 1.0]])
BOX_POLY = dict(A=np.vstack([np.eye(2), -np.eye(2)]), b=np.ones(4))
VERTICES = [np.ones(2) + np.array(s) for s in itertools.product((-1.0, 1.0), repeat=2)]


def _oracle(row):
    best = 20.0
    for v in VERTICES:
        best = min(best, float(np.min(10.0 - row(v))))
    return best


ROWS = {
    "M_at_p": (lambda x, p: x + M @ p, lambda v: M @ v),
    "p_at_MT": (lambda x, p: x + p @ M.T, lambda v: v @ M.T),
    "neg_M_at_p": (lambda x, p: x - M @ p, lambda v: -(M @ v)),
    "scaled_M_at_p": (lambda x, p: x + (M @ p) * 1.0, lambda v: M @ v),
}


@pytest.mark.parametrize("kind", ["polyhedral", "box"])
@pytest.mark.parametrize("name", sorted(ROWS))
def test_vector_matmul_row_matches_vertex_oracle(kind, name):
    row, f = ROWS[name]
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array([1.0, 1.0]))
    m.subject_to(row(x, p) <= 10)
    m.maximize(x)
    unc = (
        PolyhedralUncertaintySet(p, **BOX_POLY)
        if kind == "polyhedral"
        else BoxUncertaintySet(p, delta=np.ones(2))
    )
    RobustCounterpart(m, unc).formulate()
    r = m.solve()
    assert r.status == "optimal"
    xv = float(r.value(x))
    truth = _oracle(f)
    assert xv == pytest.approx(truth, abs=1e-4)
    for v in VERTICES:
        assert float(np.max(xv + f(v))) <= 10.0 + 1e-5


def test_unknown_static_shape_split_by_evaluation(monkeypatch):
    """With the static analysis disabled the row is still split (evaluated)."""
    import discopt.ro.formulations._common as common

    m = dm.Model("eval")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array([1.0, 1.0]))
    m.subject_to(x + M @ p <= 10)
    monkeypatch.setattr(common, "_shape_of", lambda e: None)
    rows = _split_vector_uncertain_rows(m._constraints, {"p"})
    assert len(rows) == 2


def test_unevaluable_shape_refused(monkeypatch):
    import discopt.ro.formulations._common as common

    m = dm.Model("refuse")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array([1.0, 1.0]))
    m.subject_to(x + M @ p <= 10)
    monkeypatch.setattr(common, "_shape_of", lambda e: None)

    def boom(e):
        raise NotImplementedError("no")

    monkeypatch.setattr(common, "eval_constant_expr", boom)
    with pytest.raises(NotImplementedError, match="#1833"):
        _split_vector_uncertain_rows(m._constraints, {"p"})
