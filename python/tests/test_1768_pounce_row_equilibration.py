"""``solver="pounce"`` row-equilibrates its NLP arm (#1768).

POUNCE's gradient-based scaling only scales a row down, so a row with a tiny
gradient reaches the interior-point method as written. On the COPS hanging chain
the link rows ``dx**2 + dy**2 <= L**2`` have gradients ``~2L`` and multipliers
``~1/L``; the multipliers inflate POUNCE's dual scaling factor until its scaled
convergence test passes with the barrier parameter still at ``1.8e-6``. It
reported ``Solve_Succeeded`` -- ``local_optimal`` -- with every link slack.
Scaling each row up to unit gradient at the start point removes that.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
import numpy as np
import pytest

pytestmark = [pytest.mark.requires_pounce]


def _chain(n: int):
    L = 1.5 / n
    m = dm.Model("chain")
    x = m.continuous("x", shape=(n + 1,), lb=-10, ub=10)
    y = m.continuous("y", shape=(n + 1,), lb=-10, ub=10)
    m.subject_to([x[0] == 0, y[0] == 0, x[n] == 1, y[n] == 0], name="ends")
    m.subject_to((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 <= L**2, name="link")
    m.minimize(0.5 * dm.sum(y[1:] + y[:-1]))
    return m, x, y, L


def _tiny_row():
    """``min (x-2)^2 + (y-2)^2`` on the disc ``x^2 + y^2 <= 2``, written with 1e-6 rows.

    Optimum ``(1, 1)``; the row's multiplier is ``1e6`` in the model's units. Solve it
    from ``_START``: the scales are taken from the row gradients at the start point,
    and at the default start ``(0, 0)`` the gradient vanishes (no scaling).
    """
    m = dm.Model("tiny")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((x - 2) ** 2 + (y - 2) ** 2)
    m.subject_to(1e-6 * (x**2 + y**2) <= 2e-6, name="disc")
    return m, x, y


def _start(x, y):
    return {x: 0.5, y: 0.5}


@pytest.fixture
def spy(monkeypatch):
    """Record the evaluator and warm start each POUNCE NLP solve receives."""
    calls: list[dict] = []
    real = nlp_pounce.solve_nlp

    def wrapped(evaluator, x0, **kw):
        calls.append({"evaluator": evaluator, "warm_start": kw.get("warm_start")})
        return real(evaluator, x0, **kw)

    monkeypatch.setattr(nlp_pounce, "solve_nlp", wrapped)
    return calls


def _equilibrated(call) -> bool:
    from discopt.solvers.oa import _RowEquilibratedEvaluator

    return isinstance(call["evaluator"], _RowEquilibratedEvaluator)


def test_chain_with_tiny_link_rows_converges(spy, monkeypatch):
    """n = 20,000 (L**2 = 5.6e-9). Before: ``local_optimal`` at -6053.7496 with every
    link 2e-6..6e-6 short, 19.6 s. After: -6053.78746, links within 1e-8, 3.8 s."""
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", raising=False)
    n = 20_000
    m, x, y, L = _chain(n)
    r = m.solve(solver="pounce", initial_solution={x: np.linspace(0, 1, n + 1), y: np.zeros(n + 1)})
    assert len(spy) == 1 and _equilibrated(spy[0])
    assert r.status == "local_optimal"
    seg = np.hypot(np.diff(r.value(x)), np.diff(r.value(y)))
    assert np.max(np.abs(seg / L - 1)) < 1e-7, (seg.min() / L - 1, seg.max() / L - 1)
    assert r.objective < -6053.787


def test_duals_are_reported_in_the_models_units(spy, monkeypatch):
    m, x, y = _tiny_row()
    monkeypatch.setenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", "0")
    ref = m.solve(solver="pounce", initial_solution=_start(x, y))
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE")
    r = m.solve(solver="pounce", initial_solution=_start(x, y))
    assert [_equilibrated(c) for c in spy] == [False, True]
    for res in (ref, r):
        assert res.status == "local_optimal"
        assert res.value(x) == pytest.approx(1.0, abs=1e-6)
        assert res.value(y) == pytest.approx(1.0, abs=1e-6)
    lam_ref = np.asarray(ref.constraint_duals["disc"], dtype=float)
    lam = np.asarray(r.constraint_duals["disc"], dtype=float)
    assert np.abs(lam_ref).max() == pytest.approx(1e6, rel=1e-4)
    np.testing.assert_allclose(lam, lam_ref, rtol=1e-4)


def test_warm_start_multipliers_are_scaled_into_the_rows_pounce_sees(spy, monkeypatch):
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", raising=False)
    m, x, y = _tiny_row()
    first = m.solve(solver="pounce", initial_solution=_start(x, y))
    second = m.solve(solver="pounce", warm_start=first)
    assert len(spy) == 2 and all(_equilibrated(c) for c in spy)
    ws = spy[1]["warm_start"]
    assert ws is not None and ws.lagrange is not None
    s = spy[1]["evaluator"]._s
    # POUNCE sees row ``s * g``, whose multiplier is ``lam / s``.
    lam_model = np.asarray(first.constraint_duals["disc"], dtype=float).ravel()
    np.testing.assert_allclose(np.asarray(ws.lagrange) * s, lam_model, rtol=1e-12)
    assert second.status == "local_optimal"
    assert second.objective == pytest.approx(first.objective, rel=1e-8, abs=1e-10)


def test_explicit_scaling_method_and_opt_out_leave_the_rows_alone(spy, monkeypatch):
    m, x, y = _tiny_row()
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", raising=False)
    m.solve(solver="pounce", initial_solution=_start(x, y))
    m.solve(
        solver="pounce",
        initial_solution=_start(x, y),
        pounce_options={"nlp_scaling_method": "none"},
    )
    monkeypatch.setenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", "0")
    m.solve(solver="pounce", initial_solution=_start(x, y))
    # The first solve is the control: the same start does trigger the scaling.
    assert [_equilibrated(c) for c in spy] == [True, False, False]


def test_sparse_row_norms_match_the_dense_jacobian():
    """The scales are read through the sparse structure (a dense 100k-link Jacobian is
    149 GiB); they must equal the dense ones exactly, so OA (#1537) is unchanged."""
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.oa import _jacobian_row_inf_norms

    m, x, y, _ = _chain(7)
    ev = make_evaluator(m)
    assert ev.has_sparse_structure()
    rng = np.random.default_rng(0)
    for _ in range(3):
        x0 = rng.uniform(-1, 1, ev.n_variables)
        dense = np.abs(np.asarray(ev.evaluate_jacobian(x0))).max(axis=1)
        sparse = _jacobian_row_inf_norms(ev, x0, ev.n_constraints)
        assert sparse is not None
        np.testing.assert_array_equal(sparse, dense)
