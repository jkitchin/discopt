"""Tests for the batched POUNCE node solver (`_solve_batch_pounce`).

Phase A of discopt#97: POUNCE's ``solve_nlp_batch`` (pounce#126) solves a batch
of B&B node NLP relaxations in parallel. These tests validate:

  - The batch path returns results bit-equivalent to serial per-node POUNCE
    solves (same objective, same primal, same infeasible handling).
  - End-to-end, a wide-tree MINLP solved with ``nlp_solver="pounce"`` actually
    exercises the batch path (``n_batch > 1``) and reaches the correct optimum.

Skipped entirely when POUNCE is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402
from discopt.solver import (  # noqa: E402
    _make_evaluator,
    _solve_batch_pounce,
    _solve_node_nlp_pounce,
)
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds  # noqa: E402

_SENTINEL = 1e29  # |result_lb| past this == infeasible / failed node


def _build_eval_and_boxes():
    """A small constrained NLP plus several feasible node bound-boxes."""
    m = dm.Model("batch_test")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2 + x * y)
    m.subject_to(x + y >= 1)

    ev = _make_evaluator(m)
    lb, ub = ev.variable_bounds
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    cl, cu = _infer_constraint_bounds(ev)
    cb = list(zip(cl.tolist(), cu.tolist()))

    boxes = []
    for k in range(6):
        nlb, nub = lb.copy(), ub.copy()
        nub[0] = 5 - 0.3 * k  # feasible tightening of x upper bound
        nlb[1] = -5 + 0.2 * k  # feasible tightening of y lower bound
        boxes.append((nlb, nub))
    return ev, cb, boxes


def _serial_lb(res):
    if res.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT) and np.isfinite(
        res.objective
    ):
        return float(res.objective)
    return None


@pytest.mark.correctness
def test_batch_pounce_matches_serial_feasible():
    """Batched POUNCE objectives/primals equal serial per-node POUNCE solves."""
    ev, cb, boxes = _build_eval_and_boxes()
    n = ev.n_variables
    batch_lb = [b[0].tolist() for b in boxes]
    batch_ub = [b[1].tolist() for b in boxes]
    opts = {"max_iter": 200, "max_wall_time": 30.0}

    rid, rlb, rsol, rfeas, _ = _solve_batch_pounce(
        ev, batch_lb, batch_ub, list(range(len(boxes))), n, cb, opts
    )

    # Contract: shapes, ids, and the all-False feasibility flags (Rust checks
    # integrality, not this function).
    assert rid.tolist() == list(range(len(boxes)))
    assert rsol.shape == (len(boxes), n)
    assert not rfeas.any()

    for i, (nlb, nub) in enumerate(boxes):
        x0 = 0.5 * (np.clip(nlb, -1e3, 1e3) + np.clip(nub, -1e3, 1e3))
        res = _solve_node_nlp_pounce(ev, x0, nlb, nub, cb, opts)
        serial = _serial_lb(res)
        batch = rlb[i] if abs(rlb[i]) < _SENTINEL else None

        assert (serial is None) == (batch is None), f"node {i} feasibility disagrees"
        if serial is not None:
            assert batch == pytest.approx(serial, abs=1e-4, rel=1e-4)
            assert np.allclose(rsol[i], res.x, atol=1e-4)


@pytest.mark.correctness
def test_batch_pounce_multistart_matches_single_start_on_convex_node():
    """Multistart picks the best start; on these convex nodes it ties single-start.

    Each node here is a convex QP, so every start converges to the same unique
    optimum — multistart must not change the objective, only (potentially) the
    work done. This pins the (node, start) flattening + per-node argmin reduce.
    """
    ev, cb, boxes = _build_eval_and_boxes()
    n = ev.n_variables
    batch_lb = [b[0].tolist() for b in boxes]
    batch_ub = [b[1].tolist() for b in boxes]
    opts = {"max_iter": 200, "max_wall_time": 30.0}

    _, single, _, _, _ = _solve_batch_pounce(
        ev, batch_lb, batch_ub, list(range(len(boxes))), n, cb, opts
    )
    # Force the multistart path (convex=False) even though the nodes are convex.
    _, multi, msol, _, _ = _solve_batch_pounce(
        ev,
        batch_lb,
        batch_ub,
        list(range(len(boxes))),
        n,
        cb,
        opts,
        multistart=True,
        convex=False,
    )

    assert msol.shape == (len(boxes), n)
    for i in range(len(boxes)):
        s = single[i] if abs(single[i]) < _SENTINEL else None
        mlt = multi[i] if abs(multi[i]) < _SENTINEL else None
        assert (s is None) == (mlt is None)
        if s is not None:
            # Multistart keeps the best (lowest) objective: never worse.
            assert mlt <= s + 1e-6
            assert mlt == pytest.approx(s, abs=1e-4, rel=1e-4)


@pytest.mark.correctness
def test_batch_pounce_marks_infeasible_nodes():
    """Infeasible node boxes come back at the infeasibility sentinel."""
    ev, cb, _ = _build_eval_and_boxes()
    n = ev.n_variables
    # x + y >= 1 is impossible when both are pinned near -5.
    nlb = np.array([-5.0, -5.0])
    nub = np.array([-4.9, -4.9])
    rid, rlb, rsol, rfeas, _ = _solve_batch_pounce(
        ev,
        [nlb.tolist(), nlb.tolist()],
        [nub.tolist(), nub.tolist()],
        [0, 1],
        n,
        cb,
        {"max_iter": 200, "max_wall_time": 30.0},
    )
    assert np.all(np.abs(rlb) >= _SENTINEL)


@pytest.mark.correctness
@pytest.mark.slow
def test_pounce_batch_end_to_end_pooling(monkeypatch):
    """A wide-tree nonconvex MINLP exercises the batch path and is correct.

    Pooling-Haverly is below the default ``_POUNCE_BATCH_MIN_VARS`` size gate
    (n_vars=6), so the threshold is lowered here to drive the integration; the
    gate's own behavior is covered by ``test_pounce_size_gate_keeps_small_serial``.
    """
    from discopt.modeling.examples import example_pooling_haverly

    monkeypatch.setattr(S, "_POUNCE_BATCH_MIN_VARS", 0)

    sizes: list[int] = []
    orig = S._solve_batch_pounce

    def spy(*a, **k):
        r = orig(*a, **k)
        sizes.append(len(r[0]))
        return r

    monkeypatch.setattr(S, "_solve_batch_pounce", spy)
    model = example_pooling_haverly()
    res = model.solve(nlp_solver="pounce", batch_size=16, time_limit=120)

    # Pooling-Haverly is nonconvex: the McCormick relaxation bounds do not
    # rigorously close the gap, and spatial-branch nodes whose NLP relaxation
    # fails carry no rigorous lower bound. Per the #27a soundness guard
    # (applied on the serial path and, since the batch-sentinel fix, on the
    # batch path too), such a search reports the correct incumbent as
    # "feasible" rather than claiming a certificate it does not hold. Every
    # solve path (pounce/ipm × batch/serial) agrees on "feasible" here.
    assert res.status in ("optimal", "feasible")
    # Haverly pooling global optimum (this formulation): -400 profit => 1390 obj.
    assert res.objective == pytest.approx(1390.0, abs=1.0, rel=1e-3)
    # The batch path must have actually fired with more than one node.
    assert sizes, "pounce batch path never triggered"
    assert max(sizes) > 1, f"batch never exceeded one node: {sizes}"


@pytest.mark.correctness
def test_pounce_size_gate_keeps_small_serial():
    """Below the size gate, a small problem must NOT use the batch path.

    The callback batch path is GIL-bound and only amortizes on larger node
    problems, so small models stay on the serial per-node path by default.
    """
    from discopt.modeling.examples import example_pooling_haverly

    assert S._POUNCE_BATCH_MIN_VARS > 6  # pooling has 6 variables

    fired = []
    orig = S._solve_batch_pounce

    def spy(*a, **k):
        fired.append(1)
        return orig(*a, **k)

    S._solve_batch_pounce = spy
    try:
        model = example_pooling_haverly()
        res = model.solve(nlp_solver="pounce", batch_size=16, time_limit=120)
    finally:
        S._solve_batch_pounce = orig

    assert res.objective == pytest.approx(1390.0, abs=1.0, rel=1e-3)  # still correct
    assert not fired, "small problem should stay serial under the size gate"


# --- Batched MIQP node-QP waves (solve_qp_batch) ---------------------------


def _miqp(seed: int, n: int = 10, ub: int = 6) -> dm.Model:
    """A convex integer-quadratic with a couple of coupling constraints."""
    rng = np.random.default_rng(seed)
    m = dm.Model(f"miqp_{seed}")
    x = [m.integer(f"x{i}", lb=0, ub=ub) for i in range(n)]
    t = rng.uniform(0.5, ub - 0.5, size=n)
    a = rng.uniform(1, 3, size=n)
    m.subject_to(sum(float(a[i]) * x[i] for i in range(n)) <= float(a.sum() * (ub * 0.45)))
    m.subject_to(x[0] + x[1] + x[2] >= 4)
    m.minimize(sum((x[i] - float(t[i])) ** 2 for i in range(n)))
    return m


def test_miqp_batch_qp_matches_serial_fallback(monkeypatch):
    """The batched ``solve_qp_batch`` MIQP node path matches the serial
    callback fallback bit-for-bit (objective + B&B node count), and the
    batch path is actually exercised (not silently falling back)."""
    import pounce

    calls = {"n": 0}
    real = pounce.solve_qp_batch

    def counting(problems, **kw):
        calls["n"] += 1
        return real(problems, **kw)

    for seed in range(4):
        # Batched path (default on >=0.5.0).
        monkeypatch.setattr(pounce, "solve_qp_batch", counting)
        rb = _miqp(seed).solve(nlp_solver="pounce", time_limit=60, batch_size=8)

        # Serial callback fallback (force solve_qp_batch unavailable).
        monkeypatch.setattr(pounce, "solve_qp_batch", None)
        rs = _miqp(seed).solve(nlp_solver="pounce", time_limit=60, batch_size=8)

        assert rb.status == rs.status == "optimal"
        assert rb.objective == pytest.approx(rs.objective, abs=1e-5)
        # Same search tree — only the node engine differs.
        assert rb.node_count == rs.node_count

    assert calls["n"] > 0, "batched solve_qp_batch wave was never exercised"
