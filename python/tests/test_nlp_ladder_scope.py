"""``DISCOPT_NLP_LADDER_SCOPED``: the retry ladder goes off only where a verdict is not a proof.

The scope (:mod:`discopt.solvers._nlp_ladder_scope`) covers primal-heuristic NLPs
and node NLPs of a nonconvex model. Pinned here:

* flag OFF (the default): nothing changes anywhere;
* flag ON: inside the scope the five retry options are filled with ``"no"``, an
  explicit caller value wins, and outside the scope nothing changes;
* the heuristic decorator and nonconvex B&B nodes enter the scope, and convex B&B
  nodes do not;
* the options reach POUNCE on both the serial path and the batch (Rust pool) path,
  and POUNCE accepts them.
"""

from __future__ import annotations

import contextlib

import discopt._relax.primal_heuristics as PH
import discopt.modeling as dm
import discopt.solver as S
import discopt.solvers.nlp_pounce as NP
import numpy as np
import pytest
from discopt._relax.primal_heuristics import _stops_at_solve_deadline
from discopt.solvers._nlp_ladder_scope import (
    LADDER_OFF,
    LADDER_RETRY_OPTIONS,
    apply_ladder_scope,
    ladder_off,
)

pytest.importorskip("pounce")
pytestmark = pytest.mark.unit

OFF = {k: "no" for k in LADDER_RETRY_OPTIONS}


def test_flag_off_is_a_no_op(monkeypatch):
    monkeypatch.delenv("DISCOPT_NLP_LADDER_SCOPED", raising=False)
    with ladder_off():
        assert LADDER_OFF.get() is False
        assert apply_ladder_scope({"tol": 1e-8}) == {"tol": 1e-8}


def test_flag_on_fills_defaults_inside_the_scope_only(monkeypatch):
    monkeypatch.setenv("DISCOPT_NLP_LADDER_SCOPED", "1")
    assert apply_ladder_scope({}) == {}
    with ladder_off():
        assert apply_ladder_scope({}) == OFF
        explicit = {"infeasibility_mu_strategy_retry": "yes"}
        assert apply_ladder_scope(explicit)["infeasibility_mu_strategy_retry"] == "yes"
        with ladder_off(False):
            # ``active=False`` does not clear an enclosing scope.
            assert LADDER_OFF.get() is True
    assert LADDER_OFF.get() is False
    with ladder_off(False):
        assert apply_ladder_scope({}) == {}


def test_heuristic_decorator_enters_the_scope(monkeypatch):
    monkeypatch.setenv("DISCOPT_NLP_LADDER_SCOPED", "1")
    seen = _stops_at_solve_deadline(lambda: LADDER_OFF.get())()
    assert seen is True
    assert LADDER_OFF.get() is False


def _nonconvex(n: int = 6):
    """A bilinear chain with switches: hundreds of B&B nodes, every one a node NLP."""
    m = dm.Model("ladder_nc")
    x = m.continuous("x", shape=(n,), lb=-2.0, ub=2.0)
    y = m.binary("y", shape=(n,))
    for i in range(n - 1):
        m.subject_to(x[i] * x[i + 1] >= 0.3 - y[i])
    m.subject_to(dm.sum(y) <= n // 2)
    m.subject_to(dm.sum(x**2) >= 1.5)
    m.minimize(dm.sum(x) + dm.sum(dm.sin(3 * x)) + 0.5 * dm.sum(y))
    return m


SOLVE = dict(nlp_solver="pounce", time_limit=120, deterministic=True, max_nodes=300)


def _convex(n: int = 6):
    """A convex MINLP that branches: on/off switches with a cardinality limit."""
    c = np.linspace(0.3, 1.7, n)
    m = dm.Model("ladder_cvx")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=2.0)
    y = m.binary("y", shape=(n,))
    for i in range(n):
        m.subject_to(x[i] <= 2.0 * y[i])
    m.subject_to(dm.sum(y) <= n // 2)
    m.subject_to(dm.sum(x) >= 2.5)
    m.minimize(
        dm.sum((x - c) ** 2) + dm.sum(dm.exp(0.3 * x)) + 0.1 * dm.sum(y * np.arange(1, n + 1))
    )
    return m


def _record(monkeypatch):
    """Record each POUNCE solve (serial or batch wave) and each B&B node-NLP call."""
    calls: list[tuple[str, bool]] = []
    nodes: list[tuple[object, bool]] = []
    real_apply = NP.apply_ladder_scope
    real_batch = S._solve_nlp_batch_by_deadline
    real_node = S._solve_node_nlp

    def apply(opts):
        out = real_apply(opts)
        calls.append(("serial", all(out.get(k) == "no" for k in LADDER_RETRY_OPTIONS)))
        return out

    def batch(fn, problems, x0s, options, deadline, **kw):
        calls.append(("batch", all(options.get(k) == "no" for k in LADDER_RETRY_OPTIONS)))
        return real_batch(fn, problems, x0s, options, deadline, **kw)

    def node(*a, **k):
        nodes.append((k.get("convex", "unset"), LADDER_OFF.get()))
        return real_node(*a, **k)

    monkeypatch.setattr(NP, "apply_ladder_scope", apply)
    monkeypatch.setattr(S, "_solve_nlp_batch_by_deadline", batch)
    monkeypatch.setattr(S, "_solve_node_nlp", node)
    return calls, nodes


def test_nonconvex_serial_nodes_drop_the_ladder(monkeypatch):
    monkeypatch.setenv("DISCOPT_NLP_LADDER_SCOPED", "1")
    monkeypatch.setattr(S, "_POUNCE_BATCH_MIN_VARS", 10**9)
    calls, nodes = _record(monkeypatch)
    res = _nonconvex().solve(**SOLVE)
    assert res.objective is not None and np.isfinite(res.objective)
    bb = [scoped for cvx, scoped in nodes if cvx is False]
    assert len(bb) >= 5, f"too few B&B node NLPs to test ({len(nodes)} node calls)"
    assert all(bb)
    assert sum(off for _, off in calls) >= len(bb)


def test_nonconvex_batch_waves_drop_the_ladder(monkeypatch):
    """The wave runs in POUNCE's Rust pool: the options must be in ``batch_opts``."""
    monkeypatch.setenv("DISCOPT_NLP_LADDER_SCOPED", "1")
    monkeypatch.setattr(S, "_POUNCE_BATCH_MIN_VARS", 1)
    calls, _ = _record(monkeypatch)
    res = _nonconvex().solve(**SOLVE)
    assert res.objective is not None and np.isfinite(res.objective)
    waves = [off for kind, off in calls if kind == "batch"]
    assert len(waves) >= 5, f"too few batch waves to test ({calls[:5]})"
    assert all(waves)


def test_flag_off_never_drops_the_ladder(monkeypatch):
    monkeypatch.delenv("DISCOPT_NLP_LADDER_SCOPED", raising=False)
    monkeypatch.setattr(S, "_POUNCE_BATCH_MIN_VARS", 1)
    calls, _ = _record(monkeypatch)
    _nonconvex().solve(**SOLVE)
    assert len(calls) >= 5, "no POUNCE solve was observed"
    assert not any(off for _, off in calls)


def test_convex_nodes_keep_the_ladder(monkeypatch):
    """Convex node NLPs are bound producers: their B&B solves stay outside the scope."""
    monkeypatch.setenv("DISCOPT_NLP_LADDER_SCOPED", "1")
    # Keep the model on B&B (the convex route would solve it by OA instead).
    monkeypatch.setenv("DISCOPT_CONVEX_MINLP_ROUTE", "0")
    # RINS/RENS solve a restricted sub-model with a nested B&B; its nodes are
    # heuristic work and are scoped on purpose. Count only the outer tree's nodes.
    depth = [0]
    real_stop = PH.stop_at_solve_deadline

    @contextlib.contextmanager
    def counted():
        depth[0] += 1
        try:
            with real_stop():
                yield
        finally:
            depth[0] -= 1

    monkeypatch.setattr(PH, "stop_at_solve_deadline", counted)
    outer: list[bool] = []
    real_node = S._solve_node_nlp

    def node(*a, **k):
        if k.get("convex") is True and depth[0] == 0:
            outer.append(LADDER_OFF.get())
        return real_node(*a, **k)

    monkeypatch.setattr(S, "_solve_node_nlp", node)
    monkeypatch.setattr(S, "_POUNCE_BATCH_MIN_VARS", 10**9)
    res = _convex().solve(**SOLVE, nlp_bb=True)
    assert res.objective is not None and np.isfinite(res.objective)
    assert len(outer) >= 5, f"too few convex B&B node NLPs to test ({len(outer)})"
    assert not any(outer)
