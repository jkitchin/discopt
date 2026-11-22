"""``_run_with_deep_recursion`` on an interpreter that cannot start threads.

Pyodide (wasm32) raises ``RuntimeError: can't start new thread`` from
``Thread.start``. The runner used to let that reach every caller; in the browser
the #1551 objective-error check then withdrew the certificate of any solve whose
objective needed more than the default recursion limit. Measured under Pyodide
0.28.3 with the #2295 fix applied: the issue's 60-item knapsack ``.nl`` loaded
and returned ``status='feasible'`` instead of ``optimal``. Simulated here by
making ``threading.Thread.start`` raise exactly that error.
"""

from __future__ import annotations

import sys
import threading

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import rules
from discopt.modeling.core import from_nl

pytestmark = pytest.mark.unit


def _no_threads(monkeypatch):
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)


def test_runner_falls_back_inline_when_threads_cannot_start(monkeypatch):
    _no_threads(monkeypatch)
    limit = sys.getrecursionlimit()
    stack = threading.stack_size()
    need = limit + 500
    seen = rules._run_with_deep_recursion(sys.getrecursionlimit, depth_need=need)
    assert seen == need, "fn did not run under the raised limit"
    assert sys.getrecursionlimit() == limit
    assert threading.stack_size() == stack

    # Exceptions from fn still propagate, and the limit is still restored.
    def boom():
        raise ValueError("from fn")

    with pytest.raises(ValueError, match="from fn"):
        rules._run_with_deep_recursion(boom, depth_need=need)
    assert sys.getrecursionlimit() == limit

    # A walk deeper than the default limit actually completes inline.
    def depth(k: int) -> int:
        return 0 if k == 0 else 1 + depth(k - 1)

    assert rules._run_with_deep_recursion(lambda: depth(limit + 200), depth_need=need) == (
        limit + 200
    )


def test_long_objective_keeps_its_certificate_without_threads(monkeypatch, tmp_path):
    """End to end on the #2295 shape: a 60-item knapsack read from ``.nl``."""
    rng = np.random.default_rng(1)
    v, w = rng.integers(5, 50, 60), rng.integers(3, 30, 60)
    m = dm.Model("knap")
    x = m.binary("x", shape=(60,))
    m.maximize(sum(int(v[i]) * x[i] for i in range(60)))
    m.subject_to(sum(int(w[i]) * x[i] for i in range(60)) <= int(w.sum() * 0.4))
    path = tmp_path / "knapsack.nl"
    m.to_nl(str(path))

    deep = {"n": 0}
    real = rules._run_with_deep_recursion

    def spy(fn, *, depth_need):
        if depth_need > sys.getrecursionlimit():
            deep["n"] += 1
        return real(fn, depth_need=depth_need)

    monkeypatch.setattr(rules, "_run_with_deep_recursion", spy)
    _no_threads(monkeypatch)
    # The suite runs at a limit of 3000 (pyproject.toml); CPython's default, and
    # Pyodide's, is 1000, which is where this objective needs the deep path.
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        r = from_nl(str(path)).solve(time_limit=60)
    finally:
        sys.setrecursionlimit(old_limit)
    assert deep["n"] >= 1, "the deep-recursion path was never reached; the test proves nothing"
    stats = r.solver_stats or {}
    assert "certificate/objective_eval_error_skipped" not in stats, stats.get(
        "certificate/objective_eval_error_skipped"
    )
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(1079.0)
