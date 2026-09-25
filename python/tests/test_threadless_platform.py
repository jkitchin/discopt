"""Solve without OS threads, the way Pyodide has to.

POUNCE's node waves and discopt-core's batch branches are parallel Rayon
passes. Rayon builds its global pool lazily and *panics* when that build fails,
and the panic arrives in Python as ``pyo3_runtime.PanicException``, which
derives from ``BaseException`` deliberately. So every ``except Exception``
serial fallback in ``solver.py`` is bypassed and the solve dies outright.

That is not hypothetical: it is what the in-browser page hit. Under Pyodide
0.28.3 the MIQP example aborted with

    pyo3_runtime.PanicException: The global thread pool has not been
    initialized.: ThreadPoolBuildError { kind: IOError(Os { code: 6,
    kind: WouldBlock, message: "Resource temporarily unavailable" }) }

while the other five examples passed only because they were small enough to sit
under the batch-size floors that guard the parallel branches.

This module makes a threaded machine behave like a threadless one, then solves.
It cannot reproduce the Rust panic here -- a desktop build has threads, and the
probe is what decides -- so it tests the decision instead: that the probe
reports the truth, and that with the probe reporting "no threads" the solve
still produces the same answer by the serial route.
"""

from __future__ import annotations

import threading

import discopt.modeling as dm
import pytest
from discopt import solver as solver_mod

pytestmark = pytest.mark.smoke


def test_probe_reports_true_on_a_threaded_platform() -> None:
    """The suite runs threads, so anything but True means a broken probe."""
    solver_mod._os_threads_available.cache_clear()
    assert solver_mod._os_threads_available() is True


def test_probe_reports_false_when_threads_cannot_start() -> None:
    """The failing half, which is the half that matters.

    Pyodide raises ``RuntimeError: can't start new thread`` from
    ``Thread.start()`` (measured under Pyodide 0.28.3, ``sys.platform ==
    'emscripten'``, ``os.cpu_count() == 1``). Simulated here by making
    ``start`` raise exactly that.
    """
    real_start = threading.Thread.start
    solver_mod._os_threads_available.cache_clear()
    try:

        def refuse(self: threading.Thread) -> None:
            raise RuntimeError("can't start new thread")

        threading.Thread.start = refuse  # type: ignore[method-assign]
        assert solver_mod._os_threads_available() is False
    finally:
        threading.Thread.start = real_start  # type: ignore[method-assign]
        solver_mod._os_threads_available.cache_clear()

    # And the cache must not pin the simulated answer for the rest of the run.
    assert solver_mod._os_threads_available() is True


@pytest.fixture
def threadless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report "no OS threads" for the duration of one test.

    Patching the probe rather than ``Thread.start`` itself: pytest, logging and
    the Rust extension all use threads, and a process that genuinely cannot
    start one cannot run a test. What is under test is the branch the probe
    selects.
    """
    monkeypatch.setattr(solver_mod, "_os_threads_available", lambda: False)


def _miqp() -> dm.Model:
    """A cardinality-constrained portfolio: the shape that actually died.

    A convex QP objective plus binary indicators, which is what routes through
    ``_solve_miqp_bb`` -> ``_pounce_qp_relaxation_nodes`` -> ``solve_qp_batch``
    -- the wave whose Rayon pool build panicked. Small on purpose: the POUNCE
    wave has no batch-size floor, so even three assets reach it.
    """
    cov = [[0.04, 0.012, 0.002], [0.012, 0.09, 0.003], [0.002, 0.003, 0.01]]
    ret = [0.07, 0.11, 0.05]

    m = dm.Model("threadless-miqp")
    w = m.continuous("w", shape=(3,), lb=0, ub=1)
    hold = m.binary("hold", shape=(3,))

    m.minimize(dm.sum([cov[i][j] * w[i] * w[j] for i in range(3) for j in range(3)]))
    m.subject_to(dm.sum([w[i] for i in range(3)]) == 1, name="budget")
    m.subject_to(dm.sum([ret[i] * w[i] for i in range(3)]) >= 0.08, name="return")
    m.subject_to(dm.sum([hold[i] for i in range(3)]) <= 2, name="cardinality")
    for i in range(3):
        m.subject_to(w[i] <= hold[i], name=f"on{i}")
    return m


def test_miqp_reaches_the_same_optimum_without_threads() -> None:
    """The serial route must certify the same optimum as the parallel one.

    Deliberately *not* using the ``threadless`` fixture: a fixture applies to
    the whole test, so both solves would have taken the serial route and the
    comparison would have been serial against serial -- an assertion that
    cannot fail. The baseline is taken first, at full capability, and only the
    second solve is starved.
    """
    pounce = pytest.importorskip("pounce")

    waves = {"n": 0}
    real_batch = pounce.solve_qp_batch

    def counting(*args: object, **kwargs: object) -> object:
        waves["n"] += 1
        return real_batch(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pounce, "solve_qp_batch", counting)
        threaded = _miqp().solve()
        threaded_waves = waves["n"]

        waves["n"] = 0
        mp.setattr(solver_mod, "_os_threads_available", lambda: False)
        serial = _miqp().solve()
        serial_waves = waves["n"]

    # Without this the test would pass on a build where the model never reached
    # the wave at all, which is the failure mode it exists to rule out.
    assert threaded_waves > 0, "model never reached the POUNCE QP wave"
    assert serial_waves == 0, f"wave was entered {serial_waves}x with no threads"

    assert threaded.status == "optimal", threaded.status
    assert serial.status == threaded.status
    assert serial.objective == pytest.approx(threaded.objective, abs=1e-6)


def test_lp_and_milp_solve_without_threads(threadless: None) -> None:
    """discopt-core's own batch branches sit behind the same capability."""
    checks = 0

    lp = dm.Model("threadless-lp")
    x = lp.continuous("x", lb=0.0, ub=4.0)
    y = lp.continuous("y", lb=0.0, ub=4.0)
    lp.subject_to(x + 2.0 * y <= 6.0)
    lp.maximize(3.0 * x + 2.0 * y)
    res = lp.solve()
    assert res.status == "optimal", res.status
    checks += 1

    milp = dm.Model("threadless-milp")
    a = milp.integer("a", lb=0, ub=10)
    c = milp.integer("c", lb=0, ub=10)
    milp.subject_to(2.0 * a + 3.0 * c <= 12.0)
    milp.maximize(5.0 * a + 4.0 * c)
    res = milp.solve()
    assert res.status == "optimal", res.status
    checks += 1

    assert checks == 2, checks
