"""#2221: an exception raised in a HiGHS callback reaches the caller on every highspy.

highspy 1.15 let a Python callback's exception propagate out of ``Highs.run()``. 1.16
catches it inside ``Highs::run`` and returns ``kError``, so a broken separator or
stop rule would read as an ordinary solver error. :func:`wrap_callback` records the
exception and :func:`run_checked` runs the solve and re-raises it, whatever status
HiGHS ended the run with.
"""

from __future__ import annotations

import numpy as np
import pytest

highspy = pytest.importorskip("highspy")

from discopt._optional_highs import _CALLBACK_ERRORS_ATTR, run_checked, wrap_callback  # noqa: E402


class _Boom(RuntimeError):
    pass


def _milp() -> "highspy.Highs":
    h = highspy.Highs()
    h.silent()
    h.addVars(2, np.array([0.0, 0.0]), np.array([3.0, 3.0]))
    idx = np.array([0, 1], dtype=np.int32)
    h.changeColsIntegrality(2, idx, np.array([highspy.HighsVarType.kInteger] * 2))
    h.changeColsCost(2, idx, np.array([-2.0, -3.0]))
    h.addRow(-highspy.kHighsInf, 4.5, 2, idx, np.array([1.0, 1.0]))
    return h


def _interruptible(integer: bool) -> "highspy.Highs":
    """A 40-column, 20-row model with presolve off. The two-variable ``_milp`` is solved
    before HiGHS 1.15/1.16 ever polls an interrupt callback (measured: zero calls), so an
    interrupt test on it would measure nothing."""
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("presolve", "off")
    rng = np.random.default_rng(0)
    n, m = 40, 20
    h.addVars(n, np.zeros(n), np.full(n, 10.0))
    idx = np.arange(n, dtype=np.int32)
    if integer:
        h.changeColsIntegrality(n, idx, np.array([highspy.HighsVarType.kInteger] * n))
    h.changeColsCost(n, idx, -rng.uniform(1, 10, n))
    for _ in range(m):
        h.addRow(-highspy.kHighsInf, float(rng.uniform(20, 50)), n, idx, rng.uniform(1, 10, n))
    return h


def _run(h):
    """The route's pattern. On highspy <= 1.15 the exception leaves ``run`` itself; on
    1.16 ``run`` swallows it and :func:`run_checked` re-raises it."""
    return run_checked(h)


def test_a_callback_exception_reaches_the_caller():
    h = _milp()
    calls: list = []

    def cb(callback_type, message, data_out, data_in, user_data):
        calls.append(1)
        raise _Boom("separator failed")

    h.setCallback(wrap_callback(h, cb), None)
    h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution)
    with pytest.raises(_Boom, match="separator failed"):
        _run(h)
    assert calls, "MEASURED NOTHING: the callback never fired"


def test_a_later_good_run_of_the_same_object_is_not_failed_by_a_stale_record():
    """On highspy <= 1.15 the exception leaves ``run`` itself, so the record is never
    consumed there. A later clean run of the same object must not re-raise it."""
    h = _milp()
    fail = [True]

    def cb(callback_type, message, data_out, data_in, user_data):
        if fail[0]:
            raise _Boom("first run only")

    h.setCallback(wrap_callback(h, cb), None)
    h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution)
    with pytest.raises(_Boom):
        _run(h)
    fail[0] = False
    h.clearSolver()
    # Must not raise the first run's exception. (highspy 1.15.1 leaves this object at
    # ``kNotset`` after an exception escaped ``run``; 1.16 re-solves it to ``kOptimal``.
    # Either way the old exception is not this run's.)
    _run(h)
    assert h.getModelStatus() != highspy.HighsModelStatus.kSolveError


def test_no_exception_no_raise():
    h = _milp()
    calls: list = []

    def cb(callback_type, message, data_out, data_in, user_data):
        calls.append(1)

    h.setCallback(wrap_callback(h, cb), None)
    h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution)
    _run(h)
    assert calls, "MEASURED NOTHING: the callback never fired"
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    assert h.getInfo().objective_function_value == pytest.approx(-11.0)


def test_run_checked_without_a_wrapped_callback_just_runs():
    h = _milp()
    assert run_checked(h) == highspy.HighsStatus.kOk
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal


def test_an_interrupt_callback_exception_reaches_the_caller():
    """The stop rules (``lp_milp_highs``'s ``_StopSignal``, ``milp_highs``'s claim stop,
    ``mipcc``/``_iis_backend``'s wall-clock guards) are ``kCallbackMipInterrupt``
    callbacks. One that raises must not read as an interrupted or failed solve."""
    h = _interruptible(integer=True)
    calls: list = []

    def cb(callback_type, message, data_out, data_in, user_data):
        calls.append(1)
        raise _Boom("stop rule failed")

    h.setCallback(wrap_callback(h, cb), None)
    h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipInterrupt)
    with pytest.raises(_Boom, match="stop rule failed"):
        _run(h)
    assert calls, "MEASURED NOTHING: the interrupt callback never fired"


def test_a_subscribed_interrupt_guard_exception_reaches_the_caller():
    """``_iis_milp``'s guard is subscribed with ``cbSimplexInterrupt`` (the event API)
    and runs on an LP; the same contract holds there."""
    h = _interruptible(integer=False)
    calls: list = []

    def guard(event):
        calls.append(1)
        raise _Boom("guard failed")

    h.cbSimplexInterrupt.subscribe(wrap_callback(h, guard))
    with pytest.raises(_Boom, match="guard failed"):
        _run(h)
    assert calls, "MEASURED NOTHING: the simplex interrupt guard never fired"


class _SwallowingHighs:
    """A stand-in for a highspy whose ``run`` swallows a callback's exception and ends
    with a status other than ``kSolveError`` (e.g. ``kInterrupt``)."""

    def __init__(self) -> None:
        self.cb = None

    def run(self):
        try:
            self.cb()
        except Exception:
            pass
        return "kWarning"

    def getModelStatus(self):
        return "kInterrupt"


def test_the_reraise_does_not_depend_on_the_run_status():
    h = _SwallowingHighs()
    h.cb = wrap_callback(h, lambda: (_ for _ in ()).throw(_Boom("swallowed")))
    with pytest.raises(_Boom, match="swallowed"):
        run_checked(h)
    assert getattr(h, _CALLBACK_ERRORS_ATTR) == [], "the record must be consumed"


def test_a_stale_record_is_cleared_before_the_run():
    h = _SwallowingHighs()
    h.cb = wrap_callback(h, lambda: None)
    getattr(h, _CALLBACK_ERRORS_ATTR).append(_Boom("from an earlier run"))
    assert run_checked(h) == "kWarning"
