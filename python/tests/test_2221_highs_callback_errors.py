"""#2221: an exception raised in a HiGHS callback reaches the caller on every highspy.

highspy 1.15 let a Python callback's exception propagate out of ``Highs.run()``. 1.16
catches it inside ``Highs::run`` and returns ``kError``, so a broken separator or
stop rule would read as an ordinary solver error. :func:`wrap_callback` records the
exception and :func:`raise_callback_error` re-raises it after ``run``.
"""

from __future__ import annotations

import numpy as np
import pytest

highspy = pytest.importorskip("highspy")

from discopt._optional_highs import raise_callback_error, wrap_callback  # noqa: E402


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


def _run(h) -> None:
    """The route's pattern: ``run`` then :func:`raise_callback_error`. On highspy <= 1.15
    the exception leaves ``run`` itself; on 1.16 it is re-raised by the second line."""
    h.run()
    raise_callback_error(h)


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


def test_raise_callback_error_without_a_wrapped_callback_is_a_no_op():
    raise_callback_error(highspy.Highs())
