"""#1917 -- a POUNCE node wave runs with pounce's tape evaluation serial.

Inside ``pounce.solve_nlp_batch`` a large tape (Jacobian >= 4096 nonzeros) is
evaluated with a nested Rayon ``par_iter`` that holds the GIL and the shared
``NlProblem``'s ``RefCell`` borrow; a worker waiting in it steals another wave
instance, whose callback reaches the borrowed tape: ``RefCell already
borrowed``, status -199, the node sentinel. On the syngas hull tape 13 of a
48-instance wave were lost that way. The collision needs a Rayon steal and is
not deterministic, so these tests pin the mechanism that removes it: every
``solve_nlp_batch`` call sees ``POUNCE_NL_PARALLEL_EVAL=0`` (serial and parallel
walks are bit-identical in pounce, so no number changes), the variable is gone
again afterwards, and a value the user set is left alone.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from discopt import solver as S

pytestmark = pytest.mark.smoke

ENV = "POUNCE_NL_PARALLEL_EVAL"


def _recording_batch(seen):
    def solve_nlp_batch(problems, x0s, options, **kw):
        seen.append(os.environ.get(ENV))
        return [(np.asarray(x0, dtype=np.float64), {"status": 0}) for x0 in x0s]

    return solve_nlp_batch


def _run(seen, n=3):
    problems = [object()] * n
    x0s = [np.zeros(2)] * n
    out = S._solve_nlp_batch_by_deadline(
        _recording_batch(seen),
        problems,
        x0s,
        {"max_wall_time": 5.0},
        time.perf_counter() + 60.0,
    )
    assert len(out) == n


def test_wave_runs_with_serial_tape_eval(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    seen = []
    _run(seen)
    assert seen, "solve_nlp_batch was never called"
    assert all(v == "0" for v in seen), seen
    assert ENV not in os.environ, "the wave leaked the setting past its end"


def test_wave_restores_env_when_the_batch_raises(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)

    def boom(problems, x0s, options, **kw):
        assert os.environ.get(ENV) == "0"
        raise RuntimeError("wave failed")

    with pytest.raises(RuntimeError, match="wave failed"):
        S._solve_nlp_batch_by_deadline(
            boom, [object()], [np.zeros(2)], {"max_wall_time": 5.0}, time.perf_counter() + 60.0
        )
    assert ENV not in os.environ


@pytest.mark.parametrize("user_value", ["1", "0"])
def test_user_setting_is_left_alone(monkeypatch, user_value):
    monkeypatch.setenv(ENV, user_value)
    seen = []
    _run(seen)
    assert seen and all(v == user_value for v in seen), seen
    assert os.environ.get(ENV) == user_value
