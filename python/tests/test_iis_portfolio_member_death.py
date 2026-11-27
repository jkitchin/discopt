"""A portfolio member killed from outside (the OOM killer) must not discard the
other members' answers.

Measured on MIPcc27 smallset no-ip-65059 in a 7.75 GiB Docker VM: the six members
grew to ~1.3 GB each, the kernel killed one at 14 s (exit -9), the portfolio
raised, and the entry fell back to the full model, which the checker could not
decide in 3600 s. Each member's answer is sound on its own, so the survivors'
answer is just as valid as it would have been without the death.
"""

import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._iis_milp import milp_iis_arrays

_SELF_KILL = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"


def _arrays():
    # x <= 0 and x >= 1, x integer in [-10, 10], plus a redundant row x <= 5.
    A = sp.csr_matrix(np.array([[1.0], [1.0], [1.0]]))
    return (
        A,
        np.array([-1e20, 1.0, -1e20]),
        np.array([0.0, 1e20, 5.0]),
        {0: np.array([0]), 1: np.array([1]), 2: np.array([2])},
        np.array([-10.0]),
        np.array([10.0]),
        np.array([True]),
    )


def _kill_members(monkeypatch, which):
    """Make the portfolio's ``which``-th member launches (0-based) SIGKILL
    themselves before reading their job. Returns the launch counter."""
    real = subprocess.Popen
    calls = {"n": 0, "killed": 0}

    def popen(args, *a, **kw):
        k = calls["n"]
        calls["n"] += 1
        if k in which and list(args[:2]) == [sys.executable, "-c"]:
            calls["killed"] += 1
            args = [sys.executable, "-c", _SELF_KILL]
        return real(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", popen)
    return calls


def test_survivors_answer_when_one_member_is_killed(monkeypatch):
    calls = _kill_members(monkeypatch, {0})
    with pytest.warns(RuntimeWarning, match="exited without a result"):
        res = milp_iis_arrays(
            _arrays(), iis_engine="highs", portfolio=("ICB", "ICB-r"), workers=2, threads=2
        )
    assert calls["killed"] == 1, calls
    assert sorted(res.constraints) == [0, 1]
    outcomes = {m["order"]: m["outcome"] for m in res.members}
    assert outcomes == {"ICB": "died", "ICB-r": "ok"}, res.members
    died = next(m for m in res.members if m["order"] == "ICB")
    assert "-9" in died["message"]


def test_all_members_killed_raises(monkeypatch):
    calls = _kill_members(monkeypatch, {0, 1})
    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(RuntimeError, match="no portfolio member returned an answer; 2 died"),
    ):
        milp_iis_arrays(
            _arrays(), iis_engine="highs", portfolio=("ICB", "ICB-r"), workers=2, threads=2
        )
    assert calls["killed"] == 2, calls


def test_killed_post_pass_keeps_its_input_answer(monkeypatch):
    import time

    from discopt._iis_milp import _polish_parallel

    res = milp_iis_arrays(_arrays(), iis_engine="highs", portfolio=("ICB",), workers=1)
    calls = _kill_members(monkeypatch, {0})
    with pytest.warns(RuntimeWarning, match="post-pass 0 exited without a result"):
        got = _polish_parallel(_arrays(), [res], [None], 1, time.perf_counter() + 60.0, {})
    assert calls["killed"] == 1, calls
    # None: the caller keeps the unpolished answer, which its member confirmed.
    assert got == [None]
