"""MIPCC_PORTFOLIO_WAIT=evidence: the fast wait whose 300 s floor ends early, at
t + max(10 t, 60) s after the first member completes at t.

The 2026-11-27 timing probe (3600 s, 17 smallset instances) found every run idling
to the floor: kedron's answer was final at 3.3 s and the run ended at 300.5 s.
"""

import math

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._iis_milp import (
    PORTFOLIO_FLOOR_EXITS,
    PORTFOLIO_WAITS,
    _grace_cutoff,
    milp_iis_arrays,
)


def test_every_wait_names_its_floor_exit():
    assert set(PORTFOLIO_FLOOR_EXITS) == set(PORTFOLIO_WAITS)
    assert PORTFOLIO_WAITS["evidence"] == PORTFOLIO_WAITS["fast"]
    assert PORTFOLIO_FLOOR_EXITS["fast"] is None  # the default is unchanged


@pytest.mark.parametrize(
    "t1, t_best",
    [
        (3.3, 3.3),  # kedron: nothing later beat the first
        (1.0, 29.0),  # jnh10: the smallest answer 28 s after the first
        (19.5, 148.9),  # jjliu_clique: 6.7 t after it
        (185.3, 233.0),  # stein45inf: late first completion, floor-capped
    ],
)
def test_evidence_cutoff_keeps_the_measured_late_winners(t1, t_best):
    g, w, f = PORTFOLIO_WAITS["evidence"]
    x = PORTFOLIO_FLOOR_EXITS["evidence"]
    cut = _grace_cutoff(t1, g, w, f, 3130.0, x)
    assert cut >= t_best
    # Never later than the fast wait, and never earlier than its own base wait.
    assert cut <= _grace_cutoff(t1, g, w, f, 3130.0)
    assert cut >= _grace_cutoff(t1, g, w, 0.0, 3130.0)


def test_evidence_cutoff_ends_the_floor_early():
    g, w, f = PORTFOLIO_WAITS["evidence"]
    x = PORTFOLIO_FLOOR_EXITS["evidence"]
    assert _grace_cutoff(3.3, g, w, f, 3130.0) == 300.0  # fast: pads to the floor
    assert _grace_cutoff(3.3, g, w, f, 3130.0, x) == pytest.approx(63.3)
    assert _grace_cutoff(19.5, g, w, f, 3130.0, x) == pytest.approx(214.5)
    assert _grace_cutoff(40.0, g, w, f, 3130.0, x) == 300.0  # the floor still caps it
    assert _grace_cutoff(3.3, g, w, f, None, x) == pytest.approx(13.3)  # no budget: no floor


def _arrays():
    A = sp.csr_matrix(np.array([[1.0], [1.0]]))
    return (
        A,
        np.array([-1e20, 1.0]),
        np.array([0.0, 1e20]),
        {0: np.array([0]), 1: np.array([1])},
        np.array([-10.0]),
        np.array([10.0]),
        np.array([True]),
    )


def test_floor_exit_needs_a_portfolio_and_a_valid_pair():
    with pytest.raises(ValueError, match="grace_floor_exit"):
        milp_iis_arrays(_arrays(), grace_floor_exit=(10.0, 60.0))
    with pytest.raises(ValueError, match="grace_floor_exit"):
        milp_iis_arrays(
            _arrays(),
            portfolio=("ICB",),
            grace=2.0,
            grace_floor=0.5,
            grace_floor_exit=(-1.0, 60.0),
        )
    assert math.isfinite(PORTFOLIO_FLOOR_EXITS["evidence"][0])


def test_entry_passes_the_evidence_floor_exit(tmp_path, monkeypatch):
    import discopt._iis_milp as engine
    import discopt.mipcc as mipcc

    seen = []

    def capture(*a, **k):
        seen.append(k.get("grace_floor_exit", "missing"))
        raise TimeoutError("stop here")

    monkeypatch.setattr(engine, "milp_iis_arrays", capture)
    mps = tmp_path / "m.mps"
    mps.write_text(
        "NAME m\nROWS\n N obj\n L r0\n G r1\nCOLUMNS\n"
        "    MARKER 'MARKER' 'INTORG'\n    x obj 1 r0 1\n    x r1 1\n"
        "    MARKER 'MARKER' 'INTEND'\nRHS\n    rhs r0 0 r1 1\n"
        "BOUNDS\n LO bnd x -10\n UP bnd x 10\nENDATA\n"
    )
    monkeypatch.setenv("MIPCC_TIME_LIMIT", "60")
    monkeypatch.delenv("MIPCC_PORTFOLIO_WAIT", raising=False)
    assert mipcc.main([str(mps), str(tmp_path / "a.json")]) == 0
    monkeypatch.setenv("MIPCC_PORTFOLIO_WAIT", "evidence")
    assert mipcc.main([str(mps), str(tmp_path / "b.json")]) == 0
    assert seen == [None, (10.0, 60.0)]
