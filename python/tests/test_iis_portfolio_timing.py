"""A portfolio run reports where its wall time went (MilpIIS.timing, members' t_done),
and the MIPcc27 entry logs it.

The 2026-11-26 smallset run could not say how much of each instance's wall time was
the grace wait for an answer that did not change (kedron: 12 elements, 300.5 s):
the members' times were recorded but never written out.
"""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._iis_milp import milp_iis_arrays


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


def test_portfolio_result_carries_timing():
    res = milp_iis_arrays(
        _arrays(),
        iis_engine="highs",
        portfolio=("ICB", "ICB-r"),
        workers=2,
        threads=2,
        total_time_limit=60.0,
    )
    assert sorted(res.constraints) == [0, 1]
    t = res.timing
    assert t is not None, "portfolio run without timing"
    assert t["winner"] == res.order
    assert t["pre_post_size"] >= 2
    assert 0.0 <= t["first_complete_s"] <= t["winner_done_s"] <= t["portfolio_s"] + 1e-3
    assert t["post_s"] >= 0.0
    for m in res.members:
        assert m["outcome"] == "ok" and m["t_done"] is not None and m["t_done"] >= 0.0


def test_single_order_run_has_no_timing():
    res = milp_iis_arrays(_arrays(), iis_engine="highs", total_time_limit=60.0)
    assert res.timing is None and res.members is None


def test_entry_logs_members_and_timing(tmp_path):
    # The same instance as an MPS file, through the entry program.
    mps = tmp_path / "tiny.mps"
    mps.write_text(
        "NAME tiny\nROWS\n N obj\n L r0\n G r1\n L r2\nCOLUMNS\n"
        "    MARKER 'MARKER' 'INTORG'\n"
        "    x obj 1 r0 1\n    x r1 1 r2 1\n"
        "    MARKER 'MARKER' 'INTEND'\n"
        "RHS\n    rhs r0 0 r1 1\n    rhs r2 5\n"
        "BOUNDS\n LO bnd x -10\n UP bnd x 10\nENDATA\n"
    )
    gz = tmp_path / "tiny.mps.gz"
    gz.write_bytes(gzip.compress(mps.read_bytes()))
    out = tmp_path / "out.json"
    env = dict(os.environ, MIPCC_TIME_LIMIT="120", MIPCC_THREADS="2")
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    )
    p = subprocess.run(
        [sys.executable, "-m", "discopt.mipcc", str(gz), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith('mipcc: {"size"'))
    log = json.loads(line[len("mipcc: ") :])
    assert log["members"], log
    for order, outcome, size, complete, t_done in log["members"]:
        assert isinstance(order, str) and outcome in ("ok", "timeout", "abandoned", "died")
        if outcome == "ok":
            assert size is not None and t_done is not None
    t = log["timing"]
    assert t is not None and t["winner"] == log["order"]
    assert t["portfolio_s"] <= log["wall"] + 1e-6
    pytest.importorskip("highspy")
