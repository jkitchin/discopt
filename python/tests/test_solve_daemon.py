"""Socket-free tests for the ``discopt solve`` daemon layer.

The socket protocol / lifecycle itself is covered by ``test_gams_daemon.py``
(both daemons now share ``discopt._daemon_core``); these tests exercise the
solve-specific glue: the ``DISCOPT_SOLVE_*`` env prefix, the ``_solve_request``
serializer, and the client's in-process fallback. End-to-end socket use is
exercised by ``test_cli_solve.py`` via a real subprocess.
"""

from __future__ import annotations

from pathlib import Path

import discopt.daemon as d
import numpy as np
import pytest
from discopt import _daemon_core as core
from discopt.modeling.core import SolveResult
from discopt.solver_tuning import SolverTuning

pytestmark = pytest.mark.unit


class _FakeModel:
    def __init__(self, capture=None):
        self._capture = capture

    def solve(self, **kw):
        if self._capture is not None:
            self._capture.update(kw)
        return SolveResult(status="optimal", objective=1.0, x={"x": np.array(1.0)})


def test_make_server_uses_solve_prefix_and_socket(monkeypatch):
    monkeypatch.setenv("DISCOPT_SOLVE_MAX_SOLVES", "7")
    srv = d.make_server(Path("/tmp/probe.sock"))
    assert isinstance(srv, core.DaemonServer)
    assert srv.version == core._FINGERPRINT
    assert srv.max_solves == 7  # the DISCOPT_SOLVE_* prefix resolved


def test_solve_request_serializes(monkeypatch):
    monkeypatch.setattr("discopt.modeling.core.from_nl", lambda p: _FakeModel())
    reply = d._solve_request({"nl_file": "x.nl", "options": {"time_limit": 5}})
    assert reply["ok"] is True
    assert reply["result"]["status"] == "optimal"
    assert reply["result"]["objective"] == 1.0


def test_solve_request_rebuilds_tuning(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("discopt.modeling.core.from_nl", lambda p: _FakeModel(captured))
    d._solve_request({"nl_file": "x.nl", "options": {"tuning": {"rlt_quad": False}}})
    assert isinstance(captured["tuning"], SolverTuning)
    assert captured["tuning"].rlt_quad is False


def test_client_fallback_when_daemon_unavailable(monkeypatch, tmp_path):
    # No daemon, and spawning disabled -> client signals in-process fallback.
    monkeypatch.setattr(d, "spawn_daemon", lambda *a, **k: False)
    assert d.solve_via_daemon("x.nl", {}, socket_path=tmp_path / "none.sock") is None


def test_solve_and_gams_sockets_differ():
    from discopt.gams import daemon as g

    assert d.default_socket_path() != g.default_socket_path()
    assert "discopt-solve" in str(d.default_socket_path())
