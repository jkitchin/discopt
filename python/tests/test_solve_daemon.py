"""Socket-free tests for the ``discopt solve`` daemon layer.

The socket protocol / lifecycle itself is covered by ``test_gams_daemon.py``
(both daemons now share ``discopt._daemon_core``); these tests exercise the
solve-specific glue: the ``DISCOPT_SOLVE_*`` env prefix, the ``_solve_request``
serializer, and the client's in-process fallback. End-to-end socket use is
exercised by ``test_cli_solve.py`` via a real subprocess.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import discopt.daemon as d
import numpy as np
import pytest
from discopt import _daemon_core as core
from discopt.modeling.core import SolveResult
from discopt.solver_tuning import SolverTuning

pytestmark = pytest.mark.unit


@pytest.fixture
def sock(tmp_path) -> Path:
    return tmp_path / "d.sock"


def _start(server) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(200):
        if core.ping(server.socket_path):
            break
        time.sleep(0.01)
    return t


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


# ── kill (PID-based force terminate) ─────────────────────────────────────────
def test_kill_daemon_signals_pid_and_reaps(tmp_path):
    """``kill_daemon`` SIGKILLs the process in the PID file and reaps socket/pid.

    Uses a real ``sleep`` subprocess (not a thread) so the signal goes to a
    process we own, never the test runner.
    """
    sk = tmp_path / "k.sock"
    sk.write_text("")  # stand-in socket file
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    core._pid_path(sk).write_text(str(proc.pid))

    assert core.kill_daemon(sk) is True
    assert proc.wait(timeout=5) is not None  # the process was terminated
    assert not sk.exists() and not core._pid_path(sk).exists()
    # Idempotent: nothing to kill -> False.
    assert core.kill_daemon(sk) is False


def test_recv_timeout_derivation():
    assert d._recv_timeout_for({"time_limit": 60.0}) == 60.0 + d._SOLVE_GRACE
    assert d._recv_timeout_for(None) == 3600.0 + d._SOLVE_GRACE  # default budget


# ── client timeout on a wedged solve (socket-backed; short basetemp) ─────────
def _slow_server(sock: Path, delay: float):
    return core.DaemonServer(
        sock,
        lambda req: (time.sleep(delay), {"ok": True, "result": {}})[1],
        env_prefix="DISCOPT_SOLVE",
    )


def test_request_recv_timeout_raises_solvetimeout(sock):
    srv = _slow_server(sock, delay=1.0)
    t = _start(srv)
    try:
        with pytest.raises(core.SolveTimeout):
            core.request(sock, {"cmd": "solve", "version": core._FINGERPRINT}, recv_timeout=0.2)
    finally:
        core.stop_daemon(sock)
        t.join(timeout=5)


def test_solve_via_daemon_kills_wedged_and_falls_back(sock, monkeypatch):
    srv = _slow_server(sock, delay=1.0)
    t = _start(srv)
    monkeypatch.setattr(d, "_SOLVE_GRACE", 0.1)  # recv_timeout = time_limit + 0.1
    killed = {}
    monkeypatch.setattr(d, "kill_daemon", lambda p=None: killed.setdefault("k", True))
    try:
        resp = d.solve_via_daemon("x.nl", {"time_limit": 0.0}, socket_path=sock)
        assert resp is None  # client gave up -> caller falls back in-process
        assert killed.get("k") is True  # the wedged daemon was killed
    finally:
        core.stop_daemon(sock)
        t.join(timeout=5)
