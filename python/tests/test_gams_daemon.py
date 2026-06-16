"""Lifecycle tests for the warm discopt GAMS solver daemon.

Drives the daemon over a real unix socket with a fake solve function, so no
GAMS install is required. Covers ping/solve/stop, idle timeout, max-solves,
error isolation, version eviction, and the client's spawn/fallback behaviour.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from discopt.gams import daemon as d


def _start(server: d.DaemonServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(200):  # wait until the socket answers
        if d.ping(server.socket_path):
            break
        time.sleep(0.01)
    return t


@pytest.fixture
def sock(tmp_path) -> Path:
    return tmp_path / "d.sock"


def test_ping_and_version(sock):
    calls = []
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: calls.append(cf) or 0)
    t = _start(srv)
    try:
        info = d.ping(sock)
        assert info and info["ok"] is True
        assert "version" in info
    finally:
        d.stop_daemon(sock)
        t.join(timeout=5)


def test_solve_round_trip(sock):
    seen = []
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: seen.append(cf) or 0)
    t = _start(srv)
    try:
        rc = d.solve_via_daemon("/path/to/control.dat", socket_path=sock)
        assert rc == 0
        assert seen == ["/path/to/control.dat"]
    finally:
        d.stop_daemon(sock)
        t.join(timeout=5)


def test_solve_nonzero_rc(sock):
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 1)
    t = _start(srv)
    try:
        assert d.solve_via_daemon("x", socket_path=sock) == 1
    finally:
        d.stop_daemon(sock)
        t.join(timeout=5)


def test_solve_error_is_isolated(sock):
    """A solve that raises must not take the daemon down."""

    def boom(cf, sysdir=None):
        raise RuntimeError("kaboom")

    srv = d.DaemonServer(socket_path=sock, solve_fn=boom)
    t = _start(srv)
    try:
        rc = d.solve_via_daemon("x", socket_path=sock)
        assert rc == 1
        # daemon still alive and serving
        assert d.ping(sock) is not None
    finally:
        d.stop_daemon(sock)
        t.join(timeout=5)


def test_stop(sock):
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0)
    t = _start(srv)
    assert d.stop_daemon(sock) is True
    t.join(timeout=5)
    assert not t.is_alive()
    # socket cleaned up on exit
    assert not sock.exists()


def test_idle_timeout(sock):
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0, idle_timeout=0.3)
    t = _start(srv)
    t.join(timeout=5)
    assert not t.is_alive()  # exited on its own with no requests
    assert d.ping(sock) is None


def test_max_solves(sock):
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0, max_solves=2)
    t = _start(srv)
    try:
        assert d.solve_via_daemon("a", socket_path=sock) == 0
        assert d.solve_via_daemon("b", socket_path=sock) == 0
    finally:
        t.join(timeout=5)
    assert not t.is_alive()  # stopped after hitting the cap


def test_version_eviction(sock):
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0, version="9.9.9")
    t = _start(srv)
    # client __version__ != server version -> daemon serves then exits
    rc = d.solve_via_daemon("x", socket_path=sock)
    assert rc == 0
    t.join(timeout=5)
    assert not t.is_alive()


def test_client_fallback_when_no_daemon(sock, monkeypatch):
    # No daemon running, and spawning disabled -> client signals fallback.
    monkeypatch.setattr(d, "spawn_daemon", lambda *a, **k: False)
    assert d.solve_via_daemon("x", socket_path=sock) is None


def test_reap_stale_socket(sock):
    # A leftover socket + pid file from a dead daemon should be cleared.
    sock.write_text("")
    d._pid_path(sock).write_text("999999999")  # almost certainly not alive
    d._reap_stale(sock)
    assert not sock.exists()
    assert not d._pid_path(sock).exists()


def test_default_socket_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCOPT_GAMS_SOCKET", str(tmp_path / "custom.sock"))
    assert d.default_socket_path() == tmp_path / "custom.sock"
