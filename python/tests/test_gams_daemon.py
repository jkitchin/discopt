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


def test_version_eviction(sock, monkeypatch):
    # A daemon advertising a different fingerprint is running stale code. The
    # client must recycle it BEFORE solving, rather than trusting its handshake
    # to serve-then-stop one solve later. Stub spawn so no real daemon launches;
    # we only assert the stale one was stopped and a fresh spawn was attempted.
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0, version="9.9.9")
    t = _start(srv)
    spawns = []
    monkeypatch.setattr(d, "spawn_daemon", lambda *a, **k: (spawns.append(True), False)[1])
    rc = d.solve_via_daemon("x", socket_path=sock)
    t.join(timeout=5)
    assert not t.is_alive()  # the stale daemon was stopped by the pre-check
    assert spawns  # client attempted to spawn a fresh daemon
    assert rc is None  # no live daemon after the stubbed spawn -> in-process fallback
    assert d.ping(sock) is None


def test_handle_serves_then_stops_on_version_mismatch(sock):
    # Server-side backstop for older clients that do NOT pre-check: a solve
    # request carrying a mismatched version is served, then the daemon exits.
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0)
    t = _start(srv)
    resp = d._request(
        sock, {"cmd": "solve", "control_file": "x", "sysdir": None, "version": "0.0.0"}
    )
    assert resp is not None and resp.get("rc") == 0
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


# ── hardening: RSS guard, unlimited semantics, benchmark preset ──────────────
def test_rss_guard_recycles(sock):
    # A 1 MiB ceiling is always exceeded -> daemon exits after one solve.
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0, max_rss_mb=1)
    t = _start(srv)
    assert d.solve_via_daemon("x", socket_path=sock) == 0
    t.join(timeout=5)
    assert not t.is_alive()


def test_zero_limits_mean_unlimited(sock):
    srv = d.DaemonServer(
        socket_path=sock,
        solve_fn=lambda cf, sysdir=None: 0,
        max_solves=0,
        max_lifetime=0,
        idle_timeout=0,  # block forever waiting for requests
    )
    t = _start(srv)
    try:
        for _ in range(5):
            assert d.solve_via_daemon("x", socket_path=sock) == 0
        assert d.ping(sock) is not None  # still alive after many solves
    finally:
        d.stop_daemon(sock)
        t.join(timeout=5)


def test_benchmark_preset_disables_recycling(monkeypatch, sock):
    monkeypatch.setenv("DISCOPT_GAMS_BENCHMARK", "1")
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0)
    assert srv.max_solves == 0  # unlimited
    assert srv.max_lifetime == 0
    assert srv.idle_timeout == 1800


def test_explicit_arg_overrides_benchmark_env(monkeypatch, sock):
    monkeypatch.setenv("DISCOPT_GAMS_BENCHMARK", "1")
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0, max_solves=7)
    assert srv.max_solves == 7


def test_jax_clear_every_is_safe_without_jax(sock):
    # jax not loaded -> clearing is a no-op and must not crash a solve.
    srv = d.DaemonServer(socket_path=sock, solve_fn=lambda cf, sysdir=None: 0, jax_clear_every=1)
    t = _start(srv)
    try:
        assert d.solve_via_daemon("x", socket_path=sock) == 0
    finally:
        d.stop_daemon(sock)
        t.join(timeout=5)


def test_source_fingerprint_is_stable_and_edit_sensitive(tmp_path):
    """The daemon's staleness key (its handshake ``version``) is stable across
    calls but changes when a discopt source file is edited -- so a warm daemon
    spawned from an editable install is recycled after an edit instead of serving
    stale code. Socket-free (no AF_UNIX bind)."""
    import os

    import discopt

    fp = d._source_fingerprint()
    assert fp == d._source_fingerprint()  # deterministic across calls
    assert fp.startswith(d.__version__)  # carries the package version
    # A fresh DaemonServer stamps the fingerprint as its handshake version.
    assert d.DaemonServer(socket_path=tmp_path / "x.sock").version == d._FINGERPRINT
    # Bumping a discopt source file's mtime changes the fingerprint -> recycle.
    src = os.path.join(os.path.dirname(discopt.__file__), "gams", "daemon.py")
    os.utime(src, None)
    assert d._source_fingerprint() != fp
