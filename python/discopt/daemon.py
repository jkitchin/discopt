"""Warm ``discopt solve`` daemon.

Keeps a single long-lived Python+JAX process warm so ``discopt solve <file.nl>``
is a thin socket round-trip (~0.1 s) instead of re-paying the ~1 s JAX backend
init + tracing on every invocation. The generic protocol / lifecycle / robustness
engine lives in :mod:`discopt._daemon_core`; this module is the thin solve-specific
layer (an ``.nl`` + options ``solve_fn`` that returns a serialized result, the
``discopt-solve`` socket name, and the client used by the CLI).

Client contract: :func:`solve_via_daemon` returns the reply dict
(``{"ok": True, "result": {...}}`` or ``{"ok": False, "error": ...}``), or
``None`` if the daemon is unreachable and could not be started -- the CLI then
solves in-process, so correctness never depends on the daemon being up.

Lifecycle knobs (mirroring the GAMS daemon, ``DISCOPT_SOLVE_*``): idle timeout
(default 600 s), max lifetime (3600 s), max solves (500), max RSS, JAX-clear
stride, plus the source-fingerprint recycle that evicts a daemon running stale
code after an editable-install edit.
"""

from __future__ import annotations

import sys
from pathlib import Path

from discopt._daemon_core import (
    _FINGERPRINT,
    _SPAWN_WAIT,
    DaemonServer,
    SolveTimeout,
    socket_path_for,
)
from discopt._daemon_core import (
    kill_daemon as _core_kill,
)
from discopt._daemon_core import (
    ping as _core_ping,
)
from discopt._daemon_core import (
    request as _request,
)
from discopt._daemon_core import (
    spawn_daemon as _core_spawn,
)
from discopt._daemon_core import (
    stop_daemon as _core_stop,
)

__all__ = [
    "default_socket_path",
    "ping",
    "stop_daemon",
    "kill_daemon",
    "spawn_daemon",
    "solve_via_daemon",
    "make_server",
    "main",
]

_MODULE = "discopt.daemon"
_SOCKET_ENV = "DISCOPT_SOLVE_SOCKET"
# Extra wall-clock the client allows a solve beyond its own ``time_limit`` before
# declaring the daemon wedged: covers a cold daemon's JAX warmup/compile and
# result serialization. Past ``time_limit + _SOLVE_GRACE`` the client kills the
# daemon and falls back in-process, so one runaway solve can never wedge a run.
_SOLVE_GRACE = 120.0


def default_socket_path() -> Path:
    """Per-user solve-daemon socket (honours ``DISCOPT_SOLVE_SOCKET``)."""
    return socket_path_for(_SOCKET_ENV, "discopt-solve")


def _solve_request(req: dict) -> dict:
    """Daemon ``solve_fn``: build the model from the ``.nl`` and solve it.

    Path resolution and any file output stay on the CLI side (the daemon may run
    with a different cwd); the daemon is a pure executor returning a serialized
    result. Exceptions propagate to the core, which isolates them so one bad model
    never takes the daemon down.
    """
    from typing import cast

    from discopt.modeling.core import SolveResult, from_nl
    from discopt.result_io import options_from_payload, serialize_result

    nl_file = req["nl_file"]
    options = options_from_payload(req.get("options") or {})
    # The daemon never passes stream=True, so solve() returns a SolveResult.
    result = cast(SolveResult, from_nl(nl_file).solve(**options))
    return {"ok": True, "result": serialize_result(result)}


def make_server(socket_path: Path | None = None) -> DaemonServer:
    return DaemonServer(
        socket_path or default_socket_path(),
        _solve_request,
        env_prefix="DISCOPT_SOLVE",
    )


# ── Client ───────────────────────────────────────────────────────────────────
def ping(socket_path: Path | None = None) -> dict | None:
    return _core_ping(socket_path or default_socket_path())


def stop_daemon(socket_path: Path | None = None) -> bool:
    return _core_stop(socket_path or default_socket_path())


def kill_daemon(socket_path: Path | None = None) -> bool:
    """SIGTERM/SIGKILL the solve daemon by its PID file (use when it is wedged)."""
    return _core_kill(socket_path or default_socket_path())


def spawn_daemon(socket_path: Path | None = None, wait: float = _SPAWN_WAIT) -> bool:
    return _core_spawn(socket_path or default_socket_path(), _MODULE, _SOCKET_ENV, wait)


def _recv_timeout_for(options: dict | None) -> float:
    try:
        tl = float((options or {}).get("time_limit", 3600.0))
    except (TypeError, ValueError):
        tl = 3600.0
    return max(tl, 0.0) + _SOLVE_GRACE


def solve_via_daemon(
    nl_file: str,
    options: dict | None = None,
    socket_path: Path | None = None,
    hard_deadline: float | None = None,
) -> dict | None:
    """Solve ``nl_file`` through the warm daemon, spawning it if needed.

    ``options`` is the JSON-safe solve-options payload (see
    :func:`discopt.result_io.options_to_payload`). Returns the reply dict, or
    ``None`` if the daemon is unreachable, could not be started, or **wedged** on a
    runaway solve past ``time_limit + grace`` -- in which case it is killed and the
    caller falls back to an in-process solve, so one stuck model can never block a
    run.

    ``hard_deadline`` (seconds, default ``None`` = no limit) additionally asks the
    *daemon* to fork a watchdog that ``SIGKILL``s itself if this solve overruns,
    enforcing the limit even with no client waiting (orphaned worker) and even for
    a solver wedged in a C extension.
    """
    path = socket_path or default_socket_path()
    recv_timeout = _recv_timeout_for(options)
    # Version-check the daemon before solving (evict stale code on editable
    # installs). Use a BOUNDED probe, not the unbounded ``ping``: a single-threaded
    # daemon already wedged in another solve will not answer the probe either, so
    # an unbounded wait here would hang the client forever. If the probe times out,
    # treat the daemon as wedged -- kill and respawn a fresh one.
    try:
        info = _request(path, {"cmd": "ping"}, recv_timeout=recv_timeout)
    except SolveTimeout:
        kill_daemon(path)
        spawn_daemon(path)
        info = None
    if info is not None and info.get("version") not in (None, _FINGERPRINT):
        stop_daemon(path)
        spawn_daemon(path)
    payload: dict[str, object] = {
        "cmd": "solve",
        "nl_file": nl_file,
        "options": options or {},
        "version": _FINGERPRINT,
    }
    if hard_deadline is not None and hard_deadline > 0:
        payload["hard_deadline"] = float(hard_deadline)
        # The client must wait at least as long as the daemon's own deadline,
        # else the client would give up (and kill) before the watchdog fires.
        recv_timeout = max(recv_timeout, float(hard_deadline) + 5.0)
    try:
        resp = _request(path, payload, recv_timeout=recv_timeout)
    except SolveTimeout:
        kill_daemon(path)  # wedged -> kill so the next solve gets a fresh daemon
        return None
    if resp is None:
        if not spawn_daemon(path):
            return None
        try:
            resp = _request(path, payload, recv_timeout=recv_timeout)
        except SolveTimeout:
            kill_daemon(path)
            return None
    return resp


# ── Entry points ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """``python -m discopt.daemon {serve,stop,status}``."""
    args = sys.argv[1:] if argv is None else argv
    cmd = args[0] if args else "serve"
    if cmd == "serve":
        make_server().serve_forever()
        return 0
    if cmd == "stop":
        ok = stop_daemon()
        print("stopped" if ok else "no running daemon")
        return 0 if ok else 1
    if cmd == "kill":
        ok = kill_daemon()
        print("killed" if ok else "no running daemon")
        return 0 if ok else 1
    if cmd == "status":
        info = ping()
        if info:
            print(f"running (version {info.get('version')}) at {default_socket_path()}")
            return 0
        print("not running")
        return 1
    sys.stderr.write("usage: python -m discopt.daemon {serve,stop,kill,status}\n")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
