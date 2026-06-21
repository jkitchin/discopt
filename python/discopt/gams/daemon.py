"""Warm solver daemon for the discopt GAMS link.

GAMS launches a solver as a fresh process per solve, so each solve otherwise
re-pays Python + JAX import and first-JIT warmup (seconds), even though a warm
solve is ~10 ms.  This module keeps a single long-lived process warm and turns
the per-solve launch into a thin socket round-trip.

The generic protocol / lifecycle / robustness machinery now lives in
:mod:`discopt._daemon_core`; this module is the GAMS-specific layer (the
control-file ``solve_fn``, the ``discopt-gams`` socket name, and the
``(control_file, sysdir) -> int`` client contract the GAMS link relies on). It
re-exports the core helpers under their historical names so existing imports
keep working.
"""

from __future__ import annotations

import sys
from pathlib import Path

from discopt import __version__
from discopt._daemon_core import (
    _FINGERPRINT,
    _SPAWN_WAIT,
    _clear_jax_caches,
    _current_rss_mb,
    _env_flag,
    _pid_alive,
    _pid_path,
    _reap_stale,
    _recv_line,
    _resolve,
    _source_fingerprint,
    socket_path_for,
)
from discopt._daemon_core import (
    DaemonServer as _CoreDaemonServer,
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
    "DaemonServer",
    "default_socket_path",
    "ping",
    "stop_daemon",
    "spawn_daemon",
    "solve_via_daemon",
    "main",
]

# Silence "imported but unused" for the names we re-export for back-compat.
_REEXPORTS = (
    _FINGERPRINT,
    _source_fingerprint,
    _env_flag,
    _resolve,
    _recv_line,
    _pid_path,
    _pid_alive,
    _reap_stale,
    _request,
    _clear_jax_caches,
    _current_rss_mb,
    __version__,
)

_MODULE = "discopt.gams.daemon"
_SOCKET_ENV = "DISCOPT_GAMS_SOCKET"


def default_socket_path() -> Path:
    """Per-user socket path (honours ``DISCOPT_GAMS_SOCKET`` / ``XDG_RUNTIME_DIR``)."""
    return socket_path_for(_SOCKET_ENV, "discopt-gams")


class DaemonServer(_CoreDaemonServer):
    """Warm GAMS solver server. Wraps the historical ``(control_file, sysdir) ->
    int`` solve callback into the core's ``request -> reply`` contract so the GAMS
    link and its tests keep their signature."""

    def __init__(
        self,
        socket_path: Path | None = None,
        solve_fn=None,
        idle_timeout: float | None = None,
        max_lifetime: float | None = None,
        max_solves: int | None = None,
        max_rss_mb: int | None = None,
        jax_clear_every: int | None = None,
        version: str = _FINGERPRINT,
    ):
        sp = Path(socket_path) if socket_path else default_socket_path()
        legacy = solve_fn or _default_solve_fn

        def _req_solve(req: dict) -> dict:
            cf = req.get("control_file", "")
            sysdir = req.get("sysdir")
            try:
                rc = int(legacy(cf, sysdir))
                return {"ok": rc == 0, "rc": rc}
            except Exception as exc:  # surfaced as a failed solve, daemon survives
                return {"ok": False, "rc": 1, "error": repr(exc)}

        super().__init__(
            sp,
            _req_solve,
            env_prefix="DISCOPT_GAMS",
            version=version,
            idle_timeout=idle_timeout,
            max_lifetime=max_lifetime,
            max_solves=max_solves,
            max_rss_mb=max_rss_mb,
            jax_clear_every=jax_clear_every,
        )


def _default_solve_fn(control_file: str, sysdir: str | None = None) -> int:
    # Imported lazily so the module (and tests) don't require gamsapi.
    from .link import solve_from_control_file

    return solve_from_control_file(control_file, sysdir)


# ── Client ───────────────────────────────────────────────────────────────────
def ping(socket_path: Path | None = None) -> dict | None:
    """Return the running daemon's handshake, or ``None`` if not reachable."""
    return _core_ping(socket_path or default_socket_path())


def stop_daemon(socket_path: Path | None = None) -> bool:
    """Ask a running daemon to shut down. Returns True if it acknowledged."""
    return _core_stop(socket_path or default_socket_path())


def kill_daemon(socket_path: Path | None = None) -> bool:
    """SIGTERM/SIGKILL the daemon by its PID file (use when it is wedged)."""
    return _core_kill(socket_path or default_socket_path())


def spawn_daemon(socket_path: Path | None = None, wait: float = _SPAWN_WAIT) -> bool:
    """Start a detached GAMS daemon and wait until its socket answers."""
    return _core_spawn(socket_path or default_socket_path(), _MODULE, _SOCKET_ENV, wait)


def solve_via_daemon(
    control_file: str, sysdir: str | None = None, socket_path: Path | None = None
) -> int | None:
    """Solve ``control_file`` through the warm daemon, spawning it if needed.

    Returns the solver exit code, or ``None`` if the daemon is unreachable and
    could not be started -- the caller should then solve in-process.
    """
    path = socket_path or default_socket_path()
    # Recycle a daemon running stale code BEFORE solving, so the current code is
    # used on this very solve (not one solve later). A running daemon stamps the
    # source fingerprint it was spawned with; if ours differs, stop it and spawn
    # fresh first. In a stable install the fingerprint never changes, so this is a
    # one-ping no-op. (The daemon's own version check remains a backstop.)
    info = ping(path)
    if info is not None and info.get("version") not in (None, _FINGERPRINT):
        stop_daemon(path)
        spawn_daemon(path)
    payload = {
        "cmd": "solve",
        "control_file": control_file,
        "sysdir": sysdir,
        "version": _FINGERPRINT,
    }
    resp = _request(path, payload)
    if resp is None:
        if not spawn_daemon(path):
            return None
        resp = _request(path, payload)
    if resp is None:
        return None
    return int(resp.get("rc", 1))


# ── Entry points ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """``python -m discopt.gams.daemon {serve,stop,status}``."""
    args = sys.argv[1:] if argv is None else argv
    cmd = args[0] if args else "serve"
    if cmd == "serve":
        DaemonServer().serve_forever()
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
    sys.stderr.write("usage: python -m discopt.gams.daemon {serve,stop,kill,status}\n")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
