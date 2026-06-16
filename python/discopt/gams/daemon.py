"""Warm solver daemon for the discopt GAMS link.

GAMS launches a solver as a fresh process per solve, so each solve otherwise
re-pays Python + JAX import and first-JIT warmup (seconds), even though a warm
solve is ~10 ms.  This module keeps a single long-lived process warm and turns
the per-solve launch into a thin socket round-trip.

Pieces
------
* :class:`DaemonServer` -- listens on a per-user unix socket, services one
  request at a time (solves are serialized: one warm interpreter, GIL + JAX
  thread-safety), and self-terminates on idle timeout / max lifetime / max
  solves / version mismatch.
* :func:`solve_via_daemon` -- the client used by the GAMS link: connect (lazily
  spawning a detached daemon if needed), relay the control-file path, return the
  exit code.  Returns ``None`` to signal the caller should fall back to an
  in-process solve, so correctness never depends on the daemon being up.

The lifecycle (idle timeout + lazy auto-respawn + direct-solve fallback) is
exercised in tests with a fake solve function, so it needs no GAMS install.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from discopt import __version__

# Defaults (all overridable via env vars).
DEFAULT_IDLE_TIMEOUT = float(os.environ.get("DISCOPT_GAMS_IDLE_TIMEOUT", "600"))  # 10 min
DEFAULT_MAX_LIFETIME = float(os.environ.get("DISCOPT_GAMS_MAX_LIFETIME", "3600"))  # 1 h
DEFAULT_MAX_SOLVES = int(os.environ.get("DISCOPT_GAMS_MAX_SOLVES", "500"))
_CONNECT_TIMEOUT = 5.0
_SPAWN_WAIT = 30.0


def default_socket_path() -> Path:
    """Per-user socket path (honours ``DISCOPT_GAMS_SOCKET`` / ``XDG_RUNTIME_DIR``)."""
    explicit = os.environ.get("DISCOPT_GAMS_SOCKET")
    if explicit:
        return Path(explicit)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path(os.environ.get("TMPDIR", "/tmp"))
    return base / f"discopt-gams-{os.getuid()}.sock"


def _pid_path(socket_path: Path) -> Path:
    return socket_path.with_suffix(socket_path.suffix + ".pid")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recv_line(conn: socket.socket) -> bytes:
    chunks: list[bytes] = []
    while True:
        b = conn.recv(65536)
        if not b:
            break
        chunks.append(b)
        if b.endswith(b"\n"):
            break
    return b"".join(chunks)


# ── Server ───────────────────────────────────────────────────────────────────
class DaemonServer:
    """A warm, single-request-at-a-time solver server over a unix socket."""

    def __init__(
        self,
        socket_path: Path | None = None,
        solve_fn: Callable[[str, str | None], int] | None = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        max_lifetime: float = DEFAULT_MAX_LIFETIME,
        max_solves: int = DEFAULT_MAX_SOLVES,
        version: str = __version__,
    ):
        self.socket_path = Path(socket_path) if socket_path else default_socket_path()
        self._solve_fn = solve_fn or _default_solve_fn
        self.idle_timeout = idle_timeout
        self.max_lifetime = max_lifetime
        self.max_solves = max_solves
        self.version = version
        self.solves = 0
        self._sock: socket.socket | None = None

    def _bind(self) -> None:
        # Remove a stale socket from a dead daemon before binding.
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        s.listen(16)
        self._sock = s
        _pid_path(self.socket_path).write_text(str(os.getpid()))

    def serve_forever(self) -> None:
        """Accept and service requests until a shutdown condition fires."""
        self._bind()
        started = time.monotonic()
        try:
            while True:
                assert self._sock is not None
                self._sock.settimeout(self.idle_timeout)
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    break  # idle timeout
                with conn:
                    stop = self._handle(conn)
                if stop:
                    break
                if self.solves >= self.max_solves:
                    break
                if time.monotonic() - started >= self.max_lifetime:
                    break
        finally:
            self._cleanup()

    def _handle(self, conn: socket.socket) -> bool:
        """Service one request. Returns True if the daemon should stop."""
        try:
            req = json.loads(_recv_line(conn).decode() or "{}")
        except (ValueError, OSError):
            self._reply(conn, {"ok": False, "error": "bad request"})
            return False

        cmd = req.get("cmd")
        if cmd == "ping":
            self._reply(conn, {"ok": True, "version": self.version})
            return False
        if cmd == "stop":
            self._reply(conn, {"ok": True})
            return True
        if cmd == "solve":
            # A client built against a different discopt evicts the daemon so
            # the upgrade takes effect; it serves this request first.
            stop = req.get("version") not in (None, self.version)
            control_file = req.get("control_file", "")
            sysdir = req.get("sysdir")
            try:
                rc = int(self._solve_fn(control_file, sysdir))
                self._reply(conn, {"ok": rc == 0, "rc": rc})
            except Exception as exc:  # never let one model take the daemon down
                self._reply(conn, {"ok": False, "rc": 1, "error": repr(exc)})
            self.solves += 1
            return stop
        self._reply(conn, {"ok": False, "error": f"unknown cmd {cmd!r}"})
        return False

    @staticmethod
    def _reply(conn: socket.socket, payload: dict) -> None:
        try:
            conn.sendall((json.dumps(payload) + "\n").encode())
        except OSError:
            pass

    def _cleanup(self) -> None:
        if self._sock is not None:
            self._sock.close()
        for p in (self.socket_path, _pid_path(self.socket_path)):
            try:
                p.unlink()
            except OSError:
                pass


def _default_solve_fn(control_file: str, sysdir: str | None = None) -> int:
    # Imported lazily so the module (and tests) don't require gamsapi.
    from .link import solve_from_control_file

    return solve_from_control_file(control_file, sysdir)


# ── Client ───────────────────────────────────────────────────────────────────
def _request(socket_path: Path, payload: dict, timeout: float = _CONNECT_TIMEOUT) -> dict | None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(socket_path))
            s.sendall((json.dumps(payload) + "\n").encode())
            s.settimeout(None)  # the solve itself may take a while
            data = _recv_line(s)
        return json.loads(data.decode()) if data else None
    except (OSError, ValueError):
        return None


def ping(socket_path: Path | None = None) -> dict | None:
    """Return the running daemon's handshake, or ``None`` if not reachable."""
    return _request(socket_path or default_socket_path(), {"cmd": "ping"})


def stop_daemon(socket_path: Path | None = None) -> bool:
    """Ask a running daemon to shut down. Returns True if it acknowledged."""
    resp = _request(socket_path or default_socket_path(), {"cmd": "stop"})
    return bool(resp and resp.get("ok"))


def spawn_daemon(socket_path: Path | None = None, wait: float = _SPAWN_WAIT) -> bool:
    """Start a detached daemon and wait until its socket answers."""
    path = socket_path or default_socket_path()
    _reap_stale(path)
    env = dict(os.environ, DISCOPT_GAMS_SOCKET=str(path))
    try:
        subprocess.Popen(
            [sys.executable, "-m", "discopt.gams.daemon", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # outlive the GAMS-launched client
            env=env,
        )
    except OSError:
        return False
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if ping(path):
            return True
        time.sleep(0.05)
    return False


def _reap_stale(socket_path: Path) -> None:
    """Remove a socket/pid left behind by a dead daemon."""
    pid_file = _pid_path(socket_path)
    try:
        pid = int(pid_file.read_text())
    except (OSError, ValueError):
        pid = None
    if pid is not None and _pid_alive(pid):
        return  # a live daemon owns it
    for p in (socket_path, pid_file):
        try:
            p.unlink()
        except OSError:
            pass


def solve_via_daemon(
    control_file: str, sysdir: str | None = None, socket_path: Path | None = None
) -> int | None:
    """Solve ``control_file`` through the warm daemon, spawning it if needed.

    Returns the solver exit code, or ``None`` if the daemon is unreachable and
    could not be started -- the caller should then solve in-process.
    """
    path = socket_path or default_socket_path()
    payload = {
        "cmd": "solve",
        "control_file": control_file,
        "sysdir": sysdir,
        "version": __version__,
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
    if cmd == "status":
        info = ping()
        if info:
            print(f"running (version {info.get('version')}) at {default_socket_path()}")
            return 0
        print("not running")
        return 1
    sys.stderr.write("usage: python -m discopt.gams.daemon {serve,stop,status}\n")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
