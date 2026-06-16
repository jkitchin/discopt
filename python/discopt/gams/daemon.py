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

_CONNECT_TIMEOUT = 5.0
_SPAWN_WAIT = 30.0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def _resolve(explicit, env_name: str, default, cast):
    """Pick an explicit arg, else an env override, else a default; ``cast`` typed."""
    if explicit is not None:
        return cast(explicit)
    raw = os.environ.get(env_name)
    if raw is not None:
        return cast(raw)
    return cast(default)


def _current_rss_mb() -> float:
    """Resident set size of this process in MiB (peak; monotonic, no psutil dep)."""
    import resource

    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return kb / (1024 * 1024) if sys.platform == "darwin" else kb / 1024


def _clear_jax_caches() -> None:
    """Drop JAX's compilation caches if (and only if) JAX is loaded."""
    mod = sys.modules.get("jax")
    if mod is not None:
        try:
            mod.clear_caches()
        except Exception:
            pass


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
        idle_timeout: float | None = None,
        max_lifetime: float | None = None,
        max_solves: int | None = None,
        max_rss_mb: int | None = None,
        jax_clear_every: int | None = None,
        version: str = __version__,
    ):
        # The benchmark preset disables count/age recycling by default so one
        # warm daemon spans a whole study (an RSS ceiling remains the backstop).
        bench = _env_flag("DISCOPT_GAMS_BENCHMARK")
        self.socket_path = Path(socket_path) if socket_path else default_socket_path()
        self._solve_fn = solve_fn or _default_solve_fn
        # 0 / negative means "no limit" for every guard below.
        self.idle_timeout = _resolve(
            idle_timeout, "DISCOPT_GAMS_IDLE_TIMEOUT", 1800 if bench else 600, float
        )
        self.max_lifetime = _resolve(
            max_lifetime, "DISCOPT_GAMS_MAX_LIFETIME", 0 if bench else 3600, float
        )
        self.max_solves = _resolve(max_solves, "DISCOPT_GAMS_MAX_SOLVES", 0 if bench else 500, int)
        self.max_rss_mb = _resolve(max_rss_mb, "DISCOPT_GAMS_MAX_RSS_MB", 0, int)
        self.jax_clear_every = _resolve(jax_clear_every, "DISCOPT_GAMS_JAX_CLEAR_EVERY", 0, int)
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
                # idle_timeout <= 0 means block indefinitely (no idle shutdown).
                self._sock.settimeout(self.idle_timeout if self.idle_timeout > 0 else None)
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    break  # idle timeout
                before = self.solves
                with conn:
                    stop = self._handle(conn)
                did_solve = self.solves > before
                # Recycle guards are evaluated only after an actual solve, so
                # pings/stops never trip them. A purely idle daemon exits via the
                # accept() idle timeout instead.
                if (
                    did_solve
                    and self.jax_clear_every > 0
                    and (self.solves % self.jax_clear_every == 0)
                ):
                    _clear_jax_caches()
                if stop:
                    break
                if did_solve:
                    if self.max_solves > 0 and self.solves >= self.max_solves:
                        break
                    if self.max_lifetime > 0 and time.monotonic() - started >= self.max_lifetime:
                        break
                    if self.max_rss_mb > 0 and _current_rss_mb() >= self.max_rss_mb:
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
