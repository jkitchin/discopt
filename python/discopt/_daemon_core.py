"""Generic warm-process solver daemon: protocol, lifecycle, and robustness.

This is the engine shared by the GAMS solver link (:mod:`discopt.gams.daemon`)
and the general ``discopt solve`` CLI (:mod:`discopt.daemon`). Both keep a single
long-lived Python+JAX process warm so a solve is a thin socket round-trip instead
of re-paying the multi-second import + first-JIT warmup every invocation.

The core is solver-agnostic: a :class:`DaemonServer` is parameterized by a
``solve_fn(request_dict) -> reply_dict`` callback and an env-var ``prefix`` for
its recycle guards. Everything else -- the length-prefixed JSON-line protocol,
the per-user unix socket, idle/lifetime/solve-count/RSS recycling, the
source-fingerprint staleness handshake, stale-socket reaping, and lazy
auto-spawn with in-process fallback -- is generic and lives here.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from discopt import __version__

_CONNECT_TIMEOUT = 5.0
_SPAWN_WAIT = 30.0


def _source_fingerprint() -> str:
    """A staleness key that changes whenever the installed discopt source changes.

    Returns ``__version__`` plus the newest modification time across the discopt
    package's Python and compiled-extension files. In a released / site-packages
    install those mtimes are fixed at install time, so the key is stable and the
    warm daemon is reused across solves. In an editable / development install,
    editing any source file (or rebuilding the Rust extension) bumps the max mtime,
    so a client built from the edited tree no longer matches a daemon spawned
    *before* the edit -- the solve handshake evicts and respawns it, and the change
    takes effect on the next solve without a manual ``daemon stop``. Falls back to
    ``__version__`` alone if the package tree cannot be scanned.
    """
    try:
        import discopt

        root = Path(discopt.__file__).resolve().parent
        latest = 0
        for pattern in ("*.py", "*.so", "*.pyd"):
            for p in root.rglob(pattern):
                try:
                    m = p.stat().st_mtime_ns
                except OSError:
                    continue
                if m > latest:
                    latest = m
        return f"{__version__}+{latest}" if latest else __version__
    except Exception:
        return __version__


# Computed once at import. A daemon stamps this as its version when it spawns; a
# client built from a newer source tree computes a different value and the version
# handshake recycles the stale daemon (see _source_fingerprint).
_FINGERPRINT = _source_fingerprint()


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


def socket_path_for(env_var: str, basename: str) -> Path:
    """Per-user socket path, honouring ``env_var`` then ``XDG_RUNTIME_DIR``/``TMPDIR``."""
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path(os.environ.get("TMPDIR", "/tmp"))
    return base / f"{basename}-{os.getuid()}.sock"


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


class _DeadlineWatchdog:
    """Fork a child that ``SIGKILL``s *this* process after ``deadline`` seconds
    unless cancelled. The killer is a separate process, so it stops even a solve
    wedged deep in a C extension (XLA / Rust) that no in-process signal or thread
    could interrupt -- making the deadline hard and independent of the solver.

    The child only sleeps, checks parentage, and kills (no JAX/locks touched), so
    forking it from the warm JAX process is safe. On normal completion the daemon
    cancels it (``__exit__``); on overrun the child kills the daemon, its socket
    drops, and the client falls back. ``deadline <= 0`` (or no ``os.fork``) is a
    no-op -- the default, so there is no hard limit unless one is requested.
    """

    def __init__(self, deadline: float):
        self.deadline = float(deadline)
        self.pid: int | None = None

    def __enter__(self) -> "_DeadlineWatchdog":
        if self.deadline <= 0 or not hasattr(os, "fork"):
            return self
        parent = os.getpid()
        pid = os.fork()
        if pid == 0:  # watchdog child
            try:
                time.sleep(self.deadline)
                # Guard against PID reuse: only kill if the daemon is still our
                # parent (if it already exited we were reparented away from it).
                if os.getppid() == parent:
                    os.kill(parent, signal.SIGKILL)
            except BaseException:
                pass
            finally:
                os._exit(0)
        self.pid = pid
        return self

    def __exit__(self, *exc) -> None:
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.waitpid(self.pid, 0)
            except OSError:
                pass


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
    """A warm, single-request-at-a-time solver server over a unix socket.

    ``solve_fn`` receives the full request dict (``cmd == "solve"``) and returns
    the reply payload dict to send back -- so a caller can return a rich result,
    not just an exit code. ``env_prefix`` selects which ``<PREFIX>_*`` env vars
    drive the recycle guards (e.g. ``DISCOPT_GAMS`` vs ``DISCOPT_SOLVE``).
    """

    def __init__(
        self,
        socket_path: Path,
        solve_fn: Callable[[dict], dict],
        *,
        env_prefix: str,
        version: str = _FINGERPRINT,
        idle_timeout: float | None = None,
        max_lifetime: float | None = None,
        max_solves: int | None = None,
        max_rss_mb: int | None = None,
        jax_clear_every: int | None = None,
    ):
        # The benchmark preset disables count/age recycling by default so one warm
        # daemon spans a whole study (an RSS ceiling remains the backstop).
        bench = _env_flag(f"{env_prefix}_BENCHMARK")
        self.socket_path = Path(socket_path)
        self._solve_fn = solve_fn
        # 0 / negative means "no limit" for every guard below.
        self.idle_timeout = _resolve(
            idle_timeout, f"{env_prefix}_IDLE_TIMEOUT", 1800 if bench else 600, float
        )
        self.max_lifetime = _resolve(
            max_lifetime, f"{env_prefix}_MAX_LIFETIME", 0 if bench else 3600, float
        )
        self.max_solves = _resolve(max_solves, f"{env_prefix}_MAX_SOLVES", 0 if bench else 500, int)
        self.max_rss_mb = _resolve(max_rss_mb, f"{env_prefix}_MAX_RSS_MB", 0, int)
        self.jax_clear_every = _resolve(jax_clear_every, f"{env_prefix}_JAX_CLEAR_EVERY", 0, int)
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
            # A client built against a different discopt evicts the daemon so the
            # upgrade takes effect; it serves this request first.
            stop = req.get("version") not in (None, self.version)
            # Optional daemon-side hard deadline (default: none -> no limit). A
            # forked watchdog SIGKILLs the daemon if this solve overruns, enforcing
            # the limit independently of the solver AND of any waiting client.
            try:
                deadline = float(req.get("hard_deadline") or 0.0)
            except (TypeError, ValueError):
                deadline = 0.0
            try:
                with _DeadlineWatchdog(deadline):
                    reply = self._solve_fn(req)
            except Exception as exc:  # never let one model take the daemon down
                reply = {"ok": False, "error": repr(exc)}
            self._reply(conn, reply)
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


# ── Client ───────────────────────────────────────────────────────────────────
class SolveTimeout(Exception):
    """The daemon accepted a request but did not reply within ``recv_timeout`` --
    a wedged or runaway solve. The caller should kill and recycle the daemon
    rather than wait forever (a single-threaded daemon serves one solve at a time,
    so one hang blocks every other client too)."""


def request(
    socket_path: Path,
    payload: dict,
    timeout: float = _CONNECT_TIMEOUT,
    recv_timeout: float | None = None,
) -> dict | None:
    """Send one request to a daemon socket, return its reply (or ``None``).

    ``timeout`` bounds the connect (so an absent daemon returns ``None`` fast).
    ``recv_timeout`` bounds the wait for the reply *after* sending: ``None`` (the
    default, used by fast ping/stop) blocks indefinitely; a value raises
    :class:`SolveTimeout` if no reply arrives in time -- distinct from an
    unreachable daemon -- so a stuck solve can be killed instead of hanging.
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(socket_path))
            s.sendall((json.dumps(payload) + "\n").encode())
            s.settimeout(recv_timeout)  # None -> block; else bound the solve wait
            try:
                data = _recv_line(s)
            except socket.timeout:
                raise SolveTimeout() from None
        return json.loads(data.decode()) if data else None
    except SolveTimeout:
        raise
    except (OSError, ValueError):
        return None


def ping(socket_path: Path) -> dict | None:
    """Return the running daemon's handshake, or ``None`` if not reachable."""
    return request(socket_path, {"cmd": "ping"})


def stop_daemon(socket_path: Path) -> bool:
    """Ask a running daemon to shut down. Returns True if it acknowledged.

    Graceful but cooperative: a daemon wedged in a solve is not accepting
    connections and will not hear this. Use :func:`kill_daemon` for that case.
    """
    resp = request(socket_path, {"cmd": "stop"})
    return bool(resp and resp.get("ok"))


def kill_daemon(socket_path: Path) -> bool:
    """Forcibly terminate a daemon by its PID file (SIGTERM, then SIGKILL).

    Works when the daemon is wedged in a runaway solve and cannot answer
    :func:`stop`. Reaps the socket/PID files afterward. Returns True if a live
    process was signalled.
    """
    pid_file = _pid_path(socket_path)
    try:
        pid = int(pid_file.read_text())
    except (OSError, ValueError):
        pid = None
    killed = False
    if pid is not None and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.05)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        killed = True
    for p in (socket_path, pid_file):
        try:
            p.unlink()
        except OSError:
            pass
    return killed


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


def spawn_daemon(
    socket_path: Path, module: str, socket_env: str, wait: float = _SPAWN_WAIT
) -> bool:
    """Start a detached ``python -m <module> serve`` daemon; wait for its socket."""
    _reap_stale(socket_path)
    env = dict(os.environ, **{socket_env: str(socket_path)})
    try:
        subprocess.Popen(
            [sys.executable, "-m", module, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # outlive the launching client
            env=env,
        )
    except OSError:
        return False
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if ping(socket_path):
            return True
        time.sleep(0.05)
    return False
