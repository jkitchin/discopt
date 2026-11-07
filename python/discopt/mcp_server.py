"""A Model Context Protocol (MCP) server for discopt: ``discopt mcp``.

Lets an MCP client -- Claude Desktop, Claude Code, or any other -- build and solve
discopt models on this machine. Install with ``pip install 'discopt[mcp]'`` and
register the server with the snippet ``discopt mcp --print-config`` prints.

Tools
-----
``solve_file``
    Solve a model file: ``.nl``, ``.gms``, a saved model document
    (``.dopt`` / ``.json`` / ``.json.gz``), or a ``.py`` script that builds one.
``solve_python``
    Solve a model built by a Python source string.
``model_summary``
    Size and structure of a model, without solving it.
``convert_model``
    Convert between ``.gms``, ``.nl``, ``.mps`` and ``.lp``.
``discopt_info``
    Version and install location.

Every model is loaded and solved in a **subprocess worker**
(``python -m discopt.mcp_server --worker``), never in the server process:

* The stdio transport *is* the server's stdout, so a single byte the solver or a
  user script prints there corrupts the JSON-RPC stream. The worker sends its own
  stdout (file descriptor 1, which also catches output from compiled code) to
  stderr, and writes the result on a private descriptor.
* A solve that overruns its ``time_limit`` is killed after a grace period, so a
  wedged solve cannot hang the client.
* A crash in compiled code takes down the worker, not the server.

``solve_python`` and ``.py`` files **execute arbitrary Python** with the
permissions of the user running the server. That is the point of them -- it is how
a client writes a model -- and it is no more than the client could do with a shell
tool, but register this server only with clients you would give a shell to.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Optional

#: Seconds a worker may run past the solve's own ``time_limit`` before it is killed.
#: Loading, presolve and result serialization happen outside the solver's clock.
WORKER_GRACE_S = 60.0

#: Default ``time_limit`` for a tool call, in seconds. Clients tend to wait on a
#: tool call synchronously, so this is shorter than ``Model.solve``'s default.
DEFAULT_TIME_LIMIT_S = 60.0

#: Variables reported in a result's ``x`` before it is truncated (see ``max_vars``).
DEFAULT_MAX_VARS = 200

_MODEL_EXTS = (".nl", ".gms", ".dopt", ".json", ".json.gz", ".py")
_CONVERT_IN = (".gms", ".nl")
_CONVERT_OUT = (".gms", ".nl", ".mps", ".lp")


# ---------------------------------------------------------------------------
# Worker side: runs in a fresh interpreter, one request per process.
# ---------------------------------------------------------------------------


def _ext(path: str) -> str:
    low = path.lower()
    return ".json.gz" if low.endswith(".json.gz") else os.path.splitext(low)[1]


def _model_from_namespace(ns: dict, where: str):
    """The model a script built: ``model``, else ``m``, else the only ``Model``."""
    from discopt.modeling.core import Model

    for name in ("model", "m"):
        if isinstance(ns.get(name), Model):
            return ns[name]
    found = [v for k, v in ns.items() if isinstance(v, Model) and not k.startswith("_")]
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(f"{where} defines no discopt Model (name it `model` or `m`)")
    raise ValueError(f"{where} defines {len(found)} discopt Models; name the one to use `model`")


def _load_model(spec: dict):
    if spec.get("source") is not None:
        ns: dict[str, Any] = {"__name__": "__discopt_mcp__"}
        exec(compile(spec["source"], "<python_source>", "exec"), ns)
        return _model_from_namespace(ns, "python_source")
    path = os.path.abspath(os.path.expanduser(spec["path"]))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")
    ext = _ext(path)
    if ext == ".nl":
        from discopt.modeling.core import from_nl

        return from_nl(path)
    if ext == ".gms":
        from discopt.modeling.core import from_gams

        return from_gams(path)
    if ext in (".dopt", ".json", ".json.gz"):
        from discopt.serialize import load

        return load(path)
    if ext == ".py":
        import runpy

        ns = runpy.run_path(path, run_name="__discopt_mcp__")
        return _model_from_namespace(ns, path)
    raise ValueError(f"unsupported model file {path!r}; expected one of {', '.join(_MODEL_EXTS)}")


def _truncate_x(doc: dict, max_vars: int) -> dict:
    x = doc.get("x")
    if isinstance(x, dict) and len(x) > max_vars:
        keep = dict(list(x.items())[:max_vars])
        doc["x"] = keep
        doc["x_truncated"] = {"shown": max_vars, "total": len(x)}
    return doc


def _do_solve(spec: dict) -> dict:
    from discopt.profiles import resolve_options
    from discopt.result_io import options_from_payload, options_to_payload, serialize_result

    model = _load_model(spec)
    options = resolve_options(spec.get("profile"), spec.get("options") or {})
    payload = options_to_payload(options)
    result = model.solve(**options_from_payload(payload))
    doc = serialize_result(result, options=payload)
    return _truncate_x(doc, int(spec.get("max_vars", DEFAULT_MAX_VARS)))


def _do_summary(spec: dict) -> dict:
    model = _load_model(spec)
    return {"summary": model.summary()}


def _do_convert(spec: dict) -> dict:
    src = os.path.abspath(os.path.expanduser(spec["input_path"]))
    dst = os.path.abspath(os.path.expanduser(spec["output_path"]))
    if _ext(src) not in _CONVERT_IN:
        raise ValueError(f"input must be one of {', '.join(_CONVERT_IN)}, got {src!r}")
    out_ext = _ext(dst)
    if out_ext not in _CONVERT_OUT:
        raise ValueError(f"output must be one of {', '.join(_CONVERT_OUT)}, got {dst!r}")
    model = _load_model({"path": src})
    getattr(model, {".gms": "to_gams", ".nl": "to_nl", ".mps": "to_mps", ".lp": "to_lp"}[out_ext])(
        dst
    )
    return {"wrote": dst}


_ACTIONS = {"solve": _do_solve, "summary": _do_summary, "convert": _do_convert}


def _worker_main() -> int:
    """Run one request from stdin; write ``{"ok": ..., ...}`` to the result fd."""
    # Keep a private copy of the real stdout for the result, then point fd 1 at
    # stderr so nothing else (Python prints, solver banners, Rust/C output) can
    # reach the pipe the server parses.
    result_fd = os.dup(1)
    sys.stdout.flush()
    os.dup2(2, 1)
    try:
        spec = json.loads(sys.stdin.read())
        reply = {"ok": True, **_ACTIONS[spec["action"]](spec)}
    except Exception as exc:  # reported to the client, not swallowed
        import traceback

        traceback.print_exc()
        reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.flush()
    with os.fdopen(result_fd, "w") as out:
        json.dump(reply, out, allow_nan=False)
    return 0 if reply["ok"] else 1


# ---------------------------------------------------------------------------
# Server side.
# ---------------------------------------------------------------------------


def _stderr_tail(text: str, n: int = 30) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def run_worker(spec: dict, timeout: float) -> dict:
    """Run one request in a fresh worker process; return its reply dict.

    Never raises for a failed solve: a worker error, a crash or a timeout comes
    back as ``{"ok": False, "error": ...}`` with the tail of the worker's stderr.
    """
    cmd = [sys.executable, "-m", "discopt.mcp_server", "--worker"]
    try:
        proc = subprocess.run(
            cmd, input=json.dumps(spec), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "ok": False,
            "error": f"worker killed after {timeout:g} s (time_limit plus grace)",
            "stderr_tail": _stderr_tail(err),
        }
    try:
        reply: dict = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": f"worker exited with code {proc.returncode} and no result",
            "stderr_tail": _stderr_tail(proc.stderr),
        }
    if not reply.get("ok"):
        reply["stderr_tail"] = _stderr_tail(proc.stderr)
    return reply


def _solve_spec(
    time_limit: float,
    gap_tolerance: Optional[float],
    profile: Optional[str],
    max_vars: int,
    **where: Any,
) -> tuple[dict, float]:
    options: dict[str, Any] = {"time_limit": float(time_limit)}
    if gap_tolerance is not None:
        options["gap_tolerance"] = float(gap_tolerance)
    spec = {
        "action": "solve",
        "options": options,
        "profile": profile,
        "max_vars": int(max_vars),
        **where,
    }
    return spec, float(time_limit) + WORKER_GRACE_S


def build_server():
    """The MCP server object (raises an ImportError naming the extra if the SDK is missing)."""
    try:
        # mcp 2.x renamed FastMCP to MCPServer; the decorator and run() API are
        # unchanged, so support both majors.
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as _Server
        except ImportError as exc:
            raise ImportError(
                "`discopt mcp` needs the MCP Python SDK: pip install 'discopt[mcp]' "
                f"(import failed: {exc})"
            ) from exc

    server = _Server(
        "discopt",
        instructions=(
            "discopt is a global MINLP solver. Build models in Python with "
            "`import discopt.modeling as dm; m = dm.Model()`, then call solve_python, "
            "or point solve_file at a .nl/.gms/.dopt/.py file. A result is a proven "
            "global optimum only when status is 'optimal' and gap_certified is true; "
            "'feasible' means an incumbent without a closed gap. Report the bound and "
            "gap with the objective."
        ),
    )

    @server.tool()
    def solve_file(
        path: str,
        time_limit: float = DEFAULT_TIME_LIMIT_S,
        gap_tolerance: Optional[float] = None,
        profile: Optional[str] = None,
        max_vars: int = DEFAULT_MAX_VARS,
    ) -> dict:
        """Solve a model file with discopt.

        path: a .nl or .gms file, a saved model document (.dopt, .json, .json.gz),
        or a .py script that builds a discopt Model named `model` or `m` (the script
        is executed). Returns status, objective, bound, gap, gap_certified,
        wall_time, node_count and the solution x (truncated to max_vars entries).
        """
        spec, timeout = _solve_spec(time_limit, gap_tolerance, profile, max_vars, path=path)
        return run_worker(spec, timeout)

    @server.tool()
    def solve_python(
        source: str,
        time_limit: float = DEFAULT_TIME_LIMIT_S,
        gap_tolerance: Optional[float] = None,
        profile: Optional[str] = None,
        max_vars: int = DEFAULT_MAX_VARS,
    ) -> dict:
        """Execute Python source that builds a discopt Model, then solve it.

        The source must create a Model named `model` or `m` (or exactly one Model);
        do not call solve() yourself. Example:

            import discopt.modeling as dm
            m = dm.Model("demo")
            x = m.continuous("x", lb=0, ub=4)
            n = m.integer("n", lb=0, ub=3)
            m.minimize((x - 2.5) ** 2 + (n - 1.6) ** 2)
            m.subject_to(x + n <= 4)

        Returns the same fields as solve_file. print() output goes to the server
        log, not the result.
        """
        spec, timeout = _solve_spec(time_limit, gap_tolerance, profile, max_vars, source=source)
        return run_worker(spec, timeout)

    @server.tool()
    def model_summary(path: Optional[str] = None, source: Optional[str] = None) -> dict:
        """Describe a model (variables, constraints, structure) without solving it.

        Give exactly one of `path` (as for solve_file) or `source` (as for
        solve_python).
        """
        if (path is None) == (source is None):
            return {"ok": False, "error": "give exactly one of `path` or `source`"}
        where: dict[str, Optional[str]] = {"path": path} if path is not None else {"source": source}
        return run_worker({"action": "summary", **where}, WORKER_GRACE_S)

    @server.tool()
    def convert_model(input_path: str, output_path: str) -> dict:
        """Convert a model file. Input: .gms or .nl. Output: .gms, .nl, .mps or .lp."""
        spec = {"action": "convert", "input_path": input_path, "output_path": output_path}
        return run_worker(spec, WORKER_GRACE_S)

    @server.tool()
    def discopt_info() -> dict:
        """The discopt version, its install location, and the Python running it."""
        import discopt

        return {
            "version": getattr(discopt, "__version__", "unknown"),
            "package": os.path.dirname(discopt.__file__),
            "python": sys.executable,
        }

    return server


def desktop_config() -> dict:
    """The ``mcpServers`` entry for ``claude_desktop_config.json``.

    Uses this interpreter's absolute path: Claude Desktop does not inherit the
    shell's ``PATH``, so a bare ``discopt`` command is often not found.
    """
    return {
        "mcpServers": {"discopt": {"command": sys.executable, "args": ["-m", "discopt.cli", "mcp"]}}
    }


def serve() -> None:
    """Run the server on stdio until the client disconnects."""
    build_server().run()


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        sys.exit(_worker_main())
    serve()
