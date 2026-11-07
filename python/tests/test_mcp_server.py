"""``discopt mcp``: the Model Context Protocol server (``discopt/mcp_server.py``).

The end-to-end tests drive the real server over stdio with the MCP SDK's own
client, so they exercise what Claude Desktop does: spawn ``python -m discopt.cli
mcp``, list the tools, call them, parse JSON-RPC off the server's stdout. The
noisy-model case is the one the worker design exists for -- a model that prints
to stdout during the solve must not corrupt that stream.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

import anyio  # noqa: E402  (an mcp dependency)
from discopt import mcp_server  # noqa: E402

REPO_PY = str(Path(mcp_server.__file__).resolve().parents[1])
ALAN = Path(__file__).parent / "data" / "minlplib_nl" / "alan.nl"
ALAN_OPT = 2.925  # minlplib.solu

# Optimum 0.36 at n = 1, x = 2.5 (n = 2 forces x <= 2: 0.25 + 0.16 = 0.41).
DEMO = """\
import discopt.modeling as dm
print("model script chatter on stdout")
m = dm.Model("demo")
x = m.continuous("x", lb=0, ub=4)
n = m.integer("n", lb=0, ub=3)
m.minimize((x - 2.5) ** 2 + (n - 1.6) ** 2)
m.subject_to(x + n <= 4)
"""


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_PY + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _call_tools(calls):
    """Spawn ``discopt mcp`` on stdio; return (tool names, [structured results])."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "discopt.cli", "mcp"], env=_env()
    )

    async def go():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = sorted(t.name for t in (await session.list_tools()).tools)
                out = []
                for name, arguments in calls:
                    res = await session.call_tool(name, arguments)
                    # `isError` in mcp 1.x, `is_error` in 2.x.
                    assert not getattr(res, "is_error", getattr(res, "isError", None)), res

                    out.append(json.loads(res.content[0].text))
                return names, out

    return anyio.run(go)


def test_stdio_end_to_end():
    names, (py, nl, summ, info) = _call_tools(
        [
            ("solve_python", {"source": DEMO, "time_limit": 30}),
            ("solve_file", {"path": str(ALAN), "time_limit": 30}),
            ("model_summary", {"source": DEMO}),
            ("discopt_info", {}),
        ]
    )
    assert names == [
        "convert_model",
        "discopt_info",
        "model_summary",
        "solve_file",
        "solve_python",
    ]
    # The model printed to stdout mid-solve and the stream still parsed.
    assert py["ok"] and py["status"] == "optimal" and py["gap_certified"] is True, py
    assert py["objective"] == pytest.approx(0.36, abs=1e-6)
    assert py["x"]["n"] == pytest.approx(1.0) and py["x"]["x"] == pytest.approx(2.5, abs=1e-4)
    assert nl["ok"] and nl["status"] == "optimal", nl
    assert nl["objective"] == pytest.approx(ALAN_OPT, rel=1e-6)
    assert "Variables: 2" in summ["summary"], summ
    assert Path(info["package"]).resolve() == Path(REPO_PY, "discopt").resolve(), info


def _solve(**where):
    spec, timeout = mcp_server._solve_spec(30, None, None, 200, **where)
    return mcp_server.run_worker(spec, timeout)


def test_python_file_and_model_document(tmp_path):
    py = tmp_path / "demo.py"
    py.write_text(DEMO)
    r = _solve(path=str(py))
    assert r["ok"] and r["objective"] == pytest.approx(0.36, abs=1e-6), r

    from discopt.serialize import save

    ns: dict = {}
    exec(DEMO, ns)
    doc = tmp_path / "demo.dopt"
    save(ns["m"], doc)
    r = _solve(path=str(doc))
    assert r["ok"] and r["objective"] == pytest.approx(0.36, abs=1e-6), r


def test_errors_come_back_as_replies_not_exceptions(tmp_path):
    r = _solve(source="x = 1")
    assert not r["ok"] and "defines no discopt Model" in r["error"], r
    two = "import discopt.modeling as dm\na = dm.Model()\nb = dm.Model()\n"
    r = _solve(source=two)
    assert not r["ok"] and "defines 2 discopt Models" in r["error"], r
    r = _solve(path=str(tmp_path / "missing.nl"))
    assert not r["ok"] and "FileNotFoundError" in r["error"], r
    r = _solve(source="raise RuntimeError('boom')")
    assert not r["ok"] and "RuntimeError: boom" in r["error"]
    assert "Traceback" in r["stderr_tail"], r


def test_overrunning_worker_is_killed():
    r = mcp_server.run_worker({"action": "summary", "source": "import time\ntime.sleep(60)"}, 3)
    assert not r["ok"] and "worker killed after 3 s" in r["error"], r


def test_max_vars_truncates_solution():
    src = (
        "import discopt.modeling as dm\nm = dm.Model()\n"
        "xs = [m.continuous(f'x{i}', lb=i, ub=10) for i in range(5)]\n"
        "m.minimize(sum(xs))\n"
    )
    spec, timeout = mcp_server._solve_spec(30, None, None, 2, source=src)
    r = mcp_server.run_worker(spec, timeout)
    assert r["ok"] and r["objective"] == pytest.approx(10.0), r
    assert list(r["x"]) == ["x0", "x1"] and r["x_truncated"] == {"shown": 2, "total": 5}, r


def test_convert_model(tmp_path):
    out = tmp_path / "alan.gms"
    r = mcp_server.run_worker(
        {"action": "convert", "input_path": str(ALAN), "output_path": str(out)}, 60
    )
    assert r["ok"] and Path(r["wrote"]) == out and out.stat().st_size > 0, r
    r = mcp_server.run_worker(
        {"action": "convert", "input_path": str(ALAN), "output_path": str(tmp_path / "a.txt")},
        60,
    )
    assert not r["ok"] and "output must be one of" in r["error"], r


def test_print_config_points_at_this_interpreter():
    out = subprocess.run(
        [sys.executable, "-m", "discopt.cli", "mcp", "--print-config"],
        capture_output=True,
        text=True,
        env=_env(),
        check=True,
    ).stdout
    entry = json.loads(out)["mcpServers"]["discopt"]
    assert entry == {"command": sys.executable, "args": ["-m", "discopt.cli", "mcp"]}


def test_missing_sdk_is_a_loud_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    with pytest.raises(ImportError, match=r"pip install 'discopt\[mcp\]'"):
        mcp_server.build_server()
