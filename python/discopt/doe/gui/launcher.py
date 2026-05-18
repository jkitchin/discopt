"""Launcher for the ``discopt doe`` Streamlit app.

Spawns ``streamlit run app.py`` on a free local port and (best-effort)
opens the user's default browser. The workbook path, if supplied, is
passed to the app via the ``DISCOPT_DOE_WORKBOOK`` environment variable
so the app can drop the user straight into the right campaign.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    """Poll until ``host:port`` accepts a TCP connection. Returns True on success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _streamlit_available() -> bool:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        return False
    return True


def _missing_dep_message() -> str:
    return (
        "The discopt DoE GUI requires Streamlit. Install it with:\n"
        "    pip install 'discopt[doe-gui]'\n"
        "or:\n"
        "    pip install streamlit pandas openpyxl"
    )


def launch(
    workbook: Path | str | None = None,
    *,
    port: int | None = None,
    open_browser: bool = True,
    spawn: bool = True,
) -> int:
    """Launch the Streamlit app.

    Parameters
    ----------
    workbook : path-like or None
        Optional workbook path. Forwarded to the app via the
        ``DISCOPT_DOE_WORKBOOK`` environment variable.
    port : int, optional
        TCP port to bind on ``127.0.0.1``. A free port is chosen if
        omitted.
    open_browser : bool
        Open the default browser to the app URL. Defaults to True.
    spawn : bool
        If True (default), spawn ``streamlit run`` as a subprocess and
        return its exit code (this call blocks until streamlit exits).
        If False, only resolve the command + env and return 0 — useful
        for tests.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    if not _streamlit_available():
        print(_missing_dep_message(), file=sys.stderr)
        return 1

    app_path = Path(__file__).parent / "app.py"
    if not app_path.is_file():
        print(f"streamlit app not found at {app_path}", file=sys.stderr)
        return 1

    bind_port = int(port) if port is not None else _free_port()

    env = os.environ.copy()
    if workbook is not None:
        wb_path = Path(workbook).expanduser().resolve()
        if not wb_path.is_file():
            print(f"workbook not found: {wb_path}", file=sys.stderr)
            return 1
        env["DISCOPT_DOE_WORKBOOK"] = str(wb_path)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address=127.0.0.1",
        f"--server.port={bind_port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    url = f"http://127.0.0.1:{bind_port}"
    print(f"Starting discopt doe GUI at {url} ...", file=sys.stderr)

    if not spawn:
        return 0

    proc = subprocess.Popen(cmd, env=env)
    try:
        if open_browser:
            ready = _wait_for_port("127.0.0.1", bind_port, timeout=20.0)
            if not ready:
                print(
                    f"warning: streamlit didn't bind {url} within 20s; opening browser anyway",
                    file=sys.stderr,
                )
            try:
                webbrowser.open(url, new=2)
            except Exception:
                pass
        return int(proc.wait() or 0)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0
