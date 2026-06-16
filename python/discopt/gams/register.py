"""Register discopt as a solver with a GAMS system.

A third-party solver is made known to GAMS through the ``solverConfig`` section
of ``gamsconfig.yaml`` (the modern mechanism; the legacy equivalent is an entry
in ``gmscmpun.txt``).  GAMS invokes the configured ``scriptName`` with the path
to a control file as its single argument; that script just launches the discopt
link (:mod:`discopt.gams.link`).

:func:`gamsconfig_snippet` returns the YAML block, :func:`run_script` returns
the launcher, and :func:`write_registration` writes both to a directory.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

SOLVER_NAME = "discopt"
# GAMS model types discopt can solve.
MODEL_TYPES = ("LP", "MIP", "RMIP", "NLP", "DNLP", "RMINLP", "MINLP", "QCP", "MIQCP", "RMIQCP")


def gamsconfig_snippet(script_path: str | Path = "discopt-gams") -> str:
    """Return the ``gamsconfig.yaml`` ``solverConfig`` block for discopt.

    Follows the GAMS ``gamsconfig_schema.json`` solver schema: each entry maps
    the *solver name* to its config object, with ``scriptName`` + ``modelTypes``
    required.  discopt is a control-file *script* solver, so no ``library``
    block is emitted (GAMS invokes ``scriptName`` with the control file).
    """
    model_types = "\n".join(f"        - {t}" for t in MODEL_TYPES)
    return f"""\
solverConfig:
  - {SOLVER_NAME}:
      scriptName: {script_path}
      modelTypes:
{model_types}
"""


def run_script(python_executable: str | None = None) -> str:
    """Return a POSIX shell launcher that runs the discopt GAMS link.

    GAMS calls this with the control-file path as ``$1``.
    """
    py = python_executable or sys.executable
    return f"""\
#!/bin/sh
# discopt GAMS solver link -- invoked by GAMS as: <script> <control-file>
exec "{py}" -m discopt.gams.link "$@"
"""


def _solver_entry(script_path: str) -> dict:
    """The discopt ``solverConfig`` list item: ``{discopt: {scriptName, modelTypes}}``."""
    return {SOLVER_NAME: {"scriptName": script_path, "modelTypes": list(MODEL_TYPES)}}


def render_gamsconfig(existing: str | None, script_path: str) -> tuple[str, str]:
    """Render a ``gamsconfig.yaml`` that registers discopt, merging into *existing*.

    Returns ``(text, action)`` where *action* is ``"created"`` (fresh file),
    ``"merged"`` (added to a config that had other content), or ``"replaced"``
    (an existing discopt entry was updated). Other top-level keys and other
    solvers' entries are preserved. Falls back to the standalone snippet if
    PyYAML is unavailable.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return gamsconfig_snippet(script_path), "created"

    data: dict = {}
    if existing and existing.strip():
        try:
            loaded = yaml.safe_load(existing)
            if isinstance(loaded, dict):
                data = loaded
        except yaml.YAMLError:
            data = {}

    had_content = bool(data)
    sc = data.get("solverConfig")
    if not isinstance(sc, list):
        sc = []
    replaced = any(isinstance(it, dict) and SOLVER_NAME in it for it in sc)
    sc = [it for it in sc if not (isinstance(it, dict) and SOLVER_NAME in it)]
    sc.append(_solver_entry(script_path))
    data["solverConfig"] = sc

    action = "replaced" if replaced else ("merged" if had_content else "created")
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False), action


def write_registration(directory: str | Path) -> dict:
    """Write the ``discopt-gams`` run script and a merged ``gamsconfig.yaml``.

    If ``directory/gamsconfig.yaml`` already exists, the discopt entry is merged
    into it (preserving other solvers and top-level keys) rather than
    overwritten. Returns ``{"config", "script", "action"}`` where *action* is one
    of ``"created"`` / ``"merged"`` / ``"replaced"``.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    script = out / "discopt-gams"
    script.write_text(run_script())
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    config = out / "gamsconfig.yaml"
    existing = config.read_text() if config.exists() else None
    text, action = render_gamsconfig(existing, script_path=str(script))
    config.write_text(text)

    return {"config": config, "script": script, "action": action}


def check_gamsapi() -> tuple[bool, str]:
    """Diagnose whether the GAMS expert-level Python API is usable.

    Returns ``(ok, message)``. ``ok`` is True only when the GMO/GEV core
    bindings import; the message guides the user on installing a ``gamsapi``
    that matches their GAMS system (the bindings dlopen the GAMS C libraries, so
    versions must agree).
    """
    try:
        import gams
    except Exception:
        return (
            False,
            "gamsapi is not importable. Install the one bundled with your GAMS "
            "system (GAMS_DIR/apifiles/Python/api/...), or `pip install "
            "'gamsapi[core]'` pinned to your GAMS version.",
        )
    try:
        import gams.core.gev  # noqa: F401
        import gams.core.gmo  # noqa: F401
    except Exception as exc:
        return (
            False,
            f"gamsapi is installed but its GMO/GEV core bindings failed to import "
            f"({exc}). Install `gamsapi[core]` matching your GAMS version.",
        )
    version = getattr(gams, "__version__", "unknown")
    return (
        True,
        f"gamsapi {version} with GMO/GEV core bindings is importable. Ensure this "
        "version matches your installed GAMS system.",
    )
