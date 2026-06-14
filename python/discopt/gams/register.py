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


def write_registration(directory: str | Path) -> dict[str, Path]:
    """Write ``gamsconfig.yaml`` and the ``discopt-gams`` run script.

    Returns a mapping of the files written.  The user copies/merges
    ``gamsconfig.yaml`` into their GAMS system directory (or ``$HOME/.gams``)
    and ensures the run script is on ``PATH``.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    script = out / "discopt-gams"
    script.write_text(run_script())
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    config = out / "gamsconfig.yaml"
    config.write_text(gamsconfig_snippet(script_path=str(script)))

    return {"config": config, "script": script}
