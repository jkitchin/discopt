"""discopt as a GAMS solver.

This package lets a GAMS user solve a model with discopt::

    option minlp = discopt;
    solve m using minlp minimizing z;

GAMS launches the discopt *solver link* with a control file; the link reads the
model through the GAMS Modeling Object (GMO), rebuilds it as a discopt
:class:`~discopt.modeling.core.Model` -- including a faithful translation of the
GMO nonlinear instruction lists (:mod:`discopt.gams.instructions`) -- solves it,
and writes the solution back.

Register discopt with a GAMS system via :func:`write_registration` (or the
``discopt gams-register`` CLI command).  The expert-level GAMS Python API is an
optional dependency: ``pip install discopt[gams]`` (which pulls ``gamsapi``).
"""

from __future__ import annotations

from .daemon import (
    DaemonServer,
    default_socket_path,
    ping,
    solve_via_daemon,
    spawn_daemon,
    stop_daemon,
)
from .gmo_translate import GmoView, model_from_gmo
from .instructions import (
    FUNC_CODE,
    FUNC_NAME,
    GamsOpCode,
    GamsTranslationError,
    translate_instructions,
)
from .link import is_available, solve_from_control_file, solve_view, status_to_gams
from .register import check_gamsapi, gamsconfig_snippet, run_script, write_registration
from .verify import data_dir, load_manifest, verify

__all__ = [
    "is_available",
    "solve_from_control_file",
    "solve_view",
    "status_to_gams",
    "model_from_gmo",
    "GmoView",
    "translate_instructions",
    "GamsOpCode",
    "GamsTranslationError",
    "FUNC_CODE",
    "FUNC_NAME",
    "gamsconfig_snippet",
    "run_script",
    "write_registration",
    "check_gamsapi",
    "DaemonServer",
    "solve_via_daemon",
    "spawn_daemon",
    "stop_daemon",
    "ping",
    "default_socket_path",
    "verify",
    "data_dir",
    "load_manifest",
]
