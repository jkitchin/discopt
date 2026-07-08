"""Serialize / render a :class:`~discopt.modeling.core.SolveResult`.

Shared by the ``discopt solve`` CLI, the solve daemon, and their tests. The
daemon returns a result over a socket as JSON, and the CLI renders it (or writes
it) -- both go through here so there is one implementation and it is testable
without a socket or a real solve.

Non-JSON-safe fields (``_model``, ``infeasibility_certificate``,
``validation_report``) are dropped; numpy arrays become nested lists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from discopt.modeling.core import SolveResult

SCHEMA_VERSION = 1

# Scalar SolveResult fields that round-trip as-is.
_SCALAR_FIELDS = (
    "status",
    "objective",
    "bound",
    "gap",
    "wall_time",
    "node_count",
    "mip_count",
    "rust_time",
    "jax_time",
    "python_time",
    "convex_fast_path",
    "nlp_bb",
    "gap_certified",
    "subnlp_calls",
    "subnlp_feasible",
    "subnlp_incumbent_updates",
)
_DICT_ARRAY_FIELDS = (
    "x",
    "constraint_duals",
    "bound_duals_lower",
    "bound_duals_upper",
)


def _jsonify_arrays(d: Optional[dict]) -> Optional[dict]:
    """``{name: ndarray|scalar}`` -> ``{name: list|number}`` (or ``None``)."""
    if d is None:
        return None
    return {k: np.asarray(v).tolist() for k, v in d.items()}


def serialize_result(r: SolveResult) -> dict:
    """A JSON-safe dict capturing the solver-relevant fields of *r*."""
    out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for name in _SCALAR_FIELDS:
        out[name] = getattr(r, name, None)
    for name in _DICT_ARRAY_FIELDS:
        val = _jsonify_arrays(getattr(r, name, None))
        if val is not None:
            out[name] = val
    if r.mip_nlp_trace is not None:
        out["mip_nlp_trace"] = r.mip_nlp_trace
    expl = getattr(r, "_explanation", None)
    if expl:
        out["explanation"] = str(expl)
    return out


def deserialize_result(d: dict) -> SolveResult:
    """Rebuild a :class:`SolveResult` from :func:`serialize_result` output."""
    kwargs: dict[str, Any] = {}
    for name in _SCALAR_FIELDS:
        if name in d:
            kwargs[name] = d[name]
    for name in _DICT_ARRAY_FIELDS:
        if d.get(name) is not None:
            kwargs[name] = {k: np.asarray(v) for k, v in d[name].items()}
    if d.get("mip_nlp_trace") is not None:
        kwargs["mip_nlp_trace"] = d["mip_nlp_trace"]
    r = SolveResult(**kwargs)
    if d.get("explanation"):
        r._explanation = d["explanation"]
    return r


def options_to_payload(options: dict) -> dict:
    """JSON-safe copy of solve options for the wire.

    A ``SolverTuning`` (frozen dataclass) under ``options["tuning"]`` is flattened
    to a plain dict via :func:`dataclasses.asdict`. Any callable-valued option
    (callbacks) is dropped defensively -- the CLI never sets them, and they cannot
    cross a socket.
    """
    out: dict[str, Any] = {}
    for k, v in options.items():
        if callable(v):
            continue
        if is_dataclass(v) and not isinstance(v, type):
            out[k] = asdict(v)
        else:
            out[k] = v
    return out


def options_from_payload(payload: dict) -> dict:
    """Inverse of :func:`options_to_payload` for the daemon: rebuild ``tuning``."""
    out = dict(payload)
    tuning = out.get("tuning")
    if isinstance(tuning, dict):
        from discopt.solver_tuning import SolverTuning

        valid = {f.name for f in fields(SolverTuning)}
        out["tuning"] = SolverTuning(**{k: v for k, v in tuning.items() if k in valid})
    return out


# ── Rendering / file outputs ─────────────────────────────────────────────────
def summary_text(r: SolveResult, *, max_vars: int = 20) -> str:
    """A compact human-readable summary for stdout."""
    lines = [f"status:    {r.status}"]
    if r.objective is not None:
        lines.append(f"objective: {r.objective:.8g}")
    if r.bound is not None:
        lines.append(f"bound:     {r.bound:.8g}")
    if r.gap is not None:
        cert = "" if r.gap_certified else "  (uncertified)"
        lines.append(f"gap:       {r.gap:.4g}{cert}")
    lines.append(f"nodes:     {r.node_count}   wall: {r.wall_time:.3f}s")
    if r.x:
        lines.append("solution:")
        for i, (name, val) in enumerate(r.x.items()):
            if i >= max_vars:
                lines.append(f"  ... ({len(r.x) - max_vars} more)")
                break
            arr = np.asarray(val)
            shown = arr.item() if arr.ndim == 0 else np.array2string(arr, threshold=8)
            lines.append(f"  {name} = {shown}")
    return "\n".join(lines)


def write_json(r: SolveResult, path: Path) -> None:
    path.write_text(json.dumps(serialize_result(r), indent=2) + "\n")


def write_sol(r: SolveResult, var_names: list[str], path: Path) -> None:
    """Write a minimal AMPL-style ``.sol``: a status line + one primal per variable.

    ``var_names`` must be the ``.nl`` column order (``[v.name for v in model._variables]``)
    so AMPL-side tools read the values back into the right columns.
    """
    lines = [f"discopt {r.status}"]
    if r.objective is not None:
        lines.append(f"objective {r.objective:.17g}")
    x = r.x or {}
    for name in var_names:
        if name not in x:
            continue
        arr = np.asarray(x[name]).ravel()
        for val in arr:
            lines.append(f"{float(val):.17g}")
    path.write_text("\n".join(lines) + "\n")
