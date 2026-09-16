"""Serialize / render a :class:`~discopt.modeling.core.SolveResult`.

Shared by the ``discopt solve`` CLI, the solve daemon, and their tests. The
daemon returns a result over a socket as JSON, and the CLI renders it (or writes
it) -- both go through here so there is one implementation and it is testable
without a socket or a real solve.

``_model`` and ``infeasibility_certificate`` are dropped (see below); numpy
arrays become nested lists.

Provenance (#1266)
------------------
An archived result used to carry ``wall_time`` and a node count with nothing
saying which discopt produced them, on what machine, or with which options --
a number that cannot be interpreted, let alone reproduced. Since #1264 the
model format records that; this module is the result-side half.

**The provenance of a result is the identity of the process that SOLVED it**,
which on the daemon path is not the process that writes the file. A warm daemon
serves the solve in its own interpreter (``daemon._solve_request``), so the
daemon stamps the block and the CLI carries it through to disk unchanged.
Recomputing it at the writer would record the client's version for a number the
daemon computed -- quietly wrong in exactly the way provenance exists to
prevent. Hence the rule every field below follows: **a carried value always
wins over a freshly captured one**, and ``provenance=True`` means "stamp one if
this result does not already have one", not "stamp one".

The wire format therefore does carry the block, which #1266 left open as a
question: it is one small dict per reply, and it is the only place the solving
process's identity exists.

Validation report
-----------------
``validation_report`` (the Examiner-style KKT report, populated by
``Model.solve(validate=True)``) used to be dropped as non-JSON-safe. For a
solver whose product is its certificate, an archived result that silently omits
the validation it passed is a worse gap than a missing timestamp, so it is now
carried: :class:`~discopt.validation.examiner.ExaminerReport` is a dataclass of
primitives and round-trips exactly.

Infeasibility certificate
-------------------------
``infeasibility_certificate`` is carried too. It is a three-field dataclass (a
total violation and two float arrays), so encoding it needs nothing clever --
the earlier "non-JSON-safe" grouping with ``_model`` was simply wrong about it.
An infeasible result is a *claim*, and the witness for that claim is the part
worth archiving.

ONE CAVEAT travels with it, and it is the reason to read this before using the
numbers: the violation arrays are indexed in **backend LP row order**
(inequalities then equalities, as the matrices were passed to the LP solver),
which is not the order of the constraints in the user's model, and the result
file does not carry that mapping. ``ineq_violations[3]`` is therefore "the
fourth inequality row of the LP that was solved", not "the fourth constraint I
wrote". Reconstructing the correspondence needs the model the solve ran on.

``_model`` remains dropped: it is a live object graph, not data.

Non-finite floats
-----------------
Every float written here -- the scalars, the solution and dual arrays, and the
nested blocks -- goes through :mod:`discopt.serialize`'s tagging, so a NaN or
infinity is written as a tagged string rather than the bare ``NaN`` /
``Infinity`` tokens that Python's ``json`` emits and other languages' parsers
reject outright. That is the same encoding a result already gets when embedded
in a ``.dopt`` document, so the standalone and embedded forms agree, and
:func:`write_json` dumps with ``allow_nan=False`` so a float that slipped past
the encoders raises instead of silently writing a non-standard token.

This does change what a raw reader sees for a non-finite value -- ``"inf"``
rather than ``Infinity``. It is not a compatibility break in any case that
previously worked: a document containing a non-finite value was *already*
invalid JSON, so the only readers affected are the ones that were relying on
Python's non-standard extension to parse a broken file. Finite values, which is
almost every value, are written exactly as before, and anything going through
:func:`deserialize_result` sees real floats either way.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from discopt.modeling.core import SolveResult

# The float tagging that keeps NaN/infinity out of the document as bare
# `NaN`/`Infinity` tokens. Imported rather than re-implemented so the standalone
# result file and the same result embedded in a `.dopt` use one encoding.
from discopt.serialize import _dec_float, _dec_tree, _enc_float, _enc_tree

#: Bumped to 2 by #1266 (``provenance``, ``solve_options``, ``validation_report``).
#: The change is additive: a version-1 reader ignores the new keys, and
#: :func:`deserialize_result` reads a version-1 document unchanged.
SCHEMA_VERSION = 2

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
    # #1244. Both must round-trip: ``__post_init__`` can only DERIVE
    # ``bound_valid`` from ``gap_certified``, so dropping the stored value would
    # silently downgrade every uncertified-but-valid bound (a ``time_limit``
    # exit's tree bound) to "no claim" on reload -- exactly the information a
    # consumer reads the file for.
    "bound_valid",
    "bound_source",
    "subnlp_calls",
    "subnlp_feasible",
    "subnlp_incumbent_updates",
    "algorithm_route",
    # #1246: the reason a failed solve failed. Dropping it on a round trip would
    # reload a batch's error result as a bare ``status="error"`` with no reason,
    # which is exactly what the field exists to prevent.
    "error",
)
#: ``{str: float}`` fields. Written through the same float tagging as the
#: scalars, so a non-finite residual is a tagged string rather than a bare
#: ``NaN``/``Infinity`` token.
_FLOAT_DICT_FIELDS = ("kkt",)
_DICT_ARRAY_FIELDS = (
    "x",
    "constraint_duals",
    "bound_duals_lower",
    "bound_duals_upper",
)

#: Scalar fields whose value is text, not a number. They are excluded from float
#: tag *decoding*: without this, a status or route that happened to read exactly
#: ``"inf"`` would be turned into a float on the way back in.
#:
#: ``bound_source`` was missing from this set when the tagging landed (#1266),
#: which made ``deserialize_result`` raise ``SerializationError: unknown float
#: token 'convex_proof'`` on **every** result carrying a bound source — i.e. every
#: certified result that crossed the daemon socket or was stored and read back.
#: ``test_every_string_scalar_field_is_excluded_from_float_decoding`` now derives
#: the set from the dataclass annotations, so the next string field added to
#: ``_SCALAR_FIELDS`` cannot repeat it.
_STRING_SCALAR_FIELDS = frozenset({"status", "algorithm_route", "bound_source", "error"})


def _jsonify_arrays(d: Optional[dict]) -> Optional[dict]:
    """``{name: ndarray|scalar}`` -> ``{name: list|number}`` (or ``None``).

    Routed through ``_enc_tree`` so a non-finite entry becomes a tagged string
    rather than a bare ``NaN``/``Infinity`` token. A diverged incumbent is
    exactly the case where a solution array holds one, so this is not a
    hypothetical path.
    """
    if d is None:
        return None
    return {k: _enc_tree(np.asarray(v).tolist()) for k, v in d.items()}


def _enc_certificate(cert: Any) -> dict:
    """``InfeasibilityCertificate`` -> JSON-safe dict.

    See the module docstring on row ordering: these arrays are indexed in
    backend LP row order, not the user's constraint order.
    """
    from discopt.solvers import InfeasibilityCertificate

    if not isinstance(cert, InfeasibilityCertificate):
        raise TypeError(
            f"infeasibility_certificate is a {type(cert).__name__}, not an "
            "InfeasibilityCertificate; this writer has no faithful encoding for it. "
            "Refusing rather than dropping it silently."
        )
    return {
        "total_violation": _enc_float(cert.total_violation),
        "ineq_violations": _enc_tree(np.asarray(cert.ineq_violations).tolist()),
        "eq_violations": _enc_tree(np.asarray(cert.eq_violations).tolist()),
    }


def _dec_certificate(d: Any) -> Any:
    from discopt.solvers import InfeasibilityCertificate

    if not isinstance(d, dict):
        raise TypeError(
            f"infeasibility_certificate section must be an object, got {type(d).__name__}."
        )
    return InfeasibilityCertificate(
        total_violation=_dec_float(d["total_violation"]),
        ineq_violations=np.asarray(_dec_tree(d["ineq_violations"]), dtype=float),
        eq_violations=np.asarray(_dec_tree(d["eq_violations"]), dtype=float),
    )


def _enc_validation_report(report: Any) -> dict:
    """``ExaminerReport`` -> JSON-safe dict.

    ``asdict`` carries every declared field, including the nested
    ``CheckResult`` list, so a check added to the examiner later travels without
    a change here. ``passed`` is a *property* rather than a field, so ``asdict``
    misses it; it is the report's headline, and a consumer should not have to
    re-derive the verdict by scanning checks, so it is written explicitly.

    An object that is not an ``ExaminerReport`` is refused rather than dropped:
    a report that silently vanishes from an archived result is the failure this
    change exists to fix.
    """
    from discopt.validation.examiner import ExaminerReport

    if not isinstance(report, ExaminerReport):
        raise TypeError(
            f"validation_report is a {type(report).__name__}, not an ExaminerReport; "
            "this writer has no faithful encoding for it. Refusing rather than "
            "dropping it silently."
        )
    out = asdict(report)
    out["passed"] = bool(report.passed)
    encoded: dict = _enc_tree(out)
    return encoded


def _dec_validation_report(d: Any) -> Any:
    """Inverse of :func:`_enc_validation_report`.

    Rebuilt as the real dataclass, not left as a dict: on the daemon path the
    CLI deserializes the reply and then writes the file, so a report that came
    back as a plain dict would fail the ``isinstance`` check on the way out and
    lose exactly the report this change is carrying.
    """
    from discopt.validation.examiner import CheckResult, ExaminerReport

    d = _dec_tree(d)
    if not isinstance(d, dict):
        raise TypeError(f"validation_report section must be an object, got {type(d).__name__}.")
    report_fields = {f.name for f in fields(ExaminerReport)}
    check_fields = {f.name for f in fields(CheckResult)}
    # `passed` is a property, so it is not in `fields(ExaminerReport)` and is
    # filtered out here -- passing it to the constructor would raise.
    kwargs = {k: v for k, v in d.items() if k in report_fields and k != "checks"}
    checks = []
    for c in d.get("checks", []):
        cd = {k: v for k, v in c.items() if k in check_fields}
        # `violators` is a list of (label, value) pairs; JSON gives back lists.
        cd["violators"] = [tuple(v) for v in cd.get("violators", [])]
        checks.append(CheckResult(**cd))
    return ExaminerReport(checks=checks, **kwargs)


def serialize_result(
    r: SolveResult,
    *,
    provenance: bool = False,
    options: Optional[dict] = None,
) -> dict:
    """A JSON-safe dict capturing the solver-relevant fields of *r*.

    Parameters
    ----------
    provenance : bool, default False
        Stamp a :mod:`discopt.provenance` block describing THIS process -- but
        only when *r* does not already carry one. Pass ``True`` from the process
        that ran the solve (the daemon) and from the writers that archive a
        result; leave it ``False`` where a surrounding document already records
        provenance, as :mod:`discopt.serialize` does for an embedded result.
    options : dict, optional
        The solve options actually used, JSON-safe (see
        :func:`options_to_payload`). Like ``provenance``, a value already
        carried on *r* wins: the daemon knows what it solved with, and the
        client must not overwrite that with what it *asked* for.
    """
    out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for name in _SCALAR_FIELDS:
        val = getattr(r, name, None)
        # Tag by runtime type rather than a hand-kept field list: `bool` is not a
        # `float` in Python, and ints are left alone, so this catches exactly the
        # float-valued fields and stays right if one is added later.
        out[name] = _enc_float(val) if isinstance(val, float) else val
    for name in _DICT_ARRAY_FIELDS:
        val = _jsonify_arrays(getattr(r, name, None))
        if val is not None:
            out[name] = val
    for name in _FLOAT_DICT_FIELDS:
        fval = getattr(r, name, None)
        if fval is not None:
            # Tagged like every other float here: a KKT residual is exactly the
            # kind of number that comes back non-finite from a diverged solve.
            out[name] = {k: _enc_float(float(v)) for k, v in fval.items()}
    if r.mip_nlp_trace is not None:
        out["mip_nlp_trace"] = r.mip_nlp_trace
    expl = getattr(r, "_explanation", None)
    if expl:
        out["explanation"] = str(expl)

    report = getattr(r, "validation_report", None)
    if report is not None:
        out["validation_report"] = _enc_validation_report(report)

    cert = getattr(r, "infeasibility_certificate", None)
    if cert is not None:
        out["infeasibility_certificate"] = _enc_certificate(cert)

    carried_opts = getattr(r, "_solve_options", None)
    if carried_opts is not None:
        out["solve_options"] = _enc_tree(carried_opts)
    elif options is not None:
        out["solve_options"] = _enc_tree(options)

    carried_prov = getattr(r, "_provenance", None)
    if carried_prov is not None:
        out["provenance"] = carried_prov
    elif provenance:
        from discopt.provenance import capture

        out["provenance"] = capture()

    return out


def deserialize_result(d: dict) -> SolveResult:
    """Rebuild a :class:`SolveResult` from :func:`serialize_result` output.

    Version-tolerant in both directions: a document written before #1266 simply
    has no ``provenance`` / ``solve_options`` / ``validation_report`` keys and
    reads unchanged, and the keys are restored here so a result that crosses the
    daemon socket arrives at the writer with them intact.
    """
    kwargs: dict[str, Any] = {}
    for name in _SCALAR_FIELDS:
        if name in d:
            val = d[name]
            if isinstance(val, str) and name not in _STRING_SCALAR_FIELDS:
                val = _dec_float(val)
            kwargs[name] = val
    for name in _DICT_ARRAY_FIELDS:
        if d.get(name) is not None:
            # `_dec_tree` BEFORE `np.asarray`: a tagged "inf" left as a string
            # would make an object-dtype array rather than a float one.
            kwargs[name] = {k: np.asarray(_dec_tree(v), dtype=float) for k, v in d[name].items()}
    for name in _FLOAT_DICT_FIELDS:
        if d.get(name) is not None:
            kwargs[name] = {k: _dec_float(v) for k, v in d[name].items()}
    if d.get("mip_nlp_trace") is not None:
        kwargs["mip_nlp_trace"] = d["mip_nlp_trace"]
    if d.get("validation_report") is not None:
        kwargs["validation_report"] = _dec_validation_report(d["validation_report"])
    if d.get("infeasibility_certificate") is not None:
        kwargs["infeasibility_certificate"] = _dec_certificate(d["infeasibility_certificate"])
    r = SolveResult(**kwargs)
    # #1244: restore the DECIDED validity, do not let it be re-derived.
    #
    # ``__post_init__`` runs again on this construction, and its rule 1
    # (``gap_certified`` implies ``bound_valid``) overwrites the value just
    # restored beside it. A result stored with ``gap_certified=True``, a finite
    # bound and ``bound_valid=False`` therefore reloaded as ``True`` -- the
    # persisted certificate came back STRONGER than it went in, which is the one
    # direction a round trip must never move a claim.
    #
    # That triple is not hypothetical: ``Model.solve``'s #844 merge builds it
    # whenever the primary's bound survives with a claim of False. A round trip
    # is an identity, not a re-derivation.
    if "bound_valid" in d:
        r.bound_valid = bool(d["bound_valid"])
        r.bound_source = d.get("bound_source") if r.bound_valid else None
    if d.get("explanation"):
        r._explanation = d["explanation"]
    if d.get("provenance") is not None:
        r._provenance = d["provenance"]
    if d.get("solve_options") is not None:
        r._solve_options = _dec_tree(d["solve_options"])
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


def write_json(r: SolveResult, path: Path, *, options: Optional[dict] = None) -> None:
    """Archive *r* as JSON, with provenance.

    This is the archival path, so it always records provenance: the file
    outlives the process, and a ``wall_time`` with no record of what produced it
    is not a measurement. A block already carried on *r* (stamped by the daemon
    that solved it) is preserved rather than replaced -- see the module
    docstring.
    """
    # `allow_nan=False` asserts that every float went through `_enc_float`: a
    # value that slipped past raises here instead of silently writing a bare
    # `NaN`/`Infinity` token that is not valid JSON. Same guard the model format
    # uses, for the same reason.
    path.write_text(
        json.dumps(serialize_result(r, provenance=True, options=options), indent=2, allow_nan=False)
        + "\n"
    )


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
