"""discopt as a GAMS solver: the control-file solver link.

When GAMS solves a model with ``option <type> = discopt;`` it launches this
link with a single argument -- the path to a *control file* -- and expects the
solution written back through the GAMS Modeling Object (GMO).  This module:

1. boots the GAMS environment (GEV) and model (GMO) objects from that control
   file (:func:`solve_from_control_file`),
2. translates the GMO instance into a discopt :class:`Model`
   (:mod:`discopt.gams.gmo_translate`),
3. solves it with discopt, and
4. writes the primal solution and the GAMS model/solve status back into GMO.

The GAMS-library calls live entirely in :class:`_GmoAdapter` /
:func:`solve_from_control_file`; the rest (status mapping, the solve wrapper) is
plain Python and unit-tested without a GAMS installation.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
import time
from typing import TYPE_CHECKING

from discopt.modeling.core import Model, SolveResult, VarType

from .gmo_translate import model_from_gmo
from .instructions import VAR_FIELD_OPCODES, GamsTranslationError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .gmo_translate import GmoView

# ── GAMS status constants (gmomcc) ───────────────────────────────────────────
MODELSTAT_OPTIMAL = 1
MODELSTAT_LOCALLY_OPTIMAL = 2
MODELSTAT_UNBOUNDED = 3
MODELSTAT_INFEASIBLE = 4
MODELSTAT_LOCALLY_INFEASIBLE = 5
MODELSTAT_INTERMEDIATE_INFEASIBLE = 6
MODELSTAT_FEASIBLE = 7  # feasible solution, problem WITHOUT discrete variables
MODELSTAT_INTEGER = 8  # feasible solution, problem WITH discrete variables
MODELSTAT_INTEGER_INFEASIBLE = 10  # proven infeasible, problem with discrete variables
MODELSTAT_ERROR_NO_SOLUTION = 13
MODELSTAT_NO_SOLUTION_RETURNED = 14

SOLVESTAT_NORMAL = 1
SOLVESTAT_ITERATION = 2
SOLVESTAT_RESOURCE = 3  # time / resource interrupt
SOLVESTAT_TERMINATED = 4
SOLVESTAT_CAPABILITY = 6  # the model uses something the solver does not support
SOLVESTAT_USER_INTERRUPT = 8
SOLVESTAT_SOLVER_ERROR = 10
SOLVESTAT_SYSTEM_ERROR = 13

#: GAMS's "no iteration limit" (``gevIterLim``'s default).
ITERLIM_INFINITY = 2147483647


def is_available() -> bool:
    """Return ``True`` if the GAMS expert-level Python API (gamsapi) is present."""
    try:  # pragma: no cover - depends on optional gamsapi
        import gams.core.gev  # noqa: F401
        import gams.core.gmo  # noqa: F401

        return True
    except Exception:
        return False


#: ``SolveResult.termination`` values that name an interruption, and the GAMS
#: solve status each maps to (#1888). ``gap`` / ``exhausted`` / ``presolve`` / ``None``
#: are normal completions and are absent on purpose -- except ``gap`` on a result
#: that is only ``feasible``, which ``status_to_gams`` reports as terminated (#2000).
_TERMINATION_SOLVESTAT: dict[str | None, int] = {
    "time_limit": SOLVESTAT_RESOURCE,
    "node_limit": SOLVESTAT_RESOURCE,
    "iteration_limit": SOLVESTAT_ITERATION,
    "interrupted": SOLVESTAT_USER_INTERRUPT,
}


def status_to_gams(result: SolveResult, has_discrete: bool) -> tuple[int, int]:
    """Map a discopt :class:`SolveResult` to ``(modelStat, solveStat)``.

    ``has_discrete`` selects GAMS's discrete-model codes: ``8`` (``Integer
    Solution``) for a point that is not certified optimal, where a continuous
    model gets ``2`` (local) or ``7`` (feasible), and ``10`` for a proven
    infeasibility, where a continuous model gets ``4``. A certified optimum is
    ``1`` (``Optimal``) for both.

    Globality is reported from ``result.gap_certified``: discopt sets it only
    when it produced a valid finite dual bound (McCormick, or the rigorous
    alphaBB fallback), with main's soundness guards downgrading it otherwise. A
    ``"optimal"`` result is therefore reported as GAMS ``Optimal`` (certified
    global) only when certified, else ``LocallyOptimal`` (``Integer``
    for a discrete model, since GAMS defines ``LocallyOptimal`` for NLP) -- the
    link trusts the solver's certificate rather than a static allow-list.

    A ``"feasible"`` result takes its solve status from ``result.termination``
    (#1888): ``status`` says what was proven, ``termination`` why the search
    stopped, and a feasible incumbent at a time/node/iteration limit is a GAMS
    resource/iteration interrupt, not a normal completion.
    """
    status = (result.status or "").lower()
    has_solution = result.x is not None
    termination = getattr(result, "termination", None)
    # GAMS defines 7 for a feasible point of a problem without discrete variables
    # and 8 for one with them; 4 / 10 likewise for a proven infeasibility.
    feasible_stat = MODELSTAT_INTEGER if has_discrete else MODELSTAT_FEASIBLE

    if status == "optimal":
        if not getattr(result, "gap_certified", True):
            if has_discrete:
                return MODELSTAT_INTEGER, SOLVESTAT_NORMAL
            return MODELSTAT_LOCALLY_OPTIMAL, SOLVESTAT_NORMAL
        # GAMS's 1 is "proven optimal" for any problem class. A certified
        # discrete optimum used to be reported as 8, the same code an
        # uncertified one now gets, which would hide the certificate.
        return MODELSTAT_OPTIMAL, SOLVESTAT_NORMAL
    if status == "feasible":
        if termination == "gap":
            # The search stopped on its convergence test, and the certificate it
            # stopped on was then withdrawn -- the published (incumbent, bound) pair
            # does not close the gap (the #1537 E repair moved the point, #2000).
            # That is not a normal completion: GAMS's OBJVAL/OBJEST miss optcr with
            # budget left, so say the solver terminated.
            return feasible_stat, SOLVESTAT_TERMINATED
        return feasible_stat, _TERMINATION_SOLVESTAT.get(termination, SOLVESTAT_NORMAL)
    if status == "infeasible":
        if has_discrete:
            return MODELSTAT_INTEGER_INFEASIBLE, SOLVESTAT_NORMAL
        return MODELSTAT_INFEASIBLE, SOLVESTAT_NORMAL
    if status == "unbounded":
        return MODELSTAT_UNBOUNDED, SOLVESTAT_NORMAL
    if status in ("time_limit", "node_limit"):
        model_stat = feasible_stat if has_solution else MODELSTAT_NO_SOLUTION_RETURNED
        return model_stat, SOLVESTAT_RESOURCE
    if status == "iteration_limit":
        model_stat = feasible_stat if has_solution else MODELSTAT_NO_SOLUTION_RETURNED
        return model_stat, SOLVESTAT_ITERATION
    # Local (no global claim) outcomes, discopt.status #1148.
    if status == "local_optimal":
        if has_discrete:
            return MODELSTAT_INTEGER, SOLVESTAT_NORMAL
        return MODELSTAT_LOCALLY_OPTIMAL, SOLVESTAT_NORMAL
    if status == "local_limit":
        # An iterate at a limit, neither stationary nor verified feasible: GAMS's
        # "intermediate infeasible" makes no feasibility claim it cannot back.
        model_stat = (
            MODELSTAT_INTERMEDIATE_INFEASIBLE if has_solution else MODELSTAT_NO_SOLUTION_RETURNED
        )
        return model_stat, _TERMINATION_SOLVESTAT.get(termination, SOLVESTAT_ITERATION)
    if status == "local_infeasible":
        # Not an infeasibility proof: GAMS's "locally infeasible".
        return MODELSTAT_LOCALLY_INFEASIBLE, SOLVESTAT_NORMAL
    # error / unknown
    return MODELSTAT_ERROR_NO_SOLUTION, SOLVESTAT_SOLVER_ERROR


def _has_discrete(model: Model) -> bool:
    return any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables)


def _read_gams_options(gev, gev_h) -> dict:
    """Read the job's ``reslim`` / ``optcr`` / ``optca`` / ``threads`` from GEV.

    Returns keyword arguments for :func:`solve_view`.  Unsupported values are
    rejected loudly rather than silently replaced by defaults:

    * ``reslim`` must be a positive number; ``optcr`` must be finite and ``>= 0``.
    * ``optca`` must be finite and ``>= 0``; a positive value is forwarded as
      is.  GAMS defines it as ``|PB - DB| <= optca``, so ``0`` grants no
      absolute allowance.  It used to map to ``None``, i.e. the solver's own
      ``1e-6`` (#1889), which accepted ``PB=1e-7, DB=0`` at ``optcr=1e-4``
      although neither criterion held.  Taken literally, though, ``0`` cannot
      certify any optimum at exactly 0: a rigorous dual bound carries a safety
      margin (measured: HiGHS's LP route publishes ``-1e-12`` against ``0``)
      and the relative gap is undefined there, so GAMS's default ``optca=0``
      would report every zero-optimum model as merely feasible.  ``0`` is
      therefore read as "no allowance beyond floating-point resolution": closed
      iff ``|PB - DB| <= OPTCA_ZERO_RESOLUTION * max(1, |PB|, |DB|)`` or the
      ``optcr`` test holds, which is exactly ``abs_gap=OPTCA_ZERO_RESOLUTION``
      with ``gap=max(optcr, OPTCA_ZERO_RESOLUTION)``.  That is 1000x tighter
      than the old default and still rejects the ``1e-7`` pair above.
    * ``threads`` comes from ``gevThreads(gev_h)``, which GEV has already
      resolved from the raw option (``0`` = all cores, ``-N`` = all but ``N``),
      so it is forwarded as is; a non-positive resolved count is rejected.
    * ``nodlim`` is forwarded as ``max_nodes``. GAMS's ``0`` means no node
      limit; discopt's own default is a 100,000-node cap, which GAMS never asked
      for, so ``0`` maps to :data:`NODLIM_INFINITY`.

    Options discopt cannot honour are reported by :func:`_unhonored_gams_options`.
    """
    reslim = float(gev.gevGetDblOpt(gev_h, gev.gevResLim))
    optcr = float(gev.gevGetDblOpt(gev_h, gev.gevOptCR))
    optca = float(gev.gevGetDblOpt(gev_h, gev.gevOptCA))
    # ``gevThreads`` is a function, not an option-name constant like ``gevResLim``.
    threads = int(gev.gevThreads(gev_h))
    nodlim = int(gev.gevGetIntOpt(gev_h, gev.gevNodeLim))
    if math.isnan(reslim) or reslim <= 0.0:
        raise ValueError(f"unsupported GAMS reslim={reslim!r}: must be positive")
    if not math.isfinite(optcr) or optcr < 0.0:
        raise ValueError(f"unsupported GAMS optcr={optcr!r}: must be finite and >= 0")
    if not math.isfinite(optca) or optca < 0.0:
        raise ValueError(f"unsupported GAMS optca={optca!r}: must be finite and >= 0")
    if threads < 1:
        raise ValueError(f"unsupported resolved GAMS threads={threads}: must be >= 1")
    if nodlim < 0:
        raise ValueError(f"unsupported GAMS nodlim={nodlim}: must be >= 0")
    if optca == 0.0:
        optca = OPTCA_ZERO_RESOLUTION
        optcr = max(optcr, OPTCA_ZERO_RESOLUTION)
    return {
        "time_limit": reslim,
        "gap": optcr,
        "abs_gap": optca,
        "threads": threads,
        "max_nodes": nodlim if nodlim > 0 else NODLIM_INFINITY,
    }


#: The resolution GAMS ``optca=0`` is read at, relative to ``max(1, |PB|, |DB|)``
#: (see :func:`_read_gams_options`). A chosen constant: well above the ~1e-12
#: margins rigorous bounds are published with, well below any gap a user would
#: call open.
OPTCA_ZERO_RESOLUTION = 1e-9

#: ``max_nodes`` for GAMS ``nodlim=0`` (no limit): GAMS's own limit infinity,
#: kept within a 32-bit int for every backend that takes a node budget.
NODLIM_INFINITY = 2147483647


def _unhonored_gams_options(gev, gev_h, gmo, gmo_h) -> list[str]:
    """Messages naming every GAMS option set to a value discopt does not honour.

    These used to be ignored without a word: a job with ``m.optfile=1`` or a
    ``cutoff`` ran as if neither had been set. None of them changes what is
    proven, so the solve goes ahead, but the user is told in the log and
    listing which settings had no effect.
    """
    notes = []
    iterlim = int(gev.gevGetIntOpt(gev_h, gev.gevIterLim))
    if iterlim != ITERLIM_INFINITY:
        notes.append(f"iterlim={iterlim} is ignored: discopt has no global iteration count")
    if int(gev.gevGetIntOpt(gev_h, gev.gevDomLim)) > 0:
        notes.append("domlim is ignored: discopt treats any evaluation error as infeasible")
    if int(gev.gevGetIntOpt(gev_h, gev.gevUseCutOff)):
        cutoff = float(gev.gevGetDblOpt(gev_h, gev.gevCutOff))
        notes.append(f"cutoff={cutoff:g} is ignored: discopt does not take an objective cutoff")
    if int(gev.gevGetIntOpt(gev_h, gev.gevUseCheat)):
        notes.append("cheat is ignored: discopt does not take an improvement requirement")
    if int(gmo.gmoOptFile(gmo_h)) > 0:
        notes.append(
            "optfile is ignored: the discopt GAMS link reads no solver option file; "
            "only reslim, optcr, optca, nodlim and threads are forwarded"
        )
    return [f"*** discopt: {n}" for n in notes]


def solve_view(
    view: "GmoView",
    *,
    time_limit: float = 1e10,
    gap: float = 1e-4,
    abs_gap: float | None = None,
    threads: int = 1,
    max_nodes: int = NODLIM_INFINITY,
):
    """Translate a GMO view, solve with discopt, and return ``(model, result)``.

    This is the GAMS-library-free core of the link: given anything implementing
    :class:`~discopt.gams.gmo_translate.GmoView` it produces a solved discopt
    model, so it can be exercised in tests with an in-memory fake.
    """
    model = model_from_gmo(view)
    result = model.solve(
        time_limit=time_limit,
        gap_tolerance=gap,
        abs_gap_tolerance=abs_gap,
        threads=threads,
        max_nodes=max_nodes,
    )
    return model, result


def _gams_sysdir(control_file: str) -> str | None:
    """Best-effort discovery of the GAMS system directory from a control file.

    ``gmoCreateD`` / ``gevCreateD`` need the directory holding the GAMS shared
    libraries.  The control file references it (e.g. via ``gmscmpun.txt``); we
    locate it robustly rather than relying on a fixed line number, falling back
    to ``None`` so the caller can try the search-path ``gevCreate``.
    """
    try:
        with open(control_file) as fh:
            lines = [ln.strip() for ln in fh]
    except OSError:
        return None
    # 1. Directory of any referenced GAMS solver-config file.
    for ln in lines:
        for tok in ln.split():
            if os.path.basename(tok) in ("gmscmpun.txt", "gmscmpdef.txt") and os.path.isfile(tok):
                return os.path.dirname(tok)
    # 2. A bare directory line that looks like a GAMS system directory.
    for ln in lines:
        if os.path.isdir(ln) and os.path.isfile(os.path.join(ln, "gmscmpun.txt")):
            return ln
    return None


def solve_from_control_file(
    control_file: str, sysdir: str | None = None
) -> int:  # pragma: no cover - needs GAMS
    """Entry point GAMS invokes: solve the model described by ``control_file``.

    ``sysdir`` is the GAMS system directory (passed explicitly by GAMS); when
    omitted it is discovered from the control file.  It is needed so the GAMS
    shared libraries -- and their dependencies, e.g. ``libjoatdclib`` -- load
    from the right place via ``gmoCreateD`` / ``gevCreateD``.

    Returns a process exit code (0 on success).
    """
    try:
        import gams.core.gev as gev
        import gams.core.gmo as gmo
    except Exception as exc:  # gamsapi[core] not installed
        sys.stderr.write(
            "discopt GAMS link requires the GAMS expert-level Python API.\n"
            "Install it from your GAMS system: pip install gamsapi[core]\n"
            f"({exc})\n"
        )
        return 1

    if not (sysdir and os.path.isdir(sysdir)):
        sysdir = _gams_sysdir(control_file)

    # Handles are created and *always* freed (try/finally) so the warm daemon
    # does not leak native GMO/GEV memory across solves.
    gev_h = gmo_h = None
    gev_ok = gmo_ok = False
    try:
        gev_h = gev.new_gevHandle_tp()
        rc, msg = gev.gevCreateD(gev_h, sysdir, 256) if sysdir else gev.gevCreate(gev_h, 256)
        if not rc:
            sys.stderr.write(f"gevCreate failed: {msg}\n")
            return 1
        gev_ok = True
        gev.gevInitEnvironmentLegacy(gev_h, control_file)

        gmo_h = gmo.new_gmoHandle_tp()
        rc, msg = gmo.gmoCreateD(gmo_h, sysdir, 256) if sysdir else gmo.gmoCreate(gmo_h, 256)
        if not rc:
            sys.stderr.write(f"gmoCreate failed: {msg}\n")
            return 1
        gmo_ok = True
        gmo.gmoRegisterEnvironment(gmo_h, gev.gevHandleToPtr(gev_h))
        gmo.gmoLoadDataLegacy(gmo_h)

        # Objective handled as a function (objective variable substituted out),
        # with 0-based column/row indexing.
        gmo.gmoObjStyleSet(gmo_h, gmo.gmoObjType_Fun)
        gmo.gmoIndexBaseSet(gmo_h, 0)

        view = _GmoAdapter(gmo_h, gmo)
        t0 = time.perf_counter()
        notes = _unhonored_gams_options(gev, gev_h, gmo, gmo_h)
        try:
            model, result = solve_view(view, **_read_gams_options(gev, gev_h))
        except GamsTranslationError as exc:
            # A model the link refuses (an unsupported function, row or variable
            # type) is GAMS's "capability problem": report it with the reason in
            # the log and listing rather than crashing without a status file.
            gev.gevStatCon(gev_h)
            _log_stat(gev, gev_h, f"*** discopt: {exc}")
            gmo.gmoModelStatSet(gmo_h, MODELSTAT_NO_SOLUTION_RETURNED)
            gmo.gmoSolveStatSet(gmo_h, SOLVESTAT_CAPABILITY)
            gmo.gmoUnloadSolutionLegacy(gmo_h)
            gev.gevStatEOF(gev_h)
            return 0
        resused = time.perf_counter() - t0

        # GAMS requires the solver to open, write, and finalize a status file via
        # the environment object; otherwise it reports solveStat=13 ("SOLVER DID
        # NOT WRITE A STATUS FILE") regardless of the GMO solution we unload.
        gev.gevStatCon(gev_h)
        for note in notes:
            _log_stat(gev, gev_h, note)
        _write_solution(gmo_h, gmo, model, result, resused=resused)
        gev.gevStatEOF(gev_h)
        return 0
    finally:
        _free_handle(gmo, gmo_h, gmo_ok, "gmoFree", "delete_gmoHandle_tp")
        _free_handle(gev, gev_h, gev_ok, "gevFree", "delete_gevHandle_tp")


#: Longest line written through ``gevLogStat``; a longer or multi-line message
#: was measured (GAMS 53) to vanish from the listing without an error.
_STAT_LINE_MAX = 200


def _log_stat(gev, gev_h, text: str) -> None:  # pragma: no cover - needs GAMS
    """Write ``text`` to the GAMS log and listing, one bounded line at a time."""
    for line in str(text).splitlines() or [""]:
        while True:
            gev.gevLogStat(gev_h, line[:_STAT_LINE_MAX])
            line = line[_STAT_LINE_MAX:]
            if not line:
                break


def _free_handle(mod, handle, created: bool, free_name: str, delete_name: str) -> None:
    """Release a gamsapi handle: free the native object, then the SWIG wrapper.

    ``gmoFree``/``gevFree`` are called only when the object was actually created
    (calling them on an uncreated handle can crash); the ``delete_*Handle_tp``
    wrapper deleter is best-effort and skipped if the binding lacks it.
    """
    if handle is None:
        return
    if created:
        try:
            getattr(mod, free_name)(handle)
        except Exception as exc:  # noqa: BLE001 - teardown must not fail the solve link
            # A repeatedly failing free is a native leak across GAMS calls, which
            # is invisible unless it is logged here.
            logger.debug("%s failed: %s: %s", free_name, type(exc).__name__, exc)
    deleter = getattr(mod, delete_name, None)
    if deleter is not None:
        try:
            deleter(handle)
        except Exception as exc:  # noqa: BLE001 - teardown must not fail the solve link
            logger.debug("%s failed: %s: %s", delete_name, type(exc).__name__, exc)


def _write_solution(
    gmo_h, gmo, model: Model, result: SolveResult, *, resused: float | None = None
) -> None:  # pragma: no cover
    """Write primal levels and the GAMS model/solve status back into GMO.

    resused is the seconds the link spent translating and solving; it is
    reported as the model's resusd (result.wall_time when not given).
    """
    model_stat, solve_stat = status_to_gams(result, _has_discrete(model))
    gmo.gmoModelStatSet(gmo_h, model_stat)
    gmo.gmoSolveStatSet(gmo_h, solve_stat)

    if result.x is not None:
        # Only the GMO columns, in GMO order: translation may add auxiliary
        # variables (a semicontinuous on/off binary) after them.
        columns = getattr(model, "_gams_columns", model._variables)
        x = gmo.doubleArray(len(columns))
        for j, var in enumerate(columns):
            x[j] = float(result.x[var.name])
        # Sets the variable levels AND computes every equation's level from
        # them; per-column gmoSetVarLOne left each row at its input level
        # (measured: a binding knapsack row listed at 0 instead of 100).
        gmo.gmoSetSolutionPrimal(gmo_h, x)
        if result.objective is not None:
            gmo.gmoSetHeadnTail(gmo_h, gmo.gmoHobjval, float(result.objective))
    # Export a valid global dual bound even when the gap is open and even with no
    # incumbent; an invalid/unavailable bound stays unexported.  GAMS reports the
    # gmoTmipbest tail as the model's objest (there is no gmoHobjest).
    bound = getattr(result, "bound", None)
    if getattr(result, "bound_valid", False) and bound is not None and math.isfinite(bound):
        gmo.gmoSetHeadnTail(gmo_h, gmo.gmoTmipbest, float(bound))
    gmo.gmoSetHeadnTail(
        gmo_h, gmo.gmoHresused, float(result.wall_time if resused is None else resused)
    )
    if result.node_count is not None:
        gmo.gmoSetHeadnTail(gmo_h, gmo.gmoTmipnod, float(result.node_count))
    # discopt returns no duals here: say so, rather than let GAMS list every
    # marginal as 0, which reads as a computed dual.
    gmo.gmoSetHeadnTail(gmo_h, gmo.gmoHmarginals, 0.0)
    gmo.gmoUnloadSolutionLegacy(gmo_h)


class _GmoAdapter:  # pragma: no cover - thin wrapper over gamsapi calls
    """Adapt a gamsapi GMO handle to the :class:`GmoView` protocol.

    Isolates the version-specific GMO calling convention (validated against
    GAMS 53 / gamsapi).  Index base is set to 0 by the caller, so column/row
    indices here are 0-based.  Three GMO peculiarities are handled here:

    * **Output buffers.** Vector/instruction getters fill preallocated SWIG
      ``intArray`` / ``doubleArray`` buffers and return ``[rc, length]`` (or
      ``[rc, nz, nlnz]``), rather than returning Python lists.
    * **Constant pool.** ``gmoPPool`` exposes the nonlinear constant pool as a
      raw ``double *``; we read ``gmoNLConst`` doubles from it via ``ctypes``.
    * **Column space.** Nonlinear instruction var-fields are 1-based indices in
      the *original* GAMS column space; :meth:`_remap_var_fields` maps them into
      the reduced solver column space (1-based) the translator expects.  The
      objective equation residual is sign-flipped, recovered by
      :meth:`obj_nl_sign`.
    """

    def __init__(self, gmo_h, gmo):
        self._h = gmo_h
        self._gmo = gmo
        self._n: int = int(gmo.gmoN(gmo_h))
        # Without a dictionary file (``option dictfile=0;``, as Pyomo's GAMS
        # writer emits, or a registration lacking ``dictType``) every
        # ``gmoGet*NameOne`` call prints "gmoDict failed" to stdout; check once
        # and skip the lookups (#1972).  ``gmoNameDict`` is absent only from
        # gamsapi builds that predate it; keep the unguarded lookups there.
        try:
            dict_path = str(gmo.gmoNameDict(gmo_h) or "")
        except AttributeError:
            self._has_dict = True
        else:
            self._has_dict = bool(dict_path) and os.path.isfile(dict_path)
        nc = int(gmo.gmoNLConst(gmo_h))
        self._pool: list[float]
        if nc > 0:
            buf = (ctypes.c_double * nc).from_address(int(gmo.gmoPPool(gmo_h)))
            self._pool = [float(v) for v in buf]
        else:
            self._pool = []
        # An individual function's instruction list cannot exceed the model's
        # total nonlinear code size; pad for the header/store sentinels.
        self._cap = max(int(gmo.gmoNLCodeSize(gmo_h)), 16) + 16

    def name(self) -> str:
        try:
            return str(self._gmo.gmoNameModel(self._h))
        except Exception:
            return "gams_model"

    def num_vars(self) -> int:
        return self._n

    def num_rows(self) -> int:
        return int(self._gmo.gmoM(self._h))

    def minimize(self) -> bool:
        return bool(self._gmo.gmoSense(self._h) == self._gmo.gmoObj_Min)

    def constants(self) -> list[float]:
        return self._pool

    def var_lower(self, j: int) -> float:
        return float(self._gmo.gmoGetVarLowerOne(self._h, j))

    def var_upper(self, j: int) -> float:
        return float(self._gmo.gmoGetVarUpperOne(self._h, j))

    def var_type(self, j: int) -> int:
        return int(self._gmo.gmoGetVarTypeOne(self._h, j))

    def var_name(self, j: int) -> str:
        # The GAMS dictionary may be unavailable (e.g. a control file saved
        # without it); fall back to a positional name in that case.
        if not self._has_dict:
            return ""
        try:
            name = str(self._gmo.gmoGetVarNameOne(self._h, j))
        except Exception:
            return ""
        return "" if (not name or name == "ERROR") else name

    def var_level(self, j: int) -> float:
        return float(self._gmo.gmoGetVarLOne(self._h, j))

    def obj_constant(self) -> float:
        return float(self._gmo.gmoObjConst(self._h))

    def obj_nl_sign(self) -> float:
        # Objective stored as ``objVarJac * objvar + ... + nl = rhs``; the true
        # objective nonlinear part is ``-nl / objVarJac``.
        jac = float(self._gmo.gmoObjJacVal(self._h))
        if jac == 0.0:
            raise GamsTranslationError("objective variable has a zero Jacobian in its row")
        return -1.0 / jac

    def obj_linear(self) -> dict[int, float]:
        gmo, n = self._gmo, self._n
        jac = gmo.doubleArray(n)
        nlflag = gmo.intArray(n)
        gmo.gmoGetObjVector(self._h, jac, nlflag)
        # Columns flagged nonlinear carry a reference-point derivative, not a
        # constant coefficient, so they are excluded from the linear part.
        return {j: float(jac[j]) for j in range(n) if nlflag[j] == 0 and jac[j] != 0.0}

    def obj_nl(self) -> tuple[list[int], list[int]]:
        gmo = self._gmo
        op, fld = gmo.intArray(self._cap), gmo.intArray(self._cap)
        ret = gmo.gmoDirtyGetObjFNLInstr(self._h, op, fld)
        ln = _instr_len(ret)
        opcodes = [int(op[i]) for i in range(ln)]
        fields = [int(fld[i]) for i in range(ln)]
        return opcodes, self._remap_var_fields(opcodes, fields)

    def row_name(self, i: int) -> str:
        if not self._has_dict:
            return ""
        try:
            name = str(self._gmo.gmoGetEquNameOne(self._h, i))
        except Exception:
            return ""
        return "" if (not name or name == "ERROR") else name

    def row_sense(self, i: int) -> str:
        gmo = self._gmo
        t = gmo.gmoGetEquTypeOne(self._h, i)
        sense = {gmo.gmoequ_E: "==", gmo.gmoequ_L: "<=", gmo.gmoequ_G: ">=", gmo.gmoequ_N: "free"}
        if t not in sense:
            # =X= (external), =C= (conic) and =B= (logic) rows: refuse by name
            # rather than with a bare KeyError.
            kind = {gmo.gmoequ_X: "=X= external", gmo.gmoequ_C: "=C= conic"}.get(
                t, f"=B= logic (type {t})"
            )
            raise GamsTranslationError(
                f"equation '{self.row_name(i) or i}' is an {kind} row, which the discopt "
                "GAMS link does not support"
            )
        return sense[t]

    def row_rhs(self, i: int) -> float:
        return float(self._gmo.gmoGetRhsOne(self._h, i))

    def row_constant(self, i: int) -> float:
        return 0.0

    def row_linear(self, i: int) -> dict[int, float]:
        gmo, n = self._gmo, self._n
        jidx = gmo.intArray(n)
        coef = gmo.doubleArray(n)
        nlflag = gmo.intArray(n)
        ret = gmo.gmoGetRowSparse(self._h, i, jidx, coef, nlflag)
        nz = ret[1] if isinstance(ret, (list, tuple)) else int(ret)
        return {
            int(jidx[k]): float(coef[k]) for k in range(nz) if nlflag[k] == 0 and coef[k] != 0.0
        }

    def row_nl(self, i: int) -> tuple[list[int], list[int]]:
        gmo = self._gmo
        op, fld = gmo.intArray(self._cap), gmo.intArray(self._cap)
        ret = gmo.gmoDirtyGetRowFNLInstr(self._h, i, op, fld)
        ln = _instr_len(ret)
        opcodes = [int(op[k]) for k in range(ln)]
        fields = [int(fld[k]) for k in range(ln)]
        return opcodes, self._remap_var_fields(opcodes, fields)

    def _remap_var_fields(self, opcodes: list[int], fields: list[int]) -> list[int]:
        """Remap 1-based original-column var-fields into solver column space."""
        out: list[int] = []
        for op, f in zip(opcodes, fields):
            if op in VAR_FIELD_OPCODES:
                # gmoGetjSolver maps a 0-based model column to its solver column.
                out.append(int(self._gmo.gmoGetjSolver(self._h, f - 1)) + 1)
            else:
                out.append(f)
        return out


def _instr_len(ret) -> int:
    """Extract the instruction count from a ``gmoDirtyGet*FNLInstr`` return."""
    return ret[1] if isinstance(ret, (list, tuple)) else int(ret)


def _parse_gams_args(args: list[str]) -> tuple[str | None, str | None]:
    """Resolve ``(control_file, sysdir)`` from a solver script's arguments.

    GAMS does not call a script solver with just the control file: the classic
    interface passes six arguments -- ``<scrdir> <workdir> <prmfile> <cntrfile>
    <sysdir> <solvername>`` -- so the control file is the 4th and the system
    directory the 5th.  We resolve them by *content* (the existing file vs. the
    existing GAMS system directory) so the link is robust to convention drift
    and also works when invoked directly with a single control-file argument.
    """
    # The control file is specifically the ``gamscntr*`` file; match it first so
    # we never mistake the parameter file (``gmsprmun.dat``) or a model-data file
    # for it.  Only if no such file is present do we fall back to a generic
    # ``.dat`` and finally to the first existing file (direct single-arg use).
    files = [tok for tok in args if os.path.isfile(tok)]
    control_file = next(
        (tok for tok in files if os.path.basename(tok).startswith("gamscntr")), None
    )
    if control_file is None:
        control_file = next((tok for tok in files if tok.endswith(".dat")), None)
    if control_file is None:
        control_file = files[0] if files else None

    sysdir = next(
        (
            tok
            for tok in args
            if os.path.isdir(tok) and os.path.isfile(os.path.join(tok, "gmscmpun.txt"))
        ),
        None,
    )
    return control_file, sysdir


def main(argv: list[str] | None = None) -> int:
    """Console entry point: GAMS solver link.

    Accepts either a single control-file path (direct invocation) or the full
    GAMS script-solver argument list (``<scrdir> <workdir> <prm> <cntr> <sysdir>
    <name>``).

    By default the solve is routed through a warm :mod:`discopt.gams.daemon`
    (spawned on demand) to avoid per-solve Python/JAX startup; set
    ``DISCOPT_GAMS_NO_DAEMON=1`` to always solve in-process.  If the daemon is
    unreachable and cannot be started, this falls back to an in-process solve so
    correctness never depends on the daemon.
    """
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        sys.stderr.write("usage: discopt-gams <gams-control-file>\n")
        return 0 if args else 1

    control_file, sysdir = _parse_gams_args(args)
    if control_file is None:
        sys.stderr.write(f"discopt-gams: no control file found in arguments: {args}\n")
        return 1
    if os.environ.get("DISCOPT_GAMS_NO_DAEMON", "0") != "1":
        from .daemon import solve_via_daemon

        rc = solve_via_daemon(control_file, sysdir=sysdir)
        if rc is not None:
            return rc
    return solve_from_control_file(control_file, sysdir)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
