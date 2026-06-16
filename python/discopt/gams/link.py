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
import os
import sys
from typing import TYPE_CHECKING

from discopt.modeling.core import Model, SolveResult, VarType

from .gmo_translate import model_from_gmo
from .instructions import VAR_FIELD_OPCODES

if TYPE_CHECKING:
    from .gmo_translate import GmoView

# ── GAMS status constants (gmomcc) ───────────────────────────────────────────
MODELSTAT_OPTIMAL = 1
MODELSTAT_LOCALLY_OPTIMAL = 2
MODELSTAT_UNBOUNDED = 3
MODELSTAT_INFEASIBLE = 4
MODELSTAT_FEASIBLE = 7  # intermediate non-optimal, feasible solution available
MODELSTAT_INTEGER = 8  # integer solution (optimal MIP/MINLP)
MODELSTAT_ERROR_NO_SOLUTION = 13
MODELSTAT_NO_SOLUTION_RETURNED = 14

SOLVESTAT_NORMAL = 1
SOLVESTAT_ITERATION = 2
SOLVESTAT_RESOURCE = 3  # time / resource interrupt
SOLVESTAT_TERMINATED = 4
SOLVESTAT_SOLVER_ERROR = 10
SOLVESTAT_SYSTEM_ERROR = 13


def is_available() -> bool:
    """Return ``True`` if the GAMS expert-level Python API (gamsapi) is present."""
    try:  # pragma: no cover - depends on optional gamsapi
        import gams.core.gev  # noqa: F401
        import gams.core.gmo  # noqa: F401

        return True
    except Exception:
        return False


def status_to_gams(result: SolveResult, has_discrete: bool) -> tuple[int, int]:
    """Map a discopt :class:`SolveResult` to ``(modelStat, solveStat)``.

    ``has_discrete`` selects between the continuous (``Optimal``) and discrete
    (``Integer``) optimal model-status codes, matching GAMS conventions.

    Globality is reported honestly: a discopt ``"optimal"`` result counts as a
    *certified global* optimum -- GAMS ``Optimal`` / ``Integer`` -- only when its
    gap is mathematically certified (``result.gap_certified``).  When discopt
    falls back to heuristic NLP branch-and-bound on a nonconvex model the result
    is locally optimal only, so it maps to GAMS ``LocallyOptimal`` rather than
    overstating it as a global optimum.
    """
    status = (result.status or "").lower()
    has_solution = result.x is not None

    if status == "optimal":
        if not getattr(result, "gap_certified", True):
            return MODELSTAT_LOCALLY_OPTIMAL, SOLVESTAT_NORMAL
        model_stat = MODELSTAT_INTEGER if has_discrete else MODELSTAT_OPTIMAL
        return model_stat, SOLVESTAT_NORMAL
    if status == "feasible":
        return MODELSTAT_FEASIBLE, SOLVESTAT_NORMAL
    if status == "infeasible":
        return MODELSTAT_INFEASIBLE, SOLVESTAT_NORMAL
    if status == "unbounded":
        return MODELSTAT_UNBOUNDED, SOLVESTAT_NORMAL
    if status in ("time_limit", "node_limit"):
        model_stat = MODELSTAT_FEASIBLE if has_solution else MODELSTAT_NO_SOLUTION_RETURNED
        return model_stat, SOLVESTAT_RESOURCE
    if status == "iteration_limit":
        model_stat = MODELSTAT_FEASIBLE if has_solution else MODELSTAT_NO_SOLUTION_RETURNED
        return model_stat, SOLVESTAT_ITERATION
    # error / unknown
    return MODELSTAT_ERROR_NO_SOLUTION, SOLVESTAT_SOLVER_ERROR


def _has_discrete(model: Model) -> bool:
    return any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables)


def solve_view(view: "GmoView", *, time_limit: float = 1e10, gap: float = 1e-4):
    """Translate a GMO view, solve with discopt, and return ``(model, result)``.

    This is the GAMS-library-free core of the link: given anything implementing
    :class:`~discopt.gams.gmo_translate.GmoView` it produces a solved discopt
    model, so it can be exercised in tests with an in-memory fake.
    """
    model = model_from_gmo(view)
    result = model.solve(time_limit=time_limit, gap_tolerance=gap)
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
        model, result = solve_view(view)

        # GAMS requires the solver to open, write, and finalize a status file via
        # the environment object; otherwise it reports solveStat=13 ("SOLVER DID
        # NOT WRITE A STATUS FILE") regardless of the GMO solution we unload.
        gev.gevStatCon(gev_h)
        _write_solution(gmo_h, gmo, model, result)
        gev.gevStatEOF(gev_h)
        return 0
    finally:
        _free_handle(gmo, gmo_h, gmo_ok, "gmoFree", "delete_gmoHandle_tp")
        _free_handle(gev, gev_h, gev_ok, "gevFree", "delete_gevHandle_tp")


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
        except Exception:
            pass
    deleter = getattr(mod, delete_name, None)
    if deleter is not None:
        try:
            deleter(handle)
        except Exception:
            pass


def _write_solution(gmo_h, gmo, model: Model, result: SolveResult) -> None:  # pragma: no cover
    """Write primal levels and the GAMS model/solve status back into GMO."""
    model_stat, solve_stat = status_to_gams(result, _has_discrete(model))
    gmo.gmoModelStatSet(gmo_h, model_stat)
    gmo.gmoSolveStatSet(gmo_h, solve_stat)

    if result.x is not None:
        for j, var in enumerate(model._variables):
            val = float(result.x[var.name])
            gmo.gmoSetVarLOne(gmo_h, j, val)
        if result.objective is not None:
            gmo.gmoSetHeadnTail(gmo_h, gmo.gmoHobjval, float(result.objective))
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
        # objective nonlinear part is ``-nl / objVarJac`` (objVarJac is +/-1).
        jac = float(self._gmo.gmoObjJacVal(self._h))
        return -1.0 / jac if jac else -1.0

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
        try:
            name = str(self._gmo.gmoGetEquNameOne(self._h, i))
        except Exception:
            return ""
        return "" if (not name or name == "ERROR") else name

    def row_sense(self, i: int) -> str:
        t = self._gmo.gmoGetEquTypeOne(self._h, i)
        return {self._gmo.gmoequ_E: "==", self._gmo.gmoequ_L: "<=", self._gmo.gmoequ_G: ">="}[t]

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
