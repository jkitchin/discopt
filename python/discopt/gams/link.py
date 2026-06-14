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

import sys
from typing import TYPE_CHECKING

from discopt.modeling.core import Model, SolveResult, VarType

from .gmo_translate import model_from_gmo

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
    """
    status = (result.status or "").lower()
    has_solution = result.x is not None

    if status == "optimal":
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


def solve_from_control_file(control_file: str) -> int:  # pragma: no cover - needs GAMS
    """Entry point GAMS invokes: solve the model described by ``control_file``.

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

    gev_h = gev.new_gevHandle_tp()
    rc, msg = gev.gevCreate(gev_h, 256)
    if not rc:
        sys.stderr.write(f"gevCreate failed: {msg}\n")
        return 1
    gev.gevInitEnvironmentLegacy(gev_h, control_file)

    gmo_h = gmo.new_gmoHandle_tp()
    rc, msg = gmo.gmoCreate(gmo_h, 256)
    if not rc:
        sys.stderr.write(f"gmoCreate failed: {msg}\n")
        return 1
    gmo.gmoRegisterEnvironment(gmo_h, gev.gevHandleToPtr(gev_h))
    gmo.gmoLoadDataLegacy(gmo_h)

    # Objective handled as a function (objective variable substituted out).
    gmo.gmoObjStyleSet(gmo_h, gmo.gmoObjType_Fun)
    gmo.gmoIndexBaseSet(gmo_h, 0)

    view = _GmoAdapter(gmo_h, gmo)
    model, result = solve_view(view)

    _write_solution(gmo_h, gmo, model, result)
    return 0


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

    Isolates the version-specific GMO calling convention. Index base is set to 0
    by the caller, so column/row indices here are 0-based.
    """

    def __init__(self, gmo_h, gmo):
        self._h = gmo_h
        self._gmo = gmo

    def name(self) -> str:
        return str(self._gmo.gmoNameModel(self._h))

    def num_vars(self) -> int:
        return int(self._gmo.gmoN(self._h))

    def num_rows(self) -> int:
        return int(self._gmo.gmoM(self._h))

    def minimize(self) -> bool:
        return bool(self._gmo.gmoSense(self._h) == self._gmo.gmoObj_Min)

    def constants(self) -> list[float]:
        return [float(c) for c in self._gmo.gmoPPool(self._h)]

    def var_lower(self, j: int) -> float:
        return float(self._gmo.gmoGetVarLowerOne(self._h, j))

    def var_upper(self, j: int) -> float:
        return float(self._gmo.gmoGetVarUpperOne(self._h, j))

    def var_type(self, j: int) -> int:
        return int(self._gmo.gmoGetVarTypeOne(self._h, j))

    def var_name(self, j: int) -> str:
        return str(self._gmo.gmoGetVarNameOne(self._h, j))

    def var_level(self, j: int) -> float:
        return float(self._gmo.gmoGetVarLOne(self._h, j))

    def obj_constant(self) -> float:
        return float(self._gmo.gmoObjConst(self._h))

    def obj_linear(self) -> dict[int, float]:
        grad = self._gmo.gmoGetObjVector(self._h)
        return {j: float(c) for j, c in enumerate(grad) if c != 0.0}

    def obj_nl(self) -> tuple[list[int], list[int]]:
        opcodes, fields = self._gmo.gmoGetObjFNLInstr(self._h)
        return list(opcodes), list(fields)

    def row_name(self, i: int) -> str:
        return str(self._gmo.gmoGetEquNameOne(self._h, i))

    def row_sense(self, i: int) -> str:
        t = self._gmo.gmoGetEquTypeOne(self._h, i)
        return {self._gmo.gmoequ_E: "==", self._gmo.gmoequ_L: "<=", self._gmo.gmoequ_G: ">="}[t]

    def row_rhs(self, i: int) -> float:
        return float(self._gmo.gmoGetRhsOne(self._h, i))

    def row_constant(self, i: int) -> float:
        return 0.0

    def row_linear(self, i: int) -> dict[int, float]:
        coefs, idxs, _nl, _nz = self._gmo.gmoGetRowSparse(self._h, i)
        return {int(j): float(c) for j, c in zip(idxs, coefs) if c != 0.0}

    def row_nl(self, i: int) -> tuple[list[int], list[int]]:
        opcodes, fields = self._gmo.gmoGetEquFNLInstr(self._h, i)
        return list(opcodes), list(fields)


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``discopt-gams <control-file>``)."""
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        sys.stderr.write("usage: discopt-gams <gams-control-file>\n")
        return 0 if args else 1
    return solve_from_control_file(args[0])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
