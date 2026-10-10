"""POUNCE's structured solve report, captured in memory (#1534).

POUNCE writes its ``pounce.solve-report/v1`` document -- the same JSON as
``pounce --json-output <path> --json-detail full`` -- only to a path handed to
``Problem.solve(report_path=..., report_detail=...)``. There is no in-memory
variant in the pinned release (0.12.0), so this module routes the report through a
private temporary directory, parses it, and removes the file before returning.

What the report holds (measured, not assumed). Two POUNCE versions are in range
of discopt's pin (``pounce-solver>=0.12``) and their trajectories differ; the
behavior with jkitchin/pounce#980 (the fix for pounce#979, on POUNCE ``main``
672320d; not in 0.12.0) is described first:

* ``statistics`` -- ``iteration_count``, the ``final_*`` residuals, evaluation
  counts, ``total_wallclock_time_secs``, ``restoration_calls`` /
  ``restoration_inner_iters`` / ``restoration_outer_iters`` /
  ``restoration_wall_secs``.
* ``iterations`` -- one row per **printed** iteration row, in printed order:
  ``iter``, ``phase`` (``"main"`` or ``"restoration"``), ``objective``
  (unscaled, as printed), ``inf_pr`` (the printed column: violation of the
  original constraints), ``inf_pr_internal`` (POUNCE's slack-form, scaled
  residual that its filter and convergence test use), ``inf_du``, ``mu``,
  ``d_norm``, ``regularization``, ``alpha_dual``, ``alpha_primal``,
  ``alpha_primal_char`` (the line-search flag printed after ``alpha_pr``) and
  ``ls_trials``. The inner restoration rows (printed ``24r``, ``25r``, ...) are
  included with ``phase="restoration"``. As in the printed table, one
  restoration call puts ``"R"`` on **two** rows -- the entry (a
  ``"restoration"`` row) and the return to the main phase -- so count calls with
  ``statistics["restoration_calls"]`` or the ``"main"`` -> ``"restoration"``
  transitions, never the ``"R"`` rows. Select ``phase == "main"`` for the outer
  trajectory.
* ``solution``, ``problem`` and ``fair_metadata`` (solver version, timestamps).

``iterations`` is always present in the returned document (#1585): a solve that
stops before its first iteration gets ``[]`` rather than a missing key, so a
consumer can iterate it without a key check.

On pounce-solver **0.12.0** (no pounce#980) the report differs in two places:

1. Rows carry no ``phase`` and no ``inf_pr_internal``, and the inner
   restoration-phase rows are **not** in ``iterations``; the restoration phase
   appears only as the ``"R"`` row that entered it and the ``restoration_*``
   counts in ``statistics``.
2. ``inf_pr`` is POUNCE's internal (slack-form, scaled) primal infeasibility,
   not the printed column. The two agree whenever the slacks are at the
   constraint values; at an iterate where a slack sits off its constraint value
   (the starting point after the bound push, the iterations after a restoration
   exit) they differ visibly -- 0.94 printed against 0.955 reported at iteration
   0 of a one-row problem. Every other column matches the printed table to the
   digits it prints.

A consumer that must run on both reads ``it.get("phase", "main")``.

The convex LP/QP interior-point method (``lp-ipm`` / ``qp-ipm``, reached through
:func:`pounce.qp.solve_qp`) has no ``report_path``: that Python entry point
returns a ``QpResult`` and never writes the document. :func:`convex_report`
builds it from the ``QpResult`` instead, mapping the fields exactly as POUNCE's
own CLI does when it writes ``--json-output`` for ``solver_selection=lp-ipm`` /
``qp-ipm`` (``run_convex_qp`` in ``pounce-cli``), so one consumer reads both
engines' reports the same way.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

_logger = logging.getLogger(__name__)

#: The schema this module was written and tested against.
SOLVE_REPORT_SCHEMA = "pounce.solve-report/v1"

#: ``"full"`` carries the per-iteration trajectory; ``"summary"`` drops it.
_REPORT_DETAIL = "full"


def solve_with_report(problem: Any, *args: Any, **kwargs: Any) -> tuple[Any, dict, Optional[dict]]:
    """Call ``problem.solve(*args, **kwargs)`` and also return its solve report.

    Returns ``(x, info, report)``. ``report`` is the parsed
    ``pounce.solve-report/v1`` document, or ``None`` when POUNCE wrote none.

    A missing or unreadable report is logged at WARNING with the marker
    ``[pounce-solve-report-missing]`` and returned as ``None``: the report is a
    diagnostic, nothing in the solver reads it back, so it must never be able to
    turn a solve that succeeded into one that failed. Exceptions raised by the
    solve itself propagate unchanged.
    """
    with tempfile.TemporaryDirectory(prefix="discopt-pounce-report-") as tmp:
        path = os.path.join(tmp, "report.json")
        x, info = problem.solve(*args, report_path=path, report_detail=_REPORT_DETAIL, **kwargs)
        report = _read_report(path)
    if report is not None:
        label_declined_retry(report)
    return x, info, report


#: ``statistics`` keys that, after a declined gh#884 retry, POUNCE fills from the
#: retry although the solution and every ``final_*`` metric are the base attempt's.
_RETRY_COUNT_KEYS = (
    "iteration_count",
    "num_obj_evals",
    "num_constr_evals",
    "num_obj_grad_evals",
    "num_constr_jac_evals",
    "num_hess_evals",
)


def label_declined_retry(report: dict) -> None:
    """Label, in place, the counts a declined dual-divergence retry left in ``report``.

    #1618 B-05c. When POUNCE's gh#884 dual-divergence guard fires it re-solves,
    and when it declines the retry it reports the *base* attempt's answer. The
    report it writes (and the ``info`` dict) is then mixed: ``solution`` and every
    ``final_*`` metric are the base attempt's, but ``iteration_count``, the
    evaluation counts and the ``iterations`` trajectory are the retry's. Measured
    on the issue's toll-pricing MPEC with POUNCE 0.12.0: the log prints "Number of
    Iterations....: 64" for the base attempt and "15" for the retry, then "gh#884
    dual-divergence retry: declined ... the base attempt's answer is the one
    reported", and the report carries ``iteration_count`` 15, 16 trajectory rows
    and ``final_dual_inf`` 1.89e3 (the base attempt's; the retry ended "Solved To
    Acceptable Level" with unscaled KKT error 1.69e-5).

    The base attempt's counts are not in the report, so they cannot be
    restored here (that needs POUNCE to emit per-attempt statistics). The mixed
    document is left as POUNCE wrote it and labelled: ``statistics["counts_attempt"]``
    names the attempt the counts and trajectory describe, and
    ``statistics["solution_attempt"]`` the one the answer comes from. Only the
    declined case is labelled -- the only one measured.
    """
    stats = report.get("statistics")
    if not isinstance(stats, dict):
        return
    if stats.get("dual_divergence_signature") is True and (
        stats.get("dual_divergence_retry_promoted") is False
    ):
        stats["counts_attempt"] = "declined_retry"
        stats["solution_attempt"] = "base"
        stats["counts_attempt_keys"] = [k for k in _RETRY_COUNT_KEYS if k in stats] + ["iterations"]


def _read_report(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError) as exc:
        _logger.warning(
            "POUNCE solve report could not be read from %s: %s: %s. "
            "SolveResult.solve_report is None for this solve [pounce-solve-report-missing].",
            path,
            type(exc).__name__,
            exc,
        )
        return None
    if not isinstance(report, dict):
        _logger.warning(
            "POUNCE solve report is a %s, not a JSON object; "
            "SolveResult.solve_report is None for this solve [pounce-solve-report-missing].",
            type(report).__name__,
        )
        return None
    if report.get("schema") != SOLVE_REPORT_SCHEMA:
        # Attached anyway: the document is what POUNCE produced and carries its own
        # ``schema`` key, so a consumer can branch on it. Warn so a schema bump is
        # noticed rather than discovered as a KeyError downstream.
        _logger.warning(
            "POUNCE solve report has schema %r; discopt was written against %r. "
            "Attached as-is [pounce-solve-report-schema].",
            report.get("schema"),
            SOLVE_REPORT_SCHEMA,
        )
    # #1585: a stable schema. POUNCE omits ``iterations`` when the solve stopped
    # before iteration 0 (an ``Invalid_Option`` refusal, for one), and a consumer
    # that iterates the trajectory should not need a key check for that case.
    report.setdefault("iterations", [])
    return report


#: ``QpResult.status`` -> ``(status, status_upstream, solve_result_num)``, the same
#: mapping as ``qp_status_to_ars`` / ``status_to_solve_result_num`` in POUNCE's CLI.
_CONVEX_STATUS: dict[str, tuple[str, str, int]] = {
    "optimal": ("SolveSucceeded", "Solve_Succeeded", 0),
    "optimal_inaccurate": ("SolvedToAcceptableLevel", "Solved_To_Acceptable_Level", 1),
    "primal_infeasible": ("InfeasibleProblemDetected", "Infeasible_Problem_Detected", 200),
    "dual_infeasible": ("DivergingIterates", "Diverging_Iterates", 300),
    "iteration_limit": ("MaximumIterationsExceeded", "Maximum_Iterations_Exceeded", 400),
    "time_limit": ("MaximumWallTimeExceeded", "Maximum_WallTime_Exceeded", 400),
    "numerical_failure": ("InternalError", "Internal_Error", 500),
}


def convex_report(
    res: Any, *, wall_time: float, n_constraints: int, started_unix_nanos: int
) -> dict:
    """The ``pounce.solve-report/v1`` document for one convex ``lp-ipm``/``qp-ipm`` solve.

    ``res`` is the :class:`pounce.qp.QpResult` of a solve run with
    ``collect_iterates=True``; ``res.iterates`` becomes ``report["iterations"]``
    row for row, so the trajectory is exactly the table
    ``convex_ipm_pounce._print_trace`` prints at ``print_level > 0``.

    The convex IPM has no line search, no Hessian regularization and no
    restoration phase, so -- as in POUNCE's CLI report for this engine -- each row's
    ``d_norm`` and ``regularization`` are ``0.0``, ``ls_trials`` is ``0``,
    ``alpha_primal_char`` is ``" "``, ``inf_pr_internal`` equals ``inf_pr``, and
    every ``restoration_*`` count is zero. The evaluation counts the NLP report
    carries do not exist for a matrix-form solve and are omitted, not zeroed.
    ``fair_metadata["generated_by"]`` names this function, so a consumer can tell
    the document from one POUNCE wrote itself.
    """
    import pounce

    raw = str(res.status)
    status, upstream, srn = _CONVEX_STATUS.get(raw, ("InternalError", "Internal_Error", 500))
    x = [float(v) for v in res.x]
    obj = float(res.obj)
    residuals = dict(res.residuals or {})

    def _res(key: str) -> float:
        v = residuals.get(key)
        return float("nan") if v is None else float(v)

    iterations = [
        {
            "iter": int(it["iter"]),
            "objective": float(it["objective"]),
            "inf_pr": float(it["primal_infeasibility"]),
            "inf_pr_internal": float(it["primal_infeasibility"]),
            "inf_du": float(it["dual_infeasibility"]),
            "mu": float(it["mu"]),
            "d_norm": 0.0,
            "regularization": 0.0,
            "alpha_dual": float(it["alpha_dual"]),
            "alpha_primal": float(it["alpha_primal"]),
            "alpha_primal_char": " ",
            "ls_trials": 0,
            "phase": "main",
        }
        for it in res.iterates
    ]
    return {
        "schema": SOLVE_REPORT_SCHEMA,
        "fair_metadata": {
            "created_at_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_unix_nanos / 1e9)
            ),
            "created_at_unix_nanos": int(started_unix_nanos),
            "elapsed_seconds": float(wall_time),
            "solver": {"name": "pounce", "version": str(getattr(pounce, "__version__", ""))},
            "generated_by": "discopt.solvers._pounce_report.convex_report",
        },
        "problem": {
            "n_variables": len(x),
            "n_constraints": int(n_constraints),
            "n_objectives": 1,
            "minimize": True,
        },
        "solution": {
            "engine": "cvx-qp",
            "engine_status": raw,
            "status": status,
            "status_upstream": upstream,
            "solve_result_num": srn,
            "objective": obj,
            "x": x,
        },
        "statistics": {
            "iteration_count": int(res.iters),
            "final_objective": obj,
            "final_constr_viol": _res("primal_infeasibility"),
            "final_dual_inf": _res("dual_infeasibility"),
            "final_compl": _res("complementarity"),
            "final_kkt_error": _res("kkt_error"),
            "total_wallclock_time_secs": float(wall_time),
            "restoration_calls": 0,
            "restoration_inner_iters": 0,
            "restoration_outer_iters": 0,
            "restoration_wall_secs": 0.0,
        },
        "iterations": iterations,
    }


def report_in_model_sense(report: Optional[dict], offset: float, maximize: bool) -> None:
    """Map, in place, ``report``'s objective values to the model's own objective (#1679).

    The engine minimizes ``f_eng``; the model's objective is
    ``f = sign * (f_eng + offset)`` with ``sign = -1`` for a MAXIMIZE model. The
    matrix-form LP/QP engines never see the objective's constant term (``offset``,
    the ``obj_const`` of the extracted LP/QP data), and every engine sees a
    maximized objective negated, so the document POUNCE wrote -- or
    :func:`convex_report` built -- disagreed with ``SolveResult.objective``: the
    dispatch QP of #1679 reported ``final_objective`` 7553.38 against
    ``res.objective`` 10529.92 (the omitted constant).

    Rewritten: ``solution.objective``, ``statistics.final_objective`` and every
    ``iterations[*].objective``; ``problem.minimize`` becomes ``not maximize``.
    The engine-space values are kept verbatim under ``engine_*`` keys, and
    ``problem.objective_mapping`` records ``offset`` and ``sign``. A report already
    carrying ``objective_mapping`` is left alone, so no call path can shift it twice.
    """
    if not isinstance(report, dict):
        return
    problem = report.setdefault("problem", {})
    if not isinstance(problem, dict) or "objective_mapping" in problem:
        return
    sign = -1.0 if maximize else 1.0
    off = float(offset)

    def _map(v: Any) -> Any:
        return None if v is None else sign * (float(v) + off)

    solution = report.get("solution")
    if isinstance(solution, dict) and "objective" in solution:
        solution["engine_objective"] = solution["objective"]
        solution["objective"] = _map(solution["objective"])
    stats = report.get("statistics")
    if isinstance(stats, dict) and "final_objective" in stats:
        stats["engine_final_objective"] = stats["final_objective"]
        stats["final_objective"] = _map(stats["final_objective"])
    for row in report.get("iterations") or []:
        if isinstance(row, dict) and "objective" in row:
            row["engine_objective"] = row["objective"]
            row["objective"] = _map(row["objective"])
    problem["minimize"] = not maximize
    problem["objective_mapping"] = {
        "offset": off,
        "sign": sign,
        "rule": "objective = sign * (engine_objective + offset)",
    }
