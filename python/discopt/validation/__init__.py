"""Independent post-solve validation of optimization solutions.

The :mod:`discopt.validation.examiner` module provides an Examiner-style
validator that re-evaluates the model at the returned point, recovers
multipliers from the active set, and checks the KKT optimality conditions
without trusting solver-reported residuals or duals.
"""

from discopt.validation.examiner import (
    ACTIVE_TOL,
    DUAL_CS_TOL,
    DUAL_FEAS_TOL,
    INTEGRALITY_TOL,
    OBJ_TOL,
    PRIMAL_CS_TOL,
    PRIMAL_FEAS_TOL,
    SHOW_TOL,
    CheckResult,
    ExaminerReport,
    assert_examined,
    examine,
)

__all__ = [
    "ACTIVE_TOL",
    "DUAL_CS_TOL",
    "DUAL_FEAS_TOL",
    "INTEGRALITY_TOL",
    "OBJ_TOL",
    "PRIMAL_CS_TOL",
    "PRIMAL_FEAS_TOL",
    "SHOW_TOL",
    "CheckResult",
    "ExaminerReport",
    "assert_examined",
    "examine",
]
