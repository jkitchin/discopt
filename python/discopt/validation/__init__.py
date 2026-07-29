"""Independent post-solve validation of optimization solutions.

The :mod:`discopt.validation.examiner` module provides an Examiner-style
validator that re-evaluates the model at the returned point, recovers
multipliers from the active set, and checks the KKT optimality conditions
without trusting solver-reported residuals or duals.

The :mod:`discopt.validation.feasibility` module provides
:func:`~discopt.validation.feasibility.verify_point`, the single row-scale-aware
incumbent feasibility verifier that every certificate-gating consumer uses (the
native-kernel seed gate, the convex-kernel adoption gate, and the Regime-C
differential panels).
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
from discopt.validation.feasibility import (
    DEFAULT_ABS_TOL,
    DEFAULT_INT_TOL,
    DEFAULT_REL_TOL,
    PointVerification,
    RowViolation,
    row_scales,
    verify_point,
)

__all__ = [
    "ACTIVE_TOL",
    "DEFAULT_ABS_TOL",
    "DEFAULT_INT_TOL",
    "DEFAULT_REL_TOL",
    "DUAL_CS_TOL",
    "DUAL_FEAS_TOL",
    "INTEGRALITY_TOL",
    "OBJ_TOL",
    "PRIMAL_CS_TOL",
    "PRIMAL_FEAS_TOL",
    "SHOW_TOL",
    "CheckResult",
    "ExaminerReport",
    "PointVerification",
    "RowViolation",
    "assert_examined",
    "examine",
    "row_scales",
    "verify_point",
]
