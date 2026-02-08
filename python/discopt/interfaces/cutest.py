"""
CUTEst interface for discopt.

Wraps PyCUTEst to provide:
  - CUTEstProblem: metadata extraction and problem management
  - NLPEvaluatorFromCUTEst: evaluator matching the NLPEvaluator interface
  - Problem discovery helpers: list/filter/load CUTEst problems

PyCUTEst is an optional dependency. Install with:
    pip install discopt[cutest]

Requires gfortran and the CUTEst/SIFDecode libraries.
See https://jfowkes.github.io/pycutest/ for setup instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pycutest  # type: ignore[import-untyped]

    _HAS_PYCUTEST = True
except (ImportError, RuntimeError):
    # RuntimeError is raised by pycutest when CUTEst env vars are not set
    _HAS_PYCUTEST = False


def _require_pycutest() -> None:
    """Raise ImportError with helpful message if pycutest is not installed."""
    if not _HAS_PYCUTEST:
        raise ImportError(
            "pycutest is required for CUTEst support. Install with:\n"
            "  pip install discopt[cutest]\n"
            "Also requires gfortran and CUTEst/SIFDecode libraries.\n"
            "See https://jfowkes.github.io/pycutest/ for setup."
        )


# CUTEst classification codes mapped to pycutest filter strings.
# pycutest.find_problems() expects these full string values.
_OBJECTIVE_TYPES = {
    "N": "none",
    "C": "constant",
    "L": "linear",
    "Q": "quadratic",
    "S": "sum of squares",
    "O": "other",
}

_CONSTRAINT_TYPES = {
    "U": "unconstrained",
    "X": "fixed variables",
    "B": "bounds",
    "N": "network",
    "L": "linear",
    "Q": "quadratic",
    "O": "other",
}


@dataclass
class CUTEstProblemInfo:
    """Metadata about a CUTEst problem, extracted from classification."""

    name: str
    n: int  # number of variables
    m: int  # number of constraints
    objective_type: str  # from classification
    constraint_type: str  # from classification
    regularity: str  # R=regular, I=irregular
    degree: int  # highest degree of derivatives
    is_variable_dimension: bool  # "V" problems allow sifParams to change n
    classification: str  # raw classification string


class CUTEstProblem:
    """
    Wraps a PyCUTEst problem with metadata extraction and evaluator creation.

    Usage:
        prob = CUTEstProblem("ROSENBR")
        info = prob.info
        evaluator = prob.to_evaluator()

        # Or use with discopt's Ipopt solver directly:
        from discopt.solvers.nlp_ipopt import solve_nlp
        result = solve_nlp(evaluator, prob.x0)
    """

    def __init__(self, name: str, sif_params: Optional[dict] = None) -> None:
        _require_pycutest()
        self._name = name
        self._sif_params = sif_params or {}
        self._problem = pycutest.import_problem(name, sifParams=self._sif_params)
        self._info: Optional[CUTEstProblemInfo] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def problem(self):
        """Access the underlying pycutest problem object."""
        return self._problem

    @property
    def n(self) -> int:
        """Number of variables."""
        return self._problem.n

    @property
    def m(self) -> int:
        """Number of constraints."""
        return self._problem.m

    @property
    def x0(self) -> np.ndarray:
        """Starting point."""
        return self._problem.x0.copy()

    @property
    def bl(self) -> np.ndarray:
        """Variable lower bounds."""
        return self._problem.bl.copy()

    @property
    def bu(self) -> np.ndarray:
        """Variable upper bounds."""
        return self._problem.bu.copy()

    @property
    def cl(self) -> Optional[np.ndarray]:
        """Constraint lower bounds (None if unconstrained)."""
        if self.m == 0:
            return None
        return self._problem.cl.copy()

    @property
    def cu(self) -> Optional[np.ndarray]:
        """Constraint upper bounds (None if unconstrained)."""
        if self.m == 0:
            return None
        return self._problem.cu.copy()

    @property
    def is_eq_cons(self) -> Optional[np.ndarray]:
        """Boolean array: True for equality constraints."""
        if self.m == 0:
            return None
        return self._problem.is_eq_cons.copy()

    @property
    def is_linear_cons(self) -> Optional[np.ndarray]:
        """Boolean array: True for linear constraints."""
        if self.m == 0:
            return None
        return self._problem.is_linear_cons.copy()

    @property
    def info(self) -> CUTEstProblemInfo:
        """Problem metadata from CUTEst classification."""
        if self._info is None:
            self._info = self._extract_info()
        return self._info

    def _extract_info(self) -> CUTEstProblemInfo:
        """Extract metadata using pycutest.problem_properties()."""
        props = pycutest.problem_properties(self._name)

        # problem_properties returns full strings like 'sum of squares', 'unconstrained'
        obj_type = props.get("objective", "other")
        con_type = props.get("constraints", "unconstrained")
        regular = props.get("regular", True)
        degree = props.get("degree", 2)

        # Variable-dimension problems have userN=True
        is_variable_dim = props.get("n", self.n) != self.n

        # Build a classification string from the properties
        classification = f"{obj_type}/{con_type}/d={degree}"

        return CUTEstProblemInfo(
            name=self._name,
            n=self.n,
            m=self.m,
            objective_type=obj_type,
            constraint_type=con_type,
            regularity="R" if regular else "I",
            degree=degree,
            is_variable_dimension=is_variable_dim,
            classification=classification,
        )

    def to_evaluator(self) -> "NLPEvaluatorFromCUTEst":
        """Create an NLP evaluator from this problem."""
        return NLPEvaluatorFromCUTEst(self)

    def to_instance_info(self):
        """Convert to benchmark InstanceInfo for integration with benchmark runner."""
        from benchmarks.metrics import InstanceInfo

        info = self.info
        n_lin = int(np.sum(self.is_linear_cons)) if self.is_linear_cons is not None else 0

        return InstanceInfo(
            name=self._name,
            num_variables=self.n,
            num_constraints=self.m,
            num_integer_vars=0,  # CUTEst is continuous NLP only
            num_binary_vars=0,
            num_continuous_vars=self.n,
            num_nonlinear_constraints=self.m - n_lin,
            problem_class=f"cutest_{info.constraint_type}",
            best_known_objective=None,
            is_convex=None,
            source="cutest",
        )

    def close(self) -> None:
        """Release the compiled problem resources."""
        # pycutest doesn't have an explicit close, but we can help GC
        self._problem = None

    def __repr__(self) -> str:
        return (
            f"CUTEstProblem({self._name!r}, n={self.n}, m={self.m}, "
            f"classification={self.info.classification!r})"
        )


class NLPEvaluatorFromCUTEst:
    """
    NLP evaluator wrapping a CUTEst problem via PyCUTEst callbacks.

    Provides the same interface as NLPEvaluator and NLPEvaluatorFromNl:
      - evaluate_objective(x) -> float
      - evaluate_gradient(x) -> ndarray
      - evaluate_hessian(x) -> ndarray
      - evaluate_constraints(x) -> ndarray
      - evaluate_jacobian(x) -> ndarray
      - n_variables, n_constraints, variable_bounds

    Uses PyCUTEst's analytic derivatives (not finite differences).
    """

    def __init__(self, cutest_problem: CUTEstProblem) -> None:
        self._prob = cutest_problem
        self._p = cutest_problem.problem
        self._n_variables = cutest_problem.n
        self._n_constraints = cutest_problem.m

    def evaluate_objective(self, x: np.ndarray) -> float:
        """Evaluate objective at x. Returns scalar."""
        x = np.asarray(x, dtype=np.float64)
        return float(self._p.obj(x))

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient of objective at x. Returns (n,) array."""
        x = np.asarray(x, dtype=np.float64)
        _, g = self._p.obj(x, gradient=True)
        return np.asarray(g, dtype=np.float64)

    def evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian of objective at x. Returns (n, n) dense array.

        For constrained problems this returns the objective Hessian only
        (not the Lagrangian Hessian). Use evaluate_lagrangian_hessian()
        for the full Lagrangian Hessian.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._n_constraints > 0:
            # ihess returns objective Hessian for constrained problems
            H = self._p.ihess(x)
        else:
            H = self._p.hess(x)
        return np.asarray(H, dtype=np.float64)

    def evaluate_lagrangian_hessian(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate Hessian of the Lagrangian at (x, v). Returns (n, n) dense array.

        L(x, v) = f(x) + v^T c(x)
        """
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        H = self._p.hess(x, v=v)
        return np.asarray(H, dtype=np.float64)

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all constraint bodies at x. Returns (m,) array."""
        if self._n_constraints == 0:
            return np.array([], dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        c = self._p.cons(x)
        return np.asarray(c, dtype=np.float64)

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian of constraints at x. Returns (m, n) dense array."""
        if self._n_constraints == 0:
            return np.empty((0, self._n_variables), dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        _, J = self._p.cons(x, gradient=True)
        return np.asarray(J, dtype=np.float64)

    def evaluate_sparse_hessian(
        self, x: np.ndarray, v: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate sparse Hessian of the Lagrangian. Returns (vals, rows, cols).

        For large-scale problems where dense Hessian is prohibitive.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._n_constraints > 0 and v is not None:
            v = np.asarray(v, dtype=np.float64)
            H = self._p.sphess(x, v=v)
        elif self._n_constraints > 0:
            H = self._p.isphess(x)
        else:
            H = self._p.sphess(x)
        # H is scipy.sparse.csc_matrix
        coo = H.tocoo()
        return (
            np.asarray(coo.data, dtype=np.float64),
            np.asarray(coo.row, dtype=np.int32),
            np.asarray(coo.col, dtype=np.int32),
        )

    def evaluate_sparse_jacobian(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate sparse Jacobian of constraints. Returns (vals, rows, cols).

        For large-scale problems where dense Jacobian is prohibitive.
        """
        if self._n_constraints == 0:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )
        x = np.asarray(x, dtype=np.float64)
        # slagjac returns (sparse_jac, grad_lagrangian) — we only need the Jacobian
        J_sparse, _ = self._p.slagjac(x)
        coo = J_sparse.tocoo()
        return (
            np.asarray(coo.data, dtype=np.float64),
            np.asarray(coo.row, dtype=np.int32),
            np.asarray(coo.col, dtype=np.int32),
        )

    @property
    def n_variables(self) -> int:
        """Total number of variables."""
        return self._n_variables

    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return self._n_constraints

    @property
    def variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lb, ub) arrays of shape (n,) for all variables."""
        return self._prob.bl, self._prob.bu

    @property
    def constraint_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Returns (cl, cu) arrays of shape (m,) for all constraints, or None."""
        if self._n_constraints == 0:
            return None
        return self._prob.cl, self._prob.cu


# ─────────────────────────────────────────────────────────────
# Problem discovery helpers
# ─────────────────────────────────────────────────────────────


def list_cutest_problems(
    objective: Optional[str] = None,
    constraints: Optional[str] = None,
    regular: Optional[bool] = None,
    max_n: Optional[int] = None,
    max_m: Optional[int] = None,
    userN: Optional[bool] = None,
) -> list[str]:
    """
    List available CUTEst problems matching filter criteria.

    Args:
        objective: Filter by objective type — either a single-letter CUTEst code
            (N/C/L/Q/S/O) or a pycutest string (e.g. "quadratic", "sum of squares").
        constraints: Filter by constraint type — either a single-letter CUTEst code
            (U/X/B/N/L/Q/O) or a pycutest string (e.g. "unconstrained", "bounds").
        regular: If True, only regular problems; if False, only irregular.
        max_n: Maximum number of variables (None = no limit).
        max_m: Maximum number of constraints (None = no limit).
        userN: If True, only variable-dimension problems; if False, exclude them.

    Returns:
        Sorted list of problem names matching all filters.
    """
    _require_pycutest()

    # pycutest.find_problems() expects full strings like 'unconstrained',
    # not CUTEst single-letter codes like 'U'. Map codes to strings.
    if objective is not None and len(objective) == 1:
        objective = _OBJECTIVE_TYPES.get(objective, objective)
    if constraints is not None and len(constraints) == 1:
        constraints = _CONSTRAINT_TYPES.get(constraints, constraints)

    # Build the classification filter for pycutest.find_problems()
    kwargs: dict[str, str | bool] = {}
    if objective is not None:
        kwargs["objective"] = objective
    if constraints is not None:
        kwargs["constraints"] = constraints
    if regular is not None:
        kwargs["regular"] = regular
    if userN is not None:
        kwargs["userN"] = userN

    problems = pycutest.find_problems(**kwargs)

    # Apply dimension filters if specified
    if max_n is not None or max_m is not None:
        filtered = []
        for name in problems:
            try:
                props = pycutest.problem_properties(name)
                n = props.get("n", 0)
                m = props.get("m", 0)
                if max_n is not None and n > max_n:
                    continue
                if max_m is not None and m > max_m:
                    continue
                filtered.append(name)
            except Exception:
                # Skip problems we can't query
                continue
        problems = filtered

    return sorted(problems)


def load_cutest_problem(
    name: str,
    sif_params: Optional[dict] = None,
) -> CUTEstProblem:
    """
    Load a CUTEst problem by name.

    Args:
        name: CUTEst problem name (e.g., "ROSENBR", "HS35").
        sif_params: Optional SIF parameters for variable-dimension problems.

    Returns:
        CUTEstProblem instance.
    """
    _require_pycutest()
    return CUTEstProblem(name, sif_params=sif_params)
