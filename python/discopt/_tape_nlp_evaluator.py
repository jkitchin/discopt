"""A JAX-free NLP evaluator backed by POUNCE's Rust AD tape (issue #75, Stage 3).

``_jax/nlp_evaluator.py`` is the single trigger that imports JAX on *every*
nonlinear solve — measured across 27 operator-diverse corpus instances, the first
jax import is always ``nlp_evaluator.py:22``. This module supplies the same
quantities (``f``, ``grad f``, ``g``, ``J``, and the Lagrangian Hessian, dense and
sparse) from a tape built by :mod:`discopt._nl_expr_compiler`, with no JAX at all.

**Default ON** since ``a2fb90d2``; ``DISCOPT_NLP_EVAL=jax`` is the opt-out and the
JAX evaluator is untouched beneath it. Routing derivatives through a different AD
engine is bound-CHANGING under CLAUDE.md §5 — the B&B is path-dependent, so even
last-digit differences can move the cut sequence — so the default flipped only
after the differential panel passed both bars; see :func:`tape_backend_requested`
for the numbers.

Entry experiment before any of this was written (CLAUDE.md §4), tape vs the JAX
evaluator over 66 in-repo corpus instances at 5 points each:

    f  5.48e-16   grad f  3.77e-16   g  4.55e-13   J  3.06e-15   Lagrangian H  7.82e-13

against Step 2.2's bars of 1e-10 on grad/J and 1e-8 on the Hessian.

One thing this refuses rather than approximates, because the alternative is a
silently wrong derivative:

* **Array-valued constraint bodies.** A tape node is a scalar. ``DAEBuilder``
  collocation emits one array-valued body per block, which has no scalar
  lowering.

Callers use :func:`try_build`, which returns ``None`` for it, so the JAX path
stays intact underneath.

**Gauss-Newton** (``H_obj ~ 2 JᵀJ``) *is* supported, and used to be a second
refusal here. pounce has no "residual" concept, but it has a sparse *constraint*
Jacobian, which is the same object under a different name: tape the residual
vector ``r(x)`` as the constraint rows of an auxiliary ``NlProblem`` and read its
``R x n`` Jacobian. Entry experiment (CLAUDE.md §4) over three least-squares
shapes -- dense, separable, and shared/private mixed with a constraint present --
put ``2 JᵀJ`` from that aux tape at **0.00e+00** relative error against the JAX
``_build_gauss_newton_obj_hessian``. So this needed no pounce-repo work, only the
observation that residuals and constraints tape identically.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from discopt._nl_expr_compiler import UnsupportedForTape, compile_to_nl_expr

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Constraint, Model

logger = logging.getLogger(__name__)


_POUNCE_USABLE: Optional[bool] = None


def pounce_usable() -> bool:
    """Is a POUNCE new enough to carry the tape actually installed?

    Checks the SURFACE, not just importability. Two failure modes, both hit for
    real:

    * **pounce absent.** ``pyproject`` requires ``pounce-solver>=0.9``, but a
      minimal install need not have it -- CI's AMP-coverage job installs jax,
      numpy, scipy and highspy and nothing else. Before this guard, flipping the
      default ON turned that environment from working into a hard
      ``ModuleNotFoundError`` on every solve.
    * **pounce too old.** A build predating pounce #470 exports no ``NlExpr`` at
      all. Importing succeeds and the attribute access is what explodes, deep in
      a solve. (That exact stale build also silently disabled a whole test file
      behind ``pytest.importorskip`` earlier in this work.)

    Neither is a bug to hide, so this is not exception-swallowing: it is an
    availability decision made once, logged once, and used only to choose a
    backend. Numerical failures are never routed through here.
    """
    global _POUNCE_USABLE
    if _POUNCE_USABLE is not None:
        return _POUNCE_USABLE
    try:
        import pounce
    except ImportError:
        _POUNCE_USABLE = False
        logger.info(
            "POUNCE is not installed; the tape NLP evaluator is unavailable and "
            "the JAX evaluator will be used (install `pounce-solver`)."
        )
        return _POUNCE_USABLE
    missing = [n for n in ("NlExpr", "build_nl_problem") if not hasattr(pounce, n)]
    _POUNCE_USABLE = not missing
    if missing:
        logger.warning(
            "POUNCE is installed but lacks %s, so the tape NLP evaluator is "
            "unavailable and the JAX evaluator will be used. This build predates "
            "pounce #470; rebuild the extension to enable the tape backend.",
            ", ".join(missing),
        )
    return _POUNCE_USABLE


def tape_backend_requested() -> bool:
    """True unless ``DISCOPT_NLP_EVAL`` selects the legacy JAX evaluator.

    **Default ON since the CLAUDE.md §5 panel passed both bars** (66 instances):
    cert-clean on every check -- 0 unsound bounds or statuses, 0 false optima, 0
    certification regressions, 0 infeasible incumbents over 103 cross-engine
    verifications -- and net-positive on wall, 10 faster / 0 slower, median 1.80x,
    total -43.7%, measured interleaved with a load gate and standard deviations on
    instances doing IDENTICAL work (44 of 46 node counts unchanged).

    ``DISCOPT_NLP_EVAL=jax`` is the opt-out and the JAX evaluator is untouched
    beneath it, per §5's graduation rule. Read per call rather than cached at
    import so a test can flip it without reloading the module.
    """
    if os.environ.get("DISCOPT_NLP_EVAL", "tape").strip().lower() == "jax":
        return False
    return pounce_usable()


_THREAD_SAFE_NLPROBLEM: Optional[bool] = None


def _nlproblem_is_thread_safe() -> bool:
    """Can one ``NlProblem`` be used (and dropped) from any thread?

    Fixed in pounce #477/#478, which removed the ``unsendable`` pyclass marker.
    Probed FUNCTIONALLY -- build a trivial problem and touch it from another
    thread -- rather than by version string, because ``pounce-solver>=0.9`` spans
    builds on both sides of that change and a version compare would silently pick
    the wrong answer on a dev install. Run once per process.

    Falls back to ``False`` on any failure. Being wrong in that direction costs a
    few milliseconds of extra tape building; being wrong the other way costs a
    false ``infeasible`` (CLAUDE.md §1).
    """
    global _THREAD_SAFE_NLPROBLEM
    if _THREAD_SAFE_NLPROBLEM is not None:
        return _THREAD_SAFE_NLPROBLEM

    outcome = False
    try:
        import pounce

        expr = pounce.NlExpr.var(0)
        prob = pounce.build_nl_problem(1, expr, constraints=None)
        prob.objective([1.0])  # bind it to THIS thread first
        box: list = []

        def touch() -> None:
            try:
                box.append(float(prob.objective([2.0])))
            except BaseException:  # noqa: BLE001 - PanicException is not an Exception
                box.append(None)

        t = threading.Thread(target=touch)
        t.start()
        t.join(timeout=30.0)
        outcome = bool(box) and box[0] == 2.0
    except BaseException:  # noqa: BLE001 - any probe failure means "assume not safe"
        outcome = False

    _THREAD_SAFE_NLPROBLEM = outcome
    return outcome


class TapeNLPEvaluator:
    """Tape-backed drop-in for :class:`discopt._jax.nlp_evaluator.NLPEvaluator`.

    Exposes the same surface the solver, ``nlp_ipopt``, ``oa`` and ``amp``
    consume. Conventions match the JAX evaluator exactly, because the callers
    depend on them:

    * ``evaluate_lagrangian_hessian`` returns a **full dense ``(n, n)``** matrix
      (``nlp_ipopt`` indexes it with ``h[rows, cols]``), while
      ``hessian_structure`` / ``evaluate_hessian_values`` are **lower-triangle**
      COO.
    * a MAXIMIZE objective is negated, so every caller minimizes.
    """

    def __init__(self, model: "Model") -> None:
        import pounce

        from discopt.modeling.core import Constraint, ObjectiveSense

        if model._objective is None:
            raise ValueError("Model has no objective set.")
        self._gauss_newton_requested = bool(getattr(model, "_gauss_newton_hessian", False))
        # `_build` re-runs whenever a `Parameter.value` moves, which in a fitting
        # loop is every iteration; warn once per evaluator, not once per rebuild.
        self._gn_decline_logged = False
        self._model = model
        self._pounce = pounce
        self._negate = model._objective.sense == ObjectiveSense.MAXIMIZE
        self._n_variables = sum(v.size for v in model._variables)

        # Same source list as the JAX evaluator, including the #840 fast-path
        # builder rows. Reading only ``model._constraints`` both misses those and
        # mis-indexes the ones it does read.
        self._source_constraints: list[Constraint] = [
            c
            for c in (*model._constraints, *model._builder_linear_constraints())
            if isinstance(c, Constraint)
        ]

        # Parameters are baked into the tape as constants (a tape is built for
        # fixed structure). The JAX evaluator instead plumbs them as runtime args
        # and `evaluator_fingerprint` DELIBERATELY excludes `Parameter.value` for
        # that reason. So a tape cached under that fingerprint would serve stale
        # derivatives after a re-bind. Snapshot the values and rebuild when they
        # move -- a tape build is milliseconds, unlike a JAX trace.
        self._parameters = list(model._parameters)
        self._param_snapshot = self._snapshot_params()

        self._build()

    # -- construction -------------------------------------------------------

    def _snapshot_params(self) -> tuple:
        return tuple(np.asarray(p.value, dtype=float).copy() for p in self._parameters)

    def _params_changed(self) -> bool:
        current = self._snapshot_params()
        if len(current) != len(self._param_snapshot):
            return True
        return any(
            a.shape != b.shape or not np.array_equal(a, b)
            for a, b in zip(current, self._param_snapshot)
        )

    def _build(self) -> None:
        """(Re)build the tape for this model.

        One shared ``NlProblem`` when pounce allows it, one per thread when it
        does not — see :func:`_nlproblem_is_thread_safe`. The distinction was a
        correctness issue, not a tidiness one: a shared unsendable pyclass
        touched from a solver worker thread raises a Rust ``PanicException``,
        which derives from ``BaseException`` and so slips past every
        ``except Exception`` in the solver. Measured on clay0303hfsg, that turned
        into a **false `infeasible`** where the JAX arm returned `feasible`
        (obj 26669.1). Fixed upstream in pounce #477/#478; the per-thread path
        remains for older pounce builds, because a wrong certificate is never an
        acceptable outcome (CLAUDE.md §1).
        """
        model = self._model
        objective = model._objective
        assert objective is not None  # refused in __init__
        obj = compile_to_nl_expr(objective.expression, model)
        if self._negate:
            obj = -obj

        cons = [compile_to_nl_expr(c.body, model) for c in self._source_constraints]

        # Every body that lowered is scalar by construction: the compiler refuses
        # array variables and every array reduction. So each source constraint is
        # exactly one row, and the row map is the identity -- unlike the JAX path,
        # which must call `jax.eval_shape` because an array body is one Constraint
        # and many rows.
        self._constraint_flat_sizes = np.ones(len(cons), dtype=np.intp)
        self._n_constraints = len(cons)

        lb, ub = self.variable_bounds
        self._lo = [float(v) for v in lb]
        self._hi = [float(v) for v in ub]
        self._obj_expr = obj
        self._con_exprs = cons
        self._residual_exprs = self._build_residual_exprs()
        self._shared_problem = self._new_problem() if _nlproblem_is_thread_safe() else None
        # Only used on the per-thread fallback. thread-LOCAL storage rather than a
        # dict keyed by thread id: an unsendable pyclass also refuses to be
        # DROPPED on a foreign thread, and a dict outliving its threads hands
        # every entry to whichever thread runs the GC. CPython clears a
        # threading.local on the owning thread as it exits. `_generation`
        # invalidates every thread's copy on a rebuild without enumerating them
        # (a threading.local cannot be iterated).
        self._local = threading.local()
        self._generation = getattr(self, "_generation", 0) + 1
        self._jac_struct: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._hess_struct: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._gn_base_map: Optional[np.ndarray] = None

    def _build_residual_exprs(self) -> Optional[list]:
        """Residual tapes ``r_i`` for Gauss-Newton, or ``None`` to use the exact Hessian.

        Mirrors ``NLPEvaluator._build_gauss_newton_obj_hessian``'s applicability
        rules exactly, so that flipping ``DISCOPT_NLP_EVAL`` cannot change whether
        a model gets the approximation: GN is off for a MAXIMIZE objective, and
        off when the objective is not a recognized non-negative-weighted sum of
        squares. Both decline to the exact Hessian, which is the conservative
        direction -- exact curvature is always a valid answer, an unwarranted
        ``2 JᵀJ`` is not. Both also warn rather than log at INFO, for the reason
        given at the JAX arm: the caller set an option that then did nothing.
        """
        if not self._gauss_newton_requested:
            return None
        if self._negate:
            self._warn_gn_declined(
                "gauss_newton ignored: objective is maximized (not a minimized "
                "sum of squares); using the exact tape Hessian."
            )
            return None

        from discopt._jax.least_squares import extract_residuals

        objective = self._model._objective
        assert objective is not None  # refused in __init__
        residuals = extract_residuals(objective.expression)
        if not residuals:
            self._warn_gn_declined(
                "gauss_newton requested but the objective is not a recognized "
                "non-negative-weighted sum of squares, so the exact tape Hessian "
                "is used instead. Recognized forms are `r**2`, `r*r`, `c*<square>` "
                "with c >= 0, `<square>/c` with c > 0, and sums of those."
            )
            return None
        # Every residual is a subexpression of an objective that already lowered,
        # so this cannot introduce a new UnsupportedForTape -- and if it somehow
        # does, it must propagate to `try_build` and degrade to JAX, not be caught.
        return [compile_to_nl_expr(r, self._model) for r in residuals]

    def _warn_gn_declined(self, message: str) -> None:
        if self._gn_decline_logged:
            return
        self._gn_decline_logged = True
        logger.warning(message)

    def _new_problem(self) -> Any:
        """``(main, residual_or_None)``. The pair is built and cached together so
        the per-thread fallback below cannot hand one thread's aux tape to another.
        """
        main = self._pounce.build_nl_problem(
            self._n_variables,
            self._obj_expr,
            constraints=self._con_exprs or None,
            x_l=self._lo,
            x_u=self._hi,
        )
        residual = None
        if self._residual_exprs is not None:
            # Objective is a literal 0: this tape exists only for its Jacobian
            # ∂r/∂x. pounce accepts a constant objective (verified in the entry
            # experiment -- it was kill criterion K1).
            residual = self._pounce.build_nl_problem(
                self._n_variables,
                self._pounce.NlExpr.const_(0.0),
                constraints=self._residual_exprs,
                x_l=self._lo,
                x_u=self._hi,
            )
        return (main, residual)

    @property
    def _problem_pair(self) -> Any:
        if self._shared_problem is not None:
            return self._shared_problem
        cached = getattr(self._local, "entry", None)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        pair = self._new_problem()
        self._local.entry = (self._generation, pair)
        return pair

    @property
    def _problem(self) -> Any:
        return self._problem_pair[0]

    @property
    def _residual_problem(self) -> Any:
        return self._problem_pair[1]

    def _ensure_fresh(self) -> None:
        """Rebuild if a ``Parameter.value`` moved since the tape was built."""
        if self._parameters and self._params_changed():
            self._param_snapshot = self._snapshot_params()
            self._build()

    # -- shape / structure --------------------------------------------------

    @property
    def n_variables(self) -> int:
        return self._n_variables

    @property
    def n_constraints(self) -> int:
        return self._n_constraints

    @property
    def is_gauss_newton(self) -> bool:
        """True when the objective Hessian uses the Gauss-Newton approximation."""
        return self._residual_exprs is not None

    @property
    def variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lbs, ubs = [], []
        for v in self._model._variables:
            lbs.append(np.asarray(v.lb).flatten())
            ubs.append(np.asarray(v.ub).flatten())
        if not lbs:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(lbs), np.concatenate(ubs)

    def constraint_row_map(self) -> list[tuple[int, int, "Constraint"]]:
        return [(i, i + 1, c) for i, c in enumerate(self._source_constraints)]

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_fresh()
        if self._jac_struct is None:
            r, c = self._problem.jacobian_structure()
            self._jac_struct = (
                np.asarray(r, dtype=np.int64),
                np.asarray(c, dtype=np.int64),
            )
        return self._jac_struct

    def hessian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower-triangle COO, matching the JAX evaluator's convention."""
        self._ensure_fresh()
        if self._hess_struct is None:
            r, c = self._problem.hessian_structure()
            rows = np.asarray(r, dtype=np.int64)
            cols = np.asarray(c, dtype=np.int64)
            # Two things downstream ASSUME this triangle and would fail silently
            # without it, so check it rather than trust it: `_widen_for_gauss_newton`
            # builds its union with `if a >= b`, and `evaluate_lagrangian_hessian`
            # mirrors with `lower + tril(lower, -1).T`. An upper-triangle entry
            # would make GN double-count every off-diagonal (the value would be
            # added at both (i,j) and (j,i)) and leave the dense Hessian
            # asymmetric -- a wrong curvature with no symptom, which is the
            # failure mode that reads as a pass.
            if rows.size and not bool(np.all(rows >= cols)):
                bad = np.flatnonzero(rows < cols)[:5]
                raise AssertionError(
                    "pounce returned a non-lower-triangular Hessian structure; the "
                    "Gauss-Newton union and the dense mirror both depend on it. "
                    f"First offending entries: {[(int(rows[k]), int(cols[k])) for k in bad]}"
                )
            if self._residual_exprs is not None:
                rows, cols = self._widen_for_gauss_newton(rows, cols)
            self._hess_struct = (rows, cols)
        return self._hess_struct

    def _widen_for_gauss_newton(
        self, rows: np.ndarray, cols: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Union the exact-Hessian pattern with ``2 JᵀJ``'s, and index one into the other.

        ``2 JᵀJ`` fills wherever two residuals share a variable, and the declared
        structure is fixed once. In the entry experiment the GN pattern was a
        subset of the exact pattern on all three shapes -- which is what the math
        predicts, since ∇²(Σrᵢ²) = 2JᵀJ + 2Σrᵢ∇²rᵢ and pounce's pattern is
        symbolic. But "predicted subset" is not "checked subset", and a value
        outside the declared structure is dropped *silently*, which is exactly the
        failure mode that reads as a pass. So take the union unconditionally: the
        cost is a few structural zeros, and the alternative risks a wrong Hessian
        with no symptom.
        """
        base = list(zip(rows.tolist(), cols.tolist()))
        pattern = set(base)

        jr, jc = self._residual_problem.jacobian_structure()
        cols_by_residual: dict[int, list[int]] = {}
        for i, j in zip(np.asarray(jr).tolist(), np.asarray(jc).tolist()):
            cols_by_residual.setdefault(i, []).append(j)
        for touched in cols_by_residual.values():
            for a in touched:
                for b in touched:
                    if a >= b:
                        pattern.add((a, b))

        union = sorted(pattern)
        slot = {rc: k for k, rc in enumerate(union)}
        # Duplicates in `base` (pounce may emit an entry twice, summed) map to the
        # same slot; `np.add.at` accumulates them, so this stays faithful.
        self._gn_base_map = np.array([slot[rc] for rc in base], dtype=np.int64)
        return (
            np.array([rc[0] for rc in union], dtype=np.int64),
            np.array([rc[1] for rc in union], dtype=np.int64),
        )

    def _gauss_newton_obj_hessian(self, x: np.ndarray) -> np.ndarray:
        """Dense ``(n, n)`` GN objective Hessian ``2 JᵀJ``, ``J = ∂r/∂x``."""
        prob = self._residual_problem
        rows, cols = prob.jacobian_structure()
        vals = np.asarray(prob.jacobian(self._x(x)), dtype=np.float64)
        assert self._residual_exprs is not None  # guarded by every caller
        jac = np.zeros((len(self._residual_exprs), self._n_variables), dtype=np.float64)
        np.add.at(jac, (np.asarray(rows), np.asarray(cols)), vals)
        gn: np.ndarray = 2.0 * (jac.T @ jac)
        return gn

    def has_sparse_structure(self) -> bool:
        """Always: the tape reports exact sparsity with no probing step."""
        return True

    def sparsity_pattern(self):
        return self.jacobian_structure()

    @property
    def hessian_kernel_compiled(self) -> bool:
        """A tape has no compile step, so the budget gate must never wait on one."""
        return True

    def hessian_compile_estimate_s(self) -> float:
        return 0.0

    # -- values -------------------------------------------------------------

    def _x(self, x: np.ndarray) -> list:
        return [float(v) for v in np.asarray(x, dtype=float).ravel()]

    def evaluate_objective(self, x: np.ndarray) -> float:
        self._ensure_fresh()
        return float(self._problem.objective(self._x(x)))

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        return np.asarray(self._problem.gradient(self._x(x)), dtype=np.float64)

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        if self._n_constraints == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self._problem.constraints(self._x(x)), dtype=np.float64)

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        if self._n_constraints == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self._problem.jacobian(self._x(x)), dtype=np.float64)

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense ``(m, n)`` Jacobian, scattered from the tape's COO values."""
        m, n = self._n_constraints, self._n_variables
        if m == 0:
            return np.zeros((0, n), dtype=np.float64)
        rows, cols = self.jacobian_structure()
        dense = np.zeros((m, n), dtype=np.float64)
        np.add.at(dense, (rows, cols), self.evaluate_jacobian_values(x))
        return dense

    def evaluate_hessian_values(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Lower-triangle Lagrangian Hessian values, aligned to ``hessian_structure``."""
        self._ensure_fresh()
        lam = np.asarray(lambda_, dtype=float).ravel()
        lam_list = [float(v) for v in lam]
        if self._residual_exprs is None:
            return np.asarray(
                self._problem.hessian(self._x(x), lam=lam_list, obj_factor=float(obj_factor)),
                dtype=np.float64,
            )

        # Gauss-Newton: obj_factor * 2JᵀJ + Σ λᵢ ∇²gᵢ. `obj_factor=0.0` isolates
        # the constraint term, which keeps its EXACT curvature -- only the
        # objective's second-derivative graph is replaced. Same split as the JAX
        # arm's `lagrangian_hess`.
        rows, cols = self.hessian_structure()  # also populates `_gn_base_map`
        base = np.asarray(
            self._problem.hessian(self._x(x), lam=lam_list, obj_factor=0.0),
            dtype=np.float64,
        )
        assert self._gn_base_map is not None  # set by hessian_structure above
        out = np.zeros(rows.shape[0], dtype=np.float64)
        np.add.at(out, self._gn_base_map, base)
        factor = float(obj_factor)
        if factor != 0.0:
            gn = self._gauss_newton_obj_hessian(x)
            out += factor * gn[rows, cols]
        return out

    def evaluate_lagrangian_hessian(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """FULL dense ``(n, n)`` Lagrangian Hessian.

        Full, not lower-triangular: ``nlp_ipopt.hessian`` indexes the result with
        ``h[rows, cols]`` against its own structure, and ``sipopt`` /
        ``benders/_feasibility`` treat it as a symmetric matrix. Returning a bare
        triangle here would silently zero half of every off-diagonal term.
        """
        n = self._n_variables
        rows, cols = self.hessian_structure()
        vals = self.evaluate_hessian_values(x, obj_factor, lambda_)
        lower = np.zeros((n, n), dtype=np.float64)
        np.add.at(lower, (rows, cols), vals)
        # Mirror the STRICTLY lower triangle and leave the diagonal alone, rather
        # than `lower + lower.T - diag(diag(lower))`. Identical for every finite
        # Hessian (verified elementwise), but the add-then-subtract form computes
        # `d + d - d` on the diagonal, and for an infinite `d` that is
        # `-inf + -inf - -inf = nan`: a correctly-signed infinity, which is the
        # best a double can do when the true curvature overflows, turned into a
        # nan that has lost even the sign. Reachable from ordinary operators --
        # `log''(1e-300) = -1e600` and `sqrt''(1e-300) = -2.5e449` both overflow.
        full: np.ndarray = lower + np.tril(lower, -1).T
        return full

    def evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        """Dense objective Hessian: the Lagrangian at ``obj_factor=1, lam=0``."""
        return self.evaluate_lagrangian_hessian(
            x, 1.0, np.zeros(self._n_constraints, dtype=np.float64)
        )

    # -- legacy single-argument wrappers ------------------------------------
    # `_obj_fn` / `_cons_fn` are read directly by the IPM batch path, OA, and
    # alpha estimation. They must exist and must read current parameter values.

    def _obj_fn(self, x: np.ndarray) -> float:
        return self.evaluate_objective(x)

    def _cons_fn(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_constraints(x), dtype=np.float64)


def try_build(model: "Model") -> Optional[TapeNLPEvaluator]:
    """Build a tape evaluator, or ``None`` when the model is unrepresentable.

    ``None`` means *representability* only — an array-valued body, an operator
    with no tape lowering, or ``dm.custom``. It never means a numerical failure,
    so a caller cannot confuse "no tape" with "bad point".
    """
    try:
        return TapeNLPEvaluator(model)
    except UnsupportedForTape:
        return None


def cached_tape_evaluator(model: "Model") -> Optional[TapeNLPEvaluator]:
    """Per-model LRU-cached tape evaluator, or ``None`` if unrepresentable.

    Shares :mod:`discopt._evaluator_cache` with the JAX backend so the two agree
    on what "still valid" means, on its own cache attribute so a backend switch
    cannot hand one backend's evaluator to the other.

    A ``None`` result is cached too, as a sentinel: representability is a pure
    function of the structural fingerprint, so re-attempting the build at every
    B&B node would re-walk the whole DAG only to fail again.
    """
    from discopt._evaluator_cache import cached_by_fingerprint

    built = cached_by_fingerprint(model, "_tape_evaluator_cache", _build_or_sentinel)
    return None if built is _UNREPRESENTABLE else built


_UNREPRESENTABLE = object()


def _build_or_sentinel(model: "Model") -> Any:
    ev = try_build(model)
    return _UNREPRESENTABLE if ev is None else ev


def make_evaluator(model: "Model") -> Any:
    """The cached evaluator for ``model`` under the selected backend.

    THE canonical entry point. The ``cached_evaluator`` import is inside the
    fallback on purpose: ``_jax/nlp_evaluator`` imports jax at module scope, so
    importing it eagerly puts JAX in ``sys.modules`` on every nonlinear solve and
    defeats the tape backend entirely (#75). Callers must not reach past this to
    ``cached_evaluator`` — that is what kept JAX on the path after Stage 3.
    """

    def _jax_evaluator() -> Any:
        from discopt._jax.nlp_evaluator import cached_evaluator

        return cached_evaluator(model)

    return build_evaluator(model, _jax_evaluator)


def build_evaluator(model: "Model", jax_factory: Any) -> Any:
    """The tape evaluator when opted in and representable, else ``jax_factory()``.

    Single decision point, so the fallback is one branch rather than one per call
    site. ``jax_factory`` is a zero-argument callable and is not invoked when the
    tape is used, which is what keeps JAX out of ``sys.modules`` on the tape path.
    """
    if tape_backend_requested():
        ev = cached_tape_evaluator(model)
        if ev is not None:
            return ev
    return jax_factory()
