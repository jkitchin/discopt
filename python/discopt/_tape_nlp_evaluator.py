"""A JAX-free NLP evaluator backed by POUNCE's Rust AD tape (issue #75, Stage 3).

``_relax/nlp_evaluator.py`` is the single trigger that imports JAX on *every*
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

**Array-valued constraint bodies** — one ``Constraint`` and many rows, which is
what ``DAEBuilder`` collocation emits per block — are lowered, not refused:
:func:`~discopt._nl_expr_compiler.compile_to_nl_array` returns one tape node per
element and ``_build`` fans them into rows in ``reshape(-1)`` (C) order, the same
order the JAX evaluator concatenates. ``constraint_row_map`` is therefore *not*
the identity, and callers must consume it rather than assume one row per
constraint (#908).

``dm.external`` blocks and the spectral norm of a matrix are numpy-evaluated
*holes* (:mod:`discopt._tape_holes`, #1903): extra tape columns the evaluator fills
from numpy and folds back into x-space by the chain rule (see the "holes" section
of :class:`TapeNLPEvaluator`).

What remains unrepresentable — an opaque ``dm.custom`` body, a nested hole, an
operator with no tape opcode — still raises ``UnsupportedForTape``. Callers use
:func:`try_build`, which returns ``None`` for those, so the JAX path stays intact
underneath.

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
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from discopt._arena_tape import arena_tape_enabled, try_build_arena_tape
from discopt._nl_expr_compiler import (
    UnsupportedForTape,
    compile_builder_row,
    compile_to_nl_array,
    compile_to_nl_expr,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt._tape_holes import Hole
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
    """Tape-backed drop-in for :class:`discopt._relax.nlp_evaluator.NLPEvaluator`.

    Exposes the same surface the solver, ``nlp_ipopt``, ``oa`` and ``amp``
    consume. Conventions match the JAX evaluator exactly, because the callers
    depend on them:

    * ``evaluate_lagrangian_hessian`` returns a **full dense ``(n, n)``** matrix
      (``nlp_ipopt`` indexes it with ``h[rows, cols]``), while
      ``hessian_structure`` / ``evaluate_hessian_values`` are **lower-triangle**
      COO.
    * a MAXIMIZE objective is negated, so every caller minimizes.
    """

    #: Layer bucket for :mod:`discopt._timing` (issue #74). A derivative callback
    #: on this evaluator spends its time inside POUNCE's Rust tape evaluation, so
    #: it belongs to ``rust`` -- not ``jax``, which is a subset of *Python* time
    #: and would report a library this evaluator never loads. It is deliberately
    #: ``rust`` rather than ``pounce``: ``pounce`` is the enclosing IPM's own
    #: bucket, and charging it here would net out against that frame's self time
    #: (``charge`` records self time), leaving the evaluator invisible in the
    #: profile. ``rust`` keeps it native *and* distinguishable from the solver.
    timing_bucket = "rust"

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
        # ``lazy=True`` (#1712): a builder row's body is built only if something
        # reads it; ``_build_tape_nodes`` lowers it from its coefficients instead.
        self._source_constraints: list[Constraint] = [
            c
            for c in (*model._constraints, *model._builder_linear_constraints(lazy=True))
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
        self._rebuild_param_watch()

        self._build()

    # -- construction -------------------------------------------------------

    def _snapshot_params(self) -> tuple:
        return tuple(np.asarray(p.value, dtype=float).copy() for p in self._parameters)

    def _rebuild_param_watch(self) -> None:
        """Split the snapshot into the two comparison paths ``_params_changed`` uses.

        Scalar (0-d) parameters — every parameter the relaxation layer accepts, and
        the overwhelming majority in practice — are watched as plain Python floats,
        which is what makes the staleness check cheap enough to sit in front of every
        evaluation (issue #1245). Array parameters keep the ndarray comparison.
        """
        scalar: list[tuple[Any, float]] = []
        array: list[tuple[Any, np.ndarray]] = []
        for p, snap in zip(self._parameters, self._param_snapshot):
            if snap.ndim == 0:
                scalar.append((p, float(snap)))
            else:
                array.append((p, snap))
        self._scalar_watch = scalar
        self._array_watch = array

    def _params_changed(self) -> bool:
        """Has any watched ``Parameter.value`` moved since the tape was built?

        Called once per objective/gradient/constraint/Jacobian/Hessian evaluation,
        so its cost is on the solve's hot path: measured at 11.5 us/call (5 scalar
        parameters) for the previous form, which copied every parameter array into a
        fresh snapshot tuple on every call — 4.5% of the wall of a parameterised
        CALPHAD-shaped solve (issue #1245). Comparing against the retained snapshot
        in place, with a float compare for the 0-d parameters, is 0.30 us/call.

        The verdict is unchanged, element for element: this is exact equality, the
        same predicate ``np.array_equal`` applied (so a NaN-valued parameter still
        reads as *changed* on every call, as it did before — it never compares equal
        to itself).
        """
        for p, ref in self._scalar_watch:
            v = p.value
            # ``Parameter.value`` normalises to a float64 ndarray and its setter
            # refuses a shape change, so a 0-d snapshot stays 0-d across a re-bind.
            # The ndim guard keeps the fast path exact if a value ever arrives by
            # some other route.
            if v.ndim != 0 or v[()] != ref:
                return True
        for p, ref_arr in self._array_watch:
            v = np.asarray(p.value, dtype=np.float64)
            if v.shape != ref_arr.shape or not np.array_equal(v, ref_arr):
                return True
        return False

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

        # Prefer lowering the Rust arena, which `model_to_repr` has already built
        # from the same structure, over a second walk of the Python DAG: measured
        # at foundation scale the tape build is 59% of the build -> solve-ready
        # pipeline and 93-94% of THAT is this walk (#1215). `_arena_tape` returns
        # `None` -- never a partial tape -- for anything it cannot reproduce
        # exactly, and its output is verified bit-identical to the walk below
        # (`issue1215_arena_tape_differential.py`), so this is a marshaling
        # change, not a numerical one.
        self._holes: list[Hole] = []
        obj, cons, sizes = self._build_tape_nodes(model)
        if self._negate:
            obj = -obj
        self._constraint_flat_sizes = np.asarray(sizes, dtype=np.intp)
        self._n_constraints = len(cons)

        lb, ub = self.variable_bounds
        self._lo = [float(v) for v in lb]
        self._hi = [float(v) for v in ub]
        self._obj_expr = obj
        self._con_exprs = cons
        # #1903: hole outputs are extra tape columns after the model's own; they
        # are never variables of the NLP, so their tape bounds are free.
        n_hole_vars = sum(h.n_out for h in self._holes)
        self._n_tape = self._n_variables + n_hole_vars
        self._tape_lo = self._lo + [-np.inf] * n_hole_vars
        self._tape_hi = self._hi + [np.inf] * n_hole_vars
        self._hole_args = [node for h in self._holes for node in h.arg_nodes.reshape(-1)]
        self._hole_dense = self._n_tape <= _HOLE_DENSE_MAX
        self._chain_local = threading.local()
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

    def _build_tape_nodes(self, model: "Model") -> tuple[Any, list, list[int]]:
        """``(objective, constraint rows, rows-per-constraint)``, arena path first.

        The arena lowering covers scalar-bodied expression-path models, which is
        the shape the NLP path actually meets; everything else -- array-valued
        bodies, builder-resident rows, ``dm.custom`` -- comes back as ``None``
        and falls through to the Python DAG walk unchanged.

        The row-count assertion is not defensive noise. ``_constraint_flat_sizes``
        maps solver rows back to ``Constraint`` objects for duals, the row map and
        feasibility reporting, so a lowering that returned a different number of
        rows than the evaluator enumerated would mis-attribute every one of them
        silently. The arena path already refuses on a count mismatch; this catches
        a future divergence between what it counts and what the evaluator does.
        """
        built = None
        if arena_tape_enabled():
            built = try_build_arena_tape(model, self._pounce.NlExpr)
        if built is not None:
            # Named apart from the `cons` the Python walk below builds: that one
            # is annotated `list[Any]`, and re-using one name for both makes the
            # second binding a redefinition.
            arena_obj, arena_cons = built
            if len(arena_cons) != len(self._source_constraints):
                raise AssertionError(
                    f"arena tape produced {len(arena_cons)} rows for "
                    f"{len(self._source_constraints)} constraints; row attribution "
                    f"(duals, row map, feasibility) would be wrong"
                )
            # Every arena row is scalar by construction -- the path refuses any
            # body that fans out -- so each constraint contributes exactly one.
            #
            # It deliberately does NOT use `try_build_expanded_tape`, which CAN
            # fan array bodies out (in Rust). Measured, that is 1.77x SLOWER
            # here: the Python walk below lowers array-at-a-time -- 242
            # `_lower_uncached` calls for 20 000 rows -- and lets numpy's
            # `frompyfunc` build the per-element POUNCE nodes in C, while an
            # expanded program has to be consumed one scalar instruction at a
            # time from Python. See performance-plan.md §45.
            return arena_obj, arena_cons, [1] * len(arena_cons)

        if model._objective is None:
            raise ValueError("Model has no objective set.")
        from discopt._tape_holes import hole_sink

        # #1903: numpy-evaluated holes (`dm.external`, the spectral norm) are
        # accepted on this walk; `_build` folds them back into x-space.
        with hole_sink(self._n_variables) as sink:
            obj, cons, sizes = self._walk_tape_nodes(model)
        self._holes = list(sink.holes)
        return obj, cons, sizes

    def _walk_tape_nodes(self, model: "Model") -> tuple[Any, list, list[int]]:
        """The Python DAG walk of :meth:`_build_tape_nodes` (arena path declined)."""
        obj = compile_to_nl_expr(model._objective_expression(), model)
        # A body may be ARRAY-valued -- `x <= 1` on a 3-vector is ONE Constraint
        # and THREE rows -- so fan each one out. `reshape(-1)` is C order, which is
        # exactly what the JAX evaluator concatenates
        # (`jnp.reshape(fn(x, params), (-1,))`), so row k here is row k there and
        # the two backends' duals, row maps and feasibility reports line up. An
        # earlier revision could assert one row per constraint because the
        # compiler refused every array form; it no longer does.
        cons: list[Any] = []
        sizes: list[int] = []
        from discopt.modeling.core import builder_row_of

        for c in self._source_constraints:
            pending = builder_row_of(c)
            direct = compile_builder_row(pending, model) if pending is not None else None
            if direct is None:
                direct = compile_to_nl_array(c.body, model)
            rows = direct.reshape(-1)
            sizes.append(int(rows.size))
            cons.extend(rows.tolist())
        return obj, cons, sizes

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
        if self._holes:
            self._warn_gn_declined(
                "gauss_newton ignored: the model contains a numpy-evaluated node "
                "(dm.external or a spectral matrix norm); using the exact Hessian."
            )
            return None
        if self._negate:
            self._warn_gn_declined(
                "gauss_newton ignored: objective is maximized (not a minimized "
                "sum of squares); using the exact tape Hessian."
            )
            return None

        from discopt._relax.least_squares import extract_residuals

        # refused in __init__ when there is no objective
        residuals = extract_residuals(self._model._objective_expression())
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
        #
        # An array-valued residual contributes one row per element, in the order the
        # JAX arm's `residual_vector` concatenates them (`jnp.reshape(r, (-1,))` per
        # residual): `compile_to_nl_array(r).reshape(-1)` is that order by contract.
        out: list = []
        for r in residuals:
            out.extend(compile_to_nl_array(r, self._model).reshape(-1).tolist())
        return out

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
            self._n_tape,
            self._obj_expr,
            constraints=self._con_exprs or None,
            x_l=self._tape_lo,
            x_u=self._tape_hi,
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
        args = None
        if self._holes:
            # #1903: every hole's argument, as the constraint rows of an x-space
            # tape -- its Jacobian is dA/dx, and its weighted Hessian is the
            # argument-curvature term of the chain rule.
            args = self._pounce.build_nl_problem(
                self._n_variables,
                self._pounce.NlExpr.const_(0.0),
                constraints=self._hole_args,
                x_l=self._lo,
                x_u=self._hi,
            )
        return (main, residual, args)

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

    @property
    def _args_problem(self) -> Any:
        return self._problem_pair[2]

    def _ensure_fresh(self) -> None:
        """Rebuild if a ``Parameter.value`` moved since the tape was built."""
        if self._parameters and self._params_changed():
            self._param_snapshot = self._snapshot_params()
            self._rebuild_param_watch()
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
        """``(start, stop, constraint)`` per source constraint; see the JAX arm's
        docstring for why callers must consume this rather than re-derive it.

        Built from the same ``_source_constraints`` / ``_constraint_flat_sizes``
        the tape row list is built from, so the two cannot drift. NOT the identity
        map: an array-valued body is one ``Constraint`` and many rows.
        """
        out: list[tuple[int, int, Constraint]] = []
        start = 0
        for con, size in zip(self._source_constraints, self._constraint_flat_sizes):
            stop = start + int(size)
            out.append((start, stop, con))
            start = stop
        return out

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_fresh()
        if self._jac_struct is None and self._holes:
            self._jac_struct = self._hole_jacobian_structure()
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
        if self._hess_struct is None and self._holes:
            self._hess_struct = self._hole_hessian_structure()
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

    def _x(self, x: np.ndarray) -> np.ndarray:
        """The iterate, in the form pounce's tape entry points want.

        A contiguous ``float64`` array, NOT a Python list. pounce accepts either
        and returns bit-identical values for both (verified elementwise on
        ``objective`` / ``gradient`` / ``constraints`` / ``jacobian`` /
        ``hessian``), but the list form was a per-callback ``O(n)`` Python loop
        on the hottest path in the solver: 2.11 µs to build-and-evaluate against
        0.20 µs for the array on ``nvs05`` (n=15), and 12.29 µs against 1.52 µs
        on ``4stufen`` (n=157) -- i.e. the marshaling cost *scaled with n* while
        the arithmetic underneath it did not (#1180). The array path is also
        faster inside pounce (0.199 µs vs 0.308 µs for the same evaluation), so
        the list was costing on both sides of the boundary.

        Zero-copy when the caller already has a contiguous ``float64`` vector,
        which the IPM always does; ``ravel`` copies only when it must.
        """
        return np.asarray(x, dtype=np.float64).ravel()

    def evaluate_objective(self, x: np.ndarray) -> float:
        self._ensure_fresh()
        if self._holes:
            return float(self._problem.objective(self._hole_chain(x).z))
        return float(self._problem.objective(self._x(x)))

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        if self._holes:
            ch = self._hole_chain(x)
            gz = np.asarray(self._problem.gradient(ch.z), dtype=np.float64)
            n = self._n_variables
            return np.asarray(gz[:n] + ch.dt.T @ gz[n:], dtype=np.float64)
        return np.asarray(self._problem.gradient(self._x(x)), dtype=np.float64)

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        if self._n_constraints == 0:
            return np.zeros(0, dtype=np.float64)
        if self._holes:
            return np.asarray(self._problem.constraints(self._hole_chain(x).z), dtype=np.float64)
        return np.asarray(self._problem.constraints(self._x(x)), dtype=np.float64)

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        if self._n_constraints == 0:
            return np.zeros(0, dtype=np.float64)
        if self._holes:
            rows, cols = self.jacobian_structure()
            jx = self._hole_x_jacobian(self._hole_chain(x))
            return np.asarray(jx[rows, cols], dtype=np.float64).reshape(-1)
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

    def evaluate_sparse_jacobian(self, x: np.ndarray):
        """The same Jacobian as CSR, without the scatter into a dense array.

        The tape is *natively* sparse — ``jacobian_structure()`` is COO and
        ``evaluate_jacobian_values()`` are its values — so :meth:`evaluate_jacobian`
        reaches its dense ``(m, n)`` contract by throwing the structure away. For a
        caller that only wants to multiply or factorise, that scatter is the entire
        cost: on MINLPLib ``arki0014`` it is a (17525, 19305) array, 2.707 GB, to
        carry 75,329 nonzeros — 0.022 % dense, about 0.7 MB as CSR.

        Named to match :meth:`NLPEvaluator.evaluate_sparse_jacobian` so a caller can
        ask for the sparse form by ``hasattr`` without knowing which evaluator it
        holds; ``_dual_recovery.jacobian_for_recovery`` is the first such caller.

        The result is *exactly* the dense matrix, not an approximation of it:
        ``coo_matrix`` sums duplicate ``(row, col)`` entries on conversion, which is
        the same accumulation ``np.add.at`` performs above.
        """
        import scipy.sparse as sp

        m, n = self._n_constraints, self._n_variables
        if m == 0:
            return sp.csr_matrix((0, n), dtype=np.float64)
        rows, cols = self.jacobian_structure()
        vals = self.evaluate_jacobian_values(x)
        return sp.coo_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.float64).tocsr()

    def evaluate_hessian_values(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Lower-triangle Lagrangian Hessian values, aligned to ``hessian_structure``."""
        self._ensure_fresh()
        # Same reasoning as ``_x``: the multiplier vector goes across as an
        # array, not a rebuilt Python list (#1180).
        lam_list = np.asarray(lambda_, dtype=np.float64).ravel()
        if self._holes:
            return np.asarray(
                self._hole_hessian_values(x, float(obj_factor), lam_list), dtype=np.float64
            )
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

    # -- numpy-evaluated holes (#1903) ---------------------------------------
    #
    # The tape is built over ``z = (x, t)``: ``t`` are the outputs of the holes,
    # ``t_k = f_k(a_k(x))``, with ``f_k`` numpy (`dm.external`'s callables, or
    # sigma_1) and ``a_k`` the hole's taped argument. The NLP sees only ``x``, so
    # every derivative is folded back through ``dt/dx = G J_a`` (``G`` the holes'
    # Jacobians in their arguments, ``J_a = da/dx``):
    #
    #   grad F   = F_x + Dt^T F_t                                  (Dt = dt/dx)
    #   J        = J_zx + J_zt Dt
    #   H_L      = P^T H_z P + J_a^T blkdiag(sum_q w_q H_q) J_a + sum_j (G^T w)_j d2a_j
    #
    # with ``P = [I; Dt]``, ``H_z`` the tape's Lagrangian Hessian in ``z`` and
    # ``w = dL/dt = obj_factor * F_t + J_zt^T lambda``. The last term is the
    # argument problem's own weighted Hessian, so it comes from the tape too.

    def _hole_chain(self, x: np.ndarray) -> "_HoleChain":
        """Per-point hole data, cached for the last point on this thread."""
        xv = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
        key = xv.tobytes()
        cached = getattr(self._chain_local, "entry", None)
        if cached is not None and cached[0] == self._generation and cached[1] == key:
            return cast("_HoleChain", cached[2])
        ch = _HoleChain(self, xv)
        self._chain_local.entry = (self._generation, key, ch)
        return ch

    def _hole_x_jacobian(self, ch: "_HoleChain"):
        """The constraint Jacobian in x-space (dense when small, else CSR)."""
        n, nt, m = self._n_variables, self._n_tape, self._n_constraints
        r, c = self._z_jacobian_structure()
        vals = np.asarray(self._problem.jacobian(ch.z), dtype=np.float64)
        if self._hole_dense:
            jz = np.zeros((m, nt))
            np.add.at(jz, (r, c), vals)
            return jz[:, :n] + jz[:, n:] @ ch.dt
        import scipy.sparse as sp

        jzs = sp.coo_matrix((vals, (r, c)), shape=(m, nt)).tocsc()
        return (jzs[:, :n] + jzs[:, n:] @ ch.dt).tocsr()

    def _z_jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        cached = getattr(self, "_jz_struct", None)
        if cached is None or cached[0] != self._generation:
            r, c = self._problem.jacobian_structure()
            cached = (
                self._generation,
                np.asarray(r, dtype=np.int64),
                np.asarray(c, dtype=np.int64),
            )
            self._jz_struct = cached
        return cached[1], cached[2]

    def _hole_dt_pattern(self):
        """Boolean pattern of ``dt/dx``: each output touches its argument's support."""
        import scipy.sparse as sp

        n = self._n_variables
        ar, ac = self._args_problem.jacobian_structure()
        n_args = len(self._hole_args)
        ja = sp.coo_matrix(
            (np.ones(len(ar)), (np.asarray(ar), np.asarray(ac))), shape=(n_args, n)
        ).tocsr()
        blocks = []
        offset = 0
        for h in self._holes:
            sub = ja[offset : offset + h.n_in]
            support = (abs(sub).sum(axis=0) > 0).astype(np.float64)
            blocks.append(sp.csr_matrix(np.ones((h.n_out, 1)) @ np.asarray(support)))
            offset += h.n_in
        return sp.vstack(blocks).tocsr(), ja

    def _hole_jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        import scipy.sparse as sp

        n, nt, m = self._n_variables, self._n_tape, self._n_constraints
        if m == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        r, c = self._z_jacobian_structure()
        jz = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(m, nt)).tocsc()
        dt, _ = self._hole_dt_pattern()
        pat = (abs(jz[:, :n]) + abs(jz[:, n:]) @ dt).tocoo()
        order = np.lexsort((pat.col, pat.row))
        return pat.row[order].astype(np.int64), pat.col[order].astype(np.int64)

    def _hole_hessian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower triangle of every term of ``H_L`` above, symbolically."""
        import scipy.sparse as sp

        n, nt = self._n_variables, self._n_tape
        hr, hc = self._problem.hessian_structure()
        hz = sp.coo_matrix(
            (np.ones(len(hr)), (np.asarray(hr), np.asarray(hc))), shape=(nt, nt)
        ).tocsr()
        hz = abs(hz) + abs(hz.T)
        dt, ja = self._hole_dt_pattern()
        p = sp.vstack([sp.identity(n, format="csr"), dt]).tocsr()
        total = abs(p.T @ hz @ p)
        offset = 0
        for h in self._holes:
            sub = ja[offset : offset + h.n_in]
            total = total + abs(sub.T @ sp.csr_matrix(np.ones((h.n_in, h.n_in))) @ sub)
            offset += h.n_in
        ar, ac = self._args_problem.hessian_structure()
        if len(ar):
            arg_h = sp.coo_matrix(
                (np.ones(len(ar)), (np.asarray(ar), np.asarray(ac))), shape=(n, n)
            )
            total = total + abs(arg_h) + abs(arg_h.T)
        low = sp.tril(total).tocoo()
        order = np.lexsort((low.col, low.row))
        return low.row[order].astype(np.int64), low.col[order].astype(np.int64)

    def _hole_hessian_values(self, x: np.ndarray, obj_factor: float, lam: np.ndarray):
        if self._hole_dense:
            return self._hole_hessian_values_dense(x, obj_factor, lam)
        import scipy.sparse as sp

        n, nt = self._n_variables, self._n_tape
        ch = self._hole_chain(x)
        hr, hc = self._problem.hessian_structure()
        hv = np.asarray(
            self._problem.hessian(ch.z, lam=lam, obj_factor=obj_factor), dtype=np.float64
        )
        low = sp.coo_matrix((hv, (np.asarray(hr), np.asarray(hc))), shape=(nt, nt)).tocsr()
        hz = low + sp.tril(low, -1).T
        p = sp.vstack([sp.identity(n, format="csr"), ch.dt]).tocsr()
        total = p.T @ hz @ p
        # w = dL/dt
        gz = np.asarray(self._problem.gradient(ch.z), dtype=np.float64)
        w = obj_factor * gz[n:]
        if self._n_constraints:
            r, c = self._z_jacobian_structure()
            jv = np.asarray(self._problem.jacobian(ch.z), dtype=np.float64)
            jz = sp.coo_matrix((jv, (r, c)), shape=(self._n_constraints, nt)).tocsc()
            w = w + jz[:, n:].T @ lam
        lam_args = np.zeros(len(self._hole_args))
        offset_t = offset_a = 0
        for h, g, hh in zip(self._holes, ch.g, ch.h):
            wk = w[offset_t : offset_t + h.n_out]
            # J_a^T (sum_q w_q H_q) J_a, over this hole's argument rows
            block = np.tensordot(wk, hh, axes=(0, 0))
            sub = ch.ja[offset_a : offset_a + h.n_in]
            total = total + sub.T @ sp.csr_matrix(block) @ sub
            lam_args[offset_a : offset_a + h.n_in] = wk @ g
            offset_t += h.n_out
            offset_a += h.n_in
        if np.any(lam_args != 0.0):
            ar, ac = self._args_problem.hessian_structure()
            av = np.asarray(
                self._args_problem.hessian(ch.xv, lam=lam_args, obj_factor=0.0),
                dtype=np.float64,
            )
            alow = sp.coo_matrix((av, (np.asarray(ar), np.asarray(ac))), shape=(n, n)).tocsr()
            total = total + alow + sp.tril(alow, -1).T
        rows, cols = self.hessian_structure()
        return np.asarray(total.tocsr()[rows, cols], dtype=np.float64).reshape(-1)

    def _hole_hessian_values_dense(self, x: np.ndarray, obj_factor: float, lam: np.ndarray):
        """:meth:`_hole_hessian_values` in dense numpy, for small tapes.

        The same algebra term for term; it exists because scipy's per-call
        constructor overhead dominated small models (measured 1.9x the JAX
        evaluator's wall on a 2x2 ``min sigma_1`` solve, all of it in
        ``scipy.sparse.__init__``).
        """
        n, nt = self._n_variables, self._n_tape
        ch = self._hole_chain(x)
        hr, hc = self._problem.hessian_structure()
        hv = np.asarray(
            self._problem.hessian(ch.z, lam=lam, obj_factor=obj_factor), dtype=np.float64
        )
        low = np.zeros((nt, nt))
        np.add.at(low, (np.asarray(hr), np.asarray(hc)), hv)
        hz = low + np.tril(low, -1).T
        p = np.vstack([np.eye(n), ch.dt])
        total = p.T @ hz @ p
        gz = np.asarray(self._problem.gradient(ch.z), dtype=np.float64)
        w = obj_factor * gz[n:]
        if self._n_constraints:
            r, c = self._z_jacobian_structure()
            jv = np.asarray(self._problem.jacobian(ch.z), dtype=np.float64)
            jz = np.zeros((self._n_constraints, nt))
            np.add.at(jz, (r, c), jv)
            w = w + jz[:, n:].T @ lam
        lam_args = np.zeros(len(self._hole_args))
        offset_t = offset_a = 0
        for h, g, hh in zip(self._holes, ch.g, ch.h):
            wk = w[offset_t : offset_t + h.n_out]
            block = np.tensordot(wk, hh, axes=(0, 0))
            sub = ch.ja[offset_a : offset_a + h.n_in]
            total = total + sub.T @ block @ sub
            lam_args[offset_a : offset_a + h.n_in] = wk @ g
            offset_t += h.n_out
            offset_a += h.n_in
        if np.any(lam_args != 0.0):
            ar, ac = self._args_problem.hessian_structure()
            av = np.asarray(
                self._args_problem.hessian(ch.xv, lam=lam_args, obj_factor=0.0),
                dtype=np.float64,
            )
            alow = np.zeros((n, n))
            np.add.at(alow, (np.asarray(ar), np.asarray(ac)), av)
            total = total + alow + np.tril(alow, -1).T
        rows, cols = self.hessian_structure()
        return np.asarray(total[rows, cols], dtype=np.float64).reshape(-1)

    # -- legacy single-argument wrappers ------------------------------------
    # `_obj_fn` / `_cons_fn` are read directly by the IPM batch path, OA, and
    # alpha estimation. They must exist and must read current parameter values.

    def _obj_fn(self, x: np.ndarray) -> float:
        return self.evaluate_objective(x)

    def _cons_fn(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_constraints(x), dtype=np.float64)


#: Tape width (model variables plus hole outputs) up to which the hole chain rule
#: runs in dense numpy rather than scipy.sparse. Dense ``(n, n)`` work at this size
#: is microseconds, while scipy's constructors cost ~20-50 us each per call.
_HOLE_DENSE_MAX = 300


class _HoleChain:
    """One point's hole data: ``z = (x, t)`` at once, the derivatives on first use.

    Values and derivatives are split because the IPM asks for values far more
    often: measured on a degenerate ``min sigma_1`` solve, 18,894 of 41,703
    evaluator callbacks were ``constraints`` (line-search trial points). Building
    every derivative for each of them cost 1.1 ms per call in scipy constructors
    alone; a value-only point is the argument tape plus one SVD per hole.
    """

    def __init__(self, ev: TapeNLPEvaluator, xv: np.ndarray) -> None:
        self._ev = ev
        self.xv = xv
        a_vals = np.asarray(ev._args_problem.constraints(xv), dtype=np.float64)
        self._a = a_vals
        parts = [xv]
        offset = 0
        for h in ev._holes:
            a = a_vals[offset : offset + h.n_in].reshape(h.in_shape)
            parts.append(np.asarray(h.f(a), dtype=np.float64).reshape(h.n_out))
            offset += h.n_in
        self.z = np.concatenate(parts)
        self._derivs: Optional[tuple] = None

    def _derivatives(self) -> tuple:
        if self._derivs is None:
            import scipy.sparse as sp

            ev = self._ev
            n = ev._n_variables
            args = ev._args_problem
            ar, ac = args.jacobian_structure()
            av = np.asarray(args.jacobian(self.xv), dtype=np.float64)
            if ev._hole_dense:
                jad = np.zeros((len(ev._hole_args), n))
                np.add.at(jad, (np.asarray(ar), np.asarray(ac)), av)
                gs_d: list[np.ndarray] = []
                hs_d: list[np.ndarray] = []
                dt_d = []
                off = 0
                for h in ev._holes:
                    a = self._a[off : off + h.n_in].reshape(h.in_shape)
                    g = np.asarray(h.jac(a), dtype=np.float64).reshape(h.n_out, h.n_in)
                    hh = np.asarray(h.hess(a), dtype=np.float64).reshape(h.n_out, h.n_in, h.n_in)
                    gs_d.append(g)
                    hs_d.append(hh)
                    dt_d.append(g @ jad[off : off + h.n_in])
                    off += h.n_in
                self._derivs = (np.vstack(dt_d), jad, gs_d, hs_d)
                return self._derivs
            ja = sp.coo_matrix(
                (av, (np.asarray(ar), np.asarray(ac))), shape=(len(ev._hole_args), n)
            ).tocsr()
            gs: list[np.ndarray] = []
            hs: list[np.ndarray] = []
            dt_rows = []
            offset = 0
            for h in ev._holes:
                a = self._a[offset : offset + h.n_in].reshape(h.in_shape)
                g = np.asarray(h.jac(a), dtype=np.float64).reshape(h.n_out, h.n_in)
                hh = np.asarray(h.hess(a), dtype=np.float64).reshape(h.n_out, h.n_in, h.n_in)
                gs.append(g)
                hs.append(hh)
                dt_rows.append(sp.csr_matrix(g) @ ja[offset : offset + h.n_in])
                offset += h.n_in
            self._derivs = (sp.vstack(dt_rows).tocsr(), ja, gs, hs)
        return self._derivs

    @property
    def dt(self):
        return self._derivatives()[0]

    @property
    def ja(self):
        return self._derivatives()[1]

    @property
    def g(self) -> list[np.ndarray]:
        return cast("list[np.ndarray]", self._derivatives()[2])

    @property
    def h(self) -> list[np.ndarray]:
        return cast("list[np.ndarray]", self._derivatives()[3])


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
    fallback on purpose: ``_relax/nlp_evaluator`` imports jax at module scope, so
    importing it eagerly puts JAX in ``sys.modules`` on every nonlinear solve and
    defeats the tape backend entirely (#75). Callers must not reach past this to
    ``cached_evaluator`` — that is what kept JAX on the path after Stage 3.
    """

    return build_evaluator(model, lambda: jax_fallback_evaluator(model))


def jax_fallback_evaluator(model: "Model") -> Any:
    """The cached legacy JAX evaluator, the fallback of :func:`build_evaluator`.

    JAX is the optional ``discopt[jax]`` extra (#1873). Without it there is no
    evaluator for a model the tape declines, so this refuses loudly and says why
    the tape declined -- the operator to rewrite, or the extra to install --
    rather than letting ``No module named 'jax'`` surface from deep in a solve.
    """
    from discopt._optional_jax import jax_available, missing_jax_error

    if not jax_available():
        raise missing_jax_error(
            f"Evaluating this model needs the legacy JAX NLP evaluator because "
            f"{_tape_decline_reason(model)}; it"
        )
    from discopt._relax.nlp_evaluator import cached_evaluator

    return cached_evaluator(model)


def _tape_decline_reason(model: "Model") -> str:
    """Why the tape is not evaluating ``model`` -- for an error message only."""
    if os.environ.get("DISCOPT_NLP_EVAL", "tape").strip().lower() == "jax":
        return "DISCOPT_NLP_EVAL=jax selects it"
    if not pounce_usable():
        return "the POUNCE tape backend is unavailable (install `pounce-solver`)"
    try:
        TapeNLPEvaluator(model)
    except UnsupportedForTape as exc:
        return f"the POUNCE tape has no opcode for part of it ({exc})"
    return "the POUNCE tape declined it"


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
