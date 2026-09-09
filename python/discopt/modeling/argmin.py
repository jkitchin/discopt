"""An inner NLP as a block of an outer model (issue #1216).

``argmin(inner, bind={p: outer_expr})`` embeds a whole :class:`~discopt.modeling.core.Model`
as a node of another model.  The inner problem is passed **unchanged** -- objective
and constraints as written, no reformulation:

* **Forward pass** -- a POUNCE solve of ``inner`` at the current value of the bound
  parameters.  The value of the node is the *minimizer* POUNCE returns.
* **Derivatives** -- the sIPOPT sensitivity system of
  :func:`discopt.solvers.sipopt.pounce_sensitivity`, but wired into JAX through
  :func:`jax.lax.custom_root` over the inner KKT residual instead of returned as
  arrays.  Because ``custom_root``'s rule is expressed in ordinary differentiable
  operations, **first and second order both work** (the outer solver's Lagrangian
  Hessian is a forward-over-reverse pass through this node), and the right-hand
  side is **exact** -- the parameter dependence is differentiated symbolically off
  the compiled DAG, not finite-differenced.

Why not write the follower's KKT conditions by hand
---------------------------------------------------

For a nonconvex follower, hand-writing ``∇_y L = 0, g(y) = 0`` into
:func:`~discopt.modeling.implicit.implicit_full_space` is a **relaxation**, not a
reformulation: the KKT set contains every stationary point, so the leader is free
to pick a *maximizer* of the follower when that suits its own objective.  On the
projection follower of #1216 -- ``min ||y - (p,1)||²  s.t. ||y||² = 1`` -- the
hand-written KKT block returns ``y* = (-0.9, -0.436)``, the far-side root, and the
solver certifies it, because that point really is optimal *for the model as
stated*.  It is not the projection of anything.  An ``argmin`` block cannot make
that substitution: only points POUNCE returns as minimizers are representable.

Two arms, and the trade between them
------------------------------------

The same trade :func:`~discopt.modeling.implicit.implicit` and
:func:`~discopt.modeling.implicit.implicit_full_space` make, and for the same
reason -- so it is a choice you make rather than a lowering that changes under you:

``argmin(inner, bind=...)`` -- **solve** the follower.  Built on
:func:`~discopt.modeling.core.custom`, so it inherits the ``CustomCall`` contract:
the outer model is solved on the **local NLP path only** (``status="feasible"``,
``gap_certified=False``, no ``bound``/``gap``, no ``.nl`` export, outer
integer/binary variables refused).  Takes **any** follower discopt can solve, and
its answer is a local one -- for a nonconvex follower a different POUNCE start may
land on a different branch.

``argmin_kkt(model, inner, bind=...)`` -- **lower** the follower.  Its variables
become real variables of the outer model and its KKT conditions become real
constraints, so the Rust tape, FBBT, a **global certificate** and ``.nl`` export
all come back.  The price is that KKT conditions characterize a follower's optimum
only when the follower is **convex in its own variables**, so the lowering runs
:class:`discopt.bilevel.BilevelProblem`'s convexity certifier and refuses anything
it cannot prove -- including the projection follower above, whose nonlinear
equality is exactly what makes the hand-written version unsound.  There is no flag
to override that.

For ``.nl`` export specifically, use ``method="strong_duality"``: it emits pure
algebra (stationarity, primal/dual feasibility, and the single bilinear equality
``Σ μ_i g_i == 0``).  The ``"kkt"`` arm's complementarity conditions go through the
GDP/SOS1 encodings, which discopt solves and certifies but which have no ``.nl``
form.

Soundness gates (all refuse rather than approximate)
----------------------------------------------------

* Integer/binary variables in ``inner`` -- **raises**.  KKT stationarity does not
  characterize an integer optimum.
* A maximize objective in ``inner`` -- **raises**.  The node would be an argmax.
* A failed inner solve, an inner KKT residual that does not check out at the
  returned multipliers, or (with ``verify_minimizer=True``) a reduced Hessian that
  is not positive semidefinite -- the node evaluates to **NaN**, which the outer
  NLP solver reports as a failed evaluation.  A traced solve cannot raise, so NaN
  is the refusal mechanism (the same choice :func:`implicit` documents).
* A degenerate active set (an active inequality or bound whose multiplier is ~0)
  -- ``dx*/dp`` is only one-sided there.  Warned once per block, not silenced.

The derivative is exact under the standard sIPOPT assumptions: LICQ at the inner
solution, strict complementarity, and an active set that is locally constant in the
parameters.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Callable, Optional, Sequence

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    ObjectiveSense,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
    custom,
)
from discopt.modeling.implicit import _unique_name

__all__ = ["argmin", "argmin_kkt", "argmin_layer"]

#: A constraint body sits on its bound (or a variable on its bound) within this
#: tolerance -> the row is in the active set.
ACTIVE_TOL = 1e-6

#: The KKT residual at the returned (x*, λ*) must be below this or the block
#: refuses (NaN): a residual this large means the multipliers or the active set do
#: not describe the point POUNCE returned, and every derivative built on them
#: would be wrong.
KKT_RESIDUAL_TOL = 1e-4

#: An active row whose multiplier is below this is degenerate (strict
#: complementarity fails) -> warn: the sensitivity is one-sided there.
COMPLEMENTARITY_TOL = 1e-8

#: Smallest eigenvalue of the reduced Hessian allowed by ``verify_minimizer``.
CURVATURE_TOL = -1e-6

#: The solver evaluates objective, gradient and Hessian at the same iterate, so
#: the inner solve is memoized on the parameter vector.  Small: the outer NLP
#: walks forward, it does not revisit old iterates.
_CACHE_SIZE = 16


def _flat_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Flat ``(lb, ub)`` over ``model``'s variables, in flat-vector order."""
    lo = [
        np.broadcast_to(np.asarray(v.lb, dtype=float), v.shape or ()).reshape(-1)
        for v in model._variables
    ]
    hi = [
        np.broadcast_to(np.asarray(v.ub, dtype=float), v.shape or ()).reshape(-1)
        for v in model._variables
    ]
    if not lo:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(lo), np.concatenate(hi)


def _row_bounds(constraints: Sequence[Constraint]) -> tuple[np.ndarray, np.ndarray]:
    """``(cl, cu)`` for constraint bodies compiled as ``body - rhs``."""
    table = {"<=": (-np.inf, 0.0), "==": (0.0, 0.0), ">=": (0.0, np.inf)}
    lo, hi = [], []
    for c in constraints:
        try:
            a, b = table[c.sense]
        except KeyError as e:  # pragma: no cover - Constraint validates its sense
            raise ValueError(f"Unknown constraint sense: {c.sense!r}") from e
        lo.append(a)
        hi.append(b)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _validate_inner(
    inner: Model,
    parameters: Sequence[Parameter],
    require_min: bool = True,
    where: str = "argmin()",
) -> None:
    """Refuse the inner models an argmin block cannot soundly represent.

    ``require_min=False`` is for the sensitivity API (:mod:`discopt.solvers.sipopt`),
    which differentiates a stationary point of whichever sense the model states;
    the block itself must be a minimization or its name would be a lie.
    """
    if inner._objective is None:
        raise ValueError(f"{where}: the inner model has no objective set.")
    if require_min and inner._objective.sense == ObjectiveSense.MAXIMIZE:
        raise ValueError(
            f"{where}: the inner model maximizes, so this block would be an "
            "argmax and the KKT sign convention would be inverted. State the "
            "follower as a minimization -- inner.minimize(-expr) -- and negate "
            "in the outer model if you need the value."
        )
    bad = [v.name for v in inner._variables if v.var_type != VarType.CONTINUOUS]
    if bad:
        raise ValueError(
            f"{where}: the inner model has integer/binary variables "
            f"({', '.join(bad)}). The block's derivatives come from KKT "
            "stationarity, which does not characterize an integer optimum. Model "
            "an integer follower with discopt.bilevel or an explicit "
            "reformulation instead."
        )
    for p in parameters:
        if not isinstance(p, Parameter):
            raise TypeError(
                f"{where}: bind keys must be inner dm.Parameter objects, got {type(p).__name__}"
            )
        if id(p) not in {id(q) for q in inner._parameters}:
            raise ValueError(
                f"{where}: parameter {getattr(p, 'name', p)!r} does not belong to the inner model."
            )
        if np.asarray(p.value).ndim != 0:
            raise ValueError(
                f"{where}: bound parameter {p.name!r} is not scalar "
                f"(shape {np.shape(p.value)}). Bind one scalar parameter per "
                "outer expression."
            )


class _ArgminLayer:
    """A callable ``phi(p)`` plus the handles the sensitivity API needs.

    A class rather than a closure with attributes bolted on: the sensitivity
    API reads ``n_variables``/``identify``/``objective_fn`` off it, and those
    should be part of a checked interface rather than dynamic attributes.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        n_variables: int,
        n_constraints: int,
        identify: Callable,
        params_tuple: Callable,
        objective_fn: Callable,
        state: dict,
    ) -> None:
        self._fn = fn
        self._state = state
        self.n_variables = n_variables
        self.n_constraints = n_constraints
        #: ``identify(p) -> (x*, λ*, active_row_mask, at_bound_mask)`` on the host.
        self.identify = identify
        #: ``params_tuple(p)`` -- the inner model's full parameter tuple.
        self.params_tuple = params_tuple
        #: The inner objective in the internal MINIMIZATION form, ``fn(x, params)``.
        self.objective_fn = objective_fn

    def __call__(self, p):
        return self._fn(p)

    def inner_solve_count(self) -> int:
        """POUNCE solves run so far (the forward solve is memoized on ``p``)."""
        return int(self._state["solves"])

    def last_solve_info(self) -> Optional[dict]:
        """Status/objective/iterations of the most recent inner solve."""
        info: Optional[dict] = self._state["last"]
        return info


def _build_layer(
    inner: Model,
    parameters: Sequence[Parameter],
    *,
    options: Optional[dict] = None,
    verify_minimizer: bool = True,
    require_min: bool = True,
    full: bool = False,
) -> "_ArgminLayer":
    """Build ``phi(p)``: the inner solution as a differentiable JAX function.

    The engine behind both :func:`argmin` (the modelling node) and
    :func:`discopt.solvers.sipopt.pounce_sensitivity` (the array API).  ``full``
    makes it return the whole KKT vector ``(x*, λ*)`` rather than just ``x*``, so
    the multiplier sensitivities come from the same differentiation rule as the
    primal ones instead of a second, separately-assembled system.

    Parameters
    ----------
    inner : Model
        The inner model, passed unchanged.
    parameters : sequence of Parameter
        Inner scalar parameters that ``p`` supplies values for, in order.
    options : dict, optional
        POUNCE options for the inner solve.
    verify_minimizer : bool
        Check at each inner solution that the reduced Hessian on the null space of
        the active constraints is positive semidefinite -- i.e. that the point is a
        genuine local minimizer and not merely a KKT point.  The check costs one
        small SVD and one symmetric eigenvalue decomposition per *distinct*
        parameter vector.  Turning it off does not change any derivative; it drops
        the guarantee that the forward value is an argmin at all.
    require_min : bool
        Refuse a maximize inner model.  ``False`` (the sensitivity API) accepts
        one and differentiates the stationary point of the sense as stated.
    full : bool
        Return ``(x*, λ*)`` concatenated instead of ``x*``.

    Returns
    -------
    callable
        ``phi(p)`` mapping a length-``len(parameters)`` JAX vector to the inner
        primal solution (flat, in inner variable order), or to ``(x*, λ*)`` when
        ``full``.  Differentiable twice.  Carries ``inner_solve_count()`` and
        ``last_solve_info()`` for callers that need the underlying POUNCE result.
    """
    import jax
    import jax.numpy as jnp

    from discopt._relax.dag_compiler import (
        _build_param_index,
        compile_constraint_params,
        compile_objective_params,
    )
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.nlp_pounce import solve_nlp

    params = list(parameters)
    _validate_inner(inner, params, require_min=require_min)

    objective = inner._objective
    if objective is None:  # pragma: no cover - _validate_inner refuses this first
        raise ValueError("argmin(): the inner model has no objective set.")

    param_index = _build_param_index(inner)
    _raw_obj = compile_objective_params(inner, param_index)
    if objective.sense == ObjectiveSense.MAXIMIZE:
        # POUNCE is handed the internal minimization form, so its multipliers
        # belong to ``-f``; the stationarity row below must use the same.
        def obj_fn(x_flat, prm):
            return -_raw_obj(x_flat, prm)

    else:
        obj_fn = _raw_obj
    cons = [c for c in inner._constraints if isinstance(c, Constraint)]
    con_fns = [compile_constraint_params(c, inner, param_index) for c in cons]
    lb, ub = _flat_bounds(inner)
    n = int(lb.size)

    # A constraint body may be vector-valued (one Constraint, several rows -- a
    # vectorized collocation residual, say), so the row bounds are expanded by
    # each body's flat size, exactly as the NLP evaluator does.  Getting this
    # wrong would misalign every multiplier, so the count is checked against the
    # evaluator's below rather than assumed.
    x_spec = jax.ShapeDtypeStruct((n,), jnp.float64)
    p_spec = tuple(
        jax.ShapeDtypeStruct(np.shape(prm.value), jnp.float64) for prm in inner._parameters
    )
    sizes = [int(np.prod(jax.eval_shape(f, x_spec, p_spec).shape or (1,))) for f in con_fns]
    cl_c, cu_c = _row_bounds(cons)
    cl = np.repeat(cl_c, sizes) if sizes else np.zeros(0)
    cu = np.repeat(cu_c, sizes) if sizes else np.zeros(0)
    m = int(sum(sizes))
    # The active-row target: an equality's rhs, or whichever finite side the
    # inequality can sit on (a one-sided body has exactly one finite side).
    # Bodies are compiled as ``body - rhs``, so in practice every target is 0.
    row_target = np.where(np.isfinite(cl), cl, cu)

    # Positions of ``params`` inside the inner model's full parameter tuple; the
    # unbound parameters keep their current values, read at solve time.
    bound_slots = [param_index[id(p)] for p in params]
    n_params = len(params)

    opts = dict(options or {})
    opts.setdefault("print_level", 0)

    # The multipliers and the active-set masks are indexed by the *evaluator's*
    # row order, so a disagreement between its layout and the one compiled here
    # would silently pair every multiplier with the wrong constraint.  Check it
    # once, loudly, at build time rather than discovering it as wrong numbers.
    _probe = make_evaluator(inner)
    if (_probe.n_variables, _probe.n_constraints) != (n, m):
        raise ValueError(
            "argmin(): the inner model's compiled layout "
            f"({n} variables, {m} constraint rows) disagrees with the NLP "
            f"evaluator's ({_probe.n_variables}, {_probe.n_constraints}). The "
            "block cannot line its multipliers up with the solver's, so it "
            "refuses rather than return derivatives for the wrong rows."
        )

    state: dict = {"warned_degenerate": False, "solves": 0, "last": None}
    cache: OrderedDict[bytes, tuple] = OrderedDict()

    def _warn_once(message: str) -> None:
        if not state["warned_degenerate"]:
            state["warned_degenerate"] = True
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _failed() -> tuple:
        return (
            np.full(n, np.nan),
            np.full(m, np.nan),
            np.zeros(m),
            np.zeros(n),
        )

    def _identify_host(p_np) -> tuple:
        """POUNCE forward solve + active-set identification, on the host.

        Returns ``(x*, λ*, active_row_mask, at_bound_mask)``.  All four are held
        constant by the differentiation rule: the *identification* of which
        minimizer and which active set is piecewise constant in ``p``, and the
        derivative of ``x*`` comes from the KKT system, never from here.
        """
        p_np = np.atleast_1d(np.asarray(p_np, dtype=np.float64)).reshape(-1)
        key = p_np.tobytes()
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return hit

        for k, prm in enumerate(params):
            prm.value = np.asarray(p_np[k], dtype=np.float64)
        ev = make_evaluator(inner)
        l_, u_ = ev.variable_bounds
        x0 = 0.5 * (np.clip(l_, -1e2, 1e2) + np.clip(u_, -1e2, 1e2))
        res = solve_nlp(ev, x0, options=opts)
        state["solves"] += 1
        state["last"] = {
            "status": getattr(res.status, "value", str(res.status)),
            "objective": float(res.objective) if res.objective is not None else float("nan"),
            "iterations": int(getattr(res, "iterations", 0) or 0),
        }

        out = _failed()
        if res.x is not None:
            out = _post_solve(ev, res)
        cache[key] = out
        if len(cache) > _CACHE_SIZE:
            cache.popitem(last=False)
        return out

    def _post_solve(ev, res) -> tuple:
        """Active set + the soundness gates, at a returned inner solution."""
        x = np.asarray(res.x, dtype=np.float64)
        lam = (
            np.asarray(res.multipliers, dtype=np.float64)
            if res.multipliers is not None
            else np.zeros(m)
        )
        if lam.size != m:
            return _failed()
        g = ev.evaluate_constraints(x) if m else np.zeros(0)

        # Active-set identification is primal-DUAL, not a distance threshold.
        # POUNCE is an interior-point method: a constraint it drives to its bound
        # is left a barrier's width away from it, and how wide that is depends on
        # the instance's scaling, not on how active the constraint is.  Measured
        # on the tutorial's 6-asset portfolio at a binding return target, two
        # weights sit 2.0e-6 and 1.5e-6 off their lower bound while carrying bound
        # multipliers of 1.2e-3 and 1.7e-3: an absolute 1e-6 threshold calls both
        # free, and their stationarity rows then miss the bound duals by 1e-3.
        # Comparing the primal slack against the dual instead -- active when the
        # multiplier dominates the distance -- classifies them correctly at any
        # scaling, and degrades to the absolute tolerance only when BOTH are tiny,
        # which is the degenerate case the complementarity gate below warns about.
        slack_lo = np.where(np.isfinite(cl), np.abs(g - cl), np.inf)
        slack_hi = np.where(np.isfinite(cu), np.abs(g - cu), np.inf)
        row_slack = np.minimum(slack_lo, slack_hi) if m else np.zeros(0)
        active = np.where(
            cl == cu, 1.0, (row_slack <= np.maximum(ACTIVE_TOL, np.abs(lam))).astype(float)
        )

        zl = (
            np.abs(np.asarray(res.bound_multipliers_lower, dtype=float))
            if res.bound_multipliers_lower is not None
            else np.zeros(n)
        )
        zu = (
            np.abs(np.asarray(res.bound_multipliers_upper, dtype=float))
            if res.bound_multipliers_upper is not None
            else np.zeros(n)
        )
        if zl.size != n or zu.size != n:
            zl = zu = np.zeros(n)
        at_bound = (
            (np.isfinite(lb) & ((x - lb) <= np.maximum(ACTIVE_TOL, zl)))
            | (np.isfinite(ub) & ((ub - x) <= np.maximum(ACTIVE_TOL, zu)))
        ).astype(float)

        # Gate 1: the point and multipliers must actually satisfy the reduced KKT
        # system we are about to differentiate.  Without this a wrong multiplier
        # sign convention would leave the forward value right and every
        # derivative silently wrong.
        grad = ev.evaluate_gradient(x)
        jac = ev.evaluate_jacobian(x) if m else np.zeros((0, n))
        stat = grad + (jac.T @ (lam * active) if m else 0.0)
        free = at_bound < 0.5
        resid = float(np.max(np.abs(stat[free]))) if free.any() else 0.0
        if m:
            act = active > 0.5
            if act.any():
                resid = max(resid, float(np.max(np.abs(g[act] - row_target[act]))))
            if (~act).any():
                resid = max(resid, float(np.max(np.abs(lam[~act]))))
        scale = 1.0 + float(np.max(np.abs(grad))) if grad.size else 1.0
        if not np.isfinite(resid) or resid > KKT_RESIDUAL_TOL * scale:
            _warn_once(
                f"argmin(): the inner solve returned a point whose KKT residual is "
                f"{resid:.3e} (tolerance {KKT_RESIDUAL_TOL * scale:.3e}). The "
                "multipliers or the active set do not describe it, so the block "
                "refuses (NaN) rather than return derivatives built on them."
            )
            return _failed()

        # Gate 2: strict complementarity.  A ~zero multiplier on an active row
        # means dx*/dp is one-sided there; that is a real limitation of the
        # sensitivity, so it is warned rather than hidden.
        degenerate = bool(
            m and np.any((active > 0.5) & (cl != cu) & (np.abs(lam) < COMPLEMENTARITY_TOL))
        )
        if not degenerate:
            degenerate = bool(np.any((at_bound > 0.5) & (zl + zu < COMPLEMENTARITY_TOL)))
        if degenerate:
            _warn_once(
                "argmin(): the inner solution has a degenerate active set (an "
                "active constraint or bound with a ~zero multiplier). dx*/dp is "
                "only one-sided there; treat the outer gradient as a subgradient."
            )

        # Gate 3: is it a minimizer at all?  A KKT point with indefinite reduced
        # curvature is exactly the failure the hand-written-KKT route suffers.
        if verify_minimizer and not _reduced_hessian_psd(ev, x, lam, active, at_bound):
            _warn_once(
                "argmin(): the inner solve returned a KKT point whose reduced "
                "Hessian is not positive semidefinite -- it is not a local "
                "minimizer, so the block refuses (NaN). Give the inner model "
                "tighter bounds, or solve it as a convex follower."
            )
            return _failed()

        return (x, lam, active, at_bound)

    def _reduced_hessian_psd(ev, x, lam, active, at_bound) -> bool:
        """``Zᵀ W Z ⪰ 0`` on the null space of the active constraints and bounds."""
        w = ev.evaluate_lagrangian_hessian(x, 1.0, lam * active if m else lam)
        rows: list[np.ndarray] = []
        if m:
            jac = ev.evaluate_jacobian(x)
            rows.extend(jac[j] for j in range(m) if active[j] > 0.5)
        for i in range(n):
            if at_bound[i] > 0.5:
                e = np.zeros(n)
                e[i] = 1.0
                rows.append(e)
        if rows:
            a = np.asarray(rows, dtype=float)
            _, s, vt = np.linalg.svd(a)
            rank = int(np.sum(s > max(a.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)))
            z = vt[rank:].T
        else:
            z = np.eye(n)
        if z.shape[1] == 0:  # fully determined by the active set: nothing to check
            return True
        red = z.T @ np.asarray(w, dtype=float) @ z
        red = 0.5 * (red + red.T)
        eig = float(np.min(np.linalg.eigvalsh(red)))
        scale = 1.0 + float(np.max(np.abs(red)))
        return eig >= CURVATURE_TOL * scale

    shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
    )

    # A host callback cannot be differentiated, and does not need to be: every
    # derivative below comes from the KKT system.  The explicit zero-tangent rule
    # states that -- rather than letting the callback be dragged onto the tangent
    # path, where JAX raises "Pure callbacks do not support JVP".
    @jax.custom_jvp
    def _identify(p):
        return jax.pure_callback(_identify_host, shapes, p, vmap_method="sequential")

    @_identify.defjvp
    def _identify_jvp(primals, tangents):
        out = _identify(*primals)
        return out, tuple(jnp.zeros_like(o) for o in out)

    def _cons_vec(x, pt):
        """All constraint bodies, flattened and concatenated in evaluator order."""
        if not con_fns:
            return jnp.zeros(0, dtype=jnp.float64)
        return jnp.concatenate([jnp.reshape(f(x, pt), (-1,)) for f in con_fns])

    def _params_tuple(p):
        """The inner model's full parameter tuple, with ``p`` in the bound slots."""
        vals = [jnp.asarray(prm.value, dtype=jnp.float64) for prm in inner._parameters]
        for k, slot in enumerate(bound_slots):
            vals[slot] = jnp.reshape(p[k], ())
        return tuple(vals)

    def phi(p):
        p = jnp.reshape(jnp.asarray(p, dtype=jnp.float64), (n_params,))
        x_star, lam_star, active, at_bound = _identify(p)
        active = jax.lax.stop_gradient(active)
        at_bound = jax.lax.stop_gradient(at_bound)
        x_star = jax.lax.stop_gradient(x_star)
        lam_star = jax.lax.stop_gradient(lam_star)
        pt = _params_tuple(p)

        def kkt(z):
            # Variables sitting on a bound are held at their solution value: under
            # strict complementarity the active set is locally constant, so their
            # sensitivity is zero and their stationarity row is absorbed by the
            # bound multiplier.  Their row below is replaced by ``x_i - x_i*``,
            # which keeps the system square without needing bound duals.
            x = jnp.where(at_bound > 0, x_star, z[:n])
            lam = z[n:]
            lam_active = lam * active
            # One reverse pass for the whole stationarity row: grad(f + λ_A·c) is
            # ∇f + J_Aᵀλ_A without ever forming J.
            stat = jax.grad(
                lambda xx: jnp.reshape(obj_fn(xx, pt), ()) + jnp.dot(lam_active, _cons_vec(xx, pt))
            )(x)
            stat = jnp.where(at_bound > 0, z[:n] - x_star, stat)
            # Inactive rows contribute nothing and pin their multiplier to 0,
            # which keeps the system square with the active set held fixed.
            rows = (
                jnp.where(active > 0, _cons_vec(x, pt) - row_target, lam)
                if m
                else jnp.zeros(0, dtype=stat.dtype)
            )
            return jnp.concatenate([stat, rows])

        z0 = jnp.concatenate([x_star, lam_star])

        def solve(_f, _guess):
            # The root is the point POUNCE already returned; ``custom_root`` needs
            # it only to linearize about.
            return z0

        def tangent_solve(g, y):
            return jnp.linalg.solve(jax.jacobian(g)(jnp.zeros_like(y)), y)

        z = jax.lax.custom_root(kkt, z0, solve, tangent_solve)
        return z if full else z[:n]

    return _ArgminLayer(
        phi,
        n_variables=n,
        n_constraints=m,
        identify=_identify_host,
        params_tuple=_params_tuple,
        objective_fn=obj_fn,
        state=state,
    )


def argmin_layer(
    inner: Model,
    parameters: Sequence[Parameter],
    *,
    options: Optional[dict] = None,
    verify_minimizer: bool = True,
) -> Callable:
    """``phi(p) -> x*``: the inner minimizer as a twice-differentiable JAX function.

    The numerics of :func:`argmin` without the modelling layer, so a follower's
    solution and its sensitivities are usable (and testable) on their own::

        phi = dm.argmin_layer(inner, [q])
        x_star = phi(jnp.array([0.7]))
        dx_dp = jax.jacobian(phi)(jnp.array([0.7]))
        d2x_dp2 = jax.jacfwd(jax.jacobian(phi))(jnp.array([0.7]))

    See :func:`argmin` for the arguments and the soundness contract.  ``phi``
    carries ``inner_solve_count()`` (how many POUNCE solves it has run -- the
    forward solve is memoized on the parameter vector) and ``last_solve_info()``.
    """
    return _build_layer(
        inner,
        parameters,
        options=options,
        verify_minimizer=verify_minimizer,
        require_min=True,
        full=False,
    )


def argmin(
    inner: Model,
    bind: dict,
    *,
    options: Optional[dict] = None,
    verify_minimizer: bool = True,
    name: str = "argmin",
) -> Expression:
    """Embed ``inner``'s minimizer as a block of the calling model.

    Parameters
    ----------
    inner : Model
        The inner (follower) model, used as written -- objective and constraints
        untouched.  Continuous variables only, minimization only.
    bind : dict
        ``{inner_parameter: outer_expression}``.  Each key is a scalar
        :class:`~discopt.modeling.core.Parameter` of ``inner``; each value is an
        expression of the *outer* model (a variable, or anything built from one)
        supplying that parameter's value.  Inner parameters left unbound keep
        their current ``.value``.
    options : dict, optional
        POUNCE options for the inner solve (e.g. ``{"tol": 1e-10}``).
    verify_minimizer : bool
        Verify at every inner solution that its reduced Hessian is positive
        semidefinite; see :func:`argmin_layer`.
    name : str
        Display name of the node in reprs and errors.

    Returns
    -------
    Expression
        The inner primal solution as a vector node, indexable ``v[i]`` in inner
        variable order (a vector inner variable contributes its elements in
        ``ravel`` order).  Use it anywhere an expression is allowed.

    Raises
    ------
    ValueError
        If ``inner`` has no objective, maximizes, has integer/binary variables, or
        ``bind`` names a parameter that is not a scalar parameter of ``inner``.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> inner = dm.Model("projection")                      # doctest: +SKIP
    >>> y = inner.continuous("y", shape=(2,), lb=-2, ub=2)  # doctest: +SKIP
    >>> q = inner.parameter("q", value=1.0)                 # doctest: +SKIP
    >>> inner.minimize((y[0] - q) ** 2 + (y[1] - 1) ** 2)   # doctest: +SKIP
    >>> inner.subject_to(y[0] * y[0] + y[1] * y[1] == 1)    # doctest: +SKIP
    >>> m = dm.Model("leader")                              # doctest: +SKIP
    >>> p = m.continuous("p", lb=0.05, ub=3.0)              # doctest: +SKIP
    >>> v = dm.argmin(inner, bind={q: p})                   # doctest: +SKIP
    >>> m.minimize((v[0] + 0.9) ** 2)                       # doctest: +SKIP
    >>> result = m.solve()                                  # doctest: +SKIP

    See Also
    --------
    argmin_kkt : the lowered arm -- a certified, ``.nl``-exportable reformulation
        for a follower that can be *proved* convex in its own variables.

    Notes
    -----
    The outer model is solved on the local NLP path with no global certificate
    (``status="feasible"``, ``gap_certified=False``), and cannot be exported to
    ``.nl``: see the module docstring for the full contract.
    """
    import jax.numpy as jnp

    if not isinstance(bind, dict) or not bind:
        raise ValueError(
            "argmin(): bind must be a non-empty {inner_parameter: outer_expression} "
            "dict. An inner model with nothing bound to the outer model is a "
            "constant -- solve it once and use the number."
        )
    params = list(bind.keys())
    outer_exprs = list(bind.values())
    phi = argmin_layer(inner, params, options=options, verify_minimizer=verify_minimizer)

    def fn(*vals):
        return phi(jnp.stack([jnp.reshape(jnp.asarray(v, dtype=jnp.float64), ()) for v in vals]))

    fn.__name__ = name
    node: Expression = custom(fn, name=name)(*outer_exprs)
    return node


# ---------------------------------------------------------------------------
# The lowered arm: the follower's KKT conditions, gated on convexity (#1216)
# ---------------------------------------------------------------------------


def _remap(expr, scalars: dict, arrays: dict, params: dict):
    """Rebuild ``expr`` against the outer model's variables and expressions.

    ``scalars`` maps ``id(inner scalar Variable)`` to its outer replacement;
    ``arrays`` maps ``id(inner 1-D Variable)`` to an object ndarray of the outer
    scalars standing in for its components; ``params`` maps ``id(inner
    Parameter)`` to the bound outer expression (or its frozen value).

    Every node type is handled explicitly and anything else raises. A rewriter
    that passes an unrecognized node through unchanged would leave the inner
    model's own variable in an outer constraint -- a silently wrong model rather
    than a refusal.
    """
    if isinstance(expr, Constant):
        return expr
    if isinstance(expr, Parameter):
        try:
            return params[id(expr)]
        except KeyError:
            raise ValueError(
                f"argmin_kkt(): parameter {expr.name!r} is not a parameter of the "
                "inner model, so the lowering has no value for it."
            ) from None
    if isinstance(expr, Variable):
        if id(expr) in scalars:
            return scalars[id(expr)]
        if id(expr) in arrays:
            return arrays[id(expr)]
        raise ValueError(
            f"argmin_kkt(): the inner objective/constraints reference {expr.name!r}, "
            "which is not a variable of the inner model. A follower may only depend "
            "on its own variables and its parameters."
        )
    if isinstance(expr, IndexExpression):
        base = _remap(expr.base, scalars, arrays, params)
        if isinstance(base, np.ndarray):
            return base[expr.index]
        return IndexExpression(base, expr.index)
    if isinstance(expr, BinaryOp):
        return BinaryOp(
            expr.op,
            _remap(expr.left, scalars, arrays, params),
            _remap(expr.right, scalars, arrays, params),
        )
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _remap(expr.operand, scalars, arrays, params))
    if isinstance(expr, FunctionCall):
        return FunctionCall(
            expr.func_name, *(_remap(a, scalars, arrays, params) for a in expr.args)
        )
    if isinstance(expr, MatMulExpression):
        left = _remap(expr.left, scalars, arrays, params)
        right = _remap(expr.right, scalars, arrays, params)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            # One side became an object array of scalar expressions, so the matmul
            # is folded here into ordinary scalar arithmetic rather than left as a
            # node over a variable that no longer exists.
            return _as_object_array(left) @ _as_object_array(right)
        return MatMulExpression(left, right)
    if isinstance(expr, SumExpression):
        operand = _remap(expr.operand, scalars, arrays, params)
        if isinstance(operand, np.ndarray):
            return _fold_sum(operand, expr.axis)
        return SumExpression(operand, expr.axis)
    if isinstance(expr, SumOverExpression):
        return SumOverExpression([_remap(t, scalars, arrays, params) for t in expr.terms])
    if isinstance(expr, CustomCall):
        raise ValueError(
            f"argmin_kkt(): the follower contains an opaque node ({expr.name!r}). Its "
            "KKT conditions cannot be written symbolically, which is the whole point "
            "of the lowering. Use dm.argmin() for this follower."
        )
    raise TypeError(
        f"argmin_kkt(): unhandled expression node {type(expr).__name__}. The lowering "
        "refuses rather than pass a node through unrewritten, which would leave the "
        "inner model's own variables in an outer constraint."
    )


def _as_object_array(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=object)


def _fold_sum(arr: np.ndarray, axis):
    """``dm.sum`` over an object array, folded into scalar ``+``."""
    flat = arr.ravel() if axis is None else arr.sum(axis=axis)
    if isinstance(flat, np.ndarray) and flat.ndim == 0:
        return flat.item()
    if axis is not None:
        return flat
    total = flat[0]
    for term in flat[1:]:
        total = total + term
    return total


def _lower_variables(model: Model, inner: Model, prefix: str):
    """Create the follower's variables on the outer model, one scalar per component."""
    scalars: dict = {}
    arrays: dict = {}
    ordered: list[Variable] = []
    for v in inner._variables:
        shape = tuple(v.shape or ())
        if len(shape) > 1:
            raise NotImplementedError(
                f"argmin_kkt(): follower variable {v.name!r} has shape {shape}. The "
                "KKT lowering emits one stationarity row per SCALAR follower variable, "
                "so multi-dimensional follower variables are not supported yet; "
                "declare them as 1-D, or use dm.argmin() for the opaque block."
            )
        lb = np.broadcast_to(np.asarray(v.lb, dtype=float), shape or ())
        ub = np.broadcast_to(np.asarray(v.ub, dtype=float), shape or ())
        if not shape:
            new = model.continuous(
                _unique_name(model, f"{prefix}_{v.name}"), lb=float(lb), ub=float(ub)
            )
            scalars[id(v)] = new
            ordered.append(new)
            continue
        comps = []
        for i in range(shape[0]):
            new = model.continuous(
                _unique_name(model, f"{prefix}_{v.name}_{i}"),
                lb=float(lb.reshape(-1)[i]),
                ub=float(ub.reshape(-1)[i]),
            )
            comps.append(new)
            ordered.append(new)
        arrays[id(v)] = np.array(comps, dtype=object)
    return scalars, arrays, ordered


def argmin_kkt(
    model: Model,
    inner: Model,
    bind: dict,
    *,
    method: str = "kkt",
    mpec_method: str = "gdp",
    multiplier_ub: Optional[float] = None,
    prefix: str = "argmin",
):
    """Lower a **convex** follower into ``model`` as its KKT conditions.

    The algebraic counterpart of :func:`argmin`, and the same trade
    :func:`~discopt.modeling.implicit.implicit_full_space` makes against
    :func:`~discopt.modeling.implicit.implicit`: instead of hiding the follower
    behind an opaque node, its variables become real variables of ``model`` and its
    optimality conditions become real constraints. What that buys is everything the
    opaque node forecloses -- the Rust tape, FBBT, a **global certificate**, and
    ``.nl`` export, so the same formulation can be handed to another solver.

    What it costs is generality, and the cost is not negotiable: KKT conditions
    characterize a follower's optimum only when the follower is **convex in its own
    variables**. For a nonconvex follower the KKT set contains every stationary
    point, and lowering it would reproduce exactly the failure #1216 reports -- the
    leader selecting the follower's *maximizer*. So the lowering runs
    :class:`discopt.bilevel.BilevelProblem`'s convexity certifier and **refuses**
    anything it cannot prove convex in ``y`` (LP, convex-QP, and certified
    convex-NLP followers pass; a nonlinear equality does not). There is no flag to
    override that: for an unprovable follower, use :func:`argmin`, which solves
    rather than reformulates and reports its answer as local.

    Parameters
    ----------
    model : Model
        The outer (leader) model. The follower's variables, multipliers, and
        optimality constraints are added to it **in place**.
    inner : Model
        The follower, passed unchanged. Continuous, scalar-or-1-D variables;
        minimization; no opaque nodes.
    bind : dict
        ``{inner_parameter: outer_expression}`` -- the coupling, exactly as in
        :func:`argmin`. Inner parameters left unbound are frozen at their current
        ``.value``.
    method : {"kkt", "strong_duality"}
        The single-level reduction; see
        :meth:`discopt.bilevel.BilevelProblem.formulate`.
    mpec_method : {"gdp", "sos1"}
        How the complementarity conditions are encoded (``method="kkt"`` only).
    multiplier_ub : float, optional
        A valid finite upper bound on the follower's KKT multipliers. The big-M
        complementarity encodings refuse an unbounded multiplier rather than emit a
        vacuous big-M, so a follower with inequality rows needs either this or
        ``method="strong_duality"``.
    prefix : str
        Name prefix for the emitted follower variables and multipliers.

    Returns
    -------
    numpy.ndarray
        Object array of the follower's variables on ``model``, in inner variable
        order (a 1-D inner variable contributes its components). Index it
        (``v[0]``) exactly like :func:`argmin`'s node; each entry is a real
        :class:`~discopt.modeling.core.Variable`, so ``result.value(v[0])`` works.

    Raises
    ------
    ValueError
        For the same inner models :func:`argmin` refuses, plus an opaque node or a
        foreign variable in the follower.
    NotImplementedError
        If the convexity certifier cannot prove the follower convex in its own
        variables, or the follower has a multi-dimensional variable.

    Examples
    --------
    >>> v = dm.argmin_kkt(m, inner, bind={q: p}, multiplier_ub=100.0)  # doctest: +SKIP
    >>> m.minimize((v[0] - 2.0) ** 2)                                  # doctest: +SKIP
    >>> m.to_nl("leader.nl")   # an ordinary algebraic model            # doctest: +SKIP
    """
    from discopt.bilevel import BilevelProblem

    if not isinstance(bind, dict) or not bind:
        raise ValueError(
            "argmin_kkt(): bind must be a non-empty {inner_parameter: outer_expression} "
            "dict. An inner model with nothing bound to the outer model is a constant "
            "-- solve it once and use the number."
        )
    _validate_inner(inner, list(bind.keys()), require_min=True, where="argmin_kkt()")

    upper_vars = list(model._variables)
    scalars, arrays, ordered = _lower_variables(model, inner, prefix)

    params: dict = {id(p): Constant(np.asarray(p.value, dtype=float)) for p in inner._parameters}
    for prm, outer_expr in bind.items():
        params[id(prm)] = outer_expr if isinstance(outer_expr, Expression) else Constant(outer_expr)

    inner_objective = inner._objective
    if inner_objective is None:  # pragma: no cover - _validate_inner refuses this first
        raise ValueError("argmin_kkt(): the inner model has no objective set.")
    objective = _remap(inner_objective.expression, scalars, arrays, params)
    if isinstance(objective, np.ndarray):
        raise ValueError("argmin_kkt(): the follower's objective is not scalar.")

    lower_constraints: list[Constraint] = []
    for i, con in enumerate(inner._constraints):
        if not isinstance(con, Constraint):
            raise ValueError(
                "argmin_kkt(): the follower carries a non-algebraic relation "
                f"({type(con).__name__}); the KKT lowering handles ordinary "
                "constraints only."
            )
        body = _remap(con.body, scalars, arrays, params)
        if isinstance(body, np.ndarray):
            raise NotImplementedError(
                f"argmin_kkt(): follower constraint {i} is vector-valued. The KKT "
                "lowering emits one multiplier per SCALAR row; write the rows "
                "individually, or use dm.argmin() for the opaque block."
            )
        # build_kkt takes '<=' and '==' only; '>=' is the same row negated.
        sense = con.sense
        if sense == ">=":
            body, sense = -body, "<="
        lower_constraints.append(
            Constraint(body=body, sense=sense, rhs=0.0, name=f"{prefix}_lower_{i}")
        )

    problem = BilevelProblem(
        model,
        upper_vars=upper_vars,
        lower_vars=ordered,
        lower_objective=objective,
        lower_constraints=lower_constraints,
        lower_sense="min",
        prefix=prefix,
        multiplier_ub=multiplier_ub,
    )
    problem.formulate(method=method, mpec_method=mpec_method)
    return np.array(ordered, dtype=object)
