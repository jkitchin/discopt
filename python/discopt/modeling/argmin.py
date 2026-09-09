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

What you get, and what you do not
---------------------------------

``argmin`` is built on :func:`~discopt.modeling.core.custom`, so it inherits the
``CustomCall`` contract exactly: the outer model is solved on the **local NLP path
only** -- ``status="feasible"``, ``gap_certified=False``, no ``bound``/``gap``, no
``.nl`` export, and integer/binary variables in the *outer* model are refused.  The
forward pass is a *local* solve, so the block is a local argmin; for a nonconvex
follower a different POUNCE starting point may be a different branch.  That is a
strictly weaker claim than a global bilevel certificate -- for a follower that is
provably convex in its own variables, prefer
:class:`discopt.bilevel.BilevelProblem`, which reformulates to a single-level MPEC
and keeps the global path.

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
    Constraint,
    Expression,
    Model,
    ObjectiveSense,
    Parameter,
    VarType,
    custom,
)

__all__ = ["argmin", "argmin_layer"]

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
    inner: Model, parameters: Sequence[Parameter], require_min: bool = True
) -> None:
    """Refuse the inner models an argmin block cannot soundly represent.

    ``require_min=False`` is for the sensitivity API (:mod:`discopt.solvers.sipopt`),
    which differentiates a stationary point of whichever sense the model states;
    the block itself must be a minimization or its name would be a lie.
    """
    if inner._objective is None:
        raise ValueError("argmin(): the inner model has no objective set.")
    if require_min and inner._objective.sense == ObjectiveSense.MAXIMIZE:
        raise ValueError(
            "argmin(): the inner model maximizes, so this block would be an "
            "argmax and the KKT sign convention would be inverted. State the "
            "follower as a minimization -- inner.minimize(-expr) -- and negate "
            "in the outer model if you need the value."
        )
    bad = [v.name for v in inner._variables if v.var_type != VarType.CONTINUOUS]
    if bad:
        raise ValueError(
            "argmin(): the inner model has integer/binary variables "
            f"({', '.join(bad)}). The block's derivatives come from KKT "
            "stationarity, which does not characterize an integer optimum. Model "
            "an integer follower with discopt.bilevel or an explicit "
            "reformulation instead."
        )
    for p in parameters:
        if not isinstance(p, Parameter):
            raise TypeError(
                f"argmin(): bind keys must be inner dm.Parameter objects, got {type(p).__name__}"
            )
        if id(p) not in {id(q) for q in inner._parameters}:
            raise ValueError(
                f"argmin(): parameter {getattr(p, 'name', p)!r} does not belong to the inner model."
            )
        if np.asarray(p.value).ndim != 0:
            raise ValueError(
                f"argmin(): bound parameter {p.name!r} is not scalar "
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
