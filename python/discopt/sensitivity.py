"""Unified derivative entry point: every sensitivity of a solved model, one call.

discopt already computes derivatives in four different places, each with its own
call shape and its own idea of what a "derivative of a model" is:

* :meth:`SolveResult.gradient <discopt.modeling.core.SolveResult.gradient>` --
  ``d(obj*)/dp`` only, by the envelope theorem, lazily at a solved point.
* :func:`discopt.solvers.sipopt.pounce_sensitivity` -- the full KKT/IFT system,
  returning ``dx*/dp``, ``dλ*/dp`` and (``order=2``) ``d²x*/dp²`` as raw arrays
  indexed by flat position.
* :func:`discopt.modeling.argmin_layer` -- the same numbers as a JAX callable
  ``phi(p) -> x*``, for composing into a larger differentiable pipeline.
* :func:`discopt.parametric.compile_expression` -- partial derivatives of an
  arbitrary model expression, with no solve and no solution map at all.

Picking between them, and then lining up a flat index against the variable you
actually asked about, is work the caller should not be doing.  :func:`sensitivity`
(and its method form, :meth:`Model.sensitivity
<discopt.modeling.core.Model.sensitivity>`) is the one entry point::

    s = model.sensitivity(wrt=[a, b])

    s.dx_dp                 # dx*/dp, the whole matrix
    s.d(y)                  # dy*/dp for one variable, in that variable's shape
    s.d(x * y + dm.exp(z))  # TOTAL derivative of any expression through x*(p)
    s.dobj_dp               # d(obj*)/dp
    s.predict({a: 2.1})     # first-order re-solve, no second solve
    s.as_layer()            # the JAX phi(p) -> x*, for jax.grad/jit/vmap

What ``d()`` adds over the arrays
---------------------------------
``dx_dp`` answers "how does the *solution vector* move".  The question usually
being asked is "how does *this quantity* move", where the quantity is some
expression in the model's variables -- a yield, a cost, a constraint residual.
That is a **total** derivative through the solution map::

    de/dp = ∂e/∂p |_{x*}  +  (∂e/∂x |_{x*}) · dx*/dp

which needs both halves: the partials come from the parametric compiler and
``dx*/dp`` from the KKT system.  :meth:`Sensitivity.d` is the only place the two
are joined.  For the objective specifically, that total derivative coincides with
the envelope theorem's ``∂L/∂p`` -- stationarity kills the indirect term -- and
:attr:`Sensitivity.dobj_dp` is checked against it in the test suite.

Scope and contract
------------------
The numerics are :func:`~discopt.solvers.sipopt.pounce_sensitivity`'s, unchanged
-- this module marshals, it does not re-derive.  That inherits the contract:
continuous variables only (KKT stationarity does not characterize an integer
optimum), scalar :class:`~discopt.modeling.core.Parameter` objects only, and the
derivatives belong to the KKT point POUNCE returned, which for a nonconvex model
is a local one.  ``active``/``at_bound`` record the active set the system was
assembled over, because a sensitivity is only meaningful alongside it: a
different active set is a different piecewise branch of ``x*(p)``, and at a
branch boundary the one-sided derivatives differ.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Expression, Model

__all__ = ["Sensitivity", "sensitivity"]


# ─────────────────────────────────────────────────────────────
# Parameter / variable resolution
# ─────────────────────────────────────────────────────────────


def _resolve_parameters(model: "Model", wrt) -> list:
    """Normalize ``wrt`` into a list of the model's own scalar Parameters.

    Accepts ``None`` (every declared Parameter, in declaration order), a single
    Parameter or name, or a sequence of either.  Every rejection names what was
    wrong and what to do instead -- a wrong parameter here produces a derivative
    matrix whose columns mean something other than what the caller thinks, which
    is worse than an error.
    """
    from discopt.modeling.core import Parameter

    declared = list(model._parameters)
    if not declared:
        raise ValueError(
            f"Model {model.name!r} declares no Parameters, so there is nothing to "
            "differentiate with respect to. Create one with "
            "`p = model.parameter('p', value=...)` and use it in the objective or "
            "a constraint; sensitivities are taken w.r.t. Parameters, not Variables."
        )

    if wrt is None:
        chosen = list(declared)
    else:
        if isinstance(wrt, (Parameter, str)):
            items: Sequence = [wrt]
        else:
            # A Parameter is an Expression and therefore indexable; the isinstance
            # check above has to come first or `wrt=p` would iterate the expression.
            items = list(wrt)
        by_name = {p.name: p for p in declared}
        owned = {id(p) for p in declared}
        chosen = []
        for item in items:
            if isinstance(item, str):
                if item not in by_name:
                    raise KeyError(
                        f"{item!r} is not a Parameter of model {model.name!r}. "
                        f"Declared parameters: {sorted(by_name)}"
                    )
                chosen.append(by_name[item])
            elif isinstance(item, Parameter):
                if id(item) not in owned:
                    raise ValueError(
                        f"Parameter {item.name!r} does not belong to model "
                        f"{model.name!r}. Pass parameters created by this model's "
                        "`.parameter(...)`."
                    )
                chosen.append(item)
            else:
                raise TypeError(
                    "wrt entries must be dm.Parameter objects or their names, got "
                    f"{type(item).__name__}. To differentiate w.r.t. a quantity that "
                    "is currently a Variable, make it a Parameter instead."
                )

    seen: dict[int, str] = {}
    for p in chosen:
        if id(p) in seen:
            raise ValueError(
                f"Parameter {p.name!r} appears more than once in `wrt`; each column "
                "of the sensitivity matrix must be a distinct parameter."
            )
        seen[id(p)] = p.name

    nonscalar = [(p.name, np.shape(p.value)) for p in chosen if np.asarray(p.value).ndim != 0]
    if nonscalar:
        detail = ", ".join(f"{n} (shape {s})" for n, s in nonscalar)
        raise ValueError(
            f"sensitivity() differentiates w.r.t. scalar Parameters only; {detail} "
            "is not scalar. Declare one scalar Parameter per quantity you want a "
            "derivative column for -- a vector parameter would need its own column "
            "layout, which the KKT system's right-hand side does not carry."
        )
    return chosen


def _unflatten(model: "Model", flat: np.ndarray) -> dict:
    """Split a flat variable vector into ``{name: array}`` in each variable's shape.

    Matches the layout :func:`discopt.parametric.variable_slices` documents and
    the key set of ``SolveResult.x``, so a caller can index a sensitivity the same
    way they index a solution.
    """
    out: dict[str, np.ndarray] = {}
    offset = 0
    for v in model._variables:
        chunk = np.asarray(flat[offset : offset + v.size], dtype=np.float64)
        offset += v.size
        out[v.name] = chunk.reshape(v.shape)
    return out


# ─────────────────────────────────────────────────────────────
# Result object
# ─────────────────────────────────────────────────────────────


# eq=False: the generated __eq__ would compare numpy arrays elementwise and raise
# "truth value of an array is ambiguous", and frozen=True would then build a
# __hash__ over those same unhashable arrays. Identity semantics are what a
# result object of this size wants anyway.
@dataclass(frozen=True, eq=False)
class Sensitivity:
    """Solution and every derivative of it, from one solve.

    Built by :func:`sensitivity` / :meth:`Model.sensitivity
    <discopt.modeling.core.Model.sensitivity>`; not constructed directly.

    Attributes
    ----------
    model : Model
        The model this was computed for.  Its Parameter values are restored to
        what they were before the solve, so the object does not depend on the
        model being left untouched afterwards -- but ``d()`` re-evaluates
        expressions at ``x``, so a *structural* edit to the model invalidates it.
    parameters : list of Parameter
        The parameters differentiated with respect to; column ``k`` of every
        matrix below is ``parameters[k]``.
    x : ndarray, shape (n,)
        The solution, flat, in declaration order.
    x_dict : dict[str, ndarray]
        The same solution keyed by variable name, each in the variable's shape.
    objective : float
        Objective value at ``x``.
    status : str
        POUNCE termination status of the **local** NLP this sensitivity was taken
        at -- ``"optimal"`` here means that local solve converged to a KKT point,
        NOT that the point is a certified global optimum the way
        :attr:`SolveResult.status <discopt.modeling.core.SolveResult.status>`
        does.  On a nonconvex model use :attr:`matches_reference` to tell whether
        the point agrees with what :meth:`Model.solve` certified (#1313).
    multipliers : ndarray, shape (m,)
        Constraint multipliers at ``x``, **in the internal-minimization sign
        convention**: a MAXIMIZE model is solved as ``-f``, so these multipliers
        (and the stationarity rows they satisfy) belong to ``-f``, and the same
        number comes back for ``min f`` and ``max -f``.  This is the convention
        :attr:`SolveResult.constraint_duals
        <discopt.modeling.core.SolveResult.constraint_duals>` documents and uses;
        multiply by ``-1`` for a MAXIMIZE model to read them against the
        objective as written.  :attr:`objective` and :attr:`dobj_dp` are *not*
        affected -- both are reported in the model's own sense (#1313).
    dx_dp : ndarray, shape (n, n_params)
        ``dx*/dp``. Column ``k`` is the derivative w.r.t. ``parameters[k]``.
    dlambda_dp : ndarray, shape (m, n_params)
        ``dλ*/dp``, from the same system rather than a separately assembled one.
        Carries the same internal-minimization sign convention as
        :attr:`multipliers` -- see there.
    d2x_dp2 : ndarray, shape (n, n_params, n_params), or None
        ``∂²x*/∂p_j∂p_k``; ``None`` unless ``order=2`` was requested.
    active : ndarray of bool, shape (m,)
        Which constraint rows were active and therefore entered the system.
    at_bound : ndarray of bool, shape (n,)
        Which variables sat on one of their own bounds (derivative 0).
    method : str
        ``"exact"`` (symbolic right-hand side) or ``"fd"`` (central differences).
    order : int
        1 or 2 -- the highest derivative order computed.
    p0 : ndarray, shape (n_params,)
        The parameter values the derivatives were taken at -- the expansion point
        :meth:`predict` measures ``Δp`` from.  Pinned here rather than re-read from
        ``Parameter.value``, so a later edit to the model cannot silently
        reinterpret what a prediction means.
    p_all : ndarray
        The model's *whole* flat parameter vector at that same moment, including
        parameters not differentiated with respect to.  :meth:`d` evaluates an
        expression's partials against this rather than against live values, for
        the same reason: the partials and ``dx_dp`` have to belong to one point.
    reference_objective : float or None
        Objective of the reference solution this was cross-checked against -- the
        model's last :meth:`Model.solve` result, or whatever ``at=`` named -- in
        the model's own sense.  ``None`` when the model has never been solved and
        no reference was supplied, i.e. when there is nothing to check against.
    matches_reference : bool or None
        Whether :attr:`objective` agrees with :attr:`reference_objective` to
        within tolerance.  ``None`` when there is no reference.  ``False`` means
        the derivatives describe a *different* stationary point than the one the
        model was solved to -- on a nonconvex model, a different basin -- and a
        ``RuntimeWarning`` was raised when the object was built (#1313).
    """

    model: "Model"
    parameters: list
    x: np.ndarray
    x_dict: dict
    objective: float
    status: str
    multipliers: np.ndarray
    dx_dp: np.ndarray
    dlambda_dp: np.ndarray
    d2x_dp2: Optional[np.ndarray]
    active: np.ndarray
    at_bound: np.ndarray
    method: str
    order: int
    p0: np.ndarray
    p_all: np.ndarray
    reference_objective: Optional[float] = None
    matches_reference: Optional[bool] = None
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    # ── basic descriptors ────────────────────────────────────

    @property
    def param_names(self) -> list:
        """Names of the parameters, in column order."""
        return [p.name for p in self.parameters]

    @property
    def n_parameters(self) -> int:
        """Number of parameter columns."""
        return len(self.parameters)

    def __repr__(self) -> str:  # pragma: no cover - display only
        flag = "" if self.matches_reference is not False else ", matches_reference=False"
        return (
            f"Sensitivity(status={self.status!r}, objective={self.objective:.6g}, "
            f"dx_dp{self.dx_dp.shape}, wrt={self.param_names}, "
            f"method={self.method!r}, order={self.order}{flag})"
        )

    # ── derivatives ──────────────────────────────────────────

    def d(self, target, wrt=None):
        """Total derivative of ``target`` w.r.t. the parameters, through ``x*(p)``.

        This is the entry point the other attributes are special cases of.

        Parameters
        ----------
        target : Variable, str, or Expression
            What to differentiate.  A :class:`~discopt.modeling.core.Variable` (or
            its name) is read straight out of :attr:`dx_dp` -- exact, no
            re-evaluation.  Any other expression is differentiated by the chain
            rule ``∂e/∂p + (∂e/∂x)·dx*/dp``, so the parameter dependence *inside*
            the expression and the dependence *through the solution* are both
            counted.  Forgetting the first term is the classic wrong answer here.
        wrt : Parameter, str, or sequence, optional
            Restrict to these parameters (a subset of :attr:`parameters`).  A
            single parameter drops the trailing axis, so ``s.d(y, a)`` has the
            shape of ``y``.

        Returns
        -------
        numpy.ndarray
            Shape ``target.shape + (n_selected,)``, or ``target.shape`` when a
            single parameter was named.

        Examples
        --------
        >>> s = m.sensitivity(wrt=[a, b])
        >>> s.d("y")            # dy*/d(a, b), shaped like y  # doctest: +SKIP
        >>> s.d(x * y[0], a)    # d(x*y0)/da, a scalar        # doctest: +SKIP
        """
        cols = self._select_columns(wrt)
        # `cols` is a slice (all), an int (one parameter -- the trailing axis is
        # dropped by ordinary indexing) or a list (a subset, axis kept).
        return self._total_derivative(target)[..., cols]

    def __getitem__(self, target):
        """``s["y"]`` is ``s.d("y")`` -- the derivative of one variable."""
        return self.d(target)

    @property
    def dobj_dp(self) -> np.ndarray:
        """``d(obj*)/dp``, the derivative of the optimal objective value.

        Computed as the total derivative of the objective expression through the
        solution map.  At a KKT point this equals the envelope theorem's
        ``∂L/∂p`` -- stationarity annihilates the indirect ``(∂f/∂x)·dx*/dp``
        term -- which is what
        :meth:`SolveResult.gradient <discopt.modeling.core.SolveResult.gradient>`
        returns; the test suite pins the two against each other.
        """
        if "dobj_dp" not in self._cache:
            objective = self.model._objective
            if objective is None:  # pragma: no cover - solved models have one
                raise ValueError("The model has no objective to differentiate.")
            self._cache["dobj_dp"] = np.atleast_1d(self._total_derivative(objective.expression))
        return self._cache["dobj_dp"]

    def _select_columns(self, wrt):
        """Map ``wrt`` onto column indices of :attr:`dx_dp`."""
        if wrt is None:
            return slice(None)
        from discopt.modeling.core import Parameter

        single = isinstance(wrt, (Parameter, str))
        items = [wrt] if single else list(wrt)
        index = {id(p): k for k, p in enumerate(self.parameters)}
        by_name = {p.name: k for k, p in enumerate(self.parameters)}
        cols = []
        for item in items:
            if isinstance(item, str):
                if item not in by_name:
                    raise KeyError(
                        f"{item!r} is not among the parameters this sensitivity was "
                        f"computed for: {self.param_names}"
                    )
                cols.append(by_name[item])
            elif isinstance(item, Parameter):
                if id(item) not in index:
                    raise ValueError(
                        f"Parameter {item.name!r} was not included in this "
                        f"sensitivity (computed for {self.param_names}). Re-run "
                        "sensitivity() with it in `wrt`."
                    )
                cols.append(index[id(item)])
            else:
                raise TypeError(
                    f"wrt entries must be Parameters or names, got {type(item).__name__}"
                )
        return cols[0] if single else cols

    def _total_derivative(self, target) -> np.ndarray:
        """``de/dp`` for a variable (a slice of dx_dp) or any expression (chain rule)."""
        from discopt.modeling.core import Expression, Variable

        if isinstance(target, str):
            resolved = next((v for v in self.model._variables if v.name == target), None)
            if resolved is None:
                names = [v.name for v in self.model._variables]
                raise KeyError(
                    f"{target!r} is not a variable of model {self.model.name!r}. "
                    f"Variables: {names}. To differentiate an expression, pass the "
                    "expression itself rather than a name."
                )
            target = resolved

        if isinstance(target, Variable):
            offset = 0
            for v in self.model._variables:
                if v is target:
                    rows = self.dx_dp[offset : offset + v.size]
                    return rows.reshape(v.shape + (self.n_parameters,))
                offset += v.size
            raise ValueError(
                f"Variable {target.name!r} does not belong to model {self.model.name!r}."
            )

        if not isinstance(target, Expression):
            raise TypeError(
                "d() takes a Variable, a variable name, or a model Expression, got "
                f"{type(target).__name__}."
            )
        return self._chain_rule(target)

    def _chain_rule(self, expr: "Expression") -> np.ndarray:
        """``∂e/∂p + (∂e/∂x)·dx*/dp``, both halves evaluated at the solution.

        The partials come from :func:`discopt.parametric.compile_expression`, which
        keeps Parameter values as a live argument rather than baking them in --
        without that the ``∂e/∂p`` term would silently be zero.
        """
        import jax

        from discopt.parametric import compile_expression

        fn = compile_expression(expr, self.model)
        x = np.asarray(self.x, dtype=np.float64)
        # `self.p_all`, not a fresh `flatten_params(self.model)`: the partials must
        # be taken at the same point `dx_dp` was, and a caller is free to have moved
        # `Parameter.value` since. Reading live values here would mix two points and
        # produce a derivative belonging to neither.
        p_all = np.asarray(self.p_all, dtype=np.float64)

        de_dx = np.asarray(jax.jacobian(fn, argnums=0)(x, p_all))
        de_dp_all = np.asarray(jax.jacobian(fn, argnums=1)(x, p_all))

        direct = de_dp_all[..., self._pflat_indices()]
        indirect = np.tensordot(de_dx, self.dx_dp, axes=([-1], [0]))
        return direct + indirect

    def _pflat_indices(self) -> list:
        """Positions of :attr:`parameters` inside the model's flat parameter vector."""
        from discopt._relax.differentiable import _get_param_slice

        idx = []
        for p in self.parameters:
            start, end = _get_param_slice(p, self.model)
            if end - start != 1:  # pragma: no cover - _resolve_parameters gates this
                raise ValueError(f"Parameter {p.name!r} is not scalar in the flat layout.")
            idx.append(start)
        return idx

    # ── prediction ───────────────────────────────────────────

    def _delta(self, values) -> np.ndarray:
        """Turn a mapping or sequence of new parameter values into ``Δp``."""
        from discopt.modeling.core import Parameter

        current = np.asarray(self.p0, dtype=np.float64)
        if isinstance(values, dict):
            new = current.copy()
            index = {id(p): k for k, p in enumerate(self.parameters)}
            by_name = {p.name: k for k, p in enumerate(self.parameters)}
            for key, val in values.items():
                if isinstance(key, Parameter):
                    if id(key) not in index:
                        raise ValueError(
                            f"Parameter {key.name!r} was not included in this "
                            f"sensitivity (computed for {self.param_names})."
                        )
                    new[index[id(key)]] = float(val)
                elif isinstance(key, str):
                    if key not in by_name:
                        raise KeyError(
                            f"{key!r} is not among this sensitivity's parameters: "
                            f"{self.param_names}"
                        )
                    new[by_name[key]] = float(val)
                else:
                    raise TypeError(
                        f"predict() keys must be Parameters or names, got {type(key).__name__}"
                    )
            return new - current

        seq = np.asarray(list(values), dtype=np.float64)
        if seq.size != len(self.parameters):
            raise ValueError(
                f"predict() got {seq.size} values for {len(self.parameters)} "
                f"parameters {self.param_names}. Pass a dict to set only some."
            )
        return seq - current

    def predict_flat(self, values, order: Optional[int] = None) -> np.ndarray:
        """Taylor prediction of ``x*`` at new parameter values, as a flat vector.

        Parameters
        ----------
        values : dict or sequence
            ``{parameter_or_name: new_value}`` (parameters left out keep their
            current value), or one value per parameter in column order.
        order : int, optional
            1 for ``x* + (dx*/dp)Δp``; 2 to add ``½ Δpᵀ (d²x*/dp²) Δp``, which
            requires this sensitivity to have been computed with ``order=2``.
            Defaults to whatever order is available.

        Returns
        -------
        numpy.ndarray
            Predicted flat solution.  Error is ``O(‖Δp‖²)`` at first order --
            and the prediction is only valid while the **active set** does not
            change; past a branch boundary it extrapolates the wrong piece.
        """
        use = self.order if order is None else order
        if use not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {use}")
        # Bind the second-order term to a local: it discharges the Optional once,
        # here, so the einsum below cannot be reached with `None` and the reader
        # (and the typechecker) can see that from one line rather than by pairing
        # a guard with a use several statements away.
        curvature = self.d2x_dp2 if use == 2 else None
        if use == 2 and curvature is None:
            raise ValueError(
                "Second-order prediction needs d2x_dp2; re-run with `sensitivity(..., order=2)`."
            )
        dp = self._delta(values)
        out = self.x + self.dx_dp @ dp
        if curvature is not None:
            out = out + 0.5 * np.einsum("ijk,j,k->i", curvature, dp, dp)
        return np.asarray(out, dtype=np.float64)

    def predict(self, values, order: Optional[int] = None) -> dict:
        """Taylor prediction of ``x*``, keyed by variable name like ``SolveResult.x``.

        See :meth:`predict_flat` for the arguments and the validity caveat.
        """
        return _unflatten(self.model, self.predict_flat(values, order=order))

    # ── composition ──────────────────────────────────────────

    def as_layer(self):
        """The same solution map as a differentiable JAX callable ``phi(p) -> x*``.

        For dropping this model into a larger JAX pipeline (``jax.grad``,
        ``jax.jit``, ``jax.vmap``) instead of reading matrices out here.  The rule
        is :func:`jax.lax.custom_root` over the KKT residual, so forward mode and
        second derivatives both work::

            phi = s.as_layer()
            dx_dp = jax.jacobian(phi)(jnp.array([2.0, 3.0]))   # == s.dx_dp

        Returns
        -------
        callable
            ``phi(p)`` taking a length-``n_parameters`` array, in the column order
            of :attr:`parameters`.
        """
        from discopt.modeling.argmin import _build_layer

        return _build_layer(
            self.model,
            self.parameters,
            verify_minimizer=False,
            require_min=False,
        )

    # ── display ──────────────────────────────────────────────

    def summary(self, max_rows: int = 30) -> str:
        """A formatted ``dx*/dp`` table, one row per scalar variable entry."""
        labels = []
        for v in self.model._variables:
            if v.shape == ():
                labels.append(v.name)
            else:
                for flat in range(v.size):
                    idx = np.unravel_index(flat, v.shape)
                    labels.append(f"{v.name}[{','.join(str(int(i)) for i in idx)}]")

        width = max([len(s) for s in labels] + [8])
        header = "variable".ljust(width) + "".join(f"{n:>14s}" for n in self.param_names)
        lines = [
            f"Sensitivity  status={self.status}  objective={self.objective:.6g}  "
            f"method={self.method}",
            header,
            "-" * len(header),
        ]
        shown = min(len(labels), max_rows)
        for i in range(shown):
            row = "".join(f"{self.dx_dp[i, k]:>14.6g}" for k in range(self.n_parameters))
            flag = "  (at bound)" if bool(self.at_bound[i]) else ""
            lines.append(labels[i].ljust(width) + row + flag)
        if shown < len(labels):
            lines.append(f"... {len(labels) - shown} more rows")
        lines.append(
            f"active constraint rows: {int(np.count_nonzero(self.active))}/{self.active.size}"
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────


def _reference_point(model: "Model", at):
    """Resolve the reference solution ``sensitivity()`` starts from and checks against.

    Returns ``(x0_flat, reference_objective)``, either half possibly ``None``.

    ``at`` may be a :class:`~discopt.modeling.core.SolveResult`, a ``{name: value}``
    / ``{Variable: value}`` mapping, or a flat array.  ``None`` falls back to the
    model's own last successful :meth:`Model.solve` result (#1313) -- the point of
    the fix: ``m.solve(); m.sensitivity()`` must describe the solution ``solve``
    just certified, not whatever basin a fixed midpoint start happens to reach.
    """
    from discopt.modeling.core import SolveResult

    if at is None:
        at = getattr(model, "_last_solve_result", None)
        if at is None:
            return None, None

    if isinstance(at, SolveResult):
        from discopt.warm_start import primal_point_from_result

        x0 = primal_point_from_result(model, at)
        ref = None if at.objective is None else float(at.objective)
        return x0, ref

    if isinstance(at, dict):
        from discopt.modeling.core import Variable
        from discopt.warm_start import validate_initial_solution

        by_var = {}
        by_name = {v.name: v for v in model._variables}
        for key, value in at.items():
            if isinstance(key, Variable):
                by_var[key] = value
            elif isinstance(key, str):
                if key not in by_name:
                    raise ValueError(
                        f"sensitivity(): at= names variable {key!r}, which this model "
                        "does not declare."
                    )
                by_var[by_name[key]] = value
            else:
                raise TypeError(
                    "sensitivity(): at= keys must be Variable objects or variable "
                    f"names, got {type(key).__name__}"
                )
        return validate_initial_solution(model, by_var), None

    x0 = np.asarray(at, dtype=np.float64).reshape(-1)
    n = int(sum(int(v.size) for v in model._variables))
    if x0.size != n:
        raise ValueError(
            f"sensitivity(): at= has {x0.size} entries but the model has {n} scalar variables."
        )
    return x0, None


def sensitivity(
    model: "Model",
    wrt=None,
    *,
    order: int = 1,
    method: str = "exact",
    options: Optional[dict] = None,
    at=None,
) -> Sensitivity:
    """Solve ``model`` and return every derivative of that solution.

    The unified entry point for model derivatives.  One solve, one object: the
    solution, ``dx*/dp``, ``dλ*/dp``, optionally ``d²x*/dp²``, the total
    derivative of any expression through the solution map, first-order
    prediction, and the JAX layer form.

    Parameters
    ----------
    model : Model
        A continuous model with an objective and at least one
        :class:`~discopt.modeling.core.Parameter`.  Integer or binary variables
        are refused: the derivatives come from KKT stationarity, which does not
        characterize an integer optimum.
    wrt : Parameter, str, or sequence, optional
        Which parameters to differentiate with respect to, as objects or names.
        Defaults to every Parameter the model declares, in declaration order.
    order : int, default 1
        1 for first derivatives; 2 also fills :attr:`Sensitivity.d2x_dp2`.
    method : {"exact", "fd"}, default "exact"
        ``"exact"`` differentiates the compiled model symbolically -- no
        truncation error, no step size.  ``"fd"`` re-forms the right-hand side by
        central differences and exists as an independent cross-check of the exact
        path; it cannot produce second order.
    options : dict, optional
        POUNCE options for the solve (e.g. ``{"tol": 1e-10}``).
    at : SolveResult, dict, or array, optional
        The solution the derivatives are taken at: the inner NLP starts there and
        the KKT point it returns is cross-checked against it.  Defaults to the
        model's own last successful :meth:`Model.solve` result, so
        ``m.solve(); m.sensitivity()`` describes *that* solution rather than
        whichever local one a fixed start point reaches (#1313).  Pass an explicit
        value to differentiate at a different stationary point.  A model that has
        never been solved has no reference: the inner solve then starts from the
        midpoint of the clipped box as before, and
        :attr:`Sensitivity.matches_reference` is ``None``.

    Returns
    -------
    Sensitivity

    Raises
    ------
    ValueError
        If the model declares no parameters, if ``wrt`` names a non-scalar or
        foreign parameter, or if the model has integer/binary variables.
    RuntimeError
        If the solve fails or its KKT point does not check out.

    Warns
    -----
    RuntimeWarning
        When the KKT point the derivatives belong to has a different objective
        than the reference solution -- on a nonconvex model, a different basin.
        :attr:`Sensitivity.status` stays POUNCE's ``"optimal"`` in that case,
        because the local solve did converge; ``matches_reference=False`` is what
        says the point is not the one the model was solved to.

    Notes
    -----
    Parameter values are snapshotted and restored, so the call leaves the model
    exactly as it found it -- a sensitivity that silently moved ``p.value`` would
    make every subsequent solve in the session answer a different question.

    ``status`` is the local NLP's termination status, not a global-optimality
    certificate: unlike :attr:`SolveResult.status
    <discopt.modeling.core.SolveResult.status>`, ``"optimal"`` here means only
    that the inner solve converged to a KKT point.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> m = dm.Model("plant")
    >>> cost = m.parameter("cost", value=2.0)
    >>> x = m.continuous("x", lb=0.0, ub=10.0)
    >>> m.minimize((x - 3.0) ** 2 + cost * x)
    >>> s = m.sensitivity()                      # doctest: +SKIP
    >>> s.d(x, cost)                             # doctest: +SKIP
    -0.5
    """
    params = _resolve_parameters(model, wrt)

    from discopt.modeling.core import ObjectiveSense, objective_sense_sign
    from discopt.solvers.sipopt import pounce_sensitivity

    x0, reference_objective = _reference_point(model, at)

    saved = [np.array(p.value, copy=True) for p in params]
    from discopt.parametric import flatten_params

    p_all0 = np.asarray(flatten_params(model), dtype=np.float64)
    try:
        raw = pounce_sensitivity(model, params, options=options, order=order, method=method, x0=x0)
    finally:
        # The forward solve writes trial values into `p.value` on every call; a
        # failure partway through must not leave the model describing a problem
        # nobody asked about.
        for p, value in zip(params, saved):
            p.value = value

    x = np.asarray(raw.x_star, dtype=np.float64)
    obj = objective_sense_sign(model) * float(raw.objective)

    # --- #1313: the point the derivatives belong to, checked against the one the
    # model was solved to. A nonconvex model has many KKT points and POUNCE
    # reports ``status="optimal"`` at any of them; without this check a caller
    # doing ``m.solve(); m.sensitivity()`` could be handed derivatives from an
    # entirely different basin under the same word ``optimal`` that
    # ``SolveResult.status`` uses for a certified global optimum.
    matches_reference: Optional[bool] = None
    if reference_objective is not None and np.isfinite(obj):
        tol = 1e-6 + 1e-6 * abs(reference_objective)
        matches_reference = bool(abs(obj - reference_objective) <= tol)
        if not matches_reference:
            sense = model._objective.sense if model._objective is not None else None
            worse = (
                obj > reference_objective
                if sense != ObjectiveSense.MAXIMIZE
                else obj < reference_objective
            )
            warnings.warn(
                "sensitivity(): the KKT point the derivatives were taken at has "
                f"objective {obj:.6g}, but the solution this was checked against has "
                f"{reference_objective:.6g}"
                + (" (a strictly worse point)" if worse else "")
                + ". The derivatives describe that other stationary point, not the "
                "reference solution -- on a nonconvex model, a different basin. "
                "status='optimal' below is the local NLP's, not a global certificate. "
                "Pass at= to pin the point explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )

    return Sensitivity(
        model=model,
        parameters=params,
        x=x,
        x_dict=_unflatten(model, x),
        # ``raw.objective`` is POUNCE's value for the internal minimization form
        # (#1299): a MAXIMIZE model was solved as ``-f``, so undo the flip before
        # reporting it. ``dobj_dp`` below needs no such correction -- it
        # differentiates the model's own objective expression.
        objective=obj,
        status=str(raw.status),
        multipliers=np.asarray(raw.lambda_star, dtype=np.float64),
        dx_dp=np.asarray(raw.dx_dp, dtype=np.float64),
        dlambda_dp=np.asarray(raw.dlambda_dp, dtype=np.float64),
        d2x_dp2=None if raw.d2x_dp2 is None else np.asarray(raw.d2x_dp2, dtype=np.float64),
        active=np.asarray(raw.active, dtype=bool),
        at_bound=np.asarray(raw.at_bound, dtype=bool),
        method=str(raw.method),
        order=int(order),
        p0=np.array([float(np.asarray(v)) for v in saved], dtype=np.float64),
        p_all=p_all0,
        reference_objective=reference_objective,
        matches_reference=matches_reference,
    )
