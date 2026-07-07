"""Orthogonal collocation on finite elements for ODE/DAE transcription.

This module provides :class:`DAEBuilder`, which transcribes ordinary
differential equations (ODEs), index-1 differential-algebraic equations
(DAEs), and second-order ODEs into algebraic constraints compatible with
the discopt modeling API.

The user defines state, algebraic, and control variables, supplies an ODE/DAE
right-hand side as a Python callable that builds discopt expressions, and
calls :meth:`DAEBuilder.discretize` to generate all collocation and continuity
constraints automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from discopt.dae.polynomials import collocation_matrix
from discopt.modeling.core import Constant, MatMulExpression


@dataclass
class ContinuousSet:
    """A discretized time (or spatial) domain for dynamic optimization.

    Parameters
    ----------
    name : str
        Name of the independent variable (e.g. ``"t"``).
    bounds : tuple of float
        ``(t0, tf)`` domain boundaries.
    nfe : int
        Number of finite elements.
    ncp : int
        Number of collocation points per element (1-5).
    scheme : str
        ``"radau"`` or ``"legendre"``.
    element_boundaries : np.ndarray, optional
        Explicit element boundary points. When provided, ``nfe`` is inferred
        as ``len(element_boundaries) - 1`` and ``bounds`` is inferred from
        the first and last values. Must be sorted and strictly increasing.
    """

    name: str
    bounds: tuple[float, float]
    nfe: int
    ncp: int = 3
    scheme: str = "radau"
    element_boundaries: np.ndarray | None = None

    def __post_init__(self):
        if self.element_boundaries is not None:
            eb = np.asarray(self.element_boundaries, dtype=np.float64)
            if eb.ndim != 1 or len(eb) < 2:
                raise ValueError("element_boundaries must be a 1-D array with at least 2 entries")
            if not np.all(np.diff(eb) > 0):
                raise ValueError("element_boundaries must be sorted and strictly increasing")
            self.element_boundaries = eb
            self.nfe = len(eb) - 1
            self.bounds = (float(eb[0]), float(eb[-1]))


@dataclass
class StateVar:
    """Descriptor for a differential state variable."""

    name: str
    n_components: int = 1
    bounds: tuple[float, float] = (-1e20, 1e20)
    initial: float | np.ndarray | None = None


@dataclass
class AlgebraicVar:
    """Descriptor for an algebraic variable (no time derivative)."""

    name: str
    n_components: int = 1
    bounds: tuple[float, float] = (-1e20, 1e20)


@dataclass
class ControlVar:
    """Descriptor for a piecewise-constant control variable."""

    name: str
    n_components: int = 1
    bounds: tuple[float, float] = (-1e20, 1e20)


@dataclass
class _SecondOrderInfo:
    """Internal bookkeeping for a second-order state."""

    position_name: str
    velocity_name: str
    initial_velocity: float | np.ndarray | None = None


def validate_ode_rhs_keys(deriv_keys, states, *, builder: str) -> None:
    """Require the ODE RHS to supply a derivative for *exactly* the states.

    Shared by the collocation and finite-difference transcribers (C3). A missing
    key silently drops a differential state's dynamics, leaving it unconstrained
    (an under-determined transcription the solver will happily "optimize"); an
    unknown key is a user typo the caller expects to have taken effect. Both are
    refused loudly rather than approximated silently — per the development
    philosophy, a wrong certificate is worse than a loud error.

    Parameters
    ----------
    deriv_keys : iterable of str
        Keys of the dict returned by the user's RHS callable.
    states : iterable
        The declared differential states (each with a ``.name``).
    builder : str
        Human-readable transcriber name, for the error message.
    """
    expected = {sv.name for sv in states}
    got = set(deriv_keys)
    missing = expected - got
    unknown = got - expected
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"no derivative supplied for state(s) {sorted(missing)}")
        if unknown:
            parts.append(f"unknown key(s) {sorted(unknown)} match no declared state")
        raise ValueError(
            f"{builder} RHS keys do not match the declared states "
            f"(expected exactly {sorted(expected)}): " + "; ".join(parts) + "."
        )


class DAEBuilder:
    """Transcribe an ODE/DAE system into algebraic constraints via collocation.

    Parameters
    ----------
    model : discopt.Model
        The optimization model to add variables and constraints to.
    continuous_set : ContinuousSet
        The time domain and discretization parameters.
    """

    def __init__(self, model, continuous_set: ContinuousSet):
        self._model = model
        self._cs = continuous_set
        self._states: list[StateVar] = []
        self._algebraics: list[AlgebraicVar] = []
        self._controls: list[ControlVar] = []
        self._ode_rhs: Callable | None = None
        self._alg_rhs: Callable | None = None
        self._second_order_rhs: Callable | None = None
        self._second_order_info: list[_SecondOrderInfo] = []
        self._vars: dict[str, Any] = {}  # name -> Variable
        self._discretized = False

        # Precompute collocation data
        cs = self._cs
        self._A, self._w = collocation_matrix(cs.ncp, cs.scheme)
        if cs.element_boundaries is not None:
            self._h_vec = np.diff(cs.element_boundaries)
        else:
            h_uniform = (cs.bounds[1] - cs.bounds[0]) / cs.nfe
            self._h_vec = np.full(cs.nfe, h_uniform)

    def add_state(
        self,
        name: str,
        n_components: int = 1,
        bounds: tuple[float, float] = (-1e20, 1e20),
        initial: float | np.ndarray | None = None,
    ) -> StateVar:
        """Declare a first-order differential state variable.

        Parameters
        ----------
        name : str
            Variable name.
        n_components : int
            Number of components (for vector-valued states, e.g. spatial grid).
        bounds : tuple of float
            ``(lb, ub)`` bounds on the state.
        initial : float or np.ndarray, optional
            Initial condition at ``t0``. If given, the state is fixed at this
            value at the first time point.

        Returns
        -------
        StateVar
        """
        sv = StateVar(name, n_components, bounds, initial)
        self._states.append(sv)
        return sv

    def add_second_order_state(
        self,
        name: str,
        velocity_name: str | None = None,
        n_components: int = 1,
        bounds: tuple[float, float] = (-1e20, 1e20),
        initial: float | np.ndarray | None = None,
        initial_velocity: float | np.ndarray | None = None,
        velocity_bounds: tuple[float, float] = (-1e20, 1e20),
    ) -> StateVar:
        """Declare a second-order state (position + velocity).

        Internally creates two first-order states: the position ``name`` and
        velocity ``velocity_name`` (default ``"d{name}_dt"``). The coupling
        ``d(position)/dt = velocity`` is added automatically during
        discretization.

        Parameters
        ----------
        name : str
            Position variable name.
        velocity_name : str, optional
            Velocity variable name. Defaults to ``"d{name}_dt"``.
        n_components : int
            Number of components.
        bounds : tuple of float
            Bounds on the position variable.
        initial : float or np.ndarray, optional
            Initial position.
        initial_velocity : float or np.ndarray, optional
            Initial velocity.

        Returns
        -------
        StateVar
            The position state variable descriptor.
        """
        if velocity_name is None:
            velocity_name = f"d{name}_dt"

        pos_sv = self.add_state(name, n_components, bounds, initial)
        self.add_state(velocity_name, n_components, velocity_bounds, initial_velocity)

        self._second_order_info.append(_SecondOrderInfo(name, velocity_name, initial_velocity))
        return pos_sv

    def add_algebraic(
        self,
        name: str,
        n_components: int = 1,
        bounds: tuple[float, float] = (-1e20, 1e20),
    ) -> AlgebraicVar:
        """Declare an algebraic variable (no time derivative).

        Algebraic variables exist only at collocation points, not at element
        boundaries.

        Parameters
        ----------
        name : str
            Variable name.
        n_components : int
            Number of components.
        bounds : tuple of float
            ``(lb, ub)`` bounds.

        Returns
        -------
        AlgebraicVar
        """
        av = AlgebraicVar(name, n_components, bounds)
        self._algebraics.append(av)
        return av

    def add_control(
        self,
        name: str,
        n_components: int = 1,
        bounds: tuple[float, float] = (-1e20, 1e20),
    ) -> ControlVar:
        """Declare a piecewise-constant control variable.

        Controls are constant within each finite element, shape ``(nfe,)`` or
        ``(nfe, n_components)``.

        Parameters
        ----------
        name : str
            Variable name.
        n_components : int
            Number of components.
        bounds : tuple of float
            ``(lb, ub)`` bounds.

        Returns
        -------
        ControlVar
        """
        cv = ControlVar(name, n_components, bounds)
        self._controls.append(cv)
        return cv

    def set_ode(self, rhs: Callable) -> None:
        """Set the first-order ODE right-hand side.

        Parameters
        ----------
        rhs : callable
            ``rhs(t, states, algebraics, controls) -> dict[str, Expression]``
            where each key is a state name and the value is its time derivative
            expression. ``states``, ``algebraics``, and ``controls`` are dicts
            mapping variable names to discopt expressions at the current
            collocation point.
        """
        self._ode_rhs = rhs

    def set_algebraic(self, rhs: Callable) -> None:
        """Set the algebraic equations (for index-1 DAEs).

        Parameters
        ----------
        rhs : callable
            ``rhs(t, states, algebraics, controls) -> dict[str, Expression]``
            where each value is constrained ``== 0`` at every collocation point.
        """
        self._alg_rhs = rhs

    def set_second_order_ode(self, rhs: Callable) -> None:
        """Set the second-order ODE right-hand side (acceleration).

        Parameters
        ----------
        rhs : callable
            ``rhs(t, positions, velocities, algebraics, controls) -> dict``
            where keys are position state names and values are acceleration
            expressions. The velocity coupling ``dx/dt = v`` is automatic.
        """
        self._second_order_rhs = rhs

    def discretize(self) -> dict[str, object]:
        """Generate all variables and constraints.

        Creates discopt variables for all states, algebraics, and controls,
        then adds collocation and continuity constraints to the model.

        Returns
        -------
        dict[str, Variable]
            All created discopt variables, keyed by name.
        """
        if self._discretized:
            raise RuntimeError("DAEBuilder.discretize() has already been called")
        self._discretized = True

        m = self._model
        cs = self._cs

        # Create state variables: shape (nfe, ncp+1) or (nfe, ncp+1, n_comp)
        for sv in self._states:
            shape: tuple[int, ...] = (cs.nfe, cs.ncp + 1)
            if sv.n_components > 1:
                shape = (cs.nfe, cs.ncp + 1, sv.n_components)

            lb = np.full(shape, sv.bounds[0], dtype=np.float64)
            ub = np.full(shape, sv.bounds[1], dtype=np.float64)
            if sv.initial is not None:
                if sv.n_components == 1:
                    lb[0, 0] = sv.initial
                    ub[0, 0] = sv.initial
                else:
                    lb[0, 0, ...] = sv.initial
                    ub[0, 0, ...] = sv.initial

            self._vars[sv.name] = m.continuous(f"{cs.name}_{sv.name}", shape=shape, lb=lb, ub=ub)

        # Create algebraic variables: shape (nfe, ncp) or (nfe, ncp, n_comp)
        for av in self._algebraics:
            ashape: tuple[int, ...] = (cs.nfe, cs.ncp)
            if av.n_components > 1:
                ashape = (cs.nfe, cs.ncp, av.n_components)

            self._vars[av.name] = m.continuous(
                f"{cs.name}_{av.name}",
                shape=ashape,
                lb=av.bounds[0],
                ub=av.bounds[1],
            )

        # Create control variables: shape (nfe,) or (nfe, n_comp)
        for cv in self._controls:
            cshape: tuple[int, ...] = (cs.nfe,)
            if cv.n_components > 1:
                cshape = (cs.nfe, cv.n_components)

            self._vars[cv.name] = m.continuous(
                f"{cs.name}_{cv.name}",
                shape=cshape,
                lb=cv.bounds[0],
                ub=cv.bounds[1],
            )

        # Build second-order ODE wrapper if needed
        if self._second_order_rhs is not None:
            self._build_second_order_wrapper()

        # Generate collocation constraints
        self._add_collocation_constraints()

        # Generate algebraic constraints
        if self._alg_rhs is not None:
            self._add_algebraic_constraints()

        # Generate continuity constraints (for Legendre scheme)
        if cs.scheme == "legendre":
            self._add_continuity_constraints_legendre()
        else:
            # Radau: continuity via last collocation point = next element start
            self._add_continuity_constraints_radau()

        return dict(self._vars)

    def _build_second_order_wrapper(self):
        """Wrap the second-order ODE RHS into a first-order ODE RHS."""
        accel_rhs = self._second_order_rhs
        so_info = {info.position_name: info for info in self._second_order_info}
        existing_ode = self._ode_rhs

        def combined_rhs(t, states, algebraics, controls):
            derivs = {}

            # Velocity coupling: d(position)/dt = velocity
            for info in self._second_order_info:
                derivs[info.position_name] = states[info.velocity_name]

            # Acceleration from user-supplied RHS
            positions = {
                info.position_name: states[info.position_name] for info in self._second_order_info
            }
            velocities = {
                info.velocity_name: states[info.velocity_name] for info in self._second_order_info
            }
            accels = accel_rhs(t, positions, velocities, algebraics, controls)
            for pos_name, accel_expr in accels.items():
                vel_name = so_info[pos_name].velocity_name
                derivs[vel_name] = accel_expr

            # Include any existing first-order ODE terms
            if existing_ode is not None:
                first_order = existing_ode(t, states, algebraics, controls)
                derivs.update(first_order)

            return derivs

        self._ode_rhs = combined_rhs

    def _build_vec_dicts(self):
        """Construct state/algebraic/control dicts with vector-shaped entries.

        Scalar states (``n_components == 1``) become shape ``(nfe, ncp)``
        IndexExpressions covering all finite elements and collocation points
        at once. Scalar controls are reshaped to ``(nfe, 1)`` so element-wise
        multiplication with ``(nfe, ncp)`` state/algebraic expressions
        broadcasts correctly. Multi-component variables are represented as
        Python lists of vector-shaped IndexExpressions, preserving the
        per-component access pattern the scalar path used.
        """
        states_vec: dict[str, Any] = {}
        for sv in self._states:
            var = self._vars[sv.name]
            if sv.n_components == 1:
                states_vec[sv.name] = var[:, 1:]  # (nfe, ncp)
            else:
                states_vec[sv.name] = [var[:, 1:, c] for c in range(sv.n_components)]

        alg_vec: dict[str, Any] = {}
        for av in self._algebraics:
            var = self._vars[av.name]
            if av.n_components == 1:
                alg_vec[av.name] = var[:, :]  # (nfe, ncp)
            else:
                alg_vec[av.name] = [var[:, :, c] for c in range(av.n_components)]

        ctrl_vec: dict[str, Any] = {}
        for cv in self._controls:
            var = self._vars[cv.name]
            if cv.n_components == 1:
                ctrl_vec[cv.name] = var[:, None]  # (nfe, 1) broadcasts over ncp
            else:
                ctrl_vec[cv.name] = [var[:, c, None] for c in range(cv.n_components)]

        return states_vec, alg_vec, ctrl_vec

    def _add_collocation_constraints(self):
        """Add collocation equations for all finite elements and states.

        Emits one vector-valued constraint per (state, component) of shape
        ``(nfe, ncp)``. The left-hand side ``sum_k A[j, k] * x[i, k]`` is
        built as a single ``MatMulExpression`` (``x @ A.T``), and the RHS
        ``h_i * f(...)`` is built by broadcasting ``Constant(h_vec)`` against
        the vectorized derivative expression returned from ``rhs_fn``. The
        ``rhs_fn`` callable is invoked exactly once with vector-shaped
        state/algebraic/control dicts rather than per ``(i, j)``.
        """
        m = self._model
        cs = self._cs
        A = self._A
        rhs_fn = self._ode_rhs

        # C3(a): second-order state(s) declared but no acceleration RHS set. The
        # wrapper that would populate `_ode_rhs` was never built, so dynamics
        # would be silently skipped. Refuse loudly.
        if self._second_order_info and self._second_order_rhs is None:
            raise RuntimeError(
                "Second-order state(s) "
                f"{[i.position_name for i in self._second_order_info]} were declared with "
                "add_second_order_state() but no second-order ODE was set. "
                "Call set_second_order_ode()."
            )

        if rhs_fn is None:
            raise RuntimeError("No ODE RHS set. Call set_ode() or set_second_order_ode().")

        tp = self._element_points()  # (nfe, ncp+1)
        t_arg = Constant(np.asarray(tp[:, 1:], dtype=np.float64))  # (nfe, ncp)
        h_col = Constant(np.asarray(self._h_vec, dtype=np.float64).reshape(-1, 1))  # (nfe, 1)
        A_T = Constant(np.asarray(A, dtype=np.float64).T)  # (ncp+1, ncp)

        states_vec, alg_vec, ctrl_vec = self._build_vec_dicts()
        derivs = rhs_fn(t_arg, states_vec, alg_vec, ctrl_vec)
        # C3(b): every differential state must get a derivative, and no stray keys.
        validate_ode_rhs_keys(derivs.keys(), self._states, builder="collocation ODE")

        constraints = []
        for sv in self._states:
            var = self._vars[sv.name]
            if sv.n_components == 1:
                # LHS shape (nfe, ncp) = var @ A.T
                lhs = MatMulExpression(var, A_T)
                rhs_expr = h_col * derivs[sv.name]  # (nfe, 1) * (nfe, ncp) → (nfe, ncp)
                constraints.append(lhs == rhs_expr)
            else:
                for c in range(sv.n_components):
                    lhs = MatMulExpression(var[:, :, c], A_T)
                    rhs_expr = h_col * derivs[sv.name][c]
                    constraints.append(lhs == rhs_expr)

        if constraints:
            m.subject_to(constraints, name=f"{cs.name}_collocation")

    def _add_algebraic_constraints(self):
        """Add algebraic equations at all collocation points.

        Emits one vector-valued constraint of shape ``(nfe, ncp)`` per
        algebraic residual returned by the user-supplied callable.
        """
        m = self._model
        cs = self._cs
        tp = self._element_points()
        t_arg = Constant(np.asarray(tp[:, 1:], dtype=np.float64))

        states_vec, alg_vec, ctrl_vec = self._build_vec_dicts()
        residuals = self._alg_rhs(t_arg, states_vec, alg_vec, ctrl_vec)

        constraints = [expr == 0 for expr in residuals.values()]
        if constraints:
            m.subject_to(constraints, name=f"{cs.name}_algebraic")

    def _add_continuity_constraints_radau(self):
        """Add inter-element continuity for Radau scheme.

        For Radau, the last collocation point is at tau=1, so
        ``x[i+1, 0] = x[i, ncp]``. Emitted as one vector constraint per
        state (or per state-component for vector-valued states).
        """
        m = self._model
        cs = self._cs
        if cs.nfe <= 1:
            return
        constraints = []

        for sv in self._states:
            var = self._vars[sv.name]
            if sv.n_components == 1:
                constraints.append(var[1:, 0] == var[:-1, cs.ncp])
            else:
                for c in range(sv.n_components):
                    constraints.append(var[1:, 0, c] == var[:-1, cs.ncp, c])

        if constraints:
            m.subject_to(constraints, name=f"{cs.name}_continuity")

    def _add_continuity_constraints_legendre(self):
        """Add inter-element continuity for Legendre scheme.

        Legendre points don't include tau=1, so we interpolate
        ``x[i+1, 0] = sum_k w[k] * x[i, k]``. Built as a single
        ``MatMulExpression`` per (state, component) using ``var @ w``.
        """
        m = self._model
        cs = self._cs
        if cs.nfe <= 1:
            return
        w = np.asarray(self._w, dtype=np.float64).reshape(-1, 1)  # (ncp+1, 1)
        w_const = Constant(w)
        constraints = []

        for sv in self._states:
            var = self._vars[sv.name]
            if sv.n_components == 1:
                # var[:-1, :] has shape (nfe-1, ncp+1); @ w → (nfe-1, 1)
                interp = MatMulExpression(var[:-1, :], w_const)
                # var[1:, 0:1] has shape (nfe-1, 1) to match
                constraints.append(var[1:, 0:1] == interp)
            else:
                for c in range(sv.n_components):
                    interp = MatMulExpression(var[:-1, :, c], w_const)
                    constraints.append(var[1:, 0:1, c] == interp)

        if constraints:
            m.subject_to(constraints, name=f"{cs.name}_continuity")

    def get_state(self, name: str):
        """Get the discopt Variable for a state by name.

        Parameters
        ----------
        name : str
            State variable name as passed to :meth:`add_state`.

        Returns
        -------
        Variable
        """
        if name not in self._vars:
            raise KeyError(f"Unknown variable {name!r}")
        return self._vars[name]

    def time_points(self) -> np.ndarray:
        """Return all time points as a flat array.

        Returns
        -------
        np.ndarray
            Unique, sorted time points across all elements.
        """
        tp = self._element_points()  # (nfe, ncp+1)
        flat = tp.ravel()
        return np.unique(flat)

    def integral(self, integrand: Callable) -> object:
        """Build a quadrature expression for an integral objective.

        Uses the collocation quadrature weights to approximate:
            integral_{t0}^{tf} f(t, x, z, u) dt

        Parameters
        ----------
        integrand : callable
            ``integrand(t, states, algebraics, controls) -> Expression``
            evaluated at each collocation point.

        Returns
        -------
        Expression
            A single expression suitable for use in ``m.minimize()``.
        """
        cs = self._cs
        tp = self._element_points()

        # Quadrature weights from the collocation scheme
        qw = self._quadrature_weights()

        terms = []
        for i in range(cs.nfe):
            h_i = float(self._h_vec[i])
            for j in range(cs.ncp):
                t_val = tp[i, j + 1]
                states_ij = self._state_dict_at(i, j + 1)
                alg_ij = self._algebraic_dict_at(i, j)
                ctrl_ij = self._control_dict_at(i)
                f_val = integrand(t_val, states_ij, alg_ij, ctrl_ij)
                terms.append(h_i * float(qw[j]) * f_val)

        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    def extract_solution(self, result, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series from a solve result.

        Parameters
        ----------
        result : SolveResult
            Result from ``model.solve()``.
        name : str
            State variable name.

        Returns
        -------
        t : np.ndarray
            Time points.
        x : np.ndarray
            State values at time points.
        """
        var = self._vars[name]
        val = result.value(var)
        tp = self._element_points()

        # Flatten element structure
        t_list = []
        x_list = []
        for i in range(self._cs.nfe):
            for k in range(self._cs.ncp + 1):
                if i > 0 and k == 0:
                    continue  # skip duplicate element boundary
                t_list.append(tp[i, k])
                if val.ndim == 2:
                    x_list.append(val[i, k])
                else:
                    x_list.append(val[i, k, ...])

        return np.array(t_list), np.array(x_list)

    def state_at(
        self,
        state_name: str,
        t: float,
        component: int | None = None,
    ) -> Any:
        """Return an expression for a state evaluated at an arbitrary time ``t``.

        The state is reconstructed by evaluating the collocation polynomial of
        the finite element that contains ``t``:

            ``x(t) = sum_k  L_k(t) * var[elem, k]``

        where ``L_k`` are the Lagrange basis polynomials through the element's
        node times. This is exact to the collocation order (no node-spacing
        error), making it the right primitive for observation operators, event
        constraints, path outputs, and time-accurate measurement fitting.

        Must be called after :meth:`discretize`.

        Parameters
        ----------
        state_name : str
            Name of the state variable.
        t : float
            Time at which to evaluate the state. Values outside the domain are
            clamped to the nearest element (the polynomial then extrapolates).
        component : int, optional
            For vector-valued states, which component to evaluate.

        Returns
        -------
        Expression
            A discopt expression for the interpolated state value.
        """
        if not self._discretized:
            raise RuntimeError("Call discretize() before state_at()")
        if state_name not in self._vars:
            raise KeyError(f"Unknown state {state_name!r}")

        from discopt.dae.polynomials import lagrange_basis

        t = float(t)
        var = self._vars[state_name]
        tp = self._element_points()  # (nfe, ncp+1)
        elem = self._locate_element(t)
        nodes = tp[elem]  # (ncp+1,) node times of the containing element

        expr = None
        for k in range(self._cs.ncp + 1):
            coeff = float(lagrange_basis(nodes, t, k))
            x_kc = var[elem, k, component] if component is not None else var[elem, k]
            term = coeff * x_kc
            expr = term if expr is None else expr + term
        return expr

    def least_squares(
        self,
        state_name: str,
        t_data: np.ndarray,
        y_data: np.ndarray,
        component: int | None = None,
        interpolate: bool = True,
    ) -> object:
        """Build a sum-of-squared-residuals expression for parameter estimation.

        Returns ``sum((x(t_data[i]) - y_data[i])^2)`` as a discopt expression
        suitable for ``m.minimize()``.

        Must be called after :meth:`discretize`.

        Parameters
        ----------
        state_name : str
            Name of the state variable to fit.
        t_data : np.ndarray
            Measurement times, shape ``(n_obs,)``.
        y_data : np.ndarray
            Observed values, shape ``(n_obs,)``.
        component : int, optional
            For vector-valued states, which component to fit.
        interpolate : bool, default True
            If ``True`` (default), evaluate the model at the exact measurement
            time by interpolating the element's collocation polynomial (see
            :meth:`state_at`). This is exact to the collocation order and
            removes the time-misalignment bias that arises when measurements do
            not coincide with collocation nodes. If ``False``, fall back to the
            legacy behavior of snapping each measurement to the nearest
            collocation node (kept for backwards compatibility).

        Returns
        -------
        Expression
        """
        if not self._discretized:
            raise RuntimeError("Call discretize() before least_squares()")
        if state_name not in self._vars:
            raise KeyError(f"Unknown state {state_name!r}")

        t_data = np.asarray(t_data, dtype=np.float64)
        y_data = np.asarray(y_data, dtype=np.float64)

        var = self._vars[state_name]

        terms = []
        if interpolate:
            for i in range(len(t_data)):
                x_expr = self.state_at(state_name, float(t_data[i]), component)
                terms.append((x_expr - float(y_data[i])) ** 2)
        else:
            tp = self._element_points()  # (nfe, ncp+1)
            tp_flat = tp.ravel()
            for i in range(len(t_data)):
                idx = int(np.argmin(np.abs(tp_flat - t_data[i])))
                elem = idx // (self._cs.ncp + 1)
                node = idx % (self._cs.ncp + 1)
                if component is not None:
                    x_expr = var[elem, node, component]
                else:
                    x_expr = var[elem, node]
                terms.append((x_expr - float(y_data[i])) ** 2)

        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    # ── Internal helpers ──

    def _element_boundaries(self) -> np.ndarray:
        """Return element boundary times, shape (nfe + 1,)."""
        cs = self._cs
        if cs.element_boundaries is not None:
            return np.asarray(cs.element_boundaries, dtype=np.float64)
        return np.linspace(cs.bounds[0], cs.bounds[1], cs.nfe + 1)

    def _locate_element(self, t: float) -> int:
        """Return the index of the finite element that contains time ``t``.

        Times outside the domain clamp to the first/last element. Times on an
        interior boundary resolve to the element to their right.
        """
        eb = self._element_boundaries()
        idx = int(np.searchsorted(eb, float(t), side="right") - 1)
        return max(0, min(idx, self._cs.nfe - 1))

    def _element_points(self) -> np.ndarray:
        """Return time values at all nodes, shape (nfe, ncp+1)."""
        cs = self._cs
        if cs.scheme == "radau":
            from discopt.dae.polynomials import radau_roots

            cp = radau_roots(cs.ncp)
        else:
            from discopt.dae.polynomials import legendre_roots

            cp = legendre_roots(cs.ncp)

        eb = self._element_boundaries()

        tp = np.zeros((cs.nfe, cs.ncp + 1))
        for i in range(cs.nfe):
            t_start = eb[i]
            h_i = self._h_vec[i]
            tp[i, 0] = t_start
            tp[i, 1:] = t_start + h_i * cp
        return tp

    def _state_dict_at(self, i: int, k: int) -> dict:
        """Build states dict at element i, node k."""
        d = {}
        for sv in self._states:
            var = self._vars[sv.name]
            if sv.n_components == 1:
                d[sv.name] = var[i, k]
            else:
                d[sv.name] = [var[i, k, c] for c in range(sv.n_components)]
        return d

    def _algebraic_dict_at(self, i: int, j: int) -> dict:
        """Build algebraics dict at element i, collocation point j."""
        d = {}
        for av in self._algebraics:
            var = self._vars[av.name]
            if av.n_components == 1:
                d[av.name] = var[i, j]
            else:
                d[av.name] = [var[i, j, c] for c in range(av.n_components)]
        return d

    def _control_dict_at(self, i: int) -> dict:
        """Build controls dict at element i."""
        d = {}
        for cv in self._controls:
            var = self._vars[cv.name]
            if cv.n_components == 1:
                d[cv.name] = var[i]
            else:
                d[cv.name] = [var[i, c] for c in range(cv.n_components)]
        return d

    def _quadrature_weights(self) -> np.ndarray:
        """Quadrature weights for the collocation scheme.

        For Radau IIA, uses the implicit Radau quadrature weights.
        For Gauss-Legendre, uses the Gauss-Legendre quadrature weights.
        Both are on [0, 1].
        """
        cs = self._cs
        if cs.scheme == "radau":
            from discopt.dae.polynomials import radau_roots

            # Interpolatory (Radau IIA) quadrature weights on the collocation
            # points ONLY: solve the Vandermonde moment system
            #   sum_j w_j * cp_j^k = ∫_0^1 t^k dt = 1/(k+1),  k = 0..ncp-1
            # This is exact for all ncp (including ncp=1) and, because the points
            # are Gauss-Radau, is exact to degree 2*ncp-2. The previous
            # implementation integrated the Lagrange basis over the *full* node
            # set [0, cp...] and dropped node 0's weight — only valid when that
            # weight is 0 (ncp >= 2); at ncp=1 it under-integrated by 2x. (C2)
            cp = radau_roots(cs.ncp)
            k = np.arange(cs.ncp)
            vander = cp[np.newaxis, :] ** k[:, np.newaxis]  # (ncp, ncp), rows=powers
            moments = 1.0 / (k + 1.0)
            return np.linalg.solve(vander, moments)
        else:
            _, w_gl = np.polynomial.legendre.leggauss(cs.ncp)
            return w_gl / 2.0  # transform from [-1,1] to [0,1]


def align_time_grid(
    t_span: tuple[float, float],
    nfe: int,
    measurement_times: np.ndarray,
) -> np.ndarray:
    """Adjust element boundaries so measurement times coincide with boundaries.

    Starts with a uniform grid of ``nfe + 1`` boundary points, then snaps
    each boundary to the nearest measurement time (if it is closer than half
    the original element width). This ensures that measurement times fall
    exactly on element boundaries where collocation nodes exist.

    Parameters
    ----------
    t_span : tuple of float
        ``(t0, tf)`` domain boundaries.
    nfe : int
        Number of finite elements for the initial uniform grid.
    measurement_times : np.ndarray
        Times at which measurements are available.

    Returns
    -------
    np.ndarray
        Adjusted element boundaries, sorted and strictly increasing.
        The first and last entries are always ``t_span[0]`` and ``t_span[1]``.
    """
    t0, tf = t_span
    boundaries = np.linspace(t0, tf, nfe + 1)
    meas = np.sort(np.asarray(measurement_times, dtype=np.float64))
    h = (tf - t0) / nfe

    # Snap interior boundaries to nearby measurement times
    for m_t in meas:
        if m_t <= t0 or m_t >= tf:
            continue
        # Find the nearest interior boundary
        dists = np.abs(boundaries[1:-1] - m_t)
        idx = int(np.argmin(dists))
        if dists[idx] < 0.5 * h:
            boundaries[idx + 1] = m_t

    # Ensure sorted, unique, and strictly increasing
    boundaries = np.unique(boundaries)

    # Re-fix endpoints in case of floating-point drift
    boundaries[0] = t0
    boundaries[-1] = tf
    return boundaries
