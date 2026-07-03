"""Method-of-lines (MOL) discretization for PDE systems.

Provides :class:`MOLBuilder`, which semi-discretizes PDEs with one spatial
dimension and one temporal dimension. The spatial domain is discretized with
second-order central finite differences, and the resulting ODE system is
transcribed in time using either :class:`~discopt.dae.collocation.DAEBuilder`
(orthogonal collocation) or :class:`~discopt.dae.finite_difference.FDBuilder`.

Supports Dirichlet and Neumann boundary conditions, including time-dependent
values via callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from discopt.dae.collocation import ContinuousSet, ControlVar, DAEBuilder
from discopt.dae.finite_difference import FDBuilder


@dataclass
class SpatialSet:
    """A discretized spatial domain for PDE method-of-lines.

    Parameters
    ----------
    name : str
        Name of the spatial coordinate (e.g. ``"z"``).
    bounds : tuple of float
        ``(z0, zf)`` domain boundaries.
    npts : int
        Number of interior grid points (excluding boundary points).
    """

    name: str
    bounds: tuple[float, float]
    npts: int

    def __post_init__(self):
        if self.npts < 1:
            raise ValueError(f"npts must be >= 1, got {self.npts}")
        if self.bounds[1] <= self.bounds[0]:
            raise ValueError(f"bounds must satisfy z0 < zf, got {self.bounds}")

    @property
    def dz(self) -> float:
        """Grid spacing."""
        return (self.bounds[1] - self.bounds[0]) / (self.npts + 1)

    @property
    def interior_points(self) -> np.ndarray:
        """Spatial coordinates of interior grid points."""
        z0, zf = self.bounds
        return np.linspace(z0 + self.dz, zf - self.dz, self.npts)


@dataclass
class BoundaryCondition:
    """Specification for a single boundary condition.

    Parameters
    ----------
    type : str
        ``"dirichlet"`` (fixed value) or ``"neumann"`` (fixed derivative).
    value : float or callable
        Constant value, or a callable ``value(t) -> float`` for
        time-dependent boundary conditions. For Dirichlet, this is the
        field value at the boundary. For Neumann, this is the outward
        normal derivative (du/dn) at the boundary: du/dz at the right
        boundary, -du/dz at the left boundary.
    """

    type: str
    value: float | Callable = 0.0

    def __post_init__(self):
        if self.type not in ("dirichlet", "neumann"):
            raise ValueError(f"BC type must be 'dirichlet' or 'neumann', got {self.type!r}")

    def eval(self, t: Any) -> Any:
        """Evaluate the BC value at time t (float or expression)."""
        if callable(self.value):
            return self.value(t)
        return float(self.value)


@dataclass
class FieldVar:
    """Descriptor for a spatially distributed PDE field variable.

    Parameters
    ----------
    name : str
        Variable name.
    bounds : tuple of float
        ``(lb, ub)`` bounds on the field.
    initial : callable, np.ndarray, or float, optional
        Initial condition. If callable, ``initial(z) -> value`` evaluated at
        each interior point. If array, values at interior points. If float,
        uniform initial value.
    bc_left : BoundaryCondition, optional
        Left boundary condition. Defaults to Dirichlet with value 0.
    bc_right : BoundaryCondition, optional
        Right boundary condition. Defaults to Dirichlet with value 0.
    """

    name: str
    bounds: tuple[float, float] = (-1e20, 1e20)
    initial: Callable | np.ndarray | float | None = None
    bc_left: BoundaryCondition = field(default_factory=lambda: BoundaryCondition("dirichlet", 0.0))
    bc_right: BoundaryCondition = field(default_factory=lambda: BoundaryCondition("dirichlet", 0.0))


class MOLBuilder:
    """Semi-discretize a 1-D PDE using method of lines.

    Spatial derivatives are approximated with second-order central finite
    differences. The resulting system of ODEs is then transcribed in time
    using either orthogonal collocation (:class:`DAEBuilder`) or finite
    differences (:class:`FDBuilder`).

    Parameters
    ----------
    model : discopt.Model
        The optimization model.
    time_set : ContinuousSet
        Temporal domain and discretization parameters.
    spatial_set : SpatialSet
        Spatial domain and number of interior grid points.
    time_method : str
        ``"collocation"`` (default) or ``"finite_difference"``.
    fd_method : str
        Finite-difference method when ``time_method="finite_difference"``.
        One of ``"backward"``, ``"forward"``, ``"central"``.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> from discopt.dae import ContinuousSet, MOLBuilder, SpatialSet, BoundaryCondition
    >>>
    >>> m = dm.Model("heat")
    >>> ts = ContinuousSet("t", bounds=(0, 1), nfe=10, ncp=3)
    >>> ss = SpatialSet("z", bounds=(0, 1), npts=10)
    >>> mol = MOLBuilder(m, ts, ss)
    >>> mol.add_field("u", initial=lambda z: np.sin(np.pi * z),
    ...              bc_left=BoundaryCondition("dirichlet", 0),
    ...              bc_right=BoundaryCondition("dirichlet", 0))
    >>> mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": 0.01 * fzz["u"]})
    >>> mol.discretize()
    """

    def __init__(
        self,
        model,
        time_set: ContinuousSet,
        spatial_set: SpatialSet,
        time_method: str = "collocation",
        fd_method: str = "backward",
    ):
        if time_method not in ("collocation", "finite_difference"):
            raise ValueError(
                f"time_method must be 'collocation' or 'finite_difference', got {time_method!r}"
            )
        self._model = model
        self._time_set = time_set
        self._spatial_set = spatial_set
        self._time_method = time_method
        self._fd_method = fd_method
        self._fields: list[FieldVar] = []
        self._controls: list[ControlVar] = []
        self._pde_rhs: Callable | None = None
        self._time_builder: DAEBuilder | FDBuilder | None = None
        self._discretized = False

    def add_field(
        self,
        name: str,
        bounds: tuple[float, float] = (-1e20, 1e20),
        initial: Callable | np.ndarray | float | None = None,
        bc_left: BoundaryCondition | None = None,
        bc_right: BoundaryCondition | None = None,
    ) -> FieldVar:
        """Declare a spatially distributed field variable.

        Parameters
        ----------
        name : str
            Field name.
        bounds : tuple of float
            ``(lb, ub)`` bounds.
        initial : callable, np.ndarray, float, or None
            Initial condition. Callable receives spatial coordinate z.
        bc_left : BoundaryCondition, optional
            Left boundary condition. Defaults to homogeneous Dirichlet.
        bc_right : BoundaryCondition, optional
            Right boundary condition. Defaults to homogeneous Dirichlet.

        Returns
        -------
        FieldVar
        """
        if bc_left is None:
            bc_left = BoundaryCondition("dirichlet", 0.0)
        if bc_right is None:
            bc_right = BoundaryCondition("dirichlet", 0.0)
        fv = FieldVar(name, bounds, initial, bc_left, bc_right)
        self._fields.append(fv)
        return fv

    def add_control(
        self,
        name: str,
        n_components: int = 1,
        bounds: tuple[float, float] = (-1e20, 1e20),
    ) -> ControlVar:
        """Declare a piecewise-constant control variable (not spatially distributed).

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

    def set_pde(self, rhs: Callable) -> None:
        """Set the PDE right-hand side (pointwise in space).

        The callable is evaluated at each interior spatial grid point to
        build the time derivative of each field.

        Parameters
        ----------
        rhs : callable
            ``rhs(t, z, fields, fields_z, fields_zz, controls) -> dict``

            - ``t``: time value (float or expression)
            - ``z``: spatial coordinate (float)
            - ``fields``: dict mapping field names to scalar expressions
              (field value at this spatial point)
            - ``fields_z``: dict mapping field names to scalar expressions
              (first spatial derivative du/dz, approximated by central FD)
            - ``fields_zz``: dict mapping field names to scalar expressions
              (second spatial derivative d²u/dz², approximated by central FD)
            - ``controls``: dict mapping control names to expressions

            Returns a dict mapping field names to time-derivative expressions.
        """
        self._pde_rhs = rhs

    def discretize(self) -> dict[str, Any]:
        """Generate all variables and constraints.

        Builds the spatial finite-difference stencils, wraps the PDE RHS
        into an ODE system, and delegates to the temporal discretization
        builder.

        Returns
        -------
        dict[str, Variable]
            All created discopt variables, keyed by name.
        """
        if self._discretized:
            raise RuntimeError("MOLBuilder.discretize() has already been called")
        if self._pde_rhs is None:
            raise RuntimeError("No PDE RHS set. Call set_pde() first.")
        if not self._fields:
            raise RuntimeError("No fields declared. Call add_field() first.")
        self._discretized = True

        ss = self._spatial_set
        n = ss.npts
        dz = ss.dz
        z_pts = ss.interior_points

        # Build temporal discretization builder
        tb: DAEBuilder | FDBuilder
        if self._time_method == "collocation":
            tb = DAEBuilder(self._model, self._time_set)
        else:
            tb = FDBuilder(self._model, self._time_set, method=self._fd_method)
        self._time_builder = tb

        # Declare state variables: one n_components state per field
        for fv in self._fields:
            init = self._compute_initial(fv, z_pts)
            tb.add_state(fv.name, n_components=n, bounds=fv.bounds, initial=init)

        # Declare controls
        for cv in self._controls:
            tb.add_control(cv.name, n_components=cv.n_components, bounds=cv.bounds)

        # Build ODE RHS that evaluates the PDE pointwise
        pde_rhs = self._pde_rhs
        fields_list = list(self._fields)

        def ode_rhs(t, states, algebraics, controls):
            all_derivs: dict[str, list] = {fv.name: [] for fv in fields_list}

            for k in range(n):
                z_k = float(z_pts[k])

                # Field values at this spatial point
                f_vals = {}
                for fv in fields_list:
                    u = states[fv.name]
                    f_vals[fv.name] = u[k]

                # Spatial derivatives via central FD with BC handling
                fz_vals = {}
                fzz_vals = {}
                for fv in fields_list:
                    u = states[fv.name]
                    u_left = self._get_left_value(fv, t, k, u, dz)
                    u_right = self._get_right_value(fv, t, k, n, u, dz)
                    u_center = u[k]

                    fz_vals[fv.name] = (u_right - u_left) / (2.0 * dz)
                    fzz_vals[fv.name] = (u_left - 2.0 * u_center + u_right) / (dz**2)

                derivs_k = pde_rhs(t, z_k, f_vals, fz_vals, fzz_vals, controls)

                for fv in fields_list:
                    if fv.name in derivs_k:
                        all_derivs[fv.name].append(derivs_k[fv.name])

            return all_derivs

        tb.set_ode(ode_rhs)
        return tb.discretize()

    def get_field(self, name: str):
        """Get the discopt Variable for a field by name.

        Parameters
        ----------
        name : str
            Field name as passed to :meth:`add_field`.

        Returns
        -------
        Variable
        """
        if self._time_builder is None:
            raise RuntimeError("Call discretize() first")
        return self._time_builder.get_state(name)

    def time_points(self) -> np.ndarray:
        """Return all time points as a flat array."""
        if self._time_builder is None:
            raise RuntimeError("Call discretize() first")
        return self._time_builder.time_points()

    def spatial_points(self) -> np.ndarray:
        """Return the interior spatial grid points."""
        return self._spatial_set.interior_points

    def extract_solution(self, result, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract a 2-D solution field from a solve result.

        Parameters
        ----------
        result : SolveResult
            Result from ``model.solve()``.
        name : str
            Field variable name.

        Returns
        -------
        t : np.ndarray
            Time points, shape ``(n_time,)``.
        z : np.ndarray
            Spatial points (interior), shape ``(npts,)``.
        u : np.ndarray
            Field values, shape ``(n_time, npts)``.
        """
        if self._time_builder is None:
            raise RuntimeError("Call discretize() first")
        t_pts, u_vals = self._time_builder.extract_solution(result, name)
        return t_pts, self._spatial_set.interior_points, u_vals

    @property
    def time_builder(self) -> DAEBuilder | FDBuilder:
        """The underlying temporal discretization builder.

        Useful for accessing ``integral()``, ``least_squares()``, etc.
        """
        if self._time_builder is None:
            raise RuntimeError("Call discretize() first")
        return self._time_builder

    # ── Internal helpers ──

    def _compute_initial(self, fv: FieldVar, z_pts: np.ndarray) -> np.ndarray | None:
        """Evaluate initial condition at interior points."""
        if fv.initial is None:
            return None
        if callable(fv.initial):
            return np.array([fv.initial(z) for z in z_pts], dtype=np.float64)
        if isinstance(fv.initial, np.ndarray):
            return fv.initial
        # scalar: uniform initial value
        return np.full(len(z_pts), float(fv.initial), dtype=np.float64)

    def _get_left_value(self, fv: FieldVar, t, k: int, u, dz: float):
        """Get the field value to the left of interior point k.

        For k > 0, this is simply u[k-1]. For k == 0 (leftmost interior
        point), apply the left boundary condition.
        """
        if k > 0:
            return u[k - 1]
        # k == 0: left boundary
        bc = fv.bc_left
        bc_val = bc.eval(t) if not callable(bc.value) else bc.value(t)
        if bc.type == "dirichlet":
            return bc_val
        # Neumann: bc_val = du/dn = outward normal derivative at the left
        # boundary. The outward normal points in -z, so du/dz|_boundary = -bc_val.
        # First-order Taylor from the first interior point u[0] (at z0 + dz):
        #   u(z0) ≈ u[0] - dz * (du/dz) = u[0] - dz * (-bc_val) = u[0] + dz * bc_val
        # Symmetric with the right boundary (_get_right_value, also `+`). (C1)
        return u[0] + dz * bc_val

    def _get_right_value(self, fv: FieldVar, t, k: int, n: int, u, dz: float):
        """Get the field value to the right of interior point k.

        For k < n-1, this is simply u[k+1]. For k == n-1 (rightmost interior
        point), apply the right boundary condition.
        """
        if k < n - 1:
            return u[k + 1]
        # k == n-1: right boundary
        bc = fv.bc_right
        bc_val = bc.eval(t) if not callable(bc.value) else bc.value(t)
        if bc.type == "dirichlet":
            return bc_val
        # Neumann: du/dz = bc_val at right boundary (outward normal)
        # Ghost point: u_ghost = u[n-1] + dz * bc_val
        return u[n - 1] + dz * bc_val
