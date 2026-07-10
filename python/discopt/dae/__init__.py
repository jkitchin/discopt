"""discopt.dae -- DAE discretization for dynamic optimization.

Transcribe ODE/DAE systems into algebraic constraints using orthogonal
collocation on finite elements or finite differences. Compatible with the
standard discopt modeling API -- no changes to the DAG compiler or solver.

Quick Start
-----------
>>> import discopt.modeling as dm
>>> from discopt.dae import ContinuousSet, DAEBuilder
>>>
>>> m = dm.Model("batch_reactor")
>>> cs = ContinuousSet("t", bounds=(0, 5), nfe=30, ncp=3)
>>> dae = DAEBuilder(m, cs)
>>> dae.add_state("C", initial=1.0, bounds=(0, 2))
>>> dae.set_ode(lambda t, s, a, c: {"C": -0.5 * s["C"]})
>>> dae.discretize()
>>> m.minimize(dae.get_state("C")[-1, -1] ** 2)
>>> result = m.solve()
"""

from discopt.dae.collocation import (
    AlgebraicVar,
    ContinuousSet,
    ControlVar,
    DAEBuilder,
    StateVar,
    align_time_grid,
)
from discopt.dae.finite_difference import FDBuilder
from discopt.dae.fit import Trajectory, TrajectoryFit, fit_trajectories
from discopt.dae.mol import BoundaryCondition, FieldVar, MOLBuilder, SpatialSet
from discopt.dae.polynomials import (
    collocation_matrix,
    collocation_matrix_2nd,
    legendre_roots,
    radau_roots,
)

__all__ = [
    "AlgebraicVar",
    "BoundaryCondition",
    "ContinuousSet",
    "ControlVar",
    "DAEBuilder",
    "FDBuilder",
    "FieldVar",
    "MOLBuilder",
    "SpatialSet",
    "StateVar",
    "Trajectory",
    "TrajectoryFit",
    "align_time_grid",
    "fit_trajectories",
    "collocation_matrix",
    "collocation_matrix_2nd",
    "legendre_roots",
    "radau_roots",
]
