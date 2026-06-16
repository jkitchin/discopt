"""Soundness guard for ill-conditioned fractional-power envelopes (issue #158).

A fractional power ``x**p`` with ``p < 0`` (or ``0 < p < 1``) over a box whose
lower bound reaches toward zero has tangent/secant slopes that blow up:
``|p| * lb**(p-1)`` reaches ~1e9+ for ``nvs08`` (``x0**-3.5`` on ``x0 >= 0.0121``).
Emitting an LP row with such a coefficient against an RHS of order 1 makes HiGHS
return a polytope that *excludes feasible points*. Two downstream consumers then
turn that unsound polytope into a false certificate:

  * **Root OBBT** min/maxes each variable over the relaxation. On the poisoned
    polytope it shrank ``nvs08``'s ``x0`` to the single point ``[0.0155, 0.0155]``
    — far below the true optimum ``x0 = 0.6314`` — cutting the optimum away.
  * **The per-node McCormick LP relaxer** then reports the (now genuinely
    box-empty) node ``infeasible``; the spatial-B&B driver treats that as a
    rigorous fathom and certifies the suboptimal incumbent ``12021.58`` (true
    optimum ``23.4497``) as ``optimal`` with ``gap=0``.

The fix drops any envelope cut whose slope exceeds ``_LIFT_MAX_ENVELOPE_SLOPE``.
Dropping a cut only ENLARGES the relaxation, so it is always sound; the aux
column degrades at worst to its (exact) value bounds. These tests assert the
relaxation stays a valid OUTER approximation — OBBT must never shrink a variable
past its true feasible range, and the root relaxer must not declare a feasible
problem infeasible.

Runs fast (relaxation/OBBT only, no full B&B) and is intentionally NOT marked
``correctness`` so CI — which deselects ``correctness`` — still catches a
regression here.
"""

import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import numpy as np
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root
from discopt.modeling.core import VarType
from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

_DATA = Path(__file__).parent / "data" / "minlplib"

# nvs08 published optimum (MINLPLib).
_NVS08_OPT = 23.4497
_NVS08_OPT_POINT = {"x0": 0.63138504, "x1": 4.0, "x2": 3.0}


def _int_offsets_sizes(model):
    offsets, sizes, off = [], [], 0
    for v in model._variables:
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            offsets.append(off)
            sizes.append(v.size)
        off += v.size
    return offsets, sizes


def _fbbt_then_obbt(model, deadline_s=8.0):
    """Reproduce the solver's root presolve: FBBT then OBBT over the relaxation."""
    lb, ub = flat_variable_bounds(model)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    int_offsets, int_sizes = _int_offsets_sizes(model)
    lb, ub, infeasible, _ = tighten_root_bounds_with_fbbt(
        model, lb.copy(), ub.copy(), int_offsets, int_sizes, model_repr=None
    )
    assert not infeasible, "root FBBT must not call a feasible model infeasible"
    res = obbt_tighten_root(
        model,
        lb.copy(),
        ub.copy(),
        rounds=3,
        deadline=time.perf_counter() + deadline_s,
        prefer_pounce=True,
    )
    assert not res.infeasible, "root OBBT must not call a feasible model infeasible"
    return np.maximum(lb, res.lb), np.minimum(ub, res.ub)


def test_nvs08_root_obbt_keeps_optimum_in_box():
    """OBBT over the nvs08 relaxation must not cut the true optimum out of the box.

    Reverting the conditioning guard collapses ``x0`` to ``[0.0155, 0.0155]`` — the
    optimum ``x0 = 0.6314`` falls outside, the optimum is lost, and the run
    certifies a suboptimal incumbent. The guard keeps the (loose, sound) box.
    """
    model = dm.from_nl(str(_DATA / "nvs08.nl"))
    lb, ub = _fbbt_then_obbt(model)
    x0_opt = _NVS08_OPT_POINT["x0"]
    assert lb[0] <= x0_opt + 1e-9, f"OBBT lower bound {lb[0]} cut the optimum x0={x0_opt}"
    assert ub[0] >= x0_opt - 1e-9, f"OBBT upper bound {ub[0]} cut the optimum x0={x0_opt}"
    # Integer optimum (x1=4, x2=3) must likewise survive.
    assert lb[1] <= _NVS08_OPT_POINT["x1"] <= ub[1]
    assert lb[2] <= _NVS08_OPT_POINT["x2"] <= ub[2]


def test_nvs08_root_relaxer_not_infeasible_and_bound_sound():
    """The McCormick LP relaxer at the nvs08 root is a valid outer relaxation:

    it must NOT report ``infeasible`` (the problem is feasible) and any finite
    lower bound it returns must not exceed the true optimum.
    """
    model = dm.from_nl(str(_DATA / "nvs08.nl"))
    lb, ub = _fbbt_then_obbt(model)
    relaxer = MccormickLPRelaxer(model)
    res = relaxer.solve_at_node(lb, ub)
    assert res.status != "infeasible", "feasible nvs08 root reported infeasible by relaxer"
    if res.lower_bound is not None and np.isfinite(res.lower_bound):
        assert res.lower_bound <= _NVS08_OPT + 1e-4, (
            f"root dual bound {res.lower_bound} exceeds true optimum {_NVS08_OPT}"
        )


def test_steep_negative_power_relaxation_stays_sound():
    """Minimal adversarial model: ``x**-3.5 <= rhs`` with a tiny lower bound.

    The true feasible region needs ``x >= rhs**(-1/3.5)``. The relaxation, being
    an OUTER approximation, may only UNDER-state that threshold — never report a
    box that contains feasible points as infeasible, and never bound the
    objective above the true minimum.
    """
    rhs = 100.0
    true_threshold = rhs ** (-1.0 / 3.5)  # x must be >= this to be feasible
    model = dm.Model()
    x = model.continuous("x", lb=1e-3, ub=200.0)
    model.subject_to(1.0 / ((x**3) * dm.sqrt(x)) <= rhs)
    model.minimize(x)

    relaxer = MccormickLPRelaxer(model)
    # On the full box the relaxation's min(x) is a valid LOWER bound on the true
    # minimum (= true_threshold); it must not exceed it.
    res = relaxer.solve_at_node(np.array([1e-3]), np.array([200.0]))
    assert res.status == "optimal"
    assert res.lower_bound <= true_threshold + 1e-6, (
        f"relaxation min {res.lower_bound} exceeds true feasible min {true_threshold}"
    )

    # A box strictly inside the genuinely-infeasible region stays a sound
    # infeasibility proof (truth: x must be >= 0.268, so [0.01, 0.02] is empty).
    res_empty = relaxer.solve_at_node(np.array([0.01]), np.array([0.02]))
    assert res_empty.status == "infeasible"

    # A box straddling the threshold must remain feasible in the relaxation.
    res_ok = relaxer.solve_at_node(np.array([true_threshold]), np.array([1.0]))
    assert res_ok.status != "infeasible", "feasible box reported infeasible"
