"""Regression test for an unsound spatial-B&B fathom on a fractional-power model.

Found while digging into ``nvs08`` certification. A fractional power ``x**p``
with ``|p|`` large over a domain reaching toward 0 (here ``x0 in [1e-3, 200]``,
``p=-3.5``) lifts to an LP-form McCormick relaxation whose tangent slopes and
aux-column bounds span more than 1e13. The fast Rust ``simplex`` backend reports
that ill-conditioned relaxation **infeasible**, while HiGHS solves it correctly.

The spatial-B&B driver treats a relaxation ``infeasible`` as a *rigorous fathom*
(it prunes the node's whole subtree). So the unverified false-infeasible silently
fathomed a feasible node and certified a suboptimal incumbent with a dual bound
far ABOVE the true optimum — a false-optimal:

    before:  optimal 2320.87   (bound 2320.87 >> true optimum 21.56)
    after:   optimal 21.5551   (bound 21.5544 <= obj, sound)

The fix (``MccormickLPRelaxer.solve_at_node``): a floating-point simplex
"infeasible" verdict is not a proof of an empty relaxed feasible set, so it is
re-verified with HiGHS before it is allowed to fathom a node.

True optimum (brute force over integer ``x2``; for each, the binding constraint
``x0**-3.5 <= x2**2`` fixes ``x0 = (x2**2)**(-1/3.5)``):

    x2=3, x0=0.533776  ->  obj = (x0+4)**2 + (x2-2)**2 = 21.555127   (minimum)
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import pytest

# Global optimum, brute-forced and independently verified.
_OPT = 21.555127
_OPT_X0 = 0.533776
_OPT_X2 = 3.0


def _build():
    m = dm.Model("fp_soundness")
    x0 = m.continuous("x0", lb=0.001, ub=200.0)
    x2 = m.integer("x2", lb=0, ub=200)
    m.minimize((x0 + 4) ** 2 + (x2 - 2) ** 2)
    m.subject_to(x0 ** (-3.5) - x2**2 <= 0)  # x0^-3.5 <= x2^2
    return m


@pytest.mark.correctness
def test_fp_node_lp_no_false_optimal():
    """The fractional-power model must not certify a dual bound above the optimum.

    The headline soundness invariant: a valid dual (lower) bound never exceeds
    the true optimum. The pre-fix solver reported bound 2320.87 >> 21.56.
    """
    r = _build().solve(time_limit=30, gap_tolerance=1e-4)
    if r.bound is not None:
        assert r.bound <= _OPT + 1e-2, f"invalid dual bound {r.bound} > true optimum {_OPT}"


@pytest.mark.correctness
def test_fp_certifies_true_optimum():
    """The model certifies its true optimum with a valid dual bound."""
    r = _build().solve(time_limit=30, gap_tolerance=1e-4)
    assert r.status == "optimal", f"status={r.status}"
    assert r.objective is not None
    assert abs(r.objective - _OPT) <= 1e-2, f"obj={r.objective} != {_OPT}"
    # Soundness invariant: dual bound never exceeds the optimum.
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4, f"invalid dual bound {r.bound} > obj {r.objective}"
