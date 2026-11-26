"""#2362: the native MILP engine must not propagate through a presolve cancellation residue.

``data/issue_2362/2122_p2.mps.gz`` is MIPLIB ``2122`` (a maximization) with its rows and
columns permuted by ``d2_flag_panel.py --permute`` seed 2. MILP presolve keeps a
``5.55e-17`` entry in an equality row (``Work::cancel_entry`` keeps residue in equalities on
purpose). Root probing then divided the ``6.9e-12`` round-off of the row's incremental
activity by it, "implied" ``x <= -124557`` against ``x >= 0``, fixed a binary away from
every optimum, and the solve certified ``optimal`` at node 1 with bound -189678.33. The
true optimum is -187612.94 (HiGHS, ``mip_rel_gap=0``, ``mip_abs_gap=0``, 1 thread), so
that bound was 2065 below it.

The fix (``domain.rs``: ``is_noise_coef``) refuses to derive a bound through an entry
``|a| <= 1e-9``, HiGHS's absolute ``small_matrix_value`` rule, which keeps such entries
out of its own domain. The unit test
``domain::tests::a_noise_coefficient_is_not_propagated_through`` pins the arithmetic;
this test pins the end-to-end certificate.
"""

from pathlib import Path

import discopt.modeling as dm
import pytest

DATA = Path(__file__).parent / "data" / "issue_2362" / "2122_p2.mps.gz"

# Exact HiGHS optimum of the (maximization) instance.
OPT = -187612.94419439966


@pytest.mark.parametrize("root_probe", ["1", "0"])
def test_bound_never_passes_the_optimum(monkeypatch, root_probe):
    # Before the fix both arms certified a bound below the optimum (root probe on:
    # -189678.33 at node 1; off: -187733.65 at node 1).
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    monkeypatch.setenv("DISCOPT_HIGHS_SEPA_ROOT_PROBE", root_probe)
    model = dm.from_mps(str(DATA))
    r = model.solve(max_nodes=30, deterministic=True, gap_tolerance=1e-4, time_limit=120)
    assert r.bound is not None and r.objective is not None
    tol = 1e-6 * abs(OPT)
    # Maximization: the dual bound is an upper bound on every feasible objective.
    assert r.bound >= OPT - tol, (r.status, r.objective, r.bound, r.node_count)
    assert r.objective <= OPT + tol, (r.status, r.objective, r.bound)
    if r.gap_certified:
        assert r.objective >= OPT - 1e-4 * abs(OPT) - tol
