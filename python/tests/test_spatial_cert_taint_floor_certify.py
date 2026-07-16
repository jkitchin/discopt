"""SPATIAL-CERT (gap-closing plan P0): the spatial driver certifies a closed
gap over soundly-floored sentinel removals (#604 parity).

The MILP driver's B1-FIX (#604) established the house certification
discipline: a node removed without its own proof is re-represented by its
POP-TIME inherited bound (proved at an ancestor; valid over the subtree
forever, #603), and ``Optimal`` requires the *floor-inclusive* gap to close —
every unproven removal provably within tolerance of the incumbent. The spatial
driver (``solve_model``) was not covered: one ``_nonrigorous_sentinel_fathom``
event cleared ``_gap_certified`` for the whole solve, unconditionally, even
when the removal's floor sat far ABOVE the incumbent (a rigorous cutoff fathom
in all but name).

Measured defect (docs/dev/flag-graduation-verdicts23-2026-07-10.md, verdict 3):
nvs22 with ``DISCOPT_LU_DENSITY_ROUTE=1`` solves to obj 6.05822 / bound
6.0582199951512 (relative gap ~9.8e-9, five orders inside the 1e-4 tolerance,
37 nodes) yet returned ``feasible``. Root cause (this fix's entry experiment):
at exactly one node (iteration 5) the strided per-node NLP is skipped, the
node LP solves "optimal" but *soundly declines* to certify a bound (no
Neumaier-Shcherbina safe bound computable: free lifted column; C-38-family
refusal in ``mccormick_lp.py``), the NLP-placeholder failure sentinel
survives, and the node is sentinel-fathomed carrying pop-time floor
7.40348 — far above the incumbent 6.05822. Every removed subtree was
rigorously accounted; only the label was withheld.

Same class, second probe: st_e36 with ``DISCOPT_NODE_REDUCE=1`` (BR-3 blocker,
docs/dev/br3-regate-2026-07-10.md) — one sentinel fathom whose floor
-246.00000046 sits 4.6e-7 below the incumbent -246.0000000026, well inside the
certification tolerance; the floor-inclusive gap is closed and the exit must
certify with the floor as the reported bound.

Negative control: a tree whose taint floor does NOT close the gap (nvs05,
floor 1.3521 vs incumbent 5.4709) must keep the ``feasible`` downgrade — that
is asserted by ``test_b2fix_taint_floor_bound.py`` (#27a: an open
floor-inclusive gap never certifies) and unit-tested here on the extracted
convergence predicate.

Both instance tests fail before this fix (status ``feasible``) and pass after
(status ``optimal``; bound, objective and node count byte-identical to the
pre-fix run — the change is label-only). Marked ``slow`` (~10-15 s solves).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data")
_NVS22_NL = os.path.join(_DATA, "minlplib", "nvs22.nl")
_ST_E36_NL = os.path.join(_DATA, "minlplib_nl", "st_e36.nl")

# MINLPLib reference optima (minlplib.solu =opt=). Both minimize.
_NVS22_OPT = 6.05822
_ST_E36_OPT = -246.0
# Solver-native tolerances (conftest house values).
_ABS = 1e-4
_REL = 1e-3


@pytest.mark.slow
def test_nvs22_density_route_certifies_closed_gap(monkeypatch):
    """Route ON: nvs22's closed, rigorously-accounted gap must label optimal.

    Pre-fix signature: status ``feasible`` with bound 6.0582199951512 and 37
    nodes — the gap is closed (9.8e-9) and the single sentinel-fathomed node
    carries floor 7.40348 > incumbent, but the label is withheld. This is the
    FLAG-GRAD verdict-3 blocker for lu_density_route, obj_branch_priority and
    lift_loose_products.
    """
    monkeypatch.setenv("DISCOPT_LU_DENSITY_ROUTE", "1")
    # Pin the other graduated flags OFF: this test freezes the exact measured
    # configuration (route alone) so its deterministic 37-node signature holds
    # regardless of the flags' defaults. (node_numerical_dual_bound, graduated
    # with #362, rescues the single declined node and legitimately certifies in
    # 35 nodes — pinned OFF here to keep this frozen-signature probe intact.)
    monkeypatch.setenv("DISCOPT_OBJ_BRANCH_PRIORITY", "0")
    monkeypatch.setenv("DISCOPT_LIFT_LOOSE_PRODUCTS", "0")
    monkeypatch.setenv("DISCOPT_NODE_NUMERICAL_DUAL_BOUND", "0")
    result = from_nl(_NVS22_NL).solve(time_limit=40, gap_tolerance=1e-4)

    assert result.status == "optimal", (
        f"certificate withheld on a closed, rigorously-accounted gap: "
        f"status={result.status!r}, bound={getattr(result, 'bound', None)!r} "
        f"(pre-fix signature: feasible / 6.0582199951512 / 37 nodes)"
    )

    # No false certificate: the labelled optimum must sit at the oracle.
    assert result.objective is not None
    err = abs(result.objective - _NVS22_OPT)
    assert err <= _ABS or err <= _REL * abs(_NVS22_OPT), (
        f"objective {result.objective!r} vs oracle {_NVS22_OPT}"
    )

    # Label-only fix: the reported dual bound is the same rigorous value the
    # pre-fix run reported, closed to the optimum and never past the oracle.
    bound = getattr(result, "bound", None)
    assert bound is not None
    assert bound <= _NVS22_OPT + 1e-6, f"dual bound {bound!r} crosses the oracle {_NVS22_OPT}"
    assert abs(bound - _NVS22_OPT) <= 1e-3, f"dual bound {bound!r} did not close to {_NVS22_OPT}"

    # The search itself must be untouched (certification accounting only).
    # Deterministic frozen signature, byte-identical across reps. Originally 37
    # nodes (2026-07-10 root-cause data + the SPATIAL-CERT entry experiment);
    # re-frozen at 35 on 2026-07-16 after unrelated landed changes shifted the
    # trajectory — verified pre-existing at merge-base fd5db1c (35 nodes, same
    # optimal/objective) before the #362 branch touched anything, and invariant
    # to the #362 flag (identical at =0 and =1). A drift here means some change
    # altered this frozen configuration's search — re-verify it was label-only
    # before re-freezing.
    assert result.node_count == 35, (
        f"node count {result.node_count} != 35: the frozen-config search drifted"
    )


@pytest.mark.slow
def test_st_e36_node_reduce_certifies_within_tolerance_floor(monkeypatch):
    """node_reduce ON: st_e36's floor sits 4.6e-7 under the incumbent — inside
    the certification tolerance — so the exit certifies with the floor as the
    reported bound (the BR-3 node_reduce blocker, same accounting class).
    """
    monkeypatch.setenv("DISCOPT_NODE_REDUCE", "1")
    # Pin the graduated flags OFF ("0" disables each): this test freezes the
    # exact measured configuration (node_reduce alone, historical LU routing)
    # so its deterministic signature holds regardless of the flags' defaults.
    monkeypatch.setenv("DISCOPT_LU_DENSITY_ROUTE", "0")
    monkeypatch.setenv("DISCOPT_OBJ_BRANCH_PRIORITY", "0")
    monkeypatch.setenv("DISCOPT_LIFT_LOOSE_PRODUCTS", "0")
    monkeypatch.setenv("DISCOPT_NODE_NUMERICAL_DUAL_BOUND", "0")
    result = from_nl(_ST_E36_NL).solve(time_limit=40, gap_tolerance=1e-4)

    assert result.status == "optimal", (
        f"certificate withheld: status={result.status!r}, "
        f"bound={getattr(result, 'bound', None)!r} "
        f"(pre-fix signature: feasible / -246.00000046256457 / 75 nodes)"
    )

    assert result.objective is not None
    err = abs(result.objective - _ST_E36_OPT)
    assert err <= _ABS or err <= _REL * abs(_ST_E36_OPT), (
        f"objective {result.objective!r} vs oracle {_ST_E36_OPT}"
    )

    # The certified bound must be the FLOOR-INCLUSIVE value (the honest
    # rigorous bound: the removed subtree floors at -246.00000046, below the
    # frontier), on the sound side of the oracle and of the incumbent.
    bound = getattr(result, "bound", None)
    assert bound is not None
    assert bound <= _ST_E36_OPT + 1e-9, f"dual bound {bound!r} crosses the oracle {_ST_E36_OPT}"
    assert bound <= result.objective + 1e-9, "certificate invariant: bound <= incumbent (min)"
    assert abs(bound - _ST_E36_OPT) <= 1e-3, f"dual bound {bound!r} did not close to the optimum"


@pytest.mark.smoke
def test_gap_values_converged_matches_certification_arithmetic():
    """The floor-inclusive certification test uses the exact `_gap_converged`
    arithmetic: decoupled absolute (1e-6 default) and relative tolerances,
    no 1.0 denominator floor, non-finite never converges."""
    # Imported here (not at module top) so the slow behavioral tests above
    # still collect and FAIL (not error) on a pre-fix build without the helper.
    from discopt.solver import _gap_values_converged

    # nvs22's terminal state: inc 6.05822, floor-inclusive bound = frontier
    # 6.0582199951512 -> abs gap 4.85e-9 <= 1e-6: converged.
    assert _gap_values_converged(6.05822, 6.0582199951512, 1e-4)
    # st_e36: inc -246.0000000026479, floor -246.00000046256457 -> abs gap
    # 4.6e-7 <= 1e-6: converged.
    assert _gap_values_converged(-246.0000000026479, -246.00000046256457, 1e-4)
    # nvs05: inc 5.4709, floor 1.3521 -> gap 4.1 / 5.47 = 0.75: NOT converged
    # (#27a negative control - an open floor-inclusive gap never certifies).
    assert not _gap_values_converged(5.4709341264, 1.3520892807, 1e-4)
    # Relative branch: 1e-5 relative gap at scale 1e4 (abs gap 0.1 > abs tol).
    assert _gap_values_converged(10000.0, 9999.9, 1e-4)
    assert not _gap_values_converged(10000.0, 9990.0, 1e-4)
    # Non-finite bounds never converge (bound_unresolved -inf pin).
    assert not _gap_values_converged(5.0, float("-inf"), 1e-4)
    assert not _gap_values_converged(float("inf"), 1.0, 1e-4)
