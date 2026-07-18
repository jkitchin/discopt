"""Regression test for issue #759: ``syn05hfsg`` dual bound vs. incumbent.

Issue #759 reported that ``syn05hfsg`` surfaces a dual bound *above* its
incumbent/oracle (``bound ~1310-1651 > 837.73``) and framed it as a *min-sense*
``bound <= incumbent`` invariant breach suspected to be an unsound alphaBB
underestimator (#120).

Root cause (this test pins the correct framing):

* ``syn05hfsg`` is a **MAXIMIZE** problem (``.nl`` header ``O0 1``; BARON-confirmed
  optimum ``837.7324`` from ``minlplib.solu``). For a maximization the dual bound
  is an **UPPER** bound, so ``bound >= optimum`` is the *correct-side* soundness
  invariant — a reported bound above the incumbent is exactly what an unconverged
  maximize solve is supposed to produce, not a violation. (The sibling test
  ``test_issue282_root_lp_probe.py`` asserts the same upper-bound invariant.)
* The alphaBB underestimator is **sound** here: its per-node bounds under-estimate
  the true box minimum of the internally-minimized objective (verified by
  independent sampling of the node box that yielded the highest alphaBB bound);
  the boxes it fathoms do not contain the global optimum. It is also *not* the
  bound source on the default path — the McCormick LP / spatial frontier supplies
  the reported bound.

The reported ``bound > incumbent`` observation was therefore a **false positive**
from a sense-unaware min-sense-only ``bound <= oracle`` check in some measurement
panels (e.g. ``issue280_graduation_panel.py``). The sense-aware pattern already
lives in ``gp_minlp_graduation_panel.py`` / ``global_opt_baron_vs_discopt.py`` and
in ``benchmarks.metrics.dual_bound_crosses_optimum`` (added with this fix).

This test asserts, flag-independent on the default path:
  1. the objective certifies the true optimum (``incorrect_count`` contribution 0),
  2. the reported dual bound is a valid UPPER bound (``bound >= opt - tol``) and
     never crosses the optimum from above into a *too-tight* (< opt) region,
  3. no false global certificate is emitted (an open gap stays uncertified).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
_INSTANCE = "syn05hfsg"
# BARON-confirmed optimum from minlplib.solu (=opt=); MAXIMIZE sense.
_OPT = 837.7324009
_TOL = 1e-4 * max(1.0, abs(_OPT))


@pytest.mark.slow
def test_syn05hfsg_dual_bound_is_a_sound_upper_bound():
    m = from_nl(os.path.join(_DATA, f"{_INSTANCE}.nl"))
    # The whole issue hinges on this: it is a MAXIMIZE problem, so the dual bound
    # is an UPPER bound and legitimately sits above the incumbent.
    assert m._objective.sense == ObjectiveSense.MAXIMIZE

    r = m.solve(time_limit=8.0, gap_tolerance=1e-4)

    # (1) The incumbent certifies the true optimum: no false optimal / no incumbent
    #     that beats the optimum (a MAXIMIZE incumbent never exceeds the max).
    assert r.objective is not None
    assert r.objective <= _OPT + _TOL, (
        f"incumbent {r.objective} beats the true optimum {_OPT} (false-feasible)"
    )
    assert abs(r.objective - _OPT) <= _TOL, (
        f"incumbent {r.objective} does not match oracle {_OPT} (incorrect_count > 0)"
    )

    # (2) The reported dual bound is a valid UPPER bound: it must stay >= the
    #     optimum. A bound BELOW the optimum would be the real soundness breach
    #     (a too-tight dual bound that could fathom the optimum). A bound ABOVE the
    #     incumbent — the observation that triggered #759 — is SOUND and expected
    #     for an unconverged maximize solve, NOT a violation.
    for field in ("bound", "root_bound"):
        val = getattr(r, field, None)
        if val is not None:
            assert val >= _OPT - _TOL, (
                f"{field}={val} is BELOW the true optimum {_OPT}: a too-tight "
                f"(unsound) upper bound for a MAXIMIZE problem"
            )

    # (3) No false global certificate: if the search claims the gap is closed, the
    #     bound and incumbent must actually meet (within tolerance). An open gap
    #     must NOT be reported as certified.
    if r.gap_certified and r.bound is not None:
        assert abs(r.bound - r.objective) <= 1e-3 * max(1.0, abs(_OPT)), (
            f"gap_certified=True but bound {r.bound} != incumbent {r.objective}"
        )
