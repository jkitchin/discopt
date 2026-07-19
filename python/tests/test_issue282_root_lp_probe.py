"""Regression test for the #282 tightened-box root LP probe (``DISCOPT_ROOT_LP_PROBE_TIGHT``).

Mechanism (measured, see the issue's entry experiment): the spatial path keeps the
McCormick LP relaxer for the whole search only if a one-shot probe solve yields a
valid bound at the root. That probe ran over the *raw declared* variable bounds
(``flat_variable_bounds(model)``), not the FBBT/OBBT-tightened root box. On a model
whose declared bounds are unbounded (the process-synthesis ``*hfsg`` family declares
continuous flows ``[0, inf]``) the McCormick LP is unbounded/None over that raw box,
so the relaxer is wrongly discarded (``_mc_mode = "none"``) and the whole spatial
search falls back to a far looser alphaBB/interval/NLP root bound — even though the
SAME relaxer produces a valid, much tighter bound on the already-computed tightened
box. With ``DISCOPT_ROOT_LP_PROBE_TIGHT=1`` the probe uses the tightened box, so the relaxer
is kept and the tree bounds tighten.

Soundness: the flag only changes WHETHER the (rigorous, outer-approximation) LP
relaxer is kept; every node still solves its own sub-box, and the LP is a valid dual
bound, so a tightened bound can never cross the true optimum. This test asserts the
differential-bound + soundness invariants (CLAUDE.md §5):
  * flag ON tightens the reported root dual bound vs OFF (fails before the fix, where
    the flag is a no-op and the two bounds are identical), and
  * neither bound crosses the true optimum (a valid dual bound of a MAXIMIZE problem
    is an upper bound: it must stay >= the optimum), and
  * no incumbent beats the true optimum (feasible-point check).

Graduation (#764): the ON-vs-OFF differential over the in-repo corpus's complete
affected set (all 24 unbounded-declared-box instances) was cert-clean and
net-positive (``syn05hfsg`` feasible→optimal 2x faster, ``tanksize`` root
0.8529→0.9063, byte-stable elsewhere), so the tightened-box probe is now the
**default**; ``DISCOPT_ROOT_LP_PROBE_TIGHT=0`` restores the legacy raw-box probe.
``test_default_probe_tightens_root_bound_after_graduation`` pins that the default
behaves like the ON arm.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

# syn05hfsg: smallest sibling of the #282 *hfsg* family; nonconvex, MAXIMIZE, with
# continuous flows declared [0, inf] — the structure that makes the raw-box probe
# discard the relaxer. BARON-confirmed optimum from minlplib.solu (=opt=).
_INSTANCE = "syn05hfsg"
_OPT = 837.7324009
_SENSE = "max"
_TOL = 1e-4 * max(1.0, abs(_OPT))


def _solve(flag: str, budget: float = 8.0):
    os.environ["DISCOPT_ROOT_LP_PROBE_TIGHT"] = flag
    try:
        m = from_nl(os.path.join(_DATA, f"{_INSTANCE}.nl"))
        assert m._objective.sense == ObjectiveSense.MAXIMIZE
        return m.solve(time_limit=budget, gap_tolerance=1e-4)
    finally:
        os.environ.pop("DISCOPT_ROOT_LP_PROBE_TIGHT", None)


def _assert_sound(tag: str, res) -> None:
    # A valid dual bound of a MAXIMIZE problem is an UPPER bound: never below the opt.
    for field in ("root_bound", "bound"):
        val = getattr(res, field, None)
        if val is not None:
            assert val >= _OPT - _TOL, (
                f"{tag}: {field}={val} is below the true optimum {_OPT} "
                f"(false/too-tight dual bound — soundness violation)"
            )
    # A feasible incumbent of a MAXIMIZE problem never beats the optimum.
    if res.objective is not None:
        assert res.objective <= _OPT + _TOL, (
            f"{tag}: incumbent {res.objective} beats the true optimum {_OPT} "
            f"(false-feasible — soundness violation)"
        )


@pytest.mark.slow
def test_tightened_box_probe_tightens_root_bound_soundly():
    off = _solve("0")
    on = _solve("1")

    # Both arms must be sound regardless of the flag.
    _assert_sound("flag-off", off)
    _assert_sound("flag-on", on)

    assert off.root_bound is not None and on.root_bound is not None

    # Differential bound: the flag keeps the LP relaxer, so the reported root dual
    # bound (an upper bound for MAXIMIZE) must be strictly TIGHTER (smaller) with the
    # flag on. Before the fix the flag is a no-op and the two are identical, so this
    # assertion fails -> it is the regression guard.
    assert on.root_bound < off.root_bound - _TOL, (
        f"flag ON root_bound={on.root_bound} not tighter than OFF={off.root_bound}; "
        f"the tightened-box probe did not keep the relaxer"
    )

    # And the tighter bound is still a valid upper bound on the optimum.
    assert on.root_bound >= _OPT - _TOL


def _solve_default(budget: float = 8.0):
    """Solve with NO ``DISCOPT_ROOT_LP_PROBE_TIGHT`` env var set — the shipped default."""
    os.environ.pop("DISCOPT_ROOT_LP_PROBE_TIGHT", None)
    m = from_nl(os.path.join(_DATA, f"{_INSTANCE}.nl"))
    assert m._objective.sense == ObjectiveSense.MAXIMIZE
    return m.solve(time_limit=budget, gap_tolerance=1e-4)


@pytest.mark.slow
def test_default_probe_tightens_root_bound_after_graduation():
    """The tightened-box probe is GRADUATED default-ON (#282 Workstream A).

    With no env var set, the shipped default must now produce the TIGHTER (kept-relaxer)
    root bound — i.e. the default equals the explicit-``=1`` arm and is strictly tighter
    than the explicit-``=0`` (legacy raw-box) arm. This fails before the default flip
    (default == OFF == looser) and passes after — the graduation regression guard.
    ``DISCOPT_ROOT_LP_PROBE_TIGHT=0`` remains the opt-out (asserted by the test above).
    """
    default = _solve_default()
    off = _solve("0")

    _assert_sound("default", default)
    assert default.root_bound is not None and off.root_bound is not None

    # Default (graduated ON) is strictly tighter than the legacy raw-box (=0) probe.
    assert default.root_bound < off.root_bound - _TOL, (
        f"default root_bound={default.root_bound} not tighter than legacy =0 "
        f"root_bound={off.root_bound}; the default did not graduate to the kept-relaxer probe"
    )
    # And still a valid upper bound on the optimum.
    assert default.root_bound >= _OPT - _TOL
