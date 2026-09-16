"""#1262: the native spatial kernel must not certify an open gap.

``_try_native_spatial_kernel`` set ``gap_certified=math.isfinite(bound_val)`` —
bound *finiteness*, not gap *closure* — so a budgeted ``node_limit`` exit
reported a certificate. Measured before the fix: nvs13 at ``max_nodes=5`` came
back ``node_limit`` with obj=-585.2, bound=-969.33 (66% gap) and
``gap_certified=True``. Every Python driver clears the flag on a budgeted exit,
and every consumer (``_route_result_is_certified``, phase gates) reads it as
"closed".

The fix only ever withdraws a claim; ``bound_valid`` keeps reporting that the
bound itself is usable, so the two flags now carry different information.
"""

from __future__ import annotations

import os

import discopt.solver as solver_mod
from discopt.modeling.core import from_nl

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


def _kernel_results(monkeypatch, name, **solve_kwargs):
    seen = []
    original = solver_mod._try_native_spatial_kernel

    def spy(*args, **kwargs):
        res = original(*args, **kwargs)
        if res is not None:
            seen.append(res)
        return res

    monkeypatch.setattr(solver_mod, "_try_native_spatial_kernel", spy)
    result = from_nl(os.path.join(DATA, f"{name}.nl")).solve(**solve_kwargs)
    return result, seen


def test_budgeted_kernel_exit_is_not_certified(monkeypatch):
    result, seen = _kernel_results(monkeypatch, "nvs13", time_limit=30.0, max_nodes=5)

    # The probe must actually exercise the kernel's budgeted exit (CLAUDE.md §6).
    assert seen, "native kernel never supplied a result -- probe measured nothing"
    budgeted = [r for r in seen if r.status in ("time_limit", "node_limit")]
    assert budgeted, f"kernel never took a budgeted exit: {[r.status for r in seen]}"

    for r in budgeted:
        assert r.gap_certified is False, (
            f"{r.status} exit with obj={r.objective} bound={r.bound} claims a closed gap (#1262)"
        )
        # The bound is still rigorous on a budgeted exit: the flags must differ.
        assert r.bound_valid is True

    assert result.status != "optimal"
    assert result.gap_certified is False
    assert result.bound_valid is True


def test_kernel_optimal_exit_still_certifies(monkeypatch):
    """The withdrawal is confined to budgeted exits: a proof still certifies."""
    result, seen = _kernel_results(monkeypatch, "nvs13", time_limit=120.0)

    assert seen, "native kernel never supplied a result -- probe measured nothing"
    assert result.status == "optimal"
    assert result.gap_certified is True
    assert any(r.status == "optimal" and r.gap_certified for r in seen)
