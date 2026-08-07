"""Huge-but-finite declared LP boxes must certify, not error (issue #937).

A declared variable box in ``[1e16, 1e20)`` made the Rust simplex exit
``numerical`` (→ ``SolveStatus.ERROR``). That band is exactly the band the #850
guard defers *into*: the guard distrusts an IPM ``UNBOUNDED`` whenever a declared
bound in ``[1e15, 1e20)`` was relaxed to the IPM infinity and hands off to the
exact simplex — which then could not certify either, so a correctly-detected
``unbounded`` LP came back as an undiagnosed ``status: error``. ``9.999e19`` is
the *default* ``Model.continuous`` upper bound, so this was the default path.

Three things are pinned here:

1. The simplex certifies the whole band (not one point), at the exact corner.
2. The #850 deferral is a preference, not a destruction: when no engine that
   honors the declared box can certify, the deferred ``unbounded`` is returned
   rather than ``error``; and the *simplex's own* ``UNBOUNDED`` — a verdict
   reached with the box honored — is never deferred.
3. ``lp_simplex.solve_lp`` rejects unknown keywords instead of silently
   swallowing an ``lb=``/``ub=`` box and solving over the default one.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("discopt._rust")

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.lp_simplex import solve_lp as simplex_lp  # noqa: E402

# The band the #850 guard defers into is [1e15, 1e20). 1e20 is the LP layer's INF
# sentinel — a genuinely infinite bound — and is checked separately.
_FINITE_BAND = [1e3, 1e12, 1e15, 1e16, 1e17, 1e18, 1e19, 5e19, 9.999e19]


def _unbounded_direction_lp(u: float):
    """``min -z0 - z1  s.t.  z0 + z1 >= 1,  0 <= z <= u``.

    Optimal at the corner ``z = (u, u)`` with objective ``-2u`` for any finite
    ``u``; unbounded only when the box itself is infinite.
    """
    return dict(
        c=np.array([-1.0, -1.0]),
        A_ub=sp.csr_matrix([[-1.0, -1.0]]),
        b_ub=np.array([-1.0]),
        bounds=[(0.0, u), (0.0, u)],
    )


@pytest.mark.smoke
def test_simplex_certifies_the_whole_huge_finite_box_band():
    """Every finite box below the 1e20 sentinel certifies at ``-2u``.

    Pins the band end to end rather than a point: pre-fix the failure started at
    ``1e16`` (a row residual measured only against ``|b|`` while the row's own
    terms were ~1e16) and changed character at ``9.999e19`` (the standard-form
    slack's optimal value passing its ``1e20`` sentinel "bound"), so a single
    probe would have missed half of it.
    """
    checked = 0
    for u in _FINITE_BAND:
        r = simplex_lp(**_unbounded_direction_lp(u))
        assert r.status == SolveStatus.OPTIMAL, (
            f"ub={u:.4e}: a finite declared box must certify, got {r.status.value}"
        )
        assert r.objective == pytest.approx(-2.0 * u, rel=1e-12), f"ub={u:.4e}"
        checked += 1
    assert checked == len(_FINITE_BAND), "band probe did not run every point"


@pytest.mark.smoke
def test_simplex_reports_unbounded_at_the_inf_sentinel():
    """``1e20`` is the LP layer's infinity, so there the LP really is unbounded."""
    r = simplex_lp(**_unbounded_direction_lp(1e20))
    assert r.status == SolveStatus.UNBOUNDED


@pytest.mark.smoke
def test_model_solve_on_the_default_continuous_box_is_not_error():
    """End-to-end: the issue's model must not come back ``error``.

    ``m.continuous(lb=0)`` leaves the default upper bound ``9.999e19``, so the LP
    as posed is bounded and the sound verdict is ``optimal`` at ``-2 * 9.999e19``
    — the certificate the #850 guard's own comment says the exact simplex should
    supply. What it must never be is ``error``.
    """
    import discopt.modeling as dm

    m = dm.Model("unbounded_direction")
    z = m.continuous("z", shape=(2,), lb=0)
    m.minimize(-z[0] - z[1])
    m.subject_to(z[0] + z[1] >= 1.0, name="lower")

    res = m.solve()
    assert res.status != "error", "the #937 defect: a correct verdict lost to `error`"
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-2.0 * 9.999e19, rel=1e-9)


@pytest.mark.smoke
def test_deferred_unbounded_failure_is_diagnosed_not_silently_error(monkeypatch):
    """When nothing can certify the declared box, say *why* — but do not invent a
    verdict.

    The #850 guard declines to certify an IPM ``UNBOUNDED`` on a box with a
    declared bound in ``[1e15, 1e20)``. It used to signal that with a bare ``None``
    — indistinguishable from "this engine failed" — so a simplex failure (#937's
    other half) produced a bare, undiagnosable ``error``.

    The fix is diagnosis, not promotion. Promoting the held verdict was the obvious
    reading of #937 and it is **unsound**: a ``[1e15, 1e20)`` bound is finite as
    posed, so on ``min -z0 - z1`` over the default box the truth is ``optimal`` at
    the corner (the certificate #850 decided is sound) and ``unbounded`` would be a
    false certificate on a bounded problem. So the status stays ``error`` — not a
    certificate, an honest "no engine could decide this" — and the warning carries
    the engine, the cause, and the remedy.
    """
    import discopt.modeling as dm
    import discopt.solver as solver_mod

    calls = {"simplex": 0}

    def _failing_simplex(model, t_start, time_limit=None):
        calls["simplex"] += 1
        return None

    def _deferring_pounce(model, t_start, time_limit=None):
        return solver_mod._DeferredUnbounded(
            solver_mod.SolveResult(status="unbounded", wall_time=0.0), "STUB-IPM"
        )

    monkeypatch.setattr(solver_mod, "_solve_lp_simplex", _failing_simplex)
    monkeypatch.setattr(solver_mod, "_solve_lp_pounce", _deferring_pounce)

    m = dm.Model("unbounded_direction")
    z = m.continuous("z", shape=(2,), lb=0)
    m.minimize(-z[0] - z[1])
    m.subject_to(z[0] + z[1] >= 1.0, name="lower")

    with pytest.warns(RuntimeWarning, match=r"relaxed a declared finite bound"):
        res = solver_mod._solve_lp(m, t_start=0.0)
    assert calls["simplex"] == 1, "the simplex stub must actually have been consulted"
    assert res.status == "error", (
        "a declined UNBOUNDED on a finite box must not be promoted to a certificate; "
        f"got {res.status}"
    )


@pytest.mark.smoke
def test_simplex_unbounded_verdict_is_never_deferred():
    """The guard must not discard a verdict from an engine that honors the box.

    The simplex's infinity threshold is ``1e20``, so it reads a declared bound in
    ``[1e15, 1e20)`` as finite and its ``UNBOUNDED`` is a statement about the box
    as declared. Applied engine-agnostically the guard threw that away whenever
    some *unrelated* variable carried a bound in the window — here ``w``, while
    the unboundedness comes from ``z`` at the true ``1e20`` infinity.
    """
    import discopt.modeling as dm

    m = dm.Model("genuinely_unbounded")
    z = m.continuous("z", lb=0.0, ub=1e20)  # true infinity for both engines
    w = m.continuous("w", lb=0.0, ub=1e18)  # inside the #850 deferral window
    m.minimize(-z)
    m.subject_to(z + w >= 1.0, name="lower")

    res = m.solve()
    assert res.status == "unbounded", (
        f"a genuinely unbounded direction must survive the #850 guard; got {res.status}"
    )


@pytest.mark.unit
def test_solve_lp_rejects_unknown_keywords():
    """``lb=``/``ub=`` must raise, not be swallowed into the default ``[0, 1e20]``.

    ``solve_lp`` takes ``**_kwargs`` for cross-backend signature compatibility; a
    bare catch-all made a caller-supplied box vanish and returned ``OPTIMAL`` over
    a box it never asked for (#937, side finding).
    """
    lp = _unbounded_direction_lp(5.0)
    lp.pop("bounds")
    with pytest.raises(TypeError, match=r"lb, ub"):
        simplex_lp(lb=np.zeros(2), ub=np.full(2, 5.0), **lp)

    # The inert keywords of the shared LP contract still pass through silently.
    r = simplex_lp(bounds=[(0.0, 5.0)] * 2, options=None, certificate=True, **lp)
    assert r.status == SolveStatus.OPTIMAL
    assert r.objective == pytest.approx(-10.0)
