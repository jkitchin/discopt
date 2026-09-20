"""The distance cap's round-off allowance must not scale with the box (#1397).

``improving_gradient_norms`` (#1284) credits column ``j`` with what a move of at
most ``FEASIBLE_DISTANCE_TOL`` can remove from the row's violation. Each room
carried a round-off allowance ``16·u·(|bound| + |x|)`` — honest as *room's own*
round-off, but ``room`` is then divided by an absolute ``FEASIBLE_DISTANCE_TOL``
and multiplied by ``|J_ij|``, which amplifies the allowance by up to
``|J_ij| / FEASIBLE_DISTANCE_TOL`` before it reaches the acceptance cap.

The consequence is not a slightly loose cap. A column pinned **exactly on** the
bound that blocks its improving direction has a true room of exactly zero, and
that zero is exactly representable — there is no round-off there to absorb. Yet
it was credited with phantom room proportional to ``|bound|``, and once that
phantom reaches ``FEASIBLE_DISTANCE_TOL`` the column is credited with its *full*
``|J_ij|``. At that point ``improving_gradient_norms`` returns the plain sup-norm
and #1284's tightening is a silent no-op — which matters because #1284 exists
because the untightened cap certified a point 0.87 away in ``y`` at ``z = -20``
against a true optimum of −6.699.

Two directions, both swept here:

* **unsound direction** — the phantom credit, ``test_a_blocked_column_is_never_
  credited_at_any_scale`` and ``test_the_tightening_does_not_degrade_to_the_plain
  _sup_norm``. The oracle is exact and needs no solver: a column sitting on the
  bound that blocks it can contribute nothing, so the true improving-gradient
  norm is known by hand.
* **preservation direction** — the allowance still exists for a reason, and the
  two instances #1284 named must keep passing to the ulp. Those live in
  ``test_1284_distance_cap_improving_columns.py`` and are re-asserted here at the
  level of the allowance itself, so a future edit to the cap cannot silently
  change them.

The fix is a ``min``, not a new constant: ``a <= K·u·FEASIBLE_DISTANCE_TOL``,
derived in the source comment from ``dcap = a·Σ|J_ij|`` and ``grad <= Σ|J_ij|``.
``FEASIBLE_DISTANCE_TOL`` itself is unchanged — #1397 forbids tolerance tuning.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from discopt.validation.feasibility import (
    FEASIBLE_DISTANCE_TOL,
    improving_gradient_norms,
    jacobian_row_gradient_norms,
)

U = float(np.finfo(np.float64).eps)

#: Bound magnitudes. The small end is where the old allowance was harmless (and
#: must stay byte-identical); the large end is where it defeated the cap.
BOUNDS = (1e0, 1e3, 1e6, 1e8, 1e9, 1e10, 1e11, 1e12, 1e14)

#: ``|J|`` of the blocked column. Larger than the helping column's so that the
#: plain sup-norm — the ceiling the final ``min`` imposes — is distinguishable
#: from the true answer. This is #1397 check (4): a coefficient amplifies the
#: phantom, which is why a machine-epsilon allowance becomes an O(1) weight.
W = 1000.0


def _blocked_row(bound: float):
    """One row whose only helping column is blocked, with an exact hand oracle.

    Body must DECREASE (``direction = +1``).

    * col 0 has partial ``-W`` and sits **exactly on** its upper bound. Its
      improving move is up (``step = -1·sign(-W) = +1``), which the bound blocks,
      so its true room is exactly 0 and its true contribution is exactly 0.
    * col 1 has partial ``+1``, sits interior at 0.5 with ``lb = 0``, so its room
      of 0.5 far exceeds ``FEASIBLE_DISTANCE_TOL`` and it saturates at 1.0.

    The true improving-gradient norm is therefore exactly ``1.0``, at every
    ``bound``, while the plain sup-norm is ``W``.
    """
    J = np.array([[-W, 1.0]])
    x = np.array([bound, 0.5])
    lb = np.array([0.0, 0.0])
    ub = np.array([bound, 1.0])
    return J, x, lb, ub, np.array([1.0])


TRUE_NORM = 1.0


@pytest.mark.parametrize("bound", BOUNDS)
def test_a_blocked_column_is_never_credited_at_any_scale(bound):
    """A column with exactly zero room must contribute essentially nothing."""
    got = float(improving_gradient_norms(*_blocked_row(bound))[0])
    # Allow the derived round-off allowance its full amplified effect and no
    # more: a * W / TOL on top of the true 1.0.
    ceiling = TRUE_NORM + 16.0 * U * FEASIBLE_DISTANCE_TOL * W / FEASIBLE_DISTANCE_TOL
    assert got <= ceiling, (
        f"at bound {bound:.0e} a column pinned ON the bound that blocks its "
        f"improving direction (true room exactly 0) was credited: the returned "
        f"improving-gradient norm is {got:.6f} against a true {TRUE_NORM:.1f} "
        f"({got / TRUE_NORM:.1f}x), so the acceptance cap is "
        f"{FEASIBLE_DISTANCE_TOL * got:.3e} instead of "
        f"{FEASIBLE_DISTANCE_TOL * TRUE_NORM:.3e}"
    )


def test_the_tightening_does_not_degrade_to_the_plain_sup_norm():
    """The headline failure: at large bounds the #1284 tightening became a no-op."""
    checked = 0
    degraded = []
    for bound in BOUNDS:
        J, x, lb, ub, d = _blocked_row(bound)
        got = float(improving_gradient_norms(J, x, lb, ub, d)[0])
        plain = float(jacobian_row_gradient_norms(J)[0])
        assert plain == pytest.approx(W)
        checked += 1
        if got >= plain * (1.0 - 1e-9):
            degraded.append((bound, got))
    # §6: the sweep must actually have graded verdicts.
    assert checked == len(BOUNDS) == 9, f"graded {checked} bounds, expected 9"
    assert not degraded, (
        "improving_gradient_norms collapsed to the plain sup-norm "
        f"{W:.0f} at bounds {[f'{b:.0e}' for b, _ in degraded]}, i.e. #1284's "
        "tightening is a silent no-op there"
    )


def test_the_phantom_is_monotone_in_the_bound_only_before_the_fix():
    """Pin the *class*: the returned norm must be flat in the bound, not rising.

    This is the falsifiable statement the fix rests on. The true answer does not
    depend on ``bound`` at all, so neither may the returned value.
    """
    values = [float(improving_gradient_norms(*_blocked_row(b))[0]) for b in BOUNDS]
    assert len(values) == 9
    spread = max(values) - min(values)
    assert spread <= 1e-9, (
        "the returned improving-gradient norm varies with the bound magnitude "
        f"(spread {spread:.3e} over bounds 1e0…1e14: {values}) — the true value "
        "is independent of the box, so any dependence is the absolute divisor "
        "showing through"
    )


@pytest.mark.parametrize("bound", (1e10, 1e12, 1e14))
def test_an_integer_column_already_on_its_integer_is_not_credited(bound):
    """The integer arm has the same defect via ``to_int``'s allowance.

    ``gap == 0`` — the column is already integral — yet ``to_int`` was
    ``slack·(|r| + |x|)``, pure round-off scaled by the point's magnitude.
    """
    J = np.array([[-W, 1.0]])
    x = np.array([bound, 0.5])  # col 0 integral (a power of ten) and on its ub
    lb = np.array([0.0, 0.0])
    ub = np.array([bound, 1.0])
    mask = np.array([True, False])
    got = float(improving_gradient_norms(J, x, lb, ub, np.array([1.0]), mask)[0])
    ceiling = TRUE_NORM + 16.0 * U * W
    assert got <= ceiling, (
        f"at bound {bound:.0e} an integer column already ON its integer and on "
        f"the blocking bound was credited; returned {got:.6f} vs true "
        f"{TRUE_NORM:.1f}"
    )


def test_the_two_1284_instances_keep_their_allowance_to_the_ulp():
    """Preservation: the cap must not touch the cases the allowance exists for.

    In both, ``|bound| + |x|`` is below 1e-7, so the derived ceiling
    ``16·u·FEASIBLE_DISTANCE_TOL`` is far larger and the ``min`` selects the
    original term unchanged. Asserted on the arithmetic directly so the claim is
    checked rather than assumed.
    """
    slack = 16.0 * U
    ceiling = slack * FEASIBLE_DISTANCE_TOL
    checked = 0
    # (name, |bound|, |x|) for every column the two pinned tests exercise.
    for name, b, xv in (
        ("portfol_roundlot x2 (down, lb=0)", 0.0, 7.220330978261474e-11),
        ("portfol_roundlot x11 (int, r=0)", 0.0, 0.0),
        ("clay0303hfsg x (down, lb=0)", 0.0, 9.99763018e-09),
        ("clay0303hfsg y (int, r=0)", 0.0, 1.76515511e-10),
    ):
        original = slack * (abs(b) + abs(xv))
        checked += 1
        assert min(original, ceiling) == original, (
            f"{name}: the #1397 ceiling {ceiling:.3e} clipped an allowance of "
            f"{original:.3e} that #1284 relies on — the cap is too tight"
        )
    assert checked == 4, f"graded {checked} columns, expected 4"


def test_portfol_roundlot_tie_still_breaks():
    """The exact tie the allowance was introduced for, end to end on the function.

    ``x11 - 78000 x2 >= 0`` at ``x2 = 7.22e-11`` (lb 0), ``x11 = 0`` integer.
    Moving ``x2`` onto its bound repairs the row exactly, so the point must stay
    inside the cap — and did so by a single ulp, which is why the allowance
    exists.
    """
    x2 = 7.220330978261474e-11
    J = np.array([[-78000.0, 1.0]])
    x = np.array([x2, 0.0])
    viol = 78000.0 * x2
    g = improving_gradient_norms(
        J, x, np.zeros(2), np.full(2, np.inf), [-1.0], np.array([False, True])
    )
    assert viol <= FEASIBLE_DISTANCE_TOL * float(g[0]), (
        "the #1397 allowance cap broke the tie portfol_roundlot depends on: "
        f"viol {viol!r} > cap {FEASIBLE_DISTANCE_TOL * float(g[0])!r}"
    )


def test_the_shipped_allowance_is_capped():
    """Couple the sweeps to the code they pin.

    Without this the sweeps could keep passing while the allowance went back to
    being unbounded — they would be measuring a formula nobody evaluates.
    """
    src = inspect.getsource(improving_gradient_norms)
    assert "allow = slack * FEASIBLE_DISTANCE_TOL" in src, (
        "the derived allowance ceiling is gone from improving_gradient_norms; "
        f"the sweeps in this file no longer pin the shipped formula:\n{src}"
    )
    # The uncapped forms must not return.
    for dead in (
        "np.maximum(ub - x, 0.0) + slack * (np.abs(ub) + np.abs(x))",
        "np.maximum(x - lb, 0.0) + slack * (np.abs(lb) + np.abs(x))",
        "np.abs(gap) + slack * (np.abs(r) + np.abs(x))",
    ):
        assert dead not in src, f"an uncapped allowance has returned: {dead}"
    assert src.count("np.minimum(slack *") == 3, (
        "expected the cap on all three rooms (up, down, to_int); found "
        f"{src.count('np.minimum(slack *')}"
    )
