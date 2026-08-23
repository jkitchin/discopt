"""``deterministic=True`` must neutralize the role-2 wall budgets (#1116).

``kriging_peaks-full200`` at ``max_nodes=1`` — one process, one binary, no user
time pressure — returned root dual bounds of −25371.8 / −28852.0 / −28072.6 across
three repetitions: a 14 % swing with the incumbent bit-identical. The cause is not
float noise and not hash iteration order (the whole Rust crate has three iteration
sites over hash containers, all order-insensitive, and replacing
``Variable.__hash__`` with a stable index left the drift untouched). It is
structural: the first root LP came back with a different number of COLUMNS per
repetition, because a wall-truncated tightening stage handed the builder a
different box. Replacing the clock with a deterministic counter made the same
solve reproduce exactly.

#912 named the distinction this test pins. A clock answering *"when do we stop?"*
— the user's ``time_limit`` — is role 1 and correct by definition. A clock
answering *"how much work do we do?"* is role 2, and makes the ANSWER a function of
machine speed; ``_work_budget.py`` calls that "a correctness-of-process bug, not a
performance detail".

``deterministic`` was a public parameter documented as "Ensure deterministic
results" that was read **nowhere** — a dead flag (CLAUDE.md §3) promising exactly
the guarantee #1116 measured being broken. These tests fail on the pre-fix tree:
the helpers do not exist, the tuning field does not exist, and the parameter has no
observable effect.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import discopt.modeling as dm
import discopt.solver as solver
import discopt.solver_tuning as solver_tuning
import pytest

pytestmark = [pytest.mark.relaxation]


def _model():
    m = dm.Model()
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.subject_to(x * y >= 0.5)
    m.minimize(x * x + y * y)
    return m


def test_role2_default_is_off_and_passes_the_clock_through():
    """Off by default: turning the real mode on is a bound-changing default flip."""
    assert solver_tuning.SolverTuning().deterministic is False
    token = solver_tuning.set_current(
        dataclasses.replace(solver_tuning.SolverTuning(), deterministic=False)
    )
    try:
        assert solver._role2_deadline(123.0) == 123.0
        assert solver._role2_horizon(4.0) == 4.0
    finally:
        solver_tuning.reset_current(token)


def test_role2_budgets_are_inert_under_the_flag():
    token = solver_tuning.set_current(
        dataclasses.replace(solver_tuning.SolverTuning(), deterministic=True)
    )
    try:
        assert solver._role2_deadline(123.0) is None
        assert solver._role2_horizon(4.0) == float("inf")
    finally:
        solver_tuning.reset_current(token)


def test_role1_deadline_is_not_routed_through_the_role2_helpers():
    """The user's ``time_limit`` still stops the search under the flag.

    A ``deterministic`` mode that also silenced role 1 would turn ``time_limit``
    into a suggestion. The guarantee is scoped to a run that terminates on *work*.
    """
    res = _model().solve(deterministic=True, max_nodes=50, time_limit=60.0)
    assert res.status in {"optimal", "feasible"}


@pytest.mark.parametrize("flag", [True, False])
def test_the_parameter_is_no_longer_dead(monkeypatch, flag):
    """``deterministic=`` must be *observable* on the solve path.

    Pre-fix this parameter was accepted, documented as a guarantee, and read
    nowhere. The assertion is that the role-2 helpers fire during a real solve and
    that the flag decides what they return — a dead parameter fires nothing.
    """
    seen: list[bool] = []
    real = solver._role2_deadline

    def spy(deadline):
        out = real(deadline)
        seen.append(out is None)
        return out

    monkeypatch.setattr(solver, "_role2_deadline", spy)
    _model().solve(deterministic=flag, max_nodes=50)

    assert seen, "the role-2 helper never fired — the probe measured nothing"
    assert all(s is flag for s in seen), (
        f"deterministic={flag} but suppression pattern was {set(seen)}"
    )


def test_flag_does_not_leak_out_of_the_solve():
    before = solver_tuning.current().deterministic
    _model().solve(deterministic=True, max_nodes=20)
    assert solver_tuning.current().deterministic == before


def test_role2_budget_is_the_optional_seconds_form_and_is_used():
    """``_role2_budget`` is the ``Optional[float]``-seconds sibling of the other two.

    Callees split into two conventions for "no clock": ``None`` (``subnlp``'s
    ``time_budget``, ``solve_at_node``'s ``time_limit``) and ``math.inf`` (a plain
    ``float`` parameter compared against elapsed time). Both need a helper, and a
    helper with no call site is a dead flag (CLAUDE.md §3) — so this also asserts
    every one of the three is actually wired into the solve path.
    """
    token = solver_tuning.set_current(
        dataclasses.replace(solver_tuning.SolverTuning(), deterministic=True)
    )
    try:
        assert solver._role2_budget(3.0) is None
    finally:
        solver_tuning.reset_current(token)
    assert solver._role2_budget(3.0) == 3.0

    src = Path(solver.__file__).read_text()
    for helper in ("_role2_budget", "_role2_deadline", "_role2_horizon"):
        # -1 for the ``def`` line; docstring mentions are stripped first.
        body = re.sub(r'"""(?:.|\n)*?"""', "", src)
        assert body.count(f"{helper}(") - 1 >= 1, f"{helper} has no call site"


def test_role1_phase_entry_gates_are_deliberately_left_live():
    """The documented residual, pinned so it stays a decision and not an oversight.

    Neutralizing ``_deadline_exhausted`` / ``_remaining_budget`` under the flag
    would let preprocessing overrun the user's ``time_limit`` without bound —
    trading a reproducibility bug for a broken role-1 promise (CLAUDE.md §1). The
    guarantee is therefore scoped to a solve whose role-1 budget never binds, and
    both docstrings say so. This test fails if someone later widens the flag to
    swallow role 1 without revisiting that trade.
    """
    body = re.sub(r'"""(?:.|\n)*?"""', "", Path(solver.__file__).read_text())
    for gate in ("_deadline_exhausted(", "_remaining_budget()"):
        assert gate in body
        assert f"_role2_budget({gate}" not in body
        assert f"_role2_horizon({gate}" not in body

    field_doc = Path(solver_tuning.__file__).read_text()
    assert "_deadline_exhausted" in field_doc, "the residual must stay documented"
    assert "max_wall_time" in field_doc, "the POUNCE stall backstop must stay documented"
