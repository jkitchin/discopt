"""#928 — a per-node round CUT SHORT must return the rigorous floor it holds.

§14a taught a *node LP* cut short by its deadline to bank a sound
Neumaier–Shcherbina dual floor instead of returning nothing. The coupled
graduation panel (performance-plan §14b) then failed on the analogous hole one
level up: with both the #928 LP clamp and the #966 round clamp active, a *round*
cut short produced a node result carrying **no adoptable bound at all**, and a
tree whose nodes never certify anything reports ``bound=None``.

Measured on the 19-instance binding subset with a spent round grant
(``scratchpad/issue928_round_cut_short_entry.py``): 16 of 114 cells returned
``uncertified``/no bound where the same box under an unclamped round certifies
one — in BOTH ``DISCOPT_LP_WARM_DEADLINE`` arms, so the loss is the round clamp,
not the LP clamp. In every one of them the deadline-truncated build (0 constraint
rows) still carried a valid finite ``_objective_floor``, equal to the unclamped
control bound on 4 of the 5 instances (``issue928_floor_inventory.py``):
bchoco06/07/08 -1.0, tls2 0.0, hda -172201.82.

These tests pin both halves of the fix:

* a cut-short round reports that floor rather than declining (and takes the
  tighter of it and any banked deadline dual);
* the #966 round-admission check never declines the ROOT round, which holds no
  parent bound — declining it is what leaves the whole tree bound-less.

Both fail on the pre-fix tree. The default path is asserted untouched: with the
shipped flags no build is ever truncated and no node solve reports
``time_limit``, so the new floor is never even computed.
"""

from __future__ import annotations

import time
from pathlib import Path

import discopt._relax.milp_relaxation as mr_mod
import numpy as np
import pytest
from discopt import Model
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt._relax.milp_relaxation import MilpRelaxationResult
from discopt._relax.model_utils import flat_variable_bounds
from discopt.modeling.core import from_nl

CORPUS = Path(__file__).parent / "data" / "minlplib_nl"


def _spent_round() -> float:
    """A round grant that is already gone — the state #966's clamp produces."""
    return time.perf_counter() - 1.0


def _bilinear_model() -> Model:
    m = Model()
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.minimize(x * y - 2.0 * x + y)
    m.subject_to(x + y >= 1.0)
    return m


@pytest.mark.parametrize(
    ("instance", "floor"),
    [("bchoco06", -1.0), ("tls2", 0.0)],
)
def test_cut_short_round_reports_its_rigorous_floor(instance, floor):
    """A round whose grant is spent returns a sound bound, not ``uncertified``.

    Pre-fix both instances return ``status="uncertified"`` with
    ``lower_bound=None``: the truncated build solves to LP optimality and every
    certification route then declines it for want of a safe bound.
    """
    model = from_nl(str(CORPUS / f"{instance}.nl"))
    lb, ub = flat_variable_bounds(model)
    relaxer = MccormickLPRelaxer(model)

    res = relaxer.solve_at_node(lb, ub, time_limit=5.0, round_deadline=_spent_round())

    assert res.lower_bound is not None, f"{instance}: a cut-short round returned no bound"
    assert np.isfinite(res.lower_bound)
    assert res.status == "optimal"
    assert res.lower_bound == pytest.approx(floor, rel=1e-9, abs=1e-9)


def test_cut_short_floor_never_exceeds_the_unclamped_bound():
    """Soundness: the truncated round's floor is below what the full round proves.

    The floor is the box-interval bound of the same column box the LP is solved
    over, and the LP feasible region is a subset of that box, so the clamped value
    can never exceed the unclamped certified one.
    """
    model = from_nl(str(CORPUS / "bchoco06.nl"))
    lb, ub = flat_variable_bounds(model)

    full = MccormickLPRelaxer(model).solve_at_node(lb, ub, time_limit=30.0)
    cut = MccormickLPRelaxer(model).solve_at_node(
        lb, ub, time_limit=30.0, round_deadline=_spent_round()
    )

    assert full.lower_bound is not None and cut.lower_bound is not None
    assert cut.lower_bound <= full.lower_bound + 1e-6 * max(1.0, abs(full.lower_bound))


def test_deadline_exit_takes_the_tighter_of_banked_dual_and_box_floor(monkeypatch):
    """A yielded LP's banked ``g(y)`` and the round's box floor are both valid.

    They are not ordered a priori — the hda node LP banked exactly ``g(0)``, the
    box floor itself, at every deadline fraction (§14a) — so the node must report
    the larger. Here the banked dual is deliberately the weaker of the two.
    """
    model = _bilinear_model()
    relaxer = MccormickLPRelaxer(model)
    relaxer._inc = None  # force the cold, build-and-solve path
    lb, ub = flat_variable_bounds(model)

    seen_floor: list[float] = []

    def _yield_on_deadline(self, time_limit=None, *args, **kwargs):
        # Stand in for a warm LP that ran out of budget having banked a floor
        # strictly below the relaxation's own box-interval floor.
        floor = self._objective_floor
        assert floor is not None and np.isfinite(floor), "no box floor to compare against"
        seen_floor.append(float(floor))
        return MilpRelaxationResult(
            status="time_limit",
            objective=None,
            bound=float(floor) - 100.0,
            x=None,
            safe_bound=float(floor) - 100.0,
        )

    monkeypatch.setattr(mr_mod.MilpRelaxationModel, "solve", _yield_on_deadline)
    res = relaxer.solve_at_node(lb, ub, time_limit=5.0)

    assert seen_floor, "the stubbed LP never ran"
    assert res.lower_bound == pytest.approx(seen_floor[0])


def test_default_path_still_declines_an_uncertifiable_node():
    """Neutrality: with no clamp in force the new floor is never surfaced.

    ``4stufen``'s root relaxation solves to LP optimality and every certification
    route declines it, so the shipped contract is ``uncertified`` with no bound.
    Nothing about this changes — the floor is reachable only from a truncated
    build or a ``time_limit`` LP status, neither of which the default path
    produces.
    """
    model = from_nl(str(CORPUS / "4stufen.nl"))
    lb, ub = flat_variable_bounds(model)

    res = MccormickLPRelaxer(model).solve_at_node(lb, ub, time_limit=10.0)

    assert res.status == "uncertified"
    assert res.lower_bound is None


def test_root_round_is_never_declined_by_the_admission_check(monkeypatch):
    """#966's round admission must not decline the ROOT round (rule 1).

    A branched node always carries its parent's bound, so declining its round
    forgoes tightening only. The root has no parent: declining it leaves the tree
    with no bound source at all, which is the ``bound=None`` collapse the coupled
    panel measured on contvar. Forcing the admission check to refuse every round
    (an expected build cost no grant can cover) must still let the ROOT round run.
    Pre-fix this counts zero node-loop rounds; post-fix, one.
    """
    monkeypatch.setenv("DISCOPT_NODE_ROUND_BUDGET", "1")
    monkeypatch.setattr(MccormickLPRelaxer, "expected_build_cost", lambda self: 1e6, raising=True)

    # Count the rounds the NODE LOOPS ran: those are the calls carrying a
    # ``round_deadline`` (the one-off root LP probe passes None).
    rounds: list = []
    orig = MccormickLPRelaxer.solve_at_node

    def _spy(self, node_lb, node_ub, time_limit=None, **kw):
        if kw.get("round_deadline") is not None:
            rounds.append(kw["round_deadline"])
        return orig(self, node_lb, node_ub, time_limit, **kw)

    monkeypatch.setattr(MccormickLPRelaxer, "solve_at_node", _spy)

    # nvs05 is the smallest in-repo instance whose solve actually reaches the
    # spatial node loop with an LP relaxer attached (``_bilinear_model`` closes at
    # the root and never gets there, so it cannot see this gate at all).
    res = from_nl(str(CORPUS / "nvs05.nl")).solve(time_limit=5.0)

    assert rounds, "every round was declined — the root was left with no bound source"
    assert res.bound is not None and np.isfinite(res.bound)
