"""#1236 item B: the integer-bilinear lift must earn its adoption.

The reformulation is adopted whenever it is *possible* -- the lift eliminates
every nonlinear term -- never because it was shown to help. Measured over the
population it is actually adopted on (12 instances across both in-repo corpora),
it is harmful more often than not: `nvs02` 297 -> 3 nodes without it, `nvs14`
273 -> 3, `ex1266` 1409 -> 268, `ex1263` 4997 -> 2317, `prob02` 37 -> 5,
`prob03` 7 -> 5. Four instances are unaffected. Two genuinely need it --
`ex1264` and `ex1265` certify ONLY with the lift.

Two obvious separators are measured dead (performance-plan §68.3): root-bound
tightness is *anti*-correlated with the outcome (the lift's root bound on nvs02
equals the optimum to 12 digits and its tree still takes 297 nodes), and bit
width cannot separate `ex1263`/`ex1266` from `ex1264`/`ex1265` because they are
the same family at the same widths.

What does separate them is whether the un-lifted model closes at all, so
``DISCOPT_IPX_CHEAP_FIRST`` measures that instead of predicting it. These tests
pin the two ends of that behaviour and, above all, that the flag never costs a
certificate.
"""

from __future__ import annotations

import os

import pytest
from discopt.modeling.core import from_nl

DATA = os.path.join(os.path.dirname(__file__), "data")
CORPORA = (
    os.path.join(DATA, "minlplib_nl"),
    os.path.join(DATA, "minlplib"),
)

#: Instances the un-lifted path closes quickly, with the node count it closes in.
#: The lift costs each of them nodes, so the gate must decline it.
LIFT_DECLINED = {"nvs02": 3, "nvs14": 3, "prob03": 5}

#: Instances that certify ONLY with the lift. The gate must keep it, and the
#: probe it spends first must not cost them their certificate.
LIFT_REQUIRED = {"ex1264": 8.6, "ex1265": 10.3}


def _path(name):
    for d in CORPORA:
        p = os.path.join(d, f"{name}.nl")
        if os.path.exists(p):
            return p
    raise AssertionError(f"{name}.nl not found in either corpus")


def _solve(name, monkeypatch, gate, time_limit=60.0):
    monkeypatch.setenv("DISCOPT_IPX_CHEAP_FIRST", "1" if gate else "0")
    return from_nl(_path(name)).solve(time_limit=time_limit)


@pytest.mark.smoke
@pytest.mark.parametrize("name,unlifted_nodes", sorted(LIFT_DECLINED.items()))
def test_gate_declines_a_lift_the_unlifted_path_does_not_need(name, unlifted_nodes, monkeypatch):
    off = _solve(name, monkeypatch, gate=False)
    on = _solve(name, monkeypatch, gate=True)

    # Both arms must certify, or the comparison below is between different things.
    assert off.status == "optimal" and off.gap_certified
    assert on.status == "optimal" and on.gap_certified
    assert abs(on.objective - off.objective) <= 1e-6 * max(1.0, abs(off.objective)), (
        f"{name}: the gate changed the certified objective {off.objective!r} -> {on.objective!r}"
    )
    assert on.node_count == unlifted_nodes, (
        f"{name}: gate ON explored {on.node_count} nodes, not the un-lifted "
        f"path's {unlifted_nodes} -- the lift was not declined"
    )
    assert on.node_count < off.node_count, (
        f"{name}: gate ON ({on.node_count}) is no better than OFF "
        f"({off.node_count}); this instance no longer belongs in this list"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("name,reference", sorted(LIFT_REQUIRED.items()))
def test_gate_keeps_a_lift_the_unlifted_path_cannot_close(name, reference, monkeypatch):
    """The half that matters: the probe must not cost a certificate.

    `ex1264`/`ex1265` run their whole budget uncertified without the lift. The
    gate spends up to 40% of the budget proving exactly that, then hands the rest
    to the lift -- which certifies in 1.2 s and 2.3 s respectively, so the probe
    is affordable. If this ever regresses, the flag is trading a certificate for
    node counts and must not ship.
    """
    on = _solve(name, monkeypatch, gate=True)
    assert on.status == "optimal" and on.gap_certified, (
        f"{name}: gate ON returned {on.status!r} (certified={on.gap_certified}); "
        "the cheap-first probe cost this instance its certificate"
    )
    assert abs(on.objective - reference) <= 1e-6 * max(1.0, abs(reference)), (
        f"{name}: objective {on.objective!r} differs from the reference {reference}"
    )
    assert on.bound is not None
    assert on.bound <= on.objective + 1e-6 * max(1.0, abs(on.objective))


@pytest.mark.smoke
def test_gate_is_on_by_default_with_an_opt_out(monkeypatch):
    """Default ON since the §5 graduation panel, with the `=0` opt-out intact.

    The opt-out is what keeps the legacy adopt-whenever-possible path reachable
    and testable, which CLAUDE.md §5 requires of a graduated flag.
    """
    import discopt.solver as solver_mod

    monkeypatch.delenv("DISCOPT_IPX_CHEAP_FIRST", raising=False)
    assert solver_mod._ipx_cheap_first_enabled() is True
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("DISCOPT_IPX_CHEAP_FIRST", off)
        assert solver_mod._ipx_cheap_first_enabled() is False, off
    monkeypatch.setenv("DISCOPT_IPX_CHEAP_FIRST", "1")
    assert solver_mod._ipx_cheap_first_enabled() is True
