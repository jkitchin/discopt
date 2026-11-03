"""Deletion-order portfolio for ``compute_iis(include_integrality=True)``.

The deletion filter returns *an* IIS, and which one depends on the order it visits
elements. ``portfolio=`` runs one filter per order in separate processes against
one budget and keeps the smallest complete result. Each member is an ordinary
single-order run, so these tests check the plumbing: that the order really
changes which IIS is found, that the smallest one wins, that every member's
result is reported, and that failures surface instead of disappearing.
"""

import time

import discopt.modeling as dm
import pytest
from discopt._iis_milp import DEFAULT_PORTFOLIO, _parse_order
from discopt.infeasibility import compute_iis


def _two_iis_model():
    """Two disjoint IISs: {c0, c1} (size 2) and {c2, c3, c4} (size 3).

    Constraint order puts the small one first, so the forward filter drops it
    (the big one keeps the rest infeasible) and returns the size-3 IIS; the
    reversed filter drops the big one and returns the size-2 IIS.
    """
    m = dm.Model("two_iis")
    x = m.continuous("x", lb=-100, ub=100)
    y1 = m.continuous("y1", lb=-100, ub=100)
    y2 = m.continuous("y2", lb=-100, ub=100)
    m.subject_to(x <= 0, name="c0")
    m.subject_to(x >= 1, name="c1")
    m.subject_to(y1 + y2 >= 3, name="c2")
    m.subject_to(y1 <= 1, name="c3")
    m.subject_to(y2 <= 1, name="c4")
    m.minimize(x + y1 + y2)
    return m


def _names(res):
    return sorted(c.name for c in res.constraints)


def test_order_changes_which_iis_is_found():
    fwd = compute_iis(_two_iis_model(), include_integrality=True, include_bounds=False)
    assert _names(fwd) == ["c2", "c3", "c4"]
    from discopt._iis_milp import milp_iis

    rev = milp_iis(_two_iis_model(), include_bounds=False, order="ICB-r")
    assert rev.constraints == [0, 1] and rev.order == "ICB-r"


def test_portfolio_returns_smallest_complete_member():
    res = compute_iis(
        _two_iis_model(),
        include_integrality=True,
        include_bounds=False,
        portfolio=("ICB", "ICB-r"),
    )
    assert _names(res) == ["c0", "c1"]
    assert res.complete and res.proven_irreducible and res.fully_certified
    members = res.portfolio_members
    assert [mb["order"] for mb in members] == ["ICB", "ICB-r"]
    assert [mb["outcome"] for mb in members] == ["ok", "ok"]
    assert [mb["size"] for mb in members] == [3, 2]
    assert all(mb["complete"] for mb in members)


def test_portfolio_with_fewer_workers_than_orders_runs_all():
    res = compute_iis(
        _two_iis_model(),
        include_integrality=True,
        include_bounds=False,
        portfolio=("ICB-r", "ICB", "CIB"),
        workers=1,
    )
    assert _names(res) == ["c0", "c1"]
    assert [mb["outcome"] for mb in res.portfolio_members] == ["ok"] * 3


def test_portfolio_tie_goes_to_earlier_order():
    # "ICB" and "CIB" visit constraints in the same (forward) order: both size 3.
    res = compute_iis(
        _two_iis_model(), include_integrality=True, include_bounds=False, portfolio=("CIB", "ICB")
    )
    assert [mb["size"] for mb in res.portfolio_members] == [3, 3]
    assert _names(res) == ["c2", "c3", "c4"]


def test_portfolio_member_error_is_reraised():
    m = dm.Model("feasible")
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(x >= 0.5)
    m.minimize(x)
    with pytest.raises(ValueError, match="feasible"):
        compute_iis(m, include_integrality=True, portfolio=("ICB", "CIB"))


def test_portfolio_respects_total_time_limit():
    t0 = time.perf_counter()
    res = compute_iis(
        _two_iis_model(),
        include_integrality=True,
        portfolio=DEFAULT_PORTFOLIO,
        total_time_limit=20.0,
    )
    assert time.perf_counter() - t0 < 20.0
    assert len(res) > 0


@pytest.mark.parametrize("bad", ["IC", "ICBX", "ICB-x", "IIB", ""])
def test_bad_order_is_rejected(bad):
    with pytest.raises(ValueError, match="order"):
        _parse_order(bad)


def test_portfolio_argument_validation():
    m = _two_iis_model()
    with pytest.raises(ValueError, match="include_integrality"):
        compute_iis(m, portfolio=("ICB",))
    with pytest.raises(ValueError, match="duplicate"):
        compute_iis(m, include_integrality=True, portfolio=("ICB", "ICB"))
    with pytest.raises(ValueError, match="workers"):
        compute_iis(m, include_integrality=True, portfolio=("ICB",), workers=0)
    with pytest.raises(ValueError, match="at least one"):
        compute_iis(m, include_integrality=True, portfolio=())


def test_unguarded_script_runs_once(tmp_path):
    """Members must not re-import the caller's ``__main__`` (a ``spawn`` child
    would, re-running an unguarded script once per member)."""
    import subprocess
    import sys

    marker = tmp_path / "runs.txt"
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import discopt.modeling as dm\n"
        "from discopt.infeasibility import compute_iis\n"
        f"open({str(marker)!r}, 'a').write('x')\n"
        "m = dm.Model('t')\n"
        "x = m.continuous('x', lb=-5, ub=5)\n"
        "m.subject_to(x <= 0)\n"
        "m.subject_to(x >= 1)\n"
        "m.minimize(x)\n"
        "r = compute_iis(m, include_integrality=True, portfolio=('ICB', 'CIB'))\n"
        "print('SIZE', len(r))\n"
    )
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert "SIZE 2" in p.stdout
    assert marker.read_text() == "x"
