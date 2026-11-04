"""Deletion-order portfolio for ``compute_iis(include_integrality=True)``.

The deletion filter returns *an* IIS, and which one depends on the order it visits
elements. ``portfolio=`` runs one filter per order in separate processes against
one budget and keeps the smallest complete result. Each member is an ordinary
single-order run, so these tests check the plumbing: that the order really
changes which IIS is found, that the smallest one wins, that every member's
result is reported, and that failures surface instead of disappearing.
"""

import sys
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
    with pytest.raises(ValueError, match="workers"):
        compute_iis(m, include_integrality=True, workers=2)  # no portfolio
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


def test_all_members_timing_out_reports_each_members_reason(monkeypatch):
    m = _two_iis_model()
    with pytest.raises(TimeoutError, match=r"\(ICB: .+; CIB: .+\)"):
        compute_iis(m, include_integrality=True, portfolio=("ICB", "CIB"), total_time_limit=1e-3)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="scans /proc")
def test_members_do_not_outlive_a_killed_caller(tmp_path):
    """A member blocks on the parent's pipe and exits on EOF: killing the caller
    must not leave members running (they did, at 100% CPU, before)."""
    import os
    import signal
    import subprocess
    import uuid

    marker = f"IIS_PORTFOLIO_TEST_{uuid.uuid4().hex}"
    script = tmp_path / "caller.py"
    # Pigeonhole 9 into 8: members stay busy for a while.
    script.write_text(
        "import discopt.modeling as dm\n"
        "from discopt.infeasibility import compute_iis\n"
        "m = dm.Model('php')\n"
        "x = m.binary('x', shape=(9, 8))\n"
        "for i in range(9):\n"
        "    m.subject_to(sum(x[i, j] for j in range(8)) >= 1)\n"
        "for j in range(8):\n"
        "    for a in range(9):\n"
        "        for b in range(a + 1, 9):\n"
        "            m.subject_to(x[a, j] + x[b, j] <= 1)\n"
        "m.minimize(0 * x[0, 0])\n"
        "compute_iis(m, include_integrality=True, portfolio=('ICB', 'CIB'))\n"
    )
    env = dict(os.environ, **{marker: "1"})
    caller = subprocess.Popen([sys.executable, str(script)], env=env)

    def members():
        found = []
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == caller.pid:
                continue
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    envb = f.read()
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read()
            except OSError:  # exited or not ours
                continue
            if marker.encode() in envb and b"_portfolio_child" in cmd:
                found.append(int(pid))
        return found

    deadline = time.time() + 30
    while not members() and time.time() < deadline:
        time.sleep(0.1)
    assert members(), "members never started"
    caller.send_signal(signal.SIGKILL)
    caller.wait()
    deadline = time.time() + 10
    while members() and time.time() < deadline:
        time.sleep(0.1)
    assert not members()


# Every order the parser accepts: the 6 permutations, forward and reversed.
_ALL_ORDERS = [p + s for p in ("ICB", "IBC", "CIB", "CBI", "BIC", "BCI") for s in ("", "-r")]


def _bounds_released_by_integrality():
    """LP-infeasible, so the I phase releases x and y; the IIS needs only the uppers.

    A held-integer column's box is not an element, so a bounds phase that runs
    *before* I skips x's and y's bounds. Before the fix (PR #2110 review), orders
    with B before I returned all four bounds as ``proven_irreducible=True`` although
    ``x >= 0`` and ``y >= 0`` are removable.
    """
    m = dm.Model("bci")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.subject_to(x + y >= 20, name="c0")
    m.minimize(0 * x)
    return m


def _integrality_needed():
    """LP-feasible, integer-infeasible: the IIS is {c0, integrality of x}."""
    m = dm.Model("int_needed")
    x = m.integer("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(2 * x == 1, name="c0")
    m.subject_to(y <= 3, name="c1")
    m.minimize(y)
    return m


@pytest.mark.parametrize("confirm", ["each", "final"])
@pytest.mark.parametrize("order", _ALL_ORDERS)
@pytest.mark.parametrize("build", [_bounds_released_by_integrality, _integrality_needed])
def test_every_order_returns_an_irreducible_set(build, order, confirm):
    from discopt._iis_milp import milp_iis

    ref = milp_iis(build(), order="ICB", confirm="each")
    r = milp_iis(build(), order=order, confirm=confirm)
    assert ref.complete and ref.proven_irreducible
    assert r.complete
    # A proven result must be the same irreducible set the reference finds: on these
    # models the IIS is unique, so any extra element is a removable one.
    assert (r.constraints, r.bounds, r.integrality) == (
        ref.constraints,
        ref.bounds,
        ref.integrality,
    )
    assert r.proven_irreducible
    # Every reported bound carries a certificate (it was put through the filter).
    for j, s in r.bounds:
        assert ("bound", j, s) in r.certificates


def test_member_refuses_a_different_discopt(monkeypatch):
    """A member checks it imported the parent's discopt (CLAUDE.md s8): a
    rebuilt sys.path must not silently load another copy."""
    from discopt import _iis_milp

    monkeypatch.setattr(_iis_milp, "_DISCOPT_DIR", "/nonexistent/discopt")
    with pytest.raises(RuntimeError, match="imported discopt from"):
        compute_iis(_two_iis_model(), include_integrality=True, portfolio=("ICB",))


def test_grace_extra_is_clamped_to_the_window():
    """MIPcc27 sweep: every later member that beat the first finisher did so
    within 31 s of it. A proportional wait (grace * t) cut improvements off on
    fast instances and idled for minutes on slow ones."""
    from discopt._iis_milp import _grace_extra

    assert _grace_extra(4.0, 2.0) == 4.0  # proportional, unclamped
    assert _grace_extra(4.0, 2.0, (10.0, 30.0)) == 10.0  # floor: bell3ainf's 124 at +13 s
    assert _grace_extra(20.0, 2.0, (10.0, 30.0)) == 20.0
    assert _grace_extra(285.0, 2.0, (10.0, 30.0)) == 30.0  # neos859080: was +285 s


def test_grace_window_argument_validation():
    import math

    import numpy as np
    import scipy.sparse as sp
    from discopt._iis_milp import milp_iis_arrays

    arrays = (
        sp.csr_matrix(np.array([[1.0]])),
        np.array([1.0]),
        np.array([1e20]),
        {0: np.array([0])},
        np.array([0.0]),
        np.array([0.5]),
        np.array([False]),
    )
    with pytest.raises(ValueError, match="grace_window"):
        milp_iis_arrays(arrays, portfolio=("ICB",), grace=math.inf, grace_window=(1.0, 2.0))
    with pytest.raises(ValueError, match="grace_window"):
        milp_iis_arrays(arrays, portfolio=("ICB",), grace=2.0, grace_window=(5.0, 2.0))
    with pytest.raises(ValueError, match="need portfolio"):
        milp_iis_arrays(arrays, grace_window=(1.0, 2.0))


def test_the_entry_program_uses_the_fast_wait(tmp_path, monkeypatch):
    from discopt import mipcc

    seen = {}
    import discopt._iis_milp as engine

    def capture(*a, **k):
        seen.update(k)
        raise TimeoutError("stop here")

    monkeypatch.setattr(engine, "milp_iis_arrays", capture)
    m = _two_iis_model()
    path = tmp_path / "m.mps"
    m.to_mps(str(path))
    monkeypatch.setenv("MIPCC_TIME_LIMIT", "60")
    assert mipcc.main([str(path), str(tmp_path / "s.json")]) == 0
    assert (seen["grace"], seen["grace_window"]) == (2.0, (10.0, 30.0))


@pytest.mark.parametrize(
    ("wait", "expected"), [("fast", (2.0, (10.0, 30.0))), ("all", (float("inf"), None))]
)
def test_compute_iis_portfolio_wait_reaches_the_portfolio(monkeypatch, wait, expected):
    import discopt._iis_milp as engine

    seen = {}
    real = engine._portfolio_iis

    def spy(*a, **k):
        seen["grace"], seen["window"] = k["grace"], k["grace_window"]
        return real(*a, **k)

    monkeypatch.setattr(engine, "_portfolio_iis", spy)
    res = compute_iis(
        _two_iis_model(),
        include_integrality=True,
        include_bounds=False,
        portfolio=("ICB", "ICB-r"),
        portfolio_wait=wait,
    )
    assert (seen["grace"], seen["window"]) == expected
    assert _names(res) == ["c0", "c1"]


def test_portfolio_wait_is_validated():
    with pytest.raises(ValueError, match="portfolio_wait"):
        compute_iis(
            _two_iis_model(), include_integrality=True, portfolio=("ICB",), portfolio_wait="slow"
        )


def test_flag_i_selects_the_integrality_alone_phase():
    from discopt._iis_milp import _relaxes_columns

    assert _relaxes_columns("ICB") and _relaxes_columns("CIB-rl")
    assert not _relaxes_columns("ICB-i") and not _relaxes_columns("CIB-ri")
    assert "ICB-i" in DEFAULT_PORTFOLIO and "ICB" in DEFAULT_PORTFOLIO
    with pytest.raises(ValueError, match="order"):
        _relaxes_columns("ICB-ii")


def test_column_relaxation_drops_bounds_with_integrality_and_ICB_i_does_not():
    """x0 + x1 + x2 = 1.5 over binaries plus integer columns in no row: the
    column phase drops each irrelevant column's integrality AND bounds in the
    same probe; integrality alone leaves the bounds to the bound phase. Both
    finish at the same irreducible set."""
    from discopt._iis_milp import milp_iis

    m = dm.Model("cols")
    x = m.binary("x", shape=(3,))
    m.subject_to(x[0] + x[1] + x[2] == 1.5, name="half")
    m.integer("y", shape=(8,), lb=0, ub=3)
    m.minimize(0 * x[0])
    a = milp_iis(m, order="ICB", sat_candidate=False)
    b = milp_iis(m, order="ICB-i", sat_candidate=False)
    assert (a.constraints, a.bounds, sorted(a.integrality)) == (
        b.constraints,
        b.bounds,
        sorted(b.integrality),
    )
    assert sorted(a.integrality) == [0, 1, 2] and a.bounds == []
    assert a.n_solves < b.n_solves  # bounds went with the integralities, not one by one
