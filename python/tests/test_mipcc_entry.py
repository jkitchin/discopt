"""The MIPcc27 entry program (``python -m discopt.mipcc INSTANCE SOLUTION``).

The competition discards a run that exits non-zero, times out, or leaves no
solution, and rejects a whole file for one wrong certificate
(``PROGRAM_INTERFACE.md`` / ``SOLUTION_FORMAT.md`` of the MIPcc27 kit). These
tests pin those properties: the written file follows the format (relaxed
elements listed, kept ones certified by the checker's own rule), a failing or
out-of-time engine still exits 0 with the whole model as the answer, and the
watchdog ends the process before the limit.
"""

import json
import os
import subprocess
import sys
import textwrap

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import mipcc

highspy = pytest.importorskip("highspy")


def _readme_example(tmp_path):
    """SOLUTION_FORMAT.md's example: c1 and c2 conflict, c3 is redundant."""
    m = dm.Model("ex")
    x1 = m.integer("x1", lb=0, ub=10)
    x2 = m.integer("x2", lb=0, ub=10)
    m.subject_to(x1 + x2 <= 1.5, name="c1")
    m.subject_to(x1 + x2 >= 2.5, name="c2")
    m.subject_to(x1 - x2 <= 0, name="c3")
    m.minimize(0 * x1)
    path = tmp_path / "ex.mps"
    m.to_mps(str(path))
    return str(path)


def _check(data, doc):
    """Independent re-implementation of the checker's certificate rule on the
    written file (names, not indices); returns the number of certificates checked."""
    rn, cn = data["row_names"], data["col_names"]
    col = {c: j for j, c in enumerate(cn)}
    A = data["A"].toarray()
    rows = doc.get("rows", {})
    cols = doc.get("columns", {})

    def kept(entry):
        return entry.get("action", "keep") == "keep"

    keep_row = [kept(rows.get(r, {})) for r in rn]
    held = [
        bool(data["is_int"][j]) and kept(cols.get(c, {}).get("integrality", {}))
        for j, c in enumerate(cn)
    ]
    lb_k = [held[j] or kept(cols.get(c, {}).get("lower_bound", {})) for j, c in enumerate(cn)]
    ub_k = [held[j] or kept(cols.get(c, {}).get("upper_bound", {})) for j, c in enumerate(cn)]

    def viol(x, skip):
        act = A @ x
        v = []
        for i, r in enumerate(rn):
            if keep_row[i] and ("row", r) != skip:
                lo, hi = data["row_lo"][i], data["row_hi"][i]
                v.append(max(lo - act[i] if lo > -1e20 else 0, act[i] - hi if hi < 1e20 else 0))
        for j, c in enumerate(cn):
            if lb_k[j] and ("lb", c) != skip and data["col_lb"][j] > -1e20:
                v.append(data["col_lb"][j] - x[j])
            if ub_k[j] and ("ub", c) != skip and data["col_ub"][j] < 1e20:
                v.append(x[j] - data["col_ub"][j])
            if held[j] and ("int", c) != skip:
                v.append(abs(x[j] - round(x[j])))
        return max(v, default=0.0)

    def point(cert):
        x = np.zeros(len(cn))
        for name, val in cert.items():
            x[col[name]] = val
        return x

    n = 0
    for r, e in rows.items():
        if "certificate" in e:
            x = point(e["certificate"])
            i = rn.index(r)
            act = A[i] @ x
            own = max(data["row_lo"][i] - act, act - data["row_hi"][i])
            assert viol(x, ("row", r)) <= 1e-5 and own > 1e-5, r
            n += 1
    for c, e in cols.items():
        j = col[c]
        for key, tag in (("lower_bound", "lb"), ("upper_bound", "ub"), ("integrality", "int")):
            if "certificate" in e.get(key, {}):
                x = point(e[key]["certificate"])
                if tag == "int":
                    held[j] = False
                    own = abs(x[j] - round(x[j]))
                    assert viol(x, ("int", c)) <= 1e-5 and own > 1e-5, c
                    held[j] = True
                else:
                    own = data["col_lb"][j] - x[j] if tag == "lb" else x[j] - data["col_ub"][j]
                    assert viol(x, (tag, c)) <= 1e-5 and own > 1e-5, c
                n += 1
    return n


@pytest.mark.smoke
def test_readme_example_gives_a_certified_iis_in_the_competition_format(tmp_path, monkeypatch):
    inst = _readme_example(tmp_path)
    out = tmp_path / "sol.json"
    monkeypatch.setenv("MIPCC_TIME_LIMIT", "60")
    monkeypatch.setenv("MIPCC_THREADS", "2")
    assert mipcc.main([inst, str(out)]) == 0
    doc = json.loads(out.read_text())
    assert set(doc) <= {"rows", "columns"}
    rows = doc["rows"]
    assert rows["c3"] == {"action": "relax"}
    assert rows["c1"]["action"] == "keep" and rows["c2"]["action"] == "keep"
    # Integralities and the integer columns' bounds are not needed: relaxed and
    # listed as such (unlisted would mean "keep").
    for c in ("x1", "x2"):
        assert doc["columns"][c]["integrality"] == {"action": "relax"}
        assert doc["columns"][c]["lower_bound"] == {"action": "relax"}
        assert doc["columns"][c]["upper_bound"] == {"action": "relax"}
    assert _check(mipcc.read_mps(inst), doc) == 2  # both kept rows certified


def test_certificate_rule_matches_the_checker():
    data = dict(
        A=__import__("scipy.sparse").sparse.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]])),
        row_lo=np.array([-1e20, 2.5]),
        row_hi=np.array([1.5, 1e20]),
        col_lb=np.array([0.0, 0.0]),
        col_ub=np.array([10.0, 10.0]),
        is_int=np.array([False, False]),
    )
    rows = np.array([True, True])
    none = np.zeros(2, dtype=bool)
    ok = mipcc.certificate_ok
    # Drops c1: (3, 0) satisfies c2 and violates c1 by 1.5.
    assert ok(data, rows, none, none, none, ("constraint", 0), np.array([3.0, 0.0]))
    # Within the 1e-5 tolerance of c1: it does not violate c1, so it proves nothing.
    assert not ok(data, rows, none, none, none, ("constraint", 0), np.array([1.5 + 5e-6, 0.0]))
    # Violates the other kept row (c2) by more than the tolerance.
    assert not ok(data, rows, none, none, none, ("constraint", 0), np.array([2.0, 0.0]))
    # Inside the checker's tolerance but at its edge: dropped, not risked. c1
    # violated by 1.5e-5 (> 1e-5, but not > 2e-5), and c2 satisfied but with
    # another row violated by 0.8e-5 (<= 1e-5, but not <= 0.5e-5).
    only_c1 = np.array([True, False])
    assert not ok(data, only_c1, none, none, none, ("constraint", 0), np.array([1.5 + 1.5e-5, 0.0]))
    assert ok(data, only_c1, none, none, none, ("constraint", 0), np.array([1.5 + 3e-5, 0.0]))
    data3 = dict(data, row_lo=np.array([-1e20, 2.5 + 0.8e-5]))
    assert not ok(data3, rows, none, none, none, ("constraint", 0), np.array([3.0, -0.5]))


@pytest.mark.parametrize("exc", [TimeoutError("budget"), RuntimeError("engine bug")])
def test_engine_failure_still_exits_zero_with_the_whole_model(tmp_path, monkeypatch, exc, capsys):
    inst = _readme_example(tmp_path)
    out = tmp_path / "sol.json"

    def boom(*a, **k):
        raise exc

    import discopt._iis_milp as engine

    monkeypatch.setattr(engine, "milp_iis_arrays", boom)
    monkeypatch.setenv("MIPCC_TIME_LIMIT", "60")
    assert mipcc.main([inst, str(out)]) == 0
    assert json.loads(out.read_text()) == {}  # every element kept: the full model
    assert "full model" in capsys.readouterr().out


def test_watchdog_exits_before_the_limit(tmp_path):
    inst = _readme_example(tmp_path)
    out = tmp_path / "sol.json"
    script = tmp_path / "slow.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys, time
            import discopt._iis_milp as engine
            from discopt import mipcc
            engine.milp_iis_arrays = lambda *a, **k: time.sleep(600)
            sys.exit(mipcc.main([{inst!r}, {str(out)!r}]))
            """
        )
    )
    env = dict(os.environ, MIPCC_TIME_LIMIT="12")
    p = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60, env=env
    )
    assert p.returncode == 0, p.stderr
    assert "hard stop" in p.stdout
    assert json.loads(out.read_text()) == {}
    lo, hard = mipcc.budget(12.0)
    assert hard < 12.0 and lo < hard


def test_budget_leaves_a_margin():
    for limit in (12.0, 60.0, 600.0, 3600.0):
        engine, hard = mipcc.budget(limit)
        assert engine < hard < limit
    assert mipcc.budget(3600.0)[0] >= 3480.0  # a margin, not a large share of the hour


def test_watchdog_stands_down_after_an_in_process_main(tmp_path):
    """A watchdog left behind by an in-process main() killed the pytest worker
    that ran it, mid-way through an unrelated test, when the hard stop came."""
    inst = _readme_example(tmp_path)
    out = tmp_path / "sol.json"
    script = tmp_path / "inproc.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import time
            from discopt import mipcc
            assert mipcc.main([{inst!r}, {str(out)!r}]) == 0
            time.sleep(10)  # past the hard stop (limit 12 s, launch to here)
            print("ALIVE")
            """
        )
    )
    env = dict(os.environ, MIPCC_TIME_LIMIT="12")
    p = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=90, env=env
    )
    assert p.returncode == 0, p.stderr
    assert "ALIVE" in p.stdout and "hard stop" not in p.stdout


def test_an_in_process_call_long_after_import_keeps_its_budget(tmp_path):
    """The hard stop was measured from the module's import: an in-process call
    made later than the limit after the import found it already past, and the
    watchdog os._exit'ed the caller (a pytest worker, in the smoke run)."""
    inst = _readme_example(tmp_path)
    out = tmp_path / "sol.json"
    script = tmp_path / "late.py"
    script.write_text(
        textwrap.dedent(
            f"""
            from discopt import mipcc
            mipcc._T_LAUNCH -= 1000.0  # imported "1000 s ago", limit 60 s
            assert mipcc.main([{inst!r}, {str(out)!r}]) == 0
            print("ALIVE")
            """
        )
    )
    env = dict(os.environ, MIPCC_TIME_LIMIT="60")
    p = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, env=env
    )
    assert p.returncode == 0, p.stderr
    assert "ALIVE" in p.stdout and "hard stop" not in p.stdout
    assert json.loads(out.read_text())["rows"]["c3"] == {"action": "relax"}


def test_checker_settings_catch_a_subsystem_feasible_within_tolerance():
    """#2145 review (borderline zone): a set infeasible in truth -- an integer
    column with ub = 2.9999999 and a row x >= 3 -- is feasible within the
    checker's 1e-5, so submitting it would fail the run. The entry re-solves the
    final set with the checker's own settings and rejects it; a clearly
    infeasible set is accepted."""
    from types import SimpleNamespace

    import scipy.sparse as sp

    def data(ub):
        return dict(
            A=sp.csr_matrix(np.array([[1.0]])),
            row_lo=np.array([3.0]),
            row_hi=np.array([1e20]),
            col_lb=np.array([0.0]),
            col_ub=np.array([ub]),
            is_int=np.array([True]),
        )

    res = SimpleNamespace(constraints=[0], bounds=[], integrality=[0])
    assert mipcc.checker_rejects(data(2.9999999), res, 30.0) is True
    assert mipcc.checker_rejects(data(2.5), res, 30.0) is False


def test_the_answer_is_written_before_the_checker_settings_solve(tmp_path, monkeypatch, capsys):
    """#2145 follow-up: the checker-settings solve ran before the answer was
    written, so an overrun into the watchdog (or an exception) left only the
    full-model fallback on disk. Now the answer is on disk while it runs."""
    inst = _readme_example(tmp_path)
    out = tmp_path / "sol.json"
    seen = {}

    def checking(data, res, time_limit):
        seen["on_disk"] = json.loads(out.read_text())
        seen["time_limit"] = time_limit
        raise RuntimeError("checker-settings solve failed")

    monkeypatch.setattr(mipcc, "checker_rejects", checking)
    monkeypatch.setenv("MIPCC_TIME_LIMIT", "3600")
    assert mipcc.main([inst, str(out)]) == 0  # the failure is reported, exit 0
    assert seen["on_disk"]["rows"]["c3"] == {"action": "relax"}  # the answer, not {}
    assert json.loads(out.read_text()) == seen["on_disk"]  # and it stays
    # ... and the failure message names what is on disk, not "the full model".
    printed = capsys.readouterr().out
    assert "answer is the confirmed subsystem" in printed
    assert "full model" not in printed
    assert seen["time_limit"] <= mipcc.CHECKER_SOLVE_CAP_S
