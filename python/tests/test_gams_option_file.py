"""The GAMS link reads a discopt option file (``discopt.opt``).

No GAMS installation is needed: GEV/GMO are faked as in
``test_gams_benchmark_contract.py``. The option file is GAMS's usual
``key value`` format; only ``tol``, ``constr_viol_tol`` and
``frontier_memory_mb`` are accepted, and anything else is refused loudly rather
than ignored, as ``Model.solve`` refuses an unknown keyword.
"""

import math

import pytest
from discopt import solver
from discopt.gams import link
from test_gams_benchmark_contract import _fake_gams
from test_gams_link import _FakeGmo

pytestmark = pytest.mark.smoke


def _write(tmp_path, text, name="discopt.opt"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


class TestReadOptionFile:
    def test_gams_syntax_comments_case_and_separators(self, tmp_path):
        path = _write(
            tmp_path,
            "* discopt options for a strict feasibility audit\n"
            "\n"
            "TOL 1e-9\n"
            "constr_viol_tol = 1.0E-09\n"
            "frontier_memory_mb\t30000\n",
        )
        assert link.read_option_file(path) == {
            "pounce_options": {"tol": 1e-9, "constr_viol_tol": 1e-9},
            "frontier_memory_mb": 30000,
        }

    def test_an_empty_file_sets_nothing(self, tmp_path):
        assert link.read_option_file(_write(tmp_path, "* nothing\n\n")) == {}

    def test_zero_memory_budget_is_no_budget_not_an_error(self, tmp_path):
        opts = link.read_option_file(_write(tmp_path, "frontier_memory_mb 0\n"))
        assert opts == {"frontier_memory_mb": 0}

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("tolerance 1e-9\n", "unknown option 'tolerance'"),
            ("max_iter 50\n", "unknown option 'max_iter'"),
            ("tol\n", "exactly one value"),
            ("tol 1e-9 1e-8\n", "exactly one value"),
            ("tol 0\n", "> 0"),
            ("tol -1e-9\n", "> 0"),
            ("tol nan\n", "> 0"),
            ("tol inf\n", "> 0"),
            ("tol tight\n", "must be a number"),
            ("frontier_memory_mb -1\n", ">= 0"),
            ("frontier_memory_mb 2.5\n", "must be an integer"),
            ("tol 1e-9\nTOL 1e-8\n", "set twice"),
        ],
    )
    def test_bad_lines_are_refused_with_their_line(self, tmp_path, text, match):
        path = _write(tmp_path, text)
        with pytest.raises(link.GamsOptionFileError, match=match) as exc:
            link.read_option_file(path)
        assert "discopt.opt line" in str(exc.value)

    def test_the_refusal_names_the_accepted_keys(self, tmp_path):
        with pytest.raises(link.GamsOptionFileError) as exc:
            link.read_option_file(_write(tmp_path, "gap 1e-3\n"))
        for key in link.OPTION_FILE_KEYS:
            assert key in str(exc.value)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(link.GamsOptionFileError, match="cannot read option file"):
            link.read_option_file(str(tmp_path / "absent.opt"))


def _with_option_file(monkeypatch, tmp_path, text):
    """Fake a GAMS job with ``optfile=1`` naming a file holding ``text``."""
    import sys

    _fake_gams(monkeypatch, (10.0, 1e-4, 0, 1))
    gmo = sys.modules["gams.core.gmo"]
    gev = sys.modules["gams.core.gev"]
    path = _write(tmp_path, text)
    gmo.gmoOptFile = lambda h: 1
    gmo.gmoNameOptFile = lambda h: path
    logged, stats = [], {}
    gev.gevLogStat = lambda h, line: logged.append(line)
    gmo.gmoModelStatSet = lambda h, v: stats.__setitem__("model", v)
    gmo.gmoSolveStatSet = lambda h, v: stats.__setitem__("solve", v)
    gmo.gmoUnloadSolutionLegacy = lambda h: None
    return path, logged, stats


def test_control_file_forwards_the_option_file(monkeypatch, tmp_path):
    path, logged, _ = _with_option_file(
        monkeypatch, tmp_path, "tol 1e-9\nconstr_viol_tol 1e-9\nfrontier_memory_mb 30000\n"
    )
    observed = {}

    def solve(view, **kwargs):
        observed.update(kwargs)
        return object(), object()

    monkeypatch.setattr(link, "solve_view", solve)
    assert link.solve_from_control_file("unused", sysdir=str(tmp_path)) == 0
    assert observed["pounce_options"] == {"tol": 1e-9, "constr_viol_tol": 1e-9}
    assert observed["frontier_memory_mb"] == 30000
    assert observed["time_limit"] == 10.0  # the GAMS options still apply
    # The log names what the file set, so a run is traceable to its options.
    assert any(path in ln and "tol=1e-09" in ln for ln in logged), logged


def test_a_refused_option_file_is_a_setup_failure_not_a_default_solve(monkeypatch, tmp_path):
    _, logged, stats = _with_option_file(monkeypatch, tmp_path, "tolerance 1e-9\n")
    called = []
    monkeypatch.setattr(link, "solve_view", lambda *a, **k: called.append(1))
    assert link.solve_from_control_file("unused", sysdir=str(tmp_path)) == 0
    assert called == [], "the model must not be solved at defaults"
    assert stats == {
        "model": link.MODELSTAT_NO_SOLUTION_RETURNED,
        "solve": link.SOLVESTAT_SETUP_FAILURE,
    }
    assert any("unknown option 'tolerance'" in ln for ln in logged), logged


def test_solve_view_passes_pounce_options_and_scopes_the_budget(monkeypatch):
    seen = {}
    real_solve = type(link.model_from_gmo(_FakeGmo())).solve

    def spy(self, **kwargs):
        seen["kwargs"] = kwargs
        seen["budget"] = solver._frontier_memory_bytes()
        return real_solve(self, **kwargs)

    monkeypatch.setattr(type(link.model_from_gmo(_FakeGmo())), "solve", spy)
    monkeypatch.delenv("DISCOPT_FRONTIER_MEMORY_MB", raising=False)
    _, result = link.solve_view(_FakeGmo(), pounce_options={"tol": 1e-9}, frontier_memory_mb=30000)
    assert result.status == "optimal"
    assert seen["kwargs"]["pounce_options"] == {"tol": 1e-9}
    assert seen["budget"] == 30000 << 20
    # Scoped to the one solve: the warm daemon's next job sees the default again.
    assert solver._frontier_memory_bytes() == 2 << 30


def test_solve_view_without_an_option_file_passes_nothing_extra(monkeypatch):
    seen = {}
    cls = type(link.model_from_gmo(_FakeGmo()))
    real_solve = cls.solve

    def spy(self, **kwargs):
        seen.update(kwargs)
        return real_solve(self, **kwargs)

    monkeypatch.setattr(cls, "solve", spy)
    link.solve_view(_FakeGmo())
    assert "pounce_options" not in seen


class TestFrontierMemoryOverride:
    def test_override_beats_the_environment_and_resets(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_FRONTIER_MEMORY_MB", "100")
        assert solver._frontier_memory_bytes() == 100 << 20
        with solver.frontier_memory_override(0):
            assert solver._frontier_memory_bytes() is None
            with solver.frontier_memory_override(7):
                assert solver._frontier_memory_bytes() == 7 << 20
            assert solver._frontier_memory_bytes() is None
        assert solver._frontier_memory_bytes() == 100 << 20

    def test_none_changes_nothing(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_FRONTIER_MEMORY_MB", "100")
        with solver.frontier_memory_override(None):
            assert solver._frontier_memory_bytes() == 100 << 20

    @pytest.mark.parametrize("bad", [-1, 2.5, True, math.nan, "5"])
    def test_bad_values_are_refused(self, bad):
        with pytest.raises(ValueError, match="frontier memory budget"):
            with solver.frontier_memory_override(bad):
                pass

    def test_the_override_reaches_the_tree_caps(self):
        with solver.frontier_memory_override(100):
            assert solver._tree_max_open_nodes(300) == (100 << 20) // (16 * 300 + 256)
