"""Tests for the ``discopt tutor`` CLI subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from discopt import tutor

# ──────────────────────────────────────────────────────────
# Fixtures: synthetic course tree
# ──────────────────────────────────────────────────────────


def _make_course(root: Path) -> Path:
    """Create a minimal in-memory ``course/`` tree under *root*.

    Mimics the real layout: basic/01..03, intermediate/11, advanced/21.
    """
    course = root / "course"
    (course / "basic" / "01_intro_to_optimization").mkdir(parents=True)
    (course / "basic" / "02_lp_fundamentals").mkdir(parents=True)
    (course / "basic" / "03_modeling_in_discopt").mkdir(parents=True)
    (course / "intermediate" / "11_milp_modeling").mkdir(parents=True)
    (course / "advanced" / "21_spatial_branch_and_bound").mkdir(parents=True)
    for sub in (
        "basic/01_intro_to_optimization",
        "basic/02_lp_fundamentals",
        "basic/03_modeling_in_discopt",
        "intermediate/11_milp_modeling",
        "advanced/21_spatial_branch_and_bound",
    ):
        (course / sub / "reading.ipynb").write_text("{}")
    (course / "SYLLABUS.md").write_text("# syllabus\n")
    (course / "progress.template.yaml").write_text(
        "scores: null\ncurrent_lesson: basic/01_intro_to_optimization\n"
    )
    return course


@pytest.fixture
def fake_course(tmp_path, monkeypatch):
    course = _make_course(tmp_path)
    monkeypatch.setenv("DISCOPT_COURSE_DIR", str(course))
    return course


# ──────────────────────────────────────────────────────────
# _find_course_dir
# ──────────────────────────────────────────────────────────


class TestFindCourseDir:
    def test_env_var(self, tmp_path, monkeypatch):
        course = _make_course(tmp_path)
        monkeypatch.setenv("DISCOPT_COURSE_DIR", str(course))
        assert tutor._find_course_dir() == course.resolve()

    def test_env_var_invalid_falls_back_to_package(self, tmp_path, monkeypatch):
        # An invalid env-var short-circuits to None (doesn't fall back).
        monkeypatch.setenv("DISCOPT_COURSE_DIR", str(tmp_path / "nope"))
        monkeypatch.chdir(tmp_path)
        assert tutor._find_course_dir() is None

    def test_walk_up_from_cwd(self, tmp_path, monkeypatch):
        course = _make_course(tmp_path)
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        monkeypatch.delenv("DISCOPT_COURSE_DIR", raising=False)
        monkeypatch.chdir(deep)
        assert tutor._find_course_dir() == course

    def test_packaged_fallback(self, tmp_path, monkeypatch):
        # Outside any checkout, the packaged course/ is the fallback.
        monkeypatch.delenv("DISCOPT_COURSE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        found = tutor._find_course_dir()
        assert found is not None
        assert (found / "SYLLABUS.md").exists()
        assert found == tutor._packaged_course_dir()

    def test_no_packaged_fallback(self, tmp_path, monkeypatch):
        # If the packaged copy is somehow missing, fallback returns None.
        monkeypatch.delenv("DISCOPT_COURSE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(tutor, "_packaged_course_dir", lambda: None)
        assert tutor._find_course_dir() is None


# ──────────────────────────────────────────────────────────
# _list_lessons / _resolve_lesson
# ──────────────────────────────────────────────────────────


class TestListLessons:
    def test_orders_by_track_then_id(self, fake_course):
        lessons = tutor._list_lessons(fake_course)
        assert lessons == [
            "basic/01_intro_to_optimization",
            "basic/02_lp_fundamentals",
            "basic/03_modeling_in_discopt",
            "intermediate/11_milp_modeling",
            "advanced/21_spatial_branch_and_bound",
        ]

    def test_skips_dirs_without_reading_ipynb(self, fake_course):
        (fake_course / "basic" / "99_drafty").mkdir()
        assert "basic/99_drafty" not in tutor._list_lessons(fake_course)


class TestResolveLesson:
    def test_canonical(self, fake_course):
        assert (
            tutor._resolve_lesson(fake_course, "basic/02_lp_fundamentals")
            == "basic/02_lp_fundamentals"
        )

    def test_shorthand_number(self, fake_course):
        assert tutor._resolve_lesson(fake_course, "02") == "basic/02_lp_fundamentals"

    def test_shorthand_full_id(self, fake_course):
        assert (
            tutor._resolve_lesson(fake_course, "02_lp_fundamentals") == "basic/02_lp_fundamentals"
        )

    def test_unknown(self, fake_course):
        assert tutor._resolve_lesson(fake_course, "nonsense") is None

    def test_ambiguous_returns_none(self, fake_course):
        # Add a colliding lesson under another track.
        (fake_course / "intermediate" / "02_dup").mkdir()
        (fake_course / "intermediate" / "02_dup" / "reading.ipynb").write_text("{}")
        assert tutor._resolve_lesson(fake_course, "02") is None


# ──────────────────────────────────────────────────────────
# Progress parsing
# ──────────────────────────────────────────────────────────


class TestProgress:
    def test_missing_file_returns_none(self, fake_course):
        assert tutor._load_progress(fake_course) is None

    def test_yaml_load(self, fake_course):
        pytest.importorskip("yaml")
        (fake_course / "progress.yaml").write_text(
            "current_lesson: basic/02_lp_fundamentals\n"
            "scores:\n"
            "  basic/01_intro_to_optimization:\n"
            "    total: 88\n"
            "  basic/02_lp_fundamentals:\n"
            "    total: 65\n"
        )
        data = tutor._load_progress(fake_course)
        assert data is not None
        assert data["current_lesson"] == "basic/02_lp_fundamentals"
        assert tutor._scores_for(data)["basic/01_intro_to_optimization"]["total"] == 88
        assert tutor._current_lesson(data) == "basic/02_lp_fundamentals"

    def test_is_passed_threshold(self):
        assert tutor._is_passed({"total": 70})
        assert tutor._is_passed({"total": 99})
        assert not tutor._is_passed({"total": 69})
        assert not tutor._is_passed({})
        assert not tutor._is_passed(None)


# ──────────────────────────────────────────────────────────
# _next_lesson
# ──────────────────────────────────────────────────────────


class TestNextLesson:
    def test_no_progress_returns_first(self, fake_course):
        lessons = tutor._list_lessons(fake_course)
        assert tutor._next_lesson(lessons, {}, None) == "basic/01_intro_to_optimization"

    def test_current_unpassed_returns_current(self, fake_course):
        lessons = tutor._list_lessons(fake_course)
        nxt = tutor._next_lesson(lessons, {}, "basic/02_lp_fundamentals")
        assert nxt == "basic/02_lp_fundamentals"

    def test_current_passed_advances(self, fake_course):
        lessons = tutor._list_lessons(fake_course)
        scores = {"basic/02_lp_fundamentals": {"total": 90}}
        nxt = tutor._next_lesson(lessons, scores, "basic/02_lp_fundamentals")
        assert nxt == "basic/03_modeling_in_discopt"

    def test_all_passed(self, fake_course):
        lessons = tutor._list_lessons(fake_course)
        scores = {le: {"total": 100} for le in lessons}
        assert tutor._next_lesson(lessons, scores, lessons[-1]) is None


# ──────────────────────────────────────────────────────────
# Subcommand handlers (without launching claude)
# ──────────────────────────────────────────────────────────


def _ns(**kw):
    return argparse.Namespace(**kw)


class TestDashboard:
    def test_no_progress(self, fake_course, capsys):
        rc = tutor._cmd_dashboard(_ns(), fake_course)
        out = capsys.readouterr().out
        assert rc == 0
        assert "basic" in out
        assert "0 / 3" in out
        assert "next lesson:" in out
        assert "basic/01_intro_to_optimization" in out

    def test_with_progress(self, fake_course, capsys):
        pytest.importorskip("yaml")
        (fake_course / "progress.yaml").write_text(
            "current_lesson: basic/02_lp_fundamentals\n"
            "scores:\n"
            "  basic/01_intro_to_optimization:\n"
            "    total: 80\n"
        )
        tutor._cmd_dashboard(_ns(), fake_course)
        out = capsys.readouterr().out
        assert "1 / 3" in out
        assert "current lesson:  basic/02_lp_fundamentals" in out


class TestList:
    def test_no_progress_hint(self, fake_course, capsys):
        rc = tutor._cmd_list(_ns(), fake_course)
        out = capsys.readouterr().out
        assert rc == 0
        assert "no progress.yaml yet" in out
        assert "basic/01_intro_to_optimization" in out
        # Unscored lessons show " - "
        assert " - " in out

    def test_with_scores(self, fake_course, capsys):
        pytest.importorskip("yaml")
        (fake_course / "progress.yaml").write_text(
            "current_lesson: basic/02_lp_fundamentals\n"
            "scores:\n"
            "  basic/01_intro_to_optimization:\n"
            "    total: 88\n"
        )
        tutor._cmd_list(_ns(), fake_course)
        out = capsys.readouterr().out
        assert " 88   basic/01_intro_to_optimization" in out
        # Current lesson is marked with an asterisk.
        assert "* " in out


class TestStart:
    def test_unknown_lesson(self, fake_course, capsys):
        rc = tutor._cmd_start(_ns(lesson="nonsense"), fake_course)
        assert rc == 1
        err = capsys.readouterr().err
        assert "unknown lesson" in err

    def test_resolves_and_launches(self, fake_course, monkeypatch):
        calls = []
        monkeypatch.setattr(
            tutor,
            "_launch_slash",
            lambda cmd, *args: calls.append((cmd, args)) or 0,
        )
        rc = tutor._cmd_start(_ns(lesson="02"), fake_course)
        assert rc == 0
        assert calls == [("/course:lesson basic/02_lp_fundamentals", ())]


class TestResume:
    def test_no_progress(self, fake_course, capsys):
        rc = tutor._cmd_resume(_ns(), fake_course)
        assert rc == 1
        assert "no progress.yaml yet" in capsys.readouterr().err

    def test_current_lesson(self, fake_course, monkeypatch):
        pytest.importorskip("yaml")
        (fake_course / "progress.yaml").write_text(
            "current_lesson: basic/03_modeling_in_discopt\nscores: null\n"
        )
        calls = []
        monkeypatch.setattr(tutor, "_launch_slash", lambda cmd, *args: calls.append(cmd) or 0)
        rc = tutor._cmd_resume(_ns(), fake_course)
        assert rc == 0
        assert calls == ["/course:lesson basic/03_modeling_in_discopt"]


class TestNextCmd:
    def test_launches_first_when_no_progress(self, fake_course, monkeypatch):
        calls = []
        monkeypatch.setattr(tutor, "_launch_slash", lambda cmd, *args: calls.append(cmd) or 0)
        rc = tutor._cmd_next(_ns(), fake_course)
        assert rc == 0
        assert calls == ["/course:lesson basic/01_intro_to_optimization"]

    def test_done_with_everything(self, fake_course, capsys):
        pytest.importorskip("yaml")
        (fake_course / "progress.yaml").write_text(
            "current_lesson: advanced/21_spatial_branch_and_bound\n"
            "scores:\n"
            "  basic/01_intro_to_optimization: {total: 100}\n"
            "  basic/02_lp_fundamentals: {total: 100}\n"
            "  basic/03_modeling_in_discopt: {total: 100}\n"
            "  intermediate/11_milp_modeling: {total: 100}\n"
            "  advanced/21_spatial_branch_and_bound: {total: 100}\n"
        )
        rc = tutor._cmd_next(_ns(), fake_course)
        assert rc == 0
        assert "completed every lesson" in capsys.readouterr().out


class TestReset:
    def test_no_progress_file(self, fake_course, capsys):
        rc = tutor._cmd_reset(_ns(lesson=None), fake_course)
        assert rc == 0
        assert "no progress.yaml to reset" in capsys.readouterr().out

    def test_drop_single_lesson(self, fake_course):
        pytest.importorskip("yaml")
        import yaml

        (fake_course / "progress.yaml").write_text(
            "scores:\n"
            "  basic/01_intro_to_optimization: {total: 80}\n"
            "  basic/02_lp_fundamentals: {total: 90}\n"
        )
        rc = tutor._cmd_reset(_ns(lesson="01"), fake_course)
        assert rc == 0
        data = yaml.safe_load((fake_course / "progress.yaml").read_text())
        assert "basic/01_intro_to_optimization" not in (data["scores"] or {})
        assert "basic/02_lp_fundamentals" in data["scores"]

    def test_unknown_lesson(self, fake_course, capsys):
        (fake_course / "progress.yaml").write_text("scores: null\n")
        rc = tutor._cmd_reset(_ns(lesson="nonsense"), fake_course)
        assert rc == 1
        assert "unknown lesson" in capsys.readouterr().err

    def test_refuses_on_packaged_copy(self, monkeypatch, capsys):
        pkg = tutor._packaged_course_dir()
        assert pkg is not None
        rc = tutor._cmd_reset(_ns(lesson=None), pkg)
        assert rc == 1
        assert "read-only packaged copy" in capsys.readouterr().err


class TestInstall:
    def test_missing_assets(self, fake_course, tmp_path, monkeypatch, capsys):
        # fake_course has no _claude_assets/
        monkeypatch.chdir(tmp_path)
        rc = tutor._cmd_install(_ns(force=False), fake_course)
        assert rc == 1
        assert "missing" in capsys.readouterr().err

    def test_copies_claude_assets_and_course_content(self, fake_course, tmp_path, monkeypatch):
        # Build a tiny _claude_assets tree.
        assets = fake_course / "_claude_assets"
        (assets / "commands" / "course").mkdir(parents=True)
        (assets / "commands" / "course" / "lesson.md").write_text("LESSON SLASH\n")
        (assets / "agents").mkdir()
        (assets / "agents" / "tutor.md").write_text("AGENT\n")

        # Install into a separate writable directory.
        work = tmp_path / "workspace"
        work.mkdir()
        monkeypatch.chdir(work)
        rc = tutor._cmd_install(_ns(force=False), fake_course)
        assert rc == 0
        # Claude assets land in .claude/, course/ gets the lesson tree.
        assert (work / ".claude" / "commands" / "course" / "lesson.md").read_text() == (
            "LESSON SLASH\n"
        )
        assert (work / ".claude" / "agents" / "tutor.md").read_text() == "AGENT\n"
        assert (work / "course" / "SYLLABUS.md").exists()
        assert (work / "course" / "basic" / "01_intro_to_optimization" / "reading.ipynb").exists()
        # _claude_assets must NOT be duplicated under course/.
        assert not (work / "course" / "_claude_assets").exists()

    def test_skip_without_force(self, fake_course, tmp_path, monkeypatch, capsys):
        assets = fake_course / "_claude_assets"
        (assets / "commands").mkdir(parents=True)
        (assets / "commands" / "x.md").write_text("from package\n")
        work = tmp_path / "workspace"
        work.mkdir()
        monkeypatch.chdir(work)

        (work / ".claude" / "commands").mkdir(parents=True)
        (work / ".claude" / "commands" / "x.md").write_text("preexisting\n")

        tutor._cmd_install(_ns(force=False), fake_course)
        assert (work / ".claude" / "commands" / "x.md").read_text() == "preexisting\n"
        assert "skip" in capsys.readouterr().out

        tutor._cmd_install(_ns(force=True), fake_course)
        assert (work / ".claude" / "commands" / "x.md").read_text() == "from package\n"

    def test_skips_python_artifacts(self, fake_course, tmp_path, monkeypatch):
        (fake_course / "_claude_assets").mkdir()
        (fake_course / "_claude_assets" / "ok.md").write_text("k\n")
        # Pollute the source with editable-install artifacts.
        (fake_course / "__init__.py").write_text("")
        (fake_course / "basic" / "__pycache__").mkdir()
        (fake_course / "basic" / "__pycache__" / "x.cpython.pyc").write_bytes(b"")
        (fake_course / "basic" / "01_intro_to_optimization" / "stale.pyc").write_bytes(b"")

        work = tmp_path / "workspace"
        work.mkdir()
        monkeypatch.chdir(work)
        tutor._cmd_install(_ns(force=False), fake_course)

        assert not (work / "course" / "__init__.py").exists()
        assert not (work / "course" / "basic" / "__pycache__").exists()
        assert not (work / "course" / "basic" / "01_intro_to_optimization" / "stale.pyc").exists()
        # But real content still copies.
        assert (work / "course" / "SYLLABUS.md").exists()

    def test_no_self_copy_when_install_in_course_dir(
        self, fake_course, tmp_path, monkeypatch, capsys
    ):
        # Running install from inside the course dir itself: don't copy onto self.
        (fake_course / "_claude_assets").mkdir(exist_ok=True)
        (fake_course / "_claude_assets" / "x.md").write_text("hi\n")
        # Install with cwd as the *parent* of fake_course so cwd/course == fake_course.
        monkeypatch.chdir(fake_course.parent)
        rc = tutor._cmd_install(_ns(force=False), fake_course)
        assert rc == 0
        assert "skipping course content copy" in capsys.readouterr().out


# ──────────────────────────────────────────────────────────
# run() dispatcher and argparse wiring
# ──────────────────────────────────────────────────────────


class TestRun:
    def test_missing_course_dir(self, tmp_path, monkeypatch, capsys):
        # Suppress the packaged-fallback so we exercise the not-found branch.
        monkeypatch.delenv("DISCOPT_COURSE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(tutor, "_packaged_course_dir", lambda: None)
        rc = tutor.run(_ns(tutor_cmd=None, tutor_func=tutor._cmd_dashboard))
        assert rc == 1
        assert "course/ directory not found" in capsys.readouterr().err

    def test_dispatches_to_tutor_func(self, fake_course):
        called = {}

        def fake_func(args, course_dir):
            called["course_dir"] = course_dir
            return 7

        rc = tutor.run(_ns(tutor_cmd=None, tutor_func=fake_func))
        assert rc == 7
        assert called["course_dir"] == fake_course


class TestArgparseWiring:
    def test_tutor_help(self):
        import sys

        from discopt.cli import main

        with pytest.raises(SystemExit):
            old = sys.argv[:]
            sys.argv = ["discopt", "tutor", "--help"]
            try:
                main()
            finally:
                sys.argv = old

    def test_tutor_list_runs(self, fake_course, monkeypatch, capsys):
        import sys

        from discopt.cli import main

        monkeypatch.setattr(sys, "argv", ["discopt", "tutor", "list"])
        with pytest.raises(SystemExit) as excinfo:
            main()
        # tutor.run returns 0; main calls sys.exit(...)
        assert (excinfo.value.code or 0) == 0
        assert "basic/01_intro_to_optimization" in capsys.readouterr().out


# ──────────────────────────────────────────────────────────
# _launch_slash claude-missing branch
# ──────────────────────────────────────────────────────────


class TestLaunchSlash:
    def test_no_claude_binary(self, monkeypatch, capsys):
        monkeypatch.setattr(tutor, "_claude_binary", lambda: None)
        rc = tutor._launch_slash("/course:lesson basic/01_intro_to_optimization")
        assert rc == 1
        assert "claude binary not found" in capsys.readouterr().err

    def test_invokes_subprocess(self, monkeypatch):
        captured = {}

        def fake_call(argv):
            captured["argv"] = argv
            return 0

        monkeypatch.setattr(tutor, "_claude_binary", lambda: "/fake/claude")
        monkeypatch.setattr(tutor.subprocess, "call", fake_call)
        rc = tutor._launch_slash("/course:lesson basic/02_lp_fundamentals")
        assert rc == 0
        assert captured["argv"] == ["/fake/claude", "/course:lesson basic/02_lp_fundamentals"]
