"""Tests for the ``discopt doe gui`` launcher + app module.

These intentionally avoid spawning a real Streamlit server. The app
is exercised only via import (catches syntax errors and stale
imports). The launcher is exercised through its ``spawn=False`` mode
so we cover the env/command-line wiring without ever touching a
subprocess or a TCP port the test runner doesn't own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

pytest.importorskip("streamlit")
pytest.importorskip("pandas")
pytest.importorskip("openpyxl")


def test_app_module_imports() -> None:
    """Importing the Streamlit app must not raise (catches refactors)."""
    from discopt.doe.gui import app

    assert callable(app.main)


def test_launcher_reports_missing_workbook(tmp_path: Path, capsys) -> None:
    from discopt.doe.gui.launcher import launch

    rc = launch(workbook=tmp_path / "nope.xlsx", spawn=False, open_browser=False)
    err = capsys.readouterr().err
    assert rc == 1
    assert "workbook not found" in err


def test_launcher_with_existing_workbook(tmp_path: Path, monkeypatch, capsys) -> None:
    """spawn=False resolves args + env without launching streamlit."""
    from discopt.doe.cli import NewParams, do_new
    from discopt.doe.gui.launcher import launch

    wb_path = tmp_path / "campaign.xlsx"
    do_new(
        NewParams(
            output=wb_path,
            n=3,
            inputs=[("x", 0.0, 10.0)],
            response_name="y",
            measurement_error=1.0,
            criterion="determinant",
            seed=0,
            n_starts=3,
            template="linear",
        )
    )

    rc = launch(workbook=wb_path, spawn=False, open_browser=False)
    err = capsys.readouterr().err
    assert rc == 0
    assert "Starting discopt doe GUI at http://127.0.0.1:" in err


def test_launcher_no_workbook(capsys) -> None:
    from discopt.doe.gui.launcher import launch

    rc = launch(workbook=None, spawn=False, open_browser=False)
    err = capsys.readouterr().err
    assert rc == 0
    assert "http://127.0.0.1:" in err


def test_cli_gui_help_registered() -> None:
    """`discopt doe gui --help` should not error out at argparse time."""
    import argparse

    from discopt.doe.cli import add_subparser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_subparser(sub)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["doe", "gui", "--help"])
    assert exc.value.code == 0


def test_app_runs_against_real_workbook(monkeypatch, tmp_path: Path) -> None:
    """Drive the Streamlit script through AppTest against a real workbook.

    Catches runtime errors that import-only smoke tests miss: rendering
    every panel (status banner, runs editor, fit panel, extend panel,
    history, download). No browser, no subprocess — `AppTest` runs the
    script in-process.
    """
    pytest.importorskip("streamlit.testing.v1")
    from discopt.doe.cli import NewParams, do_new
    from streamlit.testing.v1 import AppTest

    wb_path = tmp_path / "gui_smoke.xlsx"
    do_new(
        NewParams(
            output=wb_path,
            n=4,
            inputs=[("x", 0.0, 10.0)],
            response_name="y",
            measurement_error=0.5,
            criterion="determinant",
            seed=0,
            n_starts=3,
            template="linear",
        )
    )
    monkeypatch.setenv("DISCOPT_DOE_WORKBOOK", str(wb_path))

    app_path = Path(__file__).parents[1] / "discopt" / "doe" / "gui" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]

    metric_labels = {m.label for m in at.metric}
    assert {"Model", "Completed", "Pending", "Parameters"}.issubset(metric_labels)

    button_labels = {b.label for b in at.button}
    assert "Save responses" in button_labels
    assert "Reload from disk" in button_labels


def test_cli_gui_dispatches_to_launcher(monkeypatch, tmp_path: Path) -> None:
    """The CLI `gui` verb must call `launcher.launch` with parsed args."""
    import argparse

    from discopt.doe.cli import add_subparser, run
    from discopt.doe.gui import launcher

    wb = tmp_path / "x.xlsx"
    wb.write_bytes(b"placeholder")  # launcher checks isfile, not content

    captured: dict = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(launcher, "launch", fake_launch)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_subparser(sub)
    args = parser.parse_args(["doe", "gui", str(wb), "--port", "1234", "--no-browser"])
    rc = run(args)
    assert rc == 0
    assert captured["workbook"] == str(wb)
    assert captured["port"] == 1234
    assert captured["open_browser"] is False
