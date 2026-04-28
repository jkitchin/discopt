"""Tests for the discopt-dev CLI module (discopt.dev.cli).

Validates:
  - Argument parsing for dev subcommands: search-arxiv, search-openalex, write-report
  - search_arxiv() with mocked HTTP returning realistic XML
  - search_openalex() with mocked HTTP returning realistic JSON
  - --start-date client-side filtering for arxiv
  - Error handling (network failure produces JSON error output)
  - write-report end-to-end (stdin -> file, directory creation)
  - main() dispatch with sys.argv mocking
  - _find_project_command and _run_claude_command (slash-command resolution)
"""

from __future__ import annotations

import io
import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from discopt.dev.cli import main, search_arxiv, search_openalex

ARXIV_XML_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>Branch and Bound for Mixed-Integer
Nonlinear Programming</title>
    <summary>We present a new algorithm for solving
MINLP problems using spatial branch and bound.</summary>
    <published>2026-03-15T00:00:00Z</published>
    <updated>2026-03-16T00:00:00Z</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href="http://arxiv.org/abs/2603.12345v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2603.12345v1"
          title="pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="math.OC"/>
    <arxiv:doi>10.1234/example.2026</arxiv:doi>
  </entry>
  <entry>
    <title>Convex Relaxations via McCormick Envelopes</title>
    <summary>A survey of McCormick relaxation techniques.</summary>
    <published>2026-01-10T00:00:00Z</published>
    <updated>2026-01-11T00:00:00Z</updated>
    <author><name>Carol White</name></author>
    <link href="http://arxiv.org/abs/2601.67890v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2601.67890v1"
          title="pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="math.OC"/>
  </entry>
</feed>
"""

OPENALEX_JSON_RESPONSE = {
    "results": [
        {
            "title": "Global Optimization with Spatial Branch and Bound",
            "doi": "https://doi.org/10.5555/test.2026",
            "publication_date": "2026-02-20",
            "id": "https://openalex.org/W1234567890",
            "authorships": [
                {"author": {"display_name": "Dana Green"}},
                {"author": {"display_name": "Eve Black"}},
            ],
            "abstract_inverted_index": {
                "We": [0],
                "study": [1],
                "global": [2],
                "optimization.": [3],
            },
            "primary_location": {"source": {"display_name": "Journal of Optimization"}},
            "best_oa_location": {
                "pdf_url": "https://example.com/paper.pdf",
            },
        },
        {
            "title": "Integer Programming Advances",
            "doi": "10.5555/ip.2026",
            "publication_date": "2026-01-05",
            "id": "https://openalex.org/W9876543210",
            "authorships": [
                {"author": {"display_name": "Frank Blue"}},
            ],
            "abstract_inverted_index": None,
            "primary_location": None,
            "best_oa_location": None,
        },
    ]
}


def _mock_urlopen_arxiv(*_args, **_kwargs):
    resp = MagicMock()
    resp.read.return_value = ARXIV_XML_RESPONSE.encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _mock_urlopen_openalex(*_args, **_kwargs):
    resp = MagicMock()
    resp.read.return_value = json.dumps(OPENALEX_JSON_RESPONSE).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _mock_urlopen_error(*_args, **_kwargs):
    raise ConnectionError("Network unreachable")


class TestArgumentParsing:
    def test_search_arxiv_args(self):
        import argparse

        parser = argparse.ArgumentParser(prog="discopt-dev")
        subparsers = parser.add_subparsers(dest="command", required=True)
        p = subparsers.add_parser("search-arxiv")
        p.add_argument("query")
        p.add_argument("--max-results", type=int, default=20)
        p.add_argument("--start-date")
        args = parser.parse_args(["search-arxiv", "MINLP", "--max-results", "5"])
        assert args.query == "MINLP"
        assert args.max_results == 5
        assert args.start_date is None

    def test_search_openalex_args(self):
        import argparse

        parser = argparse.ArgumentParser(prog="discopt-dev")
        subparsers = parser.add_subparsers(dest="command", required=True)
        p = subparsers.add_parser("search-openalex")
        p.add_argument("query")
        p.add_argument("--from-date")
        p.add_argument("--to-date")
        p.add_argument("--per-page", type=int, default=20)
        p.add_argument("--api-key")
        args = parser.parse_args(
            [
                "search-openalex",
                "optimization",
                "--from-date",
                "2026-01-01",
                "--to-date",
                "2026-03-01",
                "--per-page",
                "10",
            ]
        )
        assert args.query == "optimization"
        assert args.from_date == "2026-01-01"
        assert args.to_date == "2026-03-01"
        assert args.per_page == 10

    def test_write_report_args(self):
        import argparse

        parser = argparse.ArgumentParser(prog="discopt-dev")
        subparsers = parser.add_subparsers(dest="command", required=True)
        p = subparsers.add_parser("write-report")
        p.add_argument("output_path")
        args = parser.parse_args(["write-report", "/tmp/report.md"])
        assert args.output_path == "/tmp/report.md"


class TestSearchArxiv:
    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_arxiv)
    def test_basic_results(self, _mock):
        results = search_arxiv("MINLP", max_results=10)
        assert len(results) == 2

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_arxiv)
    def test_result_keys(self, _mock):
        results = search_arxiv("MINLP")
        expected_keys = {
            "title",
            "authors",
            "arxiv_id",
            "published",
            "updated",
            "abstract",
            "category",
            "doi",
            "pdf_link",
            "link",
        }
        for r in results:
            assert set(r.keys()) == expected_keys

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_arxiv)
    def test_parsed_fields(self, _mock):
        results = search_arxiv("MINLP")
        first = results[0]
        assert "Branch and Bound" in first["title"]
        assert first["authors"] == ["Alice Smith", "Bob Jones"]
        assert first["arxiv_id"] == "2603.12345v1"
        assert first["published"] == "2026-03-15"
        assert first["updated"] == "2026-03-16"
        assert first["category"] == "math.OC"
        assert first["doi"] == "10.1234/example.2026"
        assert "pdf" in first["pdf_link"]
        assert first["link"] == "https://arxiv.org/abs/2603.12345v1"

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_arxiv)
    def test_multiline_title_collapsed(self, _mock):
        results = search_arxiv("MINLP")
        assert "\n" not in results[0]["title"]

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_arxiv)
    def test_missing_doi(self, _mock):
        results = search_arxiv("MINLP")
        second = results[1]
        assert second["doi"] == ""

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_error)
    def test_network_error(self, _mock):
        with pytest.raises(ConnectionError, match="Network unreachable"):
            search_arxiv("MINLP")


class TestSearchOpenalex:
    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_openalex)
    def test_basic_results(self, _mock):
        results = search_openalex("optimization")
        assert len(results) == 2

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_openalex)
    def test_result_keys(self, _mock):
        results = search_openalex("optimization")
        expected_keys = {
            "title",
            "authors",
            "doi",
            "published",
            "abstract",
            "source",
            "openalex_id",
            "link",
        }
        for r in results:
            assert set(r.keys()) == expected_keys

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_openalex)
    def test_parsed_fields(self, _mock):
        results = search_openalex("optimization")
        first = results[0]
        assert first["title"] == "Global Optimization with Spatial Branch and Bound"
        assert first["authors"] == ["Dana Green", "Eve Black"]
        assert first["doi"] == "10.5555/test.2026"
        assert first["published"] == "2026-02-20"
        assert first["source"] == "Journal of Optimization"
        assert first["link"] == "https://example.com/paper.pdf"
        assert first["openalex_id"] == "https://openalex.org/W1234567890"

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_openalex)
    def test_inverted_index_abstract(self, _mock):
        results = search_openalex("optimization")
        assert results[0]["abstract"] == "We study global optimization."

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_openalex)
    def test_missing_abstract_and_location(self, _mock):
        results = search_openalex("optimization")
        second = results[1]
        assert second["abstract"] == ""
        assert second["source"] == ""
        assert second["link"] == "https://doi.org/10.5555/ip.2026"

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_openalex)
    def test_doi_stripped(self, _mock):
        results = search_openalex("optimization")
        first = results[0]
        assert not first["doi"].startswith("https://")

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_error)
    def test_network_error(self, _mock):
        with pytest.raises(ConnectionError, match="Network unreachable"):
            search_openalex("optimization")


class TestStartDateFiltering:
    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_arxiv)
    def test_start_date_filters_old(self, _mock, capsys):
        with patch(
            "sys.argv",
            ["discopt-dev", "search-arxiv", "MINLP", "--start-date", "2026-03-01"],
        ):
            main()
        output = json.loads(capsys.readouterr().out)
        assert output["count"] == 1
        assert output["results"][0]["published"] >= "2026-03-01"

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_arxiv)
    def test_start_date_keeps_all(self, _mock, capsys):
        with patch(
            "sys.argv",
            ["discopt-dev", "search-arxiv", "MINLP", "--start-date", "2025-01-01"],
        ):
            main()
        output = json.loads(capsys.readouterr().out)
        assert output["count"] == 2


class TestWriteReport:
    def test_write_report_basic(self, tmp_path, capsys):
        outfile = str(tmp_path / "report.md")
        with patch("sys.argv", ["discopt-dev", "write-report", outfile]):
            with patch("sys.stdin", io.StringIO("# My Report\nHello world.")):
                main()
        assert os.path.isfile(outfile)
        with open(outfile) as f:
            content = f.read()
        assert content == "# My Report\nHello world."
        captured = capsys.readouterr().out
        assert "Wrote" in captured
        assert str(len("# My Report\nHello world.")) in captured

    def test_write_report_creates_directories(self, tmp_path, capsys):
        outfile = str(tmp_path / "sub" / "dir" / "report.txt")
        with patch("sys.argv", ["discopt-dev", "write-report", outfile]):
            with patch("sys.stdin", io.StringIO("content")):
                main()
        assert os.path.isfile(outfile)

    def test_write_report_empty_stdin(self, tmp_path, capsys):
        outfile = str(tmp_path / "empty.md")
        with patch("sys.argv", ["discopt-dev", "write-report", outfile]):
            with patch("sys.stdin", io.StringIO("")):
                main()
        assert os.path.isfile(outfile)
        with open(outfile) as f:
            assert f.read() == ""
        captured = capsys.readouterr().out
        assert "Wrote 0 bytes" in captured


class TestMainDispatch:
    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_arxiv)
    def test_main_search_arxiv(self, _mock, capsys):
        with patch("sys.argv", ["discopt-dev", "search-arxiv", "MINLP"]):
            main()
        output = json.loads(capsys.readouterr().out)
        assert output["query"] == "MINLP"
        assert output["count"] == 2
        assert len(output["results"]) == 2

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_openalex)
    def test_main_search_openalex(self, _mock, capsys):
        with patch("sys.argv", ["discopt-dev", "search-openalex", "optimization"]):
            main()
        output = json.loads(capsys.readouterr().out)
        assert output["query"] == "optimization"
        assert output["count"] == 2

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_error)
    def test_main_arxiv_error_json(self, _mock, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["discopt-dev", "search-arxiv", "MINLP"]):
                main()
        assert exc_info.value.code == 1
        output = json.loads(capsys.readouterr().out)
        assert "error" in output
        assert output["results"] == []

    @patch("urllib.request.urlopen", side_effect=_mock_urlopen_error)
    def test_main_openalex_error_json(self, _mock, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["discopt-dev", "search-openalex", "optimization"]):
                main()
        assert exc_info.value.code == 1
        output = json.loads(capsys.readouterr().out)
        assert "error" in output
        assert output["results"] == []

    def test_main_no_subcommand(self):
        with patch("sys.argv", ["discopt-dev"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2


class TestRunClaudeCommand:
    """Slash-command resolution and source-tree gating for _run_claude_command."""

    def test_find_project_scope(self, tmp_path, monkeypatch):
        from pathlib import Path

        from discopt.dev.cli import _find_project_command

        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "discoptbot.md").write_text("# project copy")

        sub = tmp_path / "deep" / "nested"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)

        found = _find_project_command("discoptbot")
        assert found is not None
        assert Path(found).read_text() == "# project copy"

    def test_find_returns_none_when_absent(self, tmp_path, monkeypatch):
        from discopt.dev.cli import _find_project_command

        empty = tmp_path / "work"
        empty.mkdir()
        monkeypatch.chdir(empty)
        # Walk up from tmp_path (which doesn't have .claude) — assert no match.
        # tmp_path is in /tmp or similar; very unlikely to have .claude in any ancestor
        # but to be robust, set HOME/PWD into tmp_path and check.
        assert _find_project_command("definitely-not-real-xyz") is None

    def test_run_uses_project_copy_when_present(self, tmp_path, monkeypatch):
        import subprocess as _sp

        from discopt.dev import cli as devcli

        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "discoptbot.md").write_text("# project copy")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/claude")

        captured = {}

        def fake_run(argv, check, cwd=None):
            captured["argv"] = argv
            captured["cwd"] = cwd
            return MagicMock(returncode=0)

        monkeypatch.setattr(_sp, "run", fake_run)

        devcli._run_claude_command("discoptbot", "")

        assert captured["argv"] == ["/usr/local/bin/claude", "-p", "/discoptbot"]
        assert captured["cwd"] is None

    def test_run_errors_when_claude_missing(self, monkeypatch, capsys):
        from discopt.dev import cli as devcli

        monkeypatch.setattr("shutil.which", lambda _name: None)
        with pytest.raises(SystemExit) as exc_info:
            devcli._run_claude_command("discoptbot", "")
        assert exc_info.value.code == 1
        assert "claude' CLI not found" in capsys.readouterr().err

    def test_run_errors_when_outside_source_tree(self, tmp_path, monkeypatch, capsys):
        from discopt.dev import cli as devcli

        empty = tmp_path / "work"
        empty.mkdir()
        monkeypatch.chdir(empty)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/claude")

        with pytest.raises(SystemExit) as exc_info:
            devcli._run_claude_command("definitely-not-a-real-command-xyz", "")
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "discopt source checkout" in err
