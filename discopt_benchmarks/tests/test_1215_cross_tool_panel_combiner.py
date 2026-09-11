"""The cross-tool panel's combiner must refuse to divide unlike models (#1215).

``scripts/issue1215_cross_tool_panel.py`` compares discopt, Pyomo and oximo on
identical mathematics. Its whole validity rests on one gate: every arm reads the
row and variable counts back out of the header of the ``.nl`` *it just wrote*,
and the combiner refuses to report a ratio unless the arms agree on both. An arm
that quietly built a smaller model would otherwise look fast.

CLAUDE.md's measurement discipline (§6, "prove the probe fired") says an
instrument whose check can silently degrade to a no-op is worse than no
instrument. These tests are that proof for the combiner: the gate fires, it
counts what it compared, and a malformed input raises instead of being skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

pytest.importorskip("pyomo.environ", reason="panel's Pyomo arm needs pyomo")

from scripts import issue1215_cross_tool_panel as panel  # noqa: E402


def _row(tool, idiom, family, rows, vars_, construct_s=1.0, write_s=1.0):
    return {
        "tool": tool,
        "family": family,
        "idiom": idiom,
        "rows": rows,
        "vars": vars_,
        "construct_s": construct_s,
        "write_s": write_s,
    }


def _panel(vars_oximo=3000):
    return [
        _row("discopt", "vectorised", "linear", 1000, 3000, 0.001, 0.004),
        _row("discopt", "per-element", "linear", 1000, 3000, 0.020, 0.014),
        _row("pyomo", "per-element", "linear", 1000, 3000, 0.018, 0.007),
        _row("oximo", "per-element", "linear", 1000, vars_oximo, 0.0014, 0.0013),
    ]


def test_reports_one_cell_per_arm(capsys):
    cells = panel._report(_panel())
    assert cells == 4
    out = capsys.readouterr().out
    assert "discopt vectorised" in out and "oximo" in out
    # 5.0 us/row total for the vectorised arm: (0.001 + 0.004) s over 1000 rows.
    assert "5.00" in out


def test_ratio_table_is_relative_to_oximo(capsys):
    panel._report(_panel())
    out = capsys.readouterr().out
    assert "ratio to oximo" in out
    assert "1.00x" in out  # oximo against itself
    assert "12.59x" in out  # per-element discopt: 0.034 / 0.0027


def test_identity_gate_refuses_a_smaller_model():
    with pytest.raises(AssertionError, match="not the same model"):
        panel._report(_panel(vars_oximo=2999))


def test_identity_gate_reports_how_many_comparisons_it_made(capsys):
    panel._report(_panel())
    err = capsys.readouterr().err
    # 4 arms on one cell -> 3 comparisons against the reference arm. A gate that
    # silently compared nothing would print 0 here.
    assert "# model-identity comparisons: 3" in err


def test_duplicate_measurement_is_an_error():
    rows = _panel()
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate measurement"):
        panel._report(rows)


def test_parse_tsv_round_trips(tmp_path):
    p = tmp_path / "arm.tsv"
    p.write_text(
        "\t".join(panel.COLS) + "\n"
        "oximo\tlinear\tper-element\t1000\t3000\t0.001441\t0.001303\n"
        "# a comment line is not a measurement\n"
    )
    (row,) = panel._parse_tsv(p)
    assert row["rows"] == 1000 and row["vars"] == 3000
    assert row["construct_s"] == pytest.approx(0.001441)


def test_parse_tsv_rejects_a_foreign_header(tmp_path):
    p = tmp_path / "arm.tsv"
    p.write_text("tool\tfamily\trows\noximo\tlinear\t1000\n")
    with pytest.raises(ValueError, match="header is"):
        panel._parse_tsv(p)


def test_parse_tsv_rejects_a_short_row_instead_of_skipping_it(tmp_path):
    p = tmp_path / "arm.tsv"
    p.write_text("\t".join(panel.COLS) + "\n" + "oximo\tlinear\tper-element\t1000\n")
    with pytest.raises(ValueError, match="2: 4 fields"):
        panel._parse_tsv(p)


def test_parse_tsv_rejects_a_file_with_no_measurements(tmp_path):
    p = tmp_path / "arm.tsv"
    p.write_text("\t".join(panel.COLS) + "\n")
    with pytest.raises(ValueError, match="no data rows"):
        panel._parse_tsv(p)


# ── memory mode ─────────────────────────────────────────────────────────────
#
# The memory mode shares the identity gate and the table machinery with the
# timing mode, so what is tested here is what differs: the column set, the
# per-row divisor, and the RSS-resolution note. Without that note a model too
# small to register against a 1 kB VmRSS reading prints "0", which reads as
# "free" rather than "unmeasurable".


def _mem_row(tool, idiom, family, rows, vars_, retained_b):
    return {
        "tool": tool,
        "family": family,
        "idiom": idiom,
        "rows": rows,
        "vars": vars_,
        "retained_b": retained_b,
    }


def _mem_panel():
    return [
        _mem_row("discopt", "vectorised", "linear", 100000, 300000, 1_700_000),
        _mem_row("discopt", "per-element", "linear", 100000, 300000, 130_100_000),
        _mem_row("oximo", "per-element", "linear", 100000, 300000, 107_200_000),
    ]


def test_memory_mode_reports_bytes_per_row(capsys):
    cells = panel._report(_mem_panel(), mode="memory")
    assert cells == 3
    out = capsys.readouterr().out
    assert "retained B/row" in out
    assert "17" in out  # 1_700_000 / 100_000
    assert "1072" in out  # oximo


def test_memory_mode_ratio_is_relative_to_oximo(capsys):
    panel._report(_mem_panel(), mode="memory")
    out = capsys.readouterr().out
    assert "0.02x" in out  # vectorised discopt against oximo
    assert "1.21x" in out  # per-element discopt against oximo


def test_memory_mode_flags_a_cell_below_rss_resolution(capsys):
    rows = [
        _mem_row("discopt", "vectorised", "linear", 1000, 3000, 0),
        _mem_row("oximo", "per-element", "linear", 1000, 3000, 1_188_000),
    ]
    panel._report(rows, mode="memory")
    out = capsys.readouterr().out
    assert "under the 1024 B RSS resolution" in out
    assert "not as zero" in out


def test_memory_mode_identity_gate_still_fires():
    rows = _mem_panel()
    rows[-1]["vars"] = 299999
    with pytest.raises(AssertionError, match="not the same model"):
        panel._report(rows, mode="memory")


def test_memory_tsv_round_trips(tmp_path):
    p = tmp_path / "mem.tsv"
    p.write_text(
        "\t".join(panel.MEM_COLS) + "\noximo\tlinear\tper-element\t100000\t300000\t107241472\n"
    )
    (row,) = panel._parse_tsv(p, panel.MEM_COLS)
    assert row["retained_b"] == 107241472


def test_memory_tsv_rejects_a_timing_tsv(tmp_path):
    """The two modes' TSVs must not be silently interchangeable."""
    p = tmp_path / "time.tsv"
    p.write_text(
        "\t".join(panel.COLS) + "\noximo\tlinear\tper-element\t1000\t3000\t0.001441\t0.001303\n"
    )
    with pytest.raises(ValueError, match="header is"):
        panel._parse_tsv(p, panel.MEM_COLS)
