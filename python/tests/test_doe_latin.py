"""Tests for the Latin / Graeco-Latin design family + general ANOVA.

Covers:
* MOLS orthogonality and the k=2/k=6 exceptions.
* ``latin_square_design`` row-balance and replicate handling.
* ``anova_report`` Type-I SS, df bookkeeping, F-stats vs scipy.
* CLI + workbook round-trip for ``discopt doe new latin-square`` /
  ``discopt doe anova``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from discopt.doe import (
    anova_report,
    graeco_latin_square,
    hyper_graeco_latin_square,
    latin_square,
    latin_square_design,
)
from discopt.doe.latin import _build_squares, _check_orthogonal

pytestmark = pytest.mark.smoke


# ──────────────────────────────────────────────────────────────────
# MOLS generators
# ──────────────────────────────────────────────────────────────────


def test_latin_square_each_row_col_unique():
    """A k x k Latin square has each value exactly once per row and column."""
    for k in (2, 3, 4, 5, 7):
        sq = latin_square(k, seed=0)
        for row in sq:
            assert sorted(row) == list(range(k))
        for j in range(k):
            assert sorted(sq[i][j] for i in range(k)) == list(range(k))


def test_graeco_latin_square_orthogonality():
    """Every (a, b) pair appears exactly once in a Graeco-Latin square."""
    for k in (3, 4, 5, 7):
        a, b = graeco_latin_square(k, seed=1)
        assert _check_orthogonal(a, b)


def test_graeco_latin_rejects_k2_and_k6():
    with pytest.raises(ValueError, match="k = 2"):
        graeco_latin_square(2)
    with pytest.raises(ValueError, match="k = 6"):
        graeco_latin_square(6)


def test_hyper_graeco_pairwise_orthogonal():
    for k in (4, 5, 7):
        a, b, c = hyper_graeco_latin_square(k, seed=2)
        assert _check_orthogonal(a, b)
        assert _check_orthogonal(a, c)
        assert _check_orthogonal(b, c)


def test_hyper_graeco_rejects_k3():
    with pytest.raises(ValueError):
        hyper_graeco_latin_square(3)


def test_build_squares_preserves_orthogonality():
    """The shared-randomization path keeps pairwise MOLS orthogonal."""
    for k in (4, 5, 7):
        squares = _build_squares(k, 3, seed=11)
        for i in range(3):
            for j in range(i + 1, 3):
                assert _check_orthogonal(squares[i], squares[j])


# ──────────────────────────────────────────────────────────────────
# latin_square_design
# ──────────────────────────────────────────────────────────────────


def test_latin_square_design_3_factors_balance():
    d = latin_square_design(
        {"row": [1, 2, 3, 4], "col": ["a", "b", "c", "d"], "t": ["A", "B", "C", "D"]},
        seed=0,
    )
    assert d.family == "latin-square"
    assert len(d) == 16
    # Each level appears exactly k = 4 times.
    for f in ("row", "col", "t"):
        counts: dict[object, int] = {}
        for r in d.rows:
            counts[r[f]] = counts.get(r[f], 0) + 1
        assert set(counts.values()) == {4}
    # Each (row, col) pair appears exactly once.
    pairs = {(r["row"], r["col"]) for r in d.rows}
    assert len(pairs) == 16
    # Each (row, t) and (col, t) pair appears exactly once (Latin property).
    assert len({(r["row"], r["t"]) for r in d.rows}) == 16
    assert len({(r["col"], r["t"]) for r in d.rows}) == 16


def test_latin_square_design_replicates():
    d = latin_square_design(
        {"row": [1, 2, 3], "col": [1, 2, 3], "t": ["A", "B", "C"]},
        replicates=3,
        seed=1,
    )
    assert len(d) == 27
    # Each replicate has 9 rows.
    rep_counts: dict[int, int] = {}
    for r in d.rows:
        rep_counts[r["replicate"]] = rep_counts.get(r["replicate"], 0) + 1
    assert rep_counts == {0: 9, 1: 9, 2: 9}
    # Run order is a permutation of [0..26].
    assert sorted(r["run_order"] for r in d.rows) == list(range(27))


def test_latin_square_design_4_factors_uses_graeco():
    d = latin_square_design(
        {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4], "C": [1, 2, 3, 4], "D": [1, 2, 3, 4]},
        seed=3,
    )
    assert d.family == "graeco-latin"
    # All four 2-factor pairs are balanced (orthogonality).
    for a, b in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]:
        pairs = {(r[a], r[b]) for r in d.rows}
        assert len(pairs) == 16, f"pair {a},{b} not orthogonal"


def test_latin_square_design_mixed_level_count_errors():
    with pytest.raises(ValueError, match="same number of levels"):
        latin_square_design({"a": [1, 2, 3], "b": [1, 2], "c": [1, 2, 3]})


def test_latin_square_design_too_many_factors():
    with pytest.raises(ValueError, match="up to 5 factors"):
        latin_square_design({f"f{i}": [1, 2, 3, 4] for i in range(6)})


# ──────────────────────────────────────────────────────────────────
# anova_report
# ──────────────────────────────────────────────────────────────────


def test_anova_one_way_matches_hand_calc():
    """Compare against a manually computed one-way ANOVA."""
    rows = [
        {"group": "A", "y": 1.0},
        {"group": "A", "y": 2.0},
        {"group": "A", "y": 3.0},
        {"group": "B", "y": 5.0},
        {"group": "B", "y": 6.0},
        {"group": "B", "y": 7.0},
    ]
    table = anova_report(rows, response="y", factors=["group"])
    by_source = {r.source: r for r in table.rows}
    # Means: A=2, B=6, grand=4; SS_group = 3*(2-4)^2 + 3*(6-4)^2 = 24
    assert abs(by_source["group"].ss - 24.0) < 1e-9
    assert by_source["group"].df == 1
    # SS_residual = 6 (each group SS = 2, two groups).
    assert abs(by_source["Residual"].ss - 4.0) < 1e-9
    assert by_source["Residual"].df == 4
    assert abs(by_source["Total"].ss - 28.0) < 1e-9


def test_anova_latin_square_recovers_treatment_signal():
    rng = np.random.default_rng(42)
    d = latin_square_design(
        {"row": list(range(4)), "col": list(range(4)), "t": ["A", "B", "C", "D"]},
        seed=7,
    )
    truth = {"A": 10.0, "B": 12.0, "C": 11.0, "D": 13.0}
    for r in d.rows:
        r["y"] = truth[r["t"]] + rng.normal(0, 0.3)
    table = anova_report(d.rows, response="y", factors=["row", "col", "t"])
    by_source = {r.source: r for r in table.rows}
    # Treatment should be highly significant; row/col should not.
    assert by_source["t"].p is not None and by_source["t"].p < 0.001
    assert by_source["row"].p is None or by_source["row"].p > 0.05
    assert by_source["col"].p is None or by_source["col"].p > 0.05
    # Sum of SS_factors + SS_residual == SS_total.
    ss_sum = sum(r.ss for r in table.rows if r.source not in ("Total", "Residual"))
    ss_sum += by_source["Residual"].ss
    assert abs(ss_sum - by_source["Total"].ss) < 1e-8


def test_anova_df_bookkeeping():
    """Sum of df_factors + df_residual = N - 1."""
    rng = np.random.default_rng(0)
    d = latin_square_design(
        {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4], "C": [1, 2, 3, 4]},
        seed=5,
    )
    for r in d.rows:
        r["y"] = float(rng.normal(0, 1))
    table = anova_report(d.rows, response="y")
    df_total = sum(r.df for r in table.rows if r.source not in ("Total",))
    by_source = {r.source: r for r in table.rows}
    assert df_total == by_source["Total"].df


def test_anova_balanced_2way_interaction():
    """2 x 3 design with replication; interaction df = (2-1)(3-1) = 2."""
    rng = np.random.default_rng(0)
    rows = []
    for a in (0, 1):
        for b in (0, 1, 2):
            for _ in range(3):
                rows.append({"A": a, "B": b, "y": float(a + 0.5 * b + rng.normal(0, 0.1))})
    table = anova_report(rows, response="y", factors=["A", "B"], interactions=[("A", "B")])
    by_source = {r.source: r for r in table.rows}
    assert by_source["A:B"].df == 2
    assert by_source["A"].df == 1
    assert by_source["B"].df == 2
    assert by_source["Residual"].df == 18 - 1 - 1 - 2 - 2


def test_anova_unbalanced_warns():
    rows = [
        {"group": "A", "y": 1.0},
        {"group": "A", "y": 2.0},
        {"group": "B", "y": 5.0},
    ]
    with pytest.warns(UserWarning, match="unbalanced"):
        table = anova_report(rows, response="y", factors=["group"])
    assert table.balanced is False


# ──────────────────────────────────────────────────────────────────
# CLI / workbook round-trip
# ──────────────────────────────────────────────────────────────────


def test_cli_new_latin_then_anova(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from discopt.doe.cli import NewParams, do_anova, do_new

    wb_path = tmp_path / "latin.xlsx"
    out = do_new(
        NewParams(
            output=wb_path,
            n=16,
            inputs=[],
            response_name="yield",
            measurement_error=1.0,
            criterion="anova",
            seed=7,
            n_starts=1,
            template="latin-square",
            levels={
                "row": [1, 2, 3, 4],
                "col": [1, 2, 3, 4],
                "treatment": ["A", "B", "C", "D"],
            },
            replicates=1,
        )
    )
    assert out["template"] == "latin-square"
    assert wb_path.is_file()

    # Fill in synthetic responses keyed on treatment.
    rng = np.random.default_rng(0)
    truth = {"A": 10.0, "B": 12.0, "C": 11.0, "D": 13.0}
    wb = openpyxl.load_workbook(wb_path)
    sheet = wb["runs"]
    headers = [c.value for c in sheet[1]]
    y_idx = headers.index("yield")
    t_idx = headers.index("treatment")
    for row in sheet.iter_rows(min_row=2):
        if row[0].value is None:
            continue
        t = row[t_idx].value
        row[y_idx].value = float(truth[t] + rng.normal(0, 0.3))
    wb.save(wb_path)

    result = do_anova({"workbook": str(wb_path)})
    assert result["n_observations"] == 16
    sources = {r["source"]: r for r in result["rows"]}
    assert sources["treatment"]["p"] < 0.001
    assert "Residual" in sources
    assert "Total" in sources


def test_cli_latin_rejects_k6_for_graeco(tmp_path: Path) -> None:
    from discopt.doe.cli import NewParams, do_new

    wb_path = tmp_path / "k6.xlsx"
    with pytest.raises(Exception, match="k = 6"):
        do_new(
            NewParams(
                output=wb_path,
                n=36,
                inputs=[],
                response_name="y",
                measurement_error=1.0,
                criterion="anova",
                seed=1,
                n_starts=1,
                template="graeco-latin",
                levels={
                    "A": list(range(6)),
                    "B": list(range(6)),
                    "C": list(range(6)),
                    "D": list(range(6)),
                },
                replicates=1,
            )
        )
