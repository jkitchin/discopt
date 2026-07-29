"""``select_heldout50`` must be proven to DRAW, not only proven to refuse.

On a machine without the MINLPLib snapshot — which is every CI machine and most
checkouts — the only reachable path in this script is the "SKIPPED — local only"
refusal. A selector whose selection logic has never executed anywhere is a
selector that will break silently the first time the owner runs it, and the
symptom would be a *mis-stratified graduation panel*, which is exactly the kind of
instrument this repo has been burned by (CLAUDE.md §6).

So these tests build a synthetic snapshot in ``tmp_path`` — the same file layout
``utils.corpus`` resolves (``minlplib/nl/*.nl``, ``minlplib.solu``,
``minlplib_types.csv``, ``problems_{small,short,medium,long}.txt``) — point
``$DISCOPT_MINLP_BENCH`` at it, and exercise the draw itself: determinism,
seed-sensitivity, stratification, and disjointness from the excluded sets.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from scripts import select_heldout50 as sel  # noqa: E402

_TYPES = ("MINLP", "QP", "NLP", "MIQCP", "QCQP")
_BANDS = ("small", "short", "medium", "long")

# Real names that live in global50 and/or the in-repo corpora. Planted in the
# synthetic snapshot so the exclusion filter has something to actually remove —
# a disjointness assertion over a population with no overlap proves nothing.
_PLANTED_EXCLUDED = ("alan", "casctanks", "gear4", "nvs17", "ex1221")


@pytest.fixture()
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "snap"
    (root / "minlplib" / "nl").mkdir(parents=True)
    rng = random.Random(7)
    names = [f"synth{i:04d}" for i in range(400)] + list(_PLANTED_EXCLUDED)

    rows = ["name,type,nvars"]
    solu: list[str] = []
    bands: dict[str, list[str]] = {b: [] for b in _BANDS}
    for i, n in enumerate(names):
        rows.append(f"{n},{_TYPES[i % len(_TYPES)]},{10 + i % 50}")
        solu.append(f"=opt= {n} {rng.uniform(-100, 100):.4f}")
        bands[_BANDS[i % len(_BANDS)]].append(n)
        (root / "minlplib" / "nl" / f"{n}.nl").write_text("")
    (root / "minlplib_types.csv").write_text("\n".join(rows) + "\n")
    (root / "minlplib.solu").write_text("\n".join(solu) + "\n")
    for band, ns in bands.items():
        (root / f"problems_{band}.txt").write_text("\n".join(ns) + "\n")

    monkeypatch.setenv("DISCOPT_MINLP_BENCH", str(root))
    return root


@pytest.mark.unit
def test_refuses_loudly_when_no_snapshot(monkeypatch: pytest.MonkeyPatch, capsys):
    """No snapshot => SKIPPED, non-zero. Never an empty list and exit 0."""
    monkeypatch.setenv("DISCOPT_MINLP_BENCH", "/nonexistent/definitely/not/here")
    rc = sel.main(["--dry-run"])
    assert rc != 0
    assert "SKIPPED — local only" in capsys.readouterr().out


@pytest.mark.unit
def test_draw_is_deterministic_and_seed_sensitive(snapshot: Path):
    a, _ = sel.select(50, seed=20260728)
    b, _ = sel.select(50, seed=20260728)
    c, _ = sel.select(50, seed=99)
    assert a == b, "same snapshot + same seed must give the same 50 names"
    assert a != c, "--seed must actually rotate the panel"
    assert len(a) == 50


@pytest.mark.unit
def test_draw_is_disjoint_from_global50_and_the_in_repo_corpora(snapshot: Path):
    names, meta = sel.select(50, seed=20260728)
    excluded, detail = sel._excluded()
    assert set(names).isdisjoint(excluded)
    # The exclusion must have had teeth: the planted names were candidates and
    # were removed. Without this the disjointness above is vacuous.
    assert detail["global50"] > 0
    assert set(_PLANTED_EXCLUDED) <= excluded
    assert meta["eligible"] == 400


@pytest.mark.unit
def test_draw_is_stratified_over_type_and_runtime_band(snapshot: Path):
    _, meta = sel.select(50, seed=20260728)
    drawn = {c: v for c, v in meta["cells"].items() if v["drawn"]}
    # 5 types x 4 bands, and the draw must spread over them rather than
    # concentrating in whichever cell sorts first.
    assert len(drawn) == 20
    assert sum(v["drawn"] for v in drawn.values()) == 50
    assert max(v["drawn"] for v in drawn.values()) <= 4
    for cell in drawn:
        ptype, band = cell.split("|")
        assert ptype in _TYPES
        assert band in _BANDS


@pytest.mark.unit
def test_refuses_when_the_type_csv_is_unreadable(snapshot: Path):
    """A schema change must raise, not silently collapse into one stratum."""
    (snapshot / "minlplib_types.csv").write_text("col_a,col_b\n1,2\n")
    with pytest.raises(sel.SnapshotMissingError, match="name column"):
        sel.select(50, seed=20260728)


@pytest.mark.unit
def test_refuses_rather_than_drawing_a_short_panel(snapshot: Path):
    with pytest.raises(sel.SnapshotMissingError, match="cannot draw"):
        sel.select(10_000, seed=20260728)
