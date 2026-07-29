"""Promote a Card-2a ON-arm artifact to a ``panel_baseline/1`` baseline.

Phase 0's ``panel_baseline.py --check`` gates Regime-N cards against a frozen
artifact. When a **bound-changing** card lands (Card 2a), the old baseline stops
describing the tree and the plan requires re-baselining: *"When a card legitimately
re-baselines (a new default lands), generate a fresh ``panel_baseline_<sha>.json``
and say in the PR which baseline the next card gates against."*

Re-solving 119 instances a second time to produce that file would measure exactly
what the Card 2a ON arm already measured, on the same tree, at the same budget, on
the same corpus — an hour of machine time for a duplicate. This converts the ON-arm
rows into the baseline schema instead, reusing ``panel_baseline._annotate`` so the
``comparable`` / ``comparable_reason`` / reference-oracle fields are computed by the
**same code** the producer uses rather than re-derived here.

It refuses (exit non-zero) rather than writing a baseline that cannot gate:
a row missing ``node_count``, or a zero comparable population, is a failure.

Usage::

    python -u discopt_benchmarks/scripts/adopt_panel_baseline.py \\
        reports/card2a_cascade_aux_on.json reports/panel_baseline_<sha>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])

    from scripts.panel_baseline import (  # noqa: PLC0415
        _CORPUS_DIRS,
        _MARGIN_FRAC,
        _OBJ_RTOL,
        _OBJ_TOL,
        _annotate,
        _git_dirty,
        _oracle_fn,
        _root_gap_summary,
        _short_sha,
        _status_counts,
    )

    art = json.loads(src.read_text())
    if art.get("schema") != "card2a/1":
        raise SystemExit(f"ERROR: {src} is not a card2a/1 artifact ({art.get('schema')!r}).")
    meta = art.get("meta", {})
    budget = float(meta.get("budget_seconds", 0.0))
    if budget <= 0:
        raise SystemExit("ERROR: source artifact carries no per-instance budget.")

    oracle = _oracle_fn()
    rows = []
    n_missing_nodes = 0
    for raw in art["rows"]:
        row = dict(raw)
        row.pop("obbt_sites", None)  # harness-only telemetry, not baseline schema
        if not isinstance(row.get("node_count"), int) and row.get("status") not in (
            "errored",
            "child_crashed",
            "child_timeout",
        ):
            n_missing_nodes += 1
        rows.append(_annotate(row, budget, oracle))

    n_cmp = sum(1 for r in rows if r.get("comparable"))
    out = {
        "schema": "panel_baseline/1",
        "git_sha": _short_sha(),
        "git_dirty": _git_dirty(),
        "corpus_dirs": [str(d.relative_to(_REPO_ROOT)) for d in _CORPUS_DIRS],
        "margin_frac": _MARGIN_FRAC,
        "obj_tol": _OBJ_TOL,
        "obj_rtol": _OBJ_RTOL,
        "budget_seconds": budget,
        "total_wall_seconds": meta.get("total_wall_seconds"),
        "load_start": meta.get("load_start"),
        "load_peak": meta.get("load_peak"),
        "python": sys.version.split()[0],
        "timestamp": meta.get("timestamp"),
        "adopted_from": str(src.relative_to(_REPO_ROOT)),
        "status_counts": _status_counts(rows),
        "comparable_count": n_cmp,
        "root_gap_summary": _root_gap_summary(rows),
        "rows": rows,
    }
    print(f"rows: {len(rows)}   comparable: {n_cmp}   statuses: {out['status_counts']}")
    if n_missing_nodes:
        print(f"ERROR: {n_missing_nodes} non-errored row(s) carry no node_count.")
        return 1
    if n_cmp == 0:
        print("ERROR: zero comparable rows — this baseline could never gate anything.")
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
