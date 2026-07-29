"""Phase 0 — materialize the ``heldout50`` generalization panel.

Why this exists
---------------

The architecture review found one systemic evaluation risk: ``global50`` is *both*
the iteration set and the graduation gate set (``benchmarks.toml`` ``[suites.global50]``
/ ``[gates.cert*]``). A flag tuned until ``global50`` is green and then graduated on
``global50`` has been measured against its own training set. ``heldout50`` is the
held-out half of that fix: a seeded, stratified draw from the full MINLPLib snapshot
that is **disjoint from everything the iteration loop can see**, so a Regime-C card
must show its change generalizes, not merely that it did not regress where it was
developed.

Selection is **deterministic**: the same snapshot + the same ``--seed`` produce the
same 50 names, byte for byte. That is what makes the list committable and a gate
reproducible. ``--seed`` also *is* the rotation knob the plan asks for: bumping it
draws a fresh disjoint panel from the same strata, and the seed is written into the
file header so any list can be traced back to the draw that produced it.

Exclusions (what "held out" means here)
---------------------------------------

1. ``config/baron_global50.txt`` — the certification panel itself.
2. Every stem in the two in-repo ``.nl`` corpora
   (``python/tests/data/minlplib_nl`` + ``python/tests/data/minlplib``). These are
   the Regime-N iteration corpus and the corpus every in-repo probe and panel
   script reaches for; an instance a developer can run in a loop is not held out.

An instance also has to be *usable*: a ``.nl`` must exist in the snapshot and
``minlplib.solu`` must carry a reference value, or the panel could not score it.
Every filter prints how many candidates it removed — a selector that silently
narrows to a handful and then reports "50 selected" is the failure mode
CLAUDE.md §6 is about.

Stratification
--------------

Cells are ``(problem type, runtime band)``:

* **type** from ``minlplib_types.csv`` in the snapshot (schema-tolerant: the name
  column and the type column are located by header alias, not by position).
* **runtime band** from ``problems_{small,short,medium,long}.txt``. An instance in
  none of them lands in the ``unbanded`` band rather than being dropped, so the
  draw cannot silently exclude whole families because a curated list is stale.

Quota per cell is proportional to the cell's share of the eligible population,
allocated by largest remainder (deterministic ties broken by cell name), then any
shortfall is filled round-robin over the cells in a seeded order. Cells smaller
than their quota give their remainder back to the fill.

Snapshot absence
----------------

The snapshot (``~/Dropbox/projects/discopt-minlp-benchmark`` or the local mirror,
resolved by ``utils.corpus``) is a **local-only** artifact — it does not exist in CI
and does not exist in a fresh checkout. When it is missing this script does NOT
write a list and does NOT exit 0: it prints ``heldout50: SKIPPED — local only`` with
the resolution order it tried and exits 2. Everything downstream behaves the same
way (``run_benchmarks.py`` refuses to run or gate the suite, loudly), so a
``heldout50`` arm can never appear to have passed when it never ran.

Usage
-----

::

    python discopt_benchmarks/scripts/select_heldout50.py           # write the list
    python discopt_benchmarks/scripts/select_heldout50.py --n 50 --seed 20260728
    python discopt_benchmarks/scripts/select_heldout50.py --dry-run # print, write nothing
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from utils.corpus import corpus_root, nl_dir, solu_path  # noqa: E402

_OUT = _BENCH_ROOT / "config" / "suites" / "heldout50.txt"
_GLOBAL50 = _BENCH_ROOT / "config" / "baron_global50.txt"
_IN_REPO_CORPORA = (
    _REPO_ROOT / "python" / "tests" / "data" / "minlplib_nl",
    _REPO_ROOT / "python" / "tests" / "data" / "minlplib",
)

# Fixed default so the committed list is reproducible. Bump it (via --seed) to
# rotate the panel; the seed is recorded in the file header.
_DEFAULT_SEED = 20260728
_DEFAULT_N = 50

_BANDS = ("small", "short", "medium", "long")
_UNBANDED = "unbanded"

# Header aliases for minlplib_types.csv. Located by name, never by column index:
# the snapshot is refreshed from upstream and a positional reader turns a column
# reorder into a silently mis-stratified panel.
_NAME_KEYS = ("name", "instance", "Instance", "Name", "problem")
_TYPE_KEYS = ("type", "probtype", "ProbType", "Type", "problem_type", "category")


class SnapshotMissingError(RuntimeError):
    """The MINLPLib snapshot is not installed on this machine."""


def _fail_missing(what: str) -> None:
    raise SnapshotMissingError(what)


def _read_names(path: Path) -> list[str]:
    """One name per line, ``#`` comments and blanks ignored, order preserved."""
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _load_types(root: Path) -> dict[str, str]:
    """``instance -> problem type`` from ``minlplib_types.csv``.

    Raises rather than defaulting: the whole point of the draw is stratification by
    type, and a selector that silently degrades to one giant ``unknown`` stratum
    produces a panel that looks stratified and is not.
    """
    path = root / "minlplib_types.csv"
    if not path.is_file():
        _fail_missing(f"{path} (problem-type metadata) not found")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            _fail_missing(f"{path} has no header row")
        fields = {f.strip(): f for f in reader.fieldnames if f}
        name_col = next((fields[k] for k in _NAME_KEYS if k in fields), None)
        type_col = next((fields[k] for k in _TYPE_KEYS if k in fields), None)
        if name_col is None or type_col is None:
            _fail_missing(
                f"{path}: could not locate a name column (tried {_NAME_KEYS}) and a type "
                f"column (tried {_TYPE_KEYS}) in header {reader.fieldnames}"
            )
        out: dict[str, str] = {}
        for row in reader:
            name = (row.get(name_col) or "").strip()
            ptype = (row.get(type_col) or "").strip()
            if name:
                out[name] = ptype or "unknown"
    if not out:
        _fail_missing(f"{path} parsed to zero rows")
    return out


def _load_bands(root: Path) -> dict[str, str]:
    """``instance -> runtime band`` from ``problems_{small,short,medium,long}.txt``.

    A missing band file is reported (loudly) and skipped rather than fatal — the
    bands are a curation aid, and losing one degrades the stratification's
    resolution without invalidating the draw. Losing ALL of them is fatal.
    """
    bands: dict[str, str] = {}
    found = 0
    for band in _BANDS:
        p = root / f"problems_{band}.txt"
        if not p.is_file():
            print(f"  note: runtime band file {p.name} absent — band '{band}' unavailable")
            continue
        found += 1
        for name in _read_names(p):
            # First band wins, so an instance listed in two files is assigned
            # deterministically (small < short < medium < long).
            bands.setdefault(name, band)
    if found == 0:
        _fail_missing(f"no problems_{{{','.join(_BANDS)}}}.txt found under {root}")
    return bands


def _load_solu_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in ("=opt=", "=best="):
            names.add(parts[1])
    return names


def _excluded() -> tuple[set[str], dict[str, int]]:
    """Names the panel must not contain, plus a per-source count for the header."""
    excl: set[str] = set()
    detail: dict[str, int] = {}
    if _GLOBAL50.is_file():
        g = set(_read_names(_GLOBAL50))
        detail["global50"] = len(g)
        excl |= g
    else:  # pragma: no cover - the file is vendored
        print(f"  WARNING: {_GLOBAL50} missing; heldout50 cannot be proven disjoint from it")
        detail["global50"] = 0
    for d in _IN_REPO_CORPORA:
        if d.is_dir():
            stems = {p.stem for p in d.glob("*.nl")}
            detail[f"in-repo {d.name}"] = len(stems)
            excl |= stems
    return excl, detail


def _allocate(cells: dict[str, list[str]], n: int, rng: random.Random) -> dict[str, int]:
    """Largest-remainder proportional quotas, then seeded round-robin fill.

    Deterministic given ``cells`` and ``rng``: ties in the remainder are broken by
    cell name, and the fill order is a seeded shuffle of the cell names.
    """
    total = sum(len(v) for v in cells.values())
    if total == 0:
        return {}
    exact = {c: n * len(v) / total for c, v in cells.items()}
    quota = {c: min(len(cells[c]), int(exact[c])) for c in cells}
    order = sorted(cells, key=lambda c: (-(exact[c] - int(exact[c])), c))
    i = 0
    while sum(quota.values()) < n and i < len(order):
        c = order[i]
        if quota[c] < len(cells[c]):
            quota[c] += 1
        i += 1
    # Round-robin fill for whatever the largest-remainder pass could not place
    # (cells smaller than their share give their remainder back here).
    fill = sorted(cells)
    rng.shuffle(fill)
    guard = 0
    while sum(quota.values()) < n and guard < 10_000:
        progressed = False
        for c in fill:
            if sum(quota.values()) >= n:
                break
            if quota[c] < len(cells[c]):
                quota[c] += 1
                progressed = True
        guard += 1
        if not progressed:
            break
    return quota


def select(n: int, seed: int) -> tuple[list[str], dict]:
    """Draw the panel. Raises :class:`SnapshotMissingError` when the corpus is absent."""
    root = corpus_root()
    if root is None:
        _fail_missing(
            "no usable MINLPLib snapshot: set $DISCOPT_MINLP_BENCH, or install "
            "~/projects/discopt-minlp-benchmark (mirror) or "
            "~/Dropbox/projects/discopt-minlp-benchmark (canonical)"
        )
    nls = nl_dir()
    if nls is None:
        _fail_missing(f"{root} has no minlplib/nl directory")
    solu = solu_path()
    if solu is None:  # pragma: no cover - corpus_root() already requires the sentinel
        _fail_missing(f"{root} has no minlplib.solu")

    print(f"snapshot   : {root}")
    types = _load_types(root)
    bands = _load_bands(root)
    scored = _load_solu_names(solu)
    available = {p.stem for p in nls.glob("*.nl")}
    excl, excl_detail = _excluded()

    # Every narrowing step is COUNTED and printed. This is the difference between
    # "50 of 4,800 stratified" and "50 of 61 because three filters collapsed the
    # population and nobody looked".
    pop = sorted(available)
    print(f"  .nl available            : {len(pop)}")
    pop = [p for p in pop if p in types]
    print(f"  with a type in the CSV   : {len(pop)}")
    pop = [p for p in pop if p in scored]
    print(f"  with a reference in .solu: {len(pop)}")
    eligible = [p for p in pop if p not in excl]
    print(f"  after exclusions         : {len(eligible)}  (excluded {excl_detail})")

    if len(eligible) < n:
        _fail_missing(
            f"only {len(eligible)} eligible instances after filtering; cannot draw {n}"
        )

    cells: dict[str, list[str]] = defaultdict(list)
    for name in eligible:
        cells[f"{types[name]}|{bands.get(name, _UNBANDED)}"].append(name)
    for v in cells.values():
        v.sort()

    rng = random.Random(seed)
    quota = _allocate(dict(cells), n, rng)
    picked: list[str] = []
    for cell in sorted(quota):
        k = quota[cell]
        if k <= 0:
            continue
        picked.extend(random.Random(f"{seed}:{cell}").sample(cells[cell], k))
    picked.sort()

    meta = {
        "root": str(root),
        "seed": seed,
        "n_requested": n,
        "n_selected": len(picked),
        "eligible": len(eligible),
        "cells": {c: {"pool": len(cells[c]), "drawn": quota.get(c, 0)} for c in sorted(cells)},
        "excluded_sources": excl_detail,
    }
    # Disjointness is ASSERTED, not assumed — it is the entire property the panel
    # is for, and it is one line to check.
    overlap = set(picked) & excl
    if overlap:
        raise AssertionError(f"heldout50 overlaps the excluded set: {sorted(overlap)}")
    return picked, meta


def _render(names: list[str], meta: dict) -> str:
    lines = [
        "# heldout50 — Phase 0 generalization panel (consolidation plan §Phase 0.2)",
        "#",
        "# Seeded, stratified draw from the full MINLPLib snapshot, DISJOINT from",
        "# global50 (config/baron_global50.txt) and from both in-repo .nl corpora, so a",
        "# Regime-C graduation has to generalize rather than merely not regress where the",
        "# change was developed.",
        "#",
        f"# generated by : discopt_benchmarks/scripts/select_heldout50.py --seed {meta['seed']}",
        f"# snapshot     : {meta['root']}",
        f"# eligible pop : {meta['eligible']} instances after type/oracle/exclusion filters",
        f"# strata       : {len(meta['cells'])} (problem type x runtime band)",
        f"# selected     : {meta['n_selected']}",
        "#",
        "# Regenerate (identical output for the same snapshot + seed):",
        "#     python discopt_benchmarks/scripts/select_heldout50.py",
        "# Rotate the panel by bumping --seed; the seed above records which draw this is.",
        "",
    ]
    lines.extend(names)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Materialize the heldout50 generalization panel.")
    p.add_argument("--n", type=int, default=_DEFAULT_N, help=f"panel size (default {_DEFAULT_N})")
    p.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED, help=f"draw seed (default {_DEFAULT_SEED})"
    )
    p.add_argument("--out", default=str(_OUT), help="output list file")
    p.add_argument("--dry-run", action="store_true", help="print the draw, write nothing")
    args = p.parse_args(argv)

    try:
        names, meta = select(args.n, args.seed)
    except SnapshotMissingError as exc:
        # LOUD, and non-zero. The snapshot is local-only; a run that cannot see it
        # must never look like a run that saw it and found nothing to do.
        print("heldout50: SKIPPED — local only", flush=True)
        print(f"  reason: {exc}", flush=True)
        print(
            "  The MINLPLib snapshot is not part of the repository. Install it (or set "
            "$DISCOPT_MINLP_BENCH) and re-run to materialize the panel; until then any "
            "heldout50 gate arm records SKIPPED rather than passing.",
            flush=True,
        )
        return 2

    print(f"\nstrata drawn: {len(meta['cells'])}")
    for cell, c in sorted(meta["cells"].items()):
        if c["drawn"]:
            print(f"  {cell:32s} pool={c['pool']:4d} drawn={c['drawn']}")
    print(f"\nselected {len(names)} instance(s):")
    for name in names:
        print(f"  {name}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(names, meta))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
