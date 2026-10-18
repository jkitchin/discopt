"""Round-trip every instance of a corpus through the native format (WP-5, D10/D11).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-5. For each instance:

* ``dumps`` is stable: ``dumps(loads(dumps(m))) == dumps(m)``, byte for byte;
* ``to_nl`` is byte-identical before and after (an independent writer that walks
  the whole DAG), or refuses both models with the same exception type;
* ``model_fingerprint`` is equal after ``loads(dumps(m))`` and after ``pickle``;
* the ``_source_nl_sha256`` a ``.nl``-sourced model carries is preserved (and its
  absence too: ``from_nl`` records none once it lowered complementarities).

Corpora:

* the in-repo floor (always): ``python/tests/data/minlplib_nl/*.nl``,
  ``python/tests/data/minlplib/*.nl`` (deduplicated by name), the vendored QPLIB
  instances, ``data/binary_nl/*`` and ``data/mpcc_complementarity.nl``;
* the full MINLPLib ``.nl`` set from ``DISCOPT_MINLPLIB_NL_DIR`` (default
  ``~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl``);
* the full QPLIB set from ``DISCOPT_QPLIB_DIR`` (default
  ``~/Dropbox/projects/discopt-minlp-benchmark/qplib``), filtered on
  ``qplib_manifest.csv``'s ``usable_oracle`` column -- never by name.

An absent corpus prints ``corpus X not present at <path>; skipping`` and the run
continues with the in-repo floor. One progress line per instance, flushed; the run
ends with ``executed: N, failed: F`` and exits non-zero when anything failed or
nothing executed (CLAUDE.md §6). ``faulthandler`` is NOT armed (CLAUDE.md §10,
#1758: its watchdog can wedge the process it observes).

    python -u scripts/serialize_corpus_roundtrip.py --in-repo-only
    python -u scripts/serialize_corpus_roundtrip.py \
        --out docs/dev/data/serialize-corpus-<date>.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "python" / "tests" / "data"
BENCH = Path("~/Dropbox/projects/discopt-minlp-benchmark").expanduser()


@dataclass(frozen=True)
class Instance:
    corpus: str
    name: str
    path: Path
    kind: str  # "nl" or "qplib"


# ── corpora ─────────────────────────────────────────────────────────────────


def in_repo_instances() -> list[Instance]:
    """The in-repo floor: every vendored ``.nl`` and ``.qplib`` instance."""
    out: list[Instance] = []
    seen: set[str] = set()
    for corpus, directory in (("minlplib_nl", "minlplib_nl"), ("minlplib", "minlplib")):
        for p in sorted((DATA / directory).glob("*.nl")):
            if p.stem not in seen:  # the two directories share some instances
                seen.add(p.stem)
                out.append(Instance(corpus, p.stem, p, "nl"))
    for p in sorted((DATA / "binary_nl").glob("*.nl")):
        out.append(Instance("binary_nl", p.stem, p, "nl"))
    out.append(Instance("mpcc", "mpcc_complementarity", DATA / "mpcc_complementarity.nl", "nl"))
    for p in sorted((DATA / "qplib" / "qplib").glob("*.qplib")):
        out.append(Instance("qplib_in_repo", p.stem, p, "qplib"))
    return out


def minlplib_instances() -> tuple[list[Instance], str | None]:
    root = Path(os.environ.get("DISCOPT_MINLPLIB_NL_DIR", BENCH / "minlplib" / "nl"))
    if not root.is_dir():
        return [], f"corpus minlplib_full not present at {root}; skipping"
    return [Instance("minlplib_full", p.stem, p, "nl") for p in sorted(root.glob("*.nl"))], None


def qplib_instances() -> tuple[list[Instance], str | None]:
    root = Path(os.environ.get("DISCOPT_QPLIB_DIR", BENCH / "qplib"))
    manifest = root / "qplib_manifest.csv"
    if not (root / "qplib").is_dir() or not manifest.is_file():
        return [], f"corpus qplib_full not present at {root}; skipping"
    with open(manifest, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "usable_oracle" not in rows[0]:
        raise SystemExit(f"{manifest} has no usable_oracle column; refusing to guess")
    usable = [r["name"] for r in rows if r["usable_oracle"].strip().lower() in ("1", "true")]
    out = []
    for name in sorted(usable):
        path = root / "qplib" / f"{name}.qplib"
        if path.is_file():
            out.append(Instance("qplib_full", name, path, "qplib"))
    return out, None


# ── the checks ──────────────────────────────────────────────────────────────


def _to_nl(model):
    """``.nl`` text, or the refusal's exception type: both models must agree."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ("text", model.to_nl())
    except Exception as exc:  # noqa: BLE001 - the outcome is what is compared
        return ("refused", type(exc).__name__)


def _build(inst: Instance):
    import discopt.modeling as dm
    from discopt.interfaces.qplib import from_qplib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if inst.kind == "nl":
            return dm.from_nl(str(inst.path))
        return from_qplib(str(inst.path))


def check_instance(inst: Instance) -> dict:
    """Round-trip one instance; return its jsonl row. ``status`` is ``ok`` or the
    first failed check. Exceptions propagate to the caller, which records them --
    an instrument must not swallow them (CLAUDE.md §7)."""
    from discopt.serialize import dumps, loads
    from discopt.transformations import model_fingerprint

    model = _build(inst)
    row: dict = {
        "corpus": inst.corpus,
        "name": inst.name,
        "kind": inst.kind,
        "n_vars": len(model._variables),
        "n_rows": len(model._constraints),
    }
    t0 = time.perf_counter()
    text = dumps(model, provenance=False)
    row["dumps_s"] = time.perf_counter() - t0
    raw = text.encode("utf-8")
    row["bytes"] = len(raw)
    row["gz_bytes"] = len(gzip.compress(raw, mtime=0))
    row["n_nodes"] = len(json.loads(text)["nodes"])

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded = loads(text)
    row["loads_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    copied = pickle.loads(pickle.dumps(model))
    row["pickle_s"] = time.perf_counter() - t0

    failures = []
    if dumps(reloaded, provenance=False) != text:
        failures.append("dumps not stable under reload")
    fp = model_fingerprint(model)
    if model_fingerprint(reloaded) != fp:
        failures.append("fingerprint differs after loads(dumps())")
    if model_fingerprint(copied) != fp:
        failures.append("fingerprint differs after pickle")
    if _to_nl(reloaded) != _to_nl(model):
        failures.append(".nl differs after round trip")
    # Preserved, absence included: `from_nl` records no digest for a model whose
    # complementarities it lowered (the in-memory model is no longer the file's).
    digest = getattr(model, "_source_nl_sha256", None)
    row["source_digest"] = digest is not None
    if getattr(reloaded, "_source_nl_sha256", None) != digest:
        failures.append("_source_nl_sha256 not preserved")
    row["status"] = "ok" if not failures else "; ".join(failures)
    return row


# ── driver ──────────────────────────────────────────────────────────────────


def run(instances: list[Instance], out_path: Path | None) -> tuple[int, int]:
    executed = failed = 0
    out = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for k, inst in enumerate(instances, 1):
            t0 = time.perf_counter()
            try:
                row = check_instance(inst)
            except Exception as exc:  # noqa: BLE001 - recorded, counted, printed in full
                row = {
                    "corpus": inst.corpus,
                    "name": inst.name,
                    "kind": inst.kind,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
                traceback.print_exc(file=sys.stderr)
            executed += 1
            ok = row["status"] == "ok"
            failed += not ok
            print(
                f"[{k}/{len(instances)}] {inst.corpus}/{inst.name}: "
                f"{'ok' if ok else 'FAIL ' + row['status']} "
                f"({time.perf_counter() - t0:.2f}s)",
                flush=True,
            )
            if out:
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
    finally:
        if out:
            out.close()
    return executed, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-repo-only", action="store_true", help="skip the external corpora")
    ap.add_argument("--out", type=Path, help="write one jsonl row per instance here")
    ap.add_argument("--limit", type=int, help="first N instances only (a smoke run)")
    args = ap.parse_args(argv)

    instances = in_repo_instances()
    floor = len(instances)
    if not args.in_repo_only:
        for gather in (minlplib_instances, qplib_instances):
            found, reason = gather()
            if reason:
                print(reason, flush=True)
            instances += found
    if args.limit:
        instances = instances[: args.limit]
    print(f"{len(instances)} instances (in-repo floor {floor})", flush=True)

    executed, failed = run(instances, args.out)
    print(f"executed: {executed}, failed: {failed}", flush=True)
    if executed == 0 or (not args.limit and executed < floor):
        print("nothing (or less than the in-repo floor) executed", flush=True)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
