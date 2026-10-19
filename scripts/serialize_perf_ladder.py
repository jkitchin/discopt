"""Serialization performance ladder and ratio budgets (WP-7, D13).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-7. Two parts:

* **instances** -- the in-repo ``.nl`` floor from the WP-5 harness. Per instance,
  the best of ``--repeats`` walls for ``from_nl``, ``dumps``, ``loads``,
  ``pickle.dumps`` / ``pickle.loads`` and ``copy.deepcopy``, the document size
  plain and gzipped, and the ``.nl`` size. The budgets are *ratios*, so they hold
  on any machine:

  - ``loads(dumps(m))`` wall <= ``RT_OVER_FROM_NL`` x the ``from_nl`` wall;
  - gzipped document <= ``GZ_OVER_NL`` x the ``.nl`` bytes;
  - ``pickle.dumps`` wall <= ``PICKLE_OVER_DUMPS`` x the ``dumps`` wall.

  A wall under a few milliseconds is timer noise, so every operation is timed
  ``timeit.Timer.autorange``-style: one sample repeats it until the sample lasts
  ``SAMPLE_S``, and its wall is the best sample over the call count. A wall ratio
  is judged when its denominator's *sample* lasted at least ``MIN_DENOM_S``,
  which autoranging guarantees on any machine (a fixed per-call floor did not:
  the 5 instances the budget test uses fell under it on a faster machine). A
  document carries a fixed header (schema, provenance, empty sections) that a
  tiny ``.nl`` does not, so the size budget judges ``.nl`` files of at least
  ``MIN_NL_BYTES`` (measured: the 169-byte ``mpcc_complementarity.nl`` gives a
  753-byte gzipped document, 4.5x).

* **ladder** -- synthetic models of 10^3 .. 10^6 expression nodes (one long sum
  over a 10-vector, the shape that made the deep-expression walks necessary),
  the same operations, to show how each scales.

One progress line per item, flushed. Ends with ``executed: N, over budget: B``
and exits 1 when anything is over budget, 2 when nothing executed (CLAUDE.md §6).

    python -u scripts/serialize_perf_ladder.py --out docs/dev/data/serialize-perf-<date>.jsonl
    python -u scripts/serialize_perf_ladder.py --max-nodes 100000   # a quicker ladder
"""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import importlib.util
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The budgets (plan WP-7 step 2). The owner may tighten ``PICKLE_OVER_DUMPS``
#: after O2.
RT_OVER_FROM_NL = 5.0
GZ_OVER_NL = 3.0
PICKLE_OVER_DUMPS = 1.25

#: Shortest timed sample: an operation faster than this is repeated within the
#: sample until it is not, and its wall is the sample's over the call count.
SAMPLE_S = 0.05

#: Shortest denominator *sample* a wall ratio is judged on. Autoranging makes
#: every sample at least ``SAMPLE_S``, so this fails only if the timing does.
MIN_DENOM_S = 0.005

#: Smallest ``.nl`` the size ratio is judged on.
MIN_NL_BYTES = 2048

LADDER = (1_000, 10_000, 100_000, 1_000_000)


def _harness():
    spec = importlib.util.spec_from_file_location(
        "serialize_corpus_roundtrip", REPO / "scripts" / "serialize_corpus_roundtrip.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclass resolves its module by name
    spec.loader.exec_module(mod)
    return mod


def _loops_for(fn) -> tuple[int, object]:
    """Calls per sample so one sample lasts at least ``SAMPLE_S`` (the
    ``timeit.Timer.autorange`` ladder 1, 2, 5, 10, 20, ...), and *fn*'s result."""
    loops = 1
    while True:
        for scale in (1, 2, 5):
            n = loops * scale
            t = time.perf_counter()
            for _ in range(n):
                out = fn()
            if time.perf_counter() - t >= SAMPLE_S:
                return n, out
        loops *= 10


def _best(fn, repeats: int) -> tuple[float, float, object]:
    """Per-call wall, the best sample's wall, and *fn*'s result: the best of
    *repeats* samples, each ``loops`` calls of *fn* lasting at least
    ``SAMPLE_S``, with a full collection before each sample."""
    loops, out = _loops_for(fn)
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        t = time.perf_counter()
        for _ in range(loops):
            out = fn()
        best = min(best, time.perf_counter() - t)
    return best / loops, best, out


def measure(model, repeats: int) -> dict:
    """Walls and sizes for one built model."""
    from discopt.serialize import dumps, loads

    t_dumps, s_dumps, text = _best(lambda: dumps(model, provenance=False), repeats)
    t_loads, _, _ = _best(lambda: loads(text), repeats)
    t_pdumps, _, blob = _best(lambda: pickle.dumps(model), repeats)
    t_ploads, _, _ = _best(lambda: pickle.loads(blob), repeats)
    t_deepcopy, _, _ = _best(lambda: copy.deepcopy(model), repeats)
    raw = text.encode()
    return {
        "dumps_s": t_dumps,
        "dumps_sample_s": s_dumps,
        "loads_s": t_loads,
        "pickle_dumps_s": t_pdumps,
        "pickle_loads_s": t_ploads,
        "deepcopy_s": t_deepcopy,
        "bytes": len(raw),
        "gz_bytes": len(gzip.compress(raw, mtime=0)),
        "pickle_bytes": len(blob),
        "n_nodes": len(json.loads(text)["nodes"]),
    }


def ratios(row: dict) -> dict:
    """The three budget ratios for an instance row; ``None`` where a ratio is not
    judged (a denominator sample under ``MIN_DENOM_S``, a ``.nl`` under
    ``MIN_NL_BYTES``)."""
    rt = row["dumps_s"] + row["loads_s"]
    return {
        "rt_over_from_nl": (
            rt / row["from_nl_s"] if row["from_nl_sample_s"] >= MIN_DENOM_S else None
        ),
        "gz_over_nl": (
            row["gz_bytes"] / row["nl_bytes"] if row["nl_bytes"] >= MIN_NL_BYTES else None
        ),
        "pickle_over_dumps": (
            row["pickle_dumps_s"] / row["dumps_s"] if row["dumps_sample_s"] >= MIN_DENOM_S else None
        ),
    }


def over_budget(r: dict) -> list[str]:
    out = []
    if r["rt_over_from_nl"] is not None and r["rt_over_from_nl"] > RT_OVER_FROM_NL:
        out.append(f"round trip {r['rt_over_from_nl']:.2f}x from_nl > {RT_OVER_FROM_NL}")
    if r["gz_over_nl"] is not None and r["gz_over_nl"] > GZ_OVER_NL:
        out.append(f"gz document {r['gz_over_nl']:.2f}x .nl > {GZ_OVER_NL}")
    if r["pickle_over_dumps"] is not None and r["pickle_over_dumps"] > PICKLE_OVER_DUMPS:
        out.append(f"pickle {r['pickle_over_dumps']:.2f}x dumps > {PICKLE_OVER_DUMPS}")
    return out


def measure_instance(path: Path, repeats: int) -> dict:
    import discopt.modeling as dm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_from_nl, s_from_nl, model = _best(lambda: dm.from_nl(str(path)), repeats)
        row = {
            "name": path.stem,
            "from_nl_s": t_from_nl,
            "from_nl_sample_s": s_from_nl,
            "nl_bytes": path.stat().st_size,
        }
        row.update(measure(model, repeats))
    row.update(ratios(row))
    row["over_budget"] = over_budget(row)
    return row


def synthetic(n_nodes: int):
    """A model whose one constraint is a left-deep sum of about ``n_nodes`` nodes."""
    import discopt.modeling as dm

    m = dm.Model(f"ladder_{n_nodes}")
    x = m.continuous("x", lb=-1.0, ub=1.0, shape=(10,))
    e = x[0]
    # Each step adds a product and a sum; the index nodes and the constant are
    # shared through the node table, so two nodes a step.
    for i in range(n_nodes // 2):
        e = e + 0.5 * x[i % 10]
    m.subject_to(e <= 10.0)
    m.minimize(x[0])
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="jsonl to write (one row per item)")
    ap.add_argument("--repeats", type=int, default=3, help="best of N walls (default 3)")
    ap.add_argument("--max-nodes", type=int, default=LADDER[-1], help="ladder ceiling")
    ap.add_argument("--no-ladder", action="store_true")
    args = ap.parse_args(argv)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    rows: list[dict] = []
    over = 0
    nl = [i for i in _harness().in_repo_instances() if i.kind == "nl"]
    for k, inst in enumerate(nl, 1):
        row = {
            "part": "instance",
            "corpus": inst.corpus,
            **measure_instance(inst.path, args.repeats),
        }
        rows.append(row)
        over += bool(row["over_budget"])
        print(
            f"[{k}/{len(nl)}] {inst.name}: rt/from_nl={row['rt_over_from_nl']} "
            f"gz/nl={row['gz_over_nl']} pickle/dumps={row['pickle_over_dumps']} "
            f"{'OVER ' + '; '.join(row['over_budget']) if row['over_budget'] else ''}",
            flush=True,
        )
    if not args.no_ladder:
        for n in (s for s in LADDER if s <= args.max_nodes):
            row = {"part": "ladder", "name": f"ladder_{n}", "target_nodes": n}
            row.update(measure(synthetic(n), args.repeats))
            rows.append(row)
            print(
                f"ladder {n}: nodes={row['n_nodes']} dumps={row['dumps_s']:.3f}s "
                f"loads={row['loads_s']:.3f}s pickle={row['pickle_dumps_s']:.3f}"
                f"+{row['pickle_loads_s']:.3f}s deepcopy={row['deepcopy_s']:.3f}s "
                f"gz={row['gz_bytes']}",
                flush=True,
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"executed: {len(rows)}, over budget: {over}", flush=True)
    if not rows:
        return 2
    return 1 if over else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    sys.exit(main())
