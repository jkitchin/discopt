#!/usr/bin/env python
"""Generate ``docs/reference/flags.md`` from the two flag surfaces.

The review found the flag *policy* coherent and the flag *mechanics* undocumented:
"README has zero ``DISCOPT_`` mentions; no reference doc exists". This script is
the fix — the reference page is generated, never hand-maintained, from

* :data:`discopt._flag_registry.FLAG_REGISTRY` (everything that is not a
  ``SolverTuning`` field, including the Rust flags and the 12 expanded daemon
  flags), and
* :func:`discopt._flag_registry.solver_tuning_flags` (the dataclass half, read
  straight off the ``SolverTuning`` fields and their attribute docstrings).

Usage::

    python scripts/gen_flag_docs.py            # rewrite docs/reference/flags.md
    python scripts/gen_flag_docs.py --check    # exit 1 if the committed doc is stale

``python/tests/test_flag_registry.py::test_flags_doc_is_current`` runs ``--check``'s
comparison, so a new flag that skips the doc fails CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC_PATH = REPO / "docs" / "reference" / "flags.md"

if str(REPO / "python") not in sys.path:  # running from a source checkout
    sys.path.insert(0, str(REPO / "python"))

from discopt._flag_registry import (  # noqa: E402  (path shim above)
    FLAG_REGISTRY,
    KINDS,
    solver_tuning_flags,
)

_HEADER = """\
# Environment flags

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with `python scripts/gen_flag_docs.py`. The source of truth is
     `discopt._flag_registry.FLAG_REGISTRY` plus the `SolverTuning` dataclass. -->

Every `DISCOPT_*` environment variable discopt reads, in one place.

## The truth table

All flags parse the same way (`python/discopt/_env.py`):

| value (case-insensitive, whitespace-trimmed) | result |
| --- | --- |
| `1`, `true`, `yes`, `on` | true |
| `0`, `false`, `no`, `off` | false |
| unset, or empty | the flag's default |
| anything else | `ValueError` naming the flag and the accepted values |

The Rust core (`crates/discopt-core/src/env.rs`) uses the same table; because a
solver kernel has no exception channel, an unparseable value there warns on stderr
and falls back to the default instead of aborting.

## Kinds

| kind | meaning |
| --- | --- |
| `graduated` | Default-**ON** after a differential panel; keeps its `=0` opt-out forever. |
| `parked` | Default-**OFF** opt-in: implemented and sound, awaiting graduation. |
| `permanent` | Infrastructure knob (budgets, paths, sockets, process lifecycle). |
| `debug` | Developer instrumentation or an entry-experiment lever. |

Per CLAUDE.md §5, new behavior ships default-OFF behind a `parked` flag and only
becomes `graduated` when a corpus-wide differential panel is both cert-clean and
net-positive.
"""

_TUNING_INTRO = """\
## `SolverTuning` fields

These %d flags are the *legacy* spelling of a typed
`discopt.solver_tuning.SolverTuning` field. Prefer the object — it is
per-solve, thread-safe, validated, and discoverable:

```python
from discopt import SolverTuning
model.solve(tuning=SolverTuning(rlt_quad=False, node_bound_mode="milp"))
```

The env var supplies the field's *default* when it is not passed explicitly.

| flag | field | default | description |
| --- | --- | --- | --- |
"""


def _fmt_default(value: object) -> str:
    if value is None:
        return "_unset_"
    if isinstance(value, bool):
        return "`1` (on)" if value else "`0` (off)"
    return f"`{value}`"


def _first_sentence(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    # Strip RST/MyST roles that make no sense in a table cell.
    text = text.replace("``", "`")
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip(",;") + " …"


def render_markdown() -> str:
    """The full contents of ``docs/reference/flags.md``."""
    parts = [_HEADER]

    by_kind: dict[str, list] = {k: [] for k in KINDS}
    for spec in FLAG_REGISTRY.values():
        by_kind[spec.kind].append(spec)

    parts.append(f"\n## Flags ({len(FLAG_REGISTRY)})\n")
    for kind in KINDS:
        rows = sorted(by_kind[kind], key=lambda s: s.name)
        if not rows:
            continue
        parts.append(f"\n### `{kind}` ({len(rows)})\n\n")
        parts.append("| flag | default | side | issue | description |\n")
        parts.append("| --- | --- | --- | --- | --- |\n")
        for spec in rows:
            issue = spec.issue or "—"
            parts.append(
                f"| `{spec.name}` | {_fmt_default(spec.default)} | {spec.side} | "
                f"{issue} | {_first_sentence(spec.doc)} |\n"
            )

    tuning = solver_tuning_flags()
    parts.append("\n" + _TUNING_INTRO % len(tuning))
    for name, (field, default, doc) in sorted(tuning.items()):
        parts.append(
            f"| `{name}` | `{field}` | {_fmt_default(default)} | {_first_sentence(doc)} |\n"
        )

    total = len(FLAG_REGISTRY) + len(tuning)
    parts.append(
        f"\n**Total: {total} flags** — {len(FLAG_REGISTRY)} in the registry, "
        f"{len(tuning)} `SolverTuning` fields.\n"
    )
    return "".join(parts)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed doc differs from the generated one",
    )
    args = ap.parse_args(argv)

    generated = render_markdown()
    if args.check:
        current = DOC_PATH.read_text() if DOC_PATH.exists() else ""
        if current != generated:
            print(
                f"{DOC_PATH.relative_to(REPO)} is stale; run `python scripts/gen_flag_docs.py`.",
                file=sys.stderr,
            )
            return 1
        print(f"{DOC_PATH.relative_to(REPO)} is current ({len(generated)} bytes).")
        return 0

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(generated)
    print(f"wrote {DOC_PATH.relative_to(REPO)} ({len(generated)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
