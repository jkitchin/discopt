"""#1116: does ANY std hash container in the Rust crate get iterated?

The issue's "Suggested first step" is to swap five named ``bnb`` HashMap/HashSet
sites to a deterministic container, on the theory that Rust's per-instance
``RandomState`` seed makes their iteration order vary within a process. That
only matters if the container is ever *iterated*: ``insert`` / ``get`` /
``entry`` / ``remove`` / ``len`` / ``contains`` are all order-free.

This scan is the general form of that check: it collects every identifier
declared as a ``HashMap``/``HashSet`` anywhere in ``crates/`` and then looks for
any site that iterates one (``for x in c``, ``.iter()``, ``.iter_mut()``,
``.into_iter()``, ``.keys()``, ``.values()``, ``.values_mut()``, ``.drain()``,
``.retain()``).

Prints the number of files and lines actually scanned and the number of
declarations found, and exits non-zero if it scanned nothing — a scan that
matched no declarations would report "0 iteration sites" and read as a pass
(CLAUDE.md §6).
"""

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2] / "crates"

DECL = re.compile(
    r"(?:let\s+mut\s+|let\s+|pub\s+|\s)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?:&mut\s+)?Hash(?:Map|Set)\s*<"
)
ITER = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(iter|iter_mut|into_iter|keys|values|values_mut|drain|retain)\s*\("
)
FORIN = re.compile(r"for\s+[^\n]*\s+in\s+&?\s*(?:mut\s+)?(?:self\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\{")

files = sorted(ROOT.rglob("*.rs"))
names: dict[str, set[str]] = collections.defaultdict(set)
for f in files:
    for m in DECL.finditer(f.read_text(errors="replace")):
        names[m.group(1)].add(str(f))

hits = []
lines_scanned = 0
for f in files:
    for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
        lines_scanned += 1
        for pat in (ITER, FORIN):
            m = pat.search(line)
            if m and m.group(1) in names:
                hits.append((str(f), i, line.strip()[:140]))
                break

print(f"files_scanned={len(files)} lines_scanned={lines_scanned} declarations={len(names)}")
print(f"iteration_sites={len(hits)}")
for path, i, line in sorted(hits):
    print(f"  {path}:{i}: {line}")
if not files or not names:
    print("SCAN FIRED NOTHING", flush=True)
    sys.exit(2)
