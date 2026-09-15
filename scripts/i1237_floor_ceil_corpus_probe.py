"""Entry experiment for issue #1237 (CLAUDE.md §4).

Hypothesis: `floor`/`ceil` over a VARIABLE appear in enough of the MINLPLib
corpus to justify the IR + relaxation + reformulation work.

Measurement: count instances whose expression graph contains an endogenous
floor/ceil.  Two corpora:
  * `.gms` sources     -- textual `floor(`/`ceil(` with a non-literal argument
  * `.nl`  translations -- opcodes o13 (floor) / o14 (ceil), which are emitted
    ONLY for an endogenous argument (AMPL/GAMS constant-folds literal ones)

Prints an executed-comparison count and exits non-zero if it scanned nothing
(CLAUDE.md §6: prove the probe fired).
"""

import re
import sys
from pathlib import Path

# Candidate corpus roots, in preference order: the full MINLPLib snapshot first
# (CLAUDE.md's "Benchmark instance corpus"), then this repository's own corpora.
# A root that is absent is reported, not skipped silently -- "0 hits" from a
# corpus that was never there is not a measurement.
ROOTS = [
    Path.home() / "Dropbox/projects/discopt-minlp-benchmark/minlplib",
    Path(__file__).resolve().parents[1],
]

# `floor(<something-not-a-bare-number>)`; a literal argument is constant-folded
# by both GAMS and the discopt GAMS parser and is NOT an endogenous use.
GMS_RE = re.compile(r"\b(floor|ceil)\s*\(\s*([^)]*)\)", re.IGNORECASE)
LITERAL_RE = re.compile(r"^[\s+\-]*[\d.]+([eE][+\-]?\d+)?\s*$")
# .nl opcode lines are exactly "o13" / "o14" on their own line.
NL_RE = re.compile(r"^o(13|14)\s*$")

gms_files = nl_files = 0
gms_endogenous: list[tuple[str, str]] = []
gms_literal_only: list[str] = []
nl_hits: list[str] = []
comparisons = 0

seen: set[Path] = set()
for root in ROOTS:
    if not root.is_dir():
        print(f"corpus root NOT PRESENT: {root}")
        continue
    print(f"corpus root present: {root}")
    for path in root.rglob("*"):
        if path.suffix not in (".gms", ".nl") or path in seen:
            continue
        seen.add(path)
        try:
            text = path.read_text(errors="replace")
        except OSError as e:  # deliberately not a bare except (CLAUDE.md §7)
            print(f"UNREADABLE {path}: {e}")
            continue
        if path.suffix == ".gms":
            gms_files += 1
            endo = False
            lit = False
            for m in GMS_RE.finditer(text):
                comparisons += 1
                if LITERAL_RE.match(m.group(2)):
                    lit = True
                else:
                    endo = True
                    gms_endogenous.append((str(path), m.group(0)))
            if endo:
                pass
            elif lit:
                gms_literal_only.append(str(path))
        else:
            nl_files += 1
            for line in text.splitlines():
                comparisons += 1
                if NL_RE.match(line):
                    nl_hits.append(str(path))
                    break

print()
print(f"scanned .gms files         : {gms_files}")
print(f"scanned .nl  files         : {nl_files}")
print(f"opcode/token comparisons   : {comparisons}")
print(f".gms with ENDOGENOUS floor/ceil : {len(gms_endogenous)}")
for p, frag in gms_endogenous[:20]:
    print(f"    {p}: {frag}")
print(f".gms with literal-only floor/ceil: {len(gms_literal_only)}")
print(f".nl  with o13/o14 (endogenous)   : {len(nl_hits)}")
for p in nl_hits[:20]:
    print(f"    {p}")
print()
print(f"CHECKS_EXECUTED {comparisons}")
if comparisons == 0:
    print("FAIL: the probe scanned nothing; its result is meaningless.")
    sys.exit(1)
