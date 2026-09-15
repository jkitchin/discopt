"""Entry experiment for issue #1237 (CLAUDE.md §4).

Hypothesis: `floor`/`ceil` over a VARIABLE appear in enough of the MINLPLib
corpus to justify IR + FBBT + relaxation + reformulation work.

Measurement: count instances whose expression graph contains an ENDOGENOUS
floor/ceil -- one applied to a variable, not to a literal. A literal argument is
constant-folded by GAMS, by AMPL, and (since #1237) by discopt's GAMS parser, so
`ceil(2.3)` is a number and not evidence for anything.

Three instance formats, whichever roots are present:

  * ``.gms`` -- textual ``floor(``/``ceil(`` whose argument is not a bare number
  * ``.nl``  -- opcodes o13 (floor) / o14 (ceil). AMPL/GAMS emit these ONLY for a
    non-literal argument, so any occurrence is endogenous by construction. This
    is the strongest of the three signals.
  * ``.jl``  -- JuMP models (MINLPLib.jl). JuMP's nonlinear macros accept
    ``floor``/``ceil`` over a variable, so an instance using one would translate
    rather than be dropped, and absence is meaningful.

Usage:  python scripts/i1237_floor_ceil_corpus_probe.py [extra_root ...]

Prints an executed-comparison count and exits non-zero if it scanned nothing
(CLAUDE.md §6: prove the probe fired -- "0 hits" from a corpus that was never
there is not a measurement, so an absent root is REPORTED, never skipped
silently).
"""

import re
import sys
from pathlib import Path

#: Candidate corpus roots, in preference order: the full MINLPLib snapshot named
#: in CLAUDE.md ("Benchmark instance corpus"), a local MINLPLib.jl checkout, then
#: this repository's own corpora. Extra roots may be passed on the command line.
ROOTS = [
    Path.home() / "Dropbox/projects/discopt-minlp-benchmark/minlplib",
    Path.home() / "lanl-ansi/minlplib.jl",
    Path(__file__).resolve().parents[1],
] + [Path(a) for a in sys.argv[1:]]

#: `floor(<arg>)` / `ceil(<arg>)` in GAMS or Julia source.
TEXT_RE = re.compile(r"\b(floor|ceil)\s*\(\s*([^()]*)\)", re.IGNORECASE)
#: An argument that is a bare numeric literal -- constant-folded, not endogenous.
LITERAL_RE = re.compile(r"^[\s+\-]*\d[\d.]*([eE][+\-]?\d+)?\s*$")
#: `.nl` opcode lines are exactly "o13" / "o14" on their own line.
NL_RE = re.compile(r"^o(13|14)\s*$")

#: POSITIVE CONTROL. A zero count for floor/ceil is only evidence if the scanner
#: can fire on this corpus at all -- otherwise a regex that matches nothing
#: (wrong format, wrong suffix, unreadable tree) reports "0 endogenous uses" and
#: reads as a pass. These intrinsics ARE present in every corpus format, so the
#: run fails loudly if the control count is zero. (CLAUDE.md §6.)
CONTROL_RE = re.compile(r"\b(exp|log|sqrt)\s*\(", re.IGNORECASE)
#: o43 log / o44 log10 / o45 exp / o39 sqrt -- the `.nl` control opcodes.
NL_CONTROL_RE = re.compile(r"^o(39|43|44|45)\s*$")

SUFFIXES = (".gms", ".nl", ".jl")

scanned: dict[str, int] = {s: 0 for s in SUFFIXES}
endogenous: list[tuple[str, str]] = []
literal_only: list[str] = []
comparisons = 0
control_hits = 0
seen: set[Path] = set()

for root in ROOTS:
    if not root.is_dir():
        print(f"corpus root NOT PRESENT: {root}")
        continue
    print(f"corpus root present    : {root}")
    for path in root.rglob("*"):
        if path.suffix not in SUFFIXES or path in seen:
            continue
        seen.add(path)
        try:
            text = path.read_text(errors="replace")
        except OSError as e:  # deliberately not a bare except (CLAUDE.md §7)
            print(f"UNREADABLE {path}: {e}")
            continue
        scanned[path.suffix] += 1
        if path.suffix == ".nl":
            found = False
            for line in text.splitlines():
                comparisons += 1
                if NL_CONTROL_RE.match(line):
                    control_hits += 1
                if not found and NL_RE.match(line):
                    endogenous.append((str(path), line.strip()))
                    found = True
        else:
            control_hits += len(CONTROL_RE.findall(text))
            hit = lit = False
            for m in TEXT_RE.finditer(text):
                comparisons += 1
                if LITERAL_RE.match(m.group(2)):
                    lit = True
                else:
                    hit = True
                    endogenous.append((str(path), m.group(0)))
            if lit and not hit:
                literal_only.append(str(path))

print()
for suffix in SUFFIXES:
    print(f"scanned {suffix:<5} files          : {scanned[suffix]}")
print(f"total instance files       : {sum(scanned.values())}")
print(f"token/opcode comparisons   : {comparisons}")
print(f"positive-control hits      : {control_hits} (exp/log/sqrt; proves the")
print("                             scanner fires on these formats)")
print()
print(f"instances with ENDOGENOUS floor/ceil : {len(endogenous)}")
for p, frag in endogenous[:40]:
    print(f"    {p}: {frag}")
print(f"instances with literal-only floor/ceil: {len(literal_only)}")
for p in literal_only[:20]:
    print(f"    {p}")
print()
print(f"CHECKS_EXECUTED {comparisons + control_hits}")
if sum(scanned.values()) == 0:
    print("FAIL: the probe scanned no instance files; its result is meaningless.")
    sys.exit(1)
if control_hits == 0:
    print(
        "FAIL: the positive control found no exp/log/sqrt in any scanned file. "
        "The scanner is not reading these formats, so the floor/ceil count of "
        f"{len(endogenous)} is vacuous, not a measurement."
    )
    sys.exit(1)
