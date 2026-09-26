"""#1479 bound-neutral panel driver (CLAUDE.md §5, bound-neutral regime).

Usage: python drive.py <new-tree> <base-tree> <out.jsonl>

Per case runs NEW, BASE, BASE again (interleaved, each in a fresh subprocess) and
compares status / objective / bound / node_count exactly. SAME = new equals base;
NOISE = the two base runs disagree with each other (time-limit sensitive);
DIFF = new differs from two agreeing base runs -- re-run those interleaved
several times before attributing anything to the change.
"""

import json
import os
import pathlib
import subprocess
import sys

SP = pathlib.Path(__file__).parent
NEW, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
cases = []
for r in [
    "default",
    "hull",
    "bigm",
    "mbigm",
    "auto",
    "amp",
    "mipnlp",
    "oa",
    "loa",
    "nlpbb",
    "validate",
]:
    for g in ["lin", "nl", "ind"]:
        cases.append(f"gdp:{g}/{r}")
for p in ["ib", "ib2", "im", "bm"]:
    for r in ["default", "amp"]:
        cases.append(f"prod:{p}/{r}")
cases += [
    "mpcc:gdp",
    "mpcc:sos1",
    "solve_mpec:gdp",
    "solve_mpec:sos1",
    "solve_mpec:scholtes",
    "mcp:",
    "bilevel:gdp",
    "bilevel:sos1",
]
cases += [
    f"nl:{p}" for p in sorted((pathlib.Path(NEW) / "python/tests/data/minlplib_nl").glob("*.nl"))
]
out = open(OUT, "a")  # noqa: SIM115 -- held open for the whole sweep


def key(d):
    return (
        d.get("error"),
        d.get("status"),
        d.get("objective"),
        d.get("bound"),
        d.get("node_count"),
    )


def run(tree, case):
    env = dict(os.environ, PYTHONPATH=f"{tree}/python")
    try:
        p = subprocess.run(
            [sys.executable, str(SP / "case.py"), tree, case],
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        return {"case": case, "tree": tree, "error": "timeout240"}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode or not lines:
        return {
            "case": case,
            "tree": tree,
            "error": (p.stderr.strip().splitlines() or ["?"])[-1][:300],
        }
    return json.loads(lines[-1])


for i, c in enumerate(cases):
    trio = [run(NEW, c), run(BASE, c), run(BASE, c)]
    trio[2]["tree"] = "base2"
    for t in trio:
        out.write(json.dumps(t) + "\n")
    out.flush()
    tag = (
        "SAME"
        if key(trio[0]) == key(trio[1])
        else ("NOISE" if key(trio[1]) != key(trio[2]) else "DIFF")
    )
    print(
        f"[{i + 1}/{len(cases)}] {tag} {c} new={key(trio[0])} calls={trio[0].get('calls')}",
        flush=True,
    )
