#!/usr/bin/env python3
"""One-line verdict per notebook after a full re-run: errors, output drift, smells."""

import glob
import json
import os
import re
import subprocess

RUNS = "/tmp/claude-0/-home-user-discopt/e3e50092-94db-558b-b339-ae36f8cbc9c3/scratchpad/runs"
SKIP = re.compile(r"\b(skipping|Skipping|not found|SKIP|File exists: False)\b")


def stream_texts(nb):
    return [
        "".join(
            "".join(o.get("text", []))
            for o in c.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        for c in nb["cells"]
        if c["cell_type"] == "code"
    ]


checked = 0
for p in sorted(glob.glob("docs/notebooks/*.ipynb")):
    name = os.path.basename(p)[:-6]
    nb = json.load(open(p))
    checked += 1
    flags = []
    code = [c for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    never = [
        c
        for c in code
        if c.get("execution_count") is None
        and "skip-execution" not in (c.get("metadata", {}).get("tags") or [])
    ]
    if never:
        flags.append(f"never-exec:{len(never)}")
    for c in code:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                flags.append(f"ERROR:{o.get('ename')}")
            t = "".join(o.get("text", [])) if o.get("output_type") == "stream" else ""
            if "\x1b[" in t:
                flags.append("ansi")
            if " INFO " in t:
                flags.append("rust-INFO")
            if "timing-bucket-unknown" in t:
                flags.append("timing-bucket")
            if SKIP.search(t):
                flags.append("skip-word")
    # output drift vs the committed version
    try:
        old = json.loads(
            subprocess.run(["git", "show", f"HEAD:{p}"], capture_output=True, text=True).stdout
        )
        o, n = stream_texts(old), stream_texts(nb)
        ndiff = sum(1 for a, b in zip(o, n) if a != b)
        drift = f"{ndiff}/{len(n)}"
        if len(o) != len(n):
            drift += f" (cells {len(o)}->{len(n)})"
    except Exception:
        drift = "?"
    log = os.path.join(RUNS, f"{name}.log")
    result = "no-log"
    if os.path.exists(log):
        for line in open(log):
            if line.startswith(("RESULT", "PROBE")):
                result = line.strip()
                break
    mark = "!!" if (flags or "ERROR" in result) else "  "
    print(f"{mark} {name:32s} drift={drift:10s} {result:26s} {','.join(sorted(set(flags)))}")
print(f"\n[reviewed {checked} notebooks]")
assert checked == 66, checked
