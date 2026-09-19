#!/usr/bin/env python3
"""Post-run health check over every notebook: what would read as a pass but isn't."""
import json, re, sys, glob, os

SKIP_WORDS = re.compile(r"\b(skipping|Skipping|not found|not available|File exists: False|n/a|SKIP)\b")
checks = 0
flagged = 0
for p in sorted(glob.glob("docs/notebooks/*.ipynb")):
    name = os.path.basename(p)[:-6]
    nb = json.load(open(p))
    issues = []
    code = [c for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    never = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    if never:
        issues.append(f"never-executed:{len(never)}")
    for c in code:
        for o in c.get("outputs", []):
            t = "".join(o.get("text", [])) if o.get("output_type") == "stream" else ""
            if o.get("output_type") == "error":
                issues.append(f"ERROR:{o.get('ename')}")
            if "\x1b[" in t:
                issues.append("ansi")
            if re.search(r"^\s*(INFO|DEBUG)\s", t, re.M) or " INFO " in t:
                issues.append("rust-INFO")
            if "timing-bucket-unknown" in t:
                issues.append("timing-bucket-unknown")
            if SKIP_WORDS.search(t):
                issues.append(f"skip-word:{SKIP_WORDS.search(t).group(0)!r}")
    checks += 1
    if issues:
        flagged += 1
        print(f"{name:32s} {', '.join(sorted(set(issues)))}")
print(f"\n[checked {checks} notebooks, {flagged} flagged]")
assert checks > 0, "PROBE DID NOT FIRE"
