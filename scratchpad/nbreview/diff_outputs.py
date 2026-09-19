#!/usr/bin/env python
"""Diff a notebook's stream outputs against its committed version (paths normalized)."""
import json, subprocess, sys, re
p = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
old = json.loads(subprocess.run(["git", "show", f"{ref}:{p}"], capture_output=True, text=True).stdout)
new = json.load(open(p))
def texts(nb):
    return [
        "".join("".join(o.get("text", [])) for o in c.get("outputs", []) if o.get("output_type") == "stream")
        for c in nb["cells"] if c["cell_type"] == "code"
    ]
o, n = texts(old), texts(new)
if len(o) != len(n):
    print(f"NOTE: code-cell count changed {len(o)} -> {len(n)}")
ndiff = 0
for i, (a, b) in enumerate(zip(o, n)):
    a2 = re.sub(r"/Users/jkitchin/projects/discopt", "/home/user/discopt", a)
    if a2 != b:
        ndiff += 1
        print(f"===== code cell {i} =====\n--- OLD ---\n{a[:900]}\n--- NEW ---\n{b[:900]}\n")
print(f"[compared {len(o)} cells, {ndiff} differ]")
