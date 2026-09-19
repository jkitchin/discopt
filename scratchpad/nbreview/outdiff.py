"""Which notebooks changed OUTPUT text (not just execution timestamps)?"""

import difflib
import json
import re
import subprocess
import sys

NUM = re.compile(r"\d+\.\d+")


def texts(nb):
    out = []
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        for o in c.get("outputs", []):
            t = "".join(o.get("text", []))
            if not t:
                d = o.get("data", {}).get("text/plain", "")
                t = "".join(d) if isinstance(d, list) else str(d)
            out.append(t)
    return "\n".join(out).split("\n")


changed = []
for f in sys.argv[1:]:
    old = json.loads(
        subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True).stdout
    )
    o, n = texts(old), texts(json.load(open(f)))
    d = [
        line
        for line in difflib.unified_diff(o, n, lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("---", "+++"))
    ]
    name = f.split("/")[-1][:-6]
    if d:
        changed.append(name)
        print(f"##### {name}  ({len(d)} output lines)")
        for line in d[:14]:
            print("   ", line[:150])
print("\nOUTPUT-TEXT CHANGED:", len(changed), "of", len(sys.argv) - 1)
