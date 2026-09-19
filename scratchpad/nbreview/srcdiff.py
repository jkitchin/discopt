"""Print the *source* diff of a notebook (outputs and execution metadata elided)."""

import difflib
import json
import subprocess
import sys

for f in sys.argv[1:]:
    old = json.loads(
        subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True).stdout
    )
    new = json.load(open(f))
    o = ["".join(c["source"]) for c in old["cells"]]
    n = ["".join(c["source"]) for c in new["cells"]]
    d = list(difflib.unified_diff(o, n, lineterm="", n=0))
    if not d:
        continue
    print(f"##### {f}")
    print("\n".join(d))
    print()
