"""One-line-per-notebook summary of what changed in *source* (not outputs)."""

import difflib
import json
import re
import subprocess
import sys

JAXY = re.compile(r"(JAX_PLATFORMS|JAX_ENABLE_X64|^import os$|^$)")

for f in sys.argv[1:]:
    old = json.loads(
        subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True).stdout
    )
    new = json.load(open(f))
    o = "".join("".join(c["source"]) + "\n\x00\n" for c in old["cells"]).split("\n")
    n = "".join("".join(c["source"]) + "\n\x00\n" for c in new["cells"]).split("\n")
    rem = [
        line[1:]
        for line in difflib.unified_diff(o, n, lineterm="", n=0)
        if line.startswith("-") and not line.startswith("---")
    ]
    add = [
        line[1:]
        for line in difflib.unified_diff(o, n, lineterm="", n=0)
        if line.startswith("+") and not line.startswith("+++")
    ]
    only_jax = all(JAXY.search(line) for line in rem + add)
    print(
        f"{f.split('/')[-1][:-6]:32s} -{len(rem):3d} +{len(add):3d}  "
        f"{'JAX-ENV-ONLY' if only_jax else 'SUBSTANTIVE'}"
    )
    if not only_jax:
        for line in add[:6]:
            if line.strip() and not JAXY.search(line):
                print(f"      + {line[:120]}")
