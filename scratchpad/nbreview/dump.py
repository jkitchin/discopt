#!/usr/bin/env python3
"""Dump a notebook as readable text: markdown + code + stream/text outputs."""
import json, sys, re
p = sys.argv[1]
if not p.endswith(".ipynb"): p = f"docs/notebooks/{p}.ipynb"
maxout = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
nb = json.load(open(p))
ci = 0
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        print(f"\n### [md {i}]\n{src}")
    else:
        print(f"\n### [code {i} / codeidx {ci}] exec={c.get('execution_count')}\n{src}")
        ci += 1
        for o in c.get("outputs", []):
            t = o.get("output_type")
            if t == "stream":
                txt = "".join(o.get("text", []))
            elif t in ("execute_result", "display_data"):
                txt = "".join(o.get("data", {}).get("text/plain", []))
            elif t == "error":
                txt = f"{o.get('ename')}: {o.get('evalue')}"
            else:
                txt = f"<{t}>"
            txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
            print(f"--OUT[{t}]--\n{txt[:maxout]}")
