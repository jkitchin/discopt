#!/usr/bin/env python
"""Strip dead JAX env-var boilerplate from a notebook.

JAX is not on discopt's default solve path (CLAUDE.md), so `JAX_PLATFORMS` /
`JAX_ENABLE_X64` settings are no-ops in notebooks that never import jax. Also
drops a now-unused `import os`. Refuses (exit 3) if the notebook uses jax, unless
--force is given.

Usage: strip_jax_env.py <notebook.ipynb> [--force]
"""

import json
import re
import sys

path = sys.argv[1]
force = "--force" in sys.argv
nb = json.load(open(path))

uses_jax = any(
    re.search(r"\bimport jax\b|\bjax\.|from jax\b|\bjnp\b|equinox|optax", "".join(c["source"]))
    for c in nb["cells"]
    if c["cell_type"] == "code"
)
if uses_jax and not force:
    print(f"REFUSING: {path} uses jax; strip by hand or pass --force")
    sys.exit(3)

changed = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    # Both spellings: `os.environ["JAX_X"] = v` and `os.environ.setdefault("JAX_X", v)`.
    # Only the first was matched until a sweep found the setdefault form still in
    # benchmark_dashboard and infeasibility_iis.
    if not re.search(r'os\.environ(?:\.setdefault\(|\[)\s*["\']JAX_', src):
        continue
    orig = src
    # drop the JAX_* env assignments
    jax_env = r'^\s*os\.environ\[\s*["\']JAX_[A-Z0-9_]+["\']\s*\]\s*=\s*.*\n?'
    jax_setdefault = r'^\s*os\.environ\.setdefault\(\s*["\']JAX_[A-Z0-9_]+["\']\s*,[^)]*\)\s*\n?'
    src = re.sub(jax_env, "", src, flags=re.M)
    src = re.sub(jax_setdefault, "", src, flags=re.M)
    # drop `import os` only when `os.` is unused in the WHOLE notebook. Checking
    # this cell alone broke docs/notebooks/bound_tightening.ipynb: its setup cell
    # lost `import os` while cell 10 still toggled DISCOPT_LIFTED_FBBT through
    # os.environ, so the notebook died on `NameError: name 'os' is not defined`.
    body = re.sub(r"^\s*import os\s*$", "", src, flags=re.M)
    rest = "".join(
        "".join(other["source"])
        for other in nb["cells"]
        if other is not c and other["cell_type"] == "code"
    )
    if not re.search(r"\bos\.", body) and not re.search(r"\bos\.", rest):
        src = body
    # collapse >2 blank lines and trim
    src = re.sub(r"\n{3,}", "\n\n", src).strip("\n")
    if src != orig:
        c["source"] = [line + "\n" for line in src.split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
        changed += 1

if changed == 0:
    print(f"NO-OP: no JAX env boilerplate found in {path}")
    sys.exit(1)

json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
open(path, "a").write("\n")
print(f"stripped JAX env boilerplate from {changed} cell(s) in {path}")
