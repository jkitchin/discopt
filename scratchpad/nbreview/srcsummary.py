"""One-line-per-notebook summary of what changed in *source* (not outputs)."""
import difflib, json, re, subprocess, sys

JAXY = re.compile(r'(JAX_PLATFORMS|JAX_ENABLE_X64|^import os$|^$)')

for f in sys.argv[1:]:
    old = json.loads(subprocess.run(["git","show",f"HEAD:{f}"],capture_output=True,text=True).stdout)
    new = json.load(open(f))
    o = "".join("".join(c["source"])+"\n\x00\n" for c in old["cells"]).split("\n")
    n = "".join("".join(c["source"])+"\n\x00\n" for c in new["cells"]).split("\n")
    rem = [l[1:] for l in difflib.unified_diff(o,n,lineterm="",n=0) if l.startswith("-") and not l.startswith("---")]
    add = [l[1:] for l in difflib.unified_diff(o,n,lineterm="",n=0) if l.startswith("+") and not l.startswith("+++")]
    only_jax = all(JAXY.search(l) for l in rem+add)
    print(f"{f.split('/')[-1][:-6]:32s} -{len(rem):3d} +{len(add):3d}  {'JAX-ENV-ONLY' if only_jax else 'SUBSTANTIVE'}")
    if not only_jax:
        for l in add[:6]:
            if l.strip() and not JAXY.search(l):
                print(f"      + {l[:120]}")
