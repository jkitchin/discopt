"""Population for the parity check (pre-registered rule, see docs/dev/highs-mip-replacement-plan.md):
MIPLIB 2017 easy-v2.test, minus the 62-row panel, with an =opt= reference in miplib2017-v31.solu,
ordered by sha1(name), first 120 whose .mps.gz is <= 1 MB. Prints name<TAB>opt per selected instance."""
import hashlib, os, subprocess, sys
meta, inst = sys.argv[1], sys.argv[2]
sys.path.insert(0, "discopt_benchmarks/scripts")
import p42_rens_panel as R
panel = {m["name"] for m in R.population()}
easy = [l.strip().replace(".mps.gz", "") for l in open(f"{meta}/easy-v2.test") if l.strip()]
opt = {}
for l in open(f"{meta}/miplib2017-v31.solu"):
    t = l.split()
    if len(t) >= 3 and t[0] == "=opt=":
        opt[t[1]] = float(t[2])
cand = sorted((n for n in easy if n not in panel and n in opt), key=lambda n: hashlib.sha1(n.encode()).hexdigest())
print(f"easy {len(easy)} not-in-panel-with-opt {len(cand)}", file=sys.stderr)
chosen = []
for n in cand:
    if len(chosen) >= 120:
        break
    path = f"{inst}/{n}.mps.gz"
    if not os.path.exists(path):
        r = subprocess.run(["curl", "-sS", "-o", path, "--max-time", "120", f"https://miplib.zib.de/WebData/instances/{n}.mps.gz"])
        if r.returncode != 0:
            print(f"download failed {n}", file=sys.stderr)
            continue
    if os.path.getsize(path) > 1_000_000:
        continue
    chosen.append(n)
    print(f"{n}\t{opt[n]!r}", flush=True)
print(f"chosen {len(chosen)}", file=sys.stderr)
sys.exit(0 if chosen else 1)
