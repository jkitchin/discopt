"""Symbolicate a samply --save-only profile against `nm -n` of the _rust dylib and
aggregate self / inclusive time by Rust function. Prints a resolved-frame count and
exits non-zero if nothing resolved (§6)."""

import bisect
import gzip
import json
import subprocess
import sys
from collections import defaultdict

prof_path = sys.argv[1]
dylib = sys.argv[2]

out = subprocess.run(["nm", "-n", dylib], capture_output=True, text=True, check=True).stdout
syms = []
for line in out.splitlines():
    parts = line.split(None, 2)
    if len(parts) == 3 and parts[1] in ("t", "T"):
        try:
            syms.append((int(parts[0], 16), parts[2]))
        except ValueError:
            pass
syms.sort()
addrs = [a for a, _ in syms]
print(f"nm symbols: {len(syms)}  range 0x{addrs[0]:x}..0x{addrs[-1]:x}", flush=True)


def demangle_batch(names):
    p = subprocess.run(
        ["rustfilt"], input="\n".join(names), capture_output=True, text=True
    )
    if p.returncode == 0:
        return p.stdout.splitlines()
    return names


def sym_for(addr):
    i = bisect.bisect_right(addrs, addr) - 1
    if i < 0:
        return None
    return syms[i][1]


with gzip.open(prof_path, "rt") as f:
    prof = json.load(f)

lib_names = [lb["debugName"] for lb in prof["libs"]]
rust_lib = [i for i, n in enumerate(lib_names) if n.startswith("_rust.")]
assert rust_lib, "no _rust lib in profile"
rust_lib = rust_lib[0]

total_w = 0.0
self_t = defaultdict(float)
incl_t = defaultdict(float)
resolved_frames = 0
for th in prof["threads"]:
    if not th["samples"]["length"]:
        continue
    strs = th["stringArray"]
    ft, frame, st = th["funcTable"], th["frameTable"], th["stackTable"]
    res = th["resourceTable"]
    res_lib = res.get("lib") or [None] * res["length"]
    fname = [strs[i] for i in ft["name"]]
    fres = ft["resource"]
    # frame -> label
    labels = []
    for fi in range(frame["length"]):
        fu = frame["func"][fi]
        r = fres[fu]
        lib = res_lib[r] if r is not None and r >= 0 else None
        if lib == rust_lib:
            a = frame["address"][fi]
            s = sym_for(a) if a is not None and a >= 0 else None
            if s:
                labels.append(("RUST", s))
                resolved_frames += 1
                continue
            labels.append(("RUST", fname[fu]))
        else:
            libn = lib_names[lib] if lib is not None and lib >= 0 else "?"
            labels.append((libn, fname[fu]))
    st_prefix, st_frame = st["prefix"], st["frame"]
    samples = th["samples"]
    weights = samples.get("weight")
    for i in range(samples["length"]):
        s = samples["stack"][i]
        if s is None:
            continue
        w = 1.0 if weights is None else (weights[i] or 0.0)
        total_w += w
        lib, nm_ = labels[st_frame[s]]
        self_t[(lib, nm_)] += w
        seen = set()
        cur = s
        while cur is not None:
            key = labels[st_frame[cur]]
            if key not in seen:
                seen.add(key)
                incl_t[key] += w
            cur = st_prefix[cur]

print(f"resolved rust frames: {resolved_frames}   total sample weight: {total_w:.0f}")
if resolved_frames == 0:
    sys.exit(1)

names = sorted({n for (_, n) in list(self_t) + list(incl_t)})
dem = dict(zip(names, demangle_batch(names)))

print("\n=== SELF time (top 35) ===")
for (lib, n), v in sorted(self_t.items(), key=lambda kv: -kv[1])[:35]:
    print(f"{100 * v / total_w:7.2f}%  [{lib[:20]:20s}] {dem.get(n, n)[:120]}")

print("\n=== INCLUSIVE time, rust frames (top 45) ===")
rows = [(k, v) for k, v in incl_t.items() if k[0] == "RUST"]
for (lib, n), v in sorted(rows, key=lambda kv: -kv[1])[:45]:
    print(f"{100 * v / total_w:7.2f}%  {dem.get(n, n)[:130]}")
