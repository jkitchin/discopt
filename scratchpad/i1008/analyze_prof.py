"""Aggregate a samply (Firefox Profiler format) profile into self/total time by function."""

import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
filt = sys.argv[2] if len(sys.argv) > 2 else ""
with gzip.open(path, "rt") as f:
    prof = json.load(f)

shared = prof.get("shared", {})
strings = shared.get("stringArray") or prof.get("stringArray") or []

total_samples = 0
self_time = defaultdict(float)
total_time = defaultdict(float)
n_thread = 0
for th in prof["threads"]:
    if not th.get("samples") or not th["samples"].get("length"):
        continue
    ft = th["funcTable"]
    frame = th["frameTable"]
    st = th["stackTable"]
    strs = th.get("stringArray") or strings
    fname = [strs[i] if i is not None and i < len(strs) else "?" for i in ft["name"]]
    frame_func = frame["func"]
    st_prefix = st["prefix"]
    st_frame = st["frame"]
    samples = th["samples"]
    weights = samples.get("weight")
    stacks = samples["stack"]
    n = samples["length"]
    n_thread += 1
    for i in range(n):
        s = stacks[i]
        if s is None:
            continue
        w = 1.0 if weights is None else (weights[i] or 0.0)
        total_samples += w
        # self
        leaf = fname[frame_func[st_frame[s]]]
        self_time[leaf] += w
        seen = set()
        cur = s
        while cur is not None:
            fn = fname[frame_func[st_frame[cur]]]
            if fn not in seen:
                seen.add(fn)
                total_time[fn] += w
            cur = st_prefix[cur]

print(f"threads with samples: {n_thread}   total sample weight: {total_samples:.0f}")
print("\n=== SELF time (top 30) ===")
for k, v in sorted(self_time.items(), key=lambda kv: -kv[1])[:30]:
    if filt and filt not in k:
        continue
    print(f"{100 * v / total_samples:7.2f}%  {v:9.0f}  {k[:130]}")
print("\n=== TOTAL (inclusive) time, discopt/feral frames (top 40) ===")
for k, v in sorted(total_time.items(), key=lambda kv: -kv[1])[:200]:
    if not any(t in k for t in ("simplex", "discopt", "feral", "lp::", "solve_lp")):
        continue
    print(f"{100 * v / total_samples:7.2f}%  {v:9.0f}  {k[:130]}")
