"""Aggregate the #1180 layer-split panel into the tables the report needs."""
import json, statistics, sys

d = json.load(open(sys.argv[1]))
recs = d["records"]
LAYERS = ["pounce_native", "rust_discopt", "jax", "callback_path",
          "python_discopt", "python_numpy_scipy", "python_other"]

print(f"panel: {len(recs)} instances, time_limit={d['time_limit_s']}s")
print()
# --- corpus totals (wall-weighted; the honest aggregate) ---
tot = dict.fromkeys(LAYERS, 0.0)
tot_all = 0.0
comp_tot = {}
clean_wall = 0.0
nodes = 0
for r in recs:
    p = r["cprofile"]
    for k in LAYERS:
        tot[k] += p["by_layer_s"][k]
    tot_all += p["total_self_s"]
    clean_wall += r["clean"]["wall_s"]
    nodes += r["clean"]["nodes"]
    for k, v in p["components"].items():
        if isinstance(v, dict):
            b = comp_tot.setdefault(k, {"ncalls": 0, "self_s": 0.0, "cum_s": 0.0})
            b["ncalls"] += v["ncalls"]; b["self_s"] += v["self_s"]; b["cum_s"] += v["cum_s"]

print(f"corpus profiled self-time total: {tot_all:.1f}s over {nodes} nodes "
      f"(clean-arm wall {clean_wall:.1f}s)")
print("\n--- layer split, wall-weighted over the corpus (cProfile self time) ---")
for k in LAYERS:
    print(f"  {k:22s} {tot[k]:8.2f}s   {100*tot[k]/tot_all:5.1f}%")
print("\n--- components, corpus totals (cum = upper bound, self = additive) ---")
for k, v in sorted(comp_tot.items(), key=lambda kv: -kv[1]["cum_s"]):
    print(f"  {k:24s} n={v['ncalls']:>8}  self={v['self_s']:8.2f}s  cum={v['cum_s']:8.2f}s"
          f"  ({100*v['cum_s']/tot_all:5.1f}% of total)")

# --- per-instance median share (unweighted; guards against one instance dominating) ---
print("\n--- per-instance layer share: median (min..max) across instances ---")
for k in LAYERS:
    s = [r["cprofile"]["by_layer_pct"][k] for r in recs]
    print(f"  {k:22s} median {statistics.median(s):5.1f}%   min {min(s):5.1f}%  max {max(s):5.1f}%")

# --- FFI cross-check from the clean (unprofiled) arm ---
print("\n--- clean-arm FFI boundary split (discopt._timing; no profiler) ---")
fr = sum(r["clean"]["ffi_rust_s"] for r in recs)
fp = sum(r["clean"]["ffi_python_s"] for r in recs)
fj = sum(r["clean"]["ffi_jax_s"] for r in recs)
print(f"  rust {fr:8.2f}s  python {fp:8.2f}s  jax {fj:8.2f}s   (sum {fr+fp+fj:.1f}s "
      f"vs clean wall {clean_wall:.1f}s)")
print(f"  jax_imported on any instance: {any(r['clean']['jax_imported'] for r in recs)}")

# --- profiler distortion check ---
print("\n--- cProfile distortion (profiled wall / clean wall, node counts) ---")
ratios = []
node_mismatch = 0
for r in recs:
    cw, pw = r["clean"]["wall_s"], r["cprofile"]["wall_s"]
    ratios.append(pw / cw if cw > 0 else 1.0)
    if r["clean"]["nodes"] != r["cprofile"]["nodes"]:
        node_mismatch += 1
print(f"  median {statistics.median(ratios):.2f}x  min {min(ratios):.2f}x  max {max(ratios):.2f}x"
      f"   node-count mismatches: {node_mismatch}/{len(recs)}")

# --- top instances by rust-LP share and by OBBT share ---
print("\n--- per instance (wall>=1s), sorted by OBBT cum share ---")
rows = []
for r in recs:
    p = r["cprofile"]
    if r["clean"]["wall_s"] < 1.0:
        continue
    c = p["components"]
    obbt = c.get("obbt_probe_lp", {}).get("cum_s", 0.0)
    nlp = c.get("node_nlp_solve", {}).get("cum_s", 0.0)
    lp = c.get("rust_lp_warm", {}).get("self_s", 0.0)
    ipm = c.get("pounce_ipm", {}).get("self_s", 0.0)
    t = p["total_self_s"]
    rows.append((100*obbt/t, r["instance"], r["clean"]["wall_s"], r["clean"]["nodes"],
                 100*lp/t, 100*ipm/t, 100*nlp/t, 100*p["by_layer_s"]["callback_path"]/t,
                 c.get("rust_lp_warm", {}).get("ncalls", 0),
                 c.get("pounce_ipm", {}).get("ncalls", 0)))
rows.sort(reverse=True)
print(f"{'instance':22s} {'wall':>6} {'nodes':>6} {'OBBT%':>6} {'rustLP%':>8} {'IPM%':>6} "
      f"{'nodeNLP%':>9} {'cbglue%':>8} {'nLP':>6} {'nNLP':>6}")
for o, name, w, n, lp, ipm, nlp, cb, nlpc, nipm in rows:
    print(f"{name:22s} {w:6.1f} {n:6d} {o:6.1f} {lp:8.1f} {ipm:6.1f} {nlp:9.1f} {cb:8.1f} "
          f"{nlpc:6d} {nipm:6d}")
