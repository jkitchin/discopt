"""#2170 panel: old rule (PR #2173 head e7d1a772) vs new gate, gdp_parity x {big-m, hull}, interleaved."""
import json, os, subprocess, sys, time
from pathlib import Path
SP = Path(__file__).resolve().parent
OLD = SP.parent / "old-wt" / "python"
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from benchmarks import gdplib_runner as gr
shard, nsh, tl, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
cells = [(n, m) for n in sorted(gr.GDP_PARITY) for m in ("big-m", "hull")]
only = os.environ.get("PANEL_ONLY")
if only:
    cells = [c for c in cells if c[0] in only.split(",")]
mine = [c for i, c in enumerate(cells) if i % nsh == shard]
done = set()
if out.exists():
    for ln in out.read_text().splitlines():
        r = json.loads(ln); done.add((r["instance"], r["method"], r["panel_arm"]))
print(f"shard {shard}: {len(mine)} cells", flush=True)
for i, (n, m) in enumerate(mine):
    arms = ("old", "new") if (i + shard) % 2 == 0 else ("new", "old")
    for arm in arms:
        if (n, m, arm) in done:
            continue
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        pp = [str(SP.parent / "shim")] if os.environ.get("PANEL_SHIM") else []
        if arm == "old":
            pp.insert(0, str(OLD))
        if pp:
            env["PYTHONPATH"] = os.pathsep.join(pp)
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, "-u", str(SP / "worker.py"), n, m, tl], env=env,
                           capture_output=True, text=True, timeout=float(tl) * 4 + 300)
        rows = [ln[4:] for ln in p.stdout.splitlines() if ln.startswith("ROW ")]
        if rows:
            r = json.loads(rows[-1])
        else:
            r = {"instance": n, "method": m, "status": "error", "err": (p.stderr or "")[-600:]}
        r["panel_arm"] = arm
        ok_marker = r.get("has_gate") is (arm == "new")
        r["marker_ok"] = ok_marker
        with out.open("a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(f"[{i+1}/{len(mine)}] {n:22s} {m:6s} {arm:3s} {r.get('status')!s:10s} "
              f"obj={r.get('objective')} bnd={r.get('bound')} nodes={r.get('node_count')} "
              f"marker={ok_marker} {time.perf_counter()-t0:.0f}s gate={r.get('gate_log')}", flush=True)
