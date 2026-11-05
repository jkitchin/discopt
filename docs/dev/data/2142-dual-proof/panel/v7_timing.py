"""#2142 §9 timing: tanksize and nous1 to certification, OFF/ON interleaved (order alternating),
REPS repetitions each, load-gated (1-min load <= 1.5 before each solve), one fresh process per solve.
Reports mean/sd wall and node counts.  DISCOPT_TREE=<repo>/python python -u v7_timing.py REPS"""
import json, os, statistics, subprocess, sys, time
WT = os.environ["DISCOPT_TREE"]
reps = int(sys.argv[1])
CHILD = r"""
import json, sys, time
import discopt, discopt.modeling as dm
assert discopt.__file__.startswith(sys.argv[2]), discopt.__file__
m = dm.from_nl(sys.argv[1]); t = time.perf_counter()
r = m.solve(time_limit=600, deterministic=True, gap_tolerance=1e-4)
print("T " + json.dumps({"wall": time.perf_counter() - t, "nodes": int(r.node_count or 0), "status": str(r.status), "cert": bool(r.gap_certified)}))
"""
inst = {"tanksize": "/home/user/discopt/python/tests/data/minlplib_nl/tanksize.nl",
        "nous1": "/home/user/discopt/python/tests/data/issue_1871/nous1.nl"}
res = {(n, a): [] for n in inst for a in (0, 1)}
n_runs = 0
for rep in range(reps):
    for k, (name, f) in enumerate(inst.items()):
        for a in ((0, 1) if (rep + k) % 2 == 0 else (1, 0)):
            while os.getloadavg()[0] > 1.5:
                time.sleep(5)
            load = os.getloadavg()[0]
            env = {**os.environ, "DISCOPT_DUAL_PROOF_PROPAGATION": str(a), "PYTHONHASHSEED": "0"}
            p = subprocess.run([sys.executable, "-c", CHILD, f, WT], env=env, capture_output=True, text=True, timeout=900)
            line = [l for l in p.stdout.splitlines() if l.startswith("T ")]
            assert line, p.stderr[-500:]
            d = json.loads(line[0][2:]); assert d["cert"], d
            res[(name, a)].append(d); n_runs += 1
            print(f"rep{rep} {name:9s} flag={a} wall={d['wall']:.2f}s nodes={d['nodes']} load={load:.2f}", flush=True)
for name in inst:
    for a in (0, 1):
        w = [d["wall"] for d in res[(name, a)]]
        print(f"{name:9s} flag={a} wall mean {statistics.mean(w):.2f} sd {statistics.stdev(w) if len(w) > 1 else 0:.2f} "
              f"nodes {sorted(set(d['nodes'] for d in res[(name, a)]))}")
print(f"executed assertions: {2 * n_runs}")
sys.exit(0 if n_runs else 1)
