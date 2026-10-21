"""#1775 G3.2 entry experiment: HiGHS vs the Rust simplex as the GDP master backend.

Three steps, each re-runnable on its own:

    # 1. capture every LOA master MILP on the gdp_parity panel (one process each)
    for n in $(python gdp_master_backend_entry.py names); do
        timeout 150 python -u gdp_master_backend_entry.py capture $n masters/; done
    # 2. replay each master on both backends, interleaved, 3 repeats
    python -u gdp_master_backend_entry.py replay masters/ replay.json
    # 3. apply the decision rule stated before the run
    python gdp_master_backend_entry.py analyze replay.json

Decision rule: HiGHS becomes the algorithms' master default only if it agrees with
the simplex on every master (status, and objective within abs 1e-6 / rel 1e-4), is
at least 1.5x faster by SGM over the masters both solve to optimality, and is
faster in total. Result (2026-10-21): not met; see
discopt.gdp.algorithms.master.DEFAULT_MASTER_BACKEND.
"""

import glob
import json
import math
import os
import pickle
import statistics
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def names():
    from benchmarks import gdplib_runner as gr

    print(" ".join(gr.GDP_PARITY))


def capture(name, outdir):
    import discopt.solvers.lp_backend as lpb
    from discopt.pyomo.gdp import convert_pyomo_gdp
    from discopt.solvers.gdpopt_loa import solve_gdpopt_loa

    from benchmarks import gdplib_runner as gr

    spec = gr.discover_models(include=[name])[0]
    conv = convert_pyomo_gdp(spec.builder())
    real = lpb.get_milp_solver
    captured = []

    def get_milp_solver(*a, **k):
        solve = real(*a, **k)

        def wrapped(**kw):
            t = time.perf_counter()
            r = solve(**kw)
            rec = {"kw": kw, "auto_time": time.perf_counter() - t, "auto_status": str(r.status)}
            with open(os.path.join(outdir, f"{name}__{len(captured):03d}.pkl"), "wb") as f:
                pickle.dump(rec, f)
            captured.append(rec)
            return r

        return wrapped

    lpb.get_milp_solver = get_milp_solver
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_gdpopt_loa(conv.model, time_limit=60.0, max_iterations=100)
    print(
        f"{name}: status={res.status} masters={len(captured)} wall={time.perf_counter() - t0:.2f}",
        flush=True,
    )


def replay(d, out, repeats=3, time_limit=60.0):
    from discopt.solvers.lp_backend import get_milp_solver

    backends = {
        "simplex": get_milp_solver(backend="simplex"),
        "highs": get_milp_solver(backend="highs"),
    }
    rows = []
    for i, f in enumerate(sorted(glob.glob(os.path.join(d, "*.pkl")))):
        with open(f, "rb") as fh:
            kw = dict(pickle.load(fh)["kw"])
        kw["time_limit"] = time_limit
        order = ["simplex", "highs"] if i % 2 == 0 else ["highs", "simplex"]
        rec = {"master": os.path.basename(f)[:-4], "simplex": {"t": []}, "highs": {"t": []}}
        for _ in range(repeats):
            for b in order:
                t = time.perf_counter()
                r = backends[b](**kw)
                rec[b]["t"].append(time.perf_counter() - t)
                rec[b]["status"] = str(r.status)
                rec[b]["obj"] = None if r.objective is None else float(r.objective)
        rows.append(rec)
        print(
            rec["master"],
            {b: (rec[b]["status"], statistics.median(rec[b]["t"])) for b in backends},
            flush=True,
        )
    with open(out, "w") as fh:
        json.dump(rows, fh)
    print(f"compared {len(rows)} masters", flush=True)
    return len(rows)


def analyze(path):
    with open(path) as fh:
        rows = json.load(fh)

    def close(a, b):
        return abs(a - b) <= max(1e-6, 1e-4 * max(abs(a), abs(b)))

    dis, ratios, ts, th = [], [], 0.0, 0.0
    for r in rows:
        s, h = r["simplex"], r["highs"]
        ms, mh = statistics.median(s["t"]), statistics.median(h["t"])
        ts, th = ts + ms, th + mh
        same_obj = (s["obj"] is None and h["obj"] is None) or (
            s["obj"] is not None and h["obj"] is not None and close(s["obj"], h["obj"])
        )
        if s["status"] != h["status"] or not same_obj:
            dis.append((r["master"], s["status"], s["obj"], h["status"], h["obj"]))
        if "OPTIMAL" in s["status"] and "OPTIMAL" in h["status"]:
            ratios.append(max(ms, 1e-4) / max(mh, 1e-4))
    sgm = math.exp(statistics.mean(math.log(x) for x in ratios)) if ratios else float("nan")
    print(
        f"masters {len(rows)}; both optimal {len(ratios)}; SGM simplex/highs {sgm:.3f}; "
        f"total simplex {ts:.2f}s highs {th:.2f}s"
    )
    for d in dis:
        print("DIFFERENT", d)
    ok = not dis and sgm >= 1.5 and th < ts
    print("DECISION:", "HiGHS default" if ok else "#356 stands: simplex default, HiGHS opt-in")
    return len(rows)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "names":
        names()
    elif cmd == "capture":
        capture(*args)
    elif cmd == "replay":
        sys.exit(0 if replay(*args) else 1)
    elif cmd == "analyze":
        sys.exit(0 if analyze(*args) else 1)
    else:
        raise SystemExit(f"unknown command {cmd!r}")
