"""#1920 panel: the HiGHS MILP route's concurrent cross-solve and fix-and-propagate start.

Usage:
    python -u discopt_benchmarks/scripts/panel_1920_milp_route.py OUT.jsonl [n_seeds] [ref_root]
    python discopt_benchmarks/scripts/panel_1920_milp_route.py analyze OUT.jsonl

Three configurations, interleaved per instance with the order rotating:

* ``base``: ``DISCOPT_MILP_CONCURRENT_CROSS=0 DISCOPT_MILP_FIXPROP_SEED=0`` -- the
  sequential #1634 cross-check and a cold HiGHS start (the pre-#1920 route, plus the
  ``_withdraw`` bound fix, which is not flagged);
* ``conc``: the concurrent cross-solve only;
* ``on``: both (the shipped default).

Arms, each with an oracle that does not come from the configuration under test:

* ``relu``: ``relu_bigm`` 2-H-H-1 networks (the issue's class), H in {16, 24, 32}.
  Oracle: a 1601 x 801 grid minimum of the network (an upper bound on the true
  minimum: a bound above it, or a certified value above it, is wrong) and a forward
  pass at the returned input (the published objective must equal the network's value
  there);
* ``relu-tl``: 2-64-64-1 networks at ``time_limit`` 20 s and 40 s: the issue's
  time-limited regime (incumbent present? bound monotone in the limit?);
* ``eps``, ``fam1654``, ``open``, ``check``: the #1733 panel's arms and oracles,
  re-used from ``panel_1733_row_form.py`` / ``panel_1678_implied_bounds.py``.

Every solve runs in a fresh subprocess with a branch-marker assertion (CLAUDE.md §8);
every incumbent is re-checked. Exits non-zero when no oracle check ran (§6).
Results: docs/dev/issue-1920-fixprop-seed-panel.md.
"""

import collections
import itertools
import json
import os
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import panel_1678_implied_bounds as p1678  # noqa: E402
import panel_1733_row_form as p1733  # noqa: E402

CONFIGS = {
    "base": {"DISCOPT_MILP_CONCURRENT_CROSS": "0", "DISCOPT_MILP_FIXPROP_SEED": "0"},
    "conc": {"DISCOPT_MILP_CONCURRENT_CROSS": "1", "DISCOPT_MILP_FIXPROP_SEED": "0"},
    "on": {"DISCOPT_MILP_CONCURRENT_CROSS": "1", "DISCOPT_MILP_FIXPROP_SEED": "1"},
}
MARKER = "fix_and_propagate_point"


def relu_layers(seed, width):
    import numpy as np

    rng = np.random.default_rng(seed)
    return [
        (rng.normal(size=(2, width)), rng.normal(size=width) * 0.5, "relu"),
        (rng.normal(size=(width, width)) / np.sqrt(width), rng.normal(size=width) * 0.5, "relu"),
        (rng.normal(size=(width, 1)) / np.sqrt(width), rng.normal(size=1) * 0.1, "linear"),
    ]


LB, UB = (-2.0, -1.0), (2.0, 1.0)


def relu_forward(layers, x):
    import numpy as np

    z = np.atleast_2d(x)
    for w, b, act in layers:
        z = z @ w + b
        if act == "relu":
            z = np.maximum(z, 0.0)
    return z[:, 0]


def relu_grid_min(seed, width):
    import numpy as np

    g1, g2 = np.linspace(LB[0], UB[0], 1601), np.linspace(LB[1], UB[1], 801)
    pts = np.array(np.meshgrid(g1, g2)).reshape(2, -1).T
    return float(relu_forward(relu_layers(seed, width), pts).min())


CHILD_RELU = r"""
import json, sys, time, warnings, contextlib, io
warnings.simplefilter("ignore")
sys.path.insert(0, HERE)
import numpy as np
import discopt, discopt.modeling as dm, discopt.solvers.lp_milp_highs as L
assert hasattr(L, MARKER), "branch marker missing"
assert discopt.__file__.startswith(ROOT), discopt.__file__
from discopt.ml import add_predictor
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition
from panel_1920_milp_route import relu_layers, relu_forward, LB, UB
seed, width, tl = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
layers = relu_layers(seed, width)
act = {"relu": Activation.RELU, "linear": Activation.LINEAR}
lb, ub = np.array(LB), np.array(UB)
net = NetworkDefinition([DenseLayer(W, b, act[a]) for W, b, a in layers], input_bounds=(lb, ub))
m = dm.Model("relu")
x = m.continuous("x", shape=(2,), lb=lb, ub=ub)
y, _ = add_predictor(m, x, net, method="relu_bigm")
m.minimize(y[0])
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    r = m.solve(time_limit=tl)
wall = time.perf_counter() - t
viol = None
if r.x is not None:
    xs = np.asarray(r.x["x"], float).ravel()
    viol = max(abs(float(relu_forward(layers, xs)[0]) - float(r.objective)),
               float(np.max(np.maximum(np.maximum(lb - xs, xs - ub), 0))))
st = r.solver_stats or {}
print(json.dumps(dict(status=r.status, obj=r.objective, bound=r.bound, cert=bool(r.gap_certified),
    wall=wall, viol=viol, nodes=int(r.node_count or 0),
    fixprop=st.get("milp/fixprop_found"), conc=st.get("milp/presolve_cross_concurrent"))))
""".replace("MARKER", repr(MARKER))


def _child(src):
    src = p1678._child(src).replace("_implied_bounds_enabled", MARKER)
    return src.replace("_cross_form_infeasible_check", MARKER)


def _run(args, src, cfg):
    env = dict(os.environ, PYTHONUNBUFFERED="1", **CONFIGS[cfg])
    p = subprocess.run(
        [sys.executable, "-c", _child(src), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if not line:
        return {"status": "CRASH", "err": p.stderr[-800:]}
    return json.loads(line[-1])


def _order(k):
    names = list(CONFIGS)
    return names[k % 3 :] + names[: k % 3]


def main():
    out = sys.argv[1]
    seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else range(4)
    ref_root = sys.argv[3] if len(sys.argv) > 3 else p1678.ROOT
    n_done = 0
    k = 0
    with open(out, "w") as fh:

        def emit(rec):
            nonlocal n_done
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done += 1
            print(
                n_done, rec["arm"], rec["inst"], rec.get("M"), rec["flag"], rec.get("status"),
                rec.get("obj"), rec.get("bound"), rec.get("cert"), f"{rec.get('wall', 0):.1f}s",
                flush=True,
            )  # fmt: skip

        for width in (16, 24, 32):
            for seed in seeds:
                ref = relu_grid_min(seed, width)
                k += 1
                for cfg in _order(k):
                    res = _run([str(seed), str(width), "300"], CHILD_RELU, cfg)
                    emit(dict(arm="relu", inst=f"relu{width}/{seed}", M=None, flag=cfg,
                              ref_ub=ref, **res))  # fmt: skip
        for seed in range(2):
            for tl in (20.0, 40.0):
                k += 1
                for cfg in _order(k):
                    res = _run([str(seed), "64", repr(tl)], CHILD_RELU, cfg)
                    emit(dict(arm="relu-tl", inst=f"relu64/{seed}", M=tl, flag=cfg,
                              ref_ub=relu_grid_min(seed, 64), **res))  # fmt: skip
        for seed in seeds:
            emin, opt = p1733.eps_oracle(seed)
            for shift, mode in itertools.product(p1733.SHIFTS, ("cold", "warm")):
                k += 1
                for cfg in _order(k):
                    res = _run([str(seed), repr(shift), mode], p1733.CHILD_EPS, cfg)
                    if res.get("r0") is not None:
                        assert abs(res["r0"] - emin) <= 1e-6, (seed, res["r0"], emin)
                    emit(dict(arm=f"eps-{mode}", inst=f"eps/{seed}", M=shift, flag=cfg,
                              ref=opt, **res))  # fmt: skip
        for fam, b in p1678.FAMILIES.items():
            arm = "fam1654" if fam in p1678.FAMILIES_1654 else "open"
            for seed in seeds:
                ref = _run([fam, str(seed), repr(b.M_ref)], p1678.CHILD_FAMILY, "base")
                ref_obj = ref.get("obj") if ref.get("cert") else None
                for big_m in (1e4, 1e6, 1e8, 1e10, 1e14):
                    k += 1
                    for cfg in _order(k):
                        res = _run([fam, str(seed), repr(big_m)], p1678.CHILD_FAMILY, cfg)
                        emit(dict(arm=arm, inst=f"{fam}/{seed}", M=big_m, flag=cfg,
                                  ref=ref_obj, **res))  # fmt: skip
        with open(os.path.join(p1678.ROOT, "discopt_benchmarks/data/lp_milp/manifest.json")) as mf:
            manifest = json.load(mf)
        for mem in manifest["members"]:
            p = mem["path"]
            p = os.path.join(p1678.ROOT if p.startswith("discopt_benchmarks") else ref_root, p)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            k += 1
            for cfg in _order(k):
                res = _run([p], p1678.CHILD_CHECK, cfg)
                emit(dict(arm="check", inst=mem["name"], M=None, flag=cfg,
                          ref=mem.get("oracle_objective"), **res))  # fmt: skip
    print("executed solves:", n_done)
    sys.exit(0 if analyze(out) else 1)


def analyze(path):
    with open(path) as fh:
        rows = [json.loads(ln) for ln in fh]
    bad = []
    checks = 0
    counts = collections.defaultdict(collections.Counter)
    walls = collections.defaultdict(list)
    for r in rows:
        key = (r["arm"], r["flag"])
        counts[key]["n"] += 1
        counts[key][r["status"]] += 1
        counts[key]["certified"] += bool(r.get("cert"))
        counts[key]["incumbent"] += r.get("obj") is not None
        walls[key].append(r.get("wall", 0.0))
        if r["status"] == "CRASH":
            bad.append(("crash", r))
            continue
        sgn = -1.0 if r.get("maximize") else 1.0
        if r.get("obj") is not None:
            checks += 1
            if r.get("viol") is None or r["viol"] > 1e-5:
                bad.append(("incumbent infeasible", r))
        ub = r.get("ref_ub")
        if ub is not None:  # relu: the grid minimum bounds the optimum from above
            checks += 1
            tol = 1e-6 * max(1.0, abs(ub))
            if r.get("bound") is not None and r["bound"] > ub + tol:
                bad.append(("bound above the grid minimum", r))
            if r.get("cert") and r.get("obj") is not None and r["obj"] > ub + tol:
                bad.append(("certified value above the grid minimum", r))
        ref = r.get("ref")
        if ref is not None:
            checks += 1
            tol = 1e-6 * max(1.0, abs(ref))
            if r.get("bound") is not None and sgn * r["bound"] > sgn * ref + tol:
                bad.append(("bound crosses the reference", r))
            if r.get("obj") is not None and sgn * r["obj"] < sgn * ref - tol:
                bad.append(("objective better than the reference", r))
            if (
                r.get("cert")
                and r["status"] == "optimal"
                and (r.get("obj") is None or abs(r["obj"] - ref) > 1e-4 * max(1.0, abs(ref)))
            ):
                bad.append(("certified wrong value", r))
    for key in sorted(counts):
        w = walls[key]
        print(
            f"arm={key[0]:9s} cfg={key[1]:4s} {dict(counts[key])}  "
            f"wall total {sum(w):.1f}s median {st.median(w):.2f}s"
        )
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["arm"], r["inst"], r.get("M"))][r["flag"]] = r
    for cfg in ("conc", "on"):
        reg = [
            k
            for k, d in by.items()
            if d.get("base", {}).get("cert") and not d.get(cfg, {}).get("cert")
        ]
        gain = [
            k
            for k, d in by.items()
            if d.get(cfg, {}).get("cert") and not d.get("base", {}).get("cert")
        ]
        drift = [
            k
            for k, d in by.items()
            if d.get("base", {}).get("cert")
            and d.get(cfg, {}).get("cert")
            and abs(d["base"]["obj"] - d[cfg]["obj"]) > 1e-4 * max(1.0, abs(d["base"]["obj"]))
        ]
        inc = [
            k
            for k, d in by.items()
            if d.get(cfg, {}).get("obj") is not None and d.get("base", {}).get("obj") is None
        ]
        lost = [
            k
            for k, d in by.items()
            if d.get(cfg, {}).get("obj") is None and d.get("base", {}).get("obj") is not None
        ]
        print(f"[{cfg} vs base] certification regressions: {reg}")
        print(f"[{cfg} vs base] certification gains: {len(gain)} {sorted(gain)[:10]}")
        print(f"[{cfg} vs base] certified objective drift: {drift}")
        print(f"[{cfg} vs base] incumbent gained: {len(inc)} lost: {lost}")
    print("soundness violations:", len(bad))
    for why, r in bad[:30]:
        print(
            "  ",
            why,
            {
                k: r.get(k)
                for k in (
                    "arm",
                    "inst",
                    "M",
                    "flag",
                    "status",
                    "obj",
                    "bound",
                    "ref",
                    "ref_ub",
                    "viol",
                    "err",
                )
            },
        )
    print("executed oracle checks:", checks)
    return checks


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        analyze(sys.argv[2])
    else:
        main()
